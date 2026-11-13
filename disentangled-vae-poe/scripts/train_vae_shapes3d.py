#!/usr/bin/env python3
"""
Train a VAE family on 3D Shapes (Stage 3a — Plan 03, T4).

Variants
--------
vanilla   β=1 standard VAE
beta4     β=4
beta8     β=8
beta16    β=16
factorvae Kim & Mnih 2018  (KL + γ·TC,  discriminator-estimated)
adagvae   Locatello et al. 2020  (paired data, adaptive KL masking)
sepvae    β-TC-VAE  (Chen et al. 2018, TC decomposition)
          ⚠ Architecture NOT yet finalised — see MASTER_PLAN §SepVAE.
            Uses β-TC-VAE as placeholder; do not promote to eval until resolved.

Visualisations logged to W&B every --vis-every epochs
  viz/latent_traversals        rows=latent dims, cols=traversal steps
  viz/kl_spectrum              per-dim mean KL bar chart
  viz/dci_importance_heatmap   RandomForest importance R[dim, factor]
  viz/factor_lda_histograms    LDA score distributions per factor

Checkpoint  checkpoints/shapes3d/vae/{variant}/best.pt
"""
import sys
import os
import math
import argparse
import shutil
from pathlib import Path

_HERE       = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT  = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb

import yaml
from src.models.conv_vae import ConvVAE
from src.datasets.shapes3d import Shapes3D, PairedShapes3DDataset, FACTOR_NAMES
from src.losses.adagvae_loss import adagvae_loss
from src.models.sp_vae.model import SPVAE
from src.models.sp_vae.loss import SPVAELoss
from src.training.schedulers import BetaAnneal, DannLambda

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

VARIANTS = ("vanilla", "beta4", "beta8", "beta16", "factorvae", "adagvae", "sepvae", "spvae")

def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on 3D Shapes (Stage 3a T4)")
    p.add_argument("--variant",      choices=VARIANTS, required=True)
    p.add_argument("--latent-dim",   type=int,   default=10)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--gamma",        type=float, default=35.0,
                   help="TC weight for FactorVAE (default: 35)")
    p.add_argument("--beta-tc",      type=float, default=6.0,
                   help="TC weight for SepVAE / β-TC-VAE (default: 6)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--cache-dir",    type=str,   default=None)
    p.add_argument("--wandb-project",type=str,   default="disentangled-vae-poe")
    p.add_argument("--vis-every",    type=int,   default=5,
                   help="Log visualisations every N epochs")
    p.add_argument("--val-every",    type=int,   default=1)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--no-wandb",     action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# FactorVAE discriminator
# ──────────────────────────────────────────────────────────────────────────────

class _Discriminator(nn.Module):
    """6-layer MLP discriminator for FactorVAE (Kim & Mnih 2018)."""
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        h = 1000
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, h),          nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, h),          nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, h),          nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, h),          nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, 2),
        )
    def forward(self, z): return self.net(z)


def _permute_dims(z: torch.Tensor) -> torch.Tensor:
    """Permute each latent dim independently across the batch."""
    z_perm = z.clone()
    for j in range(z.shape[1]):
        perm = torch.randperm(z.shape[0], device=z.device)
        z_perm[:, j] = z[perm, j]
    return z_perm


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def _elbo(x, x_hat, mu, lv, beta=1.0):
    B = x.shape[0]
    recon = F.mse_loss(x_hat, x, reduction="sum") / B
    kl    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
    return recon + beta * kl, recon.item(), kl.item()


def _log_density_batch(z, mu, lv):
    """log N(z_i; μ_j, σ_j²) summed over latent dims for all (i,j) pairs.

    Returns (B, B) — used by the β-TC-VAE minibatch estimator.
    """
    diff  = z.unsqueeze(1) - mu.unsqueeze(0)                              # (B,B,D)
    log_p = -0.5 * (math.log(2*math.pi) + lv.unsqueeze(0)
                    + diff.pow(2) / lv.exp().unsqueeze(0))
    return log_p.sum(-1)                                                    # (B,B)


def _sepvae_loss(x, x_hat, mu, lv, z, beta_tc=6.0):
    """β-TC-VAE loss (Chen et al. 2018): MI + β·TC + DIM-KL."""
    B, D = z.shape
    recon = F.mse_loss(x_hat, x, reduction="sum") / B

    log_q_z_given_x = (-0.5 * (math.log(2*math.pi) + lv
                                + (z - mu).pow(2) / lv.exp())).sum(-1)    # (B,)

    log_pw   = _log_density_batch(z, mu, lv)                              # (B,B) joint
    log_q_z  = torch.logsumexp(log_pw, dim=1) - math.log(B)              # (B,)

    # Product of per-dim marginals
    log_q_zj = sum(
        torch.logsumexp(
            _log_density_batch(z[:, d:d+1], mu[:, d:d+1], lv[:, d:d+1]),
            dim=1,
        ) - math.log(B)
        for d in range(D)
    )                                                                       # (B,)

    log_p_z = -0.5 * (math.log(2*math.pi) + z.pow(2)).sum(-1)            # (B,)

    mi      = (log_q_z_given_x - log_q_z).mean()
    tc      = (log_q_z - log_q_zj).mean()
    dim_kl  = (log_q_zj - log_p_z).mean()
    loss    = recon + mi + beta_tc * tc + dim_kl
    return loss, recon.item(), mi.item(), tc.item(), dim_kl.item()


# ──────────────────────────────────────────────────────────────────────────────
# Training steps
# ──────────────────────────────────────────────────────────────────────────────

def _step_standard(model, batch, opt, beta, device):
    x, _ = batch
    x = x.to(device)
    opt.zero_grad(set_to_none=True)
    x_hat, mu, lv, _ = model(x)
    loss, recon, kl = _elbo(x, x_hat, mu, lv, beta=beta)
    loss.backward()
    opt.step()
    return {"train/loss": loss.item(), "train/recon": recon, "train/kl": kl}


def _step_sepvae(model, batch, opt, beta_tc, device):
    x, _ = batch
    x = x.to(device)
    opt.zero_grad(set_to_none=True)
    x_hat, mu, lv, z = model(x)
    loss, recon, mi, tc, dim_kl = _sepvae_loss(x, x_hat, mu, lv, z, beta_tc)
    loss.backward()
    opt.step()
    return {
        "train/loss": loss.item(), "train/recon": recon,
        "train/mi": mi, "train/tc": tc, "train/dim_kl": dim_kl,
    }


def _step_factorvae(model, disc, batch, opt_vae, opt_d, gamma, device):
    x, _ = batch
    x = x.to(device)

    # ── VAE update ────────────────────────────────────────────────────────
    opt_vae.zero_grad(set_to_none=True)
    x_hat, mu, lv, z = model(x)
    _, recon, kl = _elbo(x, x_hat, mu, lv, beta=1.0)
    recon_t = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    kl_t    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
    logits  = disc(z)
    log_p   = F.log_softmax(logits, dim=1)
    # TC estimate: E[log D(z)_real − log D(z)_perm]
    tc      = (log_p[:, 0] - log_p[:, 1]).mean()
    loss_v  = recon_t + kl_t + gamma * tc
    loss_v.backward()
    opt_vae.step()

    # ── Discriminator update ──────────────────────────────────────────────
    opt_d.zero_grad(set_to_none=True)
    z_real = z.detach()
    z_perm = _permute_dims(z_real)
    B      = z_real.shape[0]
    lbls_r = torch.zeros(B, dtype=torch.long, device=device)
    lbls_p = torch.ones(B,  dtype=torch.long, device=device)
    loss_d = 0.5 * (F.cross_entropy(disc(z_real), lbls_r)
                    + F.cross_entropy(disc(z_perm), lbls_p))
    loss_d.backward()
    opt_d.step()

    return {
        "train/loss": loss_v.item(), "train/recon": recon,
        "train/kl": kl, "train/tc": tc.item(), "train/disc_loss": loss_d.item(),
    }


def _step_adagvae(model, batch, opt, beta, n_changed, device):
    x_a, x_b, _ = batch
    x_a, x_b = x_a.to(device), x_b.to(device)
    opt.zero_grad(set_to_none=True)
    x_hat_a, mu_a, lv_a, _ = model(x_a)
    x_hat_b, mu_b, lv_b, _ = model(x_b)
    loss, m = adagvae_loss(x_a, x_hat_a, mu_a, lv_a,
                           x_b, x_hat_b, mu_b, lv_b,
                           beta=beta, n_changed=n_changed)
    loss.backward()
    opt.step()
    return {
        "train/loss":          m["loss"],
        "train/recon_a":       m["recon_a"],
        "train/recon_b":       m["recon_b"],
        "train/kl":            m["kl"],
        "train/n_shared_dims": m["n_shared_dims"],
    }


def _step_spvae(model, loss_fn, batch, opt, beta_c, beta_d, dann_lambda, device):
    x, labels = batch
    x = x.to(device)
    wall_labels  = labels[:, 2].long().to(device)
    floor_labels = labels[:, 3].long().to(device)
    opt.zero_grad(set_to_none=True)
    fwd = model(x)
    terms = loss_fn(fwd, x, wall_labels, floor_labels,
                    beta_c=beta_c, beta_d=beta_d, dann_lambda=dann_lambda)
    terms["total"].backward()
    opt.step()
    return {f"train/{k}": terms[k].item() for k in terms}


# ──────────────────────────────────────────────────────────────────────────────
# Validation (always on unpaired Shapes3D)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(model, val_loader, device):
    model.eval()
    total_loss = total_recon = total_kl = total_n = 0
    for x, _ in val_loader:
        x = x.to(device)
        x_hat, mu, lv, _ = model(x)
        B = x.shape[0]
        recon = F.mse_loss(x_hat, x, reduction="sum").item()
        kl    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum().item()
        total_loss  += recon + kl
        total_recon += recon
        total_kl    += kl
        total_n     += B
    model.train()
    return {
        "val/loss":  total_loss  / total_n,
        "val/recon": total_recon / total_n,
        "val/kl":    total_kl   / total_n,
    }


@torch.no_grad()
def _validate_spvae(model, loss_fn, val_loader, device):
    model.eval()
    loss_fn.eval()
    kl_c_sum = kl_d1_sum = kl_d2_sum = total_sum = 0.0
    total_n = 0
    for x, labels in val_loader:
        x = x.to(device)
        wall_labels  = labels[:, 2].long().to(device)
        floor_labels = labels[:, 3].long().to(device)
        fwd = model(x)
        terms = loss_fn(fwd, x, wall_labels, floor_labels,
                        beta_c=1.0, beta_d=1.0, dann_lambda=0.0)
        B = x.shape[0]
        kl_c_sum  += terms["kl_c"].item()  * B
        kl_d1_sum += terms["kl_d1"].item() * B
        kl_d2_sum += terms["kl_d2"].item() * B
        total_sum  += terms["total"].item() * B
        total_n += B
    model.train()
    loss_fn.train()
    return {
        "val/loss":       total_sum  / total_n,
        "val/kl_c_mean":  kl_c_sum  / total_n,
        "val/kl_d1_mean": kl_d1_sum / total_n,
        "val/kl_d2_mean": kl_d2_sum / total_n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_subset(model, val_loader, n_max, device):
    """Encode up to n_max val samples; return (Z, labels) as numpy."""
    model.eval()
    zs, lbls = [], []
    for x, labels in val_loader:
        mu, _ = model.encode(x.to(device))
        zs.append(mu.cpu().numpy())
        lbls.append(labels.numpy())
        if sum(a.shape[0] for a in zs) >= n_max:
            break
    model.train()
    Z = np.concatenate(zs)[:n_max]
    L = np.concatenate(lbls)[:n_max]
    return Z, L


@torch.no_grad()
def _fig_traversals(model, x_seed, device, n_steps=8, z_range=3.0):
    """Latent traversal grid: rows = dims, cols = values."""
    model.eval()
    mu, _ = model.encode(x_seed.unsqueeze(0).to(device))
    z0    = mu.squeeze(0)
    D     = z0.shape[0]
    vals  = torch.linspace(-z_range, z_range, n_steps, device=device)
    imgs  = []
    for k in range(D):
        for v in vals:
            z = z0.clone(); z[k] = v
            imgs.append(model.decode(z.unsqueeze(0)).squeeze(0).cpu())
    grid = make_grid(imgs, nrow=n_steps, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(n_steps * 1.2, D * 1.2))
    ax.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.axis("off")
    ax.set_title("Latent traversals  (rows = z dims, cols = [−3 … +3])", fontsize=9)
    model.train()
    return fig


@torch.no_grad()
def _fig_kl_spectrum(model, val_loader, device, n_batches=20):
    """Per-dim mean KL bar chart."""
    model.eval()
    kl_acc, n = None, 0
    for i, (x, _) in enumerate(val_loader):
        if i >= n_batches: break
        mu, lv = model.encode(x.to(device))
        kl_dim = -0.5 * (1 + lv - mu.pow(2) - lv.exp())        # (B,D)
        kl_acc = kl_dim.mean(0).cpu() if kl_acc is None else kl_acc + kl_dim.mean(0).cpu()
        n += 1
    kl_mean = (kl_acc / n).numpy()
    D = len(kl_mean)
    order = np.argsort(kl_mean)[::-1]
    fig, ax = plt.subplots(figsize=(max(6, D * 0.5), 4))
    ax.bar(range(D), kl_mean[order], color="steelblue", alpha=0.85)
    ax.set_xticks(range(D)); ax.set_xticklabels(order, fontsize=7)
    ax.set_xlabel("Latent dim (sorted by KL)"); ax.set_ylabel("Mean KL")
    ax.set_title("KL spectrum — active dims have KL >> 0")
    ax.grid(True, axis="y", alpha=0.3)
    model.train()
    return fig


def _fig_dci_heatmap(Z, L):
    """RandomForest importance matrix R[dim, factor]. Requires sklearn."""
    from sklearn.ensemble import RandomForestClassifier
    D = Z.shape[1]
    R = np.zeros((D, 6))
    for j, name in enumerate(FACTOR_NAMES):
        clf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                     n_jobs=-1, random_state=0)
        clf.fit(Z, L[:, j])
        R[:, j] = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(7, max(4, D * 0.4)))
    im = ax.imshow(R, aspect="auto", cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(FACTOR_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(D)); ax.set_yticklabels([f"z{k}" for k in range(D)], fontsize=7)
    ax.set_xlabel("Factor"); ax.set_ylabel("Latent dim")
    ax.set_title("DCI importance  R[dim, factor]")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _fig_factor_lda(Z, L):
    """LDA score histograms — one panel per factor."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    n_factors = 6
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
    for j, (ax, name) in enumerate(zip(axes.flat, FACTOR_NAMES)):
        labels = L[:, j]
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)
            scores = lda.fit_transform(Z, labels).ravel()
        except Exception:
            ax.set_title(f"{name}\n(LDA failed)"); continue
        for val in np.unique(labels):
            ax.hist(scores[labels == val], bins=30, alpha=0.5, density=True,
                    label=str(int(val)))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("LDA score"); ax.set_yticks([])
    for ax in axes.flat[n_factors:]:
        ax.axis("off")
    fig.suptitle("Factor LDA score distributions", fontsize=10)
    fig.tight_layout()
    return fig


def _log_visuals(model, val_loader, x_seed, device, step):
    """Compute and log all four visualisation panels to W&B."""
    Z, L = _encode_subset(model, val_loader, n_max=2000, device=device)

    figs = {
        "viz/latent_traversals":      _fig_traversals(model, x_seed, device),
        "viz/kl_spectrum":            _fig_kl_spectrum(model, val_loader, device),
        "viz/dci_importance_heatmap": _fig_dci_heatmap(Z, L),
        "viz/factor_lda_histograms":  _fig_factor_lda(Z, L),
    }
    wandb.log({k: wandb.Image(fig) for k, fig in figs.items()}, step=step)
    for fig in figs.values():
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def _save(model, opt, epoch, val_loss, cfg, ckpt_path, disc=None, opt_d=None, loss_fn=None):
    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "val_loss":             val_loss,
        "config":               cfg,
    }
    if disc is not None:
        payload["disc_state_dict"]  = disc.state_dict()
        payload["opt_d_state_dict"] = opt_d.state_dict()
    if loss_fn is not None:
        payload["loss_fn_state_dict"] = loss_fn.state_dict()
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(ckpt_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Variant-specific β ──────────────────────────────────────────────
    _BETA = {"vanilla": 1.0, "beta4": 4.0, "beta8": 8.0, "beta16": 16.0,
             "factorvae": 1.0, "adagvae": 1.0, "sepvae": 1.0, "spvae": 1.0}
    beta = _BETA[args.variant]

    # ── SPVAE config (loaded from YAML when variant==spvae) ─────────────
    spvae_cfg = None
    if args.variant == "spvae":
        spvae_cfg = yaml.safe_load(
            open(Path(_SCOPE_ROOT) / "configs" / "shapes3d" / "sp_vae" / "default.yml")
        )

    # ── Datasets ────────────────────────────────────────────────────────
    kw = dict(cache_dir=args.cache_dir, seed=args.seed)
    if args.variant == "adagvae":
        train_ds = PairedShapes3DDataset(split="train", **kw)
    else:
        train_ds = Shapes3D(split="train", **kw)
    val_ds  = Shapes3D(split="val",   **kw)

    w = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=w, pin_memory=True,
                              persistent_workers=w > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=w, pin_memory=True,
                              persistent_workers=w > 0)

    # ── Model ───────────────────────────────────────────────────────────
    if args.variant == "spvae":
        model = SPVAE(z_c_dim=spvae_cfg["z_c_dim"],
                      z_d_dim=spvae_cfg["z_d_dim"]).to(device)
        loss_fn = SPVAELoss(
            z_d_dim=spvae_cfg["z_d_dim"],
            vicreg_eps=spvae_cfg["vicreg_eps"],
            orth_weight=spvae_cfg["orth_weight"],
            mi_weight=spvae_cfg["mi_weight"],
            ce_weight=spvae_cfg["ce_weight"],
            grl_weight=spvae_cfg["grl_weight"],
        ).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=spvae_cfg["lr"],
        )
        b_sched_c = BetaAnneal(warmup_frac=spvae_cfg["warmup_frac"],
                               beta_max=spvae_cfg["beta_c_max"])
        b_sched_d = BetaAnneal(warmup_frac=spvae_cfg["warmup_frac"],
                               beta_max=spvae_cfg["beta_d_max"])
        d_sched   = DannLambda(warmup_frac=spvae_cfg["warmup_frac"],
                               lambda_max=spvae_cfg["dann_lambda_max"])
    else:
        model   = ConvVAE(latent_dim=args.latent_dim).to(device)
        loss_fn = None
        opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
        b_sched_c = b_sched_d = d_sched = None

    disc   = None
    opt_d  = None
    if args.variant == "factorvae":
        disc  = _Discriminator(args.latent_dim).to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr)

    # ── Config dict for checkpoints ─────────────────────────────────────
    if args.variant == "spvae":
        cfg = {**spvae_cfg, "variant": "spvae", "epochs": args.epochs, "seed": args.seed}
    else:
        cfg = dict(
            variant   = args.variant,
            latent_dim= args.latent_dim,
            beta      = beta,
            gamma     = args.gamma,
            beta_tc   = args.beta_tc,
            lr        = args.lr,
            batch_size= args.batch_size,
            epochs    = args.epochs,
            seed      = args.seed,
        )

    # ── Checkpoint dir ──────────────────────────────────────────────────
    ckpt_dir = (Path(_SCOPE_ROOT) / "checkpoints" / "shapes3d" / "vae"
                / args.variant)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    # ── W&B ─────────────────────────────────────────────────────────────
    _tags = ["stage3", args.variant]
    if args.variant == "spvae":
        _tags.append("sp-vae")
    if not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            name    = f"vae_{args.variant}_seed{args.seed}",
            tags    = _tags,
            config  = cfg,
        )

    # ── Seed image for traversals ────────────────────────────────────────
    x_seed, _ = val_ds[0]

    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best_val = float("inf")
    model.train()

    for epoch in range(1, args.epochs + 1):
        # Ada-GVAE: shuffle pairs each epoch
        if args.variant == "adagvae":
            train_ds.set_epoch(epoch)

        # ── Train ───────────────────────────────────────────────────────
        epoch_metrics: dict[str, float] = {}
        for batch in train_loader:
            if args.variant in ("vanilla", "beta4", "beta8", "beta16"):
                m = _step_standard(model, batch, opt, beta, device)
            elif args.variant == "sepvae":
                m = _step_sepvae(model, batch, opt, args.beta_tc, device)
            elif args.variant == "factorvae":
                m = _step_factorvae(model, disc, batch, opt, opt_d, args.gamma, device)
            elif args.variant == "spvae":
                beta_c_val = b_sched_c.get(global_step, total_steps)
                beta_d_val = b_sched_d.get(global_step, total_steps)
                lam_val    = d_sched.get(global_step, total_steps)
                m = _step_spvae(model, loss_fn, batch, opt,
                                beta_c_val, beta_d_val, lam_val, device)
            else:  # adagvae
                m = _step_adagvae(model, batch, opt, beta, n_changed=1, device=device)

            for k, v in m.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            global_step += 1

        n_batches = len(train_loader)
        epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        epoch_metrics["epoch"] = epoch

        # ── Validate ─────────────────────────────────────────────────────
        if epoch % args.val_every == 0:
            if args.variant == "spvae":
                val_m = _validate_spvae(model, loss_fn, val_loader, device)
            else:
                val_m = _validate(model, val_loader, device)
            epoch_metrics.update(val_m)

            if val_m["val/loss"] < best_val:
                best_val = val_m["val/loss"]
                _save(model, opt, epoch, best_val, cfg, ckpt_path, disc, opt_d,
                      loss_fn=loss_fn)
                if not args.no_wandb:
                    wandb.run.summary["best_val_loss"] = best_val
                    wandb.run.summary["best_epoch"]    = epoch

            # convergence guard at final epoch
            if args.variant == "spvae" and epoch == args.epochs:
                for slot in ("kl_c", "kl_d1", "kl_d2"):
                    v = val_m.get(f"val/{slot}_mean", 0.0)
                    if v < 0.05:
                        print(f"WARNING: posterior collapse in slot {slot} (KL={v:.4f})")

        # ── Visualisations ────────────────────────────────────────────────
        if epoch % args.vis_every == 0 and not args.no_wandb and args.variant != "spvae":
            _log_visuals(model, val_loader, x_seed, device, step=epoch)

        if not args.no_wandb:
            wandb.log(epoch_metrics, step=epoch)

        print(
            f"[{args.variant}] epoch {epoch:3d}/{args.epochs}"
            f"  loss={epoch_metrics.get('train/loss', 0):.4f}"
            f"  val={epoch_metrics.get('val/loss', best_val):.4f}"
            f"  best={best_val:.4f}"
        )

    if not args.no_wandb:
        wandb.finish()
    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
