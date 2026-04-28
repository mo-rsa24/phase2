#!/usr/bin/env python3
"""Train FactorVAE (Kim & Mnih, 2018) on dSprites.

Mirrors scripts/train_vae.py 1:1 in structure, wandb logging keys, and
checkpoint format. The difference is the alternating update:

  1. VAE step on the first half of each batch — recon + KL + γ·TC.
  2. Discriminator step on the second half — distinguish q(z) from
     ∏_i q(z_i) (the latter via permute_dims).

Helpers (probe_device, load_config, seed_everything, make_recon_grid,
make_pca_manifold, _autocast_ctx) are reused from scripts/train_vae.py
via importlib spec-loading — the same pattern sweep_disentanglement.py
already uses.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
import wandb

from src.datasets.dsprites import (
    DSpritesDataset,
    load_dsprites,
    make_iid_split,
)
from src.datasets.correlated_dsprites import (
    make_correlated_split,
    make_heldout_pair_split,
)
from src.models.factor_vae import FactorVAE, permute_dims


# ---------------------------------------------------------------------------
# Reuse helpers from train_vae.py via spec.loader (matches sweep_disentanglement
# pattern; avoids needing scripts/ to be a package).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_VAE_PATH = REPO_ROOT / "scripts" / "train_vae.py"
_spec = importlib.util.spec_from_file_location("train_vae", _TRAIN_VAE_PATH)
_train_vae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_vae)

probe_device      = _train_vae.probe_device
load_config       = _train_vae.load_config
seed_everything   = _train_vae.seed_everything
make_recon_grid   = _train_vae.make_recon_grid
make_pca_manifold = _train_vae.make_pca_manifold
_autocast_ctx     = _train_vae._autocast_ctx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train FactorVAE on dSprites")
    p.add_argument("--config",          type=str, default="configs/factor_vae.yml")
    p.add_argument("--data-dir",        type=str, default="data")
    p.add_argument("--out-dir",         type=str, default="checkpoints/factor_vae")
    # Experiment overrides (CLI > runtime overlay > YAML).
    p.add_argument("--latent-dim",      type=int,   default=None)
    p.add_argument("--gamma",           type=float, default=None,
                   help="TC weight γ (FactorVAE-specific). Overrides training.gamma.")
    p.add_argument("--disc-lr",         type=float, default=None,
                   help="Discriminator learning rate. Overrides training.disc_lr.")
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--num-workers",     type=int,   default=None)
    # Dataset split selection (Phase 2b/2e).
    p.add_argument("--split",           type=str, default="iid",
                   choices=["iid", "correlated", "heldout"])
    p.add_argument("--corr-factor-a",   type=str, default="scale")
    p.add_argument("--corr-factor-b",   type=str, default="orientation")
    p.add_argument("--corr-direction",  type=str, default="positive",
                   choices=["positive", "negative"])
    p.add_argument("--heldout-factor-a", type=str, default="shape")
    p.add_argument("--heldout-factor-b", type=str, default="scale")
    p.add_argument("--heldout-a-vals",   type=int, nargs="*", default=[2])
    p.add_argument("--heldout-b-vals",   type=int, nargs="*", default=[4, 5])
    p.add_argument("--runtime",         type=str,   default=None,
                   help="Runtime overlay key (e.g. 'hippo', 'cluster48').")
    # W&B metadata.
    p.add_argument("--wandb-run-name",  type=str, default=None)
    p.add_argument("--wandb-group",     type=str, default=None)
    p.add_argument("--wandb-tags",      type=str, nargs="*", default=None)
    p.add_argument("--wandb-notes",     type=str, default=None)
    p.add_argument("--purpose",         type=str, default=None)
    p.add_argument("--experiment-id",   type=int, default=None)
    p.add_argument("--node",            type=str, default=None)
    # Perf escape hatch.
    p.add_argument("--no-compile",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_epoch_factorvae(
    model, device, loader, opt_vae, opt_disc, criterion,
    gamma: float, *, amp: bool,
):
    """One epoch of alternating FactorVAE / discriminator updates.

    Each batch is split in half: x1 → VAE update, x2 → discriminator update.
    Returns per-sample averages of all tracked quantities.
    """
    model.train()
    sums = {"recon": 0.0, "kl": 0.0, "tc": 0.0, "disc": 0.0,
            "d_real": 0.0, "d_fake": 0.0}
    n_vae = 0
    n_disc = 0
    ce = nn.CrossEntropyLoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        B = x.shape[0]
        if B < 2:
            continue
        x1, x2 = x[: B // 2], x[B // 2:]
        b1, b2 = x1.shape[0], x2.shape[0]

        # ---------------- VAE update on x1 ----------------
        opt_vae.zero_grad(set_to_none=True)
        with _autocast_ctx(device, amp):
            x_hat, mu, logvar, z = model(x1, return_z=True)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b1
        recon_loss = criterion(x_hat.float(), x1.float())
        # Discriminator forward in fp32; CE/BCE doesn't support bfloat16.
        d_logits   = model.discriminator(z.float())
        # Density ratio log(D/(1-D)) as a difference of class logits.
        tc_loss    = (d_logits[:, 0] - d_logits[:, 1]).mean()
        loss = recon_loss + kl_loss + gamma * tc_loss
        loss.backward()
        opt_vae.step()

        # --------------- Discriminator update on x2 -------
        opt_disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, _, _, z2 = model(x2, return_z=True)
        z2_real = z2.detach().float()
        z2_perm = permute_dims(z2_real)

        d_real = model.discriminator(z2_real)
        d_fake = model.discriminator(z2_perm)
        # Class 0 = "real (joint q(z))", Class 1 = "fake (product of marginals)".
        targets_real = torch.zeros(b2, dtype=torch.long, device=device)
        targets_fake = torch.ones (b2, dtype=torch.long, device=device)
        d_loss = 0.5 * (ce(d_real, targets_real) + ce(d_fake, targets_fake))
        d_loss.backward()
        opt_disc.step()

        # --- Track per-sample sums ---
        sums["recon"]  += recon_loss.item() * b1
        sums["kl"]     += kl_loss.item()    * b1
        sums["tc"]     += tc_loss.item()    * b1
        sums["disc"]   += d_loss.item()     * b2
        # Discriminator accuracy: P(class 0) for real, P(class 1) for fake.
        with torch.no_grad():
            d_real_p = d_real.softmax(-1)[:, 0].mean().item()
            d_fake_p = d_fake.softmax(-1)[:, 1].mean().item()
        sums["d_real"] += d_real_p * b2
        sums["d_fake"] += d_fake_p * b2

        n_vae  += b1
        n_disc += b2

    return {
        "recon":  sums["recon"]  / max(n_vae, 1),
        "kl":     sums["kl"]     / max(n_vae, 1),
        "tc":     sums["tc"]     / max(n_vae, 1),
        "disc":   sums["disc"]   / max(n_disc, 1),
        "d_real": sums["d_real"] / max(n_disc, 1),
        "d_fake": sums["d_fake"] / max(n_disc, 1),
    }


def val_epoch_factorvae(model, device, loader, criterion, *, amp: bool):
    """Validation forward pass: recon, KL, and a TC estimate from the
    discriminator (no parameter updates). Reuses standard 3-tuple forward.
    """
    model.eval()
    total_recon = total_kl = total_tc = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            with _autocast_ctx(device, amp):
                x_hat, mu, logvar, z = model(x, return_z=True)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
            recon = criterion(x_hat.float(), x.float())
            d_logits = model.discriminator(z.float())
            tc = (d_logits[:, 0] - d_logits[:, 1]).mean()
            total_recon += recon.item() * B
            total_kl    += kl.item()    * B
            total_tc    += tc.item()    * B
            total_samples += B
    return {
        "recon": total_recon / total_samples,
        "kl":    total_kl    / total_samples,
        "tc":    total_tc    / total_samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    m_cfg   = cfg["model"]
    t_cfg   = cfg["training"]
    d_cfg   = cfg["data"]
    log_cfg = cfg["logging"]
    rt_cfg  = (cfg.get("runtime") or {}).get(args.runtime, {}) if args.runtime else {}

    # --- Apply CLI overrides (CLI > runtime overlay > YAML defaults) ---
    if args.latent_dim is not None: m_cfg["latent_dim"] = args.latent_dim
    if args.gamma      is not None: t_cfg["gamma"]      = args.gamma
    if args.disc_lr    is not None: t_cfg["disc_lr"]    = args.disc_lr
    if args.seed       is not None: t_cfg["seed"]       = args.seed
    if args.epochs     is not None: t_cfg["epochs"]     = args.epochs
    if args.batch_size is not None: t_cfg["batch_size"] = args.batch_size
    elif "batch_size" in rt_cfg:    t_cfg["batch_size"] = rt_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else rt_cfg.get("num_workers", 4)

    # --- Device + perf knobs ---
    dev_info = probe_device()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = device.type == "cuda"

    seed_everything(t_cfg["seed"])

    # --- Data ---
    dataset = load_dsprites(args.data_dir)
    heldout_idx = None
    if args.split == "iid":
        train_idx, val_idx, _ = make_iid_split(
            dataset,
            train_frac=d_cfg["train_frac"],
            val_frac=d_cfg["val_frac"],
            seed=t_cfg["seed"],
        )
        print(f"[data] split=iid  train={len(train_idx)}  val={len(val_idx)}")
    elif args.split == "correlated":
        train_idx, val_idx, _ = make_correlated_split(
            dataset,
            factor_a=args.corr_factor_a,
            factor_b=args.corr_factor_b,
            correlation=args.corr_direction,
            train_frac=d_cfg["train_frac"],
            seed=t_cfg["seed"],
        )
        print(f"[data] split=correlated({args.corr_factor_a},{args.corr_factor_b},"
              f"{args.corr_direction})  train={len(train_idx)}  val={len(val_idx)}")
    elif args.split == "heldout":
        train_idx, val_idx, _, heldout_idx = make_heldout_pair_split(
            dataset,
            factor_a=args.heldout_factor_a,
            factor_b=args.heldout_factor_b,
            held_a_vals=args.heldout_a_vals,
            held_b_vals=args.heldout_b_vals,
            seed=t_cfg["seed"],
        )
        print(f"[data] split=heldout  train={len(train_idx)}  val={len(val_idx)}  heldout={len(heldout_idx)}")
    else:
        raise ValueError(f"unknown split: {args.split}")
    train_ds = DSpritesDataset(dataset, train_idx)
    val_ds   = DSpritesDataset(dataset, val_idx)
    loader_kwargs = dict(
        batch_size=t_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # --- Model ---
    img_size = torch.Size(m_cfg["img_size"])
    model = FactorVAE(
        latent_dim=m_cfg["latent_dim"],
        img_size=img_size,
        disc_hidden_dim=t_cfg.get("disc_hidden_dim", 1000),
        disc_num_layers=t_cfg.get("disc_num_layers", 6),
    ).to(device)

    # Two optimisers: VAE branch (encoder + decoder) and discriminator.
    opt_vae = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=t_cfg["lr"],
        weight_decay=t_cfg.get("weight_decay", 0.0),
        betas=(0.9, 0.999),
        fused=(device.type == "cuda"),
    )
    opt_disc = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=t_cfg["disc_lr"],
        betas=(0.5, 0.9),
        fused=(device.type == "cuda"),
    )

    def criterion(x_hat, x):
        return F.binary_cross_entropy(x_hat.float(), x.float(), reduction="sum") / x.shape[0]

    # Compile only the encoder/decoder branch — adversarial dynamics make
    # discriminator-side compile fragile; benefit there is small anyway.
    if not args.no_compile and device.type == "cuda":
        try:
            model.encoder = torch.compile(model.encoder)
            model.decoder = torch.compile(model.decoder)
            print("[perf] torch.compile enabled (encoder + decoder; discriminator eager)")
        except Exception as e:
            print(f"[perf] torch.compile skipped: {e}")
    else:
        print("[perf] torch.compile disabled")

    # --- W&B ---
    flat_cfg = {
        "latent_dim":      m_cfg["latent_dim"],
        "img_size":        str(img_size),
        "epochs":          t_cfg["epochs"],
        "batch_size":      t_cfg["batch_size"],
        "lr":              t_cfg["lr"],
        "disc_lr":         t_cfg["disc_lr"],
        "weight_decay":    t_cfg.get("weight_decay", 0.0),
        "gamma":           t_cfg["gamma"],
        "disc_hidden_dim": t_cfg.get("disc_hidden_dim", 1000),
        "disc_num_layers": t_cfg.get("disc_num_layers", 6),
        "model_type":      "factor_vae",
        "recon_loss":      "bce_sum_per_sample",
        "seed":            t_cfg["seed"],
        "train_frac":      d_cfg["train_frac"],
        "val_frac":        d_cfg["val_frac"],
        "num_workers":     num_workers,
        "amp_dtype":       "bfloat16" if amp_enabled else "fp32",
        # Dataset split metadata.
        "split":             args.split,
        "corr_factor_a":     args.corr_factor_a if args.split == "correlated" else None,
        "corr_factor_b":     args.corr_factor_b if args.split == "correlated" else None,
        "corr_direction":    args.corr_direction if args.split == "correlated" else None,
        "heldout_factor_a":  args.heldout_factor_a if args.split == "heldout" else None,
        "heldout_factor_b":  args.heldout_factor_b if args.split == "heldout" else None,
        "heldout_a_vals":    list(args.heldout_a_vals) if args.split == "heldout" else None,
        "heldout_b_vals":    list(args.heldout_b_vals) if args.split == "heldout" else None,
        "experiment_id":   args.experiment_id,
        "purpose":         args.purpose,
        "sweep_name":      args.wandb_group,
        "node":            args.node,
        "runtime":         args.runtime,
        "host":            dev_info.get("host"),
        "gpu_name":        dev_info.get("gpu_name"),
        "gpu_sm":          dev_info.get("gpu_sm"),
        "vram_gb":         dev_info.get("vram_gb"),
        "cuda_visible_devices": dev_info.get("cuda_visible_devices"),
    }
    run = wandb.init(
        project=log_cfg["wandb_project"],
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        config=flat_cfg,
    )

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pt"

    best_val_loss = float("inf")
    gamma = t_cfg["gamma"]

    for epoch in tqdm(range(1, t_cfg["epochs"] + 1), desc="Epochs"):
        train = train_epoch_factorvae(
            model, device, train_loader, opt_vae, opt_disc, criterion,
            gamma, amp=amp_enabled,
        )
        val   = val_epoch_factorvae(
            model, device, val_loader, criterion, amp=amp_enabled,
        )
        val_loss = val["recon"] + val["kl"] + gamma * val["tc"]

        log = {
            "epoch":               epoch,
            # Existing keys (kept identical to train_vae.py so dashboards work).
            "train/recon_loss":    train["recon"],
            "train/kl_loss":       train["kl"],
            "train/total_loss":    train["recon"] + train["kl"] + gamma * train["tc"],
            "val/recon_loss":      val["recon"],
            "val/kl_loss":         val["kl"],
            "val/total_loss":      val_loss,
            # New FactorVAE-specific keys.
            "train/tc_loss":       train["tc"],
            "train/disc_loss":     train["disc"],
            "train/disc_acc_real": train["d_real"],
            "train/disc_acc_fake": train["d_fake"],
            "val/tc_loss":         val["tc"],
        }

        if epoch % log_cfg["log_interval"] == 0 or epoch == 1:
            log["viz/reconstructions"] = make_recon_grid(
                model, val_loader, device, n=log_cfg["n_viz"]
            )
            manifold = make_pca_manifold(
                model, val_loader, device, n_samples=log_cfg["pca_samples"]
            )
            if manifold is not None:
                log["viz/pca_manifold"] = manifold

        wandb.log(log, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dict = {
                "epoch":                     epoch,
                "model_state_dict":          model.state_dict(),
                "encoder_state_dict":        model.encoder.state_dict(),
                "decoder_state_dict":        model.decoder.state_dict(),
                "discriminator_state_dict":  model.discriminator.state_dict(),
                "optimizer_vae_state_dict":  opt_vae.state_dict(),
                "optimizer_disc_state_dict": opt_disc.state_dict(),
                "val_loss":                  val_loss,
                "config":                    flat_cfg,
            }
            if heldout_idx is not None:
                ckpt_dict["heldout_idx"] = heldout_idx.tolist()
            torch.save(ckpt_dict, best_ckpt)

    # --- Final checkpoint + artifact ---
    final_ckpt = out_dir / "final.pt"
    final_dict = {
        "epoch":                    t_cfg["epochs"],
        "model_state_dict":         model.state_dict(),
        "encoder_state_dict":       model.encoder.state_dict(),
        "decoder_state_dict":       model.decoder.state_dict(),
        "discriminator_state_dict": model.discriminator.state_dict(),
        "config":                   flat_cfg,
    }
    if heldout_idx is not None:
        final_dict["heldout_idx"] = heldout_idx.tolist()
    torch.save(final_dict, final_ckpt)

    artifact = wandb.Artifact(name="factor-vae-checkpoint", type="model")
    artifact.add_file(str(final_ckpt))
    if best_ckpt.exists():
        artifact.add_file(str(best_ckpt))
    run.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
