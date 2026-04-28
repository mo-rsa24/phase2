#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import io
import socket
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
import yaml
import wandb

from src.datasets.dsprites import (
    DSpritesDataset,
    FACTOR_NAMES,
    load_dsprites,
    make_iid_split,
)
from src.datasets.correlated_dsprites import (
    make_correlated_split,
    make_heldout_pair_split,
)
from src.models.vae import VAE


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on dSprites")
    p.add_argument("--config",          type=str, default="configs/vae.yaml")
    p.add_argument("--data-dir",        type=str, default="data")
    p.add_argument("--out-dir",         type=str, default="checkpoints/vae")
    # Experiment overrides (CLI > YAML).
    p.add_argument("--latent-dim",      type=int,   default=None)
    p.add_argument("--beta",            type=float, default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--num-workers",     type=int,   default=None)
    # Dataset split selection (Phase 2b/2e).
    p.add_argument("--split",           type=str, default="iid",
                   choices=["iid", "correlated", "heldout"],
                   help="Which dSprites split to train on.")
    p.add_argument("--corr-factor-a",   type=str, default="scale",
                   help="First correlated factor (when --split=correlated).")
    p.add_argument("--corr-factor-b",   type=str, default="orientation",
                   help="Second correlated factor (when --split=correlated).")
    p.add_argument("--corr-direction",  type=str, default="positive",
                   choices=["positive", "negative"],
                   help="Direction of injected correlation (when --split=correlated).")
    p.add_argument("--heldout-factor-a", type=str, default="shape",
                   help="First factor of held-out cells (when --split=heldout).")
    p.add_argument("--heldout-factor-b", type=str, default="scale",
                   help="Second factor of held-out cells (when --split=heldout).")
    p.add_argument("--heldout-a-vals",   type=int, nargs="*", default=[2],
                   help="Held-out values for factor A (when --split=heldout).")
    p.add_argument("--heldout-b-vals",   type=int, nargs="*", default=[4, 5],
                   help="Held-out values for factor B (when --split=heldout).")
    # Runtime overlay key in configs/vae.yaml -> runtime.{key}.
    p.add_argument("--runtime",         type=str,   default=None,
                   help="Runtime overlay key (e.g. 'hippo', 'cluster48').")
    # W&B metadata.
    p.add_argument("--wandb-run-name",  type=str, default=None)
    p.add_argument("--wandb-group",     type=str, default=None)
    p.add_argument("--wandb-tags",      type=str, nargs="*", default=None)
    p.add_argument("--wandb-notes",     type=str, default=None)
    p.add_argument("--purpose",         type=str, default=None)
    p.add_argument("--experiment-id",   type=int, default=None)
    p.add_argument("--node",            type=str, default=None,
                   help="Node label logged with the run (hippo, mscluster106, ...).")
    # Perf escape hatch.
    p.add_argument("--no-compile",      action="store_true",
                   help="Disable torch.compile (use if compilation breaks on a node).")
    return p.parse_args()


def probe_device() -> dict:
    """Print device info; abort if installed torch wheel can't run on the GPU."""
    info = {
        "host":                  socket.gethostname(),
        "torch":                 torch.__version__,
        "cuda_visible_devices":  os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if not torch.cuda.is_available():
        info["device"] = "cpu"
        print(f"[device] cpu  host={info['host']}  torch={info['torch']}")
        return info

    props = torch.cuda.get_device_properties(0)
    sm = f"sm_{props.major}{props.minor}"
    arch_list = torch.cuda.get_arch_list()
    info.update({
        "device":     "cuda",
        "gpu_name":   props.name,
        "gpu_sm":     sm,
        "vram_gb":    round(props.total_memory / 1e9, 1),
        "arch_list":  arch_list,
    })
    print(f"[device] cuda  gpu={props.name}  sm={sm}  vram={info['vram_gb']}GB  "
          f"torch={info['torch']}  CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}")
    print(f"[device] torch arch_list={arch_list}")

    if sm not in arch_list:
        # Don't abort if PTX JIT might cover us (sm_80 PTX runs on >=sm_80) — but the
        # symptom we hit on the 5090 was a hard "no kernel image is available" error,
        # which only appears at first kernel launch. Fail fast with an actionable msg.
        sys.stderr.write(
            f"\nFATAL: installed torch supports {arch_list} but this GPU is {sm}.\n"
            f"  GPU:  {props.name}\n"
            f"  Fix:  reinstall torch with a CUDA wheel that ships {sm} kernels.\n"
            f"        e.g. pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision\n"
        )
        sys.exit(1)
    return info


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_recon_grid(vae: VAE, loader, device, n: int = 8) -> wandb.Image:
    """Two-row image strip: original on top, reconstruction below."""
    vae.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    with torch.no_grad():
        x_hat, _, _ = vae(x)
    x     = x.cpu().numpy()      # (n, 1, 64, 64)
    x_hat = x_hat.cpu().numpy()

    row_orig  = np.concatenate([x[i, 0]     for i in range(n)], axis=1)
    row_recon = np.concatenate([x_hat[i, 0] for i in range(n)], axis=1)

    gap = np.ones((4, row_orig.shape[1]), dtype=np.float32)
    grid = np.concatenate([row_orig, gap, row_recon], axis=0)
    grid_uint8 = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
    return wandb.Image(grid_uint8, caption="Top: original  |  Bottom: reconstruction")


def make_pca_manifold(
    vae: VAE, loader, device, n_samples: int = 5000
) -> wandb.Image:
    """6-panel PCA scatter of latent mu, one panel per generative factor."""
    vae.eval()
    all_mu, all_latents = [], []
    collected = 0
    with torch.no_grad():
        for x, latents in loader:
            if collected >= n_samples:
                break
            remaining = n_samples - collected
            x_batch = x[:remaining].to(device)
            _, mu, _ = vae(x_batch)
            all_mu.append(mu.cpu().numpy())
            all_latents.append(latents[:remaining].numpy())
            collected += x_batch.shape[0]

    all_mu      = np.concatenate(all_mu,      axis=0)  # (N, latent_dim)
    all_latents = np.concatenate(all_latents, axis=0)  # (N, 6)

    if all_mu.shape[1] < 2:
        return None

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_mu)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, name in enumerate(FACTOR_NAMES):
        classes = all_latents[:, i].astype(float)
        n_cls   = int(classes.max()) + 1
        cmap    = "tab20" if n_cls > 10 else "tab10"
        sc = axes[i].scatter(
            coords[:, 0], coords[:, 1],
            c=classes, cmap=cmap, s=4, alpha=0.5, rasterized=True
        )
        plt.colorbar(sc, ax=axes[i], fraction=0.03, pad=0.04)
        axes[i].set_title(f"Colored by: {name}  ({n_cls} classes)", fontsize=11)
        axes[i].set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=9)
        axes[i].set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=9)
        axes[i].tick_params(labelsize=7)

    fig.suptitle(
        f"Latent PCA Manifold  (N={len(all_mu)}, latent_dim={all_mu.shape[1]})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = wandb.Image(
        Image.open(buf),
        caption="PCA of latent μ vectors, colored by ground-truth factors",
    )
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def _autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_epoch(vae, device, loader, optimizer, criterion, beta, *, amp: bool):
    vae.train()
    total_recon = total_kl = 0.0
    total_samples = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, amp):
            x_hat, mu, logvar = vae(x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        recon_loss = criterion(x_hat.float(), x.float())
        loss       = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()
        total_recon  += recon_loss.item() * batch_size
        total_kl     += kl_loss.item() * batch_size
        total_samples += batch_size
    return total_recon / total_samples, total_kl / total_samples


def val_epoch(vae, device, loader, criterion, beta, *, amp: bool):
    vae.eval()
    total_recon = total_kl = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.shape[0]
            with _autocast_ctx(device, amp):
                x_hat, mu, logvar = vae(x)
                kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            recon = criterion(x_hat.float(), x.float())
            total_recon  += recon.item() * batch_size
            total_kl     += kl.item() * batch_size
            total_samples += batch_size
    return total_recon / total_samples, total_kl / total_samples


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
    if args.latent_dim  is not None: m_cfg["latent_dim"]  = args.latent_dim
    if args.beta        is not None: t_cfg["beta"]        = args.beta
    if args.seed        is not None: t_cfg["seed"]        = args.seed
    if args.epochs      is not None: t_cfg["epochs"]      = args.epochs
    if args.batch_size  is not None: t_cfg["batch_size"]  = args.batch_size
    elif "batch_size"  in rt_cfg:   t_cfg["batch_size"]  = rt_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else rt_cfg.get("num_workers", 4)

    # --- Device probe and perf knobs ---
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
    heldout_idx = None  # populated only when --split=heldout
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
        print(f"[data] split=heldout({args.heldout_factor_a}={args.heldout_a_vals},"
              f"{args.heldout_factor_b}={args.heldout_b_vals})  "
              f"train={len(train_idx)}  val={len(val_idx)}  heldout={len(heldout_idx)}")
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
    vae = VAE(latent_dim=m_cfg["latent_dim"], img_size=img_size).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
        fused=(device.type == "cuda"),
    )
    # dSprites pixels are binary and the decoder ends in Sigmoid, so use the
    # Bernoulli negative log-likelihood per sample. A global mean MSE makes the
    # reconstruction term about 4096x smaller than the KL term on 64x64 images.
    def criterion(x_hat, x):
        return F.binary_cross_entropy(x_hat.float(), x.float(), reduction="sum") / x.shape[0]

    if not args.no_compile and device.type == "cuda":
        try:
            vae = torch.compile(vae)
            print("[perf] torch.compile enabled")
        except Exception as e:
            print(f"[perf] torch.compile skipped: {e}")
    else:
        print("[perf] torch.compile disabled")

    # --- wandb ---
    flat_cfg = {
        "latent_dim":   m_cfg["latent_dim"],
        "img_size":     str(img_size),
        "epochs":       t_cfg["epochs"],
        "batch_size":   t_cfg["batch_size"],
        "lr":           t_cfg["lr"],
        "weight_decay": t_cfg["weight_decay"],
        "beta":         t_cfg["beta"],
        "recon_loss":   "bce_sum_per_sample",
        "seed":         t_cfg["seed"],
        "train_frac":   d_cfg["train_frac"],
        "val_frac":     d_cfg["val_frac"],
        "num_workers":  num_workers,
        "amp_dtype":    "bfloat16" if amp_enabled else "fp32",
        # Dataset split metadata.
        "split":             args.split,
        "corr_factor_a":     args.corr_factor_a if args.split == "correlated" else None,
        "corr_factor_b":     args.corr_factor_b if args.split == "correlated" else None,
        "corr_direction":    args.corr_direction if args.split == "correlated" else None,
        "heldout_factor_a":  args.heldout_factor_a if args.split == "heldout" else None,
        "heldout_factor_b":  args.heldout_factor_b if args.split == "heldout" else None,
        "heldout_a_vals":    list(args.heldout_a_vals) if args.split == "heldout" else None,
        "heldout_b_vals":    list(args.heldout_b_vals) if args.split == "heldout" else None,
        "experiment_id": args.experiment_id,
        "purpose":       args.purpose,
        "sweep_name":    args.wandb_group,
        "node":          args.node,
        "runtime":       args.runtime,
        "host":          dev_info.get("host"),
        "gpu_name":      dev_info.get("gpu_name"),
        "gpu_sm":        dev_info.get("gpu_sm"),
        "vram_gb":       dev_info.get("vram_gb"),
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
    beta = t_cfg["beta"]

    for epoch in tqdm(range(1, t_cfg["epochs"] + 1), desc="Epochs"):
        train_recon, train_kl = train_epoch(
            vae, device, train_loader, optimizer, criterion, beta, amp=amp_enabled
        )
        val_recon, val_kl = val_epoch(
            vae, device, val_loader, criterion, beta, amp=amp_enabled
        )
        val_loss = val_recon + beta * val_kl

        log = {
            "epoch":             epoch,
            "train/recon_loss":  train_recon,
            "train/kl_loss":     train_kl,
            "train/total_loss":  train_recon + beta * train_kl,
            "val/recon_loss":    val_recon,
            "val/kl_loss":       val_kl,
            "val/total_loss":    val_loss,
        }

        if epoch % log_cfg["log_interval"] == 0 or epoch == 1:
            log["viz/reconstructions"] = make_recon_grid(
                vae, val_loader, device, n=log_cfg["n_viz"]
            )
            manifold = make_pca_manifold(
                vae, val_loader, device, n_samples=log_cfg["pca_samples"]
            )
            if manifold is not None:
                log["viz/pca_manifold"] = manifold

        wandb.log(log, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dict = {
                "epoch":                epoch,
                "model_state_dict":     vae.state_dict(),
                "encoder_state_dict":   vae.encoder.state_dict(),
                "decoder_state_dict":   vae.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "config":               flat_cfg,
            }
            if heldout_idx is not None:
                ckpt_dict["heldout_idx"] = heldout_idx.tolist()
            torch.save(ckpt_dict, best_ckpt)

    # --- Final checkpoint + artifact ---
    final_ckpt = out_dir / "final.pt"
    final_dict = {
        "epoch":              t_cfg["epochs"],
        "model_state_dict":   vae.state_dict(),
        "encoder_state_dict": vae.encoder.state_dict(),
        "decoder_state_dict": vae.decoder.state_dict(),
        "config":             flat_cfg,
    }
    if heldout_idx is not None:
        final_dict["heldout_idx"] = heldout_idx.tolist()
    torch.save(final_dict, final_ckpt)

    artifact = wandb.Artifact(name="vae-checkpoint", type="model")
    artifact.add_file(str(final_ckpt))
    if best_ckpt.exists():
        artifact.add_file(str(best_ckpt))
    run.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
