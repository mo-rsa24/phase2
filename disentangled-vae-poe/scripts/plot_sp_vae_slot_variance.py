#!/usr/bin/env python3
"""SP-VAE per-slot per-dimension std histograms (Plan 04, T4).

Encodes up to 5000 val samples, computes per-dim std for z_c, z_d1, z_d2,
and plots 3 side-by-side histograms with a red dashed collapse threshold at 0.05.
Saves figures/sp_vae_slot_variance.png.
"""
import argparse
import os
import sys
from pathlib import Path

_HERE       = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT  = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.sp_vae.model import SPVAE
from src.datasets.shapes3d import Shapes3D

COLLAPSE_THRESHOLD = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples",   type=int,   default=5000)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--num-workers", type=int,   default=0)
    p.add_argument("--no-wandb",    action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    model = SPVAE(z_c_dim=cfg["z_c_dim"], z_d_dim=cfg["z_d_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds     = Shapes3D(split="val", seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    mu_cs, mu_d1s, mu_d2s = [], [], []
    n_collected = 0
    for x, _ in val_loader:
        fwd = model(x.to(device))
        mu_cs.append(fwd[1].cpu())
        mu_d1s.append(fwd[4].cpu())
        mu_d2s.append(fwd[7].cpu())
        n_collected += x.shape[0]
        if n_collected >= args.n_samples:
            break

    mu_c  = torch.cat(mu_cs)[:args.n_samples].numpy()
    mu_d1 = torch.cat(mu_d1s)[:args.n_samples].numpy()
    mu_d2 = torch.cat(mu_d2s)[:args.n_samples].numpy()

    # Per-dim std across the sample dimension
    std_c  = mu_c.std(axis=0)
    std_d1 = mu_d1.std(axis=0)
    std_d2 = mu_d2.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, std_vals, slot_name in zip(axes, [std_c, std_d1, std_d2],
                                        ["z_c", "z_d1", "z_d2"]):
        ax.bar(range(len(std_vals)), std_vals, color="steelblue", alpha=0.8)
        ax.axhline(COLLAPSE_THRESHOLD, color="red", linestyle="--",
                   linewidth=1.2, label=f"collapse ({COLLAPSE_THRESHOLD})")
        ax.set_xlabel("Dimension index")
        ax.set_ylabel("Std across val set")
        ax.set_title(f"{slot_name} per-dim std")
        ax.legend(fontsize=8)
        ax.set_xticks(range(len(std_vals)))
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"SP-VAE slot variance  (n={n_collected})", fontsize=10)
    fig.tight_layout()

    out_dir = Path(_SCOPE_ROOT) / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sp_vae_slot_variance.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Collapse check
    for name, std_vals in [("z_d1", std_d1), ("z_d2", std_d2)]:
        for i, s in enumerate(std_vals):
            if s < COLLAPSE_THRESHOLD:
                print(f"WARNING: {name}[{i}] std={s:.4f} < {COLLAPSE_THRESHOLD} (collapsed)")

    if not args.no_wandb:
        import wandb
        wandb.init(project="disentangled-vae-poe",
                   name="spvae_slot_variance", tags=["sp-vae", "eval"], resume="allow")
        wandb.log({"viz/sp_vae_slot_variance": wandb.Image(str(out_path))})
        for i, s in enumerate(std_d1):
            wandb.log({f"slot_std/d1_{i}": float(s)})
        for i, s in enumerate(std_d2):
            wandb.log({f"slot_std/d2_{i}": float(s)})
        wandb.finish()


if __name__ == "__main__":
    main()
