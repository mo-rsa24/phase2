#!/usr/bin/env python3
"""SP-VAE latent traversal grid (Plan 04, T3).

Traverses z_d1[0], z_d1[1], z_d2[0], z_d2[1] independently in 8 steps
from -3σ to +3σ while holding all other latents at their encoded mean.
Saves figures/sp_vae_traversals.png.
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

from src.models.sp_vae.model import SPVAE
from src.datasets.shapes3d import Shapes3D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-steps",    type=int,   default=8)
    p.add_argument("--z-range",    type=float, default=3.0)
    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
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

    val_ds = Shapes3D(split="val", seed=args.seed)
    x0, _  = val_ds[0]

    fwd = model(x0.unsqueeze(0).to(device))
    # (x_hat, mu_c, lv_c, z_c, mu_d1, lv_d1, z_d1, mu_d2, lv_d2, z_d2)
    mu_c  = fwd[1].squeeze(0)
    mu_d1 = fwd[4].squeeze(0)
    lv_d1 = fwd[5].squeeze(0)
    mu_d2 = fwd[7].squeeze(0)
    lv_d2 = fwd[8].squeeze(0)

    std_d1 = (0.5 * lv_d1).exp()
    std_d2 = (0.5 * lv_d2).exp()

    steps = torch.linspace(-args.z_range, args.z_range, args.n_steps, device=device)

    row_specs = [
        ("z_d1[0]", mu_d1, std_d1, "d1", 0),
        ("z_d1[1]", mu_d1, std_d1, "d1", 1),
        ("z_d2[0]", mu_d2, std_d2, "d2", 0),
        ("z_d2[1]", mu_d2, std_d2, "d2", 1),
    ]

    n_rows = len(row_specs)
    fig, axes = plt.subplots(n_rows, args.n_steps,
                             figsize=(args.n_steps * 1.3, n_rows * 1.3))

    for row_idx, (label, mu_slot, std_slot, slot_name, dim_idx) in enumerate(row_specs):
        z_c_r  = mu_c.clone()
        z_d1_r = mu_d1.clone()
        z_d2_r = mu_d2.clone()
        for col_idx, s in enumerate(steps):
            val = mu_slot[dim_idx] + s * std_slot[dim_idx]
            if slot_name == "d1":
                z_d1_r = mu_d1.clone(); z_d1_r[dim_idx] = val
            else:
                z_d2_r = mu_d2.clone(); z_d2_r[dim_idx] = val
            img = model.decode(
                z_c_r.unsqueeze(0),
                z_d1_r.unsqueeze(0),
                z_d2_r.unsqueeze(0),
            ).squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=8, rotation=0,
                                    labelpad=40, va="center")

    for col_idx, s in enumerate(steps):
        axes[0, col_idx].set_title(f"{s.item():+.1f}σ", fontsize=7)

    fig.suptitle("SP-VAE latent traversals  (rows: z_d1[0], z_d1[1], z_d2[0], z_d2[1])",
                 fontsize=9)
    fig.tight_layout()

    out_dir = Path(_SCOPE_ROOT) / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sp_vae_traversals.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    if not args.no_wandb:
        import wandb
        wandb.init(project="disentangled-vae-poe",
                   name="spvae_traversals", tags=["sp-vae", "eval"], resume="allow")
        wandb.log({"viz/sp_vae_traversals": wandb.Image(str(out_path))})
        wandb.finish()


if __name__ == "__main__":
    main()
