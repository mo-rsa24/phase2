#!/usr/bin/env python3
"""
Plot 2-D posterior Gaussian ellipses and the PoE fused posterior (Plan 01 — T5).

Three-panel figure (one column per θ).  For each θ:
  - Expert 1  q(z | x_a),  x_a = R(θ)·(1, 0)  [observing c₁=1, c₂=0]
  - Expert 2  q(z | x_b),  x_b = R(θ)·(0, 1)  [observing c₁=0, c₂=1]
  - PoE posterior  Λ_PoE = Λ₁ + Λ₂ − Λ_prior
  - True joint  q(z | x_c),  x_c = R(θ)·(1, 1)  [ground-truth c₁=c₂=1]

Reference points are constructed in OBSERVATION space using the correct R(θ)
rotation, so they represent the same underlying factors at every angle.
"""
import sys
import os
_HERE       = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT  = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

import argparse
from pathlib import Path
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import torch
import wandb

from src.models.linear_vae import LinearVAE
from src.metrics.poe import compute_poe


THETAS      = [0.0, np.pi / 8, np.pi / 4]
THETA_NAMES = ["θ = 0", "θ = π/8", "θ = π/4"]
THETA_FSTRS = ["0.0",   "0.3927",  "0.7854"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Three-panel posterior geometry figure for Plan 01 T5."
    )
    p.add_argument("--checkpoints-dir", type=str,
                   default="checkpoints/gaussian",
                   help="Directory containing linear_vae_theta*_seed*.pt files.")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out-dir", type=str, default="figures")
    p.add_argument("--wandb-project", type=str, default="disentangled-vae-poe")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _draw_ellipse(ax, mean, cov, color, alpha_outer=0.18, alpha_inner=0.40, lw=1.5):
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 1e-10)
    # eigh returns eigenvalues ascending; largest eigenvector = vecs[:,1]
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for n_std, alpha in [(2, alpha_outer), (1, alpha_inner)]:
        ell = Ellipse(
            xy    = mean,
            width = 2 * n_std * np.sqrt(vals[1]),
            height= 2 * n_std * np.sqrt(vals[0]),
            angle = angle,
            facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=lw,
        )
        ax.add_patch(ell)
    ax.plot(*mean, "x", color=color, markersize=7, markeredgewidth=2, zorder=5)


def load_model(ckpt_dir: Path, theta_str: str, seed: int) -> tuple:
    pattern = str(ckpt_dir / f"linear_vae_theta{theta_str}_seed{seed}.pt")
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern}.\n"
            "Run fit_linear_vae.py first."
        )
    ckpt  = torch.load(matches[0], map_location="cpu")
    cfg   = ckpt["config"]
    model = LinearVAE(latent_dim=cfg["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def encode_point(model, x_np: np.ndarray):
    """Encode a single 2-D observation; return (mu, cov) as numpy arrays."""
    with torch.no_grad():
        mu, lv = model.encode(torch.tensor(x_np[None, :], dtype=torch.float32))
    mu_np  = mu.squeeze().numpy()
    cov_np = np.diag(lv.exp().squeeze().numpy())
    return mu_np, cov_np


def main():
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(_SCOPE_ROOT) / args.checkpoints_dir

    COLORS = {
        "expert1":    "steelblue",
        "expert2":    "darkorange",
        "poe":        "crimson",
        "true_joint": "seagreen",
        "prior":      "gray",
    }
    LABELS = {
        "expert1":    "Expert 1  q(z|x_a)  c=(1,0)",
        "expert2":    "Expert 2  q(z|x_b)  c=(0,1)",
        "poe":        "PoE  Λ₁+Λ₂−Λ_prior",
        "true_joint": "True joint  q(z|x_c)  c=(1,1)",
        "prior":      "Prior  N(0, I)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Posterior geometry in latent space — LinearVAE trained at each θ\n"
        "Ellipses show 1σ and 2σ contours",
        fontsize=12, y=1.02,
    )

    for ax, theta, name, tfstr in zip(axes, THETAS, THETA_NAMES, THETA_FSTRS):
        model, cfg = load_model(ckpt_dir, tfstr, args.seed)
        d = cfg["latent_dim"]
        R = rotation(theta)

        # Reference observations in x-space
        x_a = R @ np.array([1.0, 0.0])   # c = (1, 0)
        x_b = R @ np.array([0.0, 1.0])   # c = (0, 1)
        x_c = R @ np.array([1.0, 1.0])   # c = (1, 1)  — true joint

        mu_a, cov_a = encode_point(model, x_a)
        mu_b, cov_b = encode_point(model, x_b)
        mu_c, cov_c = encode_point(model, x_c)

        # PoE
        lambda_a     = np.linalg.inv(cov_a)
        lambda_b     = np.linalg.inv(cov_b)
        lambda_prior = np.eye(d)
        lambda_poe, mu_poe = compute_poe(lambda_a, mu_a, lambda_b, mu_b, lambda_prior)
        cov_poe = np.linalg.inv(lambda_poe)

        # Prior
        mu_prior  = np.zeros(d)
        cov_prior = np.eye(d)

        # Draw
        _draw_ellipse(ax, mu_prior,  cov_prior, COLORS["prior"])
        _draw_ellipse(ax, mu_a,      cov_a,     COLORS["expert1"])
        _draw_ellipse(ax, mu_b,      cov_b,     COLORS["expert2"])
        _draw_ellipse(ax, mu_c,      cov_c,     COLORS["true_joint"], lw=2.5)
        _draw_ellipse(ax, mu_poe,    cov_poe,   COLORS["poe"],        lw=2.5)

        # PoE error vs true joint
        err = float(np.sum((mu_poe - mu_c) ** 2))
        ax.annotate(
            f"‖μ_PoE − μ_true‖² = {err:.4f}",
            xy=(0.04, 0.04), xycoords="axes fraction", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.axvline(0, color="k", lw=0.4, alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("z₀  (latent dim 0)", fontsize=9)
        ax.set_ylabel("z₁  (latent dim 1)", fontsize=9)
        ax.set_title(name, fontsize=11)

    # Shared legend on first axis
    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor=COLORS[k], label=LABELS[k])
            for k in ["expert1", "expert2", "poe", "true_joint", "prior"]
        ],
        fontsize=8, loc="upper right",
    )

    fig.tight_layout()
    png_path = out_dir / "stage1_posterior_geometry.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    # Per-theta PNGs for backward compat
    for theta, tfstr in zip(THETAS, THETA_FSTRS):
        src = out_dir / "stage1_posterior_geometry.png"
        dst = out_dir / f"stage1_posterior_geometry_theta{tfstr}.png"
        import shutil
        if not dst.exists():
            shutil.copy(src, dst)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"posterior_geometry_seed{args.seed}",
            tags=["stage1"],
        )
        wandb.log({"viz/posterior_geometry": wandb.Image(str(png_path))})
        wandb.finish()


if __name__ == "__main__":
    main()
