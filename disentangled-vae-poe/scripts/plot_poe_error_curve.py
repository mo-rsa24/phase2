#!/usr/bin/env python3
"""
Sweep θ and plot PoE reconstruction error vs rotation angle (Plan 01 — T6).

For each θ:
  1. Load the trained LinearVAE checkpoint.
  2. Build unit-reference posteriors:
       (mu_bar1, lambda_bar1) = encode_reference_set(c₁ ≈ 1.0)
       (mu_bar2, lambda_bar2) = encode_reference_set(c₂ ≈ 1.0)
     Linear encoder means mu_bar(alpha) = alpha * mu_bar(1.0), so the
     unit reference is computed once and scaled for the whole grid.
  3. For each (α, β) in a 10×10 grid over [−0.9, 0.9]:
       mu_poe = PoE(α·mu_bar1, λ_bar1,  β·mu_bar2, λ_bar2)
       error  = |μ_PoE[0] − α|² + |μ_PoE[1] − β|²
  4. Average error over the grid → mean_error(θ).
  5. Fit c·|sin(2θ)| and compute R².
  6. Save PNG and log to W&B.
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
import numpy as np
import torch
import wandb

from src.data.gaussian_rotation import make_rotation_data
from src.models.linear_vae import LinearVAE
from src.metrics.poe import compute_poe, poe_error, encode_reference_set


THETAS          = [0.0, np.pi / 8, np.pi / 4]
THETA_FSTRS     = ["0.0", "0.3927", "0.7854"]
ALPHA_BETA_GRID = np.linspace(-0.9, 0.9, 10)


def parse_args():
    p = argparse.ArgumentParser(description="Plot PoE error vs rotation angle (T6)")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints/gaussian")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--out-dir",         type=str, default="figures")
    p.add_argument("--wandb-project",   type=str, default="disentangled-vae-poe")
    p.add_argument("--no-wandb",        action="store_true")
    return p.parse_args()


def load_model(ckpt_dir: Path, theta_str: str, seed: int):
    pattern = str(ckpt_dir / f"linear_vae_theta{theta_str}_seed{seed}.pt")
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"Checkpoint not found: {pattern}\n"
            "Run fit_linear_vae.py for each theta first."
        )
    ckpt  = torch.load(matches[0], map_location="cpu")
    cfg   = ckpt["config"]
    model = LinearVAE(latent_dim=cfg["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main():
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(_SCOPE_ROOT) / args.checkpoints_dir

    mean_errors: list[float] = []

    for theta, tfstr in zip(THETAS, THETA_FSTRS):
        print(f"\n── θ = {theta:.4f} ({tfstr}) ──────────────────")

        model, cfg = load_model(ckpt_dir, tfstr, args.seed)
        d          = cfg["latent_dim"]

        x_np, c1_np, c2_np = make_rotation_data(
            n=20_000, theta=theta, seed=cfg["seed"]
        )
        data_c = np.stack([c1_np, c2_np], axis=1)

        # Unit-value reference posteriors (scale linearly for the grid)
        mu_bar1, lambda_bar1 = encode_reference_set(
            model, x_np, data_c, factor_id=0, factor_val=1.0, n=500
        )
        mu_bar2, lambda_bar2 = encode_reference_set(
            model, x_np, data_c, factor_id=1, factor_val=1.0, n=500
        )
        print(f"  mu_bar1={mu_bar1.round(4)},  λ_bar1={lambda_bar1.round(3)}")
        print(f"  mu_bar2={mu_bar2.round(4)},  λ_bar2={lambda_bar2.round(3)}")

        lambda1      = np.diag(lambda_bar1)
        lambda2      = np.diag(lambda_bar2)
        lambda_prior = np.eye(d)

        errors: list[float] = []
        for alpha in ALPHA_BETA_GRID:
            for beta in ALPHA_BETA_GRID:
                mu1 = alpha * mu_bar1
                mu2 = beta  * mu_bar2
                _, mu_poe = compute_poe(lambda1, mu1, lambda2, mu2, lambda_prior)
                errors.append(poe_error(mu_poe, alpha, beta))

        mean_err = float(np.mean(errors))
        mean_errors.append(mean_err)
        print(f"  mean PoE error = {mean_err:.6f}")

    # Fit c·|sin(2θ)|
    thetas_arr = np.array(THETAS)
    X      = np.abs(np.sin(2 * thetas_arr)).reshape(-1, 1)
    y      = np.array(mean_errors)
    c_fit  = float(np.linalg.lstsq(X, y, rcond=None)[0][0])
    y_pred = c_fit * np.abs(np.sin(2 * thetas_arr))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    print(f"\nFit:  c = {c_fit:.4f},  R² = {r2:.4f}")
    if not np.isnan(r2) and r2 >= 0.95:
        print("✓  R² ≥ 0.95 — |sin(2θ)| scaling confirmed.")
    else:
        print("✗  R² < 0.95 (may improve with more training or additional θ values).")

    # Plot
    theta_dense = np.linspace(0, np.pi / 2, 300)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(thetas_arr, mean_errors, s=90, zorder=5, color="steelblue",
               label="measured  E[(μ_PoE − c)²]")
    ax.plot(theta_dense, c_fit * np.abs(np.sin(2 * theta_dense)),
            color="crimson", lw=2,
            label=f"fitted  c·|sin(2θ)|,  c = {c_fit:.3f}")
    ax.set_xlabel("θ  (rotation angle, radians)", fontsize=11)
    ax.set_ylabel("mean PoE error  E[|μ_PoE − c|²]", fontsize=11)
    ax.set_title("PoE reconstruction error vs posterior entanglement", fontsize=12)
    ax.set_xticks(thetas_arr)
    ax.set_xticklabels(["0", "π/8", "π/4"])
    ax.annotate(f"R² = {r2:.4f}", xy=(0.62, 0.12), xycoords="axes fraction",
                fontsize=11, color="crimson",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = out_dir / "stage1_poe_error_vs_theta.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"poe_error_curve_seed{args.seed}",
            tags=["stage1"],
        )
        wandb.log({
            "viz/poe_error_vs_theta": wandb.Image(str(png_path)),
            "metrics/r2_sin2theta":   r2,
            "metrics/c_fit":          c_fit,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
