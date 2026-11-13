#!/usr/bin/env python3
"""Pre-training visualization: rotation data structure and PoE composition intuition.

Produces a 4-panel figure that answers two questions before any model is trained:
  1. What structure is in the data — and why can the VAE not find the factors?
  2. What does PoE try to do, and exactly where does it break with an entangled encoder?

Run:
    python scripts/visualize_rotation_intuition.py --out figures/rotation_intuition.png
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

from src.data.gaussian_rotation import make_rotation_data


# ── helpers ──────────────────────────────────────────────────────────────────

def draw_ellipse(ax, mu, cov, n_std=2.0, **kw):
    """Draw a covariance ellipse at `n_std` standard deviations."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_std * np.sqrt(vals[::-1])
    ell = Ellipse(xy=mu, width=w, height=h, angle=angle, **kw)
    ax.add_patch(ell)
    return ell


def poe_gaussian(mu1, lam1, mu2, lam2, lam_prior=None):
    """Analytic Product-of-Experts for two Gaussian experts.

    Λ_poe = Λ1 + Λ2 − Λ_prior
    μ_poe = Λ_poe⁻¹ (Λ1 μ1 + Λ2 μ2)
    """
    if lam_prior is None:
        lam_prior = np.eye(len(mu1))
    lam_poe = lam1 + lam2 - lam_prior
    mu_poe = np.linalg.solve(lam_poe, lam1 @ mu1 + lam2 @ mu2)
    cov_poe = np.linalg.inv(lam_poe)
    return mu_poe, cov_poe


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ── plotting panels ───────────────────────────────────────────────────────────

def panel_data_scatter(ax, x, c1, c2, theta_deg, show_cov=True):
    """Scatter of observed x coloured by c1 and (faintly) c2."""
    sc = ax.scatter(x[:, 0], x[:, 1], c=c1, cmap="RdBu_r", s=6, alpha=0.6,
                    vmin=-1, vmax=1, rasterized=True)
    if show_cov:
        cov = np.cov(x.T)
        draw_ellipse(ax, np.mean(x, axis=0), cov, n_std=1.5,
                     edgecolor="black", facecolor="none", lw=1.5, linestyle="--",
                     label="data 1.5σ ellipse")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"Observed data  θ = {theta_deg}°\n(colour = $c_1$)")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    return sc


def panel_poe_disentangled(ax, alpha=0.7, beta=0.7):
    """PoE composition with a disentangled encoder (θ=0).

    The encoder undoes the rotation so z ≈ c.  The reference-set posterior for
    c1=α (c2 varying) is tight on z1 and loose on z2; vice-versa for c2=β.
    Their PoE product lands very close to the target [α, β].
    """
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)

    # Expert 1: c1=α known, c2 marginalised → tight on z1, loose on z2
    mu1 = np.array([alpha, 0.0])
    lam1 = np.diag([100.0, 4.0])
    cov1 = np.linalg.inv(lam1)

    # Expert 2: c2=β known, c1 marginalised → loose on z1, tight on z2
    mu2 = np.array([0.0, beta])
    lam2 = np.diag([4.0, 100.0])
    cov2 = np.linalg.inv(lam2)

    # PoE
    mu_poe, cov_poe = poe_gaussian(mu1, lam1, mu2, lam2)

    # Target
    target = np.array([alpha, beta])

    # Draw
    draw_ellipse(ax, mu1, cov1, n_std=2, edgecolor="#e07b54", facecolor="#e07b54",
                 alpha=0.15, lw=0)
    draw_ellipse(ax, mu1, cov1, n_std=2, edgecolor="#e07b54", facecolor="none",
                 alpha=0.9, lw=1.8, label=f"Expert 1  $c_1={alpha}$")

    draw_ellipse(ax, mu2, cov2, n_std=2, edgecolor="#5b8dd9", facecolor="#5b8dd9",
                 alpha=0.15, lw=0)
    draw_ellipse(ax, mu2, cov2, n_std=2, edgecolor="#5b8dd9", facecolor="none",
                 alpha=0.9, lw=1.8, label=f"Expert 2  $c_2={beta}$")

    draw_ellipse(ax, mu_poe, cov_poe, n_std=2, edgecolor="#2ca02c", facecolor="#2ca02c",
                 alpha=0.25, lw=0)
    draw_ellipse(ax, mu_poe, cov_poe, n_std=2, edgecolor="#2ca02c", facecolor="none",
                 alpha=0.95, lw=2.0, label="PoE product")

    ax.plot(*target, marker="*", ms=14, color="gold", markeredgecolor="black",
            markeredgewidth=0.8, zorder=10, label=f"Target $[{alpha},{beta}]$")
    ax.plot(*mu_poe, marker="o", ms=8, color="#2ca02c", markeredgecolor="black",
            markeredgewidth=0.8, zorder=10)

    err = float(np.linalg.norm(mu_poe - target))
    ax.annotate(f"error = {err:.3f}", xy=(0.03, 0.04), xycoords="axes fraction",
                fontsize=9, color="#2ca02c",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title("Disentangled encoder  (θ=0°)\nPoE composition")
    ax.legend(fontsize=7.5, loc="upper left")

    # Annotate the structure
    ax.annotate("", xy=(mu1[0]+0.05, 0.55), xytext=(mu1[0]+0.05, -0.55),
                arrowprops=dict(arrowstyle="<->", color="#e07b54", lw=1.2))
    ax.text(mu1[0]+0.12, 0, "loose\n(c₂ unknown)", fontsize=7.5,
            color="#e07b54", va="center")


def panel_poe_entangled(ax, alpha=0.7, beta=0.7, theta=np.pi / 4):
    """PoE composition with an entangled encoder (θ=45°).

    The encoder does NOT undo the rotation; z = R(θ)·c.  Both reference-set
    posteriors are forced to be isotropic (diagonal VAE posterior cannot express
    directional uncertainty in the rotated basis).  PoE averages the two means
    and undershoots the target by roughly ×½.
    """
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)

    R = rotation_matrix(theta)

    # In z-space: z = R(θ)·c.  Reference mean for c1=α: average over c2 → R(θ)·[α, 0]
    mu1 = R @ np.array([alpha, 0.0])
    mu2 = R @ np.array([0.0, beta])

    # Forced isotropic precision — the diagonal VAE cannot express the directional
    # uncertainty that distinguishes c1 from c2 in the rotated basis.
    sigma_iso = 0.10
    lam1 = np.eye(2) / sigma_iso**2
    lam2 = np.eye(2) / sigma_iso**2
    cov1 = sigma_iso**2 * np.eye(2)
    cov2 = sigma_iso**2 * np.eye(2)

    # PoE
    mu_poe, cov_poe = poe_gaussian(mu1, lam1, mu2, lam2)

    # Target in z-space: R(θ)·[α, β]
    target = R @ np.array([alpha, beta])

    # Draw
    draw_ellipse(ax, mu1, cov1, n_std=2, edgecolor="#e07b54", facecolor="#e07b54",
                 alpha=0.15, lw=0)
    draw_ellipse(ax, mu1, cov1, n_std=2, edgecolor="#e07b54", facecolor="none",
                 alpha=0.9, lw=1.8, label=f"Expert 1  $c_1={alpha}$")

    draw_ellipse(ax, mu2, cov2, n_std=2, edgecolor="#5b8dd9", facecolor="#5b8dd9",
                 alpha=0.15, lw=0)
    draw_ellipse(ax, mu2, cov2, n_std=2, edgecolor="#5b8dd9", facecolor="none",
                 alpha=0.9, lw=1.8, label=f"Expert 2  $c_2={beta}$")

    draw_ellipse(ax, mu_poe, cov_poe, n_std=2, edgecolor="#d62728", facecolor="#d62728",
                 alpha=0.25, lw=0)
    draw_ellipse(ax, mu_poe, cov_poe, n_std=2, edgecolor="#d62728", facecolor="none",
                 alpha=0.95, lw=2.0, label="PoE product")

    ax.plot(*target, marker="*", ms=14, color="gold", markeredgecolor="black",
            markeredgewidth=0.8, zorder=10, label=f"Target $R(θ)·[{alpha},{beta}]$")
    ax.plot(*mu_poe, marker="o", ms=8, color="#d62728", markeredgecolor="black",
            markeredgewidth=0.8, zorder=10)

    # Arrow from PoE to target
    ax.annotate("", xy=target, xytext=mu_poe,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                connectionstyle="arc3,rad=0.15"))

    err = float(np.linalg.norm(mu_poe - target))
    ax.annotate(f"error = {err:.3f}", xy=(0.03, 0.04), xycoords="axes fraction",
                fontsize=9, color="#d62728",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title("Entangled encoder  (θ=45°)\nPoE composition — FAILS")
    ax.legend(fontsize=7.5, loc="upper left")

    ax.text(0.97, 0.55, "Both experts are\nisotropic circles:\nno directional\nuncertainty!",
            transform=ax.transAxes, ha="right", va="center", fontsize=8,
            color="#d62728", bbox=dict(boxstyle="round,pad=0.3", fc="#fdd", alpha=0.8))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figures/rotation_intuition.png")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="c1 target value for PoE composition demo")
    parser.add_argument("--beta", type=float, default=0.7,
                        help="c2 target value for PoE composition demo")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x0, c1_0, c2_0 = make_rotation_data(args.n, theta=0.0,        seed=args.seed)
    x45, c1_45, c2_45 = make_rotation_data(args.n, theta=np.pi / 4, seed=args.seed)

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ── Panel A: data at θ=0 ─────────────────────────────────────────────────
    sc_A = panel_data_scatter(ax_A, x0, c1_0, c2_0, theta_deg=0)

    # Draw the true factor axes in data space at theta=0
    for v, label in [([1.2, 0], "$c_1$ axis"), ([0, 1.2], "$c_2$ axis")]:
        ax_A.annotate("", xy=v, xytext=[0, 0],
                      arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax_A.text(v[0]+0.05, v[1]+0.05, label, fontsize=8)

    ax_A.text(0.03, 0.96,
              "Factors align with axes\nVAE can recover $c_1$, $c_2$",
              transform=ax_A.transAxes, va="top", fontsize=8.5,
              color="#1a6e1a",
              bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", alpha=0.9))

    # ── Panel B: data at θ=45° ───────────────────────────────────────────────
    sc_B = panel_data_scatter(ax_B, x45, c1_45, c2_45, theta_deg=45)

    # Draw rotated factor axes
    R45 = rotation_matrix(np.pi / 4)
    for v_fac, label in [([1.2, 0], "$c_1$ axis"), ([0, 1.2], "$c_2$ axis")]:
        v_obs = R45 @ np.array(v_fac)
        ax_B.annotate("", xy=v_obs, xytext=[0, 0],
                      arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax_B.text(v_obs[0]+0.05, v_obs[1]+0.05, label, fontsize=8)

    ax_B.text(0.03, 0.96,
              "Same cloud shape (isotropic!)\nVAE cannot distinguish from θ=0°\n"
              "→ factors end up mixed in z",
              transform=ax_B.transAxes, va="top", fontsize=8.5,
              color="#8b0000",
              bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", alpha=0.9))

    # Add a shared colourbar for the two scatter panels
    cbar = fig.colorbar(sc_B, ax=[ax_A, ax_B], orientation="vertical",
                        fraction=0.03, pad=0.02, label="$c_1$ value")

    # Annotate that both covariance ellipses are identical
    ax_A.text(0.5, -0.13, "↑  data covariance ellipse (dashed)\n"
              "Both panels have the SAME 1.5σ ellipse — VAE sees identical shape",
              transform=ax_A.transAxes, ha="center", fontsize=7.5,
              style="italic", color="gray")

    # ── Panel C: disentangled PoE ────────────────────────────────────────────
    panel_poe_disentangled(ax_C, alpha=args.alpha, beta=args.beta)

    ax_C.text(0.97, 0.04,
              "Axis-aligned ellipses\n(tight on known factor,\nloose on unknown)\n"
              "→ PoE picks c₁ from A\nand c₂ from B correctly",
              transform=ax_C.transAxes, ha="right", va="bottom", fontsize=7.5,
              color="#1a6e1a",
              bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", alpha=0.85))

    # ── Panel D: entangled PoE ───────────────────────────────────────────────
    panel_poe_entangled(ax_D, alpha=args.alpha, beta=args.beta)

    # Explain the amplitude error
    ax_D.text(0.5, -0.14,
              "PoE averages the two means.  At θ=45° the means are R(45°)·[α,0] and R(45°)·[0,β];\n"
              "average = R(45°)·[α/2, β/2] = target / 2.  Error ∝ |sin(2θ)|.",
              transform=ax_D.transAxes, ha="center", fontsize=7.5,
              style="italic", color="#8b0000")

    # ── Global title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Rotation data: what the VAE sees and why PoE breaks\n"
        r"$x = R(\theta)\,[c_1,\,c_2]^\top + \varepsilon,\quad c_i\sim\mathcal{U}[-1,1],\quad"
        r"\varepsilon\sim\mathcal{N}(0,\,0.05\,I)$",
        fontsize=12, y=1.01
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
