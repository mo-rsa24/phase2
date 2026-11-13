#!/usr/bin/env python3
"""
EDA visualization for the gaussian rotation dataset.

Produces three figures:
  01_data_scatter.png      — x-space at three rotation angles, coloured by c1 and c2
  02_latent_space.png      — what the ideal VAE encoder recovers
  03_poe_geometry.png      — PoE posterior geometry: how two partial posteriors combine

The actual trained checkpoints have collapsed posteriors (posterior ≈ prior,
std≈1).  To show meaningful geometry we use the closed-form optimal Bayesian
encoder for a linear-Gaussian model:
    q*(z|x) = N(W*x, σ²·I),  W = (20/21)·R(−θ),  σ = 1/√21 ≈ 0.218
derived from  p(x|z) = N(R(θ)·z, 0.05·I)  and  p(z) = N(0,I).
"""

import sys, os
_HERE        = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT  = os.path.dirname(os.path.dirname(_HERE))   # disentangled-vae-poe/
_PROJ_ROOT   = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm

from src.data.gaussian_rotation import make_rotation_data

OUT_DIR = os.path.dirname(_HERE)   # figures/eda/
os.makedirs(OUT_DIR, exist_ok=True)

THETAS      = [0.0, np.pi / 8, np.pi / 4]
THETA_NAMES = ["θ = 0", "θ = π/8", "θ = π/4"]
N           = 2000
SEED        = 42
RNG         = np.random.default_rng(SEED)


# ─── analytic optimal encoder ────────────────────────────────────────────────

def R(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def analytic_encode(x: np.ndarray, theta: float):
    """Return (mu, sigma) for the optimal Bayesian posterior at angle theta.

    q*(z|x) = N(W·x, sigma²·I)
    W      = (20/21)·R(−theta)
    sigma  = 1/sqrt(21)
    """
    W   = (20 / 21) * R(-theta)
    mu  = (W @ x.T).T          # (n, 2)
    sigma = 1.0 / np.sqrt(21)
    return mu, sigma


# ─── helper: draw a 1-sigma / 2-sigma ellipse ────────────────────────────────

def draw_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    e = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(e)
    return e


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — data scatter coloured by c1 (top row) and c2 (bottom row)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle(
    "Data distribution  x = R(θ)·c + ε     (ε ~ N(0, 0.05·I),  c ~ U[−1,1]²)",
    fontsize=13, y=1.01
)

for col, (theta, name) in enumerate(zip(THETAS, THETA_NAMES)):
    x, c1, c2 = make_rotation_data(N, theta=theta, seed=SEED)

    for row, (factor, flabel) in enumerate([(c1, "c₁"), (c2, "c₂")]):
        ax = axes[row, col]
        sc = ax.scatter(x[:, 0], x[:, 1], c=factor, cmap="coolwarm",
                        s=5, alpha=0.6, vmin=-1, vmax=1, rasterized=True)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        if row == 0:
            ax.set_title(name, fontsize=11)
        if col == 0:
            ax.set_ylabel(f"coloured by {flabel}", fontsize=10)
        ax.set_xlabel("x₀"); ax.set_ylabel("x₁") if col == 0 else ax.set_ylabel("")

        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

axes[0, 0].set_title("θ = 0\n(x axes ≈ factor axes)", fontsize=11)
axes[0, 1].set_title("θ = π/8\n(partial mix)", fontsize=11)
axes[0, 2].set_title("θ = π/4\n(x₀, x₁ each mix c₁ and c₂ equally)", fontsize=11)

fig.text(0.02, 0.75, "coloured by c₁", va="center", rotation=90, fontsize=11)
fig.text(0.02, 0.27, "coloured by c₂", va="center", rotation=90, fontsize=11)

fig.tight_layout()
path1 = os.path.join(OUT_DIR, "01_data_scatter.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — what the ideal VAE encoder recovers
# ══════════════════════════════════════════════════════════════════════════════
# Three columns = three θ values.
# Row 0: observation space x coloured by c1 (what the model sees).
# Row 1: latent space z = W·x coloured by c1 (what the model learns).
# Row 2: latent space z coloured by c2 (cross-check disentanglement).

fig, axes = plt.subplots(3, 3, figsize=(13, 12))
fig.suptitle(
    "Optimal VAE encoder  q*(z|x) = N(W·x, σ²·I)   W = (20/21)·R(−θ)\n"
    "The encoder undoes the rotation: z₀ ≈ c₁, z₁ ≈ c₂ at every θ.",
    fontsize=12, y=1.01
)

for col, (theta, name) in enumerate(zip(THETAS, THETA_NAMES)):
    x, c1, c2 = make_rotation_data(N, theta=theta, seed=SEED)
    mu, _ = analytic_encode(x, theta)

    for row, (pts, clr, lab, ttl) in enumerate([
        (x,  c1, "c₁", "observation space x"),
        (mu, c1, "c₁", "latent space z  (coloured by c₁)"),
        (mu, c2, "c₂", "latent space z  (coloured by c₂)"),
    ]):
        ax = axes[row, col]
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=clr, cmap="coolwarm",
                        s=5, alpha=0.6, vmin=-1, vmax=1, rasterized=True)
        lim = 1.8 if row == 0 else 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        if col == 0:
            ax.set_ylabel(ttl, fontsize=9)
        if row == 0:
            ax.set_title(name, fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
path2 = os.path.join(OUT_DIR, "02_latent_space.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PoE posterior geometry
# ══════════════════════════════════════════════════════════════════════════════
# For θ=0 and θ=π/4:
#   Blue cluster / ellipse  = encoded reference set  {c₁ ≈ +1, c₂ ~ U}
#   Red  cluster / ellipse  = encoded reference set  {c₂ ≈ +1, c₁ ~ U}
#   Green ellipse           = PoE combination  (precision-weighted product)
#   Gold star               = true target  (encoding of c₁=1, c₂=1)
#
# The reference-set EMPIRICAL precision captures the uncertainty that matters
# for PoE: tight along the "known" factor, loose along the "unknown" factor.

SHOW_THETAS = [0.0, np.pi / 4]
SHOW_NAMES  = ["θ = 0  (axis-aligned)", "θ = π/4  (maximally mixed)"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    "PoE posterior geometry in latent space\n"
    "Blue: reference set  c₁ ≈ 1  |  Red: reference set  c₂ ≈ 1\n"
    "Green: PoE product   |   ★: true target  (c₁=1, c₂=1)",
    fontsize=11, y=1.02
)

N_REF = 5000

for ax, theta, name in zip(axes, SHOW_THETAS, SHOW_NAMES):
    x, c1, c2 = make_rotation_data(N_REF, theta=theta, seed=SEED)
    mu, sigma_scalar = analytic_encode(x, theta)
    prec_scalar = 1.0 / sigma_scalar**2          # per-sample precision (scalar)

    # Reference set A: c1 in top decile
    maskA = c1 > np.percentile(c1, 90)
    muA   = mu[maskA]                            # (nA, 2)
    # Empirical precision = 1 / Var[z_i | c1 high],  per dimension
    emp_cov_A = np.cov(muA.T)                    # (2,2)  spread from c2 variation
    per_sample_cov = (sigma_scalar**2) * np.eye(2)
    cov_A = emp_cov_A + per_sample_cov            # total posterior covariance
    prec_A = np.linalg.inv(cov_A)                # (2,2)
    mu_bar_A = muA.mean(0)

    # Reference set B: c2 in top decile
    maskB = c2 > np.percentile(c2, 90)
    muB   = mu[maskB]
    emp_cov_B = np.cov(muB.T)
    cov_B = emp_cov_B + per_sample_cov
    prec_B = np.linalg.inv(cov_B)
    mu_bar_B = muB.mean(0)

    # PoE
    lam_prior = np.eye(2)
    lam_poe   = prec_A + prec_B - lam_prior
    cov_poe   = np.linalg.inv(lam_poe)
    mu_poe    = cov_poe @ (prec_A @ mu_bar_A + prec_B @ mu_bar_B)

    # True target: encode x from (c1=1, c2=1)
    x_true = R(theta) @ np.array([1.0, 1.0])
    mu_true, _ = analytic_encode(x_true[None, :], theta)
    mu_true = mu_true[0]

    # ── scatter reference sets (small alpha so ellipses are visible) ──
    ax.scatter(muA[:, 0], muA[:, 1], c="steelblue", s=4, alpha=0.25,
               label="ref-set  c₁ ≈ 1", rasterized=True)
    ax.scatter(muB[:, 0], muB[:, 1], c="tomato",    s=4, alpha=0.25,
               label="ref-set  c₂ ≈ 1", rasterized=True)

    # ── posterior ellipses ──
    draw_ellipse(ax, mu_bar_A, cov_A, n_std=1.5,
                 edgecolor="steelblue", facecolor="steelblue", alpha=0.25, lw=2)
    draw_ellipse(ax, mu_bar_B, cov_B, n_std=1.5,
                 edgecolor="tomato",    facecolor="tomato",    alpha=0.25, lw=2)
    draw_ellipse(ax, mu_poe, cov_poe,  n_std=1.5,
                 edgecolor="seagreen",  facecolor="seagreen",  alpha=0.30, lw=2.5,
                 label="PoE posterior")

    # ── centroids + target ──
    ax.plot(*mu_bar_A, "o", color="steelblue", ms=8, zorder=5)
    ax.plot(*mu_bar_B, "o", color="tomato",    ms=8, zorder=5)
    ax.plot(*mu_poe,   "s", color="seagreen",  ms=9, zorder=6, label="μ_PoE")
    ax.plot(*mu_true,  "*", color="gold",      ms=14, zorder=7,
            markeredgecolor="k", markeredgewidth=0.8, label="target z*(1,1)")

    # ── error annotation ──
    err = float(np.sum((mu_poe - mu_true)**2))
    ax.annotate(f"‖μ_PoE − z*‖² = {err:.4f}",
                xy=(0.05, 0.05), xycoords="axes fraction",
                fontsize=9, color="seagreen",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # axis
    ax.set_xlim(-1.2, 1.4); ax.set_ylim(-0.6, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("z₀  (encodes c₁)", fontsize=10)
    ax.set_ylabel("z₁  (encodes c₂)", fontsize=10)
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
path3 = os.path.join(OUT_DIR, "03_poe_geometry.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path3}")
print("Done.")
