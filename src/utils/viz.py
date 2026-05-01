"""W&B visualisation helpers shared by all VAE trainers.

`make_recon_grid` and `make_pca_manifold` were previously defined inside
`scripts/train_vae.py` and re-imported by `scripts/train_factorvae.py` via an
importlib spec hack. Both helpers operate on any nn.Module that returns
`(x_hat, mu, ...)` from `forward(x)` (FactorVAE's optional `return_z=True`
path is unaffected because the helpers ignore the trailing tuple element).
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

import wandb

from src.datasets.dsprites import FACTOR_NAMES


def make_recon_grid(model, loader, device, n: int = 8) -> wandb.Image:
    """Two-row image strip: original on top, reconstruction below."""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    with torch.no_grad():
        out = model(x)
    x_hat = out[0]
    x     = x.cpu().numpy()       # (n, 1, 64, 64)
    x_hat = x_hat.cpu().numpy()

    row_orig  = np.concatenate([x[i, 0]     for i in range(n)], axis=1)
    row_recon = np.concatenate([x_hat[i, 0] for i in range(n)], axis=1)

    gap = np.ones((4, row_orig.shape[1]), dtype=np.float32)
    grid = np.concatenate([row_orig, gap, row_recon], axis=0)
    grid_uint8 = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
    return wandb.Image(grid_uint8, caption="Top: original  |  Bottom: reconstruction")


def make_pca_manifold(model, loader, device, n_samples: int = 5000) -> Optional[wandb.Image]:
    """6-panel PCA scatter of latent μ, one panel per generative factor."""
    model.eval()
    all_mu, all_latents = [], []
    collected = 0
    with torch.no_grad():
        for x, latents in loader:
            if collected >= n_samples:
                break
            remaining = n_samples - collected
            x_batch = x[:remaining].to(device)
            out = model(x_batch)
            mu = out[1]
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
