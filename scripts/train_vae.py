#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
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
from src.models.vae import VAE


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on dSprites")
    p.add_argument("--config",          type=str, default="configs/vae.yaml")
    p.add_argument("--data-dir",        type=str, default="data")
    p.add_argument("--out-dir",         type=str, default="checkpoints/vae")
    p.add_argument("--wandb-run-name",  type=str, default=None)
    return p.parse_args()


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

def train_epoch(vae, device, loader, optimizer, criterion, beta):
    vae.train()
    total_recon = total_kl = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _, _ = vae(x)
        recon_loss = criterion(x_hat, x)
        kl_loss    = vae.encoder.kl / x.shape[0]
        loss       = recon_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()
    n = len(loader)
    return total_recon / n, total_kl / n


def val_epoch(vae, device, loader, criterion, beta):
    vae.eval()
    total_recon = total_kl = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, _, _ = vae(x)
            total_recon += criterion(x_hat, x).item()
            total_kl    += (vae.encoder.kl / x.shape[0]).item()
    n = len(loader)
    return total_recon / n, total_kl / n


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

    seed_everything(t_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    dataset = load_dsprites(args.data_dir)
    train_idx, val_idx, _ = make_iid_split(
        dataset,
        train_frac=d_cfg["train_frac"],
        val_frac=d_cfg["val_frac"],
        seed=t_cfg["seed"],
    )
    train_ds = DSpritesDataset(dataset, train_idx)
    val_ds   = DSpritesDataset(dataset, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # --- Model ---
    img_size = torch.Size(m_cfg["img_size"])
    vae = VAE(latent_dim=m_cfg["latent_dim"], img_size=img_size).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )
    criterion = nn.MSELoss()

    # --- wandb ---
    flat_cfg = {
        "latent_dim":   m_cfg["latent_dim"],
        "img_size":     str(img_size),
        "epochs":       t_cfg["epochs"],
        "batch_size":   t_cfg["batch_size"],
        "lr":           t_cfg["lr"],
        "weight_decay": t_cfg["weight_decay"],
        "beta":         t_cfg["beta"],
        "seed":         t_cfg["seed"],
        "train_frac":   d_cfg["train_frac"],
        "val_frac":     d_cfg["val_frac"],
    }
    run = wandb.init(
        project=log_cfg["wandb_project"],
        name=args.wandb_run_name,
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
            vae, device, train_loader, optimizer, criterion, beta
        )
        val_recon, val_kl = val_epoch(vae, device, val_loader, criterion, beta)
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
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     vae.state_dict(),
                    "encoder_state_dict":   vae.encoder.state_dict(),
                    "decoder_state_dict":   vae.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             val_loss,
                    "config":               flat_cfg,
                },
                best_ckpt,
            )

    # --- Final checkpoint + artifact ---
    final_ckpt = out_dir / "final.pt"
    torch.save(
        {
            "epoch":              t_cfg["epochs"],
            "model_state_dict":   vae.state_dict(),
            "encoder_state_dict": vae.encoder.state_dict(),
            "decoder_state_dict": vae.decoder.state_dict(),
            "config":             flat_cfg,
        },
        final_ckpt,
    )

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
