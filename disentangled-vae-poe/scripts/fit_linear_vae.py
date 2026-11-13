#!/usr/bin/env python3
"""Train a LinearVAE on 2-D Gaussian rotation data (Stage 1 — Plan 01)."""
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)   # Paper2/ — shared src/
sys.path.insert(0, _SCOPE_ROOT)  # disentangled-vae-poe/ — scope src/

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from src.data.gaussian_rotation import make_rotation_data
from src.models.linear_vae import LinearVAE


def parse_args():
    p = argparse.ArgumentParser(description="Fit LinearVAE on rotated Gaussian data")
    p.add_argument("--theta",           type=float, required=True,
                   help="Rotation angle in radians.")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--epochs",          type=int,   default=200)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--latent-dim",      type=int,   default=2)
    p.add_argument("--beta",            type=float, default=1.0,
                   help="KL weight. Use 0.1 to prevent posterior collapse on this "
                        "dataset (data noise σ²=0.05 << decoder-assumed σ²=1).")
    p.add_argument("--recon-sigma",     type=float, default=1.0,
                   help="Assumed decoder noise std. Scales recon loss by "
                        "1/(2σ²). Set to sqrt(0.05)≈0.224 to match data noise.")
    p.add_argument("--wandb-project",   type=str,   default="disentangled-vae-poe")
    return p.parse_args()


def elbo_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    recon_sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO = reconstruction loss / (2σ²) + β·KL divergence."""
    batch_size = x.shape[0]
    recon_scale = 1.0 / (2.0 * recon_sigma ** 2)
    recon = recon_scale * F.mse_loss(x_hat, x, reduction="sum") / batch_size
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    return recon + beta * kl, recon, kl


def train_epoch(
    model: LinearVAE,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    recon_sigma: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = total_recon = total_kl = 0.0
    total_samples = 0
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.shape[0]
        optimizer.zero_grad(set_to_none=True)
        x_hat, mu, log_var = model(x)
        loss, recon, kl = elbo_loss(x_hat, x, mu, log_var, beta=beta, recon_sigma=recon_sigma)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()  * batch_size
        total_recon   += recon.item() * batch_size
        total_kl      += kl.item()   * batch_size
        total_samples += batch_size
    return (
        total_loss    / total_samples,
        total_recon   / total_samples,
        total_kl      / total_samples,
    )


def val_epoch(
    model: LinearVAE,
    device: torch.device,
    loader: DataLoader,
    beta: float,
    recon_sigma: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    total_samples = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.shape[0]
            x_hat, mu, log_var = model(x)
            loss, recon, kl = elbo_loss(x_hat, x, mu, log_var, beta=beta, recon_sigma=recon_sigma)
            total_loss    += loss.item()  * batch_size
            total_recon   += recon.item() * batch_size
            total_kl      += kl.item()   * batch_size
            total_samples += batch_size
    return (
        total_loss    / total_samples,
        total_recon   / total_samples,
        total_kl      / total_samples,
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    x_np, _c1, _c2 = make_rotation_data(n=10_000, theta=args.theta, seed=args.seed)
    x_all = torch.tensor(x_np, dtype=torch.float32)
    n_val = int(len(x_all) * 0.1)
    x_train = x_all[n_val:]
    x_val   = x_all[:n_val]

    train_loader = DataLoader(
        TensorDataset(x_train), batch_size=256, shuffle=True,  pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        TensorDataset(x_val),   batch_size=256, shuffle=False, pin_memory=(device.type == "cuda")
    )

    # --- Model ---
    model     = LinearVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "theta":        args.theta,
        "seed":         args.seed,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "beta":         args.beta,
        "recon_sigma":  args.recon_sigma,
        "latent_dim":   args.latent_dim,
        "n_train":      len(x_train),
        "n_val":        n_val,
        "device":       str(device),
    }

    # --- W&B ---
    run_name = f"linear_vae_theta{args.theta}_seed{args.seed}"
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=["stage1"],
        config=config,
    )

    # --- Output dir ---
    ckpt_dir  = Path(_SCOPE_ROOT) / "checkpoints" / "gaussian"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"linear_vae_theta{args.theta}_seed{args.seed}.pt"

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss, train_recon, train_kl = train_epoch(
            model, device, train_loader, optimizer, args.beta, args.recon_sigma
        )
        val_loss, val_recon, val_kl = val_epoch(
            model, device, val_loader, args.beta, args.recon_sigma
        )

        wandb.log(
            {
                "epoch":            epoch,
                "train/total_loss": train_loss,
                "train/recon_loss": train_recon,
                "train/kl_loss":    train_kl,
                "val/total_loss":   val_loss,
                "val/recon_loss":   val_recon,
                "val/kl_loss":      val_kl,
            },
            step=epoch,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             best_val_loss,
                "config":               config,
            }
            torch.save(payload, ckpt_path)

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
