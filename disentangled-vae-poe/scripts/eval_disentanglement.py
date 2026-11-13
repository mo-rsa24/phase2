#!/usr/bin/env python3
"""MIG evaluation for SP-VAE and β-VAE baseline (Plan 04, T2).

Computes the Mutual Information Gap (MIG) over all 6 generative factors.
For SP-VAE: uses the concatenation of mu_d1 and mu_d2 (z_c excluded).
For ConvVAE checkpoints: uses the full mu vector.

Usage:
    python scripts/eval_disentanglement.py --checkpoint <path> [--model spvae|convvae]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

_HERE       = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT  = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.shapes3d import Shapes3D, FACTOR_NAMES, FACTOR_SIZES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model",       default="spvae",
                   choices=["spvae", "convvae"])
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-wandb",    action="store_true")
    p.add_argument("--n-bins",      type=int, default=20,
                   help="Bins for discretising continuous latents")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def encode_spvae(checkpoint_path, val_loader, device):
    from src.models.sp_vae.model import SPVAE
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    model = SPVAE(z_c_dim=cfg["z_c_dim"], z_d_dim=cfg["z_d_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mu_d1s, mu_d2s, lbls = [], [], []
    for x, labels in val_loader:
        fwd = model(x.to(device))
        mu_d1s.append(fwd[4].cpu())
        mu_d2s.append(fwd[7].cpu())
        lbls.append(labels)
    Z = torch.cat([torch.cat(mu_d1s), torch.cat(mu_d2s)], dim=0)
    # interleave so cols go d1[0], d1[1], d2[0], d2[1]
    mu_d1 = torch.cat(mu_d1s).numpy()
    mu_d2 = torch.cat(mu_d2s).numpy()
    Z_np  = np.concatenate([mu_d1, mu_d2], axis=1)
    L_np  = torch.cat(lbls).numpy()
    return Z_np, L_np


@torch.no_grad()
def encode_convvae(checkpoint_path, val_loader, device):
    from src.models.conv_vae import ConvVAE
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    model = ConvVAE(latent_dim=cfg["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mus, lbls = [], []
    for x, labels in val_loader:
        mu, _ = model.encode(x.to(device))
        mus.append(mu.cpu())
        lbls.append(labels)
    return torch.cat(mus).numpy(), torch.cat(lbls).numpy()


def _mi_discrete(z_col: np.ndarray, f_col: np.ndarray, n_bins: int) -> float:
    """Mutual information between a continuous latent dim (binned) and a factor."""
    # bin the latent
    edges = np.percentile(z_col, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-8; edges[-1] += 1e-8
    z_bin = np.digitize(z_col, edges) - 1
    n_z, n_f = n_bins, int(f_col.max()) + 1
    # joint
    joint = np.zeros((n_z, n_f))
    for zv, fv in zip(z_bin, f_col.astype(int)):
        joint[zv, fv] += 1
    joint /= joint.sum()
    p_z = joint.sum(axis=1, keepdims=True) + 1e-10
    p_f = joint.sum(axis=0, keepdims=True) + 1e-10
    mi = (joint * np.log((joint + 1e-10) / (p_z * p_f))).sum()
    return float(max(mi, 0.0))


def compute_mig(Z: np.ndarray, L: np.ndarray, n_bins: int) -> dict:
    """Returns MIG per factor and mean MIG."""
    n_factors = L.shape[1]
    n_dims    = Z.shape[1]
    mig_per_factor = {}
    for j in range(n_factors):
        f = L[:, j]
        h_f = np.log(len(np.unique(f)) + 1e-10)
        mis = [_mi_discrete(Z[:, k], f, n_bins) for k in range(n_dims)]
        mis.sort(reverse=True)
        mig_j = (mis[0] - mis[1]) / h_f if len(mis) >= 2 else 0.0
        mig_per_factor[FACTOR_NAMES[j]] = float(mig_j)
    mig_per_factor["mean"] = float(np.mean(list(mig_per_factor.values())))
    return mig_per_factor


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds     = Shapes3D(split="val", seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print(f"Encoding with {args.model} …")
    if args.model == "spvae":
        Z, L = encode_spvae(args.checkpoint, val_loader, device)
    else:
        Z, L = encode_convvae(args.checkpoint, val_loader, device)

    print(f"Computing MIG (Z shape: {Z.shape}, {args.n_bins} bins) …")
    mig = compute_mig(Z, L, n_bins=args.n_bins)

    print(f"\nMIG results ({args.model}):")
    for k, v in mig.items():
        print(f"  {k:<16}  {v:.4f}")

    # Load baseline from CSV if present
    csv_path = Path(_SCOPE_ROOT) / "results" / "shapes3d" / "spvae_disentanglement.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing_rows = list(csv.reader(f))

    # Append MIG entries tagged with model name
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not existing_rows:
            writer.writerow(["probe", "accuracy"])
        for k, v in mig.items():
            writer.writerow([f"mig_{k}_{args.model}", f"{v:.6f}"])
    print(f"\nAppended MIG entries to {csv_path}")

    # Comparison: load beta-vae results if present
    bvae_path = Path(_SCOPE_ROOT) / "checkpoints" / "shapes3d" / "vae" / "beta4" / "best.pt"
    if bvae_path.exists() and args.model == "spvae":
        print("\nComputing MIG for β-VAE (beta4) baseline …")
        Z_b, L_b = encode_convvae(str(bvae_path), val_loader, device)
        mig_b = compute_mig(Z_b, L_b, n_bins=args.n_bins)
        print(f"\n{'Factor':<16}  {'SP-VAE':>8}  {'β-VAE':>8}  {'Δ':>8}")
        print("-" * 44)
        for k in mig:
            v_sp = mig[k]
            v_bv = mig_b.get(k, float("nan"))
            delta = v_sp - v_bv
            print(f"  {k:<14}  {v_sp:>8.4f}  {v_bv:>8.4f}  {delta:>+8.4f}")
    else:
        print("\n(No β-VAE baseline found at checkpoints/shapes3d/vae/beta4/best.pt — skipping comparison)")

    if not args.no_wandb:
        import wandb
        wandb.init(project="disentangled-vae-poe",
                   name=f"mig_{args.model}", tags=["sp-vae", "eval"], resume="allow")
        wandb.log({f"mig/{k}_{args.model}": v for k, v in mig.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
