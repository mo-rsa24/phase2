#!/usr/bin/env python3
"""Train a baseline / β-VAE on dSprites.

This script is the canonical *unsupervised* trainer: it covers Exps 1–4 of
the disentanglement sweep (β=1, β=4, z=4, z=20). Targeted weak-supervision
runs live in `scripts/train_supervised_vae.py`; FactorVAE in
`scripts/train_factorvae.py`.

Shared scaffolding (device probe, dataloaders, wandb init, viz helpers,
checkpoint save) lives in `src/utils/train_runtime.py` and `src/utils/viz.py`
so all three trainers share one implementation of those concerns.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.models.vae import VAE
from src.utils.train_runtime import (
    apply_common_overrides,
    autocast_ctx,
    base_flat_config,
    build_data_loaders,
    init_wandb,
    load_config,
    maybe_compile,
    save_checkpoint,
    seed_everything,
    setup_device,
)
from src.utils.viz import make_pca_manifold, make_recon_grid


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on dSprites")
    p.add_argument("--config",          type=str, default="configs/vae.yaml")
    p.add_argument("--data-dir",        type=str, default="data")
    p.add_argument("--out-dir",         type=str, default="checkpoints/vae")
    # Experiment overrides (CLI > YAML).
    p.add_argument("--latent-dim",      type=int,   default=None)
    p.add_argument("--beta",            type=float, default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--num-workers",     type=int,   default=None)
    # Dataset split selection (Phase 2b/2e).
    p.add_argument("--split",           type=str, default="iid",
                   choices=["iid", "correlated", "heldout"],
                   help="Which dSprites split to train on.")
    p.add_argument("--corr-factor-a",   type=str, default="scale")
    p.add_argument("--corr-factor-b",   type=str, default="orientation")
    p.add_argument("--corr-direction",  type=str, default="positive",
                   choices=["positive", "negative"])
    p.add_argument("--heldout-factor-a", type=str, default="shape")
    p.add_argument("--heldout-factor-b", type=str, default="scale")
    p.add_argument("--heldout-a-vals",   type=int, nargs="*", default=[2])
    p.add_argument("--heldout-b-vals",   type=int, nargs="*", default=[4, 5])
    # Runtime overlay key in configs/vae.yaml -> runtime.{key}.
    p.add_argument("--runtime",         type=str,   default=None)
    # W&B metadata.
    p.add_argument("--wandb-run-name",  type=str, default=None)
    p.add_argument("--wandb-group",     type=str, default=None)
    p.add_argument("--wandb-tags",      type=str, nargs="*", default=None)
    p.add_argument("--wandb-notes",     type=str, default=None)
    p.add_argument("--purpose",         type=str, default=None)
    p.add_argument("--experiment-id",   type=int, default=None)
    p.add_argument("--node",            type=str, default=None)
    # Perf escape hatch.
    p.add_argument("--no-compile",      action="store_true",
                   help="Disable torch.compile (use if compilation breaks on a node).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_epoch(vae, device, loader, optimizer, criterion, beta, *, amp: bool):
    vae.train()
    total_recon = total_kl = 0.0
    total_samples = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            x_hat, mu, logvar = vae(x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        recon_loss = criterion(x_hat.float(), x.float())
        loss       = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()
        total_recon  += recon_loss.item() * batch_size
        total_kl     += kl_loss.item()    * batch_size
        total_samples += batch_size
    return total_recon / total_samples, total_kl / total_samples


def val_epoch(vae, device, loader, criterion, beta, *, amp: bool):
    vae.eval()
    total_recon = total_kl = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.shape[0]
            with autocast_ctx(device, amp):
                x_hat, mu, logvar = vae(x)
                kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            recon = criterion(x_hat.float(), x.float())
            total_recon  += recon.item() * batch_size
            total_kl     += kl.item()    * batch_size
            total_samples += batch_size
    return total_recon / total_samples, total_kl / total_samples


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
    rt_cfg  = (cfg.get("runtime") or {}).get(args.runtime, {}) if args.runtime else {}

    # CLI > runtime > YAML for the keys every trainer shares.
    num_workers = apply_common_overrides(args, m_cfg, t_cfg, rt_cfg)
    # Objective-specific override (β belongs to this trainer).
    if args.beta is not None: t_cfg["beta"] = args.beta

    device, amp_enabled, dev_info = setup_device()
    seed_everything(t_cfg["seed"])

    train_loader, val_loader, heldout_idx = build_data_loaders(
        args, d_cfg, t_cfg, device, num_workers
    )

    # --- Model ---
    img_size = torch.Size(m_cfg["img_size"])
    vae = VAE(latent_dim=m_cfg["latent_dim"], img_size=img_size).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
        fused=(device.type == "cuda"),
    )
    # dSprites pixels are binary and the decoder ends in Sigmoid → Bernoulli NLL
    # summed over pixels and averaged over the batch.
    def criterion(x_hat, x):
        return F.binary_cross_entropy(x_hat.float(), x.float(), reduction="sum") / x.shape[0]

    vae = maybe_compile(vae, args.no_compile, device)

    # --- W&B ---
    flat_cfg = base_flat_config(
        args, m_cfg, t_cfg, d_cfg, dev_info, num_workers, amp_enabled,
        img_size=img_size,
    )
    flat_cfg["beta"] = t_cfg["beta"]
    run = init_wandb(log_cfg, args, flat_cfg)

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = out_dir / "best.pt"
    final_ckpt = out_dir / "final.pt"

    best_val_loss = float("inf")
    beta = t_cfg["beta"]

    for epoch in tqdm(range(1, t_cfg["epochs"] + 1), desc="Epochs"):
        train_recon, train_kl = train_epoch(
            vae, device, train_loader, optimizer, criterion, beta, amp=amp_enabled
        )
        val_recon, val_kl = val_epoch(
            vae, device, val_loader, criterion, beta, amp=amp_enabled
        )
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
            payload = {
                "epoch":                epoch,
                "model_state_dict":     vae.state_dict(),
                "encoder_state_dict":   vae.encoder.state_dict(),
                "decoder_state_dict":   vae.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "config":               flat_cfg,
            }
            if heldout_idx is not None:
                payload["heldout_idx"] = heldout_idx.tolist()
            save_checkpoint(best_ckpt, payload)

    # --- Final checkpoint + artifact ---
    final_payload = {
        "epoch":              t_cfg["epochs"],
        "model_state_dict":   vae.state_dict(),
        "encoder_state_dict": vae.encoder.state_dict(),
        "decoder_state_dict": vae.decoder.state_dict(),
        "config":             flat_cfg,
    }
    if heldout_idx is not None:
        final_payload["heldout_idx"] = heldout_idx.tolist()
    save_checkpoint(final_ckpt, final_payload)

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
