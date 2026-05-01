#!/usr/bin/env python3
"""Train FactorVAE (Kim & Mnih, 2018) on dSprites.

Mirrors `scripts/train_vae.py` 1:1 in structure, wandb logging keys, and
checkpoint format. The difference is the alternating update:

  1. VAE step on the first half of each batch — recon + KL + γ·TC.
  2. Discriminator step on the second half — distinguish q(z) from
     ∏_i q(z_i) (the latter via permute_dims).

Shared scaffolding lives in `src/utils/train_runtime.py` and `src/utils/viz.py`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.models.factor_vae import FactorVAE, permute_dims
from src.utils.train_runtime import (
    apply_common_overrides,
    autocast_ctx,
    base_flat_config,
    build_data_loaders,
    init_wandb,
    load_config,
    save_checkpoint,
    seed_everything,
    setup_device,
)
from src.utils.viz import make_pca_manifold, make_recon_grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train FactorVAE on dSprites")
    p.add_argument("--config",          type=str, default="configs/factor_vae.yml")
    p.add_argument("--data-dir",        type=str, default="data")
    p.add_argument("--out-dir",         type=str, default="checkpoints/factor_vae")
    # Experiment overrides (CLI > runtime overlay > YAML).
    p.add_argument("--latent-dim",      type=int,   default=None)
    p.add_argument("--gamma",           type=float, default=None,
                   help="TC weight γ (FactorVAE-specific). Overrides training.gamma.")
    p.add_argument("--disc-lr",         type=float, default=None,
                   help="Discriminator learning rate. Overrides training.disc_lr.")
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--num-workers",     type=int,   default=None)
    # Dataset split selection (Phase 2b/2e).
    p.add_argument("--split",           type=str, default="iid",
                   choices=["iid", "correlated", "heldout"])
    p.add_argument("--corr-factor-a",   type=str, default="scale")
    p.add_argument("--corr-factor-b",   type=str, default="orientation")
    p.add_argument("--corr-direction",  type=str, default="positive",
                   choices=["positive", "negative"])
    p.add_argument("--heldout-factor-a", type=str, default="shape")
    p.add_argument("--heldout-factor-b", type=str, default="scale")
    p.add_argument("--heldout-a-vals",   type=int, nargs="*", default=[2])
    p.add_argument("--heldout-b-vals",   type=int, nargs="*", default=[4, 5])
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
    p.add_argument("--no-compile",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_epoch_factorvae(
    model, device, loader, opt_vae, opt_disc, criterion,
    gamma: float, *, amp: bool,
):
    """One epoch of alternating FactorVAE / discriminator updates.

    Each batch is split in half: x1 → VAE update, x2 → discriminator update.
    Returns per-sample averages of all tracked quantities.
    """
    model.train()
    sums = {"recon": 0.0, "kl": 0.0, "tc": 0.0, "disc": 0.0,
            "d_real": 0.0, "d_fake": 0.0}
    n_vae = 0
    n_disc = 0
    ce = nn.CrossEntropyLoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        B = x.shape[0]
        if B < 2:
            continue
        x1, x2 = x[: B // 2], x[B // 2:]
        b1, b2 = x1.shape[0], x2.shape[0]

        # ---------------- VAE update on x1 ----------------
        opt_vae.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            x_hat, mu, logvar, z = model(x1, return_z=True)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b1
        recon_loss = criterion(x_hat.float(), x1.float())
        d_logits   = model.discriminator(z.float())
        tc_loss    = (d_logits[:, 0] - d_logits[:, 1]).mean()
        loss = recon_loss + kl_loss + gamma * tc_loss
        loss.backward()
        opt_vae.step()

        # --------------- Discriminator update on x2 -------
        opt_disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, _, _, z2 = model(x2, return_z=True)
        z2_real = z2.detach().float()
        z2_perm = permute_dims(z2_real)

        d_real = model.discriminator(z2_real)
        d_fake = model.discriminator(z2_perm)
        targets_real = torch.zeros(b2, dtype=torch.long, device=device)
        targets_fake = torch.ones (b2, dtype=torch.long, device=device)
        d_loss = 0.5 * (ce(d_real, targets_real) + ce(d_fake, targets_fake))
        d_loss.backward()
        opt_disc.step()

        sums["recon"]  += recon_loss.item() * b1
        sums["kl"]     += kl_loss.item()    * b1
        sums["tc"]     += tc_loss.item()    * b1
        sums["disc"]   += d_loss.item()     * b2
        with torch.no_grad():
            d_real_p = d_real.softmax(-1)[:, 0].mean().item()
            d_fake_p = d_fake.softmax(-1)[:, 1].mean().item()
        sums["d_real"] += d_real_p * b2
        sums["d_fake"] += d_fake_p * b2

        n_vae  += b1
        n_disc += b2

    return {
        "recon":  sums["recon"]  / max(n_vae, 1),
        "kl":     sums["kl"]     / max(n_vae, 1),
        "tc":     sums["tc"]     / max(n_vae, 1),
        "disc":   sums["disc"]   / max(n_disc, 1),
        "d_real": sums["d_real"] / max(n_disc, 1),
        "d_fake": sums["d_fake"] / max(n_disc, 1),
    }


def val_epoch_factorvae(model, device, loader, criterion, *, amp: bool):
    """Validation: recon, KL, and a TC estimate from the discriminator."""
    model.eval()
    total_recon = total_kl = total_tc = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            with autocast_ctx(device, amp):
                x_hat, mu, logvar, z = model(x, return_z=True)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
            recon = criterion(x_hat.float(), x.float())
            d_logits = model.discriminator(z.float())
            tc = (d_logits[:, 0] - d_logits[:, 1]).mean()
            total_recon += recon.item() * B
            total_kl    += kl.item()    * B
            total_tc    += tc.item()    * B
            total_samples += B
    return {
        "recon": total_recon / total_samples,
        "kl":    total_kl    / total_samples,
        "tc":    total_tc    / total_samples,
    }


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

    num_workers = apply_common_overrides(args, m_cfg, t_cfg, rt_cfg)
    if args.gamma   is not None: t_cfg["gamma"]   = args.gamma
    if args.disc_lr is not None: t_cfg["disc_lr"] = args.disc_lr

    device, amp_enabled, dev_info = setup_device()
    seed_everything(t_cfg["seed"])

    train_loader, val_loader, heldout_idx = build_data_loaders(
        args, d_cfg, t_cfg, device, num_workers
    )

    # --- Model ---
    img_size = torch.Size(m_cfg["img_size"])
    model = FactorVAE(
        latent_dim=m_cfg["latent_dim"],
        img_size=img_size,
        disc_hidden_dim=t_cfg.get("disc_hidden_dim", 1000),
        disc_num_layers=t_cfg.get("disc_num_layers", 6),
    ).to(device)

    # Two optimisers: VAE branch (encoder + decoder) and discriminator.
    opt_vae = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=t_cfg["lr"],
        weight_decay=t_cfg.get("weight_decay", 0.0),
        betas=(0.9, 0.999),
        fused=(device.type == "cuda"),
    )
    opt_disc = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=t_cfg["disc_lr"],
        betas=(0.5, 0.9),
        fused=(device.type == "cuda"),
    )

    def criterion(x_hat, x):
        return F.binary_cross_entropy(x_hat.float(), x.float(), reduction="sum") / x.shape[0]

    # Compile only the encoder/decoder branch — adversarial dynamics make
    # discriminator-side compile fragile; benefit there is small anyway.
    if not args.no_compile and device.type == "cuda":
        try:
            model.encoder = torch.compile(model.encoder)
            model.decoder = torch.compile(model.decoder)
            print("[perf] torch.compile enabled (encoder + decoder; discriminator eager)")
        except Exception as e:
            print(f"[perf] torch.compile skipped: {e}")
    else:
        print("[perf] torch.compile disabled")

    # --- W&B ---
    flat_cfg = base_flat_config(
        args, m_cfg, t_cfg, d_cfg, dev_info, num_workers, amp_enabled,
        img_size=img_size,
    )
    # FactorVAE-specific keys.
    flat_cfg.update({
        "disc_lr":         t_cfg["disc_lr"],
        "gamma":           t_cfg["gamma"],
        "disc_hidden_dim": t_cfg.get("disc_hidden_dim", 1000),
        "disc_num_layers": t_cfg.get("disc_num_layers", 6),
        "model_type":      "factor_vae",
    })
    run = init_wandb(log_cfg, args, flat_cfg)

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = out_dir / "best.pt"
    final_ckpt = out_dir / "final.pt"

    best_val_loss = float("inf")
    gamma = t_cfg["gamma"]

    for epoch in tqdm(range(1, t_cfg["epochs"] + 1), desc="Epochs"):
        train = train_epoch_factorvae(
            model, device, train_loader, opt_vae, opt_disc, criterion,
            gamma, amp=amp_enabled,
        )
        val   = val_epoch_factorvae(
            model, device, val_loader, criterion, amp=amp_enabled,
        )
        val_loss = val["recon"] + val["kl"] + gamma * val["tc"]

        log = {
            "epoch":               epoch,
            "train/recon_loss":    train["recon"],
            "train/kl_loss":       train["kl"],
            "train/total_loss":    train["recon"] + train["kl"] + gamma * train["tc"],
            "val/recon_loss":      val["recon"],
            "val/kl_loss":         val["kl"],
            "val/total_loss":      val_loss,
            "train/tc_loss":       train["tc"],
            "train/disc_loss":     train["disc"],
            "train/disc_acc_real": train["d_real"],
            "train/disc_acc_fake": train["d_fake"],
            "val/tc_loss":         val["tc"],
        }

        if epoch % log_cfg["log_interval"] == 0 or epoch == 1:
            log["viz/reconstructions"] = make_recon_grid(
                model, val_loader, device, n=log_cfg["n_viz"]
            )
            manifold = make_pca_manifold(
                model, val_loader, device, n_samples=log_cfg["pca_samples"]
            )
            if manifold is not None:
                log["viz/pca_manifold"] = manifold

        wandb.log(log, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = {
                "epoch":                     epoch,
                "model_state_dict":          model.state_dict(),
                "encoder_state_dict":        model.encoder.state_dict(),
                "decoder_state_dict":        model.decoder.state_dict(),
                "discriminator_state_dict":  model.discriminator.state_dict(),
                "optimizer_vae_state_dict":  opt_vae.state_dict(),
                "optimizer_disc_state_dict": opt_disc.state_dict(),
                "val_loss":                  val_loss,
                "config":                    flat_cfg,
            }
            if heldout_idx is not None:
                payload["heldout_idx"] = heldout_idx.tolist()
            save_checkpoint(best_ckpt, payload)

    # --- Final checkpoint + artifact ---
    final_payload = {
        "epoch":                    t_cfg["epochs"],
        "model_state_dict":         model.state_dict(),
        "encoder_state_dict":       model.encoder.state_dict(),
        "decoder_state_dict":       model.decoder.state_dict(),
        "discriminator_state_dict": model.discriminator.state_dict(),
        "config":                   flat_cfg,
    }
    if heldout_idx is not None:
        final_payload["heldout_idx"] = heldout_idx.tolist()
    save_checkpoint(final_ckpt, final_payload)

    artifact = wandb.Artifact(name="factor-vae-checkpoint", type="model")
    artifact.add_file(str(final_ckpt))
    if best_ckpt.exists():
        artifact.add_file(str(best_ckpt))
    run.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
