#!/usr/bin/env python3
"""Train a targeted weak-supervision VAE on dSprites.

Differences from `scripts/train_vae.py`:

  * The latent has a fixed semantic partition: z[0]=z_s (scale),
    z[1:3]=z_o (orientation, parameterised as (sin kθ, cos kθ)),
    z[3:]=z_free (unconstrained). See `src/utils/factor_targets.py`.

  * Auxiliary regression losses pin scale and orientation to those slots:
        L_aux = λ_s · MSE(μ[:,0],   s_target)
              + λ_o · MSE(μ[:,1:3], o_target)
    computed against `μ` (not the sampled `z`) so the supervision is
    deterministic across the reparameterisation noise.

  * The KL term uses a per-dim β: by default `β_supervised = 0` so
    z_s and z_o are *labeled regression outputs*, not Gaussian latents,
    and the KL prior does not fight the unit-circle target. Free dims keep
    β = β_free. Set `--beta-supervised > 0` to recover the v1 behaviour
    (KL on every dim, supervision via MSE only) for ablations.

  * Optional `--free-bits F` clamps each *free* dim's KL contribution to
    a floor of F nats. Disabled (F=0) by default.

Validation logs two extra metrics that are robust to circular targets:
  - `val/scale_r2`             — explained-variance R² of normalized scale.
  - `val/orient_angular_err_deg` — mean physical-orientation error in deg,
    modulo each shape's rotational symmetry order.

See `docs/targeted_supervision.md` for theory, derivations, and run commands.
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
from src.utils.factor_targets import (
    Z_ORIENT_IDX,
    Z_SCALE_IDX,
    orient_angular_error_deg,
    orient_target,
    scale_r2,
    scale_target,
)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train targeted-supervision VAE on dSprites")
    p.add_argument("--config",          type=str, default="configs/supervised_vae.yaml")
    p.add_argument("--data-dir",        type=str, default="data")
    p.add_argument("--out-dir",         type=str, default="checkpoints/vae")
    # Experiment overrides (CLI > runtime overlay > YAML).
    p.add_argument("--latent-dim",      type=int,   default=None)
    p.add_argument("--beta",            type=float, default=None,
                   help="KL weight on FREE dims (β-VAE β). Supervised dims use --beta-supervised.")
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--num-workers",     type=int,   default=None)
    # Supervision knobs.
    p.add_argument("--lambda-scale",    type=float, default=None,
                   help="Weight on scale supervision (overrides supervision.lambda_scale).")
    p.add_argument("--lambda-orient",   type=float, default=None,
                   help="Weight on orientation supervision (overrides supervision.lambda_orient).")
    p.add_argument("--beta-supervised", type=float, default=None,
                   help="KL weight on supervised dims (z_s, z_o). 0 = treat as labeled outputs.")
    p.add_argument("--free-bits",       type=float, default=None,
                   help="Floor (nats per free dim) on the KL contribution. 0 disables.")
    # Dataset split selection.
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
    # Runtime overlay.
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
# Loss components
# ---------------------------------------------------------------------------

def _make_beta_per_dim(latent_dim: int, beta_free: float, beta_supervised: float,
                       device: torch.device) -> torch.Tensor:
    """Per-dim β: free dims get β_free, supervised dims (z_s, z_o) get β_supervised."""
    beta = torch.full((latent_dim,), float(beta_free), device=device, dtype=torch.float32)
    beta[Z_SCALE_IDX] = float(beta_supervised)
    beta[Z_ORIENT_IDX[0]:Z_ORIENT_IDX[1] + 1] = float(beta_supervised)
    return beta


def _free_dim_mask(latent_dim: int, device: torch.device) -> torch.Tensor:
    """Boolean mask, True for free dims, False for supervised dims."""
    mask = torch.ones(latent_dim, dtype=torch.bool, device=device)
    mask[Z_SCALE_IDX] = False
    mask[Z_ORIENT_IDX[0]:Z_ORIENT_IDX[1] + 1] = False
    return mask


def _kl_with_per_dim_beta(
    mu: torch.Tensor, logvar: torch.Tensor,
    beta_per_dim: torch.Tensor, free_mask: torch.Tensor, free_bits: float,
) -> torch.Tensor:
    """β-weighted KL against N(0,I) with optional free-bits floor on free dims.

    `kl_per_dim` is summed across the latent (β-weighted) and averaged over the batch,
    matching the normalisation used by the baseline trainer's KL term.
    """
    # Per-dim KL of a diagonal Gaussian against the standard prior.
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, D)
    if free_bits > 0.0:
        # Apply the floor only to free dims, in nats per dim per sample.
        floored = torch.clamp(kl_per_dim, min=float(free_bits))
        kl_per_dim = torch.where(free_mask.unsqueeze(0), floored, kl_per_dim)
    return (kl_per_dim * beta_per_dim).sum(dim=1).mean()


def _aux_losses(
    mu: torch.Tensor, latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE supervision losses on the supervised slots of `mu`."""
    shape_idx  = latents[:, 1]
    scale_idx  = latents[:, 2]
    orient_idx = latents[:, 3]
    s_target = scale_target(scale_idx)
    o_target = orient_target(orient_idx, shape_idx)
    mu32 = mu.float()
    aux_scale  = F.mse_loss(mu32[:, Z_SCALE_IDX], s_target)
    aux_orient = F.mse_loss(mu32[:, Z_ORIENT_IDX[0]:Z_ORIENT_IDX[1] + 1], o_target)
    return aux_scale, aux_orient


# ---------------------------------------------------------------------------
# Train / val
# ---------------------------------------------------------------------------

def train_epoch(
    vae, device, loader, optimizer, criterion, *,
    amp: bool, beta_per_dim: torch.Tensor, free_mask: torch.Tensor,
    free_bits: float, lambda_scale: float, lambda_orient: float,
):
    vae.train()
    sums = {"recon": 0.0, "kl": 0.0, "aux_scale": 0.0, "aux_orient": 0.0}
    total_samples = 0
    for x, latents in loader:
        x = x.to(device, non_blocking=True)
        latents = latents.to(device, non_blocking=True)
        batch_size = x.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            x_hat, mu, logvar = vae(x)
            kl = _kl_with_per_dim_beta(mu, logvar, beta_per_dim, free_mask, free_bits)
        recon = criterion(x_hat.float(), x.float())
        aux_s, aux_o = _aux_losses(mu, latents)
        loss = recon + kl + lambda_scale * aux_s + lambda_orient * aux_o
        loss.backward()
        optimizer.step()

        sums["recon"]      += recon.item() * batch_size
        sums["kl"]         += kl.item()    * batch_size
        sums["aux_scale"]  += aux_s.item() * batch_size
        sums["aux_orient"] += aux_o.item() * batch_size
        total_samples      += batch_size
    return {k: v / total_samples for k, v in sums.items()}


def val_epoch(
    vae, device, loader, criterion, *,
    amp: bool, beta_per_dim: torch.Tensor, free_mask: torch.Tensor,
    free_bits: float, lambda_scale: float, lambda_orient: float,
):
    vae.eval()
    sums = {"recon": 0.0, "kl": 0.0, "aux_scale": 0.0, "aux_orient": 0.0,
            "scale_r2_w": 0.0, "orient_err_w": 0.0}
    total_samples = 0
    with torch.no_grad():
        for x, latents in loader:
            x = x.to(device, non_blocking=True)
            latents = latents.to(device, non_blocking=True)
            batch_size = x.shape[0]
            with autocast_ctx(device, amp):
                x_hat, mu, logvar = vae(x)
                kl = _kl_with_per_dim_beta(mu, logvar, beta_per_dim, free_mask, free_bits)
            recon = criterion(x_hat.float(), x.float())
            aux_s, aux_o = _aux_losses(mu, latents)

            # Eval metrics.
            shape_idx  = latents[:, 1]
            scale_idx  = latents[:, 2]
            orient_idx = latents[:, 3]
            r2  = scale_r2(mu[:, Z_SCALE_IDX], scale_idx)
            err = orient_angular_error_deg(
                mu[:, Z_ORIENT_IDX[0]:Z_ORIENT_IDX[1] + 1], orient_idx, shape_idx,
            )

            sums["recon"]        += recon.item() * batch_size
            sums["kl"]           += kl.item()    * batch_size
            sums["aux_scale"]    += aux_s.item() * batch_size
            sums["aux_orient"]   += aux_o.item() * batch_size
            sums["scale_r2_w"]   += r2  * batch_size
            sums["orient_err_w"] += err * batch_size
            total_samples += batch_size
    return {
        "recon":      sums["recon"]      / total_samples,
        "kl":         sums["kl"]         / total_samples,
        "aux_scale":  sums["aux_scale"]  / total_samples,
        "aux_orient": sums["aux_orient"] / total_samples,
        "scale_r2":   sums["scale_r2_w"] / total_samples,
        "orient_err_deg": sums["orient_err_w"] / total_samples,
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
    s_cfg   = cfg.get("supervision", {}) or {}
    rt_cfg  = (cfg.get("runtime") or {}).get(args.runtime, {}) if args.runtime else {}

    num_workers = apply_common_overrides(args, m_cfg, t_cfg, rt_cfg)
    if args.beta            is not None: t_cfg["beta"]            = args.beta
    if args.lambda_scale    is not None: s_cfg["lambda_scale"]    = args.lambda_scale
    if args.lambda_orient   is not None: s_cfg["lambda_orient"]   = args.lambda_orient
    if args.beta_supervised is not None: s_cfg["beta_supervised"] = args.beta_supervised
    if args.free_bits       is not None: s_cfg["free_bits"]       = args.free_bits

    # Resolve final values (with defaults matching the YAML schema).
    beta_free       = float(t_cfg["beta"])
    beta_supervised = float(s_cfg.get("beta_supervised", 0.0))
    lambda_scale    = float(s_cfg.get("lambda_scale", 5.0))
    lambda_orient   = float(s_cfg.get("lambda_orient", 5.0))
    free_bits       = float(s_cfg.get("free_bits", 0.0))

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
    def criterion(x_hat, x):
        return F.binary_cross_entropy(x_hat.float(), x.float(), reduction="sum") / x.shape[0]

    vae = maybe_compile(vae, args.no_compile, device)

    # Per-dim β + free-dim mask (built once, reused every step).
    beta_per_dim = _make_beta_per_dim(
        m_cfg["latent_dim"], beta_free, beta_supervised, device,
    )
    free_mask = _free_dim_mask(m_cfg["latent_dim"], device)

    # --- W&B ---
    flat_cfg = base_flat_config(
        args, m_cfg, t_cfg, d_cfg, dev_info, num_workers, amp_enabled,
        img_size=img_size,
    )
    flat_cfg.update({
        "beta":            beta_free,
        "beta_supervised": beta_supervised,
        "lambda_scale":    lambda_scale,
        "lambda_orient":   lambda_orient,
        "free_bits":       free_bits,
        "model_type":      "supervised_vae",
        "supervise_target_factors": True,
        "z_scale_idx":     Z_SCALE_IDX,
        "z_orient_idx":    list(Z_ORIENT_IDX),
    })
    run = init_wandb(log_cfg, args, flat_cfg)

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = out_dir / "best.pt"
    final_ckpt = out_dir / "final.pt"

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, t_cfg["epochs"] + 1), desc="Epochs"):
        train = train_epoch(
            vae, device, train_loader, optimizer, criterion,
            amp=amp_enabled,
            beta_per_dim=beta_per_dim, free_mask=free_mask, free_bits=free_bits,
            lambda_scale=lambda_scale, lambda_orient=lambda_orient,
        )
        val = val_epoch(
            vae, device, val_loader, criterion,
            amp=amp_enabled,
            beta_per_dim=beta_per_dim, free_mask=free_mask, free_bits=free_bits,
            lambda_scale=lambda_scale, lambda_orient=lambda_orient,
        )
        val_loss = (
            val["recon"] + val["kl"]
            + lambda_scale  * val["aux_scale"]
            + lambda_orient * val["aux_orient"]
        )

        log = {
            "epoch":                    epoch,
            "train/recon_loss":         train["recon"],
            "train/kl_loss":            train["kl"],
            "train/aux_scale":          train["aux_scale"],
            "train/aux_orient":         train["aux_orient"],
            "train/total_loss":         (train["recon"] + train["kl"]
                                         + lambda_scale  * train["aux_scale"]
                                         + lambda_orient * train["aux_orient"]),
            "val/recon_loss":           val["recon"],
            "val/kl_loss":              val["kl"],
            "val/aux_scale":            val["aux_scale"],
            "val/aux_orient":           val["aux_orient"],
            "val/total_loss":           val_loss,
            "val/scale_r2":             val["scale_r2"],
            "val/orient_angular_err_deg": val["orient_err_deg"],
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

    artifact = wandb.Artifact(name="supervised-vae-checkpoint", type="model")
    artifact.add_file(str(final_ckpt))
    if best_ckpt.exists():
        artifact.add_file(str(best_ckpt))
    run.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
