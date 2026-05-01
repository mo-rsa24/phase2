"""Shared training scaffolding for VAE / FactorVAE / supervised-VAE trainers.

Everything in this module is objective-agnostic — config loading, device probe,
seeding, dSprites split selection, dataloaders, torch.compile, autocast, wandb
init, checkpoint save. The trainer scripts only own their argparse, model
construction, loss computation, and per-epoch logging.

Public API:
    probe_device()
    seed_everything(seed)
    load_config(path)
    setup_device() -> (device, amp_enabled, dev_info)
    autocast_ctx(device, enabled)
    apply_common_overrides(args, m_cfg, t_cfg, rt_cfg)
    build_data_loaders(args, d_cfg, t_cfg, device, num_workers) -> (train, val, heldout_idx)
    maybe_compile(model, no_compile, device)
    base_flat_config(args, m_cfg, t_cfg, d_cfg, dev_info, num_workers, amp_enabled, recon_loss="bce_sum_per_sample")
    init_wandb(log_cfg, args, flat_cfg)
    save_checkpoint(path, payload)        # thin wrapper over torch.save
"""

from __future__ import annotations

import os
import random
import socket
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data
import yaml
import wandb

from src.datasets.dsprites import DSpritesDataset, load_dsprites, make_iid_split
from src.datasets.correlated_dsprites import (
    make_correlated_split,
    make_heldout_pair_split,
)


# ---------------------------------------------------------------------------
# Device + seed + config
# ---------------------------------------------------------------------------

def probe_device() -> dict:
    """Print device info; abort if installed torch wheel can't run on the GPU."""
    info = {
        "host":                  socket.gethostname(),
        "torch":                 torch.__version__,
        "cuda_visible_devices":  os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if not torch.cuda.is_available():
        info["device"] = "cpu"
        print(f"[device] cpu  host={info['host']}  torch={info['torch']}")
        return info

    props = torch.cuda.get_device_properties(0)
    sm = f"sm_{props.major}{props.minor}"
    arch_list = torch.cuda.get_arch_list()
    info.update({
        "device":     "cuda",
        "gpu_name":   props.name,
        "gpu_sm":     sm,
        "vram_gb":    round(props.total_memory / 1e9, 1),
        "arch_list":  arch_list,
    })
    print(f"[device] cuda  gpu={props.name}  sm={sm}  vram={info['vram_gb']}GB  "
          f"torch={info['torch']}  CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}")
    print(f"[device] torch arch_list={arch_list}")
    return info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_device() -> tuple[torch.device, bool, dict]:
    """Probe + set perf knobs + decide on AMP. Returns (device, amp_enabled, dev_info)."""
    dev_info = probe_device()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark           = True
        torch.backends.cuda.matmul.allow_tf32    = True
        torch.backends.cudnn.allow_tf32          = True
    amp_enabled = device.type == "cuda"
    return device, amp_enabled, dev_info


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ---------------------------------------------------------------------------
# CLI override merge
# ---------------------------------------------------------------------------

def apply_common_overrides(args, m_cfg: dict, t_cfg: dict, rt_cfg: dict) -> int:
    """Apply CLI > runtime > YAML for the keys every trainer shares.

    Mutates m_cfg / t_cfg in place. Returns num_workers.
    Each trainer applies its own objective-specific overrides (β, γ, λ_*) after.
    """
    if getattr(args, "latent_dim", None) is not None:
        m_cfg["latent_dim"] = args.latent_dim
    if getattr(args, "seed",       None) is not None:
        t_cfg["seed"]       = args.seed
    if getattr(args, "epochs",     None) is not None:
        t_cfg["epochs"]     = args.epochs
    if getattr(args, "batch_size", None) is not None:
        t_cfg["batch_size"] = args.batch_size
    elif "batch_size" in rt_cfg:
        t_cfg["batch_size"] = rt_cfg["batch_size"]

    num_workers = (
        args.num_workers
        if getattr(args, "num_workers", None) is not None
        else rt_cfg.get("num_workers", 4)
    )
    return num_workers


# ---------------------------------------------------------------------------
# Data loaders (handles iid / correlated / heldout splits)
# ---------------------------------------------------------------------------

def build_data_loaders(args, d_cfg: dict, t_cfg: dict, device: torch.device, num_workers: int):
    """Return (train_loader, val_loader, heldout_idx_or_None)."""
    dataset = load_dsprites(args.data_dir)
    heldout_idx: Optional[torch.Tensor] = None

    if args.split == "iid":
        train_idx, val_idx, _ = make_iid_split(
            dataset,
            train_frac=d_cfg["train_frac"],
            val_frac=d_cfg["val_frac"],
            seed=t_cfg["seed"],
        )
        print(f"[data] split=iid  train={len(train_idx)}  val={len(val_idx)}")
    elif args.split == "correlated":
        train_idx, val_idx, _ = make_correlated_split(
            dataset,
            factor_a=args.corr_factor_a,
            factor_b=args.corr_factor_b,
            correlation=args.corr_direction,
            train_frac=d_cfg["train_frac"],
            seed=t_cfg["seed"],
        )
        print(f"[data] split=correlated({args.corr_factor_a},{args.corr_factor_b},"
              f"{args.corr_direction})  train={len(train_idx)}  val={len(val_idx)}")
    elif args.split == "heldout":
        train_idx, val_idx, _, heldout_idx = make_heldout_pair_split(
            dataset,
            factor_a=args.heldout_factor_a,
            factor_b=args.heldout_factor_b,
            held_a_vals=args.heldout_a_vals,
            held_b_vals=args.heldout_b_vals,
            seed=t_cfg["seed"],
        )
        print(f"[data] split=heldout({args.heldout_factor_a}={args.heldout_a_vals},"
              f"{args.heldout_factor_b}={args.heldout_b_vals})  "
              f"train={len(train_idx)}  val={len(val_idx)}  heldout={len(heldout_idx)}")
    else:
        raise ValueError(f"unknown split: {args.split}")

    train_ds = DSpritesDataset(dataset, train_idx)
    val_ds   = DSpritesDataset(dataset, val_idx)
    loader_kwargs = dict(
        batch_size=t_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader, heldout_idx


# ---------------------------------------------------------------------------
# Compile + W&B + checkpoint
# ---------------------------------------------------------------------------

def maybe_compile(model, no_compile: bool, device: torch.device):
    if not no_compile and device.type == "cuda":
        try:
            compiled = torch.compile(model)
            print("[perf] torch.compile enabled")
            return compiled
        except Exception as e:
            print(f"[perf] torch.compile skipped: {e}")
            return model
    print("[perf] torch.compile disabled")
    return model


def base_flat_config(
    args,
    m_cfg: dict,
    t_cfg: dict,
    d_cfg: dict,
    dev_info: dict,
    num_workers: int,
    amp_enabled: bool,
    *,
    img_size: torch.Size,
    recon_loss: str = "bce_sum_per_sample",
) -> dict:
    """Build the wandb config dict shared across trainers.

    Trainers add their objective-specific keys (β, γ, λ_*) on top.
    """
    return {
        "latent_dim":   m_cfg["latent_dim"],
        "img_size":     str(img_size),
        "epochs":       t_cfg["epochs"],
        "batch_size":   t_cfg["batch_size"],
        "lr":           t_cfg["lr"],
        "weight_decay": t_cfg.get("weight_decay", 0.0),
        "recon_loss":   recon_loss,
        "seed":         t_cfg["seed"],
        "train_frac":   d_cfg["train_frac"],
        "val_frac":     d_cfg["val_frac"],
        "num_workers":  num_workers,
        "amp_dtype":    "bfloat16" if amp_enabled else "fp32",
        # Dataset split metadata.
        "split":             args.split,
        "corr_factor_a":     args.corr_factor_a if args.split == "correlated" else None,
        "corr_factor_b":     args.corr_factor_b if args.split == "correlated" else None,
        "corr_direction":    args.corr_direction if args.split == "correlated" else None,
        "heldout_factor_a":  args.heldout_factor_a if args.split == "heldout" else None,
        "heldout_factor_b":  args.heldout_factor_b if args.split == "heldout" else None,
        "heldout_a_vals":    list(args.heldout_a_vals) if args.split == "heldout" else None,
        "heldout_b_vals":    list(args.heldout_b_vals) if args.split == "heldout" else None,
        # Run provenance (kept identical to the pre-refactor schema).
        "experiment_id": args.experiment_id,
        "purpose":       args.purpose,
        "sweep_name":    args.wandb_group,
        "node":          args.node,
        "runtime":       args.runtime,
        "host":          dev_info.get("host"),
        "gpu_name":      dev_info.get("gpu_name"),
        "gpu_sm":        dev_info.get("gpu_sm"),
        "vram_gb":       dev_info.get("vram_gb"),
        "cuda_visible_devices": dev_info.get("cuda_visible_devices"),
    }


def init_wandb(log_cfg: dict, args, flat_cfg: dict):
    return wandb.init(
        project=log_cfg["wandb_project"],
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        config=flat_cfg,
    )


def save_checkpoint(path: Path, payload: dict) -> None:
    """Thin wrapper around torch.save so callers can stay declarative."""
    torch.save(payload, path)
