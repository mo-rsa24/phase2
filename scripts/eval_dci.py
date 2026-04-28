#!/usr/bin/env python3
"""Compute DCI on a trained checkpoint and dump the result as JSON.

Loads encoder + decoder via the explorer's `load_encoder_decoder`,
encodes a configurable number of dSprites samples, runs `compute_dci`,
prints a summary, and writes `<ckpt-dir>/dci.json`.

Usage
-----
    python scripts/eval_dci.py \\
        --checkpoint checkpoints/vae/vae_z10_beta1.0_seed42/best.pt \\
        --n-samples 3000

If --json-out is omitted, the result is written next to the checkpoint
as `dci.json`. Pass `--print-only` to skip the file write.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.dsprites import FACTOR_NAMES, FACTOR_SIZES, load_dsprites
from src.metrics.dci import compute_dci
from src.utils.vae_inspection import load_encoder_decoder


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DCI on a trained checkpoint.")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to a .pt checkpoint (must contain encoder_state_dict).")
    p.add_argument("--data-dir",   default="data", type=str)
    p.add_argument("--device",     default="cpu",  type=str)
    p.add_argument("--n-samples",  default=3000,   type=int,
                   help="Random samples to encode for the metric.")
    p.add_argument("--seed",       default=0,      type=int)
    p.add_argument("--n-estimators", default=50,   type=int,
                   help="Random-forest n_estimators.")
    p.add_argument("--max-depth",  default=None,   type=int,
                   help="Random-forest max_depth (None = unlimited).")
    p.add_argument("--json-out",   default=None,   type=str,
                   help="Output path; defaults to <ckpt-dir>/dci.json.")
    p.add_argument("--print-only", action="store_true",
                   help="Skip writing dci.json; print to stdout only.")
    return p.parse_args()


@torch.no_grad()
def encode_samples(encoder, dataset, n: int, *, device: str, seed: int):
    """Encode N random dSprites images. Returns (mu, factors)."""
    encoder.eval()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset["imgs"]), min(n, len(dataset["imgs"])), replace=False)
    imgs    = dataset["imgs"][idx].astype(np.float32)
    factors = dataset["latents_classes"][idx]

    mu_list = []
    for i in range(0, len(imgs), 512):
        x = torch.from_numpy(imgs[i:i+512][:, np.newaxis]).float().to(device)
        mu, _ = encoder.encode(x)
        mu_list.append(mu.cpu().numpy())
    mu_arr = np.concatenate(mu_list)
    return mu_arr, factors


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    print(f"[eval_dci] loading: {ckpt_path}")
    encoder, _ = load_encoder_decoder(
        encoder_checkpoint=str(ckpt_path),
        decoder_checkpoint=str(ckpt_path),
        device=args.device,
    )
    print(f"[eval_dci] latent_dim = {encoder.latent_dim}")

    print(f"[eval_dci] loading dSprites …")
    dataset = load_dsprites(args.data_dir)
    print(f"[eval_dci] dataset ready: {len(dataset['imgs'])} images")

    print(f"[eval_dci] encoding {args.n_samples} samples …")
    mu, factors = encode_samples(
        encoder, dataset, args.n_samples,
        device=args.device, seed=args.seed,
    )

    print(f"[eval_dci] running compute_dci (RF n_estimators={args.n_estimators}) …")
    result = compute_dci(
        mu, factors,
        factor_names=FACTOR_NAMES,
        factor_sizes=FACTOR_SIZES,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    # Pretty summary
    print("\n========== DCI ==========")
    print(f"  Disentanglement (D): {result['D']:.3f}")
    print(f"  Completeness    (C): {result['C']:.3f}")
    print(f"  Informativeness (I): {result['I']:.3f}")
    print()
    print(f"  Per-factor C / I:")
    for name, c, i in zip(result["factor_names"],
                          result["C_per_factor"],
                          result["I_per_factor"]):
        print(f"    {name:14s}  C={c:.3f}  I={i:.3f}")
    print()
    print(f"  Top-3 latent dims by total importance:")
    total_imp = result["importance"].sum(axis=1)
    top = np.argsort(total_imp)[::-1][:3]
    for i in top:
        share = total_imp[i] / max(total_imp.sum(), 1e-12)
        print(f"    z{i:<3d}  importance share={share:.2%}  D={result['D_per_latent'][i]:.3f}")
    print()

    if not args.print_only:
        out_path = Path(args.json_out) if args.json_out else ckpt_path.parent / "dci.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays to lists for JSON serialisation.
        serialisable = {
            "importance":     result["importance"].tolist(),
            "D_per_latent":   result["D_per_latent"].tolist(),
            "D":              result["D"],
            "C_per_factor":   result["C_per_factor"].tolist(),
            "C":              result["C"],
            "I_per_factor":   result["I_per_factor"].tolist(),
            "I":              result["I"],
            "factor_names":   result["factor_names"],
            "n_samples":      int(args.n_samples),
            "n_estimators":   int(args.n_estimators),
            "seed":           int(args.seed),
            "checkpoint":     str(ckpt_path),
        }
        out_path.write_text(json.dumps(serialisable, indent=2))
        print(f"[eval_dci] wrote {out_path}")


if __name__ == "__main__":
    main()
