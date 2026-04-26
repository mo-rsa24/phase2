#!/usr/bin/env python3
"""Single source of truth for the 4-cell disentanglement sweep.

The bash launcher (scripts/launch_sweep.sh) calls this script once per
experiment, pinned to a specific GPU via CUDA_VISIBLE_DEVICES, and lets
train_vae.main() do the actual work in-process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

SWEEP_NAME  = "disentanglement-sweep"
SWEEP_GROUP = "disentanglement-sweep-2026-04-26"
DEFAULT_SEED = 42

EXPERIMENTS = [
    {"id": 1, "latent_dim": 10, "beta": 1.0, "purpose": "baseline",
     "notes": "Standard VAE baseline"},
    {"id": 2, "latent_dim": 10, "beta": 4.0, "purpose": "beta-vae",
     "notes": "Beta-VAE, push toward disentanglement"},
    {"id": 3, "latent_dim":  4, "beta": 1.0, "purpose": "undercomplete",
     "notes": "Undercomplete bottleneck — forces compression"},
    {"id": 4, "latent_dim": 20, "beta": 1.0, "purpose": "overcomplete",
     "notes": "Overcomplete — check for posterior collapse"},
]


def get_experiment(experiment_id: int) -> dict:
    for exp in EXPERIMENTS:
        if exp["id"] == experiment_id:
            return exp
    raise ValueError(f"Unknown experiment_id={experiment_id}; valid: {[e['id'] for e in EXPERIMENTS]}")


def run_name(exp: dict, seed: int) -> str:
    return f"vae_z{exp['latent_dim']}_beta{exp['beta']}_seed{seed}"


def build_tags(exp: dict, *, runtime: str, node: str) -> list[str]:
    return [
        "vae",
        f"sweep:{SWEEP_NAME}",
        f"z{exp['latent_dim']}",
        f"beta{exp['beta']}",
        exp["purpose"],
        f"runtime:{runtime}",
        f"node:{node}",
    ]


# ---------------------------------------------------------------------------
# CLI -> train_vae.main() bridge
# ---------------------------------------------------------------------------

def build_train_argv(args: argparse.Namespace, exp: dict) -> list[str]:
    """Build the sys.argv list that train_vae.parse_args() will consume."""
    tags = build_tags(exp, runtime=args.runtime, node=args.node)
    argv = [
        "train_vae.py",
        "--config",         args.config,
        "--data-dir",       args.data_dir,
        "--out-dir",        str(Path(args.out_dir) / run_name(exp, args.seed)),
        "--latent-dim",     str(exp["latent_dim"]),
        "--beta",           str(exp["beta"]),
        "--seed",           str(args.seed),
        "--runtime",        args.runtime,
        "--node",           args.node,
        "--purpose",        exp["purpose"],
        "--experiment-id",  str(exp["id"]),
        "--wandb-run-name", run_name(exp, args.seed),
        "--wandb-group",    SWEEP_GROUP,
        "--wandb-notes",    exp["notes"],
        "--wandb-tags",     *tags,
    ]
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        argv += ["--batch-size", str(args.batch_size)]
    if args.no_compile:
        argv += ["--no-compile"]
    return argv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one cell of the VAE disentanglement sweep.")
    p.add_argument("--experiment-id", type=int, required=True, choices=[e["id"] for e in EXPERIMENTS])
    p.add_argument("--runtime",       type=str, required=True, choices=["hippo", "cluster48"])
    p.add_argument("--node",          type=str, required=True,
                   help="Logical node label (e.g. 'hippo', 'mscluster106').")
    p.add_argument("--seed",          type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--batch-size",    type=int, default=None)
    p.add_argument("--config",        type=str, default="configs/vae.yaml")
    p.add_argument("--data-dir",      type=str, default="data")
    p.add_argument("--out-dir",       type=str, default="checkpoints/vae")
    p.add_argument("--no-compile",    action="store_true")
    p.add_argument("--print-only",    action="store_true",
                   help="Print the train_vae.py argv instead of running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp  = get_experiment(args.experiment_id)
    argv = build_train_argv(args, exp)

    if args.print_only:
        # Quote for copy-paste safety.
        import shlex
        print(" ".join(shlex.quote(a) for a in ["python"] + argv))
        return

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[sweep] experiment_id={exp['id']} purpose={exp['purpose']} "
          f"latent_dim={exp['latent_dim']} beta={exp['beta']} "
          f"node={args.node} runtime={args.runtime} CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[sweep] run_name={run_name(exp, args.seed)} group={SWEEP_GROUP}")

    # Hand off to train_vae.main() in-process. We load by file path to avoid
    # needing scripts/ to be an importable package.
    import importlib.util
    train_path = REPO_ROOT / "scripts" / "train_vae.py"
    spec = importlib.util.spec_from_file_location("train_vae", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    sys.argv = argv
    train_mod.main()


if __name__ == "__main__":
    main()
