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
    # ---- Phase 2a: IID baseline (cells 1–5) -------------------------------
    {"id": 1, "trainer": "vae",        "split": "iid", "latent_dim": 10, "beta": 1.0,
     "purpose": "baseline",
     "notes": "Standard VAE baseline (IID)"},
    {"id": 2, "trainer": "vae",        "split": "iid", "latent_dim": 10, "beta": 4.0,
     "purpose": "beta-vae",
     "notes": "Beta-VAE, push toward disentanglement (IID)"},
    {"id": 3, "trainer": "vae",        "split": "iid", "latent_dim":  4, "beta": 1.0,
     "purpose": "undercomplete",
     "notes": "Undercomplete bottleneck — forces compression (IID)"},
    {"id": 4, "trainer": "vae",        "split": "iid", "latent_dim": 20, "beta": 1.0,
     "purpose": "overcomplete",
     "notes": "Overcomplete — check for posterior collapse (IID)"},
    {"id": 5, "trainer": "factor_vae", "split": "iid", "latent_dim": 10, "gamma": 35.0,
     "purpose": "factor-vae",
     "notes": "FactorVAE (Kim & Mnih 2018), γ=35 — adversarial TC penalty (IID)"},
    # ---- Phase 2b: correlated split (cells 6–10), mirroring 1–5 ----------
    # Default correlated factors: (scale, orientation), positive correlation.
    {"id": 6, "trainer": "vae",        "split": "correlated", "latent_dim": 10, "beta": 1.0,
     "purpose": "baseline-corr",
     "notes": "Standard VAE on correlated (scale, orientation+)"},
    {"id": 7, "trainer": "vae",        "split": "correlated", "latent_dim": 10, "beta": 4.0,
     "purpose": "beta-vae-corr",
     "notes": "Beta-VAE on correlated (scale, orientation+)"},
    {"id": 8, "trainer": "vae",        "split": "correlated", "latent_dim":  4, "beta": 1.0,
     "purpose": "undercomplete-corr",
     "notes": "Undercomplete on correlated (scale, orientation+)"},
    {"id": 9, "trainer": "vae",        "split": "correlated", "latent_dim": 20, "beta": 1.0,
     "purpose": "overcomplete-corr",
     "notes": "Overcomplete on correlated (scale, orientation+)"},
    {"id": 10, "trainer": "factor_vae", "split": "correlated", "latent_dim": 10, "gamma": 35.0,
     "purpose": "factor-vae-corr",
     "notes": "FactorVAE on correlated (scale, orientation+)"},
    # ---- Phase 2c: γ × recon-quality sweep at 150 epochs (cells 11–13) ----
    # name_suffix keeps these from colliding with cell 5's checkpoint dir
    # (factorvae_z10_gamma35.0_seed42) and gives them distinct wandb run names.
    {"id": 11, "trainer": "factor_vae", "split": "iid", "latent_dim": 10, "gamma": 10.0,
     "purpose": "factor-vae-gsweep", "name_suffix": "e150",
     "notes": "FactorVAE γ=10, 150 epochs — recon/TC trade-off study"},
    {"id": 12, "trainer": "factor_vae", "split": "iid", "latent_dim": 10, "gamma": 20.0,
     "purpose": "factor-vae-gsweep", "name_suffix": "e150",
     "notes": "FactorVAE γ=20, 150 epochs — recon/TC trade-off study"},
    {"id": 13, "trainer": "factor_vae", "split": "iid", "latent_dim": 10, "gamma": 35.0,
     "purpose": "factor-vae-gsweep", "name_suffix": "e150",
     "notes": "FactorVAE γ=35, 150 epochs — recon/TC trade-off study (vs 50ep cell 5)"},
    # ---- Phase 2f: targeted weak supervision (cells 20–23) ---------------
    # 2x2 grid over (λ ∈ {1,10,50}) × (β_supervised ∈ {0,1}). Ids start at 20
    # to leave room for future Phase 2d/2e cells.
    #   sup-A (20): diagnostic — reproduce v1 failure (matches existing seed=0 ckpt).
    #   sup-B (21): "just bump λ" — KL still on supervised dims.
    #   sup-C (22): per-dim β alone — modest λ, no KL on supervised dims.
    #   sup-D (23): recommended — both fixes together.
    {"id": 20, "trainer": "supervised_vae", "split": "iid",
     "latent_dim": 10, "beta": 1.0,
     "lambda_scale": 1.0, "lambda_orient": 1.0, "beta_supervised": 1.0,
     "purpose": "supervised-A-diagnostic",
     "notes": "Supervised VAE, λ=1, β_sup=1 (diagnostic baseline; reproduces v1 failure)"},
    {"id": 21, "trainer": "supervised_vae", "split": "iid",
     "latent_dim": 10, "beta": 1.0,
     "lambda_scale": 50.0, "lambda_orient": 50.0, "beta_supervised": 1.0,
     "purpose": "supervised-B-bruteforce-lambda",
     "notes": "Supervised VAE, λ=50, β_sup=1 (brute-force λ, KL still fights MSE)"},
    {"id": 22, "trainer": "supervised_vae", "split": "iid",
     "latent_dim": 10, "beta": 1.0,
     "lambda_scale": 1.0, "lambda_orient": 1.0, "beta_supervised": 0.0,
     "purpose": "supervised-C-perdim-beta",
     "notes": "Supervised VAE, λ=1, β_sup=0 (per-dim β alone, modest λ)"},
    {"id": 23, "trainer": "supervised_vae", "split": "iid",
     "latent_dim": 10, "beta": 1.0,
     "lambda_scale": 10.0, "lambda_orient": 10.0, "beta_supervised": 0.0,
     "purpose": "supervised-D-recommended",
     "notes": "Supervised VAE, λ=10, β_sup=0 (recommended; both fixes)"},
    # ---- Phase 2g: stronger β-VAE for clean axis-alignment (cells 30–31) --
    # β=4 (cell 2) sits at the bottom of the useful range for 64×64 binary
    # dSprites — Higgins et al. recommend β∈{4,8,16,32,64}, with the cleanly
    # axis-aligned regime usually starting at β≥8. e100 keeps these distinct
    # from the 50-epoch β=4 cell so MIG/DCI can plateau before we evaluate.
    {"id": 30, "trainer": "vae", "split": "iid", "latent_dim": 10, "beta": 8.0,
     "purpose": "beta-vae-strong", "name_suffix": "e100",
     "notes": "β-VAE β=8, 100 epochs — push for axis-alignment"},
    {"id": 31, "trainer": "vae", "split": "iid", "latent_dim": 10, "beta": 16.0,
     "purpose": "beta-vae-stronger", "name_suffix": "e100",
     "notes": "β-VAE β=16, 100 epochs — strong KL pressure, expect dim collapse"},
]


# Trainer dispatch: maps each cell's "trainer" key to a script path. The
# in-process loader in main() will exec the chosen script's main() with the
# argv we construct in build_train_argv().
TRAINER_PATHS = {
    "vae":            "scripts/train_vae.py",
    "factor_vae":     "scripts/train_factorvae.py",
    "supervised_vae": "scripts/train_supervised_vae.py",
}

# Default config per trainer (used when --config is not overridden by the
# bash launcher).
TRAINER_DEFAULT_CONFIG = {
    "vae":            "configs/vae.yaml",
    "factor_vae":     "configs/factor_vae.yml",
    "supervised_vae": "configs/supervised_vae.yaml",
}

# Default checkpoint root per trainer.
TRAINER_DEFAULT_OUT_DIR = {
    "vae":            "checkpoints/vae",
    "factor_vae":     "checkpoints/factor_vae",
    "supervised_vae": "checkpoints/vae",       # supervised runs share the vae root.
}


def get_experiment(experiment_id: int) -> dict:
    for exp in EXPERIMENTS:
        if exp["id"] == experiment_id:
            return exp
    raise ValueError(f"Unknown experiment_id={experiment_id}; valid: {[e['id'] for e in EXPERIMENTS]}")


def run_name(exp: dict, seed: int) -> str:
    """Run name format depends on the trainer family and split.

    IID + VAE family       : vae_z{latent}_beta{beta}_seed{seed}                            (unchanged)
    IID + FactorVAE        : factorvae_z{latent}_gamma{gamma}_seed{seed}                    (unchanged)
    IID + Supervised VAE   : supervised_vae_z{latent}_beta{beta}_lambda{λ}_betasup{βsup}_seed{seed}
    Correlated + VAE       : correlated_vae_z{latent}_beta{beta}_seed{seed}
    Correlated + FVAE      : correlated_factorvae_z{latent}_gamma{gamma}_seed{seed}
    Heldout    + …         : heldout_…

    The IID format for vae/factor_vae is unprefixed so existing checkpoint
    paths (Exps 1–5) remain valid without renaming. Supervised runs encode
    the sweep axes (λ, β_sup) into the directory name so the explorer's
    `_seeds_by_exp` glob can discover seeds within a sweep cell.
    """
    trainer = exp.get("trainer", "vae")
    split   = exp.get("split", "iid")
    if trainer == "factor_vae":
        base = f"factorvae_z{exp['latent_dim']}_gamma{exp['gamma']}_seed{seed}"
    elif trainer == "supervised_vae":
        # λ_s and λ_o are independent in principle but identical in this sweep;
        # collapse to a single λ token in the path for brevity, falling back to
        # λ_s if they ever diverge.
        lam_s = exp.get("lambda_scale",  1.0)
        lam_o = exp.get("lambda_orient", lam_s)
        lam_token = (f"{lam_s}" if lam_s == lam_o else f"{lam_s}-{lam_o}")
        bsup = exp.get("beta_supervised", 0.0)
        base = (f"supervised_vae_z{exp['latent_dim']}"
                f"_beta{exp['beta']}_lambda{lam_token}_betasup{bsup}_seed{seed}")
    else:
        base = f"vae_z{exp['latent_dim']}_beta{exp['beta']}_seed{seed}"
    if exp.get("name_suffix"):
        base = f"{base}_{exp['name_suffix']}"
    return base if split == "iid" else f"{split}_{base}"


def build_tags(exp: dict, *, runtime: str, node: str) -> list[str]:
    trainer = exp.get("trainer", "vae")
    split   = exp.get("split", "iid")
    base = [
        trainer,
        f"sweep:{SWEEP_NAME}",
        f"z{exp['latent_dim']}",
        f"split:{split}",
        exp["purpose"],
        f"runtime:{runtime}",
        f"node:{node}",
    ]
    if trainer == "factor_vae":
        base.insert(2, f"gamma{exp['gamma']}")
    elif trainer == "supervised_vae":
        base.insert(2, f"beta{exp['beta']}")
        base += [
            f"lambda_scale{exp.get('lambda_scale', 1.0)}",
            f"lambda_orient{exp.get('lambda_orient', 1.0)}",
            f"beta_supervised{exp.get('beta_supervised', 0.0)}",
        ]
    else:
        base.insert(2, f"beta{exp['beta']}")
    return base


# ---------------------------------------------------------------------------
# CLI -> train_vae.main() bridge
# ---------------------------------------------------------------------------

def build_train_argv(args: argparse.Namespace, exp: dict) -> list[str]:
    """Build the sys.argv list that the trainer's parse_args() will consume.

    Branches on exp['trainer']: 'vae' uses --beta, 'factor_vae' uses --gamma.
    """
    trainer = exp.get("trainer", "vae")
    tags = build_tags(exp, runtime=args.runtime, node=args.node)

    # Allow sweep-level CLI to override per-trainer defaults.
    cfg     = args.config  if args.config  else TRAINER_DEFAULT_CONFIG[trainer]
    out_dir = args.out_dir if args.out_dir else TRAINER_DEFAULT_OUT_DIR[trainer]

    script = {
        "factor_vae":     "train_factorvae.py",
        "supervised_vae": "train_supervised_vae.py",
    }.get(trainer, "train_vae.py")
    argv = [
        script,
        "--config",         cfg,
        "--data-dir",       args.data_dir,
        "--out-dir",        str(Path(out_dir) / run_name(exp, args.seed)),
        "--latent-dim",     str(exp["latent_dim"]),
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
    # Trainer-specific hyperparameter flags.
    if trainer == "factor_vae":
        argv += ["--gamma", str(exp["gamma"])]
    elif trainer == "supervised_vae":
        argv += [
            "--beta",            str(exp["beta"]),
            "--lambda-scale",    str(exp.get("lambda_scale", 1.0)),
            "--lambda-orient",   str(exp.get("lambda_orient", 1.0)),
            "--beta-supervised", str(exp.get("beta_supervised", 0.0)),
        ]
        if "free_bits" in exp:
            argv += ["--free-bits", str(exp["free_bits"])]
    else:
        argv += ["--beta", str(exp["beta"])]

    # Split + per-split factor selection.
    split = exp.get("split", "iid")
    argv += ["--split", split]
    if split == "correlated":
        argv += [
            "--corr-factor-a",  exp.get("corr_factor_a",  "scale"),
            "--corr-factor-b",  exp.get("corr_factor_b",  "orientation"),
            "--corr-direction", exp.get("corr_direction", "positive"),
        ]
    elif split == "heldout":
        argv += [
            "--heldout-factor-a", exp.get("heldout_factor_a", "shape"),
            "--heldout-factor-b", exp.get("heldout_factor_b", "scale"),
        ]
        if "heldout_a_vals" in exp:
            argv += ["--heldout-a-vals", *(str(v) for v in exp["heldout_a_vals"])]
        if "heldout_b_vals" in exp:
            argv += ["--heldout-b-vals", *(str(v) for v in exp["heldout_b_vals"])]

    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        argv += ["--batch-size", str(args.batch_size)]
    if args.no_compile:
        argv += ["--no-compile"]
    return argv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one cell of the disentanglement sweep.")
    p.add_argument("--experiment-id", type=int, required=True,
                   choices=[e["id"] for e in EXPERIMENTS])
    p.add_argument("--runtime",       type=str, required=True, choices=["hippo", "cluster48"])
    p.add_argument("--node",          type=str, required=True,
                   help="Logical node label (e.g. 'hippo', 'mscluster106').")
    p.add_argument("--seed",          type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--batch-size",    type=int, default=None)
    # --config and --out-dir default to per-trainer values when left unset.
    p.add_argument("--config",        type=str, default=None,
                   help="Defaults to configs/vae.yaml (vae) or configs/factor_vae.yml (factor_vae).")
    p.add_argument("--data-dir",      type=str, default="data")
    p.add_argument("--out-dir",       type=str, default=None,
                   help="Defaults to checkpoints/vae (vae) or checkpoints/factor_vae (factor_vae).")
    p.add_argument("--no-compile",    action="store_true")
    p.add_argument("--print-only",    action="store_true",
                   help="Print the trainer argv instead of running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp  = get_experiment(args.experiment_id)
    trainer = exp.get("trainer", "vae")
    argv = build_train_argv(args, exp)

    if args.print_only:
        # Quote for copy-paste safety.
        import shlex
        print(" ".join(shlex.quote(a) for a in ["python"] + argv))
        return

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if trainer == "factor_vae":
        hp = f"gamma={exp['gamma']}"
    elif trainer == "supervised_vae":
        hp = (f"beta={exp['beta']} lambda={exp.get('lambda_scale', 1.0)}/"
              f"{exp.get('lambda_orient', 1.0)} beta_sup={exp.get('beta_supervised', 0.0)}")
    else:
        hp = f"beta={exp['beta']}"
    split = exp.get("split", "iid")
    print(f"[sweep] experiment_id={exp['id']} trainer={trainer} split={split} "
          f"purpose={exp['purpose']} latent_dim={exp['latent_dim']} {hp} "
          f"node={args.node} runtime={args.runtime} CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[sweep] run_name={run_name(exp, args.seed)} group={SWEEP_GROUP}")

    # Hand off to the chosen trainer's main() in-process. We load by file
    # path to avoid needing scripts/ to be an importable package.
    import importlib.util
    train_path = REPO_ROOT / TRAINER_PATHS[trainer]
    spec = importlib.util.spec_from_file_location(trainer, train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    sys.argv = argv
    train_mod.main()


if __name__ == "__main__":
    main()
