#!/usr/bin/env bash
# Launch the 10-cell disentanglement sweep on a chosen node.
#
# IID baseline (cells 1–5):
#   --node hippo         → 5 IID experiments sequentially on GPU 0
#   --node mscluster106  → IID exps 1+2 in parallel on GPU 0 / GPU 1
#   --node mscluster107  → IID exps 3+4 in parallel on GPU 0 / GPU 1
#   --node mscluster108  → IID exp 5 (FactorVAE) on GPU 0
#
# Phase 2b — correlated split (cells 6–10):
#   --node hippo-corr        → 5 correlated experiments sequentially on GPU 0
#   --node mscluster106-corr → corr exps 6+7 in parallel
#   --node mscluster107-corr → corr exps 8+9 in parallel
#   --node mscluster108-corr → corr exp 10 (FactorVAE) on GPU 0
#
# Phase 2c — γ × recon sweep at 150 epochs (cells 11–13):
#   --node hippo-gamma   → γ∈{10,20,35} sequentially on GPU 0 (~2h total)
#
# Assumes phase2-repr micromamba env, this repo, data/dsprites.npz, and
# `wandb login` are already set up on the node.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Defaults -------------------------------------------------------------
NODE=""
SEED=42
EPOCHS=""
OUT_DIR=""           # empty = let sweep_disentanglement pick per-trainer default
DRY_RUN=0
NO_COMPILE=0
ENV_NAME="${ENV_NAME:-phase2-repr}"

usage() {
    cat <<EOF
Usage: $0 --node {hippo|mscluster106|mscluster107} [options]

Required:
  --node NODE         Which node we are running on (drives GPU mapping + runtime overlay).

Options:
  --seed N            Seed for all runs (default: $SEED).
  --epochs N          Epoch override (default: from per-trainer YAML).
  --out-dir PATH      Checkpoints root (default: per-trainer:
                      checkpoints/vae for VAE cells,
                      checkpoints/factor_vae for FactorVAE cells).
  --no-compile        Disable torch.compile (escape hatch).
  --dry-run           Print the per-experiment commands without running.
  -h, --help          This help.

Env vars:
  ENV_NAME            micromamba env (default: phase2-repr).
EOF
}

# ---- Arg parse ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)       NODE="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --no-compile) NO_COMPILE=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$NODE" ]]; then
    echo "ERROR: --node is required." >&2
    usage
    exit 2
fi

# ---- Per-node config ------------------------------------------------------
# Each entry: "experiment_id gpu_id"
case "$NODE" in
    hippo)
        RUNTIME="hippo"
        ASSIGNMENTS=("1 0" "2 0" "3 0" "4 0" "5 0")
        PARALLEL=0
        ;;
    mscluster106)
        RUNTIME="cluster48"
        ASSIGNMENTS=("1 0" "2 1")
        PARALLEL=1
        ;;
    mscluster107)
        RUNTIME="cluster48"
        ASSIGNMENTS=("3 0" "4 1")
        PARALLEL=1
        ;;
    mscluster108)
        RUNTIME="cluster48"
        ASSIGNMENTS=("5 0")
        PARALLEL=0
        ;;
    # ---- Phase 2b: correlated split (cells 6–10) -----------------------
    hippo-corr)
        RUNTIME="hippo"
        ASSIGNMENTS=("6 0" "7 0" "8 0" "9 0" "10 0")
        PARALLEL=0
        ;;
    mscluster106-corr)
        RUNTIME="cluster48"
        ASSIGNMENTS=("6 0" "7 1")
        PARALLEL=1
        ;;
    mscluster107-corr)
        RUNTIME="cluster48"
        ASSIGNMENTS=("8 0" "9 1")
        PARALLEL=1
        ;;
    mscluster108-corr)
        RUNTIME="cluster48"
        ASSIGNMENTS=("10 0")
        PARALLEL=0
        ;;
    # ---- Phase 2c: γ-sweep at 150 epochs (cells 11–13), hippo only ------
    hippo-gamma)
        RUNTIME="hippo"
        ASSIGNMENTS=("11 0" "12 0" "13 0")
        PARALLEL=0
        ;;
    *)
        echo "ERROR: --node must be one of:" >&2
        echo "  hippo|mscluster106|mscluster107|mscluster108" >&2
        echo "  hippo-corr|mscluster106-corr|mscluster107-corr|mscluster108-corr" >&2
        echo "  hippo-gamma" >&2
        echo "(got: $NODE)" >&2
        exit 2
        ;;
esac

# The bash launcher passes --node $NODE to sweep_disentanglement.py, which
# in turn validates against a fixed list of node tags used as wandb tags.
# The "-corr" / "-gamma" variants are bash-launcher-only routing aliases;
# strip the suffix when forwarding the tag downstream.
NODE_TAG="${NODE%-corr}"
NODE_TAG="${NODE_TAG%-gamma}"

# ---- Locate micromamba ----------------------------------------------------
if command -v micromamba >/dev/null 2>&1; then
    MAMBA_BIN="$(command -v micromamba)"
elif [[ -x "$HOME/.local/bin/micromamba" ]]; then
    MAMBA_BIN="$HOME/.local/bin/micromamba"
else
    echo "ERROR: micromamba not found on PATH or at ~/.local/bin/micromamba." >&2
    exit 1
fi

# ---- Sweep table (mirror of EXPERIMENTS in sweep_disentanglement.py) -----
declare -A EXP_TRAINER EXP_SPLIT EXP_LATENT EXP_BETA EXP_GAMMA EXP_PURPOSE
# IID baseline (cells 1–5)
EXP_TRAINER[1]="vae";        EXP_SPLIT[1]="iid";        EXP_LATENT[1]=10; EXP_BETA[1]=1.0;   EXP_PURPOSE[1]="baseline"
EXP_TRAINER[2]="vae";        EXP_SPLIT[2]="iid";        EXP_LATENT[2]=10; EXP_BETA[2]=4.0;   EXP_PURPOSE[2]="beta-vae"
EXP_TRAINER[3]="vae";        EXP_SPLIT[3]="iid";        EXP_LATENT[3]=4;  EXP_BETA[3]=1.0;   EXP_PURPOSE[3]="undercomplete"
EXP_TRAINER[4]="vae";        EXP_SPLIT[4]="iid";        EXP_LATENT[4]=20; EXP_BETA[4]=1.0;   EXP_PURPOSE[4]="overcomplete"
EXP_TRAINER[5]="factor_vae"; EXP_SPLIT[5]="iid";        EXP_LATENT[5]=10; EXP_GAMMA[5]=35.0; EXP_PURPOSE[5]="factor-vae"
# Correlated split (cells 6–10)
EXP_TRAINER[6]="vae";        EXP_SPLIT[6]="correlated"; EXP_LATENT[6]=10; EXP_BETA[6]=1.0;   EXP_PURPOSE[6]="baseline-corr"
EXP_TRAINER[7]="vae";        EXP_SPLIT[7]="correlated"; EXP_LATENT[7]=10; EXP_BETA[7]=4.0;   EXP_PURPOSE[7]="beta-vae-corr"
EXP_TRAINER[8]="vae";        EXP_SPLIT[8]="correlated"; EXP_LATENT[8]=4;  EXP_BETA[8]=1.0;   EXP_PURPOSE[8]="undercomplete-corr"
EXP_TRAINER[9]="vae";        EXP_SPLIT[9]="correlated"; EXP_LATENT[9]=20; EXP_BETA[9]=1.0;   EXP_PURPOSE[9]="overcomplete-corr"
EXP_TRAINER[10]="factor_vae"; EXP_SPLIT[10]="correlated"; EXP_LATENT[10]=10; EXP_GAMMA[10]=35.0; EXP_PURPOSE[10]="factor-vae-corr"
# γ × recon-quality sweep at 150 epochs (cells 11–13) — name_suffix keeps
# these from colliding with cell 5's checkpoint dir.
declare -A EXP_SUFFIX
EXP_TRAINER[11]="factor_vae"; EXP_SPLIT[11]="iid"; EXP_LATENT[11]=10; EXP_GAMMA[11]=10.0; EXP_PURPOSE[11]="factor-vae-gsweep"; EXP_SUFFIX[11]="e150"
EXP_TRAINER[12]="factor_vae"; EXP_SPLIT[12]="iid"; EXP_LATENT[12]=10; EXP_GAMMA[12]=20.0; EXP_PURPOSE[12]="factor-vae-gsweep"; EXP_SUFFIX[12]="e150"
EXP_TRAINER[13]="factor_vae"; EXP_SPLIT[13]="iid"; EXP_LATENT[13]=10; EXP_GAMMA[13]=35.0; EXP_PURPOSE[13]="factor-vae-gsweep"; EXP_SUFFIX[13]="e150"

run_name_for() {
    local exp_id=$1
    local base
    if [[ "${EXP_TRAINER[$exp_id]}" == "factor_vae" ]]; then
        base="factorvae_z${EXP_LATENT[$exp_id]}_gamma${EXP_GAMMA[$exp_id]}_seed${SEED}"
    else
        base="vae_z${EXP_LATENT[$exp_id]}_beta${EXP_BETA[$exp_id]}_seed${SEED}"
    fi
    if [[ -n "${EXP_SUFFIX[$exp_id]:-}" ]]; then
        base="${base}_${EXP_SUFFIX[$exp_id]}"
    fi
    if [[ "${EXP_SPLIT[$exp_id]}" == "iid" ]]; then
        echo "$base"
    else
        echo "${EXP_SPLIT[$exp_id]}_${base}"
    fi
}

LOG_DIR="$PROJECT_ROOT/logs/sweep"
mkdir -p "$LOG_DIR"

# ---- Build the python command for a single experiment --------------------
# OUT_DIR is left unset by default so sweep_disentanglement.py can pick the
# per-trainer default (checkpoints/vae or checkpoints/factor_vae).
python_cmd() {
    local exp_id=$1
    local cmd=(
        "$MAMBA_BIN" run -n "$ENV_NAME"
        python -u "$PROJECT_ROOT/scripts/sweep_disentanglement.py"
        --experiment-id "$exp_id"
        --runtime "$RUNTIME"
        --node "$NODE_TAG"
        --seed "$SEED"
    )
    [[ -n "$OUT_DIR"  ]] && cmd+=(--out-dir "$OUT_DIR")
    [[ -n "$EPOCHS"   ]] && cmd+=(--epochs "$EPOCHS")
    [[ "$NO_COMPILE" == 1 ]] && cmd+=(--no-compile)
    printf '%q ' "${cmd[@]}"
}

# ---- Trap to kill background children on Ctrl-C --------------------------
PIDS=()
cleanup() {
    if (( ${#PIDS[@]} > 0 )); then
        echo
        echo "[launch_sweep] caught signal, killing background runs: ${PIDS[*]}" >&2
        kill -TERM "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

# ---- Print plan -----------------------------------------------------------
echo "==============================================="
echo "node          : $NODE"
echo "runtime       : $RUNTIME"
echo "seed          : $SEED"
echo "epochs        : ${EPOCHS:-from-yaml}"
echo "env           : $ENV_NAME ($MAMBA_BIN)"
echo "parallel      : $PARALLEL"
echo "assignments   : ${ASSIGNMENTS[*]}"
echo "log dir       : $LOG_DIR"
echo "==============================================="

# ---- Dry run --------------------------------------------------------------
if [[ "$DRY_RUN" == 1 ]]; then
    for spec in "${ASSIGNMENTS[@]}"; do
        read -r EXP_ID GPU_ID <<<"$spec"
        echo
        echo "[exp $EXP_ID, GPU $GPU_ID]  ($(run_name_for "$EXP_ID"), purpose=${EXP_PURPOSE[$EXP_ID]})"
        echo "  CUDA_VISIBLE_DEVICES=$GPU_ID $(python_cmd "$EXP_ID")"
    done
    echo
    echo "(dry-run; nothing executed)"
    exit 0
fi

# ---- Execute --------------------------------------------------------------
for spec in "${ASSIGNMENTS[@]}"; do
    read -r EXP_ID GPU_ID <<<"$spec"
    RUN_NAME="$(run_name_for "$EXP_ID")"
    LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
    CMD="$(python_cmd "$EXP_ID")"

    echo
    echo "[launch_sweep] starting exp=$EXP_ID gpu=$GPU_ID run=$RUN_NAME log=$LOG_FILE"

    if [[ "$PARALLEL" == 1 ]]; then
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$GPU_ID" bash -c "$CMD" 2>&1 | tee "$LOG_FILE" &
        PIDS+=($!)
    else
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$GPU_ID" bash -c "$CMD" 2>&1 | tee "$LOG_FILE"
    fi
done

if [[ "$PARALLEL" == 1 ]]; then
    echo
    echo "[launch_sweep] waiting on ${#PIDS[@]} background runs (pids: ${PIDS[*]})"
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=$((FAIL + 1))
    done
    if (( FAIL > 0 )); then
        echo "[launch_sweep] FAILED: $FAIL of ${#PIDS[@]} runs returned non-zero." >&2
        exit 1
    fi
fi

echo
echo "[launch_sweep] done. logs in $LOG_DIR"
