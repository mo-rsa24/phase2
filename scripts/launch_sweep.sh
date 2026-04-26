#!/usr/bin/env bash
# Launch the 4-cell VAE disentanglement sweep on a chosen node.
#
#   --node hippo         → 4 experiments sequentially on GPU 0
#   --node mscluster106  → exp 1 + exp 2 in parallel on GPU 0 / GPU 1
#   --node mscluster107  → exp 3 + exp 4 in parallel on GPU 0 / GPU 1
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
OUT_DIR="checkpoints/vae"
DRY_RUN=0
NO_COMPILE=0
ENV_NAME="${ENV_NAME:-phase2-repr}"

usage() {
    cat <<EOF
Usage: $0 --node {hippo|mscluster106|mscluster107} [options]

Required:
  --node NODE         Which node we are running on (drives GPU mapping + runtime overlay).

Options:
  --seed N            Seed for all 4 runs (default: $SEED).
  --epochs N          Epoch override (default: from configs/vae.yaml).
  --out-dir PATH      Checkpoints root (default: $OUT_DIR).
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
        ASSIGNMENTS=("1 0" "2 0" "3 0" "4 0")   # sequential on GPU 0
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
    *)
        echo "ERROR: --node must be one of hippo|mscluster106|mscluster107 (got: $NODE)." >&2
        exit 2
        ;;
esac

# ---- Locate micromamba ----------------------------------------------------
if command -v micromamba >/dev/null 2>&1; then
    MAMBA_BIN="$(command -v micromamba)"
elif [[ -x "$HOME/.local/bin/micromamba" ]]; then
    MAMBA_BIN="$HOME/.local/bin/micromamba"
else
    echo "ERROR: micromamba not found on PATH or at ~/.local/bin/micromamba." >&2
    exit 1
fi

# ---- Sweep table (id:latent_dim:beta:purpose) -----------------------------
declare -A EXP_LATENT EXP_BETA EXP_PURPOSE
EXP_LATENT[1]=10;  EXP_BETA[1]=1.0;  EXP_PURPOSE[1]="baseline"
EXP_LATENT[2]=10;  EXP_BETA[2]=4.0;  EXP_PURPOSE[2]="beta-vae"
EXP_LATENT[3]=4;   EXP_BETA[3]=1.0;  EXP_PURPOSE[3]="undercomplete"
EXP_LATENT[4]=20;  EXP_BETA[4]=1.0;  EXP_PURPOSE[4]="overcomplete"

run_name_for() {
    local exp_id=$1
    echo "vae_z${EXP_LATENT[$exp_id]}_beta${EXP_BETA[$exp_id]}_seed${SEED}"
}

LOG_DIR="$PROJECT_ROOT/logs/sweep"
mkdir -p "$LOG_DIR"

# ---- Build the python command for a single experiment --------------------
python_cmd() {
    local exp_id=$1
    local cmd=(
        "$MAMBA_BIN" run -n "$ENV_NAME"
        python -u "$PROJECT_ROOT/scripts/sweep_disentanglement.py"
        --experiment-id "$exp_id"
        --runtime "$RUNTIME"
        --node "$NODE"
        --seed "$SEED"
        --out-dir "$OUT_DIR"
    )
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
