#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-phase2-repr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CHANNEL="${TORCH_CHANNEL:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba is required but was not found on PATH." >&2
  exit 1
fi

if [[ -z "${TORCH_CHANNEL}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    # This machine reports nvcc 12.0 and a newer driver/runtime (CUDA 13.0 capable),
    # so the official PyTorch cu124 wheels are a good fit for this project.
    TORCH_CHANNEL="cu124"
  else
    TORCH_CHANNEL="cpu"
  fi
fi

case "${TORCH_CHANNEL}" in
  cu118|cu124|cu126|cpu)
    ;;
  *)
    echo "Unsupported TORCH_CHANNEL=${TORCH_CHANNEL}. Use one of: cu118, cu124, cu126, cpu." >&2
    exit 1
    ;;
esac

TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CHANNEL}"

echo "Project root: ${PROJECT_ROOT}"
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "PyTorch channel: ${TORCH_CHANNEL}"
echo "PyTorch index: ${TORCH_INDEX_URL}"

if ! micromamba run -n "${ENV_NAME}" python -V >/dev/null 2>&1; then
  echo "Creating micromamba environment ${ENV_NAME}..."
  micromamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "Micromamba environment ${ENV_NAME} already exists. Reusing it."
fi

echo "Upgrading pip tooling..."
micromamba run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "Installing project requirements..."
micromamba run -n "${ENV_NAME}" python -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  --extra-index-url https://pypi.org/simple \
  -r "${PROJECT_ROOT}/requirements.txt"

echo "Verifying PyTorch and CUDA availability..."
micromamba run -n "${ENV_NAME}" python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"device_count={torch.cuda.device_count()}")
PY

echo
echo "Install complete."
echo "Activate the environment with:"
echo "  micromamba activate ${ENV_NAME}"
echo
echo "If you want a different wheel channel later, rerun with for example:"
echo "  TORCH_CHANNEL=cu126 bash scripts.sh ${ENV_NAME}"
