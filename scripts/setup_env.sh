#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common_env.sh"
ENV_PREFIX="$(realpath -m "${ENV_PREFIX:-$REPO_ROOT/.cache/conda/opsd}")"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.cache/conda/pkgs}"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

CREATE_ARGS=(--prefix "$ENV_PREFIX")
RUN_ARGS=(-p "$ENV_PREFIX")

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Updating existing Conda environment '$ENV_PREFIX' to the pinned specification."
    conda env update "${CREATE_ARGS[@]}" --file "$REPO_ROOT/environment.yml"
else
    conda env create "${CREATE_ARGS[@]}" --file "$REPO_ROOT/environment.yml"
fi

conda run --no-capture-output "${RUN_ARGS[@]}" \
    python -m pip install flash-attn==2.8.3 --no-build-isolation

conda run --no-capture-output "${RUN_ARGS[@]}" python - <<'PY'
import accelerate
import datasets
import deepspeed
import flash_attn
import math_verify
import re
import subprocess
import torch
import transformers
import trl
import vllm

expected = {
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "trl": "0.26.0",
    "datasets": "3.6.0",
    "vllm": "0.11.0",
}
actual = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "datasets": datasets.__version__,
    "vllm": vllm.__version__,
}
for package, version in expected.items():
    if not actual[package].startswith(version):
        raise RuntimeError(f"{package}={actual[package]} (expected {version})")

print("OPSD environment import check passed")
print(f"torch={torch.__version__}, transformers={transformers.__version__}, trl={trl.__version__}, vllm={vllm.__version__}")

try:
    nvcc = subprocess.run(
        ["nvcc", "--version"], check=True, capture_output=True, text=True
    ).stdout
except (FileNotFoundError, subprocess.CalledProcessError):
    print(
        "WARNING: nvcc was not found. Training may fail when DeepSpeed builds "
        "CUDA extensions; install the CUDA toolkit matching torch.version.cuda."
    )
else:
    match = re.search(r"release\s+(\d+\.\d+)", nvcc)
    nvcc_version = match.group(1) if match else "unknown"
    if nvcc_version != torch.version.cuda:
        print(
            f"WARNING: nvcc={nvcc_version}, but PyTorch was built for CUDA "
            f"{torch.version.cuda}. Install the matching toolkit before training."
        )
PY

echo "Activate with: conda activate $ENV_PREFIX"
