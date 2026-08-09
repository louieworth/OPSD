#!/usr/bin/env bash

# Shared runtime paths. Every generated artifact except the Hugging Face model
# cache stays below the repository root so a cloned checkout is self-contained.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/.cache/huggingface/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$REPO_ROOT/.cache/vllm}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/outputs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/.cache/torch_extensions}"
export TMPDIR="${OPSD_TMPDIR:-$REPO_ROOT/.cache/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache/xdg}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$REPO_ROOT/.cache/cuda}"

mkdir -p \
    "$HF_DATASETS_CACHE" \
    "$TRITON_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" \
    "$WANDB_DIR" \
    "$PIP_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TMPDIR" \
    "$XDG_CACHE_HOME" \
    "$CUDA_CACHE_PATH"
