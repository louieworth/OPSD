#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/common_env.sh"

TACO_REVISION="d593ed0a2becbbc952230bb89be09189bf1056dc"
QWEN17_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TRD_COMMIT="5f3894d776cb2b762a44e09f8ce8293a762e21af"
EVALPLUS_COMMIT="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
PYTHON_BIN="${CODE_PYTHON_BIN:-python}"

usage() {
    echo "Usage: bash script_code/prepare_code.sh [all|train|eval|verify]" >&2
}

die() {
    echo "prepare_code.sh: $*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_python() {
    "$PYTHON_BIN" -c 'import datasets, huggingface_hub, pyarrow, transformers' >/dev/null || \
        die "activate the OPSD Python environment before running this script"
}

prepare_train_data() {
    echo "[train 1/2] Preparing pinned OpenThoughts math training data..."
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
        "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_data.py" --scope train

    echo "[train 2/2] Preparing the 18,862-row clean TACO artifact..."
    local tokenizer_source="Qwen/Qwen3-1.7B"
    if [[ -f "$REPO_ROOT/models/Qwen3-1.7B/tokenizer_config.json" ]]; then
        tokenizer_source="$REPO_ROOT/models/Qwen3-1.7B"
    fi
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 PYTHONPATH="$REPO_ROOT" \
        "$PYTHON_BIN" "$SCRIPT_DIR/prepare_data.py" \
        --output_dir "$REPO_ROOT/data/train/taco_code_clean" \
        --tokenizer_path "$tokenizer_source" \
        --tokenizer_revision "$QWEN17_REVISION" \
        --revision "$TACO_REVISION"
}

prepare_eval_sources() {
    require_command git
    mkdir -p "$REPO_ROOT/third_party"

    local evalplus_root="$REPO_ROOT/third_party/evalplus"
    if [[ ! -d "$evalplus_root/.git" ]]; then
        [[ ! -e "$evalplus_root" ]] || die "$evalplus_root exists but is not a Git checkout"
        git clone --no-checkout https://github.com/evalplus/evalplus.git "$evalplus_root"
        git -C "$evalplus_root" checkout --detach "$EVALPLUS_COMMIT"
    fi
    [[ "$(git -C "$evalplus_root" rev-parse HEAD)" == "$EVALPLUS_COMMIT" ]] || \
        die "EvalPlus checkout is not at pinned commit $EVALPLUS_COMMIT"

    local trd_root="$REPO_ROOT/third_party/trd"
    if [[ ! -d "$trd_root/.git" ]]; then
        [[ ! -e "$trd_root" ]] || die "$trd_root exists but is not a Git checkout"
        git clone --filter=blob:none --no-checkout https://github.com/louieworth/trd.git "$trd_root"
        git -C "$trd_root" sparse-checkout init --cone
        git -C "$trd_root" sparse-checkout set recipe/code_evaluation/external/LiveCodeBench
        git -C "$trd_root" checkout --detach "$TRD_COMMIT"
    fi
    [[ "$(git -C "$trd_root" rev-parse HEAD)" == "$TRD_COMMIT" ]] || \
        die "TRD checkout is not at pinned commit $TRD_COMMIT"

    local lcb_root="$trd_root/recipe/code_evaluation/external/LiveCodeBench"
    "$PYTHON_BIN" "$SCRIPT_DIR/eval/patch_livecodebench.py" "$lcb_root"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON_BIN" -m pip install "$evalplus_root" "$lcb_root"

    echo "[eval] Downloading/prefetching HumanEval+, MBPP+, and LiveCodeBench v6..."
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
        "$PYTHON_BIN" "$SCRIPT_DIR/prepare_eval_data.py" --root "$REPO_ROOT"

    local python_path
    local attention_implementation="flash_attention_2"
    python_path="$(command -v "$PYTHON_BIN")"
    if ! "$PYTHON_BIN" -c 'import flash_attn' >/dev/null 2>&1; then
        attention_implementation="sdpa"
    fi
    printf '%s\n' \
        "export CODE_PYTHON_BIN='$python_path'" \
        "export PATH='$(dirname "$python_path")':\"\$PATH\"" \
        "export CODE_ATTN_IMPLEMENTATION='${CODE_ATTN_IMPLEMENTATION:-$attention_implementation}'" \
        "export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'" \
        "export XDG_CACHE_HOME='$XDG_CACHE_HOME'" \
        "export HF_HUB_OFFLINE=1" \
        "export HF_DATASETS_OFFLINE=1" \
        "export TRANSFORMERS_OFFLINE=1" \
        "export WANDB_MODE=offline" \
        "export LCB_REPO='$lcb_root'" \
        "export LCB_CODEGEN_LITE_DIR='$REPO_ROOT/data/eval/livecodebench_code_generation_lite'" \
        > "$SCRIPT_DIR/runtime.env"
}

verify_components() {
    "$PYTHON_BIN" "$SCRIPT_DIR/preflight.py" --root "$REPO_ROOT" --scope all
    CODE_DRY_RUN=1 bash "$SCRIPT_DIR/run_matrix.sh" --dry-run >/dev/null
}

mode="${1:-all}"
(( $# <= 1 )) || { usage; exit 2; }
case "$mode" in
    -h|--help) usage; exit 0 ;;
esac
require_python
case "$mode" in
    all)
        prepare_train_data
        prepare_eval_sources
        verify_components
        ;;
    train) prepare_train_data ;;
    eval) prepare_eval_sources ;;
    verify) verify_components ;;
    *) usage; exit 2 ;;
esac

echo "Preparation complete (scope=$mode)."
