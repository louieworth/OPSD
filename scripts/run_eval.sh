#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

usage() {
    echo "Usage: bash scripts/run_eval.sh <1.7b|4b|8b> [evaluation arguments...]" >&2
}

if (( $# == 0 )); then
    usage
    exit 2
fi

model="$1"
shift

case "$model" in
    1b|1.7b|qwen3-1.7b)
        model_path="$REPO_ROOT/models/Qwen3-1.7B"
        default_experiment_dir="$REPO_ROOT/outputs/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
        model_label="qwen3-1.7b"
        ;;
    4b|qwen3-4b)
        model_path="$REPO_ROOT/models/Qwen3-4B"
        default_experiment_dir="$REPO_ROOT/outputs/opsd/qwen34b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
        model_label="qwen3-4b"
        ;;
    8b|qwen3-8b)
        model_path="$REPO_ROOT/models/Qwen3-8B"
        default_experiment_dir="$REPO_ROOT/outputs/opsd/qwen38b_gen1024_fixteacher_temp11_forwardbeta0_clip006"
        model_label="qwen3-8b"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unsupported model: $model" >&2
        usage
        exit 2
        ;;
esac

export GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"

bash "$REPO_ROOT/eval/run_model_eval.sh" \
    "$model_path" \
    "${EVAL_EXPERIMENT_DIR:-$default_experiment_dir}" \
    "$model_label" \
    "$TP_SIZE" \
    "$@"
