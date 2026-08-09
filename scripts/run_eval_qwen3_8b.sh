#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"

bash "$REPO_ROOT/eval/run_model_eval.sh" \
    "$REPO_ROOT/models/Qwen3-8B" \
    "${EVAL_EXPERIMENT_DIR:-$REPO_ROOT/outputs/opsd/qwen38b_gen1024_fixteacher_temp11_forwardbeta0_clip006}" \
    qwen3-8b \
    "$TP_SIZE" \
    "$@"
