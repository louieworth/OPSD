#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 BASE_MODEL EXPERIMENT_DIR MODEL_LABEL TENSOR_PARALLEL_SIZE [evaluate_math.py args...]" >&2
    exit 2
fi

BASE_MODEL="$1"
EXPERIMENT_DIR="$2"
MODEL_LABEL="$3"
TENSOR_PARALLEL_SIZE="$4"
shift 4
EXTRA_ARGS=("$@")

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$EVAL_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_env.sh"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

read -r -a DATASET_LIST <<< "${DATASETS:-aime24 aime25 beyond-aime hmmt25 amo-bench}"
read -r -a CHECKPOINT_LIST <<< "${CHECKPOINTS:-25 50 75 100}"

VAL_N="${VAL_N:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/outputs/eval/$MODEL_LABEL}"

for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    if [[ "$checkpoint" == "base" ]]; then
        checkpoint_args=()
        checkpoint_label="base"
    else
        checkpoint_dir="$EXPERIMENT_DIR/checkpoint-$checkpoint"
        checkpoint_args=(--checkpoint_dir "$checkpoint_dir")
        checkpoint_label="checkpoint-$checkpoint"
    fi

    for dataset in "${DATASET_LIST[@]}"; do
        output_file="$RESULT_ROOT/$checkpoint_label/$dataset.json"
        python "$EVAL_DIR/evaluate_math.py" \
            --base_model "$BASE_MODEL" \
            --data_root "$REPO_ROOT/data/eval" \
            --dataset "$dataset" \
            --val_n "$VAL_N" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --temperature 1.0 \
            --top_p 0.95 \
            --top_k -1 \
            --min_p 0.0 \
            --presence_penalty 0.0 \
            --enable_thinking \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --output_file "$output_file" \
            "${checkpoint_args[@]}" \
            "${EXTRA_ARGS[@]}"
    done
done

python "$EVAL_DIR/summarize_results.py" --result-root "$RESULT_ROOT"
