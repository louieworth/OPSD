#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

accelerate launch \
    --config_file "$REPO_ROOT/accelerate.yaml" \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    sft_train.py \
    --model_name_or_path "$REPO_ROOT/models/Qwen3-4B" \
    --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd" \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir "$REPO_ROOT/outputs/sft/qwen34b-4epochs-30k" \
    --num_train_epochs 4 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 16000 \
    --logging_steps 5 \
    --save_steps 20 \
    "$@"
