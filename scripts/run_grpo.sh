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
    grpo_train.py \
    --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd" \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path "$REPO_ROOT/models/Qwen3-4B" \
    --output_dir "$REPO_ROOT/outputs/grpo" \
    --run_config qwen4b-2epoch \
    --num_train_epochs 2 \
    --num_iterations 2 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 2048 \
    --max_completion_length 16000 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --logging_steps 10 \
    --save_steps 20 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project OPSD \
    "$@"
