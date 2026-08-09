#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPTS_DIR/common_env.sh"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RUN_CONFIG="qwen31b_gen1024_qwen38bteacher_temp11_forwardbeta0_clip005"

accelerate launch \
    --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}" \
    --num_processes 8 \
    --gradient_accumulation_steps 1 \
    --main_process_port 12949 \
    opsd_train.py \
    --alg opd \
    --model_name_or_path "$REPO_ROOT/models/Qwen3-1.7B" \
    --teacher_model_name_or_path "$REPO_ROOT/models/Qwen3-8B" \
    --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir "$REPO_ROOT/outputs/opd" \
    --run_config "$RUN_CONFIG" \
    --max_steps 100 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --student_thinking False \
    --teacher_thinking False \
    --jsd_token_clip 0.05 \
    --wandb_project OPD \
    "$@"

if [[ "${AUTO_EVAL:-1}" == "1" ]]; then
    echo "Training complete; starting Qwen3-1.7B thinking-mode evaluation."
    EVAL_EXPERIMENT_DIR="$REPO_ROOT/outputs/opd/$RUN_CONFIG" \
        bash "$SCRIPTS_DIR/run_eval.sh" 1.7b
fi
