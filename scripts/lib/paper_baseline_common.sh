#!/usr/bin/env bash

# Shared model routing and paper-locked configurations for the SFT and GRPO
# baselines in arXiv:2601.18734v3, Appendix B, Tables 6 and 7.

PAPER_BASELINE_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_BASELINE_SCRIPTS_DIR="$(cd -- "$PAPER_BASELINE_LIB_DIR/.." && pwd)"
source "$PAPER_BASELINE_SCRIPTS_DIR/common_env.sh"
# shellcheck disable=SC1091
source "$PAPER_BASELINE_LIB_DIR/math_segment_common.sh"

paper_baseline_print_command() {
    printf 'TRAIN_CMD:'
    printf ' %q' "$@"
    printf '\n'
}

paper_baseline_reject_paper_overrides() {
    local argument option
    for argument in "$@"; do
        [[ "$argument" == --* ]] || continue
        option="${argument%%=*}"
        option="${option#--}"
        option="${option//-/_}"
        case "$option" in
            num_processes|model_name_or_path|train_dataset_path|learning_rate|\
            per_device_train_batch_size|gradient_accumulation_steps|max_steps|\
            num_train_epochs|num_iterations|max_prompt_length|max_completion_length|\
            max_length|num_generations|temperature|beta|loss_type|scale_rewards|\
            use_vllm|vllm_mode|use_peft|lora_r|lora_alpha|lora_target_modules|\
            bf16|fp16|torch_dtype|attn_implementation|optim|gradient_checkpointing)
                echo "Cannot override paper-locked baseline option: $argument" >&2
                return 2
                ;;
        esac
    done
}

paper_baseline_select_model() {
    local model_scope="$1"

    case "${model_scope^^}" in
        1B)
            PAPER_MODEL_PATH="$REPO_ROOT/models/Qwen3-1.7B"
            PAPER_MODEL_KEY="1.7b"
            PAPER_RUN_MODEL_LABEL="qwen31b"
            ;;
        4B)
            PAPER_MODEL_PATH="$REPO_ROOT/models/Qwen3-4B"
            PAPER_MODEL_KEY="4b"
            PAPER_RUN_MODEL_LABEL="qwen34b"
            ;;
        8B)
            PAPER_MODEL_PATH="$REPO_ROOT/models/Qwen3-8B"
            PAPER_MODEL_KEY="8b"
            PAPER_RUN_MODEL_LABEL="qwen38b"
            ;;
        *)
            echo "Unsupported OPSD baseline model scope: $model_scope" >&2
            return 2
            ;;
    esac
}

paper_baseline_launch() {
    local recipe="$1"
    local model_scope="$2"
    shift 2
    local -a extra_args=("$@")
    local -a training_command
    local total_steps eval_interval experiment_dir result_root run_config

    paper_baseline_reject_paper_overrides "${extra_args[@]}"
    paper_baseline_select_model "$model_scope"

    case "$recipe" in
        grpo)
            total_steps=500
            eval_interval="${PAPER_GRPO_EVAL_STEPS:-20}"
            run_config="paper-v3-grpo-${PAPER_RUN_MODEL_LABEL}-500steps"
            experiment_dir="$REPO_ROOT/outputs/grpo/$run_config"
            result_root="${RESULT_ROOT:-$REPO_ROOT/outputs/eval/baselines/grpo/$run_config}"
            training_command=(
                accelerate launch
                --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
                --num_processes 8
                --gradient_accumulation_steps 4
                --main_process_port "${PAPER_BASELINE_ACCELERATE_PORT:-19346}"
                "$REPO_ROOT/grpo_train.py"
                --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd"
                --learning_rate 5e-6
                --per_device_train_batch_size 1
                --gradient_accumulation_steps 4
                --model_name_or_path "$PAPER_MODEL_PATH"
                --output_dir "$REPO_ROOT/outputs/grpo"
                --run_config "$run_config"
                --max_steps 500
                --num_iterations 2
                --gradient_checkpointing
                --bf16 True
                --torch_dtype bfloat16
                --attn_implementation flash_attention_2
                --optim adamw_torch_fused
                --lora_r 64
                --lora_alpha 128
                --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
                --max_prompt_length 2048
                --max_completion_length 16000
                --num_generations 8
                --temperature 1.2
                --use_vllm
                --use_peft
                --vllm_mode colocate
                --logging_steps 10
                --save_steps "$eval_interval"
                --save_total_limit 1
                --skip_final_model_save True
                --beta 0.0
                --loss_type grpo
                --scale_rewards group
                --wandb_project OPSD
                "${extra_args[@]}"
            )
            ;;
        sft)
            total_steps=100
            eval_interval="${PAPER_SFT_EVAL_STEPS:-100}"
            run_config="paper-v3-sft-${PAPER_RUN_MODEL_LABEL}-100steps"
            experiment_dir="$REPO_ROOT/outputs/sft/$run_config"
            result_root="${RESULT_ROOT:-$REPO_ROOT/outputs/eval/baselines/sft/$run_config}"
            training_command=(
                accelerate launch
                --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
                --num_processes 8
                --gradient_accumulation_steps 4
                --main_process_port "${PAPER_BASELINE_ACCELERATE_PORT:-19346}"
                "$REPO_ROOT/sft_train.py"
                --model_name_or_path "$PAPER_MODEL_PATH"
                --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd"
                --learning_rate 5e-6
                --per_device_train_batch_size 1
                --gradient_accumulation_steps 4
                --output_dir "$experiment_dir"
                --max_steps 100
                --gradient_checkpointing
                --bf16 True
                --torch_dtype bfloat16
                --attn_implementation flash_attention_2
                --optim adamw_torch_fused
                --use_peft
                --lora_r 64
                --lora_alpha 128
                --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
                --max_length 16000
                --logging_steps 5
                --save_steps "$eval_interval"
                --save_total_limit 1
                --skip_final_model_save True
                "${extra_args[@]}"
            )
            ;;
        *)
            echo "Unsupported paper baseline recipe: $recipe" >&2
            return 2
            ;;
    esac

    if [[ "${DISTILL_DRY_RUN:-0}" == "1" ]]; then
        paper_baseline_print_command "${training_command[@]}"
        return 0
    fi

    [[ -d "$PAPER_MODEL_PATH" ]] || {
        echo "Model directory not found: $PAPER_MODEL_PATH" >&2
        return 1
    }
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    cd "$REPO_ROOT"
    math_run_segmented_training \
        "$total_steps" \
        "$eval_interval" \
        "$experiment_dir" \
        "$PAPER_MODEL_KEY" \
        "$result_root" \
        "$run_config" \
        math_execute_command_segment \
        trainer_steps \
        "${training_command[@]}"
}
