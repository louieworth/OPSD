#!/usr/bin/env bash

# Shared source/model/variant launcher for OPD and OPSD. Public entry points
# under scripts/{OPD,OPSD} intentionally contain no duplicated training flags.

if [[ "${DISTILL_COMMON_LOADED:-0}" == "1" ]]; then
    return 0
fi
DISTILL_COMMON_LOADED=1

DISTILL_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_SCRIPTS_DIR="$(cd -- "$DISTILL_LIB_DIR/.." && pwd)"
source "$DISTILL_SCRIPTS_DIR/common_env.sh"

distill_die() {
    echo "Distillation launcher error: $*" >&2
    return 1
}

distill_usage() {
    local source_name="$1"
    local variant="$2"
    local models="1.7b|4b"
    [[ "$source_name" == "opsd" ]] && models="1.7b|4b|8b"
    echo "Usage: bash scripts/${source_name^^}/${variant}.sh <$models> [training arguments...]" >&2
}

distill_validate_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        distill_die "$name must be a positive integer; got '$value'"
    fi
}

distill_reject_structural_overrides() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --alg|--alg=*|\
            --model_name_or_path|--model-name-or-path|--model_name_or_path=*|--model-name-or-path=*|\
            --teacher_model_name_or_path|--teacher-model-name-or-path|--teacher_model_name_or_path=*|--teacher-model-name-or-path=*|\
            --fixed_teacher|--fixed-teacher|--teacher_refine|--teacher-refine|\
            --teacher_thinking|--teacher-thinking|--teacher_thinking=*|--teacher-thinking=*|\
            --student_thinking|--student-thinking|--student_thinking=*|--student-thinking=*|\
            --top_k_loss|--top-k-loss|--top_k_loss=*|--top-k-loss=*|\
            --jsd_token_clip|--jsd-token-clip|--jsd_token_clip=*|--jsd-token-clip=*|\
            --top_k|--top-k|--top_k=*|--top-k=*|\
            --beta|--beta=*|--distillation_temperature|--distillation-temperature|\
            --distillation_temperature=*|--distillation-temperature=*|\
            --max_steps|--max-steps|--max_steps=*|--max-steps=*|\
            --policy_gradient_updates|--policy-gradient-updates|--policy_gradient_updates=*|--policy-gradient-updates=*|\
            --gradient_accumulation_steps|--gradient-accumulation-steps|--gradient_accumulation_steps=*|--gradient-accumulation-steps=*|\
            --per_device_train_batch_size|--per-device-train-batch-size|--per_device_train_batch_size=*|--per-device-train-batch-size=*|\
            --max_completion_length|--max-completion-length|--max_completion_length=*|--max-completion-length=*|\
            --max_refinement_length|--max-refinement-length|--max_refinement_length=*|--max-refinement-length=*|\
            --save_steps|--save-steps|--save_steps=*|--save-steps=*|\
            --output_dir|--output-dir|--output_dir=*|--output-dir=*|\
            --run_config|--run-config|--run_config=*|--run-config=*)
                distill_die "'$argument' is launcher-owned; use the documented DISTILL_* environment variable"
                return 1
                ;;
        esac
    done
}

distill_default_save_steps() {
    local updates="$1"
    local checkpoint_count
    for checkpoint_count in 4 3 2; do
        if (( updates % checkpoint_count == 0 )); then
            echo "$(( updates / checkpoint_count ))"
            return 0
        fi
    done
    echo "$updates"
}

distill_default_checkpoints() {
    local updates="$1"
    local save_steps="$2"
    local checkpoint
    local checkpoints=()
    for (( checkpoint = save_steps; checkpoint <= updates; checkpoint += save_steps )); do
        checkpoints+=("$checkpoint")
    done
    echo "${checkpoints[*]}"
}

distill_print_command() {
    printf 'TRAIN_CMD:'
    printf ' %q' "$@"
    printf '\n'
}

distill_select_model() {
    local source_name="$1"
    local requested_model="$2"

    case "${requested_model,,}" in
        1b|1.7b|qwen3-1.7b)
            DISTILL_MODEL_KEY="1.7b"
            DISTILL_MODEL_LABEL="qwen3-1.7b"
            DISTILL_RUN_MODEL_LABEL="qwen31b"
            DISTILL_MODEL_PATH="$REPO_ROOT/models/Qwen3-1.7B"
            DISTILL_PER_DEVICE_BATCH_SIZE=4
            ;;
        4b|qwen3-4b)
            DISTILL_MODEL_KEY="4b"
            DISTILL_MODEL_LABEL="qwen3-4b"
            DISTILL_RUN_MODEL_LABEL="qwen34b"
            DISTILL_MODEL_PATH="$REPO_ROOT/models/Qwen3-4B"
            DISTILL_PER_DEVICE_BATCH_SIZE=4
            ;;
        8b|qwen3-8b)
            if [[ "$source_name" == "opd" ]]; then
                distill_die "OPD does not support an 8B student; 8B is the fixed teacher"
                return 1
            fi
            DISTILL_MODEL_KEY="8b"
            DISTILL_MODEL_LABEL="qwen3-8b"
            DISTILL_RUN_MODEL_LABEL="qwen38b"
            DISTILL_MODEL_PATH="$REPO_ROOT/models/Qwen3-8B"
            DISTILL_PER_DEVICE_BATCH_SIZE=2
            ;;
        *)
            distill_die "unsupported $source_name student model '$requested_model'"
            return 1
            ;;
    esac
}

distill_launch_standard() {
    local source_name="$1"
    local variant="$2"
    local rollout_steps="$3"
    local policy_updates="$4"
    local completion_length="$5"
    local save_steps="$6"
    local run_config="$7"
    local output_root="$8"
    shift 8
    local -a extra_training_args=("$@")
    local -a source_args=()
    local -a variant_args=()
    local -a training_command
    local clip_value="0"

    case "$source_name" in
        opsd)
            source_args=(--fixed_teacher --teacher_thinking True)
            ;;
        opd)
            source_args=(
                --teacher_model_name_or_path "$REPO_ROOT/models/Qwen3-8B"
                --teacher_thinking False
            )
            ;;
        *)
            distill_die "unsupported teacher source '$source_name'"
            return 1
            ;;
    esac

    case "$variant" in
        vanilla)
            variant_args=(--top_k_loss 0 --jsd_token_clip 0)
            ;;
        top_k)
            # This is loss-distribution truncation from arXiv:2603.07079.
            # Rollout sampling independently remains --top_k 20 below.
            variant_args=(--top_k_loss 16 --jsd_token_clip 0)
            ;;
        clip)
            if [[ "$source_name" == "opsd" && "$DISTILL_MODEL_KEY" == "8b" ]]; then
                clip_value="0.06"
            else
                clip_value="0.05"
            fi
            variant_args=(--top_k_loss 0 --jsd_token_clip "$clip_value")
            ;;
        *)
            distill_die "unsupported non-TRD variant '$variant'"
            return 1
            ;;
    esac

    training_command=(
        accelerate launch
        --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
        --num_processes 8
        --gradient_accumulation_steps 1
        --main_process_port "${DISTILL_ACCELERATE_PORT:-12949}"
        "$REPO_ROOT/opsd_train.py"
        --alg "$source_name"
        --model_name_or_path "$DISTILL_MODEL_PATH"
        --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd"
        --learning_rate "${DISTILL_LEARNING_RATE:-5e-6}"
        --max_grad_norm "${DISTILL_MAX_GRAD_NORM:-0.1}"
        --per_device_train_batch_size "$DISTILL_PER_DEVICE_BATCH_SIZE"
        --gradient_checkpointing
        --gradient_accumulation_steps 1
        --output_dir "$output_root"
        --run_config "$run_config"
        --max_steps "$rollout_steps"
        --policy_gradient_updates "$policy_updates"
        --max_completion_length "$completion_length"
        --save_steps "$save_steps"
        --logging_steps "${DISTILL_LOGGING_STEPS:-2}"
        --attn_implementation flash_attention_2
        --torch_dtype bfloat16
        --max_length "${DISTILL_MAX_LENGTH:-20000}"
        --beta 0
        --use_vllm
        --vllm_mode colocate
        --vllm_gpu_memory_utilization "${DISTILL_STUDENT_VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
        --vllm_tensor_parallel_size 1
        --vllm_sync_frequency 1
        --use_peft
        --lora_r 64
        --lora_alpha 128
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
        --temperature 1.1
        --top_p 0.95
        --top_k 20
        --lmbda 1
        --student_thinking False
        --wandb_project "${source_name^^}"
        "${source_args[@]}"
        "${variant_args[@]}"
        "${extra_training_args[@]}"
    )

    if [[ "${DISTILL_DRY_RUN:-0}" == "1" ]]; then
        distill_print_command "${training_command[@]}"
        return 0
    fi

    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    "${training_command[@]}"
}

distill_launch() {
    local source_name="$1"
    local variant="$2"
    shift 2

    if (( $# == 0 )); then
        distill_usage "$source_name" "$variant"
        return 2
    fi
    local requested_model="$1"
    shift
    local -a extra_training_args=("$@")
    local rollout_steps
    local policy_updates
    local completion_length
    local refinement_length
    local save_steps
    local run_config
    local output_root="$REPO_ROOT/outputs/$source_name"
    local eval_experiment_dir
    local result_root
    local clip_suffix=""

    if [[ "$source_name" != "opd" && "$source_name" != "opsd" ]]; then
        distill_die "unsupported teacher source '$source_name'"
        return 1
    fi
    if [[ "$variant" != "vanilla" && "$variant" != "top_k" && "$variant" != "clip" && "$variant" != "trd" ]]; then
        distill_die "unsupported variant '$variant'"
        return 1
    fi

    distill_select_model "$source_name" "$requested_model"
    distill_reject_structural_overrides "${extra_training_args[@]}"

    if [[ "$variant" == "trd" ]]; then
        rollout_steps="${DISTILL_MAX_STEPS:-${TRD_MAX_STEPS:-100}}"
        policy_updates="${DISTILL_POLICY_GRADIENT_UPDATES:-${TRD_POLICY_GRADIENT_UPDATES:-100}}"
        completion_length="${DISTILL_MAX_COMPLETION_LENGTH:-${TRD_MAX_COMPLETION_LENGTH:-1024}}"
        refinement_length="${DISTILL_MAX_REFINEMENT_LENGTH:-${TRD_MAX_REFINEMENT_LENGTH:-1024}}"
    else
        rollout_steps="${DISTILL_MAX_STEPS:-100}"
        policy_updates="${DISTILL_POLICY_GRADIENT_UPDATES:-100}"
        completion_length="${DISTILL_MAX_COMPLETION_LENGTH:-1024}"
        refinement_length="$completion_length"
    fi

    distill_validate_positive_integer DISTILL_MAX_STEPS "$rollout_steps"
    distill_validate_positive_integer DISTILL_POLICY_GRADIENT_UPDATES "$policy_updates"
    distill_validate_positive_integer DISTILL_MAX_COMPLETION_LENGTH "$completion_length"
    if (( policy_updates > rollout_steps || rollout_steps % policy_updates != 0 )); then
        distill_die "DISTILL_POLICY_GRADIENT_UPDATES must divide DISTILL_MAX_STEPS and cannot exceed it"
        return 1
    fi

    if [[ "$variant" == "trd" ]]; then
        save_steps="${DISTILL_SAVE_STEPS:-${TRD_SAVE_STEPS:-$(distill_default_save_steps "$policy_updates")}}"
    else
        save_steps="${DISTILL_SAVE_STEPS:-$(distill_default_save_steps "$policy_updates")}"
    fi
    distill_validate_positive_integer DISTILL_SAVE_STEPS "$save_steps"
    if (( save_steps > policy_updates || policy_updates % save_steps != 0 )); then
        distill_die "DISTILL_SAVE_STEPS must divide the policy update count and cannot exceed it"
        return 1
    fi

    case "$variant" in
        vanilla) clip_suffix="fullvocab" ;;
        top_k) clip_suffix="loss_topk16" ;;
        clip)
            if [[ "$source_name" == "opsd" && "$DISTILL_MODEL_KEY" == "8b" ]]; then
                clip_suffix="clip006"
            else
                clip_suffix="clip005"
            fi
            ;;
        trd) clip_suffix="step0teacher" ;;
    esac
    run_config="${RUN_CONFIG:-${source_name}_${DISTILL_RUN_MODEL_LABEL}_${variant}_gen${completion_length}_n${rollout_steps}_u${policy_updates}_${clip_suffix}}"
    eval_experiment_dir="$output_root/$run_config"
    result_root="${RESULT_ROOT:-$REPO_ROOT/outputs/eval/$source_name/$variant/$run_config}"

    if [[ "$variant" == "trd" ]]; then
        export TRD_MAX_STEPS="$rollout_steps"
        export TRD_POLICY_GRADIENT_UPDATES="$policy_updates"
        export TRD_MAX_COMPLETION_LENGTH="$completion_length"
        export TRD_MAX_REFINEMENT_LENGTH="$refinement_length"
        export TRD_SAVE_STEPS="$save_steps"
        export RUN_CONFIG="$run_config"
        export RESULT_ROOT="$result_root"
        source "$DISTILL_LIB_DIR/trd_common.sh"
        if [[ "$source_name" == "opsd" ]]; then
            DISTILL_REFINER_MODEL="$DISTILL_MODEL_PATH"
        else
            DISTILL_REFINER_MODEL="$REPO_ROOT/models/Qwen3-8B"
        fi
        trd_launch \
            "$source_name" \
            "$DISTILL_MODEL_KEY" \
            "$DISTILL_MODEL_LABEL" \
            "$DISTILL_MODEL_PATH" \
            "$DISTILL_REFINER_MODEL" \
            "$DISTILL_PER_DEVICE_BATCH_SIZE" \
            "${extra_training_args[@]}"
        return
    fi

    distill_launch_standard \
        "$source_name" \
        "$variant" \
        "$rollout_steps" \
        "$policy_updates" \
        "$completion_length" \
        "$save_steps" \
        "$run_config" \
        "$output_root" \
        "${extra_training_args[@]}"

    if [[ "${DISTILL_DRY_RUN:-0}" == "1" ]]; then
        printf 'EVAL_EXPERIMENT_DIR: %s\n' "$eval_experiment_dir"
        printf 'RESULT_ROOT: %s\n' "$result_root"
        return 0
    fi

    if [[ "${AUTO_EVAL:-1}" == "1" ]]; then
        export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
        export TP_SIZE="${TP_SIZE:-8}"
        export CHECKPOINTS="${CHECKPOINTS:-$(distill_default_checkpoints "$policy_updates" "$save_steps")}" 
        export RESULT_ROOT="$result_root"
        EVAL_EXPERIMENT_DIR="$eval_experiment_dir" \
            bash "$DISTILL_SCRIPTS_DIR/run_eval.sh" "$DISTILL_MODEL_KEY"
    fi
}
