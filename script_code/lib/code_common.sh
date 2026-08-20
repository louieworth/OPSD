#!/usr/bin/env bash

if [[ "${CODE_COMMON_LOADED:-0}" == "1" ]]; then
    return 0
fi
CODE_COMMON_LOADED=1

CODE_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$CODE_LIB_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
if [[ -f "$CODE_ROOT/runtime.env" ]]; then
    # shellcheck disable=SC1091
    source "$CODE_ROOT/runtime.env"
fi
CODE_DATA_PATH="${CODE_DATA_PATH:-$REPO_ROOT/data/train/taco_code_clean}"
CODE_ATTN_IMPLEMENTATION="${CODE_ATTN_IMPLEMENTATION:-flash_attention_2}"
CODE_AUTO_EVAL="${CODE_AUTO_EVAL:-1}"
CODE_EVAL_EVERY_STEPS="${CODE_EVAL_EVERY_STEPS:-25}"
CODE_SFT_EVAL_STEPS="${CODE_SFT_EVAL_STEPS:-100}"
CODE_GRPO_EVAL_EVERY_STEPS="${CODE_GRPO_EVAL_EVERY_STEPS:-50}"
CODE_DELETE_TRAINED_MODELS_AFTER_EVAL="${CODE_DELETE_TRAINED_MODELS_AFTER_EVAL:-1}"
CODE_EVAL_RUNNER="${CODE_EVAL_RUNNER:-$CODE_ROOT/eval/run_code_eval.sh}"
CODE_SEED="${CODE_SEED:-42}"
CODE_AUTO_PREPARE="${CODE_AUTO_PREPARE:-1}"
CODE_PREPARE_SCRIPT="${CODE_PREPARE_SCRIPT:-$CODE_ROOT/prepare_code.sh}"
CODE_PREPARE_LOCK_FILE="${CODE_PREPARE_LOCK_FILE:-$REPO_ROOT/.cache/code_prepare.lock}"

code_die() {
    echo "Code experiment launcher error: $*" >&2
    return 2
}

code_print_command() {
    printf 'TRAIN_CMD:'
    printf ' %q' "$@"
    printf '\n'
}

code_select_model() {
    case "${1,,}" in
        1b|1.7b|qwen3-1.7b)
            CODE_MODEL_KEY="1.7b"
            CODE_MODEL_SCOPE="1B"
            CODE_MODEL_LABEL="qwen3-1.7b"
            CODE_MODEL_PATH="$REPO_ROOT/models/Qwen3-1.7B"
            ;;
        4b|qwen3-4b)
            CODE_MODEL_KEY="4b"
            CODE_MODEL_SCOPE="4B"
            CODE_MODEL_LABEL="qwen3-4b"
            CODE_MODEL_PATH="$REPO_ROOT/models/Qwen3-4B"
            ;;
        8b|qwen3-8b)
            CODE_MODEL_KEY="8b"
            CODE_MODEL_SCOPE="8B"
            CODE_MODEL_LABEL="qwen3-8b"
            CODE_MODEL_PATH="$REPO_ROOT/models/Qwen3-8B"
            ;;
        *) code_die "unsupported model '$1'"; return ;;
    esac
}

code_prepare_under_lock() {
    if bash "$CODE_PREPARE_SCRIPT" verify >/dev/null 2>&1; then
        echo "[code prepare] Existing train/eval preparation verified."
        return 0
    fi

    echo "[code prepare] Train/eval inputs are missing or incomplete; running prepare_code.sh all..."
    bash "$CODE_PREPARE_SCRIPT" all
}

code_ensure_prepared() {
    if [[ "${CODE_DRY_RUN:-0}" == "1" || "$CODE_AUTO_PREPARE" == "0" ]]; then
        return 0
    fi
    [[ "$CODE_AUTO_PREPARE" == "1" ]] || {
        code_die "CODE_AUTO_PREPARE must be 0 or 1"
        return
    }
    [[ -f "$CODE_PREPARE_SCRIPT" ]] || {
        code_die "preparation script not found: $CODE_PREPARE_SCRIPT"
        return
    }

    mkdir -p "$(dirname -- "$CODE_PREPARE_LOCK_FILE")"
    if command -v flock >/dev/null 2>&1; then
        if ! (
            exec 9>"$CODE_PREPARE_LOCK_FILE"
            flock 9
            code_prepare_under_lock
        ); then
            code_die "automatic code train/eval preparation failed"
            return
        fi
    else
        echo "[code prepare] Warning: flock is unavailable; preparing without a host-local lock." >&2
        code_prepare_under_lock || {
            code_die "automatic code train/eval preparation failed"
            return
        }
    fi

    # The first preparation creates runtime.env after this library was sourced,
    # so load it again before training/evaluation starts in the current process.
    if [[ -f "$CODE_ROOT/runtime.env" ]]; then
        # shellcheck disable=SC1091
        source "$CODE_ROOT/runtime.env"
    fi
}

code_require_inputs() {
    if [[ "${CODE_DRY_RUN:-0}" == "1" ]]; then
        return
    fi
    code_ensure_prepared
    [[ -d "$CODE_MODEL_PATH" ]] || code_die "model not found: $CODE_MODEL_PATH"
    [[ -d "$CODE_DATA_PATH" ]] || code_die "prepared TACO data not found: $CODE_DATA_PATH"
}

code_remove_evaluated_checkpoint() {
    local experiment_dir="$1"
    local checkpoint_path="$2"
    local experiment_real checkpoint_parent checkpoint_name

    experiment_real="$(cd -- "$experiment_dir" && pwd -P)"
    checkpoint_parent="$(cd -- "$(dirname -- "$checkpoint_path")" && pwd -P)"
    checkpoint_name="$(basename -- "$checkpoint_path")"
    [[ "$checkpoint_parent" == "$experiment_real" ]] || {
        code_die "refusing to delete checkpoint outside experiment directory: $checkpoint_path"
        return
    }
    [[ "$checkpoint_name" =~ ^checkpoint-[1-9][0-9]*$ && -d "$checkpoint_path" ]] || {
        code_die "refusing to delete invalid checkpoint path: $checkpoint_path"
        return
    }
    rm -rf -- "$checkpoint_path"
    echo "Deleted evaluated checkpoint: $checkpoint_path"
}

code_remove_final_model_weights() {
    local experiment_dir="$1"
    local experiment_real repo_real
    local -a model_files=()

    experiment_real="$(cd -- "$experiment_dir" && pwd -P)"
    repo_real="$(cd -- "$REPO_ROOT" && pwd -P)"
    case "$experiment_real" in
        "$repo_real"/outputs/code/*) ;;
        *)
            code_die "refusing to delete model weights outside outputs/code: $experiment_real"
            return
            ;;
    esac

    mapfile -d '' -t model_files < <(
        find "$experiment_real" -mindepth 1 -maxdepth 1 -type f \
            \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \
               -o -name 'model.safetensors' -o -name 'model-*.safetensors' \
               -o -name 'model.safetensors.index.json' \
               -o -name 'pytorch_model.bin' -o -name 'pytorch_model-*.bin' \
               -o -name 'pytorch_model.bin.index.json' \) -print0
    )
    if (( ${#model_files[@]} > 0 )); then
        rm -f -- "${model_files[@]}"
        echo "Deleted final trained model weights from: $experiment_real"
    fi
}

code_latest_checkpoint() {
    local experiment_dir="$1"
    [[ -d "$experiment_dir" ]] || return 0
    local -a checkpoints=()
    mapfile -t checkpoints < <(
        find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
            -printf '%p\n' | sort -V
    )
    if (( ${#checkpoints[@]} > 0 )); then
        printf '%s\n' "${checkpoints[-1]}"
    fi
}

code_eval_checkpoint() {
    local experiment_dir="$1"
    local checkpoint_path="$2"
    local label="$3"
    local final_checkpoint="${4:-0}"
    local checkpoint_name marker

    checkpoint_name="$(basename -- "$checkpoint_path")"
    marker="$experiment_dir/.code_eval_complete_${checkpoint_name}"
    if [[ ! -f "$marker" ]]; then
        bash "$CODE_EVAL_RUNNER" \
            "$checkpoint_path" \
            "${label}_${checkpoint_name}"
        touch "$marker"
    else
        echo "Evaluation already complete: $checkpoint_name"
    fi

    if [[ "$final_checkpoint" == "1" && "$CODE_DELETE_TRAINED_MODELS_AFTER_EVAL" == "1" ]]; then
        code_remove_evaluated_checkpoint "$experiment_dir" "$checkpoint_path"
        code_remove_final_model_weights "$experiment_dir"
    fi
}

code_replace_command_option() {
    local option="$1"
    local value="$2"
    local array_name="$3"
    local -n command_ref="$array_name"
    local index
    for (( index = 0; index < ${#command_ref[@]}; index += 1 )); do
        if [[ "${command_ref[$index]}" == "$option" ]]; then
            (( index + 1 < ${#command_ref[@]} )) || {
                code_die "missing value for command option $option"
                return
            }
            command_ref[$((index + 1))]="$value"
            return 0
        fi
    done
    code_die "command option not found: $option"
}

code_execute_command_segment() {
    local schedule_kind="$1"
    local target_step="$2"
    local resume_checkpoint="$3"
    shift 3
    local -a segment_command=("$@")

    case "$schedule_kind" in
        trainer_steps)
            code_replace_command_option --max_steps "$target_step" segment_command
            ;;
        policy_updates)
            segment_command+=(--stop_after_policy_updates "$target_step")
            ;;
        *)
            code_die "unsupported segmented schedule: $schedule_kind"
            return
            ;;
    esac
    if [[ -n "$resume_checkpoint" ]]; then
        segment_command+=(--resume_from_checkpoint "$resume_checkpoint")
    fi
    "${segment_command[@]}"
}

code_execute_trd_segment() {
    local target_step="$1"
    local resume_checkpoint="$2"
    local algorithm="$3"
    local model_key="$4"
    local model_label="$5"
    local student_model="$6"
    local refiner_model="$7"
    local per_device_batch_size="$8"
    shift 8
    local -a segment_args=("$@" --stop_after_policy_updates "$target_step")
    if [[ -n "$resume_checkpoint" ]]; then
        segment_args+=(--resume_from_checkpoint "$resume_checkpoint")
    fi
    trd_launch \
        "$algorithm" \
        "$model_key" \
        "$model_label" \
        "$student_model" \
        "$refiner_model" \
        "$per_device_batch_size" \
        "${segment_args[@]}"
}

code_run_segmented_training() {
    local total_steps="$1"
    local eval_interval="$2"
    local experiment_dir="$3"
    local label="$4"
    local runner="$5"
    shift 5
    local -a runner_args=("$@")
    local target_step checkpoint_path latest_checkpoint latest_step marker

    [[ "$total_steps" =~ ^[1-9][0-9]*$ && "$eval_interval" =~ ^[1-9][0-9]*$ ]] || {
        code_die "segmented training steps and interval must be positive integers"
        return
    }
    (( total_steps % eval_interval == 0 )) || {
        code_die "evaluation interval $eval_interval must divide total steps $total_steps"
        return
    }

    if [[ "$CODE_AUTO_EVAL" != "1" ]]; then
        latest_checkpoint="$(code_latest_checkpoint "$experiment_dir")"
        "$runner" "$total_steps" "$latest_checkpoint" "${runner_args[@]}"
        return
    fi

    marker="$experiment_dir/.code_eval_complete_checkpoint-${total_steps}"
    if [[ -f "$marker" && ! -d "$experiment_dir/checkpoint-${total_steps}" ]]; then
        echo "Training and final evaluation already complete: $label"
        return 0
    fi

    for (( target_step = eval_interval; target_step <= total_steps; target_step += eval_interval )); do
        checkpoint_path="$experiment_dir/checkpoint-${target_step}"
        marker="$experiment_dir/.code_eval_complete_checkpoint-${target_step}"
        latest_checkpoint="$(code_latest_checkpoint "$experiment_dir")"
        latest_step=0
        if [[ -n "$latest_checkpoint" ]]; then
            latest_step="${latest_checkpoint##*-}"
            [[ "$latest_step" =~ ^[1-9][0-9]*$ ]] || {
                code_die "invalid checkpoint name: $latest_checkpoint"
                return
            }
        fi

        if [[ -f "$marker" && "$latest_step" -gt "$target_step" ]]; then
            continue
        fi
        if [[ -f "$marker" && "$latest_step" -eq 0 ]]; then
            code_die "cannot continue after checkpoint-$target_step evaluation without a resumable checkpoint"
            return
        fi
        if [[ ! -d "$checkpoint_path" ]]; then
            if (( latest_step > target_step )); then
                code_die "checkpoint-$target_step was rotated before its evaluation completed"
                return
            fi
            if (( latest_step == 0 && target_step > eval_interval )); then
                code_die "cannot continue segmented training without a resumable checkpoint"
                return
            fi
            echo "Training segment: $label -> step $target_step/$total_steps"
            "$runner" "$target_step" "$latest_checkpoint" "${runner_args[@]}"
        fi
        [[ -d "$checkpoint_path" ]] || {
            code_die "training segment did not create expected checkpoint: $checkpoint_path"
            return
        }

        echo "Evaluating checkpoint: $checkpoint_path"
        code_eval_checkpoint \
            "$experiment_dir" \
            "$checkpoint_path" \
            "$label" \
            "$(( target_step == total_steps ))"
    done
}

code_maybe_eval() {
    local experiment_dir="$1"
    local label="$2"
    local eval_every_steps="${3:-$CODE_EVAL_EVERY_STEPS}"
    [[ "$CODE_AUTO_EVAL" == "1" ]] || return 0
    [[ "$eval_every_steps" =~ ^[1-9][0-9]*$ ]] || {
        code_die "evaluation interval must be a positive integer"
        return
    }
    [[ -d "$experiment_dir" ]] || {
        code_die "cannot evaluate missing experiment directory: $experiment_dir"
        return
    }

    local checkpoint_name checkpoint_path checkpoint_step
    local evaluated=0
    local -a checkpoint_names=()
    mapfile -t checkpoint_names < <(
        find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
            -printf '%f\n' | sort -V
    )
    for checkpoint_name in "${checkpoint_names[@]}"; do
        checkpoint_step="${checkpoint_name#checkpoint-}"
        [[ "$checkpoint_step" =~ ^[1-9][0-9]*$ ]] || continue
        (( checkpoint_step % eval_every_steps == 0 )) || continue
        checkpoint_path="$experiment_dir/$checkpoint_name"
        if [[ "${CODE_EVAL_DRY_RUN:-0}" == "1" ]]; then
            printf 'EVAL_CMD: bash %q %q %q\n' \
                "$CODE_EVAL_RUNNER" \
                "$checkpoint_path" \
                "${label}_${checkpoint_name}"
            if [[ "$CODE_DELETE_TRAINED_MODELS_AFTER_EVAL" == "1" ]]; then
                printf 'DELETE_CHECKPOINT_AFTER_SUCCESS: %q\n' "$checkpoint_path"
            fi
        else
            bash "$CODE_EVAL_RUNNER" \
                "$checkpoint_path" \
                "${label}_${checkpoint_name}"
            if [[ "$CODE_DELETE_TRAINED_MODELS_AFTER_EVAL" == "1" ]]; then
                code_remove_evaluated_checkpoint "$experiment_dir" "$checkpoint_path"
            fi
        fi
        evaluated=$((evaluated + 1))
    done
    if (( evaluated == 0 )); then
        code_die "no checkpoint matching the ${eval_every_steps}-step evaluation interval under $experiment_dir"
        return
    fi
    if [[ "$CODE_DELETE_TRAINED_MODELS_AFTER_EVAL" == "1" && "${CODE_EVAL_DRY_RUN:-0}" != "1" ]]; then
        code_remove_final_model_weights "$experiment_dir"
    fi
}

code_launch_trd() {
    local source="$1"
    local requested_model="$2"
    shift 2
    code_select_model "$requested_model"
    if [[ "$source" != "opsd" && "$source" != "opd" ]]; then
        code_die "TRD source must be opsd or opd"
        return
    fi
    if [[ "$source" == "opd" && "$CODE_MODEL_KEY" == "8b" ]]; then
        code_die "OPD supports 1.7B and 4B students only; Qwen3-8B is the fixed teacher"
        return
    fi
    code_require_inputs

    local teacher_prompt_cap=2048
    local teacher_context_cap=6144
    local refiner_model="$REPO_ROOT/models/Qwen3-8B"
    if [[ "$source" == "opsd" ]]; then
        teacher_prompt_cap=8192
        teacher_context_cap=12288
        refiner_model="$CODE_MODEL_PATH"
    fi

    local rollout_steps="${CODE_TRD_MAX_STEPS:-400}"
    local policy_updates="${CODE_TRD_POLICY_GRADIENT_UPDATES:-100}"
    local completion_length="${CODE_TRD_MAX_COMPLETION_LENGTH:-1024}"
    local refinement_length="${CODE_TRD_MAX_REFINEMENT_LENGTH:-1024}"
    local response_reserve="$completion_length"
    if (( refinement_length > response_reserve )); then
        response_reserve="$refinement_length"
    fi
    local run_config="code_${source}_${CODE_MODEL_LABEL}_trd_gen${completion_length}_n${rollout_steps}_u${policy_updates}"
    local output_root="$REPO_ROOT/outputs/code/$source/trd"

    export DISTILL_DRY_RUN="${CODE_DRY_RUN:-0}"
    export AUTO_EVAL=0
    export TRD_TASK_TYPE=code
    export TRD_TRAIN_DATA_PATH="$CODE_DATA_PATH"
    export TRD_OUTPUT_ROOT="$output_root"
    export TRD_WANDB_PROJECT="CODE_${source^^}_TRD"
    export TRD_ATTN_IMPLEMENTATION="$CODE_ATTN_IMPLEMENTATION"
    export TRD_MAX_STEPS="$rollout_steps"
    export TRD_POLICY_GRADIENT_UPDATES="$policy_updates"
    export TRD_MAX_COMPLETION_LENGTH="$completion_length"
    export TRD_MAX_REFINEMENT_LENGTH="$refinement_length"
    export TRD_SAVE_STEPS="$CODE_EVAL_EVERY_STEPS"
    export TRD_MAX_LENGTH="${CODE_TRD_MAX_LENGTH:-$((2048 + response_reserve))}"
    export TRD_REFINEMENT_MAX_MODEL_LEN="${CODE_TRD_REFINEMENT_MAX_MODEL_LEN:-$teacher_context_cap}"
    export TRD_SERVER_LOG_DIR="${CODE_TRD_SERVER_LOG_DIR:-$REPO_ROOT/outputs/code/trd/server-logs}"
    export RUN_CONFIG="$run_config"

    # Reuse the same 4 trainer GPU + 4 rewrite-server GPU lifecycle as the
    # math scripts. Only the task adapter, dataset, and code context caps differ.
    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/lib/trd_common.sh"
    local -a trd_args=(
        --trajectory_mode student \
        --student_prompt_max_length 2048 \
        --reference_solution_max_length 4096 \
        --teacher_prompt_max_length "$teacher_prompt_cap" \
        --teacher_context_max_length "$teacher_context_cap" \
        --save_total_limit 1 \
        --skip_final_model_save True \
        "$@"
    )

    if [[ "${CODE_DRY_RUN:-0}" == "1" ]]; then
        trd_launch \
            "$source" \
            "$CODE_MODEL_KEY" \
            "$CODE_MODEL_LABEL" \
            "$CODE_MODEL_PATH" \
            "$refiner_model" \
            "${CODE_TRD_PER_DEVICE_BATCH_SIZE:-1}" \
            "${trd_args[@]}"
        return
    fi

    code_run_segmented_training \
        "$policy_updates" \
        "$CODE_EVAL_EVERY_STEPS" \
        "$output_root/$run_config" \
        "$run_config" \
        code_execute_trd_segment \
        "$source" \
        "$CODE_MODEL_KEY" \
        "$CODE_MODEL_LABEL" \
        "$CODE_MODEL_PATH" \
        "$refiner_model" \
        "${CODE_TRD_PER_DEVICE_BATCH_SIZE:-1}" \
        "${trd_args[@]}"
}

code_launch_baseline() {
    local method="$1"
    local requested_model="$2"
    shift 2
    code_select_model "$requested_model"
    if [[ "$method" != "base" && "$method" != "sft" && "$method" != "grpo" ]]; then
        code_die "unsupported baseline '$method'"
        return
    fi
    code_require_inputs

    local output="$REPO_ROOT/outputs/code/$method/$CODE_MODEL_LABEL"
    local run_config="code_${method}_${CODE_MODEL_LABEL}"
    local -a command
    case "$method" in
        base)
            command=(bash "$CODE_ROOT/eval/run_code_eval.sh" "$CODE_MODEL_PATH" "$run_config")
            ;;
        sft)
            command=(
                accelerate launch
                --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
                --num_processes 8
                --gradient_accumulation_steps 4
                --main_process_port "${CODE_ACCELERATE_PORT:-19346}"
                "$CODE_ROOT/code_sft_train.py"
                --model_name_or_path "$CODE_MODEL_PATH"
                --train_dataset_path "$CODE_DATA_PATH"
                --output_dir "$output"
                --learning_rate 5e-6
                --per_device_train_batch_size 1
                --gradient_accumulation_steps 4
                --max_steps 100
                --max_length 6144
                --gradient_checkpointing
                --bf16 True
                --torch_dtype bfloat16
                --attn_implementation "$CODE_ATTN_IMPLEMENTATION"
                --optim adamw_torch_fused
                --use_peft
                --lora_r 64
                --lora_alpha 128
                --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
                --logging_steps 5
                --save_steps "$CODE_SFT_EVAL_STEPS"
                --save_total_limit 1
                --skip_final_model_save True
                --eval_strategy no
                --seed "$CODE_SEED"
                "$@"
            )
            ;;
        grpo)
            command=(
                accelerate launch
                --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
                --num_processes 8
                --gradient_accumulation_steps 4
                --main_process_port "${CODE_ACCELERATE_PORT:-19346}"
                "$CODE_ROOT/code_grpo_train.py"
                --model_name_or_path "$CODE_MODEL_PATH"
                --train_dataset_path "$CODE_DATA_PATH"
                --output_dir "$output"
                --run_config "$run_config"
                --learning_rate 5e-6
                --per_device_train_batch_size 1
                --gradient_accumulation_steps 4
                --max_steps 500
                --num_iterations 2
                --max_prompt_length 2048
                --max_completion_length 4096
                --num_generations 8
                --temperature 1.2
                --beta 0
                --loss_type grpo
                --scale_rewards group
                --gradient_checkpointing
                --bf16 True
                --torch_dtype bfloat16
                --attn_implementation "$CODE_ATTN_IMPLEMENTATION"
                --optim adamw_torch_fused
                --use_peft
                --lora_r 64
                --lora_alpha 128
                --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
                --use_vllm
                --vllm_mode colocate
                --vllm_gpu_memory_utilization "${CODE_GRPO_VLLM_MEMORY:-0.35}"
                --logging_steps 10
                --save_steps "$CODE_GRPO_EVAL_EVERY_STEPS"
                --save_total_limit 1
                --skip_final_model_save True
                --seed "$CODE_SEED"
                "$@"
            )
            ;;
    esac

    if [[ "${CODE_DRY_RUN:-0}" == "1" ]]; then
        code_print_command "${command[@]}"
        return
    fi
    case "$method" in
        base) "${command[@]}" ;;
        sft)
            code_run_segmented_training \
                100 "$CODE_SFT_EVAL_STEPS" "$output" "$run_config" \
                code_execute_command_segment trainer_steps \
                "${command[@]}"
            ;;
        grpo)
            code_run_segmented_training \
                500 "$CODE_GRPO_EVAL_EVERY_STEPS" "$output/$run_config" "$run_config" \
                code_execute_command_segment trainer_steps \
                "${command[@]}"
            ;;
    esac
}

code_launch_kd() {
    local source="$1"
    local variant="$2"
    local requested_model="$3"
    shift 3
    code_select_model "$requested_model"
    if [[ "$source" != "opsd" && "$source" != "opd" ]]; then
        code_die "source must be opsd or opd"
        return
    fi
    if [[ "$source" == "opd" && "$CODE_MODEL_KEY" == "8b" ]]; then
        code_die "OPD supports 1.7B and 4B students only; Qwen3-8B is the fixed teacher"
        return
    fi
    case "$variant" in vanilla|clip|top_k|trd|skd) ;; *) code_die "unsupported KD variant '$variant'"; return ;; esac
    if [[ "$variant" == "trd" ]]; then
        code_launch_trd "$source" "$requested_model" "$@"
        return
    fi
    code_require_inputs

    local teacher_prompt_cap=2048
    local teacher_context_cap=6144
    local -a source_args variant_args
    if [[ "$source" == "opsd" ]]; then
        teacher_prompt_cap=8192
        teacher_context_cap=12288
        source_args=(--fixed_teacher --teacher_thinking True)
    else
        source_args=(
            --teacher_model_name_or_path "$REPO_ROOT/models/Qwen3-8B"
            --teacher_thinking False
        )
    fi
    case "$variant" in
        vanilla) variant_args=(--trajectory_mode student --top_k_loss 0 --jsd_token_clip 0) ;;
        top_k) variant_args=(--trajectory_mode student --top_k_loss 16 --jsd_token_clip 0) ;;
        clip)
            local clip_value=0.05
            [[ "$source" == "opsd" && "$CODE_MODEL_KEY" == "8b" ]] && clip_value=0.06
            variant_args=(--trajectory_mode student --top_k_loss 0 --jsd_token_clip "$clip_value")
            ;;
        skd)
            variant_args=(
                --trajectory_mode skd
                --top_k_loss 0
                --jsd_token_clip 0
                --skd_draft_length 5
                --skd_accept_top_k 25
                --skd_correction_temperature 0.2
                --skd_correction_top_p 1.0
            )
            ;;
    esac

    local run_config="code_${source}_${CODE_MODEL_LABEL}_${variant}_4k_n400_u100"
    local output="$REPO_ROOT/outputs/code/$source/$variant"
    local -a command=(
        accelerate launch
        --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
        --num_processes 8
        --gradient_accumulation_steps 1
        --main_process_port "${CODE_ACCELERATE_PORT:-12949}"
        "$REPO_ROOT/opsd_train.py"
        --task_type code
        --alg "$source"
        --model_name_or_path "$CODE_MODEL_PATH"
        --train_dataset_path "$CODE_DATA_PATH"
        --output_dir "$output"
        --run_config "$run_config"
        --learning_rate 5e-6
        --max_grad_norm 0.1
        --per_device_train_batch_size 1
        --gradient_accumulation_steps 1
        --max_steps 400
        --policy_gradient_updates 100
        --max_completion_length 4096
        --max_length 6144
        --student_prompt_max_length 2048
        --reference_solution_max_length 4096
        --teacher_prompt_max_length "$teacher_prompt_cap"
        --teacher_context_max_length "$teacher_context_cap"
        --gradient_checkpointing
        --torch_dtype bfloat16
        --attn_implementation "$CODE_ATTN_IMPLEMENTATION"
        --optim adamw_torch_fused
        --beta 0
        --distillation_temperature 1.0
        --use_peft
        --lora_r 64
        --lora_alpha 128
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
        --temperature 1.1
        --top_p 0.95
        --top_k 20
        --lmbda 1
        --student_thinking False
        --logging_steps 2
        --save_steps "$CODE_EVAL_EVERY_STEPS"
        --save_total_limit 1
        --skip_final_model_save True
        --seed "$CODE_SEED"
        --wandb_project "CODE_${source^^}"
        "${source_args[@]}"
        "${variant_args[@]}"
        "$@"
    )
    if [[ "${CODE_DRY_RUN:-0}" == "1" ]]; then
        code_print_command "${command[@]}"
        return
    fi
    code_run_segmented_training \
        100 "$CODE_EVAL_EVERY_STEPS" "$output/$run_config" "$run_config" \
        code_execute_command_segment policy_updates \
        "${command[@]}"
}
