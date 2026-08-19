#!/usr/bin/env bash

if [[ "${MATH_SEGMENT_COMMON_LOADED:-0}" == "1" ]]; then
    return 0
fi
MATH_SEGMENT_COMMON_LOADED=1

MATH_SEGMENT_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATH_SEGMENT_SCRIPTS_DIR="$(cd -- "$MATH_SEGMENT_LIB_DIR/.." && pwd)"
source "$MATH_SEGMENT_SCRIPTS_DIR/common_env.sh"

MATH_DELETE_TRAINED_MODELS_AFTER_EVAL="${MATH_DELETE_TRAINED_MODELS_AFTER_EVAL:-1}"
MATH_EVAL_RUNNER="${MATH_EVAL_RUNNER:-$MATH_SEGMENT_SCRIPTS_DIR/run_eval.sh}"

math_segment_die() {
    echo "Math segmented launcher error: $*" >&2
    return 1
}

math_latest_checkpoint() {
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

math_remove_evaluated_checkpoint() {
    local experiment_dir="$1"
    local checkpoint_path="$2"
    local experiment_real checkpoint_parent checkpoint_name repo_real

    experiment_real="$(cd -- "$experiment_dir" && pwd -P)"
    checkpoint_parent="$(cd -- "$(dirname -- "$checkpoint_path")" && pwd -P)"
    checkpoint_name="$(basename -- "$checkpoint_path")"
    repo_real="$(cd -- "$REPO_ROOT" && pwd -P)"
    [[ "$checkpoint_parent" == "$experiment_real" ]] || {
        math_segment_die "refusing to delete checkpoint outside experiment directory: $checkpoint_path"
        return
    }
    case "$experiment_real" in
        "$repo_real"/outputs/*) ;;
        *)
            math_segment_die "refusing to delete checkpoint outside outputs: $checkpoint_path"
            return
            ;;
    esac
    [[ "$checkpoint_name" =~ ^checkpoint-[1-9][0-9]*$ && -d "$checkpoint_path" ]] || {
        math_segment_die "refusing to delete invalid checkpoint path: $checkpoint_path"
        return
    }
    rm -rf -- "$checkpoint_path"
    echo "Deleted evaluated checkpoint: $checkpoint_path"
}

math_remove_final_model_weights() {
    local experiment_dir="$1"
    local experiment_real repo_real
    local -a model_files=()

    experiment_real="$(cd -- "$experiment_dir" && pwd -P)"
    repo_real="$(cd -- "$REPO_ROOT" && pwd -P)"
    case "$experiment_real" in
        "$repo_real"/outputs/*) ;;
        *)
            math_segment_die "refusing to delete model weights outside outputs: $experiment_real"
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

math_eval_checkpoint() {
    local experiment_dir="$1"
    local checkpoint_path="$2"
    local model_key="$3"
    local result_root="$4"
    local final_checkpoint="${5:-0}"
    local checkpoint_name checkpoint_step marker

    checkpoint_name="$(basename -- "$checkpoint_path")"
    checkpoint_step="${checkpoint_name#checkpoint-}"
    marker="$experiment_dir/.math_eval_complete_${checkpoint_name}"
    if [[ ! -f "$marker" ]]; then
        GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" \
        TP_SIZE="${TP_SIZE:-8}" \
        CHECKPOINTS="$checkpoint_step" \
        RESULT_ROOT="$result_root" \
        EVAL_EXPERIMENT_DIR="$experiment_dir" \
            bash "$MATH_EVAL_RUNNER" "$model_key"
        touch "$marker"
    else
        echo "Math evaluation already complete: $checkpoint_name"
    fi

    if [[ "$final_checkpoint" == "1" && "$MATH_DELETE_TRAINED_MODELS_AFTER_EVAL" == "1" ]]; then
        math_remove_evaluated_checkpoint "$experiment_dir" "$checkpoint_path"
        math_remove_final_model_weights "$experiment_dir"
    fi
}

math_replace_command_option() {
    local option="$1"
    local value="$2"
    local array_name="$3"
    local -n command_ref="$array_name"
    local index
    for (( index = 0; index < ${#command_ref[@]}; index += 1 )); do
        if [[ "${command_ref[$index]}" == "$option" ]]; then
            (( index + 1 < ${#command_ref[@]} )) || {
                math_segment_die "missing value for command option $option"
                return
            }
            command_ref[$((index + 1))]="$value"
            return 0
        fi
    done
    math_segment_die "command option not found: $option"
}

math_execute_command_segment() {
    local schedule_kind="$1"
    local target_step="$2"
    local resume_checkpoint="$3"
    shift 3
    local -a segment_command=("$@")

    case "$schedule_kind" in
        trainer_steps)
            math_replace_command_option --max_steps "$target_step" segment_command
            ;;
        policy_updates)
            segment_command+=(--stop_after_policy_updates "$target_step")
            ;;
        *)
            math_segment_die "unsupported segmented schedule: $schedule_kind"
            return
            ;;
    esac
    if [[ -n "$resume_checkpoint" ]]; then
        segment_command+=(--resume_from_checkpoint "$resume_checkpoint")
    fi
    "${segment_command[@]}"
}

math_run_segmented_training() {
    local total_steps="$1"
    local eval_interval="$2"
    local experiment_dir="$3"
    local model_key="$4"
    local result_root="$5"
    local label="$6"
    local runner="$7"
    shift 7
    local -a runner_args=("$@")
    local target_step checkpoint_path latest_checkpoint latest_step marker

    [[ "$total_steps" =~ ^[1-9][0-9]*$ && "$eval_interval" =~ ^[1-9][0-9]*$ ]] || {
        math_segment_die "segmented training steps and interval must be positive integers"
        return
    }
    (( total_steps % eval_interval == 0 )) || {
        math_segment_die "evaluation interval $eval_interval must divide total steps $total_steps"
        return
    }

    if [[ "${AUTO_EVAL:-1}" != "1" ]]; then
        latest_checkpoint="$(math_latest_checkpoint "$experiment_dir")"
        "$runner" "$total_steps" "$latest_checkpoint" "${runner_args[@]}"
        return
    fi

    marker="$experiment_dir/.math_eval_complete_checkpoint-${total_steps}"
    if [[ -f "$marker" && ! -d "$experiment_dir/checkpoint-${total_steps}" ]]; then
        echo "Math training and final evaluation already complete: $label"
        return 0
    fi

    for (( target_step = eval_interval; target_step <= total_steps; target_step += eval_interval )); do
        checkpoint_path="$experiment_dir/checkpoint-${target_step}"
        marker="$experiment_dir/.math_eval_complete_checkpoint-${target_step}"
        latest_checkpoint="$(math_latest_checkpoint "$experiment_dir")"
        latest_step=0
        if [[ -n "$latest_checkpoint" ]]; then
            latest_step="${latest_checkpoint##*-}"
            [[ "$latest_step" =~ ^[1-9][0-9]*$ ]] || {
                math_segment_die "invalid checkpoint name: $latest_checkpoint"
                return
            }
        fi

        if [[ -f "$marker" && "$latest_step" -gt "$target_step" ]]; then
            continue
        fi
        if [[ -f "$marker" && "$latest_step" -eq 0 ]]; then
            math_segment_die "cannot continue after checkpoint-$target_step evaluation without a resumable checkpoint"
            return
        fi
        if [[ ! -d "$checkpoint_path" ]]; then
            if (( latest_step > target_step )); then
                math_segment_die "checkpoint-$target_step was rotated before its evaluation completed"
                return
            fi
            if (( latest_step == 0 && target_step > eval_interval )); then
                math_segment_die "cannot continue segmented training without a resumable checkpoint"
                return
            fi
            echo "Math training segment: $label -> step $target_step/$total_steps"
            "$runner" "$target_step" "$latest_checkpoint" "${runner_args[@]}"
        fi
        [[ -d "$checkpoint_path" ]] || {
            math_segment_die "training segment did not create expected checkpoint: $checkpoint_path"
            return
        }

        echo "Evaluating Math checkpoint: $checkpoint_path"
        math_eval_checkpoint \
            "$experiment_dir" \
            "$checkpoint_path" \
            "$model_key" \
            "$result_root" \
            "$(( target_step == total_steps ))"
    done
}
