#!/usr/bin/env bash

# Shared launcher and lifecycle helpers for the single-node TRD recipes.
# Public source-specific entry points live under scripts/OPD and scripts/OPSD.

TRD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRD_SCRIPTS_DIR="$(cd -- "$TRD_SCRIPT_DIR/.." && pwd)"
source "$TRD_SCRIPTS_DIR/common_env.sh"

TRD_TRAINER_GPUS="${TRD_TRAINER_GPUS:-0,1,2,3}"
TRD_TEACHER_GPUS="${TRD_TEACHER_GPUS:-4,5,6,7}"
TRD_EXPECTED_WORLD_SIZE=4
TRD_REFINEMENT_HOST="${TRD_REFINEMENT_HOST:-127.0.0.1}"
TRD_REFINEMENT_PORT="${TRD_REFINEMENT_PORT:-8002}"
TRD_REFINEMENT_CONNECT_TIMEOUT="${TRD_REFINEMENT_CONNECT_TIMEOUT:-10}"
TRD_REFINEMENT_REQUEST_TIMEOUT="${TRD_REFINEMENT_REQUEST_TIMEOUT:-1800}"
# The default is source-specific and is resolved inside trd_launch:
# OPSD keeps a 20,000-token refinement prefix plus the y_r reserve, while
# OPD retains the existing 20,000-token total context.
TRD_REFINEMENT_MAX_MODEL_LEN="${TRD_REFINEMENT_MAX_MODEL_LEN:-}"
TRD_SERVER_STARTUP_TIMEOUT="${TRD_SERVER_STARTUP_TIMEOUT:-900}"
TRD_TEACHER_GPU_MEMORY_UTILIZATION="${TRD_TEACHER_GPU_MEMORY_UTILIZATION:-0.9}"
TRD_SERVER_LOG_DIR="${TRD_SERVER_LOG_DIR:-$REPO_ROOT/outputs/trd/server-logs}"
TRD_PYTHON_BIN="${TRD_PYTHON_BIN:-python}"
TRD_TRL_BIN="${TRD_TRL_BIN:-trl}"

TRD_TEACHER_SERVER_PID=""
TRD_TEACHER_SERVER_LOG=""
TRD_TRAINING_PID=""

trd_die() {
    echo "TRD launcher error: $*" >&2
    return 1
}

trd_require_command() {
    local command_name="$1"
    if ! command -v -- "$command_name" >/dev/null 2>&1; then
        trd_die "required command not found: $command_name"
    fi
}

trd_validate_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        trd_die "$name must be a positive integer; got '$value'"
    fi
}

trd_reject_structural_overrides() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --alg|--alg=*|\
            --model_name_or_path|--model-name-or-path|--model_name_or_path=*|--model-name-or-path=*|\
            --teacher_model_name_or_path|--teacher-model-name-or-path|--teacher_model_name_or_path=*|--teacher-model-name-or-path=*|\
            --refinement_vllm_server_host|--refinement-vllm-server-host|--refinement_vllm_server_host=*|--refinement-vllm-server-host=*|\
            --refinement_vllm_server_port|--refinement-vllm-server-port|--refinement_vllm_server_port=*|--refinement-vllm-server-port=*|\
            --refinement_vllm_max_model_len|--refinement-vllm-max-model-len|--refinement_vllm_max_model_len=*|--refinement-vllm-max-model-len=*|\
            --max_steps|--max-steps|--max_steps=*|--max-steps=*|\
            --policy_gradient_updates|--policy-gradient-updates|--policy_gradient_updates=*|--policy-gradient-updates=*|\
            --gradient_accumulation_steps|--gradient-accumulation-steps|--gradient_accumulation_steps=*|--gradient-accumulation-steps=*|\
            --max_completion_length|--max-completion-length|--max_completion_length=*|--max-completion-length=*|\
            --max_refinement_length|--max-refinement-length|--max_refinement_length=*|--max-refinement-length=*|\
            --save_steps|--save-steps|--save_steps=*|--save-steps=*|\
            --run_config|--run-config|--run_config=*|--run-config=*)
                trd_die "'$argument' is launcher-owned; choose the matching model launcher or use its documented environment variable"
                return 1
                ;;
        esac
    done
}

trd_validate_gpu_topology() {
    local -a trainer_devices teacher_devices
    local device other

    IFS=',' read -r -a trainer_devices <<< "$TRD_TRAINER_GPUS"
    IFS=',' read -r -a teacher_devices <<< "$TRD_TEACHER_GPUS"
    if (( ${#trainer_devices[@]} != TRD_EXPECTED_WORLD_SIZE )); then
        trd_die "TRD_TRAINER_GPUS must contain exactly four GPU IDs"
    fi
    if (( ${#teacher_devices[@]} != TRD_EXPECTED_WORLD_SIZE )); then
        trd_die "TRD_TEACHER_GPUS must contain exactly four GPU IDs"
    fi

    for device in "${trainer_devices[@]}" "${teacher_devices[@]}"; do
        if [[ ! "$device" =~ ^[0-9]+$ ]]; then
            trd_die "GPU IDs must be non-negative integers; got '$device'"
        fi
    done
    for device in "${trainer_devices[@]}"; do
        for other in "${teacher_devices[@]}"; do
            if [[ "$device" == "$other" ]]; then
                trd_die "trainer and teacher GPU sets overlap on device $device"
            fi
        done
    done
}

trd_assert_port_free() {
    "$TRD_PYTHON_BIN" -c '
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except OSError as exc:
    raise SystemExit(f"teacher vLLM port {host}:{port} is unavailable: {exc}")
finally:
    sock.close()
' "$TRD_REFINEMENT_HOST" "$TRD_REFINEMENT_PORT"
}

trd_server_world_size() {
    "$TRD_PYTHON_BIN" -c '
import json
import sys
import urllib.request

base_url = f"http://{sys.argv[1]}:{sys.argv[2]}"
with urllib.request.urlopen(f"{base_url}/health/", timeout=2) as response:
    health = json.load(response)
if health.get("status") != "ok":
    raise SystemExit(1)
with urllib.request.urlopen(f"{base_url}/get_world_size/", timeout=2) as response:
    world = json.load(response)
print(int(world["world_size"]))
' "$TRD_REFINEMENT_HOST" "$TRD_REFINEMENT_PORT"
}

trd_tail_server_log() {
    if [[ -n "$TRD_TEACHER_SERVER_LOG" && -f "$TRD_TEACHER_SERVER_LOG" ]]; then
        echo "Last 80 lines from teacher vLLM log ($TRD_TEACHER_SERVER_LOG):" >&2
        tail -n 80 "$TRD_TEACHER_SERVER_LOG" >&2 || true
    fi
}

trd_signal_process_group() {
    local pid="$1"
    local signal_name="$2"
    [[ -n "$pid" ]] || return 0
    kill -s "$signal_name" -- "-$pid" 2>/dev/null || kill -s "$signal_name" "$pid" 2>/dev/null || true
}

trd_stop_training() {
    local pid="$TRD_TRAINING_PID"
    local attempt
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        trd_signal_process_group "$pid" TERM
        for attempt in {1..20}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "Training process group did not stop after 20 seconds; sending SIGKILL." >&2
            trd_signal_process_group "$pid" KILL
        fi
    fi
    if wait "$pid" 2>/dev/null; then
        :
    fi
    TRD_TRAINING_PID=""
}

trd_stop_teacher_server() {
    local pid="$TRD_TEACHER_SERVER_PID"
    local attempt
    [[ -n "$pid" ]] || return 0

    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping teacher vLLM process group $pid."
        trd_signal_process_group "$pid" TERM
        for attempt in {1..20}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "Teacher vLLM did not stop after 20 seconds; sending SIGKILL." >&2
            trd_signal_process_group "$pid" KILL
        fi
    fi
    if wait "$pid" 2>/dev/null; then
        :
    fi
    TRD_TEACHER_SERVER_PID=""
}

trd_cleanup_on_exit() {
    local status="$1"
    trap - EXIT HUP INT TERM
    trd_stop_training
    trd_stop_teacher_server
    exit "$status"
}

trd_cleanup_on_signal() {
    local status="$1"
    trap - EXIT HUP INT TERM
    trd_stop_training
    trd_stop_teacher_server
    exit "$status"
}

trd_install_cleanup_traps() {
    trap 'trd_cleanup_on_exit $?' EXIT
    trap 'trd_cleanup_on_signal 129' HUP
    trap 'trd_cleanup_on_signal 130' INT
    trap 'trd_cleanup_on_signal 143' TERM
}

trd_start_teacher_server() {
    local teacher_model="$1"
    local run_name="$2"
    local refinement_max_model_len="$3"
    local safe_run_name="${run_name//\//_}"
    local startup_start world_size

    [[ -d "$teacher_model" ]] || trd_die "teacher model directory not found: $teacher_model"
    trd_require_command "$TRD_PYTHON_BIN"
    trd_require_command "$TRD_TRL_BIN"
    trd_require_command setsid
    trd_validate_gpu_topology
    trd_validate_positive_integer TRD_REFINEMENT_PORT "$TRD_REFINEMENT_PORT"
    trd_validate_positive_integer TRD_REFINEMENT_MAX_MODEL_LEN "$refinement_max_model_len"
    trd_validate_positive_integer TRD_SERVER_STARTUP_TIMEOUT "$TRD_SERVER_STARTUP_TIMEOUT"
    trd_assert_port_free

    mkdir -p "$TRD_SERVER_LOG_DIR"
    TRD_TEACHER_SERVER_LOG="$TRD_SERVER_LOG_DIR/${safe_run_name}.log"
    : > "$TRD_TEACHER_SERVER_LOG"

    echo "Starting fixed teacher vLLM on GPUs $TRD_TEACHER_GPUS (TP=$TRD_EXPECTED_WORLD_SIZE)."
    echo "Teacher log: $TRD_TEACHER_SERVER_LOG"
    CUDA_VISIBLE_DEVICES="$TRD_TEACHER_GPUS" setsid "$TRD_TRL_BIN" vllm-serve \
        --model "$teacher_model" \
        --host "$TRD_REFINEMENT_HOST" \
        --port "$TRD_REFINEMENT_PORT" \
        --tensor_parallel_size "$TRD_EXPECTED_WORLD_SIZE" \
        --data_parallel_size 1 \
        --gpu_memory_utilization "$TRD_TEACHER_GPU_MEMORY_UTILIZATION" \
        --dtype bfloat16 \
        --max_model_len "$refinement_max_model_len" \
        > "$TRD_TEACHER_SERVER_LOG" 2>&1 &
    TRD_TEACHER_SERVER_PID=$!

    startup_start=$SECONDS
    while (( SECONDS - startup_start < TRD_SERVER_STARTUP_TIMEOUT )); do
        if ! kill -0 "$TRD_TEACHER_SERVER_PID" 2>/dev/null; then
            echo "Teacher vLLM exited during startup." >&2
            trd_tail_server_log
            return 1
        fi
        if world_size="$(trd_server_world_size 2>/dev/null)"; then
            if [[ "$world_size" != "$TRD_EXPECTED_WORLD_SIZE" ]]; then
                echo "Teacher vLLM reported world_size=$world_size; expected $TRD_EXPECTED_WORLD_SIZE." >&2
                trd_stop_teacher_server
                return 1
            fi
            echo "Teacher vLLM is healthy at http://$TRD_REFINEMENT_HOST:$TRD_REFINEMENT_PORT (world_size=$world_size)."
            return 0
        fi
        sleep 2
    done

    echo "Teacher vLLM did not become healthy within $TRD_SERVER_STARTUP_TIMEOUT seconds." >&2
    trd_tail_server_log
    trd_stop_teacher_server
    return 1
}

trd_run_training() {
    local finished_pid wait_status
    (( $# > 0 )) || trd_die "trd_run_training requires a command"
    [[ -n "$TRD_TEACHER_SERVER_PID" ]] || trd_die "teacher vLLM has not been started"

    echo "Starting four-rank training on GPUs $TRD_TRAINER_GPUS."
    CUDA_VISIBLE_DEVICES="$TRD_TRAINER_GPUS" setsid "$@" &
    TRD_TRAINING_PID=$!

    if wait -n -p finished_pid "$TRD_TRAINING_PID" "$TRD_TEACHER_SERVER_PID"; then
        wait_status=0
    else
        wait_status=$?
    fi

    if [[ "$finished_pid" == "$TRD_TEACHER_SERVER_PID" ]]; then
        TRD_TEACHER_SERVER_PID=""
        echo "Teacher vLLM exited unexpectedly while training was active (status $wait_status)." >&2
        trd_stop_training
        trd_tail_server_log
        return 1
    fi

    TRD_TRAINING_PID=""
    return "$wait_status"
}

trd_default_save_steps() {
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

trd_default_checkpoints() {
    local updates="$1"
    local save_steps="$2"
    local checkpoint
    local checkpoints=()
    for (( checkpoint = save_steps; checkpoint <= updates; checkpoint += save_steps )); do
        checkpoints+=("$checkpoint")
    done
    echo "${checkpoints[*]}"
}

trd_launch() {
    local algorithm="$1"
    local model_key="$2"
    local model_label="$3"
    local student_model="$4"
    local refiner_model="$5"
    local per_device_batch_size="$6"
    shift 6
    local -a extra_training_args=("$@")
    local -a algorithm_args=()
    local -a training_command
    local rollout_steps="${TRD_MAX_STEPS:-100}"
    local policy_updates="${TRD_POLICY_GRADIENT_UPDATES:-100}"
    local completion_length="${TRD_MAX_COMPLETION_LENGTH:-1024}"
    local refinement_length="${TRD_MAX_REFINEMENT_LENGTH:-1024}"
    local training_max_length
    local refinement_max_model_len
    local student_response_reserve
    local save_steps
    local run_config
    local output_root

    trd_reject_structural_overrides "${extra_training_args[@]}"
    trd_validate_positive_integer TRD_MAX_STEPS "$rollout_steps"
    trd_validate_positive_integer TRD_POLICY_GRADIENT_UPDATES "$policy_updates"
    trd_validate_positive_integer TRD_MAX_COMPLETION_LENGTH "$completion_length"
    trd_validate_positive_integer TRD_MAX_REFINEMENT_LENGTH "$refinement_length"
    student_response_reserve="$completion_length"
    if (( refinement_length > student_response_reserve )); then
        student_response_reserve="$refinement_length"
    fi

    case "$algorithm" in
        opsd)
            # Preserve a full 20,000-token prompt on both paths, then add the
            # relevant response reserve to obtain each total context length.
            training_max_length="${TRD_MAX_LENGTH:-$((20000 + student_response_reserve))}"
            refinement_max_model_len="${TRD_REFINEMENT_MAX_MODEL_LEN:-$((20000 + refinement_length))}"
            ;;
        opd)
            training_max_length="${TRD_MAX_LENGTH:-20000}"
            refinement_max_model_len="${TRD_REFINEMENT_MAX_MODEL_LEN:-20000}"
            ;;
        *)
            trd_die "unsupported algorithm: $algorithm"
            return 1
            ;;
    esac

    trd_validate_positive_integer TRD_MAX_LENGTH "$training_max_length"
    trd_validate_positive_integer TRD_REFINEMENT_MAX_MODEL_LEN "$refinement_max_model_len"
    if (( student_response_reserve >= training_max_length )); then
        trd_die "TRD_MAX_LENGTH must leave room for the larger of y_o and y_r"
        return 1
    fi
    if (( policy_updates > rollout_steps || rollout_steps % policy_updates != 0 )); then
        trd_die "TRD_POLICY_GRADIENT_UPDATES must divide TRD_MAX_STEPS and cannot exceed it"
    fi
    if (( refinement_length >= refinement_max_model_len )); then
        trd_die "TRD_MAX_REFINEMENT_LENGTH must be smaller than TRD_REFINEMENT_MAX_MODEL_LEN"
    fi
    if [[ "${DISTILL_DRY_RUN:-0}" != "1" ]]; then
        [[ -d "$student_model" ]] || trd_die "student model directory not found: $student_model"
        trd_require_command accelerate
    fi

    save_steps="${TRD_SAVE_STEPS:-$(trd_default_save_steps "$policy_updates")}"
    trd_validate_positive_integer TRD_SAVE_STEPS "$save_steps"
    if (( save_steps > policy_updates || policy_updates % save_steps != 0 )); then
        trd_die "TRD_SAVE_STEPS must divide the policy update count and cannot exceed it"
    fi

    run_config="${RUN_CONFIG:-trd_${algorithm}_${model_label}_gen${completion_length}_n${rollout_steps}_u${policy_updates}_step0teacher}"
    output_root="$REPO_ROOT/outputs/$algorithm"

    case "$algorithm" in
        opsd)
            algorithm_args=(
                --fixed_teacher
                --teacher_thinking True
            )
            ;;
        opd)
            algorithm_args=(
                --teacher_model_name_or_path "$REPO_ROOT/models/Qwen3-8B"
                --teacher_thinking False
            )
            ;;
        *)
            trd_die "unsupported algorithm: $algorithm"
            ;;
    esac

    training_command=(
        accelerate launch
        --config_file "${ACCELERATE_CONFIG_FILE:-$REPO_ROOT/accelerate.yaml}"
        --num_processes 4
        --gradient_accumulation_steps 1
        --main_process_port "${TRD_ACCELERATE_PORT:-12949}"
        "$REPO_ROOT/opsd_train.py"
        --alg "$algorithm"
        --model_name_or_path "$student_model"
        --train_dataset_path "$REPO_ROOT/data/train/openthoughts_math_30k_opsd"
        --learning_rate "${TRD_LEARNING_RATE:-5e-6}"
        --max_grad_norm "${TRD_MAX_GRAD_NORM:-0.1}"
        --per_device_train_batch_size "$per_device_batch_size"
        --gradient_checkpointing
        --gradient_accumulation_steps 1
        --output_dir "$output_root"
        --run_config "$run_config"
        --max_steps "$rollout_steps"
        --policy_gradient_updates "$policy_updates"
        --max_completion_length "$completion_length"
        --max_refinement_length "$refinement_length"
        --save_steps "$save_steps"
        --logging_steps "${TRD_LOGGING_STEPS:-2}"
        --attn_implementation flash_attention_2
        --torch_dtype bfloat16
        --max_length "$training_max_length"
        --beta 0
        --distillation_temperature 1.0
        --top_k_loss 0
        --jsd_token_clip 0
        --use_vllm
        --vllm_mode colocate
        --vllm_gpu_memory_utilization "${TRD_STUDENT_VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
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
        --teacher_refine
        --refinement_vllm_server_host "$TRD_REFINEMENT_HOST"
        --refinement_vllm_server_port "$TRD_REFINEMENT_PORT"
        --refinement_vllm_connect_timeout "$TRD_REFINEMENT_CONNECT_TIMEOUT"
        --refinement_vllm_request_timeout "$TRD_REFINEMENT_REQUEST_TIMEOUT"
        --refinement_vllm_max_model_len "$refinement_max_model_len"
        --wandb_project TRD
        "${algorithm_args[@]}"
        "${extra_training_args[@]}"
    )

    if [[ "${DISTILL_DRY_RUN:-0}" == "1" ]]; then
        if declare -F distill_print_command >/dev/null 2>&1; then
            distill_print_command "${training_command[@]}"
        else
            printf 'TRAIN_CMD:'
            printf ' %q' "${training_command[@]}"
            printf '\n'
        fi
        printf 'EVAL_EXPERIMENT_DIR: %s\n' "$output_root/$run_config"
        printf 'RESULT_ROOT: %s\n' "${RESULT_ROOT:-$REPO_ROOT/outputs/eval/$algorithm/trd/$run_config}"
        return 0
    fi

    trd_install_cleanup_traps
    trd_start_teacher_server "$refiner_model" "$run_config" "$refinement_max_model_len"

    trd_run_training "${training_command[@]}"

    # Formal evaluation needs all eight GPUs, so release the TP=4 refiner first.
    trd_stop_teacher_server

    if [[ "${AUTO_EVAL:-1}" == "1" ]]; then
        echo "Training complete; teacher server stopped. Starting $model_label thinking-mode evaluation on all GPUs."
        export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
        export TP_SIZE="${TP_SIZE:-8}"
        export CHECKPOINTS="${CHECKPOINTS:-$(trd_default_checkpoints "$policy_updates" "$save_steps")}"
        export RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/outputs/eval/$algorithm/trd/$run_config}"
        EVAL_EXPERIMENT_DIR="$output_root/$run_config" \
            bash "$TRD_SCRIPTS_DIR/run_eval.sh" "$model_key"
    fi
}
