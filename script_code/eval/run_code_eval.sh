#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: $0 MODEL_OR_ADAPTER_PATH [LABEL]" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [[ -f "$REPO_ROOT/script_code/runtime.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/script_code/runtime.env"
fi
MODEL_PATH="$1"
MODEL_LABEL="${2:-$(basename "$MODEL_PATH")}" 
PYTHON_BIN="${CODE_PYTHON_BIN:-python}"
DATASETS="${CODE_EVAL_DATASETS:-humaneval_plus,mbpp_plus,livecodebench_v6}"
PASS_K="${CODE_EVAL_PASS_K:-12}"
TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.6}"
TOP_P="${CODE_EVAL_TOP_P:-0.95}"
MAX_TOKENS="${CODE_EVAL_MAX_RESPONSE_TOKENS:-4096}"
MAX_MODEL_LEN="${CODE_EVAL_MAX_MODEL_LEN:-6144}"
TP="${CODE_EVAL_TP:-8}"
MAX_NUM_SEQS="${CODE_EVAL_MAX_NUM_SEQS:-64}"
OUTPUT_ROOT="${CODE_EVAL_OUTPUT_ROOT:-$REPO_ROOT/outputs/code_eval/$MODEL_LABEL}"
RESULTS_FILE="${CODE_EVAL_RESULTS_FILE:-$REPO_ROOT/outputs/code_eval/results.json}"
LCB_REPO="${LCB_REPO:-$REPO_ROOT/third_party/LiveCodeBench}"

[[ -d "$MODEL_PATH" ]] || { echo "Model or adapter not found: $MODEL_PATH" >&2; exit 1; }
BASE_MODEL="$MODEL_PATH"
if [[ -f "$MODEL_PATH/adapter_config.json" ]]; then
    BASE_MODEL="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' "$MODEL_PATH/adapter_config.json")"
fi
mkdir -p "$OUTPUT_ROOT"

record_metrics() {
    local kind="$1" extract_dataset="$2" result_dataset="$3" root="$4"
    local metrics avg passed
    metrics="$($PYTHON_BIN "$SCRIPT_DIR/extract_metrics.py" --kind "$kind" --root "$root" --dataset "$extract_dataset" --pass_k "$PASS_K")"
    avg="$(awk -F= '/^avg=/{print $2}' <<< "$metrics")"
    passed="$(awk -F= '/^pass=/{print $2}' <<< "$metrics")"
    "$PYTHON_BIN" "$SCRIPT_DIR/update_results.py" \
        --file "$RESULTS_FILE" --model "$MODEL_LABEL" --dataset "$result_dataset" \
        --avg "$avg" --pass_value "$passed" --k "$PASS_K"
}

IFS=',' read -r -a dataset_list <<< "$DATASETS"
for dataset in "${dataset_list[@]}"; do
    case "$dataset" in
        humaneval_plus|mbpp_plus)
            evalplus_name="humaneval"
            [[ "$dataset" == mbpp_plus ]] && evalplus_name="mbpp"
            "$PYTHON_BIN" "$SCRIPT_DIR/run_evalplus_vllm.py" \
                --dataset "$evalplus_name" --model "$MODEL_PATH" --base_model "$BASE_MODEL" \
                --root "$OUTPUT_ROOT/evalplus" --n_samples "$PASS_K" \
                --temperature "$TEMPERATURE" --top_p "$TOP_P" \
                --max_tokens "$MAX_TOKENS" --max_model_len "$MAX_MODEL_LEN" \
                --max_num_seqs "$MAX_NUM_SEQS" --tp "$TP"
            record_metrics evalplus "$evalplus_name" "$dataset" "$OUTPUT_ROOT/evalplus"
            ;;
        livecodebench_v6)
            [[ -d "$LCB_REPO" ]] || { echo "LiveCodeBench not installed at $LCB_REPO" >&2; exit 1; }
            marker="$OUTPUT_ROOT/livecodebench/.run_start"
            mkdir -p "$(dirname "$marker")"
            touch "$marker"
            (
                cd "$LCB_REPO"
                LCB_BASE_MODEL_PATH="$BASE_MODEL" PYTHONPATH="$LCB_REPO${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON_BIN" -m lcb_runner.runner.main \
                    --model Qwen/Qwen3-235B-A22B --local_model_path "$MODEL_PATH" \
                    --trust_remote_code --scenario codegeneration --evaluate \
                    --release_version release_v6 --n "$PASS_K" \
                    --temperature "$TEMPERATURE" --top_p "$TOP_P" \
                    --max_tokens "$MAX_TOKENS" --max_model_len "$MAX_MODEL_LEN" \
                    --max_num_seqs "$MAX_NUM_SEQS" --tensor_parallel_size "$TP"
            )
            (
                cd "$LCB_REPO"
                find output -type f -newer "$marker" \
                    \( -name '*_eval.json' -o -name '*_eval_all.json' \) \
                    -exec cp --parents {} "$OUTPUT_ROOT/livecodebench" \;
            )
            record_metrics lcb livecodebench_v6 livecodebench_v6 "$OUTPUT_ROOT/livecodebench"
            ;;
        *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
    esac
done

echo "Code evaluation results: $RESULTS_FILE"
