#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
models="1.7b,4b,8b"
methods="base,sft,grpo,vanilla,clip,top_k,trd,skd"
sources="opd,opsd"
dry_run=0

while (($#)); do
    case "$1" in
        --models) models="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --sources) sources="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$dry_run" == 1 ]] && export CODE_DRY_RUN=1
IFS=',' read -r -a model_list <<< "$models"
IFS=',' read -r -a method_list <<< "$methods"
IFS=',' read -r -a source_list <<< "$sources"

baseline_methods=()
kd_methods=()
for method in "${method_list[@]}"; do
    case "$method" in
        base|sft|grpo) baseline_methods+=("$method") ;;
        vanilla|clip|top_k|trd|skd) kd_methods+=("$method") ;;
        *) echo "Unknown method: $method" >&2; exit 2 ;;
    esac
done

for source in "${source_list[@]}"; do
    case "$source" in opd|opsd) ;; *) echo "Unknown source: $source" >&2; exit 2 ;; esac
done

# Model size is the primary grouping. Within each model, run common baselines,
# then every requested OPD method, then every requested OPSD method.
for model in "${model_list[@]}"; do
    case "${model,,}" in
        1b|1.7b|qwen3-1.7b) scope="1B" ;;
        4b|qwen3-4b) scope="4B" ;;
        8b|qwen3-8b) scope="8B" ;;
        *) echo "Unknown model: $model" >&2; exit 2 ;;
    esac

    for method in "${baseline_methods[@]}"; do
        bash "$SCRIPT_DIR/Baselines/$scope/$method.sh"
    done
    for source in "${source_list[@]}"; do
        if [[ "$source" == "opd" && "$scope" == "8B" ]]; then
            continue
        fi
        for method in "${kd_methods[@]}"; do
            bash "$SCRIPT_DIR/${source^^}/$scope/$method.sh"
        done
    done
done
