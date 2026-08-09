#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: bash scripts/prepare_models.sh [all|1.7b|4b|8b]" >&2
}

prepare_model() {
    local model="$1"
    local repo_id
    local revision
    local link_name

    case "$model" in
        1.7b)
            repo_id="Qwen/Qwen3-1.7B"
            revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
            link_name="Qwen3-1.7B"
            ;;
        4b)
            repo_id="Qwen/Qwen3-4B"
            revision="1cfa9a7208912126459214e8b04321603b3df60c"
            link_name="Qwen3-4B"
            ;;
        8b)
            repo_id="Qwen/Qwen3-8B"
            revision="b968826d9c46dd6066d109eabc6255188de91218"
            link_name="Qwen3-8B"
            ;;
        *)
            echo "Unsupported model: $model" >&2
            usage
            exit 2
            ;;
    esac

    python "$SCRIPT_DIR/prepare_model.py" \
        --repo-id "$repo_id" \
        --revision "$revision" \
        --link-name "$link_name"
}

if (( $# > 1 )); then
    usage
    exit 2
fi

case "${1:-all}" in
    all)
        models=(1.7b 4b 8b)
        ;;
    1b|1.7b|qwen3-1.7b)
        models=(1.7b)
        ;;
    4b|qwen3-4b)
        models=(4b)
        ;;
    8b|qwen3-8b)
        models=(8b)
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unsupported model: $1" >&2
        usage
        exit 2
        ;;
esac

for model in "${models[@]}"; do
    prepare_model "$model"
done
