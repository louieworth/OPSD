#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/prepare_model.py" \
    --repo-id Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --link-name Qwen3-4B

