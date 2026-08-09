#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/prepare_model.py" \
    --repo-id Qwen/Qwen3-8B \
    --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --link-name Qwen3-8B

