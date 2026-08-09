#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/prepare_model_qwen3_1b.sh"
bash "$SCRIPT_DIR/prepare_model_qwen3_4b.sh"
bash "$SCRIPT_DIR/prepare_model_qwen3_8b.sh"

