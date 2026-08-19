#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/prepare_models.sh"
bash "$SCRIPT_DIR/../script_code/prepare_code.sh" all
