#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$EVAL_DIR/.." && pwd)"
bash "$REPO_ROOT/scripts/run_eval_qwen3_1b.sh" "$@"
