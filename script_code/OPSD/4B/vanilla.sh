#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../lib" && pwd)/code_common.sh"
code_launch_kd opsd vanilla 4b "$@"
