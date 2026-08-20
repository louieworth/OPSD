#!/usr/bin/env bash
set -euo pipefail

opsd_root="${OPSD_ROOT:-/workspace/OPSD}"
cd "$opsd_root"

# `all` already ran while the image was built. This is an offline integrity
# check only: no Hub/GitHub calls, no dataset preparation, and no pip install.
bash script_code/prepare_code.sh verify

if (( $# == 0 )); then
    exec bash
fi
exec "$@"
