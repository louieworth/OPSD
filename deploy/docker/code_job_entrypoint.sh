#!/usr/bin/env bash
set -euo pipefail

opsd_root="${OPSD_ROOT:-/workspace/OPSD}"
source_dir="${OPSD_ARTIFACT_SOURCE_DIR:-}"

cd "$opsd_root"
if [[ -n "$source_dir" ]]; then
    python script_code/fetch_artifacts.py \
        --target-root "$opsd_root" \
        --source-dir "$source_dir"
else
    : "${OPSD_ARTIFACT_REPO:?set OPSD_ARTIFACT_REPO for direct HF fallback}"
    : "${OPSD_ARTIFACT_RELEASE:?set OPSD_ARTIFACT_RELEASE for direct HF fallback}"
    : "${OPSD_ARTIFACT_REVISION:?set OPSD_ARTIFACT_REVISION to a pinned HF commit}"
    python script_code/fetch_artifacts.py --target-root "$opsd_root"
fi

# Recreate container-specific absolute paths, install the bundled evaluators,
# and run the same offline preflight as prepare_code.sh all.
bash script_code/prepare_code.sh artifact

if (( $# == 0 )); then
    exec bash
fi
exec "$@"
