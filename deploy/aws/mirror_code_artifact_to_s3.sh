#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

release="${OPSD_ARTIFACT_RELEASE:?set OPSD_ARTIFACT_RELEASE to the published release}"
s3_root="${OPSD_ARTIFACT_S3_URI:?set OPSD_ARTIFACT_S3_URI, for example s3://bucket/opsd}"
stage="${OPSD_ARTIFACT_STAGE:-$REPO_ROOT/.cache/code_artifact_publish/$release}"

command -v aws >/dev/null 2>&1 || { echo "aws CLI is required" >&2; exit 2; }
[[ -f "$stage/artifact_manifest.json" ]] || {
    echo "Missing staged artifact manifest: $stage/artifact_manifest.json" >&2
    exit 2
}

destination="${s3_root%/}/code-artifacts/releases/$release"
aws s3 sync "$stage/" "$destination/" --only-show-errors --no-progress
# Upload the readiness marker last. Workers reject a release without this marker.
aws s3 cp \
    "$stage/artifact_manifest.json" \
    "$destination/_READY.json" \
    --only-show-errors --no-progress

echo "Mirrored artifact to: $destination"
