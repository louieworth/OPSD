#!/usr/bin/env bash
set -euo pipefail

release="${OPSD_ARTIFACT_RELEASE:?set OPSD_ARTIFACT_RELEASE}"
s3_root="${OPSD_ARTIFACT_S3_URI:?set OPSD_ARTIFACT_S3_URI}"
host_root="${OPSD_EC2_ARTIFACT_ROOT:-/srv/opsd}"
destination="$host_root/artifacts/$release"
source_uri="${s3_root%/}/code-artifacts/releases/$release"

command -v aws >/dev/null 2>&1 || { echo "aws CLI is required" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }
mkdir -p "$host_root/artifacts" "$destination"

exec 9>"$host_root/artifacts/.fetch.lock"
flock 9
if [[ -f "$destination/artifact_manifest.json" && -f "$destination/_READY.json" ]] && \
   cmp -s "$destination/artifact_manifest.json" "$destination/_READY.json"; then
    echo "Using EC2-local artifact: $destination"
else
    aws s3 sync "$source_uri/" "$destination/" --only-show-errors --no-progress
    [[ -f "$destination/artifact_manifest.json" && -f "$destination/_READY.json" ]] || {
        echo "Incomplete S3 artifact release: $source_uri" >&2
        exit 1
    }
    cmp -s "$destination/artifact_manifest.json" "$destination/_READY.json" || {
        echo "S3 artifact readiness marker does not match the manifest" >&2
        exit 1
    }
    echo "Downloaded artifact from S3: $destination"
fi

printf '%s\n' "$destination"
