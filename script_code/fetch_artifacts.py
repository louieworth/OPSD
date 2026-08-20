#!/usr/bin/env python3
"""Fetch and install a pinned OPSD code-data artifact release."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import random
import re
import subprocess
import time
from pathlib import Path

try:
    from .artifact_utils import ArtifactError, install_release, load_manifest, verify_release
except ImportError:  # Direct execution: python script_code/fetch_artifacts.py
    from artifact_utils import ArtifactError, install_release, load_manifest, verify_release


RELEASE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PINNED_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def require(value: str | None, name: str) -> str:
    if value:
        return value
    raise SystemExit(f"{name} is required")


def verify_source_revision(target_root: Path, manifest: dict, *, allow_mismatch: bool) -> None:
    expected = manifest.get("source_git_commit")
    if not expected or allow_mismatch:
        return
    declared = os.environ.get("OPSD_GIT_REV")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target_root,
        text=True,
        capture_output=True,
    )
    actual = completed.stdout.strip() if completed.returncode == 0 else declared
    if not actual:
        raise ArtifactError(
            "Cannot verify the OPSD source revision. Use a Git checkout or set "
            "OPSD_GIT_REV in an image built from the pinned commit."
        )
    if actual != expected:
        raise ArtifactError(
            f"OPSD source revision mismatch: artifact requires {expected}, current source is {actual}"
        )


def download_hf_release(
    *,
    repo_id: str,
    release: str,
    revision: str,
    cache_root: Path,
    max_workers: int,
    attempts: int,
    initial_jitter_seconds: int,
) -> Path:
    from huggingface_hub import snapshot_download

    if not RELEASE_PATTERN.fullmatch(release):
        raise ArtifactError(f"Invalid artifact release: {release!r}")
    if not PINNED_REVISION_PATTERN.fullmatch(revision):
        raise ArtifactError(
            "HF artifact revision must be a full 40-character commit SHA; "
            "floating branches such as 'main' are not allowed in jobs"
        )
    repo_suffix = hashlib.sha256(repo_id.encode()).hexdigest()[:8]
    safe_repo = re.sub(r"[^A-Za-z0-9._-]", "--", repo_id) + f"-{repo_suffix}"
    local_root = cache_root / "hf" / safe_repo / revision
    release_root = local_root / "releases" / release
    if (release_root / "artifact_manifest.json").is_file():
        manifest = load_manifest(release_root)
        if manifest.get("release") != release:
            raise ArtifactError("Cached artifact release does not match the requested release")
        verify_release(release_root, manifest)
        print(f"Using verified local HF artifact cache: {release_root}")
        return release_root

    if initial_jitter_seconds > 0:
        delay = random.SystemRandom().uniform(0, initial_jitter_seconds)
        print(f"HF startup jitter: waiting {delay:.1f}s before the first request")
        time.sleep(delay)

    token = os.environ.get("HF_TOKEN") or None
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=local_root,
                allow_patterns=[
                    f"releases/{release}/artifact_manifest.json",
                    f"releases/{release}/**",
                ],
                token=token,
                max_workers=max_workers,
            )
            manifest = load_manifest(release_root)
            if manifest.get("release") != release:
                raise ArtifactError("Downloaded artifact release does not match the requested release")
            verify_release(release_root, manifest)
            return release_root
        except Exception as error:
            if attempt == attempts:
                raise ArtifactError(
                    f"HF download failed after {attempts} attempts: {error}"
                ) from error
            delay = min(300.0, 5.0 * (2 ** (attempt - 1))) + random.SystemRandom().uniform(0, 5)
            print(f"HF download attempt {attempt}/{attempts} failed: {error}; retrying in {delay:.1f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a verified OPSD artifact release")
    parser.add_argument("--target-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-dir", help="Verified local/S3-synced release directory; skips HF")
    parser.add_argument("--repo-id", default=os.environ.get("OPSD_ARTIFACT_REPO"))
    parser.add_argument("--release", default=os.environ.get("OPSD_ARTIFACT_RELEASE"))
    parser.add_argument("--revision", default=os.environ.get("OPSD_ARTIFACT_REVISION"))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("OPSD_HF_MAX_WORKERS", "1")))
    parser.add_argument("--attempts", type=int, default=int(os.environ.get("OPSD_HF_DOWNLOAD_ATTEMPTS", "8")))
    parser.add_argument(
        "--initial-jitter-seconds",
        type=int,
        default=int(os.environ.get("OPSD_HF_INITIAL_JITTER_SECONDS", "0")),
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Disable the source commit check (intended only for artifact migration/debugging)",
    )
    args = parser.parse_args()

    if args.max_workers < 1 or args.attempts < 1 or args.initial_jitter_seconds < 0:
        raise SystemExit("max-workers and attempts must be positive; jitter must be non-negative")

    target_root = Path(args.target_root).resolve()
    cache_root = target_root / ".cache/code_artifacts"
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "fetch.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.source_dir:
            release_root = Path(args.source_dir).resolve()
        else:
            repo_id = require(args.repo_id, "--repo-id or OPSD_ARTIFACT_REPO")
            release = require(args.release, "--release or OPSD_ARTIFACT_RELEASE")
            revision = require(args.revision, "--revision or OPSD_ARTIFACT_REVISION")
            release_root = download_hf_release(
                repo_id=repo_id,
                release=release,
                revision=revision,
                cache_root=cache_root,
                max_workers=args.max_workers,
                attempts=args.attempts,
                initial_jitter_seconds=args.initial_jitter_seconds,
            )

        manifest = load_manifest(release_root)
        verify_release(release_root, manifest)
        verify_source_revision(
            target_root,
            manifest,
            allow_mismatch=args.allow_source_mismatch,
        )
        install_release(release_root, target_root, manifest)
        installed_dir = cache_root / "installed"
        installed_dir.mkdir(parents=True, exist_ok=True)
        installed_manifest = installed_dir / f"{manifest['artifact_id']}.json"
        installed_manifest.write_text(
            (release_root / "artifact_manifest.json").read_text()
        )
        print(
            f"Installed artifact {manifest['artifact_id']} "
            f"({len(manifest['files'])} files) into {target_root}"
        )


if __name__ == "__main__":
    try:
        main()
    except ArtifactError as error:
        raise SystemExit(f"fetch_artifacts.py: {error}") from error
