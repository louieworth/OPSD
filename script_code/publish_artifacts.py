#!/usr/bin/env python3
"""Publish a prepared code-data release to a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

try:
    from .artifact_utils import ArtifactError, build_manifest, stage_release
except ImportError:  # Direct execution: python script_code/publish_artifacts.py
    from artifact_utils import ArtifactError, build_manifest, stage_release


def git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"Cannot resolve source Git commit: {completed.stderr.strip()}")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if status:
        raise SystemExit(
            "Refusing to publish from a dirty worktree. Commit the artifact scripts and "
            "all source changes first; generated data remains ignored by Git."
        )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish prepared OPSD code artifacts as an immutable HF dataset release."
    )
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, for example org/opsd-code-artifacts")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--revision", default="main", help="HF branch to update")
    parser.add_argument("--public", action="store_true", help="Create a public repo if it does not exist")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate locally without uploading")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "script_code/preflight.py"),
            "--root",
            str(repo_root),
            "--scope",
            "all",
        ],
        cwd=repo_root,
        check=True,
    )
    source_commit = git_commit(repo_root)
    manifest = build_manifest(repo_root, source_git_commit=source_commit)
    release = f"v1-{manifest['artifact_id'][:16]}"
    manifest["release"] = release

    stage_root = repo_root / ".cache/code_artifact_publish" / release
    stage_release(repo_root, stage_root, manifest)
    print(
        f"Staged {len(manifest['files'])} files ({manifest['total_bytes']} bytes) "
        f"at {stage_root}"
    )
    if args.dry_run:
        print(f"Dry run complete: release={release} artifact_id={manifest['artifact_id']}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required for upload; pass it through the environment, not the command line")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    release_path = f"releases/{release}"
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        folder_path=stage_root,
        path_in_repo=release_path,
        ignore_patterns=[".cache/.huggingface/**"],
        commit_message=f"Publish OPSD code artifact {release}",
    )
    pointer = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "release": release,
        "artifact_id": manifest["artifact_id"],
        "artifact_commit": commit.oid,
    }
    pointer_commit = api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        path_or_fileobj=(json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode(),
        path_in_repo="latest.json",
        commit_message=f"Point latest at OPSD code artifact {release}",
    )
    print("Upload complete. Pin these values in the EC2 job definition:")
    print(f"OPSD_ARTIFACT_REPO={args.repo_id}")
    print(f"OPSD_ARTIFACT_RELEASE={release}")
    print(f"OPSD_ARTIFACT_REVISION={pointer_commit.oid}")
    print(f"OPSD_GIT_REV={source_commit}")
    print(f"OPSD_ARTIFACT_STAGE={stage_root}")


if __name__ == "__main__":
    try:
        main()
    except ArtifactError as error:
        raise SystemExit(f"publish_artifacts.py: {error}") from error
