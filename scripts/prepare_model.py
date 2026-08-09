#!/usr/bin/env python3
"""Resolve a pinned Hugging Face model snapshot and link it into the repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def validate_snapshot(snapshot: Path) -> None:
    required = [snapshot / "config.json", snapshot / "tokenizer_config.json"]
    missing = [str(path) for path in required if not path.is_file()]

    index_path = snapshot / "model.safetensors.index.json"
    single_weight = snapshot / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            missing.append(f"{index_path} (empty weight_map)")
        missing.extend(str(snapshot / shard) for shard in shards if not (snapshot / shard).is_file())
    elif not single_weight.is_file():
        missing.append(f"{index_path} or {single_weight}")

    if missing:
        raise RuntimeError("Incomplete model snapshot; missing: " + ", ".join(missing))


def get_snapshot(repo_id: str, revision: str) -> Path:
    try:
        cached = snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
        print(f"Using cached Hugging Face snapshot: {cached}")
        return Path(cached).resolve()
    except Exception:
        print(f"Pinned snapshot is not cached; downloading {repo_id}@{revision} ...")
        downloaded = snapshot_download(repo_id=repo_id, revision=revision)
        return Path(downloaded).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--link-name", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    link_path = models_dir / args.link_name
    models_dir.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        existing = link_path.resolve(strict=False)
        if existing.is_dir() and args.revision in str(existing):
            validate_snapshot(existing)
            print(f"Model link is already valid: {link_path} -> {existing}")
            return
    elif link_path.exists():
        raise RuntimeError(
            f"{link_path} exists but is not a symlink. Move it aside, then rerun this script."
        )

    snapshot = get_snapshot(args.repo_id, args.revision)
    if args.revision not in str(snapshot):
        raise RuntimeError(f"Resolved snapshot does not match pinned revision {args.revision}: {snapshot}")
    validate_snapshot(snapshot)

    temporary_link = models_dir / f".{args.link_name}.tmp-{os.getpid()}"
    temporary_link.symlink_to(snapshot, target_is_directory=True)
    temporary_link.replace(link_path)
    print(f"Created model link: {link_path} -> {snapshot}")


if __name__ == "__main__":
    main()

