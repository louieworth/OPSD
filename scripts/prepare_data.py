#!/usr/bin/env python3
"""Download and validate the pinned OPSD train/evaluation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


DATASETS = {
    "train/openthoughts_math_30k_opsd": {
        "repo_id": "siyanzhao/Openthoughts_math_30k_opsd",
        "revision": "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62",
        "source_files": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
        "files": [
            "data/train-00000-of-00004.parquet",
            "data/train-00001-of-00004.parquet",
            "data/train-00002-of-00004.parquet",
            "data/train-00003-of-00004.parquet",
        ],
        "row_groups_per_file": 2,
        "max_file_bytes": 100_000_000,
        "rows": 29434,
        "fields": ["problem", "solution"],
    },
    "eval/aime24": {
        "repo_id": "HuggingFaceH4/aime_2024",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "files": ["data/train-00000-of-00001.parquet"],
        "rows": 30,
        "fields": ["id", "problem", "answer"],
    },
    "eval/aime25": {
        "repo_id": "yentinglin/aime_2025",
        "revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        "files": ["data/train-00000-of-00001-243207c6c994e1bd.parquet"],
        "rows": 30,
        "fields": ["id", "problem", "answer"],
    },
    "eval/beyond-aime": {
        "repo_id": "ByteDance-Seed/BeyondAIME",
        "revision": "c705198ae1043810b1e1693bd879250b51a7a523",
        "files": ["data/test.parquet"],
        "rows": 100,
        "fields": ["problem", "answer"],
    },
    "eval/hmmt25": {
        "repo_id": "MathArena/hmmt_feb_2025",
        "revision": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        "files": ["data/train-00000-of-00001.parquet"],
        "rows": 30,
        "fields": ["problem_idx", "problem", "answer"],
    },
    "eval/amo-bench": {
        "repo_id": "meituan-longcat/AMO-Bench",
        "revision": "2f422616c25d862984408fbbfaed63a961e8e025",
        "files": ["data/test-00000-of-00001.parquet"],
        "rows": 50,
        "fields": ["question_id", "prompt", "answer"],
    },
}


def reshard_parquet(root: Path, spec: dict) -> None:
    """Convert the two upstream shards into GitHub-safe local shards."""
    source_paths = [root / relative for relative in spec["source_files"]]
    output_paths = [root / relative for relative in spec["files"]]
    row_groups = []
    schema = None
    for source_path in source_paths:
        parquet_file = pq.ParquetFile(source_path)
        schema = schema or parquet_file.schema_arrow
        if parquet_file.schema_arrow != schema:
            raise RuntimeError(f"Parquet schema mismatch in {source_path}")
        row_groups.extend(
            (parquet_file, index) for index in range(parquet_file.metadata.num_row_groups)
        )

    groups_per_file = spec["row_groups_per_file"]
    if len(row_groups) != len(output_paths) * groups_per_file:
        raise RuntimeError(
            f"Expected {len(output_paths) * groups_per_file} row groups, found {len(row_groups)}"
        )

    for output_index, output_path in enumerate(output_paths):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with pq.ParquetWriter(temporary_path, schema, compression="snappy") as writer:
            begin = output_index * groups_per_file
            for parquet_file, row_group_index in row_groups[begin : begin + groups_per_file]:
                writer.write_table(parquet_file.read_row_group(row_group_index))
        if temporary_path.stat().st_size >= spec["max_file_bytes"]:
            raise RuntimeError(
                f"Resharded file exceeds GitHub's 100 MB limit: {temporary_path} "
                f"({temporary_path.stat().st_size} bytes)"
            )
        temporary_path.replace(output_path)

    validate_dataset(root, spec)
    for source_path in source_paths:
        source_path.unlink()


def validate_dataset(root: Path, spec: dict) -> None:
    rows = 0
    available_fields: set[str] = set()
    for relative_file in spec["files"]:
        parquet_path = root / relative_file
        if not parquet_path.is_file():
            raise RuntimeError(f"Missing dataset file: {parquet_path}")
        metadata = pq.read_metadata(parquet_path)
        rows += metadata.num_rows
        available_fields.update(metadata.schema.names)

    missing_fields = set(spec["fields"]) - available_fields
    if rows != spec["rows"] or missing_fields:
        raise RuntimeError(
            f"Dataset validation failed for {root}: rows={rows} (expected {spec['rows']}), "
            f"missing_fields={sorted(missing_fields)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate pinned OPSD train/evaluation datasets."
    )
    parser.add_argument(
        "--scope",
        choices=["all", "train", "eval"],
        default="all",
        help="Prepare all datasets, only training data, or only evaluation data.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / "manifest.json"
    try:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing_manifest = {}

    manifest = dict(existing_manifest)
    for relative_dir, spec in DATASETS.items():
        dataset_scope = relative_dir.split("/", 1)[0]
        if args.scope != "all" and dataset_scope != args.scope:
            continue

        destination = data_root / relative_dir
        existing = existing_manifest.get(relative_dir, {})
        if existing.get("revision") == spec["revision"]:
            try:
                validate_dataset(destination, spec)
                print(f"Dataset is already prepared: {relative_dir} ({spec['rows']} rows)")
            except RuntimeError:
                existing = {}

        if existing.get("revision") != spec["revision"]:
            download_files = spec.get("source_files", spec["files"])
            allow_patterns = list(download_files) + [
                "README.md",
                "LICENSE",
                "eval.yaml",
            ]
            print(f"Preparing {spec['repo_id']}@{spec['revision']} in {destination}")
            snapshot_download(
                repo_id=spec["repo_id"],
                repo_type="dataset",
                revision=spec["revision"],
                local_dir=destination,
                allow_patterns=allow_patterns,
            )
            if "source_files" in spec:
                reshard_parquet(destination, spec)
            validate_dataset(destination, spec)
            print(f"Validated {relative_dir}: {spec['rows']} rows")

        manifest[relative_dir] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "files": spec["files"],
            "rows": spec["rows"],
        }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote dataset manifest: {manifest_path} (scope={args.scope})")


if __name__ == "__main__":
    main()
