#!/usr/bin/env python3
"""Validate Git-external training data and code-evaluation components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def require_paths(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing {label}: {missing}")


def check_train(root: Path) -> None:
    math_files = sorted((root / "data/train/openthoughts_math_30k_opsd/data").glob("*.parquet"))
    require_paths(math_files, "OpenThoughts math shards")
    math_rows = sum(pq.read_metadata(path).num_rows for path in math_files)
    if math_rows != 29_434:
        raise SystemExit(f"Unexpected OpenThoughts row count: {math_rows}")

    taco_root = root / "data/train/taco_code_clean"
    require_paths(
        [taco_root / "manifest.json", taco_root / "data/train-00000-of-00001.parquet"],
        "clean TACO files",
    )
    manifest = json.loads((taco_root / "manifest.json").read_text())
    taco_rows = pq.read_metadata(taco_root / "data/train-00000-of-00001.parquet").num_rows
    if manifest.get("rows") != 18_862 or taco_rows != 18_862:
        raise SystemExit(f"Unexpected clean TACO size: manifest={manifest.get('rows')}, parquet={taco_rows}")


def check_eval(root: Path) -> None:
    math_eval = [
        root / "data/eval/aime24/data/train-00000-of-00001.parquet",
        root / "data/eval/aime25/data/train-00000-of-00001-243207c6c994e1bd.parquet",
        root / "data/eval/beyond-aime/data/test.parquet",
        root / "data/eval/hmmt25/data/train-00000-of-00001.parquet",
        root / "data/eval/amo-bench/data/test-00000-of-00001.parquet",
    ]
    require_paths(math_eval, "Git-synchronized math evaluation files")
    lcb_root = root / "data/eval/livecodebench_code_generation_lite"
    lcb_files = [lcb_root / "test.jsonl"] + [lcb_root / f"test{index}.jsonl" for index in range(2, 7)]
    require_paths(lcb_files, "LiveCodeBench v6 files")
    if any(path.stat().st_size == 0 for path in lcb_files):
        raise SystemExit("One or more LiveCodeBench v6 files are empty.")
    require_paths(
        [
            root / "third_party/evalplus/evalplus",
            root / "third_party/trd/recipe/code_evaluation/external/LiveCodeBench/lcb_runner",
        ],
        "code evaluation sources",
    )
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if not get_human_eval_plus(version="default") or not get_mbpp_plus(version="default"):
        raise SystemExit("EvalPlus datasets are empty.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scope", choices=["all", "train", "eval"], default="all")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.scope in {"all", "train"}:
        check_train(root)
    if args.scope in {"all", "eval"}:
        check_eval(root)
    print(f"Data/evaluation preflight passed (scope={args.scope}).")


if __name__ == "__main__":
    main()
