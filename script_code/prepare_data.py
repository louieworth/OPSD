#!/usr/bin/env python3
"""Prepare the single clean TACO manifest shared by all code experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from script_code.code_data import build_code_problem, render_code_prompt
from script_code.code_execution import parse_test_spec


TACO_REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"
EXPECTED_CLEAN_ROWS = 18_862


def load_prepared_if_valid(args: argparse.Namespace) -> Dataset | None:
    """Return an existing pinned artifact, or None when it must be rebuilt."""
    output = Path(args.output_dir)
    manifest_path = output / "manifest.json"
    parquet_path = output / "data/train-00000-of-00001.parquet"
    if args.force or not manifest_path.is_file() or not parquet_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "revision": args.revision,
            "tokenizer_revision": args.tokenizer_revision,
            "seed": args.seed,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_solution_tokens": args.max_solution_tokens,
            "rows": args.expected_count,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        if pq.read_metadata(parquet_path).num_rows != args.expected_count:
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return Dataset.from_parquet(str(parquet_path))


def first_solution(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
    else:
        parsed = raw
    candidates = parsed if isinstance(parsed, list) else [parsed]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def normalize_test_spec(raw: Any) -> str:
    if isinstance(raw, str):
        spec = parse_test_spec(raw)
    else:
        spec = parse_test_spec(raw or {})
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def prepare(args: argparse.Namespace) -> Dataset:
    prepared = load_prepared_if_valid(args)
    if prepared is not None:
        print(f"Clean TACO artifact is already valid: {args.output_dir}")
        return prepared

    if args.input_path:
        input_path = Path(args.input_path)
        arrow_files = sorted(input_path.rglob("*.arrow")) if input_path.is_dir() else []
        if arrow_files:
            source = concatenate_datasets(
                [Dataset.from_file(str(path)) for path in arrow_files]
            )
        else:
            source = load_from_disk(args.input_path)
            if hasattr(source, "keys") and "train" in source:
                source = source["train"]
    else:
        source = load_dataset(
            "BAAI/TACO",
            split="train",
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
    tokenizer_source = Path(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        revision=None if tokenizer_source.exists() else args.tokenizer_revision,
        local_files_only=args.offline,
    )

    rows = []
    rejection_counts = {"empty_solution": 0, "invalid_tests": 0, "prompt": 0, "solution": 0}
    for index, example in enumerate(source):
        solution = first_solution(example.get("solutions"))
        if not solution:
            rejection_counts["empty_solution"] += 1
            continue
        try:
            test_spec = normalize_test_spec(example.get("input_output"))
        except (ValueError, json.JSONDecodeError, TypeError):
            rejection_counts["invalid_tests"] += 1
            continue
        problem = build_code_problem(example.get("question", ""), example.get("starter_code", ""))
        rendered = render_code_prompt(tokenizer, problem, thinking=False)
        # We only need to know whether the limit is exceeded.  Capping at one
        # token beyond the boundary avoids fully tokenizing pathological TACO
        # rows while preserving the exact inclusion rule.
        prompt_tokens = len(
            tokenizer.encode(
                rendered,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_prompt_tokens + 1,
            )
        )
        solution_tokens = len(
            tokenizer.encode(
                solution,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_solution_tokens + 1,
            )
        )
        if prompt_tokens > args.max_prompt_tokens:
            rejection_counts["prompt"] += 1
            continue
        if solution_tokens > args.max_solution_tokens:
            rejection_counts["solution"] += 1
            continue
        stable_id = hashlib.sha256(
            f"{args.revision}:{index}:{example.get('question', '')}".encode()
        ).hexdigest()[:24]
        rows.append(
            {
                "problem_id": stable_id,
                "problem": problem,
                "solution": solution,
                "input_output": test_spec,
                "difficulty": str(example.get("difficulty", "")),
                "source": str(example.get("source", "")),
            }
        )

    if args.expected_count and len(rows) != args.expected_count:
        raise RuntimeError(
            f"Clean TACO row count changed: expected {args.expected_count}, got {len(rows)}; "
            f"rejections={rejection_counts}. Check the pinned revision and tokenizer."
        )
    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
    output = Path(args.output_dir)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(data_dir / "train-00000-of-00001.parquet")
    manifest = {
        "dataset": "BAAI/TACO",
        "revision": args.revision,
        "tokenizer": str(Path(args.tokenizer_path).resolve()),
        "tokenizer_revision": args.tokenizer_revision,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_solution_tokens": args.max_solution_tokens,
        "rows": len(dataset),
        "rejections": rejection_counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument(
        "--tokenizer_revision",
        default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument("--input_path")
    parser.add_argument("--cache_dir")
    parser.add_argument("--revision", default=TACO_REVISION)
    parser.add_argument("--max_prompt_tokens", type=int, default=2_048)
    parser.add_argument("--max_solution_tokens", type=int, default=4_096)
    parser.add_argument("--expected_count", type=int, default=EXPECTED_CLEAN_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepared = prepare(parse_args())
    print(f"Prepared {len(prepared)} clean TACO examples.")
