#!/usr/bin/env python3
"""Prepare and validate the pinned code-evaluation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


LCB_REPO_ID = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = ["test.jsonl"] + [f"test{index}.jsonl" for index in range(2, 7)]


def validate_lcb(data_dir: Path) -> None:
    missing = [str(data_dir / name) for name in LCB_FILES if not (data_dir / name).is_file()]
    empty = [str(data_dir / name) for name in LCB_FILES if (data_dir / name).is_file() and not (data_dir / name).stat().st_size]
    if missing or empty:
        raise RuntimeError(f"LiveCodeBench v6 is incomplete; missing={missing}, empty={empty}")


def validate_evalplus() -> tuple[int, int]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    human_eval = len(get_human_eval_plus(version="default"))
    mbpp = len(get_mbpp_plus(version="default"))
    if human_eval <= 0 or mbpp <= 0:
        raise RuntimeError(f"EvalPlus cache is empty: HumanEval+={human_eval}, MBPP+={mbpp}")
    return human_eval, mbpp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_dir = root / "data/eval/livecodebench_code_generation_lite"
    if not args.verify_only:
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=LCB_REPO_ID,
            repo_type="dataset",
            revision=LCB_REVISION,
            local_dir=data_dir,
            allow_patterns=LCB_FILES + ["README.md"],
        )
    validate_lcb(data_dir)
    human_eval, mbpp = validate_evalplus()
    print(
        "Code evaluation data is ready: "
        f"HumanEval+={human_eval}, MBPP+={mbpp}, LiveCodeBench files={len(LCB_FILES)}"
    )


if __name__ == "__main__":
    main()
