#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evalplus_metrics(root: Path, dataset: str, pass_k: int) -> tuple[float, float]:
    candidates = sorted((root / dataset).rglob("*eval_results*.json"))
    if not candidates:
        raise SystemExit(f"No EvalPlus results found for {dataset} under {root}")
    data = json.loads(candidates[-1].read_text()).get("eval", {})
    rows = []
    for samples in data.values():
        limited = samples[:pass_k]
        if not limited:
            continue
        correct = sum(
            1
            for sample in limited
            if str(sample.get("base_status", "")).lower() == "pass"
            and str(sample.get("plus_status", "pass")).lower() == "pass"
        )
        rows.append((correct / len(limited), float(correct > 0)))
    if not rows:
        raise SystemExit("EvalPlus result contained no samples.")
    return sum(row[0] for row in rows) / len(rows), sum(row[1] for row in rows) / len(rows)


def lcb_metrics(root: Path, pass_k: int) -> tuple[float, float]:
    candidates = sorted(root.rglob("*_eval_all.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No LiveCodeBench eval_all result found under {root}")
    rows = []
    for item in json.loads(candidates[-1].read_text()):
        graded = (item.get("graded_list") or [])[:pass_k]
        if graded:
            correct = sum(bool(value) for value in graded)
            rows.append((correct / len(graded), float(correct > 0)))
    if not rows:
        raise SystemExit("LiveCodeBench result contained no graded samples.")
    return sum(row[0] for row in rows) / len(rows), sum(row[1] for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["evalplus", "lcb"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--pass_k", type=int, default=12)
    args = parser.parse_args()
    if args.kind == "evalplus":
        avg, passed = evalplus_metrics(Path(args.root), args.dataset, args.pass_k)
    else:
        avg, passed = lcb_metrics(Path(args.root), args.pass_k)
    print(f"avg={avg}")
    print(f"pass={passed}")


if __name__ == "__main__":
    main()
