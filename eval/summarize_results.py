#!/usr/bin/env python3
"""Create compact JSON/CSV summaries from detailed evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "checkpoint",
    "dataset",
    "val_n",
    "num_problems",
    "pass_at_n_pct",
    "average_at_n_pct",
    "majority_vote_at_n_pct",
    "format_rate",
    "enable_thinking",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()

    result_root = Path(args.result_root).resolve()
    rows = []
    for result_path in sorted(result_root.glob("*/*.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        relative = result_path.relative_to(result_root)
        row = {field: data.get(field) for field in SUMMARY_FIELDS}
        row["checkpoint"] = relative.parts[0]
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No detailed result JSON files found under {result_root}")

    json_path = result_root / "summary.json"
    csv_path = result_root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} evaluation rows to {json_path} and {csv_path}")

    checkpoint_rows = [row for row in rows if row["checkpoint"].startswith("checkpoint-")]
    if checkpoint_rows:
        rows_by_dataset = defaultdict(list)
        for row in checkpoint_rows:
            rows_by_dataset[row["dataset"]].append(row)

        def checkpoint_step(row: dict) -> int:
            match = re.fullmatch(r"checkpoint-(\d+)", row["checkpoint"])
            return int(match.group(1)) if match else 10**18

        best_rows = []
        for dataset in sorted(rows_by_dataset):
            # Match the paper's reporting convention: maximize Avg@N per
            # benchmark. Prefer the earlier checkpoint when scores tie.
            best = max(
                rows_by_dataset[dataset],
                key=lambda row: (
                    float(row["average_at_n_pct"]),
                    -checkpoint_step(row),
                ),
            )
            best_rows.append(best)

        macro_average = sum(float(row["average_at_n_pct"]) for row in best_rows) / len(best_rows)
        macro_pass = sum(float(row["pass_at_n_pct"]) for row in best_rows) / len(best_rows)
        best_payload = {
            "selection_metric": "average_at_n_pct",
            "tie_break": "earliest_checkpoint",
            "num_datasets": len(best_rows),
            "macro_average_at_n_pct": macro_average,
            "macro_pass_at_n_pct": macro_pass,
            "results": best_rows,
        }

        best_json_path = result_root / "best_by_dataset.json"
        best_csv_path = result_root / "best_by_dataset.csv"
        best_json_path.write_text(
            json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with best_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(best_rows)

        val_n = best_rows[0]["val_n"]
        print(
            f"Selected the best Avg@{val_n} checkpoint for {len(best_rows)} datasets: "
            f"macro Avg@{val_n}={macro_average:.2f}%, macro Pass@{val_n}={macro_pass:.2f}%"
        )
        print(f"Best-checkpoint results: {best_json_path} and {best_csv_path}")


if __name__ == "__main__":
    main()
