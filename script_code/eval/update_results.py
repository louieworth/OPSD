#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--avg", required=True, type=float)
    parser.add_argument("--pass_value", required=True, type=float)
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()
    path = Path(args.file)
    data = json.loads(path.read_text()) if path.is_file() else {}
    entry = data.setdefault(args.model, {})
    entry[f"{args.dataset}_avg{args.k}"] = args.avg
    entry[f"{args.dataset}_pass{args.k}"] = args.pass_value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
