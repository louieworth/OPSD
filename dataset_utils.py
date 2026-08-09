"""Helpers for loading prepared, repository-local Parquet datasets."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset


REPO_ROOT = Path(__file__).resolve().parent
DATASETS_CACHE = REPO_ROOT / ".cache" / "huggingface" / "datasets"


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def load_local_parquet(
    path: str | Path,
    columns: list[str],
    split: str = "train",
) -> Dataset:
    dataset_path = resolve_repo_path(path)
    if dataset_path.is_file() and dataset_path.suffix == ".parquet":
        parquet_files = [dataset_path]
    elif dataset_path.is_dir():
        parquet_files = sorted((dataset_path / "data").glob("*.parquet"))
        if not parquet_files:
            parquet_files = sorted(dataset_path.glob("*.parquet"))
    else:
        parquet_files = []

    if not parquet_files:
        raise FileNotFoundError(
            f"No prepared Parquet files found under {dataset_path}. "
            "Run `bash scripts/prepare_data.sh` first."
        )

    DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        "parquet",
        data_files={split: [str(file) for file in parquet_files]},
        split=split,
        cache_dir=str(DATASETS_CACHE),
        columns=columns,
        features=Features({column: Value("string") for column in columns}),
    )
