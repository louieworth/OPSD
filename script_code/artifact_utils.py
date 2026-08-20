#!/usr/bin/env python3
"""Build, verify, and install immutable code-data artifact releases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable


SCHEMA_VERSION = 1
ARTIFACT_ROOTS = (
    "data/train/openthoughts_math_30k_opsd",
    "data/train/taco_code_clean",
    "data/eval/livecodebench_code_generation_lite",
    "third_party/evalplus",
    "third_party/trd/recipe/code_evaluation/external/LiveCodeBench",
    ".cache/xdg/evalplus",
)
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


class ArtifactError(RuntimeError):
    """Raised when an artifact is incomplete, unsafe, or corrupt."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArtifactError(f"Unsafe artifact path: {value!r}")
    return Path(*pure.parts)


def _is_excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    if relative.suffix in {".pyc", ".pyo"}:
        return True
    lcb_prefix = Path("third_party/trd/recipe/code_evaluation/external/LiveCodeBench")
    try:
        lcb_relative = relative.relative_to(lcb_prefix)
    except ValueError:
        return False
    return bool(lcb_relative.parts and lcb_relative.parts[0] == "output")


def _require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ArtifactError(f"{label} escapes its root: {path} -> {resolved}") from error
    return resolved


def collect_files(repo_root: Path, artifact_roots: Iterable[str] = ARTIFACT_ROOTS) -> list[Path]:
    repo_root = repo_root.resolve()
    files: list[Path] = []
    missing: list[str] = []
    for relative_root in artifact_roots:
        component = repo_root / safe_relative_path(relative_root)
        if not component.is_dir():
            missing.append(relative_root)
            continue
        for path in component.rglob("*"):
            relative = path.relative_to(repo_root)
            if _is_excluded(relative):
                continue
            if path.is_file():
                _require_within(path, repo_root, "Artifact source")
                files.append(relative)
    if missing:
        raise ArtifactError(
            "Missing prepared artifact components: "
            + ", ".join(missing)
            + ". Run 'bash script_code/prepare_code.sh all' first."
        )
    if not files:
        raise ArtifactError("No artifact files were found")
    return sorted(set(files), key=lambda path: path.as_posix())


def build_manifest(
    repo_root: Path,
    artifact_roots: Iterable[str] = ARTIFACT_ROOTS,
    *,
    source_git_commit: str | None = None,
) -> dict:
    repo_root = repo_root.resolve()
    roots = tuple(artifact_roots)
    entries = []
    for relative in collect_files(repo_root, roots):
        source = repo_root / relative
        entries.append(
            {
                "path": relative.as_posix(),
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "components": list(roots),
        "files": entries,
    }
    if source_git_commit:
        identity["source_git_commit"] = source_git_commit
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    artifact_id = hashlib.sha256(canonical).hexdigest()
    manifest = dict(identity)
    manifest["artifact_id"] = artifact_id
    manifest["total_bytes"] = sum(entry["size"] for entry in entries)
    return manifest


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported artifact schema: {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    components = manifest.get("components")
    files = manifest.get("files")
    if not isinstance(components, list) or not components:
        raise ArtifactError("Artifact manifest has no components")
    if not isinstance(files, list) or not files:
        raise ArtifactError("Artifact manifest has no files")
    allowed_roots = [safe_relative_path(value) for value in components]
    seen: set[Path] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ArtifactError("Artifact manifest contains a non-object file entry")
        relative = safe_relative_path(str(entry.get("path", "")))
        if relative in seen:
            raise ArtifactError(f"Duplicate artifact path: {relative}")
        seen.add(relative)
        if not any(relative == root or root in relative.parents for root in allowed_roots):
            raise ArtifactError(f"Artifact path is outside declared components: {relative}")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise ArtifactError(f"Invalid size for artifact path: {relative}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re_full_sha256(digest):
            raise ArtifactError(f"Invalid SHA-256 for artifact path: {relative}")
    identity = {
        "schema_version": manifest["schema_version"],
        "components": components,
        "files": files,
    }
    if manifest.get("source_git_commit"):
        identity["source_git_commit"] = manifest["source_git_commit"]
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    expected_id = hashlib.sha256(canonical).hexdigest()
    if manifest.get("artifact_id") != expected_id:
        raise ArtifactError("Artifact manifest identity does not match its contents")
    if manifest.get("total_bytes") != sum(entry["size"] for entry in files):
        raise ArtifactError("Artifact manifest total_bytes is incorrect")
    release = manifest.get("release")
    if release is not None and release != f"v1-{expected_id[:16]}":
        raise ArtifactError("Artifact release name does not match its immutable identity")


def re_full_sha256(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value) and len(value) == 64


def verify_release(release_root: Path, manifest: dict, *, hashes: bool = True) -> None:
    validate_manifest(manifest)
    release_root = release_root.resolve()
    for entry in manifest["files"]:
        relative = safe_relative_path(entry["path"])
        path = release_root / relative
        if not path.is_file():
            raise ArtifactError(f"Missing artifact file: {relative}")
        _require_within(path, release_root, "Artifact file")
        if path.stat().st_size != entry["size"]:
            raise ArtifactError(f"Artifact size mismatch: {relative}")
        if hashes and sha256_file(path) != entry["sha256"]:
            raise ArtifactError(f"Artifact SHA-256 mismatch: {relative}")


def _link_or_copy(source: Path, destination: Path, *, allow_hardlink: bool = True) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        if allow_hardlink:
            try:
                os.link(source.resolve(), temporary)
            except OSError:
                shutil.copy2(source.resolve(), temporary)
        else:
            shutil.copy2(source.resolve(), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_release(repo_root: Path, stage_root: Path, manifest: dict) -> None:
    validate_manifest(manifest)
    repo_root = repo_root.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        relative = safe_relative_path(entry["path"])
        source = repo_root / relative
        destination = stage_root / relative
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == entry["size"]
            and sha256_file(destination) == entry["sha256"]
        ):
            continue
        _link_or_copy(source, destination)
    manifest_path = stage_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_release(stage_root, manifest)


def install_release(release_root: Path, target_root: Path, manifest: dict) -> None:
    verify_release(release_root, manifest)
    release_root = release_root.resolve()
    target_root = target_root.resolve()
    for entry in manifest["files"]:
        relative = safe_relative_path(entry["path"])
        source = release_root / relative
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require_within(destination.parent, target_root, "Artifact destination")
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == entry["size"]
            and sha256_file(destination) == entry["sha256"]
        ):
            continue
        # Training/evaluation data is immutable and large, so it can share disk
        # blocks with an S3-synced release. Evaluator source and caches are
        # copied because pip/package initialization may modify them.
        _link_or_copy(source, destination, allow_hardlink=relative.parts[0] == "data")


def load_manifest(release_root: Path) -> dict:
    path = release_root / "artifact_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read artifact manifest at {path}: {error}") from error
    validate_manifest(manifest)
    return manifest
