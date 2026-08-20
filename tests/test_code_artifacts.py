from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from script_code.artifact_utils import (
    ArtifactError,
    build_manifest,
    install_release,
    load_manifest,
    stage_release,
    verify_release,
)
from script_code.fetch_artifacts import verify_source_revision


class CodeArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "data/train").mkdir(parents=True)
        (self.root / "third_party/eval").mkdir(parents=True)
        (self.root / "data/train/examples.parquet").write_bytes(b"train-data")
        (self.root / "third_party/eval/test.jsonl").write_text('{"id": 1}\n')
        self.components = ("data/train", "third_party/eval")

    def tearDown(self):
        self.temporary.cleanup()

    def test_stage_verify_and_install_release(self):
        manifest = build_manifest(self.root, self.components, source_git_commit="abc123")
        manifest["release"] = f"v1-{manifest['artifact_id'][:16]}"
        stage = self.root / "stage"
        stage_release(self.root, stage, manifest)

        loaded = load_manifest(stage)
        self.assertEqual(loaded["artifact_id"], manifest["artifact_id"])
        verify_release(stage, loaded)

        target = self.root / "target"
        install_release(stage, target, loaded)
        train_source = stage / "data/train/examples.parquet"
        train_target = target / "data/train/examples.parquet"
        evaluator_source = stage / "third_party/eval/test.jsonl"
        evaluator_target = target / "third_party/eval/test.jsonl"
        self.assertEqual(train_target.read_bytes(), b"train-data")
        self.assertEqual(evaluator_target.read_text(), '{"id": 1}\n')
        self.assertEqual(train_source.stat().st_ino, train_target.stat().st_ino)
        self.assertNotEqual(evaluator_source.stat().st_ino, evaluator_target.stat().st_ino)

    def test_verify_detects_corruption(self):
        manifest = build_manifest(self.root, self.components)
        stage = self.root / "stage"
        stage_release(self.root, stage, manifest)
        (stage / "data/train/examples.parquet").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ArtifactError, "mismatch"):
            verify_release(stage, manifest)

    def test_manifest_rejects_path_traversal(self):
        manifest = build_manifest(self.root, self.components)
        manifest["files"][0]["path"] = "../outside"
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "artifact_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ArtifactError, "Unsafe artifact path"):
            load_manifest(stage)

    def test_source_revision_must_be_verifiable_and_match(self):
        expected = "a" * 40
        manifest = {"source_git_commit": expected}
        non_repository = self.root / "not-a-repository"
        non_repository.mkdir()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ArtifactError, "Cannot verify"):
                verify_source_revision(non_repository, manifest, allow_mismatch=False)
        with patch.dict("os.environ", {"OPSD_GIT_REV": expected}, clear=True):
            verify_source_revision(non_repository, manifest, allow_mismatch=False)
        with patch.dict("os.environ", {"OPSD_GIT_REV": "b" * 40}, clear=True):
            with self.assertRaisesRegex(ArtifactError, "mismatch"):
                verify_source_revision(non_repository, manifest, allow_mismatch=False)

    def test_prepared_image_builds_once_and_jobs_only_verify(self):
        repo_root = Path(__file__).resolve().parents[1]
        direct = (repo_root / "deploy/docker/Dockerfile.prepared-code").read_text()
        from_hf = (repo_root / "deploy/docker/Dockerfile.prepared-from-hf").read_text()
        entrypoint = (repo_root / "deploy/docker/baked_code_job_entrypoint.sh").read_text()

        self.assertIn("bash script_code/prepare_code.sh all", direct)
        self.assertIn("--mount=type=secret,id=hf_token", direct)
        self.assertIn("python script_code/fetch_artifacts.py", from_hf)
        self.assertIn("bash script_code/prepare_code.sh artifact", from_hf)
        self.assertNotIn("ARG HF_TOKEN", direct)
        self.assertNotIn("ENV HF_TOKEN", direct)
        self.assertNotIn("ARG HF_TOKEN", from_hf)
        self.assertNotIn("ENV HF_TOKEN", from_hf)
        self.assertIn("bash script_code/prepare_code.sh verify", entrypoint)
        self.assertNotIn("prepare_code.sh all", entrypoint)


if __name__ == "__main__":
    unittest.main()
