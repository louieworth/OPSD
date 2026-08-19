"""Contract tests for the public distillation shell launchers.

These tests intentionally exercise only ``DISTILL_DRY_RUN=1``.  They verify
argument construction and launcher routing without starting Accelerate, vLLM,
evaluation, or a GPU workload.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import stat
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

MODEL_MATRIX = {
    "OPD": ("1.7b", "4b"),
    "OPSD": ("1.7b", "4b", "8b"),
}
MODEL_SCOPES = {
    "OPD": {"1B": "1.7b", "4B": "4b"},
    "OPSD": {"1B": "1.7b", "4B": "4b", "8B": "8b"},
}
MODEL_DIRECTORY = {
    "1.7b": "Qwen3-1.7B",
    "4b": "Qwen3-4B",
    "8b": "Qwen3-8B",
}
VARIANTS = ("vanilla", "top_k", "clip", "skd", "trd")


class DryRun:
    def __init__(self, completed: subprocess.CompletedProcess[str]):
        self.completed = completed
        self.command = self._parse_command()
        self.eval_experiment_dir = self._parse_path("EVAL_EXPERIMENT_DIR:")
        self.result_root = self._parse_path("RESULT_ROOT:")

    def _label_value(self, label: str) -> str:
        for line in self.completed.stdout.splitlines():
            if line.startswith(label):
                return line.removeprefix(label).strip()
        raise AssertionError(
            f"dry-run output omitted {label!r}:\n{self.completed.stdout}"
        )

    def _parse_command(self) -> list[str]:
        value = self._label_value("TRAIN_CMD:")
        command = shlex.split(value)
        if not command:
            raise AssertionError("TRAIN_CMD is empty")
        return command

    def _parse_path(self, label: str) -> Path:
        values = shlex.split(self._label_value(label))
        if len(values) != 1:
            raise AssertionError(f"{label} must contain exactly one shell word: {values}")
        return Path(values[0])

    def values(self, option: str) -> list[str]:
        """Return values for both ``--name value`` and ``--name=value``."""

        values: list[str] = []
        for index, token in enumerate(self.command):
            if token == option:
                if index + 1 == len(self.command):
                    raise AssertionError(f"{option} has no value in TRAIN_CMD")
                values.append(self.command[index + 1])
            elif token.startswith(f"{option}="):
                values.append(token.split("=", 1)[1])
        return values

    def value(self, option: str) -> str:
        values = self.values(option)
        if not values:
            raise AssertionError(f"TRAIN_CMD omitted {option}: {self.command}")
        return values[-1]

    def has_flag(self, option: str) -> bool:
        return option in self.command


class LauncherContractTests(unittest.TestCase):
    maxDiff = None

    def run_launcher(
        self,
        source: str,
        variant: str,
        model: str,
        *extra_args: str,
        expect_success: bool = True,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for variable in (
            "DISTILL_MAX_STEPS",
            "DISTILL_POLICY_GRADIENT_UPDATES",
            "DISTILL_MAX_COMPLETION_LENGTH",
            "DISTILL_MAX_REFINEMENT_LENGTH",
            "DISTILL_MAX_LENGTH",
            "DISTILL_SAVE_STEPS",
            "TRD_MAX_STEPS",
            "TRD_POLICY_GRADIENT_UPDATES",
            "TRD_MAX_COMPLETION_LENGTH",
            "TRD_MAX_REFINEMENT_LENGTH",
            "TRD_MAX_LENGTH",
            "TRD_REFINEMENT_MAX_MODEL_LEN",
            "TRD_SAVE_STEPS",
            "RUN_CONFIG",
            "RESULT_ROOT",
        ):
            env.pop(variable, None)
        env.update(
            {
                "AUTO_EVAL": "0",
                "DISTILL_DRY_RUN": "1",
                "WANDB_MODE": "offline",
            }
        )
        if env_overrides:
            env.update(env_overrides)
        completed = subprocess.run(
            [
                "bash",
                str(SCRIPTS_ROOT / source / f"{variant}.sh"),
                model,
                *extra_args,
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        if expect_success and completed.returncode != 0:
            self.fail(
                f"{source}/{variant}.sh {model} failed with "
                f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed

    def run_scoped_launcher(
        self,
        source: str,
        model_scope: str,
        variant: str,
        *extra_args: str,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for variable in (
            "DISTILL_MAX_STEPS",
            "DISTILL_POLICY_GRADIENT_UPDATES",
            "DISTILL_MAX_COMPLETION_LENGTH",
            "DISTILL_MAX_REFINEMENT_LENGTH",
            "DISTILL_MAX_LENGTH",
            "DISTILL_SAVE_STEPS",
            "TRD_MAX_STEPS",
            "TRD_POLICY_GRADIENT_UPDATES",
            "TRD_MAX_COMPLETION_LENGTH",
            "TRD_MAX_REFINEMENT_LENGTH",
            "TRD_MAX_LENGTH",
            "TRD_REFINEMENT_MAX_MODEL_LEN",
            "TRD_SAVE_STEPS",
            "RUN_CONFIG",
            "RESULT_ROOT",
        ):
            env.pop(variable, None)
        env.update(
            {
                "AUTO_EVAL": "0",
                "DISTILL_DRY_RUN": "1",
                "WANDB_MODE": "offline",
            }
        )
        completed = subprocess.run(
            [
                "bash",
                str(SCRIPTS_ROOT / source / model_scope / f"{variant}.sh"),
                *extra_args,
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"{source}/{model_scope}/{variant}.sh failed with "
                f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed

    def test_expected_shell_structure_and_bash_syntax(self) -> None:
        expected = [
            *(SCRIPTS_ROOT / source / f"{variant}.sh"
              for source in MODEL_MATRIX
              for variant in VARIANTS),
            *(SCRIPTS_ROOT / source / model_scope / f"{variant}.sh"
              for source, scopes in MODEL_SCOPES.items()
              for model_scope in scopes
              for variant in VARIANTS),
            SCRIPTS_ROOT / "lib" / "distill_common.sh",
            SCRIPTS_ROOT / "lib" / "trd_common.sh",
            SCRIPTS_ROOT / "lib" / "math_segment_common.sh",
        ]
        for script in expected:
            with self.subTest(script=script.relative_to(REPO_ROOT)):
                self.assertTrue(script.is_file(), f"missing launcher/helper: {script}")
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                mode = script.stat().st_mode
                self.assertTrue(
                    mode & stat.S_IXUSR,
                    f"launcher/helper is not executable: {script}",
                )

        self.assertFalse(
            (SCRIPTS_ROOT / "TRD").exists(),
            "legacy scripts/TRD must be folded into OPD/OPSD canonical launchers",
        )

    def test_allowed_model_matrix_builds_the_expected_model_command(self) -> None:
        for source, models in MODEL_MATRIX.items():
            for variant in VARIANTS:
                for model in models:
                    with self.subTest(source=source, variant=variant, model=model):
                        dry_run = DryRun(self.run_launcher(source, variant, model))
                        self.assertEqual(source.lower(), dry_run.value("--alg"))
                        self.assertEqual(
                            MODEL_DIRECTORY[model],
                            Path(dry_run.value("--model_name_or_path")).name,
                        )

    def test_model_scoped_launchers_bind_model_teacher_and_extra_arguments(self) -> None:
        observed = 0
        for source, scopes in MODEL_SCOPES.items():
            for model_scope, model in scopes.items():
                for variant in VARIANTS:
                    with self.subTest(
                        source=source, model_scope=model_scope, variant=variant
                    ):
                        dry_run = DryRun(
                            self.run_scoped_launcher(
                                source,
                                model_scope,
                                variant,
                                "--logging_steps",
                                "17",
                            )
                        )
                        observed += 1
                        self.assertEqual(source.lower(), dry_run.value("--alg"))
                        self.assertEqual(
                            MODEL_DIRECTORY[model],
                            Path(dry_run.value("--model_name_or_path")).name,
                        )
                        self.assertEqual(
                            "17",
                            dry_run.value("--logging_steps"),
                            "model-scoped launcher must forward extra arguments",
                        )

                        if source == "OPD":
                            self.assertFalse(dry_run.has_flag("--fixed_teacher"))
                            self.assertEqual(
                                "Qwen3-8B",
                                Path(
                                    dry_run.value("--teacher_model_name_or_path")
                                ).name,
                            )
                            self.assertEqual(
                                "False", dry_run.value("--teacher_thinking")
                            )
                        else:
                            self.assertTrue(dry_run.has_flag("--fixed_teacher"))
                            self.assertEqual(
                                [],
                                dry_run.values("--teacher_model_name_or_path"),
                                "OPSD fixed teacher must be the student's base model",
                            )
                            self.assertEqual(
                                "True", dry_run.value("--teacher_thinking")
                            )

        self.assertEqual(25, observed)

    def test_opd_rejects_8b_for_every_variant(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                completed = self.run_launcher(
                    "OPD", variant, "8b", expect_success=False
                )
                self.assertNotEqual(0, completed.returncode)
                diagnostic = f"{completed.stdout}\n{completed.stderr}".lower()
                self.assertIn("8b", diagnostic)
                self.assertTrue(
                    "unsupported" in diagnostic or "does not support" in diagnostic,
                    diagnostic,
                )

    def test_variant_loss_flags_do_not_confuse_sampling_top_k(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                # OPSD/1.7b is sufficient here: the common helper owns the
                # recipe flags shared by all supported source/model pairs.
                dry_run = DryRun(self.run_launcher("OPSD", variant, "1.7b"))
                self.assertEqual(
                    "20",
                    dry_run.value("--top_k"),
                    "sampling --top_k must remain independent from loss top-k",
                )

                loss_top_k = int(dry_run.value("--top_k_loss"))
                token_clip = float(dry_run.value("--jsd_token_clip"))
                if variant == "top_k":
                    self.assertEqual(16, loss_top_k)
                    self.assertEqual(0.0, token_clip)
                elif variant == "clip":
                    self.assertEqual(0, loss_top_k)
                    self.assertGreater(token_clip, 0.0)
                else:
                    self.assertEqual(0, loss_top_k)
                    self.assertEqual(0.0, token_clip)

                self.assertEqual(
                    variant == "trd", dry_run.has_flag("--teacher_refine")
                )
                if variant == "trd":
                    self.assertEqual(
                        "1.0", dry_run.value("--distillation_temperature")
                    )

    def test_skd_uses_math_prompts_local_rollout_and_official_unique_parameters(self) -> None:
        for source in MODEL_MATRIX:
            with self.subTest(source=source):
                skd = DryRun(self.run_launcher(source, "skd", "1.7b"))
                vanilla = DryRun(self.run_launcher(source, "vanilla", "1.7b"))

                self.assertEqual("math", skd.value("--task_type"))
                self.assertEqual("skd", skd.value("--trajectory_mode"))
                self.assertFalse(skd.has_flag("--use_vllm"))
                self.assertEqual([], skd.values("--vllm_mode"))
                self.assertEqual("5", skd.value("--skd_draft_length"))
                self.assertEqual("25", skd.value("--skd_accept_top_k"))
                self.assertEqual("0.2", skd.value("--skd_correction_temperature"))
                self.assertEqual("1.0", skd.value("--skd_correction_top_p"))
                self.assertEqual("0", skd.value("--top_k_loss"))
                self.assertEqual("0", skd.value("--jsd_token_clip"))

                # Strict comparison: SKD retains the same student sampler as
                # ordinary Math KD; only speculative verification differs.
                for option in ("--temperature", "--top_p", "--top_k"):
                    self.assertEqual(vanilla.value(option), skd.value(option))

    def test_skd_recipe_parameters_are_launcher_owned(self) -> None:
        for option in (
            "--trajectory_mode",
            "--skd_draft_length",
            "--skd_accept_top_k",
            "--skd_correction_temperature",
            "--skd_correction_top_p",
        ):
            with self.subTest(option=option):
                completed = self.run_launcher(
                    "OPSD", "skd", "1.7b", option, "999", expect_success=False
                )
                self.assertNotEqual(0, completed.returncode)
                self.assertIn("launcher-owned", completed.stderr)

    def test_all_variants_use_distill_rollout_update_schedule_and_ga_one(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                dry_run = DryRun(
                    self.run_launcher(
                        "OPSD",
                        variant,
                        "4b",
                        env_overrides={
                            "DISTILL_MAX_STEPS": "12",
                            "DISTILL_POLICY_GRADIENT_UPDATES": "3",
                        },
                    )
                )
                self.assertEqual("12", dry_run.value("--max_steps"))
                self.assertEqual("3", dry_run.value("--policy_gradient_updates"))
                self.assertEqual(
                    ["1", "1"], dry_run.values("--gradient_accumulation_steps")
                )
                self.assertEqual("1", dry_run.value("--save_total_limit"))
                self.assertEqual("True", dry_run.value("--skip_final_model_save"))

    def test_trd_legacy_schedule_environment_remains_supported(self) -> None:
        dry_run = DryRun(
            self.run_launcher(
                "OPD",
                "trd",
                "1.7b",
                env_overrides={
                    "TRD_MAX_STEPS": "12",
                    "TRD_POLICY_GRADIENT_UPDATES": "3",
                    "TRD_MAX_COMPLETION_LENGTH": "256",
                    "TRD_MAX_REFINEMENT_LENGTH": "128",
                },
            )
        )
        self.assertEqual("12", dry_run.value("--max_steps"))
        self.assertEqual("3", dry_run.value("--policy_gradient_updates"))
        self.assertEqual("256", dry_run.value("--max_completion_length"))
        self.assertEqual("128", dry_run.value("--max_refinement_length"))

    def test_trd_default_token_budgets_for_every_model_scoped_launcher(self) -> None:
        expected_by_source = {
            "OPD": {
                "max_length": 20_000,
                "refinement_max_model_len": 20_000,
                "refinement_prompt_budget": 18_976,
            },
            "OPSD": {
                "max_length": 21_024,
                "refinement_max_model_len": 21_024,
                "refinement_prompt_budget": 20_000,
            },
        }

        for source, scopes in MODEL_SCOPES.items():
            expected = expected_by_source[source]
            for model_scope in scopes:
                with self.subTest(source=source, model_scope=model_scope):
                    dry_run = DryRun(
                        self.run_scoped_launcher(source, model_scope, "trd")
                    )
                    completion_length = int(
                        dry_run.value("--max_completion_length")
                    )
                    refinement_length = int(
                        dry_run.value("--max_refinement_length")
                    )
                    refinement_max_model_len = int(
                        dry_run.value("--refinement_vllm_max_model_len")
                    )

                    self.assertEqual(1_024, completion_length, "y_o budget")
                    self.assertEqual(1_024, refinement_length, "y_r budget")
                    self.assertEqual(
                        expected["max_length"], int(dry_run.value("--max_length"))
                    )
                    self.assertEqual(
                        expected["refinement_max_model_len"],
                        refinement_max_model_len,
                    )
                    self.assertEqual(
                        expected["refinement_prompt_budget"],
                        refinement_max_model_len - refinement_length,
                    )

    def test_each_source_variant_and_model_has_unique_output_and_eval_namespace(self) -> None:
        training_namespaces: dict[Path, tuple[str, str, str]] = {}
        evaluation_namespaces: dict[Path, tuple[str, str, str]] = {}
        result_namespaces: dict[Path, tuple[str, str, str]] = {}

        for source, scopes in MODEL_SCOPES.items():
            for model_scope, model in scopes.items():
                for variant in VARIANTS:
                    key = (source, variant, model)
                    with self.subTest(
                        source=source, model_scope=model_scope, variant=variant
                    ):
                        dry_run = DryRun(
                            self.run_scoped_launcher(source, model_scope, variant)
                        )
                        output_dir = Path(dry_run.value("--output_dir"))
                        run_config = dry_run.value("--run_config")
                        training_namespace = output_dir / run_config

                        self.assertEqual(
                            training_namespace,
                            dry_run.eval_experiment_dir,
                            "evaluation must consume this launcher's training output",
                        )
                        self.assertNotIn(training_namespace, training_namespaces)
                        self.assertNotIn(
                            dry_run.eval_experiment_dir, evaluation_namespaces
                        )
                        self.assertNotIn(dry_run.result_root, result_namespaces)

                        training_namespaces[training_namespace] = key
                        evaluation_namespaces[dry_run.eval_experiment_dir] = key
                        result_namespaces[dry_run.result_root] = key

        expected_count = sum(len(scopes) for scopes in MODEL_SCOPES.values()) * len(
            VARIANTS
        )
        self.assertEqual(expected_count, len(training_namespaces))
        self.assertEqual(expected_count, len(evaluation_namespaces))
        self.assertEqual(expected_count, len(result_namespaces))

if __name__ == "__main__":
    unittest.main()
