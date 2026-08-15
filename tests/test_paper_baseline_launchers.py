"""Contracts for the model-scoped SFT and GRPO paper baselines."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import stat
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
MODEL_PATHS = {
    "1B": "Qwen3-1.7B",
    "4B": "Qwen3-4B",
    "8B": "Qwen3-8B",
}
RECIPES = ("grpo", "sft")
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class DryRunCommand:
    def __init__(self, stdout: str):
        line = next(
            (line for line in stdout.splitlines() if line.startswith("TRAIN_CMD:")),
            None,
        )
        if line is None:
            raise AssertionError(f"Missing TRAIN_CMD in dry-run output:\n{stdout}")
        self.tokens = shlex.split(line.removeprefix("TRAIN_CMD:").strip())

    def values(self, option: str) -> list[str]:
        values = []
        for index, token in enumerate(self.tokens):
            if token == option:
                values.append(self.tokens[index + 1])
            elif token.startswith(f"{option}="):
                values.append(token.split("=", 1)[1])
        return values

    def value(self, option: str) -> str:
        values = self.values(option)
        if not values:
            raise AssertionError(f"Missing {option}: {self.tokens}")
        return values[-1]

    def flag_values(self, option: str) -> tuple[str, ...]:
        start = self.tokens.index(option) + 1
        end = start
        while end < len(self.tokens) and not self.tokens[end].startswith("--"):
            end += 1
        return tuple(self.tokens[start:end])

    def has_flag(self, option: str) -> bool:
        return option in self.tokens


class PaperBaselineLauncherTests(unittest.TestCase):
    maxDiff = None

    def run_launcher(
        self, model_scope: str, recipe: str, *extra_args: str
    ) -> DryRunCommand:
        env = os.environ.copy()
        env.update({"DISTILL_DRY_RUN": "1", "WANDB_MODE": "offline"})
        completed = subprocess.run(
            [
                "bash",
                str(SCRIPTS_ROOT / "OPSD" / model_scope / f"{recipe}.sh"),
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
                f"{model_scope}/{recipe}.sh failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return DryRunCommand(completed.stdout)

    def assert_common_paper_configuration(self, command: DryRunCommand) -> None:
        self.assertEqual(command.value("--num_processes"), "8")
        self.assertEqual(command.values("--gradient_accumulation_steps"), ["4", "4"])
        self.assertEqual(command.value("--per_device_train_batch_size"), "1")
        self.assertEqual(command.value("--learning_rate"), "5e-6")
        self.assertEqual(command.value("--lora_r"), "64")
        self.assertEqual(command.value("--lora_alpha"), "128")
        self.assertEqual(command.flag_values("--lora_target_modules"), LORA_TARGETS)
        self.assertEqual(command.value("--bf16"), "True")
        self.assertEqual(command.value("--torch_dtype"), "bfloat16")
        self.assertEqual(command.value("--attn_implementation"), "flash_attention_2")
        self.assertEqual(command.value("--optim"), "adamw_torch_fused")
        self.assertTrue(command.has_flag("--gradient_checkpointing"))
        self.assertTrue(command.has_flag("--use_peft"))
        self.assertFalse(command.has_flag("--num_train_epochs"))

        effective_batch = (
            int(command.value("--num_processes"))
            * int(command.value("--per_device_train_batch_size"))
            * int(command.value("--gradient_accumulation_steps"))
        )
        self.assertEqual(effective_batch, 32)

    def test_all_six_model_scoped_launchers_exist_and_parse(self) -> None:
        expected = [SCRIPTS_ROOT / "lib" / "paper_baseline_common.sh"]
        expected.extend(
            SCRIPTS_ROOT / "OPSD" / model_scope / f"{recipe}.sh"
            for model_scope in MODEL_PATHS
            for recipe in RECIPES
        )
        for script in expected:
            with self.subTest(script=script.relative_to(REPO_ROOT)):
                self.assertTrue(script.is_file())
                self.assertTrue(script.stat().st_mode & stat.S_IXUSR)
                completed = subprocess.run(
                    ["bash", "-n", str(script)],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

        self.assertFalse((SCRIPTS_ROOT / "run_grpo.sh").exists())
        self.assertFalse((SCRIPTS_ROOT / "run_sft.sh").exists())

    def test_grpo_model_matrix_matches_paper_table_6(self) -> None:
        outputs = set()
        for model_scope, model_directory in MODEL_PATHS.items():
            with self.subTest(model_scope=model_scope):
                command = self.run_launcher(model_scope, "grpo")
                self.assert_common_paper_configuration(command)
                self.assertEqual(
                    Path(command.value("--model_name_or_path")).name,
                    model_directory,
                )
                self.assertEqual(command.value("--max_steps"), "500")
                self.assertEqual(command.value("--max_completion_length"), "16000")
                self.assertEqual(command.value("--num_generations"), "8")
                self.assertEqual(command.value("--temperature"), "1.2")
                self.assertEqual(command.value("--beta"), "0.0")
                self.assertEqual(command.value("--loss_type"), "grpo")
                self.assertEqual(command.value("--scale_rewards"), "group")
                self.assertNotIn("epoch", command.value("--run_config"))
                outputs.add(command.value("--run_config"))
        self.assertEqual(len(outputs), 3)

    def test_sft_model_matrix_matches_paper_table_7(self) -> None:
        outputs = set()
        for model_scope, model_directory in MODEL_PATHS.items():
            with self.subTest(model_scope=model_scope):
                command = self.run_launcher(model_scope, "sft")
                self.assert_common_paper_configuration(command)
                self.assertEqual(
                    Path(command.value("--model_name_or_path")).name,
                    model_directory,
                )
                self.assertEqual(command.value("--max_steps"), "100")
                self.assertEqual(command.value("--max_length"), "16000")
                self.assertNotIn("epoch", command.value("--output_dir"))
                outputs.add(command.value("--output_dir"))
        self.assertEqual(len(outputs), 3)

    def test_non_structural_extra_arguments_are_forwarded(self) -> None:
        command = self.run_launcher("4B", "grpo", "--logging_steps", "17")
        self.assertEqual(command.value("--logging_steps"), "17")

    def test_paper_owned_parameters_cannot_be_overridden(self) -> None:
        env = os.environ.copy()
        env.update({"DISTILL_DRY_RUN": "1", "WANDB_MODE": "offline"})
        for argument in ("--max_steps=7", "--model-name-or-path", "--lora_r"):
            with self.subTest(argument=argument):
                completed = subprocess.run(
                    [
                        "bash",
                        str(SCRIPTS_ROOT / "OPSD" / "1B" / "grpo.sh"),
                        argument,
                    ],
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("paper-locked", completed.stderr)


if __name__ == "__main__":
    unittest.main()
