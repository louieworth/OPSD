from pathlib import Path
import json
import os
import shlex
import subprocess
import tempfile
import unittest

import torch

from script_code.code_data import CodeDistillationDataCollator
from script_code.code_execution import (
    ExecutionLimits,
    evaluate_completion,
    extract_python_code,
    select_test_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "script_code"


class CharacterTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, enable_thinking=False):
        text = f"<thinking={int(enable_thinking)}>{messages[0]['content']}<assistant>"
        if len(messages) > 1:
            text += messages[1]["content"]
        self.rendered.append((enable_thinking, text))
        return text

    def __call__(self, prompts, *, padding, truncation, max_length, add_special_tokens, return_tensors=None):
        rows = [[ord(char) for char in prompt][:max_length] for prompt in prompts]
        if padding == "max_length":
            rows = [row + [0] * (max_length - len(row)) for row in rows]
        masks = [[int(token != 0) for token in row] for row in rows]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}
        return {"input_ids": rows, "attention_mask": masks}


class CodeCollatorTests(unittest.TestCase):
    def test_opsd_renders_student_nonthinking_and_teacher_thinking(self):
        tokenizer = CharacterTokenizer()
        batch = CodeDistillationDataCollator(
            tokenizer,
            alg="opsd",
            student_thinking=False,
            teacher_thinking=True,
        )([{"problem": "add", "solution": "print(1+1)"}])

        self.assertEqual([mode for mode, _ in tokenizer.rendered], [False, True])
        self.assertIn("Reference solution", tokenizer.rendered[1][1])
        self.assertEqual(batch["problems"], ["add"])
        self.assertEqual(batch["reference_solutions"], ["print(1+1)"])

    def test_opd_never_exposes_reference_solution(self):
        tokenizer = CharacterTokenizer()
        batch = CodeDistillationDataCollator(
            tokenizer,
            alg="opd",
            student_thinking=False,
            teacher_thinking=False,
        )([{"problem": "add", "solution": "SECRET"}])

        self.assertEqual([mode for mode, _ in tokenizer.rendered], [False, False])
        self.assertNotIn("SECRET", tokenizer.rendered[1][1])
        self.assertEqual(batch["reference_solutions"], [None])


class CodeExecutionTests(unittest.TestCase):
    def test_extracts_last_python_block(self):
        self.assertEqual(
            extract_python_code("text```python\nprint(1)\n```more```\nprint(2)\n```"),
            "print(2)",
        )

    def test_stdin_and_function_rewards(self):
        limits = ExecutionLimits(max_tests=4, timeout_seconds=2, memory_mb=512)
        stdin_spec = json.dumps({"inputs": ["2\n"], "outputs": ["4\n"]})
        function_spec = json.dumps(
            {"fn_name": "add", "inputs": ["[2, 3]"], "outputs": ["5"]}
        )
        self.assertEqual(
            evaluate_completion("n=int(input()); print(n*n)", stdin_spec, "stdin", limits),
            1.0,
        )
        self.assertEqual(
            evaluate_completion("def add(a,b): return a+b", function_spec, "fn", limits),
            1.0,
        )
        self.assertEqual(evaluate_completion("syntax !!!", stdin_spec, "bad", limits), 0.0)

    def test_test_cap_is_stable_and_not_a_group_size(self):
        first = select_test_indices(128, "problem", 32)
        second = select_test_indices(128, "problem", 32)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(select_test_indices(8, "problem", 32), list(range(8)))


class CodeLauncherTests(unittest.TestCase):
    def dry_run(self, relative_path):
        env = os.environ.copy()
        env["CODE_DRY_RUN"] = "1"
        completed = subprocess.run(
            ["bash", str(SCRIPT_ROOT / relative_path)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        line = next(line for line in completed.stdout.splitlines() if line.startswith("TRAIN_CMD:"))
        return shlex.split(line.removeprefix("TRAIN_CMD:"))

    @staticmethod
    def value(command, option):
        return command[command.index(option) + 1]

    def test_exact_34_model_scoped_launchers_exist(self):
        scoped = (
            list((SCRIPT_ROOT / "Baselines").glob("*B/*.sh"))
            + list((SCRIPT_ROOT / "OPD").glob("*B/*.sh"))
            + list((SCRIPT_ROOT / "OPSD").glob("*B/*.sh"))
        )
        self.assertEqual(len(scoped), 34)
        self.assertEqual(len(list(SCRIPT_ROOT.glob("*/1B/*.sh"))), 13)
        self.assertEqual(len(list(SCRIPT_ROOT.glob("*/4B/*.sh"))), 13)
        self.assertEqual(len(list(SCRIPT_ROOT.glob("*/8B/*.sh"))), 8)

    def test_common_baselines_are_not_nested_under_opsd(self):
        for scope in ("1B", "4B", "8B"):
            for method in ("base", "sft", "grpo"):
                self.assertTrue(
                    (SCRIPT_ROOT / "Baselines" / scope / f"{method}.sh").is_file()
                )
                self.assertFalse((SCRIPT_ROOT / "OPSD" / scope / f"{method}.sh").exists())

    def test_default_matrix_is_grouped_by_model_then_baselines_opd_opsd(self):
        completed = subprocess.run(
            ["bash", str(SCRIPT_ROOT / "run_matrix.sh"), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        commands = [
            shlex.split(line.removeprefix("TRAIN_CMD:"))
            for line in completed.stdout.splitlines()
            if line.startswith("TRAIN_CMD:")
        ]
        self.assertEqual(len(commands), 34)
        expected_groups = [
            (commands[0:3], None),
            (commands[3:8], "opd"),
            (commands[8:13], "opsd"),
            (commands[13:16], None),
            (commands[16:21], "opd"),
            (commands[21:26], "opsd"),
            (commands[26:29], None),
            (commands[29:34], "opsd"),
        ]
        for group, source in expected_groups:
            if source is None:
                self.assertTrue(all("--alg" not in command for command in group))
            else:
                self.assertEqual(
                    [self.value(command, "--alg") for command in group],
                    [source] * len(group),
                )

    def test_kd_sampler_and_h100_defaults_are_strictly_shared(self):
        commands = {
            variant: self.dry_run(f"OPSD/1B/{variant}.sh")
            for variant in ("vanilla", "clip", "top_k", "skd")
        }
        for option, expected in {
            "--temperature": "1.1",
            "--top_p": "0.95",
            "--top_k": "20",
            "--max_completion_length": "4096",
            "--per_device_train_batch_size": "1",
            "--max_steps": "400",
            "--policy_gradient_updates": "100",
            "--teacher_context_max_length": "12288",
        }.items():
            self.assertEqual({self.value(command, option) for command in commands.values()}, {expected})
        self.assertEqual(self.value(commands["skd"], "--skd_draft_length"), "5")
        self.assertEqual(self.value(commands["skd"], "--skd_accept_top_k"), "25")
        self.assertNotIn("--use_vllm", commands["vanilla"])
        self.assertNotIn("--use_vllm", commands["skd"])

    def test_model_matrix_uses_base_checkpoints_and_no_opd_8b(self):
        self.assertEqual(
            Path(self.value(self.dry_run("OPSD/4B/skd.sh"), "--model_name_or_path")).name,
            "Qwen3-4B",
        )
        self.assertFalse((SCRIPT_ROOT / "OPD/8B").exists())

    def test_code_trd_matches_scripts_lifecycle_and_separates_rewrite_from_scoring(self):
        commands = {
            source: self.dry_run(f"{source}/1B/trd.sh")
            for source in ("OPD", "OPSD")
        }
        for command in commands.values():
            self.assertEqual(self.value(command, "--task_type"), "code")
            self.assertEqual(self.value(command, "--max_completion_length"), "1024")
            self.assertEqual(self.value(command, "--max_refinement_length"), "1024")
            self.assertEqual(self.value(command, "--max_length"), "3072")
            self.assertEqual(self.value(command, "--refinement_thinking"), "False")
            self.assertEqual(self.value(command, "--num_processes"), "4")
            self.assertEqual(self.value(command, "--per_device_train_batch_size"), "1")
            self.assertEqual(self.value(command, "--save_steps"), "25")
            self.assertEqual(self.value(command, "--save_total_limit"), "1")
            self.assertEqual(self.value(command, "--skip_final_model_save"), "True")
            self.assertIn("--teacher_refine", command)
            self.assertIn("taco_code_clean", self.value(command, "--train_dataset_path"))

        self.assertEqual(self.value(commands["OPSD"], "--teacher_thinking"), "True")
        self.assertEqual(self.value(commands["OPD"], "--teacher_thinking"), "False")
        self.assertEqual(
            self.value(commands["OPSD"], "--refinement_vllm_max_model_len"),
            "12288",
        )
        self.assertEqual(
            self.value(commands["OPD"], "--refinement_vllm_max_model_len"),
            "6144",
        )

    def test_sft_evaluates_only_at_100_grpo_every_50_and_kd_every_25(self):
        grpo = self.dry_run("Baselines/1B/grpo.sh")
        sft = self.dry_run("Baselines/1B/sft.sh")
        kd_commands = [
            self.dry_run("OPD/1B/vanilla.sh"),
            self.dry_run("OPD/1B/trd.sh"),
            self.dry_run("OPSD/1B/skd.sh"),
        ]
        self.assertEqual(self.value(sft, "--save_steps"), "100")
        self.assertEqual(self.value(grpo, "--save_steps"), "50")
        self.assertEqual(
            {self.value(command, "--save_steps") for command in kd_commands},
            {"25"},
        )
        for command in [sft, grpo, *kd_commands]:
            self.assertEqual(self.value(command, "--save_total_limit"), "1")
            self.assertEqual(self.value(command, "--skip_final_model_save"), "True")
            self.assertNotIn("--save_only_model", command)

    def test_automatic_eval_sweeps_only_25_step_checkpoints_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            for step in (100, 7, 50, 25):
                (experiment / f"checkpoint-{step}").mkdir()
            env = os.environ.copy()
            env.pop("CODE_AUTO_EVAL", None)
            env.update({"CODE_EVAL_DRY_RUN": "1", "CODE_EVAL_EVERY_STEPS": "25"})
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; code_maybe_eval "$2" code_test',
                    "code-eval-test",
                    str(SCRIPT_ROOT / "lib/code_common.sh"),
                    str(experiment),
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
        commands = [
            line for line in completed.stdout.splitlines() if line.startswith("EVAL_CMD:")
        ]
        self.assertEqual(len(commands), 3)
        self.assertIn("checkpoint-25", commands[0])
        self.assertIn("checkpoint-50", commands[1])
        self.assertIn("checkpoint-100", commands[2])
        deletions = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("DELETE_CHECKPOINT_AFTER_SUCCESS:")
        ]
        self.assertEqual(len(deletions), 3)

    def test_code_eval_defaults_to_twelve_samples(self):
        wrapper = (SCRIPT_ROOT / "eval/run_code_eval.sh").read_text()
        decoder = (SCRIPT_ROOT / "eval/run_evalplus_vllm.py").read_text()
        self.assertIn('CODE_EVAL_PASS_K:-12', wrapper)
        self.assertIn('default=12', decoder)

    def test_post_eval_cleanup_removes_weights_but_keeps_metadata(self):
        output_root = REPO_ROOT / "outputs/code/test-cleanup"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            experiment = Path(temporary)
            checkpoint = experiment / "checkpoint-50"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
            (experiment / "adapter_model.safetensors").write_bytes(b"weights")
            metadata = experiment / "trainer_state.json"
            metadata.write_text("{}")
            subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; code_remove_evaluated_checkpoint "$2" "$3"; '
                    'code_remove_final_model_weights "$2"',
                    "code-cleanup-test",
                    str(SCRIPT_ROOT / "lib/code_common.sh"),
                    str(experiment),
                    str(checkpoint),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertFalse(checkpoint.exists())
            self.assertFalse((experiment / "adapter_model.safetensors").exists())
            self.assertTrue(metadata.exists())

    def test_segmented_training_evaluates_each_point_and_deletes_final_checkpoint(self):
        output_root = REPO_ROOT / "outputs/code/test-segments"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            experiment = Path(temporary)
            eval_log = experiment / "eval.log"
            eval_runner = experiment / "fake_eval.sh"
            eval_runner.write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$1" >> "$CODE_FAKE_EVAL_LOG"\n'
            )
            eval_runner.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "CODE_EVAL_RUNNER": str(eval_runner),
                    "CODE_FAKE_EVAL_LOG": str(eval_log),
                }
            )
            subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; '
                    'fake_train() { '
                    'local target="$1" latest="$2" root="$3"; '
                    '[[ -z "$latest" ]] || rm -rf -- "$latest"; '
                    'mkdir -p "$root/checkpoint-$target"; '
                    'touch "$root/checkpoint-$target/adapter_model.safetensors"; '
                    '}; '
                    'code_run_segmented_training 100 25 "$2" code_test fake_train "$2"',
                    "code-segment-test",
                    str(SCRIPT_ROOT / "lib/code_common.sh"),
                    str(experiment),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            evaluated = eval_log.read_text().splitlines()
            self.assertEqual(
                [Path(path).name for path in evaluated],
                ["checkpoint-25", "checkpoint-50", "checkpoint-75", "checkpoint-100"],
            )
            self.assertEqual(list(experiment.glob("checkpoint-*")), [])
            self.assertEqual(len(list(experiment.glob(".code_eval_complete_checkpoint-*"))), 4)

    def test_math_trd_rewrite_is_nonthinking_1024_but_opsd_scoring_thinks(self):
        env = os.environ.copy()
        env["DISTILL_DRY_RUN"] = "1"
        commands = {}
        for source in ("OPSD", "OPD"):
            completed = subprocess.run(
                ["bash", str(REPO_ROOT / f"scripts/{source}/1B/trd.sh")],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            line = next(
                line for line in completed.stdout.splitlines() if line.startswith("TRAIN_CMD:")
            )
            commands[source] = shlex.split(line.removeprefix("TRAIN_CMD:"))

        for command in commands.values():
            self.assertEqual(self.value(command, "--max_refinement_length"), "1024")
            self.assertEqual(self.value(command, "--refinement_thinking"), "False")
        self.assertEqual(self.value(commands["OPSD"], "--teacher_thinking"), "True")
        self.assertEqual(self.value(commands["OPD"], "--teacher_thinking"), "False")


class GitPreparationTests(unittest.TestCase):
    def test_large_data_is_ignored_but_small_math_eval_is_trackable(self):
        ignored = subprocess.run(
            [
                "git",
                "check-ignore",
                "data/train/example.parquet",
                "data/eval/livecodebench_code_generation_lite/test.jsonl",
                "third_party/trd/.git/HEAD",
                "script_code/runtime.env",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(ignored.returncode, 0, ignored.stderr)

        math_eval = subprocess.run(
            ["git", "check-ignore", "data/eval/aime24/data/train-00000-of-00001.parquet"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(math_eval.returncode, 0)

    def test_one_command_preparer_exposes_scoped_modes(self):
        completed = subprocess.run(
            ["bash", str(SCRIPT_ROOT / "prepare_code.sh"), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("all|train|eval|verify", completed.stderr)

        source = (SCRIPT_ROOT / "prepare_code.sh").read_text()
        self.assertIn('"$REPO_ROOT/scripts/prepare_data.py" --scope train', source)
        self.assertIn('"$SCRIPT_DIR/prepare_data.py"', source)
        self.assertIn('"$SCRIPT_DIR/prepare_eval_data.py"', source)


if __name__ == "__main__":
    unittest.main()
