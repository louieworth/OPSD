from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data_collator import WindowDataCollator
from opsd_train import configure_policy_update_schedule, validate_algorithm_config
from opsd_trainer import OPSDTrainer
from trl.trainer.sft_trainer import SFTTrainer


def _script_args(*, updates=None, teacher_refine=False):
    return SimpleNamespace(
        policy_gradient_updates=updates,
        teacher_refine=teacher_refine,
    )


def _training_args(
    *,
    max_steps=100,
    gradient_accumulation_steps=1,
    ignore_data_skip=False,
    use_vllm=True,
    vllm_sync_frequency=1,
):
    return SimpleNamespace(
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=ignore_data_skip,
        use_vllm=use_vllm,
        vllm_sync_frequency=vllm_sync_frequency,
    )


class PolicyUpdateScheduleTests(unittest.TestCase):
    def test_default_preserves_legacy_max_steps(self):
        script_args = _script_args()
        training_args = _training_args(max_steps=100, gradient_accumulation_steps=7)

        configure_policy_update_schedule(script_args, training_args)

        self.assertEqual(training_args.max_steps, 100)
        self.assertFalse(training_args.windowed_policy_updates)
        self.assertIsNone(training_args.total_rollout_steps)
        self.assertIsNone(training_args.policy_gradient_updates)
        self.assertEqual(training_args.rollouts_per_update, 1)
        self.assertIsNone(script_args.policy_gradient_updates)

    def test_translates_rollout_steps_to_outer_optimizer_steps(self):
        for updates, expected_window_size in ((1, 100), (50, 2), (100, 1)):
            with self.subTest(updates=updates):
                script_args = _script_args(updates=updates)
                training_args = _training_args(max_steps=100)

                configure_policy_update_schedule(script_args, training_args)

                self.assertTrue(training_args.windowed_policy_updates)
                self.assertEqual(training_args.total_rollout_steps, 100)
                self.assertEqual(training_args.policy_gradient_updates, updates)
                self.assertEqual(training_args.rollouts_per_update, expected_window_size)
                self.assertEqual(training_args.max_steps, updates)
                self.assertEqual(script_args.policy_gradient_updates, updates)

    def test_teacher_refine_defaults_to_one_update_per_rollout(self):
        script_args = _script_args(teacher_refine=True)
        training_args = _training_args(max_steps=100)

        configure_policy_update_schedule(script_args, training_args)

        self.assertEqual(training_args.total_rollout_steps, 100)
        self.assertEqual(training_args.policy_gradient_updates, 100)
        self.assertEqual(training_args.rollouts_per_update, 1)
        self.assertEqual(training_args.max_steps, 100)

    def test_rejects_invalid_rollout_or_update_counts(self):
        cases = (
            (_script_args(updates=1), _training_args(max_steps=0), "max_steps > 0"),
            (_script_args(updates=0), _training_args(), "1 <= U <= max_steps"),
            (_script_args(updates=101), _training_args(), "1 <= U <= max_steps"),
            (_script_args(updates=30), _training_args(), "must be divisible"),
        )
        for script_args, training_args, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                configure_policy_update_schedule(script_args, training_args)

    def test_rejects_incompatible_trainer_accumulation_and_resume_settings(self):
        cases = (
            (_training_args(gradient_accumulation_steps=2), "gradient_accumulation_steps 1"),
            (_training_args(ignore_data_skip=True), "ignore_data_skip=False"),
            (_training_args(vllm_sync_frequency=2), "vllm_sync_frequency 1"),
        )
        for training_args, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                configure_policy_update_schedule(_script_args(updates=50), training_args)

    def test_non_vllm_schedule_does_not_require_sync_frequency_one(self):
        training_args = _training_args(use_vllm=False, vllm_sync_frequency=17)

        configure_policy_update_schedule(_script_args(updates=50), training_args)

        self.assertEqual(training_args.rollouts_per_update, 2)


class AlgorithmConfigTests(unittest.TestCase):
    @staticmethod
    def trd_args(**overrides):
        values = {
            "alg": "opsd",
            "teacher_refine": True,
            "reason_first": False,
            "use_tinker_loss": False,
            "top_k_loss": 0,
            "jsd_token_clip": 0,
            "distillation_temperature": None,
            "use_ema_teacher": False,
            "max_refinement_length": None,
            "refinement_vllm_max_model_len": None,
            "fixed_teacher": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def training_args():
        return SimpleNamespace(
            beta=0,
            max_completion_length=1_024,
            temperature=1.1,
            teacher_model_name_or_path=None,
        )

    def test_trd_defaults_distillation_temperature_to_paper_value(self):
        script_args = self.trd_args()

        validate_algorithm_config(script_args, self.training_args(), SimpleNamespace())

        self.assertEqual(script_args.distillation_temperature, 1.0)
        self.assertEqual(script_args.max_refinement_length, 1_024)
        self.assertEqual(script_args.refinement_vllm_max_model_len, 21_024)

    def test_trd_rejects_non_unit_distillation_temperature(self):
        script_args = self.trd_args(distillation_temperature=0.9)

        with self.assertRaisesRegex(ValueError, "requires --distillation_temperature 1.0"):
            validate_algorithm_config(script_args, self.training_args(), SimpleNamespace())

    def test_skd_accepts_math_task_with_local_rollout_backend(self):
        script_args = self.trd_args(
            task_type="math",
            trajectory_mode="skd",
            teacher_refine=False,
            policy_gradient_updates=100,
        )
        training_args = self.training_args()
        training_args.use_vllm = False

        validate_algorithm_config(script_args, training_args, SimpleNamespace())

        self.assertEqual(script_args.task_type, "math")
        self.assertEqual(script_args.trajectory_mode, "skd")


class TRDLengthBudgetTests(unittest.TestCase):
    class StopAfterBaseTrainerInit(Exception):
        pass

    class TokenizedPrompts:
        def __init__(self, input_ids):
            self.input_ids = input_ids

        def to(self, _device):
            return self

    class CapturingTokenizer:
        pad_token = None
        pad_token_id = 0

        def __init__(self):
            self.max_length = None

        def batch_decode(self, token_ids, *, skip_special_tokens):
            del skip_special_tokens
            return ["p" * 21_024 for _ in token_ids]

        def __call__(self, prompts, **kwargs):
            self.max_length = kwargs["max_length"]
            return TRDLengthBudgetTests.TokenizedPrompts(
                torch.ones((len(prompts), self.max_length), dtype=torch.long)
            )

        def decode(self, token_ids, *, skip_special_tokens):
            del skip_special_tokens
            return "/".join(map(str, token_ids))

    def test_teacher_refine_reserves_larger_student_response_budget(self):
        cases = (
            (1_024, 1_024),
            (512, 1_024),
            (1_024, 512),
        )
        for max_completion_length, max_refinement_length in cases:
            with self.subTest(
                max_completion_length=max_completion_length,
                max_refinement_length=max_refinement_length,
            ):
                trainer = OPSDTrainer.__new__(OPSDTrainer)
                args = SimpleNamespace(
                    max_length=21_024,
                    max_completion_length=max_completion_length,
                    student_model_revision=None,
                    dataset_kwargs=None,
                )
                model = SimpleNamespace(config=SimpleNamespace(_name_or_path="student"))
                collator = object()

                with patch(
                    "opsd_trainer.SelfDistillationDataCollator", return_value=collator
                ) as collator_factory, patch.object(
                    SFTTrainer,
                    "__init__",
                    side_effect=self.StopAfterBaseTrainerInit,
                ), self.assertRaises(self.StopAfterBaseTrainerInit):
                    OPSDTrainer.__init__(
                        trainer,
                        model=model,
                        args=args,
                        processing_class=object(),
                        teacher_refine=True,
                        max_refinement_length=max_refinement_length,
                        refinement_vllm_max_model_len=21_024,
                    )

                expected_prompt_cap = 21_024 - max(
                    max_completion_length, max_refinement_length
                )
                self.assertEqual(trainer.student_prompt_max_length, expected_prompt_cap)
                self.assertEqual(
                    trainer.refinement_prompt_max_length,
                    21_024 - max_refinement_length,
                )
                collator_factory.assert_called_once()
                self.assertEqual(
                    collator_factory.call_args.kwargs["max_length"], expected_prompt_cap
                )

    @patch("opsd_trainer.broadcast_object_list")
    @patch("opsd_trainer.gather_object", side_effect=lambda values: values)
    def test_vllm_retokenization_keeps_larger_refinement_reserve(
        self, _gather, _broadcast
    ):
        tokenizer = self.CapturingTokenizer()
        trainer = OPSDTrainer.__new__(OPSDTrainer)
        trainer.teacher_refine = True
        trainer.student_prompt_max_length = 20_000
        trainer.processing_class = tokenizer
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"), is_main_process=True, process_index=0
        )
        trainer.vllm_mode = "server"
        trainer.vllm_guided_decoding_regex = None
        trainer.vllm_client = SimpleNamespace(
            generate=lambda **kwargs: {
                "completion_ids": [[7] for _ in kwargs["prompts"]]
            }
        )
        trainer.args = SimpleNamespace(max_length=21_024)
        generation_config = SimpleNamespace(
            max_new_tokens=512,
            temperature=1.0,
            top_k=0,
        )

        with patch("builtins.print"):
            generated_ids, *_ = OPSDTrainer._generate_on_policy_outputs_vllm.__wrapped__(
                trainer,
                {"student_prompts": torch.tensor([[1]])},
                generation_config,
                pad_token_id=0,
            )

        self.assertEqual(tokenizer.max_length, 20_000)
        self.assertEqual(generated_ids.shape[1], 20_000 + 512)


class WindowDataCollatorTests(unittest.TestCase):
    def test_splits_exact_window_into_contiguous_microbatches(self):
        calls = []

        def base_collator(features):
            values = [feature["value"] for feature in features]
            calls.append(values)
            return {"values": values, "batch_size": len(values)}

        collator = WindowDataCollator(base_collator, micro_batch_size=2, window_size=3)
        result = collator([{"value": value} for value in range(6)])

        self.assertEqual(calls, [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(
            result,
            {
                "rollout_batches": [
                    {"values": [0, 1], "batch_size": 2},
                    {"values": [2, 3], "batch_size": 2},
                    {"values": [4, 5], "batch_size": 2},
                ]
            },
        )

    def test_rejects_partial_window_without_calling_base_collator(self):
        calls = []
        collator = WindowDataCollator(calls.append, micro_batch_size=2, window_size=3)

        with self.assertRaisesRegex(ValueError, "requires exactly 6 examples"):
            collator([{"value": value} for value in range(5)])

        self.assertEqual(calls, [])

    def test_rejects_non_positive_dimensions(self):
        for micro_batch_size, window_size in ((0, 1), (1, 0), (-1, 2), (2, -1)):
            with self.subTest(
                micro_batch_size=micro_batch_size,
                window_size=window_size,
            ), self.assertRaisesRegex(ValueError, "must be positive"):
                WindowDataCollator(lambda features: features, micro_batch_size, window_size)

    def test_window_dataloader_rejects_launch_level_accumulation(self):
        trainer = OPSDTrainer.__new__(OPSDTrainer)
        trainer.windowed_policy_updates = True
        trainer.train_dataset = [object()]
        trainer.accelerator = SimpleNamespace(
            split_batches=False,
            dispatch_batches=False,
            gradient_accumulation_steps=2,
        )

        with self.assertRaisesRegex(ValueError, "launch-level gradient accumulation 1"):
            trainer.get_train_dataloader()


class DistillationBatchTests(unittest.TestCase):
    def setUp(self):
        self.trainer = OPSDTrainer.__new__(OPSDTrainer)
        self.trainer.processing_class = SimpleNamespace(pad_token_id=0)
        self.trainer.student_context_max_length = 21_024
        self.trainer.refinement_vllm_max_model_len = 21_024
        self.rollout = {
            "inputs": {
                "student_prompts": torch.tensor([[1, 2, 0], [3, 4, 5]]),
                "student_prompt_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                "teacher_prompts": torch.tensor([[6, 7, 8, 0], [9, 10, 11, 12]]),
                "teacher_prompt_attention_mask": torch.tensor(
                    [[1, 1, 1, 0], [1, 1, 1, 1]]
                ),
            }
        }

    def test_builds_different_width_prefixes_with_one_aligned_target(self):
        batch = self.trainer._build_distillation_batch(
            self.rollout,
            target_ids=[[20, 21], [30]],
        )

        self.assertEqual(batch["student_prompt_length"], 3)
        self.assertEqual(batch["teacher_prompt_length"], 4)
        torch.testing.assert_close(
            batch["student_input_ids"],
            torch.tensor([[0, 1, 2, 20, 21], [3, 4, 5, 30, 0]]),
        )
        torch.testing.assert_close(
            batch["student_attention_mask"],
            torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        )
        torch.testing.assert_close(
            batch["teacher_input_ids"],
            torch.tensor([[0, 6, 7, 8, 20, 21], [9, 10, 11, 12, 30, 0]]),
        )
        torch.testing.assert_close(
            batch["teacher_attention_mask"],
            torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
        )
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([[-100, -100, -100, 20, 21], [-100, -100, -100, 30, -100]]),
        )

    def test_teacher_refine_uses_canonical_teacher_prompt_ids(self):
        batch = self.trainer._build_distillation_batch(
            self.rollout,
            target_ids=[[20, 21], [30]],
            teacher_prompt_ids=[[50, 51], [60]],
        )

        self.assertEqual(batch["teacher_prompt_length"], 2)
        torch.testing.assert_close(
            batch["teacher_input_ids"],
            torch.tensor([[50, 51, 20, 21], [0, 60, 30, 0]]),
        )
        torch.testing.assert_close(
            batch["teacher_attention_mask"],
            torch.tensor([[1, 1, 1, 1], [0, 1, 1, 0]]),
        )
        # The student target labels remain aligned to the student prefix,
        # independent of the teacher prefix width.
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([[-100, -100, -100, 20, 21], [-100, -100, -100, 30, -100]]),
        )

    def test_trd_kl_inputs_fit_exact_student_and_teacher_context_caps(self):
        prompt_length = 20_000
        refinement_length = 1_024
        rollout = {
            "inputs": {
                "student_prompts": torch.full((1, prompt_length), 1),
                "student_prompt_attention_mask": torch.ones(
                    (1, prompt_length), dtype=torch.long
                ),
                "teacher_prompts": torch.tensor([[9]]),
                "teacher_prompt_attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        }

        batch = self.trainer._build_distillation_batch(
            rollout,
            target_ids=[[3] * refinement_length],
            teacher_prompt_ids=[[2] * prompt_length],
        )

        self.assertEqual(batch["student_input_ids"].shape[1], 21_024)
        self.assertEqual(batch["teacher_input_ids"].shape[1], 21_024)
        self.assertLessEqual(
            batch["student_input_ids"].shape[1],
            self.trainer.student_context_max_length,
        )
        self.assertLessEqual(
            batch["teacher_input_ids"].shape[1],
            self.trainer.refinement_vllm_max_model_len,
        )

    def test_trd_kl_inputs_reject_one_token_context_overflow(self):
        refinement_ids = [[3] * 1_024]
        overflowing_prompt = [2] * 20_001

        student_overflow_rollout = {
            "inputs": {
                "student_prompts": torch.full((1, 20_001), 1),
                "student_prompt_attention_mask": torch.ones(
                    (1, 20_001), dtype=torch.long
                ),
                "teacher_prompts": torch.tensor([[9]]),
                "teacher_prompt_attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        }
        with self.assertRaisesRegex(RuntimeError, "student KL sequence"):
            self.trainer._build_distillation_batch(
                student_overflow_rollout,
                target_ids=refinement_ids,
                teacher_prompt_ids=[[2]],
            )

        teacher_overflow_rollout = {
            "inputs": {
                "student_prompts": torch.tensor([[1]]),
                "student_prompt_attention_mask": torch.ones((1, 1), dtype=torch.long),
                "teacher_prompts": torch.tensor([[9]]),
                "teacher_prompt_attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        }
        with self.assertRaisesRegex(RuntimeError, "teacher KL sequence"):
            self.trainer._build_distillation_batch(
                teacher_overflow_rollout,
                target_ids=refinement_ids,
                teacher_prompt_ids=[overflowing_prompt],
            )

    def test_rejects_empty_target_batches_or_rows(self):
        for target_ids in ([], [[20], []]):
            with self.subTest(target_ids=target_ids), self.assertRaisesRegex(
                RuntimeError, "at least one token"
            ):
                self.trainer._build_distillation_batch(self.rollout, target_ids)

    def test_variable_length_targets_require_a_pad_token(self):
        self.trainer.processing_class.pad_token_id = None

        with self.assertRaisesRegex(ValueError, "pad token"):
            self.trainer._build_distillation_batch(self.rollout, [[20], [30]])

    def test_vanilla_builder_preserves_exact_legacy_student_sequence(self):
        generated_ids = torch.tensor([[1, 2, 0, 20, 21], [3, 4, 5, 30, 0]])
        generated_attention_mask = torch.tensor(
            [[1, 1, 0, 1, 1], [1, 1, 1, 1, 0]]
        )
        self.rollout.update(
            {
                "generated_ids": generated_ids,
                "generated_attention_mask": generated_attention_mask,
                "generated_prompt_width": 3,
            }
        )
        self.rollout["inputs"]["student_prompt_lengths_per_example"] = torch.tensor([2, 3])

        batch = self.trainer._build_vanilla_distillation_batch(self.rollout)

        torch.testing.assert_close(batch["student_input_ids"], generated_ids)
        torch.testing.assert_close(
            batch["student_attention_mask"], generated_attention_mask
        )
        self.assertEqual(batch["student_prompt_length"], 3)
        self.assertEqual(batch["teacher_prompt_length"], 4)
        torch.testing.assert_close(
            batch["teacher_input_ids"],
            torch.tensor([[6, 7, 8, 0, 20, 21], [9, 10, 11, 12, 30, 0]]),
        )
        torch.testing.assert_close(
            batch["teacher_attention_mask"],
            torch.tensor([[1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0]]),
        )
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([[-100, -100, -100, 20, 21], [-100, -100, -100, 30, -100]]),
        )


class LocalRefinementResultValidationTests(unittest.TestCase):
    class Tokenizer:
        prompt_ids = {"prompt-a": [1, 2], "prompt-b": [3]}

        def __len__(self):
            return 100

        def encode(self, prompt, *, add_special_tokens):
            if add_special_tokens:
                raise AssertionError("Refinement prompts must not add special tokens twice")
            return self.prompt_ids[prompt]

        def batch_decode(self, completion_ids, *, skip_special_tokens):
            if not skip_special_tokens:
                raise AssertionError("Refined text logging should skip special tokens")
            return ["/".join(map(str, row)) for row in completion_ids]

    def setUp(self):
        self.trainer = OPSDTrainer.__new__(OPSDTrainer)
        self.trainer.processing_class = self.Tokenizer()
        self.records = [
            {"request_id": (0, 0, 0, 0), "prompt": "prompt-a"},
            {"request_id": (0, 0, 0, 1), "prompt": "prompt-b"},
        ]
        self.results = {
            (0, 0, 0, 0): {"prompt_ids": [1, 2], "completion_ids": [10, 11]},
            (0, 0, 0, 1): {"prompt_ids": [3], "completion_ids": [12]},
        }

    def test_returns_canonical_ids_and_decoded_text(self):
        prompt_ids, completion_ids, texts = self.trainer._validate_local_refinement_results(
            self.records, self.results
        )

        self.assertEqual(prompt_ids, [[1, 2], [3]])
        self.assertEqual(completion_ids, [[10, 11], [12]])
        self.assertEqual(texts, ["10/11", "12"])

    def test_returns_independent_teacher_scoring_prompt_ids(self):
        self.records[0]["scoring_prompt_ids"] = [91, 92]
        self.records[1]["scoring_prompt_ids"] = [93]

        prompt_ids, _, _ = self.trainer._validate_local_refinement_results(
            self.records, self.results
        )

        self.assertEqual(prompt_ids, [[91, 92], [93]])

    def test_rejects_missing_request_before_next_collective(self):
        del self.results[(0, 0, 0, 1)]

        with self.assertRaisesRegex(RuntimeError, "result IDs do not match requests"):
            self.trainer._validate_local_refinement_results(self.records, self.results)

    def test_rejects_teacher_and_local_tokenizer_mismatch(self):
        self.results[(0, 0, 0, 1)]["prompt_ids"] = [999]

        with self.assertRaisesRegex(RuntimeError, "prompt IDs differ"):
            self.trainer._validate_local_refinement_results(self.records, self.results)

    def test_rejects_completion_tokens_outside_shared_vocabulary(self):
        self.results[(0, 0, 0, 1)]["completion_ids"] = [100]

        with self.assertRaisesRegex(RuntimeError, "outside the student/teacher shared vocabulary"):
            self.trainer._validate_local_refinement_results(self.records, self.results)

    def test_rejects_unexpected_local_result(self):
        self.results[(0, 0, 0, 2)] = {"prompt_ids": [4], "completion_ids": [13]}

        with self.assertRaisesRegex(RuntimeError, r"unexpected=\[\(0, 0, 0, 2\)\]"):
            self.trainer._validate_local_refinement_results(self.records, self.results)


class TRDGlobalTokenMeanTests(unittest.TestCase):
    class FakeAccelerator:
        def __init__(self, global_token_count, num_processes):
            self.global_token_count = global_token_count
            self.num_processes = num_processes
            self.local_counts = []

        def reduce(self, value, *, reduction):
            if reduction != "sum":
                raise AssertionError(reduction)
            self.local_counts.append(int(value.item()))
            return torch.tensor(self.global_token_count, device=value.device)

    @staticmethod
    def trainer(global_token_count, num_processes):
        trainer = OPSDTrainer.__new__(OPSDTrainer)
        trainer.accelerator = TRDGlobalTokenMeanTests.FakeAccelerator(
            global_token_count, num_processes
        )
        return trainer

    def test_dp_average_of_scaled_local_sums_equals_global_token_mean(self):
        labels_by_rank = (
            torch.tensor([[1, 2, -100]]),
            torch.tensor([[3, 4, 5, 6]]),
        )
        local_sums = (torch.tensor(6.0), torch.tensor(10.0))
        scaled_losses = []
        for labels, local_sum in zip(labels_by_rank, local_sums, strict=True):
            trainer = self.trainer(global_token_count=6, num_processes=2)
            scaled_losses.append(trainer._trd_global_token_mean(local_sum, labels))
            self.assertEqual(trainer.accelerator.local_counts, [(labels != -100).sum().item()])

        ddp_averaged_loss = torch.stack(scaled_losses).mean()
        torch.testing.assert_close(ddp_averaged_loss, torch.tensor(16.0 / 6.0))

    def test_single_rank_reduces_to_local_token_mean(self):
        trainer = self.trainer(global_token_count=3, num_processes=1)

        loss = trainer._trd_global_token_mean(
            torch.tensor(9.0), torch.tensor([[1, 2, -100, 3]])
        )

        torch.testing.assert_close(loss, torch.tensor(3.0))


class DistributedStageStatusTests(unittest.TestCase):
    def setUp(self):
        self.trainer = OPSDTrainer.__new__(OPSDTrainer)
        self.trainer.accelerator = SimpleNamespace(process_index=0)

    @patch("opsd_trainer.gather_object")
    def test_all_successful_ranks_continue(self, gather):
        gather.return_value = [
            {"ok": True, "rank": 0, "error": None},
            {"ok": True, "rank": 1, "error": None},
        ]

        self.trainer._raise_if_any_rank_failed("Student rollout 0", None)

        gather.assert_called_once_with([{"ok": True, "rank": 0, "error": None}])

    @patch("opsd_trainer.gather_object")
    def test_one_rank_failure_is_raised_on_every_rank(self, gather):
        gather.return_value = [
            {"ok": True, "rank": 0, "error": None},
            {"ok": False, "rank": 1, "error": "RuntimeError('empty y_o')"},
        ]

        with self.assertRaisesRegex(RuntimeError, "Student rollout 0 failed.*rank.*1"):
            self.trainer._raise_if_any_rank_failed("Student rollout 0", None)


class WindowedTrainingStepOrderTests(unittest.TestCase):
    class GradientState:
        def __init__(self, events):
            self.events = events

        def _set_sync_gradients(self, enabled):
            self.events.append(("sync", enabled))

    class NoSync:
        def __init__(self, events):
            self.events = events

        def __enter__(self):
            self.events.append(("no_sync_enter",))

        def __exit__(self, exc_type, exc_value, traceback):
            self.events.append(("no_sync_exit",))

    def test_collects_whole_window_before_refine_and_backward(self):
        events = []
        trainer = OPSDTrainer.__new__(OPSDTrainer)
        trainer.rollouts_per_update = 3
        trainer.teacher_refine = True
        trainer.state = SimpleNamespace(global_step=4)
        trainer.args = SimpleNamespace(device=torch.device("cpu"))
        trainer.current_gradient_accumulation_steps = 1
        trainer.is_deepspeed_enabled = False
        trainer._on_policy_loss_total = 0.0
        trainer._on_policy_step_equiv = 0.0
        gradient_state = self.GradientState(events)
        trainer.accelerator = SimpleNamespace(
            process_index=0,
            gradient_state=gradient_state,
            no_sync=lambda model: self.NoSync(events),
        )
        trainer._ensure_student_vllm_current = lambda: events.append(("sync_vllm",))
        trainer._prepare_inputs = lambda raw: {"raw": raw}

        def collect(model, prepared, rollout_step):
            events.append(("collect", prepared["raw"], rollout_step))
            return {"id": prepared["raw"]}

        def refine(rollout, offset):
            events.append(("refine", rollout["id"], offset))
            return [[1]], [[20]], ["refined"]

        def build(rollout, *, target_ids, teacher_prompt_ids):
            events.append(("build", rollout["id"]))
            return {"id": rollout["id"]}

        trainer._collect_original_rollout = collect
        trainer._generate_refined_completions = refine
        trainer._build_distillation_batch = build
        trainer._record_rollout_outputs = lambda rollout, refined: events.append(
            ("record", rollout["id"])
        )

        def raise_if_failed(stage, error):
            if error is not None:
                raise error

        trainer._raise_if_any_rank_failed = raise_if_failed

        def parent_training_step(self, model, batch, num_items_in_batch=None):
            events.append(("backward", batch["id"], self.current_gradient_accumulation_steps))
            return torch.tensor(1.0 / self.current_gradient_accumulation_steps)

        with patch.object(SFTTrainer, "training_step", new=parent_training_step), patch(
            "opsd_trainer.empty_cache", return_value=None
        ):
            loss = trainer._windowed_training_step(object(), {"rollout_batches": ["a", "b", "c"]})

        collect_positions = [index for index, event in enumerate(events) if event[0] == "collect"]
        refine_positions = [index for index, event in enumerate(events) if event[0] == "refine"]
        build_positions = [index for index, event in enumerate(events) if event[0] == "build"]
        backward_positions = [index for index, event in enumerate(events) if event[0] == "backward"]
        self.assertEqual([events[index][1] for index in collect_positions], ["a", "b", "c"])
        self.assertLess(max(collect_positions), min(refine_positions))
        self.assertLess(max(build_positions), min(backward_positions))
        self.assertEqual(
            [events[index][1:] for index in backward_positions],
            [("a", 3), ("b", 3), ("c", 3)],
        )
        self.assertEqual(
            [event for event in events if event[0] == "sync"],
            [("sync", False), ("sync", False), ("sync", True), ("sync", True)],
        )
        self.assertEqual(sum(event[0] == "no_sync_enter" for event in events), 2)
        self.assertEqual(trainer.current_gradient_accumulation_steps, 1)
        torch.testing.assert_close(loss, torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
