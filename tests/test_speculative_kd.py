from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import torch

from script_code.speculative_kd import SKDConfig, SpeculativeKDGenerator


class FakeCache:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def crop(self, length):
        self.tokens = self.tokens[:length]


class DeterministicModel:
    def __init__(self, prompt_length, preferred, vocab_size=16):
        self.prompt_length = prompt_length
        self.preferred = preferred
        self.vocab_size = vocab_size

    def __call__(self, *, input_ids, past_key_values=None, use_cache, return_dict):
        self.assertions = (use_cache, return_dict)
        previous = [] if past_key_values is None else past_key_values.tokens
        incoming = input_ids[0].tolist()
        complete = previous + incoming
        logits = []
        for offset in range(len(incoming)):
            prefix_length = len(previous) + offset + 1
            generated = max(0, prefix_length - self.prompt_length)
            preferred = self.preferred[min(generated, len(self.preferred) - 1)]
            scores = torch.full((self.vocab_size,), -100.0)
            scores[preferred] = 100.0
            logits.append(scores)
        return SimpleNamespace(
            logits=torch.stack(logits).unsqueeze(0),
            past_key_values=FakeCache(complete),
        )


class SpeculativeKDTests(unittest.TestCase):
    def generator(self, *, max_tokens=3, draft_length=2):
        return SpeculativeKDGenerator(
            SKDConfig(
                max_new_tokens=max_tokens,
                draft_length=draft_length,
                accept_top_k=1,
                student_temperature=1.0,
                student_top_p=1.0,
                student_top_k=1,
                correction_temperature=0.2,
                correction_top_p=1.0,
            )
        )

    def test_accepts_complete_student_trajectory(self):
        student = DeterministicModel(1, [1, 2, 3, 4])
        teacher = DeterministicModel(2, [1, 2, 3, 4])
        completion, stats = self.generator().generate_one(
            student_model=student,
            teacher_model=teacher,
            student_prompt_ids=torch.tensor([[10]]),
            teacher_prompt_ids=torch.tensor([[11, 12]]),
            eos_token_ids=15,
        )

        self.assertEqual(completion, [1, 2, 3])
        self.assertEqual(stats.accepted_tokens, 3)
        self.assertEqual(stats.corrections, 0)
        self.assertEqual(stats.acceptance_rate, 1.0)

    def test_rejects_suffix_and_inserts_teacher_correction(self):
        student = DeterministicModel(1, [1, 2, 3, 4])
        teacher = DeterministicModel(2, [1, 9, 3, 4])
        completion, stats = self.generator().generate_one(
            student_model=student,
            teacher_model=teacher,
            student_prompt_ids=torch.tensor([[10]]),
            teacher_prompt_ids=torch.tensor([[11, 12]]),
            eos_token_ids=15,
            teacher_context=nullcontext,
        )

        self.assertEqual(completion, [1, 9, 3])
        self.assertEqual(stats.accepted_tokens, 2)
        self.assertEqual(stats.corrections, 1)
        self.assertEqual(stats.proposed_tokens, 3)

    def test_teacher_eos_correction_ends_generation(self):
        student = DeterministicModel(1, [1, 2])
        teacher = DeterministicModel(1, [7, 2])
        completion, stats = self.generator(max_tokens=8).generate_one(
            student_model=student,
            teacher_model=teacher,
            student_prompt_ids=torch.tensor([[10]]),
            teacher_prompt_ids=torch.tensor([[11]]),
            eos_token_ids=7,
        )

        self.assertEqual(completion, [7])
        self.assertEqual(stats.corrections, 1)

    def test_config_rejects_invalid_unique_parameters(self):
        for kwargs in ({"draft_length": 0}, {"accept_top_k": 0}, {"correction_top_p": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SKDConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()

