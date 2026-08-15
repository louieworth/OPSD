import unittest

from trd_refinement import (
    RefinementPromptError,
    RefinementServerError,
    TeacherVLLMClient,
    build_refinement_prompt,
)


class CharacterTokenizer:
    """Small reversible tokenizer that makes prompt-budget tests exact."""

    def __init__(self):
        self.last_enable_thinking = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.last_enable_thinking = enable_thinking
        self.assertions = (tokenize, add_generation_prompt)
        return f"<user>{messages[0]['content']}<assistant thinking={int(enable_thinking)}>"

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Rendered prompts must not receive extra special tokens")
        return [ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeResponse:
    def __init__(self, body, status_code=200, text=""):
        self.body = body
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, generation_body):
        self.generation_body = generation_body
        self.get_calls = []
        self.post_calls = []
        self.closed = False

    def get(self, url, *, timeout):
        self.get_calls.append((url, timeout))
        if url.endswith("/health/"):
            return FakeResponse({"status": "ok"})
        if url.endswith("/get_world_size/"):
            return FakeResponse({"world_size": 4})
        raise AssertionError(url)

    def post(self, url, *, json, timeout):
        self.post_calls.append((url, json, timeout))
        return FakeResponse(self.generation_body)

    def close(self):
        self.closed = True


class RefinementPromptTests(unittest.TestCase):
    def test_opd_uses_initial_response_without_reference(self):
        tokenizer = CharacterTokenizer()
        rendered = build_refinement_prompt(
            tokenizer,
            alg="opd",
            problem="Find x.",
            initial_response="x=3",
            reference_solution="THIS MUST NOT APPEAR",
            teacher_thinking=False,
            max_model_len=2_000,
            max_refinement_length=128,
        )

        self.assertIn("**Problem:**\nFind x.", rendered)
        self.assertIn("**Your Initial Solution:**\nx=3", rendered)
        self.assertNotIn("Reference Solution", rendered)
        self.assertNotIn("THIS MUST NOT APPEAR", rendered)
        self.assertIn("Please reason step by step", rendered)
        self.assertFalse(tokenizer.last_enable_thinking)
        self.assertEqual(tokenizer.assertions, (False, True))

    def test_opsd_contains_reference_and_initial_response(self):
        rendered = build_refinement_prompt(
            CharacterTokenizer(),
            alg="opsd",
            problem="Compute 1+1.",
            reference_solution="It equals two.",
            initial_response="Maybe three.",
            max_model_len=2_000,
            max_refinement_length=128,
        )

        self.assertIn("**Reference Solution:**\nIt equals two.", rendered)
        self.assertIn("**Your Initial Solution:**\nMaybe three.", rendered)
        self.assertIn("Maybe three.\n\n**Instructions:**", rendered)

    def test_truncates_initial_then_reference_and_reserves_output(self):
        tokenizer = CharacterTokenizer()
        rendered = build_refinement_prompt(
            tokenizer,
            alg="opsd",
            problem="p",
            reference_solution="R" * 500,
            initial_response="I" * 500,
            max_model_len=720,
            max_refinement_length=64,
        )

        self.assertIn("[... initial solution truncated ...]", rendered)
        self.assertIn("[... reference solution truncated ...]", rendered)
        self.assertLessEqual(len(tokenizer.encode(rendered)) + 64, 720)
        self.assertNotIn("I" * 500, rendered)
        self.assertNotIn("R" * 500, rendered)

    def test_opsd_allows_exact_twenty_thousand_token_prompt_budget(self):
        tokenizer = CharacterTokenizer()
        empty_response_prompt = build_refinement_prompt(
            tokenizer,
            alg="opsd",
            problem="p",
            reference_solution="",
            initial_response="",
            max_model_len=21_024,
            max_refinement_length=1_024,
        )
        exact_initial_response = "I" * (
            20_000 - len(tokenizer.encode(empty_response_prompt))
        )

        rendered = build_refinement_prompt(
            tokenizer,
            alg="opsd",
            problem="p",
            reference_solution="",
            initial_response=exact_initial_response,
            max_model_len=21_024,
            max_refinement_length=1_024,
        )

        self.assertEqual(len(tokenizer.encode(rendered)), 20_000)
        self.assertIn(exact_initial_response, rendered)
        self.assertNotIn("[... initial solution truncated ...]", rendered)

    def test_raises_instead_of_truncating_problem_or_instructions(self):
        with self.assertRaises(RefinementPromptError):
            build_refinement_prompt(
                CharacterTokenizer(),
                alg="opd",
                problem="P" * 1_000,
                initial_response="i",
                max_model_len=500,
                max_refinement_length=64,
            )


class TeacherVLLMClientTests(unittest.TestCase):
    @staticmethod
    def valid_body():
        return {
            "prompt_ids": [[1, 2], [3]],
            "completion_ids": [[4, 5], [6]],
            "logprobs": [[-0.1, -0.2], [-0.3]],
        }

    def test_health_generation_payload_and_canonical_result(self):
        session = FakeSession(self.valid_body())
        client = TeacherVLLMClient(
            host="teacher",
            server_port=8002,
            connect_timeout=3,
            read_timeout=30,
            expected_world_size=4,
            max_model_len=20,
            session=session,
        )
        result = client.generate(
            ["prompt-a", "prompt-b"],
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.1,
            presence_penalty=0.25,
            max_tokens=4,
        )

        self.assertEqual(result.prompt_ids, [[1, 2], [3]])
        self.assertEqual(result.completion_ids, [[4, 5], [6]])
        self.assertEqual(client.world_size, 4)
        self.assertEqual(len(session.get_calls), 2)
        _, payload, timeout = session.post_calls[0]
        self.assertEqual(timeout, (3.0, 30.0))
        self.assertIsNone(payload["truncate_prompt_tokens"])
        self.assertEqual(payload["generation_kwargs"], {"presence_penalty": 0.25})
        top_level_payload = {
            key: value for key, value in payload.items() if key != "generation_kwargs"
        }
        self.assertNotIn("presence_penalty", top_level_payload)

    def test_rejects_empty_completion(self):
        body = self.valid_body()
        body["completion_ids"][0] = []
        body["logprobs"][0] = []
        client = TeacherVLLMClient(expected_world_size=4, session=FakeSession(body))

        with self.assertRaisesRegex(RefinementServerError, "completion_ids\\[0\\] is empty"):
            client.generate(["a", "b"])

    def test_rejects_world_size_mismatch(self):
        with self.assertRaisesRegex(RefinementServerError, "world size is 4, expected 8"):
            TeacherVLLMClient(expected_world_size=8, session=FakeSession(self.valid_body()))


if __name__ == "__main__":
    unittest.main()
