"""Shared TACO prompt adapters and distillation collator.

The public dataset produced by :mod:`script_code.prepare_data` has a compact
schema shared by every baseline: ``problem``, ``solution``, ``input_output``
and ``problem_id``.  Keeping the prompt construction here prevents SFT, GRPO,
OPSD, OPD, and SKD from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


CODE_INSTRUCTION = (
    "You will be given a programming problem. Write a correct Python program "
    "that solves it. Return only the code inside a single ```python code block."
)

CODE_TEACHER_TRANSITION = (
    "Use the reference program to understand the required algorithm and edge "
    "cases. Then independently produce a correct Python solution. Return only "
    "the code inside a single ```python code block."
)


def build_code_problem(question: str, starter_code: str = "") -> str:
    """Render the task body exactly once before chat-template application."""
    problem = (question or "").strip()
    starter_code = (starter_code or "").strip()
    if starter_code:
        problem += f"\n\nStarter code:\n```python\n{starter_code}\n```"
    return problem


def code_user_content(problem: str) -> str:
    return f"{problem.strip()}\n\n{CODE_INSTRUCTION}"


def code_teacher_content(problem: str, solution: str) -> str:
    return (
        f"Programming problem:\n{problem.strip()}\n\n"
        "Reference solution:\n"
        f"```python\n{solution.strip()}\n```\n\n"
        f"{CODE_TEACHER_TRANSITION}"
    )


def render_code_prompt(tokenizer: Any, problem: str, *, thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": code_user_content(problem)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def render_code_teacher_prompt(
    tokenizer: Any,
    problem: str,
    solution: str | None,
    *,
    alg: str,
    thinking: bool,
) -> str:
    if alg == "opd":
        return render_code_prompt(tokenizer, problem, thinking=thinking)
    if alg != "opsd":
        raise ValueError(f"Unsupported distillation source: {alg!r}")
    if not solution:
        raise ValueError("OPSD code distillation requires a non-empty reference solution.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": code_teacher_content(problem, solution)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def render_code_sft_example(tokenizer: Any, problem: str, solution: str) -> str:
    target = f"```python\n{solution.strip()}\n```"
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": code_user_content(problem)},
            {"role": "assistant", "content": target},
        ],
        tokenize=False,
        enable_thinking=False,
    )


@dataclass
class CodeLengthConfig:
    student_prompt: int = 2_048
    reference_solution: int = 4_096
    teacher_prompt: int = 8_192

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")


class CodeDistillationDataCollator:
    """Create aligned student and teacher prompts for code KD.

    Prepared examples are filtered before training, so truncation here is a
    final safety boundary rather than normal data processing.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        alg: str,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        lengths: CodeLengthConfig | None = None,
    ) -> None:
        if alg not in {"opsd", "opd"}:
            raise ValueError(f"Unsupported algorithm: {alg!r}")
        self.tokenizer = tokenizer
        self.alg = alg
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.lengths = lengths or CodeLengthConfig()
        self.tokenizer.padding_side = "right"

    def _encode_prompts(self, prompts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        raw = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        lengths = [len(ids) for ids in raw["input_ids"]]
        width = max(lengths)
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=width,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"], lengths

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        problems = [str(feature["problem"]) for feature in features]
        references = [
            str(feature.get("solution") or "") if self.alg == "opsd" else None
            for feature in features
        ]
        student_prompts = [
            render_code_prompt(
                self.tokenizer,
                problem,
                thinking=self.student_thinking,
            )
            for problem in problems
        ]
        teacher_prompts = [
            render_code_teacher_prompt(
                self.tokenizer,
                problem,
                reference,
                alg=self.alg,
                thinking=self.teacher_thinking,
            )
            for problem, reference in zip(problems, references, strict=True)
        ]

        student_ids, student_mask, student_lengths = self._encode_prompts(
            student_prompts, self.lengths.student_prompt
        )
        teacher_ids, teacher_mask, teacher_lengths = self._encode_prompts(
            teacher_prompts, self.lengths.teacher_prompt
        )
        return {
            "student_prompts": student_ids,
            "student_prompt_attention_mask": student_mask,
            "student_prompt_length": student_ids.shape[1],
            "student_prompt_lengths_per_example": torch.tensor(student_lengths),
            "teacher_prompts": teacher_ids,
            "teacher_prompt_attention_mask": teacher_mask,
            "teacher_prompt_length": teacher_ids.shape[1],
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths),
            "problems": problems,
            "reference_solutions": references,
        }

