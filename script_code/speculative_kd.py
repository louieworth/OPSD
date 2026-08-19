"""Repository-local Speculative Knowledge Distillation trajectory sampler.

This implementation follows Google Research SKD's interleaved sampling rule
without patching the installed Transformers package.  Student and teacher may
use different prompts, which is required by OPSD.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterable

import torch


@dataclass(frozen=True)
class SKDConfig:
    max_new_tokens: int = 4_096
    draft_length: int = 5
    accept_top_k: int = 25
    student_temperature: float = 1.1
    student_top_p: float = 0.95
    student_top_k: int = 20
    correction_temperature: float = 0.2
    correction_top_p: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "draft_length", "accept_top_k", "student_top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        for name in ("student_temperature", "correction_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("student_top_p", "correction_top_p"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")


@dataclass
class SKDStats:
    proposals: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    corrections: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0

    @property
    def average_accepted_prefix(self) -> float:
        return self.accepted_tokens / self.proposals if self.proposals else 0.0

    @property
    def teacher_token_ratio(self) -> float:
        total = self.accepted_tokens + self.corrections
        return self.corrections / total if total else 0.0


def _sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> int:
    scores = logits.float() / temperature
    if top_k is not None and top_k > 0:
        keep = min(top_k, scores.shape[-1])
        threshold = torch.topk(scores, keep).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(
            -1, sorted_indices, sorted_scores
        )
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _crop_cache(cache: Any, length: int) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    if isinstance(cache, tuple):
        cropped_layers = []
        for layer in cache:
            if not isinstance(layer, tuple) or len(layer) < 2:
                raise TypeError("Unsupported legacy KV-cache structure.")
            cropped_layers.append(
                tuple(
                    value[..., :length, :] if torch.is_tensor(value) and value.ndim >= 3 else value
                    for value in layer
                )
            )
        return tuple(cropped_layers)
    raise TypeError(f"Unsupported KV-cache type: {type(cache).__name__}")


def _normalize_eos(eos_token_ids: int | Iterable[int] | None) -> set[int]:
    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, int):
        return {eos_token_ids}
    return {int(token_id) for token_id in eos_token_ids}


class SpeculativeKDGenerator:
    def __init__(self, config: SKDConfig) -> None:
        self.config = config

    @staticmethod
    def _forward(
        model: Any,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
    ) -> Any:
        return model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    @torch.no_grad()
    def generate_one(
        self,
        *,
        student_model: Any,
        teacher_model: Any,
        student_prompt_ids: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        eos_token_ids: int | Iterable[int] | None,
        generator: torch.Generator | None = None,
        teacher_context: Callable[[], ContextManager[Any]] | None = None,
    ) -> tuple[list[int], SKDStats]:
        """Generate one interleaved trajectory; prompts must have batch size one."""
        if student_prompt_ids.ndim != 2 or student_prompt_ids.shape[0] != 1:
            raise ValueError("student_prompt_ids must have shape [1, sequence].")
        if teacher_prompt_ids.ndim != 2 or teacher_prompt_ids.shape[0] != 1:
            raise ValueError("teacher_prompt_ids must have shape [1, sequence].")
        if student_prompt_ids.device != teacher_prompt_ids.device:
            raise ValueError("Student and teacher prompts must be on the same device.")

        teacher_context = teacher_context or nullcontext
        eos = _normalize_eos(eos_token_ids)
        stats = SKDStats()
        completion: list[int] = []

        student_initial = self._forward(student_model, student_prompt_ids)
        student_cache = student_initial.past_key_values
        student_next = student_initial.logits[:, -1, :]
        with teacher_context():
            teacher_initial = self._forward(teacher_model, teacher_prompt_ids)
        teacher_cache = teacher_initial.past_key_values
        teacher_next = teacher_initial.logits[:, -1, :]

        student_prefix = student_prompt_ids.shape[1]
        teacher_prefix = teacher_prompt_ids.shape[1]

        while len(completion) < self.config.max_new_tokens:
            remaining = self.config.max_new_tokens - len(completion)
            proposal_limit = min(self.config.draft_length, remaining)
            proposal: list[int] = []

            for _ in range(proposal_limit):
                token = _sample(
                    student_next,
                    temperature=self.config.student_temperature,
                    top_p=self.config.student_top_p,
                    top_k=self.config.student_top_k,
                    generator=generator,
                )
                proposal.append(token)
                token_tensor = torch.tensor([[token]], device=student_prompt_ids.device)
                student_step = self._forward(student_model, token_tensor, student_cache)
                student_cache = student_step.past_key_values
                student_next = student_step.logits[:, -1, :]
                if token in eos:
                    break

            stats.proposals += 1
            stats.proposed_tokens += len(proposal)
            proposal_tensor = torch.tensor([proposal], device=teacher_prompt_ids.device)
            with teacher_context():
                teacher_block = self._forward(teacher_model, proposal_tensor, teacher_cache)

            accepted = 0
            for index, candidate in enumerate(proposal):
                verification_logits = (
                    teacher_next if index == 0 else teacher_block.logits[:, index - 1, :]
                )
                support = torch.topk(
                    verification_logits,
                    min(self.config.accept_top_k, verification_logits.shape[-1]),
                    dim=-1,
                ).indices
                if bool((support == candidate).any()):
                    accepted += 1
                else:
                    break

            stats.accepted_tokens += accepted
            old_completion_length = len(completion)
            completion.extend(proposal[:accepted])

            if accepted == len(proposal):
                teacher_cache = teacher_block.past_key_values
                teacher_next = teacher_block.logits[:, -1, :]
                if proposal and proposal[-1] in eos:
                    break
                continue

            # Discard the rejected candidate and its suffix from both caches.
            student_cache = _crop_cache(
                student_cache, student_prefix + old_completion_length + accepted
            )
            teacher_cache = _crop_cache(
                teacher_block.past_key_values,
                teacher_prefix + old_completion_length + accepted,
            )
            correction_logits = (
                teacher_next if accepted == 0 else teacher_block.logits[:, accepted - 1, :]
            )
            correction = _sample(
                correction_logits,
                temperature=self.config.correction_temperature,
                top_p=self.config.correction_top_p,
                top_k=None,
                generator=generator,
            )
            completion.append(correction)
            stats.corrections += 1
            correction_tensor = torch.tensor([[correction]], device=student_prompt_ids.device)

            student_step = self._forward(student_model, correction_tensor, student_cache)
            student_cache = student_step.past_key_values
            student_next = student_step.logits[:, -1, :]
            with teacher_context():
                teacher_step = self._forward(teacher_model, correction_tensor, teacher_cache)
            teacher_cache = teacher_step.past_key_values
            teacher_next = teacher_step.logits[:, -1, :]
            if correction in eos:
                break

        return completion, stats

