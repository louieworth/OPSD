#!/usr/bin/env python3
"""EvalPlus generation with explicit Qwen3 non-thinking and vLLM LoRA."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evalplus.codegen import codegen
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.evaluate import evaluate
from evalplus.provider.base import DecoderBase


def resolve_model(path: str, explicit_base: str | None) -> tuple[str, str | None, int]:
    adapter_config = Path(path) / "adapter_config.json"
    if not adapter_config.is_file():
        return path, None, 64
    data = json.loads(adapter_config.read_text())
    base = explicit_base or data.get("base_model_name_or_path")
    if not base:
        raise ValueError("LoRA adapter_config.json does not identify its base model.")
    return str(base), path, int(data.get("r", 64))


class NonThinkingEvalPlusDecoder(DecoderBase):
    def __init__(
        self,
        model_path: str,
        dataset: str,
        tensor_parallel_size: int,
        max_model_len: int,
        max_num_seqs: int,
        top_p: float,
        base_model: str | None,
        **kwargs,
    ) -> None:
        super().__init__(model_path, **kwargs)
        self.top_p = top_p
        resolved_base, adapter_path, lora_rank = resolve_model(model_path, base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_base, use_fast=False, trust_remote_code=self.trust_remote_code
        )
        # This is an instruction/chat decoder.  Direct-completion stops such
        # as ``\ndef `` would truncate an ordinary fenced solution at its first
        # function definition.
        self.eos += ["\n```\n"]
        llm_kwargs = {
            "model": resolved_base,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enable_prefix_caching": True,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
        }
        self.lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            llm_kwargs.update(enable_lora=True, max_lora_rank=max(64, lora_rank))
            self.lora_request = LoRARequest("code-eval", 1, adapter_path)
        self.llm = LLM(**llm_kwargs)
        self.total_generations = 0
        self.length_capped = 0

    def is_direct_completion(self) -> bool:
        return False

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        batch_size = min(self.batch_size, num_samples)
        content = (
            "Provide a self-contained Python solution to the following task. "
            "Return only Python code inside one markdown code block.\n\n" + prompt
        )
        raw_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = self.llm.generate(
            [raw_prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature if do_sample else 0.0,
                max_tokens=self.max_new_tokens,
                top_p=self.top_p if do_sample else 1.0,
                stop=self.eos,
            ),
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        self.total_generations += len(outputs)
        self.length_capped += sum(
            int(output.outputs[0].finish_reason == "length") for output in outputs
        )
        return [output.outputs[0].text.replace("\t", "    ") for output in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--root", required=True)
    parser.add_argument("--n_samples", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4_096)
    parser.add_argument("--max_model_len", type=int, default=6_144)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--parallel", type=int)
    args = parser.parse_args()

    decoder = NonThinkingEvalPlusDecoder(
        model_path=args.model,
        base_model=args.base_model,
        dataset=args.dataset,
        batch_size=args.bs,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        dtype=args.dtype,
        trust_remote_code=True,
        instruction_prefix="",
        response_prefix="",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        top_p=args.top_p,
    )
    identifier = Path(args.model.rstrip("/")).name + "_nonthinking_4k"
    target = Path(args.root) / args.dataset / f"{identifier}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = (
        get_human_eval_plus(version="default")
        if args.dataset == "humaneval"
        else get_mbpp_plus(version="default")
    )
    codegen(
        target_path=str(target),
        model=decoder,
        dataset=dataset,
        n_samples=args.n_samples,
        resume=True,
    )
    truncation = {
        "total": decoder.total_generations,
        "length_capped": decoder.length_capped,
        "rate": decoder.length_capped / decoder.total_generations if decoder.total_generations else 0.0,
    }
    target.with_suffix(".truncation.json").write_text(json.dumps(truncation, indent=2) + "\n")
    del decoder
    gc.collect()
    evaluate(
        dataset=args.dataset,
        samples=str(target),
        parallel=args.parallel,
        version="default",
        i_just_wanna_run=False,
    )


if __name__ == "__main__":
    main()
