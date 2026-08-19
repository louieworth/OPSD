#!/usr/bin/env python3
"""Idempotently add Qwen3 non-thinking and LoRA support to pinned LCB v6."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Pinned LiveCodeBench source changed; cannot patch {path}")
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()
    root = Path(args.repo)
    prompt_file = root / "lcb_runner/prompts/code_generation.py"
    runner_file = root / "lcb_runner/runner/vllm_runner.py"

    replace_once(
        prompt_file,
        '    prompt += f"<|im_start|>assistant\\n"\n',
        '    # Qwen3 non-thinking: the empty think block is part of the prompt.\n'
        '    prompt += f"<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"\n',
    )
    replace_once(
        runner_file,
        "try:\n    from transformers import AutoTokenizer\n    from vllm import LLM, SamplingParams\n",
        "try:\n    import json\n    import os\n    from transformers import AutoTokenizer\n"
        "    from vllm import LLM, SamplingParams\n    from vllm.lora.request import LoRARequest\n",
    )
    replace_once(
        runner_file,
        "        self.llm = LLM(\n",
        "        self.lora_request = None\n"
        "        adapter_config = os.path.join(model_tokenizer_path, 'adapter_config.json')\n"
        "        if os.path.isfile(adapter_config):\n"
        "            with open(adapter_config) as handle:\n"
        "                adapter = json.load(handle)\n"
        "            adapter_path = model_tokenizer_path\n"
        "            model_tokenizer_path = os.environ.get(\n"
        "                'LCB_BASE_MODEL_PATH', adapter.get('base_model_name_or_path')\n"
        "            )\n"
        "            self.lora_request = LoRARequest('code-eval', 1, adapter_path)\n"
        "        self.llm = LLM(\n",
    )
    replace_once(
        runner_file,
        "            max_num_seqs=args.max_num_seqs,\n        )\n",
        "            max_num_seqs=args.max_num_seqs,\n"
        "            enable_lora=self.lora_request is not None,\n"
        "            max_lora_rank=128,\n        )\n",
    )
    replace_once(
        runner_file,
        "            vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)\n",
        "            vllm_outputs = self.llm.generate(\n"
        "                remaining_prompts, self.sampling_params, lora_request=self.lora_request\n"
        "            )\n",
    )
    print(f"Patched LiveCodeBench at {root}")


if __name__ == "__main__":
    main()

