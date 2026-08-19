#!/usr/bin/env python3
"""GRPO code baseline with deterministic TACO pass-ratio reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dataset_utils import load_local_parquet
from script_code.code_data import render_code_prompt
from script_code.code_execution import reward_code_correctness


@dataclass
class CodeGRPOArguments(ScriptArguments):
    train_dataset_path: str = field(default="data/train/taco_code_clean")
    run_config: str | None = field(default=None)
    skip_final_model_save: bool = field(default=False)


def latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    checkpoints = sorted(
        path.glob("checkpoint-*"),
        key=lambda item: int(item.name.rsplit("-", 1)[-1]),
    ) if path.is_dir() else []
    return str(checkpoints[-1]) if checkpoints else None


def main() -> None:
    parser = TrlParser((CodeGRPOArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if script_args.run_config and not training_args.output_dir.endswith(script_args.run_config):
        training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation or "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "use_cache": not training_args.gradient_checkpointing,
    }
    quantization = get_quantization_config(model_args)
    if quantization is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization
    training_args.model_init_kwargs = model_kwargs

    dataset = load_local_parquet(
        script_args.train_dataset_path,
        columns=["problem", "input_output", "problem_id"],
    )
    dataset = dataset.map(
        lambda row: {
            "prompt": render_code_prompt(tokenizer, row["problem"], thinking=False),
            "input_output": row["input_output"],
            "problem_id": row["problem_id"],
        },
        remove_columns=["problem"],
    )
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_code_correctness,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    resume_checkpoint = training_args.resume_from_checkpoint or latest_checkpoint(
        training_args.output_dir
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    if not script_args.skip_final_model_save:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
