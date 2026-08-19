#!/usr/bin/env python3
"""LoRA SFT baseline on the prepared clean TACO manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dataset_utils import load_local_parquet
from script_code.code_data import render_code_sft_example


@dataclass
class CodeSFTArguments(ScriptArguments):
    train_dataset_path: str = field(default="data/train/taco_code_clean")
    skip_final_model_save: bool = field(default=False)


def latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    checkpoints = sorted(
        path.glob("checkpoint-*"),
        key=lambda item: int(item.name.rsplit("-", 1)[-1]),
    ) if path.is_dir() else []
    return str(checkpoints[-1]) if checkpoints else None


def main() -> None:
    parser = TrlParser((CodeSFTArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
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
        columns=["problem", "solution"],
    )
    dataset = dataset.map(
        lambda row: {
            "text": render_code_sft_example(tokenizer, row["problem"], row["solution"])
        },
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=min(256, max(1, len(dataset) // 100)), seed=42)
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
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
