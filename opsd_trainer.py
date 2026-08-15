# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from data_collator import SelfDistillationDataCollator, WindowDataCollator
from trd_refinement import TeacherVLLMClient, build_refinement_prompt


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


class EMAUpdateCallback(TrainerCallback):
    """Update EMA teacher weights after each optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when the optimizer actually stepped (end of a gradient accumulation cycle)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class GenerationOutputCallback(TrainerCallback):
    """Persist generation records at completed update boundaries and at train end."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0 and state.global_step % self.trainer._generation_save_frequency == 0:
            self.trainer._save_generation_outputs(state.global_step)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            self.trainer._save_generation_outputs(state.global_step)
        finally:
            client = self.trainer.refinement_vllm_client
            if client is not None:
                client.close()


class OPSDTrainer(SFTTrainer):
    _tag_names = ["trl", "opsd"]
    _name = "OPSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        fixed_teacher: bool = False,
        reason_first: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        teacher_refine: bool = False,
        max_refinement_length: int | None = None,
        distillation_temperature: float | None = None,
        refinement_vllm_server_host: str = "127.0.0.1",
        refinement_vllm_server_port: int = 8002,
        refinement_vllm_connect_timeout: float = 10.0,
        refinement_vllm_request_timeout: float = 1800.0,
        refinement_vllm_max_model_len: int | None = None,
        alg: str = "opsd",
        teacher_model: PreTrainedModel | nn.Module | str | None = None,
    ):
        if alg not in {"opsd", "opd"}:
            raise ValueError(f"Unsupported algorithm: {alg!r}. Expected 'opsd' or 'opd'.")
        if alg == "opd" and teacher_model is None:
            raise ValueError("OPD requires an external teacher_model.")
        if alg == "opsd" and teacher_model is not None:
            raise ValueError("OPSD uses the student as its teacher; teacher_model must be None.")
        if alg == "opd" and reason_first:
            raise ValueError("reason_first is only supported by OPSD because OPD has no privileged solution y*.")
        if alg == "opd" and use_ema_teacher:
            raise ValueError("use_ema_teacher is only supported by OPSD; the OPD teacher is external and fixed.")
        if alg == "opd" and fixed_teacher:
            raise ValueError("fixed_teacher is an OPSD option; the external OPD teacher is always fixed.")

        self.alg = alg
        self.teacher_refine = teacher_refine
        self.teacher_thinking = teacher_thinking
        self.max_refinement_length = max_refinement_length or args.max_completion_length
        if refinement_vllm_max_model_len is None:
            refinement_vllm_max_model_len = (
                20_000 + self.max_refinement_length if alg == "opsd" else 20_000
            )
        self.refinement_vllm_max_model_len = refinement_vllm_max_model_len
        self.student_context_max_length = int(args.max_length)
        student_response_reserve = max(
            int(args.max_completion_length), int(self.max_refinement_length)
        )
        if self.teacher_refine:
            if student_response_reserve >= self.student_context_max_length:
                raise ValueError(
                    "TRD requires --max_length to leave room for both y_o and y_r; "
                    f"got max_length={self.student_context_max_length}, "
                    f"max_completion_length={args.max_completion_length}, and "
                    f"max_refinement_length={self.max_refinement_length}."
                )
            self.student_prompt_max_length = (
                self.student_context_max_length - student_response_reserve
            )
        else:
            self.student_prompt_max_length = self.student_context_max_length
        if self.teacher_refine and self.max_refinement_length >= self.refinement_vllm_max_model_len:
            raise ValueError(
                "TRD requires refinement_vllm_max_model_len to leave room for y_r; "
                f"got refinement_vllm_max_model_len={self.refinement_vllm_max_model_len} and "
                f"max_refinement_length={self.max_refinement_length}."
            )
        self.refinement_prompt_max_length = (
            self.refinement_vllm_max_model_len - self.max_refinement_length
        )
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Custom data collator for self-distillation
        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class,
                # TRD uses the same x prefix for y_o generation and student
                # KL on y_r, so reserve the larger response budget up front.
                max_length=self.student_prompt_max_length,
                reason_first=reason_first,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
                alg=alg,
            )

        # OPSD/OPD use a custom collator over the raw dataset columns.
        # TRL's SFTTrainer otherwise tries to tokenize a non-existent `text`
        # field and Transformers may remove these columns before collation.
        args.dataset_kwargs = dict(args.dataset_kwargs or {})
        args.dataset_kwargs["skip_prepare_dataset"] = True
        args.remove_unused_columns = False

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.distillation_temperature = (
            args.temperature if distillation_temperature is None else distillation_temperature
        )
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None  # lazily initialized on first optimizer step
        self.windowed_policy_updates = bool(getattr(args, "windowed_policy_updates", False))
        self.total_rollout_steps = getattr(args, "total_rollout_steps", None)
        self.policy_gradient_updates = getattr(args, "policy_gradient_updates", None)
        self.rollouts_per_update = int(getattr(args, "rollouts_per_update", 1))
        self.rollout_micro_batch_size = int(args.per_device_train_batch_size)
        self._schedule_metadata_written = False

        # Validate fixed_teacher option
        if self.alg == "opsd" and self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.alg == "opsd" and self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'='*80}\n")

        self.refinement_vllm_client = None
        if self.teacher_refine and self.accelerator.is_main_process:
            self.refinement_vllm_client = TeacherVLLMClient(
                host=refinement_vllm_server_host,
                port=refinement_vllm_server_port,
                connect_timeout=refinement_vllm_connect_timeout,
                read_timeout=refinement_vllm_request_timeout,
                expected_world_size=4,
                max_model_len=refinement_vllm_max_model_len,
            )

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        # Track generation outputs for saving
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps
        self.add_callback(GenerationOutputCallback(self))

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase (when reason_first=True)
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

        self.teacher_model = None
        if self.alg == "opd":
            teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs or {})
            if not isinstance(teacher_model, str) and teacher_model_init_kwargs:
                raise ValueError(
                    "teacher_model_init_kwargs can only be used when teacher_model is a model path."
                )
            if isinstance(teacher_model, str):
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    teacher_model, **teacher_model_init_kwargs
                )
            if teacher_model.config.vocab_size != self.model.config.vocab_size:
                raise ValueError(
                    "OPD requires student and teacher to share a vocabulary; got "
                    f"{self.model.config.vocab_size} and {teacher_model.config.vocab_size}."
                )
            teacher_model.requires_grad_(False)
            disable_dropout_in_model(teacher_model)
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(
                    teacher_model, evaluation_mode=True
                )
            self.teacher_model.eval()
            print(f"\n{'='*80}")
            print("OPD EXTERNAL TEACHER ENABLED")
            print(f"Teacher: {args.teacher_model_name_or_path}")
            print("Teacher and student use the identical x prompt; no privileged solution y* is used.")
            print(f"{'='*80}\n")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = ["problem"]
        if self.alg == "opsd":
            required_columns.append("solution")
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def get_train_dataloader(self):
        """Return ordinary batches in legacy mode and whole update windows otherwise."""
        if not self.windowed_policy_updates:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train dataset.")
        if isinstance(self.train_dataset, IterableDataset):
            raise ValueError("Windowed policy updates do not support IterableDataset.")
        if self.accelerator.split_batches:
            raise ValueError("Windowed policy updates require Accelerate split_batches=False.")
        if self.accelerator.dispatch_batches is True:
            raise ValueError("Windowed policy updates require Accelerate dispatch_batches=False.")
        if int(self.accelerator.gradient_accumulation_steps) != 1:
            raise ValueError(
                "Windowed policy updates require launch-level gradient accumulation 1; "
                f"Accelerate is configured with {self.accelerator.gradient_accumulation_steps}."
            )

        micro_batch_size = self.rollout_micro_batch_size
        window_size = self.rollouts_per_update
        loader_batch_size = micro_batch_size * window_size
        minimum_examples = loader_batch_size * self.accelerator.num_processes
        if len(self.train_dataset) < minimum_examples:
            raise ValueError(
                "The dataset is too small for one complete distributed policy-update window: "
                f"need at least {minimum_examples} examples, found {len(self.train_dataset)}."
            )

        # The DataLoader carries B*R CPU examples, but DeepSpeed still executes
        # one B-sized model microbatch at a time inside training_step.
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is not None:
            ds_config = deepspeed_plugin.deepspeed_config
            ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
            ds_config["gradient_accumulation_steps"] = 1
            ds_config["train_batch_size"] = micro_batch_size * self.accelerator.num_processes

        dataloader_params = {
            "batch_size": loader_batch_size,
            "collate_fn": WindowDataCollator(
                self.data_collator,
                micro_batch_size=micro_batch_size,
                window_size=window_size,
            ),
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "sampler": self._get_train_sampler(self.train_dataset),
            "drop_last": True,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(self.train_dataset, **dataloader_params)
        return self.accelerator.prepare_data_loader(dataloader, device_placement=False)

    def get_total_train_batch_size(self, args) -> int:
        batch_size = super().get_total_train_batch_size(args)
        return batch_size * self.rollouts_per_update if self.windowed_policy_updates else batch_size

    def _window_schedule_metadata(self) -> dict[str, Any]:
        return {
            "total_rollout_steps": self.total_rollout_steps,
            "policy_gradient_updates": self.policy_gradient_updates,
            "rollouts_per_update": self.rollouts_per_update,
            "per_device_train_batch_size": self.rollout_micro_batch_size,
            "world_size": self.accelerator.num_processes,
            "dataset_length": len(self.train_dataset) if self.train_dataset is not None else None,
            "dataset_fingerprint": getattr(self.train_dataset, "_fingerprint", None),
            "alg": self.alg,
            "teacher_refine": self.teacher_refine,
            "model_name_or_path": str(self.model_name_or_path),
            "teacher_model_name_or_path": getattr(self.args, "teacher_model_name_or_path", None),
            "max_length": self.student_context_max_length,
            "student_prompt_max_length": self.student_prompt_max_length,
            "max_completion_length": self.args.max_completion_length,
            "max_refinement_length": self.max_refinement_length,
            "refinement_prompt_max_length": self.refinement_prompt_max_length,
            "refinement_vllm_max_model_len": self.refinement_vllm_max_model_len,
        }

    def _write_window_schedule(self, directory: str | os.PathLike[str]) -> None:
        if not self.windowed_policy_updates or not self.args.should_save:
            return
        import json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "window_schedule.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._window_schedule_metadata(), handle, indent=2, sort_keys=True)

    def prepare_window_schedule(self, resume_from_checkpoint: str | None = None) -> None:
        """Persist schedule metadata and reject incompatible resume settings."""
        if not self.windowed_policy_updates:
            return
        import json

        expected = self._window_schedule_metadata()
        if resume_from_checkpoint is not None:
            schedule_path = Path(resume_from_checkpoint) / "window_schedule.json"
            if not schedule_path.is_file():
                raise ValueError(
                    f"Cannot exactly resume a windowed run: missing {schedule_path}."
                )
            with schedule_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved != expected:
                differing = {
                    key: {"checkpoint": saved.get(key), "current": expected.get(key)}
                    for key in sorted(set(saved) | set(expected))
                    if saved.get(key) != expected.get(key)
                }
                raise ValueError(f"Window schedule differs from checkpoint: {differing}")
        self._write_window_schedule(self.args.output_dir)
        self.accelerator.wait_for_everyone()

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.windowed_policy_updates:
            run_dir = self._get_output_dir(trial=trial)
            checkpoint_dir = Path(run_dir) / f"checkpoint-{self.state.global_step}"
            self._write_window_schedule(checkpoint_dir)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, approximate forward KL over the teacher's top-k tokens. The teacher distribution is
                renormalized over those tokens, while the student probabilities retain their full-vocabulary
                denominator, as in Eq. (10) of arXiv:2603.07079. This mode requires beta=0.
            token_clip:
                if set, clips per-token divergence values to this maximum before reduction. Prevents style tokens from dominating the gradient signal over math tokens.

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if top_k is not None and top_k > 0:
            if beta != 0:
                raise ValueError("top-k distillation is defined only for forward KL (beta=0).")
            if top_k > teacher_logits.size(-1):
                raise ValueError(
                    f"top_k={top_k} exceeds the teacher vocabulary size {teacher_logits.size(-1)}."
                )

            if logits_are_probs:
                student_log_probs_full = torch.log(student_logits.clamp_min(1e-8))
                teacher_top_k_probs, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                teacher_top_k_probs = teacher_top_k_probs / teacher_top_k_probs.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                teacher_log_probs = torch.log(teacher_top_k_probs.clamp_min(1e-8))
            else:
                student_scaled_logits = student_logits / temperature
                teacher_scaled_logits = teacher_logits / temperature
                _, top_k_indices = torch.topk(teacher_scaled_logits, k=top_k, dim=-1)
                teacher_top_k_logits = torch.gather(
                    teacher_scaled_logits, dim=-1, index=top_k_indices
                )
                teacher_log_probs = F.log_softmax(teacher_top_k_logits, dim=-1)
                student_log_probs_full = F.log_softmax(student_scaled_logits, dim=-1)

            # Eq. (10): only the teacher is renormalized on S_k. The student
            # probability keeps the full-vocabulary partition function.
            student_log_probs = torch.gather(
                student_log_probs_full, dim=-1, index=top_k_indices
            )
        elif logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Per-token clipping: cap each token's divergence value
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _trd_global_token_mean(
        self, local_loss_sum: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Scale a local TRD token sum so DP gradient averaging yields a global token mean."""
        local_token_count = (labels != -100).sum().to(device=local_loss_sum.device)
        global_token_count = self.accelerator.reduce(local_token_count, reduction="sum")
        if int(global_token_count.item()) <= 0:
            raise RuntimeError("TRD requires at least one unmasked target token globally.")
        return (
            local_loss_sum
            * self.accelerator.num_processes
            / global_token_count.to(dtype=local_loss_sum.dtype)
        )

    def _update_ema(self):
        """Update EMA parameters after an optimizer step.

        On the very first call this lazily initializes the EMA state as an exact copy of the
        current (trainable) model parameters, then returns without applying a decay step.
        Subsequent calls apply: ema = decay * ema + (1 - decay) * student.

        Only trainable parameters are tracked (i.e. LoRA adapter weights for PEFT models,
        or all parameters for full fine-tuning).

        ZeRO-3 note: with ZeRO-3 each rank only holds a shard of every parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        every rank sees the full parameter tensor when snapshotting / updating the EMA.
        The EMA tensors are therefore full-sized copies, which is also required by
        `_ema_teacher_context` when it swaps the gathered student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None → read-only gather; original partitions are restored on exit.
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # first call = initialization only, no decay update

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                # Lazy init: snapshot the current weights as the initial EMA state.
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # first call = initialization only, no decay update

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to the same device as the live param (handles multi-GPU setups)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """Context manager that temporarily loads EMA weights for the teacher forward pass.

        Swaps `param.data` of every tracked (trainable) parameter with its EMA counterpart,
        runs the body (teacher forward), then restores the student weights unconditionally.
        Safe to use inside `torch.no_grad()`.  If EMA has not been initialized yet (step 0),
        this is a no-op and the current student weights are used instead.

        ZeRO-3 note: direct `param.data` assignment bypasses ZeRO-3's shard lifecycle and
        corrupts its internal state, causing size-mismatch errors during gradient-checkpoint
        recomputation.  When ZeRO-3 is active we therefore wrap the swap inside
        `deepspeed.zero.GatheredParameters` so the parameters are fully materialised on every
        rank before we touch them, and ZeRO-3 re-partitions cleanly when the context exits.
        """
        if self._ema_params is None:
            yield  # EMA not yet initialized; fall back to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to re-partition from rank-0's param.data on exit,
            # which will be the restored student weights.
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the self-distillation loss with memory-efficient log-prob extraction.

        Memory optimization: Extract only needed log-probs immediately and free large tensors.
        """
        # Get batch-level prompt lengths
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === STUDENT FORWARD - Extract log-probs immediately ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )

        # Extract only what we need and convert to log-probs immediately
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            # For reverse KL, we only need log-probs of sampled tokens
            student_log_probs = F.log_softmax(student_logits / self.distillation_temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs  # Free immediately!
        else:
            # For JSD, keep logits (temperature will be applied in generalized_jsd_loss)
            student_logits_for_loss = student_logits
            del student_logits

        # Free the full outputs (but keep reference for return_outputs if needed)
        if return_outputs:
            # Create a minimal output object to return (just the loss, no logits)
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        # === TEACHER FORWARD - Extract log-probs immediately ===
        # Choose teacher model/context based on mode:
        #   OPD              → fixed external Qwen3-8B teacher
        #   use_ema_teacher  → swap in EMA weights temporarily
        #   fixed_teacher    → disable LoRA adapters (base model = initial policy)
        #   default (dynamic)→ no-op, use current student weights
        if self.alg == "opd":
            teacher_forward_model = self.teacher_model
            adapter_context = nullcontext()
        elif self.use_ema_teacher:
            teacher_forward_model = model
            adapter_context = self._ema_teacher_context(model)
        elif self.fixed_teacher and is_peft_model(model):
            teacher_forward_model = model
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            teacher_forward_model = model
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            if self.alg == "opd":
                teacher_forward_model.eval()
            outputs_teacher = teacher_forward_model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                use_cache=False,
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.distillation_temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs  # Free immediately!
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        # === COMPUTE LOSS with only small tensors ===
        if self.use_thinking_machines_loss:
            # Thinking Machines uses RL-style policy gradient:
            # Advantage = log π_teacher(x) - log π_student(x)
            # Loss = -E[Advantage * log π_student(x)]
            #
            # CRITICAL: advantage must be detached to prevent gradients flowing through it.
            # We want: ∇θ L = -E[A(x) * ∇θ log π_student(x)]
            # NOT: ∇θ L = -E[(T(x) - S(x)) * ∇θ S(x)] where both terms differentiate

            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            # Apply masking before computing loss
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            # Policy gradient loss: -advantage * log π_student
            # Negative because we minimize loss (gradient descent), but want to maximize reward
            loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            # Temperature is applied inside generalized_jsd_loss
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.distillation_temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
                reduction="sum" if self.teacher_refine else "batchmean",
            )
            if self.teacher_refine:
                loss = self._trd_global_token_mean(loss, shifted_labels)
            del student_logits_for_loss, teacher_logits_for_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning about the solution."""
        if self.use_vllm:
            # Use vLLM for fast reasoning generation
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            # Use transformers generation (slower)
            with torch.no_grad():
                # Temporarily enable KV cache
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                # If fixed_teacher=True, disable LoRA adapters
                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=teacher_reasoning_prompts,
                            attention_mask=teacher_reasoning_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts only."""
        import time

        start_time = time.time()

        # Temporarily enable KV cache for generation if it was disabled for training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'='*80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Batch size: {inputs['student_prompts'].shape[0]}")
        print(f"  Prompt length: {inputs['student_prompts'].shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'='*80}\n")

        # Generate output with respect to the student prompt only
        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts using vLLM."""
        import time

        device = self.accelerator.device

        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        # Start timing for vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                server_result = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                    generation_kwargs={"presence_penalty": presence_penalty},
                )
                completion_ids = server_result["completion_ids"]
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_holder = [completion_ids]
            broadcast_object_list(completion_holder, from_process=0)
            completion_ids = completion_holder[0]
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Calculate and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        if self.teacher_refine:
            # The same x prefix is later paired with y_r.  Respect the larger
            # of the y_o/y_r reserves selected during trainer initialization.
            prompt_max_length = self.student_prompt_max_length
        else:
            prompt_max_length = (
                max(1, self.args.max_length - max_completion_length)
                if self.args.max_length
                else None
            )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning using vLLM."""
        import time

        device = self.accelerator.device

        # Decode prompts for vLLM
        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                server_result = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
                completion_ids = server_result["completion_ids"]
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_holder = [completion_ids]
            broadcast_object_list(completion_holder, from_process=0)
            completion_ids = completion_holder[0]
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        # Combine prompt + completion
        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    @staticmethod
    def _detach_to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: OPSDTrainer._detach_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [OPSDTrainer._detach_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(OPSDTrainer._detach_to_cpu(item) for item in value)
        return value

    def _ensure_student_vllm_current(self):
        """Synchronize before a window, including the first window after resume."""
        if self.use_vllm and self._last_vllm_sync_step != self.state.global_step:
            self._move_model_to_vllm()
            self._last_vllm_sync_step = self.state.global_step

    def _apply_reason_first(self, model, inputs):
        """Build the legacy reason-first teacher prompt without performing backward."""
        if not self.reason_first:
            return inputs
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            teacher_reasoning_ids = self.generate_teacher_reasoning(
                unwrapped_model,
                inputs["teacher_reasoning_prompts"],
                inputs.get("teacher_reasoning_attention_mask"),
            )
        reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
        reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
        teacher_prompts = torch.cat(
            [
                inputs["teacher_reasoning_prompts"],
                reasoning_completions,
                inputs["teacher_transition_tokens"],
            ],
            dim=1,
        )
        inputs["teacher_prompts"] = teacher_prompts
        teacher_attention_mask = torch.ones_like(teacher_prompts)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_prompts == self.processing_class.pad_token_id] = 0
        inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
        inputs["teacher_prompt_length"] = teacher_prompts.shape[1]
        inputs["teacher_prompt_lengths_per_example"] = torch.full(
            (teacher_prompts.shape[0],),
            teacher_prompts.shape[1],
            dtype=torch.long,
            device=teacher_prompts.device,
        )
        return inputs

    def _collect_original_rollout(self, model, inputs, rollout_step: int) -> dict[str, Any]:
        """Generate y_o and return a graph-free CPU record."""
        inputs = self._apply_reason_first(model, dict(inputs))
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
            generated_prompt_width = generated_ids.shape[1] - self.generation_config.max_new_tokens
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            generated_ids, generated_attention_mask, _ = result
            prompt_texts = self.processing_class.batch_decode(
                inputs["student_prompts"], skip_special_tokens=False
            )
            generated_prompt_width = int(inputs["student_prompt_length"])
            completion_texts = self.processing_class.batch_decode(
                generated_ids[:, generated_prompt_width:], skip_special_tokens=True
            )

        generated_completion_ids = generated_ids[:, generated_prompt_width:]
        generated_completion_mask = generated_attention_mask[:, generated_prompt_width:]
        if self.teacher_refine and generated_ids.shape[1] > self.student_context_max_length:
            raise RuntimeError(
                "TRD student rollout exceeded its configured context: "
                f"sequence width {generated_ids.shape[1]} > "
                f"max_length {self.student_context_max_length}."
            )
        completion_ids = []
        clean_completion_texts = []
        for token_ids, token_mask in zip(generated_completion_ids, generated_completion_mask, strict=True):
            ids = token_ids[token_mask.bool()].detach().cpu().tolist()
            if not ids:
                raise RuntimeError(f"Student generated an empty y_o at rollout step {rollout_step}.")
            completion_ids.append(ids)
            clean_completion_texts.append(
                self.processing_class.decode(ids, skip_special_tokens=True)
            )

        return {
            "inputs": self._detach_to_cpu(inputs),
            # Preserve the exact legacy student sequence for vanilla windowed
            # updates.  Re-tokenizing or re-padding it would make U=N differ
            # from the historical one-rollout-per-update path.
            "generated_ids": generated_ids.detach().cpu(),
            "generated_attention_mask": generated_attention_mask.detach().cpu(),
            "generated_prompt_width": generated_prompt_width,
            "prompt_texts": list(prompt_texts),
            "original_completion_ids": completion_ids,
            "original_completion_texts": clean_completion_texts,
            "raw_completion_texts": list(completion_texts),
            "rollout_step": rollout_step,
        }

    def _pad_token_lists(
        self, sequences: list[list[int]], *, padding_side: str = "right"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences or any(len(sequence) == 0 for sequence in sequences):
            raise RuntimeError("Every distillation target must contain at least one token.")
        if padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding side: {padding_side}.")
        width = max(len(sequence) for sequence in sequences)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            raise ValueError("A pad token is required for variable-length distillation batches.")
        input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            target_slice = slice(0, len(sequence)) if padding_side == "right" else slice(width - len(sequence), width)
            input_ids[row, target_slice] = sequence_tensor
            attention_mask[row, target_slice] = 1
        return input_ids, attention_mask

    def _build_distillation_batch(
        self,
        rollout: dict[str, Any],
        target_ids: list[list[int]],
        teacher_prompt_ids: list[list[int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build aligned student/teacher inputs that train only on target_ids."""
        source = rollout["inputs"]
        target_tensor, target_mask = self._pad_token_lists(target_ids)

        source_student_prompts = source["student_prompts"].cpu()
        source_student_mask = source["student_prompt_attention_mask"].cpu()
        student_prompt_sequences = [
            token_ids[token_mask.bool()].tolist()
            for token_ids, token_mask in zip(
                source_student_prompts, source_student_mask, strict=True
            )
        ]
        student_prompts, student_prompt_mask = self._pad_token_lists(
            student_prompt_sequences, padding_side="left"
        )
        student_prompt_width = student_prompts.shape[1]
        student_input_ids = torch.cat([student_prompts, target_tensor], dim=1)
        student_attention_mask = torch.cat([student_prompt_mask, target_mask], dim=1)
        if student_input_ids.shape[1] > self.student_context_max_length:
            raise RuntimeError(
                "TRD student KL sequence exceeded its configured context: "
                f"sequence width {student_input_ids.shape[1]} > "
                f"max_length {self.student_context_max_length}."
            )

        if teacher_prompt_ids is None:
            source_teacher_prompts = source["teacher_prompts"].cpu()
            source_teacher_mask = source["teacher_prompt_attention_mask"].cpu()
            teacher_prompt_sequences = [
                token_ids[token_mask.bool()].tolist()
                for token_ids, token_mask in zip(
                    source_teacher_prompts, source_teacher_mask, strict=True
                )
            ]
            teacher_prompts, teacher_prompt_mask = self._pad_token_lists(
                teacher_prompt_sequences, padding_side="left"
            )
        else:
            teacher_prompts, teacher_prompt_mask = self._pad_token_lists(
                teacher_prompt_ids, padding_side="left"
            )
        teacher_prompt_width = teacher_prompts.shape[1]
        teacher_input_ids = torch.cat([teacher_prompts, target_tensor], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, target_mask], dim=1)
        if teacher_input_ids.shape[1] > self.refinement_vllm_max_model_len:
            raise RuntimeError(
                "TRD teacher KL sequence exceeded the refinement context: "
                f"sequence width {teacher_input_ids.shape[1]} > "
                f"refinement_vllm_max_model_len {self.refinement_vllm_max_model_len}."
            )

        labels = torch.full_like(student_input_ids, -100)
        target_labels = target_tensor.clone()
        target_labels[target_mask == 0] = -100
        labels[:, student_prompt_width:] = target_labels

        return {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "student_prompt_length": student_prompt_width,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_prompt_length": teacher_prompt_width,
            "labels": labels,
        }

    def _build_vanilla_distillation_batch(
        self, rollout: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Rebuild the legacy y_o batch without changing its token layout."""
        source = rollout["inputs"]
        generated_ids = rollout["generated_ids"].cpu()
        generated_attention_mask = rollout["generated_attention_mask"].cpu()
        student_prompt_width = int(rollout["generated_prompt_width"])
        generation_ids = generated_ids[:, student_prompt_width:]

        teacher_prompts = source["teacher_prompts"].cpu()
        teacher_prompt_width = teacher_prompts.shape[1]
        teacher_input_ids = torch.cat([teacher_prompts, generation_ids], dim=1)
        teacher_attention_mask = torch.ones_like(teacher_input_ids)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is not None:
            teacher_attention_mask[teacher_input_ids == pad_token_id] = 0

        labels = generated_ids.clone()
        prompt_lengths = source["student_prompt_lengths_per_example"]
        for row in range(labels.shape[0]):
            labels[row, : int(prompt_lengths[row].item())] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return {
            "student_input_ids": generated_ids,
            "student_attention_mask": generated_attention_mask,
            "student_prompt_length": student_prompt_width,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_prompt_length": teacher_prompt_width,
            "labels": labels,
        }

    def _generate_refined_completions(
        self,
        rollout: dict[str, Any],
        rollout_offset: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        """Generate y_r on rank 0 and scatter results by stable request ID."""
        rank = self.accelerator.process_index
        inputs = rollout["inputs"]
        try:
            prompts = [
                build_refinement_prompt(
                    tokenizer=self.processing_class,
                    alg=self.alg,
                    problem=problem,
                    initial_response=initial_response,
                    reference_solution=reference_solution,
                    teacher_thinking=self.teacher_thinking,
                    max_model_len=self.refinement_vllm_max_model_len,
                    max_refinement_length=self.max_refinement_length,
                )
                for problem, initial_response, reference_solution in zip(
                    inputs["problems"],
                    rollout["original_completion_texts"],
                    inputs["reference_solutions"],
                    strict=True,
                )
            ]
            local_records = [
                {
                    "request_id": (
                        self.state.global_step,
                        rollout_offset,
                        rank,
                        local_index,
                    ),
                    "prompt": prompt,
                }
                for local_index, prompt in enumerate(prompts)
            ]
            local_status = {"ok": True, "rank": rank, "records": local_records}
        except Exception as exc:
            prompts = []
            local_records = []
            local_status = {"ok": False, "rank": rank, "error": repr(exc), "records": []}

        statuses = gather_object([local_status])
        response_payload = None
        if self.accelerator.is_main_process:
            try:
                failures = [status for status in statuses if not status["ok"]]
                if failures:
                    raise RuntimeError(f"Failed to build refinement prompts: {failures}")
                all_records = sorted(
                    [record for status in statuses for record in status["records"]],
                    key=lambda record: record["request_id"],
                )
                generation = self.refinement_vllm_client.generate(
                    prompts=[record["prompt"] for record in all_records],
                    n=1,
                    repetition_penalty=getattr(self.args, "repetition_penalty", 1.0),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.args.top_k if self.args.top_k > 0 else -1,
                    min_p=getattr(self.args, "min_p", 0.0),
                    max_tokens=self.max_refinement_length,
                    presence_penalty=getattr(self.args, "presence_penalty", 0.0),
                )
                results = {}
                for record, prompt_ids, completion_ids in zip(
                    all_records,
                    generation.prompt_ids,
                    generation.completion_ids,
                    strict=True,
                ):
                    results[record["request_id"]] = {
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion_ids,
                    }
                response_payload = {"ok": True, "results": results}
            except Exception as exc:
                response_payload = {"ok": False, "error": repr(exc)}

        payload_holder = [response_payload]
        broadcast_object_list(payload_holder, from_process=0)
        response_payload = payload_holder[0]
        if not response_payload["ok"]:
            raise RuntimeError(f"Teacher refinement failed: {response_payload['error']}")

        try:
            expected_local_ids = {record["request_id"] for record in local_records}
            local_results = {
                request_id: result
                for request_id, result in response_payload["results"].items()
                if request_id in expected_local_ids
            }
            teacher_prompt_ids, refined_completion_ids, refined_texts = (
                self._validate_local_refinement_results(local_records, local_results)
            )
            validation_status = {"ok": True, "rank": rank}
        except Exception as exc:
            teacher_prompt_ids, refined_completion_ids, refined_texts = [], [], []
            validation_status = {"ok": False, "rank": rank, "error": repr(exc)}

        # A tokenizer mismatch can be data/rank specific.  Never let one rank
        # raise while its peers continue into the next gather/backward.
        validation_statuses = gather_object([validation_status])
        validation_payload = None
        if self.accelerator.is_main_process:
            failures = [status for status in validation_statuses if not status["ok"]]
            validation_payload = {
                "ok": not failures,
                "error": f"Local refinement result validation failed: {failures}" if failures else None,
            }
        validation_holder = [validation_payload]
        broadcast_object_list(validation_holder, from_process=0)
        validation_payload = validation_holder[0]
        if not validation_payload["ok"]:
            raise RuntimeError(f"Teacher refinement failed: {validation_payload['error']}")

        return teacher_prompt_ids, refined_completion_ids, refined_texts

    def _validate_local_refinement_results(
        self,
        local_records: list[dict[str, Any]],
        results: dict[tuple[int, int, int, int], dict[str, list[int]]],
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        """Validate and decode this rank's refinement results without collectives."""
        teacher_prompt_ids = []
        refined_completion_ids = []
        vocab_size = len(self.processing_class)
        expected_ids = [record["request_id"] for record in local_records]
        if len(set(expected_ids)) != len(expected_ids):
            raise RuntimeError("Local refinement request IDs must be unique.")
        if set(results) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(results))
            unexpected = sorted(set(results) - set(expected_ids))
            raise RuntimeError(
                "Local refinement result IDs do not match requests: "
                f"missing={missing}, unexpected={unexpected}."
            )
        for record in local_records:
            result = results.get(record["request_id"])
            # Exact set equality above makes this lookup total while keeping a
            # defensive check useful for custom mapping implementations.
            if result is None:
                raise RuntimeError(f"Missing refinement result for {record['request_id']}.")
            local_prompt_ids = self.processing_class.encode(
                record["prompt"], add_special_tokens=False
            )
            if local_prompt_ids != result["prompt_ids"]:
                raise RuntimeError(
                    "Teacher vLLM prompt IDs differ from the local tokenizer; "
                    "the endpoint model/tokenizer or prompt rendering is incorrect."
                )
            if any(token_id >= vocab_size for token_id in result["completion_ids"]):
                raise RuntimeError(
                    "Teacher vLLM returned a completion token outside the student/teacher shared "
                    f"vocabulary of size {vocab_size}."
                )
            teacher_prompt_ids.append(result["prompt_ids"])
            refined_completion_ids.append(result["completion_ids"])

        refined_texts = self.processing_class.batch_decode(
            refined_completion_ids, skip_special_tokens=True
        )
        return teacher_prompt_ids, refined_completion_ids, list(refined_texts)

    def _raise_if_any_rank_failed(self, stage: str, error: Exception | None) -> None:
        """Turn a rank-local pre-forward failure into one symmetric distributed failure."""
        local_status = {
            "ok": error is None,
            "rank": self.accelerator.process_index,
            "error": None if error is None else repr(error),
        }
        statuses = gather_object([local_status])
        failures = [status for status in statuses if not status["ok"]]
        if failures:
            raise RuntimeError(f"{stage} failed on one or more ranks: {failures}")

    def _record_rollout_outputs(
        self,
        rollout: dict[str, Any],
        refined_texts: list[str] | None,
    ) -> None:
        target_texts = refined_texts or rollout["original_completion_texts"]
        gathered_prompts = gather_object(list(rollout["prompt_texts"]))
        gathered_original = gather_object(list(rollout["original_completion_texts"]))
        gathered_target = gather_object(list(target_texts))
        self._textual_logs["prompt"].extend(gathered_prompts)
        self._textual_logs["completion"].extend(gathered_target)
        if self.accelerator.is_main_process:
            for prompt, original, target in zip(
                gathered_prompts, gathered_original, gathered_target, strict=True
            ):
                self._generation_outputs_buffer.append(
                    {
                        "policy_update": self.state.global_step + 1,
                        "rollout_step": rollout["rollout_step"],
                        "prompt": prompt,
                        "original_completion": original,
                        "refined_completion": target if self.teacher_refine else None,
                        "training_completion": target,
                    }
                )

    def _windowed_training_step(self, model, inputs) -> torch.Tensor:
        raw_batches = inputs.get("rollout_batches")
        if not isinstance(raw_batches, list) or len(raw_batches) != self.rollouts_per_update:
            raise ValueError(
                f"Expected {self.rollouts_per_update} rollout batches in a policy window."
            )

        self._ensure_student_vllm_current()
        rollouts = []
        # Phase A: collect every y_o before any teacher rewrite or backward.
        for offset, raw_batch in enumerate(raw_batches):
            rollout = None
            local_error = None
            try:
                prepared = self._prepare_inputs(raw_batch)
                rollout_step = self.state.global_step * self.rollouts_per_update + offset + 1
                with torch.no_grad():
                    rollout = self._collect_original_rollout(model, prepared, rollout_step)
                del prepared
            except Exception as exc:
                local_error = exc
            if rollout is None and local_error is None:
                local_error = RuntimeError("Student rollout returned no cache record.")
            self._raise_if_any_rank_failed(f"Student rollout {offset}", local_error)
            # The synchronized status above guarantees every rank has a value.
            assert rollout is not None
            rollouts.append(rollout)
            empty_cache()

        # Phase B: collect every y_r (or build vanilla y_o batches) on CPU.
        cached_training_batches = []
        for offset, rollout in enumerate(rollouts):
            cached_batch = None
            local_error = None
            try:
                if self.teacher_refine:
                    teacher_prompt_ids, refined_ids, refined_texts = self._generate_refined_completions(
                        rollout, offset
                    )
                    cached_batch = self._build_distillation_batch(
                        rollout,
                        target_ids=refined_ids,
                        teacher_prompt_ids=teacher_prompt_ids,
                    )
                else:
                    refined_texts = None
                    cached_batch = self._build_vanilla_distillation_batch(rollout)
            except Exception as exc:
                local_error = exc
                refined_texts = None
            if cached_batch is None and local_error is None:
                local_error = RuntimeError("Distillation builder returned no batch.")
            self._raise_if_any_rank_failed(f"Distillation batch {offset}", local_error)
            assert cached_batch is not None
            cached_training_batches.append(cached_batch)
            self._record_rollout_outputs(rollout, refined_texts)

        # Phase C: recompute logits and accumulate gradients only after the
        # complete window's trajectories are frozen in CPU memory.
        window_loss = torch.zeros((), device=self.args.device)
        previous_accumulation = self.current_gradient_accumulation_steps
        self.current_gradient_accumulation_steps = self.rollouts_per_update
        try:
            for offset, cached_batch in enumerate(cached_training_batches):
                is_last = offset == self.rollouts_per_update - 1
                self.accelerator.gradient_state._set_sync_gradients(is_last)
                if self.is_deepspeed_enabled:
                    if not hasattr(model, "set_gradient_accumulation_boundary"):
                        raise RuntimeError("DeepSpeed model lacks set_gradient_accumulation_boundary().")
                    model.set_gradient_accumulation_boundary(is_last)
                    sync_context = nullcontext()
                else:
                    sync_context = nullcontext() if is_last else self.accelerator.no_sync(model)
                with sync_context:
                    sub_loss = super().training_step(model, cached_batch, num_items_in_batch=None)
                window_loss += sub_loss
                empty_cache()
        finally:
            self.current_gradient_accumulation_steps = previous_accumulation
            self.accelerator.gradient_state._set_sync_gradients(True)
            if self.is_deepspeed_enabled and hasattr(model, "set_gradient_accumulation_boundary"):
                model.set_gradient_accumulation_boundary(True)

        loss_scalar = float(window_loss.detach())
        self._on_policy_loss_total += loss_scalar
        self._on_policy_step_equiv += 1.0
        return window_loss

    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        # Create generations directory in output_dir
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        # Clear buffer after saving
        self._generation_outputs_buffer.clear()

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step with self-distillation.

        If reason_first=True:
        1. Generate teacher's reasoning about the solution
        2. Append reasoning to teacher prompt
        3. Generate completions from student prompts
        4. Compute JSD loss

        Otherwise:
        1. Generate completions from student prompts
        2. Construct full sequences for both student and teacher with the generation
        3. Compute JSD loss on the generation tokens
        """
        if self.windowed_policy_updates:
            return self._windowed_training_step(model, inputs)

        on_policy = True

        # === REASONING PHASE (if enabled) ===
        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'='*80}\n")

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # Generate teacher's reasoning
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                # Decode reasoning
                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                # Occasionally print reasoning
                if random.random() < 0.01:
                    print(f"\n{'='*80}")
                    print(f"TEACHER REASONING SAMPLE (Step {self.state.global_step}):")
                    print(f"{'='*80}")
                    sample_idx = random.randint(0, len(reasoning_texts) - 1)
                    print(f"\n{'='*80}")
                    # Decode the prompt from token IDs to text
                    sample_prompt = self.processing_class.decode(
                        inputs["teacher_reasoning_prompts"][sample_idx], skip_special_tokens=False
                    )
                    print(f"PROMPT:\n{sample_prompt}")
                    print(f"\nReasoning:\n{reasoning_texts[sample_idx]}")
                    print(f"{'='*80}\n")

                # Update teacher prompts with reasoning
                # Construct: [teacher_reasoning_prompt][reasoning][transition_to_teaching]
                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                # Update inputs with new teacher prompts
                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        # === GENERATION PHASE ===
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                # Decode for logging
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        # Get batch-level student prompt length
        student_prompt_len = inputs["student_prompt_length"]

        # Extract generation part (same slice for all examples since prompts are padded)
        generation_ids = generated_ids[:, student_prompt_len:]

        # Construct student full sequence: [student_prompt][generation]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        # Construct teacher full sequence: [teacher_prompt][generation]
        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

        # Create attention mask for teacher
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        # Create labels for generation tokens
        # Mask prompt tokens (use per-example lengths for accurate masking)
        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100  # Mask actual prompt

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        # Collect generation outputs for saving
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        # Occasionally print student's generation with 1% probability
        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"{'='*80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # Track on/off-policy loss statistics
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
