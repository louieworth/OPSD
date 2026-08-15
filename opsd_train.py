import os
import wandb

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field
from dataset_utils import load_local_parquet

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    alg: str = field(
        default="opsd",
        metadata={
            "help": "Distillation algorithm: 'opsd' (privileged self-teacher) or "
            "'opd' (external Qwen3-8B teacher)."
        },
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    train_dataset_path: str = field(
        default="data/train/openthoughts_math_30k_opsd",
        metadata={
            "help": "Repository-relative path to the prepared OpenThoughts Parquet dataset."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Approximate forward KL over the top-k tokens of the teacher distribution. The teacher is "
            "renormalized over those k tokens while the student retains its full-vocabulary softmax denominator, "
            "matching Eq. (10) of arXiv:2603.07079. Requires --beta 0. Set to 0 for full-vocabulary KL."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )
    policy_gradient_updates: int | None = field(
        default=None,
        metadata={
            "help": "Number of optimizer/policy updates to perform across --max_steps rollout microbatches. "
            "When omitted, the legacy Trainer max_steps semantics are preserved."
        },
    )
    teacher_refine: bool = field(
        default=False,
        metadata={
            "help": "Generate a teacher rewrite y_r for every student rollout y_o and train on y_r (TRD)."
        },
    )
    max_refinement_length: int | None = field(
        default=None,
        metadata={"help": "Maximum y_r tokens. Defaults to --max_completion_length."},
    )
    distillation_temperature: float | None = field(
        default=None,
        metadata={
            "help": "Temperature used only by the distillation loss. Defaults to --temperature for legacy runs; "
            "TRD requires 1.0."
        },
    )
    refinement_vllm_server_host: str = field(
        default="127.0.0.1",
        metadata={"help": "Host of the fixed teacher vLLM service used to generate y_r."},
    )
    refinement_vllm_server_port: int = field(
        default=8002,
        metadata={"help": "Port of the fixed teacher vLLM service used to generate y_r."},
    )
    refinement_vllm_connect_timeout: float = field(
        default=10.0,
        metadata={"help": "Teacher vLLM HTTP connection timeout in seconds."},
    )
    refinement_vllm_request_timeout: float = field(
        default=1800.0,
        metadata={"help": "Teacher vLLM generation read timeout in seconds."},
    )
    refinement_vllm_max_model_len: int = field(
        default=20000,
        metadata={"help": "Context limit configured on the teacher vLLM service."},
    )


def validate_algorithm_config(script_args, training_args, model_args) -> None:
    """Validate the model roles before allocating any model weights."""
    if script_args.alg not in {"opsd", "opd"}:
        raise ValueError(f"Unsupported --alg {script_args.alg!r}; expected 'opsd' or 'opd'.")

    if script_args.teacher_refine:
        if script_args.reason_first:
            raise ValueError("--teacher_refine and --reason_first are mutually exclusive.")
        if script_args.use_tinker_loss:
            raise ValueError("TRD Eq. (6) requires full-vocabulary forward KL; disable --use_tinker_loss.")
        if training_args.beta != 0:
            raise ValueError("TRD Eq. (6) requires --beta 0 (forward KL).")
        if script_args.top_k_loss != 0:
            raise ValueError("TRD Eq. (6) requires --top_k_loss 0 (full vocabulary).")
        if script_args.jsd_token_clip != 0:
            raise ValueError("TRD Eq. (6) requires --jsd_token_clip 0.")
        if script_args.distillation_temperature is None:
            script_args.distillation_temperature = 1.0
        elif script_args.distillation_temperature != 1.0:
            raise ValueError("TRD Eq. (6) requires --distillation_temperature 1.0.")
        if script_args.use_ema_teacher:
            raise ValueError("--teacher_refine uses a fixed step-0 teacher and is incompatible with EMA.")
        if script_args.max_refinement_length is None:
            script_args.max_refinement_length = training_args.max_completion_length
        if script_args.max_refinement_length <= 0:
            raise ValueError("--max_refinement_length must be positive.")
        if script_args.refinement_vllm_max_model_len <= script_args.max_refinement_length:
            raise ValueError(
                "--refinement_vllm_max_model_len must be larger than --max_refinement_length."
            )

    if script_args.distillation_temperature is None:
        script_args.distillation_temperature = training_args.temperature

    if script_args.top_k_loss < 0:
        raise ValueError("--top_k_loss must be non-negative; use 0 for full-vocabulary distillation.")
    if script_args.top_k_loss > 0 and training_args.beta != 0:
        raise ValueError("--top_k_loss implements top-k forward KL and therefore requires --beta 0.")

    if script_args.alg == "opsd":
        if training_args.teacher_model_name_or_path is not None:
            raise ValueError("--teacher_model_name_or_path is only valid with --alg opd.")
        if script_args.teacher_refine and not script_args.fixed_teacher:
            raise ValueError("OPSD teacher refinement requires --fixed_teacher (the step-0 base model).")
        return

    if script_args.reason_first:
        raise ValueError("--reason_first is incompatible with OPD because OPD does not use y*.")
    if script_args.use_ema_teacher:
        raise ValueError("--use_ema_teacher is only valid with --alg opsd.")
    if script_args.fixed_teacher:
        raise ValueError("--fixed_teacher is only valid with --alg opsd; the OPD teacher is always fixed.")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if training_args.teacher_model_name_or_path is None:
        training_args.teacher_model_name_or_path = os.path.join(repo_root, "models", "Qwen3-8B")

    config_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
    }
    student_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    teacher_config = AutoConfig.from_pretrained(training_args.teacher_model_name_or_path, **config_kwargs)

    student_signature = (
        student_config.model_type,
        student_config.hidden_size,
        student_config.num_hidden_layers,
    )
    allowed_students = {
        ("qwen3", 2048, 28): "Qwen3-1.7B",
        ("qwen3", 2560, 36): "Qwen3-4B",
    }
    if student_signature not in allowed_students:
        raise ValueError(
            "OPD supports only Qwen3-1.7B and Qwen3-4B students; got architecture "
            f"{student_signature}."
        )

    teacher_signature = (
        teacher_config.model_type,
        teacher_config.hidden_size,
        teacher_config.num_hidden_layers,
    )
    if teacher_signature != ("qwen3", 4096, 36):
        raise ValueError(
            "OPD requires a Qwen3-8B teacher; got architecture "
            f"{teacher_signature}."
        )
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("The OPD student and teacher must use the same vocabulary.")


def configure_policy_update_schedule(script_args, training_args) -> None:
    """Translate public rollout/update counts into Trainer outer-step semantics."""
    schedule_enabled = script_args.policy_gradient_updates is not None or script_args.teacher_refine
    if not schedule_enabled:
        training_args.windowed_policy_updates = False
        training_args.total_rollout_steps = None
        training_args.policy_gradient_updates = None
        training_args.rollouts_per_update = 1
        return

    total_rollout_steps = int(training_args.max_steps)
    if total_rollout_steps <= 0:
        raise ValueError("Windowed policy updates require an explicit --max_steps > 0.")

    policy_updates = (
        total_rollout_steps
        if script_args.policy_gradient_updates is None
        else int(script_args.policy_gradient_updates)
    )
    if not 1 <= policy_updates <= total_rollout_steps:
        raise ValueError(
            "--policy_gradient_updates must satisfy 1 <= U <= max_steps; "
            f"got U={policy_updates}, max_steps={total_rollout_steps}."
        )
    if total_rollout_steps % policy_updates != 0:
        raise ValueError(
            "--max_steps must be divisible by --policy_gradient_updates; "
            f"got {total_rollout_steps} % {policy_updates}."
        )
    if int(training_args.gradient_accumulation_steps) != 1:
        raise ValueError(
            "Windowed policy updates require --gradient_accumulation_steps 1; "
            "the window itself performs the requested accumulation."
        )
    if getattr(training_args, "ignore_data_skip", False):
        raise ValueError("Windowed policy updates require ignore_data_skip=False for exact resume semantics.")
    if training_args.use_vllm and training_args.vllm_sync_frequency != 1:
        raise ValueError("Windowed on-policy rollouts require --vllm_sync_frequency 1.")

    training_args.windowed_policy_updates = True
    training_args.total_rollout_steps = total_rollout_steps
    training_args.policy_gradient_updates = policy_updates
    training_args.rollouts_per_update = total_rollout_steps // policy_updates
    training_args.max_steps = policy_updates
    script_args.policy_gradient_updates = policy_updates


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    validate_algorithm_config(script_args, training_args, model_args)
    configure_policy_update_schedule(script_args, training_args)

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Report rollout and optimizer-window batch sizes separately. In legacy mode
    # effective_batch_size retains its historical gradient-accumulation meaning.
    rollout_batch_size = training_args.per_device_train_batch_size * num_processes
    if training_args.windowed_policy_updates:
        effective_batch_size = rollout_batch_size * training_args.rollouts_per_update
    else:
        effective_batch_size = rollout_batch_size * training_args.gradient_accumulation_steps

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"{script_args.alg}_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    if training_args.windowed_policy_updates:
        print(
            "Rollout/update schedule: "
            f"N={training_args.total_rollout_steps}, "
            f"U={training_args.policy_gradient_updates}, "
            f"R={training_args.rollouts_per_update}"
        )
        print(f"Rollout batch size: {rollout_batch_size}")
        print(f"Policy update batch size: {effective_batch_size}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "alg": script_args.alg,
                "teacher_model_name_or_path": training_args.teacher_model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "total_rollout_steps": training_args.total_rollout_steps,
                "policy_gradient_updates": training_args.policy_gradient_updates,
                "rollouts_per_update": training_args.rollouts_per_update,
                "rollout_batch_size": rollout_batch_size,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "teacher_refine": script_args.teacher_refine,
                "max_refinement_length": script_args.max_refinement_length,
                "distillation_temperature": script_args.distillation_temperature,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    if script_args.alg == "opd":
        training_args.teacher_model_init_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation or "flash_attention_2",
            torch_dtype=model_dtype,
            use_cache=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # OPSD loads reference solutions; OPD intentionally loads problems only.
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    train_dataset = load_local_parquet(
        script_args.train_dataset_path,
        columns=["problem", "solution"] if script_args.alg == "opsd" else ["problem"],
    )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        teacher_refine=script_args.teacher_refine,
        max_refinement_length=script_args.max_refinement_length,
        distillation_temperature=script_args.distillation_temperature,
        refinement_vllm_server_host=script_args.refinement_vllm_server_host,
        refinement_vllm_server_port=script_args.refinement_vllm_server_port,
        refinement_vllm_connect_timeout=script_args.refinement_vllm_connect_timeout,
        refinement_vllm_request_timeout=script_args.refinement_vllm_request_timeout,
        refinement_vllm_max_model_len=script_args.refinement_vllm_max_model_len,
        alg=script_args.alg,
        teacher_model=(
            training_args.teacher_model_name_or_path if script_args.alg == "opd" else None
        ),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.prepare_window_schedule(training_args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
