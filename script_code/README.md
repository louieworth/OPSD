# Code experiment suite

This directory contains the ready-to-run code matrix. Every trainable run starts
from a Qwen3 **base** checkpoint; SFT, GRPO, OPSD, and OPD do not depend on one
another's outputs.

## Experiment matrix

| Block | Models | Methods |
|---|---|---|
| Common baselines | Qwen3-1.7B / 4B / 8B | Base, SFT, GRPO |
| OPD | Qwen3-1.7B / 4B students; Qwen3-8B teacher | vanilla, clip, top_k, TRD, SKD |
| OPSD | Qwen3-1.7B / 4B / 8B | vanilla, clip, top_k, TRD, SKD |

There are exactly 34 model-scoped experiment launchers. The layout mirrors
`scripts/`: common baselines, OPD, and OPSD are the top-level blocks, and each
block is classified by model size:

```bash
script_code/Baselines/{1B,4B,8B}   # 3 methods x 3 models = 9
script_code/OPD/{1B,4B}             # 5 methods x 2 models = 10
script_code/OPSD/{1B,4B,8B}         # 5 methods x 3 models = 15
```

Every file fixes both the model and experiment, so it takes no model argument:

```bash
bash script_code/Baselines/1B/sft.sh
bash script_code/OPD/4B/trd.sh
bash script_code/OPSD/8B/top_k.sh
```

Use a dry run to inspect commands without loading a model:

```bash
bash script_code/run_matrix.sh --dry-run
```

`run_matrix.sh` runs jobs sequentially by model size. Within each model it runs
common baselines first, then all OPD methods, then all OPSD methods. Narrow a
real run explicitly when desired:

```bash
bash script_code/run_matrix.sh \
  --models 1.7b,4b \
  --methods vanilla,clip,top_k,trd,skd \
  --sources opd,opsd
```

## Locked comparison settings

- Dataset: pinned TACO train revision, deterministically filtered to 18,862
  rows with seed 42.
- Student prompt: non-thinking, at most 2,048 tokens.
- Completion/reference solution: at most 4,096 tokens.
- OPSD teacher: the same base checkpoint with LoRA disabled, privileged
  reference-solution prompt, thinking enabled during scoring.
- OPD teacher: Qwen3-8B base checkpoint, no reference solution, non-thinking.
- KD: 400 rollout steps / 100 policy updates, 8 GPUs, one example per GPU,
  four-rollout reuse window, effective update batch 32.
- Common KD sampler: temperature 1.1, top-p 0.95, top-k 20.
- SKD-only settings: draft length 5, acceptance top-k 25, correction
  temperature 0.2, correction top-p 1.0. SKD otherwise shares the KD sampler
  and optimization configuration.
- TRD: 1,024-token non-thinking student response `y_o`, followed by a
  1,024-token non-thinking rewrite `y_r`. It uses four trainer GPUs and a
  four-GPU fixed rewrite server. Subsequent KL scoring is thinking for OPSD
  and non-thinking for OPD.
- SFT: 100 updates. GRPO: 500 updates, group size 8, temperature 1.2,
  beta 0, group reward scaling.
- Code evaluation: Qwen3 non-thinking mode, 4,096 generated tokens, 12 samples,
  temperature 0.6 and top-p 0.95.

Here `400 / 100` means `N=400` distributed rollout micro-batches and `U=100`
optimizer steps, so `R=N/U=4` rollout batches are collected with the same
frozen policy before one gradient update. The trainer therefore reports global
steps 1--100, while the rollout counter reaches 400. For the ordinary 8-GPU KD
launchers with batch size 1, each optimizer step covers `4 x 8 = 32`
trajectories. TRD reserves four GPUs for its rewrite server, so its default
four-rank training side covers `4 x 4 = 16` trajectories per update.

The 6,144 student context is deliberate for code: 2,048 prompt + 4,096
completion. It replaces the previous 16K setting. OPSD's privileged teacher
has a separate 12,288-token cap because its prompt also contains the reference
solution; OPD uses 6,144.

## Evaluation

SFT trains for 100 steps and evaluates only checkpoint-100. All OPD/OPSD
launchers stop for evaluation every 25 optimizer updates. GRPO stops every 50
steps, so its 500-step run is evaluated at
50/100/.../500 (10 checkpoints). At each point the training process exits to
release all GPUs, the checkpoint is evaluated on HumanEval+, MBPP+, and
LiveCodeBench v6, and training resumes from that checkpoint. Checkpoints
include optimizer state because exact segmented continuation requires it.

`save_total_limit=1` replaces the previous checkpoint when the next segment
finishes, so at most one resumable checkpoint remains on disk. The final
checkpoint is deleted after its three evaluations succeed. Thus SFT and the
100-policy-update OPD/OPSD runs delete their last model after step 100;
GRPO must retain one intermediate checkpoint until step 500 because it still
needs that state to continue training. Evaluation JSON, completion markers,
and WandB logs remain. Base checkpoints under `models/` are never deleted.

Set `CODE_AUTO_EVAL=0` to disable automatic evaluation, or set
`CODE_DELETE_TRAINED_MODELS_AFTER_EVAL=0` to retain the final trained
checkpoint. `CODE_EVAL_EVERY_STEPS` controls the KD interval;
`CODE_SFT_EVAL_STEPS` defaults to 100 and `CODE_GRPO_EVAL_EVERY_STEPS`
defaults to 50. Successful evaluations write
small completion markers, so restarting a launcher continues from the single
remaining checkpoint instead of repeating finished evaluations.
If checkpoint retention was explicitly enabled, invoke the evaluation wrapper
directly on one checkpoint with:

```bash
bash script_code/eval/run_code_eval.sh \
  outputs/code/opsd/skd/code_opsd_qwen3-1.7b_skd_4k_n400_u100/checkpoint-25 \
  code_opsd_qwen3-1.7b_skd_checkpoint-25
```

The default suite is HumanEval+, MBPP+, and LiveCodeBench v6. Results are
merged into `outputs/code_eval/results.json`. Evaluation supports both full
models and PEFT adapters.

## Git synchronization and one-command preparation

Git contains the complete experiment code and the small math evaluation sets.
It intentionally does **not** contain training data, model weights,
LiveCodeBench v6, cloned evaluator repositories, caches, or outputs. After
pulling this repository on the 8xH100 server, activate the OPSD Python
environment and run:

```bash
git pull
conda activate opsd
bash script_code/prepare_code.sh all
```

That single command prepares and validates all Git-external train/eval
components:

- pinned OpenThoughts math train data (29,434 rows);
- pinned TACO, filtered once into the shared 18,862-row clean code artifact;
- pinned EvalPlus source plus HumanEval+ and MBPP+ data;
- the TRD-pinned LiveCodeBench evaluator with the local Qwen/PEFT patch;
- all six pinned LiveCodeBench v6 JSONL splits;
- `script_code/runtime.env`, including local evaluator paths and
  FlashAttention 2/SDPA selection;
- all 34 launcher commands through a no-model dry run.

The command is idempotent: validated datasets and pinned checkouts are reused.
It requires network access on the target server. To prepare only one side, use
`train`, `eval`, or `verify` instead of `all`.

Model weights are also intentionally outside Git. Prepare all three pinned
base checkpoints separately:

```bash
bash scripts/prepare_models.sh all
```

Or use the repository-wide wrapper to prepare models, train data, and code
evaluation components in one call:

```bash
bash scripts/prepare_all.sh
```

After preparation, launch directly; both training and evaluation wrappers
automatically source the generated `runtime.env`:

```bash
bash script_code/OPSD/1B/skd.sh
```

## Math TRD behavior

The shared math launcher now keeps rewrite generation separate from KL
scoring: both OPSD and OPD rewrite teachers are non-thinking with a 1,024-token
limit. The subsequent OPSD KL teacher prompt is independently rendered in
thinking mode; OPD KL remains non-thinking.
