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

## Git synchronization and automatic preparation

Git contains the complete experiment code, the 29,434-row OpenThoughts math
training artifact, and the small math evaluation sets. It intentionally does
**not** contain the prepared TACO code artifact, model weights, LiveCodeBench
v6, cloned evaluator repositories, caches, or outputs.

After pulling this repository on a worker, activate the OPSD Python environment
and launch the requested code experiment directly:

```bash
git pull
conda activate opsd
bash script_code/OPSD/1B/skd.sh
```

Every real launcher passes through the shared `code_common.sh` input check. It
first runs an offline `prepare_code.sh verify`; when verification fails, it
automatically runs `prepare_code.sh all` before starting the job. A repo-local
`flock` prevents duplicate preparation by concurrent launchers sharing the same
checkout. `CODE_DRY_RUN=1` never prepares or downloads anything. Set
`CODE_AUTO_PREPARE=0` only when the container entrypoint has already prepared
and verified all inputs.

Automatic preparation covers these Git-external train/eval components:

- pinned TACO, filtered once into the shared 18,862-row clean code artifact;
- pinned EvalPlus source plus HumanEval+ and MBPP+ data;
- the TRD-pinned LiveCodeBench evaluator with the local Qwen/PEFT patch;
- all six pinned LiveCodeBench v6 JSONL splits;
- `script_code/runtime.env`, including local evaluator paths and
  FlashAttention 2/SDPA selection;
- all 34 launcher commands through a no-model dry run.

Preparation is idempotent: validated datasets and pinned checkouts are reused.
The first real code run on a fresh worker requires network access. Manual
`bash script_code/prepare_code.sh all` remains available for image builds and
diagnostics; `train`, `eval`, and `verify` retain their scoped behavior.

## Optional prepared Docker image for isolated EC2 jobs

If job-start download time later becomes important, an alternative is to run
`prepare_code.sh all` once during `docker build`, push that immutable prepared
image to ECR, and run every job from its digest. Those job containers perform
only an offline `verify`; they do not download datasets or repeat TACO
filtering. See `deploy/docker/README.md` and
`deploy/docker/Dockerfile.prepared-code`.

If even an ECR layer pull is unacceptable at job startup, pull the image before
creating the EBS-backed launch AMI. The Docker storage layer will then already
exist on every new instance.

## Optional immutable HF artifact releases and S3 fallback

For fleets that later need centralized artifacts, prepare once, publish one
immutable release to a Hugging Face **dataset** repository, and pin both the
release name and the returned Hub commit:

```bash
bash script_code/prepare_code.sh all

export HF_TOKEN='<write token used only by the publisher>'
python script_code/publish_artifacts.py \
  --repo-id YOUR_ORG/opsd-code-artifacts
unset HF_TOKEN
```

The publisher runs the full preflight, hashes every file, stages hard links
below `.cache/code_artifact_publish/`, and uploads these components below
`releases/<content-id>/`:

- the OpenThoughts and clean TACO training artifacts;
- all six LiveCodeBench v6 data files;
- the pinned, patched EvalPlus and LiveCodeBench sources;
- the prefetched HumanEval+ and MBPP+ cache.

It intentionally excludes models, checkpoints, outputs, transient compiler
caches, nested Git metadata, and `runtime.env`. The last item is regenerated
inside each container because it contains absolute Python and repository paths.

The command prints four deployment values. Keep all four in the job
definition; never deploy from the floating `main` revision:

```text
OPSD_ARTIFACT_REPO=YOUR_ORG/opsd-code-artifacts
OPSD_ARTIFACT_RELEASE=v1-...
OPSD_ARTIFACT_REVISION=<full HF commit SHA>
OPSD_GIT_REV=<full OPSD Git commit SHA>
```

Direct HF download is supported and deliberately uses one worker, a
process-local file lock, bounded exponential backoff, and optional startup
jitter:

```bash
export OPSD_ARTIFACT_REPO='YOUR_ORG/opsd-code-artifacts'
export OPSD_ARTIFACT_RELEASE='v1-...'
export OPSD_ARTIFACT_REVISION='<full HF commit SHA>'
export OPSD_HF_MAX_WORKERS=1
export OPSD_HF_INITIAL_JITTER_SECONDS=600
python script_code/fetch_artifacts.py
bash script_code/prepare_code.sh artifact
```

This reduces bursts but cannot coordinate separate EC2 instances. When baking
the artifact into an ECR image is unsuitable, mirror the already-staged release
to S3 exactly once:

```bash
export OPSD_ARTIFACT_RELEASE='v1-...'
export OPSD_ARTIFACT_STAGE="$PWD/.cache/code_artifact_publish/$OPSD_ARTIFACT_RELEASE"
export OPSD_ARTIFACT_S3_URI='s3://YOUR_BUCKET/opsd'
bash deploy/aws/mirror_code_artifact_to_s3.sh
```

Grant worker instances `s3:GetObject` on only that prefix through an EC2 IAM
role. Do not put `HF_TOKEN` or long-lived AWS keys in the image or job
definition. Each EC2 bootstrap downloads from S3 into its instance-local
directory; `flock` prevents duplicate downloads if several containers land on
the same host:

```bash
export OPSD_ARTIFACT_RELEASE='v1-...'
export OPSD_ARTIFACT_S3_URI='s3://YOUR_BUCKET/opsd'
export OPSD_EC2_ARTIFACT_ROOT='/srv/opsd'
bash deploy/aws/ec2_fetch_code_artifact.sh
```

Mount the host root once so the artifact installer can use hard links instead
of storing a second copy, then start the job container:

```bash
docker run --rm --gpus all --ipc=host \
  -v /srv/opsd:/workspace/opsd \
  -v /srv/opsd/models:/workspace/opsd/repo/models:ro \
  -e OPSD_ROOT=/workspace/opsd/repo \
  -e OPSD_ARTIFACT_SOURCE_DIR=/workspace/opsd/artifacts/$OPSD_ARTIFACT_RELEASE \
  YOUR_ECR_IMAGE:TAG \
  bash script_code/OPSD/1B/skd.sh
```

The host checkout must be at `/srv/opsd/repo`. The image entrypoint template is
`deploy/docker/code_job_entrypoint.sh`; an example wrapper Dockerfile is at
`deploy/docker/Dockerfile.code-job.example`. With the S3 path, workers make no
HF requests and need no HF token. If S3 is not configured, the same entrypoint
can fall back to the three pinned HF environment variables above.

Base Qwen weights are not part of this artifact. A fleet must also avoid having
every EC2 call `prepare_models.sh`: mirror each pinned model snapshot once to
S3, bake it into an AMI/EBS snapshot, or provide a shared read-only model
volume. For short-lived GPU instances, an EBS snapshot or a same-region S3
mirror is preferable to repeated Hub downloads.

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
