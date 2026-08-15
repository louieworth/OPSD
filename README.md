# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-Post-blue.svg"></a>
</p>

---
## Overview

**On-Policy Self-Distillation (OPSD)** trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.

This repository also supports **On-Policy Distillation (OPD)** through the same
training entry point. OPD uses Qwen3-8B as a fixed external teacher and scores
the student's on-policy completion with the same prompt and condition,
\(\pi_T(y\mid x)\), without exposing the ground-truth solution to the teacher.

It additionally implements the teacher-refinement stage from
**Trajectory-Refined Distillation (TRD)** ([paper](https://arxiv.org/abs/2606.08432),
[reference code](https://github.com/louieworth/trd)). The student first produces
an on-policy answer \(y_o\), a fixed step-0 teacher rewrites it as \(y_r\), and
the student is trained with full-vocabulary forward KL along \(y_r\). Both OPD
and OPSD conditioning are supported.


## Updates

- **Mar 18, 2026**: Released updated code. 

  (1) Fixed chat template and zero2 bugs (see [template issue](https://github.com/huggingface/trl/issues/5241)), we re-ran experiments with updated results (detailed results & ablations updated on arxiv/blog). The fixes yield improved OPSD performance, most notably on Qwen3-1.7B.

  (2) Added a new training stabilization strategy 🚀: per-token point-wise KL clipping. We find style tokens (such as 'wait', 'think') can exhibit 6–15× higher KL divergence than math-related tokens, and dominates the training signal. Clipping stablizes training and improves performance.


-  **Mar 3, 2026**: Initial code release.

## Installation


```bash
bash scripts/setup_env.sh
conda activate "$PWD/.cache/conda/opsd"
```
If you encounter difficulties installing flash-attn, you can check the version matching your CUDA and PyTorch versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

The code uses `trl`'s experimental GOLD trainer as a base.

## Portable 1-node reproduction

The scripts below are designed for a fresh GitHub checkout. Prepared datasets,
checkpoints, logs, and caches stay below the repository root. Base-model weights
remain in the Hugging Face cache and are exposed through links in `models/`.

When the target platform already provides the environment, prepare the pinned
training data, five evaluation datasets, and Hugging Face model links:

```bash
bash scripts/prepare_train_data.sh
bash scripts/prepare_eval_data.sh
bash scripts/prepare_models.sh
```

The combined data command is equivalent and prepares both scopes:

```bash
bash scripts/prepare_data.sh
```

All preparation commands are idempotent: existing files are validated against
their pinned revisions, row counts, and required fields before downloading.

Model preparation defaults to all three models. A single model can also be
selected with the positional model argument:

```bash
bash scripts/prepare_models.sh 1.7b
bash scripts/prepare_models.sh 4b
bash scripts/prepare_models.sh 8b
```

### Teacher source and loss variants

The two launcher directories identify only the **teacher source**:

- `scripts/OPSD/`: the teacher is the same step-0 student base model with
  privileged reference-solution context.
- `scripts/OPD/`: the teacher is a stronger fixed Qwen3-8B model without
  privileged reference-solution context.

The next directory level selects the student size. `1B` is the short launcher
name for the Qwen3-1.7B checkpoint:

| Directory | Student | Teacher source |
|---|---|---|
| `scripts/OPSD/1B/` | Qwen3-1.7B | Same Qwen3-1.7B step-0 base + privileged solution |
| `scripts/OPSD/4B/` | Qwen3-4B | Same Qwen3-4B step-0 base + privileged solution |
| `scripts/OPSD/8B/` | Qwen3-8B | Same Qwen3-8B step-0 base + privileged solution |
| `scripts/OPD/1B/` | Qwen3-1.7B | Fixed external Qwen3-8B, without privileged solution |
| `scripts/OPD/4B/` | Qwen3-4B | Fixed external Qwen3-8B, without privileged solution |

Every OPD/OPSD model directory exposes the same four distillation algorithms
and requires no model argument:

| Launcher | Distillation support | Pointwise clip | Training trajectory |
|---|---|---|---|
| `vanilla.sh` | Full vocabulary | Off | Student rollout \(y_o\) |
| `top_k.sh` | Teacher top-16 | Off | Student rollout \(y_o\) |
| `clip.sh` | Full vocabulary | 0.05 (OPSD 8B: 0.06) | Student rollout \(y_o\) |
| `trd.sh` | Full vocabulary | Off | Teacher rewrite \(y_r\) |

The three OPSD model directories additionally contain `sft.sh` and `grpo.sh`
as paper-comparison baselines. Their location does not give either baseline an
OPSD teacher; SFT uses reference trajectories and GRPO uses correctness reward.

For example:

```bash
bash scripts/OPSD/1B/vanilla.sh
bash scripts/OPSD/4B/top_k.sh
bash scripts/OPSD/8B/clip.sh
bash scripts/OPSD/1B/trd.sh

bash scripts/OPD/1B/vanilla.sh
bash scripts/OPD/4B/top_k.sh
bash scripts/OPD/4B/clip.sh
bash scripts/OPD/1B/trd.sh
```

`top_k.sh` uses loss-side `k=16`, the main value reported by
[Entropy-Aware On-Policy Distillation](https://arxiv.org/abs/2603.07079).
It implements the paper's top-k forward-KL component: the teacher is
renormalized on its top-16 tokens while the student keeps its full-vocabulary
softmax denominator. It is a controlled top-k forward-KL variant, not the full
entropy-gated EOPD objective.

`clip.sh` uses the vocabulary-entry pointwise clipping operator from
[Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734): 0.05 for 1.7B/4B
and 0.06 for OPSD 8B. Applying 0.05 to OPD is this repository's corresponding
external-teacher adaptation; the clipping paper evaluated the OPSD source.
The source-level form (`scripts/OPSD/clip.sh 1.7b`) remains available as a
generic compatibility interface; new runs should use the model-scoped paths
above.

All four variants deliberately share rollout sampling at temperature 1.1,
top-p 0.95, and generation `top_k=20`. Generation `top_k` controls how
trajectories are sampled and is independent of loss-side `top_k_loss`; therefore
`vanilla.sh`, `clip.sh`, and `trd.sh` have no **loss top-k** even though their
common rollout sampler still uses top-k sampling.

### Rollout/update cadence

Every launcher defaults to 100 rollout microbatches and 100 policy updates.
`DISTILL_MAX_STEPS=N` counts rollout microbatches and
`DISTILL_POLICY_GRADIENT_UPDATES=U` counts optimizer/policy updates. `N` must be
divisible by `U`; one update consumes a complete window of `N/U` rollout
microbatches, all generated before that window's backward pass.

```bash
# One update over all 100 rollout microbatches.
DISTILL_MAX_STEPS=100 DISTILL_POLICY_GRADIENT_UPDATES=1 \
    AUTO_EVAL=0 bash scripts/OPSD/1B/vanilla.sh

# One update per two rollout microbatches.
DISTILL_MAX_STEPS=100 DISTILL_POLICY_GRADIENT_UPDATES=50 \
    AUTO_EVAL=0 bash scripts/OPD/4B/top_k.sh

# One update after every rollout microbatch.
DISTILL_MAX_STEPS=100 DISTILL_POLICY_GRADIENT_UPDATES=100 \
    bash scripts/OPSD/4B/clip.sh
```

Checkpoint numbers are policy-update numbers. `DISTILL_SAVE_STEPS` must divide
`U`; the launcher derives the automatic evaluation checkpoint list from it.
Each source/variant/run writes to a separate evaluation namespace under
`outputs/eval/<source>/<variant>/<run-config>/`.

### Trajectory-Refined Distillation

`trd.sh` adds the teacher-refinement stage and explicitly disables loss top-k
and clipping. OPD refines with \(x+y_o\); OPSD refines with
\(x+y^\star+y_o\). Student KL is evaluated on \(x+y_r\), while the teacher
reuses the exact canonical refinement prefix followed by \(y_r\).

TRD uses a fixed 4/4 topology on one **8×H100-80GB** node. Trainer ranks and
their colocated student vLLM engines use GPUs 0–3; a tensor-parallel fixed
teacher service uses GPUs 4–7. Both sources default \(y_o\) and \(y_r\) to 1,024
tokens, but their context budgets are source-specific: OPSD allows 20,000-token
student and refinement prompts and therefore uses 21,024-token student/refiner
limits; OPD retains 18,976-token prompts and 20,000-token total limits. These
budgets are an H100-80GB assumption, not a supported capacity claim for 40GB
A100 GPUs. For OPD, each trainer rank also retains the frozen Qwen3-8B
Transformers teacher needed for full-vocabulary KL; the TP=4 service supplies
refinement generation only.

The service starts on port 8002, is health/world-size checked, monitored during
training, cleaned up on exit or signal, and stopped before optional all-eight-GPU
evaluation. Useful overrides include `TRD_REFINEMENT_HOST`,
`TRD_REFINEMENT_PORT`, `TRD_REFINEMENT_MAX_MODEL_LEN`,
`TRD_SERVER_STARTUP_TIMEOUT`, `TRD_REFINEMENT_REQUEST_TIMEOUT`, and
`DISTILL_MAX_REFINEMENT_LENGTH`. Override the student total length with
`TRD_MAX_LENGTH` and the refiner total length with
`TRD_REFINEMENT_MAX_MODEL_LEN`; response budgets use
`DISTILL_MAX_COMPLETION_LENGTH` and `DISTILL_MAX_REFINEMENT_LENGTH` (with their
`TRD_MAX_COMPLETION_LENGTH` and `TRD_MAX_REFINEMENT_LENGTH` fallbacks). This
remains a fixed-step-0 TRD adaptation on the repository's current Qwen3
backbones and thinking-mode choices.

After training exits successfully, each canonical launcher automatically starts
its matching five-dataset thinking-mode evaluation. Set `AUTO_EVAL=0` to train
without post-training evaluation. Evaluation scope can be adjusted with
`CHECKPOINTS`, `DATASETS`, and `VAL_N`.

Evaluate checkpoints 25/50/75/100 on AIME24, AIME25, BeyondAIME, HMMT25,
and AMO-Bench:

```bash
bash scripts/run_eval.sh 1.7b
bash scripts/run_eval.sh 4b
bash scripts/run_eval.sh 8b
```

Formal evaluation uses thinking mode, 12 samples per problem, temperature 1.0,
top-p 0.95, disabled top-k, and 38,912 maximum new tokens. Detailed generations
and aggregate `summary.json`/`summary.csv` files are written under
`outputs/eval/<model>/`. `Pass@12` is the fraction of problems with at least one
correct sample; `Avg@12` is accuracy over all 12 samples per problem.
After all checkpoints finish, `best_by_dataset.json` and
`best_by_dataset.csv` select the highest-Avg@12 checkpoint independently for
each dataset and report the macro average, matching the paper's Table 2
reporting convention.

Long runs are resumable: completed result files are skipped. The evaluation
scope can be narrowed without editing a script, for example:

```bash
CHECKPOINTS="25 100" DATASETS="aime24 beyond-aime" \
    bash scripts/run_eval.sh 8b
```

Set `HF_HOME` before model preparation if the target platform keeps its model
cache on a dedicated volume. The resulting `models/Qwen3-*` links are checked
before any download is attempted.

The pinned PyTorch build uses CUDA 12.8. The NVIDIA driver may be newer, but
the `nvcc` toolkit used to build DeepSpeed/FlashAttention extensions should be
CUDA 12.8 as well; `setup_env.sh` prints a warning when it detects a mismatch.

## Repository Structure

```
├── opsd_trainer.py          # OPSDTrainer: core self-distillation trainer
├── data_collator.py         # Data collator for self-distillation
├── opsd_train.py            # OPSD training entry point
├── sft_train.py             # SFT baseline training entry point
├── grpo_train.py            # GRPO baseline training entry point
├── accelerate.yaml          # Accelerate config (multi-GPU)
├── scripts/
│   ├── prepare_all.sh       # Pinned local data + HF model links
│   ├── OPSD/               # Privileged same-base teacher source
│   │   ├── 1B/             # Qwen3-1.7B; distillation + SFT/GRPO baselines
│   │   ├── 4B/             # Qwen3-4B; distillation + SFT/GRPO baselines
│   │   └── 8B/             # Qwen3-8B; distillation + SFT/GRPO baselines
│   ├── OPD/                # Fixed stronger Qwen3-8B teacher source
│   │   ├── 1B/             # Qwen3-1.7B student; vanilla/top_k/clip/trd
│   │   └── 4B/             # Qwen3-4B student; vanilla/top_k/clip/trd
│   ├── lib/                # Shared distillation, baseline, and TRD helpers
│   ├── prepare_models.sh    # Prepare all models or one selected model
│   └── run_eval.sh          # Model-parameterized formal evaluation
└── eval/
    ├── evaluate_math.py     # Evaluation script (vLLM)
    ├── run_model_eval.sh    # Model/checkpoint/dataset matrix runner
    ├── summarize_results.py # JSON/CSV aggregate metrics
    └── run_eval.sh          # Example evaluation script
```

## Quick Start

Run the paper-style clipped OPSD configuration on Qwen3-1.7B:

```bash
bash scripts/OPSD/1B/clip.sh
```
Evaluation: (evaluation takes ~ 30-50 minutes on 4xh100 for each checkpoint) 
```bash
cd eval
bash run_eval.sh
```

### Evaluation Results across Tasks on Qwen3-1.7B

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 51.5% |
| 25 | 51.4% |
| 50 | 52.8% |
| 75 | 54.4% |
| 100 | 57.2% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 36.7% |
| 25 | 42.5% |
| 50 | 43.9% |
| 75 | 40.6% |
| 100 | 41.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 25 | 24.7% |
| 50 | 27.8% |
| 75 | 26.9% |
| 100 | 29.2% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, thinking mode enabled, max new tokens=38912, top-p=none, top-k disabled, min-p=0, presence penalty=0, num samples=12


## Non-Thinking Mode

OPSD can also run in non-thinking setting where both the Qwen student and teacher are enabled_thinking=False during training (`--student_thinking False --teacher_thinking False`) and evaluated with non-thinking inference (`--no_thinking`), with faster evaluation time than thinking mode.

The unified OPSD launchers intentionally use the main student-off/teacher-on
role configuration. The tables below retain the upstream both-nonthinking
ablation results; launch that ablation through `opsd_train.py` directly if
needed rather than treating it as one of the four loss/trajectory variants.

Evaluation:
```bash
cd eval
bash run_eval_nonthink.sh
```

### Evaluation Results with Non-Thinking Mode across Models

#### Qwen3-8B (`--jsd_token_clip 1e-7`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 26.4% |
| 50 | 49.7% |
| 75 | 45.3% |
| 100 | 38.3% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 19.7% |
| 50 | 35.0% |
| 75 | 26.9% |
| 100 | 27.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 18.3% |
| 75 | 17.5% |
| 100 | 15.3% |

</td>
</tr>
</table>
</div>

#### Qwen3-4B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 50 | 20.3% |
| 75 | 27.5% |
| 100 | 31.1% |
| 150 | 32.8% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 21.4% |
| 50 | 21.4% |
| 75 | 20.8% |
| 100 | 21.1% |
| 150 | 21.9% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 11.1% |
| 75 | 13.1% |
| 100 | 16.4% |
| 150 | 14.4% |

</td>
</tr>
</table>
</div>

#### Qwen3-1.7B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 11.9% |
| 50 | 15.0% |
| 75 | 13.9% |
| 100 | 12.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 9.2% |
| 50 | 6.2% |
| 75 | 8.3% |
| 100 | 8.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 5.0% |
| 25 | 7.2% |
| 50 | 5.8% |
| 75 | 5.0% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, non-thinking mode, num samples=12.



## Key distillation arguments

| Argument | Default | Description |
|---|---|---|
| `--alg` | `opsd` | Select `opsd` for the privileged self-teacher or `opd` for the fixed external Qwen3-8B teacher. OPD accepts only Qwen3-1.7B and Qwen3-4B students. |
| `--teacher_model_name_or_path` | `models/Qwen3-8B` for OPD | External teacher used by OPD. Its architecture is validated as Qwen3-8B. This argument is rejected for OPSD. |
| `--fixed_teacher` | `False` | Fix the teacher to the initial policy (step 0). Requires --use_peft. Note ❗ If you disable PEFT, the teacher will keep updating at every training step, which may make training unstable. Our main results use the fixed teacher, which is currently implemented with LoRA adapter weights. |
| `--use_tinker_loss` | `False` | Use sampled-token policy-gradient objective instead of full-vocabulary JSD. More memory efficient. Currently no clipped implemented for this variant, could be unstable. |
| `--max_completion_length` | — | Student generation length for distillation. We use 1024 in our main experiments. |
| `--beta` | — | Interpolation weight for the JSD mixture distribution. Beta=0 means forward KL and 1 means reverse KL. |
| `--top_k_loss` | `0` | Loss-side teacher support size. A positive value requires `beta=0`; only the teacher is renormalized on its top-k tokens while student probabilities retain the full-vocabulary denominator. |
| `--jsd_token_clip` | 0.05 | Cap every vocabulary-entry divergence contribution before summation. Set to `0` to disable it. Since negative contributions are not clipped, the summed loss can be negative. |
| `--policy_gradient_updates` | unset | Number of optimizer updates across `max_steps` rollout microbatches. When set, it must divide `max_steps` and requires gradient accumulation 1. |
| `--reason_first` | `False` | Prepend an explicit rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Custom name suffix for the output directory and WandB run. |

### SFT Baseline

The three model-scoped SFT launchers follow Table 7 of the paper: 100 optimizer
steps, effective batch size 32 on eight ranks, learning rate `5e-6`, LoRA
rank/alpha `64/128`, and a 16,000-token maximum sequence length.

```bash
bash scripts/OPSD/1B/sft.sh
bash scripts/OPSD/4B/sft.sh
bash scripts/OPSD/8B/sft.sh
```

### GRPO Baseline

The three model-scoped GRPO launchers follow Table 6 of the paper: 500 optimizer
steps, effective batch size 32 on eight ranks, eight generations per prompt,
learning rate `5e-6`, temperature `1.2`, zero reference-KL coefficient, LoRA
rank/alpha `64/128`, and a 16,000-token maximum completion length.

```bash
bash scripts/OPSD/1B/grpo.sh
bash scripts/OPSD/4B/grpo.sh
bash scripts/OPSD/8B/grpo.sh
```

### Acknowledgements
Our implementation builds on [TRL GOLD Trainer](https://huggingface.co/docs/trl/gold_trainer). We sincerely thank [@simran135](https://github.com/simran135) and [@beanie00](https://github.com/beanie00) for identifying the prompt template bugs and the zero-2 issue, respectively!

## Citation
If you find this useful, please consider citing:
```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
