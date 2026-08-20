# S3 fallback for EC2-per-job artifact deployment

The preferred one-job-per-EC2 path bakes the prepared artifact into an ECR
image; see `deploy/docker/README.md`. Use this S3 path when the prepared data
must remain outside the image.

The worker path is intentionally HF-token-free:

```text
prepare_code.sh all
        |
        v
immutable HF dataset release (source of truth, pinned commit)
        |
        v  one publisher/mirror operation
same-region S3 prefix
        |
        +----> EC2 job A local disk ----> Docker A
        +----> EC2 job B local disk ----> Docker B
        +----> EC2 job N local disk ----> Docker N
```

`max_workers=1`, jitter, and retry are available for a direct-HF fallback, but
they do not form a distributed rate limiter. S3 removes Hub requests from the
worker fleet and is the default production path.

## One-time publisher flow

Run on the prepared OPSD checkout:

```bash
bash script_code/prepare_code.sh all

export HF_TOKEN='<publisher write token>'
python script_code/publish_artifacts.py \
  --repo-id YOUR_ORG/opsd-code-artifacts
unset HF_TOKEN
```

Copy the printed `OPSD_ARTIFACT_*` values and `OPSD_GIT_REV` to the immutable
job configuration. Then mirror the local staged release once:

```bash
export OPSD_ARTIFACT_RELEASE='v1-...'
export OPSD_ARTIFACT_STAGE="$PWD/.cache/code_artifact_publish/$OPSD_ARTIFACT_RELEASE"
export OPSD_ARTIFACT_S3_URI='s3://YOUR_BUCKET/opsd'
bash deploy/aws/mirror_code_artifact_to_s3.sh
```

The upload token belongs only on this publisher. Never bake it into an image,
AMI, launch template, EC2 user-data, or job environment.

## EC2 bootstrap

The instance profile needs `s3:GetObject` and `s3:ListBucket` limited to the
artifact prefix. The relevant policy statements are:

```json
[
  {
    "Effect": "Allow",
    "Action": ["s3:ListBucket"],
    "Resource": "arn:aws:s3:::YOUR_BUCKET",
    "Condition": {
      "StringLike": {"s3:prefix": ["opsd/code-artifacts/*", "opsd/models/*"]}
    }
  },
  {
    "Effect": "Allow",
    "Action": ["s3:GetObject"],
    "Resource": [
      "arn:aws:s3:::YOUR_BUCKET/opsd/code-artifacts/*",
      "arn:aws:s3:::YOUR_BUCKET/opsd/models/*"
    ]
  }
]
```

Clone/update the small Git repository and fetch the artifact on the host:

```bash
mkdir -p /srv/opsd
export OPSD_GIT_REV='<publisher output: full commit SHA>'
if [[ -d /srv/opsd/repo/.git ]]; then
  git -C /srv/opsd/repo fetch origin "$OPSD_GIT_REV"
else
  git clone --filter=blob:none --no-checkout \
    https://github.com/louieworth/OPSD.git /srv/opsd/repo
fi
git -C /srv/opsd/repo checkout --detach "$OPSD_GIT_REV"

export OPSD_ARTIFACT_RELEASE='v1-...'
export OPSD_ARTIFACT_S3_URI='s3://YOUR_BUCKET/opsd'
export OPSD_EC2_ARTIFACT_ROOT='/srv/opsd'
bash /srv/opsd/repo/deploy/aws/ec2_fetch_code_artifact.sh
```

The script downloads a readiness marker last and uses `flock`, so concurrent
containers on one EC2 do not duplicate the S3 transfer. Separate EC2 instances
download from S3 independently and never share an HF account limit.

## Job image

In the other repository, either add OPSD as a pinned Git submodule named
`OPSD`, or clone a pinned OPSD commit during its image build. Adapt
`deploy/docker/Dockerfile.code-job.example` to the existing CUDA image. The
image must contain the Python packages in `environment.yml`, plus `git` and
`pip`; it must not contain `HF_TOKEN` or AWS keys.

When copying the source into the image, pass the same revision as build
metadata so the entrypoint can enforce that the code and artifact match:

```bash
docker build \
  --build-arg BASE_IMAGE=YOUR_EXISTING_IMAGE:TAG \
  --build-arg OPSD_GIT_REV="$OPSD_GIT_REV" \
  -f OPSD/deploy/docker/Dockerfile.code-job.example .
```

The EC2 command mounts the common host root once. Keeping the artifact and
checkout on the same filesystem lets `fetch_artifacts.py` hard-link large files
instead of copying them:

```bash
docker run --rm --gpus all --ipc=host \
  -v /srv/opsd:/workspace/opsd \
  -v /srv/opsd/models:/workspace/opsd/repo/models:ro \
  -e OPSD_ROOT=/workspace/opsd/repo \
  -e OPSD_ARTIFACT_SOURCE_DIR=/workspace/opsd/artifacts/$OPSD_ARTIFACT_RELEASE \
  YOUR_ACCOUNT.dkr.ecr.YOUR_REGION.amazonaws.com/YOUR_JOB_IMAGE:TAG \
  bash script_code/OPSD/1B/skd.sh
```

The entrypoint verifies every SHA-256, installs the files, recreates
container-specific `runtime.env`, installs the bundled pinned evaluator source,
and runs the offline preflight before starting the requested launcher.

## Base models

`prepare_code.sh all` does not include Qwen weights. Do not let every EC2 call
`prepare_models.sh`, because that recreates the same Hub fan-out for much larger
files. Mirror each needed pinned snapshot once:

```bash
aws s3 sync "$(readlink -f models/Qwen3-1.7B)/" \
  s3://YOUR_BUCKET/opsd/models/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/ \
  --only-show-errors --no-progress
```

At EC2 bootstrap, download the required model to `/srv/opsd/models/Qwen3-1.7B`
and mount the model root read-only over the checkout's `models/` directory:

```bash
mkdir -p /srv/opsd/models/Qwen3-1.7B
aws s3 sync \
  s3://YOUR_BUCKET/opsd/models/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/ \
  /srv/opsd/models/Qwen3-1.7B/ \
  --only-show-errors --no-progress

# Add this mount to docker run:
# -v /srv/opsd/models:/workspace/opsd/repo/models:ro
```

OPD jobs also need the pinned Qwen3-8B teacher. For a large number of short-lived
instances, place the S3-prefetched artifact and model directories in an EBS
snapshot and create one volume per job; this removes repeated 28 GB model
downloads while preserving EC2 isolation.
