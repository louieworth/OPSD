# Prepared Docker image for one-job-per-EC2

This is the preferred deployment when every submitted job launches an isolated
EC2 instance. Preparation happens once during `docker build`; job containers do
not download or prepare code datasets.

```text
one image build: prepare/fetch -> verify -> push immutable image to ECR
                                      |
                                      +--> EC2 A: pull layer -> run job
                                      +--> EC2 B: pull layer -> run job
                                      +--> EC2 N: pull layer -> run job
```

Every new EC2 must still receive the image bytes from ECR. To eliminate even
that transfer, create the launch AMI after pulling the immutable image so its
EBS-backed Docker storage is already populated.

## Recommended: bake the custom HF artifact once

First publish the prepared artifact from the OPSD repository with
`script_code/publish_artifacts.py`. It prints these immutable values:

```text
OPSD_ARTIFACT_REPO=YOUR_ORG/opsd-code-artifacts
OPSD_ARTIFACT_RELEASE=v1-...
OPSD_ARTIFACT_REVISION=<full HF commit SHA>
OPSD_GIT_REV=<full OPSD Git commit SHA>
```

Build the other repository's normal CUDA/job image first and use its immutable
digest as `BASE_IMAGE`. Keep an OPSD submodule checked out at `OPSD_GIT_REV`,
then run this once:

```bash
export HF_TOKEN='<read token used only by this image build>'

docker buildx build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg BASE_IMAGE='YOUR_ECR_BASE_IMAGE@sha256:...' \
  --build-arg OPSD_GIT_REV="$OPSD_GIT_REV" \
  --build-arg OPSD_ARTIFACT_REPO="$OPSD_ARTIFACT_REPO" \
  --build-arg OPSD_ARTIFACT_RELEASE="$OPSD_ARTIFACT_RELEASE" \
  --build-arg OPSD_ARTIFACT_REVISION="$OPSD_ARTIFACT_REVISION" \
  -f OPSD/deploy/docker/Dockerfile.prepared-from-hf \
  -t YOUR_ACCOUNT.dkr.ecr.YOUR_REGION.amazonaws.com/opsd-job-prepared:$OPSD_ARTIFACT_RELEASE \
  --push \
  OPSD

unset HF_TOKEN
```

`HF_TOKEN` is a BuildKit secret, not a build argument or environment variable,
and is not stored in the image. This single build is the only consumer of the
private HF artifact. Pin the ECR digest printed by Buildx in every job instead
of a mutable tag.

The build uses `Dockerfile.prepared-from-hf`, verifies all SHA-256 values,
installs the bundled evaluator source, generates `runtime.env`, removes download
caches in the same layer, and retains only the prepared runtime files.

## Alternative: run `prepare_code.sh all` in the image build

If the custom HF artifact repository is unnecessary, build directly from the
pinned upstream datasets:

```bash
export HF_TOKEN='<optional read token used only by this image build>'

docker buildx build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg BASE_IMAGE='YOUR_ECR_BASE_IMAGE@sha256:...' \
  --build-arg OPSD_GIT_REV="$OPSD_GIT_REV" \
  -f deploy/docker/Dockerfile.prepared-code \
  -t YOUR_ACCOUNT.dkr.ecr.YOUR_REGION.amazonaws.com/opsd-job-prepared:$OPSD_GIT_REV \
  --push \
  .

unset HF_TOKEN
```

Here `prepare_code.sh all` runs exactly once in a Docker build layer. No EC2 job
runs it again.

## EC2 job command

After authenticating Docker to ECR, run the image by immutable digest:

```bash
docker pull 'YOUR_ECR_PREPARED_IMAGE@sha256:...'

docker run --rm --gpus all --ipc=host \
  -v /mnt/opsd-models:/workspace/OPSD/models:ro \
  -v /mnt/opsd-outputs:/workspace/OPSD/outputs \
  'YOUR_ECR_PREPARED_IMAGE@sha256:...' \
  bash script_code/OPSD/1B/skd.sh
```

The inherited `baked_code_job_entrypoint.sh` runs only
`prepare_code.sh verify`, an offline integrity/dry-run check. It performs no HF
or GitHub requests, no TACO filtering, no evaluator installation, and no pip
operation.

`prepare_code.sh all` does not include Qwen weights. Either bake only the
required model(s) into a model-specific derived image, mount an EBS snapshot at
`/workspace/OPSD/models`, or pre-populate that path in the launch AMI. Avoid one
universal image containing all three models unless every job needs all 28 GB.

## Zero network transfer at job startup

An ECR-backed job still pulls missing image layers on a brand-new EC2. For zero
startup transfer:

1. Launch one EC2 using the same base AMI and root-volume layout as workers.
2. Pull the prepared image by digest and verify it with `docker image inspect`.
3. Stop/quiesce that instance and create an EBS-backed AMI.
4. Launch jobs from that AMI and run with `--pull=never`.

The AMI snapshots `/var/lib/docker`, so the prepared data layer is already on
each new instance. The same principle applies to a separate EBS snapshot for
large model weights.
