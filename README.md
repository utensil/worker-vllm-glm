# worker-vllm-glm

[![Deploy GLM-5.3-Flash NVFP4 on RunPod](https://img.shields.io/badge/Deploy-GLM--5.3--Flash%20NVFP4-673AB7?style=for-the-badge)](https://console.runpod.io/deploy?template=yp08gkkdz2&ref=km0th85l)
[![Deploy GLM-5.3-Flash NVFP4 Serverless](https://img.shields.io/badge/Deploy-Serverless%20GLM--5.3--Flash-00A67E?style=for-the-badge)](https://console.runpod.io/deploy?template=nxl09wfv94&ref=km0th85l)

Public, secret-free vLLM worker images and RunPod templates for GLM models.

The current release serves
[`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
as an OpenAI-compatible API on one NVIDIA B300.

## Deploy the public Pod template

The public RunPod template is named:

```text
GLM-5.3-Flash-NVFP4 (Public vLLM 1xB300 TP1)
```

Select exactly **1 x NVIDIA B300 SXM6 AC**, expose port 8000, and allow roughly
16 minutes for a fresh direct-download cold start. The demonstrated baseline is
TP1, 8,192-token context, `max-num-seqs=2`, and a 300 GB container disk.

The checkpoint is public. No Hugging Face, GitHub, RunPod, Discord, or API
credential is stored in the image or template. Add `VLLM_API_KEY` when deploying
if the OpenAI-compatible server should require authentication.

To recreate the public template under your own RunPod account:

```bash
python3 -m pip install requests
python3 create_pod_template.py             # dry render, no mutation
export RUNPOD_API_KEY=...                   # never commit this value
python3 create_pod_template.py --create    # create/reuse + verify
```

## Image

```text
ghcr.io/utensil/worker-vllm-glm@sha256:8a025ed9775fd57dcd66ec11d80296401e2609afcda483ed64345bc4a3328816
```

The public template pins this exact image digest. The mutable convenience tag is
`glm-5.3-flash-nvfp4-v1`; deployment should use the digest. The Dockerfile
derives from a pinned vLLM amd64 digest. Its pinned GLM5
processor implementation incorrectly opens `processor_config.json` as a local
path when the model is a Hugging Face repository. The image applies the tested
Hub-aware `cached_file` lookup at build time. The patch asserts one exact source
match and fails the build if the upstream image changes.

The model revision is pinned to:

```text
36c184c6cda000a481711306df5adde42f63321a
```

## API smoke

After the Pod becomes healthy:

```bash
curl "https://<POD_ID>-8000.proxy.runpod.net/v1/models"

curl "https://<POD_ID>-8000.proxy.runpod.net/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/GLM-5.3-Flash-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32
  }'
```

If `VLLM_API_KEY` is set, include `Authorization: Bearer <YOUR_KEY>`.

## Claim boundary

The configuration has demonstrated checkpoint load, health, model listing, and
one basic chat completion on one B300. It is not yet a full quality, multimodal,
tool-use, long-context, concurrency, cost-efficiency, or optimized-production
evaluation.

This repository publishes the proven **Pod** image/template and a separately
reviewed Serverless queue-worker image. The Serverless image is pinned at:

```text
ghcr.io/utensil/worker-vllm-glm@sha256:3f31ef919c98ad370f4bfdddafd6f20ae92f1208fc96616bb1e9cbd877e09c09
```

The Serverless endpoint cold start and request paths remain **unvalidated until
the independent live gates below pass**.

## Public Serverless template (endpoint not yet live-validated)

The public template is named `GLM-5.3-Flash-NVFP4 (Public Serverless 1xB300
TP1)` and currently has template ID `nxl09wfv94`. Template IDs can change if a
public template is replaced; prefer the name when auditing an account.

The `serverless` Docker target starts the same pinned GLM vLLM server on
loopback, waits for `/health`, and only then registers its RunPod queue handler.
It supports these OpenAI routes, including SSE for chat/text completions:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/models`

The handler accepts RunPod's `openai_route` / `openai_input` passthrough and the
native `messages` or `prompt` shorthand. Routes, methods, body size, model name,
backend liveness, startup time, and request time are checked fail-closed. SSE
chunks are forwarded for streaming jobs. Tool definitions and remote image-URL
message content pass through without being rewritten; their model behavior is
not claimed until the live acceptance gate. The handler is an async generator
and uses `aiohttp`, so the configured two in-flight jobs do not block each
other's event-loop progress. A post-health watchdog terminates the worker if
the owned vLLM child exits after queue registration.

The process ownership and queue-proxy pattern is grounded in official RunPod
[`worker-vllm` commit `84a6b5e8`](https://github.com/runpod-workers/worker-vllm/commit/84a6b5e8fc9f7beb4def70b823706262e047b315).
This repository does **not** inherit that worker's vLLM 0.28 base. Both Docker
targets continue from this repository's GLM-specific digest-pinned base and
fail-closed processor patch.

The desired hardware remains exactly **1 x NVIDIA B300 SXM6 AC at TP1**. RunPod
Serverless endpoint APIs select GPU pools, which can contain multiple GPU types.
The template itself does not select hardware, so template publication/readback
is independent of current GPU stock. Before any endpoint action,
`create_serverless_template.py --check-gpu` reads the current Serverless GPU
catalog and requires the exact B300 type to have a Serverless pool, price,
non-`NONE` availability, and an available CUDA 13.0 placement. It constructs
exclusions for every other type in that pool and pins CUDA 13.0. It never
substitutes B200, changes to TP2, or creates an endpoint.

**Current endpoint blocker (2026-08-29 catalog readback):** exact
`NVIDIA B300 SXM6 AC` has `pool=null`, no Serverless price, and availability
`NONE`; CUDA 13.0 and 13.2 are listed but unavailable. The public template can
still be published after its image digest is pinned, but exact-B300 live
validation cannot start until the mandatory read-only gate passes.

The public template candidate is volume-free with a 300 GB ephemeral container
disk. This preserves maximum placement availability and matches the proved
direct-download path. A network volume or RunPod cached model should be adopted
only after a measured cold-start comparison; a volume adds storage cost and
restricts workers to its data center.

Both initialization gates are set to 1,800 seconds, above the measured 922.7s
Pod cold start:

```text
RUNPOD_INIT_TIMEOUT=1800
VLLM_STARTUP_TIMEOUT=1800
```

The backend request timeout is 600 seconds. Endpoint policy is separate from a
template: use job TTL of at least 3,600 seconds so provisioning and model load
fit. The first bounded validation must use a temporary endpoint with
`workersMin=0`, `workersMax=1`, queue-delay scaling at 1 second, FlashBoot,
300-second idle timeout, and REST v2 `timeout` of 1,800,000 ms. The renderer
validates the exact allowed top-level and nested key sets before printing it.

Local, no-spend checks:

```bash
python3 -m pip install requests -r serverless/requirements.txt
python3 -m unittest discover -v
python3 create_serverless_template.py
docker build --target serverless -t worker-vllm-glm:serverless-candidate .
```

`create_serverless_template.py --create` is deliberately locked until CI has
published the separate Serverless image and the script's image constant is
replaced with its immutable digest. Once pinned, `--create` creates or reuses
the public Serverless template and reads back every persisted field. It does
not require GPU stock, create an endpoint, or start compute.

Endpoint preflight is separate and mandatory:

```bash
export RUNPOD_API_KEY=...  # keep this out of shell history and source control
python3 create_serverless_template.py --check-gpu
python3 create_serverless_template.py \
  --render-temporary-endpoint PUBLIC_TEMPLATE_ID
```

The second command is render-only. It carries the admitted pool, exclusions of
all other current pool members, one GPU, and CUDA 13.0 directly into the bounded
endpoint payload. Both commands currently fail closed at the exact-B300 gate.

Live acceptance requires separate approval and evidence: anonymous digest pull;
public template readback; a temporary endpoint with `workersMin=0`; one bounded
cold start, model listing, chat completion, and negative request; endpoint
deletion; and a fresh absence check. Until then, this repository makes no
Serverless deployment, quality, latency, throughput, or cost claim.

## 2xB200 TP2 Serverless feasibility candidate

`create_serverless_b200_template.py` defines a distinct public candidate named
`GLM-5.3-Flash-NVFP4 (Public Serverless 2xB200 TP2)`. It reuses the same
reviewed queue worker while setting `TENSOR_PARALLEL_SIZE=2`; the B300 template
continues to omit that variable and therefore retains the worker's TP1 default.

The B200 endpoint renderer fails closed unless a live catalog query for exactly
two `NVIDIA B200` GPUs admits CUDA 13.0, a Serverless pool, a positive finite
price, and `LOW`, `MEDIUM`, or `HIGH` availability. It pins count 2, excludes
every other type in the selected pool, permits only one worker, and never falls
back to another GPU count or type. TP2 remains unproven until its bounded live
cold-start, model-list, and chat checks pass.

No-spend render and exact live-catalog check:

```bash
python3 create_serverless_b200_template.py
python3 create_serverless_b200_template.py --check-gpu
```

Template creation remains digest-locked. After CI publishes the B200-tagged
Serverless image and `IMAGE` is replaced by that immutable digest,
`--create` creates or reuses only the distinct B200 template and verifies every
persisted field. It does not alter the existing B300 template or create an
endpoint.

## Build

GitHub Actions tests every change. Image-source changes build the default `pod`
target and the separate `serverless` target for `linux/amd64`. Pod tags remain
`latest` and `glm-5.3-flash-nvfp4-v1`; the Serverless target uses
`glm-5.3-flash-nvfp4-serverless-v1` and
`glm-5.3-flash-nvfp4-serverless-b200-tp2-v1`. Deployment uses immutable digests. The
repository and package must remain public so RunPod can pull either image
without registry credentials.
