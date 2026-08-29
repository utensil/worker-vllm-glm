# worker-vllm-glm

[![Deploy GLM-5.3-Flash NVFP4 on RunPod](https://img.shields.io/badge/Deploy-GLM--5.3--Flash%20NVFP4-673AB7?style=for-the-badge)](https://console.runpod.io/deploy?template=yp08gkkdz2&ref=km0th85l)

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

This repository currently publishes a **Pod** image/template only. The vLLM HTTP
server is not a RunPod Serverless queue worker; Serverless needs a separate
handler-compatible image and independent worker/cold-start validation.

## Build

GitHub Actions tests every change. Image-source changes build `linux/amd64` and
publish both `latest` and the versioned convenience tag to GHCR. Deployment uses
the digest above. The repository and package must remain public so RunPod can
pull the image without registry credentials.
