#!/usr/bin/env python3
"""Render/create the distinct public 2xB200 TP2 Serverless template.

Dry render is the default. Template creation remains locked until IMAGE is an
immutable digest. Endpoint rendering requires a live exact-count B200 catalog
gate and never creates compute or substitutes another GPU type/topology.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import requests

import create_serverless_template as common

IMAGE = (
    "ghcr.io/utensil/worker-vllm-glm@"
    "sha256:7a39b66b69597bddfc06106d3c74daf959b447c48b9cc1c46b0b45bfbcc9e529"
)
NAME = "GLM-5.3-Flash-NVFP4 (Public Serverless 2xB200 TP2)"
GPU_TYPE_ID = "NVIDIA B200"
GPU_COUNT = 2
CUDA_VERSION = "13.0"

README = """# GLM-5.3-Flash-NVFP4 Serverless 2xB200 TP2 candidate

Queue-worker candidate for RedHatAI/GLM-5.3-Flash-NVFP4 on exactly two NVIDIA
B200 GPUs at tensor parallel size 2. The worker owns a loopback vLLM server,
waits for health, then registers with RunPod. It supports model listing,
non-streaming and SSE chat/text completions, tool-call payloads, and remote
image-URL chat content.

The template uses a 300 GB ephemeral disk, no network volume, no exposed port,
and stores no credentials. Both RunPod and vLLM initialization timeouts are
1800 seconds. The template does not select or create an endpoint.

Any live endpoint must independently admit exactly NVIDIA B200 count=2 with
CUDA 13.0, exclude every other GPU type in the selected pool, and retain one
worker maximum. There is no B300, TP1, TP4, or other-hardware fallback.

This TP2 topology remains a feasibility candidate until a bounded live cold
start, model-list, and chat check passes. Source:
https://github.com/utensil/worker-vllm-glm
"""


def payload() -> dict[str, Any]:
    value = common.payload()
    value.update({"name": NAME, "imageName": IMAGE, "readme": README})
    value["env"] = dict(value["env"], TENSOR_PARALLEL_SIZE="2")
    return value


def exact_serverless_gpu_selection(
    session: requests.Session, key: str
) -> dict[str, Any]:
    response = session.get(
        common.API_V2 + "/catalog/gpus",
        headers=common._headers(key),
        params={
            "include": "AVAILABILITY",
            "product": "SERVERLESS",
            "count": GPU_COUNT,
            "cudaVersions": CUDA_VERSION,
        },
        timeout=30,
    )
    body = common._response_json(response, "serverless GPU catalog read")
    gpus = body.get("gpus") if isinstance(body, dict) else None
    if not isinstance(gpus, list):
        raise SystemExit("serverless GPU catalog response has no gpus list")
    exact = [gpu for gpu in gpus if gpu.get("id") == GPU_TYPE_ID]
    if len(exact) != 1:
        raise SystemExit(f"exact Serverless GPU type unavailable: {GPU_TYPE_ID}")
    gpu = exact[0]
    availability = gpu.get("availability")
    if availability not in {"LOW", "MEDIUM", "HIGH"}:
        raise SystemExit(
            f"exact GPU count has no current Serverless availability: "
            f"{GPU_TYPE_ID} count={GPU_COUNT} availability={availability}"
        )
    pool = gpu.get("pool")
    price = (gpu.get("price") or {}).get("serverless")
    if (
        not isinstance(pool, str)
        or not pool
        or isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(price)
        or price <= 0
    ):
        raise SystemExit(
            f"exact GPU count is not admitted to Serverless: "
            f"{GPU_TYPE_ID} count={GPU_COUNT}"
        )
    cuda_versions = gpu.get("cudaVersions") or []
    if (
        sum(
            item.get("version") == CUDA_VERSION and item.get("available") is True
            for item in cuda_versions
        )
        != 1
    ):
        raise SystemExit(
            f"exact GPU count lacks available CUDA {CUDA_VERSION}: "
            f"{GPU_TYPE_ID} count={GPU_COUNT}"
        )
    exclusions = sorted(
        candidate["id"]
        for candidate in gpus
        if candidate.get("pool") == pool and candidate.get("id") != GPU_TYPE_ID
    )
    return {
        "pools": [pool],
        "excludedTypes": exclusions,
        "count": GPU_COUNT,
        "allowedCudaVersions": [CUDA_VERSION],
    }


def temporary_endpoint_payload(
    template_id: str, gpu_selection: dict[str, Any]
) -> dict[str, Any]:
    endpoint = common.temporary_endpoint_payload(
        template_id,
        {
            **gpu_selection,
            "count": 1,
        },
    )
    endpoint["name"] = "GLM-5.3-Flash-NVFP4 temporary 2xB200 TP2 validation"
    endpoint["gpu"] = dict(gpu_selection)
    validate_temporary_endpoint_payload(endpoint)
    return endpoint


def validate_temporary_endpoint_payload(endpoint: dict[str, Any]) -> None:
    if set(endpoint) != common.ENDPOINT_ALLOWED_KEYS:
        raise ValueError("temporary endpoint has unsupported or missing top-level keys")
    gpu = endpoint.get("gpu")
    if not isinstance(gpu, dict) or set(gpu) != common.ENDPOINT_GPU_KEYS:
        raise ValueError("temporary endpoint GPU contract mismatch")
    if type(gpu.get("count")) is not int or gpu["count"] != GPU_COUNT:
        raise ValueError("temporary endpoint exact-2xB200 selection mismatch")
    check = dict(endpoint)
    check["gpu"] = dict(gpu, count=1)
    common.validate_temporary_endpoint_payload(check)


def create_or_reuse(
    expected: dict[str, Any], key: str, session: requests.Session
) -> tuple[str, str]:
    common.require_immutable_image(expected["imageName"])
    matches = [
        item for item in common._templates(session, key) if item.get("name") == NAME
    ]
    if len(matches) > 1:
        raise SystemExit("multiple templates have the requested name")
    if matches:
        common.verify(matches[0], expected)
        return matches[0]["id"], "reused"
    created = common._response_json(
        session.post(
            common.REST + "/templates",
            headers=common._headers(key),
            json=expected,
            timeout=60,
        ),
        "template create",
    )
    template_id = created.get("id") or (created.get("template") or {}).get("id")
    if not template_id:
        raise SystemExit("template create response has no id")
    stored = [
        item
        for item in common._templates(session, key)
        if item.get("id") == template_id
    ]
    if len(stored) != 1:
        raise SystemExit("created template missing from GraphQL readback")
    common.verify(stored[0], expected)
    return template_id, "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--create", action="store_true")
    action.add_argument("--check-gpu", action="store_true")
    action.add_argument("--render-temporary-endpoint", metavar="TEMPLATE_ID")
    args = parser.parse_args()
    expected = payload()
    if not (args.create or args.check_gpu or args.render_temporary_endpoint):
        print(json.dumps(expected, indent=2))
        print("PUBLIC_SERVERLESS_B200_TP2_TEMPLATE(dry): PASS")
        return 0
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("set RUNPOD_API_KEY")
    session = requests.Session()
    if args.check_gpu:
        selection = exact_serverless_gpu_selection(session, key)
        print(
            "EXACT_2XB200_SERVERLESS_GPU: PASS " + json.dumps(selection, sort_keys=True)
        )
        return 0
    if args.render_temporary_endpoint:
        selection = exact_serverless_gpu_selection(session, key)
        endpoint = temporary_endpoint_payload(args.render_temporary_endpoint, selection)
        print(json.dumps(endpoint, indent=2, sort_keys=True))
        print("TEMPORARY_2XB200_ENDPOINT(render-only): PASS")
        return 0
    template_id, result = create_or_reuse(expected, key, session)
    print(f"PUBLIC_SERVERLESS_B200_TP2_TEMPLATE: PASS action={result} id={template_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
