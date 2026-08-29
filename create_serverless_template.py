#!/usr/bin/env python3
"""Render, create, and read back the public GLM Serverless worker template.

Dry render is the default. Template creation requires an immutable image digest
and an API key from the environment, but is deliberately independent of GPU
availability because templates do not select hardware. The separate endpoint
preflight requires an exact, currently admitted B300. Nothing here creates an
endpoint or starts compute.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import requests

REST = "https://rest.runpod.io/v1"
API_V2 = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = (
    "ghcr.io/utensil/worker-vllm-glm@"
    "sha256:3f31ef919c98ad370f4bfdddafd6f20ae92f1208fc96616bb1e9cbd877e09c09"
)
NAME = "GLM-5.3-Flash-NVFP4 (Public Serverless 1xB300 TP1)"
GPU_TYPE_ID = "NVIDIA B300 SXM6 AC"
CUDA_VERSION = "13.0"
ENDPOINT_ALLOWED_KEYS = frozenset(
    {"name", "templateId", "type", "gpu", "workers", "scaling", "flashboot", "timeout"}
)
ENDPOINT_GPU_KEYS = frozenset(
    {"pools", "excludedTypes", "count", "allowedCudaVersions"}
)
ENDPOINT_WORKER_KEYS = frozenset({"min", "max", "idleTimeout"})
ENDPOINT_SCALING_KEYS = frozenset({"type", "queueDelay"})

README = """# GLM-5.3-Flash-NVFP4 Serverless worker

Queue worker for RedHatAI/GLM-5.3-Flash-NVFP4 on exactly one NVIDIA B300 at
TP1. The image owns a loopback vLLM server, waits for health, then registers
with RunPod. It supports model listing, non-streaming and SSE chat/text
completions, tool-call payloads, and remote image-URL chat content.

The template uses a 300 GB ephemeral disk, no network volume, no exposed port,
and stores no credentials. Both RunPod and vLLM initialization timeouts are
1800 seconds. This template does not select or create an endpoint; endpoint
creation must independently pin the exact B300 type with no hardware fallback.

At the 2026-08-29 catalog readback, exact-B300 endpoint validation is blocked:
its Serverless pool and price are null, availability is NONE, and its listed
CUDA 13.0/13.2 placements are unavailable. Re-run the repository's mandatory
read-only GPU gate before endpoint work; never substitute B200 or change to TP2.

Serverless remains unvalidated until the repository's bounded live gate passes.
Source: https://github.com/utensil/worker-vllm-glm
"""


def payload() -> dict[str, Any]:
    return {
        "name": NAME,
        "imageName": IMAGE,
        "category": "NVIDIA",
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "containerDiskInGb": 300,
        "volumeInGb": 0,
        "volumeMountPath": "/runpod-volume",
        "ports": [],
        "env": {
            "RUNPOD_INIT_TIMEOUT": "1800",
            "VLLM_STARTUP_TIMEOUT": "1800",
            "BACKEND_REQUEST_TIMEOUT_SECONDS": "600",
            "MAX_CONCURRENCY": "2",
            "MAX_MODEL_LEN": "8192",
            "MAX_NUM_SEQS": "2",
            "GPU_MEMORY_UTILIZATION": "0.90",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "isServerless": True,
        "isPublic": True,
        "readme": README,
    }


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": "Bearer " + key, "Content-Type": "application/json"}


def _response_json(response: requests.Response, operation: str) -> Any:
    if response.status_code < 200 or response.status_code >= 300:
        raise SystemExit(f"{operation} failed with HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise SystemExit(f"{operation} returned invalid JSON") from exc


def require_immutable_image(image: str) -> None:
    if not re.fullmatch(
        r"ghcr\.io/[a-z0-9_.-]+/[a-z0-9_.-]+@sha256:[0-9a-f]{64}", image
    ):
        raise SystemExit(
            "live creation is locked until IMAGE is replaced by the published digest"
        )


def exact_serverless_gpu_selection(
    session: requests.Session, key: str
) -> dict[str, Any]:
    response = session.get(
        API_V2 + "/catalog/gpus",
        headers=_headers(key),
        params={
            "include": "AVAILABILITY",
            "product": "SERVERLESS",
            "count": 1,
            "cudaVersions": CUDA_VERSION,
        },
        timeout=30,
    )
    body = _response_json(response, "serverless GPU catalog read")
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
            f"exact GPU has no current Serverless availability: {GPU_TYPE_ID} "
            f"availability={availability}"
        )
    pool = gpu.get("pool")
    serverless_price = (gpu.get("price") or {}).get("serverless")
    if not isinstance(pool, str) or not pool or serverless_price is None:
        raise SystemExit(f"exact GPU is not admitted to Serverless: {GPU_TYPE_ID}")
    cuda_versions = gpu.get("cudaVersions") or []
    cuda_match = [
        item
        for item in cuda_versions
        if item.get("version") == CUDA_VERSION and item.get("available") is True
    ]
    if len(cuda_match) != 1:
        raise SystemExit(
            f"exact GPU lacks available CUDA {CUDA_VERSION}: {GPU_TYPE_ID}"
        )

    exclusions = sorted(
        candidate["id"]
        for candidate in gpus
        if candidate.get("pool") == pool and candidate.get("id") != GPU_TYPE_ID
    )
    return {
        "pools": [pool],
        "excludedTypes": exclusions,
        "count": 1,
        "allowedCudaVersions": [CUDA_VERSION],
    }


def temporary_endpoint_payload(
    template_id: str, gpu_selection: dict[str, Any]
) -> dict[str, Any]:
    """Render the approved bounded endpoint configuration without creating it."""
    if not template_id.strip():
        raise ValueError("template_id must be non-empty")
    if set(gpu_selection) != ENDPOINT_GPU_KEYS:
        raise ValueError("gpu selection was not produced by the exact-B300 gate")
    endpoint = {
        "name": "GLM-5.3-Flash-NVFP4 temporary validation",
        "templateId": template_id,
        "type": "QUEUE",
        "gpu": gpu_selection,
        "workers": {"min": 0, "max": 1, "idleTimeout": 300},
        "scaling": {"type": "QUEUE_DELAY", "queueDelay": 1},
        "flashboot": "FLASHBOOT",
        "timeout": 1_800_000,
    }
    validate_temporary_endpoint_payload(endpoint)
    return endpoint


def validate_temporary_endpoint_payload(endpoint: dict[str, Any]) -> None:
    """Fail closed if the REST v2 CreateEndpointRequest contract drifts."""
    if set(endpoint) != ENDPOINT_ALLOWED_KEYS:
        raise ValueError("temporary endpoint has unsupported or missing top-level keys")
    if (
        not isinstance(endpoint["name"], str)
        or not endpoint["name"].strip()
        or not isinstance(endpoint["templateId"], str)
        or not endpoint["templateId"].strip()
        or endpoint["type"] != "QUEUE"
        or endpoint["flashboot"] != "FLASHBOOT"
        or type(endpoint["timeout"]) is not int
        or endpoint["timeout"] != 1_800_000
    ):
        raise ValueError("temporary endpoint scalar contract mismatch")
    gpu = endpoint["gpu"]
    if not isinstance(gpu, dict) or set(gpu) != ENDPOINT_GPU_KEYS:
        raise ValueError("temporary endpoint GPU contract mismatch")
    if (
        not isinstance(gpu["pools"], list)
        or len(gpu["pools"]) != 1
        or not isinstance(gpu["pools"][0], str)
        or not gpu["pools"][0]
        or not isinstance(gpu["excludedTypes"], list)
        or any(not isinstance(item, str) or not item for item in gpu["excludedTypes"])
        or gpu["excludedTypes"] != sorted(set(gpu["excludedTypes"]))
        or type(gpu["count"]) is not int
        or gpu["count"] != 1
        or gpu["allowedCudaVersions"] != [CUDA_VERSION]
    ):
        raise ValueError("temporary endpoint exact-B300 selection mismatch")
    workers = endpoint["workers"]
    if (
        not isinstance(workers, dict)
        or set(workers) != ENDPOINT_WORKER_KEYS
        or workers != {"min": 0, "max": 1, "idleTimeout": 300}
    ):
        raise ValueError("temporary endpoint worker bounds mismatch")
    scaling = endpoint["scaling"]
    if (
        not isinstance(scaling, dict)
        or set(scaling) != ENDPOINT_SCALING_KEYS
        or scaling != {"type": "QUEUE_DELAY", "queueDelay": 1}
    ):
        raise ValueError("temporary endpoint scaling contract mismatch")


def _stored_view(value: dict[str, Any]) -> dict[str, Any]:
    ports = value.get("ports") or []
    if isinstance(ports, str):
        ports = [item for item in ports.split(",") if item]
    env = value.get("env") or {}
    if isinstance(env, list):
        env = {item["key"]: item["value"] for item in env}
    entrypoint = value.get("dockerEntrypoint") or []
    commands = value.get("dockerStartCmd") or []
    docker_args = value.get("dockerArgs")
    if docker_args:
        try:
            wrapped = json.loads(docker_args)
        except (TypeError, json.JSONDecodeError):
            wrapped = None
        if isinstance(wrapped, dict):
            entrypoint = wrapped.get("entrypoint") or []
            commands = wrapped.get("cmd") or []
        elif not commands:
            commands = [docker_args]
    keys = (
        "name",
        "imageName",
        "category",
        "containerDiskInGb",
        "volumeInGb",
        "volumeMountPath",
        "isServerless",
        "isPublic",
        "readme",
    )
    view = {key: value.get(key) for key in keys}
    if isinstance(view.get("readme"), str):
        view["readme"] = view["readme"].rstrip("\n")
    view.update(
        {
            "dockerEntrypoint": entrypoint,
            "dockerStartCmd": commands,
            "ports": ports,
            "env": env,
        }
    )
    return view


def verify(stored: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = _stored_view(stored)
    wanted = _stored_view(expected)
    mismatch = {
        key: {"expected": wanted[key], "actual": actual[key]}
        for key in wanted
        if wanted[key] != actual[key]
    }
    if mismatch:
        raise SystemExit("template readback mismatch: " + json.dumps(mismatch))
    if actual["isPublic"] is not True or actual["isServerless"] is not True:
        raise SystemExit("template is not public Serverless")
    if (
        actual["ports"]
        or actual["volumeInGb"] != 0
        or actual["dockerStartCmd"]
        or "HF_TOKEN" in actual["env"]
        or "VLLM_API_KEY" in actual["env"]
    ):
        raise SystemExit(
            "template unexpectedly stores command, volume, ports, or credentials"
        )


def _templates(session: requests.Session, key: str) -> list[dict[str, Any]]:
    query = """query {
      myself {
        podTemplates {
          id category name imageName dockerArgs containerDiskInGb volumeInGb
          volumeMountPath ports env { key value } isPublic isServerless readme
        }
      }
    }"""
    body = _response_json(
        session.post(
            GRAPHQL,
            headers=_headers(key),
            json={"query": query},
            timeout=30,
        ),
        "template GraphQL read",
    )
    if body.get("errors"):
        raise SystemExit(body["errors"][0].get("message", "GraphQL error"))
    return body["data"]["myself"]["podTemplates"]


def create_or_reuse(
    expected: dict[str, Any], key: str, session: requests.Session
) -> tuple[str, str]:
    require_immutable_image(expected["imageName"])
    matches = [item for item in _templates(session, key) if item.get("name") == NAME]
    if len(matches) > 1:
        raise SystemExit("multiple templates have the requested name")
    if matches:
        verify(matches[0], expected)
        return matches[0]["id"], "reused"

    created = _response_json(
        session.post(
            REST + "/templates",
            headers=_headers(key),
            json=expected,
            timeout=60,
        ),
        "template create",
    )
    template_id = created.get("id") or (created.get("template") or {}).get("id")
    if not template_id:
        raise SystemExit("template create response has no id")
    matches = [
        item for item in _templates(session, key) if item.get("id") == template_id
    ]
    if len(matches) != 1:
        raise SystemExit("created template missing from GraphQL readback")
    verify(matches[0], expected)
    return template_id, "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--create", action="store_true", help="create/reuse and verify")
    action.add_argument(
        "--check-gpu", action="store_true", help="read-only exact B300 Serverless check"
    )
    action.add_argument(
        "--render-temporary-endpoint",
        metavar="TEMPLATE_ID",
        help="read live GPU catalog and render, but do not create, a bounded endpoint",
    )
    args = parser.parse_args()
    expected = payload()
    if not (args.create or args.check_gpu or args.render_temporary_endpoint):
        print(json.dumps(expected, indent=2))
        print("PUBLIC_SERVERLESS_TEMPLATE(dry): PASS")
        print(
            "ENDPOINT_LIVE(2026-08-29): BLOCKED exact B300 is not currently "
            "admitted/available on Serverless"
        )
        return 0
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("set RUNPOD_API_KEY")
    session = requests.Session()
    if args.check_gpu:
        selection = exact_serverless_gpu_selection(session, key)
        print("EXACT_SERVERLESS_GPU: PASS " + json.dumps(selection, sort_keys=True))
        return 0
    if args.render_temporary_endpoint:
        selection = exact_serverless_gpu_selection(session, key)
        endpoint = temporary_endpoint_payload(args.render_temporary_endpoint, selection)
        print(json.dumps(endpoint, indent=2, sort_keys=True))
        print("TEMPORARY_ENDPOINT(render-only): PASS")
        return 0
    template_id, result = create_or_reuse(expected, key, session)
    print(f"PUBLIC_SERVERLESS_TEMPLATE: PASS action={result} id={template_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
