#!/usr/bin/env python3
"""Render, create, and verify the public GLM-5.3-Flash NVFP4 Pod template."""
import argparse
import json
import os
import sys

import requests


REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = (
    "ghcr.io/utensil/worker-vllm-glm@"
    "sha256:8a025ed9775fd57dcd66ec11d80296401e2609afcda483ed64345bc4a3328816"
)
MODEL = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "36c184c6cda000a481711306df5adde42f63321a"
NAME = "GLM-5.3-Flash-NVFP4 (Public vLLM 1xB300 TP1)"

SERVE = (
    'api_args=(); if [ -n "${VLLM_API_KEY:-}" ]; then '
    'api_args=(--api-key "$VLLM_API_KEY"); fi; '
    "exec python3 -m vllm.entrypoints.openai.api_server "
    f"--model {MODEL} --revision {REVISION} --served-model-name {MODEL} "
    "--host 0.0.0.0 --port 8000 --tensor-parallel-size 1 "
    "--max-model-len 8192 --max-num-seqs 2 --gpu-memory-utilization 0.90 "
    "--no-enable-flashinfer-autotune --tool-call-parser glm47 "
    '--enable-auto-tool-choice --reasoning-parser glm45 "${api_args[@]}"'
)

README = """# GLM-5.3-Flash-NVFP4 on vLLM

Runs RedHatAI/GLM-5.3-Flash-NVFP4 as an OpenAI-compatible server on one NVIDIA
B300 at TP1. Exposes port 8000 and uses a 300 GB container disk with no volume.

The public worker image derives from a pinned vLLM amd64 image and bakes a
fail-closed fix for that image's GLM processor Hub lookup. The model revision is
pinned. The checkpoint is public and this template stores no credentials.

Add VLLM_API_KEY when deploying if API authentication is required. Select exactly
one NVIDIA B300 SXM6 AC for the demonstrated baseline.

This is a deployment and basic-API baseline, not a full quality, throughput, or
optimized-production claim. Source: https://github.com/utensil/worker-vllm-glm
"""


def payload():
    return {
        "name": NAME,
        "imageName": IMAGE,
        "category": "NVIDIA",
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [SERVE],
        "containerDiskInGb": 300,
        "volumeInGb": 0,
        "volumeMountPath": "/runpod-volume",
        "ports": ["8000/http"],
        "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "isPublic": True,
        "isServerless": False,
        "readme": README,
    }


def _stored_view(template):
    ports = template.get("ports") or []
    if isinstance(ports, str):
        ports = [item for item in ports.split(",") if item]
    env = template.get("env") or {}
    if isinstance(env, list):
        env = {item["key"]: item["value"] for item in env}
    docker_args = template.get("dockerArgs")
    entrypoint = template.get("dockerEntrypoint") or []
    if docker_args is None:
        commands = template.get("dockerStartCmd") or []
        docker_args = commands[0] if commands else ""
    else:
        try:
            wrapped = json.loads(docker_args)
        except (TypeError, json.JSONDecodeError):
            wrapped = None
        if isinstance(wrapped, dict):
            commands = wrapped.get("cmd") or []
            docker_args = commands[0] if commands else ""
            entrypoint = wrapped.get("entrypoint") or []
    keys = (
        "category", "name", "imageName", "containerDiskInGb", "volumeInGb",
        "volumeMountPath", "isPublic", "isServerless", "readme",
    )
    view = {key: template.get(key) for key in keys}
    if isinstance(view.get("readme"), str):
        view["readme"] = view["readme"].rstrip("\n")
    view.update({"ports": ports, "env": env, "dockerArgs": docker_args,
                 "dockerEntrypoint": entrypoint})
    return view


def verify(template, expected):
    actual_view = _stored_view(template)
    expected_view = _stored_view(expected)
    mismatch = {key: {"expected": expected_view[key], "actual": actual_view[key]}
                for key in expected_view if actual_view[key] != expected_view[key]}
    if mismatch:
        raise SystemExit("template readback mismatch: " + json.dumps(mismatch))
    if actual_view["isPublic"] is not True or actual_view["isServerless"] is not False:
        raise SystemExit("template is not public Pod-only")


def _templates(key):
    query = """query {
      myself {
        podTemplates {
          id category name imageName dockerArgs containerDiskInGb volumeInGb
          volumeMountPath ports env { key value } isPublic isServerless readme
        }
      }
    }"""
    response = requests.post(
        GRAPHQL,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        json={"query": query}, timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise SystemExit(body["errors"][0].get("message", "GraphQL error"))
    return body["data"]["myself"]["podTemplates"]


def create_or_reuse(expected, key):
    matches = [item for item in _templates(key)
               if item.get("name") == NAME
               and item.get("isPublic") is True
               and item.get("isServerless") is False]
    if len(matches) > 1:
        raise SystemExit("multiple public Pod templates have the requested name")
    if matches:
        verify(matches[0], expected)
        return matches[0]["id"], "reused"

    response = requests.post(
        REST + "/templates",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        json=expected, timeout=60,
    )
    if response.status_code not in (200, 201):
        raise SystemExit(
            f"template create failed {response.status_code}: {response.text[:300]}"
        )
    created = response.json()
    template_id = created.get("id") or (created.get("template") or {}).get("id")
    matches = [item for item in _templates(key) if item.get("id") == template_id]
    if len(matches) != 1:
        raise SystemExit("created template missing from GraphQL readback")
    verify(matches[0], expected)
    return template_id, "created"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true",
                        help="create/reuse and verify the public template")
    args = parser.parse_args()
    expected = payload()
    if not args.create:
        print(json.dumps(expected, indent=2))
        print("PUBLIC_POD_TEMPLATE(dry): PASS")
        return 0
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("set RUNPOD_API_KEY")
    template_id, action = create_or_reuse(expected, key)
    print(f"PUBLIC_POD_TEMPLATE: PASS action={action} id={template_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
