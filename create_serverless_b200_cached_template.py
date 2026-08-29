#!/usr/bin/env python3
"""Render/create the fail-closed cached-model 2xB200 TP2 template."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

import create_serverless_b200_template as b200

IMAGE = (
    "ghcr.io/utensil/worker-vllm-glm@"
    "sha256:577394a87422627b322d7c64bb3d589fd5a9b2f1d25bda87e7048a1855364669"
)
NAME = "GLM-5.3-Flash-NVFP4 (Public Serverless Cached 2xB200 TP2)"

README = """# GLM-5.3-Flash-NVFP4 cached-model Serverless worker

Queue-worker template for exactly two NVIDIA B200 GPUs at TP2. It requires the
exact RedHatAI/GLM-5.3-Flash-NVFP4 revision to be attached through RunPod's
Hugging Face model-cache feature. The worker resolves only the pinned local
snapshot and fails closed rather than downloading from the Hub at runtime.

The template alone does not attach the model. Create the endpoint with current
runpodctl and `--model-reference` using the exact URL/revision documented in the
source repository. Never remove `RUNPOD_MODEL_CACHE_REQUIRED`, substitute a GPU,
or change the GPU count. No credential is stored in this public template.
"""


def payload() -> dict[str, Any]:
    value = b200.payload()
    value.update({"name": NAME, "imageName": IMAGE, "readme": README})
    value["env"] = dict(
        value["env"],
        RUNPOD_MODEL_CACHE_REQUIRED="true",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return value


def create_or_reuse(
    expected: dict[str, Any], key: str, session: requests.Session
) -> tuple[str, str]:
    b200.common.require_immutable_image(expected["imageName"])
    matches = [
        item
        for item in b200.common._templates(session, key)
        if item.get("name") == NAME
    ]
    if len(matches) > 1:
        raise SystemExit("multiple templates have the requested name")
    if matches:
        b200.common.verify(matches[0], expected)
        return matches[0]["id"], "reused"
    created = b200.common._response_json(
        session.post(
            b200.common.REST + "/templates",
            headers=b200.common._headers(key),
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
        for item in b200.common._templates(session, key)
        if item.get("id") == template_id
    ]
    if len(stored) != 1:
        raise SystemExit("created template missing from GraphQL readback")
    b200.common.verify(stored[0], expected)
    return template_id, "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    expected = payload()
    if not args.create:
        print(json.dumps(expected, indent=2))
        print("PUBLIC_SERVERLESS_CACHED_B200_TP2_TEMPLATE(dry): PASS")
        return 0
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("set RUNPOD_API_KEY")
    template_id, result = create_or_reuse(expected, key, requests.Session())
    print(
        "PUBLIC_SERVERLESS_CACHED_B200_TP2_TEMPLATE: "
        f"PASS action={result} id={template_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
