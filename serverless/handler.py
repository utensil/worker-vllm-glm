"""Async RunPod queue adapter for the local GLM vLLM OpenAI server.

The worker supports model listing plus non-streaming and SSE chat/text
completions. RunPod's OpenAI-compatible route uses ``openai_route`` and
``openai_input``. Native queue clients may instead send ``messages`` or
``prompt`` with an optional ``sampling_params`` object.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)
MODEL = "RedHatAI/GLM-5.3-Flash-NVFP4"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SUPPORTED_ROUTES = {
    "/v1/chat/completions": "POST",
    "/v1/completions": "POST",
    "/v1/models": "GET",
}
MAX_BODY_BYTES = 2 * 1024 * 1024

# Assigned by main.py only after the backend passes its health gate.
backend_process: Any | None = None


class RequestValidationError(ValueError):
    """The queue payload is outside the deliberately small public contract."""


@dataclass(frozen=True)
class RequestSpec:
    route: str
    method: str
    body: dict[str, Any] | None
    stream: bool = False


def _timeout_seconds() -> float:
    raw = os.getenv("BACKEND_REQUEST_TIMEOUT_SECONDS", "600")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("BACKEND_REQUEST_TIMEOUT_SECONDS must be numeric") from exc
    if not 1 <= value <= 3600:
        raise RuntimeError("BACKEND_REQUEST_TIMEOUT_SECONDS must be between 1 and 3600")
    return value


def _validate_body(route: str, body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise RequestValidationError("request body must be an object")
    requested_model = body.get("model")
    if requested_model not in (None, MODEL):
        raise RequestValidationError(f"model must be {MODEL}")

    normalized = dict(body)
    normalized.setdefault("model", MODEL)
    if route == "/v1/chat/completions":
        messages = normalized.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RequestValidationError("chat completions require non-empty messages")
        if any(not isinstance(message, dict) for message in messages):
            raise RequestValidationError("each chat message must be an object")
    elif route == "/v1/completions" and "prompt" not in normalized:
        raise RequestValidationError("text completions require prompt")

    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise RequestValidationError("request body exceeds 2 MiB")
    return normalized


def normalize_job(job: Any) -> RequestSpec:
    if not isinstance(job, dict):
        raise RequestValidationError("job must be an object")
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        raise RequestValidationError("job.input must be an object")

    if "openai_route" in job_input or "openai_input" in job_input:
        route = job_input.get("openai_route", "/v1/chat/completions")
        body = job_input.get("openai_input")
    elif "route" in job_input or "body" in job_input:
        route = job_input.get("route")
        body = job_input.get("body")
        supplied_method = job_input.get("method")
        if supplied_method is not None and (
            not isinstance(supplied_method, str)
            or supplied_method.upper() != SUPPORTED_ROUTES.get(route)
        ):
            raise RequestValidationError("method does not match the supported route")
    elif "messages" in job_input or "prompt" in job_input:
        sampling = job_input.get("sampling_params", {})
        if not isinstance(sampling, dict):
            raise RequestValidationError("sampling_params must be an object")
        body = dict(sampling)
        if "messages" in job_input:
            route = "/v1/chat/completions"
            body["messages"] = job_input["messages"]
        else:
            route = "/v1/completions"
            body["prompt"] = job_input["prompt"]
        body["stream"] = job_input.get("stream", False)
    else:
        raise RequestValidationError(
            "input must contain OpenAI passthrough, route/body, messages, or prompt"
        )

    if route not in SUPPORTED_ROUTES:
        raise RequestValidationError("unsupported OpenAI route")
    method = SUPPORTED_ROUTES[route]
    if method == "GET":
        if body not in (None, {}):
            raise RequestValidationError("model listing does not accept a body")
        return RequestSpec(route, method, None, False)
    normalized = _validate_body(route, body)
    return RequestSpec(route, method, normalized, normalized.get("stream") is True)


def _error(message: str, error_type: str, code: int | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def _backend_alive() -> bool:
    return backend_process is not None and backend_process.poll() is None


async def _proxy(spec: RequestSpec) -> AsyncIterator[Any]:
    timeout = aiohttp.ClientTimeout(total=_timeout_seconds())
    headers = {"Content-Type": "application/json"}
    if spec.stream:
        headers["Accept"] = "text/event-stream"
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.request(
                spec.method,
                DEFAULT_BASE_URL + spec.route,
                json=spec.body,
                headers=headers,
            ) as response,
        ):
            if response.status >= 400:
                LOGGER.warning(
                    "vLLM returned HTTP %s for %s", response.status, spec.route
                )
                yield _error(
                    f"vLLM returned HTTP {response.status}",
                    "backend_error",
                    response.status,
                )
                return
            if spec.stream:
                async for chunk in response.content.iter_any():
                    yield chunk.decode("utf-8", errors="replace")
                return
            payload = await response.content.read(MAX_BODY_BYTES + 1)
            if len(payload) > MAX_BODY_BYTES:
                yield _error("vLLM response exceeds 2 MiB", "backend_error")
                return
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                yield _error(
                    "vLLM returned a non-object JSON response", "backend_error"
                )
                return
            yield parsed
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        LOGGER.warning("vLLM request failed for %s: %s", spec.route, type(exc).__name__)
        yield _error("vLLM backend request failed or timed out", "backend_error")
    except (UnicodeDecodeError, json.JSONDecodeError):
        LOGGER.warning("vLLM returned invalid JSON for %s", spec.route)
        yield _error("vLLM returned invalid JSON", "backend_error")


async def handler(job: Any) -> AsyncIterator[Any]:
    try:
        spec = normalize_job(job)
    except RequestValidationError as exc:
        yield _error(str(exc), "validation_error")
        return

    if not _backend_alive():
        yield _error("vLLM backend is not running", "worker_unhealthy")
        return
    try:
        async for output in _proxy(spec):
            yield output
    except RuntimeError as exc:
        LOGGER.error("Invalid worker configuration: %s", exc)
        yield _error("worker request timeout configuration is invalid", "worker_error")
