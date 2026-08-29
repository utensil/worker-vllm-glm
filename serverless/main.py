"""Fail-closed process supervisor for the GLM RunPod queue worker."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)
MODEL = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "36c184c6cda000a481711306df5adde42f63321a"
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
HOST = "127.0.0.1"
PORT = 8000
HEALTH_POLL_SECONDS = 2.0

backend_process: subprocess.Popen[Any] | None = None
watchdog_stop_event = threading.Event()


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _strict_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true"}:
        return True
    if raw in {"0", "false"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def resolve_model_source() -> tuple[str, bool]:
    """Return the exact model source and whether a Hub revision is required."""
    if not _strict_bool("RUNPOD_MODEL_CACHE_REQUIRED"):
        return MODEL, True
    cache_root = Path(os.getenv("RUNPOD_MODEL_CACHE_ROOT", str(CACHE_ROOT)))
    snapshot = (
        cache_root / "models--RedHatAI--GLM-5.3-Flash-NVFP4" / "snapshots" / REVISION
    )
    required = (snapshot / "config.json", snapshot / "model.safetensors.index.json")
    if not snapshot.is_dir() or not all(path.is_file() for path in required):
        raise RuntimeError("exact pinned RunPod cached-model snapshot is unavailable")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return str(snapshot), False


def build_vllm_argv() -> list[str]:
    tensor_parallel_size = _bounded_int("TENSOR_PARALLEL_SIZE", 1, 1, 8)
    max_model_len = _bounded_int("MAX_MODEL_LEN", 8192, 1024, 131072)
    max_num_seqs = _bounded_int("MAX_NUM_SEQS", 2, 1, 64)
    gpu_memory = _bounded_float("GPU_MEMORY_UTILIZATION", 0.90, 0.50, 0.98)
    model_source, needs_revision = resolve_model_source()
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_source,
        "--served-model-name",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory),
        "--no-enable-flashinfer-autotune",
        "--tool-call-parser",
        "glm47",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "glm45",
    ]
    if needs_revision:
        argv[5:5] = ["--revision", REVISION]
    return argv


def startup_timeout_seconds() -> int:
    return _bounded_int("VLLM_STARTUP_TIMEOUT", 1800, 60, 3600)


def wait_until_healthy(
    process: subprocess.Popen[Any],
    *,
    timeout: int | None = None,
    poll_seconds: float = HEALTH_POLL_SECONDS,
) -> None:
    timeout = startup_timeout_seconds() if timeout is None else timeout
    deadline = time.monotonic() + timeout
    health_url = f"http://{HOST}:{PORT}/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vLLM exited before becoming healthy with code {return_code}"
            )
        try:
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    LOGGER.info("vLLM health gate passed")
                    return
        except OSError:
            pass
        time.sleep(poll_seconds)
    raise RuntimeError(f"vLLM did not become healthy within {timeout}s")


def stop_backend(process: subprocess.Popen[Any], grace_seconds: float = 15) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def watch_backend(
    process: subprocess.Popen[Any],
    stop_event: threading.Event,
    *,
    poll_seconds: float = 2.0,
) -> None:
    """Terminate this worker if the healthy vLLM child later exits."""
    while not stop_event.wait(poll_seconds):
        return_code = process.poll()
        if return_code is None:
            continue
        if stop_event.is_set():
            return
        LOGGER.critical(
            "vLLM exited after queue registration with code %s; terminating worker",
            return_code,
        )
        os.kill(os.getpid(), signal.SIGTERM)
        return


def start_backend_watchdog(
    process: subprocess.Popen[Any], stop_event: threading.Event
) -> threading.Thread:
    thread = threading.Thread(
        target=watch_backend,
        args=(process, stop_event),
        name="vllm-child-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _forward_signal(signum: int, _frame: Any) -> None:
    LOGGER.info("received signal %s", signum)
    watchdog_stop_event.set()
    if backend_process is not None:
        stop_backend(backend_process)
    raise SystemExit(128 + signum)


def main() -> None:
    global backend_process
    argv = build_vllm_argv()
    LOGGER.info("starting pinned GLM vLLM backend")
    backend_process = subprocess.Popen(argv)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)
    try:
        wait_until_healthy(backend_process)
    except Exception as exc:
        LOGGER.error("startup health gate failed: %s", exc)
        stop_backend(backend_process)
        raise SystemExit(1) from exc

    import handler as queue_handler
    import runpod

    queue_handler.backend_process = backend_process
    max_concurrency = _bounded_int("MAX_CONCURRENCY", 2, 1, 2)
    watchdog_stop_event.clear()
    start_backend_watchdog(backend_process, watchdog_stop_event)
    try:
        runpod.serverless.start(
            {
                "handler": queue_handler.handler,
                "concurrency_modifier": lambda _current: max_concurrency,
                "return_aggregate_stream": True,
            }
        )
    finally:
        watchdog_stop_event.set()
        stop_backend(backend_process)


if __name__ == "__main__":
    main()
