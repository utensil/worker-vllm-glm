import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serverless import main


class FakeProcess:
    def __init__(self, return_code=None):
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.terminated and not self.killed and timeout == 15:
            return 0
        return 0


class HealthyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ServerlessMainTest(unittest.TestCase):
    def test_command_defaults_to_tp1_and_loopback(self):
        argv = main.build_vllm_argv()
        joined = " ".join(argv)
        self.assertIn(main.MODEL, argv)
        self.assertIn(main.REVISION, argv)
        self.assertIn("--host 127.0.0.1", joined)
        self.assertIn("--tensor-parallel-size 1", joined)
        self.assertIn("--max-model-len 8192", joined)
        self.assertIn("--no-enable-flashinfer-autotune", argv)

    def test_command_accepts_explicit_tp2(self):
        with mock.patch.dict(main.os.environ, {"TENSOR_PARALLEL_SIZE": "2"}):
            argv = main.build_vllm_argv()
        self.assertIn("--tensor-parallel-size 2", " ".join(argv))

    def test_cached_command_requires_and_uses_exact_pinned_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = (
                Path(tmp)
                / "models--RedHatAI--GLM-5.3-Flash-NVFP4"
                / "snapshots"
                / main.REVISION
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").touch()
            (snapshot / "model.safetensors.index.json").touch()
            with mock.patch.dict(
                main.os.environ,
                {
                    "RUNPOD_MODEL_CACHE_REQUIRED": "true",
                    "RUNPOD_MODEL_CACHE_ROOT": tmp,
                },
                clear=False,
            ):
                argv = main.build_vllm_argv()
                self.assertEqual(argv[argv.index("--model") + 1], str(snapshot))
                self.assertNotIn("--revision", argv)
                self.assertEqual(main.os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(main.os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_cached_command_never_falls_back_to_hub(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(
                main.os.environ,
                {
                    "RUNPOD_MODEL_CACHE_REQUIRED": "1",
                    "RUNPOD_MODEL_CACHE_ROOT": tmp,
                },
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "exact pinned"),
        ):
            main.build_vllm_argv()

    def test_cached_command_rejects_ambiguous_boolean(self):
        with (
            mock.patch.dict(
                main.os.environ,
                {"RUNPOD_MODEL_CACHE_REQUIRED": "sometimes"},
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "must be true or false"),
        ):
            main.build_vllm_argv()

    def test_command_rejects_invalid_tensor_parallel_size(self):
        for value in ("0", "9", "two"):
            with (
                self.subTest(value=value),
                mock.patch.dict(main.os.environ, {"TENSOR_PARALLEL_SIZE": value}),
                self.assertRaisesRegex(RuntimeError, "TENSOR_PARALLEL_SIZE"),
            ):
                main.build_vllm_argv()

    def test_health_gate_succeeds_only_on_200(self):
        with mock.patch.object(
            main.urllib.request, "urlopen", return_value=HealthyResponse()
        ):
            main.wait_until_healthy(FakeProcess(), timeout=1, poll_seconds=0)

    def test_health_gate_fails_when_backend_exits(self):
        with self.assertRaisesRegex(RuntimeError, "exited before"):
            main.wait_until_healthy(FakeProcess(7), timeout=1, poll_seconds=0)

    def test_health_gate_has_hard_deadline(self):
        with (
            mock.patch.object(main.time, "monotonic", side_effect=[0, 2]),
            self.assertRaisesRegex(RuntimeError, "within 1s"),
        ):
            main.wait_until_healthy(FakeProcess(), timeout=1, poll_seconds=0)

    def test_stop_backend_terminates_running_child(self):
        process = FakeProcess()
        main.stop_backend(process)
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_stop_backend_kills_after_grace_timeout(self):
        process = FakeProcess()
        process.wait = mock.Mock(side_effect=[subprocess.TimeoutExpired("vllm", 15), 0])
        main.stop_backend(process)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_watchdog_terminates_worker_when_healthy_child_later_exits(self):
        process = FakeProcess(17)
        stop_event = threading.Event()
        with (
            mock.patch.object(main.os, "getpid", return_value=1234),
            mock.patch.object(main.os, "kill") as kill,
        ):
            main.watch_backend(process, stop_event, poll_seconds=0)
        kill.assert_called_once_with(1234, main.signal.SIGTERM)

    def test_watchdog_does_not_signal_during_intentional_shutdown(self):
        stop_event = threading.Event()
        stop_event.set()
        with mock.patch.object(main.os, "kill") as kill:
            main.watch_backend(FakeProcess(0), stop_event, poll_seconds=0)
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
