import subprocess
import unittest
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
    def test_command_is_pinned_tp1_and_loopback(self):
        argv = main.build_vllm_argv()
        joined = " ".join(argv)
        self.assertIn(main.MODEL, argv)
        self.assertIn(main.REVISION, argv)
        self.assertIn("--host 127.0.0.1", joined)
        self.assertIn("--tensor-parallel-size 1", joined)
        self.assertIn("--max-model-len 8192", joined)
        self.assertIn("--no-enable-flashinfer-autotune", argv)

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


if __name__ == "__main__":
    unittest.main()
