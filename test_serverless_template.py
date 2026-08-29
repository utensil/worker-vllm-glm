import json
import unittest

import create_serverless_template as template


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, gpus):
        self.gpus = gpus

    def get(self, _url, **_kwargs):
        return FakeResponse({"gpus": self.gpus})


class ServerlessTemplateTest(unittest.TestCase):
    def test_payload_is_public_secret_free_volume_free_serverless(self):
        value = template.payload()
        self.assertIs(value["isPublic"], True)
        self.assertIs(value["isServerless"], True)
        self.assertEqual(value["ports"], [])
        self.assertEqual(value["volumeInGb"], 0)
        self.assertEqual(value["dockerStartCmd"], [])
        self.assertEqual(value["env"]["RUNPOD_INIT_TIMEOUT"], "1800")
        self.assertEqual(value["env"]["VLLM_STARTUP_TIMEOUT"], "1800")
        self.assertEqual(value["env"]["MAX_CONCURRENCY"], "2")
        self.assertNotIn("HF_TOKEN", value["env"])
        self.assertNotIn("VLLM_API_KEY", value["env"])

    def test_live_create_is_locked_until_image_digest_is_pinned(self):
        with self.assertRaisesRegex(SystemExit, "published digest"):
            template.require_immutable_image(template.IMAGE)
        template.require_immutable_image(
            "ghcr.io/utensil/worker-vllm-glm@sha256:" + "a" * 64
        )

    def test_exact_b300_gate_builds_pool_exclusions(self):
        gpus = [
            {
                "id": template.GPU_TYPE_ID,
                "pool": "BLACKWELL",
                "price": {"serverless": 6.0},
            },
            {
                "id": "NVIDIA B200",
                "pool": "BLACKWELL",
                "price": {"serverless": 4.0},
            },
            {"id": "NVIDIA H200", "pool": "HOPPER", "price": {"serverless": 3.0}},
        ]
        selection = template.exact_serverless_gpu_selection(
            FakeSession(gpus), "redacted"
        )
        self.assertEqual(selection["pools"], ["BLACKWELL"])
        self.assertEqual(selection["excludedTypes"], ["NVIDIA B200"])
        self.assertEqual(selection["count"], 1)

    def test_exact_b300_gate_does_not_substitute(self):
        gpus = [
            {"id": "NVIDIA B200", "pool": "BLACKWELL", "price": {"serverless": 4.0}}
        ]
        with self.assertRaisesRegex(
            SystemExit, "exact Serverless GPU type unavailable"
        ):
            template.exact_serverless_gpu_selection(FakeSession(gpus), "redacted")

    def test_exact_b300_must_be_admitted_to_serverless(self):
        gpus = [{"id": template.GPU_TYPE_ID, "pool": None, "price": {}}]
        with self.assertRaisesRegex(SystemExit, "not admitted"):
            template.exact_serverless_gpu_selection(FakeSession(gpus), "redacted")

    def test_readback_verifies_all_persisted_fields(self):
        value = template.payload()
        stored = dict(value)
        template.verify(stored, value)
        stored["isServerless"] = False
        with self.assertRaisesRegex(SystemExit, "readback mismatch"):
            template.verify(stored, value)

    def test_graphql_storage_normalization_keeps_empty_start_command(self):
        value = template.payload()
        stored = dict(value)
        stored.pop("dockerEntrypoint")
        stored.pop("dockerStartCmd")
        stored["dockerArgs"] = json.dumps({"entrypoint": [], "cmd": []})
        stored["ports"] = ""
        stored["env"] = [
            {"key": key, "value": item} for key, item in value["env"].items()
        ]
        stored["readme"] = value["readme"].rstrip("\n")
        template.verify(stored, value)


if __name__ == "__main__":
    unittest.main()
