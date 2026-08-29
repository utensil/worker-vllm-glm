import json
import unittest
from unittest import mock

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
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return FakeResponse({"gpus": self.gpus})


def admitted_gpu(gpu_id, pool="BLACKWELL", price=6.0):
    return {
        "id": gpu_id,
        "pool": pool,
        "price": {"serverless": price},
        "availability": "HIGH",
        "cudaVersions": [{"version": "13.0", "available": True}],
    }


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

    def test_template_reuse_is_independent_of_gpu_catalog(self):
        expected = template.payload()
        expected["imageName"] = "ghcr.io/utensil/worker-vllm-glm@sha256:" + "a" * 64
        stored = dict(expected, id="public-template")
        with (
            mock.patch.object(template, "_templates", return_value=[stored]),
            mock.patch.object(
                template,
                "exact_serverless_gpu_selection",
                side_effect=AssertionError("template path must not read GPU catalog"),
            ),
        ):
            template_id, action = template.create_or_reuse(
                expected, "redacted", mock.Mock()
            )
        self.assertEqual((template_id, action), ("public-template", "reused"))

    def test_exact_b300_gate_builds_pool_exclusions(self):
        gpus = [
            admitted_gpu(template.GPU_TYPE_ID),
            admitted_gpu("NVIDIA B200", price=4.0),
            admitted_gpu("NVIDIA H200", pool="HOPPER", price=3.0),
        ]
        session = FakeSession(gpus)
        selection = template.exact_serverless_gpu_selection(session, "redacted")
        self.assertEqual(selection["pools"], ["BLACKWELL"])
        self.assertEqual(selection["excludedTypes"], ["NVIDIA B200"])
        self.assertEqual(selection["count"], 1)
        self.assertEqual(selection["allowedCudaVersions"], ["13.0"])
        self.assertEqual(session.get_calls[0][1]["params"]["cudaVersions"], "13.0")

    def test_exact_b300_gate_does_not_substitute(self):
        gpus = [admitted_gpu("NVIDIA B200", price=4.0)]
        with self.assertRaisesRegex(
            SystemExit, "exact Serverless GPU type unavailable"
        ):
            template.exact_serverless_gpu_selection(FakeSession(gpus), "redacted")

    def test_exact_b300_must_be_admitted_to_serverless(self):
        gpu = admitted_gpu(template.GPU_TYPE_ID)
        gpu.update({"pool": None, "price": {}})
        gpus = [gpu]
        with self.assertRaisesRegex(SystemExit, "not admitted"):
            template.exact_serverless_gpu_selection(FakeSession(gpus), "redacted")

    def test_exact_b300_availability_none_fails_closed(self):
        gpu = admitted_gpu(template.GPU_TYPE_ID)
        gpu["availability"] = "NONE"
        with self.assertRaisesRegex(SystemExit, "no current Serverless availability"):
            template.exact_serverless_gpu_selection(FakeSession([gpu]), "redacted")

    def test_exact_b300_requires_available_cuda_13(self):
        gpu = admitted_gpu(template.GPU_TYPE_ID)
        gpu["cudaVersions"] = [
            {"version": "13.0", "available": False},
            {"version": "13.2", "available": True},
        ]
        with self.assertRaisesRegex(SystemExit, "lacks available CUDA 13.0"):
            template.exact_serverless_gpu_selection(FakeSession([gpu]), "redacted")

    def test_temporary_endpoint_pins_admitted_selection_and_bounds_compute(self):
        selection = {
            "pools": ["BLACKWELL"],
            "excludedTypes": ["NVIDIA B200"],
            "count": 1,
            "allowedCudaVersions": ["13.0"],
        }
        endpoint = template.temporary_endpoint_payload("template-public", selection)
        self.assertEqual(endpoint["templateId"], "template-public")
        self.assertEqual(endpoint["gpu"], selection)
        self.assertEqual(endpoint["workers"], {"min": 0, "max": 1, "idleTimeout": 300})
        self.assertEqual(endpoint["scaling"], {"type": "QUEUE_DELAY", "queueDelay": 1})
        self.assertEqual(endpoint["flashboot"], "FLASHBOOT")
        self.assertEqual(endpoint["timeout"], 1_800_000)
        self.assertEqual(set(endpoint), template.ENDPOINT_ALLOWED_KEYS)
        self.assertEqual(set(endpoint["gpu"]), template.ENDPOINT_GPU_KEYS)
        self.assertEqual(set(endpoint["workers"]), template.ENDPOINT_WORKER_KEYS)
        self.assertEqual(set(endpoint["scaling"]), template.ENDPOINT_SCALING_KEYS)
        template.validate_temporary_endpoint_payload(endpoint)

    def test_temporary_endpoint_rejects_unknown_rest_v2_key(self):
        selection = {
            "pools": ["BLACKWELL"],
            "excludedTypes": ["NVIDIA B200"],
            "count": 1,
            "allowedCudaVersions": ["13.0"],
        }
        endpoint = template.temporary_endpoint_payload("template-public", selection)
        endpoint["executionTimeout"] = endpoint.pop("timeout")
        with self.assertRaisesRegex(ValueError, "top-level keys"):
            template.validate_temporary_endpoint_payload(endpoint)

    def test_temporary_endpoint_rejects_unknown_nested_key(self):
        selection = {
            "pools": ["BLACKWELL"],
            "excludedTypes": ["NVIDIA B200"],
            "count": 1,
            "allowedCudaVersions": ["13.0"],
        }
        endpoint = template.temporary_endpoint_payload("template-public", selection)
        endpoint["workers"]["unsupported"] = True
        with self.assertRaisesRegex(ValueError, "worker bounds"):
            template.validate_temporary_endpoint_payload(endpoint)

    def test_temporary_endpoint_rejects_unpinned_selection(self):
        with self.assertRaisesRegex(SystemExit, "not admitted"):
            template.exact_serverless_gpu_selection(
                FakeSession(
                    [
                        {
                            "id": template.GPU_TYPE_ID,
                            "pool": None,
                            "price": {},
                            "availability": "HIGH",
                            "cudaVersions": [{"version": "13.0", "available": True}],
                        }
                    ]
                ),
                "redacted",
            )
        with self.assertRaisesRegex(ValueError, "exact-B300 gate"):
            template.temporary_endpoint_payload(
                "template-public", {"pools": ["BLACKWELL"], "count": 1}
            )

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
