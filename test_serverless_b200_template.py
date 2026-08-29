import copy
import unittest

import create_serverless_b200_template as template


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, gpus):
        self.gpus = gpus
        self.params = None

    def get(self, _url, **kwargs):
        self.params = kwargs["params"]
        return FakeResponse({"gpus": self.gpus})


def gpu(gpu_id, *, pool="BLACKWELL_180", availability="LOW", price=8.64):
    return {
        "id": gpu_id,
        "pool": pool,
        "availability": availability,
        "price": {"serverless": price},
        "cudaVersions": [{"version": "13.0", "available": True}],
    }


class ServerlessB200TemplateTest(unittest.TestCase):
    def test_payload_is_distinct_tp2_secret_free_and_volume_free(self):
        value = template.payload()
        self.assertIn("2xB200 TP2", value["name"])
        self.assertEqual(value["env"]["TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(value["ports"], [])
        self.assertEqual(value["volumeInGb"], 0)
        self.assertIs(value["isPublic"], True)
        self.assertIs(value["isServerless"], True)
        self.assertNotIn("HF_TOKEN", value["env"])
        self.assertNotIn("RUNPOD_API_KEY", value["env"])

    def test_creation_requires_published_digest(self):
        with self.assertRaisesRegex(SystemExit, "published digest"):
            template.common.require_immutable_image(
                "ghcr.io/utensil/worker-vllm-glm:serverless-candidate"
            )
        template.common.require_immutable_image(template.IMAGE)

    def test_exact_count2_b200_gate_and_endpoint(self):
        session = FakeSession([gpu("NVIDIA B200"), gpu("NVIDIA B300 SXM6 AC")])
        selection = template.exact_serverless_gpu_selection(session, "redacted")
        self.assertEqual(session.params["count"], 2)
        self.assertEqual(
            selection,
            {
                "pools": ["BLACKWELL_180"],
                "excludedTypes": ["NVIDIA B300 SXM6 AC"],
                "count": 2,
                "allowedCudaVersions": ["13.0"],
            },
        )
        endpoint = template.temporary_endpoint_payload("public-b200", selection)
        self.assertEqual(endpoint["gpu"]["count"], 2)
        self.assertEqual(endpoint["workers"], {"min": 0, "max": 1, "idleTimeout": 300})
        template.validate_temporary_endpoint_payload(endpoint)

    def test_gate_never_substitutes_and_rejects_unavailable_or_bad_price(self):
        with self.assertRaisesRegex(
            SystemExit, "exact Serverless GPU type unavailable"
        ):
            template.exact_serverless_gpu_selection(
                FakeSession([gpu("NVIDIA H200")]), "redacted"
            )
        with self.assertRaisesRegex(SystemExit, "no current Serverless availability"):
            template.exact_serverless_gpu_selection(
                FakeSession([gpu("NVIDIA B200", availability="NONE")]), "redacted"
            )
        for price in (None, 0, float("nan"), True):
            with (
                self.subTest(price=price),
                self.assertRaisesRegex(SystemExit, "not admitted"),
            ):
                template.exact_serverless_gpu_selection(
                    FakeSession([gpu("NVIDIA B200", price=price)]), "redacted"
                )

    def test_endpoint_rejects_count_or_unknown_key_mutations(self):
        selection = {
            "pools": ["BLACKWELL_180"],
            "excludedTypes": [],
            "count": 2,
            "allowedCudaVersions": ["13.0"],
        }
        endpoint = template.temporary_endpoint_payload("public-b200", selection)
        changed = copy.deepcopy(endpoint)
        changed["gpu"]["count"] = 1
        with self.assertRaisesRegex(ValueError, "exact-2xB200"):
            template.validate_temporary_endpoint_payload(changed)
        changed = copy.deepcopy(endpoint)
        changed["gpu"]["fallback"] = True
        with self.assertRaisesRegex(ValueError, "GPU contract"):
            template.validate_temporary_endpoint_payload(changed)


if __name__ == "__main__":
    unittest.main()
