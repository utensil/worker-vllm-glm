import unittest

import create_serverless_b200_cached_template as template


class ServerlessB200CachedTemplateTest(unittest.TestCase):
    def test_payload_is_distinct_cache_required_and_secret_free(self):
        value = template.payload()
        self.assertIn("Cached 2xB200 TP2", value["name"])
        self.assertEqual(value["env"]["TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(value["env"]["RUNPOD_MODEL_CACHE_REQUIRED"], "true")
        self.assertEqual(value["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(value["env"]["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(value["ports"], [])
        self.assertEqual(value["volumeInGb"], 0)
        self.assertIs(value["isPublic"], True)
        self.assertIs(value["isServerless"], True)
        self.assertNotIn("HF_TOKEN", value["env"])
        self.assertNotIn("RUNPOD_API_KEY", value["env"])

    def test_creation_uses_published_amd64_digest(self):
        template.b200.common.require_immutable_image(template.IMAGE)
        self.assertEqual(
            template.IMAGE,
            "ghcr.io/utensil/worker-vllm-glm@"
            "sha256:577394a87422627b322d7c64bb3d589fd5a9b2f1d25bda87e7048a1855364669",
        )


if __name__ == "__main__":
    unittest.main()
