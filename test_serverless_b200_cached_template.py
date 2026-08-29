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

    def test_creation_stays_locked_until_digest_is_pinned(self):
        with self.assertRaisesRegex(SystemExit, "published digest"):
            template.b200.common.require_immutable_image(template.IMAGE)


if __name__ == "__main__":
    unittest.main()
