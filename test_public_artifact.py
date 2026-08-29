import json
import unittest

import create_pod_template as template


class PublicArtifactTest(unittest.TestCase):
    def test_template_is_public_secret_free_pod(self):
        value = template.payload()
        self.assertIs(value["isPublic"], True)
        self.assertIs(value["isServerless"], False)
        self.assertEqual(value["env"], {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        })
        self.assertEqual(value["ports"], ["8000/http"])
        self.assertIn(template.MODEL, value["dockerStartCmd"][0])
        self.assertIn(template.REVISION, value["dockerStartCmd"][0])

    def test_graphql_storage_normalization(self):
        expected = template.payload()
        stored = dict(expected)
        stored.pop("dockerEntrypoint")
        stored.pop("dockerStartCmd")
        stored["dockerArgs"] = json.dumps({
            "entrypoint": expected["dockerEntrypoint"],
            "cmd": expected["dockerStartCmd"],
        })
        stored["ports"] = ",".join(expected["ports"])
        stored["env"] = [
            {"key": key, "value": value}
            for key, value in expected["env"].items()
        ]
        stored["readme"] = expected["readme"].rstrip("\n")
        template.verify(stored, expected)


if __name__ == "__main__":
    unittest.main()
