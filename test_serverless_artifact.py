import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent


class ServerlessArtifactTest(unittest.TestCase):
    def test_docker_targets_preserve_pod_default_and_separate_serverless(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertTrue(
            dockerfile.startswith(
                "FROM vllm/vllm-openai@sha256:"
                "2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703 AS base"
            )
        )
        self.assertIn("AS base", dockerfile)
        self.assertIn("FROM base AS serverless", dockerfile)
        self.assertTrue(dockerfile.rstrip().endswith("FROM base AS pod"))
        self.assertIn(
            'ENTRYPOINT ["python3", "-u", "/opt/glm-serverless/main.py"]',
            dockerfile,
        )

    def test_ci_uses_distinct_targets_and_tags(self):
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text()
        self.assertIn("target: pod", workflow)
        self.assertIn("target: serverless", workflow)
        self.assertIn("glm-5.3-flash-nvfp4-serverless-v1", workflow)

    def test_runpod_sdk_is_pinned(self):
        requirements = (ROOT / "serverless" / "requirements.txt").read_text().strip()
        self.assertEqual(requirements, "runpod==1.12.0")


if __name__ == "__main__":
    unittest.main()
