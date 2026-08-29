import json
import os
import unittest
import urllib.error
from unittest import mock

from serverless import handler


class FakeProcess:
    def __init__(self, return_code=None):
        self.return_code = return_code

    def poll(self):
        return self.return_code


class FakeResponse:
    def __init__(self, value, status=200):
        self.value = value
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=-1):
        return json.dumps(self.value).encode("utf-8")

    def readline(self, _limit=-1):
        if getattr(self, "_read", False):
            return b""
        self._read = True
        return self.value if isinstance(self.value, bytes) else self.read()


class ServerlessHandlerTest(unittest.TestCase):
    def setUp(self):
        handler.backend_process = FakeProcess()

    def tearDown(self):
        handler.backend_process = None

    def test_openai_chat_injects_pinned_model_and_calls_loopback(self):
        job = {
            "input": {
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "messages": [{"role": "user", "content": "2+2"}],
                    "max_tokens": 8,
                },
            }
        }
        with mock.patch.object(
            handler.urllib.request, "urlopen", return_value=FakeResponse({"id": "ok"})
        ) as opened:
            self.assertEqual(list(handler.handler(job)), [{"id": "ok"}])
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        body = json.loads(request.data)
        self.assertEqual(body["model"], handler.MODEL)
        self.assertNotIn("Authorization", dict(request.header_items()))

    def test_models_is_get_without_body(self):
        spec = handler.normalize_job({"input": {"openai_route": "/v1/models"}})
        self.assertEqual(
            (spec.route, spec.method, spec.body), ("/v1/models", "GET", None)
        )

    def test_native_messages_and_sampling_params(self):
        spec = handler.normalize_job(
            {
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "sampling_params": {"temperature": 0.2},
                }
            }
        )
        self.assertEqual(spec.route, "/v1/chat/completions")
        self.assertEqual(spec.body["temperature"], 0.2)

    def test_sse_streaming_forwards_raw_chunks(self):
        response = FakeResponse(b'data: {"choices":[]}\n\n')
        with mock.patch.object(
            handler.urllib.request, "urlopen", return_value=response
        ):
            chunks = list(
                handler.handler(
                    {
                        "input": {
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": True,
                        }
                    }
                )
            )
        self.assertEqual(chunks, ['data: {"choices":[]}\n\n'])

    def test_arbitrary_routes_fail_closed(self):
        route = list(handler.handler({"input": {"route": "/metrics"}}))
        self.assertEqual(route[0]["error"]["type"], "validation_error")

    def test_tool_and_remote_image_url_payloads_pass_through(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        },
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
        spec = handler.normalize_job(
            {"input": {"openai_route": "/v1/chat/completions", "openai_input": body}}
        )
        self.assertEqual(spec.body["tools"], body["tools"])
        self.assertEqual(spec.body["messages"], body["messages"])

    def test_wrong_model_and_bad_method_fail_closed(self):
        wrong_model = list(
            handler.handler(
                {
                    "input": {
                        "openai_input": {
                            "model": "another/model",
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    }
                }
            )
        )
        bad_method = list(
            handler.handler({"input": {"route": "/v1/models", "method": "POST"}})
        )
        self.assertEqual(wrong_model[0]["error"]["type"], "validation_error")
        self.assertEqual(bad_method[0]["error"]["type"], "validation_error")

    def test_dead_backend_fails_before_http(self):
        handler.backend_process = FakeProcess(1)
        with mock.patch.object(handler.urllib.request, "urlopen") as opened:
            result = list(handler.handler({"input": {"openai_route": "/v1/models"}}))
        self.assertEqual(result[0]["error"]["type"], "worker_unhealthy")
        opened.assert_not_called()

    def test_backend_http_error_is_bounded_and_does_not_echo_body(self):
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8000/v1/chat/completions", 400, "bad", {}, None
        )
        with mock.patch.object(handler.urllib.request, "urlopen", side_effect=error):
            result = list(
                handler.handler(
                    {
                        "input": {
                            "messages": [{"role": "user", "content": "private prompt"}]
                        }
                    }
                )
            )
        self.assertEqual(result[0]["error"]["code"], 400)
        self.assertNotIn("private prompt", json.dumps(result))

    def test_invalid_timeout_configuration_returns_worker_error(self):
        with mock.patch.dict(os.environ, {"BACKEND_REQUEST_TIMEOUT_SECONDS": "bad"}):
            result = list(handler.handler({"input": {"openai_route": "/v1/models"}}))
        self.assertEqual(result[0]["error"]["type"], "worker_error")


if __name__ == "__main__":
    unittest.main()
