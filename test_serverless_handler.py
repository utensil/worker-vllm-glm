import asyncio
import inspect
import json
import os
import unittest
from unittest import mock

from serverless import handler


class FakeProcess:
    def __init__(self, return_code=None):
        self.return_code = return_code

    def poll(self):
        return self.return_code


class FakeContent:
    def __init__(self, payload=b"", chunks=None, read_chunks=None):
        self.payload = payload
        self.chunks = chunks or []
        self.read_chunks = read_chunks

    async def read(self, limit=-1):
        return self.payload if limit < 0 else self.payload[:limit]

    async def iter_any(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk

    async def iter_chunked(self, size):
        chunks = self.read_chunks
        if chunks is None:
            chunks = [
                self.payload[index : index + size]
                for index in range(0, len(self.payload), size)
            ]
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk


class FakeResponse:
    def __init__(self, value=None, *, status=200, chunks=None, read_chunks=None):
        payload = b"" if value is None else json.dumps(value).encode("utf-8")
        self.status = status
        self.content = FakeContent(payload, chunks, read_chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.request_call = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def request(self, *args, **kwargs):
        self.request_call = (args, kwargs)
        return self.response


async def collect(job):
    return [item async for item in handler.handler(job)]


class ServerlessHandlerTest(unittest.TestCase):
    def setUp(self):
        handler.backend_process = FakeProcess()

    def tearDown(self):
        handler.backend_process = None

    def test_handler_is_async_generator(self):
        self.assertTrue(inspect.isasyncgenfunction(handler.handler))

    def test_pinned_runpod_sdk_consumes_async_generator(self):
        from runpod.serverless.modules.rp_job import run_job_generator

        async def collect_with_sdk():
            job = {"id": "unit-job", "input": {"openai_route": "/v1/models"}}
            return [item async for item in run_job_generator(handler.handler, job)]

        session = FakeSession(FakeResponse({"id": "ok"}))
        with mock.patch.object(handler.aiohttp, "ClientSession", return_value=session):
            result = asyncio.run(collect_with_sdk())
        self.assertEqual(result, [{"output": {"id": "ok"}}])

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
        session = FakeSession(FakeResponse({"id": "ok"}))
        with mock.patch.object(handler.aiohttp, "ClientSession", return_value=session):
            self.assertEqual(asyncio.run(collect(job)), [{"id": "ok"}])
        args, kwargs = session.request_call
        self.assertEqual(args, ("POST", "http://127.0.0.1:8000/v1/chat/completions"))
        self.assertEqual(kwargs["json"]["model"], handler.MODEL)
        self.assertNotIn("Authorization", kwargs["headers"])

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
        response = FakeResponse(
            chunks=[b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"]
        )
        with mock.patch.object(
            handler.aiohttp, "ClientSession", return_value=FakeSession(response)
        ):
            chunks = asyncio.run(
                collect(
                    {
                        "input": {
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": True,
                        }
                    }
                )
            )
        self.assertEqual(chunks, ['data: {"choices":[]}\n\n', "data: [DONE]\n\n"])

    def test_sse_streaming_preserves_utf8_split_across_transport_chunks(self):
        event = 'data: {"delta":"\u4f60"}\n\n'.encode()
        split_at = event.index("\u4f60".encode()) + 1
        response = FakeResponse(chunks=[event[:split_at], event[split_at:]])
        with mock.patch.object(
            handler.aiohttp, "ClientSession", return_value=FakeSession(response)
        ):
            chunks = asyncio.run(
                collect(
                    {
                        "input": {
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": True,
                        }
                    }
                )
            )
        self.assertEqual("".join(chunks), event.decode())
        self.assertNotIn("\ufffd", "".join(chunks))

    def test_nonstream_json_accumulates_multiple_transport_chunks_to_eof(self):
        response = FakeResponse(read_chunks=[b'{"id":', b'"split",', b'"value":4}'])
        with mock.patch.object(
            handler.aiohttp, "ClientSession", return_value=FakeSession(response)
        ):
            result = asyncio.run(collect({"input": {"openai_route": "/v1/models"}}))
        self.assertEqual(result, [{"id": "split", "value": 4}])

    def test_nonstream_json_fails_as_soon_as_accumulated_body_exceeds_limit(self):
        response = FakeResponse(
            read_chunks=[b"x" * handler.MAX_BODY_BYTES, b"overflow"]
        )
        with mock.patch.object(
            handler.aiohttp, "ClientSession", return_value=FakeSession(response)
        ):
            result = asyncio.run(collect({"input": {"openai_route": "/v1/models"}}))
        self.assertEqual(result[0]["error"]["message"], "vLLM response exceeds 2 MiB")

    def test_two_jobs_overlap_without_blocking_event_loop(self):
        tracker = {"active": 0, "max_active": 0}

        class OverlapResponse(FakeResponse):
            async def __aenter__(self):
                tracker["active"] += 1
                tracker["max_active"] = max(tracker["max_active"], tracker["active"])
                await asyncio.sleep(0.02)
                return self

            async def __aexit__(self, *_args):
                tracker["active"] -= 1
                return False

        def session_factory(**_kwargs):
            return FakeSession(OverlapResponse({"id": "ok"}))

        async def run_both():
            job = {"input": {"openai_route": "/v1/models"}}
            return await asyncio.gather(collect(job), collect(job))

        with mock.patch.object(
            handler.aiohttp, "ClientSession", side_effect=session_factory
        ):
            results = asyncio.run(run_both())
        self.assertEqual(results, [[{"id": "ok"}], [{"id": "ok"}]])
        self.assertEqual(tracker["max_active"], 2)

    def test_arbitrary_routes_fail_closed(self):
        route = asyncio.run(collect({"input": {"route": "/metrics"}}))
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
        wrong_model = asyncio.run(
            collect(
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
        bad_method = asyncio.run(
            collect({"input": {"route": "/v1/models", "method": "POST"}})
        )
        self.assertEqual(wrong_model[0]["error"]["type"], "validation_error")
        self.assertEqual(bad_method[0]["error"]["type"], "validation_error")

    def test_dead_backend_fails_before_http(self):
        handler.backend_process = FakeProcess(1)
        with mock.patch.object(handler.aiohttp, "ClientSession") as session:
            result = asyncio.run(collect({"input": {"openai_route": "/v1/models"}}))
        self.assertEqual(result[0]["error"]["type"], "worker_unhealthy")
        session.assert_not_called()

    def test_backend_http_error_is_bounded_and_does_not_echo_body(self):
        session = FakeSession(FakeResponse(status=400))
        with mock.patch.object(handler.aiohttp, "ClientSession", return_value=session):
            result = asyncio.run(
                collect(
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
            result = asyncio.run(collect({"input": {"openai_route": "/v1/models"}}))
        self.assertEqual(result[0]["error"]["type"], "worker_error")


if __name__ == "__main__":
    unittest.main()
