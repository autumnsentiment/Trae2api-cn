import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from src import main as main_module
from src.main import app


AUTH_HEADERS = {"Authorization": "Bearer smoke-key"}


class TraeWorkCustomIngressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_keys = patch.object(main_module, "API_KEYS", ["smoke-key"])
        cls.api_keys.start()
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.__exit__(None, None, None)
        finally:
            cls.api_keys.stop()

    def setUp(self):
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()

    def test_custom_dispatch_defaults_to_verified_remote_transport(self):
        remote = AsyncMock(return_value=JSONResponse({"route": "remote"}))
        raw = AsyncMock(side_effect=AssertionError("raw must be explicitly selected"))

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRAEWORK_CUSTOM_UPSTREAM_MODE", None)
            with (
                patch("src.main._run_remote_with_retry", remote),
                patch("src.main.run_raw_chat", raw),
            ):
                response = asyncio.run(
                    main_module._dispatch_chat(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {"_traework_custom_model": True},
                    )
                )

        self.assertEqual(json.loads(response.body), {"route": "remote"})
        remote.assert_awaited_once()
        raw.assert_not_awaited()

    def test_custom_dispatch_keeps_explicit_raw_as_diagnostic_route(self):
        raw = AsyncMock(return_value=JSONResponse({"route": "raw"}))
        remote = AsyncMock(side_effect=AssertionError("explicit raw must stay raw"))

        with (
            patch("src.main.run_raw_chat", raw),
            patch("src.main._run_remote_with_retry", remote),
        ):
            response = asyncio.run(
                main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    True,
                    {
                        "_traework_custom_model": True,
                        "_traework_upstream_mode": "raw",
                    },
                )
            )

        self.assertEqual(json.loads(response.body), {"route": "raw"})
        raw.assert_awaited_once()
        remote.assert_not_awaited()

    def test_explicit_raw_late_stream_failure_is_not_falsely_cross_retried(self):
        async def failing_body():
            raise RuntimeError("raw iterator failed")
            yield ""  # pragma: no cover

        raw = AsyncMock(
            return_value=StreamingResponse(
                failing_body(), media_type="text/event-stream"
            )
        )
        remote = AsyncMock(
            side_effect=AssertionError("a started raw stream must not cross-retry")
        )

        async def collect():
            with (
                patch("src.main.run_raw_chat", raw),
                patch("src.main._run_remote_with_retry", remote),
            ):
                return [
                    chunk
                    async for chunk in main_module._deferred_dispatch_stream(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        {
                            "_traework_custom_model": True,
                            "_traework_upstream_mode": "raw",
                        },
                    )
                ]

        joined = "".join(asyncio.run(collect()))
        self.assertIn("raw iterator failed", joined)
        self.assertIn("data: [DONE]", joined)
        raw.assert_awaited_once()
        remote.assert_not_awaited()

    def test_openai_agent_metadata_keeps_chat_completions_contract(self):
        async def fake_dispatch(messages, model, stream, options=None):
            return JSONResponse(
                {
                    "id": "chatcmpl-openai",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )

        with (
            patch("src.main._dispatch_chat", new=fake_dispatch),
            patch("src.main._fetch_used_credits", new=_async_none),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "glm-5.3",
                    "messages": [{"role": "user", "content": "hello"}],
                    "agent_type": "coding",
                    "chat_session_id": "session-1",
                    "render_context": {"theme": "dark"},
                    "mcp_tool_list": [{"name": "read_file"}],
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "pong")
        self.assertNotIn("response", body)

    def test_connectivity_endpoint_returns_model_capabilities(self):
        with patch(
            "src.main.trae_client.get_models",
            new=lambda force=False: _async_value(
                [{"id": "glm-5.3", "object": "model", "owned_by": "trae"}]
            ),
        ):
            response = self.client.post(
                "/api/agent/v3/custom_model_connectivity_check",
                json={"custom_model": {"config_name": "glm-5.3"}},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["config_name"], "glm-5.3")
        self.assertFalse(body["model"]["is_preset"])
        self.assertTrue(body["model"]["is_custom_base_url"])
        selected = next(
            item for item in body["models"] if item["id"] == "glm-5.3"
        )
        self.assertFalse(selected["is_preset"])
        self.assertTrue(selected["is_custom_base_url"])
        self.assertTrue(body["model"]["features"]["tool_calls"]["enable"])

    def test_raw_stream_converts_openai_tool_call_for_traework_toolhost(self):
        captured = {}

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            captured["model"] = model
            captured["options"] = options

            async def chunks():
                yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_write","type":"function","function":{"name":"writeFile","arguments":"{\\"path\\":\\"C:/workspace/output.txt\\",\\"content\\":\\"ok\\"}"}}]},"finish_reason":null}]}\n\n'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")

        request_body = {
            "config_name": "glm-5.3",
            "conversation_id": "tw-session-1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "create output.txt"}],
                }
            ],
            "model_name": "glm-5.3",
            "session_id": "tw-session-1",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "writeFile",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
        }
        with (
            patch("src.main._dispatch_chat", new=fake_dispatch),
            patch("src.main._fetch_used_credits", new=_async_none),
        ):
            response = self.client.post(
                "/api/ide/v2/llm_raw_chat",
                json=request_body,
                headers={**AUTH_HEADERS, "x-bridge-transport": "aha"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["model"], "glm-5.3")
        self.assertTrue(captured["options"]["_traework_custom_model"])
        self.assertEqual(captured["options"]["session_id"], "tw-session-1")
        self.assertIn("event: output", response.text)
        self.assertIn('"id":"call_write"', response.text)
        self.assertIn('"name":"writeFile"', response.text)
        self.assertIn('C:/workspace/output.txt', response.text)
        self.assertIn("event: token_usage", response.text)
        self.assertIn("event: done", response.text)
        self.assertNotIn("chat.completion.chunk", response.text)

    def test_nonstream_renderer_tool_result_reaches_continuation(self):
        captured = {}

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            return JSONResponse(
                {
                    "id": "chatcmpl-tw",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "file confirmed",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 3,
                        "total_tokens": 11,
                    },
                }
            )

        with (
            patch("src.main._dispatch_chat", new=fake_dispatch),
            patch("src.main._fetch_used_credits", new=_async_none),
        ):
            response = self.client.post(
                "/api/ide/v2/llm_raw_chat",
                headers=AUTH_HEADERS,
                json={
                    "config_name": "glm-5.3",
                    "model_name": "glm-5.3",
                    "session_id": "tw-continuation-1",
                    "stream": False,
                    "messages": [
                        {
                            "role": 2,
                            "content": [
                                {
                                    "type": "tool_use",
                                    "toolCallId": "call_write_1",
                                    "name": "writeFile",
                                    "parameters": {"path": "C:/workspace/a.txt"},
                                }
                            ],
                        },
                        {
                            "role": 1,
                            "content": [
                                {
                                    "type": "tool_result",
                                    "toolCallId": "call_write_1",
                                    "value": [{"type": "text", "value": "written"}],
                                    "isError": False,
                                }
                            ],
                        },
                    ],
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["messages"][0]["content"][0]["type"], "tool_use")
        self.assertEqual(captured["messages"][1]["content"][0]["type"], "tool_result")
        body = response.json()
        self.assertEqual(body["response"], "file confirmed")
        self.assertEqual(body["finish_reason"], "stop")
        self.assertEqual(body["usage"]["total_tokens"], 11)

    def test_two_round_tool_loop_uses_renderer_result_and_returns_final_text(self):
        """Exercise the actual TraeWork toolhost loop across two requests.

        The first request must expose a callable renderer tool block.  The
        second request is the client-owned toolhost continuation and must keep
        the same session/call identity while carrying the renderer result back
        to the model.  This is intentionally a single test so a regression in
        either half cannot hide behind two independent unit tests.
        """
        dispatch_calls = []

        async def fake_dispatch(messages, model, stream, options=None):
            dispatch_calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "stream": stream,
                    "options": dict(options or {}),
                }
            )
            if len(dispatch_calls) == 1:
                self.assertTrue(stream)

                async def chunks():
                    yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                    yield (
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_loop_1",'
                        '"type":"function","function":{"name":"writeFile","arguments":"{\\"path\\":\\"C:/workspace/loop.txt\\",\\"content\\":\\"ok\\"}"}}]},'
                        '"finish_reason":null}]}\n\n'
                    )
                    yield (
                        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],'
                        '"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(chunks(), media_type="text/event-stream")

            self.assertFalse(stream)
            self.assertEqual(model, "glm-5.3")
            self.assertEqual(options.get("session_id"), "tw-loop-1")
            self.assertEqual(len(messages), 3)
            assistant_blocks = messages[1]["content"]
            result_blocks = messages[2]["content"]
            self.assertEqual(assistant_blocks[0]["type"], "tool_use")
            self.assertEqual(assistant_blocks[0]["toolCallId"], "call_loop_1")
            self.assertEqual(assistant_blocks[0]["name"], "writeFile")
            self.assertEqual(
                assistant_blocks[0]["parameters"],
                {"path": "C:/workspace/loop.txt", "content": "ok"},
            )
            self.assertEqual(result_blocks[0]["type"], "tool_result")
            self.assertEqual(result_blocks[0]["toolCallId"], "call_loop_1")
            self.assertFalse(result_blocks[0]["isError"])

            return JSONResponse(
                {
                    "id": "chatcmpl-loop-final",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "loop file written",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                }
            )

        first_request = {
            "config_name": "glm-5.3",
            "conversation_id": "tw-loop-1",
            "model_name": "glm-5.3",
            "session_id": "tw-loop-1",
            "stream": True,
            "messages": [
                {
                    "role": 1,
                    "content": [{"type": "text", "text": "create loop.txt"}],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "writeFile",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
        }

        with (
            patch("src.main._dispatch_chat", new=fake_dispatch),
            patch("src.main._fetch_used_credits", new=_async_none),
        ):
            first_response = self.client.post(
                "/api/ide/v2/llm_raw_chat",
                json=first_request,
                headers={**AUTH_HEADERS, "x-bridge-transport": "aha"},
            )

            self.assertEqual(first_response.status_code, 200, first_response.text)
            self.assertIn("event: output", first_response.text)
            self.assertIn('"id":"call_loop_1"', first_response.text)
            self.assertIn('"name":"writeFile"', first_response.text)
            self.assertIn("event: done", first_response.text)
            self.assertIn('"finish_reason":"tool_calls"', first_response.text)

            second_response = self.client.post(
                "/api/ide/v2/llm_raw_chat",
                json={
                    "config_name": "glm-5.3",
                    "conversation_id": "tw-loop-1",
                    "model_name": "glm-5.3",
                    "session_id": "tw-loop-1",
                    "stream": False,
                    "messages": [
                        first_request["messages"][0],
                        {
                            "role": 2,
                            "content": [
                                {
                                    "type": "tool_use",
                                    "toolCallId": "call_loop_1",
                                    "name": "writeFile",
                                    "parameters": {
                                        "path": "C:/workspace/loop.txt",
                                        "content": "ok",
                                    },
                                }
                            ],
                        },
                        {
                            "role": 1,
                            "content": [
                                {
                                    "type": "tool_result",
                                    "toolCallId": "call_loop_1",
                                    "value": [{"type": "text", "value": "written"}],
                                    "isError": False,
                                }
                            ],
                        },
                    ],
                },
                headers={**AUTH_HEADERS, "x-bridge-transport": "aha"},
            )

        self.assertEqual(len(dispatch_calls), 2)
        self.assertEqual(second_response.status_code, 200, second_response.text)
        final = second_response.json()
        self.assertEqual(final["response"], "loop file written")
        self.assertEqual(final["finish_reason"], "stop")
        self.assertEqual(final["usage"]["total_tokens"], 16)


async def _async_none(*_args, **_kwargs):
    return None


async def _async_value(value):
    return value


if __name__ == "__main__":
    unittest.main()
