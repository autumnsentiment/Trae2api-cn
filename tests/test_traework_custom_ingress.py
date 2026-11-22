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

    def test_trae344_raw_function_routes_to_shared_traecode_chain(self):
        ide = AsyncMock(return_value=JSONResponse({"route": "traecode-raw-v2"}))
        legacy_raw = AsyncMock(
            side_effect=AssertionError("3.4.4 raw-v2 must use TraeCode adapter")
        )
        remote = AsyncMock(
            side_effect=AssertionError("3.4.4 raw-v2 must not use Remote")
        )

        with (
            patch("src.main.run_ide_chat", ide),
            patch("src.main.run_raw_chat", legacy_raw),
            patch("src.main._run_remote_with_retry", remote),
        ):
            response = asyncio.run(
                main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.2",
                    False,
                    {
                        "_traework_custom_model": True,
                        "_traework_upstream_mode": "raw",
                        "_traecode_raw_chat_function": "chat_v3",
                        "request_seq": 0,
                    },
                )
            )

        self.assertEqual(json.loads(response.body), {"route": "traecode-raw-v2"})
        ide.assert_awaited_once()
        self.assertEqual(ide.await_args.args[1], "glm-5.2")
        self.assertEqual(ide.await_args.args[3]["request_seq"], 0)
        legacy_raw.assert_not_awaited()
        remote.assert_not_awaited()

    def test_native_raw_catalog_rejection_returns_scope_error(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                error = main_module.raw_client.TraeCodeModelUnavailable(
                    "glm-5.3",
                    "chat_v3",
                    reason="custom_provider_requires_client_direct",
                )
                ide = AsyncMock(side_effect=error)
                remote = AsyncMock()
                with (
                    patch("src.main._run_ide_with_retry", ide),
                    patch("src.main._run_remote_with_retry", remote),
                ):
                    response = asyncio.run(
                        main_module._dispatch_chat(
                            [{"role": "user", "content": "hello"}],
                            "glm-5.3",
                            stream,
                            {
                                "_traework_custom_model": True,
                                "_traecode_raw_chat_function": "chat_v3",
                            },
                        )
                    )
                self.assertEqual(response.status_code, 403)
                payload = json.loads(response.body)
                self.assertEqual(
                    payload["error"]["type"], "model_not_available_in_scope"
                )
                self.assertIn(
                    "custom_provider_requires_client_direct",
                    payload["error"]["message"],
                )
                ide.assert_awaited_once()
                remote.assert_not_awaited()

    def test_native_scope_stream_error_does_not_fall_back_to_remote(self):
        for late in (False, True):
            with self.subTest(late=late):
                error = main_module.raw_client.TraeCodeModelUnavailable(
                    "glm-5.3", "chat_v3"
                )

                async def failing_body():
                    raise error
                    yield ""  # pragma: no cover

                ide = (
                    AsyncMock(
                        return_value=StreamingResponse(
                            failing_body(), media_type="text/event-stream"
                        )
                    )
                    if late
                    else AsyncMock(side_effect=error)
                )
                remote = AsyncMock(
                    side_effect=AssertionError("scope rejection must not replay")
                )

                async def collect():
                    with (
                        patch("src.main._run_ide_with_retry", ide),
                        patch("src.main._run_remote_with_retry", remote),
                        patch(
                            "src.main._remote_fallback_allowed_for_request",
                            return_value=True,
                        ),
                    ):
                        return [
                            chunk
                            async for chunk in main_module._deferred_dispatch_stream(
                                [{"role": "user", "content": "hello"}],
                                "glm-5.3",
                                {
                                    "_traework_custom_model": True,
                                    "_traework_upstream_mode": "ide",
                                    "_upstream_mode": "ide",
                                    "_traecode_raw_chat_function": "chat_v3",
                                },
                            )
                        ]

                joined = "".join(asyncio.run(collect()))
                errors = [
                    item["error"]
                    for item in main_module._iter_sse_payloads(joined)
                    if "error" in item
                ]
                self.assertEqual(len(errors), 1)
                self.assertEqual(
                    errors[0]["type"], "model_not_available_in_scope"
                )
                self.assertIn("data: [DONE]", joined)
                ide.assert_awaited_once()
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

    def test_explicit_raw_stream_error_preserves_upstream_code_and_request_id(self):
        async def failing_body():
            raise main_module.UpstreamEventError(
                "We're sorry, the param is invalid.",
                code=4001,
                retryable=True,
            )
            yield ""  # pragma: no cover

        raw = AsyncMock(
            return_value=StreamingResponse(
                failing_body(), media_type="text/event-stream"
            )
        )
        remote = AsyncMock(
            side_effect=AssertionError("explicit raw stream must not cross-retry")
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
                            "_relay_request_id": "req-tool-4001",
                        },
                    )
                ]

        joined = "".join(asyncio.run(collect()))
        payloads = list(main_module._iter_sse_payloads(joined))
        errors = [item["error"] for item in payloads if "error" in item]
        self.assertTrue(errors)
        self.assertEqual(errors[0]["code"], 4001)
        self.assertEqual(errors[0]["upstream_code"], 4001)
        self.assertEqual(errors[0]["request_id"], "req-tool-4001")
        self.assertTrue(errors[0]["retryable"])
        self.assertIn("4001", errors[0]["message"])
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
        self.assertNotIn("reasoning", body["model"]["features"])
        self.assertNotIn("context_window_size", body["model"])
        self.assertNotIn("prompt_max_tokens", body["model"])

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

    def test_native_raw_ingress_keeps_tools_and_tool_history_on_wire(self):
        captured = {}
        response_payload = {
            "id": "chatcmpl-native-tools",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "read ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            captured["model"] = model
            captured["options"] = dict(options or {})
            return JSONResponse(response_payload)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        tool_calls = [
            {
                "id": "call-native-1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
            }
        ]
        request_body = {
            "config_name": "DeepSeek-V4-Pro-Official",
            "model_name": "DeepSeek-V4-Pro-Official",
            "raw_chat_function": "chat_v3",
            "conversation_id": "native-tools-1",
            "session_id": "native-tools-1",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Read README.md"}],
                },
                {"role": "assistant", "content": "", "tool_calls": tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": "call-native-1",
                    "name": "read_file",
                    "content": "relay",
                },
            ],
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }

        with (
            patch("src.main._dispatch_chat", new=fake_dispatch),
            patch("src.main._fetch_used_credits", new=_async_none),
        ):
            response = self.client.post(
                "/api/ide/v2/llm_raw_chat",
                json=request_body,
                headers={
                    **AUTH_HEADERS,
                    "x-bridge-transport": "aha",
                    "x-ide-function": "chat_v3",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["model"], "DeepSeek-V4-Pro-Official")
        options = captured["options"]
        self.assertTrue(options["_traework_custom_model"])
        self.assertEqual(options["_traecode_raw_chat_function"], "chat_v3")
        self.assertTrue(options["_tool_protocol_requested"])
        body = main_module.raw_client.build_traecode_chat_body(
            captured["messages"],
            captured["model"],
            options={**options, "_traecode_native_tools": True},
        )
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(body["tool_choice"], "auto")
        # raw-v2 rejects ``parallel_tool_calls`` at the gateway; the relay
        # enforces the caller's one-call policy while accumulating the stream.
        self.assertNotIn("parallel_tool_calls", body)
        self.assertEqual(body["messages"][1]["tool_calls"][0]["id"], "call-native-1")
        self.assertEqual(body["messages"][2]["role"], "tool")
        self.assertEqual(body["messages"][2]["tool_call_id"], "call-native-1")
        self.assertEqual(json.loads(response.text)["response"], "read ok")

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
