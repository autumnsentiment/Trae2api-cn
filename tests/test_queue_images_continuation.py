"""Queue telemetry, image inputs, auto-continuation, and /v1/messages."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from src import anthropic_api, continuation, images, main as main_module, sse
from src.main import app


def _parse_chunks(chunks):
    parsed, done = [], False
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            done = True
            continue
        if chunk.startswith("data: "):
            parsed.append(json.loads(chunk[6:].strip()))
    return parsed, done


def _visible_text(chunks):
    return "".join(
        choice.get("delta", {}).get("content", "")
        for event in chunks
        for choice in event.get("choices", [])
    )


async def _async_chunks(chunks):
    """Adapt a synchronous fixture iterator to the async stream contract."""

    for chunk in chunks:
        yield chunk


class _LineResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        for line in self._lines:
            yield line


class QueueTelemetryTests(unittest.TestCase):
    def test_queue_position_is_assistant_text(self):
        response = _LineResponse(
            [
                'data: {"event":"request_wait_in_queue","position":2}',
                'data: {"event":"request_wait_in_queue","position":1}',
                'data: {"response":"ready"}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        metadata: dict = {}
        chunks = asyncio.run(
            _collect(sse.translate_ide_stream(response, "m", True, upstream_metadata=metadata))
        )
        parsed, done = _parse_chunks(chunks)
        text = _visible_text(parsed)
        self.assertEqual(text, "排队中，当前位置：2\n排队中，当前位置：1\nready")
        self.assertTrue(done)
        # The raw protocol exposes queue status as assistant content. It is
        # intentionally not emitted as a synthetic SSE comment.
        self.assertFalse(any(chunk.startswith(":") for chunk in chunks))
        self.assertEqual(metadata.get("queue_position"), 1)

    def test_queue_position_is_recorded_for_nonstream(self):
        response = _LineResponse(
            [
                'data: {"event":"request_wait_in_queue","position":4}',
                'data: {"response":"ready"}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        metadata: dict = {}
        result = asyncio.run(
            sse.collect_nonstream_ide(response, "m", upstream_metadata=metadata)
        )
        # The non-stream collector follows the historical behavior and
        # exposes queue position through metadata while returning only the
        # completed answer content.
        self.assertEqual(result["choices"][0]["message"]["content"], "ready")
        self.assertEqual(metadata.get("queue_position"), 4)


async def _collect(stream):
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


class ImageNormalizationTests(unittest.TestCase):
    def test_openai_data_url_becomes_renderer_block(self):
        block = images.from_openai_part(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,QUJD"},
            }
        )
        self.assertEqual(block["type"], "image_url")
        self.assertEqual(block["value"]["mimeType"], "image/png")
        self.assertEqual(block["value"]["data"], "QUJD")

    def test_anthropic_base64_block_becomes_renderer_block(self):
        block = images.from_anthropic_block(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
            }
        )
        self.assertEqual(block["value"]["mimeType"], "image/jpeg")

    def test_messages_have_images_detects_openai_and_anthropic_parts(self):
        self.assertTrue(
            images.messages_have_images(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hi"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,QUJD"},
                            },
                        ],
                    }
                ]
            )
        )
        self.assertTrue(
            images.messages_have_images(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "QUJD",
                                },
                            }
                        ],
                    }
                ]
            )
        )
        self.assertFalse(
            images.messages_have_images([{"role": "user", "content": "plain"}])
        )

    def test_raw_history_keeps_image_blocks(self):
        from src import raw_client

        built = raw_client.build_raw_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        },
                    ],
                }
            ]
        )
        user_message = built[-1]
        self.assertEqual(user_message["role"], "user")
        kinds = [block["type"] for block in user_message["content"]]
        self.assertIn("text", kinds)
        self.assertIn("image_url", kinds)
        image_block = next(
            block for block in user_message["content"] if block["type"] == "image_url"
        )
        self.assertEqual(image_block["value"]["data"], "QUJD")

    def test_multimodal_switch_prefers_configured_override(self):
        with patch.dict(
            main_module.os.environ, {"TRAE_MULTIMODAL_MODEL": "kimi-k3"}, clear=False
        ):
            switched = main_module._apply_multimodal_switch(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,QUJD"},
                            }
                        ],
                    }
                ],
                "glm-5.3",
                {
                    "_trae_model_capability_catalog": {
                        "kimi-k3": {"multimodal": True},
                        "glm-5.3": {"multimodal": False},
                    }
                },
            )
        self.assertEqual(switched, "kimi-k3")

    def test_multimodal_switch_skips_when_model_already_vision(self):
        self.assertEqual(
            main_module._apply_multimodal_switch(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,QUJD"},
                            }
                        ],
                    }
                ],
                "kimi-k3",
                {},
            ),
            "kimi-k3",
        )

    def test_multimodal_switch_skips_text_only_requests(self):
        self.assertEqual(
            main_module._apply_multimodal_switch(
                [{"role": "user", "content": "hello"}], "glm-5.3", {}
            ),
            "glm-5.3",
        )

    def test_explicit_model_is_not_silently_repointed_for_images(self):
        trace = {}
        image_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    }
                ],
            }
        ]
        with patch.dict(
            main_module.os.environ,
            {"TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT": ""},
            clear=False,
        ):
            selected = main_module._apply_multimodal_switch(
                image_messages,
                "glm-5.2",
                {"_requested_model": "glm-5.2", "_upstream_trace": trace},
            )
        self.assertEqual(selected, "glm-5.2")
        self.assertEqual(
            trace["multimodal_switch_skipped"]["reason"],
            "explicit_model_pinned",
        )


class TruncationDetectionTests(unittest.TestCase):
    def test_unclosed_code_fence_is_truncated(self):
        self.assertTrue(continuation.looks_truncated("text\n```python\nprint(1)"))

    def test_closed_answer_is_not_truncated(self):
        self.assertFalse(continuation.looks_truncated("A complete answer."))

    def test_finish_reason_length_is_truncated(self):
        self.assertTrue(continuation.looks_truncated("done", "length"))

    def test_tool_call_turns_are_never_continued(self):
        self.assertFalse(
            continuation.looks_truncated("x", "stop", has_tool_calls=True)
        )

    def test_reasoning_only_response_is_detected(self):
        self.assertTrue(
            continuation.looks_reasoning_only("", reasoning="thinking hard")
        )
        self.assertFalse(
            continuation.looks_reasoning_only("answer", reasoning="thinking")
        )

    def test_continued_messages_replay_partial_answer(self):
        built = continuation.continued_messages(
            [{"role": "user", "content": "write code"}], "def f():"
        )
        self.assertEqual(built[-2]["role"], "assistant")
        self.assertEqual(built[-2]["content"], "def f():")
        self.assertEqual(built[-1]["role"], "user")
        self.assertIn("继续", built[-1]["content"])


class AnthropicTranslationTests(unittest.TestCase):
    def test_request_with_tools_and_images_translates(self):
        messages, options = anthropic_api.normalize_request(
            {
                "model": "glm-5.3",
                "max_tokens": 1024,
                "system": [{"type": "text", "text": "be brief"}],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "read",
                        "input_schema": {"type": "object"},
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "QUJD",
                                },
                            },
                        ],
                    }
                ],
            }
        )
        self.assertEqual(messages[0], {"role": "system", "content": "be brief"})
        self.assertEqual(messages[1]["content"][0], {"type": "text", "text": "look"})
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")
        self.assertEqual(options["max_tokens"], 1024)
        self.assertEqual(options["tools"][0]["function"]["name"], "read_file")

    def test_tool_result_becomes_tool_message(self):
        messages, _ = anthropic_api.normalize_request(
            {
                "model": "m",
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "read_file",
                                "input": {"path": "a.txt"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "file body",
                            }
                        ],
                    },
                ],
            }
        )
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "toolu_1")
        self.assertEqual(messages[1]["content"], "file body")

    def test_thinking_budget_maps_to_native_level(self):
        _, options = anthropic_api.normalize_request(
            {
                "model": "m",
                "max_tokens": 64,
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        self.assertEqual(options["thinking"]["effort"], "light")

    def test_completion_to_message_maps_tool_calls(self):
        message = anthropic_api.completion_to_message(
            {
                "model": "glm-5.3",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
            "glm-5.3",
        )
        self.assertEqual(message["stop_reason"], "tool_use")
        self.assertEqual(message["content"][0]["type"], "tool_use")
        self.assertEqual(message["content"][0]["input"], {"path": "a.txt"})
        self.assertEqual(message["usage"]["input_tokens"], 5)

    def test_stream_translation_emits_content_blocks(self):
        async def body():
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":" there"}}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        async def run():
            events = []
            async for event in anthropic_api.translate_chat_stream(body(), "m"):
                events.append(event)
            return events

        events = asyncio.run(run())
        joined = "".join(events)
        self.assertIn("event: message_start", joined)
        self.assertIn("content_block_delta", joined)
        self.assertIn('"text_delta"', joined)
        self.assertIn("event: message_stop", joined)
        self.assertIn('"stop_reason":"end_turn"', joined.replace(' ', ''))

    def test_stream_translation_reports_tool_use(self):
        async def body():
            yield (
                'data: {"choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_1","function":{"name":"read_file",'
                '"arguments":"{}"}}]}}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        async def run():
            events = []
            async for event in anthropic_api.translate_chat_stream(body(), "m"):
                events.append(event)
            return events

        joined = "".join(asyncio.run(run()))
        self.assertIn("event: content_block_start", joined)
        self.assertIn('"tool_use"', joined)
        self.assertIn('"stop_reason":"tool_use"', joined.replace(' ', ''))


class AnthropicEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_messages_endpoint_returns_anthropic_message(self):
        async def fake_dispatch(messages, model, stream, options=None):
            return JSONResponse(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch), patch.object(
            main_module, "_continuation_allowed", return_value=False
        ):
            response = self.client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.3",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"x-api-key": "smoke-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["content"][0]["text"], "pong")
        self.assertEqual(payload["stop_reason"], "end_turn")
        self.assertEqual(payload["usage"]["output_tokens"], 2)

    def test_messages_endpoint_rejects_missing_messages(self):
        response = self.client.post(
            "/v1/messages",
            json={"model": "glm-5.3", "max_tokens": 16},
            headers={"x-api-key": "smoke-key"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["type"], "error")

    def test_nonstream_messages_invalid_tool_arguments_return_502(self):
        for arguments in ('{"path":', '["not-an-object"]'):
            with self.subTest(arguments=arguments):
                async def fake_dispatch(messages, model, stream, options=None):
                    self.assertFalse(stream)
                    return JSONResponse(
                        {
                            "id": "chatcmpl-invalid-tool",
                            "object": "chat.completion",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "call_invalid",
                                                "type": "function",
                                                "function": {
                                                    "name": "read_file",
                                                    "arguments": arguments,
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        }
                    )

                with patch.object(
                    main_module, "_dispatch_chat", new=fake_dispatch
                ):
                    response = self.client.post(
                        "/v1/messages",
                        json={
                            "model": "glm-5.3",
                            "max_tokens": 32,
                            "stream": False,
                            "tools": [
                                {
                                    "name": "read_file",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {"path": {"type": "string"}},
                                    },
                                }
                            ],
                            "messages": [{"role": "user", "content": "Read a.txt"}],
                        },
                        headers={"x-api-key": "smoke-key"},
                    )
                self.assertEqual(response.status_code, 502, response.text)
                payload = response.json()
                self.assertEqual(payload["type"], "error")
                self.assertEqual(payload["error"]["type"], "api_error")
                self.assertIn("Tool input for read_file", payload["error"]["message"])
                self.assertNotIn("content", payload)

    def test_count_tokens_endpoint_returns_estimate_without_dispatch(self):
        response = self.client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "glm-5.3",
                "system": "system",
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"x-api-key": "smoke-key"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["input_tokens"], 0)


class ContinuationStreamTests(unittest.TestCase):
    def test_coalesced_text_and_terminal_frames_keep_text_for_continuation(self):
        calls = []

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)
            return StreamingResponse(
                iter(
                    [
                        'data: {"choices":[{"index":0,"delta":{"content":"rest."}}]}\n\n',
                        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                        "data: [DONE]\n\n",
                    ]
                ),
                media_type="text/event-stream",
            )

        async def run():
            first = (
                'data: {"choices":[{"index":0,"delta":{"content":"partial,"}}]}\n\n'
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
                "data: [DONE]\n\n"
            )

            async def first_round():
                yield first

            stream = main_module._stream_with_continuation(
                [{"role": "user", "content": "write"}],
                "m",
                {"_relay_request_id": "req-coalesced"},
                first_round(),
            )
            return [chunk async for chunk in stream]

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch), patch.dict(
            main_module.os.environ, {"TRAE_AUTO_CONTINUE": "1"}, clear=False
        ):
            chunks = asyncio.run(run())
        parsed, done = _parse_chunks(chunks)
        self.assertEqual(_visible_text(parsed), "partial,rest.")
        self.assertTrue(done)
        self.assertEqual(len(calls), 1)
        self.assertIn("partial,", calls[0][-2]["content"])

    def test_coalesced_text_and_finish_frames_are_preserved_across_continuation(self):
        calls = []

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)

            async def second_round():
                yield 'data: {"choices":[{"index":0,"delta":{"content":" second half."}}]}\n\n'
                yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(second_round(), media_type="text/event-stream")

        async def run():
            async def first_round():
                # Simulate a proxy coalescing three SSE blocks into one read.
                yield (
                    'data: {"choices":[{"index":0,"delta":{"content":"first half,"}}]}\n\n'
                    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
                    "data: [DONE]\n\n"
                )

            return [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "user", "content": "write"}],
                    "m",
                    {"_relay_request_id": "req-coalesced"},
                    first_round(),
                )
            ]

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch), patch.dict(
            main_module.os.environ, {"TRAE_AUTO_CONTINUE": "1"}, clear=False
        ):
            chunks = asyncio.run(run())

        parsed, done = _parse_chunks(chunks)
        self.assertEqual(_visible_text(parsed), "first half, second half.")
        self.assertTrue(done)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-2]["content"], "first half,")

    def test_truncated_stream_is_continued_and_finish_is_deferred(self):
        calls = []

        def make_response(text, finish="stop"):
            def gen():
                yield (
                    'data: {"choices":[{"index":0,"delta":{"content":"%s"}}]}\n\n' % text
                )
                yield (
                    'data: {"choices":[{"index":0,"delta":{},'
                    '"finish_reason":"%s"}]}\n\n' % finish
                )
                yield "data: [DONE]\n\n"

            return gen()

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)
            # The first round is the stream handed in by the caller; every
            # dispatch call after that is one continuation round.
            return StreamingResponse(
                make_response(" second half."),
                media_type="text/event-stream",
            )

        async def run():
            options = {"_relay_request_id": "req-test"}
            stream = main_module._stream_with_continuation(
                [{"role": "user", "content": "write"}],
                "m",
                options,
                make_response("first half,", finish="stop"),
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            return chunks

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch), patch.dict(
            main_module.os.environ, {"TRAE_AUTO_CONTINUE": "1"}, clear=False
        ):
            chunks = asyncio.run(run())

        parsed, done = _parse_chunks(chunks)
        text = _visible_text(parsed)
        self.assertEqual(text, "first half, second half.")
        self.assertTrue(done)
        # The first round's finish frame must not reach the client.
        finish_frames = [
            event
            for event in parsed
            if event.get("choices")
            and event["choices"][0].get("finish_reason")
        ]
        self.assertEqual(len(finish_frames), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1]["role"], "user")
        self.assertIn("继续", calls[0][-1]["content"])

    def test_complete_stream_is_not_continued(self):
        calls = []

        def make_response(text):
            def gen():
                yield (
                    'data: {"choices":[{"index":0,"delta":{"content":"%s"}}]}\n\n' % text
                )
                yield (
                    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                )
                yield "data: [DONE]\n\n"

            return gen()

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)
            return StreamingResponse(
                make_response("done."), media_type="text/event-stream"
            )

        async def run():
            stream = main_module._stream_with_continuation(
                [{"role": "user", "content": "hi"}],
                "m",
                {},
                _async_chunks(make_response("A complete sentence.")),
            )
            return [chunk async for chunk in stream]

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            chunks = asyncio.run(run())
        parsed, done = _parse_chunks(chunks)
        self.assertEqual(_visible_text(parsed), "A complete sentence.")
        self.assertTrue(done)
        self.assertEqual(calls, [])

    def test_tool_call_stream_is_not_continued(self):
        calls = []

        def gen():
            yield (
                'data: {"choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"c1","function":{"name":"t","arguments":"{}"}}]}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)
            return StreamingResponse(gen(), media_type="text/event-stream")

        async def run():
            stream = main_module._stream_with_continuation(
                [{"role": "user", "content": "hi"}],
                "m",
                {},
                _async_chunks(gen()),
            )
            return [chunk async for chunk in stream]

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            chunks = asyncio.run(run())
        self.assertEqual(calls, [])
        self.assertIn("tool_calls", "".join(chunks))


class ContinuationNonstreamTests(unittest.TestCase):
    def test_distinct_short_rounds_are_not_compared_to_accumulated_text(self):
        completions = [
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "alpha"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "beta"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "gamma."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        ]
        calls = []

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)
            return JSONResponse(content=completions[len(calls) - 1])

        class DummyTracker:
            async def begin(self):
                return None

            async def finish(self, status=None):
                return None

            def update(self, usage):
                return None

            def mark_tool_calls(self, value):
                return None

        async def run():
            return await main_module._nonstream_with_continuation(
                [{"role": "user", "content": "write"}],
                "m",
                {},
                DummyTracker(),
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch), patch.dict(
            main_module.os.environ,
            {
                "TRAE_AUTO_CONTINUE": "1",
                "TRAE_TRUNCATION_TEXT_THRESHOLD": "200",
            },
            clear=False,
        ):
            response = asyncio.run(run())

        payload = json.loads(response.body)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            payload["choices"][0]["message"]["content"], "alphabetagamma."
        )

    def test_truncated_completion_is_continued_and_usage_summed(self):
        completions = [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "part one,"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {
                        "cached_tokens": 6,
                        "cache_write_tokens": 4,
                    },
                },
            },
            {
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": " part two."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                    "prompt_tokens_details": {
                        "cached_tokens": 8,
                        "cache_write_tokens": 4,
                    },
                },
            },
        ]
        calls = []

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append(messages)
            return JSONResponse(content=completions[len(calls) - 1])

        class DummyTracker:
            def __init__(self):
                self.saw_usage = False

            async def begin(self):
                return None

            async def finish(self, status=None):
                return None

            def update(self, usage):
                self.saw_usage = True

            def mark_tool_calls(self, value):
                return None

        async def run():
            return await main_module._nonstream_with_continuation(
                [{"role": "user", "content": "write"}],
                "m",
                {"_relay_request_id": "req-1"},
                DummyTracker(),
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            response = asyncio.run(run())
        payload = json.loads(response.body)
        self.assertEqual(
            payload["choices"][0]["message"]["content"], "part one, part two."
        )
        self.assertEqual(payload["usage"]["completion_tokens"], 9)
        self.assertEqual(
            payload["usage"]["prompt_tokens_details"],
            {"cached_tokens": 14, "cache_write_tokens": 8},
        )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
