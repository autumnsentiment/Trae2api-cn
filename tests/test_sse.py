import asyncio
import json
import os
import time
import unittest
from unittest.mock import patch

from src import sse
from src.cli_client import CliEvent


async def _cli_events():
    yield CliEvent(
        type="json",
        data={"message": {"content": [{"type": "text", "text": "hello"}]}},
    )
    yield CliEvent(
        type="json",
        data={
            "message": {"content": [{"type": "text", "text": "hello world"}]},
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        },
    )


async def _cli_tool_events():
    yield CliEvent(
        type="json",
        data={
            "message": {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"filePath":"README.md"}',
                        },
                    }
                ],
            }
        },
    )


async def _collect(chunks):
    return [chunk async for chunk in chunks]


class _LineResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)

    def __iter__(self):
        return iter(self._lines)


def _real_incremental_tool_response():
    fragments = [
        '<opencode',
        '_tool_call>{"id":"1"',
        ',"name":"read_',
        'client_file","input":{"path":"C:/demo/',
        'tool.txt"}}</opencode_',
        'tool_call>',
    ]
    lines = []
    for fragment in fragments:
        lines.extend(("event: output", "data: " + json.dumps({"response": fragment})))
    lines.extend(("event: done", 'data: {"finish_reason":"stop"}'))
    return _LineResponse(lines)


def _trae_native_split_tool_response():
    return _LineResponse(
        [
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_native_1",
                            "type": "function",
                            "function_call": {"name": "Read", "args": ""},
                        }
                    ]
                }
            ),
            # Trae can report the reason before its final argument delta. The
            # translator must keep consuming until the stream terminates.
            'data: {"finish_reason":"tool_calls"}',
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "",
                            "type": "function",
                            "function_call": {
                                "name": "",
                                "arguments": '{"filePath":"README.md"}',
                            },
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
    )


def _parse_chunks(chunks):
    parsed = []
    done = False
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            done = True
            continue
        if chunk.startswith("data: "):
            parsed.append(json.loads(chunk[6:].strip()))
    return parsed, done


class TranslateCliStreamTests(unittest.TestCase):
    def test_provider_qualified_deepseek_model_matches_requested_family(self):
        sse._check_provider_model("DeepSeek-V4-Pro", "ali-deepseek-v4-pro")

    def test_provider_family_normalizes_only_known_trae_decorations(self):
        matching = [
            ("DeepSeek-V4-Pro", "ali-deepseek-v4-pro-0813"),
            ("DeepSeek-V4-Pro", "ali-deepseek-v4-pro-0813__dev"),
            ("DeepSeek-V4-Pro-Official", "DeepSeek-V4-Pro__v2"),
            ("DeepSeek-V4-Pro 正式版", "ali-deepseek-v4-pro-0813"),
            ("DeepSeek-V4-Flash", "ali-deepseek-v4-flash"),
            (
                "DeepSeek-V4-Flash-Official",
                "ali-deepseek-v4-flash-Official__v2",
            ),
            ("glm-5.3", "glm-5.3__dev"),
            ("glm-5.3", "glm-5.3__v2"),
        ]
        for requested, actual in matching:
            with self.subTest(requested=requested, actual=actual):
                sse._check_provider_model(requested, actual)

        with self.assertRaises(sse.ModelProviderMismatch):
            sse._check_provider_model(
                "DeepSeek-V4-Pro-Official", "kimi-k2.6__v2"
            )

    def test_native_pascal_and_camel_case_events_are_normalized(self):
        class GuardedResponse:
            def iter_lines(self):
                yield "Event: modelConfig"
                yield 'Data: {"modelName":"glm-5.3__v2"}'
                yield "Event: output"
                yield 'Data: {"response":"native complete"}'
                yield "Event: tokenUsage"
                yield (
                    'Data: {"inputToken":5,"outputToken":2,'
                    '"totalToken":7,"creditsFloat":0.2}'
                )
                yield "Event: Done"
                yield 'Data: {"stopReason":"stop"}'
                raise AssertionError("read past native Done event")

        result = asyncio.run(
            sse.collect_nonstream_ide(GuardedResponse(), "glm-5.3")
        )
        self.assertEqual(
            result["choices"][0]["message"]["content"], "native complete"
        )
        self.assertEqual(result["provider_model_name"], "glm-5.3__v2")
        self.assertEqual(
            result["usage"],
            {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "credits_consumed": 0.2,
            },
        )

    def test_data_event_type_camel_case_is_normalized(self):
        response = _LineResponse(
            [
                'data: {"eventType":"requestWaitInQueue","position":1}',
                'data: {"eventType":"output","response":"ready"}',
                'data: {"eventType":"Done","finishReason":"stop"}',
            ]
        )
        chunks = asyncio.run(
            _collect(sse.translate_ide_stream(response, "m", True))
        )
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertIn("位置：1", content)
        self.assertTrue(content.endswith("ready"))
        self.assertTrue(done)

    def test_ide_provider_mismatch_is_not_presented_as_requested_model(self):
        response = _LineResponse(
            [
                'data: {"response":"fallback"}',
                'data: {"timing_cost":{"provider_model_name":"kimi-k2.6"}}',
                "data: [DONE]",
            ]
        )

        with self.assertRaises(sse.ModelProviderMismatch):
            asyncio.run(_collect(sse.translate_ide_stream(response, "glm-5.3", True)))

    def test_ide_provider_name_is_exposed_on_terminal_chunk(self):
        response = _LineResponse(
            [
                'data: {"response":"ok"}',
                'data: {"timing_cost":{"provider_model_name":"glm-5.3__dev"}}',
                "data: [DONE]",
            ]
        )
        chunks = asyncio.run(_collect(sse.translate_ide_stream(response, "glm-5.3", True)))
        parsed, done = _parse_chunks(chunks)
        terminal = [item for item in parsed if item.get("provider_model_name")]
        self.assertEqual(terminal[-1]["provider_model_name"], "glm-5.3__dev")
        self.assertTrue(done)

    def test_timing_events_internal_model_id_is_ignored(self):
        response = _LineResponse(
            [
                'data: {"response":"ok"}',
                'data: {"event":"timing_events","model_name":"eFs8axOnhnVU4aIQm9b4"}',
                "data: [DONE]",
            ]
        )
        chunks = asyncio.run(_collect(sse.translate_ide_stream(response, "glm-5.3", True)))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        self.assertFalse(any(item.get("provider_model_name") for item in parsed))

    def test_nested_internal_model_id_is_ignored(self):
        response = _LineResponse(
            [
                'data: {"response":"ok"}',
                'data: {"data":{"model_name":"eFs8axOnhnVU4aIQm9b4"}}',
                "data: [DONE]",
            ]
        )
        chunks = asyncio.run(_collect(sse.translate_ide_stream(response, "glm-5.3", True)))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        self.assertFalse(any(item.get("provider_model_name") for item in parsed))

    def test_model_config_model_name_is_exposed(self):
        response = _LineResponse(
            [
                'data: {"event":"model_config","model_name":"glm-5.3__dev"}',
                'data: {"response":"ok"}',
                "data: [DONE]",
            ]
        )
        chunks = asyncio.run(_collect(sse.translate_ide_stream(response, "glm-5.3", True)))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        self.assertEqual(
            [item["provider_model_name"] for item in parsed if item.get("provider_model_name")][-1],
            "glm-5.3__dev",
        )

    def test_model_config_reply_to_message_id_is_captured_privately(self):
        response = _LineResponse(
            [
                "event: model_config",
                'data: {"model_name":"glm-5.3__dev","reply_to_message_id":"turn-1"}',
                'data: {"response":"ok"}',
                "data: [DONE]",
            ]
        )
        metadata = {}
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    response,
                    "glm-5.3",
                    True,
                    upstream_metadata=metadata,
                )
            )
        )
        self.assertEqual(metadata, {"usage_turn_id": "turn-1"})
        self.assertNotIn("turn-1", "".join(chunks))

    def test_stream_usage_turn_id_variants_are_captured_privately(self):
        cases = [
            (
                "reply snake",
                {"reply_to_message_id": "stream-reply-snake"},
                "stream-reply-snake",
            ),
            (
                "reply camel",
                {"replyToMessageId": "stream-reply-camel"},
                "stream-reply-camel",
            ),
            (
                "user snake",
                {"user_message_id": "stream-user-snake"},
                "stream-user-snake",
            ),
            (
                "user camel",
                {"userMessageId": "stream-user-camel"},
                "stream-user-camel",
            ),
            (
                "nested snake",
                {"model_config": {"reply_to_message_id": "stream-nested-snake"}},
                "stream-nested-snake",
            ),
            (
                "nested camel",
                {"modelConfig": {"userMessageId": "stream-nested-camel"}},
                "stream-nested-camel",
            ),
            (
                "nested data",
                {
                    "data": {
                        "modelConfig": {
                            "replyToMessageId": "stream-nested-data"
                        }
                    }
                },
                "stream-nested-data",
            ),
        ]

        for label, payload, turn_id in cases:
            with self.subTest(label=label):
                response = _LineResponse(
                    [
                        "event: model_config",
                        "data: " + json.dumps(payload),
                        'data: {"response":"ok"}',
                        "data: [DONE]",
                    ]
                )
                metadata = {}
                chunks = asyncio.run(
                    _collect(
                        sse.translate_ide_stream(
                            response,
                            "m",
                            True,
                            upstream_metadata=metadata,
                        )
                    )
                )
                wire = "".join(chunks)

                self.assertEqual(metadata, {"usage_turn_id": turn_id})
                self.assertNotIn(turn_id, wire)
                self.assertNotIn("reply_to_message_id", wire)
                self.assertNotIn("replyToMessageId", wire)
                self.assertNotIn("user_message_id", wire)
                self.assertNotIn("userMessageId", wire)
                parsed, done = _parse_chunks(chunks)
                self.assertTrue(done)
                self.assertTrue(
                    any(
                        choice.get("delta", {}).get("content") == "ok"
                        for item in parsed
                        for choice in item.get("choices", [])
                    )
                )

    def test_done_message_id_is_not_used_as_usage_turn_id(self):
        for key in ("message_id", "messageId"):
            with self.subTest(key=key):
                response = _LineResponse(
                    [
                        'data: {"response":"ok"}',
                        "event: done",
                        "data: " + json.dumps({key: "agent-message-1"}),
                    ]
                )
                metadata = {}
                chunks = asyncio.run(
                    _collect(
                        sse.translate_ide_stream(
                            response,
                            "m",
                            True,
                            upstream_metadata=metadata,
                        )
                    )
                )

                self.assertEqual(metadata, {})
                self.assertNotIn("agent-message-1", "".join(chunks))

    def test_translate_cli_stream(self):
        chunks = asyncio.run(_collect(sse.translate_cli_stream(_cli_events(), "m", True)))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        content = "".join(
            choice["delta"].get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content, "hello world")
        last = parsed[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            last["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    def test_text_chunks_are_always_treated_as_incremental(self):
        async def events():
            yield CliEvent(type="text", text="a")
            yield CliEvent(type="text", text="ab")

        chunks = asyncio.run(_collect(sse.translate_cli_stream(events(), "m", True)))
        parsed, _ = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content, "aab")

    def test_intermediate_finish_reason_does_not_truncate_cli_snapshot(self):
        closed = False

        async def events():
            nonlocal closed
            try:
                yield CliEvent(
                    type="json",
                    data={"message": {"content": "once"}},
                )
                yield CliEvent(
                    type="json",
                    data={
                        "message": {"content": "once"},
                        "finish_reason": "stop",
                    },
                )
                # The CLI can send a later cumulative snapshot even though an
                # earlier snapshot already carried finish_reason.
                yield CliEvent(
                    type="json",
                    data={"message": {"content": "once and continued"}},
                )
            finally:
                closed = True

        chunks = asyncio.run(_collect(sse.translate_cli_stream(events(), "m", True)))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        terminal = [
            choice.get("finish_reason")
            for event in parsed
            for choice in event.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        self.assertEqual(content, "once and continued")
        self.assertEqual(terminal, ["stop"])
        self.assertTrue(done)
        self.assertTrue(closed)

    def test_echoed_truncated_history_marker_is_not_visible(self):
        async def events():
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": (
                            'Previous client tool request(s):\n'
                            '[{"id":"new","name":"read_file","input":"{\\"path\\":\\"C:\\\\work\\\\'
                        ),
                        "tool_calls": [
                            {
                                "id": "new",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"C:\\\\work\\"}',
                                },
                            }
                        ],
                    }
                },
            )

        chunks = asyncio.run(_collect(sse.translate_cli_stream(events(), "m")))
        parsed, _ = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertNotIn("Previous client tool request", content)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas[0]["function"]["name"], "read_file")

    def test_split_partial_history_marker_after_answer_is_never_streamed(self):
        async def events():
            yield CliEvent(
                type="text",
                text="让我执行 sudo 部署脚本。\n\nPrev",
            )
            yield CliEvent(type="text", text="ious client tool request(s")
            yield CliEvent(
                type="text",
                text="\nPrevious client tool request(s",
            )

        chunks = asyncio.run(_collect(sse.translate_cli_stream(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )

        self.assertEqual(content, "让我执行 sudo 部署脚本。")
        self.assertNotIn("Previous client", content)
        self.assertTrue(done)

    def test_cumulative_raw_snapshots_hold_split_history_marker(self):
        response = _LineResponse(
            [
                "data: "
                + json.dumps({"response": "让我执行 sudo 部署脚本。\n\nPrev"}),
                "data: "
                + json.dumps(
                    {
                        "response": (
                            "让我执行 sudo 部署脚本。\n\n"
                            "Previous client tool request(s"
                        )
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "response": (
                            "让我执行 sudo 部署脚本。\n\n"
                            "Previous client tool request(s\n"
                            "Previous client tool request(s"
                        )
                    }
                ),
                "data: [DONE]",
            ]
        )

        chunks = asyncio.run(_collect(sse.translate_ide_stream(response, "m", True)))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )

        self.assertEqual(content, "让我执行 sudo 部署脚本。")
        self.assertTrue(done)

    def test_json_snapshot_revision_does_not_repeat_common_prefix(self):
        async def events():
            yield CliEvent(
                type="json",
                data={"message": {"content": "hello world"}},
            )
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": "hello there"},
                    "finish_reason": "stop",
                },
            )

        chunks = asyncio.run(_collect(sse.translate_cli_stream(events(), "m", True)))
        parsed, _ = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content.count("hello "), 1)
        self.assertNotEqual(content, "hello worldhello there")

    def test_intermediate_finish_reason_does_not_truncate_ide_stream(self):
        class GuardedResponse:
            def __iter__(self):
                yield "data: " + json.dumps(
                    {"response": "once", "finish_reason": "stop"}
                )
                yield "data: " + json.dumps({"response": "once and continued"})
                yield "data: [DONE]"

        chunks = asyncio.run(
            _collect(sse.translate_ide_stream(GuardedResponse(), "m", True))
        )
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        terminal = [
            choice.get("finish_reason")
            for event in parsed
            for choice in event.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        self.assertEqual(content, "once and continued")
        self.assertEqual(terminal, ["stop"])
        self.assertTrue(done)
        self.assertEqual(chunks.count("data: [DONE]\n\n"), 1)

    def test_ide_eof_without_terminal_is_reported_as_incomplete(self):
        response = _LineResponse(
            [
                'data: {"response":"partial","stop_reason":"max_tokens"}',
                'data: {"response":"partial and still growing"}',
            ]
        )
        with self.assertRaises(sse.IncompleteUpstreamResponse):
            asyncio.run(_collect(sse.translate_ide_stream(response, "m", True)))

    def test_raw_v2_eof_is_a_valid_terminal_boundary(self):
        response = _LineResponse(
            [
                'event: output',
                'data: {"response":"complete at eof"}',
                'event: token_usage',
                'data: {"input_token":5,"output_token":2}',
            ]
        )
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    response,
                    "m",
                    True,
                    fail_on_empty=True,
                    require_terminal=False,
                )
            )
        )
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for item in parsed
            for choice in item.get("choices", [])
        )
        self.assertEqual(content, "complete at eof")
        self.assertTrue(done)

    def test_sync_ide_reader_does_not_block_event_loop(self):
        class SlowResponse:
            def iter_lines(self):
                time.sleep(0.08)
                yield 'data: {"response":"ready"}'
                time.sleep(0.08)
                yield "data: [DONE]"

        async def run():
            finished = False
            ticks = 0

            async def ticker():
                nonlocal ticks
                while not finished:
                    await asyncio.sleep(0.01)
                    ticks += 1

            task = asyncio.create_task(ticker())
            chunks = await _collect(
                sse.translate_ide_stream(SlowResponse(), "m", True)
            )
            finished = True
            await task
            return ticks, chunks

        with patch.dict(os.environ, {"SSE_HEARTBEAT_SECONDS": "0"}):
            ticks, chunks = asyncio.run(run())
        parsed, done = _parse_chunks(chunks)
        self.assertGreaterEqual(ticks, 5)
        self.assertTrue(done)
        self.assertEqual(parsed[1]["choices"][0]["delta"]["content"], "ready")

    def test_ide_stream_emits_comment_keepalive_during_upstream_gap(self):
        class SlowResponse:
            def iter_lines(self):
                time.sleep(0.06)
                yield 'data: {"response":"ready"}'
                yield "data: [DONE]"

        with patch.dict(os.environ, {"SSE_HEARTBEAT_SECONDS": "0.01"}):
            chunks = asyncio.run(
                _collect(sse.translate_ide_stream(SlowResponse(), "m", True))
            )
        self.assertIn(": relay-keepalive\n\n", chunks)
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content, "ready")
        self.assertTrue(done)

    def test_repeated_queue_position_is_emitted_once(self):
        response = _LineResponse(
            [
                'data: {"event":"request_wait_in_queue","position":2}',
                'data: {"event":"request_wait_in_queue","position":2}',
                'data: {"event":"request_wait_in_queue","position":1}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        chunks = asyncio.run(_collect(sse.translate_ide_stream(response, "m", True)))
        parsed, _ = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content.count("位置：2"), 1)
        self.assertEqual(content.count("位置：1"), 1)

    def test_ide_empty_probe_emits_nothing_before_retry_signal(self):
        response = _LineResponse(
            [
                'data: {"event":"request_wait_in_queue","position":1}',
                'data: {"event":"token_usage","usage":{"total_tokens":2}}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )

        async def run():
            chunks = []
            try:
                async for chunk in sse.translate_ide_stream(
                    response, "m", True, fail_on_empty=True
                ):
                    chunks.append(chunk)
            except sse.EmptyUpstreamResponse:
                return chunks
            raise AssertionError("empty probe did not raise")

        self.assertEqual(asyncio.run(run()), [])

    def test_ide_empty_probe_with_usage_is_not_retryable(self):
        response = _LineResponse(
            [
                'event: token_usage',
                'data: {"input_token":12,"output_token":1,"credits_float":0.25}',
                'event: done',
                'data: {"finish_reason":"stop"}',
            ]
        )

        async def run():
            async for _chunk in sse.translate_ide_stream(
                response, "m", True, fail_on_empty=True
            ):
                pass

        with self.assertRaises(sse.EmptyUpstreamResponse) as raised:
            asyncio.run(run())

        self.assertFalse(raised.exception.retryable)
        self.assertTrue(raised.exception.observed_model_event)
        self.assertEqual(
            raised.exception.usage,
            {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "total_tokens": 13,
                "credits_consumed": 0.25,
            },
        )

    def test_ide_empty_probe_with_usage_turn_id_is_not_retryable(self):
        response = _LineResponse(
            [
                "event: model_config",
                'data: {"replyToMessageId":"accepted-turn"}',
                "event: done",
                'data: {"finish_reason":"stop"}',
            ]
        )
        metadata = {}

        async def run():
            async for _chunk in sse.translate_ide_stream(
                response,
                "m",
                True,
                fail_on_empty=True,
                upstream_metadata=metadata,
            ):
                pass

        with self.assertRaises(sse.EmptyUpstreamResponse) as raised:
            asyncio.run(run())

        self.assertFalse(raised.exception.retryable)
        self.assertTrue(raised.exception.observed_model_event)
        self.assertEqual(metadata["usage_turn_id"], "accepted-turn")

    def test_ide_empty_probe_with_only_nested_usage_turn_id_is_not_retryable(self):
        response = _LineResponse(
            [
                "event: model_config",
                'data: {"model_config":{"reply_to_message_id":"accepted-nested"}}',
                "event: done",
                "data: {}",
            ]
        )
        metadata = {}

        async def run():
            chunks = []
            try:
                async for chunk in sse.translate_ide_stream(
                    response,
                    "m",
                    True,
                    fail_on_empty=True,
                    upstream_metadata=metadata,
                ):
                    chunks.append(chunk)
            except sse.EmptyUpstreamResponse as exc:
                return chunks, exc
            raise AssertionError("empty probe did not raise")

        chunks, error = asyncio.run(run())
        self.assertEqual(chunks, [])
        self.assertFalse(error.retryable)
        self.assertTrue(error.observed_model_event)
        self.assertIsNone(error.usage)
        self.assertEqual(metadata, {"usage_turn_id": "accepted-nested"})

    def test_ide_incomplete_after_visible_output_is_not_retryable(self):
        response = _LineResponse(['data: {"response":"partial"}'])

        async def run():
            async for _chunk in sse.translate_ide_stream(
                response, "m", True, fail_on_empty=True
            ):
                pass

        with self.assertRaises(sse.IncompleteUpstreamResponse) as raised:
            asyncio.run(run())

        self.assertFalse(raised.exception.retryable)
        self.assertTrue(raised.exception.observed_model_event)

    def test_ide_probe_starts_stream_as_soon_as_text_arrives(self):
        response = _LineResponse(
            [
                'data: {"event":"request_wait_in_queue","position":1}',
                'data: {"response":"ready"}',
                'data: {"finish_reason":"stop"}',
            ]
        )

        async def first_three():
            stream = sse.translate_ide_stream(
                response, "m", True, fail_on_empty=True
            )
            chunks = [await anext(stream), await anext(stream), await anext(stream)]
            await stream.aclose()
            return chunks

        parsed, _ = _parse_chunks(asyncio.run(first_three()))
        self.assertEqual(parsed[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertIn("位置：1", parsed[1]["choices"][0]["delta"]["content"])
        self.assertEqual(parsed[2]["choices"][0]["delta"]["content"], "ready")

    def test_ide_error_event_is_propagated(self):
        response = _LineResponse(
            [
                "event: error",
                'data: {"code":"bad_request","message":"native stream failed"}',
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "native stream failed"):
            asyncio.run(_collect(sse.translate_ide_stream(response, "m", True)))

    def test_translate_cli_stream_error(self):
        async def events():
            yield CliEvent(type="error", error="boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(
                _collect(sse.translate_cli_stream(events(), "m", True))
            )

    def test_translate_cli_stream_tool_call(self):
        chunks = asyncio.run(_collect(sse.translate_cli_stream(_cli_tool_events(), "m", True)))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        tool_deltas = [
            tool_call
            for event in parsed
            for choice in event.get("choices", [])
            for tool_call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas[0]["id"], "call_read_1")
        self.assertEqual(tool_deltas[0]["function"]["name"], "Read")
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_unadvertised_tool_call_is_suppressed(self):
        tools = [{"type": "function", "function": {"name": "Bash"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_cli_stream(
                    _cli_tool_events(), "m", True, allowed_tools=tools
                )
            )
        )
        parsed, _ = _parse_chunks(chunks)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas, [])
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "stop")

    def test_parallel_false_limits_calls_across_multiple_events(self):
        async def events():
            for call_id, name in (("call_1", "Read"), ("call_2", "Bash")):
                yield CliEvent(
                    type="json",
                    data={
                        "message": {
                            "content": [],
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": "{}"},
                                }
                            ],
                        }
                    },
                )

        tools = [
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Bash"}},
        ]
        chunks = asyncio.run(
            _collect(
                sse.translate_cli_stream(
                    events(),
                    "m",
                    True,
                    allowed_tools=tools,
                    parallel_tool_calls=False,
                )
            )
        )
        parsed, _ = _parse_chunks(chunks)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
            if call.get("id")
        ]
        self.assertEqual([call["id"] for call in tool_deltas], ["call_1"])

    def test_tool_choice_none_suppresses_call_and_uses_stop_reason(self):
        response = _LineResponse(
            [
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Read", "arguments": "{}"},
                            }
                        ],
                        "finish_reason": "tool_calls",
                    }
                ),
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    response,
                    "m",
                    True,
                    allowed_tools=tools,
                    tool_choice="none",
                )
            )
        )
        parsed, _ = _parse_chunks(chunks)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas, [])
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "stop")

    def test_split_raw_tool_block_never_leaks_as_content(self):
        complete = (
            '<opencode_tool_call>{"id":"call_1","name":"Read",'
            '"input":{"filePath":"README.md"}}</opencode_tool_call>'
        )
        response = _LineResponse([
            "event: output",
            'data: {"response":"<opencode_tool"}',
            "event: output",
            "data: " + json.dumps({"response": complete}),
            "event: done",
            'data: {"finish_reason":"stop"}',
        ])
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    response, "m", True, allowed_tools=tools
                )
            )
        )
        parsed, _ = _parse_chunks(chunks)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertNotIn("opencode_tool", visible)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas[0]["function"]["name"], "Read")
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_incremental_split_tool_block_never_leaks_as_content(self):
        response = _LineResponse(
            [
                'data: {"response":"<tool_"}',
                "data: "
                + json.dumps(
                    {
                        "response": (
                            'call>{"id":"call_1","name":"Read",'
                            '"arguments":{"filePath":"README.md"}}</tool_call>'
                        )
                    }
                ),
                'data: {"finish_reason":"tool_calls"}',
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(response, "m", True, allowed_tools=tools)
            )
        )
        parsed, _ = _parse_chunks(chunks)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertNotIn("tool_call", visible)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
            if call.get("id")
        ]
        self.assertEqual([call["id"] for call in tool_deltas], ["call_1"])

    def test_real_opencode_incremental_tool_block_preserves_windows_path(self):
        tools = [{"type": "function", "function": {"name": "read_client_file"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    _real_incremental_tool_response(),
                    "m",
                    True,
                    allowed_tools=tools,
                )
            )
        )
        parsed, _ = _parse_chunks(chunks)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertNotIn("opencode", visible)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas[0]["id"], "1")
        self.assertEqual(tool_deltas[0]["function"]["name"], "read_client_file")
        arguments = "".join(
            call.get("function", {}).get("arguments", "") for call in tool_deltas
        )
        self.assertEqual(json.loads(arguments)["path"], "C:/demo/tool.txt")
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_reasoning_tool_block_is_extracted_not_rendered(self):
        block = (
            '<tool_call>{"id":"call_1","name":"Read",'
            '"arguments":{"filePath":"README.md"}}</tool_call>'
        )
        response = _LineResponse(
            [
                "data: " + json.dumps({"reasoning_content": block}),
                'data: {"finish_reason":"tool_calls"}',
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(response, "m", True, allowed_tools=tools)
            )
        )
        parsed, _ = _parse_chunks(chunks)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertNotIn("tool_call", visible)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
            if call.get("id")
        ]
        self.assertEqual([call["id"] for call in tool_deltas], ["call_1"])

    def test_native_cumulative_call_without_id_keeps_one_stable_call(self):
        response = _LineResponse(
            [
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"filePath":',
                                },
                            }
                        ]
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"filePath":"README.md"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                    }
                ),
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(response, "m", True, allowed_tools=tools)
            )
        )
        parsed, _ = _parse_chunks(chunks)
        deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        ids = [call["id"] for call in deltas if call.get("id")]
        arguments = "".join(
            call.get("function", {}).get("arguments", "") for call in deltas
        )
        self.assertEqual(len(ids), 1)
        self.assertEqual(arguments, '{"filePath":"README.md"}')

    def test_distinct_same_name_calls_without_ids_stay_separate(self):
        response = _LineResponse(
            [
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"filePath":"a.txt"}',
                                },
                            }
                        ]
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"filePath":"b.txt"}',
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                    }
                ),
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(response, "m", True, allowed_tools=tools)
            )
        )
        parsed, _ = _parse_chunks(chunks)
        deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
            if call.get("id")
        ]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(
            [call["function"]["arguments"] for call in deltas],
            ['{"filePath":"a.txt"}', '{"filePath":"b.txt"}'],
        )

    def test_native_stream_keeps_empty_initial_args_and_nameless_delta(self):
        response = _LineResponse(
            [
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "index": 0,
                                "type": "function",
                                "function": {"name": "Read", "arguments": ""},
                            }
                        ]
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "arguments": '{"filePath":"README.md"}'
                                },
                            }
                        ],
                        "finish_reason": "tool_calls",
                    }
                ),
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(response, "m", True, allowed_tools=tools)
            )
        )
        parsed, _ = _parse_chunks(chunks)
        deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual([call["id"] for call in deltas if call.get("id")], ["call_1"])
        arguments = "".join(
            call.get("function", {}).get("arguments", "") for call in deltas
        )
        self.assertEqual(arguments, '{"filePath":"README.md"}')

    def test_trae_native_function_call_delta_merges_after_finish_frame(self):
        tools = [{"type": "function", "function": {"name": "Read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    _trae_native_split_tool_response(),
                    "m",
                    True,
                    allowed_tools=tools,
                )
            )
        )
        parsed, done = _parse_chunks(chunks)
        deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        finish_reasons = [
            choice["finish_reason"]
            for event in parsed
            for choice in event.get("choices", [])
            if choice.get("finish_reason") is not None
        ]

        self.assertTrue(done)
        self.assertEqual(
            [call["id"] for call in deltas if call.get("id")],
            ["call_native_1"],
        )
        self.assertEqual(
            [
                call.get("function", {}).get("name")
                for call in deltas
                if call.get("function", {}).get("name")
            ],
            ["Read"],
        )
        self.assertEqual(
            "".join(call.get("function", {}).get("arguments", "") for call in deltas),
            '{"filePath":"README.md"}',
        )
        self.assertEqual(finish_reasons, ["tool_calls"])

    def test_tool_names_are_case_sensitive(self):
        tools = [{"type": "function", "function": {"name": "read"}}]
        chunks = asyncio.run(
            _collect(
                sse.translate_cli_stream(
                    _cli_tool_events(), "m", True, allowed_tools=tools
                )
            )
        )
        parsed, _ = _parse_chunks(chunks)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(tool_deltas, [])
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "stop")


class CollectNonstreamCliTests(unittest.TestCase):
    def test_collect_nonstream_cli(self):
        result = asyncio.run(sse.collect_nonstream_cli(_cli_events(), "m"))
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(
            result["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    def test_collect_nonstream_cli_tool_call(self):
        result = asyncio.run(sse.collect_nonstream_cli(_cli_tool_events(), "m"))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_read_1")

    def test_collect_nonstream_cli_suppresses_completed_duplicate_signature(self):
        async def events():
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-new-id",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    }
                },
            )

        completed = {
            'Read\x00{"path":"README.md"}',
        }
        result = asyncio.run(
            sse.collect_nonstream_cli(
                events(), "m", completed_tool_signatures=completed
            )
        )
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            result["choices"][0]["message"]["content"],
            "(trae cli returned an empty response)",
        )
        self.assertNotIn("tool_calls", result["choices"][0]["message"])

    def test_collect_nonstream_cli_uses_latest_json_snapshot(self):
        async def events():
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": "hello"},
                    "finish_reason": "stop",
                },
            )
            yield CliEvent(
                type="json",
                data={"message": {"content": "hello there"}},
            )

        result = asyncio.run(sse.collect_nonstream_cli(events(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"], "hello there"
        )

    def test_collect_nonstream_cli_keeps_snapshot_after_finish_reason(self):
        async def events():
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": "first"},
                    "finish_reason": "stop",
                },
            )
            yield CliEvent(
                type="json",
                data={"message": {"content": "first and final"}},
            )

        result = asyncio.run(sse.collect_nonstream_cli(events(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"], "first and final"
        )


class CollectNonstreamIdeTests(unittest.TestCase):
    def test_usage_turn_id_variants_are_captured_privately(self):
        cases = [
            (
                "reply snake",
                {"reply_to_message_id": "nonstream-reply-snake"},
                "nonstream-reply-snake",
            ),
            (
                "reply camel",
                {"replyToMessageId": "nonstream-reply-camel"},
                "nonstream-reply-camel",
            ),
            (
                "user snake",
                {"user_message_id": "nonstream-user-snake"},
                "nonstream-user-snake",
            ),
            (
                "user camel",
                {"userMessageId": "nonstream-user-camel"},
                "nonstream-user-camel",
            ),
            (
                "nested snake",
                {
                    "model_config": {
                        "user_message_id": "nonstream-nested-snake"
                    }
                },
                "nonstream-nested-snake",
            ),
            (
                "nested camel",
                {
                    "modelConfig": {
                        "replyToMessageId": "nonstream-nested-camel"
                    }
                },
                "nonstream-nested-camel",
            ),
            (
                "nested data",
                {
                    "data": {
                        "model_config": {
                            "userMessageId": "nonstream-nested-data"
                        }
                    }
                },
                "nonstream-nested-data",
            ),
        ]

        for label, payload, turn_id in cases:
            with self.subTest(label=label):
                response = _LineResponse(
                    [
                        "event: model_config",
                        "data: " + json.dumps(payload),
                        'data: {"response":"ok"}',
                        "data: [DONE]",
                    ]
                )
                metadata = {}
                result = asyncio.run(
                    sse.collect_nonstream_ide(
                        response,
                        "m",
                        upstream_metadata=metadata,
                    )
                )
                public_json = json.dumps(result)

                self.assertEqual(metadata, {"usage_turn_id": turn_id})
                self.assertEqual(
                    result["choices"][0]["message"]["content"], "ok"
                )
                self.assertNotIn(turn_id, public_json)
                self.assertNotIn("reply_to_message_id", public_json)
                self.assertNotIn("replyToMessageId", public_json)
                self.assertNotIn("user_message_id", public_json)
                self.assertNotIn("userMessageId", public_json)

    def test_done_message_id_variants_are_not_used_as_usage_turn_id(self):
        for key in ("message_id", "messageId"):
            with self.subTest(key=key):
                response = _LineResponse(
                    [
                        'data: {"response":"ok"}',
                        "event: done",
                        "data: " + json.dumps({key: "assistant-message"}),
                    ]
                )
                metadata = {}
                result = asyncio.run(
                    sse.collect_nonstream_ide(
                        response,
                        "m",
                        upstream_metadata=metadata,
                    )
                )

                self.assertEqual(metadata, {})
                self.assertNotIn("assistant-message", json.dumps(result))

    def test_empty_response_with_only_nested_usage_turn_id_is_not_retryable(self):
        response = _LineResponse(
            [
                "event: model_config",
                'data: {"modelConfig":{"replyToMessageId":"nonstream-accepted"}}',
                "event: done",
                "data: {}",
            ]
        )
        metadata = {}

        with self.assertRaises(sse.EmptyUpstreamResponse) as raised:
            asyncio.run(
                sse.collect_nonstream_ide(
                    response,
                    "m",
                    fail_on_empty=True,
                    upstream_metadata=metadata,
                )
            )

        self.assertFalse(raised.exception.retryable)
        self.assertTrue(raised.exception.observed_model_event)
        self.assertIsNone(raised.exception.usage)
        self.assertEqual(metadata, {"usage_turn_id": "nonstream-accepted"})

    def test_eof_without_terminal_is_reported_as_incomplete(self):
        response = _LineResponse(
            [
                'data: {"response":"partial","stop_reason":"max_tokens"}',
                'data: {"response":"partial and still growing"}',
            ]
        )
        with self.assertRaises(sse.IncompleteUpstreamResponse):
            asyncio.run(sse.collect_nonstream_ide(response, "m"))

    def test_raw_v2_nonstream_accepts_eof_after_output(self):
        response = _LineResponse(
            [
                'event: output',
                'data: {"response":"complete at eof"}',
                'event: token_usage',
                'data: {"input_token":5,"output_token":2}',
            ]
        )
        result = asyncio.run(
            sse.collect_nonstream_ide(
                response,
                "m",
                fail_on_empty=True,
                require_terminal=False,
            )
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "complete at eof")
        self.assertEqual(result["usage"]["total_tokens"], 7)

    def test_empty_probe_raises_before_placeholder_is_created(self):
        response = _LineResponse(
            ['data: {"finish_reason":"stop"}', "data: [DONE]"]
        )
        with self.assertRaises(sse.EmptyUpstreamResponse):
            asyncio.run(
                sse.collect_nonstream_ide(
                    response, "m", fail_on_empty=True
                )
            )

    def test_done_sentinel_stops_before_replayed_data(self):
        class GuardedResponse:
            def iter_lines(self):
                yield "data: " + json.dumps({"response": "once"})
                yield "data: [DONE]"
                raise AssertionError("read past DONE")

        result = asyncio.run(sse.collect_nonstream_ide(GuardedResponse(), "m"))
        self.assertEqual(result["choices"][0]["message"]["content"], "once")

    def test_done_event_stops_before_replayed_data(self):
        class GuardedResponse:
            def iter_lines(self):
                yield "data: " + json.dumps({"response": "once"})
                yield "event: done"
                yield "data: " + json.dumps({"finish_reason": "stop"})
                raise AssertionError("read past done event")

        result = asyncio.run(sse.collect_nonstream_ide(GuardedResponse(), "m"))
        self.assertEqual(result["choices"][0]["message"]["content"], "once")

    def test_intermediate_finish_reason_does_not_truncate_nonstream_ide(self):
        response = _LineResponse(
            [
                'data: {"response":"first","finish_reason":"stop"}',
                'data: {"response":"first and final"}',
                "data: [DONE]",
            ]
        )
        result = asyncio.run(sse.collect_nonstream_ide(response, "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"], "first and final"
        )

    def test_split_tool_block_never_leaks_as_content(self):
        complete = (
            '<opencode_tool_call>{"id":"call_1","name":"Read",'
            '"input":{"filePath":"README.md"}}</opencode_tool_call>'
        )
        response = _LineResponse(
            [
                'data: {"response":"<opencode_tool"}',
                "data: " + json.dumps({"response": complete}),
                'data: {"finish_reason":"tool_calls"}',
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        result = asyncio.run(
            sse.collect_nonstream_ide(response, "m", allowed_tools=tools)
        )
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_1")

    def test_real_opencode_incremental_tool_block_nonstream(self):
        tools = [{"type": "function", "function": {"name": "read_client_file"}}]
        result = asyncio.run(
            sse.collect_nonstream_ide(
                _real_incremental_tool_response(),
                "m",
                allowed_tools=tools,
            )
        )
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "read_client_file")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"])["path"],
            "C:/demo/tool.txt",
        )

    def test_trae_native_function_call_delta_merges_after_finish_frame(self):
        tools = [{"type": "function", "function": {"name": "Read"}}]
        result = asyncio.run(
            sse.collect_nonstream_ide(
                _trae_native_split_tool_response(), "m", allowed_tools=tools
            )
        )
        choice = result["choices"][0]

        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(len(choice["message"]["tool_calls"]), 1)
        self.assertEqual(
            choice["message"]["tool_calls"][0],
            {
                "id": "call_native_1",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": '{"filePath":"README.md"}',
                },
            },
        )

    def test_required_tool_choice_fails_when_upstream_returns_no_call(self):
        response = _LineResponse(
            [
                'data: {"response":"I will answer without a tool."}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        tools = [{"type": "function", "function": {"name": "Read"}}]
        with self.assertRaisesRegex(RuntimeError, "required tool call"):
            asyncio.run(
                sse.collect_nonstream_ide(
                    response,
                    "m",
                    allowed_tools=tools,
                    tool_choice="required",
                )
            )


class WebEventTests(unittest.TestCase):
    def test_web_hyphenated_keepalive_is_ignored(self):
        async def events():
            yield "keep-alive", {"message": "heartbeat"}
            yield "planItem", {"id": "answer", "thought": "complete"}
            yield "Done", {"stopReason": "stop"}

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(result["choices"][0]["message"]["content"], "complete")

    def test_web_camel_case_events_stop_without_reading_replayed_data(self):
        async def events():
            yield "planItem", {"id": "answer", "thought": "complete"}
            yield "tokenUsage", {"inputToken": 3, "outputToken": 1}
            yield "Done", {"stopReason": "max_tokens"}
            raise AssertionError("read past web Done event")

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(result["choices"][0]["message"]["content"], "complete")
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.assertEqual(result["usage"]["total_tokens"], 4)

    def test_interleaved_plan_updates_do_not_replay_later_items(self):
        async def events():
            yield "plan_item", {"id": "a", "thought": "A"}
            yield "plan_item", {"id": "b", "thought": "B"}
            yield "plan_item", {"id": "a", "thought": "A+"}
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content, "AB+")
        self.assertEqual(content.count("B"), 1)
        self.assertTrue(done)

    def test_finish_plan_item_is_summary_not_tool_call(self):
        async def events():
            yield "plan_item", {
                "id": "finish-1",
                "tool_call_info": {
                    "name": "finish",
                    "params": {"summary": "done"},
                },
            }
            yield "done", {}

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], "done")
        self.assertNotIn("tool_calls", choice["message"])

    def test_plan_thought_tool_block_is_translated_to_client_call(self):
        block = (
            '<opencode_tool_call>{"id":"call_dl_1","name":"download",'
            '"input":{"url":"https://example.com/file.zip","dest":"file.zip"}}'
            "</opencode_tool_call>"
        )

        async def events():
            yield "plan_item", {
                "id": "plan-1",
                "thought": "Downloading the file.\n" + block,
            }
            yield "done", {"status": "succeeded"}

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "download",
                    "description": "Download a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "dest": {"type": "string"},
                        },
                        "required": ["url", "dest"],
                    },
                },
            }
        ]
        chunks = asyncio.run(
            _collect(
                sse.translate_web_events(
                    events(),
                    "m",
                    True,
                    allowed_tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                )
            )
        )
        parsed, done = _parse_chunks(chunks)
        tool_deltas = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertTrue(done)
        self.assertEqual(tool_deltas[0]["id"], "call_dl_1")
        self.assertEqual(tool_deltas[0]["function"]["name"], "download")
        arguments = "".join(
            call.get("function", {}).get("arguments", "")
            for call in tool_deltas
        )
        self.assertEqual(json.loads(arguments)["dest"], "file.zip")
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")

        result = asyncio.run(
            sse.collect_nonstream_web(
                events(),
                "m",
                allowed_tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        )
        call = result["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "download")
        self.assertIn("file.zip", call["function"]["arguments"])

    def test_done_max_tokens_maps_to_openai_length(self):
        async def events():
            yield "plan_item", {"id": "answer-1", "thought": "partial"}
            yield "done", {"stop_reason": "max_tokens"}

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(result["choices"][0]["finish_reason"], "length")

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "length")



    def test_context_window_exceeded_maps_to_openai_length(self):
        """A truncated turn must not be reported to the client as ``stop``."""

        for reason in (
            "model_context_window_exceeded",
            "context_window_exceeded",
            "context_length_exceeded",
        ):
            with self.subTest(reason=reason):

                async def events(reason=reason):
                    yield "plan_item", {"id": "answer-1", "thought": "partial"}
                    yield "done", {"stop_reason": reason}

                result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
                self.assertEqual(result["choices"][0]["finish_reason"], "length")

    def test_plan_item_content_is_not_dropped(self):
        """plan_item carries the reply in ``content`` next to ``thought``."""

        async def both():
            yield "plan_item", {
                "id": "p1",
                "thought": "thinking",
                "content": "the actual reply",
            }
            yield "done", {}

        async def content_only():
            yield "plan_item", {"id": "p1", "content": "the actual reply"}
            yield "done", {}

        result = asyncio.run(sse.collect_nonstream_web(both(), "m"))
        content = result["choices"][0]["message"]["content"]
        self.assertIn("thinking", content)
        self.assertIn("the actual reply", content)

        result = asyncio.run(sse.collect_nonstream_web(content_only(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"], "the actual reply"
        )

    def test_plan_item_content_streams_without_duplication(self):
        async def events():
            yield "plan_item", {"id": "p1", "thought": "step 1"}
            yield "plan_item", {"id": "p1", "thought": "step 1 step 2"}
            yield "plan_item", {
                "id": "p1",
                "thought": "step 1 step 2",
                "content": "final answer",
            }
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertTrue(done)
        self.assertEqual(content, "step 1 step 2\n\nfinal answer")

    def test_reasoning_narration_is_dropped_from_visible_answer(self):
        """The remote agent ships reasoning and reply in one ``thought`` field."""

        async def events():
            yield "plan_item", {"id": "p1", "thought": "The user wants a reversal."}
            yield "plan_item", {
                "id": "p1",
                "thought": "The user wants a reversal.\nUse s[::-1].",
            }
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        streamed = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertTrue(done)
        self.assertEqual(streamed, "Use s[::-1].")

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"], "Use s[::-1]."
        )

    def test_narration_only_turn_is_returned_not_reported_empty(self):
        """Holding back every sentence would look like an empty upstream."""

        async def events():
            yield "plan_item", {"id": "p1", "thought": "The user wants X."}
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, _ = _parse_chunks(chunks)
        streamed = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(streamed, "The user wants X.")
        self.assertNotIn("empty response", streamed)

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"], "The user wants X."
        )

    def test_split_narration_prefix_never_leaks_a_dangling_fragment(self):
        """Cumulative snapshots arrive mid-sentence ("The", "The user")."""

        snapshots = [
            "The",
            "The user",
            "The user wants the arrival time.",
            "The user wants the arrival time. Leg 1: 340 km = 4h.",
            "The user wants the arrival time. Leg 1: 340 km = 4h.\n20:45",
        ]

        async def events():
            for snapshot in snapshots:
                yield "plan_item", {"id": "p1", "thought": snapshot}
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, _ = _parse_chunks(chunks)
        streamed = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(streamed, "Leg 1: 340 km = 4h.\n20:45")
        self.assertFalse(streamed.startswith("The"))

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"],
            "Leg 1: 340 km = 4h.\n20:45",
        )

    def test_plain_answer_is_never_altered_by_the_filter(self):
        for answer in ("20:45", "340/85 = 4 hours.\n20:45", "Use s[::-1] to reverse."):
            with self.subTest(answer=answer):
                self.assertEqual(sse.strip_reasoning_narration(answer), answer)

    def test_verbose_reasoning_opt_out_keeps_narration(self):
        narrated = "The user wants a reversal.\nUse s[::-1]."
        with patch.dict(os.environ, {"TRAE_VERBOSE_REASONING": "1"}, clear=False):
            self.assertEqual(sse.strip_reasoning_narration(narrated), narrated)

    def test_raw_cache_read_tokens_are_reported(self):
        usage = sse._map_usage(
            {
                "input_tokens": 29558,
                "output_tokens": 11,
                "cache_read_tokens": 28544,
                "cache_write_tokens": 0,
            }
        )

        self.assertEqual(usage["prompt_tokens"], 29558)
        self.assertEqual(usage["completion_tokens"], 11)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 28544)

        plain = sse._map_usage({"input_token": 5, "output_token": 2})
        self.assertNotIn("prompt_tokens_details", plain)

    def test_web_message_event_content_is_translated(self):
        async def events():
            yield "message", {
                "content": [{"type": "text", "text": "Hello "}],
            }
            yield "message", {
                "content": [{"type": "text", "text": "Hello world"}],
            }
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertIn("Hello world", content)
        self.assertTrue(done)

    def test_web_empty_stream_yields_visible_placeholder(self):
        async def events():
            yield "heartbeat", {"ts": 1}
            yield "done", {}

        chunks = asyncio.run(_collect(sse.translate_web_events(events(), "m")))
        parsed, done = _parse_chunks(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertIn("trae upstream returned an empty response", content)
        self.assertTrue(done)

    def test_web_empty_nonstream_uses_placeholder(self):
        async def events():
            yield "done", {}

        result = asyncio.run(sse.collect_nonstream_web(events(), "m"))
        self.assertEqual(
            result["choices"][0]["message"]["content"],
            "(trae upstream returned an empty response)",
        )
    def test_web_empty_stream_fail_on_empty_raises(self):
        async def events():
            yield "heartbeat", {"ts": 1}
            yield "done", {}

        with self.assertRaises(sse.EmptyUpstreamResponse) as raised:
            asyncio.run(_collect(sse.translate_web_events(events(), "m", fail_on_empty=True)))
        self.assertTrue(raised.exception.retryable)
        self.assertIsNone(raised.exception.usage)

    def test_web_empty_stream_fail_on_empty_emits_no_data_frame(self):
        async def events():
            yield "done", {}

        collected = []

        async def run():
            async for chunk in sse.translate_web_events(
                events(), "m", fail_on_empty=True
            ):
                collected.append(chunk)

        with self.assertRaises(sse.EmptyUpstreamResponse):
            asyncio.run(run())
        self.assertEqual(collected, [])

    def test_web_nonempty_fail_on_empty_emits_role_then_content(self):
        async def events():
            yield "message", {"content": [{"type": "text", "text": "hi"}]}
            yield "done", {}

        chunks = asyncio.run(
            _collect(
                sse.translate_web_events(events(), "m", fail_on_empty=True)
            )
        )
        parsed, done = _parse_chunks(chunks)
        roles = [
            choice["delta"].get("role")
            for event in parsed
            for choice in event.get("choices", [])
            if choice.get("delta", {}).get("role")
        ]
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(roles, ["assistant"])
        self.assertIn("hi", content)
        self.assertTrue(done)

    def test_web_observed_usage_marks_empty_response_not_retryable(self):
        async def events():
            yield "token_usage", {"prompt_tokens": 3, "completion_tokens": 0}
            yield "done", {}

        with self.assertRaises(sse.EmptyUpstreamResponse) as raised:
            asyncio.run(
                _collect(
                    sse.translate_web_events(events(), "m", fail_on_empty=True)
                )
            )
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(raised.exception.usage.get("prompt_tokens"), 3)

    def test_web_empty_nonstream_fail_on_empty_raises(self):
        async def events():
            yield "done", {}

        with self.assertRaises(sse.EmptyUpstreamResponse) as raised:
            asyncio.run(
                sse.collect_nonstream_web(events(), "m", fail_on_empty=True)
            )
        self.assertTrue(raised.exception.retryable)
        self.assertIsNone(raised.exception.usage)

if __name__ == "__main__":
    unittest.main()
