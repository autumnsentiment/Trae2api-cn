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


if __name__ == "__main__":
    unittest.main()
