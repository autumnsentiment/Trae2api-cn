"""Regression coverage for SOLO tool deltas and indexed call correlation."""

from __future__ import annotations

import json
import unittest
from typing import Any

from src import cli_client, sse


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]


class LineResponse:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def iter_lines(self):
        return iter(self.lines)


def _call(
    arguments: Any,
    *,
    name: str = "",
    call_id: str = "",
    index: int = 0,
    representation: str = "function_call",
) -> dict:
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        representation: {"name": name, "arguments": arguments},
    }


def _response(frames: list[list[dict]], *, terminal: bool = True) -> LineResponse:
    lines: list[str] = []
    for calls in frames:
        lines.extend(["event: output", "data: " + json.dumps({"tool_calls": calls})])
    lines.extend(
        ["event: token_usage", 'data: {"input_token":3,"output_token":5}']
    )
    if terminal:
        lines.extend(["event: done", 'data: {"finish_reason":"tool_calls"}'])
    return LineResponse(lines)


class IdeNativeToolDeltaTests(unittest.IsolatedAsyncioTestCase):
    async def _calls(self, frames: list[list[dict]], stream: bool) -> list[dict]:
        if not stream:
            result = await sse.collect_nonstream_ide(
                _response(frames), "m", allowed_tools=TOOLS, fail_on_empty=True
            )
            self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
            return result["choices"][0]["message"]["tool_calls"]
        merged: dict[int, dict] = {}
        finishes: list[str] = []
        done_count = 0
        async for chunk in sse.translate_ide_stream(
            _response(frames), "m", allowed_tools=TOOLS, fail_on_empty=True
        ):
            if chunk == "data: [DONE]\n\n":
                done_count += 1
                continue
            if not chunk.startswith("data: "):
                continue
            payload = json.loads(chunk[6:])
            for choice in payload["choices"]:
                if choice["finish_reason"]:
                    finishes.append(choice["finish_reason"])
                for delta in choice["delta"].get("tool_calls", []):
                    self.assertFalse(
                        any(key.startswith("_") for key in delta),
                        "Internal representation markers must not reach API callers",
                    )
                    index = delta["index"]
                    if index not in merged:
                        merged[index] = {
                            "id": delta["id"],
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    elif "id" in delta:
                        self.assertEqual(delta["id"], merged[index]["id"])
                    function = delta.get("function") or {}
                    for key in ("name", "arguments"):
                        merged[index]["function"][key] += function.get(key) or ""
        self.assertEqual(finishes, ["tool_calls"])
        self.assertEqual(done_count, 1)
        return list(merged.values())

    async def test_native_nested_fragments_are_appended_even_when_they_match_a_prefix(self):
        frames = [
            [_call("", name="read_file", call_id="call_native")],
            *[
                [_call(fragment)]
                for fragment in ("{", '"outer":', "{", '"x":1', "}", "}")
            ],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["id"], "call_native")
                self.assertEqual(calls[0]["function"]["arguments"], '{"outer":{"x":1}}')

    async def test_named_fragment_without_id_keeps_the_initial_call_identity(self):
        frames = [
            [_call('{"path":', name="read_file", call_id="call_native")],
            [_call('"README.md"}', name="read_file")],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["id"], "call_native")
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"README.md"}')

    async def test_idless_parallel_fragments_follow_the_explicit_index(self):
        frames = [
            [
                _call('{"path":', name="read_file", index=0),
                _call('{"path":', name="read_file", index=1),
            ],
            [_call('"b.txt"}', name="read_file", index=1)],
            [_call('"a.txt"}', index=0)],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len({call["id"] for call in calls}), 2)
                self.assertEqual(
                    [call["function"]["arguments"] for call in calls],
                    ['{"path":"a.txt"}', '{"path":"b.txt"}'],
                )

    async def test_standard_native_split_preserves_empty_first_args_and_nameless_tail(self):
        head = _call("", name="read_file", call_id="call_native")
        head["function_call"] = {"name": "read_file", "args": ""}
        frames = [[head], [_call('{"path":"README.md"}')]]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(calls[0]["id"], "call_native")
                self.assertEqual(calls[0]["function"]["name"], "read_file")
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"README.md"}')

    async def test_name_only_native_metadata_does_not_prefix_a_fabricated_object(self):
        head = _call("", name="read_file", call_id="call_native")
        head["function_call"].pop("arguments")
        frames = [[head], [_call('{"path":"README.md"}')]]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"README.md"}')

    async def test_cumulative_function_snapshots_remain_cumulative(self):
        frames = [
            [
                _call(
                    '{"path":', name="read_file", representation="function"
                )
            ],
            [
                _call(
                    '{"path":"README.md"}',
                    name="read_file",
                    representation="function",
                )
            ],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"README.md"}')

    async def test_parallel_cumulative_snapshots_cannot_merge_different_indexes(self):
        frames = [
            [
                _call(
                    '{"path":', name="read_file", index=0, representation="function"
                ),
                _call(
                    '{"path":', name="read_file", index=1, representation="function"
                ),
            ],
            [
                _call(
                    '{"path":"b.txt"}',
                    name="read_file",
                    index=1,
                    representation="function",
                )
            ],
            [
                _call(
                    '{"path":"a.txt"}',
                    name="read_file",
                    index=0,
                    representation="function",
                )
            ],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    [call["function"]["arguments"] for call in calls],
                    ['{"path":"a.txt"}', '{"path":"b.txt"}'],
                )

    async def test_explicit_native_snapshot_representation_is_preserved(self):
        first = _call('{"path":', name="read_file", call_id="call_native")
        first["_arguments_mode"] = "snapshot"
        second = _call('{"path":"README.md"}', name="read_file")
        second["_arguments_mode"] = "snapshot"
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls([[first], [second]], stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"README.md"}')

    async def test_native_structured_argument_objects_keep_snapshot_semantics(self):
        frames = [
            [
                _call(
                    {"path": "README.md"}, name="read_file", call_id="call_native"
                )
            ]
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "README.md"})

    async def test_invalid_final_native_json_does_not_emit_success(self):
        frames = [
            [_call("", name="read_file", call_id="call_native")],
            [_call('{"path":')],
        ]
        with self.assertRaises(sse.InvalidNativeToolArguments) as nonstream:
            await self._calls(frames, False)
        self.assertFalse(nonstream.exception.retryable)
        self.assertTrue(nonstream.exception.observed_model_event)
        self.assertEqual(nonstream.exception.usage["total_tokens"], 8)
        chunks: list[str] = []
        with self.assertRaises(sse.InvalidNativeToolArguments):
            async for chunk in sse.translate_ide_stream(
                _response(frames), "m", allowed_tools=TOOLS, fail_on_empty=True
            ):
                chunks.append(chunk)
        self.assertNotIn("data: [DONE]\n\n", chunks)
        self.assertFalse(
            any('"finish_reason": "tool_calls"' in chunk for chunk in chunks)
        )

    async def test_missing_terminal_is_still_incomplete(self):
        frames = [[_call("{}", name="read_file", call_id="call_native")]]
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(sse.IncompleteUpstreamResponse):
                    if stream:
                        async for _ in sse.translate_ide_stream(
                            _response(frames, terminal=False),
                            "m",
                            allowed_tools=TOOLS,
                        ):
                            pass
                    else:
                        await sse.collect_nonstream_ide(
                            _response(frames, terminal=False), "m", allowed_tools=TOOLS
                        )

    def test_normalization_preserves_native_representation_through_replays(self):
        native = cli_client.normalize_tool_call(
            _call("{", name="read_file", call_id="call_native")
        )
        self.assertEqual(native["_arguments_mode"], "delta")
        replay = cli_client.normalize_tool_call(native)
        self.assertEqual(replay["_arguments_mode"], "delta")
        snapshot = cli_client.normalize_tool_call(
            _call("{}", name="read_file", representation="function")
        )
        self.assertNotIn("_arguments_mode", snapshot)

    def test_empty_initial_native_record_keeps_its_mode_after_accumulation(self):
        native = cli_client.extract_tool_calls(
            {"tool_calls": [_call("", name="read_file", call_id="call_native")]}
        )
        accumulator = sse.ToolCallAccumulator()
        accumulator.add(accumulator.prepare(native))
        self.assertEqual(accumulator.calls()[0]["_arguments_mode"], "delta")
        tail = cli_client.extract_tool_calls({"tool_calls": [_call("{")]})
        accumulator.add(accumulator.prepare(tail))
        self.assertEqual(accumulator.calls()[0]["function"]["arguments"], "{")
        with self.assertRaises(sse.InvalidNativeToolArguments):
            sse._ensure_native_tool_arguments(accumulator.calls())

    async def test_zero_argument_native_call_still_returns_a_json_object(self):
        frames = [[_call("{}", name="read_file", call_id="call_native")]]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(calls[0]["function"]["arguments"], "{}")


if __name__ == "__main__":
    unittest.main()
