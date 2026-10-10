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

    async def test_replayed_complete_native_call_with_same_real_id_is_emitted_once(self):
        frames = [
            [
                _call(
                    '{"path":"README.md"}',
                    name="read_file",
                    call_id="call_native",
                )
            ],
            [
                _call(
                    '{"path":"README.md"}',
                    name="read_file",
                    call_id="call_native",
                )
            ],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["id"], "call_native")
                self.assertEqual(
                    calls[0]["function"]["arguments"], '{"path":"README.md"}'
                )

    async def test_replayed_native_call_uses_canonical_json_for_duplicate_detection(self):
        first = '{"path":"README.md","line":1}'
        replay = ' { "line": 1, "path": "README.md" } '
        frames = [
            [_call(first, name="read_file", call_id="call_native")],
            [_call(replay, name="read_file", call_id="call_native")],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                # Keep the first closed payload byte-for-byte; the replay is
                # only a duplicate even though its whitespace/key order differs.
                self.assertEqual(calls[0]["function"]["arguments"], first)

    async def test_conflicting_complete_native_replay_keeps_first_closed_arguments(self):
        frames = [
            [_call('{"path":"a.txt"}', name="read_file", call_id="call_native")],
            [_call('{"path":"b.txt"}', name="read_file", call_id="call_native")],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(
                    calls[0]["function"]["arguments"], '{"path":"a.txt"}'
                )

    async def test_idless_replay_with_same_explicit_index_is_emitted_once(self):
        frames = [
            [_call('{"path":"README.md"}', name="read_file", index=3)],
            [_call('{"path":"README.md"}', name="read_file", index=3)],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"README.md"}')

    async def test_distinct_real_ids_or_explicit_indexes_keep_identical_calls(self):
        arguments = '{"path":"README.md"}'
        first = _call(arguments, name="read_file", call_id="call_a")
        second = _call(arguments, name="read_file", call_id="call_b")
        first.pop("index")
        second.pop("index")
        cases = [
            (
                "real ids",
                [[
                    first,
                    second,
                ]],
                {"call_a", "call_b"},
            ),
            (
                "explicit indexes",
                [[
                    _call(arguments, name="read_file", index=0),
                    _call(arguments, name="read_file", index=1),
                ]],
                None,
            ),
        ]
        for stream in (False, True):
            for label, frames, expected_ids in cases:
                with self.subTest(stream=stream, case=label):
                    calls = await self._calls(frames, stream)
                    self.assertEqual(len(calls), 2)
                    if expected_ids is not None:
                        self.assertEqual(
                            {call["id"] for call in calls}, expected_ids
                        )
                    else:
                        self.assertEqual(
                            [call["function"]["arguments"] for call in calls],
                            [arguments, arguments],
                        )

    async def test_idless_unindexed_calls_dedup_only_when_arguments_match(self):
        def no_identity(arguments: str) -> dict:
            call = _call(arguments, name="read_file", call_id="")
            call.pop("index")
            return call

        same_frames = [
            [no_identity('{"path":"README.md"}')],
            [no_identity('{"path":"README.md"}')],
        ]
        different_frames = [
            [no_identity('{"path":"a.txt"}')],
            [no_identity('{"path":"b.txt"}')],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream, case="same"):
                calls = await self._calls(same_frames, stream)
                self.assertEqual(len(calls), 1)
            with self.subTest(stream=stream, case="different"):
                calls = await self._calls(different_frames, stream)
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    [call["function"]["arguments"] for call in calls],
                    ['{"path":"a.txt"}', '{"path":"b.txt"}'],
                )

    async def test_repeated_short_nested_fragments_remain_valid(self):
        frames = [
            [_call("", name="read_file", call_id="call_nested")],
            *[
                [_call(fragment)]
                for fragment in (
                    "{",
                    '"a":',
                    "{",
                    '"b":',
                    "{",
                    '"c":1',
                    "}",
                    "}",
                    "}",
                )
            ],
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]),
                    {"a": {"b": {"c": 1}}},
                )

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

    async def test_invalid_final_native_json_is_dropped_without_tool_success(self):
        frames = [
            [_call("", name="read_file", call_id="call_native")],
            [_call('{"path":')],
        ]
        with self.assertRaises(sse.EmptyUpstreamResponse) as nonstream:
            await self._calls(frames, False)
        self.assertFalse(nonstream.exception.retryable)
        self.assertTrue(nonstream.exception.observed_model_event)
        self.assertEqual(nonstream.exception.usage["total_tokens"], 8)
        chunks: list[str] = []
        with self.assertRaises(sse.EmptyUpstreamResponse):
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
        self.assertTrue(native["_explicit_id"])
        replay = cli_client.normalize_tool_call(native)
        self.assertEqual(replay["_arguments_mode"], "delta")
        self.assertTrue(replay["_explicit_id"])
        snapshot = cli_client.normalize_tool_call(
            _call("{}", name="read_file", representation="function")
        )
        self.assertNotIn("_arguments_mode", snapshot)
        self.assertFalse(snapshot["_explicit_id"])

    def test_prepare_preserves_inherited_id_as_non_explicit(self):
        head = cli_client.normalize_tool_call(
            _call("", name="read_file", call_id="call_native")
        )
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([head])

        tail = cli_client.normalize_tool_call(_call("{", name="read_file"))
        self.assertFalse(tail["_explicit_id"])
        prepared = accumulator.prepare([tail])[0]
        self.assertEqual(prepared["id"], "call_native")
        self.assertFalse(prepared["_explicit_id"])

        renormalized = cli_client.normalize_tool_call(prepared)
        self.assertEqual(renormalized["id"], "call_native")
        self.assertFalse(renormalized["_explicit_id"])

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
        repaired = sse._ensure_native_tool_arguments(accumulator.calls())
        self.assertEqual(repaired[0]["function"]["arguments"], "{}")

    def test_stacked_native_json_arguments_are_split_into_calls(self):
        call = cli_client.normalize_tool_call(
            _call('{"path":"a.txt"}{"path":"b.txt"}', name="read_file")
        )
        repaired = sse._ensure_native_tool_arguments([call])
        self.assertEqual(
            [item["function"]["arguments"] for item in repaired],
            ['{"path": "a.txt"}', '{"path": "b.txt"}'],
        )
        self.assertEqual(repaired[1]["id"], f"{call['id']}:1")

    def test_orphan_native_closing_fragment_is_dropped(self):
        call = cli_client.normalize_tool_call(_call("}", name="read_file"))
        self.assertEqual(sse._ensure_native_tool_arguments([call]), [])

    async def test_zero_argument_native_call_still_returns_a_json_object(self):
        frames = [[_call("{}", name="read_file", call_id="call_native")]]
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await self._calls(frames, stream)
                self.assertEqual(calls[0]["function"]["arguments"], "{}")


if __name__ == "__main__":
    unittest.main()
