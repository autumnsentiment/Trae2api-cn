"""Kimi K3 native special-token tool calls through the text protocol."""

from __future__ import annotations

import asyncio
import json
import unittest

from src import cli_client, raw_client, sse


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def _block(name: str, arguments: str, index: int = 0) -> str:
    return (
        "<|tool_call_begin|>"
        f"functions.{name}:{index}"
        "<|tool_call_argument_begin|>"
        f"{arguments}"
        "<|tool_call_end|>"
    )


def _section(*blocks: str) -> str:
    return (
        "<|tool_calls_section_begin|>"
        + "".join(blocks)
        + "<|tool_calls_section_end|>"
    )


class _LineResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


async def _collect(chunks):
    return [chunk async for chunk in chunks]


def _parse_chunks(chunks: list[str]) -> tuple[list[dict], bool]:
    parsed: list[dict] = []
    done = False
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            done = True
        elif chunk.startswith("data: "):
            parsed.append(json.loads(chunk[6:].strip()))
    return parsed, done


class KimiK3TextProtocolTests(unittest.TestCase):
    def test_native_section_parses_a_single_call_and_is_not_visible(self):
        content = "thinking out loud\n" + _section(
            _block("Read", '{"filePath":"README.md"}')
        )

        calls = cli_client.extract_text_tool_calls(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "Read")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"filePath": "README.md"},
        )
        self.assertEqual(
            cli_client.strip_tool_call_blocks(content), "thinking out loud\n"
        )

    def test_multiple_native_calls_stay_distinct(self):
        content = _section(
            _block("Read", '{"filePath":"README.md"}', 0),
            _block("Bash", '{"command":"git status"}', 1),
        )

        calls = cli_client.extract_text_tool_calls(content)

        self.assertEqual(
            [call["function"]["name"] for call in calls], ["Read", "Bash"]
        )
        self.assertEqual(len({call["id"] for call in calls}), 2)
        self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_optional_functions_prefix_and_empty_arguments(self):
        cases = (
            (_block("Read", '{"filePath":"a.txt"}'), "Read", {"filePath": "a.txt"}),
            (_block("Read", "{}"), "Read", {}),
            (
                "<|tool_call_begin|>Bash:0<|tool_call_argument_begin|>"
                '{"command":"dir"}<|tool_call_end|>',
                "Bash",
                {"command": "dir"},
            ),
        )
        for content, expected_name, expected_arguments in cases:
            with self.subTest(content=content):
                calls = cli_client.extract_text_tool_calls(content)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["name"], expected_name)
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]),
                    expected_arguments,
                )
                self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_malformed_arguments_are_never_treated_as_a_second_call(self):
        content = _section(
            _block(
                "Bash",
                '{"command":"echo hi","nested":{"name":"Read",'
                '"arguments":{"filePath":"fake.txt"}}} trailing',
            )
        )

        calls = cli_client.extract_text_tool_calls(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "Bash")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["command"],
            "echo hi",
        )
        self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_truncated_native_marker_is_removed_from_visible_text(self):
        content = (
            "answer before\n"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.Read:0"
            "<|tool_call_argument_begin|>{\"filePath\":\"README.md\"}"
        )

        self.assertEqual(
            cli_client.strip_tool_call_blocks(content), "answer before\n"
        )
        self.assertEqual(cli_client.extract_text_tool_calls(content), [])


class KimiK3StreamingTests(unittest.TestCase):
    def test_native_function_call_completed_at_eof_is_emitted_as_valid_json(self):
        lines = [
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_kimi_eof",
                            "type": "function",
                            "function_call": {"name": "Read", "arguments": ""},
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
                            "function_call": {
                                "arguments": '{"filePath":"README.md"}'
                            },
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]

        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    _LineResponse(lines),
                    "kimi-k3",
                    allowed_tools=TOOLS,
                )
            )
        )
        parsed, done = _parse_chunks(chunks)
        calls = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
            if call.get("id")
        ]

        self.assertTrue(done)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_kimi_eof")
        self.assertEqual(calls[0]["function"]["name"], "Read")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"filePath": "README.md"},
        )
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_split_native_markers_never_leak_and_emit_tool_call(self):
        fragments = [
            "prefix ",
            "<|tool_calls_section_",
            "begin|><|tool_call_",
            'begin|>functions.Read:0<|tool_call_argument_begin|>{"filePath":',
            '"README.md"}<|tool_call_end|>',
            "<|tool_calls_section_end|>",
        ]
        lines: list[str] = []
        for fragment in fragments:
            lines.append("data: " + json.dumps({"response": fragment}))
        lines.extend(('data: {"finish_reason":"tool_calls"}', "data: [DONE]"))

        chunks = asyncio.run(
            _collect(
                sse.translate_ide_stream(
                    _LineResponse(lines),
                    "kimi-k3",
                    allowed_tools=TOOLS,
                )
            )
        )
        parsed, done = _parse_chunks(chunks)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        calls = [
            call
            for event in parsed
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
            if call.get("id")
        ]

        self.assertTrue(done)
        self.assertEqual(visible, "prefix ")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "Read")
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_nonstream_multiple_native_calls_have_valid_json_and_tool_reason(self):
        content = "working\n" + _section(
            _block("Read", '{"filePath":"README.md"}', 0),
            _block("Bash", '{"command":"git status"}', 1),
        )
        response = _LineResponse(
            [
                "data: " + json.dumps({"response": content}),
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )

        result = asyncio.run(
            sse.collect_nonstream_ide(
                response,
                "kimi-k3",
                allowed_tools=TOOLS,
            )
        )
        choice = result["choices"][0]

        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "working\n")
        self.assertEqual(
            [call["function"]["name"] for call in choice["message"]["tool_calls"]],
            ["Read", "Bash"],
        )
        for call in choice["message"]["tool_calls"]:
            json.loads(call["function"]["arguments"])

    def test_native_tool_call_in_reasoning_is_extracted_without_leaking(self):
        content = _section(_block("Read", '{"filePath":"README.md"}'))
        response = _LineResponse(
            [
                "data: " + json.dumps({"reasoning_content": content}),
                "data: [DONE]",
            ]
        )

        result = asyncio.run(
            sse.collect_nonstream_ide(
                response,
                "kimi-k3",
                allowed_tools=TOOLS,
            )
        )
        choice = result["choices"][0]

        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            choice["message"]["tool_calls"][0]["function"]["name"], "Read"
        )

    def test_native_call_respects_allowed_tool_catalog(self):
        content = _section(_block("Bash", '{"command":"git status"}'))
        response = _LineResponse(
            [
                "data: " + json.dumps({"response": content}),
                "data: [DONE]",
            ]
        )

        result = asyncio.run(
            sse.collect_nonstream_ide(
                response,
                "kimi-k3",
                allowed_tools=TOOLS[:1],
            )
        )
        choice = result["choices"][0]

        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertEqual(
            choice["message"]["content"],
            "(trae upstream returned an empty response)",
        )

    def test_tool_result_continuation_keeps_call_id_and_name(self):
        call = cli_client.extract_text_tool_calls(
            _section(_block("Read", '{"filePath":"README.md"}'))
        )[0]
        body = raw_client.build_raw_chat_body(
            [
                {"role": "user", "content": "Read the project README"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call],
                },
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "Read",
                    "content": "# Project\n\nContents.",
                },
            ],
            "kimi-k3",
            {"tools": TOOLS},
        )
        serialized = json.dumps(body["messages"], ensure_ascii=False)

        self.assertIn(f"Client tool call [{call['id']}] Read", serialized)
        self.assertIn(
            f"Client tool result [{call['id']}] Read status=succeeded",
            serialized,
        )
        self.assertIn("# Project", serialized)
        self.assertNotIn("<|tool_call_begin|>", serialized)


if __name__ == "__main__":
    unittest.main()
