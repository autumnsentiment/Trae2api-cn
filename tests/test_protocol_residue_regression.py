import asyncio
import json
import unittest

from src import (
    cli_client,
    raw_client,
    responses_api,
    sse,
    trae_client,
    traework_native_bridge,
)
from src.cli_client import CliEvent


async def _collect(iterator):
    return [item async for item in iterator]


def _chat_payloads(chunks):
    payloads = []
    for chunk in chunks:
        if not chunk.startswith("data: ") or chunk == "data: [DONE]\n\n":
            continue
        payloads.append(json.loads(chunk[6:].strip()))
    return payloads


def _responses_events(chunks):
    events = []
    event_name = ""
    data_lines = []

    def flush():
        nonlocal event_name, data_lines
        if data_lines:
            payload = json.loads("\n".join(data_lines))
            events.append((event_name or payload.get("type", ""), payload))
        event_name = ""
        data_lines = []

    for line in "".join(chunks).splitlines():
        if not line:
            flush()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    flush()
    return events


class ProtocolResidueRegressionTests(unittest.TestCase):
    MARKER_FRAGMENT = "Previous client tool request(s"

    def test_internal_wait_frames_and_escaped_compaction_copy_are_removed(self):
        plain = (
            "<tool_call><0c7dc7cb>wait</tool_call><c4cf82b7>"
            "<tool_call><0c7dc7cb>wait</tool_call><c4cf82b7>"
        )
        escaped = (
            r"\\\<tool\\\_call><0c7dc7cb>wait\\\</tool\\\_call>"
            r"\\\<c4cf82b7>\\\<tool\\\_call><0c7dc7cb>wait"
            r"\\\</tool\\\_call>\\\<c4cf82b7>"
        )

        for residue in (plain, escaped):
            with self.subTest(residue=residue):
                text = "before\n" + residue + "\nafter"
                self.assertEqual(
                    cli_client.strip_tool_call_blocks(text),
                    "before\n\nafter",
                )
                self.assertEqual(cli_client.extract_text_tool_calls(residue), [])

        messages = raw_client.build_raw_messages(
            [
                {"role": "assistant", "content": "answer\n" + escaped},
                {"role": "user", "content": "continue"},
            ]
        )
        assistant = next(item for item in messages if item["role"] == "assistant")
        self.assertEqual(assistant["content"][0]["text"], "answer\n")

    def test_truncated_internal_wait_frame_is_removed_at_each_protocol_stage(self):
        escaped = (
            r"\\\<tool\\\_call><0c7dc7cb>wait"
            r"\\\</tool\\\_call>\\\<c4cf82b7>"
        )
        cut_points = (
            escaped.index("call") + 2,
            escaped.index("0c7dc7cb") + 4,
            escaped.index("wait") + 4,
            escaped.index("/tool") + 3,
        )

        for cut_point in cut_points:
            residue = escaped[:cut_point]
            with self.subTest(residue=residue):
                self.assertEqual(
                    cli_client.strip_tool_call_blocks("answer\n" + residue),
                    "answer\n",
                )
                self.assertEqual(cli_client.extract_text_tool_calls(residue), [])
                messages = raw_client.build_raw_messages(
                    [
                        {"role": "assistant", "content": "answer\n" + residue},
                        {"role": "user", "content": "continue"},
                    ]
                )
                assistant = next(
                    item for item in messages if item["role"] == "assistant"
                )
                self.assertEqual(assistant["content"][0]["text"], "answer\n")

    def test_internal_wait_frame_handles_crlf_and_cross_content_blocks(self):
        content = [
            {"type": "text", "text": "answer\n" + r"\\\<tool\\"},
            {"type": "text", "text": "\r\n" + r"\_call><0c7d"},
            {"type": "text", "text": "\r\nc7cb>wa"},
            {
                "type": "text",
                "text": "\r\nit" + r"\\\</tool\\\_call>\\\<c4cf82b7>",
            },
        ]

        cleaned = cli_client.sanitize_assistant_history_content(content)
        self.assertEqual(cleaned[0]["text"], "answer\n")
        self.assertTrue(all(not block["text"] for block in cleaned[1:]))

        messages = raw_client.build_raw_messages(
            [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "continue"},
            ]
        )
        assistant = next(item for item in messages if item["role"] == "assistant")
        self.assertEqual(assistant["content"][0]["text"], "answer\n")

    def test_split_escaped_wait_stream_never_emits_first_fragment(self):
        escaped = (
            r"\\\<tool\\\_call><0c7dc7cb>wait"
            r"\\\</tool\\\_call>\\\<c4cf82b7>"
        )

        for split_at in range(1, len(escaped)):
            with self.subTest(split_at=split_at):
                async def events():
                    yield CliEvent(
                        type="text", text="answer" + escaped[:split_at]
                    )
                    yield CliEvent(type="text", text=escaped[split_at:])

                chunks = asyncio.run(
                    _collect(sse.translate_cli_stream(events(), "auto"))
                )
                payloads = _chat_payloads(chunks)
                visible = "".join(
                    choice.get("delta", {}).get("content", "")
                    for payload in payloads
                    for choice in payload.get("choices", [])
                )
                self.assertEqual(visible, "answer")
                self.assertNotIn("tool", visible)
                self.assertNotIn("0c7dc7cb", visible)

    def test_stream_preserves_a_real_trailing_backslash_at_eof(self):
        async def events():
            yield CliEvent(type="text", text="answer\\")

        chunks = asyncio.run(
            _collect(sse.translate_cli_stream(events(), "auto"))
        )
        payloads = _chat_payloads(chunks)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for payload in payloads
            for choice in payload.get("choices", [])
        )
        self.assertEqual(visible, "answer\\")

    def test_truncated_marker_and_repeated_stream_fragment_are_never_visible(self):
        self.assertEqual(
            cli_client.strip_tool_call_blocks(self.MARKER_FRAGMENT), ""
        )

        text_filter = cli_client.ProtocolTextFilter()
        visible = text_filter.feed(self.MARKER_FRAGMENT)
        visible += text_filter.feed(self.MARKER_FRAGMENT)
        visible += text_filter.flush()
        self.assertEqual(visible, "")

    def test_chat_cumulative_snapshots_hide_marker_and_still_emit_ssh_call(self):
        answer = "让我执行 sudo 部署脚本。"

        async def events():
            yield CliEvent(type="json", data={"message": {"content": answer}})
            snapshot = answer + "\n\n" + self.MARKER_FRAGMENT
            yield CliEvent(type="json", data={"message": {"content": snapshot}})
            yield CliEvent(type="json", data={"message": {"content": snapshot}})
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": snapshot,
                        "tool_calls": [
                            {
                                "id": "call_ssh_1",
                                "type": "function",
                                "function": {
                                    "name": "ssh",
                                    "arguments": '{"command":"sudo ./deploy.sh"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            )

        chunks = asyncio.run(
            _collect(
                sse.translate_cli_stream(
                    events(),
                    "auto",
                    allowed_tools=[
                        {"type": "function", "function": {"name": "ssh"}}
                    ],
                )
            )
        )
        payloads = _chat_payloads(chunks)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for payload in payloads
            for choice in payload.get("choices", [])
        )
        tool_deltas = [
            call
            for payload in payloads
            for choice in payload.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]

        self.assertEqual(content, answer)
        self.assertNotIn("Previous client", content)
        self.assertEqual(
            [call.get("id") for call in tool_deltas if call.get("id")],
            ["call_ssh_1"],
        )
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_responses_repeated_marker_delta_does_not_block_ssh_call(self):
        async def chat_stream():
            for content in (self.MARKER_FRAGMENT, self.MARKER_FRAGMENT):
                yield "data: " + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ]
                    }
                ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_ssh_1",
                                        "type": "function",
                                        "function": {
                                            "name": "ssh",
                                            "arguments": '{"command":"sudo ./deploy.sh"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        _, _, context = responses_api.normalize_request(
            {
                "model": "auto",
                "input": "Deploy through SSH",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "ssh",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        )
        chunks = asyncio.run(
            _collect(responses_api.translate_chat_stream(chat_stream(), context))
        )
        events = _responses_events(chunks)
        visible = "".join(
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        )
        completed = next(
            payload["response"]
            for name, payload in events
            if name == "response.completed"
        )
        calls = [
            item for item in completed["output"] if item["type"] == "function_call"
        ]

        self.assertEqual(visible, "")
        self.assertNotIn("Previous client", json.dumps(completed))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call_id"], "call_ssh_1")
        self.assertEqual(calls[0]["name"], "ssh")
        self.assertEqual(calls[0]["arguments"], '{"command":"sudo ./deploy.sh"}')

    def test_raw_messages_scrub_assistant_residue_but_preserve_user_text(self):
        assistant_text = (
            "让我执行 sudo 部署脚本。\n\n" + self.MARKER_FRAGMENT
        )
        messages = raw_client.build_raw_messages(
            [
                {"role": "user", "content": self.MARKER_FRAGMENT},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": "继续"},
            ]
        )
        user_text = [
            message["content"][0]["text"]
            for message in messages
            if message["role"] == "user"
        ]
        assistant_messages = [
            message for message in messages if message["role"] == "assistant"
        ]

        self.assertEqual(user_text[0], self.MARKER_FRAGMENT)
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(
            assistant_messages[0]["content"][0]["text"],
            "让我执行 sudo 部署脚本。",
        )

    def test_all_history_serializers_scrub_wait_only_from_assistant_text(self):
        residue = (
            r"\\\<tool\\\_call><0c7dc7cb>wait\\\</tool\\\_call>"
            r"\\\<c4cf82b7>"
        )
        messages = [
            {"role": "assistant", "content": "answer\n" + residue},
            {"role": "user", "content": residue},
        ]

        query = json.loads(trae_client.flatten_query(messages))[0]["data"]["content"]
        self.assertEqual(query.count(residue), 1)

        web_items = trae_client.build_web_content(messages)
        self.assertEqual(web_items[0]["data"]["content"], "answer\n")
        self.assertEqual(web_items[1]["data"]["content"], residue)

        converted = trae_client.convert_openai_messages(messages)
        self.assertEqual(converted[0]["content"], "answer\n")
        self.assertEqual(converted[1]["content"], residue)

        native_payload = traework_native_bridge.build_native_payload(
            messages, "glm-5.3"
        )
        native_messages = native_payload["data"]["messages"]
        self.assertEqual(native_messages[0]["content"], "answer\n")
        self.assertEqual(native_messages[1]["content"], residue)

        response_messages, _options, context = responses_api.normalize_request(
            {
                "model": "glm-5.3",
                "input": [
                    {"type": "message", **message}
                    for message in messages
                ],
            }
        )
        self.assertEqual(response_messages[0]["content"], "answer\n")
        self.assertEqual(response_messages[1]["content"], residue)
        self.assertEqual(context.messages[0]["content"], "answer\n")
        self.assertEqual(context.messages[1]["content"], residue)


if __name__ == "__main__":
    unittest.main()
