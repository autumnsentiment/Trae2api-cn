import asyncio
import json
import unittest

from src import cli_client, raw_client, responses_api, sse
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


if __name__ == "__main__":
    unittest.main()
