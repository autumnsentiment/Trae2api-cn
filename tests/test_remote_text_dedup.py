import asyncio
import json
import unittest

from src import sse


def _event_factory(events):
    async def iterator():
        for event, data in events:
            yield event, data

    return iterator


def _stream_payloads(events):
    chunks = asyncio.run(
        _collect(
            sse.translate_web_events(
                _event_factory(events)(),
                "test-model",
            )
        )
    )
    payloads = []
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.startswith("data: "):
            continue
        raw = chunk[6:].strip()
        if raw == "[DONE]":
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


async def _collect(source):
    return [chunk async for chunk in source]


def _stream_text(events):
    text = []
    for payload in _stream_payloads(events):
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            value = delta.get("content")
            if isinstance(value, str):
                text.append(value)
    return "".join(text)


def _stream_tool_calls(events):
    calls = []
    for payload in _stream_payloads(events):
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            calls.extend(delta.get("tool_calls") or [])
    return calls


def _nonstream_message(events):
    result = asyncio.run(
        sse.collect_nonstream_web(
            _event_factory(events)(),
            "test-model",
        )
    )
    return result["choices"][0]["message"]


def _plan(text, plan_id="p1"):
    # The split-reasoning shape makes ``thought`` the public answer snapshot.
    return (
        "plan_item",
        {
            "id": plan_id,
            "thought": text,
            "reasoning_content": "",
        },
    )


def _message(text, message_id="m1"):
    return "message", {"id": message_id, "content": text}


class RemoteTextDedupTests(unittest.TestCase):
    def test_exact_plan_and_message_are_emitted_once_in_stream_and_nonstream(self):
        events = [
            _plan("Hello world"),
            _message("Hello world"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "Hello world")
        self.assertEqual(_nonstream_message(events)["content"], "Hello world")

    def test_prefix_growth_and_mirrored_message_emit_only_new_suffix(self):
        events = [
            _plan("Hello"),
            _plan("Hello world"),
            _message("Hello world"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "Hello world")
        self.assertEqual(_nonstream_message(events)["content"], "Hello world")

    def test_suffix_message_replay_does_not_repeat_the_plan_answer(self):
        events = [
            _plan("Hello world"),
            _message("world"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "Hello world")
        self.assertEqual(_nonstream_message(events)["content"], "Hello world")

    def test_distinct_plan_ids_with_same_text_are_preserved(self):
        events = [
            _plan("same", "p1"),
            _plan("same", "p2"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "samesame")
        self.assertEqual(_nonstream_message(events)["content"], "same\n\nsame")

    def test_message_before_plan_mirror_is_not_replayed(self):
        events = [
            _message("Hello world"),
            _plan("Hello world"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "Hello world")
        self.assertEqual(_nonstream_message(events)["content"], "Hello world")

    def test_unrelated_plan_and_message_are_both_kept(self):
        events = [
            _plan("Plan output."),
            _message("Final answer."),
            ("done", {}),
        ]

        expected = "Plan output.\n\nFinal answer."
        self.assertEqual(_stream_text(events), expected)
        self.assertEqual(_nonstream_message(events)["content"], expected)

    def test_xml_tool_extraction_survives_cross_source_text_deduplication(self):
        block = (
            '<opencode_tool_call>{"id":"call_read_1","name":"read_file",'
            '"input":{"path":"README.md"}}</opencode_tool_call>'
        )
        events = [
            _plan("Before\n" + block),
            _message("Before\n" + block),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "Before\n")
        stream_calls = _stream_tool_calls(events)
        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_calls[0]["id"], "call_read_1")

        message = _nonstream_message(events)
        self.assertEqual(message["content"], "Before")
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["id"], "call_read_1")

    def test_trailing_backslash_is_released_once_on_finalize(self):
        events = [
            _plan("answer\\"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "answer\\")
        self.assertEqual(_nonstream_message(events)["content"], "answer\\")

    def test_message_trailing_backslash_is_released_once_on_finalize(self):
        events = [
            _message("answer\\"),
            ("done", {}),
        ]

        self.assertEqual(_stream_text(events), "answer\\")
        self.assertEqual(_nonstream_message(events)["content"], "answer\\")


if __name__ == "__main__":
    unittest.main()
