"""Replay the full public ingress -> SOLO -> caller tool result contract."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from src import main, responses_api, trae_client


FUNCTION = {
    "name": "read_file",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
CALL = {
    "id": "call_native_ingress",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path":"nonce.txt"}'},
}


class _ReplayResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def iter_lines(self):
        return iter(
            [
                "event: output",
                "data: " + json.dumps(self.payload),
                "",
                "event: done",
                'data: {"finish_reason":"stop"}',
                "",
            ]
        )


def _chat_stream_message(text):
    message = {"role": "assistant", "content": ""}
    calls = {}
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[6:])
        if payload.get("error"):
            raise AssertionError(payload["error"])
        for choice in payload.get("choices", []):
            delta = choice.get("delta", {})
            message["content"] += delta.get("content") or ""
            for raw in delta.get("tool_calls", []):
                call = calls.setdefault(
                    raw["index"],
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if raw.get("id"):
                    call["id"] = raw["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += raw.get("function", {}).get(key) or ""
    if calls:
        message["tool_calls"] = list(calls.values())
    return message


def _responses_result(response, stream):
    if not stream:
        return response.json()
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "response.completed":
            return payload["response"]
        if payload.get("type") in {"error", "response.failed"}:
            raise AssertionError(payload)
    raise AssertionError("Missing response.completed")


class IdeNativeIngressTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(main, "_USAGE_RECORDS_PATH", root / "usage.json"))
        self.stack.enter_context(patch.object(main, "_USAGE_HISTORY", []))
        self.stack.enter_context(patch.object(main, "API_KEYS", []))
        self.stack.enter_context(patch.object(main, "UPSTREAM_MODE", "ide"))
        self.stack.enter_context(patch.object(main, "_remote_only_models", return_value=set()))
        self.stack.enter_context(patch.object(main, "_auto_route_enabled", return_value=False))
        self.stack.enter_context(
            patch.object(main.auth, "get_polling_status", return_value={"enabled": False})
        )
        self.stack.enter_context(patch.object(main, "_bind_usage_turn_from_metadata", Mock()))
        self.stack.enter_context(patch.object(main, "_track_usage_from_result", Mock()))
        self.stack.enter_context(patch.object(main, "_track_usage_from_chunk", Mock()))
        self.stack.enter_context(
            patch.object(main, "_run_remote_with_retry", AsyncMock(side_effect=AssertionError(
                "Native ingress replay must not use Remote fallback"
            )))
        )
        responses_api._RESPONSE_SESSIONS.clear()
        self.addCleanup(responses_api._RESPONSE_SESSIONS.clear)
        self.addCleanup(self.stack.close)
        self.client = self.stack.enter_context(TestClient(main.app))
        self.bodies = []
        self.closers = []

        async def send(messages, model, stream, options=None):
            self.bodies.append(
                trae_client.build_llm_chat_body(messages, model, stream, options=options)
            )
            close = Mock()
            self.closers.append(close)
            payload = (
                {"tool_calls": [{
                    "id": CALL["id"],
                    "index": 0,
                    "type": "function",
                    "function_call": CALL["function"],
                }]}
                if len(self.bodies) == 1
                else {"response": "nonce-confirmed"}
            )
            return SimpleNamespace(response=_ReplayResponse(payload), close=close)

        self.stack.enter_context(patch.object(trae_client, "send_chat_request", new=send))

    def assert_native_second_round(self):
        history = self.bodies[1]["messages"]
        assistant = next(message for message in history if message.get("tool_calls"))
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], CALL["id"])
        self.assertEqual(call["function_call"], CALL["function"])
        result = next(message for message in history if message["role"] == "tool")
        self.assertEqual(result["tool_call_id"], CALL["id"])
        self.assertEqual(result["content"], [{"type": "text", "text": "nonce-confirmed"}])
        for body in self.bodies:
            self.assertEqual(body["function"], "solo_work_lite")
            self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        for close in self.closers:
            close.assert_called_once()

    def _chat_loop(self, stream):
        messages = [{"role": "user", "content": "Read nonce.txt and report its contents."}]
        options = {
            "model": "glm-5.3",
            "tools": [{"type": "function", "function": FUNCTION}],
            "parallel_tool_calls": False,
            "stream": stream,
        }
        first = self.client.post("/v1/chat/completions", json={**options, "messages": messages})
        self.assertEqual(first.status_code, 200, first.text)
        assistant = (
            _chat_stream_message(first.text)
            if stream
            else first.json()["choices"][0]["message"]
        )
        self.assertEqual(assistant["tool_calls"], [CALL])
        second = self.client.post(
            "/v1/chat/completions",
            json={
                **options,
                "messages": [
                    *messages,
                    assistant,
                    {"role": "tool", "tool_call_id": CALL["id"], "content": "nonce-confirmed"},
                ],
            },
        )
        self.assertEqual(second.status_code, 200, second.text)
        final = (
            _chat_stream_message(second.text)
            if stream
            else second.json()["choices"][0]["message"]
        )
        self.assertEqual(final["content"], "nonce-confirmed")
        self.assertFalse(final.get("tool_calls"))
        self.assert_native_second_round()

    def _responses_loop(self, stream):
        options = {
            "model": "glm-5.3",
            "tools": [{"type": "function", **FUNCTION}],
            "parallel_tool_calls": False,
            "stream": stream,
        }
        first = self.client.post(
            "/v1/responses", json={**options, "input": "Read nonce.txt and report its contents."}
        )
        self.assertEqual(first.status_code, 200, first.text)
        initial = _responses_result(first, stream)
        call = next(item for item in initial["output"] if item["type"] == "function_call")
        self.assertEqual(call["call_id"], CALL["id"])
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(json.loads(call["arguments"]), {"path": "nonce.txt"})
        second = self.client.post(
            "/v1/responses",
            json={
                **options,
                "previous_response_id": initial["id"],
                "input": [{
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "nonce-confirmed",
                }],
            },
        )
        self.assertEqual(second.status_code, 200, second.text)
        final = _responses_result(second, stream)
        self.assertFalse(any(item["type"] == "function_call" for item in final["output"]))
        content = [
            part["text"]
            for item in final["output"] if item["type"] == "message"
            for part in item["content"] if part["type"] == "output_text"
        ]
        self.assertEqual(content, ["nonce-confirmed"])
        self.assert_native_second_round()

    def test_chat_nonstream_native_tool_loop(self):
        self._chat_loop(False)

    def test_chat_stream_native_tool_loop(self):
        self._chat_loop(True)

    def test_responses_nonstream_native_tool_loop(self):
        self._responses_loop(False)

    def test_responses_stream_native_tool_loop(self):
        self._responses_loop(True)
