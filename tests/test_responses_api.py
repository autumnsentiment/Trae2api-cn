import asyncio
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch


FAKE_CMD = str(Path(__file__).resolve().parent / "fake" / "fake_cli.cmd")

os.environ.setdefault("TRAE_AUTH_SOURCE", "cli")
os.environ.setdefault("UPSTREAM_MODE", "cli")
os.environ.setdefault("TRAE_CLI_COMMAND", FAKE_CMD)

from fastapi.testclient import TestClient

from src import main, responses_api
from src.cli_client import CliEvent


AUTH_HEADERS = {"Authorization": "Bearer responses-key"}


def _responses_tool() -> dict:
    return {
        "type": "function",
        "name": "read_file",
        "description": "Read a UTF-8 file from the caller workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _expected_chat_tool() -> dict:
    tool = _responses_tool()
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": tool["strict"],
        },
    }


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") in ("text", "input_text")
    )


def _response_output_text(body: dict) -> str:
    parts = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _parse_responses_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    event_name = None
    data_lines = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return
        payload_text = "\n".join(data_lines)
        event_name_local = event_name
        event_name = None
        data_lines = []
        if payload_text == "[DONE]":
            return
        payload = json.loads(payload_text)
        name = event_name_local or payload.get("type")
        events.append((name, payload))

    for line in raw.splitlines():
        if not line:
            flush()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    flush()
    return events


class ResponsesApiCompatibilityTests(unittest.TestCase):
    def setUp(self):
        responses_api._RESPONSE_SESSIONS.clear()
        self.client = TestClient(main.app)
        self.api_keys = patch("src.main.API_KEYS", ["responses-key"])
        self.mode = patch("src.main.UPSTREAM_MODE", "cli")
        self.api_keys.start()
        self.mode.start()

    def tearDown(self):
        self.mode.stop()
        self.api_keys.stop()
        self.client.close()
        responses_api._RESPONSE_SESSIONS.clear()

    def test_previous_response_id_restores_nonstream_conversation(self):
        captured = []
        captured_options = []

        async def fake_stream(messages, model, options=None):
            captured.append(messages)
            captured_options.append(options or {})
            text = "first answer" if len(captured) == 1 else "second answer"
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": [{"type": "text", "text": text}]},
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={"model": "auto", "input": "first question"},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "follow-up question",
                    "previous_response_id": first.json()["id"],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            [message["role"] for message in captured[1]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(captured[1][0]["content"], "first question")
        self.assertEqual(captured[1][1]["content"], "first answer")
        self.assertEqual(captured[1][2]["content"], "follow-up question")
        self.assertEqual(
            second.json()["previous_response_id"], first.json()["id"]
        )
        self.assertEqual(captured_options[0]["session_id"], first.json()["id"])
        self.assertEqual(
            captured_options[1]["session_id"], first.json()["id"]
        )

    def test_previous_response_id_deduplicates_replayed_full_history(self):
        captured = []

        async def fake_stream(messages, model, options=None):
            captured.append(messages)
            text = "first answer" if len(captured) == 1 else "second answer"
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": [{"type": "text", "text": text}]},
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={"model": "auto", "input": "first question"},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "previous_response_id": first.json()["id"],
                    "input": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first answer"},
                        {"role": "user", "content": "follow-up question"},
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            [message["role"] for message in captured[1]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(
            [message["content"] for message in captured[1]],
            ["first question", "first answer", "follow-up question"],
        )

    def test_previous_response_id_does_not_replay_request_instructions(self):
        captured = []

        async def fake_stream(messages, model, options=None):
            captured.append(messages)
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [{"type": "text", "text": "answer"}]
                    },
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "instructions": "old request instructions",
                    "input": "first question",
                },
                headers=AUTH_HEADERS,
            )
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "instructions": "new request instructions",
                    "input": "follow-up question",
                    "previous_response_id": first.json()["id"],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            [message["role"] for message in captured[1]],
            ["developer", "user", "assistant", "user"],
        )
        contents = [message.get("content") for message in captured[1]]
        self.assertEqual(contents[0], "new request instructions")
        self.assertNotIn("old request instructions", contents)

    def test_previous_response_id_restores_tool_call_for_output_only_input(self):
        captured = []

        async def fake_stream(messages, model, options=None):
            captured.append((messages, options))
            if len(captured) == 1:
                yield CliEvent(
                    type="json",
                    data={
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "tool_calls": [
                                {
                                    "id": "call_cached_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                )
                return
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [
                            {"type": "text", "text": "used cached tool result"}
                        ]
                    },
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "Read README.md",
                    "tools": [_responses_tool()],
                },
                headers=AUTH_HEADERS,
            )
            self.assertEqual(first.status_code, 200, first.text)
            call = next(
                item
                for item in first.json()["output"]
                if item["type"] == "function_call"
            )
            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "previous_response_id": first.json()["id"],
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": "cached README contents",
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(second.status_code, 200, second.text)
        messages, options = captured[1]
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(
            messages[1]["tool_calls"][0]["id"], "call_cached_read"
        )
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"], "read_file"
        )
        self.assertEqual(messages[2]["tool_call_id"], "call_cached_read")
        self.assertEqual(messages[2]["name"], "read_file")
        self.assertEqual(messages[2]["content"], "cached README contents")
        self.assertNotIn("tools", options)
        self.assertEqual(options["_inherited_tools"], [_expected_chat_tool()])
        self.assertTrue(options["_tool_protocol_requested"])

    def test_multi_tool_continuation_keeps_one_upstream_session(self):
        """A Responses tool chain must not mint a raw Trae session per turn."""
        captured = []
        list_tool = {
            "type": "function",
            "name": "list_files",
            "description": "List files from the caller workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        }

        async def fake_stream(messages, model, options=None):
            captured.append((messages, dict(options or {})))
            turn = len(captured)
            if turn == 1:
                yield CliEvent(
                    type="json",
                    data={
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                )
                return
            if turn == 2:
                yield CliEvent(
                    type="json",
                    data={
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "tool_calls": [
                                {
                                    "id": "call_list",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": '{"path":"src"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                )
                return
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "README and source files inspected.",
                            }
                        ]
                    },
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "Inspect the README and source files.",
                    "tools": [_responses_tool(), list_tool],
                },
                headers=AUTH_HEADERS,
            )
            self.assertEqual(first.status_code, 200, first.text)
            first_call = next(
                item
                for item in first.json()["output"]
                if item["type"] == "function_call"
            )

            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "previous_response_id": first.json()["id"],
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": first_call["call_id"],
                            "output": "README contents",
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )
            self.assertEqual(second.status_code, 200, second.text)
            second_call = next(
                item
                for item in second.json()["output"]
                if item["type"] == "function_call"
            )

            third = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "previous_response_id": second.json()["id"],
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": second_call["call_id"],
                            "output": "main.py\nraw_client.py",
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(third.status_code, 200, third.text)
        self.assertEqual(
            [options["session_id"] for _, options in captured],
            [first.json()["id"]] * 3,
        )
        self.assertNotEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(
            [message["role"] for message in captured[2][0]],
            ["user", "assistant", "tool", "assistant", "tool"],
        )
        self.assertEqual(
            _response_output_text(third.json()),
            "README and source files inspected.",
        )

    def test_previous_response_id_restores_completed_stream(self):
        captured = []

        async def fake_stream(messages, model, options=None):
            captured.append(messages)
            text = "streamed first" if len(captured) == 1 else "continued"
            yield CliEvent(
                type="json",
                data={"message": {"content": [{"type": "text", "text": text}]}},
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={"model": "auto", "input": "stream this", "stream": True},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(first.status_code, 200, first.text)
            events = _parse_responses_sse(first.text)
            response_id = events[-1][1]["response"]["id"]
            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "continue it",
                    "previous_response_id": response_id,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            [message["role"] for message in captured[1]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(captured[1][1]["content"], "streamed first")

    def test_unknown_previous_response_id_returns_parameter_error(self):
        response = self.client.post(
            "/v1/responses",
            json={
                "model": "auto",
                "input": "continue",
                "previous_response_id": "resp_missing",
            },
            headers=AUTH_HEADERS,
        )

        self.assertEqual(response.status_code, 400, response.text)
        error = response.json()["error"]
        self.assertEqual(error["param"], "previous_response_id")
        self.assertIn("not found or has expired", error["message"])

    def test_store_false_response_cannot_be_used_as_previous_response(self):
        async def fake_stream(messages, model, options=None):
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": [{"type": "text", "text": "answer"}]},
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            first = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "do not retain this turn",
                    "store": False,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(first.status_code, 200, first.text)
        response_id = first.json()["id"]
        self.assertIsNone(responses_api._RESPONSE_SESSIONS.get(response_id))

        second = self.client.post(
            "/v1/responses",
            json={
                "model": "auto",
                "input": "continue",
                "previous_response_id": response_id,
            },
            headers=AUTH_HEADERS,
        )
        self.assertEqual(second.status_code, 400, second.text)
        self.assertEqual(second.json()["error"]["param"], "previous_response_id")

    def test_unknown_tool_call_output_is_downgraded_to_inert_history(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["messages"] = messages
            captured["options"] = options or {}
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [{"type": "text", "text": "continued"}]
                    },
                    "finish_reason": "stop",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_missing",
                            "output": "result from the caller",
                        },
                        {"type": "input_text", "text": "continue"},
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(captured["options"]["_tool_protocol_requested"])
        self.assertEqual([item["role"] for item in captured["messages"]], ["user", "user"])
        stale = captured["messages"][0]
        self.assertNotIn("tool_calls", stale)
        self.assertIn("Untrusted client tool result", stale["content"])
        self.assertIn("call_missing", stale["content"])
        self.assertIn("result from the caller", stale["content"])
        self.assertEqual(captured["messages"][1]["content"], "continue")

    def test_replayed_assistant_tool_call_recovers_binding_for_output(self):
        messages, options, _ = responses_api.normalize_request(
            {
                "model": "auto",
                "tools": [_responses_tool()],
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_replayed",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_replayed",
                        "output": "README contents",
                    },
                ],
            }
        )

        self.assertEqual([item["role"] for item in messages], ["assistant", "tool"])
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_replayed")
        self.assertEqual(messages[1]["tool_call_id"], "call_replayed")
        self.assertEqual(messages[1]["name"], "read_file")
        self.assertEqual(messages[1]["content"], "README contents")
        self.assertTrue(options["_tool_protocol_requested"])

    def test_unknown_replayed_call_never_synthesizes_executable_tool_call(self):
        messages, options, _ = responses_api.normalize_request(
            {
                "model": "auto",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_unbound",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"whoami"}',
                                },
                            }
                        ],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_unbound",
                        "output": "Windows user",
                    },
                ],
            }
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_unbound")
        self.assertEqual(messages[1]["role"], "user")
        self.assertNotIn("tool_call_id", messages[1])
        self.assertIn("history only", messages[1]["content"])
        self.assertTrue(options["_tool_protocol_requested"])

    def test_normalize_accepts_chat_nested_tool_shape(self):
        chat_tool = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 file from the caller workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        messages, options, context = responses_api.normalize_request(
            {
                "model": "auto",
                "input": "Use read_file.",
                "tools": [chat_tool],
                "tool_choice": {"type": "function", "name": "read_file"},
            }
        )
        self.assertEqual(options["tools"][0]["type"], "function")
        self.assertEqual(options["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(context.bindings["read_file"].response_type, "function")
        self.assertEqual(context.bindings["read_file"].name, "read_file")

    def test_inherited_tool_names_are_reserved_for_new_tools(self):
        _, _, context = responses_api.normalize_request(
            {
                "model": "auto",
                "input": "call the namespaced tool",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "x",
                        "tools": [
                            {
                                "type": "function",
                                "name": "y",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                ],
            }
        )
        first = responses_api.completion_to_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_old_namespace",
                                    "type": "function",
                                    "function": {
                                        "name": "x__y",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            context,
        )
        messages, options, _ = responses_api.normalize_request(
            {
                "model": "auto",
                "previous_response_id": first["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_old_namespace",
                        "output": "done",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "x__y",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        )

        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"], "x__y"
        )
        self.assertEqual(messages[2]["name"], "x__y")
        self.assertEqual(
            options["tools"][0]["function"]["name"], "x__y_2"
        )

    def test_nonstream_text_response_uses_responses_shape(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["messages"] = messages
            captured["model"] = model
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [{"type": "text", "text": "hello from responses"}]
                    },
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 3,
                        "total_tokens": 7,
                    },
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={"model": "auto", "input": "hello", "stream": False},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertTrue(str(body["id"]).startswith("resp_"))
        self.assertNotIn("choices", body)
        self.assertEqual(_response_output_text(body), "hello from responses")
        self.assertEqual(body["usage"]["input_tokens"], 4)
        self.assertEqual(body["usage"]["output_tokens"], 3)
        self.assertEqual(body["usage"]["total_tokens"], 7)
        self.assertEqual(captured["model"], "auto")
        self.assertEqual([message["role"] for message in captured["messages"]], ["user"])
        self.assertEqual(_content_text(captured["messages"][0]["content"]), "hello")

    def test_stream_text_emits_typed_responses_events(self):
        async def fake_stream(messages, model, options=None):
            yield CliEvent(
                type="json",
                data={
                    "message": {"content": [{"type": "text", "text": "hello"}]}
                },
            )
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [{"type": "text", "text": "hello world"}]
                    },
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 2,
                        "total_tokens": 4,
                    },
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={"model": "auto", "input": "hello", "stream": True},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        events = _parse_responses_sse(response.text)
        event_types = [name for name, _ in events]
        self.assertEqual(event_types[0], "response.created")
        self.assertIn("response.output_item.added", event_types)
        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("response.output_text.done", event_types)
        self.assertIn("response.output_item.done", event_types)
        self.assertEqual(event_types[-1], "response.completed")
        for name, payload in events:
            self.assertEqual(payload.get("type"), name)
        text = "".join(
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        )
        self.assertEqual(text, "hello world")
        created = events[0][1]["response"]
        completed = events[-1][1]["response"]
        self.assertEqual(created["id"], completed["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(_response_output_text(completed), "hello world")

    def test_stream_preserves_sse_keepalive_without_creating_output(self):
        async def inner():
            yield ": relay-keepalive\n\n"
            yield (
                'data: {"choices":[{"delta":{"content":"hello"},'
                '"finish_reason":null}]}\n\n'
            )
            yield 'data: [DONE]\n\n'

        _, _, context = responses_api.normalize_request(
            {"model": "auto", "input": "hello", "stream": True}
        )
        async def collect():
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        raw = "".join(asyncio.run(collect()))
        self.assertIn(": relay-keepalive\n\n", raw)
        events = _parse_responses_sse(raw)
        text = "".join(
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        )
        self.assertEqual(text, "hello")
        self.assertEqual(
            sum(name == "response.completed" for name, _ in events), 1
        )

    def test_stream_hides_history_marker_split_over_chat_sse_chunks(self):
        async def inner():
            for content, finish_reason in (
                ("Previous", None),
                (' client tool request(s):\n[{"id":"old","name":"shell","input":"{}"}]', None),
                ("done", "stop"),
            ):
                yield "data: " + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": content},
                                "finish_reason": finish_reason,
                            }
                        ]
                    }
                ) + "\n\n"
            yield "data: [DONE]\n\n"

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(inner(), context)
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        text = "".join(
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        )
        self.assertEqual(text, "done")

    def test_normalize_request_clamps_max_output_tokens(self):
        _, options, _ = responses_api.normalize_request(
            {
                "model": "deepseek-v4-flash",
                "input": "hello",
                "max_output_tokens": 384000,
            }
        )
        self.assertEqual(options["max_tokens"], 64000)

    def test_stream_function_call_emits_argument_delta_and_done_events(self):
        captured = {}
        arguments = '{"path":"README.md"}'

        async def fake_stream(messages, model, options=None):
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "tool_calls": [
                            {
                                "id": "call_fake_read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "Read README.md",
                    "tools": [_responses_tool()],
                    "tool_choice": "required",
                    "stream": True,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["options"]["tools"], [_expected_chat_tool()])
        events = _parse_responses_sse(response.text)
        event_types = [name for name, _ in events]
        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(event_types[-1], "response.completed")
        argument_deltas = "".join(
            payload.get("delta", "")
            for name, payload in events
            if name == "response.function_call_arguments.delta"
        )
        self.assertEqual(argument_deltas, arguments)
        done = next(
            payload
            for name, payload in events
            if name == "response.function_call_arguments.done"
        )
        self.assertEqual(done["arguments"], arguments)
        self.assertEqual(done["name"], "read_file")
        completed = events[-1][1]["response"]
        call = next(item for item in completed["output"] if item["type"] == "function_call")
        self.assertEqual(done["item_id"], call["id"])
        self.assertTrue(call["id"])
        self.assertEqual(call["call_id"], "call_fake_read")
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(call["arguments"], arguments)

    def test_function_call_output_continues_via_chat_tool_history(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["messages"] = messages
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Used README contents from the Windows client.",
                            }
                        ]
                    },
                    "usage": {
                        "input_tokens": 8,
                        "output_tokens": 4,
                        "total_tokens": 12,
                    },
                },
            )

        responses_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "FAKE_TOOL_CALL_REQUEST: inspect README.md",
                    }
                ],
            },
            {
                "type": "function_call",
                "id": "fc_fake_read",
                "call_id": "call_fake_read",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_fake_read",
                "output": "README contents from the Windows client",
            },
        ]

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": responses_input,
                    "tools": [_responses_tool()],
                    "stream": False,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        messages = captured["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "tool"])
        assistant = messages[1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_fake_read")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            '{"path":"README.md"}',
        )
        tool_result = messages[2]
        self.assertEqual(tool_result["tool_call_id"], "call_fake_read")
        self.assertEqual(tool_result["name"], "read_file")
        self.assertEqual(tool_result["content"], "README contents from the Windows client")
        self.assertEqual(captured["options"]["tools"], [_expected_chat_tool()])
        self.assertEqual(
            _response_output_text(response.json()),
            "Used README contents from the Windows client.",
        )

    def test_stream_custom_tool_call_uses_custom_responses_events(self):
        captured = {}
        patch_text = "*** Begin Patch\n*** End Patch"

        async def fake_stream(messages, model, options=None):
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "tool_calls": [
                            {
                                "id": "call_apply_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": json.dumps({"input": patch_text}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            )

        custom_tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch in the caller workspace.",
            "format": {"type": "text"},
        }
        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "Update the file",
                    "tools": [custom_tool],
                    "stream": True,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        chat_function = captured["options"]["tools"][0]["function"]
        self.assertEqual(chat_function["name"], "apply_patch")
        self.assertEqual(
            chat_function["parameters"]["required"], ["input"]
        )
        events = _parse_responses_sse(response.text)
        event_types = [name for name, _ in events]
        self.assertIn("response.custom_tool_call_input.delta", event_types)
        self.assertIn("response.custom_tool_call_input.done", event_types)
        self.assertNotIn("response.function_call_arguments.done", event_types)
        self.assertNotIn("data: [DONE]", response.text)
        completed = events[-1][1]["response"]
        item = completed["output"][0]
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["call_id"], "call_apply_patch")
        self.assertEqual(item["name"], "apply_patch")
        self.assertEqual(item["input"], patch_text)

    def test_custom_tool_output_continues_via_chat_tool_history(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["messages"] = messages
            yield CliEvent(
                type="json",
                data={"message": {"content": [{"type": "text", "text": "patched"}]}},
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": [
                        {"role": "user", "content": "Patch the file"},
                        {
                            "type": "custom_tool_call",
                            "id": "ctc_apply_patch",
                            "call_id": "call_apply_patch",
                            "name": "apply_patch",
                            "input": "*** Begin Patch\n*** End Patch",
                        },
                        {
                            "type": "custom_tool_call_output",
                            "call_id": "call_apply_patch",
                            "output": "Done!",
                        },
                    ],
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "Apply a patch.",
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        assistant = captured["messages"][1]
        arguments = json.loads(
            assistant["tool_calls"][0]["function"]["arguments"]
        )
        self.assertEqual(arguments["input"], "*** Begin Patch\n*** End Patch")
        self.assertEqual(captured["messages"][2]["role"], "tool")
        self.assertEqual(
            captured["messages"][2]["tool_call_id"], "call_apply_patch"
        )
        self.assertEqual(captured["messages"][2]["content"], "Done!")

    def test_namespace_function_call_preserves_namespace_and_continuation(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["messages"] = messages
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "content": [],
                        "tool_calls": [
                            {
                                "id": "call_send",
                                "type": "function",
                                "function": {
                                    "name": "collaboration__send_message",
                                    "arguments": '{"target":"/root","message":"done"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            )

        namespace_tool = {
            "type": "namespace",
            "name": "collaboration",
            "description": "Coordinate agents.",
            "tools": [
                {
                    "type": "function",
                    "name": "send_message",
                    "description": "Send a message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["target", "message"],
                    },
                }
            ],
        }
        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "Notify the parent",
                    "tools": [namespace_tool],
                    "stream": True,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            captured["options"]["tools"][0]["function"]["name"],
            "collaboration__send_message",
        )
        discovery = captured["options"]["client_context"]["tool_discovery"]
        self.assertIn("collaboration", discovery["plugin_namespaces"])
        events = _parse_responses_sse(response.text)
        item = events[-1][1]["response"]["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["name"], "send_message")
        self.assertEqual(item["namespace"], "collaboration")

        second_capture = {}

        async def second_stream(messages, model, options=None):
            second_capture["messages"] = messages
            yield CliEvent(
                type="json",
                data={"message": {"content": [{"type": "text", "text": "continued"}]}},
            )

        with patch("src.main.cli_client.stream_cli_chat", new=second_stream):
            second = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": [
                        {"role": "user", "content": "Notify the parent"},
                        item,
                        {
                            "type": "function_call_output",
                            "call_id": "call_send",
                            "output": "message delivered",
                        },
                    ],
                    "tools": [namespace_tool],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(second.status_code, 200, second.text)
        messages = second_capture["messages"]
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"],
            "collaboration__send_message",
        )
        self.assertEqual(messages[2]["name"], "collaboration__send_message")

    def test_client_metadata_drives_automatic_environment_discovery(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={"message": {"content": [{"type": "text", "text": "ready"}]}},
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": "Inspect the local project",
                    "tools": [_responses_tool()],
                    "client_metadata": {
                        "system_type": "Windows",
                        "cwd": r"D:\\work\\client-project",
                        "terminal_context": [{"shell": "PowerShell"}],
                    },
                },
                headers={**AUTH_HEADERS, "User-Agent": "codex_cli_rs/1.0"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        context = captured["options"]["client_context"]
        self.assertEqual(context["client_name"], "Codex")
        self.assertEqual(context["system_type"], "Windows")
        self.assertEqual(context["workspace_path"], r"D:\\work\\client-project")
        self.assertEqual(context["terminal_context"][0]["shell"], "PowerShell")

    def test_replayed_additional_tools_are_available_to_the_upstream(self):
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={"message": {"content": [{"type": "text", "text": "ready"}]}},
            )

        with patch("src.main.cli_client.stream_cli_chat", new=fake_stream):
            response = self.client.post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": [
                        {"role": "user", "content": "Use the loaded tool"},
                        {
                            "type": "additional_tools",
                            "tools": [_responses_tool()],
                        },
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["options"]["tools"], [_expected_chat_tool()])

    def test_closing_responses_stream_closes_inner_chat_iterator(self):
        async def scenario():
            state = {"closed": False}

            async def inner():
                try:
                    yield (
                        'data: {"choices":[{"delta":{"content":"hello"},'
                        '"finish_reason":null}]}\n\n'
                    )
                    await asyncio.sleep(60)
                finally:
                    state["closed"] = True

            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            stream = responses_api.translate_chat_stream(inner(), context)
            await anext(stream)
            await anext(stream)
            await anext(stream)
            await stream.aclose()
            return state["closed"]

        self.assertTrue(asyncio.run(scenario()))

    def test_done_sentinel_stops_before_replayed_chat_chunks(self):
        async def inner():
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":"once"},'
                '"finish_reason":null}]}\n\n'
            )
            yield "data: [DONE]\n\n"
            raise AssertionError("read past DONE")

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        deltas = [
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        ]
        self.assertEqual(deltas, ["once"])
        self.assertEqual(
            sum(name == "response.completed" for name, _ in events), 1
        )

    def test_unterminated_chat_stream_fails_and_is_not_cached(self):
        async def inner():
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
                '"finish_reason":null}]}\n\n'
            )

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            chunks = [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]
            return context.response_id, chunks

        response_id, chunks = asyncio.run(run())
        events = _parse_responses_sse("".join(chunks))
        self.assertEqual(events[-1][0], "response.failed")
        self.assertEqual(
            events[-1][1]["response"]["error"]["code"],
            "upstream_stream_incomplete",
        )
        self.assertIsNone(responses_api._RESPONSE_SESSIONS.get(response_id))

    def test_finish_reason_snapshot_does_not_truncate_later_chat_chunks(self):
        closed = False

        async def inner():
            nonlocal closed
            try:
                yield (
                    'data: {"choices":[{"index":0,"delta":{"content":"once"},'
                    '"finish_reason":null}]}\n\n'
                )
                yield (
                    'data: {"choices":[{"index":0,"delta":{},'
                    '"finish_reason":"stop"}]}\n\n'
                )
                yield (
                    'data: {"choices":[{"index":0,"delta":{"content":" later"},'
                    '"finish_reason":null}]}\n\n'
                )
                yield 'data: [DONE]\n\n'
            finally:
                closed = True

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        deltas = [
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        ]
        self.assertEqual(deltas, ["once", " later"])
        self.assertEqual(
            sum(name == "response.completed" for name, _ in events), 1
        )
        self.assertTrue(closed)

    def test_stream_reconciles_cumulative_text_snapshots_and_replays(self):
        async def inner():
            for content in ("hello", "hello world", "hello world"):
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"content": content}}]}
                ) + "\n\n"
            yield "data: [DONE]\n\n"

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        deltas = [
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        ]
        self.assertEqual("".join(deltas), "hello world")

    def test_stream_keeps_short_incremental_text_repetition(self):
        async def inner():
            for content in ("ha", "ha"):
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"content": content}}]}
                ) + "\n\n"
            yield "data: [DONE]\n\n"

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        self.assertEqual(
            "".join(
                payload.get("delta", "")
                for name, payload in events
                if name == "response.output_text.delta"
            ),
            "haha",
        )

    def test_stream_reconciles_cumulative_tool_arguments(self):
        async def inner():
            for arguments in ('{"path":"R', '{"path":"README.md"}', '{"path":"README.md"}'):
                yield "data: " + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_read",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": arguments,
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ) + "\n\n"
            yield "data: [DONE]\n\n"

        async def run():
            _, _, context = responses_api.normalize_request(
                {
                    "model": "auto",
                    "input": "read",
                    "stream": True,
                    "tools": [_responses_tool()],
                }
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        done = next(
            payload
            for name, payload in events
            if name == "response.function_call_arguments.done"
        )
        self.assertEqual(done["arguments"], '{"path":"README.md"}')
        self.assertEqual(json.loads(done["arguments"]), {"path": "README.md"})

    def test_finish_reason_without_done_fails_and_is_not_cached(self):
        async def inner():
            yield (
                'data: {"choices":[{"delta":{"content":"partial"},'
                '"finish_reason":"stop"}]}\n\n'
            )

        async def run():
            _, _, context = responses_api.normalize_request(
                {"model": "auto", "input": "hello", "stream": True}
            )
            chunks = [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]
            return context.response_id, chunks

        response_id, chunks = asyncio.run(run())
        events = _parse_responses_sse("".join(chunks))
        self.assertEqual(events[-1][0], "response.failed")
        self.assertNotIn("response.completed", [name for name, _ in events])
        self.assertIsNone(responses_api._RESPONSE_SESSIONS.get(response_id))

    def test_text_before_tool_call_is_not_prematurely_final_answer(self):
        async def inner():
            yield (
                'data: {"choices":[{"delta":{"content":"I will inspect."}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"call_read","function":{"name":"read_file",'
                '"arguments":"{}"}}]}}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        async def run():
            _, _, context = responses_api.normalize_request(
                {
                    "model": "auto",
                    "input": "read",
                    "stream": True,
                    "tools": [_responses_tool()],
                }
            )
            return [
                event
                async for event in responses_api.translate_chat_stream(
                    inner(), context
                )
            ]

        events = _parse_responses_sse("".join(asyncio.run(run())))
        added = next(
            payload
            for name, payload in events
            if name == "response.output_item.added"
            and payload["item"]["type"] == "message"
        )
        self.assertNotIn("phase", added["item"])
        done = next(
            payload
            for name, payload in events
            if name == "response.output_item.done"
            and payload["item"]["type"] == "message"
        )
        self.assertEqual(done["item"]["phase"], "commentary")

    def test_raw_empty_retry_emits_one_responses_lifecycle(self):
        class LineSource:
            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self):
                return iter(self.lines)

            def __iter__(self):
                return iter(self.lines)

        class FakeRawResponse:
            def __init__(self, lines):
                self.response = LineSource(lines)
                self.closed = False

            def close(self):
                self.closed = True

        first = FakeRawResponse(
            ['data: {"finish_reason":"stop"}', "data: [DONE]"]
        )
        second = FakeRawResponse(
            [
                'data: {"response":"recovered once"}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        send_raw = AsyncMock(side_effect=[first, second])

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
        ):
            response = self.client.post(
                "/v1/responses",
                json={"model": "auto", "input": "hello", "stream": True},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        events = _parse_responses_sse(response.text)
        event_types = [name for name, _ in events]
        text = "".join(
            payload.get("delta", "")
            for name, payload in events
            if name == "response.output_text.delta"
        )
        self.assertEqual(text, "recovered once")
        self.assertNotIn("trae upstream returned an empty response", text)
        self.assertEqual(event_types.count("response.created"), 1)
        self.assertEqual(event_types.count("response.in_progress"), 1)
        self.assertEqual(event_types.count("response.completed"), 1)
        self.assertNotIn("response.failed", event_types)
        self.assertEqual(send_raw.await_count, 2)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)


class ResponsesSessionCacheTests(unittest.TestCase):
    def _session(self, text: str) -> responses_api._ResponseSession:
        return responses_api._ResponseSession(
            messages=[{"role": "user", "content": text}],
            bindings={},
            call_bindings={},
        )

    def test_cache_uses_ttl_and_lru_entry_bound(self):
        now = [0.0]
        cache = responses_api._ResponseSessionCache(
            ttl_seconds=10,
            max_entries=2,
            clock=lambda: now[0],
        )
        cache.put("one", self._session("one"))
        cache.put("two", self._session("two"))
        self.assertEqual(len(cache), 2)
        self.assertIsNotNone(cache.get("one"))
        cache.put("three", self._session("three"))
        self.assertIsNone(cache.get("two"))
        self.assertIsNotNone(cache.get("one"))
        now[0] = 10.001
        self.assertIsNone(cache.get("one"))
        self.assertIsNone(cache.get("three"))
        self.assertEqual(len(cache), 0)

    def test_cache_returns_deep_copies(self):
        cache = responses_api._ResponseSessionCache(
            ttl_seconds=60,
            max_entries=4,
        )
        cache.put("copy", self._session("original"))
        first = cache.get("copy")
        self.assertIsNotNone(first)
        first.messages[0]["content"] = "mutated"
        second = cache.get("copy")
        self.assertEqual(second.messages[0]["content"], "original")

    def test_cache_rejects_session_over_message_count(self):
        cache = responses_api._ResponseSessionCache(
            ttl_seconds=60,
            max_entries=4,
            max_messages=2,
        )
        session = responses_api._ResponseSession(
            messages=[
                {"role": "user", "content": f"message-{index}"}
                for index in range(5)
            ],
            bindings={},
            call_bindings={},
        )
        cache.put("bounded", session)
        restored = cache.get("bounded")
        self.assertEqual(restored.messages, [])
        self.assertIn("exceeded", restored.unavailable_reason)

    def test_cache_marks_single_oversized_session_unavailable(self):
        cache = responses_api._ResponseSessionCache(
            ttl_seconds=60,
            max_entries=4,
            max_session_bytes=1024,
        )
        cache.put("oversized", self._session("x" * 4096))
        restored = cache.get("oversized")
        self.assertEqual(restored.messages, [])
        self.assertIn("exceeded", restored.unavailable_reason)

    def test_cache_is_safe_under_concurrent_put_and_get(self):
        cache = responses_api._ResponseSessionCache(
            ttl_seconds=60,
            max_entries=16,
        )

        def worker(worker_id: int) -> None:
            for index in range(100):
                response_id = f"resp_{worker_id}_{index}"
                cache.put(response_id, self._session(response_id))
                cache.get(response_id)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(worker, range(8)))
        self.assertLessEqual(len(cache), 16)


if __name__ == "__main__":
    unittest.main()
