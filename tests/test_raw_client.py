import asyncio
import json
import os
import threading
import unittest
import uuid
from unittest.mock import AsyncMock, Mock, patch

import httpx

from src import raw_client


REAL_HTTPX_CLIENT = httpx.Client


class TrackingByteStream(httpx.SyncByteStream):
    def __init__(self, content: bytes):
        self.content = content
        self.close_calls = 0

    def __iter__(self):
        yield self.content

    def close(self):
        self.close_calls += 1


class OfflineRawResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text
        self.read_calls = 0
        self.close_calls = 0

    def read(self):
        self.read_calls += 1
        return self.text.encode()

    def close(self):
        self.close_calls += 1


class OfflineRawClient:
    def __init__(self, send):
        self._send = send
        self.close_calls = 0
        self.requests = []

    def build_request(self, method, url, **kwargs):
        request = {"method": method, "url": url, **kwargs}
        self.requests.append(request)
        return request

    def send(self, request, stream=False):
        return self._send(request, stream)

    def close(self):
        self.close_calls += 1


async def wait_for_condition(predicate, timeout=1.0):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


def raw_gate_users(session_id):
    with raw_client._RAW_SESSION_GATES_LOCK:
        gate = raw_client._RAW_SESSION_GATES.get(session_id)
        return gate.users if gate is not None else 0


class RawClientBuildTests(unittest.TestCase):
    def test_runtime_prompt_requires_client_confirmation_for_file_changes(self):
        prompt = raw_client.build_runtime_system_prompt(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "download_file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            {"workspace_path": r"C:\workspace"},
        )

        self.assertIn("cannot write into the caller workspace", prompt)
        self.assertIn("matching client tool result", prompt)

    def test_compacted_catalog_prioritizes_namespaced_exec_and_download(self):
        tools = []
        for index in range(100):
            tools.append(
                {
                    "name": f"bulk__tool_{index}",
                    "description": "x" * 600,
                    "input_schema": {
                        "type": "object",
                        "properties": {f"field_{index}": {"type": "string"}},
                    },
                }
            )
        tools.extend(
            [
                {
                    "name": "functions__exec",
                    "description": "Run a caller-side command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
                {
                    "name": "browser__download_file",
                    "description": "Download a caller-side file.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["url", "path"],
                    },
                },
            ]
        )

        with patch.dict(os.environ, {"TRAE_RAW_MAX_TOOL_SCHEMA_CHARS": "4096"}):
            payload, compacted, _ = raw_client._tool_definitions_prompt(tools)

        parsed = json.loads(payload)
        by_name = {item["name"]: item for item in parsed["signatures"]}
        self.assertTrue(compacted)
        self.assertEqual(by_name["functions__exec"]["required"], ["cmd"])
        self.assertEqual(
            by_name["browser__download_file"]["required"], ["url", "path"]
        )

    def test_build_body_has_exact_new_protocol_keys(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the client workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        options = {
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "client_context": {
                "workspace_path": r"D:\code\demo",
                "system_type": "Windows",
                "terminal_context": [{"shell": "PowerShell", "cwd": r"D:\code\demo"}],
            },
        }

        body = raw_client.build_raw_chat_body(
            [{"role": "user", "content": "Inspect the project"}],
            "trae/DeepSeek-V4-Pro",
            options,
            session_id="session-1",
        )

        self.assertEqual(
            set(body),
            {
                "config_name",
                "conversation_id",
                "messages",
                "model_name",
                "session_id",
                "stream",
            },
        )
        self.assertEqual(body["config_name"], "DeepSeek-V4-Pro")
        self.assertEqual(body["model_name"], "DeepSeek-V4-Pro__v2")
        self.assertEqual(body["session_id"], "session-1")
        self.assertEqual(body["conversation_id"], "session-1")
        self.assertTrue(body["stream"])
        # New protocol: tools are not in the body; they are in the system prompt
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertNotIn("parallel_tool_calls", body)

        runtime_prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn(r"D:\\code\\demo", runtime_prompt)
        self.assertIn("Windows", runtime_prompt)
        self.assertIn("read_file", runtime_prompt)
        self.assertIn("external client", runtime_prompt)
        self.assertIn("do not describe that server's Linux filesystem", runtime_prompt)
        self.assertIn("opencode_tool_call", runtime_prompt)

    def test_named_tool_choice_emits_prompt_guidance(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "run_shell", "parameters": {"type": "object"}},
            },
        ]

        body = raw_client.build_raw_chat_body(
            [{"role": "user", "content": "Inspect the project"}],
            "auto",
            {
                "tools": tools,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "run_shell"},
                },
            },
        )

        # New protocol: no tool_choice or tools in the body
        self.assertNotIn("tool_choice", body)
        self.assertNotIn("tools", body)
        prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn("Tool choice requires the client tool named run_shell", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("run_shell", prompt)

    def test_build_body_preserves_tool_result_as_text_history(self):
        messages = [
            {"role": "user", "content": "Inspect the project"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "project docs",
            },
        ]

        body = raw_client.build_raw_chat_body(messages, "auto")

        # New protocol: assistant tool calls are serialized as text in the content
        assistant_message = next(
            message for message in body["messages"] if message["role"] == "assistant"
        )
        self.assertIn("content", assistant_message)
        self.assertNotIn("tool_calls", assistant_message)
        content_text = assistant_message["content"][0]["text"]
        self.assertIn("Client tool calls already issued", content_text)
        self.assertIn("call-1", content_text)
        self.assertIn("read_file", content_text)

        # New protocol: tool results are converted to user messages with text
        tool_texts = [
            block["text"]
            for message in body["messages"]
            if message["role"] == "user"
            for block in message.get("content", [])
            if "Client tool result" in block.get("text", "")
        ]
        self.assertTrue(tool_texts, "Expected a user message with Client tool result")
        self.assertIn("call-1", tool_texts[0])
        self.assertIn("project docs", tool_texts[0])

        # No 'tool' role
        self.assertFalse(
            [m for m in body["messages"] if m["role"] == "tool"],
            "Unexpected tool role message",
        )

    def test_build_body_preserves_renderer_tool_use_and_tool_result(self):
        """Trae's renderer carries tool history as content blocks, not arrays."""

        messages = [
            {"role": "user", "content": "download it"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "toolCallId": "call_abc",
                        "name": "download",
                        "parameters": {
                            "url": "https://example.com/a.zip",
                            "dest": "a.zip",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "toolCallId": "call_abc",
                "name": "download",
                "content": [
                    {
                        "type": "tool_result",
                        "toolCallId": "call_abc",
                        "value": [{"type": "text", "value": "Downloaded 4096 bytes"}],
                        "isError": False,
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

        body = raw_client.build_raw_chat_body(messages, "auto")

        assistant_message = next(
            message for message in body["messages"] if message["role"] == "assistant"
        )
        assistant_text = assistant_message["content"][0]["text"]
        self.assertIn("Client tool calls already issued", assistant_text)
        self.assertIn("call_abc", assistant_text)
        self.assertIn("download", assistant_text)
        self.assertIn("a.zip", assistant_text)

        tool_texts = [
            block["text"]
            for message in body["messages"]
            if message["role"] == "user"
            for block in message.get("content", [])
            if "Client tool result" in block.get("text", "")
        ]
        self.assertTrue(tool_texts, "Expected a renderer tool result in the history")
        self.assertIn("call_abc", tool_texts[0])
        self.assertIn("Downloaded 4096 bytes", tool_texts[0])

    def test_build_body_does_not_synthesize_missing_native_tool_result(self):
        body = raw_client.build_raw_chat_body(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "mcp__node_repl__js:92",
                            "type": "function",
                            "function": {
                                "name": "mcp__node_repl__js",
                                "arguments": '{"code":"1+1"}',
                            },
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            "auto",
        )

        assistant_index = next(
            index
            for index, message in enumerate(body["messages"])
            if message["role"] == "assistant"
        )
        following = body["messages"][assistant_index + 1]
        self.assertEqual(following["role"], "user")
        self.assertNotIn("tool_call_id", following)

    def test_generation_parameters_not_forwarded_in_new_protocol(self):
        """New protocol: generation parameters are not forwarded to the raw body."""
        options = {
            "temperature": 0.25,
            "top_p": 0.85,
            "stop": ["END", "DONE"],
            "presence_penalty": 0.1,
            "frequency_penalty": -0.2,
            "seed": 42,
            "reasoning_effort": "high",
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"},
            "service_tier": "priority",
            "user": "client-user",
            "logprobs": True,
            "top_logprobs": 3,
        }

        body = raw_client.build_raw_chat_body(
            [{"role": "user", "content": "hello"}],
            "auto",
            options,
        )

        # New protocol body has only the 6 keys
        self.assertEqual(
            set(body),
            {"config_name", "conversation_id", "messages", "model_name", "session_id", "stream"},
        )

    def test_model_aliases_use_shared_trae_mapping(self):
        self.assertEqual(
            raw_client.resolve_raw_model("claude-sonnet-4").raw_model_name,
            "glm-5.2",
        )
        self.assertEqual(
            raw_client.resolve_raw_model("gpt-4o").raw_model_name,
            "DeepSeek-V4-Pro__v2",
        )
        self.assertEqual(raw_client.resolve_raw_model("auto").raw_model_name, "glm-5.2")
        self.assertEqual(
            raw_client.resolve_raw_model("glm-5.3").raw_model_name,
            "glm-5.3",
        )

    def test_deepseek_v4_flash_raw_model_mappings_are_pinned(self):
        flash = raw_client.resolve_raw_model("trae/DeepSeek-V4-Flash")
        official = raw_client.resolve_raw_model("DeepSeek-V4-Flash-Official")

        self.assertEqual(flash.config_name, "DeepSeek-V4-Flash")
        self.assertEqual(flash.raw_model_name, "DeepSeek-V4-Flash__v2")
        self.assertEqual(flash.display_name, "DeepSeek-V4-Flash")
        self.assertEqual(official.config_name, "DeepSeek-V4-Flash-Official")
        self.assertEqual(official.raw_model_name, "DeepSeek-V4-Flash-Official")
        self.assertEqual(official.display_name, "DeepSeek-V4-Flash \u6b63\u5f0f\u7248")

    def test_stable_session_has_no_request_id_in_body(self):
        messages = [{"role": "user", "content": "hello"}]
        with patch("src.raw_client.uuid.uuid4", side_effect=[uuid.UUID(int=1), uuid.UUID(int=2)]):
            first = raw_client.build_raw_chat_body(
                messages, "auto", session_id="client-session"
            )
        with patch("src.raw_client.uuid.uuid4", side_effect=[uuid.UUID(int=3), uuid.UUID(int=4)]):
            second = raw_client.build_raw_chat_body(
                messages, "auto", session_id="client-session"
            )

        self.assertEqual(first["session_id"], "client-session")
        self.assertEqual(second["session_id"], "client-session")
        self.assertEqual(first["conversation_id"], "client-session")
        self.assertEqual(second["conversation_id"], "client-session")
        self.assertNotIn("request_id", first)
        self.assertNotIn("request_id", second)

    def test_max_tokens_not_forwarded_in_new_protocol(self):
        """New protocol: max_tokens/clamping is not forwarded to the raw body."""
        body = raw_client.build_raw_chat_body(
            [{"role": "user", "content": "hello"}],
            "deepseek-v4-flash",
            {"max_tokens": 384000},
        )
        self.assertNotIn("max_tokens", body)

    def test_model_overrides_and_explicit_client_context(self):
        with patch.dict(
            os.environ,
            {
                "TRAE_CLIENT_WORKSPACE_PATH": r"C:\Users\demo\project",
                "TRAE_CLIENT_SYSTEM_TYPE": "Windows 11",
            },
            clear=False,
        ):
            body = raw_client.build_raw_chat_body(
                [{"role": "user", "content": "hello"}],
                "new-model",
                {
                    "configName": "config-model",
                    "traeRawModelName": "raw-model-v2",
                    "displayName": "Display Model",
                    "client_context": {
                        "workspace_path": r"C:\Users\demo\project",
                        "system_type": "Windows 11",
                    },
                },
                session_id="fixed",
            )
        self.assertEqual(body["model_name"], "raw-model-v2")
        self.assertEqual(body["config_name"], "config-model")
        self.assertEqual(body["conversation_id"], "fixed")
        self.assertNotIn("client_context", body)
        prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn("Windows 11", prompt)

    def test_auto_discovery_catalog_guides_environment_and_plugin_lookup(self):
        options = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser__open",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]
        }
        context = raw_client.build_client_context(
            options,
            request_headers={
                "User-Agent": "codex_cli_rs/0.148.0",
                "X-Stainless-OS": "Windows",
            },
        )

        self.assertEqual(context["client_name"], "Codex")
        self.assertEqual(context["system_type"], "Windows")
        self.assertEqual(context["terminal_context"][0]["shell"], "PowerShell")
        self.assertTrue(context["tool_discovery"]["tool_search_available"])
        self.assertIn("browser", context["tool_discovery"]["plugin_namespaces"])

        prompt = raw_client.build_runtime_system_prompt(
            options["tools"], context
        )
        self.assertIn("Discover the local environment", prompt)
        self.assertIn("call it proactively", prompt)
        self.assertIn("Never repeat a completed tool call", prompt)
        self.assertEqual(prompt.count('"tool_discovery"'), 1)
        self.assertNotIn("Auto-discovered client tool and plugin catalog", prompt)

    def test_runtime_prompt_deduplicates_tool_definitions_by_name(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "description": "first",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "description": "duplicate",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        prompt = raw_client.build_runtime_system_prompt(tools, {})

        self.assertEqual(prompt.count('"input_schema"'), 1)
        self.assertIn('"description":"first"', prompt)
        self.assertNotIn('"description":"duplicate"', prompt)

    def test_runtime_prompt_keeps_full_schema_within_budget(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        with patch.dict(
            os.environ, {"TRAE_RAW_MAX_TOOL_SCHEMA_CHARS": "48000"}, clear=False
        ):
            prompt = raw_client.build_runtime_system_prompt(tools, {})

        self.assertIn("Available client tool definitions and input schemas", prompt)
        self.assertIn('"description":"Read a file"', prompt)
        self.assertIn('"required":["path"]', prompt)

    def test_runtime_prompt_compacts_oversized_tool_catalog(self):
        tools = []
        for index in range(20):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{index}",
                        "description": "x" * 300,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                f"field_{index}": {
                                    "type": "string",
                                    "description": "y" * 300,
                                }
                            },
                        },
                    },
                }
            )
        with patch.dict(
            os.environ, {"TRAE_RAW_MAX_TOOL_SCHEMA_CHARS": "1024"}, clear=False
        ):
            prompt = raw_client.build_runtime_system_prompt(tools, {})

        self.assertIn("compact input signatures", prompt)
        self.assertIn('"available_tools"', prompt)
        for index in range(20):
            self.assertIn(f'"tool_{index}"', prompt)
        self.assertNotIn('"description":"' + ("x" * 50), prompt)

    def test_raw_history_compaction_preserves_system_and_latest_turns(self):
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "latest question"},
        ]
        with patch.dict(
            os.environ,
            {"TRAE_RAW_MAX_MESSAGES": "2", "TRAE_RAW_MAX_HISTORY_CHARS": "1000"},
            clear=False,
        ):
            raw_messages = raw_client.build_raw_messages(messages)

        texts = [message["content"][0]["text"] for message in raw_messages]
        self.assertEqual(texts, ["policy", "recent answer", "latest question"])
        self.assertNotIn("old question", texts)
        self.assertNotIn("old answer", texts)

    def test_raw_history_strips_multi_escaped_wait_residue_from_assistant_only(self):
        residue = (
            r"\\\<tool\\\_call><0c7dc7cb>wait\\\</tool\\\_call>"
            r"\\\<c4cf82b7>\\\<tool\\\_call><0c7dc7cb>wait"
            r"\\\</tool\\\_call>\\\<c4cf82b7>"
        )
        raw_messages = raw_client.build_raw_messages(
            [
                {"role": "assistant", "content": "answer\n" + residue},
                {"role": "user", "content": "literal\n" + residue},
            ]
        )

        assistant = next(item for item in raw_messages if item["role"] == "assistant")
        user = next(item for item in raw_messages if item["role"] == "user")
        self.assertEqual(assistant["content"][0]["text"], "answer\n")
        self.assertEqual(user["content"][0]["text"], "literal\n" + residue)

    def test_raw_headers_use_traework_reference_defaults(self):
        model = raw_client.RawModel("config", "raw-model", "Display")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            headers = raw_client.build_raw_headers(
                "https://raw.example",
                "token",
                model,
                "request-id",
                session_id="session-1",
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-app-id"], raw_client.RAW_APP_ID)
        self.assertEqual(normalized["x-ide-version-code"], raw_client.RAW_IDE_VERSION_CODE)
        self.assertEqual(normalized["x-ide-version"], raw_client.RAW_IDE_VERSION)
        self.assertEqual(normalized["x-ide-function"], raw_client.RAW_IDE_FUNCTION)
        self.assertEqual(raw_client.RAW_APP_ID, "7b3f9dc2-8a4e-5c6d-2f1b-9e4a3c5b7df0")
        self.assertEqual(raw_client.RAW_IDE_VERSION_CODE, "20260206")
        self.assertEqual(raw_client.RAW_IDE_FUNCTION, "chat")

    def test_raw_header_environment_is_read_at_call_time(self):
        model = raw_client.RawModel("config", "raw-model", "Display")
        with (
            patch.dict(
                os.environ,
                {
                    "TRAE_RAW_APP_ID": "runtime-app-id",
                    "TRAE_RAW_IDE_VERSION_CODE": "20991231",
                    "TRAE_RAW_IDE_VERSION": "9.8.7",
                    "TRAE_RAW_IDE_FUNCTION": "runtime-chat",
                },
                clear=False,
            ),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch(
                "src.trae_client.build_headers",
                return_value={"x-app-id": "old", "x-request-id": "old"},
            ),
        ):
            headers = raw_client.build_raw_headers(
                "https://raw.example",
                "token",
                model,
                "request-from-body",
                {"request_id": "request-fixed"},
                session_id="session-1",
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-app-id"], "runtime-app-id")
        self.assertEqual(normalized["x-ide-version-code"], "20991231")
        self.assertEqual(normalized["x-ide-version"], "9.8.7")
        self.assertEqual(normalized["x-request-id"], "request-from-body")
        self.assertEqual(normalized["x-uid"], "user-1")
        self.assertEqual(normalized["authorization"], "Cloud-IDE-JWT token")
        # New protocol: Extra and x-ide-function are always present
        self.assertIn("extra", normalized)
        self.assertEqual(normalized["x-ide-function"], "runtime-chat")
        extra = json.loads(normalized["extra"])
        self.assertEqual(extra["config_name"], "config")
        self.assertEqual(extra["model_name"], "raw-model")
        self.assertEqual(extra["display_name"], "Display")
        self.assertEqual(extra["config_source"], 1)
        self.assertEqual(extra["session_id"], "session-1")
        self.assertEqual(extra["agent_loop_id"], "session-1")
        self.assertEqual(extra["user_prompt_submit_id"], "session-1")

    def test_raw_extra_header_escapes_non_ascii_model_display_name(self):
        model = raw_client.RawModel(
            "DeepSeek-V4-Flash-Official",
            "DeepSeek-V4-Flash-Official",
            "DeepSeek-V4-Flash 正式版",
        )
        with (
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            headers = raw_client.build_raw_headers(
                "https://raw.example",
                "token",
                model,
                "request-id",
                session_id="session-1",
            )

        extra_header = headers["Extra"]
        extra_header.encode("ascii")
        extra = json.loads(extra_header)
        self.assertEqual(extra["display_name"], "DeepSeek-V4-Flash 正式版")

    def test_raw_header_prefers_bound_billing_identity(self):
        model = raw_client.RawModel("config", "raw-model", "Display")
        with (
            patch.object(raw_client.auth, "get_user_id", return_value="global-user"),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            headers = raw_client.build_raw_headers(
                "https://raw.example",
                "token",
                model,
                "request-id",
                {"_auth_user_id": "billing-account"},
                session_id="session-1",
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-uid"], "billing-account")

    def test_raw_session_id_is_stable_per_model_per_account(self):
        """Session is derived from (account, model) only, not from caller session."""
        model = raw_client.RawModel("glm-5.1", "glm-5__v2", "GLM-5.1")
        opts = {"_billing_id": "account-1"}
        sid1 = raw_client.raw_session_id(model, opts)
        sid2 = raw_client.raw_session_id(model, opts)
        self.assertEqual(sid1, sid2)

        # Different model -> different session
        model2 = raw_client.RawModel("glm-5.3", "glm-5.3", "GLM-5.3")
        sid3 = raw_client.raw_session_id(model2, opts)
        self.assertNotEqual(sid1, sid3)

        # Different account -> different session
        opts2 = {"_billing_id": "account-2"}
        sid4 = raw_client.raw_session_id(model, opts2)
        self.assertNotEqual(sid1, sid4)

        # Public/raw aliases cannot override the account/model lock.
        opts3 = {"rawSessionId": "explicit-override"}
        sid5 = raw_client.raw_session_id(model, opts3)
        self.assertEqual(sid5, raw_client.raw_session_id(model, {}))
        self.assertNotEqual(sid5, "explicit-override")

    def test_raw_session_id_without_session_id_is_stable(self):
        """Caller conversation ids do not change the model-bound raw session."""
        model = raw_client.RawModel("glm-5.1", "glm-5__v2", "GLM-5.1")
        opts = {"_billing_id": "account-1"}
        sid1 = raw_client.raw_session_id(model, opts)
        opts2 = {"_billing_id": "account-1", "session_id": "some-session"}
        sid2 = raw_client.raw_session_id(model, opts2)
        self.assertEqual(sid1, sid2)
        sid3 = raw_client.raw_session_id(model, opts)
        self.assertEqual(sid1, sid3)

    def test_send_dedupes_session_seed_but_not_model(self):
        """Verify send_raw_chat_request derives an upstream session and model lock."""
        model = raw_client.RawModel("glm-5.1", "glm-5__v2", "GLM-5.1")
        opts = {"_billing_id": "account-1"}
        sid1 = raw_client.raw_session_id(model, opts)
        opts2 = {"_billing_id": "account-1", "_auth_user_id": "account-1"}
        sid2 = raw_client.raw_session_id(model, opts2)
        self.assertEqual(sid1, sid2)

    def test_raw_chat_response_close_is_idempotent_and_always_releases(self):
        for failing_owner in ("response", "client"):
            with self.subTest(failing_owner=failing_owner):
                response = Mock()
                client = Mock()
                release = Mock()
                getattr(response if failing_owner == "response" else client, "close").side_effect = (
                    RuntimeError(f"{failing_owner} close failed")
                )
                wrapped = raw_client.RawChatResponse(
                    response=response,
                    client=client,
                    release_session=release,
                )

                with self.assertRaisesRegex(RuntimeError, f"{failing_owner} close failed"):
                    wrapped.close()
                wrapped.close()

                response.close.assert_called_once_with()
                client.close.assert_called_once_with()
                release.assert_called_once_with()


class RawClientRequestTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def raw_options(account="account-1"):
        return {
            "_auth_token": "offline-token",
            "_auth_user_id": account,
            "_billing_id": account,
            "base_url": "https://raw.example",
        }

    def assert_no_raw_session_gates(self):
        with raw_client._RAW_SESSION_GATES_LOCK:
            self.assertEqual(raw_client._RAW_SESSION_GATES, {})

    async def test_model_lookup_uses_bound_billing_identity(self):
        with patch(
            "src.trae_client.resolve_model_config",
            new=AsyncMock(
                return_value={
                    "name": "DeepSeek-V4-Pro-Official",
                    "config_name": "DeepSeek-V4-Pro-Official",
                }
            ),
        ) as lookup:
            resolved = await raw_client.resolve_raw_model_for_request(
                "A future model display label",
                {
                    "_auth_token": "token-b",
                    "_auth_user_id": "account-b",
                },
            )

        self.assertEqual(resolved.raw_model_name, "DeepSeek-V4-Pro-Official")
        lookup.assert_awaited_once_with(
            "A future model display label",
            token_override="token-b",
            user_id_override="account-b",
        )

    async def test_display_model_label_uses_config_name_in_body(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b'event: output\ndata: {"response":"ok"}\n\n',
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        with (
            patch.object(raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)),
            patch.object(raw_client.auth, "get_token", return_value="jwt-token"),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch.dict(os.environ, {"TRAE_RAW_BASE_URL": "https://raw.example"}, clear=False),
            patch("src.trae_client.build_headers", return_value={}),
            patch(
                "src.trae_client.resolve_model_config",
                new=AsyncMock(
                    return_value={
                        "name": "DeepSeek-V4-Pro-Official",
                        "config_name": "DeepSeek-V4-Pro-Official",
                        "raw_model_name": "DeepSeek-V4-Pro-Official",
                        "display_name": "DeepSeek-V4-Pro-Official",
                    }
                ),
            ) as lookup,
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            wrapped = await raw_client.send_raw_chat_request(
                [{"role": "user", "content": "hello"}],
                "A future model display label",
            )

        self.assertEqual(captured["body"]["model_name"], "DeepSeek-V4-Pro-Official")
        self.assertEqual(captured["body"]["config_name"], "DeepSeek-V4-Pro-Official")
        lookup.assert_awaited_once_with(
            "A future model display label", token_override="jwt-token"
        )
        wrapped.close()

    async def test_display_model_label_uses_known_offline_mapping(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b'event: output\ndata: {"response":"ok"}\n\n',
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        with (
            patch.object(raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)),
            patch.object(raw_client.auth, "get_token", return_value="jwt-token"),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch.dict(os.environ, {"TRAE_RAW_BASE_URL": "https://raw.example"}, clear=False),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.trae_client.resolve_model_config", new=AsyncMock()) as lookup,
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            wrapped = await raw_client.send_raw_chat_request(
                [{"role": "user", "content": "hello"}],
                "DeepSeek-V4-Pro 正式版",
            )

        self.assertEqual(captured["body"]["model_name"], "DeepSeek-V4-Pro-Official")
        self.assertEqual(captured["body"]["config_name"], "DeepSeek-V4-Pro-Official")
        lookup.assert_not_awaited()
        wrapped.close()

    async def test_send_raw_chat_request_uses_llm_raw_chat_protocol(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b'event: output\ndata: {"response":"ok"}\n\n',
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        options = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "client_context": {
                "workspace_path": r"E:\work",
                "system_type": "Windows",
                "terminal_context": [{"shell": "pwsh"}],
            },
            "session_id": "session-fixed",
            "_billing_id": "billing-1",
        }
        with (
            patch.object(raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)),
            patch.object(raw_client.auth, "get_token", return_value="jwt-token"),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch.dict(os.environ, {"TRAE_RAW_BASE_URL": "https://raw.example/"}, clear=False),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            wrapped = await raw_client.send_raw_chat_request(
                [{"role": "user", "content": "hello"}], "Kimi-K2.6", options
            )

        request = captured["request"]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url.path, raw_client.RAW_CHAT_ENDPOINT)
        self.assertEqual(request.headers["Authorization"], "Cloud-IDE-JWT jwt-token")
        self.assertTrue(request.headers["X-Request-Id"])
        self.assertEqual(request.headers["X-Uid"], "user-1")
        # New protocol: Extra and X-Ide-Function headers are present
        self.assertIn("Extra", request.headers)
        self.assertEqual(request.headers["X-Ide-Function"], "chat")

        body = json.loads(request.content)
        self.assertEqual(
            set(body),
            {
                "config_name",
                "conversation_id",
                "messages",
                "model_name",
                "session_id",
                "stream",
            },
        )
        self.assertEqual(body["model_name"], "kimi-k2.6__v2")
        self.assertEqual(body["config_name"], "kimi-k2.6")
        self.assertNotIn("request_id", body)
        self.assertNotIn("tools", body)
        self.assertNotIn("function", body)
        # The session id is derived from the account+model hash.
        expected = raw_client.raw_session_id(
            raw_client.RawModel("kimi-k2.6", "kimi-k2.6__v2", "Kimi-K2.6"),
            {"_billing_id": "billing-1", "session_id": "session-fixed"},
        )
        self.assertEqual(body["session_id"], expected)
        self.assertEqual(body["conversation_id"], expected)
        self.assertTrue(body["stream"])
        self.assertNotIn("tools", body)
        self.assertNotIn("client_context", body)
        runtime_prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn("shell", runtime_prompt)
        self.assertIn(r"E:\\work", runtime_prompt)
        self.assertIn("opencode_tool_call", runtime_prompt)

        self.assertEqual(
            list(wrapped.response.iter_lines()),
            ['event: output', 'data: {"response":"ok"}', ""],
        )
        wrapped.close()
        self.assertTrue(wrapped.client.is_closed)

    async def test_non_200_response_is_not_retried_and_closes_client(self):
        request_count = 0
        streams = []
        clients = []

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            stream = TrackingByteStream(b"invalid token")
            streams.append(stream)
            return httpx.Response(401, stream=stream)

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            client = REAL_HTTPX_CLIENT(transport=transport, **kwargs)
            clients.append(client)
            return client

        with (
            patch.object(raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)),
            patch.object(raw_client.auth, "get_token", return_value="bad-token"),
            patch.object(raw_client.auth, "get_user_id", return_value=""),
            patch.dict(os.environ, {"TRAE_RAW_BASE_URL": "https://raw.example"}, clear=False),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            with self.assertRaisesRegex(RuntimeError, "401: invalid token"):
                await raw_client.send_raw_chat_request(
                    [{"role": "user", "content": "hello"}], "auto"
                )

        self.assertEqual(request_count, 1)
        self.assertEqual(len(clients), 1)
        self.assertTrue(clients[0].is_closed)
        self.assertEqual(streams[0].close_calls, 1)
        self.assert_no_raw_session_gates()

    async def test_same_account_model_waits_for_previous_response_close(self):
        options = self.raw_options()
        raw_model = raw_client.resolve_raw_model("glm-5.1")
        session_id = raw_client.raw_session_id(raw_model, options)
        clients = []
        responses = []
        send_count = 0
        state_lock = threading.Lock()

        def send(_request, stream):
            nonlocal send_count
            self.assertTrue(stream)
            response = OfflineRawResponse()
            with state_lock:
                send_count += 1
                responses.append(response)
            return response

        def client_factory(**_kwargs):
            client = OfflineRawClient(send)
            clients.append(client)
            return client

        with (
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            first = await raw_client.send_raw_chat_request(
                [{"role": "user", "content": "first"}], "glm-5.1", options
            )
            second_task = asyncio.create_task(
                raw_client.send_raw_chat_request(
                    [{"role": "user", "content": "second"}], "glm-5.1", options
                )
            )
            await wait_for_condition(lambda: raw_gate_users(session_id) == 2)
            with state_lock:
                self.assertEqual(send_count, 1)
            self.assertEqual(len(clients), 1)

            first.close()
            second = await asyncio.wait_for(second_task, 1.0)
            with state_lock:
                self.assertEqual(send_count, 2)
            second.close()

        self.assertEqual(len(clients), 2)
        self.assertEqual([item.close_calls for item in responses], [1, 1])
        self.assert_no_raw_session_gates()

    async def test_different_model_or_account_sessions_open_concurrently(self):
        async def exercise(model_a, options_a, model_b, options_b):
            entered_sessions = set()
            entered_lock = threading.Lock()
            both_entered = threading.Event()
            allow_return = threading.Event()

            def send(request, stream):
                self.assertTrue(stream)
                session_id = request["json"]["session_id"]
                with entered_lock:
                    entered_sessions.add(session_id)
                    if len(entered_sessions) == 2:
                        both_entered.set()
                allow_return.wait(2.0)
                return OfflineRawResponse()

            def client_factory(**_kwargs):
                return OfflineRawClient(send)

            tasks = []
            results = []
            with (
                patch("src.trae_client.build_headers", return_value={}),
                patch("src.raw_client.httpx.Client", side_effect=client_factory),
            ):
                tasks = [
                    asyncio.create_task(
                        raw_client.send_raw_chat_request(
                            [{"role": "user", "content": "a"}], model_a, options_a
                        )
                    ),
                    asyncio.create_task(
                        raw_client.send_raw_chat_request(
                            [{"role": "user", "content": "b"}], model_b, options_b
                        )
                    ),
                ]
                try:
                    await wait_for_condition(both_entered.is_set, timeout=0.75)
                    with entered_lock:
                        self.assertEqual(len(entered_sessions), 2)
                finally:
                    allow_return.set()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, raw_client.RawChatResponse):
                            result.close()

            for result in results:
                if isinstance(result, BaseException):
                    raise result
            self.assert_no_raw_session_gates()

        await exercise(
            "glm-5.1",
            self.raw_options("account-1"),
            "deepseek-v4-flash",
            self.raw_options("account-1"),
        )
        await exercise(
            "glm-5.1",
            self.raw_options("account-1"),
            "glm-5.1",
            self.raw_options("account-2"),
        )

    async def test_send_exception_closes_client_and_releases_session_gate(self):
        clients = []

        def send(_request, _stream):
            raise RuntimeError("offline send failed")

        def client_factory(**_kwargs):
            client = OfflineRawClient(send)
            clients.append(client)
            return client

        with (
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline send failed"):
                await raw_client.send_raw_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.1",
                    self.raw_options(),
                )

        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0].close_calls, 1)
        self.assert_no_raw_session_gates()

    async def test_cancelling_while_waiting_for_session_gate_unregisters_waiter(self):
        options = self.raw_options()
        session_id = raw_client.raw_session_id(
            raw_client.resolve_raw_model("glm-5.1"), options
        )
        send_count = 0
        state_lock = threading.Lock()

        def send(_request, _stream):
            nonlocal send_count
            with state_lock:
                send_count += 1
            return OfflineRawResponse()

        with (
            patch("src.trae_client.build_headers", return_value={}),
            patch(
                "src.raw_client.httpx.Client",
                side_effect=lambda **_kwargs: OfflineRawClient(send),
            ),
        ):
            first = await raw_client.send_raw_chat_request(
                [{"role": "user", "content": "first"}], "glm-5.1", options
            )
            waiter = asyncio.create_task(
                raw_client.send_raw_chat_request(
                    [{"role": "user", "content": "waiter"}], "glm-5.1", options
                )
            )
            await wait_for_condition(lambda: raw_gate_users(session_id) == 2)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter

            self.assertEqual(raw_gate_users(session_id), 1)
            with state_lock:
                self.assertEqual(send_count, 1)
            first.close()

        self.assert_no_raw_session_gates()

    async def test_cancelling_during_threaded_open_closes_late_response_and_releases(self):
        entered_send = threading.Event()
        allow_return = threading.Event()
        clients = []
        responses = []

        def send(_request, _stream):
            entered_send.set()
            allow_return.wait(2.0)
            response = OfflineRawResponse()
            responses.append(response)
            return response

        def client_factory(**_kwargs):
            client = OfflineRawClient(send)
            clients.append(client)
            return client

        with (
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            request_task = asyncio.create_task(
                raw_client.send_raw_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.1",
                    self.raw_options(),
                )
            )
            await wait_for_condition(entered_send.is_set)
            request_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request_task
            allow_return.set()
            await wait_for_condition(
                lambda: bool(responses)
                and responses[0].close_calls == 1
                and clients[0].close_calls == 1
                and not raw_client._RAW_SESSION_GATES
            )

        self.assertEqual(len(clients), 1)
        self.assertEqual(len(responses), 1)
        self.assert_no_raw_session_gates()


if __name__ == "__main__":
    unittest.main()
