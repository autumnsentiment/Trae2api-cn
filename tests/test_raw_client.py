import json
import os
import unittest
import uuid
from unittest.mock import AsyncMock, patch

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


class RawClientBuildTests(unittest.TestCase):
    def test_build_body_forwards_native_tools_on_first_turn(self):
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
                "messages",
                "model",
                "function",
                "request_id",
                "session_id",
                "stream",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
            },
        )
        self.assertEqual(body["model"], "DeepSeek-V4-Pro")
        self.assertEqual(body["function"], "inline_chat")
        uuid.UUID(body["request_id"])
        self.assertNotEqual(body["request_id"], "session-1")
        self.assertEqual(body["session_id"], "session-1")
        self.assertTrue(body["stream"])
        self.assertEqual(body["tool_choice"], "auto")
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        parameters = body["tools"][0]["function"]["parameters"]
        self.assertIsInstance(parameters, str)
        self.assertEqual(json.loads(parameters), tools[0]["function"]["parameters"])

        runtime_prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn(r"D:\\code\\demo", runtime_prompt)
        self.assertIn("Windows", runtime_prompt)
        self.assertIn("read_file", runtime_prompt)
        self.assertIn("external client", runtime_prompt)
        self.assertIn("do not describe that server's Linux filesystem", runtime_prompt)

    def test_named_tool_choice_becomes_required_single_tool_for_trae(self):
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

        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(len(body["tools"]), 1)
        self.assertEqual(body["tools"][0]["function"]["name"], "run_shell")

    def test_build_body_preserves_native_tool_result_continuation(self):
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

        assistant_message = next(
            message for message in body["messages"] if message["role"] == "assistant"
        )
        self.assertNotIn("content", assistant_message)
        self.assertEqual(
            assistant_message["tool_calls"],
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function_call": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        )

        tool_message = next(
            message for message in body["messages"] if message["role"] == "tool"
        )
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertEqual(tool_message["name"], "read_file")
        self.assertEqual(
            tool_message["content"], [{"type": "text", "text": "project docs"}]
        )

        history = "\n".join(
            block["text"]
            for message in body["messages"]
            for block in message.get("content", [])
        )
        self.assertNotIn("Previous client tool request", history)
        self.assertNotIn("Client tool calls already issued", history)
        self.assertNotIn("Client tool call [call-1]", history)
        self.assertNotIn("Client tool result [call-1]", history)

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

    def test_generation_parameters_are_forwarded_without_rewriting(self):
        generation_options = {
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
            generation_options,
        )

        for key, value in generation_options.items():
            self.assertEqual(body[key], value)

    def test_model_aliases_use_shared_trae_mapping(self):
        self.assertEqual(
            raw_client.resolve_raw_model("claude-sonnet-4").raw_model_name,
            "glm-5.2",
        )
        self.assertEqual(
            raw_client.resolve_raw_model("gpt-4o").raw_model_name,
            "DeepSeek-V4-Pro",
        )
        self.assertEqual(raw_client.resolve_raw_model("auto").raw_model_name, "glm-5.2")
        self.assertEqual(
            raw_client.resolve_raw_model("glm-5.3").raw_model_name,
            "glm-5.3",
        )

    def test_stable_client_session_still_gets_unique_upstream_request_ids(self):
        messages = [{"role": "user", "content": "hello"}]
        first = raw_client.build_raw_chat_body(
            messages, "auto", session_id="client-session"
        )
        second = raw_client.build_raw_chat_body(
            messages, "auto", session_id="client-session"
        )

        self.assertEqual(first["session_id"], "client-session")
        self.assertEqual(second["session_id"], "client-session")
        self.assertNotEqual(first["request_id"], second["request_id"])

    def test_max_tokens_is_clamped_to_upstream_completion_limit(self):
        body = raw_client.build_raw_chat_body(
            [{"role": "user", "content": "hello"}],
            "deepseek-v4-flash",
            {"max_tokens": 384000},
        )
        self.assertEqual(body["max_tokens"], 131072)

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
                    "rawModelName": "raw-model-v2",
                    "displayName": "Display Model",
                    "client_context": {
                        "workspace_path": r"C:\Users\demo\project",
                        "system_type": "Windows 11",
                    },
                },
                session_id="fixed",
            )
        self.assertEqual(body["model"], "raw-model-v2")
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

    def test_raw_header_environment_is_read_at_call_time(self):
        model = raw_client.RawModel("config", "raw-model", "Display")
        with (
            patch.dict(
                os.environ,
                {
                    "TRAE_RAW_APP_ID": "runtime-app-id",
                    "TRAE_RAW_IDE_VERSION_CODE": "20991231",
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
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-app-id"], "runtime-app-id")
        self.assertEqual(normalized["x-ide-version-code"], "20991231")
        self.assertEqual(normalized["x-request-id"], "request-from-body")
        self.assertEqual(normalized["x-uid"], "user-1")
        self.assertEqual(normalized["authorization"], "Cloud-IDE-JWT token")
        self.assertNotIn("extra", normalized)
        self.assertNotIn("x-ide-function", normalized)

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
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-uid"], "billing-account")


class RawClientRequestTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_display_model_label_is_sent_as_exact_upstream_config(self):
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
                    }
                ),
            ) as lookup,
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            wrapped = await raw_client.send_raw_chat_request(
                [{"role": "user", "content": "hello"}],
                "A future model display label",
            )

        self.assertEqual(captured["body"]["model"], "DeepSeek-V4-Pro-Official")
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

        self.assertEqual(captured["body"]["model"], "DeepSeek-V4-Pro-Official")
        lookup.assert_not_awaited()
        wrapped.close()

    async def test_send_raw_chat_request_uses_llm_utils_protocol(self):
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
        self.assertNotIn("Extra", request.headers)
        self.assertNotIn("X-Ide-Function", request.headers)

        body = json.loads(request.content)
        self.assertEqual(
            set(body),
            {
                "messages",
                "model",
                "function",
                "request_id",
                "session_id",
                "stream",
                "tools",
            },
        )
        self.assertEqual(body["model"], "kimi-k2.6")
        self.assertEqual(body["function"], "inline_chat")
        uuid.UUID(body["request_id"])
        self.assertNotEqual(body["request_id"], "session-fixed")
        self.assertEqual(body["session_id"], "session-fixed")
        self.assertEqual(request.headers["X-Request-Id"], body["request_id"])
        self.assertTrue(body["stream"])
        self.assertEqual(body["tools"][0]["function"]["name"], "shell")
        self.assertIsInstance(body["tools"][0]["function"]["parameters"], str)
        self.assertEqual(
            json.loads(body["tools"][0]["function"]["parameters"]),
            {"type": "object", "properties": {}},
        )
        self.assertNotIn("client_context", body)
        runtime_prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn("shell", runtime_prompt)
        self.assertIn(r"E:\\work", runtime_prompt)

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


if __name__ == "__main__":
    unittest.main()
