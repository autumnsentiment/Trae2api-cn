import base64
import json
import os
import threading
import unittest
import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from src import raw_client, sse


REAL_HTTPX_CLIENT = httpx.Client


def _jwt(payload):
    def segment(value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f'{segment({"alg": "none", "typ": "JWT"})}.{segment(payload)}.signature'


class TraeCodeRawBodyTests(unittest.TestCase):
    def test_enterprise_body_forces_enterprise_config_source(self):
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "hello"}],
            "glm-5.2",
            options={
                "_enterprise_mode": True,
                "application_config": {"app_id": "enterprise-app"},
            },
            session_id="enterprise-session",
        )

        self.assertEqual(body["config_source"], 2)
        self.assertEqual(
            body["application_config"],
            {"app_id": "enterprise-app"},
        )

    def test_prompt_cache_key_affinity_overrides_relay_local_session(self):
        model = raw_client.RawModel("glm-5.2", "glm-5.2", "glm-5.2")
        common = {
            "_billing_id": "account-1",
            "session_id": "relay-random-session",
            "_prompt_cache_key_affinity": "stable-prefix-a",
        }
        first = raw_client.traecode_session_id(model, common)
        second = raw_client.traecode_session_id(
            model,
            {**common, "session_id": "another-relay-session"},
        )
        other = raw_client.traecode_session_id(
            model,
            {**common, "_prompt_cache_key_affinity": "stable-prefix-b"},
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, "relay-random-session")
        self.assertEqual(
            raw_client.traecode_session_id(
                model,
                {**common, "_prompt_cache_key_affinity": None},
            ),
            "relay-random-session",
        )

    def test_raw_session_id_includes_prompt_cache_affinity_only_when_present(self):
        model = raw_client.RawModel("glm-5.2", "glm-5.2", "glm-5.2")
        base = {"_billing_id": "account-1"}
        self.assertNotEqual(
            raw_client.raw_session_id(model, base),
            raw_client.raw_session_id(
                model,
                {**base, "_prompt_cache_key_affinity": "stable-prefix-a"},
            ),
        )

    def test_ide_body_uses_traecode_function_and_provider_model_id(self):
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "hello"}],
            "glm-5.1",
            options={"_traecode_raw_chat_function": "chat_v3"},
            session_id="ide-session",
        )

        self.assertEqual(body["config_name"], "glm-5.1")
        self.assertEqual(body["model_name"], "glm-5.1")
        self.assertEqual(body["raw_chat_function"], "chat_v3")
        self.assertEqual(body["config_source"], 1)
        self.assertEqual(body["conversation_id"], "ide-session")
        self.assertEqual(body["session_id"], "ide-session")
        self.assertTrue(body["stream"])
        self.assertEqual(
            body["messages"],
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )

    def test_default_body_uses_six_field_prompt_tool_contract(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                },
            }
        ]
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "Inspect the workspace"}],
            "glm-5.2",
            options={"tools": tools},
            session_id="tool-session",
        )

        self.assertEqual(
            set(body),
            {
                "config_name",
                "model_name",
                "conversation_id",
                "session_id",
                "stream",
                "raw_chat_function",
                "config_source",
                "messages",
            },
        )
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn("read_file", prompt)
        self.assertIn("opencode_tool_call", prompt)

    def test_solo_body_preserves_native_tools_and_tool_history(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a caller file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        messages = [
            {"role": "user", "content": "Read x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"x"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "contents",
            },
        ]
        body = raw_client.build_traecode_chat_body(
            messages,
            "glm-5.3",
            options={
                "_traecode_solo": True,
                "_traecode_raw_chat_function": "solo_agent",
                "_traecode_native_tools": True,
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
            session_id="solo-session",
        )

        self.assertEqual(body["raw_chat_function"], "solo_agent")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        self.assertIsInstance(body["tools"][0]["function"]["parameters"], dict)
        assistant = body["messages"][1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"], '{"path":"x"}'
        )
        self.assertEqual(body["messages"][2]["tool_call_id"], "call_1")

    def test_requested_effort_is_sent_in_expanded_raw_v2_body(self):
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "think"}],
            "deepseek-v4.1-flash",
            options={
                "_traecode_raw_chat_function": "solo_agent",
                "reasoning_effort": "ultra",
            },
            session_id="effort-session",
        )

        self.assertEqual(body["reasoning_effort"], "ultra")

    def test_live_descriptor_clamps_effort_and_uses_declared_wire_key(self):
        descriptor = {
            "config_name": "deepseek-v4.1-flash",
            "reasoning_effort_config": {
                "support_thinking": True,
                "options": ["light", "high"],
            },
            "reasoning_effort_level": "light",
        }
        trace = {}
        options = {
            "reasoning_effort": "ultra",
            "_traecode_model_descriptor": descriptor,
            "_upstream_trace": trace,
        }
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "think"}],
            "deepseek-v4.1-flash",
            options=options,
            session_id="descriptor-effort-session",
        )

        self.assertEqual(body["reasoning_effort_level"], "high")
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(trace["reasoning_effort_applied"], "high")
        self.assertEqual(descriptor["reasoning_effort_level"], "light")
        self.assertNotIn("reasoning_effort", descriptor)
        self.assertEqual(options["_traecode_model_descriptor"], descriptor)

    def test_unsupported_live_descriptor_omits_effort_and_clears_applied_trace(self):
        descriptor = {
            "config_name": "text-only-model",
            "reasoning_effort_config": {
                "support_thinking": False,
                "options": ["high", "extra_high"],
            },
        }
        trace = {"reasoning_effort_applied": "extra_high"}
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "think"}],
            "text-only-model",
            options={
                "reasoning_effort": "high",
                "reasoning_effort_level": "extra_high",
                "_traecode_model_descriptor": descriptor,
                "_upstream_trace": trace,
            },
            session_id="unsupported-effort-session",
        )

        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("reasoning_effort_level", body)
        self.assertNotIn("reasoning_effort_applied", trace)
        self.assertEqual(descriptor["reasoning_effort_config"]["support_thinking"], False)

    def test_expanded_raw_v2_fields_are_whitelisted_and_deep_copied(self):
        metadata = {"nested": {"source": "test"}}
        request_envelope = {"client_info": {"version": "3.4.4"}}
        options = {
            "message": {"role": "user", "content": "hello"},
            "isCustomModel": False,
            "provider": "volcengine",
            "Request": request_envelope,
            "encryptedModelParams": {"ciphertext": "encrypted"},
            "userInput": "hello",
            "request_type": "chat",
            "metadata": metadata,
            "queueId": "queue-1",
            "request_seq": 0,
            "modeType": "ide",
            "chat_mode": "chat",
            "accessType": "default",
            "image_config": {"detail": "auto"},
            "estimatePromptTokens": False,
            "slash_command": {"name": "review"},
            "messageOrigin": "input-box",
            "reasoning_effort": "high",
            "retryTimes": 0,
            "queue_retry": False,
            "videoConfig": {"fps": 1},
            "connect_check_param": {"timeout": 3},
            "userSelectedModelName": "GLM-5.2",
            "reasoning_effort_level": "high",
            "estimatedTokenUsage": {"prompt": 42},
            "custom_model_uniq_id": "custom-1",
            "providerContext": {"region": "cn"},
            "biz_context": {"scene": "code"},
            "extraInfo": {"feature": "raw-v2"},
            "use_fast_request": False,
            "internal_scheduler_state": {"must_not": "leak"},
        }

        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "hello"}],
            "glm-5.2",
            options=options,
            session_id="expanded-session",
        )

        self.assertFalse(body["is_custom_model"])
        self.assertEqual(body["request_seq"], 0)
        self.assertFalse(body["estimate_prompt_tokens"])
        self.assertFalse(body["queue_retry"])
        self.assertFalse(body["use_fast_request"])
        self.assertEqual(body["Request"], request_envelope)
        self.assertEqual(body["metadata"], metadata)
        self.assertEqual(body["reasoning_effort_level"], "high")
        self.assertEqual(body["custom_model_uniq_id"], "custom-1")
        self.assertNotIn("internal_scheduler_state", body)

        metadata["nested"]["source"] = "mutated"
        request_envelope["client_info"]["version"] = "mutated"
        self.assertEqual(body["metadata"]["nested"]["source"], "test")
        self.assertEqual(body["Request"]["client_info"]["version"], "3.4.4")

    def test_explicit_config_source_is_preserved(self):
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "hello"}],
            "custom-model",
            options={
                "traecode_config_name": "custom-model",
                "traecode_model_name": "provider-model-v2",
                "traecode_config_source": 2,
            },
            session_id="enterprise-session",
        )

        self.assertEqual(body["config_name"], "custom-model")
        self.assertEqual(body["model_name"], "provider-model-v2")
        self.assertEqual(body["config_source"], 2)

    def test_traecode_headers_use_new_app_id_without_legacy_extra(self):
        with patch("src.trae_client.build_headers", return_value={}):
            headers = raw_client.build_traecode_headers(
                "https://gateway.example",
                "jwt",
                raw_client.resolve_traecode_model("glm-5.2"),
                "request-id",
                {"_auth_user_id": "user-1"},
                session_id="session-1",
                raw_chat_function="chat_v3",
            )

        self.assertEqual(headers["X-App-Id"], raw_client.TRAECODE_APP_ID)
        self.assertEqual(headers["Authorization"], "Cloud-IDE-JWT jwt")
        self.assertEqual(headers["X-Ide-Token"], "jwt")
        self.assertEqual(headers["X-App-Function"], "chat_v3")
        self.assertEqual(headers["X-Ide-Function"], "chat_v3")
        self.assertNotIn("Extra", headers)
        self.assertTrue(headers["X-Custom-Trace-Id"])
        self.assertTrue(headers["X-Flow-Traceparent"].startswith("04-"))

    def test_traecode_headers_fall_back_to_billing_identity(self):
        with (
            patch.object(raw_client.auth, "get_user_id", return_value="stale-global-user"),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            headers = raw_client.build_traecode_headers(
                "https://gateway.example",
                "jwt",
                raw_client.resolve_traecode_model("glm-5.2"),
                "request-id",
                {"_billing_id": "jwt-billing-user"},
                session_id="session-1",
                raw_chat_function="chat_v3",
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-uid"], "jwt-billing-user")

    def test_traecode_headers_prefer_identity_from_parseable_jwt(self):
        token = _jwt({"sub": "jwt-user", "data": {"id": "nested-user"}})
        with (
            patch.object(raw_client.auth, "get_user_id", return_value="global-user"),
            patch("src.trae_client.build_headers", return_value={}) as build_headers,
        ):
            headers = raw_client.build_traecode_headers(
                "https://gateway.example",
                token,
                raw_client.resolve_traecode_model("glm-5.2"),
                "request-id",
                {"_auth_user_id": "stale-bound-user"},
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-uid"], "nested-user")
        self.assertEqual(
            build_headers.call_args.kwargs["user_id_override"], "nested-user"
        )

    def test_traecode_headers_include_encoded_client_identity_context(self):
        options = {
            "client_context": {
                "machine_id": "machine / one",
                "device_model": "ROG Strix (2026)",
            },
            "provider_specific": {
                "common_params": json.dumps(
                    {
                        "cpu": "Intel Core/Ultra",
                        "device_id": "设备 1",
                        "os_name": "Windows Desktop",
                        "os_version": "Windows 11/24H2",
                    }
                ),
                "webId": "web-123",
                "authorizedServices": ["chat", "builder"],
                "ppeEnvName": "boe-test",
            },
        }
        with patch("src.trae_client.build_headers", return_value={}):
            headers = raw_client.build_traecode_headers(
                "https://gateway.example",
                "opaque-token",
                raw_client.resolve_traecode_model("glm-5.2"),
                "request-id",
                options,
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["x-device-cpu"], "Intel%20Core%2FUltra")
        self.assertEqual(normalized["x-device-id"], "%E8%AE%BE%E5%A4%87%201")
        self.assertEqual(normalized["x-machine-id"], "machine%20%2F%20one")
        self.assertEqual(normalized["x-device-brand"], "ROG%20Strix%20(2026)")
        self.assertEqual(normalized["x-device-type"], "Windows%20Desktop")
        self.assertEqual(normalized["x-os-version"], "Windows%2011%2F24H2")
        self.assertEqual(normalized["x-web-id"], "web-123")
        self.assertEqual(normalized["x-tt-env"], "boe-test")
        self.assertEqual(normalized["x-use-ppe"], "1")
        self.assertEqual(normalized["x-trae-authorized-services"], "chat,builder")
        self.assertEqual(normalized["x-plugin-channel"], "desktop")

    def test_traecode_default_client_version_matches_344_report(self):
        self.assertEqual(raw_client.TRAECODE_IDE_VERSION, "3.4.4")
        self.assertEqual(raw_client.TRAECODE_IDE_VERSION_CODE, "20260212")

    def test_traecode_custom_headers_cannot_override_bound_identity_or_function(self):
        token = _jwt({"data": {"id": "bound-user"}})
        with patch("src.trae_client.build_headers", return_value={}):
            headers = raw_client.build_traecode_headers(
                "https://gateway.example",
                token,
                raw_client.resolve_traecode_model("glm-5.2"),
                "request-id",
                {
                    "headers": {
                        "Authorization": "Cloud-IDE-JWT attacker",
                        "x-ide-token": "attacker",
                        "X-Uid": "attacker",
                        "X-App-Id": "attacker-app",
                        "X-Ide-Function": "solo_work_remote",
                        "X-Diagnostic-Tag": "allowed",
                    }
                },
                raw_chat_function="chat_v3",
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["authorization"], f"Cloud-IDE-JWT {token}")
        self.assertEqual(normalized["x-ide-token"], token)
        self.assertEqual(normalized["x-uid"], "bound-user")
        self.assertEqual(normalized["x-app-id"], raw_client.TRAECODE_APP_ID)
        self.assertEqual(normalized["x-ide-function"], "chat_v3")
        self.assertEqual(normalized["x-diagnostic-tag"], "allowed")


class TraeCodeRawTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_text_delta_events_are_translated(self):
        class Stream:
            def iter_lines(self):
                return iter(
                    [
                        "event: text_delta",
                        'data: {"delta":"Hel"}',
                        "",
                        "event: text_delta",
                        'data: {"delta":"lo"}',
                        "",
                        "event: reasoning_delta",
                        'data: {"delta":"internal"}',
                        "",
                        "event: done",
                        "data: {}",
                        "",
                    ]
                )

        chunks = []
        async for chunk in sse.translate_ide_stream(
            Stream(),
            "glm-5.2",
            fail_on_empty=True,
            require_terminal=True,
            include_reasoning=True,
        ):
            chunks.append(chunk)

        wire = "".join(chunks)
        self.assertIn('"content": "Hel"', wire)
        self.assertIn('"content": "lo"', wire)
        self.assertIn('"reasoning_content": "internal"', wire)
        self.assertIn("data: [DONE]", wire)

    async def test_v2_request_falls_back_to_v1_and_closes(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/v2/llm_raw_chat"):
                return httpx.Response(404, text="not found")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'event: output\ndata: {"response":"ok"}\n\nevent: done\ndata: {}\n\n',
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        with (
            patch.object(
                raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)
            ),
            patch.object(raw_client.auth, "get_token", return_value="jwt"),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            response = await raw_client.send_traecode_chat_request(
                [{"role": "user", "content": "hello"}],
                "glm-5.2",
                options={
                    "_account_id": "account-1",
                    "_auth_token": "jwt",
                    "_auth_user_id": "user-1",
                },
            )

        self.assertEqual([request.url.path for request in requests], [
            "/api/ide/v2/llm_raw_chat",
            "/api/ide/v1/llm_raw_chat",
        ])
        self.assertEqual(requests[-1].headers["X-App-Id"], raw_client.TRAECODE_APP_ID)
        body = json.loads(requests[-1].content)
        self.assertEqual(body["raw_chat_function"], "chat_v3")
        self.assertEqual(body["model_name"], "glm-5.2")
        self.assertNotIn("Extra", requests[-1].headers)
        response.close()
        self.assertTrue(response.client.is_closed)

    async def test_http_rate_limit_stops_host_and_endpoint_fallback(self):
        cases = [
            (429, "busy", 429),
            (400, {"code": 3004, "message": "busy"}, 3004),
            (403, {"code": "3004", "message": "busy"}, "3004"),
            (500, {"error_code": 3004}, 3004),
            (502, {"errorCode": "3004"}, "3004"),
            (503, {"error": {"code": 3004}}, 3004),
            (400, {"error": {"error_code": "3004"}}, "3004"),
            (403, {"error": {"errorCode": 3004}}, 3004),
        ]
        for status_code, payload, expected_code in cases:
            with self.subTest(status_code=status_code, payload=payload):
                requests = []
                clients = []
                responses = []
                session_id = "http-rate-limit-test"

                def handler(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    response = (
                        httpx.Response(status_code, text=payload)
                        if isinstance(payload, str)
                        else httpx.Response(status_code, json=payload)
                    )
                    responses.append(response)
                    return response

                transport = httpx.MockTransport(handler)

                def client_factory(**kwargs):
                    client = REAL_HTTPX_CLIENT(transport=transport, **kwargs)
                    clients.append(client)
                    return client

                with (
                    patch("src.trae_client.build_headers", return_value={}),
                    patch.object(
                        raw_client,
                        "resolve_traecode_base_urls",
                        return_value=[
                            "https://trae-api-cn.mchost.guru",
                            "https://api5-normal.mchost.guru",
                        ],
                    ),
                    patch.object(
                        raw_client,
                        "resolve_traecode_model_for_request",
                        new=AsyncMock(
                            return_value=raw_client.resolve_traecode_model("glm-5.2")
                        ),
                    ),
                    patch("src.raw_client.httpx.Client", side_effect=client_factory),
                ):
                    with self.assertRaises(raw_client.TraeCodeRateLimitError) as raised:
                        await raw_client.send_traecode_chat_request(
                            [{"role": "user", "content": "hello"}],
                            "glm-5.2",
                            options={
                                "_account_id": "account-1",
                                "_auth_token": "jwt",
                                "_auth_user_id": "user-1",
                                "_traecode_session_id": session_id,
                            },
                        )

                self.assertIsInstance(raised.exception, RuntimeError)
                self.assertEqual(raised.exception.status_code, status_code)
                self.assertEqual(raised.exception.code, expected_code)
                self.assertIn("rate limit", str(raised.exception))
                self.assertEqual(len(requests), 1)
                self.assertEqual(len(clients), 1)
                self.assertTrue(clients[0].is_closed)
                self.assertTrue(responses[0].is_closed)
                with raw_client._RAW_SESSION_GATES_LOCK:
                    self.assertNotIn(session_id, raw_client._RAW_SESSION_GATES)

    async def test_http_429_body_read_error_preserves_rate_limit_and_releases_slot(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                requests = []
                clients = []
                responses = []
                session_id = "http-rate-limit-broken-body"

                class BrokenBody(httpx.SyncByteStream):
                    def __iter__(self):
                        yield b'{"error":'
                        raise httpx.ReadError("error body disconnected")

                def handler(request):
                    requests.append(request)
                    response = httpx.Response(429, stream=BrokenBody())
                    responses.append(response)
                    return response

                def client_factory(**kwargs):
                    client = REAL_HTTPX_CLIENT(
                        transport=httpx.MockTransport(handler), **kwargs
                    )
                    clients.append(client)
                    return client

                with (
                    patch("src.trae_client.build_headers", return_value={}),
                    patch.object(
                        raw_client,
                        "resolve_traecode_base_urls",
                        return_value=["https://first.example", "https://second.example"],
                    ),
                    patch.object(
                        raw_client,
                        "resolve_traecode_model_for_request",
                        new=AsyncMock(
                            return_value=raw_client.resolve_traecode_model("glm-5.2")
                        ),
                    ),
                    patch("src.raw_client.httpx.Client", side_effect=client_factory),
                ):
                    with self.assertRaises(raw_client.TraeCodeRateLimitError) as raised:
                        await raw_client.send_traecode_chat_request(
                            [{"role": "user", "content": "hello"}],
                            "glm-5.2",
                            stream=stream,
                            options={
                                "_auth_token": "jwt",
                                "_auth_user_id": "user-1",
                                "_traecode_session_id": session_id,
                            },
                        )

                self.assertEqual(raised.exception.code, 429)
                self.assertEqual(raised.exception.status_code, 429)
                self.assertEqual(len(requests), 1)
                self.assertTrue(clients[0].is_closed)
                self.assertTrue(responses[0].is_closed)
                with raw_client._RAW_SESSION_GATES_LOCK:
                    self.assertNotIn(session_id, raw_client._RAW_SESSION_GATES)

    async def test_unstructured_rate_limit_number_preserves_endpoint_fallback(self):
        cases = [
            "gateway reference 3004",
            {"code": 4001, "message": "gateway reference 3004"},
            [{"code": 3004}],
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                requests = []

                def handler(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    if len(requests) == 1:
                        return (
                            httpx.Response(502, text=payload)
                            if isinstance(payload, str)
                            else httpx.Response(502, json=payload)
                        )
                    return httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        content=b"event: done\ndata: {}\n\n",
                    )

                transport = httpx.MockTransport(handler)

                def client_factory(**kwargs):
                    return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

                with (
                    patch("src.trae_client.build_headers", return_value={}),
                    patch.object(
                        raw_client,
                        "resolve_traecode_base_urls",
                        return_value=["https://gateway.example"],
                    ),
                    patch.object(
                        raw_client,
                        "resolve_traecode_model_for_request",
                        new=AsyncMock(
                            return_value=raw_client.resolve_traecode_model("glm-5.2")
                        ),
                    ),
                    patch("src.raw_client.httpx.Client", side_effect=client_factory),
                ):
                    response = await raw_client.send_traecode_chat_request(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.2",
                        options={
                            "_account_id": "account-1",
                            "_auth_token": "jwt",
                            "_auth_user_id": "user-1",
                        },
                    )

                self.assertEqual(
                    [request.url.path for request in requests],
                    [
                        "/api/ide/v2/llm_raw_chat",
                        "/api/ide/v1/llm_raw_chat",
                    ],
                )
                response.close()
                self.assertTrue(response.client.is_closed)

    async def test_cancellation_keeps_session_gate_until_worker_finishes(self):
        entered_send = threading.Event()
        allow_return = threading.Event()
        clients = []

        class BlockingClient:
            def __init__(self):
                self.close_calls = 0

            def build_request(self, method, url, **kwargs):
                return {"method": method, "url": url, **kwargs}

            def send(self, _request, stream=False):
                entered_send.set()
                allow_return.wait(2.0)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=b"event: done\ndata: {}\n\n",
                )

            def close(self):
                self.close_calls += 1

        def client_factory(**_kwargs):
            client = BlockingClient()
            clients.append(client)
            return client

        with (
            patch.object(
                raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)
            ),
            patch.object(raw_client.auth, "get_token", return_value="jwt"),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            request_task = asyncio.create_task(
                raw_client.send_traecode_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.2",
                    options={
                        "_account_id": "account-1",
                        "_billing_id": "account-1",
                        "_auth_token": "jwt",
                        "_auth_user_id": "user-1",
                    },
                )
            )
            await asyncio.to_thread(entered_send.wait, 1.0)
            request_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request_task

            # The worker still owns the socket and the model-bound gate while
            # ``httpx.Client.send`` is blocked.
            with raw_client._RAW_SESSION_GATES_LOCK:
                self.assertTrue(raw_client._RAW_SESSION_GATES)
            allow_return.set()
            for _ in range(200):
                with raw_client._RAW_SESSION_GATES_LOCK:
                    done = not raw_client._RAW_SESSION_GATES
                if clients and clients[0].close_calls == 1 and done:
                    break
                await asyncio.sleep(0.005)

        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0].close_calls, 1)
        with raw_client._RAW_SESSION_GATES_LOCK:
            self.assertFalse(raw_client._RAW_SESSION_GATES)

    async def test_custom_base_url_strips_remote_api_prefix(self):
        with patch.dict(
            os.environ,
            {"TRAE_CODE_BASE_URL": "https://gateway.example/api/remote/v1"},
            clear=False,
        ):
            self.assertEqual(
                raw_client.resolve_traecode_base_url({}),
                "https://gateway.example",
            )

    async def test_default_gateway_uses_boot_config_host_and_known_alternate(self):
        with patch.dict(
            os.environ,
            {"TRAE_CODE_BASE_URL": "", "TRAE_MODEL_API_HOST": ""},
            clear=False,
        ):
            self.assertEqual(
                raw_client.resolve_traecode_base_url({}),
                "https://trae-api-cn.mchost.guru",
            )
            self.assertEqual(
                raw_client.resolve_traecode_base_urls({}),
                [
                    "https://trae-api-cn.mchost.guru",
                    "https://api5-normal.mchost.guru",
                ],
            )

    async def test_custom_gateway_does_not_add_unrelated_hosts(self):
        self.assertEqual(
            raw_client.resolve_traecode_base_urls(
                {"traecode_base_url": "https://custom.example"}
            ),
            ["https://custom.example"],
        )

    async def test_live_model_mapping_is_kept_when_config_name_matches(self):
        live = raw_client.RawModel(
            config_name="model-alias",
            raw_model_name="actual-provider-model",
            display_name="Actual provider display",
            config_source=2,
            provider="provider-x",
        )
        with patch.object(
            raw_client,
            "resolve_raw_model_for_request",
            new=AsyncMock(return_value=live),
        ):
            resolved = await raw_client.resolve_traecode_model_for_request(
                "model-alias"
            )

        self.assertEqual(resolved.config_name, "model-alias")
        self.assertEqual(resolved.raw_model_name, "actual-provider-model")
        self.assertEqual(resolved.display_name, "Actual provider display")
        self.assertEqual(resolved.config_source, 2)
        self.assertEqual(resolved.provider, "provider-x")

    async def test_native_function_prefers_live_traecode_provider_model_id(self):
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(
                    return_value={
                        "config_name": "glm-5",
                        "model_name": "glm-5__dev",
                        "display_name": "GLM-5.2",
                        "config_source": 1,
                        "provider": "glm-5",
                        "reasoning_effort_config": {
                            "support_thinking": False,
                        },
                    }
                ),
            ) as lookup,
            patch.object(
                raw_client,
                "resolve_raw_model_for_request",
                new=AsyncMock(
                    return_value=raw_client.resolve_traecode_model("glm-5.2")
                ),
            ),
        ):
            resolved = await raw_client.resolve_traecode_model_for_request(
                "glm-5.2",
                {
                    "_traecode_raw_chat_function": "chat_v3",
                    "_auth_token": "token-x",
                    "_auth_user_id": "user-x",
                },
            )

        self.assertEqual(resolved.config_name, "glm-5")
        self.assertEqual(resolved.raw_model_name, "glm-5__dev")
        self.assertEqual(resolved.display_name, "GLM-5.2")
        self.assertFalse(resolved.descriptor["reasoning_effort_config"]["support_thinking"])
        lookup.assert_awaited_once()
        self.assertEqual(lookup.await_args.kwargs["raw_chat_function"], "chat_v3")

    async def test_native_function_selects_max_provider_model_id(self):
        with patch(
            "src.trae_client.resolve_traecode_model_config",
            new=AsyncMock(
                return_value={
                    "config_name": "glm-5.2",
                    "model_name": "glm-5.2__max",
                    "display_name": "GLM-5.2",
                    "config_source": 1,
                }
            ),
        ) as lookup:
            resolved = await raw_client.resolve_traecode_model_for_request(
                "glm-5.2",
                {
                    "_traecode_raw_chat_function": "solo_agent",
                    "trae_max_mode": True,
                    "_auth_token": "token-x",
                    "_auth_user_id": "user-x",
                },
            )

        self.assertEqual(resolved.raw_model_name, "glm-5.2__max")
        self.assertEqual(lookup.await_args.kwargs["max_mode"], True)
        self.assertEqual(
            lookup.await_args.kwargs["raw_chat_function"], "solo_agent"
        )

    async def test_v2_raw_chat_retries_known_gateway_after_http_rejection(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "trae-api-cn.mchost.guru":
                return httpx.Response(502, text="gateway unavailable")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"event: done\ndata: {}\n\n",
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        with (
            patch.object(
                raw_client.auth, "maybe_refresh", new=AsyncMock(return_value=False)
            ),
            patch.object(raw_client.auth, "get_token", return_value="jwt"),
            patch.object(raw_client.auth, "get_user_id", return_value="user-1"),
            patch("src.trae_client.build_headers", return_value={}),
            patch.dict(
                os.environ,
                {"TRAE_CODE_BASE_URL": "", "TRAE_MODEL_API_HOST": ""},
                clear=False,
            ),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
            patch.object(
                raw_client,
                "resolve_traecode_model_for_request",
                new=AsyncMock(return_value=raw_client.resolve_traecode_model("glm-5.2")),
            ),
        ):
            response = await raw_client.send_traecode_chat_request(
                [{"role": "user", "content": "hello"}],
                "glm-5.2",
                options={
                    "_account_id": "account-1",
                    "_auth_token": "jwt",
                    "_auth_user_id": "user-1",
                },
            )

        self.assertEqual(
            [(request.url.host, request.url.path) for request in requests],
            [
                (
                    "trae-api-cn.mchost.guru",
                    "/api/ide/v2/llm_raw_chat",
                ),
                (
                    "trae-api-cn.mchost.guru",
                    "/api/ide/v1/llm_raw_chat",
                ),
                (
                    "api5-normal.mchost.guru",
                    "/api/ide/v2/llm_raw_chat",
                ),
            ],
        )
        response.close()


if __name__ == "__main__":
    unittest.main()
