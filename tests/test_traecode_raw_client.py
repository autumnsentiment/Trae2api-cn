import base64
import json
import os
import threading
import unittest
import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from src import raw_client, sse, trae_client


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
        self.assertNotIn("raw_chat_function", body)
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

        self.assertEqual(body["tool_choice"], "auto")
        self.assertNotIn("parallel_tool_calls", body)
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        parameters = body["tools"][0]["function"]["parameters"]
        self.assertIsInstance(parameters, str)
        self.assertEqual(json.loads(parameters)["required"], ["path"])
        assistant = body["messages"][1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"], '{"path":"x"}'
        )
        self.assertEqual(body["messages"][2]["tool_call_id"], "call_1")

    def test_raw_v2_tool_choice_is_always_a_string(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": '{"type":"object"}',
                },
            }
        ]
        cases = (
            ("auto", "auto"),
            (" required ", "required"),
            ({"type": "auto"}, "auto"),
            ({"type": "function", "function": {"name": "read_file"}}, "read_file"),
            ({"name": "read_file"}, "read_file"),
            (True, "auto"),
            (False, "none"),
            (None, None),
        )
        for supplied, expected in cases:
            with self.subTest(choice=supplied):
                body = raw_client.build_traecode_chat_body(
                    [{"role": "user", "content": "read"}],
                    "glm-5.2",
                    options={
                        "_traecode_native_tools": True,
                        "tools": tools,
                        "tool_choice": supplied,
                    },
                    session_id="choice-session",
                )
                self.assertEqual(body.get("tool_choice"), expected)

    def test_raw_v2_schema_string_is_encoded_exactly_once(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        body = raw_client.build_traecode_chat_body(
            [{"role": "user", "content": "read"}],
            "glm-5.2",
            options={
                "_traecode_native_tools": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": json.dumps(schema),
                        },
                    }
                ],
            },
            session_id="schema-session",
        )

        encoded = body["tools"][0]["function"]["parameters"]
        self.assertIsInstance(encoded, str)
        self.assertEqual(json.loads(encoded), schema)

    def test_raw_v2_invalid_schema_is_not_silently_replaced(self):
        for parameters in ("{", '"object"', "true", "null", "[]", [], 7, True):
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, "tool parameters"):
                    raw_client.build_traecode_chat_body(
                        [{"role": "user", "content": "read"}],
                        "glm-5.2",
                        options={
                            "_traecode_native_tools": True,
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "parameters": parameters,
                                    },
                                }
                            ],
                        },
                        session_id="invalid-schema-session",
                    )

    def test_native_tool_definition_helpers_require_object_schemas(self):
        for parameters in ("{", "[]", "false", "1", [], [1], 7, True):
            tool = {
                "type": "function",
                "function": {"name": "read_file", "parameters": parameters},
            }
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, "tool parameters"):
                    raw_client._traecode_tool_definitions([tool])
                with self.assertRaisesRegex(ValueError, "tool parameters"):
                    raw_client._native_tool_definition(tool)
        for parameters in ({}, "{}"):
            tool = {
                "type": "function",
                "function": {"name": "read_file", "parameters": parameters},
            }
            with self.subTest(parameters=parameters):
                self.assertEqual(
                    json.loads(
                        raw_client._traecode_tool_definitions([tool])[0]["function"][
                            "parameters"
                        ]
                    ),
                    {},
                )
                self.assertEqual(
                    json.loads(
                        raw_client._native_tool_definition(tool)["function"]["parameters"]
                    ),
                    {},
                )

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
    async def test_kimi_catalog_miss_does_not_guess_provider_id(self):
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.trae_client._cached_model_catalog",
                new=AsyncMock(
                    return_value={
                        "kimi-k3": {
                            "config_name": "kimi-k3",
                            "model_name": "kimi-k3",
                        }
                    }
                ),
            ),
            patch.object(
                raw_client,
                "resolve_raw_model_for_request",
                new=AsyncMock(
                    side_effect=AssertionError(
                        "Kimi must not fall through to a legacy provider alias"
                    )
                ),
            ),
        ):
            with self.assertRaises(raw_client.TraeCodeModelUnavailable) as raised:
                await raw_client.resolve_traecode_model_for_request(
                    "kimi-k3",
                    {
                        "_account_id": "account-1",
                        "_auth_token": "token-x",
                        "_auth_user_id": "user-1",
                        "_traecode_raw_chat_function": "solo_agent",
                    },
                )

        self.assertEqual(raised.exception.model, "kimi-k3")
        self.assertEqual(raised.exception.raw_chat_function, "solo_agent")

    async def test_kimi_catalog_miss_skips_legacy_gateway_replay(self):
        main_module = __import__("src.main", fromlist=["main"])
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.trae_client._cached_model_catalog",
                new=AsyncMock(
                    return_value={
                        "kimi-k3": {
                            "config_name": "kimi-k3",
                            "model_name": "kimi-k3",
                        }
                    }
                ),
            ),
            patch.object(
                main_module.trae_client,
                "send_chat_request",
                new=AsyncMock(
                    side_effect=AssertionError(
                        "A missing Kimi catalog scope must not replay legacy"
                    )
                ),
            ) as legacy,
        ):
            with self.assertRaises(raw_client.TraeCodeModelUnavailable):
                await main_module.run_ide_chat(
                    [{"role": "user", "content": "hello"}],
                    "kimi-k3",
                    True,
                    {
                        "_account_id": "account-1",
                        "_auth_token": "token-x",
                        "_auth_user_id": "user-1",
                        "_traecode_raw_chat_function": "solo_agent",
                    },
                )

        legacy.assert_not_awaited()

    async def test_kimi_catalog_transport_failure_does_not_become_scope_denial(self):
        provider_model = raw_client.RawModel(
            config_name="kimi-k3",
            raw_model_name="kimi-k3__dev",
            display_name="Kimi-K3",
        )
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.trae_client._cached_model_catalog",
                new=AsyncMock(
                    return_value=trae_client._ModelCatalog(fetch_succeeded=False)
                ),
            ),
            patch.object(
                raw_client,
                "resolve_raw_model_for_request",
                new=AsyncMock(return_value=provider_model),
            ),
        ):
            resolved = await raw_client.resolve_traecode_model_for_request(
                "kimi-k3",
                {
                    "_account_id": "account-1",
                    "_auth_token": "token-x",
                    "_auth_user_id": "user-1",
                    "_traecode_raw_chat_function": "solo_agent",
                },
            )

        self.assertEqual(resolved.raw_model_name, "kimi-k3__dev")

    async def test_catalog_success_without_model_fails_closed(self):
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.trae_client._cached_model_catalog",
                new=AsyncMock(
                    return_value={
                        "some-other-model": {
                            "config_name": "some-other-model",
                            "model_name": "some-other-model__dev",
                        }
                    }
                ),
            ),
            patch.object(
                raw_client,
                "resolve_raw_model_for_request",
                new=AsyncMock(
                    side_effect=AssertionError(
                        "A successful catalog miss must not synthesize provider ids"
                    )
                ),
            ),
        ):
            with self.assertRaises(raw_client.TraeCodeModelUnavailable):
                await raw_client.resolve_traecode_model_for_request(
                    "Doubao-Seed-2.0-Code",
                    {
                        "_account_id": "account-1",
                        "_auth_token": "token-x",
                        "_auth_user_id": "user-1",
                        "_traecode_raw_chat_function": "chat_v3",
                    },
                )

    async def test_successful_empty_catalog_does_not_guess_provider_id(self):
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.trae_client._cached_model_catalog",
                new=AsyncMock(
                    return_value=trae_client._ModelCatalog(fetch_succeeded=True)
                ),
            ),
            patch.object(
                raw_client,
                "resolve_raw_model_for_request",
                new=AsyncMock(),
            ) as legacy_lookup,
        ):
            with self.assertRaises(raw_client.TraeCodeModelUnavailable):
                await raw_client.resolve_traecode_model_for_request(
                    "missing-model",
                    {
                        "_auth_token": "token-x",
                        "_account_id": "account-1",
                        "_traecode_raw_chat_function": "chat_v3",
                    },
                )

        legacy_lookup.assert_not_awaited()

    async def test_custom_provider_without_binding_fails_before_wire(self):
        catalog = {
            "mimo-v2.5": {
                "config_name": "mimo-v2.5",
                "model_name": "mimo-v2.5",
                "provider": "custom_openai_compatible",
                "base_url": "",
                "ak": "",
            }
        }
        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=catalog["mimo-v2.5"]),
            ),
            patch.object(
                raw_client,
                "send_traecode_chat_request",
                new=AsyncMock(
                    side_effect=AssertionError(
                        "unbound custom provider must not reach the wire"
                    )
                ),
            ) as send,
        ):
            with self.assertRaises(
                raw_client.TraeCodeModelUnavailable
            ) as raised:
                await raw_client.resolve_traecode_model_for_request(
                    "mimo-v2.5",
                    {
                        "_account_id": "account-1",
                        "_auth_token": "token-x",
                        "_auth_user_id": "user-1",
                        "_traecode_raw_chat_function": "chat_v3",
                    },
                )

        self.assertEqual(
            raised.exception.reason, "custom_provider_requires_client_direct"
        )
        send.assert_not_awaited()

    async def test_custom_provider_nested_binding_requires_client_direct(self):
        catalog = {
            "mimo-v2.5": {
                "config_name": "mimo-v2.5",
                "model_name": "mimo-v2.5",
                "_traecode_model_detail_list": [
                    {
                        "model_name": "mimo-v2.5",
                        "provider": "custom_openai_compatible",
                        "custom_config": {
                            "base_url": "https://upstream.example/v1/chat/completions",
                            "ak": "x",
                            "client_connect": True,
                        },
                    }
                ],
            }
        }
        with patch(
            "src.trae_client.resolve_traecode_model_config",
            new=AsyncMock(return_value=catalog["mimo-v2.5"]),
        ):
            with self.assertRaises(
                raw_client.TraeCodeModelUnavailable
            ) as raised:
                await raw_client.resolve_traecode_model_for_request(
                    "mimo-v2.5",
                    {
                        "_account_id": "account-1",
                        "_auth_token": "token-x",
                        "_auth_user_id": "user-1",
                        "_traecode_raw_chat_function": "chat_v3",
                    },
                )

        self.assertEqual(
            raised.exception.reason, "custom_provider_requires_client_direct"
        )

    async def test_custom_provider_guard_uses_selected_detail_only(self):
        native = {
            "model_name": "mixed-model__dev",
            "provider": "native-provider",
        }
        custom = {
            "model_name": "mixed-model__max",
            "provider": "custom_openai_compatible",
            "custom_config": {"client_connect": True},
        }
        for details in ([custom, native], [native, custom]):
            for selected, denied in (
                ("mixed-model__dev", False),
                ("mixed-model__max", True),
            ):
                with self.subTest(details=details, selected=selected):
                    descriptor = {
                        "config_name": "mixed-model",
                        "raw_model_name": selected,
                        "provider": "custom_openai_compatible",
                        "_traecode_model_detail_list": details,
                    }
                    with patch(
                        "src.trae_client.resolve_traecode_model_config",
                        new=AsyncMock(return_value=descriptor),
                    ):
                        options = {
                            "_auth_token": "token-x",
                            "_account_id": "account-1",
                            "_traecode_raw_chat_function": "chat_v3",
                        }
                        if denied:
                            with self.assertRaises(
                                raw_client.TraeCodeModelUnavailable
                            ) as raised:
                                await raw_client.resolve_traecode_model_for_request(
                                    "mixed-model", options
                                )
                            self.assertEqual(
                                raised.exception.reason,
                                "custom_provider_requires_client_direct",
                            )
                        else:
                            resolved = (
                                await raw_client.resolve_traecode_model_for_request(
                                    "mixed-model", options
                                )
                            )
                            self.assertEqual(resolved.raw_model_name, selected)

    async def test_kimi_k3_uses_solo_scope_in_body_and_headers(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"event: done\ndata: {}\n\n",
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        live_model = raw_client.RawModel(
            config_name="kimi-k3",
            raw_model_name="kimi-k3__dev",
            display_name="Kimi-K3",
            descriptor={"raw_chat_function": "solo_agent"},
        )
        with (
            patch.object(
                raw_client,
                "resolve_traecode_model_for_request",
                new=AsyncMock(return_value=live_model),
            ),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            response = await raw_client.send_traecode_chat_request(
                [{"role": "user", "content": "Read README.md"}],
                "kimi-k3",
                options={
                    "_account_id": "account-1",
                    "_auth_token": "jwt",
                    "_auth_user_id": "user-1",
                    "_traecode_raw_chat_function": "solo_agent",
                    "_traecode_native_tools": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        self.assertEqual(len(requests), 1)
        request = requests[0]
        body = json.loads(request.content)
        self.assertEqual(body["config_name"], "kimi-k3")
        self.assertEqual(body["model_name"], "kimi-k3__dev")
        self.assertEqual(request.headers["X-App-Function"], "solo_agent")
        self.assertEqual(request.headers["X-Ide-Function"], "solo_agent")
        self.assertTrue(body["stream"])
        response.close()

    async def test_normalized_k3_hint_uses_solo_scope_on_wire(self):
        """The raw transport must honor the dispatcher's K3 override."""

        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"event: done\ndata: {}\n\n",
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        live_model = raw_client.RawModel(
            config_name="kimi-k3",
            raw_model_name="kimi-k3__dev",
            display_name="Kimi-K3",
            descriptor={"raw_chat_function": "solo_agent"},
        )
        function = __import__("src.main", fromlist=["main"])._effective_traecode_function(
            "kimi-k3", {"_traecode_raw_chat_function": "chat_v3"}
        )
        with (
            patch.object(
                raw_client,
                "resolve_traecode_model_for_request",
                new=AsyncMock(return_value=live_model),
            ),
            patch("src.trae_client.build_headers", return_value={}),
            patch("src.raw_client.httpx.Client", side_effect=client_factory),
        ):
            response = await raw_client.send_traecode_chat_request(
                [{"role": "user", "content": "Read README.md"}],
                "kimi-k3",
                options={
                    "_account_id": "account-1",
                    "_auth_token": "jwt",
                    "_auth_user_id": "user-1",
                    "_traecode_raw_chat_function": function,
                    "_traecode_native_tools": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        self.assertEqual(function, "solo_agent")
        self.assertEqual(len(requests), 1)
        request = requests[0]
        body = json.loads(request.content)
        self.assertEqual(
            request.url.path, "/api/ide/v2/llm_raw_chat"
        )
        self.assertEqual(body["config_name"], "kimi-k3")
        self.assertEqual(body["model_name"], "kimi-k3__dev")
        self.assertEqual(request.headers["X-App-Function"], "solo_agent")
        self.assertEqual(request.headers["X-Ide-Function"], "solo_agent")
        response.close()

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
        self.assertNotIn("raw_chat_function", body)
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
            (400, {"code": 4011, "message": "busy"}, 4011),
            (403, {"error": {"error_code": "4011"}}, "4011"),
            (400, {"error": {"errorCode": 429}}, 429),
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

    async def test_http_failures_preserve_metadata_and_stop_fatal_model_retries(self):
        cases = [
            (400, {"code": 4023}, 4023, 1),
            (403, {"error": {"errorCode": "4023"}}, "4023", 1),
            (403, {"code": 0, "error": {"code": 4023}}, 4023, 1),
            (403, {"code": "model_not_entitled"}, "model_not_entitled", 1),
            (
                403,
                {"code": 1005, "error": {"type": "model_not_entitled"}},
                "model_not_entitled",
                1,
            ),
            (403, {"error": "model_not_entitled"}, "model_not_entitled", 1),
            (400, {"code": 4001}, 4001, 4),
            (400, {"code": 0, "error": {"code": 4001}}, 4001, 4),
            (403, {"error": {"error_code": "1005"}}, "1005", 4),
            (403, {"code": 1, "error": {"code": "1005"}}, "1005", 4),
            (502, {"code": 9876}, 9876, 4),
            (404, "missing endpoint", None, 4),
            (503, "x" * 1200, None, 4),
        ]
        for status_code, payload, expected_code, expected_attempts in cases:
            with self.subTest(status_code=status_code, payload=payload):
                requests = []
                clients = []
                responses = []
                session_id = "http-failure-metadata-test"

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
                    with self.assertRaises(raw_client.TraeCodeRequestError) as raised:
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

                error = raised.exception
                self.assertNotIsInstance(error, raw_client.TraeCodeRateLimitError)
                self.assertEqual(error.status_code, status_code)
                self.assertEqual(error.code, expected_code)
                self.assertLessEqual(len(error.detail), 800)
                self.assertEqual(len(requests), expected_attempts)
                self.assertTrue(all(response.is_closed for response in responses))
                self.assertTrue(clients[0].is_closed)
                with raw_client._RAW_SESSION_GATES_LOCK:
                    self.assertNotIn(session_id, raw_client._RAW_SESSION_GATES)

    async def test_later_transport_error_does_not_reuse_prior_http_metadata(self):
        for disable_fallback in (False, True):
            with self.subTest(disable_fallback=disable_fallback):
                requests = []
                clients = []
                responses = []

                def handler(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    if len(requests) > 1:
                        raise httpx.ReadError("synthetic transport failure", request=request)
                    response = httpx.Response(400, json={"code": 4001})
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
                    with self.assertRaises(RuntimeError) as raised:
                        await raw_client.send_traecode_chat_request(
                            [{"role": "user", "content": "hello"}],
                            "glm-5.2",
                            options={
                                "_auth_token": "jwt",
                                "_disable_upstream_fallback": disable_fallback,
                            },
                        )

                if disable_fallback:
                    self.assertIsInstance(raised.exception, raw_client.TraeCodeRequestError)
                    self.assertEqual(raised.exception.code, 4001)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(
                        str(requests[0].url),
                        "https://trae-api-cn.mchost.guru/api/ide/v2/llm_raw_chat",
                    )
                else:
                    self.assertNotIsInstance(
                        raised.exception, raw_client.TraeCodeRequestError
                    )
                    self.assertIn("synthetic transport failure", str(raised.exception))
                    self.assertEqual(len(requests), 4)
                self.assertTrue(clients[0].is_closed)
                self.assertTrue(all(response.is_closed for response in responses))

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

    async def test_max_model_suffix_selects_live_max_provider_model_id(self):
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
                "glm-5.2-max",
                {
                    "_traecode_raw_chat_function": "solo_agent",
                    "_auth_token": "token-x",
                    "_auth_user_id": "user-x",
                },
            )

        self.assertEqual(resolved.config_name, "glm-5.2")
        self.assertEqual(resolved.raw_model_name, "glm-5.2__max")
        self.assertEqual(lookup.await_args.args[0], "glm-5.2")
        self.assertTrue(lookup.await_args.kwargs["max_mode"])
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
