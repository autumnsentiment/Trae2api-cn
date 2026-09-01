import asyncio
import base64
import json
import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src import trae_client


def _fake_jwt(user_id: str, issued_at: int) -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return ".".join(
        (
            encode({"alg": "none", "typ": "JWT"}),
            encode({"data": {"id": user_id}, "iat": issued_at}),
            f"signature-{issued_at}",
        )
    )


class CheckinDeviceTests(unittest.TestCase):
    def test_device_id_survives_jwt_refresh_for_same_account(self):
        first = _fake_jwt("account-42", 100)
        refreshed = _fake_jwt("account-42", 200)

        first_id = trae_client.checkin_device_id_for(first)
        refreshed_id = trae_client.checkin_device_id_for(refreshed)

        self.assertEqual(first_id, refreshed_id)
        self.assertRegex(first_id, r"^\d{16}$")

    def test_device_ids_are_distinct_between_accounts(self):
        first = _fake_jwt("account-42", 100)
        second = _fake_jwt("account-43", 100)

        self.assertNotEqual(
            trae_client.checkin_device_id_for(first),
            trae_client.checkin_device_id_for(second),
        )

    def test_jwt_identity_wins_over_legacy_account_store_key(self):
        first = _fake_jwt("account-42", 100)
        refreshed = _fake_jwt("account-42", 200)

        self.assertEqual(
            trae_client.checkin_device_id_for(first, "legacy-row-key"),
            trae_client.checkin_device_id_for(refreshed, "renamed-row-key"),
        )
        self.assertEqual(
            trae_client.build_checkin_headers(first, "legacy-row-key")["x-device-id"],
            trae_client.build_checkin_headers(first)["x-device-id"],
        )

    def test_headers_include_client_checkin_metadata(self):
        headers = trae_client.build_checkin_headers(_fake_jwt("account-44", 100))

        self.assertEqual(headers["x-device-type"], "windows")
        self.assertRegex(headers["x-device-id"], r"^\d{16}$")
        self.assertNotIn("x-os-version", headers)
        self.assertNotIn("x-app-version", headers)

    def test_global_traework_device_id_override_wins(self):
        token = _fake_jwt("account-override", 100)

        with patch.dict(os.environ, {"TRAE_CHECKIN_DEVICE_ID": "traework-machine-did"}):
            self.assertEqual(
                trae_client.checkin_device_id_for(token), "traework-machine-did"
            )

    def test_per_account_traework_device_id_override_wins(self):
        first = _fake_jwt("account-first", 100)
        second = _fake_jwt("account-second", 100)
        mapping = json.dumps(
            {
                "account-first": "first-machine-did",
                "legacy-second-row": "second-machine-did",
            }
        )

        with patch.dict(
            os.environ,
            {
                "TRAE_CHECKIN_DEVICE_ID": "fallback-machine-did",
                "TRAE_CHECKIN_DEVICE_IDS_JSON": mapping,
            },
        ):
            self.assertEqual(
                trae_client.checkin_device_id_for(first), "first-machine-did"
            )
            self.assertEqual(
                trae_client.checkin_device_id_for(second, "legacy-second-row"),
                "second-machine-did",
            )

    def test_device_already_checked_code_does_not_claim_account_checked(self):
        token = _fake_jwt("account-45", 100)

        class FakeResponse:
            status_code = 200
            text = '{"code":9095,"message":"device already checked"}'

            @staticmethod
            def json():
                return {"code": 9095, "message": "device already checked"}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json, headers):
                self.headers = headers
                return FakeResponse()

        fake_client = FakeClient()
        with patch("src.trae_client.httpx.AsyncClient", return_value=fake_client):
            result = asyncio.run(
                trae_client._post_checkin(
                    "/trae/api/v2/ug/checkin_credits/claim",
                    token,
                    account_id="account-45",
                )
            )

        self.assertEqual(result["code"], 9095)
        self.assertNotIn("checked_in", result)
        self.assertNotIn("success", result)
        self.assertEqual(
            fake_client.headers["x-device-id"],
            trae_client.checkin_device_id_for(token, "account-45"),
        )

    def test_rate_limit_is_returned_without_immediate_retry(self):
        upstream = AsyncMock(
            return_value={"code": 9074, "message": "operation too frequent"}
        )
        with patch("src.trae_client._post_checkin", new=upstream):
            result = asyncio.run(
                trae_client.claim_checkin_credits("token-1", "account-1")
            )

        self.assertEqual(result["code"], 9074)
        upstream.assert_awaited_once_with(
            "/trae/api/v2/ug/checkin_credits/claim", "token-1", account_id="account-1"
        )

    def test_status_rejects_non_boolean_protocol_fields(self):
        class FakeResponse:
            status_code = 200
            text = '{"code":0,"enable":1,"checked_in":false}'

            @staticmethod
            def json():
                return {"code": 0, "enable": 1, "checked_in": False}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json, headers):
                return FakeResponse()

        with patch("src.trae_client.httpx.AsyncClient", return_value=FakeClient()):
            with self.assertRaisesRegex(RuntimeError, "must be boolean"):
                asyncio.run(
                    trae_client._post_checkin(
                        "/trae/api/v2/ug/checkin_credits/status",
                        _fake_jwt("account-status", 100),
                    )
                )

    def test_status_drops_non_positive_credits(self):
        class FakeResponse:
            status_code = 200
            text = '{"code":0,"enable":true,"checked_in":false,"credits":0}'

            @staticmethod
            def json():
                return {"code": 0, "enable": True, "checked_in": False, "credits": 0}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json, headers):
                return FakeResponse()

        with patch("src.trae_client.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(
                trae_client._post_checkin(
                    "/trae/api/v2/ug/checkin_credits/status",
                    _fake_jwt("account-status", 100),
                )
            )

        self.assertNotIn("credits", result)


class WebModelCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_bound_account_model_config_cache_does_not_cross_contaminate(self):
        token_a = _fake_jwt("account-a", 100)
        token_b = _fake_jwt("account-b", 100)
        fetched = []

        async def fetch(token):
            fetched.append(token)
            return {
                "requested-model": {
                    "config_name": "config-a" if token == token_a else "config-b"
                }
            }

        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", side_effect=fetch),
            patch.object(trae_client.auth, "get_user_id", return_value="account-a"),
        ):
            model_b = await trae_client._get_web_custom_model(
                "requested-model",
                token_override=token_b,
                user_id_override="account-b",
            )
            # A second B request should reuse B's own cache entry.
            cached_b = await trae_client._get_web_custom_model(
                "requested-model",
                token_override=token_b,
                user_id_override="account-b",
            )
            model_a = await trae_client._get_web_custom_model(
                "requested-model",
                token_override=token_a,
                user_id_override="account-a",
            )

        self.assertEqual(model_b["config_name"], "config-b")
        self.assertEqual(cached_b["config_name"], "config-b")
        self.assertEqual(model_a["config_name"], "config-a")
        self.assertEqual(fetched, [token_b, token_a])

    async def test_web_session_resolves_custom_model_with_bound_credentials(self):
        token_b = _fake_jwt("account-b", 100)

        class FakeResponse:
            status_code = 200
            text = "{}"

            @staticmethod
            def json():
                return {"data": {"chat_session_id": "session-b", "message_id": "message-b"}}

        class FakeClient:
            request = None

            async def post(self, *_args, **_kwargs):
                self.request = _kwargs
                return FakeResponse()

        lookup = AsyncMock(
            return_value={"name": "requested-model", "config_name": "config-b"}
        )
        provider_specific = {
            "webId": "bound-web",
            "bizUserId": "bound-biz",
            "appLanguage": "ja-JP",
            "userRegion": "JP",
        }
        fake_client = FakeClient()
        with (
            patch.object(trae_client, "_get_web_custom_model", new=lookup),
            patch.object(
                trae_client.auth,
                "get_psd",
                return_value={
                    "webId": "wrong-global-web",
                    "appLanguage": "wrong-global-language",
                    "userRegion": "wrong-global-region",
                },
            ),
        ):
            session_id, message_id = await trae_client.create_web_session(
                fake_client,
                "requested-model",
                [{"role": "user", "content": "hello"}],
                options={
                    "_auth_token": token_b,
                    "_auth_user_id": "account-b",
                    "provider_specific": provider_specific,
                },
            )

        self.assertEqual((session_id, message_id), ("session-b", "message-b"))
        lookup.assert_awaited_once_with(
            "requested-model",
            token_override=token_b,
            user_id_override="account-b",
            provider_specific=provider_specific,
        )
        common = json.loads(
            fake_client.request["json"]["initial_message"]["common_params"]
        )
        self.assertEqual(common["web_id"], "bound-web")
        self.assertEqual(common["biz_user_id"], "bound-biz")
        self.assertEqual(fake_client.request["headers"]["X-Preferenced-Language"], "ja-JP")
        self.assertEqual(fake_client.request["headers"]["x-user-region"], "JP")

    def test_web_headers_preserve_bound_empty_provider_metadata(self):
        with patch.object(
            trae_client.auth,
            "get_psd",
            return_value={"appLanguage": "wrong", "userRegion": "wrong"},
        ):
            headers = trae_client.build_web_headers(
                "token", {"provider_specific": {}}
            )

        self.assertEqual(headers["X-Preferenced-Language"], "zh-CN")
        self.assertEqual(headers["x-user-region"], "CN")


class SessionUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_session_usage_matches_traework_request_and_response(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "code": 0,
                    "data": {
                        "user_usage_group_by_session": {
                            "credits_float": 0.92,
                            "extra_info": {
                                "input_token": 29558,
                                "output_token": 11,
                            },
                        }
                    },
                }

        class Client:
            request = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, url, **kwargs):
                self.request = (url, kwargs)
                return Response()

        client = Client()
        with patch("src.trae_client.httpx.AsyncClient", return_value=client):
            result = await trae_client.fetch_session_usage(
                "user-message-1",
                "jwt-token",
                base_url="https://usage.example",
            )

        url, request = client.request
        self.assertEqual(
            url,
            "https://usage.example/api/v1/commercial/get_session_usage",
        )
        self.assertEqual(request["json"], {"session_id": "user-message-1"})
        self.assertEqual(
            request["headers"],
            {
                "Authorization": "Cloud-IDE-JWT jwt-token",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(
            result,
            {"credits_consumed": 0.92, "credits_source": "session_usage"},
        )
        self.assertNotIn("extra_info", result)

    async def test_fetch_session_usage_preserves_explicit_zero(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"user_usage_group_by_session": {"credits_float": 0}}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return Response()

        with patch("src.trae_client.httpx.AsyncClient", return_value=Client()):
            result = await trae_client.fetch_session_usage("turn-0", "jwt-token")

        self.assertEqual(result["credits_consumed"], 0)

    async def test_fetch_session_usage_rejects_business_error(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"code": 2001, "message": "record not found"}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return Response()

        with patch("src.trae_client.httpx.AsyncClient", return_value=Client()):
            with self.assertRaisesRegex(RuntimeError, "record not found"):
                await trae_client.fetch_session_usage("missing", "jwt-token")


class AccountCreditsParserTests(unittest.TestCase):
    def test_preserves_fractional_limits_usage_and_remaining(self):
        result = trae_client.parse_account_credits(
            {
                "user_entitlement_pack_list": [
                    {
                        "entitlement_base_info": {
                            "available_endpoint": 0,
                            "quota": {"credits_limit": 100.75},
                        },
                        "usage": {"credits_amount": 12.34},
                    },
                    {
                        "entitlement_base_info": {
                            "available_endpoint": 0,
                            "quota": {"credits_limit": 10.25},
                        },
                        "usage": {"credits_amount": 0.56},
                    },
                ]
            }
        )

        self.assertEqual(result["total_limit"], 111)
        self.assertAlmostEqual(result["used"], 12.9)
        self.assertAlmostEqual(result["remaining"], 98.1)
        self.assertIsInstance(result["total_limit"], int)
        self.assertIsInstance(result["used"], float)
        json.dumps(result)

    def test_accepts_numeric_strings_without_truncating(self):
        result = trae_client.parse_account_credits(
            {
                "user_entitlement_pack_list": [
                    {
                        "entitlement_base_info": {
                            "available_endpoint": 1,
                            "quota": {"credits_limit": "20.50"},
                        },
                        "usage": {"credits_amount": "3.25"},
                    }
                ]
            },
            available_endpoint_filter=1,
        )

        self.assertEqual(result["total_limit"], 20.5)
        self.assertEqual(result["used"], 3.25)
        self.assertEqual(result["remaining"], 17.25)
        self.assertTrue(all(not isinstance(value, Decimal) for value in result.values()))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the caller workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

CLIENT_CONTEXT = {
    "workspace_path": r"D:\work\demo",
    "system_type": "Windows 11",
    "terminal_context": [
        {"shell": "PowerShell", "cwd": r"D:\work\demo"}
    ],
}


class ConvertOpenAiMessagesTests(unittest.TestCase):
    def test_preserves_tool_request_result_and_client_runtime(self):
        messages = [
            {"role": "user", "content": "Inspect README.md"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
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
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "project documentation",
            },
        ]

        converted = trae_client.convert_openai_messages(
            messages,
            {"tools": TOOLS, "client_context": CLIENT_CONTEXT},
        )

        self.assertEqual(converted[0]["role"], "system")
        self.assertIn("read_file", converted[0]["content"])
        self.assertIn(r"D:\\work\\demo", converted[0]["content"])
        self.assertIn("Client tool history (already handled; do not repeat)", converted[2]["content"])
        self.assertIn("call_1", converted[2]["content"])
        self.assertEqual(converted[3]["role"], "user")
        self.assertIn("Client tool result [call_1] read_file", converted[3]["content"])
        self.assertIn("project documentation", converted[3]["content"])

    def test_empty_tool_policy_injects_no_tools_runtime(self):
        converted = trae_client.convert_openai_messages(
            [{"role": "user", "content": "answer directly"}],
            {"tools": [], "tool_choice": "none"},
        )
        self.assertEqual(converted[0]["role"], "system")
        self.assertIn("No client tools are available", converted[0]["content"])
        self.assertIn("Tool choice is none", converted[0]["content"])

    def test_none_assistant_content_does_not_become_literal_none(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
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
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "contents",
            },
        ]
        flattened = trae_client.flatten_query(messages)
        structured = json.dumps(
            trae_client.build_web_content(messages), ensure_ascii=False
        )
        self.assertNotIn("None", flattened)
        self.assertNotIn("None", structured)
        self.assertIn("call_1 read_file", structured)

    def test_renderer_tool_use_and_tool_result_survive_remote_flattening(self):
        """Trae's renderer keeps tool history in content blocks, not arrays."""

        messages = [
            {"role": "user", "content": "Download release.zip"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "toolCallId": "call_abc",
                        "name": "download",
                        "parameters": {
                            "url": "https://example.com/release.zip",
                            "dest": "release.zip",
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
        ]

        flattened = trae_client.flatten_query(messages)
        structured = json.dumps(
            trae_client.build_web_content(messages), ensure_ascii=False
        )

        for blob in (flattened, structured):
            self.assertIn("call_abc", blob)
            self.assertIn("download", blob)
            self.assertIn("Downloaded 4096 bytes", blob)
            self.assertNotIn("None", blob)


class IdeRequestContextTests(unittest.TestCase):
    def test_runtime_prompt_uses_inherited_response_tools(self):
        messages = trae_client._messages_with_client_runtime(
            [{"role": "user", "content": "继续"}],
            {
                "_inherited_tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "download_file",
                            "description": "Download a client-side file.",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "_tool_protocol_requested": True,
            },
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("download_file", messages[0]["content"])
        self.assertNotIn("No client tools are available", messages[0]["content"])

    def test_ide_request_uses_bound_account_credential_snapshot(self):
        calls = []

        class FakeResponse:
            status_code = 200

            def close(self):
                pass

        class FakeClient:
            def __init__(self, *, headers, timeout, http2):
                self.headers = headers
                self.timeout = timeout
                self.http2 = http2
                calls.append(self)

            def build_request(self, method, url, **kwargs):
                self.request = (method, url, kwargs)
                return self.request

            def send(self, request, stream=True):
                self.sent = (request, stream)
                return FakeResponse()

            def close(self):
                self.closed = True

        bound_options = {
            "_account_id": "charged-account",
            "_auth_token": "bound-token",
        }
        with (
            patch(
                "src.trae_client.auth.get_account_record",
                return_value={
                    "user_id": "charged-account",
                    "host": "https://charged.example",
                    "token": "stale-token",
                },
            ),
            patch("src.trae_client.auth.get_token", return_value="selected-token"),
            patch("src.trae_client.auth.get_user_id", return_value="selected-account"),
            patch("src.trae_client.auth.get_auth", return_value=SimpleNamespace(host="https://selected.example")),
            patch("src.trae_client.auth.maybe_refresh", new=AsyncMock()) as refresh,
            patch("src.trae_client.httpx.Client", side_effect=FakeClient),
            patch(
                "src.trae_client.build_headers",
                side_effect=lambda **kwargs: {
                    "Authorization": "Cloud-IDE-JWT " + kwargs["token_override"],
                    "x-uid": kwargs["user_id_override"],
                },
            ) as build_headers,
        ):
            response = asyncio.run(
                trae_client.send_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "auto",
                    False,
                    bound_options,
                )
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers["Authorization"], "Cloud-IDE-JWT bound-token")
        self.assertEqual(calls[0].headers["x-uid"], "charged-account")
        self.assertEqual(calls[0].request[1], "https://charged.example/api/agent/v3/llm_utils_chat")
        build_headers.assert_called_once_with(
            token_override="bound-token",
            user_id_override="charged-account",
        )
        refresh.assert_not_awaited()
        response.close()

    def test_unbound_ide_request_can_refresh_selected_account(self):
        class FakeResponse:
            status_code = 200

            def close(self):
                pass

        class FakeClient:
            def __init__(self, **kwargs):
                self.headers = kwargs["headers"]

            def build_request(self, *args, **kwargs):
                return (args, kwargs)

            def send(self, request, stream=True):
                return FakeResponse()

            def close(self):
                pass

        with (
            patch("src.trae_client.auth.maybe_refresh", new=AsyncMock()) as refresh,
            patch("src.trae_client.auth.get_token", return_value="selected-token"),
            patch("src.trae_client.auth.get_user_id", return_value="selected-account"),
            patch("src.trae_client.auth.get_auth", return_value=SimpleNamespace(host="https://selected.example")),
            patch("src.trae_client.httpx.Client", side_effect=FakeClient),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            response = asyncio.run(
                trae_client.send_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "auto",
                    False,
                )
            )

        refresh.assert_awaited_once()
        response.close()

    def test_uses_caller_workspace_and_terminal_context(self):
        _, request = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "hello"}],
                "auto",
                options={"client_context": CLIENT_CONTEXT},
            )
        )

        variables = json.loads(request["variables"])
        self.assertEqual(variables["workspace_path"], r"D:\work\demo")
        self.assertEqual(variables["system_type"], "Windows 11")
        terminal_resolver = next(
            item
            for item in request["context_resolvers"]
            if item["resolver_id"] == "terminal_context"
        )
        terminal_variables = json.loads(terminal_resolver["variables"])
        self.assertEqual(
            terminal_variables["terminal_context"],
            CLIENT_CONTEXT["terminal_context"],
        )

    def test_explicit_session_id_is_preserved(self):
        _, request = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "hello"}],
                "auto",
                options={
                    "client_context": CLIENT_CONTEXT,
                    "session_id": "client-session-1",
                },
            )
        )
        self.assertEqual(request["session_id"], "client-session-1")
        self.assertEqual(request["conversation_id"], "client-session-1")

    def test_ide_and_llm_requests_clamp_completion_tokens(self):
        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hello"}],
            "deepseek-v4-flash",
            True,
            384000,
        )
        _, ide_request = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "hello"}],
                "deepseek-v4-flash",
                384000,
            )
        )
        self.assertEqual(body["max_tokens"], 64000)
        self.assertEqual(ide_request["max_output_tokens"], 64000)

    def test_implicit_sessions_do_not_collide_on_same_prompt(self):
        first = trae_client.generate_session_id_from_messages(
            [{"role": "user", "content": "same"}]
        )
        second = trae_client.generate_session_id_from_messages(
            [{"role": "user", "content": "same"}]
        )
        self.assertNotEqual(first, second)

    def test_web_and_ide_helpers_fail_fast_for_tool_policy(self):
        options = {"tools": [], "tool_choice": "none"}

        with self.assertRaisesRegex(RuntimeError, "web remote"):
            asyncio.run(
                trae_client.create_web_session(
                    None,
                    "auto",
                    [{"role": "user", "content": "hello"}],
                    options,
                )
            )

        with self.assertRaisesRegex(RuntimeError, "IDE endpoints"):
            asyncio.run(
                trae_client.send_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "auto",
                    False,
                    options,
                )
            )

    def test_web_and_ide_helpers_fail_fast_for_tool_history(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "done",
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "web remote"):
            asyncio.run(trae_client.create_web_session(None, "auto", messages))
        with self.assertRaisesRegex(RuntimeError, "IDE endpoints"):
            asyncio.run(trae_client.send_chat_request(messages, "auto", False))


class WebModelGroupPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_remote_group_wins_over_same_name_work(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "list": [
                            {
                                "function": "solo_work_remote",
                                "models": [
                                    {
                                        "name": "glm-5.3",
                                        "max_tokens": 32000,
                                        "context_window_size": {"default": 200000},
                                    }
                                ],
                            },
                            {
                                "function": "solo_agent_remote",
                                "models": [
                                    {
                                        "name": "glm-5.3",
                                        "max_tokens": 64000,
                                        "context_window_size": {
                                            "default": 200000,
                                            "max": 1000000,
                                        },
                                    }
                                ],
                            },
                        ]
                    }
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, url):
                return FakeResponse()

        with patch.object(trae_client.httpx, "AsyncClient", return_value=FakeClient()):
            configs = await trae_client._fetch_web_model_configs(token_override="token-x")

        cfg = configs["glm-5.3"]
        self.assertEqual(cfg["max_tokens"], 64000)
        self.assertEqual(cfg["context_window_size"]["max"], 1000000)

    async def test_fallback_keeps_work_group_when_agent_absent(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "list": [
                            {
                                "function": "solo_work_remote",
                                "models": [{"name": "glm-5.3", "max_tokens": 32000}],
                            }
                        ]
                    }
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, url):
                return FakeResponse()

        with patch.object(trae_client.httpx, "AsyncClient", return_value=FakeClient()):
            configs = await trae_client._fetch_web_model_configs(token_override="token-x")

        self.assertEqual(configs["glm-5.3"]["max_tokens"], 32000)


if __name__ == "__main__":
    unittest.main()
