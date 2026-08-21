import asyncio
import base64
import json
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
        self.assertEqual(headers["x-os-version"], trae_client.CHECKIN_OS_VERSION)
        self.assertEqual(headers["x-app-version"], trae_client.CHECKIN_APP_VERSION)
        self.assertRegex(headers["x-device-id"], r"^\d{16}$")

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


class IdeRequestContextTests(unittest.TestCase):
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
        self.assertEqual(body["max_tokens"], 131072)
        self.assertEqual(ide_request["max_output_tokens"], 131072)

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


if __name__ == "__main__":
    unittest.main()
