import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient

# Keep importing this module from changing the process-wide default expected by
# the existing smoke suite. Individual replay cases opt into raw mode locally.
os.environ["TRAE_AUTH_SOURCE"] = "cli"
os.environ["UPSTREAM_MODE"] = "cli"

from src import main, raw_client
from src.sse import EmptyUpstreamResponse


REAL_HTTPX_CLIENT = httpx.Client
AUTH_HEADERS = {"Authorization": "Bearer offline-chain-key"}


def _chat_sse_payloads(raw: str) -> list[dict]:
    payloads = []
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payloads.append(json.loads(line[6:]))
    return payloads


class _ReplayResponse:
    def __init__(self, lines):
        self._lines = list(lines)

    def iter_lines(self):
        return iter(self._lines)


class _ReplayRawChatResponse:
    def __init__(self, lines):
        self.response = _ReplayResponse(lines)
        self.auth_token = "offline-token"
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class OfflineRawChainReplayTests(unittest.TestCase):
    def setUp(self):
        self._usage_history = main._USAGE_HISTORY
        main._USAGE_HISTORY = []
        main._USAGE_ENRICH_TASKS.clear()
        main._USAGE_SNAPSHOT_TASKS.clear()
        main._USAGE_ACTIVE_ACCOUNTS.clear()
        main._USAGE_UNSAFE_ACCOUNTS.clear()

    def tearDown(self):
        main._USAGE_HISTORY = self._usage_history
        main._USAGE_ENRICH_TASKS.clear()
        main._USAGE_SNAPSHOT_TASKS.clear()
        main._USAGE_ACTIVE_ACCOUNTS.clear()
        main._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_http_to_raw_v2_to_openai_sse_is_single_request_and_offline(self):
        captured: dict[str, object] = {"request_count": 0}
        residue = (
            r"\<tool\_call><0c7dc7cb>wait\</tool\_call>\<c4cf82b7>"
            r"\<tool\_call><0c7dc7cb>wait\</tool\_call>\<c4cf82b7>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request_count"] = int(captured["request_count"]) + 1
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            extra = json.loads(request.headers["Extra"])
            captured["extra_model"] = extra.get("model_name")
            stream = "\n".join(
                (
                    "event: output",
                    "data: "
                    + json.dumps(
                        {
                            "response": "chain-ok" + residue,
                            "timing_cost": {
                                "provider_model_name": "glm-5.3__dev"
                            },
                        }
                    ),
                    "",
                    "event: token_usage",
                    "data: "
                    + json.dumps(
                        {
                            "usage": {
                                "input_tokens": 12,
                                "output_tokens": 3,
                                "total_tokens": 15,
                                "credits_consumed": 0.31,
                            }
                        }
                    ),
                    "",
                    "event: done",
                    'data: {"finish_reason":"stop"}',
                    "",
                )
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=stream.encode("utf-8"),
            )

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            return REAL_HTTPX_CLIENT(transport=transport, **kwargs)

        model_lookup = AsyncMock(
            return_value={
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "raw_model_name": "glm-5.3",
                "display_name": "GLM-5.3",
            }
        )
        credit_probe = AsyncMock(return_value=None)
        forbidden_claim = AsyncMock(
            side_effect=AssertionError("model requests must never claim check-in credits")
        )
        account_record = {"token": "offline-token", "user_id": "offline-account"}

        with tempfile.TemporaryDirectory() as temp_dir:
            usage_path = Path(temp_dir) / "usage_records.json"
            client = TestClient(main.app)
            try:
                with (
                    patch("src.main.API_KEYS", ["offline-chain-key"]),
                    patch("src.main.UPSTREAM_MODE", "raw"),
                    patch.object(main, "_USAGE_RECORDS_PATH", usage_path),
                    patch.dict(
                        os.environ,
                        {
                            "TRAE_RAW_BASE_URL": "https://raw.invalid",
                            "TRAE_RAW_V2_MODELS": "glm-5.3",
                            "TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0",
                        },
                        clear=False,
                    ),
                    patch.object(
                        main.auth,
                        "get_active_account_snapshot",
                        return_value=("offline-account", account_record),
                    ),
                    patch.object(
                        main.auth,
                        "get_polling_status",
                        return_value={"enabled": False},
                    ),
                    patch.object(
                        main.auth,
                        "get_active_account_id",
                        return_value="offline-account",
                    ),
                    patch.object(
                        main.auth,
                        "get_account_record",
                        return_value=account_record,
                    ),
                    patch.object(main.auth, "get_token", return_value="offline-token"),
                    patch.object(
                        main.auth, "get_user_id", return_value="offline-account"
                    ),
                    patch.object(
                        main.auth,
                        "maybe_refresh",
                        new=AsyncMock(return_value=False),
                    ),
                    patch(
                        "src.trae_client.resolve_model_config", new=model_lookup
                    ),
                    patch("src.trae_client.build_headers", return_value={}),
                    patch("src.raw_client.httpx.Client", side_effect=client_factory),
                    patch.object(main, "_fetch_used_credits", new=credit_probe),
                    patch.object(
                        main.trae_client,
                        "claim_checkin_credits",
                        new=forbidden_claim,
                    ),
                ):
                    response = client.post(
                        "/v1/chat/completions",
                        headers=AUTH_HEADERS,
                        json={
                            "model": "glm-5.3",
                            "stream": True,
                            "session_id": "offline-chain-session",
                            "messages": [
                                {"role": "user", "content": "Run the chain test"}
                            ],
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "path": {"type": "string"}
                                            },
                                        },
                                    },
                                }
                            ],
                        },
                    )
            finally:
                client.close()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["request_count"], 1)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], raw_client.RAW_CHAT_ENDPOINT)
        body = captured["body"]
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
        self.assertEqual(body["config_name"], "glm-5.3")
        self.assertEqual(body["model_name"], "glm-5.3")
        self.assertEqual(captured["extra_model"], "glm-5.3")
        self.assertNotIn("tools", body)
        self.assertIn("read_file", body["messages"][0]["content"][0]["text"])

        payloads = _chat_sse_payloads(response.text)
        visible = "".join(
            choice.get("delta", {}).get("content", "")
            for payload in payloads
            for choice in payload.get("choices", [])
        )
        self.assertEqual(visible, "chain-ok")
        self.assertNotIn("tool_call", visible)
        self.assertNotIn("0c7dc7cb", visible)
        terminal = payloads[-1]
        self.assertEqual(terminal["choices"][0]["finish_reason"], "stop")
        self.assertEqual(terminal["provider_model_name"], "glm-5.3__dev")
        self.assertEqual(terminal["usage"]["credits_consumed"], 0.31)

        self.assertLessEqual(credit_probe.await_count, 1)
        forbidden_claim.assert_not_awaited()
        self.assertEqual(len(main._USAGE_HISTORY), 1)
        self.assertEqual(main._USAGE_HISTORY[0]["credits_consumed"], 0.31)
        self.assertEqual(main._USAGE_HISTORY[0]["credits_source"], "upstream")

    def test_terminal_usage_evidence_blocks_a_second_model_request(self):
        upstream = _ReplayRawChatResponse(
            [
                "event: token_usage",
                'data: {"usage":{"input_token":9,"output_token":0,'
                '"total_token":9,"credits_float":0.12}}',
                "",
                "event: done",
                'data: {"finish_reason":"stop"}',
                "",
            ]
        )
        send = AsyncMock(return_value=upstream)
        track_usage = Mock()

        async def scenario():
            response = await main.run_raw_chat(
                [{"role": "user", "content": "offline empty replay"}],
                "glm-5.3",
                True,
                {"session_id": "offline-empty-usage"},
            )
            return [chunk async for chunk in response.body_iterator]

        with (
            patch.object(main.raw_client, "send_raw_chat_request", new=send),
            patch.object(main, "_track_usage_from_result", new=track_usage),
        ):
            with self.assertRaises(EmptyUpstreamResponse) as raised:
                asyncio.run(scenario())

        self.assertFalse(raised.exception.retryable)
        self.assertTrue(raised.exception.observed_model_event)
        self.assertEqual(send.await_count, 1)
        self.assertEqual(upstream.close_calls, 1)
        track_usage.assert_called_once_with(
            {
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 0,
                    "total_tokens": 9,
                    "credits_consumed": 0.12,
                }
            },
            "glm-5.3",
        )

    def test_eventless_eof_retries_once_but_never_creates_a_third_request(self):
        first = _ReplayRawChatResponse([])
        second = _ReplayRawChatResponse([])
        send = AsyncMock(
            side_effect=[
                first,
                second,
                AssertionError("a third model request must never be created"),
            ]
        )

        async def scenario():
            response = await main.run_raw_chat(
                [{"role": "user", "content": "offline EOF replay"}],
                "glm-5.3",
                True,
                {"session_id": "offline-empty-eof"},
            )
            return [chunk async for chunk in response.body_iterator]

        with patch.object(main.raw_client, "send_raw_chat_request", new=send):
            with self.assertRaises(EmptyUpstreamResponse):
                asyncio.run(scenario())

        self.assertEqual(send.await_count, 2)
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(second.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
