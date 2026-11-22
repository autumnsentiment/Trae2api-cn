"""Kimi model-scope and reused-index tool-call regressions."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from src import raw_client, sse


class KimiModelScopeFailClosedTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_hit_without_provider_detail_fails_closed(self):
        with patch(
            "src.trae_client.resolve_traecode_model_config",
            new=AsyncMock(
                return_value={
                    "config_name": "kimi-k3",
                    "name": "kimi-k3",
                }
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

    async def test_catalog_hit_with_picker_as_provider_id_fails_closed(self):
        with patch(
            "src.trae_client.resolve_traecode_model_config",
            new=AsyncMock(
                return_value={
                    "config_name": "kimi-k3",
                    "model_name": "kimi-k3",
                }
            ),
        ):
            with self.assertRaises(raw_client.TraeCodeModelUnavailable):
                await raw_client.resolve_traecode_model_for_request(
                    "kimi-k3",
                    {
                        "_account_id": "account-1",
                        "_auth_token": "token-x",
                        "_auth_user_id": "user-1",
                        "_traecode_raw_chat_function": "solo_agent",
                    },
                )

    async def test_catalog_provider_detail_is_accepted(self):
        with patch(
            "src.trae_client.resolve_traecode_model_config",
            new=AsyncMock(
                return_value={
                    "config_name": "kimi-k3",
                    "model_name": "kimi-k3__dev",
                    "display_name": "Kimi-K3",
                }
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

        self.assertEqual(resolved.config_name, "kimi-k3")
        self.assertEqual(resolved.raw_model_name, "kimi-k3__dev")


class ReusedIndexToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_id_head_then_idless_replay_keeps_one_call(self):
        lines = [
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_kimi_replay",
                            "type": "function",
                            "function_call": {
                                "name": "Read",
                                "arguments": '{"filePath":"README.md"}',
                            },
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "type": "function",
                            "function_call": {
                                "name": "Read",
                                "arguments": '{"filePath":"README.md"}',
                            },
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]

        class LineResponse:
            def iter_lines(self):
                return iter(lines)

        result = await sse.collect_nonstream_ide(
            LineResponse(),
            "kimi-k3",
            allowed_tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            fail_on_empty=True,
        )
        calls = result["choices"][0]["message"]["tool_calls"]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_kimi_replay")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"filePath": "README.md"},
        )

    async def test_reused_explicit_index_with_distinct_complete_args_keeps_both_calls(self):
        lines = [
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_kimi_scope_1",
                            "type": "function",
                            "function_call": {
                                "name": "Read",
                                "arguments": '{"filePath":"a.txt"}',
                            },
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "type": "function",
                            "function_call": {
                                "name": "Read",
                                "arguments": '{"filePath":"b.txt"}',
                            },
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]

        class LineResponse:
            def iter_lines(self):
                return iter(lines)

        result = await sse.collect_nonstream_ide(
            LineResponse(),
            "kimi-k3",
            allowed_tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            fail_on_empty=True,
        )
        calls = result["choices"][0]["message"]["tool_calls"]

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [json.loads(call["function"]["arguments"]) for call in calls],
            [{"filePath": "a.txt"}, {"filePath": "b.txt"}],
        )
        self.assertEqual(len({call["id"] for call in calls}), 2)


if __name__ == "__main__":
    unittest.main()
