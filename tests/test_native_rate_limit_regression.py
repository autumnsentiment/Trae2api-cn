import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src import main as main_module
from src import sse
from src.sse import (
    EmptyUpstreamResponse,
    IncompleteUpstreamResponse,
    UpstreamEventError,
)


class _ReplayResponse:
    status_code = 200

    def __init__(self, events):
        self.events = list(events)

    def iter_lines(self):
        for event, payload in self.events:
            yield "event: " + event
            yield "data: " + json.dumps(payload)
            yield ""


def _wrapper(*events):
    return SimpleNamespace(response=_ReplayResponse(events), close=Mock())


class NativeRateLimitClassificationTests(unittest.TestCase):
    def test_structured_3004_is_recognized_without_rate_limit_text(self):
        for code in (3004, "3004"):
            with self.subTest(code=code):
                error = UpstreamEventError("busy", code=code)
                self.assertTrue(main_module._is_native_rate_limit_error(error))

    def test_structured_4008_and_quota_message_are_recognized(self):
        for error in (
            UpstreamEventError("Your requests have exceeded the quota", code=4008),
            UpstreamEventError("busy", account_quota_exhausted=True),
            RuntimeError("Your requests have exceeded the quota"),
        ):
            with self.subTest(error=str(error)):
                self.assertTrue(main_module._is_native_rate_limit_error(error))

    def test_unrelated_3004_text_is_not_a_rate_limit(self):
        self.assertFalse(
            main_module._is_native_rate_limit_error(
                RuntimeError("input contains 3004 characters")
            )
        )

    def test_nonreplayable_3004_is_not_eligible_for_account_retry(self):
        for guard in (
            {"retryable": False},
            {"observed_model_event": True},
            {"usage": {"prompt_tokens": 3}},
        ):
            with self.subTest(guard=guard):
                self.assertFalse(
                    main_module._is_native_rate_limit_error(
                        UpstreamEventError("quota exhausted", code=4008, **guard)
                    )
                )

    def test_4023_is_not_a_rate_limit(self):
        for error in (
            UpstreamEventError(
                "We're sorry, the model is unknown.", code=4023
            ),
            RuntimeError(
                "Trae upstream error 4023: We're sorry, the model is unknown."
            ),
            RuntimeError("the model is unknown"),
        ):
            with self.subTest(error=str(error)):
                self.assertTrue(
                    main_module._is_native_model_unknown_error(error)
                )
                self.assertFalse(
                    main_module._is_native_rate_limit_error(error)
                )

    def test_unrelated_numbers_are_not_model_unknown_errors(self):
        for error in (
            RuntimeError("input exceeds context 14023 tokens"),
            RuntimeError("request id 40230"),
            RuntimeError("model is available"),
        ):
            with self.subTest(error=str(error)):
                self.assertFalse(
                    main_module._is_native_model_unknown_error(error)
                )


class NativeRateLimitChainTests(unittest.IsolatedAsyncioTestCase):
    def _options(self):
        return {
            "_account_id": "account-a",
            "_auth_token": "token-a",
            "_upstream_mode": "ide",
            "_traecode_raw_chat_function": "chat_v3",
            "_ide_endpoint": "/api/agent/v3/llm_utils_chat",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }

    def _patch_chain(self, raw, legacy, *, polling=True):
        stack = ExitStack()
        stack.enter_context(
            patch.object(main_module.raw_client, "send_traecode_chat_request", raw)
        )
        stack.enter_context(
            patch.object(main_module.trae_client, "send_chat_request", legacy)
        )
        stack.enter_context(
            patch.object(
                main_module.auth, "get_polling_status", return_value={"enabled": polling}
            )
        )
        stack.enter_context(
            patch.object(main_module, "_polling_retry_limit", return_value=2)
        )
        stack.enter_context(
            patch.object(
                main_module,
                "_next_retry_account_snapshot",
                return_value=("account-b", {"token": "token-b"}),
            )
        )
        stack.enter_context(
            patch.object(main_module, "_account_id_from_token", return_value="")
        )
        stack.enter_context(patch.object(main_module, "_rebind_chat_session_account"))
        stack.enter_context(patch.object(main_module, "_bind_usage_turn_from_metadata"))
        return stack

    async def _request(self, stream):
        response = await main_module._run_ide_with_retry(
            [{"role": "user", "content": "Reply pong"}],
            "glm-5.2",
            stream,
            self._options(),
        )
        if stream:
            return "".join([chunk async for chunk in response.body_iterator])
        return response.body.decode("utf-8")

    async def _custom_request(self, stream, **extra_options):
        response = await main_module._dispatch_chat(
            [{"role": "user", "content": "Reply pong"}],
            "glm-5.2",
            stream,
            {
                **self._options(),
                "_traework_custom_model": True,
                "_tool_protocol_requested": True,
                **extra_options,
            },
        )
        if stream:
            return "".join([chunk async for chunk in response.body_iterator])
        return response.body.decode("utf-8")

    async def test_custom_raw_v2_4011_rotates_before_output(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                limited = _wrapper(
                    ("model_config", {"model_name": "glm-5.2__dev"}),
                    ("error", {"code": 4011, "message": "rate limit"}),
                )
                completed = _wrapper(
                    ("text_delta", {"delta": "pong"}),
                    ("done", {"finish_reason": "stop"}),
                )
                raw = AsyncMock(side_effect=[limited, completed])
                legacy = AsyncMock()
                with self._patch_chain(raw, legacy):
                    result = await self._custom_request(stream)

                self.assertIn("pong", result)
                self.assertEqual(raw.await_count, 2)
                self.assertEqual(
                    [call.kwargs["options"]["_account_id"] for call in raw.await_args_list],
                    ["account-a", "account-b"],
                )
                for call in raw.await_args_list:
                    self.assertEqual(call.args[1], "glm-5.2")
                    self.assertEqual(
                        call.kwargs["options"]["_traecode_raw_chat_function"],
                        "chat_v3",
                    )
                    self.assertEqual(
                        call.kwargs["options"]["tools"], self._options()["tools"]
                    )
                    self.assertTrue(call.kwargs["options"]["_traecode_native_tools"])
                legacy.assert_not_awaited()
                limited.close.assert_called_once()
                completed.close.assert_called_once()

    async def test_custom_raw_v2_4011_after_output_or_usage_never_replays(self):
        for stream in (False, True):
            for event in (
                ("text_delta", {"delta": "partial"}),
                ("reasoning_delta", {"delta": "reasoning"}),
                ("token_usage", {"input_token": 3, "output_token": 1}),
            ):
                with self.subTest(stream=stream, event=event[0]):
                    limited = _wrapper(
                        event, ("error", {"code": 4011, "message": "rate limit"})
                    )
                    raw = AsyncMock(return_value=limited)
                    legacy = AsyncMock()
                    with (
                        self._patch_chain(raw, legacy),
                        self.assertRaises(UpstreamEventError) as caught,
                    ):
                        await self._custom_request(stream)

                    self.assertEqual(caught.exception.code, 4011)
                    self.assertFalse(caught.exception.retryable)
                    raw.assert_awaited_once()
                    legacy.assert_not_awaited()
                    limited.close.assert_called_once()

    async def test_custom_raw_v2_4011_respects_disabled_fallback(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                limited = _wrapper(
                    ("error", {"code": 4011, "message": "rate limit"})
                )
                raw = AsyncMock(return_value=limited)
                legacy = AsyncMock()
                with (
                    self._patch_chain(raw, legacy),
                    self.assertRaises(UpstreamEventError) as caught,
                ):
                    await self._custom_request(
                        stream, _disable_upstream_fallback=True
                    )

                self.assertEqual(caught.exception.code, 4011)
                raw.assert_awaited_once()
                legacy.assert_not_awaited()
                limited.close.assert_called_once()

    async def test_3004_after_model_metadata_rotates_without_legacy_replay(self):
        for stream in (False, True):
            for error_payload in (
                {"code": 3004, "message": "busy"},
                {"error": {"errorCode": "3004", "message": "busy"}},
                {"code": "3004"},
            ):
                with self.subTest(stream=stream, error_payload=error_payload):
                    limited = _wrapper(
                        (
                            "model_config",
                            {
                                "model_name": "glm-5.2__dev",
                                "reply_to_message_id": "setup-turn",
                            },
                        ),
                        ("error", error_payload),
                    )
                    completed = _wrapper(
                        ("text_delta", {"delta": "pong"}),
                        ("done", {"finish_reason": "stop"}),
                    )
                    raw = AsyncMock(side_effect=[limited, completed])
                    legacy = AsyncMock()
                    with self._patch_chain(raw, legacy):
                        result = await self._request(stream)

                    self.assertIn("pong", result)
                    self.assertEqual(raw.await_count, 2)
                    self.assertEqual(
                        [call.kwargs["options"]["_account_id"] for call in raw.await_args_list],
                        ["account-a", "account-b"],
                    )
                    legacy.assert_not_awaited()
                    limited.close.assert_called_once()
                    completed.close.assert_called_once()

    async def test_4008_after_model_metadata_rotates_without_legacy_replay(self):
        for stream in (False, True):
            for error_payload in (
                {"code": 4008, "message": "Your requests have exceeded the quota."},
                {"error": {"errorCode": "4008", "message": "quota exceeded"}},
            ):
                with self.subTest(stream=stream, error_payload=error_payload):
                    limited = _wrapper(
                        (
                            "model_config",
                            {
                                "model_name": "glm-5.2__dev",
                                "reply_to_message_id": "setup-turn",
                            },
                        ),
                        ("error", error_payload),
                    )
                    completed = _wrapper(
                        ("text_delta", {"delta": "pong"}),
                        ("done", {"finish_reason": "stop"}),
                    )
                    raw = AsyncMock(side_effect=[limited, completed])
                    legacy = AsyncMock()
                    with self._patch_chain(raw, legacy):
                        result = await self._request(stream)

                    self.assertIn("pong", result)
                    self.assertEqual(raw.await_count, 2)
                    self.assertEqual(
                        [call.kwargs["options"]["_account_id"] for call in raw.await_args_list],
                        ["account-a", "account-b"],
                    )
                    legacy.assert_not_awaited()
                    limited.close.assert_called_once()
                    completed.close.assert_called_once()

    async def test_4008_without_polling_preserves_error_without_legacy_replay(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                limited = _wrapper(
                    ("error", {"code": 4008, "message": "quota exceeded"})
                )
                raw = AsyncMock(return_value=limited)
                legacy = AsyncMock()
                with self._patch_chain(raw, legacy, polling=False):
                    with self.assertRaises(UpstreamEventError) as error:
                        await self._request(stream)

                self.assertEqual(error.exception.code, 4008)
                self.assertTrue(error.exception.retryable)
                self.assertTrue(error.exception.account_quota_exhausted)
                raw.assert_awaited_once()
                legacy.assert_not_awaited()
                limited.close.assert_called_once()

    async def test_3004_without_polling_preserves_error_without_legacy_replay(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                limited = _wrapper(("error", {"code": 3004, "message": "busy"}))
                raw = AsyncMock(return_value=limited)
                legacy = AsyncMock()
                with self._patch_chain(raw, legacy, polling=False):
                    with self.assertRaises(UpstreamEventError) as error:
                        await self._request(stream)

                self.assertEqual(error.exception.code, 3004)
                self.assertTrue(error.exception.retryable)
                self.assertTrue(error.exception.account_quota_exhausted)
                raw.assert_awaited_once()
                legacy.assert_not_awaited()
                limited.close.assert_called_once()

    async def test_4023_does_not_rotate_or_replay(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                error = RuntimeError(
                    "Trae upstream error 4023: We're sorry, the model is unknown."
                )
                raw = AsyncMock(side_effect=error)
                legacy = AsyncMock()
                with self._patch_chain(raw, legacy):
                    with self.assertRaises(RuntimeError) as caught:
                        await self._request(stream)
                self.assertIs(caught.exception, error)
                raw.assert_awaited_once()
                legacy.assert_not_awaited()

    async def test_3004_after_model_output_or_usage_never_replays(self):
        for stream in (False, True):
            for event in (
                ("text_delta", {"delta": "partial"}),
                ("reasoning_delta", {"delta": "reasoning"}),
                (
                    "output",
                    {"response": "partial"},
                ),
                (
                    "output",
                    {
                        "tool_calls": [
                            {
                                "id": "call-read",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"x"}',
                                },
                            }
                        ]
                    },
                ),
                ("token_usage", {"input_token": 3, "output_token": 1}),
            ):
                with self.subTest(stream=stream, event=event[0]):
                    limited = _wrapper(
                        event, ("error", {"code": 3004, "message": "rate limit"})
                    )
                    raw = AsyncMock(return_value=limited)
                    legacy = AsyncMock()
                    with self._patch_chain(raw, legacy):
                        with self.assertRaises(UpstreamEventError) as error:
                            await self._request(stream)

                    self.assertFalse(error.exception.retryable)
                    self.assertTrue(error.exception.observed_model_event)
                    raw.assert_awaited_once()
                    legacy.assert_not_awaited()
                    limited.close.assert_called_once()

    async def test_hidden_reasoning_only_never_reopens_legacy_or_another_account(self):
        for stream in (False, True):
            for terminal in (False, True):
                with self.subTest(stream=stream, terminal=terminal):
                    events = [("reasoning_delta", {"delta": "private reasoning"})]
                    if terminal:
                        events.append(("done", {}))
                    response = _wrapper(*events)
                    raw = AsyncMock(return_value=response)
                    legacy = AsyncMock()
                    with (
                        self._patch_chain(raw, legacy),
                        patch.object(
                            main_module.responses_api,
                            "reasoning_output_requested",
                            return_value=False,
                        ),
                    ):
                        with self.assertRaises(EmptyUpstreamResponse) as caught:
                            await self._request(stream)
                    self.assertFalse(caught.exception.retryable)
                    self.assertTrue(caught.exception.observed_model_event)
                    raw.assert_awaited_once()
                    legacy.assert_not_awaited()
                    response.close.assert_called_once()


class HiddenReasoningReplayGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_reasoning_only_done_is_observed_but_not_visible(self):
        response = _ReplayResponse(
            [
                ("reasoning_delta", {"delta": "private reasoning"}),
                ("done", {}),
            ]
        )
        chunks = []
        with self.assertRaises(EmptyUpstreamResponse) as caught:
            async for chunk in sse.translate_ide_stream(
                response,
                "glm-5.2",
                fail_on_empty=True,
                require_terminal=True,
            ):
                chunks.append(chunk)
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.observed_model_event)
        self.assertIsNone(caught.exception.usage)
        self.assertNotIn("private reasoning", "".join(chunks))

    async def test_stream_reasoning_only_eof_is_not_safe_to_replay(self):
        response = _ReplayResponse(
            [("reasoning_delta", {"delta": "private reasoning"})]
        )
        chunks = []
        with self.assertRaises(IncompleteUpstreamResponse) as caught:
            async for chunk in sse.translate_ide_stream(
                response,
                "glm-5.2",
                fail_on_empty=True,
                require_terminal=True,
            ):
                chunks.append(chunk)
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.observed_model_event)
        self.assertNotIn("private reasoning", "".join(chunks))

    async def test_nonstream_reasoning_only_done_is_not_empty_retry(self):
        event_iter = [
            ("reasoning_delta", {"delta": "private reasoning"}),
            ("done", {}),
        ]
        with self.assertRaises(EmptyUpstreamResponse) as caught:
            await sse.collect_nonstream_ide(
                _ReplayResponse(event_iter),
                "glm-5.2",
                fail_on_empty=True,
                require_terminal=True,
            )
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.observed_model_event)

    async def test_nonstream_reasoning_only_eof_is_not_safe_to_replay(self):
        with self.assertRaises(IncompleteUpstreamResponse) as caught:
            await sse.collect_nonstream_ide(
                _ReplayResponse([("reasoning_delta", {"delta": "private reasoning"})]),
                "glm-5.2",
                fail_on_empty=True,
                require_terminal=True,
            )
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.observed_model_event)


if __name__ == "__main__":
    unittest.main()
