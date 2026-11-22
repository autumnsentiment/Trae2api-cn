import unittest
from unittest.mock import patch

from src import trae_remote_client
from src.sse import EmptyUpstreamResponse


class RemoteFirstEventErrorGuardTests(unittest.IsolatedAsyncioTestCase):
    async def _first_error(self, payload):
        async def upstream():
            yield "error", payload

        with patch.object(
            trae_remote_client,
            "_stream_events_unbounded",
            return_value=upstream(),
        ), self.assertRaises(
            (
                trae_remote_client.RemoteFirstEventError,
                trae_remote_client.RemoteModelNotEntitled,
            )
        ) as caught:
            _ = [
                event
                async for event in trae_remote_client.stream_events(
                    object(),
                    "token",
                    "session",
                    "message",
                )
            ]
        return caught.exception

    async def test_account_capacity_error_preserves_replay_metadata(self):
        cases = (
            ({"code": 429, "message": "busy"}, 429),
            ({"error": {"errorCode": "3004", "message": "busy"}}, "3004"),
            ({"message": "parallel limit reached"}, None),
        )
        for payload, expected_code in cases:
            with self.subTest(payload=payload):
                error = await self._first_error(payload)

                self.assertIsInstance(error, EmptyUpstreamResponse)
                self.assertEqual(error.code, expected_code)
                self.assertTrue(error.retryable)
                self.assertFalse(error.observed_model_event)
                self.assertIsNone(error.usage)
                self.assertTrue(error.account_quota_exhausted)

    async def test_error_attached_usage_blocks_replay(self):
        payload = {
            "code": 429,
            "message": "parallel limit",
            "usage": {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9},
        }
        for error_payload in (payload, {"error": payload}):
            with self.subTest(payload=error_payload):
                error = await self._first_error(error_payload)

                self.assertEqual(error.code, 429)
                self.assertFalse(error.retryable)
                self.assertTrue(error.observed_model_event)
                self.assertEqual(
                    error.usage,
                    {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                )
                self.assertTrue(error.account_quota_exhausted)

    async def test_other_setup_error_remains_fallback_eligible(self):
        error = await self._first_error(
            {"code": 2001, "message": "app config record not found"}
        )

        self.assertEqual(error.code, 2001)
        self.assertTrue(error.retryable)
        self.assertFalse(error.observed_model_event)
        self.assertIsNone(error.usage)
        self.assertFalse(error.account_quota_exhausted)

    async def test_model_access_error_is_not_replayed_on_same_account(self):
        error = await self._first_error(
            {"code": 1005, "message": "model access denied"}
        )

        self.assertIsInstance(error, trae_remote_client.RemoteModelNotEntitled)
        self.assertTrue(error.retryable)
        self.assertFalse(error.observed_model_event)
        self.assertIsNone(error.usage)
        self.assertFalse(error.account_quota_exhausted)
        self.assertEqual(error.upstream_code, 1005)
        self.assertIsNotNone(error.__cause__)

    async def test_string_model_access_error_is_terminal(self):
        error = await self._first_error(
            {"errorCode": "1005", "detail": "restricted model"}
        )

        self.assertIsInstance(error, trae_remote_client.RemoteModelNotEntitled)
        self.assertEqual(error.upstream_code, "1005")
        self.assertIsNone(error.usage)


if __name__ == "__main__":
    unittest.main()
