"""Remote billing enrichment must preserve streaming and usage contracts."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src import session_usage, sse


class SessionUsageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.closed = []
        self.usage = {
            "prompt_tokens": 300,
            "completion_tokens": 4,
            "total_tokens": 304,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": 2,
        }
        self.bill = {
            "prompt_tokens": 300,
            "cached_tokens": 240,
            "cache_write_tokens": 60,
        }

    async def events(self, usage=None):
        try:
            yield "plan_item", {"id": "answer", "content": "complete answer"}
            yield "tokenUsage", self.usage if usage is None else usage
            yield "done", {"stopReason": "stop"}
            raise AssertionError("enrichment consumed an event after done")
        finally:
            self.closed.append(True)

    async def collect(self, source=None):
        return [item async for item in session_usage.enrich_remote_events(
            source if source is not None else self.events(), "test-token", "turn-123"
        )]

    async def test_same_turn_matching_prompt_recovers_cache_without_double_count(self):
        fetch = AsyncMock(return_value=self.bill)
        with patch.object(session_usage.trae_client, "fetch_session_usage", fetch):
            result = await sse.collect_nonstream_web(
                session_usage.enrich_remote_events(self.events(), "test-token", "turn-123"),
                "m",
            )
        fetch.assert_awaited_once_with("turn-123", "test-token")
        self.assertEqual(result["choices"][0]["message"]["content"], "complete answer")
        self.assertEqual(result["usage"], {
            "prompt_tokens": 300, "completion_tokens": 4, "total_tokens": 304,
            "prompt_tokens_details": {"cached_tokens": 240, "cache_write_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 2},
        })
        self.assertEqual(self.closed, [True])

    async def test_existing_read_or_write_cache_does_not_query_billing(self):
        for field in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            with self.subTest(field=field):
                fetch = AsyncMock()
                with patch.object(session_usage.trae_client, "fetch_session_usage", fetch):
                    result = await self.collect(self.events({**self.usage, field: 12}))
                fetch.assert_not_awaited()
                self.assertEqual(len(result), 3)
                self.assertEqual(result[1][1][field], 12)

    async def test_mismatched_prompt_or_impossible_cache_is_not_recovered(self):
        for bill in (
            {**self.bill, "prompt_tokens": 301},
            {**self.bill, "cached_tokens": 301},
            {"cached_tokens": 240},
        ):
            with self.subTest(bill=bill):
                fetch = AsyncMock(return_value=bill)
                with patch.object(session_usage.trae_client, "fetch_session_usage", fetch), \
                     patch.object(session_usage.asyncio, "sleep", AsyncMock()):
                    result = await self.collect()
                self.assertEqual(len(result), 3)
                self.assertEqual(result[1][1], self.usage)

    async def test_missing_cache_never_creates_zero_cache_snapshot(self):
        for bill in ({"prompt_tokens": 300}, {"prompt_tokens": 300, "reasoning_tokens": 2}):
            with self.subTest(bill=bill):
                with patch.object(session_usage.trae_client, "fetch_session_usage", AsyncMock(return_value=bill)), \
                     patch.object(session_usage.asyncio, "sleep", AsyncMock()):
                    result = await self.collect()
                self.assertEqual(len(result), 3)
                self.assertEqual(result[1][1], self.usage)

    async def test_partial_cache_details_do_not_invent_the_other_counter(self):
        with patch.object(session_usage.trae_client, "fetch_session_usage",
                          AsyncMock(return_value={"prompt_tokens": 300, "cached_tokens": 240})):
            result = await self.collect()
        self.assertEqual(result[-2][1]["prompt_tokens_details"], {"cached_tokens": 240})

    async def test_settling_bill_can_retry_without_duplicate_recovery(self):
        fetch = AsyncMock(side_effect=[{}, self.bill])
        with patch.object(session_usage.trae_client, "fetch_session_usage", fetch), \
             patch.object(session_usage.asyncio, "sleep", AsyncMock()):
            result = await self.collect()
        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-2][0], "token_usage")
        self.assertEqual(result[-1][0], "done")

    async def test_body_is_yielded_before_billing_is_started(self):
        fetch = AsyncMock(return_value=self.bill)
        with patch.object(session_usage.trae_client, "fetch_session_usage", fetch):
            enriched = session_usage.enrich_remote_events(self.events(), "test-token", "turn-123")
            first = await anext(enriched)
            self.assertEqual(first[1]["content"], "complete answer")
            fetch.assert_not_awaited()
            await enriched.aclose()
        self.assertEqual(self.closed, [True])

    async def test_lookup_timeout_keeps_body_and_original_usage(self):
        async def pending(*args):
            await asyncio.Event().wait()
        fetch = AsyncMock(side_effect=pending)
        with patch.object(session_usage.trae_client, "fetch_session_usage", fetch), \
             patch.object(session_usage, "_lookup_timeout", return_value=0.01):
            result = await asyncio.wait_for(self.collect(), timeout=1)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][1]["content"], "complete answer")
        self.assertEqual(result[1][1], self.usage)
        self.assertEqual(result[-1][0], "done")
        self.assertEqual(self.closed, [True])

    async def test_lookup_exception_keeps_original_frames(self):
        with patch.object(session_usage.trae_client, "fetch_session_usage",
                          AsyncMock(side_effect=RuntimeError("unavailable"))):
            result = await self.collect()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1][1], self.usage)
        self.assertEqual(self.closed, [True])

    async def test_cancellation_propagates_and_closes_source(self):
        entered = asyncio.Event()
        async def pending(*args):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(session_usage.trae_client, "fetch_session_usage", AsyncMock(side_effect=pending)):
            task = asyncio.create_task(self.collect())
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.closed, [True])

    async def test_disabled_lookup_and_missing_identity_skip_billing(self):
        for token, turn, timeout in (("", "turn", 3), ("token", "", 3), ("token", "turn", 0)):
            with self.subTest(token=bool(token), turn=bool(turn), timeout=timeout):
                fetch = AsyncMock()
                with patch.object(session_usage.trae_client, "fetch_session_usage", fetch), \
                     patch.object(session_usage, "_lookup_timeout", return_value=timeout):
                    result = [item async for item in session_usage.enrich_remote_events(self.events(), token, turn)]
                fetch.assert_not_awaited()
                self.assertEqual(len(result), 3)

    async def test_streamed_final_usage_contains_recovery_once(self):
        with patch.object(session_usage.trae_client, "fetch_session_usage", AsyncMock(return_value=self.bill)):
            wire = [item async for item in sse.translate_web_events(
                session_usage.enrich_remote_events(self.events(), "test-token", "turn-123"), "m"
            )]
        payloads = [json.loads(line[6:]) for chunk in wire for line in chunk.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"]
        usages = [item["usage"] for item in payloads if item.get("usage")]
        self.assertEqual(len(usages), 1)
        self.assertEqual(usages[0]["prompt_tokens"], 300)
        self.assertEqual(usages[0]["prompt_tokens_details"]["cached_tokens"], 240)
        self.assertEqual(sum(chunk.count("data: [DONE]") for chunk in wire), 1)

    async def test_legacy_bill_requires_both_input_and_output_to_match(self):
        original = sse._map_usage(self.usage)
        for completion in (None, 5):
            with self.subTest(completion=completion):
                bill = dict(self.bill)
                if completion is not None:
                    bill["completion_tokens"] = completion
                with patch.object(session_usage.trae_client, "fetch_session_usage",
                                  AsyncMock(return_value=bill)), \
                     patch.object(session_usage.asyncio, "sleep", AsyncMock()):
                    result = await session_usage.enrich_usage(
                        original, "bound-token", "isolated-turn", match_completion=True
                    )
                self.assertIs(result, original)

    async def test_legacy_callback_captures_identity_and_binds_only_at_completion(self):
        wrapper = SimpleNamespace(auth_token="bound-token", usage_turn_id="isolated-turn")
        metadata = {}
        enrich = session_usage.legacy_usage_enricher(wrapper, metadata)
        self.assertEqual(metadata, {})
        wrapper.auth_token = "different-account-token"
        wrapper.usage_turn_id = "different-turn"
        fetch = AsyncMock(return_value={**self.bill, "completion_tokens": 4})
        original = sse._map_usage(self.usage)
        with patch.object(session_usage.trae_client, "fetch_session_usage", fetch):
            result = await enrich(original)
        fetch.assert_awaited_once_with("isolated-turn", "bound-token")
        self.assertEqual(metadata, {"usage_turn_id": "isolated-turn"})
        self.assertEqual(result["prompt_tokens_details"]["cached_tokens"], 240)
        self.assertEqual(original["prompt_tokens_details"]["cached_tokens"], 0)

    def test_legacy_callback_rejects_missing_adapter_billing_identity(self):
        for wrapper in (object(), SimpleNamespace(auth_token="token"),
                        SimpleNamespace(auth_token="", usage_turn_id="turn")):
            with self.subTest(wrapper=type(wrapper).__name__):
                self.assertIsNone(session_usage.legacy_usage_enricher(wrapper, {}))
