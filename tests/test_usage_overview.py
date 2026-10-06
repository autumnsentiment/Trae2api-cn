"""Dashboard overview: persistent daily token aggregation and /api/overview."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import main as main_module


class _UsageSandbox:
    """Point usage history and stats at a temp dir with empty state."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patches = [
            patch.object(main_module, "_USAGE_RECORDS_PATH", self.root / "usage_records.json"),
            patch.object(main_module, "_USAGE_STATS_PATH_OVERRIDE", ""),
            patch.object(main_module, "_USAGE_HISTORY", []),
            patch.object(main_module, "_USAGE_STATS", {"days": {}}),
        ]
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in reversed(self._patches):
            item.stop()
        self._tmp.cleanup()

    @property
    def stats_path(self) -> Path:
        return self.root / "usage_stats.json"


class UsageValuesTests(unittest.TestCase):
    def test_cached_tokens_read_from_flat_and_nested_shapes(self):
        flat = main_module._usage_values({"prompt_tokens": 10, "cache_read_tokens": 4})
        nested = main_module._usage_values(
            {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 6}}
        )
        responses = main_module._usage_values(
            {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 3}}
        )
        self.assertEqual(flat["cached_tokens"], 4)
        self.assertEqual(nested["cached_tokens"], 6)
        self.assertEqual(responses["cached_tokens"], 3)
        self.assertEqual(main_module._usage_values({})["cached_tokens"], 0)

    def test_explicit_empty_tools_does_not_restore_inherited_catalog(self):
        inherited = [
            {
                "type": "function",
                "function": {"name": "read_file"},
            }
        ]
        self.assertFalse(
            main_module._tools_requested_from_options(
                {
                    "tools": [],
                    "_inherited_tools": inherited,
                    "_tool_protocol_requested": True,
                }
            )
        )
        self.assertTrue(
            main_module._tools_requested_from_options(
                {
                    "_inherited_tools": inherited,
                    "_tool_protocol_requested": True,
                }
            )
        )

    def test_context_mode_defaults_to_standard_for_a_plain_request(self):
        with _UsageSandbox():
            tracker = main_module._UsageTracker(
                "glm-5.3",
                "/v1/chat/completions",
                False,
                {},
            )
            tracker.request_id = "standard-context-request"
            tracker.update({"prompt_tokens": 1, "completion_tokens": 1})
            asyncio.run(tracker.finish("completed"))
            record = main_module._USAGE_HISTORY[0]
            self.assertEqual(record["context_mode_requested"], "standard")
            self.assertEqual(record["context_mode_actual"], "standard")
            self.assertEqual(record["context_mode"], "standard")
            self.assertIs(record["max_mode_requested"], False)
            self.assertIs(record["max_mode_enabled"], False)

    def test_request_metadata_is_normalized_and_old_rows_stay_unknown(self):
        current = main_module._normalize_usage_record(
            {
                "account_id": "acct",
                "model": "glm-5.3",
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "reasoning_effort": "high",
                "reasoning_effort_applied": "extra_high",
                "context_mode_requested": "max_1m",
                "context_mode": "max_1m",
                "context_mode_actual": "max_1m",
                "context_window_tokens": 1_000_000,
                "max_mode_applied": True,
                "max_mode_requested": True,
                "max_mode_enabled": True,
                "tools_requested": True,
                "tool_used": False,
                "tool_calls_returned": False,
                "tool_called": False,
            }
        )
        self.assertEqual(current["reasoning_effort"], "high")
        self.assertEqual(current["reasoning_effort_applied"], "extra_high")
        self.assertEqual(current["context_mode_actual"], "max_1m")
        self.assertEqual(current["context_window_tokens"], 1_000_000)
        self.assertIs(current["tools_requested"], True)
        self.assertIs(current["tool_used"], False)
        self.assertIs(current["tool_calls_returned"], False)
        self.assertIs(current["max_mode_requested"], True)
        self.assertIs(current["max_mode_enabled"], True)
        self.assertIs(current["tool_called"], False)
        self.assertIs(current["max_mode"], True)
        self.assertIs(current["tool"], False)

        historical = main_module._normalize_usage_record(
            {"account_id": "old", "model": "m", "prompt_tokens": 1, "completion_tokens": 1}
        )
        for key in (
            "reasoning_effort",
            "reasoning_effort_applied",
            "context_mode",
            "context_mode_actual",
            "tools_requested",
            "tool_used",
        ):
            self.assertIsNone(historical[key], key)
        unknown_context = main_module._normalize_usage_record(
            {
                "account_id": "old",
                "model": "m",
                "context_mode_requested": "vendor_custom",
                "context_mode_actual": "vendor_custom",
            }
        )
        self.assertIsNone(unknown_context["max_mode_requested"])
        self.assertIsNone(unknown_context["max_mode_enabled"])

    def test_legacy_standard_request_exposes_explicit_non_max_state(self):
        record = main_module._normalize_usage_record(
            {
                "account_id": "old",
                "model": "m",
                "context_mode_requested": "standard",
                "tools_requested": False,
                "tool_used": False,
            }
        )
        self.assertEqual(record["context_mode"], "standard")
        self.assertEqual(record["context_mode_actual"], "standard")
        self.assertIs(record["max_mode_requested"], False)
        self.assertIs(record["max_mode_enabled"], False)
        self.assertIs(record["max_mode"], False)
        self.assertIs(record["tool"], False)


class DailyStatsTests(unittest.TestCase):
    def test_records_aggregate_into_today_and_persist(self):
        with _UsageSandbox() as box:
            main_module._record_usage(
                "a",
                "glm-5.3",
                100,
                20,
                cached_tokens=40,
                request_id="r1",
                context_mode_requested="max_1m",
                max_mode_requested=True,
                max_mode_enabled=True,
                tools_requested=True,
                tool_called=True,
            )
            main_module._record_usage("a", "kimi", 50, 5, status="error", request_id="r2")
            overview = main_module._usage_overview(7)
            today = overview["today"]
            self.assertEqual(today["requests"], 2)
            self.assertEqual(today["failed"], 1)
            self.assertEqual(today["completed"], 1)
            self.assertEqual(today["input_tokens"], 150)
            self.assertEqual(today["output_tokens"], 25)
            self.assertEqual(today["cached_tokens"], 40)
            self.assertEqual(today["total_tokens"], 175)
            self.assertEqual(today["max_mode_requests"], 1)
            self.assertEqual(today["max_mode_enabled_requests"], 1)
            self.assertEqual(today["tool_requests"], 1)
            self.assertEqual(today["tool_calls"], 1)
            self.assertEqual(overview["totals"]["total_tokens"], 175)
            self.assertEqual(len(overview["daily"]), 7)
            self.assertEqual({m["model"] for m in today["models"]}, {"glm-5.3", "kimi"})
            saved = json.loads(box.stats_path.read_text("utf-8"))
            self.assertIn(today["date"], saved["days"])

    def test_old_record_context_and_tool_aliases_feed_canonical_metadata(self):
        with _UsageSandbox():
            main_module._record_usage(
                "a",
                "glm-5.3",
                1,
                1,
                context_mode_requested="max_1m",
                max_mode_applied=True,
                tools_requested=True,
                tool_calls_returned=True,
                request_id="canonical-fallback",
            )
            record = main_module._USAGE_HISTORY[0]
            self.assertIs(record["max_mode_requested"], True)
            self.assertIs(record["max_mode_enabled"], True)
            self.assertIs(record["tool_called"], True)
            today = main_module._usage_overview(1)["today"]
            self.assertEqual(today["max_mode_requests"], 1)
            self.assertEqual(today["max_mode_enabled_requests"], 1)
            self.assertEqual(today["tool_requests"], 1)
            self.assertEqual(today["tool_calls"], 1)

    def test_tracker_persists_request_metadata_and_actual_tool_result(self):
        with _UsageSandbox():
            tracker = main_module._UsageTracker(
                "glm-5.3",
                "/v1/chat/completions",
                False,
                {
                    "reasoning_effort": "high",
                    "trae_max_mode": True,
                    "_upstream_trace": {
                        "max_mode_applied": True,
                        "max_context_tokens": 1_000_000,
                        "reasoning_effort_applied": "extra_high",
                    },
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "read_file"},
                        }
                    ],
                },
            )
            tracker.request_id = "metadata-request"
            tracker.update({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
            tracker.mark_tool_calls(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "read_file"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
            asyncio.run(tracker.finish("completed"))
            record = main_module._USAGE_HISTORY[0]
            self.assertEqual(record["reasoning_effort"], "high")
            self.assertEqual(record["reasoning_effort_applied"], "extra_high")
            self.assertEqual(record["context_mode"], "max_1m")
            self.assertEqual(record["context_mode_actual"], "max_1m")
            self.assertEqual(record["context_window_tokens"], 1_000_000)
            self.assertIs(record["tools_requested"], True)
            self.assertIs(record["tool_used"], True)
            self.assertIs(record["tool_calls_returned"], True)
            self.assertIs(record["max_mode_requested"], True)
            self.assertIs(record["max_mode_enabled"], True)
            self.assertIs(record["tool_called"], True)

    def test_requested_max_without_upstream_evidence_is_not_marked_applied(self):
        with _UsageSandbox():
            tracker = main_module._UsageTracker(
                "glm-5.3", "/v1/responses", False, {"trae_max_mode": True}
            )
            asyncio.run(tracker.finish("error"))
            record = main_module._USAGE_HISTORY[0]
            self.assertEqual(record["context_mode_requested"], "max_1m")
            self.assertIsNone(record["context_mode_actual"])
            self.assertIsNone(record["max_mode_applied"])
            self.assertIs(record["max_mode_requested"], True)
            self.assertIsNone(record["max_mode_enabled"])
            self.assertIsNone(record["tool_called"])
            self.assertIsNone(record["context_window_tokens"])

    def test_remote_fallback_keeps_shared_trace_for_actual_context_metadata(self):
        with _UsageSandbox():
            options = {"trae_max_mode": True}
            tracker = main_module._UsageTracker(
                "glm-5.3", "/v1/responses", False, options
            )
            fallback = main_module._remote_fallback_options(options, "ide")
            self.assertIs(fallback["_upstream_trace"], options["_upstream_trace"])
            fallback["_upstream_trace"].update(
                max_mode_applied=False, reasoning_effort_applied="medium"
            )
            asyncio.run(tracker.finish("completed"))
            record = main_module._USAGE_HISTORY[0]
            self.assertEqual(record["context_mode_requested"], "max_1m")
            self.assertEqual(record["context_mode_actual"], "standard")
            self.assertIs(record["max_mode_applied"], False)
            self.assertEqual(record["reasoning_effort_applied"], "medium")

    def test_tracker_marks_offered_but_unused_tools_as_false_on_completion(self):
        with _UsageSandbox():
            tracker = main_module._UsageTracker(
                "glm-5.3",
                "/v1/chat/completions",
                False,
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "read_file"},
                        }
                    ]
                },
            )
            tracker.request_id = "unused-tools-request"
            tracker.update({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
            asyncio.run(tracker.finish("completed"))
            record = main_module._USAGE_HISTORY[0]
            self.assertIs(record["tools_requested"], True)
            self.assertIs(record["tool_used"], False)
            self.assertIs(record["tool_calls_returned"], False)
            self.assertIs(record["tool_called"], False)

    def test_stream_tool_delta_marks_actual_tool_call(self):
        class FakeTracker:
            def __init__(self):
                self.marked = []
                self.usage = []

            def mark_tool_calls(self, value):
                self.marked.append(value)

            def update(self, value):
                self.usage.append(value)

        tracker = FakeTracker()
        token = main_module._USAGE_TRACKER.set(tracker)
        try:
            main_module._track_usage_from_chunk(
                (
                    'data: {"choices":[{"delta":{"tool_calls":['
                    '{"id":"call_1","type":"function","function":{"name":"read_file"}}'
                    ']}}]}\n\n'
                ),
                "glm-5.3",
            )
        finally:
            main_module._USAGE_TRACKER.reset(token)
        self.assertEqual(len(tracker.marked), 1)
        self.assertEqual(tracker.usage, [])

    def test_rewriting_same_request_does_not_double_count(self):
        with _UsageSandbox():
            main_module._record_usage("a", "glm-5.3", 10, 2, request_id="same")
            main_module._record_usage("a", "glm-5.3", 30, 6, request_id="same")
            main_module._update_usage_record("same", credits_consumed=1.5)
            today = main_module._usage_overview(1)["today"]
            self.assertEqual(today["requests"], 1)
            self.assertEqual(today["input_tokens"], 30)
            self.assertEqual(today["output_tokens"], 6)
            self.assertEqual(today["credits"], 1.5)

    def test_totals_survive_history_cap(self):
        with _UsageSandbox():
            for index in range(main_module._USAGE_MAX_HISTORY + 20):
                main_module._record_usage("a", "m", 1, 1, request_id=f"r{index}")
            self.assertEqual(len(main_module._USAGE_HISTORY), main_module._USAGE_MAX_HISTORY)
            totals = main_module._usage_overview(1)["totals"]
            self.assertEqual(totals["requests"], main_module._USAGE_MAX_HISTORY + 20)

    def test_load_seeds_stats_from_existing_history(self):
        with _UsageSandbox() as box:
            yesterday = time.time() - 86400
            main_module._USAGE_HISTORY.extend(
                [
                    main_module._normalize_usage_record(
                        {"model": "m", "prompt_tokens": 7, "completion_tokens": 3, "timestamp": yesterday}
                    )
                ]
            )
            main_module._load_usage_stats()
            daily = main_module._usage_overview(3)["daily"]
            self.assertEqual(daily[1]["total_tokens"], 10)
            self.assertTrue(box.stats_path.exists())


class OverviewEndpointTests(unittest.TestCase):
    def test_usage_records_endpoint_exposes_request_capability_metadata(self):
        with _UsageSandbox():
            main_module._record_usage(
                "acct",
                "glm-5.3",
                7,
                5,
                request_id="api-metadata",
                reasoning_effort="high",
                reasoning_effort_applied="extra_high",
                context_mode_requested="max_1m",
                context_mode_actual="max_1m",
                context_window_tokens=1_000_000,
                max_mode_requested=True,
                max_mode_enabled=True,
                tools_requested=True,
                tool_used=True,
                tool_called=True,
                tool_calls_returned=True,
            )
            body = TestClient(main_module.app).get("/api/usage/records")
            self.assertEqual(body.status_code, 200)
            record = next(
                item for item in body.json() if item["request_id"] == "api-metadata"
            )
            expected = {
                "reasoning_effort": "high",
                "requested_reasoning_effort": "high",
                "reasoning_effort_applied": "extra_high",
                "context_mode_requested": "max_1m",
                "context_mode": "max_1m",
                "context_mode_actual": "max_1m",
                "context_window_tokens": 1_000_000,
                "max_mode_applied": True,
                "max_mode_requested": True,
                "max_mode_enabled": True,
                "tools_requested": True,
                "tool_used": True,
                "tool_called": True,
                "tool_calls_returned": True,
            }
            for key, value in expected.items():
                with self.subTest(key=key):
                    self.assertEqual(record[key], value)

    def test_usage_records_endpoint_normalizes_legacy_rows_with_unknown_metadata(self):
        with _UsageSandbox():
            main_module._USAGE_HISTORY.append(
                {
                    "request_id": "legacy",
                    "model": "glm-5.3",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "timestamp": time.time(),
                }
            )
            body = TestClient(main_module.app).get("/api/usage/records")
            self.assertEqual(body.status_code, 200)
            record = body.json()[0]
            self.assertEqual(record["request_id"], "legacy")
            for key in (
                "reasoning_effort",
                "context_mode",
                "context_mode_actual",
                "tools_requested",
                "tool_used",
                "tool_called",
            ):
                with self.subTest(key=key):
                    self.assertIsNone(record[key])

    def test_overview_endpoint_and_home_page(self):
        with _UsageSandbox():
            main_module._record_usage("a", "glm-5.3", 12, 3, cached_tokens=2, request_id="x")
            client = TestClient(main_module.app)
            body = client.get("/api/overview?days=14").json()
            self.assertTrue(body["success"])
            self.assertEqual(len(body["usage"]["daily"]), 14)
            self.assertEqual(body["usage"]["today"]["cached_tokens"], 2)
            for key in ("total", "valid", "checked_in", "credits_remaining"):
                self.assertIn(key, body["accounts"])
            for key in ("version", "uptime_seconds", "upstream_mode", "in_flight"):
                self.assertIn(key, body["service"])
            html = client.get("/web/login").text
        self.assertIn('data-page="overview"', html)
        self.assertIn('id="ov-chart"', html)
        self.assertLess(html.index('data-tab="overview"'), html.index('data-tab="accounts"'))


if __name__ == "__main__":
    unittest.main()
