import asyncio
import base64
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

FAKE_CMD = str(Path(__file__).resolve().parent / "fake" / "fake_cli.cmd")
WORKDIR = tempfile.mkdtemp(prefix="trae-relay-test-")

os.environ["TRAE_AUTH_SOURCE"] = "cli"
os.environ["UPSTREAM_MODE"] = "cli"
os.environ["TRAE_CLI_COMMAND"] = FAKE_CMD
os.environ["TRAE_CLI_WORKDIR"] = WORKDIR
os.environ["TRAE_CLI_PROMPT_MODE"] = "stdin"
os.environ["TRAE_CLI_OUTPUT_MODE"] = "json"
os.environ["TRAE_CLI_DISABLE_TOOLS"] = "false"
os.environ["RELAY_API_KEYS"] = "smoke-key"

from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse, StreamingResponse

from src import main as main_module
from src.main import app, _checkin_claim_error, _checkin_claim_ok
from src.cli_client import CliEvent

AUTH_HEADERS = {"Authorization": "Bearer smoke-key"}




class CheckinResultTests(unittest.TestCase):
    def setUp(self):
        main_module._CHECKIN_ACCOUNT_LOCKS.clear()
        main_module._CHECKIN_CLAIM_GATE = None
        main_module._CHECKIN_CLAIM_GATE_LOOP = None
        main_module._CHECKIN_NEXT_CLAIM_AT = 0.0
        main_module._CHECKIN_COOLDOWN_UNTIL.clear()
        main_module._CHECKIN_ACCEPTED_UNTIL.clear()

    def test_only_zero_business_code_is_success(self):
        self.assertTrue(_checkin_claim_ok({"code": 0, "message": "success"}))
        self.assertFalse(_checkin_claim_ok({"code": 9074, "message": "too frequent"}))
        self.assertFalse(_checkin_claim_ok({"code": 9095, "message": "device already checked"}))
        self.assertFalse(_checkin_claim_ok({}))

    def test_business_error_keeps_upstream_code_and_message(self):
        message = _checkin_claim_error({"code": 9074, "message": "too frequent"})
        self.assertIn("[9074]", message)
        self.assertIn("too frequent", message)

    def test_status_9074_starts_cooldown_and_suppresses_second_probe(self):
        status = AsyncMock(return_value={"code": 9074, "message": "too frequent"})
        record = {
            "token": "test-token",
            "checkin": {"checked_in": False, "credits": 200},
        }
        with (
            patch("src.main.auth.get_account_record", return_value=record),
            patch("src.main.trae_client.fetch_checkin_credits_status", new=status),
            patch("src.main.CHECKIN_RETRY_AFTER", 60),
            patch("src.main.time.monotonic", return_value=100.0),
        ):
            first = asyncio.run(main_module.api_checkin_account_status("account-1"))
            second = asyncio.run(main_module.api_checkin_account_status("account-1"))

        first_body = json.loads(first.body)
        second_body = json.loads(second.body)
        self.assertFalse(first_body["success"])
        self.assertEqual(first_body["data"]["code"], 9074)
        self.assertFalse(second_body["success"])
        self.assertTrue(second_body["stale"])
        self.assertIn("9074", second_body["error"])
        self.assertEqual(status.await_count, 1)

    def test_checked_in_status_retires_stale_9074_backoff(self):
        """A leftover backoff would block tomorrow's first claim."""

        record = {
            "token": "test-token",
            "checkin": {
                "checked_in": False,
                "retry_backoff": 3600.0,
                "retry_9074_count": 22,
            },
        }
        cleared = []
        with (
            patch("src.main.auth.get_account_record", return_value=record),
            patch("src.main.auth.merge_account_checkin", return_value={}),
            patch(
                "src.main.trae_client.fetch_checkin_credits_status",
                new=AsyncMock(return_value={"code": 0, "enable": True, "checked_in": True}),
            ),
            patch(
                "src.main._checkin_clear_retry_state",
                side_effect=lambda aid: cleared.append(aid),
            ),
            patch.dict(
                main_module._CHECKIN_COOLDOWN_UNTIL,
                {"account-1": time.monotonic() + 3600},
                clear=True,
            ),
        ):
            row = asyncio.run(
                main_module._fetch_checkin_status_snapshot(
                    "account-1", record, use_cached_on_cooldown=True
                )
            )

        self.assertIs(row["checked_in"], True)
        self.assertEqual(cleared, ["account-1"])
        self.assertNotIn("account-1", main_module._CHECKIN_COOLDOWN_UNTIL)

    def test_claim_cooldown_does_not_block_status_read(self):
        """A claim-side 9074 must not freeze status on a stale checked_in."""

        status = AsyncMock(
            return_value={
                "code": 0,
                "enable": True,
                "checked_in": False,
                "credits": 200,
            }
        )
        record = {
            "token": "test-token",
            "checkin": {"checked_in": False, "credits": 200},
        }
        with (
            patch("src.main.auth.get_account_record", return_value=record),
            patch("src.main.auth.merge_account_checkin", return_value={}),
            patch("src.main.trae_client.fetch_checkin_credits_status", new=status),
            patch("src.main.CHECKIN_RETRY_AFTER", 60),
            patch.dict(
                main_module._CHECKIN_COOLDOWN_UNTIL,
                {"account-1": time.monotonic() + 600},
                clear=True,
            ),
            patch.dict(
                main_module._CHECKIN_STATUS_COOLDOWN_UNTIL, {}, clear=True
            ),
        ):
            row = asyncio.run(
                main_module._fetch_checkin_status_snapshot(
                    "account-1", record, use_cached_on_cooldown=True
                )
            )

        # The status endpoint was still queried and its live value returned.
        self.assertEqual(status.await_count, 1)
        self.assertIs(row["checked_in"], False)
        self.assertEqual(row["credits"], 200)
        self.assertNotIn("stale", row)
        # The pending claim window stays visible to the caller.
        self.assertTrue(row["claim_rate_limited"])
        self.assertGreater(row["retry_after_seconds"], 0)

    def test_checkin_cache_uses_china_business_day_and_status_timestamp(self):
        china = timezone(timedelta(hours=8))
        now = datetime(2026, 8, 20, 0, 10, tzinfo=china).timestamp()
        previous_day = datetime(2026, 8, 19, 23, 50, tzinfo=china).timestamp()
        record = {
            "checkin_status_updated_at": previous_day,
            "checkin_updated_at": now,
            "credits_updated_at": now,
        }

        with patch("src.main.time.time", return_value=now):
            self.assertFalse(main_module._checkin_cache_is_today(record))

        legacy_record = {
            "checkin_updated_at": now,
            "credits_updated_at": now,
        }
        with patch("src.main.time.time", return_value=now):
            self.assertFalse(main_module._checkin_cache_is_today(legacy_record))

    def test_bulk_credit_refresh_reports_when_both_upstreams_fail(self):
        raw_accounts = [("account-1", {"token": "test-token"})]
        with (
            patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
            patch("src.main.auth.get_active_account_id", return_value="account-1"),
            patch(
                "src.main.trae_client.fetch_account_credits",
                new=AsyncMock(side_effect=RuntimeError("general failed")),
            ),
            patch(
                "src.main.trae_client.fetch_account_total_credits",
                new=AsyncMock(side_effect=RuntimeError("total failed")),
            ),
        ):
            response = asyncio.run(main_module.api_checkin_credits_accounts())

        body = json.loads(response.body)
        self.assertTrue(body["success"])
        self.assertIn("credits query failed", body["accounts"][0]["error"])

    def test_bulk_credits_static_route_is_not_captured_as_account_id(self):
        with patch("src.main.auth.get_accounts_raw", return_value=[]):
            with TestClient(app) as client:
                response = client.get(
                    "/api/checkin/credits/accounts", headers=AUTH_HEADERS
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "accounts": []})

    def test_single_account_endpoint_does_not_report_rate_limit_as_success(self):
        claim = AsyncMock(return_value={"code": 9074, "message": "too frequent"})
        with (
            patch(
                "src.main.auth.get_account_record",
                return_value={"token": "test-token"},
            ),
            patch(
                "src.main.trae_client.claim_checkin_credits",
                new=claim,
            ),
            patch(
                "src.main.trae_client.fetch_checkin_credits_status",
                new=AsyncMock(return_value={"checked_in": False, "credits": 200}),
            ),
            patch(
                "src.main.trae_client.fetch_account_credits",
                new=AsyncMock(side_effect=RuntimeError("skip credits")),
            ),
            patch(
                "src.main.trae_client.fetch_account_work_credits",
                new=AsyncMock(side_effect=RuntimeError("skip work credits")),
            ),
            patch("src.main.auth.set_account_checkin"),
            # A persistent 9074 must still surface; rotation is covered
            # separately.
            patch("src.main._checkin_device_rotation_enabled", return_value=False),
        ):
            response = asyncio.run(main_module.api_checkin_account("account-1"))

        body = json.loads(response.body)
        self.assertFalse(body["success"])
        self.assertEqual(body["data"]["code"], 9074)
        self.assertEqual(body["data"]["message"], "too frequent")
        self.assertIn("9074", body["error"])
        self.assertTrue(body["retryable"])
        self.assertEqual(body["retry_after_seconds"], 60)
        self.assertEqual(claim.await_count, 1)

    def test_single_account_endpoint_skips_claim_when_already_checked(self):
        claim = AsyncMock(side_effect=AssertionError("claim must be skipped"))
        with (
            patch(
                "src.main.auth.get_account_record",
                return_value={
                    "token": "test-token",
                    "user_id": "user-1",
                    "checkin": {"checked_in": True},
                    "checkin_status_updated_at": __import__("time").time(),
                    "checkin_updated_at": __import__("time").time(),
                },
            ),
            patch(
                "src.main.trae_client.fetch_checkin_credits_status",
                new=AsyncMock(return_value={"checked_in": True, "credits": 200}),
            ),
            patch(
                "src.main._fetch_full_credits",
                new=AsyncMock(
                    return_value={
                        "account_credits": {"remaining": 10},
                        "work_credits": {"remaining": 20},
                        "total_credits": {"remaining": 30},
                    }
                ),
            ),
            patch("src.main.auth.set_account_checkin"),
            patch("src.main.trae_client.claim_checkin_credits", new=claim),
        ):
            response = asyncio.run(main_module.api_checkin_account("account-1"))

        body = json.loads(response.body)
        self.assertTrue(body["success"])
        self.assertTrue(body["skipped"])
        self.assertTrue(body["checked_in"])
        self.assertEqual(claim.await_count, 0)

    def test_single_account_status_endpoint_only_refreshes_requested_account(self):
        status = AsyncMock(return_value={"checked_in": True, "credits": 200})
        full = AsyncMock(
            return_value={
                "account_credits": {"remaining": 10},
                "work_credits": {"remaining": 20},
                "total_credits": {"remaining": 30},
            }
        )
        with (
            patch(
                "src.main.auth.get_account_record",
                return_value={"token": "only-this-token", "user_id": "user-1"},
            ),
            patch(
                "src.main.trae_client.fetch_checkin_credits_status", new=status
            ),
            patch("src.main._fetch_full_credits", new=full),
            patch(
                "src.main.auth.merge_account_checkin",
                side_effect=lambda _aid, data: dict(data),
            ) as save,
        ):
            response = asyncio.run(
                main_module.api_checkin_account_status("account-1")
            )

        body = json.loads(response.body)
        self.assertTrue(body["success"])
        self.assertEqual(body["id"], "account-1")
        status.assert_awaited_once_with("only-this-token", "account-1")
        full.assert_not_awaited()
        save.assert_called_once()

    def test_single_account_endpoint_keeps_code_zero_success_when_status_lags(self):
        with (
            patch(
                "src.main.auth.get_account_record",
                return_value={"token": "test-token"},
            ),
            patch(
                "src.main.trae_client.claim_checkin_credits",
                new=AsyncMock(return_value={"code": 0, "message": "success"}),
            ),
            patch(
                "src.main.trae_client.fetch_checkin_credits_status",
                new=AsyncMock(return_value={"checked_in": False, "credits": 200}),
            ),
            patch(
                "src.main.trae_client.fetch_account_credits",
                new=AsyncMock(side_effect=RuntimeError("skip credits")),
            ),
            patch(
                "src.main.trae_client.fetch_account_work_credits",
                new=AsyncMock(side_effect=RuntimeError("skip work credits")),
            ),
            patch("src.main.auth.set_account_checkin"),
            patch("src.main._CHECKIN_VERIFY_DELAYS", ()),
        ):
            response = asyncio.run(main_module.api_checkin_account("account-1"))

        body = json.loads(response.body)
        self.assertTrue(body["success"])
        self.assertTrue(body["checked_in"])
        self.assertTrue(body["verification_pending"])

    def test_claim_all_uses_cached_state_without_status_or_credit_probes(self):
        raw_accounts = [
            (f"acct-{index}", {"token": f"token-{index}", "user_id": f"user-{index}"})
            for index in range(3)
        ]
        records = dict(raw_accounts)

        async def scenario():
            async def status(token, _account_id):
                raise AssertionError(f"status probe must not run: {token}")

            async def full(token):
                raise AssertionError(f"credit probe must not run: {token}")

            with (
                patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
                patch("src.main.auth.get_account_record", side_effect=lambda aid: dict(records.get(aid, {}))),
                patch("src.main.auth.get_active_account_id", return_value="acct-0"),
                patch("src.main.trae_client.fetch_checkin_credits_status", new=status),
                patch("src.main._fetch_full_credits", new=full),
                patch("src.main.auth.set_account_checkin"),
                 patch(
                     "src.main.trae_client.claim_checkin_credits",
                     new=AsyncMock(return_value={"code": 0, "message": "success"}),
                 ),
                patch("src.main.CHECKIN_INTERVAL", 0),
            ):
                response = await main_module.api_checkin_claim_all()
            return response

        response = asyncio.run(scenario())
        body = json.loads(response.body)
        self.assertTrue(body["success"])
        self.assertEqual(
            [row["id"] for row in body["accounts"]],
            [item[0] for item in raw_accounts],
        )
        self.assertTrue(all(row["success"] and not row["skipped"] for row in body["accounts"]))

    def test_claim_all_claims_unchecked_in_order_with_interval(self):
        raw_accounts = [
            (f"acct-{index}", {"token": f"token-{index}", "user_id": f"user-{index}"})
            for index in range(3)
        ]
        records = dict(raw_accounts)

        async def scenario():
            status_calls = {}
            claim_order = []
            sleep_calls = []
            clock = [100.0]

            claimed = set()

            async def status(token, _account_id):
                status_calls[token] = status_calls.get(token, 0) + 1
                return {
                    "checked_in": token in claimed,
                    "credits": 100,
                }

            async def full(_token):
                return {
                    "account_credits": {"remaining": 90},
                    "work_credits": {"remaining": 10},
                    "total_credits": {"remaining": 100},
                }

            async def claim(token, account_id):
                claim_order.append((token, account_id))
                claimed.add(token)
                return {"code": 0, "message": "success"}

            async def fake_sleep(delay):
                sleep_calls.append(delay)
                clock[0] += delay

            with (
                patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
                patch("src.main.auth.get_account_record", side_effect=lambda aid: dict(records.get(aid, {}))),
                patch("src.main.auth.get_active_account_id", return_value="acct-0"),
                patch("src.main.trae_client.fetch_checkin_credits_status", new=status),
                patch("src.main._fetch_full_credits", new=full),
                patch("src.main.trae_client.claim_checkin_credits", new=claim),
                patch("src.main.auth.set_account_checkin"),
                patch("src.main.CHECKIN_INTERVAL", 7),
                patch("src.main.time.monotonic", side_effect=lambda: clock[0]),
                patch("src.main.asyncio.sleep", new=fake_sleep),
            ):
                response = await main_module.api_checkin_claim_all()
            return response, claim_order, sleep_calls, status_calls

        response, claim_order, sleep_calls, status_calls = asyncio.run(scenario())
        body = json.loads(response.body)
        self.assertTrue(body["success"])
        self.assertEqual(
            claim_order,
            [("token-0", "acct-0"), ("token-1", "acct-1"), ("token-2", "acct-2")],
        )
        self.assertEqual(sleep_calls, [7, 7])
        self.assertEqual(status_calls, {})
        self.assertTrue(
            all(row["success"] and not row["skipped"] for row in body["accounts"])
        )

    def test_9074_cooldown_suppresses_repeated_upstream_claim(self):
        record = {"token": "test-token", "user_id": "user-1"}

        async def scenario():
            clock = [100.0]
            claim = AsyncMock(
                side_effect=[
                    {"code": 9074, "message": "too frequent"},
                    {"code": 0, "message": "success"},
                ]
            )
            with (
                patch("src.main.auth.get_account_record", return_value=dict(record)),
                patch(
                    "src.main.trae_client.fetch_checkin_credits_status",
                    new=AsyncMock(return_value={"checked_in": False, "credits": 200}),
                ),
                patch(
                    "src.main._fetch_full_credits",
                    new=AsyncMock(return_value={"account_credits": {"remaining": 10}}),
                ),
                patch("src.main.auth.set_account_checkin"),
                patch("src.main.trae_client.claim_checkin_credits", new=claim),
                patch("src.main.CHECKIN_INTERVAL", 0),
                patch("src.main.CHECKIN_RETRY_AFTER", 60),
                patch("src.main._CHECKIN_VERIFY_DELAYS", ()),
                patch("src.main.time.monotonic", side_effect=lambda: clock[0]),
                # Device rotation is covered separately; keep this focused on
                # the cooldown gate.
                patch("src.main._checkin_device_rotation_enabled", return_value=False),
            ):
                first = await main_module.api_checkin_account("account-1")
                second = await main_module.api_checkin_account("account-1")
                clock[0] = 161.0
                third = await main_module.api_checkin_account("account-1")
            return [json.loads(item.body) for item in (first, second, third)], claim

        bodies, claim = asyncio.run(scenario())
        self.assertFalse(bodies[0]["success"])
        self.assertTrue(bodies[0]["retryable"])
        self.assertFalse(bodies[1]["success"])
        self.assertFalse(bodies[1]["claim_sent"])
        self.assertTrue(bodies[1]["skipped"])
        self.assertTrue(bodies[2]["success"])
        self.assertEqual(claim.await_count, 2)

    def test_9074_rotates_device_id_and_retries_once(self):
        """9074 is device-scoped: a fresh id lets the same account claim."""

        claim = AsyncMock(
            side_effect=[
                {"code": 9074, "message": "too frequent"},
                {"code": 0, "message": "success"},
            ]
        )
        rotate = MagicMock(return_value="9999888877776666")
        with (
            patch(
                "src.main.auth.get_account_record",
                return_value={"token": "test-token"},
            ),
            patch("src.main.trae_client.claim_checkin_credits", new=claim),
            patch("src.main.trae_client.rotate_checkin_device_id", new=rotate),
            patch(
                "src.main.trae_client.fetch_checkin_credits_status",
                new=AsyncMock(return_value={"checked_in": False, "credits": 200}),
            ),
            patch(
                "src.main._fetch_full_credits",
                new=AsyncMock(return_value={"account_credits": {"remaining": 10}}),
            ),
            patch("src.main.auth.set_account_checkin"),
            patch("src.main.CHECKIN_INTERVAL", 0),
            patch.dict(main_module._CHECKIN_COOLDOWN_UNTIL, {}, clear=True),
        ):
            data, retry_after = asyncio.run(
                main_module._claim_checkin_throttled("account-1", "test-token")
            )

        self.assertEqual(rotate.call_count, 1)
        self.assertEqual(claim.await_count, 2)
        self.assertEqual(data.get("code"), 0)
        self.assertEqual(retry_after, 0)

    def test_persistent_9074_still_falls_back_to_cooldown(self):
        claim = AsyncMock(return_value={"code": 9074, "message": "too frequent"})
        rotate = MagicMock(return_value="9999888877776666")
        with (
            patch(
                "src.main.auth.get_account_record",
                return_value={"token": "test-token"},
            ),
            patch("src.main.trae_client.claim_checkin_credits", new=claim),
            patch("src.main.trae_client.rotate_checkin_device_id", new=rotate),
            patch("src.main.CHECKIN_INTERVAL", 0),
            patch("src.main.CHECKIN_RETRY_AFTER", 60),
            patch.dict(main_module._CHECKIN_COOLDOWN_UNTIL, {}, clear=True),
        ):
            data, retry_after = asyncio.run(
                main_module._claim_checkin_throttled("account-1", "test-token")
            )

        self.assertEqual(claim.await_count, 2)
        self.assertEqual(data.get("code"), 9074)
        self.assertGreater(retry_after, 0)

    def test_9074_backoff_escalates_after_repeated_failures(self):
        def scenario():
            record = {"token": "test-token", "user_id": "user-1"}
            with (
                patch("src.main.auth.get_account_record", return_value=record),
                patch("src.main.CHECKIN_RETRY_AFTER", 60),
                patch("src.main.CHECKIN_9074_MAX_BACKOFF", 3600),
            ):
                first = main_module._checkin_next_backoff("account-1")
                second = main_module._checkin_next_backoff("account-1")
            return first, second

        # With no persisted streak both settle on the base window.
        first, second = scenario()
        self.assertEqual(first, 60)
        self.assertEqual(second, 60)

        def with_streak(count):
            record = {
                "token": "test-token",
                "user_id": "user-1",
                "checkin": {"retry_9074_count": count},
            }
            with (
                patch("src.main.auth.get_account_record", return_value=record),
                patch("src.main.CHECKIN_RETRY_AFTER", 60),
                patch("src.main.CHECKIN_9074_MAX_BACKOFF", 3600),
            ):
                return main_module._checkin_next_backoff("account-1")

        self.assertEqual(with_streak(0), 60)
        self.assertEqual(with_streak(1), 120)
        self.assertEqual(with_streak(2), 240)
        # Doubling stops at the exponent cap so a long streak keeps retrying
        # within the day instead of parking on the hour-long ceiling.
        self.assertEqual(with_streak(3), 480)
        self.assertEqual(with_streak(6), 480)
        self.assertEqual(with_streak(24), 480)

        # A larger max window is still respected when the cap allows it.
        record = {
            "token": "test-token",
            "checkin": {"retry_9074_count": 24},
        }
        with (
            patch("src.main.auth.get_account_record", return_value=record),
            patch("src.main.CHECKIN_RETRY_AFTER", 60),
            patch("src.main.CHECKIN_9074_MAX_BACKOFF", 120),
            patch("src.main.CHECKIN_9074_BACKOFF_EXPONENT_CAP", 10),
        ):
            self.assertEqual(main_module._checkin_next_backoff("account-1"), 120)

    def test_9074_cooldown_honors_persisted_wall_clock_deadline(self):
        now = 1_000_000.0
        record = {
            "token": "test-token",
            "checkin": {
                "retry_backoff": 300,
                "retry_updated_at": now,
            },
        }
        with (
            patch("src.main.auth.get_account_record", return_value=record),
            patch("src.main.time.monotonic", return_value=now),
            patch("src.main.time.time", return_value=now + 120),
        ):
            remaining = main_module._checkin_cooldown_remaining("account-1")
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 180)

        with (
            patch("src.main.auth.get_account_record", return_value=record),
            patch("src.main.time.monotonic", return_value=now),
            patch("src.main.time.time", return_value=now + 400),
        ):
            remaining = main_module._checkin_cooldown_remaining("account-1")
        self.assertEqual(remaining, 0)

    def test_start_cooldown_persists_escalating_backoff(self):
        state = {
            "record": {
                "token": "test-token",
                "user_id": "user-1",
                "checkin": {},
            }
        }

        def fake_merge(account_id, data):
            state["record"]["checkin"].update(data)
            return dict(state["record"]["checkin"])

        with (
            patch(
                "src.main.auth.get_account_record",
                side_effect=lambda _: dict(state["record"]),
            ),
            patch("src.main.auth.merge_account_retry", side_effect=fake_merge),
            patch("src.main.CHECKIN_RETRY_AFTER", 60),
            patch("src.main.CHECKIN_9074_MAX_BACKOFF", 3600),
            patch("src.main.time.monotonic", return_value=100.0),
            patch("src.main.time.time", return_value=1_000_000.0),
        ):
            first = main_module._checkin_start_cooldown("account-1")
            second = main_module._checkin_start_cooldown("account-1")

        self.assertEqual(first, 60)
        self.assertEqual(second, 120)
        self.assertEqual(state["record"]["checkin"].get("retry_9074_count"), 2)
        self.assertEqual(state["record"]["checkin"].get("retry_backoff"), 120)

    def test_auto_retry_loop_retries_only_persisted_backoff_accounts(self):
        record = {
            "token": "test-token",
            "user_id": "user-1",
            "checkin": {"retry_backoff": 60, "retry_9074_count": 1},
        }
        raw_accounts = [("account-1", record), ("plain-account", {"token": "tok"})]
        claimed = []

        async def fake_claim(account_id):
            claimed.append(account_id)
            return {
                "success": True,
                "skipped": False,
                "claim_sent": True,
                "checked_in": True,
            }

        async def cycle():
            with (
                patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
                patch(
                    "src.main._claim_checkin_account",
                    side_effect=fake_claim,
                ),
                patch("src.main._checkin_cooldown_remaining", return_value=0),
                patch("src.main._checkin_cache_is_today", return_value=False),
                patch("src.main._checkin_clear_retry_state"),
            ):
                await main_module._checkin_auto_retry_cycle()

        asyncio.run(cycle())
        self.assertIn("account-1", claimed)
        self.assertNotIn("plain-account", claimed)

    def test_auto_retry_still_sees_account_after_backoff_is_cleared(self):
        """Clearing the backoff must not hide a still-unclaimed 9074 account."""

        rotated = {
            "token": "test-token",
            # Backoff was reset, but the account has not claimed yet and its
            # device id was already rotated once.
            "checkin": {"retry_backoff": 0, "device_generation": 1},
        }
        raw_accounts = [("rotated-account", rotated), ("plain-account", {"token": "t"})]
        claimed = []

        async def fake_claim(account_id):
            claimed.append(account_id)
            return {"success": True, "skipped": False, "checked_in": True}

        async def cycle():
            with (
                patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
                patch("src.main._claim_checkin_account", side_effect=fake_claim),
                patch("src.main._checkin_cooldown_remaining", return_value=0),
                patch("src.main._checkin_cache_is_today", return_value=False),
                patch("src.main._checkin_clear_retry_state"),
            ):
                await main_module._checkin_auto_retry_cycle()

        asyncio.run(cycle())
        self.assertIn("rotated-account", claimed)
        # Still not a general poller.
        self.assertNotIn("plain-account", claimed)

    def test_single_and_claim_all_share_account_lock(self):
        record = {"token": "test-token", "user_id": "user-1"}
        raw_accounts = [("account-1", record)]

        async def scenario():
            state = {"checked": False}
            claim_started = asyncio.Event()
            release_claim = asyncio.Event()
            claim_count = 0

            async def status(_token, _account_id):
                return {"checked_in": state["checked"], "credits": 200}

            async def full(_token):
                return {"account_credits": {"remaining": 10}}

            async def claim(_token, _account_id):
                nonlocal claim_count
                claim_count += 1
                claim_started.set()
                await release_claim.wait()
                state["checked"] = True
                return {"code": 0, "message": "success"}

            with (
                patch("src.main.auth.get_account_record", return_value=dict(record)),
                patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
                patch("src.main.auth.get_active_account_id", return_value="account-1"),
                patch("src.main.trae_client.fetch_checkin_credits_status", new=status),
                patch("src.main._fetch_full_credits", new=full),
                patch("src.main.auth.set_account_checkin"),
                patch("src.main.trae_client.claim_checkin_credits", new=claim),
                patch("src.main.CHECKIN_INTERVAL", 0),
                patch("src.main._CHECKIN_VERIFY_DELAYS", ()),
            ):
                single_task = asyncio.create_task(main_module.api_checkin_account("account-1"))
                await claim_started.wait()
                bulk_task = asyncio.create_task(main_module.api_checkin_claim_all())
                await asyncio.sleep(0)
                release_claim.set()
                single, bulk = await asyncio.gather(single_task, bulk_task)
            return single, bulk, claim_count

        single, bulk, claim_count = asyncio.run(scenario())
        self.assertTrue(json.loads(single.body)["success"])
        bulk_body = json.loads(bulk.body)
        self.assertTrue(bulk_body["accounts"][0]["success"])
        self.assertTrue(bulk_body["accounts"][0]["skipped"])
        self.assertEqual(claim_count, 1)

    def test_claim_credits_preserves_fresh_checkin_and_cached_fields(self):
        raw_accounts = [
            (
                "account-1",
                {
                    "token": "test-token",
                    "user_id": "user-1",
                    "checkin": {"checked_in": True, "keep": "yes"},
                    "checkin_status_updated_at": __import__("time").time(),
                    "checkin_updated_at": __import__("time").time(),
                },
            )
        ]
        saved = []

        async def scenario():
            with (
                patch("src.main.auth.get_accounts_raw", return_value=raw_accounts),
                patch("src.main.auth.get_active_account_id", return_value="account-1"),
                patch(
                    "src.main.trae_client.fetch_checkin_credits_status",
                    new=AsyncMock(side_effect=AssertionError("claim-credits must not query status")),
                ),
                patch(
                    "src.main._fetch_full_credits",
                    new=AsyncMock(
                        return_value={
                            "account_credits": {"remaining": 10},
                            "work_credits": None,
                            "total_credits": {"remaining": 10},
                        }
                    ),
                ),
                patch("src.main.auth.merge_account_credits", side_effect=lambda aid, data: saved.append(dict(data)) or dict(data)),
                patch("src.main.trae_client.claim_checkin_credits", new=AsyncMock()),
            ):
                response = await main_module.api_checkin_claim_credits()
            return response

        body = json.loads(asyncio.run(scenario()).body)
        self.assertTrue(body["accounts"][0]["success"])
        self.assertTrue(body["accounts"][0]["skipped"])
        self.assertTrue(saved[-1].get("checked_in") is True)
        self.assertEqual(saved[-1]["keep"], "yes")
        self.assertNotIn("work_credits", saved[-1])


class UsageRecordTests(unittest.TestCase):
    @staticmethod
    def _jwt_for_account(account_id: str) -> str:
        payload = base64.urlsafe_b64encode(
            json.dumps({"data": {"id": account_id}}).encode("utf-8")
        ).decode("ascii").rstrip("=")
        return "header." + payload + ".signature"

    def test_token_identity_helper_uses_jwt_data_id(self):
        token = self._jwt_for_account("jwt-account")
        self.assertEqual(
            main_module._account_id_from_token(token),
            "jwt-account",
        )

    def test_usage_tracker_keeps_explicit_credit_after_later_token_only_frame(self):
        tracker = object.__new__(main_module._UsageTracker)
        tracker.usage = main_module._usage_values({})
        tracker.saw_usage = False

        tracker.update(
            {
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
                "credits_consumed": 0.31,
            }
        )
        tracker.update(
            {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}
        )

        self.assertEqual(tracker.usage["total_tokens"], 13)
        self.assertEqual(tracker.usage["credits_consumed"], 0.31)

    def test_usage_tracker_preserves_explicit_zero_credit(self):
        tracker = object.__new__(main_module._UsageTracker)
        tracker.usage = main_module._usage_values({})
        tracker.saw_usage = False

        tracker.update({"total_tokens": 1, "credits_consumed": 0})
        tracker.update({"total_tokens": 2})

        self.assertEqual(tracker.usage["credits_consumed"], 0)

    def test_token_only_tracker_uses_bound_jwt_account_not_active_account(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()
        token = self._jwt_for_account("charged-account")

        async def scenario(path):
            with (
                patch("src.main.auth.get_active_account_id", return_value="selected-account"),
                patch("src.main.auth.get_user_id", return_value="selected-account"),
                patch("src.main.auth.get_token", return_value=self._jwt_for_account("selected-account")),
                patch("src.main._fetch_used_credits", new=AsyncMock(return_value=None)),
            ):
                tracker = main_module._UsageTracker(
                    "glm-5.3",
                    "/v1/responses",
                    False,
                    {
                        "_account_id": "selected-account",
                        "_auth_token": token,
                    },
                )
                self.assertEqual(tracker.account_id, "charged-account")
                self.assertEqual(tracker.token, token)
                tracker.update({"input_tokens": 1, "output_tokens": 1})
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish("completed")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                asyncio.run(scenario(path))
                self.assertEqual(len(main_module._USAGE_HISTORY), 1)
                self.assertEqual(
                    main_module._USAGE_HISTORY[0]["account_id"],
                    "charged-account",
                )
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_usage_record_normalizes_tokens_and_unknown_credit(self):
        record = main_module._normalize_usage_record(
            {"account_id": "acct", "model": "m", "prompt_tokens": 3, "completion_tokens": 4}
        )
        self.assertEqual(record["input_tokens"], 3)
        self.assertEqual(record["output_tokens"], 4)
        self.assertEqual(record["total_tokens"], 7)
        self.assertIsNone(record["credits_consumed"])
        self.assertEqual(record["credits_source"], "unknown")

    def test_tracker_writes_one_record_and_enriches_credit_delta(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        async def scenario(path):
            with (
                patch("src.main.auth.get_active_account_id", return_value="acct-1"),
                patch("src.main.auth.get_user_id", return_value="user-1"),
                patch("src.main.auth.get_token", return_value="jwt-token"),
                patch(
                    "src.main._fetch_used_credits",
                    new=AsyncMock(side_effect=[10, 13]),
                ),
                patch.dict(os.environ, {"TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0"}),
            ):
                tracker = main_module._UsageTracker("glm-5.3", "/v1/chat/completions", False)
                await tracker.begin()
                tracker.update({"input_tokens": 3, "output_tokens": 5, "total_tokens": 8})
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                asyncio.run(scenario(path))
                self.assertEqual(len(main_module._USAGE_HISTORY), 1)
                record = main_module._USAGE_HISTORY[0]
                self.assertEqual(record["total_tokens"], 8)
                self.assertEqual(record["credits_consumed"], 3)
                self.assertEqual(record["credits_source"], "snapshot_delta")
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_tracker_prefers_turn_usage_over_account_snapshot_delta(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        async def scenario(path):
            credit_snapshot = AsyncMock(return_value=10)
            turn_usage = AsyncMock(
                return_value={
                    "credits_consumed": 0.92,
                    "credits_source": "session_usage",
                }
            )
            with (
                patch("src.main._fetch_used_credits", new=credit_snapshot),
                patch(
                    "src.main.trae_client.fetch_session_usage",
                    new=turn_usage,
                ),
                patch.dict(
                    os.environ,
                    {
                        "TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0",
                        "TRAE_USAGE_SESSION_QUERY": "true",
                    },
                ),
            ):
                tracker = main_module._UsageTracker(
                    "glm-5.3",
                    "/v1/chat/completions",
                    False,
                    {"_account_id": "acct-1", "_auth_token": "bound-token"},
                )
                await tracker.begin()
                tracker.bind_usage_turn("turn-1")
                tracker.update({"input_tokens": 3, "output_tokens": 5})
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish()
                    tasks = list(main_module._USAGE_ENRICH_TASKS)
                    if tasks:
                        await asyncio.gather(*tasks)
            return credit_snapshot, turn_usage

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                credit_snapshot, turn_usage = asyncio.run(scenario(path))
                record = main_module._USAGE_HISTORY[0]
                self.assertEqual(record["credits_consumed"], 0.92)
                self.assertEqual(record["credits_source"], "session_usage")
                turn_usage.assert_awaited_once_with("turn-1", "bound-token")
                self.assertEqual(credit_snapshot.await_count, 1)
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_tracker_explicit_zero_credit_skips_turn_usage_query(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        async def scenario(path):
            turn_usage = AsyncMock(
                return_value={
                    "credits_consumed": 9.9,
                    "credits_source": "session_usage",
                }
            )
            with (
                patch(
                    "src.main._fetch_used_credits",
                    new=AsyncMock(return_value=None),
                ),
                patch(
                    "src.main.trae_client.fetch_session_usage",
                    new=turn_usage,
                ),
                patch.dict(
                    os.environ,
                    {
                        "TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0",
                        "TRAE_USAGE_SESSION_QUERY": "true",
                    },
                ),
            ):
                tracker = main_module._UsageTracker(
                    "glm-5.3",
                    "/v1/chat/completions",
                    False,
                    {"_account_id": "acct-zero", "_auth_token": "bound-token"},
                )
                await tracker.begin()
                tracker.bind_usage_turn("turn-zero")
                tracker.update(
                    {
                        "input_tokens": 3,
                        "output_tokens": 1,
                        "credits_consumed": 0,
                    }
                )
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish()
                return turn_usage

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                turn_usage = asyncio.run(scenario(path))
                record = main_module._USAGE_HISTORY[0]
                self.assertEqual(record["credits_consumed"], 0)
                self.assertEqual(record["credits_source"], "upstream")
                turn_usage.assert_not_awaited()
                self.assertFalse(main_module._USAGE_ENRICH_TASKS)
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_concurrent_unsafe_snapshot_still_uses_turn_usage_query(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        async def scenario(path):
            credit_snapshot = AsyncMock(return_value=10)
            turn_usage = AsyncMock(
                return_value={
                    "credits_consumed": 0.44,
                    "credits_source": "session_usage",
                }
            )
            with (
                patch("src.main._fetch_used_credits", new=credit_snapshot),
                patch(
                    "src.main.trae_client.fetch_session_usage",
                    new=turn_usage,
                ),
                patch.dict(
                    os.environ,
                    {
                        "TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0",
                        "TRAE_USAGE_SESSION_QUERY": "true",
                    },
                ),
            ):
                options = {
                    "_account_id": "acct-shared",
                    "_auth_token": "bound-token",
                }
                first = main_module._UsageTracker(
                    "glm-5.3", "/v1/chat/completions", False, options
                )
                second = main_module._UsageTracker(
                    "glm-5.3", "/v1/chat/completions", False, options
                )
                await first.begin()
                await second.begin()
                self.assertTrue(first._credit_safe)
                self.assertFalse(second._credit_safe)

                second.bind_usage_turn("turn-concurrent")
                second.update({"input_tokens": 2, "output_tokens": 1})
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await second.finish()
                    tasks = list(main_module._USAGE_ENRICH_TASKS)
                    if tasks:
                        await asyncio.gather(*tasks)

                    first.update(
                        {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "credits_consumed": 0,
                        }
                    )
                    await first.finish()
            return credit_snapshot, turn_usage, second.request_id

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                credit_snapshot, turn_usage, request_id = asyncio.run(
                    scenario(path)
                )
                record = next(
                    item
                    for item in main_module._USAGE_HISTORY
                    if item["request_id"] == request_id
                )
                self.assertEqual(record["credits_consumed"], 0.44)
                self.assertEqual(record["credits_source"], "session_usage")
                turn_usage.assert_awaited_once_with(
                    "turn-concurrent", "bound-token"
                )
                self.assertEqual(credit_snapshot.await_count, 1)
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_raw_retry_binds_second_turn_id_and_queries_usage_once(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        class LineSource:
            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self):
                return iter(self.lines)

        class FakeRawResponse:
            def __init__(self, lines):
                self.response = LineSource(lines)
                self.closed = False
                self.auth_token = "bound-token"

            def close(self):
                self.closed = True

        first = FakeRawResponse(
            ['data: {"finish_reason":"stop"}', "data: [DONE]"]
        )
        second = FakeRawResponse(
            [
                "event: model_config",
                'data: {"reply_to_message_id":"turn-second"}',
                "event: output",
                'data: {"response":"recovered"}',
                "event: done",
                'data: {"finish_reason":"stop"}',
            ]
        )

        async def scenario(path):
            send_raw = AsyncMock(side_effect=[first, second])
            turn_usage = AsyncMock(
                return_value={
                    "credits_consumed": 0.71,
                    "credits_source": "session_usage",
                }
            )
            with (
                patch(
                    "src.main.raw_client.send_raw_chat_request",
                    new=send_raw,
                ),
                patch(
                    "src.main._fetch_used_credits",
                    new=AsyncMock(return_value=None),
                ),
                patch(
                    "src.main.trae_client.fetch_session_usage",
                    new=turn_usage,
                ),
                patch.dict(
                    os.environ,
                    {
                        "TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0",
                        "TRAE_USAGE_SESSION_QUERY": "true",
                    },
                ),
            ):
                options = {
                    "_account_id": "acct-raw",
                    "_auth_token": "bound-token",
                    "session_id": "raw-fixed-session",
                }
                tracker = main_module._UsageTracker(
                    "glm-5.3", "/v1/chat/completions", False, options
                )
                context_token = main_module._USAGE_TRACKER.set(tracker)
                try:
                    await tracker.begin()
                    response = await main_module.run_raw_chat(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        options,
                    )
                finally:
                    main_module._USAGE_TRACKER.reset(context_token)
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish()
                    tasks = list(main_module._USAGE_ENRICH_TASKS)
                    if tasks:
                        await asyncio.gather(*tasks)
            return response, send_raw, turn_usage, tracker.request_id

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                response, send_raw, turn_usage, request_id = asyncio.run(
                    scenario(path)
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    json.loads(response.body)["choices"][0]["message"]["content"],
                    "recovered",
                )
                self.assertEqual(send_raw.await_count, 2)
                turn_usage.assert_awaited_once_with(
                    "turn-second", "bound-token"
                )
                record = next(
                    item
                    for item in main_module._USAGE_HISTORY
                    if item["request_id"] == request_id
                )
                self.assertEqual(record["credits_consumed"], 0.71)
                self.assertEqual(record["credits_source"], "session_usage")
                self.assertTrue(first.closed)
                self.assertTrue(second.closed)
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_tracker_falls_back_to_snapshot_when_turn_usage_fails(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        async def scenario(path):
            credit_snapshot = AsyncMock(side_effect=[10, 13])
            with (
                patch("src.main._fetch_used_credits", new=credit_snapshot),
                patch(
                    "src.main.trae_client.fetch_session_usage",
                    new=AsyncMock(side_effect=RuntimeError("usage unavailable")),
                ),
                patch.dict(
                    os.environ,
                    {
                        "TRAE_USAGE_CREDIT_SETTLE_SECONDS": "0",
                        "TRAE_USAGE_SESSION_QUERY": "true",
                    },
                ),
            ):
                tracker = main_module._UsageTracker(
                    "glm-5.3",
                    "/v1/chat/completions",
                    False,
                    {"_account_id": "acct-1", "_auth_token": "bound-token"},
                )
                await tracker.begin()
                tracker.bind_usage_turn("turn-1")
                tracker.update({"input_tokens": 3, "output_tokens": 5})
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish()
                    tasks = list(main_module._USAGE_ENRICH_TASKS)
                    if tasks:
                        await asyncio.gather(*tasks)
            return credit_snapshot

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                credit_snapshot = asyncio.run(scenario(path))
                record = main_module._USAGE_HISTORY[0]
                self.assertEqual(record["credits_consumed"], 3)
                self.assertEqual(record["credits_source"], "snapshot_delta")
                self.assertEqual(credit_snapshot.await_count, 2)
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_tracker_cancels_before_snapshot_for_explicit_upstream_credits(self):
        original_history = main_module._USAGE_HISTORY
        original_path = main_module._USAGE_RECORDS_PATH
        main_module._USAGE_HISTORY = []
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()
        main_module._USAGE_ACTIVE_ACCOUNTS.clear()
        main_module._USAGE_UNSAFE_ACCOUNTS.clear()

        async def scenario(path):
            started = asyncio.Event()
            stopped = asyncio.Event()

            async def pending_snapshot(_token):
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    stopped.set()

            with (
                patch("src.main.auth.get_active_account_id", return_value="acct-1"),
                patch("src.main.auth.get_user_id", return_value="user-1"),
                patch("src.main.auth.get_token", return_value="jwt-token"),
                patch("src.main._fetch_used_credits", new=pending_snapshot),
            ):
                tracker = main_module._UsageTracker(
                    "glm-5.3", "/v1/chat/completions", False
                )
                await tracker.begin()
                await started.wait()
                before_task = tracker._before_task
                tracker.update(
                    {
                        "input_tokens": 3,
                        "output_tokens": 5,
                        "total_tokens": 8,
                        "credits_consumed": 2,
                    }
                )
                with patch.object(main_module, "_USAGE_RECORDS_PATH", path):
                    await tracker.finish()
                return before_task, stopped.is_set()

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "usage_records.json"
            try:
                before_task, stopped = asyncio.run(scenario(path))
                self.assertTrue(before_task.cancelled())
                self.assertTrue(stopped)
                self.assertFalse(main_module._USAGE_SNAPSHOT_TASKS)
                self.assertFalse(main_module._USAGE_ENRICH_TASKS)
                self.assertFalse(main_module._USAGE_ACTIVE_ACCOUNTS)
                self.assertEqual(len(main_module._USAGE_HISTORY), 1)
                record = main_module._USAGE_HISTORY[0]
                self.assertEqual(record["credits_consumed"], 2)
                self.assertEqual(record["credits_source"], "upstream")
            finally:
                main_module._USAGE_HISTORY = original_history
                main_module._USAGE_RECORDS_PATH = original_path
                main_module._USAGE_ENRICH_TASKS.clear()
                main_module._USAGE_SNAPSHOT_TASKS.clear()
                main_module._USAGE_ACTIVE_ACCOUNTS.clear()
                main_module._USAGE_UNSAFE_ACCOUNTS.clear()

    def test_lifespan_cancels_and_waits_for_usage_tasks(self):
        main_module._USAGE_ENRICH_TASKS.clear()
        main_module._USAGE_SNAPSHOT_TASKS.clear()

        async def scenario():
            stopped = set()

            async def pending(label):
                try:
                    await asyncio.Future()
                finally:
                    stopped.add(label)

            async with main_module.lifespan(main_module.app):
                enrich_task = main_module._spawn_usage_task(pending("enrich"))
                snapshot_task = main_module._spawn_usage_task(
                    pending("snapshot"), main_module._USAGE_SNAPSHOT_TASKS
                )
                await asyncio.sleep(0)
            return enrich_task, snapshot_task, stopped

        with patch("src.main.init_app", new=AsyncMock()):
            enrich_task, snapshot_task, stopped = asyncio.run(scenario())

        self.assertTrue(enrich_task.cancelled())
        self.assertTrue(snapshot_task.cancelled())
        self.assertEqual(stopped, {"enrich", "snapshot"})
        self.assertFalse(main_module._USAGE_ENRICH_TASKS)
        self.assertFalse(main_module._USAGE_SNAPSHOT_TASKS)

    def test_rendered_console_keeps_javascript_newline_escapes(self):
        html = main_module._web_login_html()
        self.assertIn("requestJSON", html)
        self.assertIn("消耗积分", html)
        self.assertIn("join('\\n')", html)
        self.assertNotIn("失败：\n'+", html)
        self.assertIn("[hidden] { display:none !important; }", html)
        self.assertIn("setAccountCheckinBusy(id,true)", html)
        self.assertIn("checkinAccountBusy=new Set()", html)
        self.assertIn("checkinAccountBusy.has", html)
        self.assertIn("查询签到状态", html)
        self.assertIn("查询全部积分", html)
        self.assertIn("async function creditsRefreshAll()", html)
        single_start = html.index("async function checkinAccount(id)")
        single_end = html.index("async function checkinClaimAll()", single_start)
        single_account = html[single_start:single_end]
        self.assertNotIn("checkinRefreshAll()", single_account)
        self.assertNotIn("var refresh=await requestJSON", single_account)
        status_start = html.index("async function checkinRefreshAll()")
        credits_start = html.index("async function creditsRefreshAll()", status_start)
        status_refresh = html[status_start:credits_start]
        self.assertIn("/api/checkin/accounts", status_refresh)
        self.assertIn("updateAccountCheckinRow", status_refresh)
        credits_end = html.index("async function checkinAccount(id)", credits_start)
        credits_refresh = html[credits_start:credits_end]
        self.assertIn("/api/checkin/credits/accounts", credits_refresh)
        self.assertIn("updateAccountCreditsRow", credits_refresh)
        self.assertNotIn("updateAccountCheckinRow", credits_refresh)


class _FakeUsageTracker:
    def __init__(self):
        self.statuses = []

    async def begin(self):
        pass

    async def finish(self, status):
        self.statuses.append(status)


class UsageStreamTerminalTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_after_done_records_completed(self):
        tracker = _FakeUsageTracker()
        gate = asyncio.Event()
        started = asyncio.Event()
        got = []

        async def source():
            yield "data: [DONE]\n\n"
            started.set()
            await gate.wait()

        async def consume():
            async for chunk in main_module._tracked_stream(source(), tracker):
                got.append(chunk)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(got, ["data: [DONE]\n\n"])
        self.assertEqual(tracker.statuses, ["completed"])

    async def test_cancel_before_done_records_cancelled(self):
        tracker = _FakeUsageTracker()
        gate = asyncio.Event()
        started = asyncio.Event()
        got = []

        async def source():
            yield 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            started.set()
            await gate.wait()

        async def consume():
            async for chunk in main_module._tracked_stream(source(), tracker):
                got.append(chunk)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(got), 1)
        self.assertIn("content", got[0])
        self.assertEqual(tracker.statuses, ["cancelled"])

    async def test_close_after_done_records_completed(self):
        tracker = _FakeUsageTracker()

        async def source():
            yield "data: [DONE]\n\n"
            await asyncio.Event().wait()

        stream = main_module._tracked_stream(source(), tracker)
        await anext(stream)
        await stream.aclose()
        self.assertEqual(tracker.statuses, ["completed"])

    async def test_responses_terminal_event_counts_as_done(self):
        tracker = _FakeUsageTracker()
        gate = asyncio.Event()
        started = asyncio.Event()
        got = []

        async def source():
            yield "event: response.completed\ndata: {}\n\n"
            started.set()
            await gate.wait()

        async def consume():
            async for chunk in main_module._tracked_stream(source(), tracker):
                got.append(chunk)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(got), 1)
        self.assertIn("response.completed", got[0])
        self.assertEqual(tracker.statuses, ["completed"])


class SessionLeaseTests(unittest.TestCase):
    def setUp(self):
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()

    def tearDown(self):
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()

    def test_chat_normalization_repairs_missing_tool_result(self):
        normalized = main_module._normalize_chat_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "mcp__node_repl__js:92",
                            "type": "function",
                            "function": {
                                "name": "mcp__node_repl__js",
                                "arguments": '{"code":"1+1"}',
                            },
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ]
        )

        self.assertEqual([item["role"] for item in normalized], ["assistant", "tool", "user"])
        repaired = normalized[1]
        self.assertEqual(repaired["tool_call_id"], "mcp__node_repl__js:92")
        self.assertTrue(repaired["is_error"])

    def test_continuation_reuses_initial_account_and_token_snapshot(self):
        messages = [{"role": "user", "content": "inspect the workspace"}]
        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.auth.next_polling_account") as rotate,
            patch("src.main.auth.get_active_account_id", return_value="account-1"),
            patch(
                "src.main.auth.get_active_account_snapshot",
                return_value=("account-1", {"token": "first-jwt"}),
            ),
            patch(
                "src.main.auth.get_account_record",
                return_value={"token": "first-jwt"},
            ) as account,
            patch("src.main.auth.get_token", return_value="fallback-jwt"),
        ):
            first = main_module._bind_chat_session(
                messages, {}, requested_session_id="terminal-1"
            )
            second = main_module._bind_chat_session(
                [
                    *messages,
                    {"role": "assistant", "content": "tool requested"},
                    {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                ],
                {},
                requested_session_id="terminal-1",
            )

        self.assertEqual(first["_account_id"], "account-1")
        self.assertEqual(first["_auth_token"], "first-jwt")
        self.assertEqual(second["_account_id"], "account-1")
        self.assertEqual(second["_auth_token"], "first-jwt")
        self.assertEqual(rotate.call_count, 1)
        # The atomic snapshot supplies the credential and avoids a second
        # mutable account-store read.
        self.assertEqual(account.call_count, 0)

    def test_session_binding_uses_jwt_billing_identity_for_raw_uid(self):
        token = UsageRecordTests._jwt_for_account("billing-account")
        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.auth.next_polling_account"),
            patch(
                "src.main.auth.get_active_account_snapshot",
                return_value=("store-account", {"token": token}),
            ),
        ):
            bound = main_module._bind_chat_session(
                [{"role": "user", "content": "hello"}],
                {},
                requested_session_id="terminal-billing",
            )

        self.assertEqual(bound["_account_id"], "store-account")
        self.assertEqual(bound["_billing_id"], "billing-account")
        self.assertEqual(bound["_auth_user_id"], "billing-account")

    def test_inferred_history_rebinds_after_account_switch_but_explicit_session_stays_pinned(self):
        messages = [{"role": "user", "content": "inspect the workspace"}]
        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.auth.next_polling_account"),
            patch(
                "src.main.auth.get_active_account_snapshot",
                side_effect=[
                    ("account-a", {"token": "token-a"}),
                    ("account-b", {"token": "token-b"}),
                ],
            ) as snapshot,
        ):
            first = main_module._bind_chat_session(messages, {})
            inferred = main_module._bind_chat_session(
                [
                    *messages,
                    {"role": "assistant", "content": "tool requested"},
                    {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                ],
                {},
            )

        self.assertEqual(first["_account_id"], "account-a")
        self.assertEqual(first["_auth_token"], "token-a")
        self.assertEqual(inferred["_account_id"], "account-b")
        self.assertEqual(inferred["_auth_token"], "token-b")
        self.assertEqual(snapshot.call_count, 2)

        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()
        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.auth.next_polling_account"),
            patch(
                "src.main.auth.get_active_account_snapshot",
                side_effect=[
                    ("account-a", {"token": "token-a"}),
                ],
            ),
        ):
            pinned = main_module._bind_chat_session(
                messages,
                {},
                requested_session_id="terminal-1",
            )
            with patch(
                "src.main.auth.get_active_account_snapshot",
                return_value=("account-b", {"token": "token-b"}),
            ):
                continuation = main_module._bind_chat_session(
                    [
                        *messages,
                        {"role": "assistant", "content": "tool requested"},
                        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                    ],
                    {},
                    requested_session_id="terminal-1",
                )

        self.assertEqual(pinned["_account_id"], "account-a")
        self.assertEqual(continuation["_account_id"], "account-a")

    def test_idle_session_lease_is_reaped_and_active_stream_is_retained(self):
        main_module._UPSTREAM_SESSION_LEASES["idle"] = main_module._UpstreamSessionLease(
            account_id="account-1",
            billing_id="account-1",
            auth_token="token-1",
            last_client_activity=10.0,
        )
        main_module._UPSTREAM_SESSION_LEASES["streaming"] = main_module._UpstreamSessionLease(
            account_id="account-2",
            billing_id="account-2",
            auth_token="token-2",
            last_client_activity=10.0,
            active_streams=1,
        )

        main_module._prune_chat_sessions(10.0 + main_module._CHAT_SESSION_TTL + 1)

        self.assertNotIn("idle", main_module._UPSTREAM_SESSION_LEASES)
        self.assertIn("streaming", main_module._UPSTREAM_SESSION_LEASES)

    def test_leased_stream_marks_client_activity_until_sse_finishes(self):
        main_module._UPSTREAM_SESSION_LEASES["stream-1"] = main_module._UpstreamSessionLease(
            account_id="account-1",
            billing_id="account-1",
            auth_token="token-1",
            last_client_activity=0.0,
        )

        async def source():
            yield "data: first\\n\\n"
            yield "data: [DONE]\\n\\n"

        async def consume():
            values = []
            async for value in main_module._lease_stream(source(), "stream-1"):
                values.append(value)
                self.assertEqual(
                    main_module._UPSTREAM_SESSION_LEASES["stream-1"].active_streams,
                    1,
                )
            return values

        values = asyncio.run(consume())
        lease = main_module._UPSTREAM_SESSION_LEASES["stream-1"]
        self.assertEqual(values, ["data: first\\n\\n", "data: [DONE]\\n\\n"])
        self.assertEqual(lease.active_streams, 0)
        self.assertGreater(lease.last_client_activity, 0)


class MainCliSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream_mode = patch.object(main_module, "UPSTREAM_MODE", "cli")
        cls.upstream_mode.start()
        try:
            cls.client = TestClient(app)
            cls.client.__enter__()
        except Exception:
            cls.upstream_mode.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.__exit__(None, None, None)
        finally:
            cls.upstream_mode.stop()

    def test_status(self):
        response = self.client.get("/v1/status", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "cli")
        self.assertEqual(body["upstream_mode"], "cli")
        self.assertEqual(body["tool_execution"], "client")
        self.assertTrue(body["capabilities"]["openai_tool_calls"])
        self.assertFalse(body["capabilities"]["server_executes_caller_tools"])
        self.assertTrue(body["cli"]["available"])

    def test_chat_nonstream(self):
        with patch(
            "src.main._fetch_used_credits", new=AsyncMock(return_value=None)
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                headers=AUTH_HEADERS,
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        self.assertIn("fake reply", content)
        self.assertEqual(body["usage"]["total_tokens"], 17)

    def test_chat_tools_are_forwarded_and_returned(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        expected_options = {
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "client_context": {
                "workspace_path": r"C:\work\demo",
                "system_type": "Windows",
                "terminal_context": [{"shell": "PowerShell"}],
            },
            "session_id": "session-tools-1",
        }
        tool_calls = [
            {
                "id": "call_weather_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Shanghai"}',
                },
            }
        ]
        captured = {}

        async def fake_stream(messages, model, options=None):
            captured["messages"] = messages
            captured["model"] = model
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "tool_calls": tool_calls,
                    }
                },
            )

        with patch("src.main.cli_client.stream_cli_chat", side_effect=fake_stream):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    **expected_options,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {
                key: value
                for key, value in captured["options"].items()
                if key not in ("_relay_request_id", "_billing_id")
            },
            expected_options,
        )
        self.assertRegex(captured["options"]["_relay_request_id"], r"^req-[0-9a-f]+$")
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"], tool_calls)

    def test_chat_tools_auto_discover_client_environment_and_plugins(self):
        captured = {}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_search",
                    "description": "Load deferred client tools",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser__open",
                    "description": "Open a page with the browser plugin",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "description": "Execute a command on the client",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        async def fake_stream(messages, model, options=None):
            captured["options"] = options
            yield CliEvent(
                type="json",
                data={"message": {"content": [{"type": "text", "text": "ready"}]}},
            )

        headers = {
            **AUTH_HEADERS,
            "User-Agent": "codex_cli_rs/0.148.0",
            "X-Stainless-OS": "Windows",
        }
        with patch("src.main.cli_client.stream_cli_chat", side_effect=fake_stream):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Inspect the project"}],
                    "tools": tools,
                },
                headers=headers,
            )

        self.assertEqual(response.status_code, 200, response.text)
        context = captured["options"]["client_context"]
        self.assertEqual(context["client_name"], "Codex")
        self.assertEqual(context["system_type"], "Windows")
        self.assertEqual(context["terminal_context"][0]["shell"], "PowerShell")
        discovery = context["tool_discovery"]
        self.assertTrue(discovery["tool_search_available"])
        self.assertIn("browser", discovery["plugin_namespaces"])
        self.assertIn("shell_exec", discovery["environment_probe_tools"])

    def test_two_turn_tool_loop_through_real_cli_subprocess(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        user_message = {
            "role": "user",
            "content": "FAKE_TOOL_CALL_REQUEST: inspect README.md",
        }
        first = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [user_message],
                "tools": tools,
                "client_context": {
                    "workspace_path": r"C:\work\demo",
                    "system_type": "Windows",
                    "terminal_context": [{"shell": "PowerShell"}],
                },
            },
            headers=AUTH_HEADERS,
        )

        self.assertEqual(first.status_code, 200)
        assistant = first.json()["choices"][0]["message"]
        self.assertEqual(first.json()["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_fake_read")

        second = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    user_message,
                    assistant,
                    {
                        "role": "tool",
                        "tool_call_id": "call_fake_read",
                        "name": "read_file",
                        "content": "README contents from the Windows client",
                    },
                ],
                "tools": tools,
            },
            headers=AUTH_HEADERS,
        )

        self.assertEqual(second.status_code, 200)
        choice = second.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIn("fake reply", choice["message"]["content"])

    def test_auto_tools_do_not_fall_back_when_raw_fails(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        raw = AsyncMock(side_effect=RuntimeError("raw unavailable"))
        cli = AsyncMock(return_value=JSONResponse({"route": "cli"}))
        web = AsyncMock(side_effect=AssertionError("web must not run external tools"))
        ide = AsyncMock(side_effect=AssertionError("ide must not run external tools"))

        with (
            patch("src.main.UPSTREAM_MODE", "auto"),
            patch("src.main.run_raw_chat", raw),
            patch("src.main.run_cli_chat", cli),
            patch("src.main._run_web_with_retry", web),
            patch("src.main.run_ide_chat", ide),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Read README.md"}],
                    "tools": tools,
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("raw: raw unavailable", response.json()["error"]["message"])
        raw.assert_awaited_once()
        cli.assert_not_awaited()
        web.assert_not_awaited()
        ide.assert_not_awaited()

    def test_auto_tools_never_fall_back_to_web_or_ide_after_failures(self):
        raw = AsyncMock(side_effect=RuntimeError("raw unavailable"))
        cli = AsyncMock(side_effect=RuntimeError("cli unavailable"))
        web = AsyncMock(side_effect=AssertionError("web must not run external tools"))
        ide = AsyncMock(side_effect=AssertionError("ide must not run external tools"))

        with (
            patch("src.main.UPSTREAM_MODE", "auto"),
            patch("src.main.run_raw_chat", raw),
            patch("src.main.run_cli_chat", cli),
            patch("src.main._run_web_with_retry", web),
            patch("src.main.run_ide_chat", ide),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Read README.md"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502)
        message = response.json()["error"]["message"]
        self.assertIn("raw: raw unavailable", message)
        self.assertNotIn("cli:", message)
        self.assertNotIn("web:", message)
        self.assertNotIn("ide:", message)
        raw.assert_awaited_once()
        cli.assert_not_awaited()
        web.assert_not_awaited()
        ide.assert_not_awaited()

    def test_empty_tools_and_none_choice_still_use_only_safe_routes(self):
        raw = AsyncMock(side_effect=RuntimeError("raw unavailable"))
        cli = AsyncMock(return_value=JSONResponse({"route": "cli"}))
        web = AsyncMock(side_effect=AssertionError("web must not run tool policy"))
        ide = AsyncMock(side_effect=AssertionError("ide must not run tool policy"))

        with (
            patch("src.main.UPSTREAM_MODE", "auto"),
            patch("src.main.run_raw_chat", raw),
            patch("src.main.run_cli_chat", cli),
            patch("src.main._run_web_with_retry", web),
            patch("src.main.run_ide_chat", ide),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "answer directly"}],
                    "tools": [],
                    "tool_choice": "none",
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("raw: raw unavailable", response.json()["error"]["message"])
        raw.assert_awaited_once()
        cli.assert_not_awaited()
        web.assert_not_awaited()
        ide.assert_not_awaited()

    def test_tool_result_continuation_without_tools_stays_on_safe_routes(self):
        raw = AsyncMock(side_effect=RuntimeError("raw unavailable"))
        cli = AsyncMock(return_value=JSONResponse({"route": "cli"}))
        web = AsyncMock(side_effect=AssertionError("web must not run tool history"))
        ide = AsyncMock(side_effect=AssertionError("ide must not run tool history"))
        messages = [
            {"role": "user", "content": "read it"},
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

        with (
            patch("src.main.UPSTREAM_MODE", "auto"),
            patch("src.main.run_raw_chat", raw),
            patch("src.main.run_cli_chat", cli),
            patch("src.main._run_web_with_retry", web),
            patch("src.main.run_ide_chat", ide),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": messages},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502)
        raw.assert_awaited_once()
        cli.assert_not_awaited()
        web.assert_not_awaited()
        ide.assert_not_awaited()

    def test_auto_plain_chat_uses_only_raw(self):
        raw = AsyncMock(return_value=JSONResponse({"route": "raw"}))
        remote = AsyncMock(side_effect=AssertionError("remote must not run in auto mode"))
        cli = AsyncMock(side_effect=AssertionError("cli must not run in auto mode"))
        with (
            patch("src.main.UPSTREAM_MODE", "auto"),
            patch("src.main.run_raw_chat", raw),
            patch("src.main._run_remote_with_retry", remote),
            patch("src.main.run_cli_chat", cli),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"route": "raw"})
        raw.assert_awaited_once()
        remote.assert_not_awaited()
        cli.assert_not_awaited()

    def test_web_mode_accepts_caller_owned_tools(self):
        web = AsyncMock(return_value=JSONResponse({"route": "web"}))
        with patch("src.main.UPSTREAM_MODE", "web"), patch(
            "src.main._run_web_with_retry", web
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Read README.md"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        web.assert_awaited_once()

    def test_ide_mode_accepts_caller_owned_tools(self):
        ide = AsyncMock(return_value=JSONResponse({"route": "ide"}))
        with patch("src.main.UPSTREAM_MODE", "ide"), patch(
            "src.main.run_ide_chat", ide
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Read README.md"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        ide.assert_awaited_once()

    def test_web_mode_accepts_empty_tool_policy(self):
        web = AsyncMock(return_value=JSONResponse({"route": "web"}))
        with patch("src.main.UPSTREAM_MODE", "web"), patch(
            "src.main._run_web_with_retry", web
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [],
                    "tool_choice": "none",
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        web.assert_awaited_once()

    def test_malformed_tools_are_rejected_before_upstream_routing(self):
        raw = AsyncMock(side_effect=AssertionError("raw must not receive invalid tools"))
        cli = AsyncMock(side_effect=AssertionError("cli must not receive invalid tools"))

        with (
            patch("src.main.UPSTREAM_MODE", "auto"),
            patch("src.main.run_raw_chat", raw),
            patch("src.main.run_cli_chat", cli),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": ["bad"],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "tools.0")
        raw.assert_not_awaited()
        cli.assert_not_awaited()

    def test_chat_accepts_flat_responses_tool_shape(self):
        tools = [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file from the caller workspace.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        captured = {}

        async def fake_raw(messages, model, stream=False, options=None):
            captured["messages"] = messages
            captured["model"] = model
            captured["options"] = options
            from fastapi.responses import JSONResponse
            return JSONResponse({
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with (
            patch.object(main_module, "UPSTREAM_MODE", "raw"),
            patch("src.main.run_raw_chat", new=fake_raw),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hello"}], "tools": tools},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["options"]["tools"][0]["function"]["name"], "read_file")

    def test_tool_choice_must_reference_a_declared_tool(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "shell"},
                },
            },
            headers=AUTH_HEADERS,
        )

        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["param"], "tool_choice")
        self.assertIn("undeclared tool", error["message"])

    def test_conflicting_session_aliases_are_rejected(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "session_id": "one",
                "sessionId": "two",
            },
            headers=AUTH_HEADERS,
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "session_id")

    def test_raw_stream_returns_openai_tool_call_deltas(self):
        tool_call = {
            "id": "call_read_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        }

        class FakeRawResponse:
            def __init__(self):
                self.response = [
                    "event: output",
                    "data: " + json.dumps(
                        {"tool_calls": [tool_call], "finish_reason": "tool_calls"}
                    ),
                    "data: [DONE]",
                ]
                self.closed = False

            def close(self):
                self.closed = True

        wrapped = FakeRawResponse()
        send_raw = AsyncMock(return_value=wrapped)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
            self.client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Read README.md"}],
                    "tools": tools,
                    "client_context": {
                        "workspace_path": r"C:\work\demo",
                        "system_type": "Windows",
                        "terminal_context": [{"shell": "PowerShell"}],
                    },
                },
                headers=AUTH_HEADERS,
            ) as response,
        ):
            self.assertEqual(response.status_code, 200)
            lines = list(response.iter_lines())

        events = [
            json.loads(line[6:])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        deltas = [
            call
            for event in events
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        finish_reasons = [
            choice.get("finish_reason")
            for event in events
            for choice in event.get("choices", [])
            if choice.get("finish_reason")
        ]

        self.assertEqual(deltas, [{"index": 0, **tool_call}])
        self.assertIn("tool_calls", finish_reasons)
        self.assertTrue(wrapped.closed)
        send_raw.assert_awaited_once()

    def test_raw_stream_retries_empty_turn_without_leaking_placeholder(self):
        class LineSource:
            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self):
                return iter(self.lines)

            def __iter__(self):
                return iter(self.lines)

        class FakeRawResponse:
            def __init__(self, lines):
                self.response = LineSource(lines)
                self.closed = False

            def close(self):
                self.closed = True

        first = FakeRawResponse(
            ['data: {"finish_reason":"stop"}', "data: [DONE]"]
        )
        second = FakeRawResponse(
            [
                'data: {"response":"recovered"}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        send_raw = AsyncMock(side_effect=[first, second])

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
            self.client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers=AUTH_HEADERS,
            ) as response,
        ):
            self.assertEqual(response.status_code, 200)
            lines = list(response.iter_lines())

        events = [
            json.loads(line[6:])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        contents = [
            choice.get("delta", {}).get("content", "")
            for event in events
            for choice in event.get("choices", [])
        ]
        roles = [
            choice.get("delta", {}).get("role")
            for event in events
            for choice in event.get("choices", [])
            if choice.get("delta", {}).get("role")
        ]
        finishes = [
            choice.get("finish_reason")
            for event in events
            for choice in event.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        self.assertEqual("".join(contents), "recovered")
        self.assertNotIn("trae upstream returned an empty response", "".join(contents))
        self.assertEqual(roles, ["assistant"])
        self.assertEqual(finishes, ["stop"])
        self.assertEqual(lines.count("data: [DONE]"), 1)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(send_raw.await_count, 2)

    def test_raw_nonstream_retries_empty_turn_once(self):
        class LineSource:
            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self):
                return iter(self.lines)

        class FakeRawResponse:
            def __init__(self, lines):
                self.response = LineSource(lines)
                self.closed = False

            def close(self):
                self.closed = True

        first = FakeRawResponse(
            ['data: {"finish_reason":"stop"}', "data: [DONE]"]
        )
        second = FakeRawResponse(
            [
                'data: {"response":"recovered"}',
                'data: {"finish_reason":"stop"}',
                "data: [DONE]",
            ]
        )
        send_raw = AsyncMock(side_effect=[first, second])

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"], "recovered"
        )
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(send_raw.await_count, 2)

    def test_raw_nonstream_does_not_retry_empty_turn_after_usage(self):
        class LineSource:
            def iter_lines(self):
                return iter(
                    [
                        "event: token_usage",
                        'data: {"input_token":8,"output_token":1,"credits_float":0.2}',
                        "event: done",
                        'data: {"finish_reason":"stop"}',
                    ]
                )

        class FakeRawResponse:
            def __init__(self):
                self.response = LineSource()
                self.closed = False

            def close(self):
                self.closed = True

        first = FakeRawResponse()
        send_raw = AsyncMock(return_value=first)

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502, response.text)
        send_raw.assert_awaited_once()
        self.assertTrue(first.closed)

    def test_raw_nonstream_second_incomplete_attempt_never_opens_third_request(self):
        class LineSource:
            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self):
                return iter(self.lines)

        class FakeRawResponse:
            def __init__(self, lines):
                self.response = LineSource(lines)
                self.closed = False

            def close(self):
                self.closed = True

        first = FakeRawResponse(
            ['data: {"finish_reason":"stop"}', "data: [DONE]"]
        )
        second = FakeRawResponse([])
        send_raw = AsyncMock(side_effect=[first, second])

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(send_raw.await_count, 2)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)

    def test_raw_nonstream_does_not_replay_completed_tool_repeat(self):
        class LineSource:
            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self):
                return iter(self.lines)

        class FakeRawResponse:
            def __init__(self, lines):
                self.response = LineSource(lines)
                self.closed = False

            def close(self):
                self.closed = True

        repeated_call = {
            "id": "call-new-id",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        }
        first = FakeRawResponse(
            [
                "data: "
                + json.dumps(
                    {"tool_calls": [repeated_call], "finish_reason": "tool_calls"}
                ),
                "data: [DONE]",
            ]
        )
        send_raw = AsyncMock(return_value=first)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        messages = [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-old-id",
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
                "tool_call_id": "call-old-id",
                "name": "read_file",
                "content": "README contents",
            },
        ]

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": messages, "tools": tools},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(send_raw.await_count, 1)
        self.assertTrue(first.closed)

    def test_raw_stream_stops_after_one_empty_retry(self):
        class LineSource:
            def iter_lines(self):
                return iter(['data: {"finish_reason":"stop"}', "data: [DONE]"])

            def __iter__(self):
                return self.iter_lines()

        class FakeRawResponse:
            def __init__(self):
                self.response = LineSource()
                self.closed = False

            def close(self):
                self.closed = True

        attempts = [FakeRawResponse(), FakeRawResponse()]
        send_raw = AsyncMock(side_effect=attempts)

        with (
            patch("src.main.UPSTREAM_MODE", "raw"),
            patch("src.main.raw_client.send_raw_chat_request", send_raw),
            self.client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers=AUTH_HEADERS,
            ) as response,
        ):
            self.assertEqual(response.status_code, 200)
            lines = list(response.iter_lines())

        events = [
            json.loads(line[6:])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in events
            for choice in event.get("choices", [])
        )
        roles = sum(
            choice.get("delta", {}).get("role") == "assistant"
            for event in events
            for choice in event.get("choices", [])
        )
        finishes = sum(
            choice.get("finish_reason") is not None
            for event in events
            for choice in event.get("choices", [])
        )
        errors = [event["error"] for event in events if "error" in event]
        self.assertEqual(content, "")
        self.assertEqual(roles, 0)
        self.assertEqual(finishes, 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("returned no text or tool call", errors[0]["message"])
        self.assertEqual(lines.count("data: [DONE]"), 1)
        self.assertEqual(send_raw.await_count, 2)
        self.assertTrue(all(item.closed for item in attempts))

    def test_chat_stream(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=AUTH_HEADERS,
        ) as response:
            self.assertEqual(response.status_code, 200)
            lines = list(response.iter_lines())

        events = []
        done = False
        for line in lines:
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload))

        self.assertTrue(done)
        content = "".join(
            choice["delta"].get("content", "")
            for event in events
            for choice in event.get("choices", [])
        )
        self.assertIn("fake reply", content)
        self.assertIn(17, [event["usage"]["total_tokens"] for event in events if "usage" in event])

    def test_deferred_stream_starts_upstream_before_first_keepalive(self):
        calls = []

        async def fake_dispatch(messages, model, stream, options=None):
            calls.append((messages, model, stream, options))

            async def body():
                yield "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"

            return StreamingResponse(body(), media_type="text/event-stream")

        async def collect_first():
            stream = main_module._deferred_dispatch_stream(
                [{"role": "user", "content": "long task"}],
                "auto",
                {"session_id": "session-deferred"},
            )
            first = await anext(stream)
            await stream.aclose()
            return first

        with patch("src.main._dispatch_chat", new=fake_dispatch):
            first = asyncio.run(collect_first())

        self.assertEqual(
            first,
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], True)

    def test_model_test_endpoint_reports_text_and_tool_probes(self):
        text_reply = JSONResponse(
            {
                "choices": [
                    {"message": {"content": "pong"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
                "provider_model_name": "glm-5.3__max",
            }
        )
        with patch("src.main._dispatch_chat", new=AsyncMock(return_value=text_reply)):
            body = self.client.post(
                "/api/model-test",
                json={"model": "glm-5.3", "mode": "text"},
                headers=AUTH_HEADERS,
            ).json()
        self.assertTrue(body["success"])
        self.assertEqual(body["reply"], "pong")
        self.assertEqual(body["provider_model_name"], "glm-5.3__max")
        self.assertIsNone(body["error"])

        tool_reply = JSONResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "relay_probe",
                                        "arguments": '{"token":"pong"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }
        )
        with patch("src.main._dispatch_chat", new=AsyncMock(return_value=tool_reply)):
            body = self.client.post(
                "/api/model-test",
                json={"model": "glm-5.3", "mode": "tool"},
                headers=AUTH_HEADERS,
            ).json()
        self.assertTrue(body["success"])
        self.assertEqual(
            [call["name"] for call in body["tool_calls"]], ["relay_probe"]
        )

    def test_model_test_endpoint_surfaces_failures(self):
        empty = JSONResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "(trae upstream returned an empty response)"
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        )
        with patch("src.main._dispatch_chat", new=AsyncMock(return_value=empty)):
            body = self.client.post(
                "/api/model-test", json={"model": "m"}, headers=AUTH_HEADERS
            ).json()
        self.assertFalse(body["success"])
        self.assertIn("empty response", body["error"])

        upstream_502 = JSONResponse(
            {"error": {"message": "All upstream paths failed: raw", "type": "api_error"}},
            status_code=502,
        )
        with patch("src.main._dispatch_chat", new=AsyncMock(return_value=upstream_502)):
            body = self.client.post(
                "/api/model-test", json={"model": "m"}, headers=AUTH_HEADERS
            ).json()
        self.assertFalse(body["success"])
        self.assertIn("All upstream paths failed", body["error"])

        with patch(
            "src.main._dispatch_chat", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            body = self.client.post(
                "/api/model-test", json={"model": "m"}, headers=AUTH_HEADERS
            ).json()
        self.assertFalse(body["success"])
        self.assertIn("boom", body["error"])

        self.assertEqual(
            self.client.post(
                "/api/model-test", json={}, headers=AUTH_HEADERS
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                "/api/model-test",
                json={"model": "m", "mode": "bogus"},
                headers=AUTH_HEADERS,
            ).status_code,
            400,
        )

    def test_dashboard_exposes_model_connectivity_panel(self):
        html = main_module._web_login_html()
        for token in (
            "conn-run-btn",
            "runConnTest",
            "/api/model-test",
            "conn-tbody",
            "conn-mode-tool",
        ):
            with self.subTest(token=token):
                self.assertIn(token, html)

    def test_stream_start_event_is_a_standard_assistant_chunk(self):
        raw = main_module._stream_start_event("glm-5.3")
        payload = json.loads(raw[len("data: "):].strip())

        self.assertEqual(payload["model"], "glm-5.3")
        self.assertEqual(len(payload["choices"]), 1)
        choice = payload["choices"][0]
        self.assertEqual(choice["delta"]["content"], "")
        self.assertIsNone(choice["finish_reason"])
        # The assistant role belongs to the translator's opening frame; sending
        # it here too would emit it twice in one stream.
        self.assertNotIn("role", choice["delta"])

    def test_get_v1_models(self):
        response = self.client.get("/v1", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertTrue(body["data"])

    def test_get_v1_models_with_refresh(self):
        response = self.client.get("/v1/models?refresh=true", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertGreaterEqual(len(body["data"]), 1)

    def test_web_login_has_model_refresh_button(self):
        response = self.client.get("/web/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("refreshModels", response.text)
        self.assertIn("获取模型列表", response.text)

    def test_web_login_usage_panel_is_below_account_list(self):
        response = self.client.get("/web/login")
        self.assertEqual(response.status_code, 200)
        account_index = response.text.index('<div class="section-title">账号列表</div>')
        usage_index = response.text.index('<div class="section-title">消费记录</div>')
        polling_index = response.text.index('<div class="section-title">多账号轮询</div>')
        self.assertLess(account_index, usage_index)
        self.assertLess(usage_index, polling_index)
        self.assertIn('<div class="panel-card" id="usage-panel">', response.text)

    def test_usage_records_are_public_for_the_login_console(self):
        response = self.client.get("/api/usage/records")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_healthz_without_auth(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)

    def test_web_login_public(self):
        response = self.client.get("/web/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("\u7528 Trae \u7f51\u9875\u6388\u6743\u767b\u5f55", response.text)

        download = self.client.get("/web/login/download")
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.content.startswith(b"#!/usr/bin/env python3"))

    def test_web_auth_callback(self):
        user_jwt = json.dumps({
            "ClientID": "ono9krqynydwx5",
            "Token": "oauth-token",
            "RefreshToken": "oauth-refresh",
            "TokenExpireAt": "1900000000000",
        })
        user_info = json.dumps({
            "UserID": "uid-1",
            "TenantID": "tenant-1",
            "Region": "CN",
            "AIRegion": "cn",
        })
        response = self.client.get(
            "/authorize",
            params={
                "userJwt": user_jwt,
                "userInfo": user_info,
                "host": "https://trae-api-cn.mchost.guru",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("\u767b\u5f55\u6210\u529f", response.text)
        status = self.client.get("/v1/status").json()
        self.assertTrue(status["has_token"])
        self.assertEqual(status["source"], "web-login")


if __name__ == "__main__":
    unittest.main()
