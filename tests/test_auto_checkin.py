"""Scheduled daily check-in: settings, due-time logic, cycle and API."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src import auth
from src import main as main_module

TZ = main_module._CHECKIN_TIMEZONE


def _fresh_state(**overrides):
    state = {
        "running": False,
        "last_run_at": 0.0,
        "last_run_date": "",
        "last_trigger": "",
        "summary": None,
    }
    state.update(overrides)
    return state


class AutoCheckinSettingsTests(unittest.TestCase):
    def test_round_trip_persists_and_console_wins_over_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            env_path.write_text("UPSTREAM_MODE=remote\n", "utf-8")
            with (
                patch.object(auth, "ACCOUNTS_PATH", root / "accounts.json"),
                patch.object(auth, "ENV_PATH", env_path),
                patch.object(auth, "_settings", {}),
                patch.dict(
                    os.environ,
                    {"TRAE_AUTO_CHECKIN": "1", "TRAE_AUTO_CHECKIN_TIME": "7:05"},
                    clear=False,
                ),
            ):
                self.assertEqual(
                    auth.get_auto_checkin_settings(),
                    {"enabled": True, "time": "07:05", "source": "env"},
                )
                result = auth.set_auto_checkin_settings(False, "21:40")
                self.assertEqual(result, {"enabled": False, "time": "21:40", "source": "console"})
                # Omitting the time keeps the saved one.
                result = auth.set_auto_checkin_settings(True)
                self.assertEqual(result["time"], "21:40")
                self.assertTrue(result["enabled"])
                text = env_path.read_text("utf-8")
                self.assertIn("TRAE_AUTO_CHECKIN=1", text)
                self.assertIn("TRAE_AUTO_CHECKIN_TIME=21:40", text)
                self.assertIn("UPSTREAM_MODE=remote", text)

    def test_invalid_time_rejected(self):
        for bad in ("24:00", "12:60", "noon", "", "1230"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                auth.normalize_checkin_time(bad)
        self.assertEqual(auth.normalize_checkin_time(" 8:05 "), "08:05")


class AutoCheckinScheduleTests(unittest.TestCase):
    def test_due_and_next_run(self):
        settings = {"enabled": True, "time": "08:30"}
        before = datetime(2026, 10, 1, 8, 0, tzinfo=TZ)
        after = datetime(2026, 10, 1, 9, 0, tzinfo=TZ)
        with patch.object(main_module, "_AUTO_CHECKIN_STATE", _fresh_state()):
            self.assertFalse(main_module._auto_checkin_due(settings, before))
            self.assertTrue(main_module._auto_checkin_due(settings, after))
            self.assertEqual(
                main_module._auto_checkin_next_run(settings, before),
                datetime(2026, 10, 1, 8, 30, tzinfo=TZ),
            )
            # Late start: due right away rather than tomorrow.
            self.assertEqual(main_module._auto_checkin_next_run(settings, after), after)
            self.assertFalse(main_module._auto_checkin_due({"enabled": False, "time": "08:30"}, after))
        with patch.object(
            main_module, "_AUTO_CHECKIN_STATE", _fresh_state(last_run_date="2026-10-01")
        ):
            self.assertFalse(main_module._auto_checkin_due(settings, after))
            self.assertEqual(
                main_module._auto_checkin_next_run(settings, after),
                datetime(2026, 10, 2, 8, 30, tzinfo=TZ),
            )


class AutoCheckinCycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cycle_skips_checked_in_and_tokenless_accounts(self):
        accounts = [
            ("a1", {"token": "t1"}),
            ("a2", {"token": ""}),
            (
                "a3",
                {
                    "token": "t3",
                    "checkin": {"checked_in": True},
                    "checkin_status_updated_at": time.time(),
                },
            ),
            ("a4", {"token": "t4"}),
        ]
        claim = AsyncMock(
            side_effect=lambda aid: {"success": aid == "a1", "error": None if aid == "a1" else "9074"}
        )
        state = _fresh_state()
        with (
            patch.object(main_module, "_AUTO_CHECKIN_STATE", state),
            patch.object(auth, "get_accounts_raw", return_value=accounts),
            patch.object(main_module, "_claim_checkin_account", claim),
        ):
            summary = await main_module._auto_checkin_cycle("schedule")
        self.assertEqual([c.args[0] for c in claim.await_args_list], ["a1", "a4"])
        self.assertEqual(
            summary, {"total": 4, "ok": 1, "skipped": 1, "failed": 1, "no_token": 1}
        )
        self.assertFalse(state["running"])
        self.assertEqual(state["last_trigger"], "schedule")
        self.assertTrue(state["last_run_date"])

    async def test_manual_run_keeps_daily_schedule_slot(self):
        state = _fresh_state()
        with (
            patch.object(main_module, "_AUTO_CHECKIN_STATE", state),
            patch.object(auth, "get_accounts_raw", return_value=[]),
        ):
            await main_module._auto_checkin_cycle("manual")
        self.assertEqual(state["last_trigger"], "manual")
        self.assertEqual(state["last_run_date"], "")


class AutoCheckinApiTests(unittest.TestCase):
    def test_api_round_trip_and_console_markup(self):
        saved = {"enabled": False, "time": "08:30", "source": "console"}

        def fake_set(enabled, time_text=None):
            if time_text:
                saved["time"] = auth.normalize_checkin_time(time_text)
            saved["enabled"] = enabled
            return dict(saved)

        with (
            patch.object(auth, "set_auto_checkin_settings", side_effect=fake_set),
            patch.object(auth, "get_auto_checkin_settings", side_effect=lambda: dict(saved)),
            patch.object(main_module, "_AUTO_CHECKIN_STATE", _fresh_state()),
            patch.dict(os.environ, {"RELAY_API_KEYS": ""}, clear=False),
        ):
            # No ``with``: skip app startup so the real accounts are untouched.
            client = TestClient(main_module.app)
            self.assertEqual(client.post("/api/auto-checkin", json={}).status_code, 400)
            bad = client.post("/api/auto-checkin", json={"enabled": True, "time": "25:00"})
            self.assertEqual(bad.status_code, 400)
            ok = client.post("/api/auto-checkin", json={"enabled": True, "time": "06:15"}).json()
            self.assertTrue(ok["enabled"])
            self.assertEqual(ok["time"], "06:15")
            self.assertTrue(ok["next_run"])
            got = client.get("/api/auto-checkin").json()
            self.assertEqual(got["timezone"], "Asia/Shanghai")
            self.assertTrue(got["enabled"])
            html = client.get("/web/login").text
        self.assertIn('id="auto-checkin-toggle"', html)
        self.assertIn('id="auto-checkin-time"', html)
        self.assertIn("saveAutoCheckin()", html)
        self.assertIn('value="06:15"', html)


if __name__ == "__main__":
    unittest.main()
