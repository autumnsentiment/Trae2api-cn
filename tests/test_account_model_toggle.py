"""Per-account model-request switch regression tests.

The switch is deliberately independent from daily check-in.  These tests use
synthetic, model-shaped JWT strings only; no real credentials or repository
``data/`` files are touched.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src import auth
from src import main as main_module


_UNSET = object()


def _token(account_id: str) -> str:
    # _record_model_valid only needs a JWT-shaped value of sufficient length.
    return f"header.{account_id}{'p' * 140}.signature"


def _record(account_id: str, *, enabled=_UNSET, checked_in: bool = False) -> dict:
    record = {
        "user_id": account_id,
        "label": account_id,
        "token": _token(account_id),
        "expired_at": "2099-01-01T00:00:00Z",
        "source": "web-login",
        "provider_specific": {},
        "checkin": {"checked_in": checked_in},
    }
    if enabled is not _UNSET:
        record["model_enabled"] = enabled
    return record

class AccountModelToggleAuthTests(unittest.TestCase):
    def test_legacy_account_defaults_to_model_enabled(self):
        record = _record("legacy")
        with patch.object(auth, "_accounts", {"legacy": record}):
            row = auth.list_accounts()[0]
        self.assertTrue(row["model_enabled"])
        self.assertTrue(row["model_eligible"])

    def test_toggle_persists_and_reimport_keeps_disabled_state(self):
        record = _record("account-a")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "accounts.json"
            with (
                patch.object(auth, "ACCOUNTS_PATH", path),
                patch.object(auth, "_accounts", {"account-a": record}),
                patch.object(auth, "_active_account", "account-a"),
                patch.object(auth, "_save_env_snapshot"),
            ):
                self.assertTrue(auth.set_account_model_enabled("account-a", False))
                persisted = json.loads(path.read_text("utf-8"))
                self.assertFalse(
                    persisted["accounts"]["account-a"]["model_enabled"]
                )

                # Re-adding credentials for the same account must not reset the
                # operator's model-request preference.
                auth.add_account(
                    {
                        "user_id": "account-a",
                        "token": _token("account-a-new"),
                        "source": "web-login",
                    }
                )
                self.assertFalse(auth.get_account_record("account-a")["model_enabled"])

    def test_polling_skips_disabled_accounts_but_keeps_enabled_account(self):
        accounts = {
            "disabled-a": _record("disabled-a", enabled=False),
            "enabled-b": _record("enabled-b", enabled=True),
            "disabled-c": _record("disabled-c", enabled=False),
        }
        with (
            patch.object(auth, "_accounts", accounts),
            patch.object(auth, "_active_account", "disabled-a"),
            patch.object(auth, "_poll_enabled", True),
            patch.object(auth, "_rotation_cursor", 0),
            patch.object(auth, "_save_accounts"),
        ):
            auth.next_polling_account()
            self.assertEqual(auth.get_active_account_id(), "enabled-b")

    def test_all_disabled_rejects_new_session(self):
        accounts = {"account-a": _record("account-a", enabled=False)}
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()
        try:
            with (
                patch.object(auth, "_accounts", accounts),
                patch.object(auth, "_active_account", "account-a"),
                patch.object(auth, "_poll_enabled", False),
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
            ):
                with self.assertRaises(RuntimeError):
                    main_module._bind_chat_session(
                        [{"role": "user", "content": "hello"}],
                        {},
                        requested_session_id="disabled-new-session",
                    )
        finally:
            main_module._CHAT_HISTORY_SESSIONS.clear()
            main_module._UPSTREAM_SESSION_LEASES.clear()

    def test_existing_session_is_rejected_after_account_is_disabled(self):
        accounts = {"account-a": _record("account-a", enabled=True)}
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()
        try:
            with (
                patch.object(auth, "_accounts", accounts),
                patch.object(auth, "_active_account", "account-a"),
                patch.object(auth, "_poll_enabled", False),
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(auth, "_save_accounts"),
            ):
                main_module._bind_chat_session(
                    [{"role": "user", "content": "first turn"}],
                    {},
                    requested_session_id="pinned-session",
                )
                self.assertTrue(
                    auth.set_account_model_enabled("account-a", False)
                )
                with self.assertRaises(RuntimeError):
                    main_module._bind_chat_session(
                        [
                            {"role": "user", "content": "first turn"},
                            {"role": "assistant", "content": "answer"},
                            {"role": "user", "content": "next turn"},
                        ],
                        {},
                        requested_session_id="pinned-session",
                    )
        finally:
            main_module._CHAT_HISTORY_SESSIONS.clear()
            main_module._UPSTREAM_SESSION_LEASES.clear()


class AccountModelToggleApiTests(unittest.TestCase):
    def _client(self, accounts):
        self._patches = [
            patch.object(auth, "_accounts", accounts),
            patch.object(auth, "_active_account", next(iter(accounts), "")),
            patch.object(auth, "_poll_enabled", False),
            patch.dict("os.environ", {"RELAY_API_KEYS": ""}, clear=False),
        ]
        for item in self._patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(self._patches)])
        return TestClient(main_module.app, raise_server_exceptions=False)

    def test_model_enabled_api_rejects_non_boolean_container_values(self):
        client = self._client({"account-a": _record("account-a", enabled=True)})
        for value in ([], {}, 1, 0):
            with self.subTest(value=value):
                response = client.post(
                    "/api/accounts/model-enabled",
                    json={"account_id": "account-a", "enabled": value},
                )
                self.assertEqual(response.status_code, 400)

    def test_chat_and_responses_return_503_when_all_accounts_disabled(self):
        client = self._client({"account-a": _record("account-a", enabled=False)})
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "glm-5.3",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        responses = client.post(
            "/v1/responses",
            json={"model": "glm-5.3", "input": "hello"},
        )
        self.assertEqual(chat.status_code, 503)
        self.assertEqual(responses.status_code, 503)
        self.assertIn("enabled", chat.text.lower())
        self.assertIn("enabled", responses.text.lower())


class DisabledAccountCheckinTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_and_scheduled_checkin_still_include_disabled_accounts(self):
        raw_accounts = [
            ("disabled-a", _record("disabled-a", enabled=False)),
            ("enabled-b", _record("enabled-b", enabled=True)),
        ]
        claimed = AsyncMock(
            side_effect=lambda account_id: {
                "success": True,
                "skipped": False,
                "id": account_id,
            }
        )
        with (
            patch.object(auth, "get_accounts_raw", return_value=raw_accounts),
            patch.object(auth, "get_active_account_id", return_value="enabled-b"),
            patch.object(main_module, "_claim_checkin_account", claimed),
            patch.object(main_module, "_checkin_cache_is_today", return_value=False),
            patch.object(main_module, "_AUTO_CHECKIN_STATE", {
                "running": False,
                "last_run_at": 0.0,
                "last_run_date": "",
                "last_trigger": "",
                "summary": None,
            }),
        ):
            manual = await main_module.api_checkin_claim_all()
            self.assertEqual(
                [call.args[0] for call in claimed.await_args_list],
                ["disabled-a", "enabled-b"],
            )
            claimed.reset_mock()
            scheduled = await main_module._auto_checkin_cycle("schedule")

        self.assertEqual(
            [call.args[0] for call in claimed.await_args_list],
            ["disabled-a", "enabled-b"],
        )
        self.assertEqual(manual.status_code, 200)
        self.assertEqual(scheduled["ok"], 2)


if __name__ == "__main__":
    unittest.main()
