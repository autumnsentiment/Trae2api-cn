from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from src import auth, main as main_module


def _token(account_id: str) -> str:
    return f"header.{account_id * 40}.signature"


def _record(account_id: str) -> dict:
    return {
        "user_id": account_id,
        "source": "web-login",
        "token": _token(account_id),
        "expired_at": "2099-01-01T00:00:00Z",
        "provider_specific": {},
        "checkin": {"account_credits": {"remaining": 100}},
    }


class EnterpriseRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.accounts = {
            "personal": _record("personal"),
            "enterprise": _record("enterprise"),
        }
        self.patches = [
            patch.object(auth, "_accounts", self.accounts),
            patch.object(auth, "_active_account", "personal"),
            patch.object(auth, "_poll_enabled", False),
            patch.object(auth, "_polling_mode", "round-robin"),
            patch.object(auth, "_MODEL_ACCOUNT_INFLIGHT", {}),
            patch.object(auth, "_MODEL_ACCOUNT_RESERVED", {}),
            patch.object(auth, "_MODEL_ACCOUNT_RESERVATION_OWNERS", {}),
            patch.object(auth, "_save_accounts"),
            patch.object(main_module, "UPSTREAM_MODE", "remote"),
            patch.object(
                auth,
                "get_enterprise_settings",
                return_value={
                    "enabled": True,
                    "account_id": "enterprise",
                    "source": "console",
                },
            ),
        ]
        for item in self.patches:
            item.start()
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()

    def tearDown(self):
        main_module._CHAT_HISTORY_SESSIONS.clear()
        main_module._UPSTREAM_SESSION_LEASES.clear()
        for item in reversed(self.patches):
            item.stop()

    def test_enterprise_route_pins_raw_and_disables_fallback(self):
        options = main_module._apply_auto_route(
            [{"role": "user", "content": "hello"}],
            {"_upstream_mode": "remote"},
        )
        self.assertEqual(options["_upstream_mode"], "raw")
        self.assertTrue(options["_disable_upstream_fallback"])
        self.assertEqual(options["_auto_route"], "enterprise")

    def test_enterprise_session_is_pinned_and_does_not_change_personal_active(self):
        first = main_module._bind_chat_session(
            [{"role": "user", "content": "first"}],
            {},
            requested_session_id="shared-session",
        )
        self.assertEqual(first["_account_id"], "enterprise")
        self.assertTrue(first["_enterprise_mode"])
        self.assertEqual(auth.get_active_account_id(), "personal")
        enterprise_session = first["session_id"]
        self.assertNotEqual(enterprise_session, "shared-session")
        self.assertTrue(main_module._cancel_unclaimed_account_slot(first))

        continuation = main_module._bind_chat_session(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "thinking"},
                {"role": "user", "content": "continue"},
            ],
            {},
            requested_session_id="shared-session",
        )
        self.assertEqual(continuation["session_id"], enterprise_session)
        self.assertEqual(continuation["_account_id"], "enterprise")
        self.assertTrue(main_module._cancel_unclaimed_account_slot(continuation))

    async def test_enterprise_dispatch_does_not_fallback_to_personal_endpoints(self):
        raw = AsyncMock(side_effect=RuntimeError("enterprise raw unavailable"))
        ide = AsyncMock(side_effect=AssertionError("enterprise must not use IDE"))
        remote = AsyncMock(side_effect=AssertionError("enterprise must not use Remote"))
        with (
            patch.object(main_module, "run_raw_chat", raw),
            patch.object(main_module, "run_ide_chat", ide),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "hello"}],
                "glm-5.3",
                False,
                {},
            )
        self.assertEqual(result.status_code, 502)
        raw.assert_awaited_once()
        ide.assert_not_awaited()
        remote.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
