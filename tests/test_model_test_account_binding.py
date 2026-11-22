"""Keep model probes on the normal initial-account scheduler."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.responses import JSONResponse

from src import auth, main
from src.sse import UpstreamEventError


def _record(account_id, credits):
    return {
        "user_id": account_id,
        "token": f"header.{account_id}{'p' * 140}.signature",
        "source": "web-login",
        "expired_at": "2099-01-01T00:00:00Z",
        "provider_specific": {"account": account_id},
        "checkin": {"account_credits": {"remaining": credits}},
    }


def _request(**body):
    return Mock(json=AsyncMock(return_value={"model": "glm-5.2", **body}))


def _reply():
    return JSONResponse(
        {"choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}]}
    )


class ModelTestAccountBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.accounts = {
            "rich": _record("rich", 100),
            "poor": _record("poor", 10),
        }
        self.patches = [
            patch.object(auth, "_accounts", self.accounts),
            patch.object(auth, "_active_account", "rich"),
            patch.object(auth, "_poll_enabled", True),
            patch.object(auth, "_polling_mode", "credit-priority"),
            patch.object(auth, "_rotation_cursor", 0),
            patch.object(auth, "_MODEL_ACCOUNT_INFLIGHT", {}),
            patch.object(auth, "_MODEL_ACCOUNT_RESERVED", {}),
            patch.object(auth, "_MODEL_ACCOUNT_RESERVATION_OWNERS", {}),
            patch.object(auth, "_save_accounts"),
            patch.object(auth, "_switch_record"),
            patch.object(auth, "get_enterprise_settings", return_value={"enabled": False}),
            patch.object(main, "_current_upstream_mode", return_value="ide"),
            patch.object(main, "_CHAT_HISTORY_SESSIONS", main.OrderedDict()),
            patch.object(main, "_UPSTREAM_SESSION_LEASES", main.OrderedDict()),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    async def test_overlapping_isolated_probes_select_available_initial_slots(self):
        release = asyncio.Event()
        all_entered = asyncio.Event()
        selected = []

        async def dispatch(_messages, _model, stream, options):
            self.assertFalse(stream)
            self.assertTrue(options["_disable_upstream_fallback"])
            selected.append(dict(options))
            if len(selected) == 3:
                all_entered.set()
            await release.wait()
            return _reply()

        with patch.object(main, "_dispatch_chat", side_effect=dispatch) as dispatch_mock:
            tasks = [
                asyncio.create_task(
                    main.api_model_test(
                        _request(endpoint="ide", disable_fallback=True, mode="tool")
                    )
                )
                for _ in range(3)
            ]
            try:
                await asyncio.wait_for(all_entered.wait(), timeout=1)
                self.assertEqual(
                    [options["_account_id"] for options in selected],
                    ["rich", "rich", "poor"],
                )
                self.assertEqual(auth.account_capacity_snapshot("rich")["inflight"], 2)
                self.assertEqual(auth.account_capacity_snapshot("poor")["inflight"], 1)
                self.assertEqual(
                    len({options["_account_slot_reservation_id"] for options in selected}),
                    3,
                )
            finally:
                release.set()
                await asyncio.gather(*tasks)

        self.assertEqual(dispatch_mock.await_count, 3)
        self.assertEqual(auth.account_capacity_snapshot("rich")["inflight"], 0)
        self.assertEqual(auth.account_capacity_snapshot("poor")["inflight"], 0)

    async def test_isolated_4011_does_not_retry_or_release_another_request_slot(self):
        account_id, reservation = auth.reserve_model_account()
        error = UpstreamEventError("concurrency limit", code=4011)
        dispatch = AsyncMock(side_effect=error)
        next_account = Mock()
        with (
            patch.object(main, "_dispatch_chat", dispatch),
            patch.object(main, "_next_retry_account_snapshot", next_account),
        ):
            response = await main.api_model_test(
                _request(endpoint="ide", disable_fallback=True)
            )

        payload = json.loads(response.body)
        self.assertFalse(payload["success"])
        self.assertIn("4011", payload["error"])
        dispatch.assert_awaited_once()
        next_account.assert_not_called()
        options = dispatch.await_args.args[3]
        self.assertEqual(options["_account_id"], "rich")
        self.assertTrue(options["_disable_upstream_fallback"])
        self.assertEqual(auth.account_capacity_snapshot("rich")["inflight"], 1)
        self.assertIn(
            reservation["_slot_reservation_id"],
            auth._MODEL_ACCOUNT_RESERVATION_OWNERS,
        )
        self.assertTrue(
            auth.cancel_account_slot_reservation(
                account_id, reservation["_slot_reservation_id"]
            )
        )

    async def test_failed_and_timed_out_probes_release_unclaimed_reservations(self):
        for error in (RuntimeError("failed"), TimeoutError()):
            with self.subTest(error=type(error).__name__):
                dispatch = AsyncMock(side_effect=error)
                with patch.object(main, "_dispatch_chat", dispatch):
                    response = await main.api_model_test(_request(disable_fallback=True))
                self.assertFalse(json.loads(response.body)["success"])
                dispatch.assert_awaited_once()
                self.assertEqual(auth.account_capacity_snapshot("rich")["inflight"], 0)
                self.assertFalse(auth._MODEL_ACCOUNT_RESERVATION_OWNERS)

    async def test_cancelled_probe_releases_unclaimed_reservation(self):
        entered = asyncio.Event()

        async def dispatch(*_args):
            entered.set()
            await asyncio.Event().wait()

        with patch.object(main, "_dispatch_chat", side_effect=dispatch):
            task = asyncio.create_task(main.api_model_test(_request(disable_fallback=True)))
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(auth.account_capacity_snapshot("rich")["inflight"], 0)
        self.assertFalse(auth._MODEL_ACCOUNT_RESERVATION_OWNERS)

    async def test_enterprise_probe_binds_its_config_without_rotating_active_account(self):
        dispatch = AsyncMock(return_value=_reply())
        with (
            patch.object(
                auth,
                "get_enterprise_settings",
                return_value={"enabled": True, "account_id": "poor"},
            ),
            patch.object(main, "_dispatch_chat", dispatch),
        ):
            response = await main.api_model_test(
                _request(endpoint="ide", disable_fallback=True)
            )

        self.assertTrue(json.loads(response.body)["success"])
        options = dispatch.await_args.args[3]
        self.assertEqual(options["_account_id"], "poor")
        self.assertEqual(options["_auth_token"], self.accounts["poor"]["token"])
        self.assertEqual(options["_enterprise_account_id"], "poor")
        self.assertEqual(options["config_source"], 2)
        self.assertEqual(options["_traecode_raw_chat_function"], "chat_v3")
        self.assertEqual(auth._active_account, "rich")
        self.assertEqual(auth.account_capacity_snapshot("poor")["inflight"], 0)

    async def test_unavailable_enterprise_account_is_reported_before_dispatch(self):
        dispatch = AsyncMock()
        with (
            patch.object(
                auth,
                "get_enterprise_settings",
                return_value={"enabled": True, "account_id": "missing"},
            ),
            patch.object(main, "_dispatch_chat", dispatch),
        ):
            response = await main.api_model_test(_request(disable_fallback=True))

        self.assertFalse(json.loads(response.body)["success"])
        dispatch.assert_not_awaited()
        self.assertFalse(auth._MODEL_ACCOUNT_RESERVATION_OWNERS)

    async def test_empty_account_store_keeps_cli_probes_usable(self):
        dispatch = AsyncMock(return_value=_reply())
        with (
            patch.object(auth, "_accounts", {}),
            patch.object(auth, "_active_account", None),
            patch.object(auth, "get_active_account_id", return_value=""),
            patch.object(auth, "get_active_account_snapshot", return_value=("", {})),
            patch.object(auth, "get_model_account_snapshot", return_value=("", {})),
            patch.object(auth, "get_token", return_value=""),
            patch.object(auth, "reserve_model_account") as reserve,
            patch.object(main, "_current_upstream_mode", return_value="cli"),
            patch.object(main, "_dispatch_chat", dispatch),
        ):
            response = await main.api_model_test(
                _request(endpoint="cli", disable_fallback=True)
            )

        self.assertTrue(json.loads(response.body)["success"])
        dispatch.assert_awaited_once()
        reserve.assert_not_called()
        self.assertTrue(dispatch.await_args.args[3]["session_id"])
