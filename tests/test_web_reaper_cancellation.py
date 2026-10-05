"""Use the actual idle reaper and lease registry to test cleanup ownership."""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio

from src import trae_client


class WebReaperCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.enterContext(patch.object(trae_client, "_WEB_LEASES", {}))
        self.enterContext(patch.object(trae_client, "_WEB_SLOTS", {}))
        self.enterContext(patch.object(trae_client, "_WEB_PARALLEL_LIMIT", 1))
        self.enterContext(patch.object(trae_client, "_WEB_IDLE_TIMEOUT", 1))
        self.release = self.enterContext(
            patch.object(
                trae_client,
                "release_web_slot",
                wraps=trae_client.release_web_slot,
            )
        )

    async def _register_idle(
        self, client, *, session_id: str = "idle-session", account_id: str = "account-1"
    ):
        await trae_client.acquire_web_slot(account_id)
        trae_client.register_web_lease(
            account_id,
            session_id,
            "user-message",
            client,
            token="pinned-test-token",
            provider_specific={"tenant": "test"},
        )
        trae_client._WEB_LEASES[session_id]["last_activity"] = time.monotonic() - 2
        self.assertFalse(trae_client.web_slot_available(account_id))

    def _assert_released(self, client, *, account_id: str = "account-1"):
        client.aclose.assert_awaited_once()
        self.release.assert_called_once_with(account_id)
        self.assertTrue(trae_client.web_slot_available(account_id))
        self.assertEqual(trae_client._web_slot(account_id)._value, 1)

    async def test_cancelled_stop_finishes_close_and_slot_release_before_raising(self):
        stopping = asyncio.Event()
        finish_stop = asyncio.Event()
        finished: list[str] = []

        async def stop(*_args, **_kwargs):
            stopping.set()
            await finish_stop.wait()
            finished.append("stop")

        async def close():
            await asyncio.sleep(0)
            finished.append("close")

        client = SimpleNamespace(aclose=AsyncMock(side_effect=close))
        await self._register_idle(client)
        with patch.object(
            trae_client, "stop_web_session", AsyncMock(side_effect=stop)
        ) as stop_mock:
            task = asyncio.create_task(trae_client.reap_idle_web_sessions())
            try:
                await asyncio.wait_for(stopping.wait(), timeout=1)
                self.assertNotIn("idle-session", trae_client._WEB_LEASES)
                task.cancel("reaper shutdown")
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                client.aclose.assert_not_awaited()
                self.release.assert_not_called()
                finish_stop.set()
                with self.assertRaises(asyncio.CancelledError) as cancelled:
                    await task
                self.assertEqual(cancelled.exception.args, ("reaper shutdown",))
                self.assertEqual(finished, ["stop", "close"])
                self._assert_released(client)
                stop_mock.assert_awaited_once_with(
                    client,
                    "idle-session",
                    "user-message",
                    options={
                        "_auth_token": "pinned-test-token",
                        "provider_specific": {"tenant": "test"},
                    },
                )
                self.assertEqual(await trae_client.reap_idle_web_sessions(), 0)
                self._assert_released(client)
            finally:
                finish_stop.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_repeated_cancels_during_stop_and_close_preserve_first_cancel(self):
        stopping = asyncio.Event()
        finish_stop = asyncio.Event()
        closing = asyncio.Event()
        finish_close = asyncio.Event()

        async def stop(*_args, **_kwargs):
            stopping.set()
            await finish_stop.wait()

        async def close():
            closing.set()
            await finish_close.wait()

        client = SimpleNamespace(aclose=AsyncMock(side_effect=close))
        await self._register_idle(client)
        with patch.object(
            trae_client, "stop_web_session", AsyncMock(side_effect=stop)
        ) as stop_mock:
            task = asyncio.create_task(trae_client.reap_idle_web_sessions())
            try:
                await asyncio.wait_for(stopping.wait(), timeout=1)
                task.cancel("first cancel")
                await asyncio.sleep(0)
                task.cancel("second cancel")
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.release.assert_not_called()
                finish_stop.set()
                await asyncio.wait_for(closing.wait(), timeout=1)
                task.cancel("cancel during close")
                await asyncio.sleep(0)
                task.cancel("fourth cancel")
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertFalse(trae_client.web_slot_available("account-1"))
                finish_close.set()
                with self.assertRaises(asyncio.CancelledError) as cancelled:
                    await task
                self.assertEqual(cancelled.exception.args, ("first cancel",))
                stop_mock.assert_awaited_once()
                self._assert_released(client)
            finally:
                finish_stop.set()
                finish_close.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_cancelled_anyio_scope_cannot_interrupt_owned_cleanup(self):
        stopped: list[bool] = []

        async def stop(*_args, **_kwargs):
            scope.cancel()
            await anyio.sleep(0)
            stopped.append(True)

        async def close():
            await anyio.sleep(0)

        client = SimpleNamespace(aclose=AsyncMock(side_effect=close))
        await self._register_idle(client)
        with patch.object(trae_client, "stop_web_session", AsyncMock(side_effect=stop)):
            with anyio.CancelScope() as scope:
                await trae_client.reap_idle_web_sessions()
                self.fail("A cancelled reaper must not return successfully")
        self.assertEqual(stopped, [True])
        self._assert_released(client)

    async def test_stop_exception_does_not_prevent_close_or_slot_release(self):
        client = SimpleNamespace(aclose=AsyncMock())
        await self._register_idle(client)
        with patch.object(
            trae_client, "stop_web_session", AsyncMock(side_effect=RuntimeError("stop failed"))
        ):
            self.assertEqual(await trae_client.reap_idle_web_sessions(), 1)
        self._assert_released(client)

    async def test_close_exception_does_not_prevent_slot_release(self):
        client = SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError("close failed")))
        await self._register_idle(client)
        with patch.object(trae_client, "stop_web_session", AsyncMock()):
            self.assertEqual(await trae_client.reap_idle_web_sessions(), 1)
        self._assert_released(client)

    async def test_actual_stop_request_exception_keeps_its_warning_log(self):
        client = SimpleNamespace(
            post=AsyncMock(side_effect=RuntimeError("stop transport failed")),
            aclose=AsyncMock(),
        )
        await self._register_idle(client)
        with self.assertLogs("src.trae_client", level="WARNING") as logs:
            self.assertEqual(await trae_client.reap_idle_web_sessions(), 1)
        self.assertTrue(
            any(
                "stop web session idle-session failed: stop transport failed" in log
                for log in logs.output
            )
        )
        client.post.assert_awaited_once()
        self._assert_released(client)

    async def test_cancel_does_not_claim_later_idle_leases(self):
        stopping = asyncio.Event()
        finish_stop = asyncio.Event()

        async def stop(*_args, **_kwargs):
            stopping.set()
            await finish_stop.wait()

        first = SimpleNamespace(aclose=AsyncMock())
        second = SimpleNamespace(aclose=AsyncMock())
        await self._register_idle(first)
        await self._register_idle(second, session_id="second-session", account_id="account-2")
        with patch.object(trae_client, "stop_web_session", AsyncMock(side_effect=stop)):
            task = asyncio.create_task(trae_client.reap_idle_web_sessions())
            try:
                await asyncio.wait_for(stopping.wait(), timeout=1)
                task.cancel()
                finish_stop.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self._assert_released(first)
                self.assertIn("second-session", trae_client._WEB_LEASES)
                second.aclose.assert_not_awaited()
                self.assertFalse(trae_client.web_slot_available("account-2"))
                self.assertEqual(await trae_client.reap_idle_web_sessions(), 1)
                second.aclose.assert_awaited_once()
                self.assertTrue(trae_client.web_slot_available("account-2"))
                self.assertEqual(self.release.call_count, 2)
            finally:
                finish_stop.set()
                await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
