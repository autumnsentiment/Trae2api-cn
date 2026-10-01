"""Auto endpoint routing: tools -> IDE Agent, chat -> Remote, fallback Remote."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src import auth
from src import main as main_module

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]
CHAT = [{"role": "user", "content": "hi"}]


class AutoRouteDispatchTests(unittest.IsolatedAsyncioTestCase):
    def _patches(self, enabled: bool, mode: str = "raw"):
        return (
            patch.object(main_module, "_auto_route_enabled", return_value=enabled),
            patch.object(main_module, "UPSTREAM_MODE", mode),
            patch.object(main_module, "_remote_only_models", return_value=set()),
        )

    async def test_tool_request_goes_to_ide_agent(self):
        ide = AsyncMock(return_value={"route": "ide"})
        raw = AsyncMock(side_effect=AssertionError("raw must be skipped"))
        a, b, c = self._patches(True)
        trace = {}
        with a, b, c, patch.object(main_module, "run_ide_chat", ide), patch.object(
            main_module, "run_raw_chat", raw
        ):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False, {"tools": TOOLS, "_upstream_trace": trace}
            )
        self.assertEqual(result, {"route": "ide"})
        self.assertEqual(ide.await_args.args[3]["tools"], TOOLS)
        self.assertTrue(trace["auto_route"])
        self.assertEqual(trace["auto_route_reason"], "tools")
        self.assertEqual(trace["requested_mode"], "ide")

    async def test_tool_history_without_tools_field_counts_as_tools(self):
        ide = AsyncMock(return_value={"route": "ide"})
        messages = [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
        ]
        a, b, c = self._patches(True)
        with a, b, c, patch.object(main_module, "run_ide_chat", ide):
            result = await main_module._dispatch_chat(messages, "glm-5.3", False, {})
        self.assertEqual(result, {"route": "ide"})

    async def test_plain_chat_goes_to_remote(self):
        remote = AsyncMock(return_value={"route": "remote"})
        ide = AsyncMock(side_effect=AssertionError("chat must not use IDE"))
        a, b, c = self._patches(True)
        with a, b, c, patch.object(main_module, "_run_remote_with_retry", remote), patch.object(
            main_module, "run_ide_chat", ide
        ):
            result = await main_module._dispatch_chat(CHAT, "glm-5.3", False, {})
        self.assertEqual(result, {"route": "remote"})
        self.assertEqual(remote.await_args.args[3]["_upstream_mode"], "remote")

    async def test_ide_failure_falls_back_to_remote(self):
        ide = AsyncMock(side_effect=RuntimeError("ide down"))
        remote = AsyncMock(return_value={"route": "remote"})
        a, b, c = self._patches(True)
        with a, b, c, patch.object(main_module, "run_ide_chat", ide), patch.object(
            main_module, "_run_remote_with_retry", remote
        ):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False, {"tools": TOOLS}
            )
        self.assertEqual(result, {"route": "remote"})
        fallback = remote.await_args.args[3]
        self.assertEqual(fallback["_upstream_fallback_from"], "ide")
        self.assertEqual(fallback["tools"], TOOLS)

    async def test_switch_off_keeps_preset_endpoint(self):
        raw = AsyncMock(return_value={"route": "raw"})
        a, b, c = self._patches(False)
        with a, b, c, patch.object(main_module, "run_raw_chat", raw):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False, {"tools": TOOLS}
            )
        self.assertEqual(result, {"route": "raw"})

    async def test_explicit_endpoint_wins_over_switch(self):
        raw = AsyncMock(return_value={"route": "raw"})
        a, b, c = self._patches(True, mode="remote")
        with a, b, c, patch.object(main_module, "run_raw_chat", raw):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False, {"_upstream_mode": "raw"}
            )
        self.assertEqual(result, {"route": "raw"})

    async def test_explicit_auto_route_works_with_switch_off(self):
        ide = AsyncMock(return_value={"route": "ide"})
        a, b, c = self._patches(False, mode="remote")
        with a, b, c, patch.object(main_module, "run_ide_chat", ide):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False, {"_upstream_mode": "auto-route", "tools": TOOLS}
            )
        self.assertEqual(result, {"route": "ide"})

    def test_cli_mode_is_not_rerouted(self):
        with patch.object(main_module, "_auto_route_enabled", return_value=True), patch.object(
            main_module, "UPSTREAM_MODE", "cli"
        ):
            options = main_module._apply_auto_route(CHAT, {})
        self.assertNotIn("_upstream_mode", options)


class AutoRouteSettingsTests(unittest.TestCase):
    def test_switch_persists_and_console_wins_over_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            env_path.write_text("UPSTREAM_MODE=remote\n", "utf-8")
            with (
                patch.object(auth, "ACCOUNTS_PATH", root / "accounts.json"),
                patch.object(auth, "ENV_PATH", env_path),
                patch.object(auth, "_settings", {}),
                patch.dict(os.environ, {"TRAE_AUTO_ROUTE": "1"}, clear=False),
            ):
                self.assertEqual(auth.get_auto_route_settings(), {"enabled": True, "source": "env"})
                result = auth.set_auto_route_settings(False)
                self.assertEqual(result, {"enabled": False, "source": "console"})
                self.assertEqual(os.environ["TRAE_AUTO_ROUTE"], "0")
                self.assertIn("TRAE_AUTO_ROUTE=0", env_path.read_text("utf-8"))
                self.assertIn("UPSTREAM_MODE=remote", env_path.read_text("utf-8"))

    def test_api_round_trip_and_console_markup(self):
        state = {}

        def fake_set(enabled):
            state["enabled"] = enabled
            return {"enabled": enabled, "source": "console"}

        with (
            patch.object(auth, "set_auto_route_settings", side_effect=fake_set),
            patch.object(
                auth,
                "get_auto_route_settings",
                side_effect=lambda: {"enabled": state.get("enabled", False), "source": "console"},
            ),
            patch.dict(os.environ, {"RELAY_API_KEYS": ""}, clear=False),
        ):
            # No ``with``: skip app startup so init_auth never reads the
            # developer's real .env / data/accounts.json.
            client = TestClient(main_module.app)
            if True:
                headers = {"Authorization": "Bearer smoke-key"}
                bad = client.post("/api/auto-route", json={}, headers=headers)
                self.assertEqual(bad.status_code, 400)
                ok = client.post("/api/auto-route", json={"enabled": True}, headers=headers)
                self.assertTrue(ok.json()["enabled"])
                got = client.get("/api/auto-route", headers=headers).json()
                self.assertTrue(got["enabled"])
                self.assertEqual(got["tool_endpoint"], "ide")
                self.assertEqual(got["chat_endpoint"], "remote")
                html = client.get("/web/login", headers=headers).text
        self.assertIn('id="auto-route-toggle"', html)
        self.assertIn("saveAutoRoute()", html)
        self.assertIn('value="auto-route"', html)


if __name__ == "__main__":
    unittest.main()
