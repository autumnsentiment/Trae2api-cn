"""Tool transport constraints and optional image/text auto routing."""

from __future__ import annotations

import json
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
IMAGE_INPUTS = {
    "openai": {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    },
    "anthropic": {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "QUJD",
        },
    },
}


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

    async def test_image_requests_go_to_ide_without_changing_explicit_model(self):
        for source, image_part in IMAGE_INPUTS.items():
            with self.subTest(source=source):
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this."}, image_part],
                }]
                ide = AsyncMock(return_value={"route": "ide"})
                remote = AsyncMock(side_effect=AssertionError("images must use IDE"))
                raw = AsyncMock(side_effect=AssertionError("raw must be skipped"))
                trace = {}
                a, b, c = self._patches(True, mode="remote")
                with (
                    a, b, c,
                    patch.dict(os.environ, {"TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT": "0"}),
                    patch.object(main_module, "run_ide_chat", ide),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                    patch.object(main_module, "run_raw_chat", raw),
                ):
                    result = await main_module._dispatch_chat(
                        messages, "glm-5.2", False, {"_upstream_trace": trace}
                    )
                self.assertEqual(result, {"route": "ide"})
                ide.assert_awaited_once()
                self.assertEqual(ide.await_args.args[0], messages)
                self.assertEqual(ide.await_args.args[1], "glm-5.2")
                self.assertEqual(ide.await_args.args[3]["_upstream_mode"], "ide")
                self.assertEqual(trace["auto_route_reason"], "images")
                self.assertEqual(trace["effective_model"], "glm-5.2")
                remote.assert_not_awaited()
                raw.assert_not_awaited()

    async def test_image_explicit_remote_endpoint_is_not_rerouted(self):
        messages = [{"role": "user", "content": [IMAGE_INPUTS["openai"]]}]
        remote = AsyncMock(return_value={"route": "remote"})
        ide = AsyncMock(side_effect=AssertionError("explicit remote must win"))
        a, b, c = self._patches(True)
        with (
            a, b, c,
            patch.dict(os.environ, {"TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT": "0"}),
            patch.object(main_module, "_run_remote_with_retry", remote),
            patch.object(main_module, "run_ide_chat", ide),
        ):
            result = await main_module._dispatch_chat(
                messages, "glm-5.2", False, {"_upstream_mode": "remote"}
            )
        self.assertEqual(result, {"route": "remote"})
        remote.assert_awaited_once()
        self.assertEqual(remote.await_args.args[0], messages)
        self.assertEqual(remote.await_args.args[1], "glm-5.2")
        self.assertEqual(remote.await_args.args[3]["_upstream_mode"], "remote")
        self.assertNotIn("_auto_route", remote.await_args.args[3])
        ide.assert_not_awaited()

    async def test_image_switch_off_keeps_preset_endpoint(self):
        messages = [{"role": "user", "content": [IMAGE_INPUTS["openai"]]}]
        for mode in ("raw", "remote"):
            with self.subTest(mode=mode):
                raw = AsyncMock(return_value={"route": "raw"})
                remote = AsyncMock(return_value={"route": "remote"})
                ide = AsyncMock(side_effect=AssertionError("routing is disabled"))
                a, b, c = self._patches(False, mode=mode)
                with (
                    a, b, c,
                    patch.dict(os.environ, {"TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT": "0"}),
                    patch.object(main_module, "run_raw_chat", raw),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                    patch.object(main_module, "run_ide_chat", ide),
                ):
                    result = await main_module._dispatch_chat(
                        messages, "glm-5.2", False, {}
                    )
                self.assertEqual(result, {"route": mode})
                selected = raw if mode == "raw" else remote
                skipped = remote if mode == "raw" else raw
                selected.assert_awaited_once()
                self.assertEqual(selected.await_args.args[0], messages)
                self.assertEqual(selected.await_args.args[1], "glm-5.2")
                self.assertNotIn("_auto_route", selected.await_args.args[3])
                skipped.assert_not_awaited()
                ide.assert_not_awaited()

    async def test_image_ide_failure_does_not_fallback_to_remote_query_limit(self):
        """An image auto-route failure must preserve the IDE error.

        The image payload is intentionally not sent through Remote as a
        fallback: flattening its base64 data can exceed Remote's hard query
        limit and turns the useful IDE failure into a misleading 413.
        """

        large_image = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + ("A" * 600_000),
            },
        }
        messages = [{"role": "user", "content": [large_image]}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                ide = AsyncMock(side_effect=RuntimeError("ide down"))
                remote = AsyncMock(
                    side_effect=main_module.RemoteQueryTooLarge(2_895_151, 480_000)
                )
                a, b, c = self._patches(True, mode="remote")
                with (
                    a, b, c,
                    patch.dict(
                        os.environ,
                        {"TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT": "0"},
                    ),
                    patch.object(main_module, "run_ide_chat", ide),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                    patch.object(main_module, "run_remote_session", remote),
                    patch.object(
                        main_module.auth,
                        "get_polling_status",
                        return_value={"enabled": True},
                    ),
                    patch.object(main_module, "_polling_retry_limit", return_value=3),
                    patch.object(
                        main_module, "_next_retry_account_snapshot"
                    ) as next_account,
                    patch.object(main_module.auth, "next_polling_account") as rotate,
                ):
                    result = await main_module._dispatch_chat(
                        messages, "glm-5.2", stream, {}
                    )
                self.assertEqual(result.status_code, 502)
                error = json.loads(result.body)["error"]
                self.assertEqual(error["type"], "api_error")
                self.assertIn("ide", error["message"].lower())
                self.assertNotEqual(result.status_code, 413)
                ide.assert_awaited_once()
                remote.assert_not_awaited()
                next_account.assert_not_called()
                rotate.assert_not_called()

    async def test_small_image_ide_failure_can_fallback_to_remote(self):
        """Images that fit Remote's cap retain the normal endpoint fallback."""

        messages = [{"role": "user", "content": [IMAGE_INPUTS["openai"]]}]
        ide = AsyncMock(side_effect=RuntimeError("ide down"))
        remote = AsyncMock(return_value={"route": "remote"})
        a, b, c = self._patches(True, mode="remote")
        with (
            a,
            b,
            c,
            patch.dict(
                os.environ,
                {"TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT": "0"},
            ),
            patch.object(main_module, "run_ide_chat", ide),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                messages, "glm-5.2", False, {}
            )
        self.assertEqual(result, {"route": "remote"})
        ide.assert_awaited_once()
        remote.assert_awaited_once()
        fallback = remote.await_args.args[3]
        self.assertEqual(fallback["_upstream_fallback_from"], "ide")

    async def test_ide_failure_does_not_fallback_to_remote_for_tools(self):
        ide = AsyncMock(side_effect=RuntimeError("ide down"))
        remote = AsyncMock(return_value={"route": "remote"})
        a, b, c = self._patches(True)
        with a, b, c, patch.object(main_module, "run_ide_chat", ide), patch.object(
            main_module, "_run_remote_with_retry", remote
        ):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False, {"tools": TOOLS}
            )
        self.assertEqual(result.status_code, 502)
        remote.assert_not_awaited()

    async def test_switch_off_still_routes_tools_to_ide(self):
        messages = [
            {"role": "user", "content": "Read README.md"},
            {"role": "assistant", "tool_calls": [{
                "id": "call-readme", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-readme", "content": "Project readme"},
        ]
        cases = [(mode, {}) for mode in ("remote", "raw", "direct", "auto")]
        cases.append(("remote", {"_upstream_mode": "auto"}))
        for mode, override in cases:
            with self.subTest(mode=mode, override=override):
                ide = AsyncMock(return_value={"route": "ide"})
                raw = AsyncMock(side_effect=AssertionError("tool requests must skip the text bridge"))
                remote = AsyncMock(side_effect=AssertionError("remote cannot own caller tools"))
                a, b, c = self._patches(False, mode=mode)
                with (
                    a, b, c,
                    patch.object(main_module, "run_ide_chat", ide),
                    patch.object(main_module, "run_raw_chat", raw),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                ):
                    result = await main_module._dispatch_chat(
                        messages, "glm-5.3", False, {"tools": TOOLS, **override}
                    )
                self.assertEqual(result, {"route": "ide"})
                ide.assert_awaited_once()
                options = ide.await_args.args[3]
                self.assertEqual(options["_upstream_mode"], "ide")
                self.assertTrue(options["_traecode_native_tools"])
                body = main_module.raw_client.build_traecode_chat_body(
                    messages, "glm-5.3", options=options
                )
                self.assertEqual(body["tools"][0]["function"]["parameters"], TOOLS[0]["function"]["parameters"])
                self.assertEqual(body["messages"][1]["tool_calls"][0]["id"], "call-readme")
                self.assertEqual(body["messages"][2]["role"], "tool")
                self.assertEqual(body["messages"][2]["tool_call_id"], "call-readme")
                raw.assert_not_awaited()
                remote.assert_not_awaited()

    async def test_explicit_remote_with_tools_is_corrected_to_ide(self):
        ide = AsyncMock(return_value={"route": "ide"})
        remote = AsyncMock(side_effect=AssertionError("remote cannot own caller tools"))
        a, b, c = self._patches(False, mode="remote")
        with a, b, c, patch.object(main_module, "run_ide_chat", ide), patch.object(
            main_module, "_run_remote_with_retry", remote
        ):
            result = await main_module._dispatch_chat(
                CHAT, "glm-5.3", False,
                {"_upstream_mode": "remote", "tools": TOOLS},
            )
        self.assertEqual(result, {"route": "ide"})
        ide.assert_awaited_once()
        remote.assert_not_awaited()

    async def test_explicit_endpoint_wins_over_switch(self):
        a, b, c = self._patches(True, mode="remote")
        for mode in ("raw", "direct"):
            with self.subTest(mode=mode):
                selected = AsyncMock(return_value={"route": mode})
                with a, b, c, patch.object(main_module, "run_raw_chat", selected):
                    result = await main_module._dispatch_chat(
                        CHAT, "glm-5.3", False, {"_upstream_mode": mode}
                    )
                self.assertEqual(result, {"route": mode})
                selected.assert_awaited_once()

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

    def _console_html(self, enabled: bool) -> str:
        with (
            patch.object(
                auth,
                "get_auto_route_settings",
                side_effect=lambda: {"enabled": enabled, "source": "console"},
            ),
            patch.dict(os.environ, {"RELAY_API_KEYS": ""}, clear=False),
        ):
            client = TestClient(main_module.app)
            return client.get("/web/login").text

    def test_console_renders_endpoint_lock_contract(self):
        """The settings panel must expose the lock hooks the JS drives."""

        html = self._console_html(True)
        # Every element ``applyEndpointLock`` toggles must exist, otherwise the
        # lock silently no-ops and the form keeps accepting a dead selection.
        for marker in (
            'id="endpoint-fields"',
            'id="endpoint-lock-note"',
            'id="endpoint-lock-state"',
            'id="settings-endpoint-preset"',
            'id="settings-web"',
            'id="settings-model-base"',
        ):
            self.assertIn(marker, html)
        self.assertIn("function applyEndpointLock(enabled)", html)
        self.assertIn("applyEndpointLock(true)", html)
        self.assertIn("applyEndpointLock(d.enabled)", html)
        # The lock must be visible before the first paint, not only after a save.
        self.assertIn('class="state-pill lock"', html)

    def test_console_endpoint_lock_follows_auto_route_state(self):
        on_html = self._console_html(True)
        off_html = self._console_html(False)
        self.assertIn("applyEndpointLock(true)", on_html)
        self.assertIn("applyEndpointLock(false)", off_html)
        self.assertIn("自动路由已接管", on_html)
        self.assertIn("手动选择", off_html)
        self.assertIn('id="endpoint-fields" class="endpoint-locked"', on_html)
        self.assertIn('id="endpoint-fields" class=""', off_html)
        # The persisted enabled state must be non-interactive on the first
        # response, before the browser runs applyEndpointLock().
        for element_id in (
            "settings-endpoint-preset",
            "settings-web",
            "settings-model-base",
        ):
            self.assertRegex(
                on_html,
                rf'<(?:select|input)[^>]*id="{element_id}"[^>]*\sdisabled(?:\s|=|>)',
            )
            self.assertNotRegex(
                off_html,
                rf'<(?:select|input)[^>]*id="{element_id}"[^>]*(?<!aria-)\sdisabled(?:\s|=|>)',
            )


if __name__ == "__main__":
    unittest.main()
