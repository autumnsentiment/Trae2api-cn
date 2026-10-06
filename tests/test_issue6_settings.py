import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from src import auth
from src import main as main_module


class Issue6SettingsTests(unittest.TestCase):
    def test_settings_api_round_trip_preserves_shared_url_preset_identity(self):
        """IDE/Work presets must survive the settings API, not just the UI."""

        saved = {}

        def save_settings(**values):
            saved.update(values)

        def get_settings():
            return {
                "web_base_url": saved.get("web_base_url", ""),
                "upstream_mode": saved.get("upstream_mode", ""),
                "endpoint_id": saved.get("endpoint_id", ""),
                "relay_port": 0,
            }

        old_mode = main_module.UPSTREAM_MODE
        old_env_mode = os.environ.get("UPSTREAM_MODE")
        try:
            with (
                patch.object(main_module.auth, "set_relay_settings", side_effect=save_settings),
                patch.object(main_module.auth, "get_settings", side_effect=get_settings),
                patch.object(main_module, "UPSTREAM_MODE", "remote"),
                patch.dict(os.environ, {"RELAY_API_KEYS": "", "UPSTREAM_MODE": "remote"}, clear=False),
            ):
                client = TestClient(main_module.app)
                ide = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://trae-api-cn.mchost.guru",
                        "upstream_mode": "ide",
                        "endpoint_id": "ide",
                    },
                )
                self.assertEqual(ide.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "ide")
                self.assertEqual(saved["upstream_mode"], "ide")
                self.assertEqual(ide.json()["settings"]["endpoint_id"], "ide")
                self.assertEqual(main_module.UPSTREAM_MODE, "ide")

                solo = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://trae-api-cn.mchost.guru",
                        "upstream_mode": "solo",
                        "endpoint_id": "solo",
                    },
                )
                self.assertEqual(solo.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "solo")
                self.assertEqual(saved["upstream_mode"], "solo")
                self.assertEqual(solo.json()["settings"]["endpoint_id"], "solo")

                work = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
                        "upstream_mode": "work-agent",
                        "endpoint_id": "agent",
                    },
                )
                self.assertEqual(work.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "agent")
                self.assertEqual(saved["upstream_mode"], "work-agent")
                self.assertEqual(work.json()["settings"]["endpoint_id"], "agent")
                self.assertEqual(main_module.UPSTREAM_MODE, "work-agent")

                # A stale URL/mode must not silently overwrite the selected
                # preset and turn Work Agent back into the Remote preset.
                mismatch = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
                        "upstream_mode": "remote",
                        "endpoint_id": "agent",
                    },
                )
                self.assertEqual(mismatch.status_code, 400)
        finally:
            main_module.UPSTREAM_MODE = old_mode
            if old_env_mode is None:
                os.environ.pop("UPSTREAM_MODE", None)
            else:
                os.environ["UPSTREAM_MODE"] = old_env_mode

    def test_solo_preset_points_to_the_verified_llm_utils_chat_url(self):
        solo = next(
            item
            for item in main_module.UPSTREAM_ENDPOINT_PRESETS
            if item["id"] == "solo"
        )
        self.assertEqual(
            solo["endpoint_url"],
            "https://trae-api-cn.mchost.guru/api/agent/v3/llm_utils_chat",
        )
        self.assertEqual(solo["endpoint_path"], "/api/agent/v3/llm_utils_chat")
        self.assertEqual(solo["base_url"], "https://trae-api-cn.mchost.guru")
        # Solo is dispatched by the existing IDE client, which already builds
        # the same native function/body contract used by Jeff's implementation.
        self.assertEqual(solo["mode"], "solo")

    def test_ide_raw_is_not_a_formal_endpoint_preset(self):
        """Personal accounts cannot satisfy Raw's enterprise app-config check."""

        self.assertNotIn(
            "raw",
            {str(item.get("id") or "").strip().lower() for item in main_module.UPSTREAM_ENDPOINT_PRESETS},
        )

    def test_legacy_settings_update_migrates_or_clears_stale_endpoint_id(self):
        """URL/mode-only callers must not resurrect a previous shared URL tier."""

        saved = {
            "web_base_url": "https://trae-api-cn.mchost.guru",
            "upstream_mode": "ide",
            "endpoint_id": "ide",
        }

        def save_settings(**values):
            saved.update(values)

        def get_settings():
            return {
                "web_base_url": saved.get("web_base_url", ""),
                "upstream_mode": saved.get("upstream_mode", ""),
                "endpoint_id": saved.get("endpoint_id", ""),
                "relay_port": 0,
            }

        old_mode = main_module.UPSTREAM_MODE
        try:
            with (
                patch.object(main_module.auth, "set_relay_settings", side_effect=save_settings),
                patch.object(main_module.auth, "get_settings", side_effect=get_settings),
                patch.object(main_module, "UPSTREAM_MODE", "ide"),
                patch.dict(os.environ, {"RELAY_API_KEYS": "", "UPSTREAM_MODE": "ide"}, clear=False),
            ):
                client = TestClient(main_module.app)

                # A legacy URL/mode-only Work Agent update can be mapped to its
                # unambiguous preset even though it shares Remote's URL.
                response = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
                        "upstream_mode": "work-agent",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "agent")

                # A custom URL cannot identify a preset; explicitly clear the
                # old Work Agent id so it cannot be restored after restart.
                response = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://custom.example/v1",
                        "upstream_mode": "remote",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "")
        finally:
            main_module.UPSTREAM_MODE = old_mode

    def test_legacy_endpoint_update_migrates_preset_but_port_only_keeps_it(self):
        """URL/mode-only clients must preserve an unambiguous preset tier."""

        saved = {
            "web_base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
            "upstream_mode": "work-agent",
            "endpoint_id": "agent",
        }

        def save_settings(**values):
            saved.update(
                {key: value for key, value in values.items() if value is not None}
            )

        def get_settings():
            return dict(saved)

        old_mode = main_module.UPSTREAM_MODE
        old_env_mode = os.environ.get("UPSTREAM_MODE")
        try:
            with (
                patch.object(main_module.auth, "set_relay_settings", side_effect=save_settings),
                patch.object(main_module.auth, "get_settings", side_effect=get_settings),
                patch.object(main_module, "UPSTREAM_MODE", "work-agent"),
                patch.dict(os.environ, {"RELAY_API_KEYS": "", "UPSTREAM_MODE": "work-agent"}, clear=False),
            ):
                client = TestClient(main_module.app)
                # A legacy settings client selects the ordinary Remote preset
                # without knowing endpoint_id.  Persist the unambiguous Remote
                # identity instead of allowing the old Work identity to win.
                legacy = client.post(
                    "/api/settings",
                    json={
                        "web_base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
                        "upstream_mode": "remote",
                    },
                )
                self.assertEqual(legacy.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "remote")
                self.assertEqual(saved["upstream_mode"], "remote")

                # A port-only update is unrelated to endpoint selection and
                # must retain the selected preset identity.
                saved.update(
                    {
                        "web_base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
                        "upstream_mode": "work-agent",
                        "endpoint_id": "agent",
                    }
                )
                port = client.post("/api/settings", json={"relay_port": 8100})
                self.assertEqual(port.status_code, 200)
                self.assertEqual(saved["endpoint_id"], "agent")
        finally:
            main_module.UPSTREAM_MODE = old_mode
            if old_env_mode is None:
                os.environ.pop("UPSTREAM_MODE", None)
            else:
                os.environ["UPSTREAM_MODE"] = old_env_mode

    def test_model_test_trace_keeps_explicit_ide_and_work_endpoint(self):
        """The connectivity probe must report the selected tier, not a URL alias."""

        seen = []

        async def fake_dispatch(_messages, _model, _stream, options):
            seen.append(dict(options))
            trace = options["_upstream_trace"]
            mode = options["_upstream_mode"]
            trace.update(
                requested_mode=mode,
                requested_endpoint=mode,
                actual_mode=mode,
                actual_endpoint="remote-work" if mode == "work-agent" else "ide",
                fallback_used=False,
            )
            return JSONResponse(
                {"choices": [{"message": {"content": "pong"}}], "usage": {}}
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            client = TestClient(main_module.app)
            for endpoint, actual in (("ide", "ide"), ("work-agent", "remote-work")):
                with self.subTest(endpoint=endpoint):
                    response = client.post(
                        "/api/model-test",
                        json={
                            "model": "glm-5.3",
                            "endpoint": endpoint,
                            "disable_fallback": True,
                        },
                    )
                    body = response.json()
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(body["success"])
                    self.assertEqual(body["requested_endpoint"], endpoint)
                    self.assertEqual(body["actual_endpoint"], actual)
                    self.assertFalse(body["fallback_used"])

        self.assertEqual([item["_upstream_mode"] for item in seen], ["ide", "work-agent"])
        self.assertTrue(all(item["_disable_upstream_fallback"] for item in seen))

    def test_model_test_returns_endpoint_fallback_diagnostics(self):
        async def fake_dispatch(_messages, _model, _stream, options):
            trace = options["_upstream_trace"]
            trace.update(
                requested_mode="ide",
                requested_endpoint="ide",
                actual_mode="remote",
                actual_endpoint="remote",
                fallback_used=True,
                fallback_target="remote",
                failed_endpoint="ide",
                fallback_reason="ide: llm_utils_chat is unavailable",
            )
            return JSONResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            client = TestClient(main_module.app)
            response = client.post(
                "/api/model-test",
                json={
                    "model": "glm-5.3",
                    "endpoint": "ide",
                    "disable_fallback": False,
                },
            )

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertTrue(body["fallback_used"])
        self.assertEqual(body["fallback_target"], "remote")
        self.assertEqual(body["failed_endpoint"], "ide")
        self.assertIn("llm_utils_chat", body["fallback_reason"])

    def test_endpoint_markup_uses_mode_when_raw_and_ide_share_url(self):
        state = auth.AuthState(source="web-login")
        settings = {
            "web_base_url": "https://trae-api-cn.mchost.guru",
            "upstream_mode": "ide",
            "relay_port": 8000,
        }
        with (
            patch.object(main_module.auth, "get_auth", return_value=state),
            patch.object(main_module.auth, "list_accounts", return_value=[]),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": False, "mode": "round-robin"},
            ),
            patch.object(main_module.auth, "get_settings", return_value=settings),
            patch.object(
                main_module.auth,
                "get_max_mode_settings",
                return_value={"enabled": False, "models": ""},
            ),
            patch.object(
                main_module.auth,
                "get_auto_route_settings",
                return_value={"enabled": False},
            ),
            patch.object(
                main_module.auth,
                "get_auto_checkin_settings",
                return_value={"enabled": False, "time": "08:30"},
            ),
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
        ):
            html = main_module._web_login_html()

        self.assertIn(
            '<option value="ide" selected>IDE Agent / llm_utils_chat</option>',
            html,
        )
        self.assertNotIn(
            '<option value="raw">IDE Raw / llm_raw_chat</option>',
            html,
        )
        self.assertIn(
            '<option value="solo">Solo / api/agent/v3/llm_utils_chat</option>',
            html,
        )
        self.assertNotIn(
            '<option value="raw" selected>IDE Raw / llm_raw_chat</option>',
            html,
        )
        self.assertIn("saveRelayServiceSettings", html)
        settings_start = html.index("async function saveSettings()")
        service_start = html.index("async function saveRelayServiceSettings()", settings_start)
        self.assertNotIn("relay_port", html[settings_start:service_start])
        self.assertIn("relay_port", html[service_start:])
        self.assertIn("宿主机映射端口（RELAY_PORT）", html)
        settings_markup = html[
            html.index('<div class="tab-page" data-page="settings">') :
            html.index('<div class="tab-page" data-page="models">')
        ]
        self.assertIn('onchange="applyEndpointPreset()"', settings_markup)
        self.assertIn('oninput="markEndpointDirty()"', settings_markup)
        self.assertIn('id="settings-save-btn"', settings_markup)
        self.assertIn('onclick="saveSettings()"', settings_markup)
        self.assertIn('id="settings-save-btn" onclick="saveSettings()" disabled', settings_markup)
        self.assertNotIn('onchange="saveSettings()"', settings_markup)
        self.assertNotIn('onchange="togglePolling()"', settings_markup)
        self.assertIn("endpointDiagnosticText", html)
        self.assertIn("fallback_target", html)
        self.assertIn("失败端点=", html)
        self.assertIn("回落目标=", html)
        self.assertIn("原因=", html)

    @staticmethod
    def _extract_js_function(html, name):
        """Return one generated JS function body for static UI contract tests."""
        marker = f"function {name}("
        start = html.index(marker)
        opening = html.index("{", start)
        depth = 0
        for index in range(opening, len(html)):
            char = html[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return html[start : index + 1]
        raise AssertionError(f"unterminated generated function: {name}")

    def test_settings_changes_are_draft_only_until_explicit_save(self):
        """Changing a setting must not POST or reload before its Save button."""
        state = auth.AuthState(source="web-login")
        settings = {
            "web_base_url": "https://trae-api-cn.mchost.guru",
            "upstream_mode": "ide",
            "endpoint_id": "ide",
            "relay_port": 8000,
        }
        account = {
            "id": "account-1",
            "user_id": "user-1",
            "label": "测试账号",
            "is_valid": True,
            "is_active": True,
            "model_enabled": True,
            "checked_in": True,
            "expires": "",
            "account_credits": {},
        }
        with (
            patch.object(main_module.auth, "get_auth", return_value=state),
            patch.object(main_module.auth, "list_accounts", return_value=[account]),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": False, "mode": "round-robin"},
            ),
            patch.object(main_module.auth, "get_settings", return_value=settings),
            patch.object(
                main_module.auth,
                "get_max_mode_settings",
                return_value={"enabled": False, "models": ""},
            ),
            patch.object(
                main_module.auth,
                "get_auto_route_settings",
                return_value={"enabled": False},
            ),
            patch.object(
                main_module.auth,
                "get_auto_checkin_settings",
                return_value={"enabled": False, "time": "08:30"},
            ),
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
        ):
            html = main_module._web_login_html()

        settings_start = html.index('<div class="tab-page" data-page="settings">')
        settings_end = html.index('<div class="tab-page" data-page="models">')
        settings_markup = html[settings_start:settings_end]
        accounts_start = html.index('<div class="tab-page" data-page="accounts">')
        accounts_end = html.index('<div class="tab-page" data-page="usage">')
        accounts_markup = html[accounts_start:accounts_end]

        # Every mutable settings control is draft-only.  The only calls from
        # the controls are dirty markers; API writes are bound to Save buttons.
        self.assertIn('onchange="markPollingDirty()"', settings_markup)
        self.assertIn('id="poll-save-btn"', settings_markup)
        self.assertIn('onclick="savePollingSettings()"', settings_markup)
        self.assertIn('id="poll-save-btn" onclick="savePollingSettings()" disabled', settings_markup)
        self.assertIn('onchange="markAutoRouteDirty()"', settings_markup)
        self.assertIn('id="auto-route-save-btn"', settings_markup)
        self.assertIn('onclick="saveAutoRoute()"', settings_markup)
        self.assertIn('id="auto-route-save-btn" onclick="saveAutoRoute()" disabled', settings_markup)
        self.assertIn('onchange="markMaxModeDirty()"', settings_markup)
        self.assertIn('oninput="markMaxModeDirty()"', settings_markup)
        self.assertIn('oninput="markRelayServiceDirty()"', settings_markup)
        self.assertIn('onchange="markAutoCheckinDirty()"', accounts_markup)
        self.assertIn('oninput="markAutoCheckinDirty()"', accounts_markup)
        self.assertIn('onchange="markAccountModelDirty(', accounts_markup)
        self.assertIn('id="account-model-save-btn"', accounts_markup)
        self.assertIn('onclick="saveAccountModelSettings()"', accounts_markup)
        self.assertNotIn('onchange="togglePolling()"', settings_markup)
        self.assertNotIn('onchange="saveAutoRoute()"', settings_markup)
        self.assertNotIn('onchange="toggleAccountModel(', accounts_markup)

        # The draft handlers must not write to the API or trigger a page
        # refresh.  Save handlers also update the page in place.
        for name in (
            "markPollingDirty",
            "markEndpointDirty",
            "markAutoRouteDirty",
            "markMaxModeDirty",
            "markRelayServiceDirty",
            "markAutoCheckinDirty",
            "markAccountModelDirty",
        ):
            body = self._extract_js_function(html, name)
            self.assertNotIn("postJSON(", body, name)
            self.assertNotIn("location.reload()", body, name)

        for name in (
            "applyEndpointPreset",
            "savePollingSettings",
            "saveSettings",
            "saveAutoRoute",
            "saveMaxMode",
            "saveRelayServiceSettings",
            "saveAutoCheckin",
            "saveAccountModelSettings",
        ):
            body = self._extract_js_function(html, name)
            self.assertNotIn("location.reload()", body, name)

        endpoint_preview = self._extract_js_function(html, "applyEndpointPreset")
        self.assertNotIn("saveSettings()", endpoint_preview)
        self.assertNotIn("postJSON(", endpoint_preview)

    def test_endpoint_markup_prefers_persisted_endpoint_id(self):
        """A shared URL must not make IDE Agent render as IDE Raw."""

        state = auth.AuthState(source="web-login")
        settings = {
            "web_base_url": "https://trae-api-cn.mchost.guru",
            "upstream_mode": "ide",
            "endpoint_id": "ide",
            "relay_port": 8000,
        }
        with (
            patch.object(main_module.auth, "get_auth", return_value=state),
            patch.object(main_module.auth, "list_accounts", return_value=[]),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": False, "mode": "round-robin"},
            ),
            patch.object(main_module.auth, "get_settings", return_value=settings),
            patch.object(
                main_module.auth,
                "get_max_mode_settings",
                return_value={"enabled": False, "models": ""},
            ),
            patch.object(
                main_module.auth,
                "get_auto_route_settings",
                return_value={"enabled": False},
            ),
            patch.object(
                main_module.auth,
                "get_auto_checkin_settings",
                return_value={"enabled": False, "time": "08:30"},
            ),
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
        ):
            html = main_module._web_login_html()

        self.assertIn(
            '<option value="ide" selected>IDE Agent / llm_utils_chat</option>',
            html,
        )
        self.assertNotIn(
            '<option value="raw" selected>IDE Raw / llm_raw_chat</option>',
            html,
        )

    def test_startup_restores_persisted_mode_and_base_url(self):
        original_mode = main_module.UPSTREAM_MODE
        original_base = main_module.WEB_BASE
        old_mode = os.environ.get("UPSTREAM_MODE")
        old_base = os.environ.get("TRAE_WEB_BASE_URL")
        try:
            with (
                patch.object(
                    main_module.auth,
                    "get_settings",
                    return_value={
                        "upstream_mode": "ide",
                        "web_base_url": "https://relay.example/api/remote/v1/",
                    },
                ),
                patch.object(main_module, "UPSTREAM_MODE", "remote"),
                patch.object(
                    main_module,
                    "WEB_BASE",
                    "https://trae-api-cn.mchost.guru/api/remote/v1",
                ),
            ):
                settings = main_module._restore_persisted_relay_settings()
                self.assertEqual(settings["upstream_mode"], "ide")
                self.assertEqual(main_module.UPSTREAM_MODE, "ide")
                self.assertEqual(
                    main_module.WEB_BASE,
                    "https://relay.example/api/remote/v1",
                )
                self.assertEqual(os.environ["UPSTREAM_MODE"], "ide")
                self.assertEqual(
                    os.environ["TRAE_WEB_BASE_URL"],
                    "https://relay.example/api/remote/v1",
                )
        finally:
            main_module.UPSTREAM_MODE = original_mode
            main_module.WEB_BASE = original_base
            if old_mode is None:
                os.environ.pop("UPSTREAM_MODE", None)
            else:
                os.environ["UPSTREAM_MODE"] = old_mode
            if old_base is None:
                os.environ.pop("TRAE_WEB_BASE_URL", None)
            else:
                os.environ["TRAE_WEB_BASE_URL"] = old_base

    def test_startup_restores_endpoint_id_before_shared_url_or_mode(self):
        original_mode = main_module.UPSTREAM_MODE
        original_base = main_module.WEB_BASE
        old_mode = os.environ.get("UPSTREAM_MODE")
        old_base = os.environ.get("TRAE_WEB_BASE_URL")
        persisted = {
            "endpoint_id": "agent",
            "upstream_mode": "remote",
            "web_base_url": "https://trae-api-cn.mchost.guru",
        }

        def save_endpoint(**values):
            persisted.update(
                {key: value for key, value in values.items() if value not in ("", None, 0)}
            )

        try:
            with (
                patch.object(
                    main_module.auth,
                    "get_settings",
                    side_effect=lambda: dict(persisted),
                ),
                patch.object(main_module.auth, "set_relay_settings", side_effect=save_endpoint),
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(
                    main_module,
                    "WEB_BASE",
                    "https://old.example",
                ),
            ):
                settings = main_module._restore_persisted_relay_settings()
                self.assertEqual(settings["endpoint_id"], "agent")
                self.assertEqual(settings["upstream_mode"], "work-agent")
                self.assertEqual(
                    main_module.WEB_BASE,
                    "https://trae-api-cn.mchost.guru/api/remote/v1",
                )
                self.assertEqual(main_module.UPSTREAM_MODE, "work-agent")
                self.assertEqual(os.environ["UPSTREAM_MODE"], "work-agent")
        finally:
            main_module.UPSTREAM_MODE = original_mode
            main_module.WEB_BASE = original_base
            if old_mode is None:
                os.environ.pop("UPSTREAM_MODE", None)
            else:
                os.environ["UPSTREAM_MODE"] = old_mode
            if old_base is None:
                os.environ.pop("TRAE_WEB_BASE_URL", None)
            else:
                os.environ["TRAE_WEB_BASE_URL"] = old_base

    def test_relay_env_update_is_partial_and_keeps_other_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accounts = root / "accounts.json"
            env_path = root / ".env"
            env_path.write_text(
                "TRAE_WEB_BASE_URL=https://old.example/api/remote/v1\n"
                "UPSTREAM_MODE=remote\n"
                "RELAY_PORT=8000\n"
                "TRAE_AUTO_ROUTE=1\n",
                "utf-8",
            )
            with (
                patch.object(auth, "ACCOUNTS_PATH", accounts),
                patch.object(auth, "ENV_PATH", env_path),
                patch.object(auth, "_settings", {}),
            ):
                auth.set_relay_settings(upstream_mode="ide")

            text = env_path.read_text("utf-8")
            self.assertIn("TRAE_WEB_BASE_URL=https://old.example/api/remote/v1", text)
            self.assertIn("UPSTREAM_MODE=ide", text)
            self.assertIn("RELAY_PORT=8000", text)
            self.assertIn("TRAE_AUTO_ROUTE=1", text)

    def test_relay_settings_persist_endpoint_id_without_touching_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accounts = root / "accounts.json"
            env_path = root / ".env"
            env_path.write_text(
                "TRAE_WEB_BASE_URL=https://old.example/api/remote/v1\n"
                "UPSTREAM_MODE=remote\n"
                "RELAY_PORT=8000\n",
                "utf-8",
            )
            with (
                patch.object(auth, "ACCOUNTS_PATH", accounts),
                patch.object(auth, "ENV_PATH", env_path),
                patch.object(auth, "_settings", {}),
            ):
                auth.set_relay_settings(
                    web_base_url="https://trae-api-cn.mchost.guru",
                    upstream_mode="ide",
                    endpoint_id="ide",
                )
                settings = auth.get_settings()

            self.assertEqual(settings["endpoint_id"], "ide")
            self.assertEqual(settings["upstream_mode"], "ide")
            self.assertIn(
                "UPSTREAM_MODE=ide",
                env_path.read_text("utf-8"),
            )
            self.assertNotIn("ENDPOINT_ID", env_path.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
