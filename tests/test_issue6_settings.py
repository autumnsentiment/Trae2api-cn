import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import auth
from src import main as main_module


class Issue6SettingsTests(unittest.TestCase):
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
        self.assertIn(
            '<option value="raw">IDE Raw / llm_raw_chat</option>',
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


if __name__ == "__main__":
    unittest.main()
