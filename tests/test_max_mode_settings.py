import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("TRAE_AUTH_SOURCE", "cli")
os.environ.setdefault("RELAY_API_KEYS", "smoke-key")

from fastapi.testclient import TestClient

from src import auth, trae_remote_client
from src import main as main_module


class _StoreSandbox:
    """Point the account store and .env at a temp dir for one test."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.accounts_path = root / "accounts.json"
        self.env_path = root / ".env"
        self.env_path.write_text("UPSTREAM_MODE=remote\nTRAE_REMOTE_MAX_MODE=1\n", "utf-8")
        self._patches = [
            patch.object(auth, "ACCOUNTS_PATH", self.accounts_path),
            patch.object(auth, "ENV_PATH", self.env_path),
            patch.object(auth, "_settings", {}),
            patch.dict(os.environ, {}, clear=False),
        ]

    def __enter__(self):
        for item in self._patches:
            item.start()
        os.environ.pop("TRAE_REMOTE_MAX_MODE", None)
        os.environ.pop("TRAE_REMOTE_MAX_MODELS", None)
        return self

    def __exit__(self, *exc):
        for item in reversed(self._patches):
            item.stop()
        self._tmp.cleanup()
        return False


class MaxModeSettingsTests(unittest.TestCase):
    def test_env_is_default_until_console_saves(self):
        with _StoreSandbox():
            os.environ["TRAE_REMOTE_MAX_MODE"] = "1"
            os.environ["TRAE_REMOTE_MAX_MODELS"] = "glm-5.3, GLM-5.3 ,deepseek-v4-pro"
            settings = auth.get_max_mode_settings()
            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["models"], "glm-5.3,deepseek-v4-pro")
            self.assertEqual(settings["source"], "env")

    def test_console_switch_persists_and_updates_runtime_env(self):
        with _StoreSandbox() as box:
            os.environ["TRAE_REMOTE_MAX_MODE"] = "1"
            settings = auth.set_max_mode_settings(False, ["glm-5.3", ""])
            self.assertFalse(settings["enabled"])
            self.assertEqual(settings["source"], "console")
            self.assertEqual(os.environ["TRAE_REMOTE_MAX_MODE"], "0")
            self.assertEqual(os.environ["TRAE_REMOTE_MAX_MODELS"], "glm-5.3")
            stored = json.loads(box.accounts_path.read_text("utf-8"))
            self.assertEqual(
                stored["settings"]["max_mode"], {"enabled": False, "models": "glm-5.3"}
            )
            env_text = box.env_path.read_text("utf-8")
            self.assertIn("TRAE_REMOTE_MAX_MODE=0", env_text)
            self.assertIn("TRAE_REMOTE_MAX_MODELS=glm-5.3", env_text)
            self.assertIn("UPSTREAM_MODE=remote", env_text)
            self.assertEqual(env_text.count("TRAE_REMOTE_MAX_MODE="), 1)

    def test_saved_console_value_overrides_container_env_after_restart(self):
        with _StoreSandbox():
            auth._settings["max_mode"] = {"enabled": True, "models": ""}
            os.environ["TRAE_REMOTE_MAX_MODE"] = "0"
            auth.apply_max_mode_settings()
            self.assertEqual(os.environ["TRAE_REMOTE_MAX_MODE"], "1")


class MaxModeApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main_module.app)

    def test_toggle_endpoint_round_trip(self):
        with _StoreSandbox():
            response = self.client.post(
                "/api/max-mode", json={"enabled": True, "models": "glm-5.3"}
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertTrue(body["enabled"])
            self.assertEqual(body["models"], "glm-5.3")
            current = self.client.get("/api/max-mode").json()
            self.assertTrue(current["enabled"])
            self.assertEqual(current["source"], "console")

    def test_toggle_requires_enabled_field(self):
        with _StoreSandbox():
            response = self.client.post("/api/max-mode", json={"models": "glm-5.3"})
            self.assertEqual(response.status_code, 400)
            response = self.client.post(
                "/api/max-mode", json={"enabled": True, "models": 5}
            )
            self.assertEqual(response.status_code, 400)

    def test_settings_page_renders_max_mode_switch(self):
        with _StoreSandbox():
            auth.set_max_mode_settings(True, "glm-5.3")
            html = self.client.get("/web/login").text
        self.assertIn('id="max-mode-toggle" checked', html)
        self.assertIn('id="max-mode-models" value="glm-5.3"', html)
        self.assertIn("saveMaxMode()", html)
        self.assertIn('id="conn-max"', html)
        polling_index = html.index('<div class="section-title">多账号轮询</div>')
        max_index = html.index("1M 上下文（Max 模式）")
        models_index = html.index('data-page="models"')
        self.assertLess(polling_index, max_index)
        self.assertLess(max_index, models_index)

    def test_detect_lists_only_max_capable_agent_models(self):
        configs = {
            "glm-5.3": {
                "max_mode": True,
                "context_window_size": {"default": 200000, "max": [1000000]},
            },
            "kimi-k2.7-code": {"max_mode": False},
        }
        fetch = AsyncMock(return_value=configs)
        with (
            patch.object(
                auth,
                "get_active_account_snapshot",
                return_value=("acct-1", {"token": "jwt", "provider_specific": {}}),
            ),
            patch.object(main_module.trae_client, "_fetch_web_model_configs", new=fetch),
        ):
            body = self.client.get("/api/max-mode/models").json()
        self.assertTrue(body["success"])
        self.assertEqual(
            body["models"],
            [{"name": "glm-5.3", "display_name": "glm-5.3", "max_context": 1000000}],
        )
        self.assertEqual(fetch.await_args.kwargs["agent_type"], "solo_agent_remote")


class _Client:
    def __init__(self):
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))

        class _Resp:
            status_code = 200
            text = ""

            def json(self_inner):
                return {"data": {"chat_session_id": "s1", "message_id": "m1"}}

        return _Resp()


class MaxModeTraceTests(unittest.TestCase):
    def _create(self, env, options):
        custom = {
            "name": "glm-5.3",
            "config_name": "glm-5.3",
            "model_name": "glm-5.3",
            "config_source": 1,
            "max_mode": True,
            "context_window_size": {"default": 200000, "max": [1000000]},
        }

        async def run():
            client = _Client()
            with (
                patch.dict(trae_remote_client.os.environ, env),
                patch.object(
                    trae_remote_client.trae_client,
                    "resolve_model_config",
                    new=AsyncMock(return_value=custom),
                ),
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options=options,
                )
            return client.posts[0][1]["json"]["initial_message"]

        return asyncio.run(run())

    def test_trace_reports_applied_max_context(self):
        trace = {}
        initial = self._create(
            {"TRAE_REMOTE_MAX_MODE": "1", "TRAE_REMOTE_MAX_MODELS": ""},
            {"_account_id": "a1", "_auth_token": "jwt-token", "_upstream_trace": trace},
        )
        self.assertEqual(initial["model_selection_strategy"], "max")
        self.assertTrue(trace["max_mode_applied"])
        self.assertEqual(trace["max_context_tokens"], 1000000)

    def test_per_request_flag_enables_max_when_switch_is_off(self):
        trace = {}
        initial = self._create(
            {"TRAE_REMOTE_MAX_MODE": "0", "TRAE_REMOTE_MAX_MODELS": ""},
            {
                "_account_id": "a1",
                "_auth_token": "jwt-token",
                "trae_max_mode": True,
                "_upstream_trace": trace,
            },
        )
        self.assertEqual(initial["model_selection_strategy"], "max")
        self.assertTrue(trace["max_mode_applied"])

    def test_switch_off_reports_not_applied(self):
        trace = {}
        initial = self._create(
            {"TRAE_REMOTE_MAX_MODE": "0", "TRAE_REMOTE_MAX_MODELS": ""},
            {"_account_id": "a1", "_auth_token": "jwt-token", "_upstream_trace": trace},
        )
        self.assertNotEqual(initial["model_selection_strategy"], "max")
        self.assertFalse(trace["max_mode_applied"])
        self.assertNotIn("max_context_tokens", trace)


if __name__ == "__main__":
    unittest.main()
