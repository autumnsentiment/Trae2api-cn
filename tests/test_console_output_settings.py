import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import auth
from src import main as main_module


def _account(account_id, **overrides):
    record = {
        "user_id": account_id,
        "source": "web-login",
        "token": f"header.{account_id * 32}.signature",
        "expired_at": "2099-01-01T00:00:00Z",
        "provider_specific": {},
    }
    record.update(overrides)
    return record


class _StoreSandbox:
    def __enter__(self):
        self._stack = ExitStack()
        root = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        self.accounts_path = root / "accounts.json"
        self.env_path = root / ".env"
        self.env_path.write_text("UPSTREAM_MODE=ide\nUNRELATED=keep\n", "utf-8")
        self.accounts = {
            "personal": _account("personal"),
            "enterprise": _account("enterprise", account_tier="free"),
        }
        values = {
            "ACCOUNTS_PATH": self.accounts_path,
            "ENV_PATH": self.env_path,
            "_accounts": self.accounts,
            "_settings": {},
            "_active_account": "personal",
            "_auth": auth._record_to_state(self.accounts["personal"]),
            "_poll_enabled": False,
            "_polling_mode": "round-robin",
            "_rotation_cursor": 0,
            "_MODEL_ACCOUNT_INFLIGHT": {},
            "_MODEL_ACCOUNT_RESERVED": {},
            "_MODEL_ACCOUNT_RESERVATION_OWNERS": {},
        }
        for key, value in values.items():
            self._stack.enter_context(patch.object(auth, key, value))
        self._stack.enter_context(patch.dict(os.environ, {}, clear=False))
        for key in (
            "TRAE_VERBOSE_REASONING",
            "TRAE_ENTERPRISE_ENABLED",
            "TRAE_ENTERPRISE_ACCOUNT_ID",
        ):
            os.environ.pop(key, None)
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def saved_settings(self):
        return json.loads(self.accounts_path.read_text("utf-8"))["settings"]


class ReasoningOutputSettingsTests(unittest.TestCase):
    def test_environment_is_used_until_console_saves(self):
        with _StoreSandbox():
            self.assertEqual(
                auth.get_reasoning_output_settings(),
                {"enabled": False, "source": "env"},
            )
            for value in ("1", "true", "YES", " on "):
                with self.subTest(value=value):
                    os.environ["TRAE_VERBOSE_REASONING"] = value
                    self.assertTrue(auth.get_reasoning_output_settings()["enabled"])

    def test_console_value_persists_and_overrides_runtime_environment(self):
        with _StoreSandbox() as box:
            os.environ["TRAE_VERBOSE_REASONING"] = "1"
            result = auth.set_reasoning_output_settings(False)
            self.assertEqual(result, {"enabled": False, "source": "console"})
            self.assertEqual(os.environ["TRAE_VERBOSE_REASONING"], "0")
            self.assertEqual(
                box.saved_settings()["reasoning_output"], {"enabled": False}
            )
            env_text = box.env_path.read_text("utf-8")
            self.assertIn("TRAE_VERBOSE_REASONING=0", env_text)
            self.assertIn("UPSTREAM_MODE=ide", env_text)
            self.assertIn("UNRELATED=keep", env_text)

            os.environ["TRAE_VERBOSE_REASONING"] = "1"
            auth._settings.clear()
            auth._bootstrap_account_store()
            self.assertEqual(auth.get_reasoning_output_settings(), result)

    def test_setter_requires_boolean_without_mutating_state(self):
        with _StoreSandbox() as box:
            original_env = box.env_path.read_text("utf-8")
            for value in (None, "false", 0, 1, [], {}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    auth.set_reasoning_output_settings(value)
            self.assertEqual(auth._settings, {})
            self.assertFalse(box.accounts_path.exists())
            self.assertEqual(box.env_path.read_text("utf-8"), original_env)
            self.assertNotIn("TRAE_VERBOSE_REASONING", os.environ)


class EnterpriseSettingsTests(unittest.TestCase):
    def test_environment_defaults_do_not_infer_account_entitlement(self):
        with _StoreSandbox():
            self.assertEqual(
                auth.get_enterprise_settings(),
                {"enabled": False, "account_id": "", "source": "env"},
            )
            os.environ["TRAE_ENTERPRISE_ENABLED"] = "true"
            os.environ["TRAE_ENTERPRISE_ACCOUNT_ID"] = " enterprise "
            self.assertEqual(
                auth.get_enterprise_settings(),
                {"enabled": True, "account_id": "enterprise", "source": "env"},
            )

    def test_explicit_account_persists_without_changing_active_selection(self):
        with _StoreSandbox() as box:
            result = auth.set_enterprise_settings(True, " enterprise ")
            self.assertEqual(
                result,
                {"enabled": True, "account_id": "enterprise", "source": "console"},
            )
            self.assertEqual(
                box.saved_settings()["enterprise"],
                {"enabled": True, "account_id": "enterprise"},
            )
            self.assertEqual(os.environ["TRAE_ENTERPRISE_ENABLED"], "1")
            self.assertEqual(os.environ["TRAE_ENTERPRISE_ACCOUNT_ID"], "enterprise")
            env_text = box.env_path.read_text("utf-8")
            self.assertIn("TRAE_ENTERPRISE_ENABLED=1", env_text)
            self.assertIn("TRAE_ENTERPRISE_ACCOUNT_ID=enterprise", env_text)
            self.assertIn("UNRELATED=keep", env_text)
            self.assertNotIn(box.accounts["enterprise"]["token"], env_text)
            self.assertNotIn(box.accounts["enterprise"]["token"], repr(result))
            self.assertEqual(auth.get_active_account_id(), "personal")
            self.assertEqual(auth.get_token(), box.accounts["personal"]["token"])

            os.environ["TRAE_ENTERPRISE_ENABLED"] = "0"
            os.environ["TRAE_ENTERPRISE_ACCOUNT_ID"] = "personal"
            auth._settings.clear()
            auth._bootstrap_account_store()
            self.assertEqual(auth.get_enterprise_settings(), result)

    def test_omitted_account_retains_saved_selection_before_environment(self):
        with _StoreSandbox():
            auth.set_enterprise_settings(False, "enterprise")
            os.environ["TRAE_ENTERPRISE_ACCOUNT_ID"] = "personal"
            self.assertEqual(
                auth.set_enterprise_settings(True),
                {"enabled": True, "account_id": "enterprise", "source": "console"},
            )

    def test_omitted_account_can_use_explicit_environment_selection(self):
        with _StoreSandbox():
            os.environ["TRAE_ENTERPRISE_ACCOUNT_ID"] = "enterprise"
            self.assertEqual(
                auth.set_enterprise_settings(True)["account_id"], "enterprise"
            )

    def test_enable_rejects_missing_disabled_expired_opaque_and_cli_accounts(self):
        with _StoreSandbox() as box:
            box.accounts.update({
                "disabled": _account("disabled", model_enabled=False),
                "expired": _account("expired", expired_at="2000-01-01T00:00:00Z"),
                "opaque": _account("opaque", token="opaque-session"),
                "cli": _account("cli", source="cli"),
            })
            for account_id in ("", "missing", "disabled", "expired", "opaque", "cli"):
                with self.subTest(account_id=account_id), self.assertRaises(ValueError):
                    auth.set_enterprise_settings(True, account_id)
            self.assertEqual(auth._settings, {})
            self.assertFalse(box.accounts_path.exists())
            self.assertNotIn("TRAE_ENTERPRISE_ENABLED", os.environ)

    def test_disable_remains_possible_after_selected_account_is_removed(self):
        with _StoreSandbox():
            auth.set_enterprise_settings(True, "enterprise")
            del auth._accounts["enterprise"]
            self.assertEqual(
                auth.set_enterprise_settings(False),
                {"enabled": False, "account_id": "enterprise", "source": "console"},
            )
            self.assertEqual(
                auth.set_enterprise_settings(False, "")["account_id"], ""
            )

    def test_setter_rejects_invalid_input_before_saving(self):
        with _StoreSandbox() as box:
            for value in (None, "true", 0, 1, [], {}):
                with self.subTest(enabled=value), self.assertRaises(ValueError):
                    auth.set_enterprise_settings(value, "enterprise")
            for value in (0, [], {}, "enterprise\nOTHER=value"):
                with self.subTest(account_id=value), self.assertRaises(ValueError):
                    auth.set_enterprise_settings(False, value)
            self.assertEqual(auth._settings, {})
            self.assertFalse(box.accounts_path.exists())
            self.assertNotIn("TRAE_ENTERPRISE_ENABLED", os.environ)

    def test_get_settings_includes_public_output_and_enterprise_state(self):
        with _StoreSandbox() as box:
            auth.set_reasoning_output_settings(True)
            auth.set_enterprise_settings(True, "enterprise")
            settings = auth.get_settings()
            self.assertEqual(
                settings["reasoning_output"], {"enabled": True, "source": "console"}
            )
            self.assertEqual(
                settings["enterprise"],
                {"enabled": True, "account_id": "enterprise", "source": "console"},
            )
            for record in box.accounts.values():
                self.assertNotIn(record["token"], repr(settings))


class DedicatedAccountReservationTests(unittest.TestCase):
    def test_nonactivating_reservation_preserves_selected_account_and_capacity(self):
        with _StoreSandbox() as box, patch.object(
            auth, "_switch_record", wraps=auth._switch_record
        ) as switch:
            personal_state = auth._auth
            reservations = []
            for _ in range(2):
                account_id, snapshot = auth.reserve_model_account(
                    preferred_account_id="enterprise", activate=False
                )
                self.assertEqual(account_id, "enterprise")
                self.assertEqual(snapshot["token"], box.accounts["enterprise"]["token"])
                reservations.append(snapshot["_slot_reservation_id"])
            self.assertEqual(
                auth.reserve_model_account(
                    preferred_account_id="enterprise", activate=False
                ),
                ("", {}),
            )
            self.assertEqual(auth.get_active_account_id(), "personal")
            self.assertIs(auth._auth, personal_state)
            switch.assert_not_called()
            self.assertEqual(auth._MODEL_ACCOUNT_INFLIGHT, {"enterprise": 2})
            self.assertEqual(auth._MODEL_ACCOUNT_RESERVED, {"enterprise": 2})
            for reservation_id in reservations:
                self.assertEqual(
                    auth._MODEL_ACCOUNT_RESERVATION_OWNERS[reservation_id], "enterprise"
                )
                self.assertTrue(
                    auth.cancel_account_slot_reservation("enterprise", reservation_id)
                )
            self.assertEqual(auth._MODEL_ACCOUNT_INFLIGHT, {})
            self.assertEqual(auth._MODEL_ACCOUNT_RESERVED, {})
            self.assertEqual(auth._MODEL_ACCOUNT_RESERVATION_OWNERS, {})

    def test_default_reservation_keeps_existing_activation_behavior(self):
        with _StoreSandbox() as box:
            account_id, snapshot = auth.reserve_model_account(
                preferred_account_id="enterprise"
            )
            self.assertEqual(account_id, "enterprise")
            self.assertEqual(auth.get_active_account_id(), "enterprise")
            self.assertEqual(auth.get_token(), box.accounts["enterprise"]["token"])
            self.assertTrue(
                auth.cancel_account_slot_reservation(
                    account_id, snapshot["_slot_reservation_id"]
                )
            )


class ConsoleSwitchApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main_module.app)

    def test_reasoning_output_api_requires_boolean_and_round_trips(self):
        with patch.object(
            main_module.auth,
            "set_reasoning_output_settings",
            return_value={"enabled": True, "source": "console"},
        ) as setter:
            response = self.client.post(
                "/api/reasoning-output", json={"enabled": "true"}
            )
            self.assertEqual(response.status_code, 400)
            response = self.client.post(
                "/api/reasoning-output", json={"enabled": True}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["enabled"], True)
        setter.assert_called_once_with(True)

    def test_enterprise_api_rejects_invalid_account_id_and_passes_selection(self):
        with patch.object(
            main_module.auth,
            "set_enterprise_settings",
            return_value={
                "enabled": True,
                "account_id": "enterprise",
                "source": "console",
            },
        ) as setter:
            response = self.client.post(
                "/api/enterprise",
                json={"enabled": True, "account_id": 123},
            )
            self.assertEqual(response.status_code, 400)
            response = self.client.post(
                "/api/enterprise",
                json={"enabled": True, "account_id": "enterprise"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["account_id"], "enterprise")
        setter.assert_called_once_with(True, "enterprise")


if __name__ == "__main__":
    unittest.main()
