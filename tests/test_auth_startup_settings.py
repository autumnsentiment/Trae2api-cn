import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from src import auth, cli_client


class AuthFallbackSettingsTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.accounts_path = root / "accounts.json"
        env_path = root / ".env"
        env_path.write_text("TRAE_VERBOSE_REASONING=0\n", "utf-8")
        self.saved_settings = {
            "reasoning_output": {"enabled": True},
            "endpoint_id": "ide",
        }
        self.saved_account = {
            "user_id": "saved-account",
            "label": "Saved account",
            "source": "web-login",
            "token": "header." + ("s" * 140) + ".signature",
            "expired_at": "2099-01-01T00:00:00Z",
            "provider_specific": {},
            "checkin": {"checked_in": True},
        }
        self.accounts_path.write_text(
            json.dumps({
                "accounts": {"saved-account": self.saved_account},
                "active": "saved-account",
                "poll_enabled": True,
                "polling_mode": "credit-priority",
                "rotation_cursor": 3,
                "settings": self.saved_settings,
            }),
            "utf-8",
        )
        for name, value in {
            "ACCOUNTS_PATH": self.accounts_path,
            "ENV_PATH": env_path,
            "_auth": auth.AuthState(),
            "_accounts": {},
            "_active_account": "",
            "_poll_enabled": False,
            "_polling_mode": "round-robin",
            "_rotation_cursor": 0,
            "_settings": {},
        }.items():
            stack.enter_context(patch.object(auth, name, value))
        stack.enter_context(patch.dict(
            os.environ,
            {"TRAE_AUTH_SOURCE": "auto", "TRAE_VERBOSE_REASONING": "0"},
            clear=True,
        ))
        stack.enter_context(patch.object(
            auth, "_safe_read_env_value", return_value=""
        ))
        stack.enter_context(patch.object(
            auth.trae_decrypt, "try_auto_discover", return_value=(None, "cn")
        ))
        self.resolve_cli = stack.enter_context(patch.object(
            cli_client, "resolve_cli_command", return_value=""
        ))

    def assert_saved_settings_restored(self):
        self.assertEqual(
            auth.get_reasoning_output_settings(),
            {"enabled": True, "source": "console"},
        )
        self.assertEqual(auth.get_settings()["endpoint_id"], "ide")
        self.assertEqual(auth.get_polling_status(), {
            "enabled": True,
            "active_account": "saved-account",
            "account_count": 1,
            "mode": "credit-priority",
        })
        self.assertEqual(auth._rotation_cursor, 3)
        record = auth.get_account_record("saved-account")
        self.assertEqual(record["label"], "Saved account")
        self.assertEqual(record["checkin"], {"checked_in": True})
        persisted = json.loads(self.accounts_path.read_text("utf-8"))
        self.assertEqual(persisted["settings"], self.saved_settings)

    def test_auto_to_env_restores_saved_settings_before_auth_snapshot(self):
        token = "header." + ("e" * 140) + ".signature"
        os.environ.update({
            "TRAE_TOKEN": token,
            "TRAE_USER_ID": "saved-account",
            "TRAE_TOKEN_EXPIRES": "2099-01-01T00:00:00Z",
        })

        state = auth.init_auth()

        self.assertEqual(state.source, "env")
        self.assert_saved_settings_restored()
        self.assertEqual(auth.get_account_record("saved-account")["token"], token)
        self.resolve_cli.assert_not_called()

    def test_auto_to_cli_restores_saved_settings_and_polling_accounts(self):
        self.resolve_cli.return_value = "fake-trae-cli"

        state = auth.init_auth()

        self.assertEqual(state.source, "cli")
        self.assert_saved_settings_restored()
        self.assertEqual(auth.get_account_record("saved-account"), self.saved_account)
        self.resolve_cli.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
