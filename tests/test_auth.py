import unittest
from unittest.mock import patch

from src import auth
from src.auth import AuthState, _merge_state_record


class AuthStateMergeTests(unittest.TestCase):
    def test_merge_state_record_preserves_cached_account_metadata(self):
        previous = {
            "user_id": "account-1",
            "label": "Custom label",
            "token": "old-token",
            "checkin": {"credits": 1234, "checked_in": True},
            "checkin_updated_at": 1786723200.0,
            "future_cache_field": {"value": "keep"},
        }
        state = AuthState(
            edition="cn",
            source="web-login",
            token="new-token",
            refresh_token="new-refresh-token",
            user_id="account-1",
            host="https://example.invalid",
            client_id="client-1",
            provider_specific={"screenName": "Upstream label"},
        )

        merged = _merge_state_record(state, previous)

        self.assertEqual(merged["token"], "new-token")
        self.assertEqual(merged["refresh_token"], "new-refresh-token")
        self.assertEqual(merged["label"], "Custom label")
        self.assertEqual(merged["checkin"], previous["checkin"])
        self.assertEqual(merged["checkin_updated_at"], previous["checkin_updated_at"])
        self.assertEqual(merged["future_cache_field"], previous["future_cache_field"])

    def test_credit_merge_does_not_refresh_daily_checkin_timestamp(self):
        record = {
            "checkin": {"checked_in": True},
            "checkin_status_updated_at": 100.0,
            "checkin_updated_at": 100.0,
        }
        with (
            patch.object(auth, "_accounts", {"account-1": record}),
            patch.object(auth, "_save_accounts"),
            patch("src.auth.time.time", return_value=200.0),
        ):
            merged = auth.merge_account_credits(
                "account-1", {"account_credits": {"remaining": 10}}
            )

        self.assertTrue(merged["checked_in"])
        self.assertEqual(record["checkin_status_updated_at"], 100.0)
        self.assertEqual(record["checkin_updated_at"], 100.0)
        self.assertEqual(record["credits_updated_at"], 200.0)


if __name__ == "__main__":
    unittest.main()
