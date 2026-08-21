import asyncio
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


class RefreshTokenRaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_updates_captured_account_after_active_account_switch(self):
        request_started = asyncio.Event()
        release_response = asyncio.Event()
        posted = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "Result": {
                        "Token": "new-token-a",
                        "RefreshToken": "new-refresh-a",
                        "TokenExpireAt": "1900000000000",
                    }
                }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json):
                posted.update({"url": url, "json": json})
                request_started.set()
                await release_response.wait()
                return FakeResponse()

        account_a = {
            "user_id": "account-a",
            "token": "old-token-a",
            "refresh_token": "refresh-a",
            "expired_at": "2026-01-01T00:00:00Z",
            "host": "https://account-a.example",
            "client_id": "client-a",
            "source": "web-login",
            "edition": "cn",
            "provider_specific": {},
        }
        account_b = {
            "user_id": "account-b",
            "token": "token-b",
            "refresh_token": "refresh-b",
            "expired_at": "2026-01-01T00:00:00Z",
            "host": "https://account-b.example",
            "client_id": "client-b",
            "source": "web-login",
            "edition": "cn",
            "provider_specific": {},
        }
        accounts = {"account-a": account_a, "account-b": account_b}

        with (
            patch.object(auth, "_accounts", accounts),
            patch.object(auth, "_active_account", "account-a"),
            patch.object(auth, "_auth", auth._record_to_state(account_a)),
            patch.object(auth, "_save_accounts"),
            patch.object(auth, "_save_env_snapshot"),
            patch("src.auth.httpx.AsyncClient", FakeClient),
        ):
            refresh_task = asyncio.create_task(auth.refresh_token())
            await request_started.wait()

            self.assertTrue(auth.switch_account("account-b"))
            release_response.set()

            self.assertTrue(await refresh_task)
            self.assertEqual(auth.get_active_account_id(), "account-b")
            self.assertEqual(auth.get_token(), "token-b")
            self.assertEqual(accounts["account-b"]["token"], "token-b")
            self.assertEqual(accounts["account-b"]["refresh_token"], "refresh-b")
            self.assertEqual(accounts["account-a"]["token"], "new-token-a")
            self.assertEqual(
                accounts["account-a"]["refresh_token"], "new-refresh-a"
            )

        self.assertEqual(posted["url"], "https://account-a.example/cloudide/api/v3/trae/oauth/ExchangeToken")
        self.assertEqual(posted["json"]["RefreshToken"], "refresh-a")
        self.assertEqual(posted["json"]["UserID"], "account-a")


if __name__ == "__main__":
    unittest.main()
