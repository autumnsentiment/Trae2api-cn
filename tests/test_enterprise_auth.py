import asyncio
import hashlib
import unittest
from typing import Any, Mapping

from src.enterprise_auth import (
    EnterpriseAuthFlow,
    EnterpriseAuthState,
    EnterpriseTokens,
    classify_enterprise_account,
)


class FakeTransport:
    def __init__(self):
        self.token_payload: Mapping[str, Any] = {
            "accessToken": "test-access-token",
            "refreshToken": "test-refresh-token",
            "expiresIn": 3600,
        }
        self.profile_payload: Mapping[str, Any] = {
            "userId": "test-user",
            "tenantId": "test-tenant",
            "scope": "enterprise",
            "region": "CN",
        }
        self.refresh_payload: Mapping[str, Any] = {
            "accessToken": "refreshed-access-token",
            "expiresIn": 3600,
        }
        self.refresh_error = False

    async def exchange_authorization_code(
        self, code: str, redirect_uri: str
    ) -> Mapping[str, Any]:
        return self.token_payload

    async def fetch_user_info(self, access_token: str) -> Mapping[str, Any]:
        return self.profile_payload

    async def refresh_access_token(self, refresh_token: str) -> Mapping[str, Any]:
        if self.refresh_error:
            raise RuntimeError("refresh rejected")
        return self.refresh_payload


class EnterpriseAuthFlowTests(unittest.TestCase):
    def test_login_resolves_tenant_and_reaches_account_ready(self):
        async def scenario():
            flow = EnterpriseAuthFlow(FakeTransport())
            await flow.begin()
            result = await flow.complete_login("code", "app://callback")
            return result

        result = asyncio.run(scenario())
        self.assertEqual(result.state, EnterpriseAuthState.ACCOUNT_READY)
        self.assertEqual(result.identity.tenant_id, "test-tenant")
        self.assertEqual(result.identity.user_id, "test-user")
        self.assertEqual(
            result.credential_fingerprint(),
            hashlib.sha256(b"test-access-token").hexdigest()[:12],
        )

    def test_missing_tenant_fails_closed(self):
        transport = FakeTransport()
        transport.profile_payload = {"userId": "test-user"}

        async def scenario():
            flow = EnterpriseAuthFlow(transport)
            await flow.begin()
            return await flow.complete_login("code", "app://callback")

        result = asyncio.run(scenario())
        self.assertEqual(result.state, EnterpriseAuthState.FAILED)
        self.assertEqual(result.error_code, "tenant_missing")
        self.assertIsNone(result.identity)

    def test_missing_code_fails_before_transport_call(self):
        async def scenario():
            flow = EnterpriseAuthFlow(FakeTransport())
            await flow.begin()
            return await flow.complete_login("", "app://callback")

        result = asyncio.run(scenario())
        self.assertEqual(result.state, EnterpriseAuthState.FAILED)
        self.assertEqual(result.error_code, "authorization_code_missing")

    def test_refresh_preserves_refresh_token_when_omitted(self):
        async def scenario():
            flow = EnterpriseAuthFlow(FakeTransport())
            await flow.begin()
            await flow.complete_login("code", "app://callback")
            flow.context.tokens.expires_at = 0
            flow.context.tokens.access_token = ""
            return await flow.ensure_fresh_token()

        result = asyncio.run(scenario())
        self.assertEqual(result.state, EnterpriseAuthState.ACCOUNT_READY)
        self.assertEqual(result.tokens.access_token, "refreshed-access-token")
        self.assertEqual(result.tokens.refresh_token, "test-refresh-token")

    def test_refresh_failure_expires_account(self):
        transport = FakeTransport()
        transport.refresh_error = True

        async def scenario():
            flow = EnterpriseAuthFlow(transport)
            await flow.begin()
            await flow.complete_login("code", "app://callback")
            flow.context.tokens.access_token = ""
            return await flow.ensure_fresh_token()

        result = asyncio.run(scenario())
        self.assertEqual(result.state, EnterpriseAuthState.EXPIRED)
        self.assertEqual(result.error_code, "refresh_failed")

    def test_logout_clears_identity_and_tokens(self):
        async def scenario():
            flow = EnterpriseAuthFlow(FakeTransport())
            await flow.begin()
            await flow.complete_login("code", "app://callback")
            return flow.logout()

        result = asyncio.run(scenario())
        self.assertEqual(result.state, EnterpriseAuthState.SIGNED_OUT)
        self.assertIsNone(result.identity)
        self.assertEqual(result.tokens, EnterpriseTokens())

    def test_classify_tenant_member_account(self):
        result = classify_enterprise_account(
            {"Result": {"TenantID": "tenant-a", "UserID": "member-a"}},
            {
                "user_entitlement_pack_list": [
                    {"entitlement_base_info": {"product_type": 2}},
                    {"entitlement_base_info": {"product_type": 0}},
                ]
            },
        )
        self.assertTrue(result.is_enterprise_candidate)
        self.assertTrue(result.has_tenant)
        self.assertTrue(result.has_package)
        self.assertTrue(result.has_free)
        self.assertEqual(result.entitlements, (2, 0))
        self.assertEqual(len(result.tenant_fingerprint), 12)

    def test_classify_free_account_is_not_enterprise(self):
        result = classify_enterprise_account(
            {"Result": {"UserID": "free-user"}},
            {
                "user_entitlement_pack_list": [
                    {"entitlement_base_info": {"product_type": 0}}
                ]
            },
        )
        self.assertFalse(result.is_enterprise_candidate)
        self.assertFalse(result.has_tenant)
        self.assertFalse(result.has_package)
        self.assertTrue(result.has_free)
        self.assertEqual(result.tenant_fingerprint, "")

    def test_same_tenant_members_share_fingerprint(self):
        packs = {
            "user_entitlement_pack_list": [
                {"entitlement_base_info": {"product_type": 2}},
                {"entitlement_base_info": {"product_type": 0}},
            ]
        }
        first = classify_enterprise_account(
            {"Result": {"TenantID": "tenant-a", "UserID": "member-a"}}, packs
        )
        second = classify_enterprise_account(
            {"Result": {"TenantID": "tenant-a", "UserID": "member-b"}}, packs
        )
        self.assertEqual(first.tenant_fingerprint, second.tenant_fingerprint)
        self.assertNotEqual("member-a", first.tenant_fingerprint)


if __name__ == "__main__":
    unittest.main()
