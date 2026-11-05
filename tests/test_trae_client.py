import asyncio
import base64
import copy
import json
import os
import threading
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src import raw_client, trae_client
from src.model_limits import descriptor_multimodal


class ModelDescriptorCapabilityTests(unittest.TestCase):
    def test_missing_capability_fields_remain_unknown(self):
        capabilities = trae_client._descriptor_capabilities({"name": "unknown"})

        self.assertEqual(capabilities, {})

    def test_explicitly_unsupported_capabilities_remain_false(self):
        capabilities = trae_client._descriptor_capabilities(
            {
                "name": "text-only",
                "multimodal": False,
                "max_mode": False,
                "reasoning_effort_config": {"support_thinking": False},
            }
        )

        self.assertEqual(
            capabilities,
            {
                "multimodal": False,
                "reasoning_effort": [],
                "max_mode": False,
            },
        )

    def test_incomplete_positive_capabilities_remain_unknown(self):
        capabilities = trae_client._descriptor_capabilities(
            {
                "name": "incomplete",
                "max_mode": True,
                "reasoning_effort_config": {"support_thinking": True},
            }
        )

        self.assertEqual(capabilities, {})

    def test_complete_positive_capabilities_are_exposed(self):
        capabilities = trae_client._descriptor_capabilities(
            {
                "name": "complete",
                "max_mode": True,
                "context_window_size": {"max": 1_000_000},
                "prompt_max_tokens": 936_000,
                "max_tokens": 64_000,
                "reasoning_effort_config": {
                    "support_thinking": True,
                    "options": ["light", "high", "extra_high"],
                },
            }
        )

        self.assertEqual(
            capabilities,
            {
                "reasoning_effort": ["light", "high", "extra_high"],
                "max_mode": True,
                "max_context": 1_000_000,
            },
        )

    def test_multimodal_uses_nested_feature_when_top_level_is_absent(self):
        config = trae_client._build_web_model_config(
            {"name": "vision", "features": '{"multimodal":{"enable":true}}'}
        )
        self.assertIs(descriptor_multimodal(config), True)
        self.assertIs(config["multimodal"], True)

    def test_support_read_image_is_capability_evidence(self):
        config = trae_client._build_web_model_config(
            {"name": "vision", "support_read_image": True}
        )
        self.assertIs(descriptor_multimodal(config), True)

    def test_explicit_top_level_false_wins_over_nested_true(self):
        config = trae_client._build_web_model_config(
            {
                "name": "text",
                "multimodal": False,
                "support_read_image": True,
                "features": {"multimodal": {"enable": True}},
            }
        )
        self.assertIs(descriptor_multimodal(config), False)
        self.assertIs(config["multimodal"], False)


class LegacyReasoningEffortTests(unittest.TestCase):
    SUPPORTED = {
        "config_name": "reasoning-model",
        "reasoning_effort_config": {
            "support_thinking": True,
            "options": ["light", "high"],
        },
    }
    UNSUPPORTED = {
        "config_name": "text-only-model",
        "reasoning_effort_config": {"support_thinking": False},
    }

    def test_legacy_builders_send_effort_only_for_supported_descriptor(self):
        options = {
            "reasoning_effort": "ultra",
            "model_descriptor": self.SUPPORTED,
        }

        llm_body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "think"}],
            "reasoning-model",
            True,
            options=options,
        )
        work_body = trae_client.build_work_agent_body(
            [{"role": "user", "content": "think"}],
            "reasoning-model",
            True,
            options=options,
        )
        _, ide_body = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "think"}],
                "reasoning-model",
                options=options,
            )
        )

        for body in (llm_body, work_body, ide_body):
            self.assertEqual(body["reasoning_effort"], "high")
            self.assertNotIn("reasoning_effort_level", body)

    def test_legacy_builders_strip_effort_for_unsupported_descriptor(self):
        trace = {"reasoning_effort_applied": "high"}
        options = {
            "reasoning_effort": "high",
            "reasoning_effort_level": "extra_high",
            "model_descriptor": self.UNSUPPORTED,
            "_upstream_trace": trace,
        }

        llm_body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "think"}],
            "text-only-model",
            True,
            options=options,
        )
        work_body = trae_client.build_work_agent_body(
            [{"role": "user", "content": "think"}],
            "text-only-model",
            True,
            options=options,
        )
        _, ide_body = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "think"}],
                "text-only-model",
                options=options,
            )
        )

        for body in (llm_body, work_body, ide_body):
            self.assertNotIn("reasoning_effort", body)
            self.assertNotIn("reasoning_effort_level", body)
        self.assertNotIn("reasoning_effort_applied", trace)


class ModelRequestParallelLimitTests(unittest.TestCase):
    def test_default_capacity_is_four(self):
        self.assertEqual(
            trae_client._configured_model_request_parallel_limit({}),
            4,
        )

    def test_new_setting_precedes_legacy_alias(self):
        self.assertEqual(
            trae_client._configured_model_request_parallel_limit(
                {
                    "TRAE_MODEL_REQUEST_PARALLEL_LIMIT": "6",
                    "TRAE_WEB_PARALLEL_LIMIT": "2",
                }
            ),
            6,
        )

    def test_legacy_alias_and_bounds_remain_supported(self):
        self.assertEqual(
            trae_client._configured_model_request_parallel_limit(
                {"TRAE_WEB_PARALLEL_LIMIT": "5"}
            ),
            5,
        )
        self.assertEqual(
            trae_client._configured_model_request_parallel_limit(
                {"TRAE_MODEL_REQUEST_PARALLEL_LIMIT": "0"}
            ),
            1,
        )
        self.assertEqual(
            trae_client._configured_model_request_parallel_limit(
                {"TRAE_MODEL_REQUEST_PARALLEL_LIMIT": "1000"}
            ),
            64,
        )
        self.assertEqual(
            trae_client._configured_model_request_parallel_limit(
                {"TRAE_MODEL_REQUEST_PARALLEL_LIMIT": "invalid"}
            ),
            4,
        )


def _fake_jwt(user_id: str, issued_at: int) -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return ".".join(
        (
            encode({"alg": "none", "typ": "JWT"}),
            encode({"data": {"id": user_id}, "iat": issued_at}),
            f"signature-{issued_at}",
        )
    )


class CheckinDeviceTests(unittest.TestCase):
    def test_device_id_survives_jwt_refresh_for_same_account(self):
        first = _fake_jwt("account-42", 100)
        refreshed = _fake_jwt("account-42", 200)

        first_id = trae_client.checkin_device_id_for(first)
        refreshed_id = trae_client.checkin_device_id_for(refreshed)

        self.assertEqual(first_id, refreshed_id)
        self.assertRegex(first_id, r"^\d{16}$")

    def test_device_ids_are_distinct_between_accounts(self):
        first = _fake_jwt("account-42", 100)
        second = _fake_jwt("account-43", 100)

        self.assertNotEqual(
            trae_client.checkin_device_id_for(first),
            trae_client.checkin_device_id_for(second),
        )

    def test_jwt_identity_wins_over_legacy_account_store_key(self):
        first = _fake_jwt("account-42", 100)
        refreshed = _fake_jwt("account-42", 200)

        self.assertEqual(
            trae_client.checkin_device_id_for(first, "legacy-row-key"),
            trae_client.checkin_device_id_for(refreshed, "renamed-row-key"),
        )
        self.assertEqual(
            trae_client.build_checkin_headers(first, "legacy-row-key")["x-device-id"],
            trae_client.build_checkin_headers(first)["x-device-id"],
        )

    def test_headers_include_client_checkin_metadata(self):
        headers = trae_client.build_checkin_headers(_fake_jwt("account-44", 100))

        self.assertEqual(headers["x-device-type"], "windows")
        self.assertRegex(headers["x-device-id"], r"^\d{16}$")
        self.assertNotIn("x-os-version", headers)
        self.assertNotIn("x-app-version", headers)

    def test_global_traework_device_id_override_wins(self):
        token = _fake_jwt("account-override", 100)

        with patch.dict(os.environ, {"TRAE_CHECKIN_DEVICE_ID": "traework-machine-did"}):
            self.assertEqual(
                trae_client.checkin_device_id_for(token), "traework-machine-did"
            )

    def test_per_account_traework_device_id_override_wins(self):
        first = _fake_jwt("account-first", 100)
        second = _fake_jwt("account-second", 100)
        mapping = json.dumps(
            {
                "account-first": "first-machine-did",
                "legacy-second-row": "second-machine-did",
            }
        )

        with patch.dict(
            os.environ,
            {
                "TRAE_CHECKIN_DEVICE_ID": "fallback-machine-did",
                "TRAE_CHECKIN_DEVICE_IDS_JSON": mapping,
            },
        ):
            self.assertEqual(
                trae_client.checkin_device_id_for(first), "first-machine-did"
            )
            self.assertEqual(
                trae_client.checkin_device_id_for(second, "legacy-second-row"),
                "second-machine-did",
            )

    def test_device_already_checked_code_does_not_claim_account_checked(self):
        token = _fake_jwt("account-45", 100)

        class FakeResponse:
            status_code = 200
            text = '{"code":9095,"message":"device already checked"}'

            @staticmethod
            def json():
                return {"code": 9095, "message": "device already checked"}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json, headers):
                self.headers = headers
                return FakeResponse()

        fake_client = FakeClient()
        with patch("src.trae_client.httpx.AsyncClient", return_value=fake_client):
            result = asyncio.run(
                trae_client._post_checkin(
                    "/trae/api/v2/ug/checkin_credits/claim",
                    token,
                    account_id="account-45",
                )
            )

        self.assertEqual(result["code"], 9095)
        self.assertNotIn("checked_in", result)
        self.assertNotIn("success", result)
        self.assertEqual(
            fake_client.headers["x-device-id"],
            trae_client.checkin_device_id_for(token, "account-45"),
        )

    def test_rate_limit_is_returned_without_immediate_retry(self):
        upstream = AsyncMock(
            return_value={"code": 9074, "message": "operation too frequent"}
        )
        with patch("src.trae_client._post_checkin", new=upstream):
            result = asyncio.run(
                trae_client.claim_checkin_credits("token-1", "account-1")
            )

        self.assertEqual(result["code"], 9074)
        upstream.assert_awaited_once_with(
            "/trae/api/v2/ug/checkin_credits/claim", "token-1", account_id="account-1"
        )

    def test_status_rejects_non_boolean_protocol_fields(self):
        class FakeResponse:
            status_code = 200
            text = '{"code":0,"enable":1,"checked_in":false}'

            @staticmethod
            def json():
                return {"code": 0, "enable": 1, "checked_in": False}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json, headers):
                return FakeResponse()

        with patch("src.trae_client.httpx.AsyncClient", return_value=FakeClient()):
            with self.assertRaisesRegex(RuntimeError, "must be boolean"):
                asyncio.run(
                    trae_client._post_checkin(
                        "/trae/api/v2/ug/checkin_credits/status",
                        _fake_jwt("account-status", 100),
                    )
                )

    def test_status_drops_non_positive_credits(self):
        class FakeResponse:
            status_code = 200
            text = '{"code":0,"enable":true,"checked_in":false,"credits":0}'

            @staticmethod
            def json():
                return {"code": 0, "enable": True, "checked_in": False, "credits": 0}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json, headers):
                return FakeResponse()

        with patch("src.trae_client.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(
                trae_client._post_checkin(
                    "/trae/api/v2/ug/checkin_credits/status",
                    _fake_jwt("account-status", 100),
                )
            )

        self.assertNotIn("credits", result)


class WebModelCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_cache_isolates_gateway_and_provider_context(self):
        fetch = AsyncMock(side_effect=[
            {"model": {"config_name": name}}
            for name in ("base", "remote", "native", "provider")
        ])
        contexts = (
            ("https://remote-a.example", "https://native-a.example", {"region": "cn"}),
            ("https://remote-b.example", "https://native-a.example", {"region": "cn"}),
            ("https://remote-a.example", "https://native-b.example", {"region": "cn"}),
            ("https://remote-a.example", "https://native-a.example", {"region": "global"}),
        )
        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_MODEL_CATALOG_INFLIGHT", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=fetch),
            patch.object(trae_client, "remote_base_url") as remote_gateway,
            patch.object(trae_client, "_model_gateway_base") as native_gateway,
        ):
            names = []
            for remote, native, provider in (*contexts, contexts[0]):
                remote_gateway.return_value = remote
                native_gateway.return_value = native
                result = await trae_client._cached_model_catalog(
                    "web", "token-context", user_id_override="account",
                    provider_specific=provider,
                )
                names.append(result["model"]["config_name"])

        self.assertEqual(names, ["base", "remote", "native", "provider", "base"])
        self.assertEqual(fetch.await_count, 4)

    async def test_catalog_cache_preserves_nested_snapshot_and_returns_copies(self):
        upstream = {"model": {"capabilities": {"reasoning_effort": ["high"]}}}
        fetch = AsyncMock(return_value=upstream)
        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_MODEL_CATALOG_INFLIGHT", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=fetch),
        ):
            first = await trae_client._cached_model_catalog(
                "web", "token-copy", user_id_override="account",
                provider_specific={},
            )
            first["model"]["capabilities"]["reasoning_effort"].append("caller")
            upstream["model"]["capabilities"]["reasoning_effort"].append("upstream")
            second = await trae_client._cached_model_catalog(
                "web", "token-copy", user_id_override="account",
                provider_specific={},
            )
            self.assertEqual(second["model"]["capabilities"]["reasoning_effort"], ["high"])
            second["model"]["capabilities"]["reasoning_effort"].append("cached-caller")
            third = await trae_client._cached_model_catalog(
                "web", "token-copy", user_id_override="account",
                provider_specific={},
            )

        self.assertEqual(third["model"]["capabilities"]["reasoning_effort"], ["high"])
        self.assertEqual(fetch.await_count, 1)

    async def test_catalog_cache_coalesces_simultaneous_cold_reads(self):
        started = asyncio.Event()
        release = asyncio.Event()
        fetch = AsyncMock()

        async def fetch_catalog(_token, **_kwargs):
            started.set()
            await release.wait()
            return {"same-model": {"config_name": "same-model"}}

        fetch.side_effect = fetch_catalog
        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_MODEL_CATALOG_INFLIGHT", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=fetch),
        ):
            first = asyncio.create_task(
                trae_client._cached_model_catalog("web", "token-coalesce", user_id_override="account")
            )
            second = asyncio.create_task(
                trae_client._cached_model_catalog("web", "token-coalesce", user_id_override="account")
            )
            await started.wait()
            release.set()
            result_one, result_two = await asyncio.gather(first, second)

        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(result_one, result_two)

    async def test_cancelled_catalog_waiter_does_not_cancel_shared_fetch(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def fetch_catalog(_token, **_kwargs):
            started.set()
            await release.wait()
            return {"same-model": {"config_name": "same-model"}}

        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_MODEL_CATALOG_INFLIGHT", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=fetch_catalog),
        ):
            cancelled = asyncio.create_task(
                trae_client._cached_model_catalog("web", "token-cancel", user_id_override="account")
            )
            survivor = asyncio.create_task(
                trae_client._cached_model_catalog("web", "token-cancel", user_id_override="account")
            )
            await started.wait()
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            release.set()
            result = await survivor

        self.assertEqual(result["same-model"]["config_name"], "same-model")

    async def test_warm_web_catalog_still_discovers_solo_only_model(self):
        web_fetch = AsyncMock(return_value={
            "remote-model": {"config_name": "remote-model"},
        })
        solo_fetch = AsyncMock(return_value={
            "solo-only": {"config_name": "solo-only", "raw_model_name": "solo-only__dev"},
        })
        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_MODEL_CATALOG_INFLIGHT", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=web_fetch),
            patch.object(trae_client, "_fetch_solo_model_configs", new=solo_fetch),
        ):
            remote = await trae_client._get_web_custom_model(
                "remote-model", token_override="token-solo", user_id_override="account"
            )
            solo = await trae_client._get_web_custom_model(
                "solo-only", token_override="token-solo", user_id_override="account"
            )

        self.assertEqual(remote["config_name"], "remote-model")
        self.assertEqual(solo["config_name"], "solo-only")
        self.assertEqual(web_fetch.await_count, 1)
        self.assertEqual(solo_fetch.await_count, 1)

    async def test_empty_catalog_is_retried_instead_of_cached(self):
        fetch = AsyncMock(
            side_effect=[
                {},
                {
                    "glm-5.3-flash": {
                        "config_name": "GLM-5.3-Flash",
                        "model_name": "glm-5.3-flash",
                    }
                },
            ]
        )

        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=fetch),
            patch.object(
                trae_client, "_fetch_solo_model_configs", new=AsyncMock(return_value={})
            ),
        ):
            first = await trae_client._get_web_custom_model(
                "glm-5.3-flash",
                token_override="token-retry",
                user_id_override="account-retry",
            )
            second = await trae_client._get_web_custom_model(
                "glm-5.3-flash",
                token_override="token-retry",
                user_id_override="account-retry",
            )

        self.assertIsNone(first)
        self.assertEqual(second["config_name"], "GLM-5.3-Flash")
        self.assertEqual(fetch.await_count, 2)

    async def test_lowercase_request_matches_wire_config_without_model_list_warmup(self):
        """A lower-case public id must resolve the exact live config directly."""

        async def fetch(_token, **_kwargs):
            # Older catalog payloads used a display label in ``name`` while
            # exposing the request identifiers in separate fields.
            return {
                "GLM-5.3 Flash": {
                    "name": "GLM-5.3 Flash",
                    "config_name": "GLM-5.3-Flash",
                    "model_name": "glm-5.3-flash",
                    "display_name": "GLM-5.3 Flash",
                }
            }

        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", side_effect=fetch),
            patch.object(
                trae_client, "_fetch_solo_model_configs", new=AsyncMock(return_value={})
            ) as solo_fetch,
        ):
            resolved = await trae_client._get_web_custom_model(
                "glm-5.3-flash",
                token_override="token-lower",
                user_id_override="account-lower",
            )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["config_name"], "GLM-5.3-Flash")
        self.assertEqual(resolved["model_name"], "glm-5.3-flash")
        solo_fetch.assert_not_awaited()

    async def test_bound_account_model_config_cache_does_not_cross_contaminate(self):
        token_a = _fake_jwt("account-a", 100)
        token_b = _fake_jwt("account-b", 100)
        fetched = []

        async def fetch(token):
            fetched.append(token)
            return {
                "requested-model": {
                    "config_name": "config-a" if token == token_a else "config-b"
                }
            }

        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", side_effect=fetch),
            patch.object(trae_client.auth, "get_user_id", return_value="account-a"),
        ):
            model_b = await trae_client._get_web_custom_model(
                "requested-model",
                token_override=token_b,
                user_id_override="account-b",
            )
            # A second B request should reuse B's own cache entry.
            cached_b = await trae_client._get_web_custom_model(
                "requested-model",
                token_override=token_b,
                user_id_override="account-b",
            )
            model_a = await trae_client._get_web_custom_model(
                "requested-model",
                token_override=token_a,
                user_id_override="account-a",
            )

        self.assertEqual(model_b["config_name"], "config-b")
        self.assertEqual(cached_b["config_name"], "config-b")
        self.assertEqual(model_a["config_name"], "config-a")
        self.assertEqual(fetched, [token_b, token_a])

    async def test_web_session_resolves_custom_model_with_bound_credentials(self):
        token_b = _fake_jwt("account-b", 100)

        class FakeResponse:
            status_code = 200
            text = "{}"

            @staticmethod
            def json():
                return {"data": {"chat_session_id": "session-b", "message_id": "message-b"}}

        class FakeClient:
            request = None

            async def post(self, *_args, **_kwargs):
                self.request = _kwargs
                return FakeResponse()

        lookup = AsyncMock(
            return_value={"name": "requested-model", "config_name": "config-b"}
        )
        provider_specific = {
            "webId": "bound-web",
            "bizUserId": "bound-biz",
            "appLanguage": "ja-JP",
            "userRegion": "JP",
        }
        fake_client = FakeClient()
        with (
            patch.object(trae_client, "_get_web_custom_model", new=lookup),
            patch.object(
                trae_client.auth,
                "get_psd",
                return_value={
                    "webId": "wrong-global-web",
                    "appLanguage": "wrong-global-language",
                    "userRegion": "wrong-global-region",
                },
            ),
        ):
            session_id, message_id = await trae_client.create_web_session(
                fake_client,
                "requested-model",
                [{"role": "user", "content": "hello"}],
                options={
                    "_auth_token": token_b,
                    "_auth_user_id": "account-b",
                    "provider_specific": provider_specific,
                },
            )

        self.assertEqual((session_id, message_id), ("session-b", "message-b"))
        lookup.assert_awaited_once_with(
            "requested-model",
            token_override=token_b,
            user_id_override="account-b",
            provider_specific=provider_specific,
            raw_chat_function="solo_agent_remote",
        )
        common = json.loads(
            fake_client.request["json"]["initial_message"]["common_params"]
        )
        self.assertEqual(common["web_id"], "bound-web")
        self.assertEqual(common["biz_user_id"], "bound-biz")
        self.assertEqual(fake_client.request["headers"]["X-Preferenced-Language"], "ja-JP")
        self.assertEqual(fake_client.request["headers"]["x-user-region"], "JP")

    def test_web_headers_preserve_bound_empty_provider_metadata(self):
        with patch.object(
            trae_client.auth,
            "get_psd",
            return_value={"appLanguage": "wrong", "userRegion": "wrong"},
        ):
            headers = trae_client.build_web_headers(
                "token", {"provider_specific": {}}
            )

        self.assertEqual(headers["X-Preferenced-Language"], "zh-CN")
        self.assertEqual(headers["x-user-region"], "CN")


class SessionUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_session_usage_matches_traework_request_and_response(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "code": 0,
                    "data": {
                        "user_usage_group_by_session": {
                            "credits_float": 0.92,
                            "extra_info": {
                                "input_token": 29558,
                                "output_token": 11,
                            },
                        }
                    },
                }

        class Client:
            request = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, url, **kwargs):
                self.request = (url, kwargs)
                return Response()

        client = Client()
        with patch("src.trae_client.httpx.AsyncClient", return_value=client):
            result = await trae_client.fetch_session_usage(
                "user-message-1",
                "jwt-token",
                base_url="https://usage.example",
            )

        url, request = client.request
        self.assertEqual(
            url,
            "https://usage.example/api/v1/commercial/get_session_usage",
        )
        self.assertEqual(request["json"], {"session_id": "user-message-1"})
        self.assertEqual(
            request["headers"],
            {
                "Authorization": "Cloud-IDE-JWT jwt-token",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(
            result,
            {
                "credits_consumed": 0.92,
                "credits_source": "session_usage",
                "prompt_tokens": 29558,
                "completion_tokens": 11,
                "total_tokens": 29569,
            },
        )
        self.assertNotIn("extra_info", result)

    async def test_fetch_session_usage_preserves_cache_aggregate_without_duplicate_details(self):
        counters = {
            "cache_read_token": 22784, "cache_write_token": 0,
            "input_token": 22857, "output_token": 211,
            "prompt": "sensitive prompt must not escape",
        }
        payload = {"user_usage_group_by_session": {
            "credits_float": 0.11,
            "extra_info": counters,
            "usage_group_details": [{"extra_info": counters}],
        }}
        class Response:
            status_code = 200
            def json(self):
                return payload

        class Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_args):
                return False
            async def post(self, *_args, **_kwargs):
                return Response()

        with patch("src.trae_client.httpx.AsyncClient", return_value=Client()):
            result = await trae_client.fetch_session_usage("turn-cache", "jwt-token")
        self.assertEqual(result, {
            "credits_consumed": 0.11, "credits_source": "session_usage",
            "prompt_tokens": 22857, "completion_tokens": 211,
            "total_tokens": 23068,
            "prompt_tokens_details": {"cached_tokens": 22784, "cache_write_tokens": 0},
        })

    def test_session_token_usage_rejects_invalid_and_missing_values(self):
        for invalid in (True, False, -1, 1.5, float("nan"), float("inf"), "-2", "NaN", "secret"):
            with self.subTest(invalid=invalid):
                self.assertEqual(trae_client._session_token_usage({
                    "extra_info": {"input_token": invalid, "cache_read_token": invalid},
                    "usage_group_details": [{"extra_info": {"cache_read_token": 100}}],
                }, "turn"), {})
        self.assertEqual(trae_client._session_token_usage({"extra_info": {"input_token": 100}}, "turn"), {"prompt_tokens": 100})
        self.assertEqual(trae_client._session_token_usage({"arbitrary_metadata": {"cache_read_token": 100}}, "turn"), {})

    def test_session_token_usage_aggregate_zero_beats_detail_and_ignores_unrelated_data(self):
        result = trae_client._session_token_usage({"data": {"usage": {
            "userUsageGroupBySession": {
                "extraInfo": {"cacheReadToken": "0", "cacheWriteToken": 0},
                "usageGroupDetails": [{"extra_info": {"cache_read_token": 100}}],
            }
        }}}, "turn")
        self.assertEqual(result, {"prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}})

    def test_session_token_usage_detail_fallback_requires_turn_evidence_for_multiple_rows(self):
        first = {"session_id": "turn", "extra_info": {"input_token": 100, "output_token": 10, "cache_read_token": 80}}
        second = {"session_id": "turn", "extra_info": {"input_token": 200, "output_token": 20, "cache_read_token": 160}}
        result = trae_client._session_token_usage({"usage_group_details": [first, second]}, "turn")
        self.assertEqual(result, {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330, "prompt_tokens_details": {"cached_tokens": 240}})
        for identity in (None, "other-turn"):
            with self.subTest(identity=identity):
                self.assertEqual(trae_client._session_token_usage({"usage_group_details": [first, {**second, "session_id": identity}]}, "turn"), {})
        self.assertEqual(trae_client._session_token_usage({"usage_group_details": [first]}, "turn"), {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "prompt_tokens_details": {"cached_tokens": 80}})
        self.assertEqual(trae_client._session_token_usage({"usage_group_details": [first]}, "different-turn"), {})

    async def test_fetch_session_usage_preserves_explicit_zero(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"user_usage_group_by_session": {"credits_float": 0}}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return Response()

        with patch("src.trae_client.httpx.AsyncClient", return_value=Client()):
            result = await trae_client.fetch_session_usage("turn-0", "jwt-token")

        self.assertEqual(result["credits_consumed"], 0)

    async def test_fetch_session_usage_accepts_nested_camel_case_string_amount(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "usage": {
                            "userUsageGroupBySession": {
                                "amountFloat": "0.37",
                            }
                        }
                    }
                }

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return Response()

        with patch("src.trae_client.httpx.AsyncClient", return_value=Client()):
            result = await trae_client.fetch_session_usage("turn-camel", "jwt-token")

        self.assertEqual(result["credits_consumed"], 0.37)
        self.assertEqual(result["credits_source"], "session_usage")

    async def test_fetch_session_usage_rejects_business_error(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"code": 2001, "message": "record not found"}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return Response()

        with patch("src.trae_client.httpx.AsyncClient", return_value=Client()):
            with self.assertRaisesRegex(RuntimeError, "record not found"):
                await trae_client.fetch_session_usage("missing", "jwt-token")


class AccountCreditsParserTests(unittest.TestCase):
    def test_preserves_fractional_limits_usage_and_remaining(self):
        result = trae_client.parse_account_credits(
            {
                "user_entitlement_pack_list": [
                    {
                        "entitlement_base_info": {
                            "available_endpoint": 0,
                            "quota": {"credits_limit": 100.75},
                        },
                        "usage": {"credits_amount": 12.34},
                    },
                    {
                        "entitlement_base_info": {
                            "available_endpoint": 0,
                            "quota": {"credits_limit": 10.25},
                        },
                        "usage": {"credits_amount": 0.56},
                    },
                ]
            }
        )

        self.assertEqual(result["total_limit"], 111)
        self.assertAlmostEqual(result["used"], 12.9)
        self.assertAlmostEqual(result["remaining"], 98.1)
        self.assertIsInstance(result["total_limit"], int)
        self.assertIsInstance(result["used"], float)
        json.dumps(result)

    def test_accepts_numeric_strings_without_truncating(self):
        result = trae_client.parse_account_credits(
            {
                "user_entitlement_pack_list": [
                    {
                        "entitlement_base_info": {
                            "available_endpoint": 1,
                            "quota": {"credits_limit": "20.50"},
                        },
                        "usage": {"credits_amount": "3.25"},
                    }
                ]
            },
            available_endpoint_filter=1,
        )

        self.assertEqual(result["total_limit"], 20.5)
        self.assertEqual(result["used"], 3.25)
        self.assertEqual(result["remaining"], 17.25)
        self.assertTrue(all(not isinstance(value, Decimal) for value in result.values()))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the caller workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

CLIENT_CONTEXT = {
    "workspace_path": r"D:\work\demo",
    "system_type": "Windows 11",
    "terminal_context": [
        {"shell": "PowerShell", "cwd": r"D:\work\demo"}
    ],
}


class ConvertOpenAiMessagesTests(unittest.TestCase):
    def test_preserves_tool_request_result_and_client_runtime(self):
        messages = [
            {"role": "user", "content": "Inspect README.md"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "project documentation",
            },
        ]

        converted = trae_client.convert_openai_messages(
            messages,
            {"tools": TOOLS, "client_context": CLIENT_CONTEXT},
        )

        self.assertEqual(converted[0]["role"], "system")
        self.assertIn("read_file", converted[0]["content"])
        self.assertIn(r"D:\\work\\demo", converted[0]["content"])
        self.assertIn("Client tool history (already handled; do not repeat)", converted[2]["content"])
        self.assertIn("call_1", converted[2]["content"])
        self.assertEqual(converted[3]["role"], "user")
        self.assertIn("Client tool result [call_1] read_file", converted[3]["content"])
        self.assertIn("project documentation", converted[3]["content"])

    def test_empty_tool_policy_injects_no_tools_runtime(self):
        converted = trae_client.convert_openai_messages(
            [{"role": "user", "content": "answer directly"}],
            {"tools": [], "tool_choice": "none"},
        )
        self.assertEqual(converted[0]["role"], "system")
        self.assertIn("No client tools are available", converted[0]["content"])
        self.assertIn("Tool choice is none", converted[0]["content"])

    def test_none_assistant_content_does_not_become_literal_none(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "contents",
            },
        ]
        flattened = trae_client.flatten_query(messages)
        structured = json.dumps(
            trae_client.build_web_content(messages), ensure_ascii=False
        )
        self.assertNotIn("None", flattened)
        self.assertNotIn("None", structured)
        self.assertIn("call_1 read_file", structured)

    def test_response_style_directive_reaches_plain_and_tool_requests(self):
        """The remote agent returns reasoning as the answer without this."""

        plain = trae_client._messages_with_client_runtime(
            [{"role": "user", "content": "why is TCP slower than UDP?"}], {}
        )
        self.assertEqual(plain[0]["role"], "system")
        directive = raw_client.response_style_instruction()
        self.assertTrue(directive)
        self.assertIn(directive, plain[0]["content"])
        self.assertIn("only the final answer", plain[0]["content"])
        self.assertIn("only the final answer", trae_client.flatten_query(plain))

        with_tools = trae_client._messages_with_client_runtime(
            [{"role": "user", "content": "read the readme"}],
            {"tools": TOOLS},
        )
        self.assertEqual(with_tools[0]["role"], "system")
        self.assertIn(directive, with_tools[0]["content"])
        # The tool runtime description must survive alongside the directive.
        self.assertIn("read_file", with_tools[0]["content"])

    def test_verbose_reasoning_opt_out_removes_directive(self):
        with patch.dict(os.environ, {"TRAE_VERBOSE_REASONING": "1"}, clear=False):
            messages = trae_client._messages_with_client_runtime(
                [{"role": "user", "content": "hello"}], {}
            )

        self.assertEqual([m["role"] for m in messages], ["user"])

    def test_renderer_tool_use_and_tool_result_survive_remote_flattening(self):
        """Trae's renderer keeps tool history in content blocks, not arrays."""

        messages = [
            {"role": "user", "content": "Download release.zip"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "toolCallId": "call_abc",
                        "name": "download",
                        "parameters": {
                            "url": "https://example.com/release.zip",
                            "dest": "release.zip",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "toolCallId": "call_abc",
                "name": "download",
                "content": [
                    {
                        "type": "tool_result",
                        "toolCallId": "call_abc",
                        "value": [{"type": "text", "value": "Downloaded 4096 bytes"}],
                        "isError": False,
                    }
                ],
            },
        ]

        flattened = trae_client.flatten_query(messages)
        structured = json.dumps(
            trae_client.build_web_content(messages), ensure_ascii=False
        )

        for blob in (flattened, structured):
            self.assertIn("call_abc", blob)
            self.assertIn("download", blob)
            self.assertIn("Downloaded 4096 bytes", blob)
            self.assertNotIn("None", blob)

    def test_renderer_user_tool_result_preserves_status_and_call_id(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "toolCallId": "call_download_1",
                        "name": "download_file",
                        "parameters": {
                            "url": "https://example.com/a.zip",
                            "path": "a.zip",
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "toolCallId": "call_download_1",
                        "value": [{"type": "text", "value": "download denied"}],
                        "isError": True,
                    }
                ],
            },
        ]

        flattened = trae_client.flatten_query(messages)
        structured = json.dumps(
            trae_client.build_web_content(messages), ensure_ascii=False
        )
        converted = trae_client.convert_openai_messages(messages)

        for blob in (flattened, structured, json.dumps(converted, ensure_ascii=False)):
            self.assertIn("call_download_1", blob)
            self.assertIn("download_file", blob)
            self.assertIn("status=failed", blob)
            self.assertIn("download denied", blob)

        converted_blob = json.dumps(converted, ensure_ascii=False)
        self.assertEqual(converted_blob.count("Client tool result [call_download_1]"), 1)

    def test_renderer_tool_history_marks_remote_tool_protocol_requested(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "toolCallId": "call_write_1",
                        "name": "write_file",
                        "parameters": {"path": "README.md", "content": "ok"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "toolCallId": "call_write_1",
                        "value": [{"type": "text", "value": "wrote"}],
                        "isError": False,
                    }
                ],
            },
        ]

        self.assertTrue(trae_client._tool_protocol_requested({}, messages))
        prepared = trae_client._messages_with_client_runtime(messages, {})
        self.assertEqual(prepared[0]["role"], "system")
        self.assertIn("external client", prepared[0]["content"])


class IdeRequestContextTests(unittest.TestCase):
    def test_legacy_ide_response_holds_shared_slot_until_close(self):
        class FakeResponse:
            status_code = 200

            def close(self):
                self.closed = True

        class FakeClient:
            def __init__(self, **kwargs):
                self.headers = kwargs["headers"]
                self.response = None

            def build_request(self, *args, **kwargs):
                return (args, kwargs)

            def send(self, request, stream=True):
                self.response = FakeResponse()
                return self.response

            def close(self):
                self.closed = True

        with (
            patch.object(trae_client, "_WEB_SLOTS", {}),
            patch.object(trae_client, "_WEB_PARALLEL_LIMIT", 1),
            patch(
                "src.trae_client.auth.get_account_record",
                return_value={"user_id": "acct"},
            ),
            patch("src.trae_client.auth.get_auth", return_value=SimpleNamespace(host="https://example.test")),
            patch("src.trae_client.httpx.Client", side_effect=FakeClient),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            response = asyncio.run(
                trae_client.send_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "auto",
                    True,
                    {"_account_id": "acct", "_auth_token": "bound-token"},
                )
            )
            sem = trae_client._WEB_SLOTS["acct"]
            self.assertTrue(sem.locked())
            response.close()
            response.close()
            self.assertFalse(sem.locked())

    def test_legacy_ide_cancel_waits_for_sync_send_before_releasing_slot(self):
        started = threading.Event()
        finish = threading.Event()

        class FakeResponse:
            status_code = 200

            def close(self):
                self.closed = True

        class FakeClient:
            def __init__(self, **kwargs):
                self.headers = kwargs["headers"]

            def build_request(self, *args, **kwargs):
                return (args, kwargs)

            def send(self, request, stream=True):
                started.set()
                finish.wait(2)
                return FakeResponse()

            def close(self):
                self.closed = True

        async def exercise():
            with (
                patch.object(trae_client, "_WEB_SLOTS", {}),
                patch.object(trae_client, "_WEB_PARALLEL_LIMIT", 1),
                patch(
                    "src.trae_client.auth.get_account_record",
                    return_value={"user_id": "acct"},
                ),
                patch("src.trae_client.auth.get_auth", return_value=SimpleNamespace(host="https://example.test")),
                patch("src.trae_client.httpx.Client", side_effect=FakeClient),
                patch("src.trae_client.build_headers", return_value={}),
            ):
                task = asyncio.create_task(
                    trae_client.send_chat_request(
                        [{"role": "user", "content": "hello"}],
                        "auto",
                        True,
                        {"_account_id": "acct", "_auth_token": "bound-token"},
                    )
                )
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                sem = trae_client._WEB_SLOTS["acct"]
                self.assertTrue(sem.locked())
                finish.set()
                for _ in range(100):
                    if not sem.locked():
                        break
                    await asyncio.sleep(0.01)
                self.assertFalse(sem.locked())

        asyncio.run(exercise())

    def test_runtime_prompt_uses_inherited_response_tools(self):
        messages = trae_client._messages_with_client_runtime(
            [{"role": "user", "content": "继续"}],
            {
                "_inherited_tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "download_file",
                            "description": "Download a client-side file.",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "_tool_protocol_requested": True,
            },
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("download_file", messages[0]["content"])
        self.assertNotIn("No client tools are available", messages[0]["content"])

    def test_ide_request_uses_bound_account_credential_snapshot(self):
        calls = []

        class FakeResponse:
            status_code = 200

            def close(self):
                pass

        class FakeClient:
            def __init__(self, *, headers, timeout, http2):
                self.headers = headers
                self.timeout = timeout
                self.http2 = http2
                calls.append(self)

            def build_request(self, method, url, **kwargs):
                self.request = (method, url, kwargs)
                return self.request

            def send(self, request, stream=True):
                self.sent = (request, stream)
                return FakeResponse()

            def close(self):
                self.closed = True

        bound_options = {
            "_account_id": "charged-account",
            "_auth_token": "bound-token",
        }
        with (
            patch(
                "src.trae_client.auth.get_account_record",
                return_value={
                    "user_id": "charged-account",
                    "host": "https://charged.example",
                    "token": "stale-token",
                },
            ),
            patch("src.trae_client.auth.get_token", return_value="selected-token"),
            patch("src.trae_client.auth.get_user_id", return_value="selected-account"),
            patch("src.trae_client.auth.get_auth", return_value=SimpleNamespace(host="https://selected.example")),
            patch("src.trae_client.auth.maybe_refresh", new=AsyncMock()) as refresh,
            patch("src.trae_client.httpx.Client", side_effect=FakeClient),
            patch(
                "src.trae_client.build_headers",
                side_effect=lambda **kwargs: {
                    "Authorization": "Cloud-IDE-JWT " + kwargs["token_override"],
                    "x-uid": kwargs["user_id_override"],
                },
            ) as build_headers,
        ):
            response = asyncio.run(
                trae_client.send_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "auto",
                    False,
                    bound_options,
                )
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers["Authorization"], "Cloud-IDE-JWT bound-token")
        self.assertEqual(response.auth_token, "bound-token")
        self.assertNotIn("bound-token", repr(response))
        self.assertEqual(
            response.usage_turn_id, calls[0].request[2]["json"]["session_id"]
        )
        self.assertEqual(calls[0].headers["x-uid"], "charged-account")
        self.assertEqual(calls[0].request[1], "https://charged.example/api/agent/v3/llm_utils_chat")
        self.assertEqual(calls[0].headers["x-device-type"], trae_client.SOLO_DEVICE_TYPE)
        self.assertEqual(
            calls[0].headers["x-device-platform"],
            trae_client.SOLO_PLATFORM,
        )
        self.assertEqual(
            calls[0].headers["x-version-code"],
            trae_client.SOLO_GATEWAY_VERSION_CODE,
        )
        self.assertEqual(calls[0].headers["package-type"], trae_client.SOLO_PACKAGE_TYPE)
        self.assertEqual(calls[0].headers["x-lscbd-aid"], trae_client.SOLO_LSCBD_AID)
        self.assertEqual(calls[0].headers["x-lscbd-platform"], trae_client.SOLO_PLATFORM)
        self.assertEqual(
            calls[0].headers["Referer"],
            "https://charged.example/api/agent/v3/llm_utils_chat",
        )
        build_headers.assert_called_once_with(
            token_override="bound-token",
            user_id_override="charged-account",
        )
        refresh.assert_not_awaited()
        response.close()

    def test_unbound_ide_request_can_refresh_selected_account(self):
        class FakeResponse:
            status_code = 200

            def close(self):
                pass

        class FakeClient:
            def __init__(self, **kwargs):
                self.headers = kwargs["headers"]

            def build_request(self, *args, **kwargs):
                return (args, kwargs)

            def send(self, request, stream=True):
                return FakeResponse()

            def close(self):
                pass

        with (
            patch("src.trae_client.auth.maybe_refresh", new=AsyncMock()) as refresh,
            patch("src.trae_client.auth.get_token", return_value="selected-token"),
            patch("src.trae_client.auth.get_user_id", return_value="selected-account"),
            patch("src.trae_client.auth.get_auth", return_value=SimpleNamespace(host="https://selected.example")),
            patch("src.trae_client.httpx.Client", side_effect=FakeClient),
            patch("src.trae_client.build_headers", return_value={}),
        ):
            response = asyncio.run(
                trae_client.send_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "auto",
                    False,
                )
            )

        refresh.assert_awaited_once()
        self.assertEqual(response.auth_token, "selected-token")
        response.close()

    def test_usage_turn_id_is_bound_only_for_successful_llm_utils_endpoint(self):
        cases = (
            ("/api/agent/v3/llm_utils_chat", False, True),
            ("/api/ide/v1/chat", False, False),
            ("/api/agent/v3/create_agent_task", False, False),
            ("", True, False),
        )
        for endpoint, reject_first, has_turn_id in cases:
            with self.subTest(endpoint=endpoint, fallback=reject_first):
                requests = []

                class FakeResponse:
                    text = "unavailable"

                    def __init__(self, status_code):
                        self.status_code = status_code

                    def read(self):
                        return b"unavailable"

                    def close(self):
                        pass

                class FakeClient:
                    def __init__(self, **kwargs):
                        pass

                    def build_request(self, method, url, **kwargs):
                        return (url, kwargs["json"])

                    def send(self, request, stream=True):
                        requests.append(request)
                        status = 503 if reject_first and len(requests) == 1 else 200
                        return FakeResponse(status)

                    def close(self):
                        pass

                with (
                    patch.object(trae_client, "_WEB_SLOTS", {}),
                    patch("src.trae_client.auth.get_account_record", return_value={}),
                    patch(
                        "src.trae_client.auth.get_auth",
                        return_value=SimpleNamespace(host="https://example.test"),
                    ),
                    patch("src.trae_client.httpx.Client", side_effect=FakeClient),
                    patch("src.trae_client.build_headers", return_value={}),
                    patch(
                        "src.trae_client._resolve_request_model_name",
                        new=AsyncMock(return_value="auto"),
                    ),
                ):
                    response = asyncio.run(
                        trae_client.send_chat_request(
                            [{"role": "user", "content": "hello"}],
                            "auto",
                            True,
                            {
                                "_account_id": "acct",
                                "_auth_token": "bound-token-hidden",
                                "_ide_endpoint": endpoint,
                                "session_id": "request-session",
                            },
                        )
                    )
                    try:
                        if has_turn_id:
                            self.assertTrue(response.usage_turn_id)
                            self.assertNotEqual(response.usage_turn_id, "request-session")
                            self.assertEqual(
                                response.usage_turn_id, requests[-1][1]["session_id"]
                            )
                            self.assertEqual(
                                response.usage_turn_id, requests[-1][1]["request_id"]
                            )
                        else:
                            self.assertEqual(response.usage_turn_id, "")
                            self.assertEqual(requests[-1][1]["session_id"], "request-session")
                        self.assertEqual(response.auth_token, "bound-token-hidden")
                        self.assertNotIn("bound-token-hidden", repr(response))
                        self.assertEqual(len(requests), 2 if reject_first else 1)
                    finally:
                        response.close()

    def test_legacy_requests_use_distinct_bill_ids_without_mutating_tool_history(self):
        bodies = []

        class FakeResponse:
            status_code = 200

            def close(self):
                pass

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def build_request(self, method, url, **kwargs):
                bodies.append(kwargs["json"])
                return object()

            def send(self, request, stream=True):
                return FakeResponse()

            def close(self):
                pass

        tool_call = {
            "id": "call_marker_1",
            "type": "function",
            "function": {"name": "marker", "arguments": '{"value":"one"}'},
        }
        messages = [
            {"role": "user", "content": "Read the marker."},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_marker_1", "content": "one"},
        ]
        options = {
            "_account_id": "acct",
            "_auth_token": "bound-token",
            "_ide_endpoint": "/api/agent/v3/llm_utils_chat",
            "session_id": "logical-conversation",
            "_legacy_usage_turn_id": "caller-stale-turn",
            "tools": [{
                "type": "function",
                "function": {"name": "marker", "parameters": {"type": "object"}},
            }],
        }
        before_messages = copy.deepcopy(messages)
        before_options = copy.deepcopy(options)
        bill_ids = []
        with (
            patch.object(trae_client, "_WEB_SLOTS", {}),
            patch("src.trae_client.auth.get_account_record", return_value={}),
            patch(
                "src.trae_client.auth.get_auth",
                return_value=SimpleNamespace(host="https://example.test"),
            ),
            patch("src.trae_client.httpx.Client", side_effect=FakeClient),
            patch("src.trae_client.build_headers", return_value={}),
            patch(
                "src.trae_client._resolve_request_model_name",
                new=AsyncMock(return_value="auto"),
            ),
        ):
            for _ in range(2):
                response = asyncio.run(
                    trae_client.send_chat_request(messages, "auto", True, options)
                )
                bill_ids.append(response.usage_turn_id)
                response.close()

        self.assertEqual(len(set(bill_ids)), 2)
        self.assertNotIn("logical-conversation", bill_ids)
        self.assertNotIn("caller-stale-turn", bill_ids)
        self.assertEqual(messages, before_messages)
        self.assertEqual(options, before_options)
        self.assertEqual(bodies[0]["messages"], bodies[1]["messages"])
        for body, bill_id in zip(bodies, bill_ids):
            self.assertEqual(body["session_id"], bill_id)
            self.assertEqual(body["request_id"], bill_id)
            assistant = next(item for item in body["messages"] if item["role"] == "assistant")
            result = next(item for item in body["messages"] if item["role"] == "tool")
            self.assertEqual(len(assistant["tool_calls"]), 1)
            self.assertEqual(assistant["tool_calls"][0]["id"], tool_call["id"])
            self.assertEqual(
                assistant["tool_calls"][0]["function_call"], tool_call["function"]
            )
            self.assertEqual(result["tool_call_id"], "call_marker_1")
            self.assertEqual(result["content"], [{"type": "text", "text": "one"}])
            self.assertEqual(len(body["tools"]), 1)
            self.assertEqual(body["tools"][0]["function"]["name"], "marker")
            self.assertEqual(
                json.loads(body["tools"][0]["function"]["parameters"]),
                options["tools"][0]["function"]["parameters"],
            )

    def test_llm_body_preserves_explicit_session_without_internal_bill_override(self):
        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hello"}],
            "auto",
            True,
            options={"session_id": "caller-session"},
        )
        self.assertEqual(body["session_id"], "caller-session")
        self.assertEqual(body["request_id"], "caller-session")

    def test_uses_caller_workspace_and_terminal_context(self):
        _, request = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "hello"}],
                "auto",
                options={"client_context": CLIENT_CONTEXT},
            )
        )

        variables = json.loads(request["variables"])
        self.assertEqual(variables["workspace_path"], r"D:\work\demo")
        self.assertEqual(variables["system_type"], "Windows 11")
        terminal_resolver = next(
            item
            for item in request["context_resolvers"]
            if item["resolver_id"] == "terminal_context"
        )
        terminal_variables = json.loads(terminal_resolver["variables"])
        self.assertEqual(
            terminal_variables["terminal_context"],
            CLIENT_CONTEXT["terminal_context"],
        )

    def test_explicit_session_id_is_preserved(self):
        _, request = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "hello"}],
                "auto",
                options={
                    "client_context": CLIENT_CONTEXT,
                    "session_id": "client-session-1",
                },
            )
        )
        self.assertEqual(request["session_id"], "client-session-1")
        self.assertEqual(request["conversation_id"], "client-session-1")

    def test_ide_and_llm_requests_clamp_completion_tokens(self):
        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hello"}],
            "deepseek-v4-flash",
            True,
            384000,
        )
        _, ide_request = asyncio.run(
            trae_client.build_trae_ide_request(
                [{"role": "user", "content": "hello"}],
                "deepseek-v4-flash",
                384000,
            )
        )
        self.assertEqual(body["max_tokens"], 64000)
        self.assertEqual(ide_request["max_output_tokens"], 64000)

    def test_solo_llm_utils_body_uses_native_solo_function_and_model_lock(self):
        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hello"}],
            "glm-5.3",
            True,
            options={"session_id": "solo-session"},
        )

        self.assertEqual(body["function"], "solo_work_lite")
        self.assertEqual(body["config_name"], "glm-5.3")
        self.assertEqual(body["model"], "glm-5.3")
        self.assertEqual(body["session_id"], "solo-session")
        self.assertEqual(body["request_id"], "solo-session")
        self.assertTrue(body["stream"])

    def test_legacy_bodies_gate_reasoning_by_model_descriptor(self):
        unsupported = {
            "config_name": "text-only",
            "reasoning_effort_config": {"support_thinking": False},
        }
        unsupported_body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hello"}],
            "text-only",
            True,
            options={
                "reasoning_effort": "high",
                "_traecode_model_descriptor": unsupported,
            },
        )
        self.assertNotIn("reasoning_effort", unsupported_body)
        self.assertNotIn("reasoning_effort_level", unsupported_body)

        supported = {
            "config_name": "thinking-model",
            "reasoning_effort_level": "high",
            "reasoning_effort_config": {
                "support_thinking": True,
                "options": ["light", "high"],
            },
        }
        trace = {}
        supported_body = trae_client.build_work_agent_body(
            [{"role": "user", "content": "hello"}],
            "thinking-model",
            True,
            options={
                "reasoning_effort": "ultra",
                "_traecode_model_descriptor": supported,
                "_upstream_trace": trace,
            },
        )
        self.assertEqual(supported_body["reasoning_effort_level"], "high")
        self.assertNotIn("reasoning_effort", supported_body)
        self.assertEqual(trace["reasoning_effort_applied"], "high")

    def test_solo_body_serializes_tool_schema_and_normalizes_choice(self):
        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "call read_file"}],
            "glm-5.3",
            True,
            options={
                "session_id": "solo-tool-session",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a caller file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "read_file"},
                },
                "parallel_tool_calls": False,
            },
        )

        self.assertEqual(body["function"], "solo_work_lite")
        self.assertEqual(body["tool_choice"], "read_file")
        self.assertFalse(body["parallel_tool_calls"])
        schema = body["tools"][0]["function"]["parameters"]
        self.assertIsInstance(schema, str)
        self.assertEqual(json.loads(schema)["required"], ["path"])

    def test_implicit_sessions_do_not_collide_on_same_prompt(self):
        first = trae_client.generate_session_id_from_messages(
            [{"role": "user", "content": "same"}]
        )
        second = trae_client.generate_session_id_from_messages(
            [{"role": "user", "content": "same"}]
        )
        self.assertNotEqual(first, second)

    def test_web_rejects_but_ide_encodes_tool_policy_in_messages(self):
        options = {"tools": [], "tool_choice": "none"}

        with self.assertRaisesRegex(RuntimeError, "web remote"):
            asyncio.run(
                trae_client.create_web_session(
                    None,
                    "auto",
                    [{"role": "user", "content": "hello"}],
                    options,
                )
            )

        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hello"}],
            "auto",
            False,
            options=options,
        )
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertIn("Tool choice is none", body["messages"][0]["content"][0]["text"])

    def test_web_rejects_but_ide_preserves_tool_history(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "done",
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "web remote"):
            asyncio.run(trae_client.create_web_session(None, "auto", messages))
        body = trae_client.build_llm_chat_body(messages, "auto", False)
        assistant = next(
            message for message in body["messages"] if message["role"] == "assistant"
        )
        self.assertIsNone(assistant["content"])
        self.assertEqual(
            assistant["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function_call": {"name": "read_file", "arguments": "{}"},
                }
            ],
        )
        tool_result = next(
            message for message in body["messages"] if message["role"] == "tool"
        )
        self.assertEqual(tool_result["tool_call_id"], "call_1")
        self.assertEqual(tool_result["name"], "read_file")
        self.assertEqual(tool_result["content"], [{"type": "text", "text": "done"}])
        prompt = body["messages"][0]["content"][0]["text"]
        self.assertIn("same call id confirms success", prompt)
        self.assertNotIn("opencode_tool_call", prompt)
        self.assertNotIn("Client tool history", json.dumps(body))

    def test_model_gateway_does_not_reuse_oauth_host(self):
        with patch.dict("os.environ", {"TRAE_MODEL_API_HOST": ""}, clear=False):
            self.assertEqual(
                trae_client._model_gateway_base("https://api.trae.com.cn"),
                "https://trae-api-cn.mchost.guru",
            )
            self.assertEqual(
                trae_client._model_gateway_base("https://custom.example"),
                "https://custom.example",
            )
            self.assertEqual(
                trae_client._model_gateway_base(
                    "https://trae-api-cn.mchost.guru/api/remote/v1"
                ),
                "https://trae-api-cn.mchost.guru",
            )

    def test_remote_base_url_repairs_host_only_gateway(self):
        """A legacy bare gateway must not 404 the Remote transport (#9)."""

        cases = (
            "https://trae-api-cn.mchost.guru",
            "https://trae-api-cn.mchost.guru/",
            "https://core-normal.trae.cn",
        )
        for value in cases:
            with patch.dict("os.environ", {"TRAE_WEB_BASE_URL": value}, clear=False):
                self.assertEqual(
                    trae_client.remote_base_url(),
                    value.rstrip("/") + "/api/remote/v1",
                )

    def test_remote_base_url_preserves_path_and_custom_host(self):
        with patch.dict(
            "os.environ",
            {"TRAE_WEB_BASE_URL": "https://trae-api-cn.mchost.guru/api/remote/v1/"},
            clear=False,
        ):
            self.assertEqual(
                trae_client.remote_base_url(),
                "https://trae-api-cn.mchost.guru/api/remote/v1",
            )
        with patch.dict(
            "os.environ", {"TRAE_WEB_BASE_URL": "https://custom.example/v1"}, clear=False
        ):
            self.assertEqual(
                trae_client.remote_base_url(), "https://custom.example/v1"
            )
        for value in (
            "https://custom.example/proxy/trae.ai",
            "https://custom.example/v1?next=trae.ai",
        ):
            with patch.dict("os.environ", {"TRAE_WEB_BASE_URL": value}, clear=False):
                self.assertEqual(trae_client.remote_base_url(), value)

    def test_remote_base_url_defaults_when_unset(self):
        with patch.dict("os.environ", {"TRAE_WEB_BASE_URL": ""}, clear=False):
            self.assertEqual(
                trae_client.remote_base_url(), trae_client.REMOTE_BASE_DEFAULT
            )

    def test_work_agent_body_carries_native_tools(self):
        options = {
            "_auth_user_id": "uid-1",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        body = trae_client.build_work_agent_body(
            [{"role": "user", "content": "Read README.md"}],
            "glm-5.3",
            True,
            options,
        )
        self.assertEqual(body["agent_type"], "solo_work_lite")
        self.assertEqual(body["config_name"], "glm-5.3")
        self.assertEqual(body["model_name"], "glm-5.3__dev")
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        self.assertIsInstance(body["tools"][0]["function"]["parameters"], str)
        self.assertFalse(body["parallel_tool_calls"])

    def test_work_agent_host_defaults_to_work_gateway(self):
        self.assertEqual(
            trae_client.WORK_AGENT_HOST,
            "https://api5-normal.mchost.guru",
        )


class WebModelGroupPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_indexes_exact_config_name_when_name_is_display_label(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "list": [
                            {
                                "function": "solo_agent_remote",
                                "models": [
                                    {
                                        "name": "GLM-5.3 Flash",
                                        "config_name": "GLM-5.3-Flash",
                                        "model_name": "glm-5.3-flash",
                                    }
                                ],
                            }
                        ]
                    }
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, _url):
                return FakeResponse()

        with patch.object(trae_client.httpx, "AsyncClient", return_value=FakeClient()):
            configs = await trae_client._fetch_web_model_configs(
                token_override="token-catalog"
            )

        self.assertIn("GLM-5.3-Flash", configs)
        self.assertEqual(configs["GLM-5.3-Flash"]["config_name"], "GLM-5.3-Flash")
        self.assertEqual(configs["GLM-5.3-Flash"]["model_name"], "glm-5.3-flash")

    async def test_agent_remote_group_wins_over_same_name_work(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "list": [
                            {
                                "function": "solo_work_remote",
                                "models": [
                                    {
                                        "name": "glm-5.3",
                                        "multimodal": False,
                                        "max_mode": False,
                                        "prompt_max_tokens": 168000,
                                        "max_tokens": 32000,
                                        "context_window_size": {"default": 200000},
                                    }
                                ],
                            },
                            {
                                "function": "solo_agent_remote",
                                "models": [
                                    {
                                        "name": "glm-5.3",
                                        "multimodal": True,
                                        "max_mode": True,
                                        "prompt_max_tokens": 936000,
                                        "max_tokens": 64000,
                                        "context_window_size": {
                                            "default": 200000,
                                            "max": 1000000,
                                        },
                                        "reasoning_effort_config": {
                                            "support_thinking": True,
                                            "options": ["light", "high", "extra_high"],
                                        },
                                    }
                                ],
                            },
                        ]
                    }
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, url):
                return FakeResponse()

        with patch.object(trae_client.httpx, "AsyncClient", return_value=FakeClient()):
            configs = await trae_client._fetch_web_model_configs(token_override="token-x")

        cfg = configs["glm-5.3"]
        self.assertEqual(cfg["max_tokens"], 64000)
        self.assertEqual(cfg["context_window_size"]["max"], 1000000)
        scopes = cfg["_trae_capabilities_by_scope"]
        self.assertTrue(scopes["solo_agent_remote"]["multimodal"])
        self.assertTrue(scopes["solo_agent_remote"]["max_mode"])
        self.assertEqual(
            scopes["solo_agent_remote"]["reasoning_effort"],
            ["light", "high", "extra_high"],
        )
        self.assertFalse(scopes["solo_work_remote"]["multimodal"])
        self.assertFalse(scopes["solo_work_remote"]["max_mode"])
        self.assertNotIn("reasoning_effort", scopes["solo_work_remote"])

    async def test_fallback_keeps_work_group_when_agent_absent(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "list": [
                            {
                                "function": "solo_work_remote",
                                "models": [{"name": "glm-5.3", "max_tokens": 32000}],
                            }
                        ]
                    }
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, url):
                return FakeResponse()

        with patch.object(trae_client.httpx, "AsyncClient", return_value=FakeClient()):
            configs = await trae_client._fetch_web_model_configs(token_override="token-x")

        self.assertEqual(configs["glm-5.3"]["max_tokens"], 32000)

    async def test_raw_function_filter_does_not_borrow_sibling_descriptor(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": {
                        "list": [
                            {
                                "function": "solo_agent_remote",
                                "models": [
                                    {
                                        "name": "chat-model",
                                        "raw_chat_function": "chat_v3",
                                        "multimodal": True,
                                    },
                                    {
                                        "name": "agent-model",
                                        "raw_chat_function": "solo_agent",
                                        "multimodal": False,
                                    },
                                ],
                            }
                        ]
                    }
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, _url):
                return FakeResponse()

        with patch.object(trae_client.httpx, "AsyncClient", return_value=FakeClient()):
            configs = await trae_client._fetch_web_model_configs(
                token_override="token-x",
                raw_chat_function="chat_v3",
            )

        self.assertEqual(set(configs), {"chat-model"})
        self.assertEqual(
            set(configs["chat-model"]["_trae_capabilities_by_scope"]),
            {"chat_v3"},
        )


class SoloModelListTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_solo_catalog_parses_flash_and_filters_routing_profiles(self):
        calls = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "config_info_list": [
                        {
                            "config_name": "glm-5.3-flash",
                            "usage": "chat_completion",
                            "model_detail_list": [
                                {"model_name": "glm-5.3-flash__dev"},
                            ],
                            "display_config": {
                                "display_name": "GLM-5.3-Flash",
                                "max_mode": False,
                            },
                        },
                        {
                            "config_name": "custom_model_1M_text",
                            "usage": "custom_model",
                            "custom_models": ["provider//glm-5.3"],
                        },
                    ]
                }

        class FakeClient:
            def __init__(self, **kwargs):
                calls["timeout"] = kwargs.get("timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, url, headers=None, json=None):
                calls.update(url=url, headers=headers, body=json)
                return FakeResponse()

        with (
            patch.object(trae_client.httpx, "AsyncClient", FakeClient),
            patch.object(
                trae_client,
                "build_headers",
                return_value={"Authorization": "Cloud-IDE-JWT token-x"},
            ),
            patch.object(
                trae_client,
                "_model_gateway_base",
                return_value="https://trae-api-cn.mchost.guru",
            ),
        ):
            configs = await trae_client._fetch_solo_model_configs(
                token_override="token-x",
                raw_chat_function="chat_v3",
            )

        self.assertIn("glm-5.3-flash", configs)
        self.assertNotIn("custom_model_1M_text", configs)
        self.assertEqual(configs["glm-5.3-flash"]["display_name"], "GLM-5.3-Flash")
        self.assertEqual(
            configs["glm-5.3-flash"]["model_name"],
            "glm-5.3-flash__dev",
        )
        self.assertEqual(
            configs["glm-5.3-flash"]["_traecode_model_detail_list"][0]["model_name"],
            "glm-5.3-flash__dev",
        )
        self.assertEqual(
            calls["url"],
            "https://trae-api-cn.mchost.guru/api/ide/v1/get_detail_param",
        )
        self.assertEqual(calls["body"]["function"], "chat_v3")
        self.assertLessEqual(calls["timeout"], 20)

    async def test_native_catalog_resolves_dev_and_max_provider_ids(self):
        seen_functions = []

        async def fake_fetch(
            *,
            token_override="",
            provider_specific=None,
            user_id_override="",
            raw_chat_function="",
        ):
            seen_functions.append(raw_chat_function)
            return {
                "glm-5.2": {
                    "config_name": "glm-5.2",
                    "display_name": "GLM-5.2",
                    "config_source": 1,
                    "model_detail_list": [
                        {"model_name": "glm-5.2__max"},
                        {"model_name": "glm-5.2__dev"},
                    ],
                    "_traecode_model_detail_list": [
                        {"model_name": "glm-5.2__max"},
                        {"model_name": "glm-5.2__dev"},
                    ],
                }
            }

        with patch.object(trae_client, "_fetch_solo_model_configs", fake_fetch):
            dev = await trae_client.resolve_traecode_model_config(
                "glm-5.2",
                token_override="token-x",
                user_id_override="user-x",
                max_mode=False,
            )
            max_cfg = await trae_client.resolve_traecode_model_config(
                "glm-5.2",
                token_override="token-x",
                user_id_override="user-x",
                max_mode=True,
                raw_chat_function="chat_v3",
            )

        self.assertEqual(dev["config_name"], "glm-5.2")
        self.assertEqual(dev["raw_model_name"], "glm-5.2__dev")
        self.assertEqual(max_cfg["raw_model_name"], "glm-5.2__max")
        self.assertEqual(seen_functions, ["solo_work_lite", "chat_v3"])

    async def test_model_cache_is_partitioned_by_tier_and_raw_function(self):
        fetch = AsyncMock(
            side_effect=[
                {"same-model": {"config_name": "same-model", "multimodal": True}},
                {"same-model": {"config_name": "same-model", "multimodal": False}},
            ]
        )
        with (
            patch.object(trae_client, "_WEB_MODEL_CACHE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", new=fetch),
        ):
            agent = await trae_client._get_web_custom_model(
                "same-model",
                token_override="token",
                user_id_override="account",
                agent_type="solo_agent_remote",
                raw_chat_function="solo_agent_remote",
            )
            work = await trae_client._get_web_custom_model(
                "same-model",
                token_override="token",
                user_id_override="account",
                agent_type="solo_work_remote",
                raw_chat_function="solo_work_remote",
            )

        self.assertTrue(agent["multimodal"])
        self.assertFalse(work["multimodal"])
        self.assertEqual(fetch.await_count, 2)


if __name__ == "__main__":
    unittest.main()


class UserInfoTests(unittest.TestCase):
    def test_fetch_user_info_uses_cloudide_token_header(self):
        calls = {}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                calls.update(url=url, json=json, headers=headers)
                body = {"Result": {"UserID": "u-1", "ScreenName": "name"}}
                return SimpleNamespace(status_code=200, text="{}", json=lambda: body)

        with patch("src.trae_client.httpx.AsyncClient", FakeClient):
            result = asyncio.run(trae_client.fetch_user_info("jwt-token"))

        self.assertEqual(result["ScreenName"], "name")
        self.assertTrue(calls["url"].endswith("/cloudide/api/v3/trae/GetUserInfo"))
        self.assertEqual(calls["headers"]["x-cloudide-token"], "jwt-token")
        self.assertNotIn("Authorization", calls["headers"])
        self.assertEqual(calls["json"], {"ReqSource": "IDE"})

    def test_fetch_user_info_rejects_empty_result(self):
        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return SimpleNamespace(status_code=200, text="{}", json=lambda: {"Result": {}})

        with patch("src.trae_client.httpx.AsyncClient", FakeClient):
            with self.assertRaises(RuntimeError):
                asyncio.run(trae_client.fetch_user_info("jwt-token"))
