import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.responses import StreamingResponse

from src import trae_remote_client
from src import main as main_module
from src.sse import EmptyUpstreamResponse


class _Response:
    def __init__(self, payload=None, status_code=200, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        return self._payload

    async def aread(self):
        return self.text.encode()


class _StreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        # Deliberately split frames across chunks like a real proxy/TLS stream.
        for chunk in (
            b"event: plan_item\ndata: {\"id\":\"p1\",\"thought\":\"hel",
            b"lo\"}\n\n",
            b"event: token_usage\ndata: {\"prompt_tokens\":1}\n\n",
            b"event: done\ndata: {\"finish_reason\":\"stop\"}\n\n",
        ):
            yield chunk


class _SilentStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        await asyncio.sleep(60)
        if False:
            yield b""


class _EmptyStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        if False:
            yield b""


class _ReadTimeoutStreamResponse:
    status_code = 200

    def __init__(self, chunks=()):
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk
        raise httpx.ReadTimeout("silent upstream")


class _TrailingSseStreamResponse:
    status_code = 200

    def __init__(self, payload: bytes):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        yield self.payload


class _Client:
    def __init__(self):
        self.posts = []
        self.streams = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _Response({"code": 0, "data": {"chat_session_id": "s1", "message_id": "m1"}})

    def stream(self, method, url, **kwargs):
        self.streams.append((method, url, kwargs))
        return _StreamResponse()


class RemoteClientTests(unittest.TestCase):
    def test_model_session_id_uses_prompt_cache_key_affinity(self):
        base = {"_account_id": "account-1", "_session_variant": "default"}
        first = trae_remote_client.model_session_id(
            "glm-5.2", {**base, "_prompt_cache_key_affinity": "stable-prefix-a"}
        )
        second = trae_remote_client.model_session_id(
            "glm-5.2", {**base, "_prompt_cache_key_affinity": "stable-prefix-a"}
        )
        other = trae_remote_client.model_session_id(
            "glm-5.2", {**base, "_prompt_cache_key_affinity": "stable-prefix-b"}
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_work_common_params_uses_bound_paid_account_identity(self):
        params = json.loads(
            trae_remote_client.common_params(
                {
                    "userIdentity": "Free",
                    "userPayIdentity": 4,
                    "userPayIdentityText": "Pro+",
                },
                "work",
                options={"_account_id": "paid-account"},
            )
        )

        self.assertEqual(params["identity"], "4")
        self.assertEqual(params["identity_str"], "Pro+")
        self.assertEqual(params["user_identity"], "Pro+")

    def test_work_common_params_reads_only_the_bound_account_record(self):
        records = {
            "paid-account": {
                "pay_identity": 1,
                "pay_identity_str": "Pro",
                "provider_specific": {},
            },
            "other-account": {
                "pay_identity": 4,
                "pay_identity_str": "Pro+",
                "provider_specific": {},
            },
        }
        with (
            patch.object(
                trae_remote_client.auth,
                "resolve_model_account_id",
                side_effect=lambda account_id: account_id,
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                side_effect=lambda account_id: records.get(account_id, {}),
            ),
        ):
            params = json.loads(
                trae_remote_client.common_params(
                    {},
                    "work",
                    options={"_account_id": "paid-account"},
                )
            )

        self.assertEqual(params["identity"], "1")
        self.assertEqual(params["identity_str"], "Pro")
        self.assertEqual(params["user_identity"], "Pro")

    def test_work_common_params_does_not_trust_spoofed_request_identity(self):
        with (
            patch.object(
                trae_remote_client.auth,
                "resolve_model_account_id",
                side_effect=lambda account_id: account_id,
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={
                    "pay_identity": 1,
                    "pay_identity_str": "Pro",
                    "provider_specific": {},
                },
            ),
        ):
            params = json.loads(
                trae_remote_client.common_params(
                    {
                        "userPayIdentity": 99,
                        "userPayIdentityText": "Enterprise",
                    },
                    "work",
                    options={"_account_id": "paid-account"},
                )
            )

        self.assertEqual(params["identity"], "1")
        self.assertEqual(params["identity_str"], "Pro")

    def test_create_session_matches_remote_executor_shape(self):
        async def run():
            client = _Client()
            result = await trae_remote_client.create_session(
                client,
                "jwt-token",
                "auto",
                [{"role": "user", "content": "hello"}],
            )
            return result, client.posts[0]

        (session, (url, request)) = asyncio.run(run())
        self.assertEqual(session, ("s1", "m1"))
        self.assertTrue(url.endswith("/chat_sessions"))
        self.assertEqual(request["headers"]["Authorization"], "Cloud-IDE-JWT jwt-token")
        body = request["json"]
        self.assertEqual(body["initial_message"]["agent_type"], "solo_agent_remote")
        self.assertEqual(body["initial_message"]["agent_id"], "solo_agent_remote")
        self.assertEqual(body["initial_message"]["model_selection_strategy"], "auto")
        self.assertIn("hello", body["initial_message"]["query"])

    def test_create_session_normalizes_nested_non_json_values(self):
        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
                "adapter_metadata": {
                    "flags": {"zeta", "alpha"},
                    "limits": frozenset({3, 1}),
                    "raw": b"legacy",
                },
            }
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(return_value=custom),
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_trae_common_params": {
                            "adapter_flags": {"zeta", "alpha"},
                        }
                    },
                )
            return client.posts[0][1]["json"]

        body = asyncio.run(run())
        # The exact object sent to the transport must be JSON encodable and
        # must use stable ordering for set values.
        self.assertEqual(
            body["initial_message"]["custom_model"]["adapter_metadata"]["flags"],
            ["alpha", "zeta"],
        )
        self.assertEqual(
            body["initial_message"]["custom_model"]["adapter_metadata"]["limits"],
            [1, 3],
        )
        self.assertEqual(
            body["initial_message"]["custom_model"]["adapter_metadata"]["raw"],
            "b'legacy'",
        )
        self.assertEqual(
            json.loads(body["initial_message"]["common_params"])["adapter_flags"],
            ["alpha", "zeta"],
        )
        self.assertEqual(json.loads(json.dumps(body)), body)

    def test_work_mode_sends_work_executor_type(self):
        async def run():
            client = _Client()
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(
                    return_value={
                        "name": "glm-5.3",
                        "config_name": "glm-5.3",
                        "model_name": "glm-5.3",
                        "config_source": 1,
                    }
                ),
            ):
                result = await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={"_remote_agent_type": "solo_work_remote"},
                )
            return result, client.posts[0]

        (_session, (_url, request)) = asyncio.run(run())
        initial = request["json"]["initial_message"]
        self.assertEqual(initial["agent_type"], "solo_work_remote")
        self.assertEqual(initial["agent_id"], "solo_work_remote")
        self.assertEqual(initial["model_selection_strategy"], "manual")

    def test_work_model_lookup_uses_work_tier(self):
        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
            }
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(return_value=custom),
            ) as resolve:
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={"_remote_agent_type": "solo_work_remote"},
                )
            return resolve

        resolve = asyncio.run(run())
        self.assertEqual(resolve.await_args.kwargs["agent_type"], "solo_work_remote")

    def test_tier_aware_model_resolver_receives_default_agent_type(self):
        async def run():
            client = _Client()
            captured = {}

            async def resolve(
                model_name,
                *,
                token_override="",
                user_id_override="",
                provider_specific=None,
                agent_type="",
            ):
                captured.update(
                    model_name=model_name,
                    token=token_override,
                    user_id=user_id_override,
                    provider_specific=provider_specific,
                    agent_type=agent_type,
                )
                return {
                    "name": model_name,
                    "config_name": model_name,
                    "model_name": model_name,
                    "config_source": 1,
                }

            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=resolve,
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={"_account_id": "account-1", "provider_specific": {}},
                )
            return captured

        captured = asyncio.run(run())
        self.assertEqual(captured["agent_type"], "solo_agent_remote")
        self.assertEqual(captured["model_name"], "glm-5.3")

    def test_legacy_model_resolver_without_agent_type_remains_supported(self):
        async def run():
            client = _Client()
            captured = {}

            async def resolve(
                model_name,
                *,
                token_override="",
                user_id_override="",
                provider_specific=None,
            ):
                captured.update(
                    model_name=model_name,
                    token=token_override,
                    user_id=user_id_override,
                    provider_specific=provider_specific,
                )
                return {
                    "name": model_name,
                    "config_name": model_name,
                    "model_name": model_name,
                    "config_source": 1,
                }

            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=resolve,
            ):
                result = await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={"_account_id": "account-1", "provider_specific": {}},
                )
            return result, captured

        result, captured = asyncio.run(run())
        self.assertEqual(result, ("s1", "m1"))
        self.assertEqual(captured["model_name"], "glm-5.3")

    def test_mismatched_model_config_is_rejected_before_session_create(self):
        async def run():
            client = _Client()
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(
                    return_value={
                        "name": "kimi-k2.6",
                        "config_name": "kimi-k2.6",
                        "model_name": "kimi-k2.6",
                    }
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "model binding mismatch"):
                    await trae_remote_client.create_session(
                        client,
                        "jwt-token",
                        "glm-5.3",
                        [{"role": "user", "content": "hello"}],
                    )
            return client

        client = asyncio.run(run())
        self.assertEqual(client.posts, [])

    def test_manual_model_sends_account_bound_custom_model(self):
        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
            }
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(return_value=custom),
            ) as resolve:
                result = await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "1329782626198720",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                    },
                )
            return result, client.posts[0], resolve

        (session, (_url, request), resolve) = asyncio.run(run())
        self.assertEqual(session, ("s1", "m1"))
        resolve.assert_awaited_once_with(
            "glm-5.3",
            token_override="jwt-token",
            user_id_override="1329782626198720",
            provider_specific={},
        )
        initial = request["json"]["initial_message"]
        self.assertEqual(initial["model_selection_strategy"], "manual")
        self.assertEqual(initial["model_name"], "glm-5.3")
        self.assertEqual(initial["model_config_source"], 1)
        self.assertTrue(initial["model_is_preset"])
        self.assertEqual(initial["model_provider"], "")
        self.assertEqual(initial["custom_model"], {
            "name": "glm-5.3",
            "config_name": "glm-5.3",
            "model_name": "glm-5.3",
            "config_source": 1,
        })
        common = json.loads(initial["common_params"])
        self.assertEqual(
            common["biz_session_id"],
            trae_remote_client.model_session_id(
                "glm-5.3", {"_account_id": "1329782626198720"}
            ),
        )
        self.assertEqual(
            json.loads(initial["query"]),
            [{"type": "text", "data": {"content": "hello"}}],
        )


    def test_manual_model_requests_max_mode_fields_when_enabled(self):
        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
                "is_preset": True,
                "provider": "",
                "max_mode": True,
                "context_window_size": {"default": 200000, "max": [1000000]},
                "prompt_max_tokens": 936000,
                "max_tokens": 64000,
            }
            with (
                patch.dict(
                    trae_remote_client.os.environ,
                    {
                        "TRAE_REMOTE_MAX_MODE": "1",
                        "TRAE_REMOTE_MAX_MODE_TYPE": "2",
                    },
                ),
                patch.object(
                    trae_remote_client.trae_client,
                    "resolve_model_config",
                    new=AsyncMock(return_value=custom),
                ),
            ):
                result = await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "1329782626198720",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                    },
                )
            return result, client.posts[0]

        (_session, (_url, request)) = asyncio.run(run())
        initial = request["json"]["initial_message"]
        self.assertEqual(initial["model_selection_strategy"], "max")
        self.assertEqual(initial["model_auto_selection"]["strategy"], "max")
        self.assertEqual(initial["mode_type"], 2)
        self.assertEqual(initial["context_window_size"], 1000000)
        self.assertEqual(initial["prompt_max_tokens"], 936000)
        self.assertEqual(initial["max_tokens"], 64000)
        common = json.loads(initial["common_params"])
        self.assertEqual(
            common["biz_session_id"],
            trae_remote_client.model_session_id(
                "glm-5.3",
                {
                    "_account_id": "1329782626198720",
                    "_session_variant": "max",
                },
            ),
        )

    def test_max_mode_does_not_enrich_incomplete_account_config(self):
        async def run():
            client = _Client()
            slim = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
                "is_preset": True,
                "provider": "",
                "max_mode": True,
                "context_window_tokens": {"dev": 200000, "max": 1000000},
            }
            with (
                patch.dict(
                    trae_remote_client.os.environ,
                    {"TRAE_REMOTE_MAX_MODE": "1"},
                ),
                patch.object(
                    trae_remote_client.trae_client,
                    "resolve_model_config",
                    new=AsyncMock(return_value=slim),
                ),
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "1329782626198720",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                    },
                )
            return client.posts[0]

        (_url, request) = asyncio.run(run())
        initial = request["json"]["initial_message"]
        cm = initial["custom_model"]
        self.assertEqual(initial["model_selection_strategy"], "manual")
        self.assertNotIn("context_window_size", initial)
        self.assertNotIn("prompt_max_tokens", initial)
        self.assertNotIn("max_tokens", initial)
        self.assertNotIn("context_window_size", cm)
        self.assertEqual(cm["context_window_tokens"], {"dev": 200000, "max": 1000000})
        self.assertNotIn("prompt_max_tokens", cm)
        self.assertNotIn("max_tokens", cm)
        self.assertNotIn("max_turn", cm)
        self.assertNotIn("features", cm)

    def test_max_mode_stays_off_when_account_config_lacks_max_mode(self):
        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
                "max_mode": False,
            }
            with (
                patch.dict(
                    trae_remote_client.os.environ,
                    {"TRAE_REMOTE_MAX_MODE": "1"},
                ),
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
                    options={
                        "_account_id": "1329782626198720",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                    },
                )
            return client.posts[0]

        (_url, request) = asyncio.run(run())
        initial = request["json"]["initial_message"]
        self.assertEqual(initial["model_selection_strategy"], "manual")
        self.assertNotIn("model_auto_selection", initial)
        self.assertNotIn("mode_type", initial)

    def test_work_fallback_clears_max_wire_fields_and_trace(self):
        """The final Work attempt must not inherit Agent Max accounting state."""

        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
                "is_preset": True,
                "provider": "",
                "max_mode": True,
                "context_window_size": {"default": 200000, "max": [1000000]},
                "prompt_max_tokens": 936000,
                "max_tokens": 64000,
            }
            trace = {}
            with (
                patch.dict(
                    trae_remote_client.os.environ,
                    {"TRAE_REMOTE_MAX_MODE": "1"},
                ),
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
                    options={
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                        "_upstream_trace": trace,
                    },
                )
                agent_initial = client.posts[-1][1]["json"]["initial_message"]
                self.assertEqual(agent_initial["agent_type"], "solo_agent_remote")
                self.assertEqual(agent_initial["context_window_size"], 1000000)
                self.assertTrue(trace["max_mode_applied"])
                self.assertEqual(trace["max_context_tokens"], 1000000)

                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                        "_remote_agent_type": "solo_work_remote",
                        "_session_variant": "work-fallback",
                        "_upstream_trace": trace,
                    },
                )
            return client.posts[-1][1]["json"]["initial_message"], trace

        work_initial, trace = asyncio.run(run())
        self.assertEqual(work_initial["agent_type"], "solo_work_remote")
        self.assertEqual(work_initial["agent_id"], "solo_work_remote")
        self.assertEqual(work_initial["model_selection_strategy"], "manual")
        self.assertNotIn("model_auto_selection", work_initial)
        self.assertNotIn("mode_type", work_initial)
        self.assertNotIn("context_window_size", work_initial)
        self.assertNotIn("prompt_max_tokens", work_initial)
        self.assertNotIn("max_tokens", work_initial)
        self.assertFalse(trace["max_mode_applied"])
        self.assertNotIn("max_context_tokens", trace)

    def test_remote_bound_empty_provider_metadata_does_not_use_global_state(self):
        with patch.object(
            trae_remote_client.auth,
            "get_psd",
            return_value={"appLanguage": "wrong", "userRegion": "wrong"},
        ):
            headers = trae_remote_client.build_headers(
                "jwt-token", options={"provider_specific": {}}, stream=False
            )

        self.assertEqual(headers["X-Preferenced-Language"], "zh-CN")
        self.assertEqual(headers["x-user-region"], "CN")

    def test_work_billing_overlay_is_opt_in_and_injects_wire_fields(self):
        async def run():
            client = _Client()
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(
                    return_value={
                        "name": "glm-5.3",
                        "config_name": "glm-5.3",
                        "model_name": "glm-5.3",
                        "config_source": 1,
                    }
                ),
            ):
                result = await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "1523296676357308",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                        "_remote_agent_type": "solo_work_remote",
                        "_trae_mode": "work",
                        "_work_billing_overlay": True,
                    },
                )
            return result, client.posts[0]

        (session, (url, request)) = asyncio.run(run())
        self.assertEqual(session, ("s1", "m1"))
        self.assertTrue(url.endswith("/chat_sessions"))
        headers = request["headers"]
        self.assertEqual(headers["X-Trae-Client-Type"], "lite")
        self.assertEqual(headers["request-traffic-type"], "prod")
        self.assertEqual(headers["x-trae-work"], "1")
        self.assertEqual(headers["x-bridge-transport"], "aha")
        body = request["json"]
        self.assertEqual(body["origin"], "lite")
        self.assertEqual(body["product_code"], "SOLO_Lite")
        self.assertEqual(body["billing_source"], "work")
        initial = body["initial_message"]
        self.assertEqual(initial["entitlement_id"], "335006824194")
        self.assertEqual(initial["available_endpoint"], 1)
        self.assertEqual(initial["billing_source"], "work")
        self.assertEqual(initial["product_code"], "SOLO_Lite")
        self.assertEqual(
            initial["model_auto_selection"],
            {
                "strategy": "manual",
                "fallback_to_advance_model": None,
                "entitlement_id": "335006824194",
            },
        )
        common = json.loads(initial["common_params"])
        self.assertEqual(common["entitlement_id"], "335006824194")
        self.assertEqual(common["available_endpoint"], 1)
        self.assertEqual(common["billing_source"], "work")
        self.assertEqual(common["product_code"], "SOLO_Lite")

    def test_work_billing_overlay_disabled_by_default(self):
        async def run():
            client = _Client()
            with (
                patch.dict(
                    trae_remote_client.os.environ,
                    {"TRAE_REMOTE_WORK_BILLING": "0"},
                ),
                patch.object(
                    trae_remote_client.trae_client,
                    "resolve_model_config",
                    new=AsyncMock(
                        return_value={
                            "name": "glm-5.3",
                            "config_name": "glm-5.3",
                            "model_name": "glm-5.3",
                            "config_source": 1,
                        }
                    ),
                ),
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "1329782626198720",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                        "_remote_agent_type": "solo_work_remote",
                        "_trae_mode": "work",
                    },
                )
            return client.posts[0]

        (_url, request) = asyncio.run(run())
        headers = request["headers"]
        self.assertEqual(headers["X-Trae-Client-Type"], "web")
        self.assertNotIn("x-trae-work", headers)
        body = request["json"]
        self.assertEqual(body["origin"], "web")
        initial = body["initial_message"]
        self.assertNotIn("entitlement_id", initial)
        self.assertNotIn("available_endpoint", initial)
        self.assertNotIn("billing_source", initial)
        self.assertNotIn("model_auto_selection", initial)

    def test_work_billing_env_header_enables_lite_without_body_overlay(self):
        async def run():
            client = _Client()
            with (
                patch.dict(
                    trae_remote_client.os.environ,
                    {"TRAE_REMOTE_WORK_BILLING": "1"},
                ),
                patch.object(
                    trae_remote_client.trae_client,
                    "resolve_model_config",
                    new=AsyncMock(
                        return_value={
                            "name": "glm-5.3",
                            "config_name": "glm-5.3",
                            "model_name": "glm-5.3",
                            "config_source": 1,
                        }
                    ),
                ),
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "1523296676357308",
                        "_auth_token": "jwt-token",
                        "provider_specific": {},
                        "_remote_agent_type": "solo_work_remote",
                        "_trae_mode": "work",
                    },
                )
            return client.posts[0]

        (_url, request) = asyncio.run(run())
        self.assertEqual(request["headers"]["X-Trae-Client-Type"], "lite")
        self.assertNotIn("x-trae-work", request["headers"])
        body = request["json"]
        self.assertEqual(body["origin"], "web")
        initial = body["initial_message"]
        self.assertNotIn("entitlement_id", initial)
        self.assertNotIn("available_endpoint", initial)
        self.assertNotIn("billing_source", initial)
        self.assertNotIn("model_auto_selection", initial)

    def test_stream_events_handles_split_sse_frames(self):
        async def run():
            client = _Client()
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        events = asyncio.run(run())
        self.assertEqual(events[0], ("plan_item", {"id": "p1", "thought": "hello"}))
        self.assertEqual(events[1][0], "token_usage")
        self.assertEqual(events[2], ("done", {"finish_reason": "stop"}))

    def test_stream_events_preserves_done_event_without_trailing_blank_line(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _TrailingSseStreamResponse(
                b'event: plan_item\ndata: {"id":"p1","thought":"complete"}\n\n'
                b'event: done\ndata: {"finish_reason":"stop"}'
            )
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        events = asyncio.run(run())
        self.assertEqual(events[-1], ("done", {"finish_reason": "stop"}))

    def test_stream_events_recognizes_done_sentinel_without_trailing_blank_line(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _TrailingSseStreamResponse(
                b'event: plan_item\ndata: {"id":"p1","thought":"complete"}\n\n'
                b"data: [DONE]"
            )
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        events = asyncio.run(run())
        self.assertEqual(events[-1], ("done", {}))

    def test_stream_events_times_out_before_first_event(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _SilentStreamResponse()
            with patch.dict(
                trae_remote_client.os.environ,
                {"TRAE_REMOTE_FIRST_EVENT_TIMEOUT_SECONDS": "0.01"},
            ):
                return [
                    item
                    async for item in trae_remote_client.stream_events(
                        client, "jwt-token", "s1", "m1"
                    )
                ]

        with self.assertRaises(trae_remote_client.RemoteFirstEventTimeout) as raised:
            asyncio.run(run())
        self.assertIsInstance(raised.exception, EmptyUpstreamResponse)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.observed_model_event)

    def test_stream_events_treats_eof_before_first_event_as_retryable(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _EmptyStreamResponse()
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        with self.assertRaises(trae_remote_client.RemoteFirstEventTimeout) as raised:
            asyncio.run(run())
        self.assertTrue(raised.exception.retryable)

    def test_stream_events_treats_read_timeout_before_first_event_as_retryable(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _ReadTimeoutStreamResponse()
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        with self.assertRaises(trae_remote_client.RemoteFirstEventTimeout) as raised:
            asyncio.run(run())
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.observed_model_event)

    def test_stream_events_eof_after_model_event_without_done_is_incomplete(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _StreamResponseWithEvents([
                b'event: plan_item\ndata: {"id":"p1","thought":"partial"}\n\n',
            ])
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        with self.assertRaises(trae_remote_client.RemoteStreamIncomplete):
            asyncio.run(run())

    def test_stream_events_does_not_retry_read_timeout_after_first_event(self):
        async def run():
            client = _Client()
            client.stream = lambda *_args, **_kwargs: _ReadTimeoutStreamResponse(
                (b'event: plan_item\ndata: {"thought":"started"}\n\n',)
            )
            return [
                item
                async for item in trae_remote_client.stream_events(
                    client, "jwt-token", "s1", "m1"
                )
            ]

        with self.assertRaises(trae_remote_client.RemoteStreamReadTimeout) as raised:
            asyncio.run(run())
        self.assertFalse(raised.exception.retryable)
        self.assertTrue(raised.exception.observed_model_event)

    def test_stream_events_treats_early_error_as_retryable_setup_failure(self):
        async def upstream():
            yield "heartbeat", {}
            yield "error", {"code": 2001, "message": "app config record not found"}

        async def run():
            with patch.object(
                trae_remote_client,
                "_stream_events_unbounded",
                return_value=upstream(),
            ):
                return [
                    item
                    async for item in trae_remote_client.stream_events(
                        _Client(), "jwt-token", "s1", "m1"
                    )
                ]

        with self.assertRaises(trae_remote_client.RemoteFirstEventError) as raised:
            asyncio.run(run())
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.observed_model_event)
        self.assertIn("record not found", str(raised.exception))

    def test_heartbeat_does_not_consume_first_model_event_timeout(self):
        async def upstream():
            yield "heartbeat", {}
            await asyncio.sleep(60)

        async def run():
            with (
                patch.object(
                    trae_remote_client,
                    "_stream_events_unbounded",
                    return_value=upstream(),
                ),
                patch.dict(
                    trae_remote_client.os.environ,
                    {"TRAE_REMOTE_FIRST_EVENT_TIMEOUT_SECONDS": "0.01"},
                ),
            ):
                return [
                    item
                    async for item in trae_remote_client.stream_events(
                        _Client(), "jwt-token", "s1", "m1"
                    )
                ]

        with self.assertRaises(trae_remote_client.RemoteFirstEventTimeout):
            asyncio.run(run())

    def test_remote_stream_without_done_does_not_emit_public_done(self):
        async def run():
            async def events(*_args, **_kwargs):
                yield "plan_item", {"id": "p1", "thought": "partial"}
                raise trae_remote_client.RemoteStreamIncomplete(
                    "Trae remote event stream ended without a done event"
                )

            async def fake_create(client, token, model, msgs, *, options=None):
                return ("session-1", "message-1")

            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(
                    main_module.trae_client,
                    "acquire_web_slot",
                    new=AsyncMock(),
                ),
                patch.object(
                    main_module.trae_client,
                    "release_web_slot",
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "create_session",
                    new=fake_create,
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "stream_events",
                    side_effect=lambda *_args, **_kwargs: events(),
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "stop_session",
                    new=AsyncMock(),
                ),
            ):
                response = await main_module.run_remote_session(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    True,
                    {
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                    },
                )
                chunks = [chunk async for chunk in response.body_iterator]
            return chunks

        with self.assertRaises(trae_remote_client.RemoteStreamIncomplete):
            asyncio.run(run())

    def test_stream_work_first_event_timeout_rotates_account(self):
        async def run():
            create_calls = []
            release_calls = []
            translate_calls = 0

            async def fake_create(client, token, model, msgs, *, options=None):
                create_calls.append((token, dict(options or {})))
                index = len(create_calls)
                return (f"session-{index}", f"message-{index}")

            async def fake_events(*_args, **_kwargs):
                if False:
                    yield None

            async def fake_translate(*_args, **_kwargs):
                nonlocal translate_calls
                translate_calls += 1
                if translate_calls == 1:
                    raise trae_remote_client.RemoteFirstEventTimeout("silent")
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield "data: [DONE]\n\n"

            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "token-a", "provider_specific": {}},
                ),
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module,
                    "_next_retry_account_snapshot",
                    return_value=(
                        "account-b",
                        {"token": "token-b", "provider_specific": {"webId": "b"}},
                    ),
                ),
                patch.object(
                    main_module.trae_client,
                    "acquire_web_slot",
                    new=AsyncMock(),
                ) as acquire,
                patch.object(
                    main_module.trae_client,
                    "release_web_slot",
                    side_effect=release_calls.append,
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "create_session",
                    new=fake_create,
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "stream_events",
                    side_effect=lambda *_args, **_kwargs: fake_events(),
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "stop_session",
                    new=AsyncMock(),
                ),
                patch.object(main_module, "translate_web_events", new=fake_translate),
                patch.object(main_module, "_track_usage_from_chunk"),
            ):
                response = await main_module.run_remote_session(
                    [{"role": "user", "content": "download the file"}],
                    "glm-5.3",
                    True,
                    {
                        "_account_id": "account-a",
                        "_auth_token": "token-a",
                        "_remote_agent_type": "solo_work_remote",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "download_file",
                                    "parameters": {"type": "object"},
                                },
                            }
                        ],
                    },
                )
                chunks = [chunk async for chunk in response.body_iterator]
            return chunks, create_calls, acquire, release_calls

        chunks, create_calls, acquire, release_calls = asyncio.run(run())
        self.assertIn("data: [DONE]", "".join(chunks))
        self.assertEqual([item[0] for item in create_calls], ["token-a", "token-b"])
        self.assertEqual(create_calls[1][1]["_account_id"], "account-b")
        self.assertEqual(create_calls[1][1]["provider_specific"], {"webId": "b"})
        self.assertEqual(
            [call.args[0] for call in acquire.await_args_list],
            ["account-a", "account-b"],
        )
        self.assertEqual(release_calls, ["account-a", "account-b"])

    def test_stream_retry_rotates_after_same_account_work_fallback_is_empty(self):
        """Body-level stream failures must reach every enabled account once."""

        async def run():
            calls = []
            rotations = iter(
                [
                    ("account-b", {"token": "token-b"}),
                    ("account-c", {"token": "token-c"}),
                ]
            )

            async def fake_remote(_messages, _model, _stream, options):
                account = str((options or {}).get("_account_id") or "")
                calls.append((account, dict(options or {})))

                async def body():
                    if account != "account-c":
                        raise trae_remote_client.RemoteFirstEventTimeout(
                            f"silent {account}"
                        )
                    yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    body(),
                    media_type="text/event-stream",
                    headers=main_module._sse_headers(),
                )

            def next_account(_attempted, _limit):
                try:
                    return next(rotations)
                except StopIteration:
                    return None

            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "account-a", "is_valid": True},
                        {"id": "account-b", "is_valid": True},
                        {"id": "account-c", "is_valid": True},
                    ],
                ),
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "token-a"},
                ),
                patch.object(
                    main_module,
                    "_next_retry_account_snapshot",
                    side_effect=next_account,
                ),
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                response = await main_module._run_remote_with_retry(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    True,
                    {"_account_id": "account-a", "_auth_token": "token-a"},
                )
                chunks = [chunk async for chunk in response.body_iterator]
            return calls, chunks

        calls, chunks = asyncio.run(run())
        self.assertEqual(
            [account for account, _options in calls],
            ["account-a", "account-b", "account-c"],
        )
        self.assertTrue(
            all(
                options.get("_disable_remote_stream_account_retry") is True
                for _account, options in calls
            )
        )
        self.assertIn("data: [DONE]", "".join(chunks))

    def test_stream_retry_closes_eager_session_before_first_body_iteration(self):
        """A client disconnect before iteration must release the first slot."""

        async def run():
            closed = []

            async def fake_remote(_messages, _model, _stream, options):
                class Body:
                    def __aiter__(self):
                        return self

                    async def __anext__(self):
                        await asyncio.sleep(60)
                        raise StopAsyncIteration

                    async def aclose(self):
                        closed.append(
                            str((options or {}).get("_account_id") or "")
                        )

                return StreamingResponse(
                    Body(),
                    media_type="text/event-stream",
                    headers=main_module._sse_headers(),
                )

            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[{"id": "account-a", "is_valid": True}],
                ),
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                response = await main_module._run_remote_with_retry(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    True,
                    {"_account_id": "account-a", "_auth_token": "token-a"},
                )
                await response.body_iterator.aclose()
            return closed

        self.assertEqual(asyncio.run(run()), ["account-a"])

    def test_remote_session_compacts_oversized_history_before_create(self):
        messages = []
        for index in range(300):
            messages.append(
                {
                    "role": "user" if index % 2 else "assistant",
                    "content": "long history message %d " % index + "x" * 300,
                }
            )
        messages.append({"role": "user", "content": "continue"})

        async def empty_events():
            if False:
                yield None

        captured = {}

        async def fake_create(client, token, model, msgs, *, options=None):
            captured["messages"] = msgs
            return ("session-1", "message-1")

        async def run():
            with (
                patch.dict(
                    main_module.os.environ,
                    {
                        "TRAE_REMOTE_MAX_MESSAGES": "20",
                        "TRAE_REMOTE_MAX_HISTORY_CHARS": "4000",
                    },
                ),
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=fake_create),
                patch.object(
                    main_module.trae_remote_client,
                    "stream_events",
                    return_value=empty_events(),
                ),
                patch.object(main_module.trae_remote_client, "stop_session", new=AsyncMock()),
                patch.object(
                    main_module,
                    "collect_nonstream_web",
                    new=AsyncMock(return_value={"usage": {}}),
                ),
                patch.object(main_module, "_track_usage_from_result", return_value=None),
            ):
                await main_module.run_remote_session(
                    messages,
                    "glm-5.3",
                    False,
                    {
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                    },
                )

        asyncio.run(run())
        compacted = captured["messages"]
        self.assertLessEqual(len(compacted), 20)
        self.assertLessEqual(
            sum(len(str(item.get("content") or "")) for item in compacted),
            4500,
        )

    def test_bounded_remote_query_trims_oldest_non_system_messages(self):
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "old turn " + "x" * 40_000},
            {
                "role": "assistant",
                "content": "old answer",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "file_edit",
                            "arguments": '{"path":"a.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "file_edit", "content": "ok"},
            {"role": "user", "content": "second turn " + "y" * 40_000},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "continue"},
        ]
        bounded, removed = main_module._bounded_remote_query(
            messages, {"trae_remote_query_max_chars": 12_000}
        )
        self.assertGreater(removed, 0)
        self.assertEqual(bounded[0]["content"], "You are a coding agent.")
        self.assertEqual(bounded[-1]["content"], "continue")
        self.assertLessEqual(len(main_module.trae_client.flatten_query(bounded)), 12_000)

    def test_bounded_remote_query_trims_single_oversized_message(self):
        messages = [
            {"role": "system", "content": "Keep the active task and answer concisely."},
            {
                "role": "user",
                "content": "BEGIN TASK\n" + ("x" * 100_000) + "\nEND TASK",
            },
        ]

        bounded, removed = main_module._bounded_remote_query(
            messages, {"trae_remote_query_max_chars": 4_000}
        )

        query = main_module.trae_client.flatten_query(bounded)
        self.assertEqual(removed, 0)
        self.assertLessEqual(len(query), 4_000)
        self.assertIn("BEGIN TASK", bounded[-1]["content"])
        self.assertIn("END TASK", bounded[-1]["content"])
        self.assertIn("older context omitted", bounded[-1]["content"])

    def test_bounded_remote_query_rejects_unfittable_fixed_overhead(self):
        messages = [
            {"role": "system", "content": "Keep the task."},
            {"role": "user", "content": "x" * 100},
        ]

        with self.assertRaises(main_module.RemoteQueryTooLarge):
            main_module._bounded_remote_query(
                messages, {"trae_remote_query_max_chars": 1}
            )

    def test_bounded_remote_query_handles_megabyte_single_turns(self):
        for size in (1_000_000, 7_000_000):
            with self.subTest(size=size):
                messages = [
                    {"role": "system", "content": "Keep the active task."},
                    {"role": "user", "content": "A" * size},
                ]
                bounded, removed = main_module._bounded_remote_query(
                    messages, {"trae_remote_query_max_chars": 480_000}
                )
                self.assertEqual(removed, 0)
                self.assertLessEqual(
                    len(main_module.trae_client.flatten_query(bounded)),
                    480_000,
                )
                self.assertIn("older context omitted", bounded[-1]["content"])

    def test_bounded_remote_query_counts_large_images_without_truncating_them(self):
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 600_000},
        }
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "describe"}, image],
            }
        ]
        original = json.dumps(messages)
        self.assertLess(len(main_module.trae_client.flatten_query(messages)), 1_000)

        with self.assertRaises(main_module.RemoteQueryTooLarge) as raised:
            main_module._bounded_remote_query(
                messages, {"trae_remote_query_max_chars": 480_000}
            )

        self.assertGreater(raised.exception.query_chars, 600_000)
        self.assertIn("Attached images", str(raised.exception))
        self.assertEqual(json.dumps(messages), original)

    def test_bounded_remote_query_rejects_image_only_overflow(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "A" * 8_000,
                        },
                    }
                ],
            }
        ]
        with self.assertRaises(main_module.RemoteQueryTooLarge):
            main_module._bounded_remote_query(
                messages, {"trae_remote_query_max_chars": 4_000}
            )

    def test_bounded_remote_query_trims_text_but_preserves_image_data(self):
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 1_000},
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "START\n" + "x" * 10_000 + "\nEND"},
                    image,
                ],
            }
        ]
        bounded, removed = main_module._bounded_remote_query(
            messages, {"trae_remote_query_max_chars": 4_000}
        )

        self.assertEqual(removed, 0)
        self.assertLessEqual(
            len(main_module.trae_client.flatten_query_parts(bounded)), 4_000
        )
        self.assertEqual(bounded[0]["content"][-1], image)
        text = bounded[0]["content"][0]["text"]
        self.assertIn("START", text)
        self.assertIn("END", text)
        self.assertIn("older context omitted", text)
        parts = json.loads(main_module.trae_client.flatten_query_parts(bounded))
        media = next(part for part in parts if part["type"] == "image_url")
        self.assertEqual(media["data"]["data"], "A" * 1_000)

    def test_bounded_remote_query_removes_old_image_turn_as_a_whole(self):
        latest = {
            "role": "user",
            "content": [
                {"type": "text", "text": "latest"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 8_000},
                    }
                ],
            },
            latest,
        ]
        bounded, removed = main_module._bounded_remote_query(
            messages, {"trae_remote_query_max_chars": 4_000}
        )
        self.assertEqual(removed, 1)
        self.assertEqual(bounded, [latest])
        self.assertLessEqual(
            len(main_module.trae_client.flatten_query_parts(bounded)), 4_000
        )

    def test_oversized_image_returns_413_before_session_or_account_rotation(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 600_000},
                    }
                ],
            }
        ]

        async def run(stream):
            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module, "_ensure_model_account_enabled"),
                patch.object(
                    main_module.auth, "get_polling_status", return_value={"enabled": True}
                ),
                patch.object(main_module, "_polling_retry_limit", return_value=2),
                patch.object(main_module, "_next_retry_account_snapshot") as rotate,
                patch.object(
                    main_module.trae_client, "acquire_web_slot", new=AsyncMock()
                ) as acquire,
                patch.object(
                    main_module.trae_remote_client, "create_session", new=AsyncMock()
                ) as create,
                patch.dict(os.environ, {"TRAE_REMOTE_QUERY_MAX_CHARS": "480000"}),
            ):
                result = await main_module._dispatch_chat(
                    messages,
                    "glm-5.2",
                    stream,
                    {
                        "_upstream_mode": "remote",
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                    },
                )
                rotate.assert_not_called()
                acquire.assert_not_awaited()
                create.assert_not_awaited()
                return result

        for stream in (False, True):
            with self.subTest(stream=stream):
                response = asyncio.run(run(stream))
                self.assertEqual(response.status_code, 413)
                error = json.loads(response.body)["error"]
                self.assertEqual(error["type"], "context_length_exceeded")
                self.assertIn("IDE/Solo", error["message"])

    def test_short_continuation_retains_active_client_tool_task(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "Download https://example.com/archive.zip to "
                    "C:/workspace/output/archive.zip and verify the file."
                ),
            },
            {"role": "assistant", "content": "I will continue."},
            {"role": "user", "content": "继续"},
        ]
        options = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "download_file",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }

        anchor = main_module._remote_client_tool_task_anchor(messages, options)

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["role"], "system")
        self.assertIn("https://example.com/archive.zip", anchor["content"])
        self.assertIn("C:/workspace/output/archive.zip", anchor["content"])
        self.assertIn("emit the matching client tool call", anchor["content"])

    def test_task_anchor_survives_remote_query_trimming(self):
        anchor = {
            "role": "system",
            "content": (
                "Active caller task retained during history compaction.\n"
                "Download https://example.com/archive.zip to "
                "C:/workspace/output/archive.zip."
            ),
        }
        messages = [
            {"role": "system", "content": "client tool runtime"},
            anchor,
            {"role": "user", "content": "old " + "x" * 40_000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "继续"},
        ]

        bounded, removed = main_module._bounded_remote_query(
            messages, {"trae_remote_query_max_chars": 4_000}
        )

        self.assertGreater(removed, 0)
        self.assertIn(anchor, bounded)
        self.assertEqual(bounded[-1]["content"], "继续")

    def test_remote_caller_tools_default_to_agent_executor(self):
        async def empty_events():
            if False:
                yield None

        async def run():
            captured = {}

            async def fake_create(client, token, model, msgs, *, options=None):
                captured.update(dict(options or {}))
                return ("work-session", "work-message")

            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=fake_create),
                patch.object(main_module.trae_remote_client, "stream_events", return_value=empty_events()),
                patch.object(main_module.trae_remote_client, "stop_session", new=AsyncMock()),
                patch.object(main_module, "collect_nonstream_web", new=AsyncMock(return_value={"choices": []})),
                patch.object(main_module, "_track_usage_from_result", return_value=None),
            ):
                await main_module.run_remote_session(
                    [{"role": "user", "content": "download the file"}],
                    "glm-5.3",
                    False,
                    {
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "download_file",
                                    "parameters": {"type": "object"},
                                },
                            }
                        ],
                    },
                )
            return captured

        captured = asyncio.run(run())
        # No explicit tier marker is needed: create_session resolves the
        # default remote executor to Agent, including for tool-bearing turns.
        self.assertNotIn("_remote_agent_type", captured)
        self.assertNotIn("_session_variant", captured)
        self.assertEqual(
            trae_remote_client.remote_agent_type("glm-5.3", captured),
            "solo_agent_remote",
        )

    def test_remote_caller_tools_can_explicitly_use_work_executor(self):
        async def empty_events():
            if False:
                yield None

        async def run():
            captured = {}

            async def fake_create(client, token, model, msgs, *, options=None):
                captured.update(dict(options or {}))
                return ("work-session", "work-message")

            with (
                patch.dict(
                    main_module.os.environ,
                    {"TRAE_REMOTE_CALLER_TOOLS_USE_WORK": "1"},
                ),
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=fake_create),
                patch.object(main_module.trae_remote_client, "stream_events", return_value=empty_events()),
                patch.object(main_module.trae_remote_client, "stop_session", new=AsyncMock()),
                patch.object(main_module, "collect_nonstream_web", new=AsyncMock(return_value={"choices": []})),
                patch.object(main_module, "_track_usage_from_result", return_value=None),
            ):
                await main_module.run_remote_session(
                    [{"role": "user", "content": "download the file"}],
                    "glm-5.3",
                    False,
                    {
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "download_file",
                                    "parameters": {"type": "object"},
                                },
                            }
                        ],
                    },
                )
            return captured

        captured = asyncio.run(run())
        self.assertEqual(captured["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(captured["_session_variant"], "caller-tools-work")

    def test_remote_slot_is_released_once_when_nonstream_translation_fails(self):
        async def empty_events():
            if False:
                yield None

        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(
                    main_module.trae_client,
                    "acquire_web_slot",
                    new=AsyncMock(),
                ) as acquire,
                patch.object(
                    main_module.trae_client,
                    "release_web_slot",
                ) as release,
                patch.object(
                    main_module.trae_remote_client,
                    "create_session",
                    new=AsyncMock(return_value=("session-1", "message-1")),
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "stream_events",
                    return_value=empty_events(),
                ),
                patch.object(
                    main_module.trae_remote_client,
                    "stop_session",
                    new=AsyncMock(),
                ),
                patch.object(
                    main_module,
                    "collect_nonstream_web",
                    new=AsyncMock(side_effect=RuntimeError("translation failed")),
                ),
            ):
                try:
                    await main_module.run_remote_session(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {"_account_id": "account-1", "_auth_token": "jwt-token"},
                    )
                except RuntimeError as exc:
                    self.assertEqual(str(exc), "translation failed")
                else:
                    self.fail("run_remote_session should propagate translation failure")
                return acquire, release

        acquire, release = asyncio.run(run())
        acquire.assert_awaited_once()
        release.assert_called_once_with("account-1")

    def test_agent_create_failure_falls_back_to_work_once(self):
        async def empty_events():
            if False:
                yield None

        async def run():
            create_calls = []
            trace = {}

            async def fake_create(client, token, model, msgs, *, options=None):
                create_calls.append(dict(options or {}))
                if len(create_calls) == 1:
                    raise RuntimeError("agent create failed")
                return ("work-session", "work-message")

            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=fake_create),
                patch.object(main_module.trae_remote_client, "stream_events", return_value=empty_events()),
                patch.object(main_module.trae_remote_client, "stop_session", new=AsyncMock()),
                patch.object(main_module, "collect_nonstream_web", new=AsyncMock(return_value={"choices": []})),
                patch.object(main_module, "_track_usage_from_result", return_value=None),
            ):
                result = await main_module.run_remote_session(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    False,
                    {
                        "_account_id": "account-1",
                        "_auth_token": "jwt-token",
                        "_upstream_trace": trace,
                    },
                )
            return result, create_calls, trace

        result, create_calls, trace = asyncio.run(run())
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(create_calls), 2)
        self.assertNotIn("_remote_agent_type", create_calls[0])
        self.assertEqual(create_calls[1]["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(create_calls[1]["_session_variant"], "work-fallback")
        self.assertEqual(trace["actual_mode"], "work")
        self.assertEqual(trace["actual_endpoint"], "work")
        self.assertEqual(trace["product"], "traework")
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["fallback_target"], "work")

    def test_agent_model_binding_failure_does_not_fallback_to_work(self):
        """A deterministic model/config mismatch must reach account rotation."""

        async def run():
            create = AsyncMock(
                side_effect=RuntimeError(
                    "Trae remote model binding mismatch: requested 'glm-5.3', "
                    "resolved 'kimi-k2.6'"
                )
            )
            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(
                    main_module.trae_remote_client,
                    "create_session",
                    new=create,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "model binding mismatch"):
                    await main_module.run_remote_session(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {"_account_id": "account-1", "_auth_token": "jwt-token"},
                    )
            return create

        create = asyncio.run(run())
        create.assert_awaited_once()

    def test_nonstream_empty_agent_falls_back_to_work(self):
        async def empty_events():
            if False:
                yield None

        async def run():
            create_calls = []

            async def fake_create(client, token, model, msgs, *, options=None):
                create_calls.append(dict(options or {}))
                return (f"session-{len(create_calls)}", f"message-{len(create_calls)}")

            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=fake_create),
                patch.object(main_module.trae_remote_client, "stream_events", side_effect=lambda *args, **kwargs: empty_events()),
                patch.object(main_module.trae_remote_client, "stop_session", new=AsyncMock()),
                patch.object(
                    main_module,
                    "collect_nonstream_web",
                    new=AsyncMock(
                        side_effect=[
                            main_module.EmptyUpstreamResponse(
                                "empty", retryable=True
                            ),
                            {"choices": [{"message": {"content": "ok"}}]},
                        ]
                    ),
                ),
                patch.object(main_module, "_track_usage_from_result", return_value=None),
            ):
                result = await main_module.run_remote_session(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    False,
                    {"_account_id": "account-1", "_auth_token": "jwt-token"},
                )
            return result, create_calls

        result, create_calls = asyncio.run(run())
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(create_calls), 2)
        self.assertEqual(create_calls[1]["_remote_agent_type"], "solo_work_remote")

    def test_nonstream_fallback_rebinds_usage_to_work_message(self):
        async def empty_events():
            if False:
                yield None

        async def run():
            tracker = main_module._UsageTracker(
                "glm-5.3",
                "/v1/chat/completions",
                False,
                {"_account_id": "account-1", "_auth_token": "jwt-token"},
            )
            create_count = 0

            async def fake_create(client, token, model, msgs, *, options=None):
                nonlocal create_count
                create_count += 1
                return (
                    f"session-{create_count}",
                    "agent-empty-message" if create_count == 1 else "work-success-message",
                )

            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=fake_create),
                patch.object(
                    main_module.trae_remote_client,
                    "stream_events",
                    side_effect=lambda *args, **kwargs: empty_events(),
                ),
                patch.object(main_module.trae_remote_client, "stop_session", new=AsyncMock()),
                patch.object(
                    main_module,
                    "collect_nonstream_web",
                    new=AsyncMock(
                        side_effect=[
                            main_module.EmptyUpstreamResponse("empty", retryable=True),
                            {"choices": [{"message": {"content": "ok"}}]},
                        ]
                    ),
                ),
                patch.object(main_module, "_track_usage_from_result", return_value=None),
                patch.object(main_module, "_USAGE_TRACKER", main_module.ContextVar("test_usage_tracker")),
            ):
                context_token = main_module._USAGE_TRACKER.set(tracker)
                try:
                    await main_module.run_remote_session(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {"_account_id": "account-1", "_auth_token": "jwt-token"},
                    )
                finally:
                    main_module._USAGE_TRACKER.reset(context_token)
            return tracker.usage_turn_id

        self.assertEqual(asyncio.run(run()), "work-success-message")

    def test_explicit_work_does_not_fallback_to_itself(self):
        async def run():
            create = AsyncMock(side_effect=RuntimeError("work create failed"))
            with (
                patch.object(
                    main_module.auth,
                    "get_account_record",
                    return_value={"token": "jwt-token", "provider_specific": {}},
                ),
                patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
                patch.object(main_module.trae_client, "release_web_slot"),
                patch.object(main_module.trae_remote_client, "create_session", new=create),
            ):
                with self.assertRaisesRegex(RuntimeError, "work create failed"):
                    await main_module.run_remote_session(
                        [{"role": "user", "content": "hello"}],
                        "work",
                        False,
                        {"_account_id": "account-1", "_auth_token": "jwt-token"},
                    )
            return create

        create = asyncio.run(run())
        create.assert_awaited_once()

    def test_session_binding_keeps_provider_metadata_with_bound_credential(self):
        main_module._UPSTREAM_SESSION_LEASES.clear()
        main_module._CHAT_HISTORY_SESSIONS.clear()
        provider_specific = {
            "webId": "web-account-1",
            "bizUserId": "biz-account-1",
            "region": "cn",
        }
        with (
            patch.object(main_module, "UPSTREAM_MODE", "remote"),
            patch.object(main_module.auth, "next_polling_account"),
            patch.object(
                main_module.auth,
                "get_active_account_snapshot",
                return_value=(
                    "store-account-1",
                    {
                        "token": "jwt-account-1",
                        "provider_specific": provider_specific,
                    },
                ),
            ),
        ):
            bound = main_module._bind_chat_session(
                [{"role": "user", "content": "hello"}],
                {},
                requested_session_id="provider-bound-session",
            )

        self.assertEqual(bound["_account_id"], "store-account-1")
        self.assertEqual(bound["_auth_token"], "jwt-account-1")
        self.assertEqual(bound["provider_specific"], provider_specific)
        self.assertIsNot(bound["provider_specific"], provider_specific)

    def test_remote_retry_rebinds_provider_metadata_to_rotated_account(self):
        main_module._UPSTREAM_SESSION_LEASES.clear()
        main_module._UPSTREAM_SESSION_LEASES["retry-session"] = main_module._UpstreamSessionLease(
            account_id="old-account",
            billing_id="old-account",
            auth_token="old-token",
            last_client_activity=0.0,
            provider_specific={"webId": "old-web"},
        )
        calls = []

        async def fake_remote(_messages, _model, _stream, options):
            calls.append(dict(options or {}))
            if len(calls) == 1:
                raise RuntimeError("Trae remote create_session [429]: parallel limit")
            return "ok"

        next_provider = {"webId": "new-web", "bizUserId": "new-biz"}
        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "old-account", "is_valid": True},
                        {"id": "new-account", "is_valid": True},
                    ],
                ),
                patch.object(main_module.auth, "next_polling_account"),
                patch.object(
                    main_module.auth,
                    "get_active_account_snapshot",
                    return_value=(
                        "new-account",
                        {"token": "new-token", "provider_specific": next_provider},
                    ),
                ),
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                return await main_module._run_remote_with_retry(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    False,
                    {
                        "session_id": "retry-session",
                        "_account_id": "old-account",
                        "_billing_id": "old-account",
                        "_auth_token": "old-token",
                        "provider_specific": {"webId": "old-web"},
                    },
                )

        self.assertEqual(asyncio.run(run()), "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["_account_id"], "new-account")
        self.assertEqual(calls[1]["_auth_token"], "new-token")
        self.assertEqual(calls[1]["_auth_user_id"], "new-account")
        self.assertEqual(calls[1]["provider_specific"], next_provider)
        lease = main_module._UPSTREAM_SESSION_LEASES["retry-session"]
        self.assertEqual(lease.account_id, "new-account")
        self.assertEqual(lease.provider_specific, next_provider)

    def test_remote_retry_requests_each_valid_account_at_most_once(self):
        calls = []

        async def fake_remote(_messages, _model, _stream, options):
            calls.append(str((options or {}).get("_account_id") or ""))
            raise RuntimeError("Trae remote create_session [429]: parallel limit")

        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "account-a", "is_valid": True},
                        {"id": "expired", "is_valid": False},
                        {"id": "account-b", "is_valid": True},
                    ],
                ),
                patch.object(main_module.auth, "next_polling_account") as rotate,
                patch.object(
                    main_module.auth,
                    "get_active_account_snapshot",
                    return_value=("account-b", {"token": "token-b"}),
                ),
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                with self.assertRaisesRegex(RuntimeError, "All remote accounts busy"):
                    await main_module._run_remote_with_retry(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {"_account_id": "account-a", "_auth_token": "token-a"},
                    )
                return rotate

        rotate = asyncio.run(run())
        self.assertEqual(calls, ["account-a", "account-b"])
        rotate.assert_called_once()

    def test_remote_retry_does_not_rotate_deterministic_query_limit_error(self):
        calls = []

        async def fake_remote(_messages, _model, _stream, options):
            calls.append(str((options or {}).get("_account_id") or ""))
            raise main_module.RemoteQueryTooLarge(700_000, 480_000)

        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "account-a", "is_valid": True},
                        {"id": "account-b", "is_valid": True},
                    ],
                ),
                patch.object(main_module.auth, "next_polling_account") as rotate,
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                with self.assertRaises(main_module.RemoteQueryTooLarge):
                    await main_module._run_remote_with_retry(
                        [{"role": "user", "content": "oversized"}],
                        "kimi-k2.6",
                        False,
                        {"_account_id": "account-a", "_auth_token": "token-a"},
                    )
                return rotate

        rotate = asyncio.run(run())
        self.assertEqual(calls, ["account-a"])
        rotate.assert_not_called()

    def test_remote_retry_rotates_when_model_is_not_bound(self):
        calls = []

        async def fake_remote(_messages, _model, _stream, options):
            calls.append(str((options or {}).get("_account_id") or ""))
            if len(calls) == 1:
                raise RuntimeError(
                    "Trae remote model is not available for the bound account: glm-5.3"
                )
            return "ok"

        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "account-a", "is_valid": True},
                        {"id": "account-b", "is_valid": True},
                    ],
                ),
                patch.object(main_module.auth, "next_polling_account") as rotate,
                patch.object(
                    main_module.auth,
                    "get_active_account_snapshot",
                    return_value=("account-b", {"token": "token-b"}),
                ),
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                return await main_module._run_remote_with_retry(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    False,
                    {"_account_id": "account-a", "_auth_token": "token-a"},
                )

        result = asyncio.run(run())
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["account-a", "account-b"])

    def test_remote_retry_preserves_model_unavailable_after_all_accounts(self):
        calls = []

        async def fake_remote(_messages, _model, _stream, options):
            calls.append(str((options or {}).get("_account_id") or ""))
            raise RuntimeError(
                "Trae remote model is not available for the bound account: glm-5.3"
            )

        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "account-a", "is_valid": True},
                        {"id": "account-b", "is_valid": True},
                    ],
                ),
                patch.object(main_module.auth, "next_polling_account") as rotate,
                patch.object(
                    main_module.auth,
                    "get_active_account_snapshot",
                    return_value=("account-b", {"token": "token-b"}),
                ),
                patch.object(main_module, "run_remote_session", new=fake_remote),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "remote model is not available for the bound account",
                ):
                    await main_module._run_remote_with_retry(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {"_account_id": "account-a", "_auth_token": "token-a"},
                    )
                return rotate

        rotate = asyncio.run(run())
        self.assertEqual(calls, ["account-a", "account-b"])
        rotate.assert_called_once()

    def test_web_retry_requests_each_valid_account_at_most_once(self):
        main_module._UPSTREAM_SESSION_LEASES.clear()
        main_module._UPSTREAM_SESSION_LEASES["web-retry-session"] = (
            main_module._UpstreamSessionLease(
                account_id="account-a",
                billing_id="account-a",
                auth_token="token-a",
                last_client_activity=0.0,
                provider_specific={"webId": "old-web"},
            )
        )
        calls = []
        tracker = AsyncMock()
        next_provider = {"webId": "new-web", "bizUserId": "new-biz"}

        async def fake_web(_messages, _model, _stream, options):
            calls.append(str((options or {}).get("_account_id") or ""))
            raise RuntimeError("Trae web [429]: solo_agent_parallel_limit")

        async def run():
            with (
                patch.object(
                    main_module.auth,
                    "get_polling_status",
                    return_value={"enabled": True},
                ),
                patch.object(
                    main_module.auth,
                    "list_accounts",
                    return_value=[
                        {"id": "account-a", "is_valid": True},
                        {"id": "account-b", "is_valid": True},
                    ],
                ),
                patch.object(main_module.auth, "next_polling_account") as rotate,
                patch.object(
                    main_module.auth,
                    "get_active_account_snapshot",
                    return_value=(
                        "account-b",
                        {"token": "token-b", "provider_specific": next_provider},
                    ),
                ),
                patch.object(main_module, "run_web_session", new=fake_web),
            ):
                with self.assertRaisesRegex(RuntimeError, "All web accounts busy"):
                    await main_module._run_web_with_retry(
                        [{"role": "user", "content": "hello"}],
                        "glm-5.3",
                        False,
                        {
                            "session_id": "web-retry-session",
                            "_account_id": "account-a",
                            "_auth_token": "token-a",
                        },
                        tracker=tracker,
                    )
                return rotate

        rotate = asyncio.run(run())
        self.assertEqual(calls, ["account-a", "account-b"])
        rotate.assert_called_once()
        tracker.rebind.assert_awaited_once()
        rebound = tracker.rebind.await_args.args[0]
        self.assertEqual(rebound["_account_id"], "account-b")
        self.assertEqual(rebound["provider_specific"], next_provider)
        lease = main_module._UPSTREAM_SESSION_LEASES["web-retry-session"]
        self.assertEqual(lease.account_id, "account-b")
        self.assertEqual(lease.provider_specific, next_provider)

    def test_cn_defaults_are_used_without_international_switch(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertIn("trae-api-cn", trae_remote_client.base_url())
            headers = trae_remote_client.build_headers("jwt")
        self.assertEqual(headers["x-user-region"], "CN")
        self.assertIn("solo.trae.cn", headers["Referer"])

    def test_bare_trae_host_is_repaired_with_remote_path(self):
        """A host-only base must not 404 every chat_sessions/models call (#9)."""

        cases = (
            "https://trae-api-cn.mchost.guru",
            "https://trae-api-cn.mchost.guru/",
            "https://core-normal.trae.cn",
        )
        for value in cases:
            self.assertEqual(
                trae_remote_client._normalize_remote_base(value),
                value.rstrip("/") + "/api/remote/v1",
            )

    def test_existing_remote_path_and_custom_hosts_are_preserved(self):
        self.assertEqual(
            trae_remote_client._normalize_remote_base(
                "https://trae-api-cn.mchost.guru/api/remote/v1/"
            ),
            "https://trae-api-cn.mchost.guru/api/remote/v1",
        )
        self.assertEqual(
            trae_remote_client._normalize_remote_base("https://custom.example/v1"),
            "https://custom.example/v1",
        )
        self.assertEqual(
            trae_remote_client._normalize_remote_base(
                "https://custom.example/proxy/trae.ai"
            ),
            "https://custom.example/proxy/trae.ai",
        )
        self.assertEqual(
            trae_remote_client._normalize_remote_base(
                "https://custom.example/v1?next=trae.ai"
            ),
            "https://custom.example/v1?next=trae.ai",
        )

    def test_base_url_normalizes_a_legacy_bare_gateway_override(self):
        """``base_url()`` must repair a stale bare-domain option, not just env."""

        self.assertEqual(
            trae_remote_client.base_url({"base_url": "https://trae-api-cn.mchost.guru"}),
            "https://trae-api-cn.mchost.guru/api/remote/v1",
        )

    def test_dispatch_uses_remote_alias_for_plain_chat(self):
        async def fake_remote(messages, model, stream, options):
            return "remote-ok"

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "9router"),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}], "auto", False, {}
                )

        self.assertEqual(asyncio.run(run()), "remote-ok")

    def test_raw_mode_uses_raw_for_explicit_glm_by_default(self):
        async def fake_raw(messages, model, stream, options):
            return {"route": "raw", "model": model}

        async def fake_remote(*args, **kwargs):
            raise AssertionError("explicit models default to raw v2")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(main_module, "run_raw_chat", new=fake_raw),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}], "glm-5.3", False, {}
                )

        self.assertEqual(asyncio.run(run()), {"route": "raw", "model": "glm-5.3"})

    def test_raw_mode_keeps_verified_deepseek_on_raw_v2(self):
        async def fake_raw(messages, model, stream, options):
            return {"route": "raw", "model": model}

        async def fake_remote(*args, **kwargs):
            raise AssertionError("verified raw-v2 model should remain on raw")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(main_module, "run_raw_chat", new=fake_raw),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}],
                    "DeepSeek-V4-Pro",
                    False,
                    {},
                )

        self.assertEqual(
            asyncio.run(run()),
            {"route": "raw", "model": "DeepSeek-V4-Pro"},
        )

    def test_remote_only_models_can_still_force_remote(self):
        async def fake_remote(messages, model, stream, options):
            return {"route": "remote", "model": model}

        async def fake_raw(*args, **kwargs):
            raise AssertionError("forced remote model must not use raw IDE routing")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(
                    main_module,
                    "_remote_only_models",
                    return_value={"glm-5.3"},
                ),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
                patch.object(main_module, "run_raw_chat", new=fake_raw),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}],
                    "glm-5.3",
                    False,
                    {},
                )

        self.assertEqual(
            asyncio.run(run()),
            {"route": "remote", "model": "glm-5.3"},
        )

    def test_remote_only_wildcard_forces_auto_to_remote(self):
        async def fake_remote(messages, model, stream, options):
            return {"route": "remote", "model": model}

        async def fake_raw(*args, **kwargs):
            raise AssertionError("wildcard remote override must bypass raw")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(main_module, "_remote_only_models", return_value={"*"}),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
                patch.object(main_module, "run_raw_chat", new=fake_raw),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}], "auto", False, {}
                )

        self.assertEqual(
            asyncio.run(run()),
            {"route": "remote", "model": "auto"},
        )

    def test_raw_mode_keeps_auto_on_native_route(self):
        async def fake_raw(messages, model, stream, options):
            return {"route": "raw", "model": model}

        async def fake_remote(*args, **kwargs):
            raise AssertionError("auto should retain native automatic selection")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(main_module, "run_raw_chat", new=fake_raw),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}], "auto", False, {}
                )

        self.assertEqual(
            asyncio.run(run()),
            {"route": "raw", "model": "auto"},
        )

    def test_display_alias_is_compared_against_effective_remote_model(self):
        from src.sse import _check_provider_model

        # The web catalog reports the concrete official config, while clients
        # may send the Chinese display label.
        _check_provider_model(
            "DeepSeek-V4-Pro 正式版",
            "DeepSeek-V4-Pro-Official__dev",
        )

    def test_remote_only_provider_mismatch_returns_502(self):
        async def fake_remote(messages, model, stream, options):
            from src.sse import ModelProviderMismatch

            raise ModelProviderMismatch("Trae selected provider model 'kimi-k2.6' for requested model 'glm-5.3'")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "raw"),
                patch.object(
                    main_module,
                    "_remote_only_models",
                    return_value={"glm-5.3"},
                ),
                patch.object(main_module, "_run_remote_with_retry", new=fake_remote),
            ):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "hello"}], "glm-5.3", False, {}
                )

        response = asyncio.run(run())
        self.assertEqual(response.status_code, 502)
        body = json.loads(response.body)
        self.assertEqual(body["error"]["type"], "upstream_model_mismatch")

    def test_dispatch_routes_remote_tool_policy_to_ide(self):
        """Remote tool policies must use an endpoint preserving caller tools."""
        messages = [{"role": "user", "content": "inspect"}]
        options = {"tools": []}
        remote = AsyncMock(return_value="remote-ok")
        ide = AsyncMock(return_value="ide-ok")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "remote"),
                patch.object(main_module, "_run_remote_with_retry", remote),
                patch.object(main_module, "run_ide_chat", ide),
            ):
                return await main_module._dispatch_chat(
                    messages, "auto", False, options
                )

        response = asyncio.run(run())
        self.assertEqual(response, "ide-ok")
        ide.assert_awaited_once()
        remote.assert_not_awaited()
        called_messages, called_model, called_stream, called_options = (
            ide.await_args.args
        )
        self.assertEqual(called_messages, messages)
        # Tool-bearing automatic requests are classified as complex.  When
        # 6.1sol is not present in the loaded catalog, the safe fallback is
        # provider-owned auto selection rather than an implicit Kimi mapping.
        self.assertEqual(called_model, "auto")
        self.assertFalse(called_stream)
        self.assertIsNot(called_options, options)
        self.assertEqual(called_options["tools"], [])
        self.assertEqual(called_options["_task_model_reason"], "tools")
        self.assertEqual(called_options["_task_model_candidate"], "6.1sol")
        self.assertEqual(called_options["_task_model_effective"], "auto")
        self.assertEqual(called_options["reasoning_effort"], "ultra")


if __name__ == "__main__":
    unittest.main()


class _StreamResponseWithEvents:
    status_code = 200

    def __init__(self, chunks):
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class RemoteReasoningEffortTests(unittest.TestCase):
    def test_reasoning_effort_reaches_custom_model(self):
        async def run():
            client = _Client()
            custom = {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "config_source": 1,
                "reasoning_effort_config": {
                    "support_thinking": True,
                    "options": ["light", "high"],
                    "default_level": "high",
                },
            }
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=AsyncMock(return_value=custom),
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={"reasoning_effort": "low"},
                )
            return client.posts[0]

        _url, request = asyncio.run(run())
        model = request["json"]["initial_message"]["custom_model"]
        self.assertEqual(model["reasoning_effort"], "light")

    def test_work_tier_does_not_borrow_agent_tier_effort_config(self):
        work_model = {
            "name": "glm-5.3",
            "config_name": "glm-5.3",
            "model_name": "glm-5.3",
            "config_source": 1,
        }
        agent_model = dict(
            work_model,
            reasoning_effort_config={
                "support_thinking": True,
                "options": ["light", "high", "extra_high"],
            },
        )

        async def resolver(_name, **kwargs):
            if kwargs.get("agent_type") == "solo_agent_remote":
                return agent_model
            return work_model

        async def run():
            client = _Client()
            trace = {}
            with patch.object(
                trae_remote_client.trae_client,
                "resolve_model_config",
                new=resolver,
            ):
                await trae_remote_client.create_session(
                    client,
                    "jwt-token",
                    "glm-5.3",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "reasoning_effort": "xhigh",
                        "_remote_agent_type": "solo_work_remote",
                        "_upstream_trace": trace,
                    },
                )
            return client.posts[0], trace

        (_url, request), trace = asyncio.run(run())
        message = request["json"]["initial_message"]
        self.assertEqual(message["agent_type"], "solo_work_remote")
        self.assertNotIn("reasoning_effort", message["custom_model"])
        self.assertNotIn("reasoning_effort_level", message["custom_model"])
        self.assertNotIn("reasoning_effort_applied", trace)
