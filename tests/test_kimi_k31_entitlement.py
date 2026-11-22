import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from src import main as main_module
from src import trae_remote_client


AUTH_HEADERS = {"Authorization": "Bearer kimi-entitlement-key"}


def _descriptor(identities):
    return {
        "name": "kimi-k3.1-pro",
        "config_name": "kimi-k3.1-pro",
        "model_name": "kimi-k3.1-pro",
        "features": {
            "access": {
                "data": {
                    "identity_list": identities,
                }
            }
        },
    }


class KimiK31EntitlementTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_identity_text_is_treated_as_denied_pay_identity(self):
        client = AsyncMock()
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=_descriptor([5, 1, 2, 3, 100])),
            ),
            patch.object(
                trae_remote_client.auth,
                "resolve_model_account_id",
                side_effect=lambda account_id: account_id,
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={
                    "provider_specific": {"userIdentity": "Free"},
                },
            ),
        ):
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await trae_remote_client.create_session(
                    client,
                    "token-a",
                    "kimi-k3.1-pro",
                    [{"role": "user", "content": "hello"}],
                    options={"_account_id": "account-a"},
                )
        client.post.assert_not_awaited()

    async def test_free_identity_text_is_denied_by_native_solo_descriptor(self):
        client = AsyncMock()
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=_descriptor([5, 1, 2, 3, 100])),
            ),
            patch.object(
                trae_remote_client.auth,
                "resolve_model_account_id",
                side_effect=lambda account_id: account_id,
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={
                    "provider_specific": {"userIdentity": "Free"},
                },
            ),
        ):
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await trae_remote_client.create_session(
                    client,
                    "token-a",
                    "kimi-k3.1-pro",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "account-a",
                        "_remote_agent_type": "solo_agent",
                    },
                )
        client.post.assert_not_awaited()

    async def test_bound_account_identity_wins_over_global_provider_metadata(self):
        client = AsyncMock()
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=_descriptor([5, 1, 2, 3, 100])),
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={"user_pay_identity": 0},
            ) as get_record,
        ):
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await trae_remote_client.create_session(
                    client,
                    "token-a",
                    "kimi-k3.1-pro",
                    [{"role": "user", "content": "hello"}],
                    options={
                        "_account_id": "account-a",
                        "provider_specific": {"userPayIdentity": 5},
                    },
                )
        get_record.assert_called_once_with("account-a")
        client.post.assert_not_awaited()

    async def test_entitlement_error_never_falls_back_to_work(self):
        async def fake_create(*args, **kwargs):
            raise trae_remote_client.RemoteModelNotEntitled(
                "Trae model_not_entitled: Kimi-K3.1-Pro"
            )

        create = AsyncMock(side_effect=fake_create)
        with (
            patch.object(
                main_module.auth,
                "get_account_record",
                return_value={
                    "token": "token-a",
                    "provider_specific": {},
                    "user_pay_identity": 0,
                },
            ),
            patch.object(main_module.trae_client, "acquire_web_slot", new=AsyncMock()),
            patch.object(main_module.trae_client, "release_web_slot"),
            patch.object(
                main_module.trae_remote_client,
                "create_session",
                new=create,
            ),
        ):
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await main_module.run_remote_session(
                    [{"role": "user", "content": "hello"}],
                    "kimi-k3.1-pro",
                    False,
                    {"_account_id": "account-a", "_auth_token": "token-a"},
                )
        create.assert_awaited_once()

    async def test_create_session_rejects_known_denied_pay_identity_before_post(self):
        client = AsyncMock()
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=_descriptor([5, 1, 2, 3, 100])),
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={"user_pay_identity": 0},
            ),
        ):
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await trae_remote_client.create_session(
                    client,
                    "token-a",
                    "kimi-k3.1-pro",
                    [{"role": "user", "content": "hello"}],
                    options={"_account_id": "account-a"},
                )
        client.post.assert_not_awaited()

    async def test_create_session_allows_known_entitled_pay_identity(self):
        client = AsyncMock()
        client.post.return_value = type(
            "Response",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": lambda self: {
                    "code": 0,
                    "data": {
                        "chat_session_id": "session-a",
                        "message_id": "message-a",
                    },
                },
            },
        )()
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=_descriptor([5, 1, 2, 3, 100])),
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={"user_pay_identity": 5},
            ),
        ):
            session_id, message_id = await trae_remote_client.create_session(
                client,
                "token-a",
                "kimi-k3.1-pro",
                [{"role": "user", "content": "hello"}],
                options={"_account_id": "account-a"},
            )
        self.assertEqual((session_id, message_id), ("session-a", "message-a"))
        client.post.assert_awaited_once()

    async def test_same_account_ordinary_kimi_remains_available(self):
        client = AsyncMock()
        client.post.return_value = type(
            "Response",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": lambda self: {
                    "code": 0,
                    "data": {
                        "chat_session_id": "session-k3",
                        "message_id": "message-k3",
                    },
                },
            },
        )()
        descriptor = {
            "name": "kimi-k3",
            "config_name": "kimi-k3",
            "model_name": "kimi-k3",
            "features": {
                "access": {
                    "data": {
                        "identity_list": [0, 5, 1, 2, 3, 100],
                    }
                }
            },
        }
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=descriptor),
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={"user_pay_identity": 0},
            ),
        ):
            session_id, message_id = await trae_remote_client.create_session(
                client,
                "token-a",
                "kimi-k3",
                [{"role": "user", "content": "hello"}],
                options={"_account_id": "account-a"},
            )
        self.assertEqual((session_id, message_id), ("session-k3", "message-k3"))
        client.post.assert_awaited_once()

    async def test_kimi_k3_comparison_does_not_relax_k31_entitlement(self):
        """A descriptive kimi-k3 entry must not validate kimi-k3.1-pro."""

        client = AsyncMock()
        client.post.return_value = type(
            "Response",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": lambda self: {
                    "code": 0,
                    "data": {
                        "chat_session_id": "session-k3",
                        "message_id": "message-k3",
                    },
                },
            },
        )()
        ordinary = {
            "name": "kimi-k3",
            "config_name": "kimi-k3",
            "model_name": "kimi-k3",
            "features": {
                "access": {"data": {"identity_list": [0, 5, 1, 2, 3, 100]}}
            },
        }
        with (
            patch.object(
                trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(
                    side_effect=[
                        ordinary,
                        _descriptor([5, 1, 2, 3, 100]),
                    ]
                ),
            ),
            patch.object(
                trae_remote_client.auth,
                "get_account_record",
                return_value={"user_pay_identity": 0},
            ),
        ):
            await trae_remote_client.create_session(
                client,
                "token-a",
                "kimi-k3",
                [{"role": "user", "content": "hello"}],
                options={"_account_id": "account-a"},
            )
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await trae_remote_client.create_session(
                    client,
                    "token-a",
                    "kimi-k3.1-pro",
                    [{"role": "user", "content": "hello"}],
                    options={"_account_id": "account-a"},
                )
        client.post.assert_awaited_once()

    async def test_dispatcher_maps_entitlement_error_to_403(self):
        internal_account = "account-sensitive-a"
        with (
            patch.object(
                main_module,
                "run_remote_session",
                new=AsyncMock(
                    side_effect=trae_remote_client.RemoteModelNotEntitled(
                        "Trae model_not_entitled: Kimi-K3.1-Pro "
                        f"for account {internal_account}"
                    )
                ),
            ),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": False},
            ),
        ):
            response = await main_module._dispatch_chat(
                [{"role": "user", "content": "hello"}],
                "kimi-k3.1-pro",
                False,
                {
                    "_upstream_mode": "remote",
                    "_disable_upstream_fallback": True,
                    "_disable_traecode_legacy_fallback": True,
                },
            )
        self.assertEqual(response.status_code, 403)
        self.assertIn(b"model_not_entitled", response.body)
        self.assertNotIn(internal_account.encode(), response.body)

    async def test_native_descriptor_denial_fails_before_wire(self):
        descriptor = _descriptor([5, 1, 2, 3, 100])
        descriptor["config_name"] = "kimi-k3.1-pro-auto"
        descriptor["model_name"] = "kimi-k3.1-pro__dev"
        send_started = []

        class ForbiddenClient:
            def __init__(self, **_kwargs):
                pass

            def build_request(self, *_args, **_kwargs):
                send_started.append(True)
                raise AssertionError("denied model must not reach the wire")

            def close(self):
                pass

        with (
            patch(
                "src.trae_client.resolve_traecode_model_config",
                new=AsyncMock(return_value=descriptor),
            ),
            patch.object(
                main_module.auth,
                "resolve_model_account_id",
                side_effect=lambda account_id: account_id,
            ),
            patch.object(
                main_module.auth,
                "get_account_record",
                return_value={"user_pay_identity": 0},
            ),
            patch("src.raw_client.httpx.Client", side_effect=ForbiddenClient),
        ):
            with self.assertRaises(main_module.raw_client.TraeCodeModelNotEntitled):
                await main_module.raw_client.send_traecode_chat_request(
                    [{"role": "user", "content": "hello"}],
                    "kimi-k3.1-pro",
                    stream=False,
                    options={
                        "_account_id": "account-a",
                        "_auth_token": "token-a",
                        "_traecode_raw_chat_function": "solo_agent",
                        "_disable_upstream_fallback": True,
                    },
                )
        self.assertEqual(send_started, [])

    async def test_native_descriptor_denial_maps_to_403_without_rotation(self):
        denied = main_module.raw_client.TraeCodeModelNotEntitled(
            "kimi-k3.1-pro", "solo_agent", 0
        )
        with (
            patch.object(
                main_module,
                "run_ide_chat",
                new=AsyncMock(side_effect=denied),
            ) as run,
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": True},
            ),
            patch.object(main_module, "_polling_retry_limit", return_value=15),
            patch.object(
                main_module,
                "_next_retry_account_snapshot",
            ) as rotate,
        ):
            with self.assertRaises(main_module.raw_client.TraeCodeModelNotEntitled):
                await main_module._run_ide_with_retry(
                    [{"role": "user", "content": "hello"}],
                    "kimi-k3.1-pro",
                    False,
                    {
                        "_account_id": "account-a",
                        "_auth_token": "token-a",
                        "_upstream_trace": {},
                    },
                )
        run.assert_awaited_once()
        rotate.assert_not_called()

        with patch.object(
            main_module,
            "run_ide_chat",
            new=AsyncMock(side_effect=denied),
        ):
            response = await main_module._dispatch_chat(
                [{"role": "user", "content": "hello"}],
                "kimi-k3.1-pro",
                False,
                {
                    "_upstream_mode": "ide",
                    "_disable_upstream_fallback": True,
                    "_disable_traecode_legacy_fallback": True,
                },
            )
        self.assertEqual(response.status_code, 403)
        self.assertIn(b"model_not_entitled", response.body)
        self.assertNotIn(b"account-a", response.body)

    async def test_entitlement_denial_rotates_at_most_once(self):
        """A model-access denial is account metadata, not a transient retry."""

        create = AsyncMock(
            side_effect=trae_remote_client.RemoteModelNotEntitled(
                "Trae model_not_entitled: Kimi-K3.1-Pro"
            )
        )
        with (
            patch.object(
                main_module,
                "run_remote_session",
                new=create,
            ),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": True},
            ),
            patch.object(
                main_module,
                "_next_retry_account_snapshot",
            ) as rotate,
            patch.object(main_module, "_polling_retry_limit", return_value=15),
        ):
            with self.assertRaises(trae_remote_client.RemoteModelNotEntitled):
                await main_module._run_remote_with_retry(
                    [{"role": "user", "content": "hello"}],
                    "kimi-k3.1-pro",
                    False,
                    {
                        "_account_id": "account-a",
                        "_auth_token": "token-a",
                        "_upstream_trace": {},
                    },
                )
        create.assert_awaited_once()
        rotate.assert_not_called()

    async def test_polling_remote_accepts_omitted_request_options(self):
        expected = JSONResponse({"choices": []})
        request = AsyncMock(return_value=expected)
        with (
            patch.object(main_module, "run_remote_session", new=request),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": True},
            ),
            patch.object(main_module, "_polling_retry_limit", return_value=1),
            patch.object(main_module, "_cancel_unclaimed_account_slot"),
        ):
            response = await main_module._run_remote_with_retry(
                [{"role": "user", "content": "hello"}],
                "kimi-k3",
                False,
            )
        self.assertIs(response, expected)
        request.assert_awaited_once_with(
            [{"role": "user", "content": "hello"}], "kimi-k3", False, {}
        )

    async def test_stream_disable_fallback_does_not_enter_work(self):
        """An empty Remote turn must stay Remote when fallback is disabled."""

        async def empty_events():
            if False:
                yield None

        created = AsyncMock(return_value=("session-a", "message-a"))
        trace = {}
        with (
            patch.object(
                main_module.trae_remote_client,
                "create_session",
                new=created,
            ),
            patch.object(
                main_module.trae_remote_client,
                "_stream_events_unbounded",
                side_effect=lambda *args, **kwargs: empty_events(),
            ),
            patch.object(
                main_module.trae_client,
                "acquire_web_slot",
                new=AsyncMock(return_value="lease-a"),
            ),
            patch.object(main_module.trae_client, "release_web_slot_owned"),
            patch.object(
                main_module.trae_remote_client,
                "_resolve_model_config",
                new=AsyncMock(return_value=_descriptor([5, 1, 2, 3, 100])),
            ),
            patch.object(
                main_module.trae_remote_client,
                "stop_session",
                new=AsyncMock(),
            ),
        ):
            response = await main_module.run_remote_session(
                [{"role": "user", "content": "hello"}],
                "kimi-k3.1-pro",
                True,
                {
                    "_account_id": "account-a",
                    "_auth_token": "token-a",
                    "_disable_upstream_fallback": True,
                    "_upstream_trace": trace,
                },
            )
            with self.assertRaises(trae_remote_client.RemoteFirstEventTimeout):
                _ = [chunk async for chunk in response.body_iterator]

        created.assert_awaited_once()
        self.assertFalse(trace.get("fallback_used"))

    async def test_deferred_stream_preserves_entitlement_error_type(self):
        async def denied(*_args, **_kwargs):
            raise trae_remote_client.RemoteModelNotEntitled(
                "Trae model_not_entitled: Kimi-K3.1-Pro"
            )

        options = {
            "_upstream_mode": "remote",
            "_disable_upstream_fallback": True,
            "_disable_traecode_legacy_fallback": True,
        }
        with (
            patch.object(main_module, "_dispatch_chat", new=denied),
            patch.object(main_module, "_apply_auto_route", return_value=options),
            patch.object(main_module, "_cancel_unclaimed_account_slot"),
        ):
            result = "".join(
                [
                    chunk
                    async for chunk in main_module._deferred_dispatch_stream(
                        [{"role": "user", "content": "hello"}],
                        "kimi-k3.1-pro",
                        options,
                    )
                ]
            )
        self.assertEqual(result.count('"type": "model_not_entitled"'), 1)
        self.assertEqual(result.count("data: [DONE]"), 1)
        self.assertLess(
            result.index("model_not_entitled"),
            result.index("data: [DONE]"),
        )


class KimiK31PublicEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main_module.app)
        self.api_keys = patch.object(
            main_module, "API_KEYS", ["kimi-entitlement-key"]
        )
        self.upstream_mode = patch.object(main_module, "UPSTREAM_MODE", "remote")
        self.api_keys.start()
        self.upstream_mode.start()

    def tearDown(self):
        self.upstream_mode.stop()
        self.api_keys.stop()
        self.client.close()

    def _assert_isolated_dispatch(
        self, path, payload, *, expected_status=403, expected_marker="model_not_entitled"
    ):
        seen = []

        async def fake_dispatch(_messages, _model, _stream, options):
            seen.append(dict(options))
            return main_module._openai_error(
                403,
                "This model is not included in the bound account's "
                "Trae entitlement.",
                "model_not_entitled",
                "model",
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            response = self.client.post(
                path,
                json=payload,
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, expected_status, response.text)
        self.assertIn(expected_marker, response.text)
        self.assertNotIn("secret-account", response.text)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].get("_disable_upstream_fallback"))
        return response

    def test_chat_completion_public_switch_is_mapped(self):
        self._assert_isolated_dispatch(
            "/v1/chat/completions",
            {
                "model": "kimi-k3.1-pro",
                "messages": [{"role": "user", "content": "hello"}],
                "disable_fallback": True,
            },
        )

    def test_chat_completion_compat_alias_is_mapped(self):
        self._assert_isolated_dispatch(
            "/chat/completions",
            {
                "model": "kimi-k3.1-pro",
                "messages": [{"role": "user", "content": "hello"}],
                "disable_fallback": "true",
            },
        )

    def test_responses_public_switch_is_mapped(self):
        self._assert_isolated_dispatch(
            "/v1/responses",
            {
                "model": "kimi-k3.1-pro",
                "input": "hello",
                "disable_fallback": True,
            },
        )

    def test_messages_public_switch_is_mapped(self):
        self._assert_isolated_dispatch(
            "/v1/messages",
            {
                "model": "kimi-k3.1-pro",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "disable_fallback": True,
            },
            expected_marker="Trae entitlement",
        )

    def test_traework_public_switch_is_mapped(self):
        self._assert_isolated_dispatch(
            "/v1/traework/chat",
            {
                "model": "kimi-k3.1-pro",
                "messages": [{"role": "user", "content": "hello"}],
                "disable_fallback": True,
            },
            expected_status=200,
            expected_marker="event: error",
        )

    def test_streaming_chat_returns_single_entitlement_error(self):
        async def denied(*_args, **_kwargs):
            raise trae_remote_client.RemoteModelNotEntitled(
                "Trae model_not_entitled: Kimi-K3.1-Pro"
            )

        with patch.object(main_module, "_dispatch_chat", new=denied):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi-k3.1-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                    "disable_fallback": True,
                },
                headers=AUTH_HEADERS,
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.text.count('"type": "model_not_entitled"'), 1)
        self.assertEqual(response.text.count("data: [DONE]"), 1)


if __name__ == "__main__":
    unittest.main()
