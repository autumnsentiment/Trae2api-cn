import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

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
        self.assertEqual(body["initial_message"]["model_selection_strategy"], "auto")
        self.assertIn("hello", body["initial_message"]["query"])

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

    def test_max_mode_enriches_slim_account_config(self):
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
        self.assertEqual(initial["context_window_size"], 1000000)
        self.assertEqual(initial["prompt_max_tokens"], 936000)
        self.assertEqual(initial["max_tokens"], 64000)
        self.assertEqual(
            cm["context_window_size"], {"default": 200000, "max": [1000000]}
        )
        self.assertEqual(cm["context_window_tokens"], {"dev": 200000, "max": 1000000})
        self.assertEqual(cm["prompt_max_tokens"], 936000)
        self.assertEqual(cm["max_tokens"], 64000)
        self.assertEqual(cm["max_turn"], 500)
        self.assertTrue(cm["features"]["context_windows"]["enable"])
        self.assertEqual(
            cm["features"]["context_windows"]["data"]["max_context"], 1000000
        )

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

    def test_remote_caller_tools_use_work_executor(self):
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
                    {"_account_id": "account-1", "_auth_token": "jwt-token"},
                )
            return result, create_calls

        result, create_calls = asyncio.run(run())
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(create_calls), 2)
        self.assertNotIn("_remote_agent_type", create_calls[0])
        self.assertEqual(create_calls[1]["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(create_calls[1]["_session_variant"], "work-fallback")

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

        self.assertEqual(asyncio.run(run()), {"route": "raw", "model": "auto"})

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

    def test_dispatch_accepts_tools_in_remote_mode(self):
        """Tools are no longer rejected in remote mode; they are injected as a
        system prompt and the upstream response is filtered for tool calls."""
        messages = [{"role": "user", "content": "inspect"}]
        options = {"tools": []}
        remote = AsyncMock(return_value="remote-ok")

        async def run():
            with (
                patch.object(main_module, "UPSTREAM_MODE", "remote"),
                patch.object(main_module, "_run_remote_with_retry", remote),
            ):
                return await main_module._dispatch_chat(
                    messages, "auto", False, options
                )

        response = asyncio.run(run())
        self.assertEqual(response, "remote-ok")
        remote.assert_awaited_once_with(messages, "auto", False, options)


if __name__ == "__main__":
    unittest.main()
