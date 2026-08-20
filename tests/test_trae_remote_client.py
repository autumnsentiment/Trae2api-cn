import asyncio
import json
import unittest
from unittest.mock import patch

from src import trae_remote_client
from src import main as main_module


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

    def test_dispatch_rejects_tools_in_remote_mode(self):
        async def run():
            with patch.object(main_module, "UPSTREAM_MODE", "remote"):
                return await main_module._dispatch_chat(
                    [{"role": "user", "content": "inspect"}],
                    "auto",
                    False,
                    {"tools": []},
                )

        response = asyncio.run(run())
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
