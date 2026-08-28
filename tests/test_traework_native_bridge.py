import asyncio
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src import traework_native_bridge as native


class _FakeClient:
    def __init__(self, transport=None, **kwargs):
        self.transport = transport
        self.kwargs = kwargs
        self.requests = []
        self.is_closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def build_request(self, method, url, **kwargs):
        request = httpx.Request(method, url, **kwargs)
        self.requests.append(request)
        return request

    def send(self, request, stream=False):
        self.requests.append(request)
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=b"event: done\ndata: {\"response\":\"ok\"}\n\n",
        )

    def get(self, url):
        request = httpx.Request("GET", url)
        return httpx.Response(200, request=request, json={"status": "ok"})

    def close(self):
        self.is_closed = True


class NativeBridgePacketTests(unittest.TestCase):
    def test_packet_matches_report_and_does_not_copy_internal_options(self):
        packet = native.build_aha_packet(
            [{"role": "user", "content": "hello"}],
            "glm-5.3",
            options={
                "_auth_token": "secret-token",
                "_account_id": "account-1",
                "connect_session_id": "connect-1",
                "native_channel_id": "channel-1",
                "tools": [{"type": "function", "function": {"name": "Read"}}],
                "workspace_folder": r"C:\demo",
            },
        )
        self.assertEqual(packet["packet_type"], "request")
        self.assertEqual(packet["channel_id"], "channel-1")
        self.assertEqual(packet["session_id"], "connect-1")
        params = packet["params"]
        self.assertEqual(params["service"], "chat")
        self.assertEqual(params["method"], "chat")
        self.assertEqual(params["client_info"]["connect_session_id"], "connect-1")
        self.assertEqual(params["data"]["model_name"], "glm-5.3")
        self.assertEqual(params["data"]["tools"][0]["function"]["name"], "Read")
        self.assertNotIn("_account_id", json.dumps(packet))

    def test_length_prefixed_decoder_handles_split_and_multiple_frames(self):
        first = {"event": "token", "data": {"text": "a"}}
        second = {"event": "Done", "data": {}}
        encoded = native.encode_aha_frame(first) + native.encode_aha_frame(second)
        decoder = native.AhaFrameDecoder(max_frame_bytes=4096)
        self.assertEqual(decoder.feed(encoded[:3]), [])
        self.assertEqual(decoder.feed(encoded[3:11]), [])
        self.assertEqual(decoder.feed(encoded[11:]), [first, second])
        decoder.finish()

    def test_frame_limit_rejects_oversized_payload(self):
        decoder = native.AhaFrameDecoder(max_frame_bytes=1024)
        with self.assertRaises(native.NativeBridgeProtocolError):
            decoder.feed(struct.pack(">I", 2048))

    def test_installation_inspection_is_non_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = root / "modules" / "ai-agent"
            agent.mkdir(parents=True)
            for name in ("ai_agent.dll", "sscronet.dll", "meta.json", "start.bat"):
                (agent / name).write_bytes(b"x")
            status = native.inspect_installation(root)
        self.assertTrue(status["available"])
        self.assertNotIn("token", json.dumps(status).lower())


class NativeBridgeClientTests(unittest.TestCase):
    def test_linux_is_rejected_by_default(self):
        config = native.NativeBridgeConfig(enabled=True)
        bridge = native.TraeWorkNativeBridge(config)
        with patch.object(native.sys, "platform", "linux"):
            with self.assertRaises(native.NativeBridgeUnavailable):
                asyncio.run(
                    bridge.send_chat_request(
                        [{"role": "user", "content": "hello"}], "auto"
                    )
                )

    def test_http_helper_receives_packet_and_auth_header(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=b"event: done\ndata: {\"response\":\"ok\"}\n\n",
            )

        transport = httpx.MockTransport(handler)

        def factory(**kwargs):
            return httpx.Client(transport=transport, **kwargs)

        config = native.NativeBridgeConfig(
            enabled=True,
            allow_non_windows=True,
            bridge_url="https://bridge.example",
        )
        bridge = native.TraeWorkNativeBridge(config, client_factory=factory)
        response = asyncio.run(
            bridge.send_chat_request(
                [{"role": "user", "content": "hello"}],
                "glm-5.3",
                options={"_auth_token": "jwt-token", "connect_session_id": "s1"},
            )
        )
        self.assertEqual(captured["request"].url.path, "/v1/traework/request_stream")
        self.assertEqual(captured["request"].headers["Authorization"], "Cloud-IDE-JWT jwt-token")
        self.assertEqual(captured["body"]["packet_type"], "request")
        self.assertEqual(captured["body"]["session_id"], "s1")
        response.close()


if __name__ == "__main__":
    unittest.main()
