import json
import os
import tempfile
import unittest
from pathlib import Path

FAKE_CMD = str(Path(__file__).resolve().parent / "fake" / "fake_cli.cmd")
WORKDIR = tempfile.mkdtemp(prefix="trae-relay-test-")

os.environ["TRAE_AUTH_SOURCE"] = "cli"
os.environ["UPSTREAM_MODE"] = "cli"
os.environ["TRAE_CLI_COMMAND"] = FAKE_CMD
os.environ["TRAE_CLI_WORKDIR"] = WORKDIR
os.environ["TRAE_CLI_PROMPT_MODE"] = "stdin"
os.environ["TRAE_CLI_OUTPUT_MODE"] = "json"
os.environ["TRAE_CLI_DISABLE_TOOLS"] = "false"
os.environ["RELAY_API_KEYS"] = "smoke-key"

from fastapi.testclient import TestClient

from src.main import app

AUTH_HEADERS = {"Authorization": "Bearer smoke-key"}




class MainCliSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_status(self):
        response = self.client.get("/v1/status", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "cli")
        self.assertEqual(body["upstream_mode"], "cli")
        self.assertTrue(body["cli"]["available"])

    def test_chat_nonstream(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        self.assertIn("fake reply", content)
        self.assertEqual(body["usage"]["total_tokens"], 17)

    def test_chat_stream(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=AUTH_HEADERS,
        ) as response:
            self.assertEqual(response.status_code, 200)
            lines = list(response.iter_lines())

        events = []
        done = False
        for line in lines:
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload))

        self.assertTrue(done)
        content = "".join(
            choice["delta"].get("content", "")
            for event in events
            for choice in event.get("choices", [])
        )
        self.assertIn("fake reply", content)
        self.assertIn(17, [event["usage"]["total_tokens"] for event in events if "usage" in event])

    def test_get_v1_models(self):
        response = self.client.get("/v1", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertTrue(body["data"])

    def test_healthz_without_auth(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)

    def test_web_login_public(self):
        response = self.client.get("/web/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("\u7528 Trae \u7f51\u9875\u6388\u6743\u767b\u5f55", response.text)

        download = self.client.get("/web/login/download")
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.content.startswith(b"#!/usr/bin/env python3"))

    def test_web_auth_callback(self):
        user_jwt = json.dumps({
            "ClientID": "ono9krqynydwx5",
            "Token": "oauth-token",
            "RefreshToken": "oauth-refresh",
            "TokenExpireAt": "1900000000000",
        })
        user_info = json.dumps({
            "UserID": "uid-1",
            "TenantID": "tenant-1",
            "Region": "CN",
            "AIRegion": "cn",
        })
        response = self.client.get(
            "/authorize",
            params={
                "userJwt": user_jwt,
                "userInfo": user_info,
                "host": "https://trae-api-cn.mchost.guru",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("\u767b\u5f55\u6210\u529f", response.text)
        status = self.client.get("/v1/status").json()
        self.assertTrue(status["has_token"])
        self.assertEqual(status["source"], "env")


if __name__ == "__main__":
    unittest.main()
