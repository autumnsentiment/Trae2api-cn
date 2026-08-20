import gzip
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src import main as main_module
from src.main import app


class RequestBodyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def _chat_payload(self, message):
        return {"model": "auto", "messages": [message]}

    def test_gzip_chat_body_is_decoded_and_forwarded(self):
        payload = self._chat_payload({"role": "user", "content": "gzip prompt"})
        compressed = gzip.compress(json.dumps(payload).encode("utf-8"))
        captured = {}

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            return JSONResponse(
                {
                    "id": "chatcmpl-body",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            response = self.client.post(
                "/v1/chat/completions",
                content=compressed,
                headers={
                    "Authorization": "Bearer smoke-key",
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["messages"][-1]["content"], "gzip prompt")

    def test_chat_parts_and_string_messages_are_not_dropped(self):
        payload = {
            "model": "auto",
            "messages": [
                "plain string prompt",
                {"role": "user", "parts": [{"type": "text", "text": "parts prompt"}]},
            ],
        }
        captured = {}

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            return JSONResponse({"choices": []})

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            response = self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer smoke-key"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [item["content"] for item in captured["messages"]],
            ["plain string prompt", [{"type": "text", "text": "parts prompt"}]],
        )

    def test_chat_prompt_alias_is_forwarded(self):
        captured = {}

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            return JSONResponse({"choices": []})

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "auto", "prompt": "alias prompt"},
                headers={"Authorization": "Bearer smoke-key"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["messages"][-1]["content"], "alias prompt")

    def test_responses_bare_input_text_item_reaches_dispatch(self):
        payload = {
            "model": "auto",
            "input": [{"type": "input_text", "text": "bare responses prompt"}],
        }
        captured = {}

        async def fake_dispatch(messages, model, stream, options=None):
            captured["messages"] = messages
            return JSONResponse(
                {
                    "id": "chatcmpl-body",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        with patch.object(main_module, "_dispatch_chat", new=fake_dispatch):
            response = self.client.post(
                "/v1/responses",
                json=payload,
                headers={"Authorization": "Bearer smoke-key"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["messages"][-1]["content"], "bare responses prompt")

    def test_empty_body_is_reported_without_dispatching_upstream(self):
        dispatch = AsyncMock(side_effect=AssertionError("upstream must not run"))
        with patch.object(main_module, "_dispatch_chat", new=dispatch):
            response = self.client.post(
                "/v1/chat/completions",
                content=b"",
                headers={
                    "Authorization": "Bearer smoke-key",
                    "Content-Type": "application/json",
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["error"]["message"])
        dispatch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
