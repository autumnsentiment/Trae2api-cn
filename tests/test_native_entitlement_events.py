"""Keep explicit native entitlement failures separate from generic model access."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src import main, raw_client, sse

ENTITLEMENT_PAYLOADS = [
    {"code": 1005, "type": "model_not_entitled", "message": "Upgrade required"},
    {
        "code": 1005,
        "error": {"type": "model_not_entitled", "message": "Upgrade required"},
    },
    {"code": "1005", "message": "MODEL_NOT_ENTITLED: upgrade required"},
    {"code": 1005, "message": "Access denied", "error": {"reason": "model_not_entitled"}},
    {"code": 1005, "error": "model_not_entitled"},
]


def _response(payload):
    return SimpleNamespace(
        iter_lines=lambda: iter(["event: error", "data: " + json.dumps(payload), ""])
    )


class NativeEntitlementEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_and_nonstream_preserve_numeric_code_and_entitlement_type(self):
        for stream in (False, True):
            for payload in ENTITLEMENT_PAYLOADS:
                with self.subTest(stream=stream, payload=payload):
                    with self.assertRaises(sse.UpstreamEventError) as raised:
                        if stream:
                            async for _ in sse.translate_ide_stream(
                                _response(payload), "glm-5.2", fail_on_empty=True
                            ):
                                pass
                        else:
                            await sse.collect_nonstream_ide(
                                _response(payload), "glm-5.2", fail_on_empty=True
                            )
                    self.assertEqual(raised.exception.code, payload["code"])
                    self.assertEqual(raised.exception.error_type, "model_not_entitled")

    async def test_generic_1005_access_failure_is_not_reclassified_as_entitlement(self):
        payload = {"code": 1005, "message": "Model access denied in this scope"}
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(sse.UpstreamEventError) as raised:
                    if stream:
                        async for _ in sse.translate_ide_stream(
                            _response(payload), "glm-5.2", fail_on_empty=True
                        ):
                            pass
                    else:
                        await sse.collect_nonstream_ide(
                            _response(payload), "glm-5.2", fail_on_empty=True
                        )
                self.assertEqual(raised.exception.code, 1005)
                self.assertIsNone(raised.exception.error_type)

    async def test_explicit_entitlement_never_replays_through_legacy(self):
        for stream in (False, True):
            for payload in ENTITLEMENT_PAYLOADS:
                with self.subTest(stream=stream, payload=payload):
                    wrapper = SimpleNamespace(response=_response(payload), close=Mock())
                    legacy = AsyncMock()
                    with (
                        patch.object(
                            raw_client,
                            "send_traecode_chat_request",
                            AsyncMock(return_value=wrapper),
                        ),
                        patch.object(main.trae_client, "send_chat_request", legacy),
                        self.assertRaises(sse.UpstreamEventError) as raised,
                    ):
                        result = await main.run_ide_chat(
                            [{"role": "user", "content": "read"}],
                            "glm-5.2",
                            stream,
                            {"_traecode_raw_chat_function": "chat_v3"},
                        )
                        if stream:
                            async for _ in result.body_iterator:
                                pass
                    self.assertEqual(raised.exception.code, payload["code"])
                    self.assertEqual(raised.exception.error_type, "model_not_entitled")
                    legacy.assert_not_awaited()
                    wrapper.close.assert_called_once()

    def test_error_type_metadata_is_bounded(self):
        error = sse.UpstreamEventError("Failure", code=1005, error_type="x" * 1000)
        self.assertEqual(error.code, 1005)
        self.assertEqual(error.error_type, "x" * 128)
