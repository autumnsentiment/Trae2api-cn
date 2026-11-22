import asyncio
import threading
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpcore
import httpx

from src import trae_client


LLM_ENDPOINT = "/api/agent/v3/llm_utils_chat"


class FakeResponse:
    def __init__(self, status_code=200, read_error=None):
        self.status_code = status_code
        self.text = "unavailable"
        self.read_error = read_error
        self.close_count = 0

    def read(self):
        if self.read_error is not None:
            raise self.read_error
        return b"unavailable"

    def close(self):
        self.close_count += 1


class LegacyTransportRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.requests = []
        self.clients = []
        self.actions = []
        self.real_client = httpx.Client
        owner = self

        class FakeClient:
            def __init__(self, **kwargs):
                self.close_count = 0
                owner.clients.append(self)

            def build_request(self, method, url, **kwargs):
                return httpx.Request(method, url, **kwargs)

            def send(self, request, stream=True):
                owner.requests.append(request)
                action = owner.actions.pop(0)
                return action(request) if callable(action) else action

            def close(self):
                self.close_count += 1

        self.stack.enter_context(
            patch("src.trae_client.auth.get_account_record", return_value={"user_id": "acct"})
        )
        self.stack.enter_context(
            patch(
                "src.trae_client.auth.get_auth",
                return_value=SimpleNamespace(host="https://example.test"),
            )
        )
        self.stack.enter_context(patch.object(trae_client, "build_headers", return_value={}))
        self.stack.enter_context(
            patch.object(
                trae_client,
                "_resolve_request_model_name",
                new=AsyncMock(side_effect=lambda model, **kwargs: model),
            )
        )
        self.stack.enter_context(
            patch.object(
                trae_client, "resolve_traecode_model_config", new=AsyncMock(return_value=None)
            )
        )
        self.stack.enter_context(
            patch.object(trae_client, "acquire_web_slot", new=AsyncMock(return_value="lease"))
        )
        self.release_slot = self.stack.enter_context(
            patch.object(trae_client, "release_web_slot_owned")
        )
        self.stack.enter_context(patch.object(trae_client.httpx, "Client", FakeClient))
        self.stack.enter_context(
            patch.object(
                trae_client,
                "build_llm_chat_body",
                return_value={"session_id": "billing-turn", "request_id": "billing-turn"},
            )
        )
        self.stack.enter_context(
            patch.object(
                trae_client,
                "build_trae_ide_request",
                new=AsyncMock(return_value=(None, {"session_id": "ide-turn"})),
            )
        )
        self.stack.enter_context(
            patch.object(
                trae_client, "build_work_agent_body", return_value={"session_id": "work-turn"}
            )
        )

    async def send(self, **options):
        return await trae_client.send_chat_request(
            [{"role": "user", "content": "hello"}],
            "auto",
            True,
            {"_account_id": "acct", "_auth_token": "synthetic-token", **options},
        )

    @staticmethod
    def fail(error_type, events=()):
        def action(request):
            trace = request.extensions["trace"]
            for event in events:
                trace(event, {})
            raise error_type("synthetic transport failure", request=request)

        return action

    def assert_closed_once(self):
        self.release_slot.assert_called_once()
        self.assertTrue(self.clients)
        self.assertTrue(all(client.close_count == 1 for client in self.clients))

    async def test_connect_error_retries_once_with_same_body_and_billing(self):
        upstream = FakeResponse()
        self.actions = [self.fail(httpx.ConnectError), upstream]

        response = await self.send(_ide_endpoint=LLM_ENDPOINT)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0].content, self.requests[1].content)
        self.assertEqual(self.requests[0].url, self.requests[1].url)
        self.assertEqual(response.usage_turn_id, "billing-turn")
        self.release_slot.assert_not_called()
        response.close()
        response.close()
        self.assertEqual(upstream.close_count, 1)
        self.assert_closed_once()

    async def test_tls_phase_read_and_protocol_failure_retry_once(self):
        for error_type in (httpx.ReadError, httpx.RemoteProtocolError):
            with self.subTest(error_type=error_type.__name__):
                self.requests.clear()
                self.clients.clear()
                self.release_slot.reset_mock()
                self.actions = [
                    self.fail(
                        error_type,
                        ("connection.start_tls.started", "connection.start_tls.failed"),
                    ),
                    FakeResponse(),
                ]
                response = await self.send(_ide_endpoint=LLM_ENDPOINT)
                self.assertEqual(len(self.requests), 2)
                response.close()
                self.assert_closed_once()

    async def test_repeated_connection_failure_stops_after_second_attempt(self):
        self.actions = [self.fail(httpx.ConnectError), self.fail(httpx.ConnectError)]
        with self.assertRaisesRegex(RuntimeError, "synthetic transport failure"):
            await self.send()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.clients), 1)
        self.assert_closed_once()

    async def test_repeated_tls_failure_stops_after_second_attempt(self):
        failure = self.fail(httpx.ReadError, ("connection.start_tls.failed",))
        self.actions = [failure, failure]
        with self.assertRaisesRegex(RuntimeError, "synthetic transport failure"):
            await self.send()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.clients), 1)
        self.assert_closed_once()

    async def test_disable_fallback_blocks_connection_retry(self):
        self.actions = [self.fail(httpx.ConnectError)]
        with self.assertRaises(RuntimeError):
            await self.send(_disable_upstream_fallback=True)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.clients), 1)
        self.assert_closed_once()

    async def test_disable_fallback_blocks_next_endpoint_after_http_error(self):
        self.actions = [FakeResponse(status_code=503)]
        with self.assertRaises(RuntimeError):
            await self.send(_disable_upstream_fallback=True)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.clients), 1)
        self.assert_closed_once()

    async def test_other_endpoints_do_not_add_connection_retry(self):
        for endpoint in ("/api/ide/v1/chat", "/api/agent/v3/create_agent_task"):
            with self.subTest(endpoint=endpoint):
                self.requests.clear()
                self.clients.clear()
                self.release_slot.reset_mock()
                self.actions = [self.fail(httpx.ConnectError)]
                with self.assertRaises(RuntimeError):
                    await self.send(_ide_endpoint=endpoint)
                self.assertEqual(len(self.requests), 1)
                self.assert_closed_once()

    async def test_ambiguous_send_failure_does_not_replay_on_another_endpoint(self):
        for error_type in (
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.WriteError,
            httpx.ReadTimeout,
        ):
            with self.subTest(error_type=error_type.__name__):
                self.requests.clear()
                self.clients.clear()
                self.release_slot.reset_mock()
                self.actions = [self.fail(error_type)]
                with self.assertRaises(RuntimeError):
                    await self.send()
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(len(self.clients), 1)
                self.assert_closed_once()

    async def test_unexpected_send_exception_does_not_replay(self):
        def fail(_request):
            raise RuntimeError("synthetic unexpected failure")

        self.actions = [fail]
        with self.assertRaisesRegex(RuntimeError, "synthetic unexpected failure"):
            await self.send()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.clients), 1)
        self.assert_closed_once()

    async def test_started_request_blocks_retry_even_with_connection_failure_trace(self):
        for error_type in (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            with self.subTest(error_type=error_type.__name__):
                self.requests.clear()
                self.clients.clear()
                self.release_slot.reset_mock()
                self.actions = [
                    self.fail(
                        error_type,
                        (
                            "connection.start_tls.failed",
                            "http11.send_request_headers.started",
                        ),
                    )
                ]
                with self.assertRaises(RuntimeError):
                    await self.send()
                self.assertEqual(len(self.requests), 1)
                self.assert_closed_once()

    async def test_connection_start_without_failure_is_insufficient_evidence(self):
        self.actions = [
            self.fail(httpx.ReadError, ("connection.start_tls.started",))
        ]
        with self.assertRaises(RuntimeError):
            await self.send()
        self.assertEqual(len(self.requests), 1)
        self.assert_closed_once()

    async def test_error_response_read_failure_does_not_replay(self):
        self.actions = [
            FakeResponse(status_code=503, read_error=httpx.ReadError("synthetic EOF"))
        ]
        with self.assertRaisesRegex(RuntimeError, "synthetic EOF"):
            await self.send()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.clients), 1)
        self.assert_closed_once()

    async def test_successful_response_stream_read_failure_does_not_replay(self):
        upstream = FakeResponse(read_error=httpx.ReadError("synthetic stream EOF"))
        self.actions = [upstream]
        response = await self.send()
        with self.assertRaises(httpx.ReadError):
            response.response.read()
        self.assertEqual(len(self.requests), 1)
        response.close()
        self.assert_closed_once()

    async def test_http_error_endpoint_fallback_remains_available_when_enabled(self):
        self.actions = [FakeResponse(status_code=503), FakeResponse()]
        response = await self.send()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0].url.path, LLM_ENDPOINT)
        self.assertEqual(self.requests[1].url.path, "/api/ide/v1/chat")
        response.close()
        self.assert_closed_once()

    async def test_cancel_during_retry_holds_slot_until_worker_finishes(self):
        started = threading.Event()
        finish = threading.Event()
        upstream = FakeResponse()

        def delayed(_request):
            started.set()
            finish.wait(2)
            return upstream

        self.actions = [self.fail(httpx.ConnectError), delayed]
        task = asyncio.create_task(self.send(_ide_endpoint=LLM_ENDPOINT))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.release_slot.assert_not_called()
        finally:
            finish.set()
        for _ in range(100):
            if self.release_slot.called:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(upstream.close_count, 1)
        self.assert_closed_once()

    async def test_real_httpx_trace_distinguishes_tls_eof_from_response_eof(self):
        for failure_phase in ("tls", "response"):
            with self.subTest(failure_phase=failure_phase):
                streams = []
                self.release_slot.reset_mock()

                class FakeNetworkStream:
                    def __init__(self):
                        self.writes = []
                        self.response_sent = False

                    def start_tls(self, **kwargs):
                        if failure_phase == "tls" and len(streams) == 1:
                            raise httpcore.ReadError("synthetic TLS EOF")
                        return self

                    def get_extra_info(self, name):
                        return None

                    def write(self, buffer, timeout=None):
                        self.writes.append(bytes(buffer))

                    def read(self, max_bytes, timeout=None):
                        if failure_phase == "response":
                            raise httpcore.ReadError("synthetic response EOF")
                        if self.response_sent:
                            return b""
                        self.response_sent = True
                        return b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

                    def close(self):
                        pass

                def connect_tcp(_backend, *args, **kwargs):
                    stream = FakeNetworkStream()
                    streams.append(stream)
                    return stream

                with (
                    patch.object(
                        trae_client.httpx,
                        "Client",
                        side_effect=lambda **kwargs: self.real_client(
                            trust_env=False, **kwargs
                        ),
                    ),
                    patch(
                        "httpcore._backends.sync.SyncBackend.connect_tcp",
                        new=connect_tcp,
                    ),
                ):
                    if failure_phase == "tls":
                        response = await self.send(_ide_endpoint=LLM_ENDPOINT)
                        self.assertEqual(len(streams), 2)
                        self.assertEqual(streams[0].writes, [])
                        self.assertTrue(streams[1].writes)
                        response.close()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "synthetic response EOF"):
                            await self.send()
                        self.assertEqual(len(streams), 1)
                        self.assertTrue(streams[0].writes)
                self.release_slot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
