import asyncio
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import anyio

from src import main as main_module


async def _empty_events():
    if False:
        yield None


async def _translated_events(*_args, **_kwargs):
    yield "data: first\n\n"
    yield "data: [DONE]\n\n"


class SessionCancellationTests(unittest.IsolatedAsyncioTestCase):
    paths = ("remote", "web")
    messages = [{"role": "user", "content": "hello"}]
    options = {"_account_id": "account-1", "_auth_token": "jwt-token"}

    @contextmanager
    def mocks(self, path):
        client = SimpleNamespace(aclose=AsyncMock())
        state = SimpleNamespace(
            client=client,
            factory=Mock(return_value=client),
            acquire=AsyncMock(),
            release=Mock(),
            create=AsyncMock(return_value=("session-1", "message-1")),
            stop=AsyncMock(),
            collect=AsyncMock(return_value={"choices": []}),
            leases={},
        )
        transport = (
            main_module.trae_remote_client
            if path == "remote"
            else main_module.trae_client
        )
        names = (
            ("create_session", "stream_events", "stop_session")
            if path == "remote"
            else ("create_web_session", "stream_web_events", "stop_web_session")
        )
        with ExitStack() as stack:
            for target, name, replacement in (
                (main_module.httpx, "AsyncClient", state.factory),
                (main_module.auth, "get_account_record", Mock(return_value={})),
                (main_module.auth, "get_polling_status", Mock(return_value={"enabled": False})),
                (main_module.trae_client, "acquire_web_slot", state.acquire),
                (main_module.trae_client, "release_web_slot", state.release),
                (main_module.trae_client, "_WEB_LEASES", state.leases),
                (transport, names[0], state.create),
                (transport, names[1], Mock(side_effect=lambda *_a, **_k: _empty_events())),
                (transport, names[2], state.stop),
                (main_module, "collect_nonstream_web", state.collect),
                (main_module, "translate_web_events", _translated_events),
                (main_module, "_bind_usage_turn", Mock()),
                (main_module, "_track_usage_from_result", Mock()),
                (main_module, "_track_usage_from_chunk", Mock()),
            ):
                stack.enter_context(patch.object(target, name, replacement))
            yield state

    async def run_session(self, path, stream=False, *, model="work"):
        run = (
            main_module.run_remote_session
            if path == "remote"
            else main_module.run_web_session
        )
        return await run(self.messages, model, stream, self.options)

    def assert_released_once(self, state, *, stop=True):
        state.client.aclose.assert_awaited_once()
        state.release.assert_called_once_with("account-1")
        self.assertFalse(state.leases)
        if stop:
            state.stop.assert_awaited_once()
        else:
            state.stop.assert_not_awaited()

    async def test_cancelled_slot_wait_does_not_release_unowned_resources(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                started = asyncio.Event()

                async def acquire(*_args, **_kwargs):
                    started.set()
                    await asyncio.Event().wait()

                state.acquire.side_effect = acquire
                task = asyncio.create_task(self.run_session(path))
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                state.factory.assert_not_called()
                state.release.assert_not_called()

    async def test_cancelled_create_closes_client_and_releases_slot(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                started = asyncio.Event()

                async def create(*_args, **_kwargs):
                    started.set()
                    await asyncio.Event().wait()

                state.create.side_effect = create
                task = asyncio.create_task(self.run_session(path))
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                state.acquire.assert_awaited_once()
                self.assert_released_once(state, stop=False)

    async def test_repeated_cancel_during_create_cleanup_finishes_before_return(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                creating = asyncio.Event()
                closing = asyncio.Event()
                finish_close = asyncio.Event()

                async def create(*_args, **_kwargs):
                    creating.set()
                    await asyncio.Event().wait()

                async def close():
                    closing.set()
                    await finish_close.wait()

                state.create.side_effect = create
                state.client.aclose.side_effect = close
                task = asyncio.create_task(self.run_session(path))
                await asyncio.wait_for(creating.wait(), timeout=1)
                task.cancel()
                await asyncio.wait_for(closing.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                state.release.assert_not_called()
                finish_close.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_released_once(state, stop=False)

    async def test_anyio_cancel_scope_does_not_interrupt_create_cleanup(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                async def close():
                    await asyncio.sleep(0)

                state.client.aclose.side_effect = close
                with anyio.CancelScope() as scope:
                    async def create(*_args, **_kwargs):
                        scope.cancel()
                        await anyio.sleep(0)

                    state.create.side_effect = create
                    await self.run_session(path)
                    self.fail("cancelled create must not return a response")
                self.assert_released_once(state, stop=False)

    async def test_client_constructor_failure_releases_slot(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                state.factory.side_effect = RuntimeError("client constructor failed")
                with self.assertRaisesRegex(RuntimeError, "client constructor failed"):
                    await self.run_session(path)
                state.acquire.assert_awaited_once()
                state.release.assert_called_once_with("account-1")
                state.create.assert_not_awaited()
                state.client.aclose.assert_not_awaited()

    async def test_translation_setup_failure_does_not_acquire_slot(self):
        for path in self.paths:
            with (
                self.subTest(path=path),
                self.mocks(path) as state,
                patch.object(
                    main_module,
                    "_tool_translation_options",
                    side_effect=RuntimeError("translation setup failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "translation setup failed"):
                    await self.run_session(path)
                state.acquire.assert_not_awaited()
                state.release.assert_not_called()
                state.factory.assert_not_called()

    async def test_remote_history_preparation_failure_does_not_acquire_slot(self):
        with (
            self.mocks("remote") as state,
            patch.object(
                main_module.raw_client,
                "_compact_raw_history",
                side_effect=RuntimeError("history preparation failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "history preparation failed"):
                await self.run_session("remote")
            state.acquire.assert_not_awaited()
            state.release.assert_not_called()
            state.factory.assert_not_called()

    async def test_create_failure_closes_client_and_releases_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                state.create.side_effect = RuntimeError("create failed")
                with self.assertRaisesRegex(RuntimeError, "create failed"):
                    await self.run_session(path)
                self.assert_released_once(state, stop=False)

    async def test_nonstream_success_stops_closes_and_releases_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                result = await self.run_session(path)
                self.assertEqual(result.status_code, 200)
                self.assert_released_once(state)

    async def test_nonstream_failure_stops_closes_and_releases_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                state.collect.side_effect = RuntimeError("translation failed")
                with self.assertRaisesRegex(RuntimeError, "translation failed"):
                    await self.run_session(path)
                self.assert_released_once(state)

    async def test_nonstream_cancel_after_create_stops_known_session(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                collecting = asyncio.Event()

                async def collect(*_args, **_kwargs):
                    collecting.set()
                    await asyncio.Event().wait()

                state.collect.side_effect = collect
                task = asyncio.create_task(self.run_session(path))
                await asyncio.wait_for(collecting.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_released_once(state)

    async def test_repeated_cancel_during_stop_still_closes_and_releases_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                stopping = asyncio.Event()
                finish_stop = asyncio.Event()

                async def stop(*_args, **_kwargs):
                    stopping.set()
                    await finish_stop.wait()

                state.stop.side_effect = stop
                task = asyncio.create_task(self.run_session(path))
                await asyncio.wait_for(stopping.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                state.client.aclose.assert_not_awaited()
                finish_stop.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_released_once(state)

    async def test_cancelled_create_preserves_cancel_when_close_fails(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                creating = asyncio.Event()

                async def create(*_args, **_kwargs):
                    creating.set()
                    await asyncio.Event().wait()

                state.create.side_effect = create
                state.client.aclose.side_effect = RuntimeError("cleanup failed")
                task = asyncio.create_task(self.run_session(path))
                await asyncio.wait_for(creating.wait(), timeout=1)
                task.cancel("client disconnected")
                with self.assertRaises(asyncio.CancelledError) as cancelled:
                    await task
                self.assertEqual(cancelled.exception.args, ("client disconnected",))
                self.assert_released_once(state, stop=False)

    async def test_cancelled_nonstream_preserves_cancel_when_cleanup_fails(self):
        for path in self.paths:
            for failure in ("stop", "close"):
                with self.subTest(path=path, failure=failure), self.mocks(path) as state:
                    collecting = asyncio.Event()

                    async def collect(*_args, **_kwargs):
                        collecting.set()
                        await asyncio.Event().wait()

                    state.collect.side_effect = collect
                    cleanup_mock = state.stop if failure == "stop" else state.client.aclose
                    cleanup_mock.side_effect = RuntimeError("cleanup failed")
                    task = asyncio.create_task(self.run_session(path))
                    await asyncio.wait_for(collecting.wait(), timeout=1)
                    task.cancel("client disconnected")
                    with self.assertRaises(asyncio.CancelledError) as cancelled:
                        await task
                    self.assertEqual(cancelled.exception.args, ("client disconnected",))
                    self.assert_released_once(state)

    async def test_stop_failure_still_closes_and_releases_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                state.stop.side_effect = RuntimeError("stop failed")
                with self.assertRaisesRegex(RuntimeError, "stop failed"):
                    await self.run_session(path)
                self.assert_released_once(state)

    async def test_client_close_failure_still_releases_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                state.client.aclose.side_effect = RuntimeError("close failed")
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    await self.run_session(path)
                self.assert_released_once(state)

    async def test_stream_handoff_keeps_resources_until_consumer_closes(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_session(path, stream=True)
                state.stop.assert_not_awaited()
                state.client.aclose.assert_not_awaited()
                state.release.assert_not_called()
                iterator = response.body_iterator
                self.assertEqual(await anext(iterator), "data: first\n\n")
                await iterator.aclose()
                await iterator.aclose()
                self.assert_released_once(state)

    async def test_unstarted_stream_close_releases_resources_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_session(path, stream=True)
                await response.body_iterator.aclose()
                await response.body_iterator.aclose()
                self.assert_released_once(state)

    async def test_stream_exhaustion_releases_resources_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_session(path, stream=True)
                chunks = [chunk async for chunk in response.body_iterator]
                self.assertEqual(chunks, ["data: first\n\n", "data: [DONE]\n\n"])
                await response.body_iterator.aclose()
                self.assert_released_once(state)

    async def test_stream_consumption_cancel_releases_resources_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                reading = asyncio.Event()

                async def translate(*_args, **_kwargs):
                    reading.set()
                    await asyncio.Event().wait()
                    yield "unreachable"

                with patch.object(main_module, "translate_web_events", translate):
                    response = await self.run_session(path, stream=True)
                    task = asyncio.create_task(anext(response.body_iterator))
                    await asyncio.wait_for(reading.wait(), timeout=1)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    await response.body_iterator.aclose()
                self.assert_released_once(state)

    async def test_stream_cancel_preserves_cancel_when_cleanup_fails(self):
        for path in self.paths:
            for failure in ("stop", "close"):
                with self.subTest(path=path, failure=failure), self.mocks(path) as state:
                    reading = asyncio.Event()

                    async def translate(*_args, **_kwargs):
                        reading.set()
                        await asyncio.Event().wait()
                        yield "unreachable"

                    cleanup_mock = state.stop if failure == "stop" else state.client.aclose
                    cleanup_mock.side_effect = RuntimeError("cleanup failed")
                    with patch.object(main_module, "translate_web_events", translate):
                        response = await self.run_session(path, stream=True)
                        task = asyncio.create_task(anext(response.body_iterator))
                        await asyncio.wait_for(reading.wait(), timeout=1)
                        task.cancel("client disconnected")
                        with self.assertRaises(asyncio.CancelledError) as cancelled:
                            await task
                        self.assertEqual(cancelled.exception.args, ("client disconnected",))
                    self.assert_released_once(state)

    async def test_web_reaper_and_stream_close_do_not_both_own_cleanup(self):
        with self.mocks("web") as state:
            response = await self.run_session("web", stream=True)
            self.assertTrue(main_module.trae_client.unregister_web_lease("session-1"))
            await state.stop()
            await state.client.aclose()
            state.release("account-1")
            await response.body_iterator.aclose()
            self.assert_released_once(state)

    async def test_remote_create_fallback_cancel_closes_each_attempt_once(self):
        with self.mocks("remote") as state:
            fallback_creating = asyncio.Event()
            retry_client = SimpleNamespace(aclose=AsyncMock())
            state.factory.side_effect = [state.client, retry_client]
            calls = 0

            async def create(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("agent create failed")
                fallback_creating.set()
                await asyncio.Event().wait()

            state.create.side_effect = create
            task = asyncio.create_task(self.run_session("remote", model="glm-5.3"))
            await asyncio.wait_for(fallback_creating.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assert_released_once(state, stop=False)
            retry_client.aclose.assert_awaited_once()

    async def test_remote_create_fallback_cancel_during_old_client_close(self):
        with self.mocks("remote") as state:
            closing = asyncio.Event()
            finish_close = asyncio.Event()
            state.create.side_effect = RuntimeError("agent create failed")

            async def close():
                closing.set()
                await finish_close.wait()

            state.client.aclose.side_effect = close
            task = asyncio.create_task(self.run_session("remote", model="glm-5.3"))
            await asyncio.wait_for(closing.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            finish_close.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assert_released_once(state, stop=False)
            state.factory.assert_called_once()
            state.create.assert_awaited_once()

    async def test_remote_nonstream_fallback_create_cancel_closes_slot_once(self):
        with self.mocks("remote") as state:
            creating_retry = asyncio.Event()
            calls = 0

            async def create(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return "session-1", "message-1"
                creating_retry.set()
                await asyncio.Event().wait()

            state.create.side_effect = create
            state.collect.side_effect = main_module.EmptyUpstreamResponse(
                "empty", retryable=True
            )
            task = asyncio.create_task(self.run_session("remote", model="glm-5.3"))
            await asyncio.wait_for(creating_retry.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assert_released_once(state)

    async def test_remote_nonstream_fallback_cancel_during_stop_does_not_stop_twice(self):
        with self.mocks("remote") as state:
            stopping = asyncio.Event()
            finish_stop = asyncio.Event()
            state.collect.side_effect = main_module.EmptyUpstreamResponse(
                "empty", retryable=True
            )

            async def stop(*_args, **_kwargs):
                stopping.set()
                await finish_stop.wait()

            state.stop.side_effect = stop
            task = asyncio.create_task(self.run_session("remote", model="glm-5.3"))
            await asyncio.wait_for(stopping.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            finish_stop.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assert_released_once(state)
            state.create.assert_awaited_once()

    async def test_remote_stream_retry_create_cancel_closes_each_acquired_slot(self):
        with self.mocks("remote") as state:
            creating_retry = asyncio.Event()
            retry_client = SimpleNamespace(aclose=AsyncMock())
            state.factory.side_effect = [state.client, retry_client]
            calls = 0

            async def create(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return "session-1", "message-1"
                creating_retry.set()
                await asyncio.Event().wait()

            async def translate(*_args, **_kwargs):
                raise main_module.EmptyUpstreamResponse("empty", retryable=True)
                yield "unreachable"

            state.create.side_effect = create
            with patch.object(main_module, "translate_web_events", translate):
                response = await self.run_session("remote", stream=True, model="glm-5.3")
                task = asyncio.create_task(anext(response.body_iterator))
                await asyncio.wait_for(creating_retry.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await response.body_iterator.aclose()
            state.client.aclose.assert_awaited_once()
            retry_client.aclose.assert_awaited_once()
            self.assertEqual(state.acquire.await_count, 2)
            self.assertEqual(state.release.call_count, 2)
            state.stop.assert_awaited_once()

    async def test_public_stream_close_cancels_and_drains_pending_dispatch(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                creating = asyncio.Event()
                create_exited = asyncio.Event()

                async def create(*_args, **_kwargs):
                    creating.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        create_exited.set()

                async def dispatch(*_args, **_kwargs):
                    return await self.run_session(path, stream=True)

                state.create.side_effect = create
                with (
                    patch.object(main_module, "_dispatch_chat", dispatch),
                    patch.object(main_module, "_apply_auto_route", return_value=self.options),
                ):
                    stream = main_module._deferred_dispatch_stream(
                        self.messages, "work", self.options
                    )
                    first = await anext(stream)
                    self.assertIn("chat.completion.chunk", first)
                    await asyncio.wait_for(creating.wait(), timeout=1)
                    await stream.aclose()
                self.assertTrue(create_exited.is_set())
                self.assert_released_once(state, stop=False)

    async def test_public_stream_cancel_before_first_frame_drains_dispatch(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                consume_task = None

                async def create(*_args, **_kwargs):
                    consume_task.cancel()
                    await asyncio.Event().wait()

                async def dispatch(*_args, **_kwargs):
                    return await self.run_session(path, stream=True)

                state.create.side_effect = create
                with (
                    patch.object(main_module, "_dispatch_chat", dispatch),
                    patch.object(main_module, "_apply_auto_route", return_value=self.options),
                ):
                    stream = main_module._deferred_dispatch_stream(
                        self.messages, "work", self.options
                    )
                    consume_task = asyncio.create_task(anext(stream))
                    with self.assertRaises(asyncio.CancelledError):
                        await consume_task
                    await stream.aclose()
                self.assert_released_once(state, stop=False)

    async def test_public_stream_closes_completed_but_unclaimed_response(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                finish_create = asyncio.Event()
                response_ready = asyncio.Event()

                async def create(*_args, **_kwargs):
                    await finish_create.wait()
                    return "session-1", "message-1"

                async def dispatch(*_args, **_kwargs):
                    response = await self.run_session(path, stream=True)
                    response_ready.set()
                    return response

                state.create.side_effect = create
                with (
                    patch.object(main_module, "_dispatch_chat", dispatch),
                    patch.object(main_module, "_apply_auto_route", return_value=self.options),
                ):
                    stream = main_module._deferred_dispatch_stream(
                        self.messages, "work", self.options
                    )
                    await anext(stream)
                    finish_create.set()
                    await asyncio.wait_for(response_ready.wait(), timeout=1)
                    state.client.aclose.assert_not_awaited()
                    await stream.aclose()
                self.assert_released_once(state)

    async def test_public_stream_active_response_close_releases_resources_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                async def dispatch(*_args, **_kwargs):
                    return await self.run_session(path, stream=True)

                with (
                    patch.object(main_module, "_dispatch_chat", dispatch),
                    patch.object(main_module, "_apply_auto_route", return_value=self.options),
                ):
                    stream = main_module._deferred_dispatch_stream(
                        self.messages, "work", self.options
                    )
                    self.assertEqual(await anext(stream), "data: first\n\n")
                    await stream.aclose()
                    await stream.aclose()
                self.assert_released_once(state)

    async def test_public_stream_repeated_cancel_waits_for_dispatch_cleanup(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                creating = asyncio.Event()
                closing = asyncio.Event()
                finish_close = asyncio.Event()

                async def create(*_args, **_kwargs):
                    creating.set()
                    await asyncio.Event().wait()

                async def close():
                    closing.set()
                    await finish_close.wait()

                async def dispatch(*_args, **_kwargs):
                    return await self.run_session(path, stream=True)

                state.create.side_effect = create
                state.client.aclose.side_effect = close
                with (
                    patch.object(main_module, "_dispatch_chat", dispatch),
                    patch.object(main_module, "_apply_auto_route", return_value=self.options),
                ):
                    stream = main_module._deferred_dispatch_stream(
                        self.messages, "work", self.options
                    )
                    await anext(stream)
                    await asyncio.wait_for(creating.wait(), timeout=1)
                    close_task = asyncio.create_task(stream.aclose())
                    await asyncio.wait_for(closing.wait(), timeout=1)
                    close_task.cancel()
                    await asyncio.sleep(0)
                    close_task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(close_task.done())
                    finish_close.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await close_task
                self.assert_released_once(state, stop=False)

    async def test_public_stream_close_retrieves_unclaimed_dispatch_failure(self):
        with self.mocks("remote") as state:
            finish_create = asyncio.Event()
            dispatch_exited = asyncio.Event()

            async def create(*_args, **_kwargs):
                await finish_create.wait()
                raise RuntimeError("create failed")

            async def dispatch(*_args, **_kwargs):
                try:
                    return await self.run_session("remote", stream=True)
                finally:
                    dispatch_exited.set()

            state.create.side_effect = create
            with (
                patch.object(main_module, "_dispatch_chat", dispatch),
                patch.object(main_module, "_apply_auto_route", return_value=self.options),
            ):
                stream = main_module._deferred_dispatch_stream(
                    self.messages, "work", self.options
                )
                await anext(stream)
                finish_create.set()
                await asyncio.wait_for(dispatch_exited.wait(), timeout=1)
                await stream.aclose()
            self.assert_released_once(state, stop=False)

    async def test_public_stream_does_not_reclaim_closed_response_after_fallback(self):
        class FailedIterator:
            aclose = AsyncMock()

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("initial stream failed")

        initial = SimpleNamespace(
            status_code=200, body_iterator=FailedIterator(), close=Mock()
        )
        dispatch = AsyncMock(side_effect=[initial, RuntimeError("fallback failed")])
        with (
            patch.object(main_module, "_dispatch_chat", dispatch),
            patch.object(main_module, "_apply_auto_route", return_value={"_upstream_mode": "ide"}),
        ):
            stream = main_module._deferred_dispatch_stream(self.messages, "work", {})
            chunks = [chunk async for chunk in stream]
        self.assertTrue(any("initial stream failed" in chunk for chunk in chunks))
        self.assertEqual(dispatch.await_count, 2)
        initial.body_iterator.aclose.assert_awaited_once()
        initial.close.assert_called_once()

    async def test_public_stream_closes_error_response_before_opening_fallback(self):
        class EmptyIterator:
            def __init__(self):
                self.aclose = AsyncMock()

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        initial = SimpleNamespace(
            status_code=502,
            body=b'{"error":{"message":"initial failed"}}',
            body_iterator=EmptyIterator(),
            close=Mock(),
        )
        fallback = SimpleNamespace(
            status_code=200, body_iterator=EmptyIterator(), close=Mock()
        )
        calls = 0

        async def dispatch(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return initial
            initial.close.assert_called_once()
            return fallback

        with (
            patch.object(main_module, "_dispatch_chat", dispatch),
            patch.object(
                main_module, "_apply_auto_route", return_value={"_upstream_mode": "ide"}
            ),
        ):
            stream = main_module._deferred_dispatch_stream(self.messages, "work", {})
            async for _chunk in stream:
                pass
        initial.close.assert_called_once()
        fallback.close.assert_called_once()
        initial.body_iterator.aclose.assert_awaited_once()
        fallback.body_iterator.aclose.assert_awaited_once()


class ChatResponseOwnershipTests(unittest.IsolatedAsyncioTestCase):
    paths = ("ide", "raw", "native")
    messages = [{"role": "user", "content": "hello"}]
    options = {"_account_id": "account-1", "_auth_token": "jwt-token"}

    @contextmanager
    def mocks(self, path):
        owned = SimpleNamespace(response=object(), close=Mock(), auth_token="")
        state = SimpleNamespace(
            owned=owned,
            send=AsyncMock(return_value=owned),
            collect=AsyncMock(return_value={"choices": []}),
        )
        transport, name = {
            "ide": (main_module.trae_client, "send_chat_request"),
            "raw": (main_module.raw_client, "send_raw_chat_request"),
            "native": (main_module.traework_native_bridge, "send_native_chat_request"),
        }[path]
        with (
            patch.object(transport, name, state.send),
            patch.object(main_module, "translate_ide_stream", _translated_events),
            patch.object(main_module, "collect_nonstream_ide", state.collect),
            patch.object(main_module, "_bind_usage_turn_from_metadata"),
            patch.object(main_module, "_track_usage_from_chunk"),
            patch.object(main_module, "_track_usage_from_result"),
            patch.object(main_module, "_capture_chat_session_auth"),
        ):
            yield state

    async def run_chat(self, path, stream=True):
        run = {
            "ide": main_module.run_ide_chat,
            "raw": main_module.run_raw_chat,
            "native": main_module.run_traework_native_chat,
        }[path]
        return await run(self.messages, "work", stream, self.options)

    async def test_translation_setup_failure_does_not_open_response(self):
        for path in self.paths:
            with (
                self.subTest(path=path),
                self.mocks(path) as state,
                patch.object(
                    main_module,
                    "_tool_translation_options",
                    side_effect=RuntimeError("setup failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    await self.run_chat(path)
                state.send.assert_not_awaited()
                state.owned.close.assert_not_called()

    async def test_stream_response_constructor_failure_closes_owned_response(self):
        for path in self.paths:
            with (
                self.subTest(path=path),
                self.mocks(path) as state,
                patch.object(
                    main_module, "StreamingResponse", side_effect=RuntimeError("response failed")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "response failed"):
                    await self.run_chat(path)
                state.owned.close.assert_called_once()

    async def test_unstarted_stream_close_releases_response_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_chat(path)
                state.owned.close.assert_not_called()
                await response.body_iterator.aclose()
                await response.body_iterator.aclose()
                state.owned.close.assert_called_once()

    async def test_active_stream_close_releases_response_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_chat(path)
                self.assertEqual(await anext(response.body_iterator), "data: first\n\n")
                await response.body_iterator.aclose()
                await response.body_iterator.aclose()
                state.owned.close.assert_called_once()

    async def test_exhausted_stream_closes_response_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_chat(path)
                chunks = [chunk async for chunk in response.body_iterator]
                self.assertEqual(chunks, ["data: first\n\n", "data: [DONE]\n\n"])
                await response.body_iterator.aclose()
                state.owned.close.assert_called_once()

    async def test_nonstream_success_closes_response_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                response = await self.run_chat(path, stream=False)
                self.assertEqual(response.status_code, 200)
                state.owned.close.assert_called_once()

    async def test_nonstream_failure_closes_response_once(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                state.collect.side_effect = RuntimeError("translation failed")
                with self.assertRaisesRegex(RuntimeError, "translation failed"):
                    await self.run_chat(path, stream=False)
                state.owned.close.assert_called_once()

    async def test_stream_cancel_preserved_when_sync_close_fails(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                reading = asyncio.Event()

                async def translate(*_args, **_kwargs):
                    reading.set()
                    await asyncio.Event().wait()
                    yield "unreachable"

                state.owned.close.side_effect = RuntimeError("close failed")
                with patch.object(main_module, "translate_ide_stream", translate):
                    response = await self.run_chat(path)
                    task = asyncio.create_task(anext(response.body_iterator))
                    await asyncio.wait_for(reading.wait(), timeout=1)
                    task.cancel("client disconnected")
                    with self.assertRaises(asyncio.CancelledError) as cancelled:
                        await task
                    self.assertEqual(cancelled.exception.args, ("client disconnected",))
                state.owned.close.assert_called_once()

    async def test_public_stream_closes_unclaimed_sync_response(self):
        for path in self.paths:
            with self.subTest(path=path), self.mocks(path) as state:
                finish_send = asyncio.Event()
                response_ready = asyncio.Event()

                async def send(*_args, **_kwargs):
                    await finish_send.wait()
                    return state.owned

                async def dispatch(*_args, **_kwargs):
                    response = await self.run_chat(path)
                    response_ready.set()
                    return response

                state.send.side_effect = send
                with (
                    patch.object(main_module, "_dispatch_chat", dispatch),
                    patch.object(main_module, "_apply_auto_route", return_value=self.options),
                ):
                    stream = main_module._deferred_dispatch_stream(
                        self.messages, "work", self.options
                    )
                    await anext(stream)
                    finish_send.set()
                    await asyncio.wait_for(response_ready.wait(), timeout=1)
                    await stream.aclose()
                state.owned.close.assert_called_once()

    async def test_raw_retry_closes_each_owned_response_once(self):
        with self.mocks("raw") as state:
            retry = SimpleNamespace(response=object(), close=Mock(), auth_token="")
            state.send.side_effect = [state.owned, retry]

            async def translate(response, *_args, **_kwargs):
                if response is state.owned.response:
                    raise main_module.EmptyUpstreamResponse("empty", retryable=True)
                yield "data: first\n\n"

            with patch.object(main_module, "translate_ide_stream", translate):
                response = await self.run_chat("raw")
                self.assertEqual(await anext(response.body_iterator), "data: first\n\n")
                state.owned.close.assert_called_once()
                retry.close.assert_not_called()
                await response.body_iterator.aclose()
                await response.body_iterator.aclose()
            state.owned.close.assert_called_once()
            retry.close.assert_called_once()

    async def test_raw_retry_send_failure_does_not_reclose_old_response(self):
        with self.mocks("raw") as state:
            state.send.side_effect = [state.owned, RuntimeError("retry failed")]

            async def translate(*_args, **_kwargs):
                raise main_module.EmptyUpstreamResponse("empty", retryable=True)
                yield "unreachable"

            with patch.object(main_module, "translate_ide_stream", translate):
                response = await self.run_chat("raw")
                with self.assertRaisesRegex(RuntimeError, "retry failed"):
                    await anext(response.body_iterator)
                await response.body_iterator.aclose()
            state.owned.close.assert_called_once()

    async def test_raw_retry_setup_failure_closes_new_response(self):
        for stream in (True, False):
            with self.subTest(stream=stream), self.mocks("raw") as state:
                retry = SimpleNamespace(response=object(), close=Mock(), auth_token="")
                state.send.side_effect = [state.owned, retry]
                state.collect.side_effect = main_module.EmptyUpstreamResponse(
                    "empty", retryable=True
                )

                async def translate(*_args, **_kwargs):
                    raise main_module.EmptyUpstreamResponse("empty", retryable=True)
                    yield "unreachable"

                with (
                    patch.object(main_module, "translate_ide_stream", translate),
                    patch.object(
                        main_module,
                        "_capture_chat_session_auth",
                        side_effect=[None, RuntimeError("retry setup failed")],
                    ),
                ):
                    if stream:
                        response = await self.run_chat("raw")
                        with self.assertRaisesRegex(RuntimeError, "retry setup failed"):
                            await anext(response.body_iterator)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "retry setup failed"):
                            await self.run_chat("raw", stream=False)
                state.owned.close.assert_called_once()
                retry.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
