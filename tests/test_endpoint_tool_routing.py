"""Routing contracts for explicit Trae endpoint modes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src import main as main_module
from src.sse import (
    EmptyUpstreamResponse,
    InvalidNativeToolArguments,
    RepeatedCompletedToolResponse,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the caller workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _request_options() -> dict:
    return {
        "tools": TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "client_context": {"workspace_path": r"C:\\workspace"},
    }


class ExplicitEndpointToolRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_are_passed_to_the_selected_raw_ide_and_work_routes(self):
        routes = {
            "raw": "run_raw_chat",
            "ide": "run_ide_chat",
            "solo": "run_ide_chat",
            "traework-native": "run_traework_native_chat",
        }

        for mode, route_name in routes.items():
            with self.subTest(mode=mode):
                selected = AsyncMock(return_value={"route": mode})
                remote = AsyncMock(
                    side_effect=AssertionError(
                        f"{mode} tool request must not be served by remote"
                    )
                )
                options = _request_options()

                with (
                    patch.object(main_module, "UPSTREAM_MODE", mode),
                    patch.object(main_module, "_remote_only_models", return_value=set()),
                    patch.object(main_module, route_name, selected),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                ):
                    result = await main_module._dispatch_chat(
                        [{"role": "user", "content": "Read README.md"}],
                        "auto",
                        False,
                        options,
                    )

                self.assertEqual(result, {"route": mode})
                selected.assert_awaited_once()
                called_options = selected.await_args.args[3]
                self.assertEqual(called_options["tools"], TOOLS)
                self.assertEqual(called_options["tool_choice"], "auto")
                self.assertFalse(called_options["parallel_tool_calls"])
                if mode == "solo":
                    self.assertEqual(
                        called_options["_ide_endpoint"],
                        "/api/agent/v3/llm_utils_chat",
                    )
                remote.assert_not_awaited()

    async def test_explicit_ide_and_native_failures_fallback_to_remote(self):
        routes = {
            "ide": "run_ide_chat",
            "traework-native": "run_traework_native_chat",
        }

        for mode, route_name in routes.items():
            with self.subTest(mode=mode):
                selected = AsyncMock(side_effect=RuntimeError(f"{mode} unavailable"))
                remote = AsyncMock(return_value={"route": "remote"})

                with (
                    patch.object(main_module, "UPSTREAM_MODE", mode),
                    patch.object(main_module, "_remote_only_models", return_value=set()),
                    patch.object(main_module, route_name, selected),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                ):
                    response = await main_module._dispatch_chat(
                        [{"role": "user", "content": "Read README.md"}],
                        "auto",
                        False,
                        _request_options(),
                    )

                self.assertEqual(response, {"route": "remote"})
                selected.assert_awaited_once()
                remote.assert_awaited_once()
                fallback_options = remote.await_args.args[3]
                self.assertEqual(fallback_options["_upstream_mode"], "remote")
                self.assertEqual(fallback_options["_upstream_fallback_from"], mode)
                self.assertEqual(
                    fallback_options["_remote_agent_type"],
                    "solo_work_remote"
                    if mode == "traework-native"
                    else "solo_agent_remote",
                )

    async def test_raw_app_config_error_uses_ide_before_remote(self):
        raw = AsyncMock(side_effect=RuntimeError("app config: record not found"))
        ide = AsyncMock(return_value={"route": "ide"})
        remote = AsyncMock(side_effect=AssertionError("IDE path was available"))
        trace = {}
        with (
            patch.object(main_module, "UPSTREAM_MODE", "raw"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "run_raw_chat", raw),
            patch.object(main_module, "run_ide_chat", ide),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "glm-5.3",
                False,
                {**_request_options(), "_upstream_trace": trace},
            )
        self.assertEqual(result, {"route": "ide"})
        self.assertEqual(ide.await_args.args[3]["_ide_endpoint"], "/api/agent/v3/llm_utils_chat")
        self.assertEqual(ide.await_args.args[3]["tools"], TOOLS)
        self.assertEqual(trace["actual_endpoint"], "ide")
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["fallback_target"], "ide")
        self.assertIn("raw:", trace["fallback_reason"])
        self.assertEqual(trace["failed_endpoint"], "raw")
        remote.assert_not_awaited()

    async def test_cached_raw_app_config_skip_reports_fallback_diagnostic(self):
        raw = AsyncMock(side_effect=AssertionError("cached raw path must be skipped"))
        ide = AsyncMock(return_value={"route": "ide"})
        remote = AsyncMock(side_effect=AssertionError("IDE path was available"))
        trace = {}
        with (
            patch.object(main_module, "UPSTREAM_MODE", "raw"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "_raw_app_config_known_missing", return_value=True),
            patch.object(main_module, "run_raw_chat", raw),
            patch.object(main_module, "run_ide_chat", ide),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "glm-5.3",
                False,
                {**_request_options(), "_upstream_trace": trace},
            )

        self.assertEqual(result, {"route": "ide"})
        raw.assert_not_awaited()
        remote.assert_not_awaited()
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["fallback_target"], "ide")
        self.assertEqual(trace["failed_endpoint"], "raw")
        self.assertIn("app config record not found", trace["fallback_reason"])

    async def test_raw_falls_back_to_remote_after_ide_error(self):
        raw = AsyncMock(side_effect=RuntimeError("app config missing"))
        ide = AsyncMock(side_effect=RuntimeError("IDE unavailable"))
        remote = AsyncMock(return_value={"route": "remote"})
        with (
            patch.object(main_module, "UPSTREAM_MODE", "raw"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "run_raw_chat", raw),
            patch.object(main_module, "run_ide_chat", ide),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "auto",
                False,
                _request_options(),
            )
        self.assertEqual(result, {"route": "remote"})
        self.assertEqual(remote.await_args.args[3]["_upstream_fallback_from"], "raw")

    async def test_work_agent_uses_bound_work_remote_executor(self):
        remote = AsyncMock(return_value={"route": "work"})
        ide = AsyncMock(side_effect=AssertionError("legacy Work endpoint must not run"))
        trace = {}
        with (
            patch.object(main_module, "UPSTREAM_MODE", "work-agent"),
            patch.object(main_module, "_remote_only_models", return_value={"*"}),
            patch.object(main_module, "run_ide_chat", ide),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "glm-5.3",
                False,
                {**_request_options(), "_upstream_trace": trace,
                 "_disable_upstream_fallback": True},
            )
        self.assertEqual(result, {"route": "work"})
        work_options = remote.await_args.args[3]
        self.assertEqual(work_options["_trae_mode"], "work")
        self.assertEqual(work_options["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(work_options["tools"], TOOLS)
        self.assertEqual(trace["actual_endpoint"], "remote-work")
        self.assertFalse(trace["fallback_used"])
        ide.assert_not_awaited()

    async def test_work_agent_fallback_switches_to_ordinary_remote_tier(self):
        """A Work failure must not issue a second Work-tier request as fallback."""

        work_failure = RuntimeError("work model unavailable")
        remote = AsyncMock(return_value={"route": "remote"})
        trace = {}
        with (
            patch.object(main_module, "UPSTREAM_MODE", "work-agent"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(
                main_module,
                "_run_remote_with_retry",
                AsyncMock(side_effect=[work_failure, {"route": "remote"}]),
            ) as dispatch_remote,
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Use the workspace tool"}],
                "auto",
                False,
                {**_request_options(), "_upstream_trace": trace},
            )

        self.assertEqual(result, {"route": "remote"})
        self.assertEqual(dispatch_remote.await_count, 2)
        work_options = dispatch_remote.await_args_list[0].args[3]
        fallback_options = dispatch_remote.await_args_list[1].args[3]
        self.assertEqual(work_options["_trae_mode"], "work")
        self.assertEqual(work_options["_remote_agent_type"], "solo_work_remote")
        self.assertNotIn("_trae_mode", fallback_options)
        self.assertEqual(fallback_options["_remote_agent_type"], "solo_agent_remote")
        self.assertEqual(fallback_options["_upstream_fallback_from"], "work-agent")
        self.assertEqual(trace["requested_endpoint"], "work-agent")
        self.assertEqual(trace["actual_endpoint"], "remote")
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["fallback_target"], "remote")
        self.assertEqual(trace["failed_endpoint"], "work-agent")
        self.assertIn("work-agent: work model unavailable", trace["fallback_reason"])

    async def test_tool_request_falls_back_without_dropping_tool_schema(self):
        selected = AsyncMock(side_effect=RuntimeError("ide unavailable"))
        remote = AsyncMock(return_value={"route": "remote"})
        with (
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "auto",
                False,
                _request_options(),
            )

        self.assertEqual(result, {"route": "remote"})
        fallback_options = remote.await_args.args[3]
        self.assertEqual(fallback_options["tools"], TOOLS)
        self.assertEqual(fallback_options["tool_choice"], "auto")
        self.assertEqual(fallback_options["_remote_agent_type"], "solo_agent_remote")

    async def test_explicit_disable_fallback_keeps_native_error(self):
        selected = AsyncMock(side_effect=RuntimeError("raw unavailable"))
        remote = AsyncMock(return_value={"route": "remote"})
        options = _request_options()
        options["_disable_upstream_fallback"] = True
        with (
            patch.object(main_module, "UPSTREAM_MODE", "raw"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "run_raw_chat", selected),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            response = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "auto",
                False,
                options,
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("raw unavailable", response.body.decode())
        remote.assert_not_awaited()

    async def test_streaming_native_empty_body_falls_back_before_public_error(self):
        class EmptyNativeResponse:
            status_code = 200

            @staticmethod
            async def _body():
                yield ": native-wait\n\n"
                raise RuntimeError("native stream disconnected")

            def __init__(self):
                self.body_iterator = self._body()

            def close(self):
                return None

        async def remote_body():
            yield 'data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
            yield "data: [DONE]\n\n"

        selected = AsyncMock(return_value=EmptyNativeResponse())
        remote_response = type(
            "RemoteResponse",
            (),
            {
                "status_code": 200,
                "body_iterator": remote_body(),
                "close": lambda self: None,
            },
        )()
        remote = AsyncMock(return_value=remote_response)
        options = _request_options()
        options["_upstream_mode"] = "ide"

        with (
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            chunks = [
                chunk
                async for chunk in main_module._deferred_dispatch_stream(
                    [{"role": "user", "content": "Reply pong"}],
                    "auto",
                    options,
                )
            ]

        joined = "".join(chunks)
        self.assertIn('"content":"pong"', joined)
        self.assertIn("data: [DONE]", joined)
        remote.assert_awaited_once()
        fallback_options = remote.await_args.args[3]
        self.assertEqual(fallback_options["tools"], TOOLS)
        self.assertEqual(fallback_options["_remote_agent_type"], "solo_agent_remote")

    async def test_nonstream_model_event_error_does_not_replay_other_routes(self):
        routes = {
            "raw": "run_raw_chat",
            "ide": "run_ide_chat",
            "traework-native": "run_traework_native_chat",
        }
        for mode, route_name in routes.items():
            for retryable, observed in ((False, True), (False, False), (True, True)):
                with self.subTest(mode=mode, retryable=retryable, observed=observed):
                    exc = InvalidNativeToolArguments(
                        "invalid native tool arguments",
                        retryable=retryable,
                        observed_model_event=observed,
                        usage={"prompt_tokens": 10, "completion_tokens": 3},
                    )
                    selected = AsyncMock(side_effect=exc)
                    remote = AsyncMock(return_value={"route": "remote"})
                    ide = AsyncMock(return_value={"route": "ide"})
                    with (
                        patch.object(main_module, "UPSTREAM_MODE", mode),
                        patch.object(main_module, "_remote_only_models", return_value=set()),
                        patch.object(main_module, "_raw_app_config_known_missing", return_value=False),
                        patch.object(main_module, "run_ide_chat", ide),
                        patch.object(main_module, route_name, selected),
                        patch.object(main_module, "_run_remote_with_retry", remote),
                        patch.object(main_module, "_track_usage_from_result") as usage,
                    ):
                        response = await main_module._dispatch_chat(
                            [{"role": "user", "content": "Read README.md"}],
                            "auto",
                            False,
                            _request_options(),
                        )
                    self.assertEqual(response.status_code, 502)
                    self.assertIn("invalid native tool arguments", response.body.decode())
                    selected.assert_awaited_once()
                    remote.assert_not_awaited()
                    if mode == "raw":
                        ide.assert_not_awaited()
                    usage.assert_called_once_with({"usage": exc.usage}, "auto")

    async def test_retryable_truly_empty_nonstream_still_falls_back(self):
        for mode, route_name in {
            "raw": "run_raw_chat",
            "ide": "run_ide_chat",
            "traework-native": "run_traework_native_chat",
        }.items():
            with self.subTest(mode=mode):
                exc = EmptyUpstreamResponse(
                    "truly empty",
                    retryable=True,
                    observed_model_event=False,
                    usage={"prompt_tokens": 10},
                )
                selected = AsyncMock(side_effect=exc)
                fallback = AsyncMock(return_value={"route": "fallback"})
                with (
                    patch.object(main_module, "UPSTREAM_MODE", mode),
                    patch.object(main_module, "_remote_only_models", return_value=set()),
                    patch.object(main_module, "_raw_app_config_known_missing", return_value=False),
                    patch.object(main_module, "run_ide_chat", fallback),
                    patch.object(main_module, route_name, selected),
                    patch.object(main_module, "_run_remote_with_retry", fallback),
                    patch.object(main_module, "_track_usage_from_result") as usage,
                ):
                    response = await main_module._dispatch_chat(
                        [{"role": "user", "content": "Read README.md"}],
                        "auto",
                        False,
                        _request_options(),
                    )
                self.assertEqual(response, {"route": "fallback"})
                selected.assert_awaited_once()
                fallback.assert_awaited_once()
                usage.assert_called_once_with({"usage": exc.usage}, "auto")

    async def test_stream_model_event_error_does_not_fallback_or_double_count_usage(self):
        for creation_error in (False, True):
            for retryable, observed in ((False, True), (False, False), (True, True)):
                with self.subTest(
                    creation_error=creation_error, retryable=retryable, observed=observed
                ):
                    exc = InvalidNativeToolArguments(
                        "invalid native tool arguments",
                        retryable=retryable,
                        observed_model_event=observed,
                        usage={"prompt_tokens": 10, "completion_tokens": 3},
                    )

                    async def invalid_body():
                        raise exc
                        yield "unreachable"

                    selected = AsyncMock(
                        side_effect=exc if creation_error else None,
                        return_value=main_module.StreamingResponse(invalid_body()),
                    )
                    remote = AsyncMock(return_value={"route": "remote"})
                    with (
                        patch.object(main_module, "UPSTREAM_MODE", "ide"),
                        patch.object(main_module, "_remote_only_models", return_value=set()),
                        patch.object(main_module, "run_ide_chat", selected),
                        patch.object(main_module, "_run_remote_with_retry", remote),
                        patch.object(main_module, "_track_usage_from_result") as usage,
                    ):
                        chunks = [
                            chunk
                            async for chunk in main_module._deferred_dispatch_stream(
                                [{"role": "user", "content": "Read README.md"}],
                                "auto",
                                {**_request_options(), "_upstream_mode": "ide"},
                            )
                        ]
                    self.assertIn("invalid native tool arguments", "".join(chunks))
                    selected.assert_awaited_once()
                    remote.assert_not_awaited()
                    usage.assert_called_once_with({"usage": exc.usage}, "auto")

    async def test_retryable_truly_empty_stream_still_falls_back(self):
        exc = EmptyUpstreamResponse(
            "truly empty",
            retryable=True,
            observed_model_event=False,
            usage={"prompt_tokens": 10},
        )

        async def empty_body():
            raise exc
            yield "unreachable"

        async def remote_body():
            yield 'data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
            yield "data: [DONE]\n\n"

        selected = AsyncMock(return_value=main_module.StreamingResponse(empty_body()))
        remote = AsyncMock(return_value=main_module.StreamingResponse(remote_body()))
        with (
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(main_module, "_run_remote_with_retry", remote),
            patch.object(main_module, "_track_usage_from_result") as usage,
        ):
            chunks = [
                chunk
                async for chunk in main_module._deferred_dispatch_stream(
                    [{"role": "user", "content": "Read README.md"}],
                    "auto",
                    {**_request_options(), "_upstream_mode": "ide"},
                )
            ]
        self.assertIn('"content":"pong"', "".join(chunks))
        selected.assert_awaited_once()
        remote.assert_awaited_once()
        usage.assert_called_once_with({"usage": exc.usage}, "auto")

    async def test_transport_and_router_record_same_error_usage_only_once(self):
        exc = InvalidNativeToolArguments(
            "invalid native tool arguments",
            retryable=False,
            observed_model_event=True,
            usage={"prompt_tokens": 10},
        )

        async def selected(*_args, **_kwargs):
            main_module._track_usage_from_exception(exc, "auto")
            raise exc

        remote = AsyncMock()
        with (
            patch.object(main_module, "UPSTREAM_MODE", "raw"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "_raw_app_config_known_missing", return_value=False),
            patch.object(main_module, "run_raw_chat", selected),
            patch.object(main_module, "_run_remote_with_retry", remote),
            patch.object(main_module, "_track_usage_from_result") as usage,
        ):
            response = await main_module._dispatch_chat(
                [{"role": "user", "content": "Read README.md"}],
                "auto",
                False,
                _request_options(),
            )
        self.assertEqual(response.status_code, 502)
        remote.assert_not_awaited()
        usage.assert_called_once_with({"usage": exc.usage}, "auto")

    async def test_actual_raw_repeated_completed_tool_error_never_replays_routes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                exc = RepeatedCompletedToolResponse(
                    "repeated completed tool",
                    retryable=False,
                    observed_model_event=True,
                    usage={"prompt_tokens": 10, "completion_tokens": 3},
                )
                owned = SimpleNamespace(response=object(), close=Mock(), auth_token="")
                send = AsyncMock(return_value=owned)
                ide = AsyncMock(return_value={"route": "ide"})
                remote = AsyncMock(return_value={"route": "remote"})

                async def translate(*_args, **_kwargs):
                    raise exc
                    yield "unreachable"

                with (
                    patch.object(main_module, "UPSTREAM_MODE", "raw"),
                    patch.object(main_module, "_remote_only_models", return_value=set()),
                    patch.object(main_module, "_raw_app_config_known_missing", return_value=False),
                    patch.object(main_module.raw_client, "send_raw_chat_request", send),
                    patch.object(main_module, "translate_ide_stream", translate),
                    patch.object(
                        main_module, "collect_nonstream_ide", AsyncMock(side_effect=exc)
                    ),
                    patch.object(main_module, "_capture_chat_session_auth"),
                    patch.object(main_module, "_bind_usage_turn_from_metadata"),
                    patch.object(main_module, "run_ide_chat", ide),
                    patch.object(main_module, "_run_remote_with_retry", remote),
                    patch.object(main_module, "_track_usage_from_result") as usage,
                ):
                    if stream:
                        chunks = [
                            chunk
                            async for chunk in main_module._deferred_dispatch_stream(
                                [{"role": "user", "content": "Read README.md"}],
                                "auto",
                                {**_request_options(), "_upstream_mode": "raw"},
                            )
                        ]
                        self.assertIn("repeated completed tool", "".join(chunks))
                    else:
                        response = await main_module._dispatch_chat(
                            [{"role": "user", "content": "Read README.md"}],
                            "auto",
                            False,
                            _request_options(),
                        )
                        self.assertEqual(response.status_code, 502)
                        self.assertIn("repeated completed tool", response.body.decode())
                send.assert_awaited_once()
                owned.close.assert_called_once()
                ide.assert_not_awaited()
                remote.assert_not_awaited()
                usage.assert_called_once_with({"usage": exc.usage}, "auto")


if __name__ == "__main__":
    unittest.main()
