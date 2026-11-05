"""Routing contracts for explicit Trae endpoint modes."""

from __future__ import annotations

import json
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
                options["_upstream_mode"] = mode

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
                if mode in {"ide", "solo"}:
                    self.assertTrue(called_options["_traecode_native_tools"])
                remote.assert_not_awaited()

    async def test_ide_and_solo_dispatch_preserve_native_tool_round_trip(self):
        messages = [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-readme",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call-readme", "content": "Project readme"},
        ]
        for mode in ("ide", "solo"):
            with self.subTest(mode=mode):
                selected = AsyncMock(return_value={"route": mode})
                with (
                    patch.object(main_module, "UPSTREAM_MODE", mode),
                    patch.object(main_module, "_remote_only_models", return_value=set()),
                    patch.object(main_module, "run_ide_chat", selected),
                ):
                    await main_module._dispatch_chat(
                        messages, "glm-5.3", False, _request_options()
                    )
                body = main_module.raw_client.build_traecode_chat_body(
                    messages, "glm-5.3", options=selected.await_args.args[3]
                )
                self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
                self.assertEqual(body["tool_choice"], "auto")
                self.assertFalse(body["parallel_tool_calls"])
                self.assertEqual(body["messages"][1]["tool_calls"][0]["id"], "call-readme")
                self.assertEqual(body["messages"][2]["tool_call_id"], "call-readme")
                self.assertEqual(body["messages"][2]["role"], "tool")

    async def test_tool_failures_do_not_downgrade_to_remote(self):
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

                self.assertEqual(response.status_code, 502)
                selected.assert_awaited_once()
                remote.assert_not_awaited()

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
                {**_request_options(), "_upstream_mode": "raw", "_upstream_trace": trace},
            )
        self.assertEqual(result, {"route": "ide"})
        self.assertEqual(
            ide.await_args.args[3]["_ide_endpoint"],
            "/api/agent/v3/llm_utils_chat",
        )
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
                {**_request_options(), "_upstream_mode": "raw", "_upstream_trace": trace},
            )

        self.assertEqual(result, {"route": "ide"})
        raw.assert_not_awaited()
        remote.assert_not_awaited()
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["fallback_target"], "ide")
        self.assertEqual(trace["failed_endpoint"], "raw")
        self.assertIn("app config record not found", trace["fallback_reason"])

    async def test_raw_tools_report_failure_after_ide_error(self):
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
                {**_request_options(), "_upstream_mode": "raw"},
            )
        self.assertEqual(result.status_code, 502)
        remote.assert_not_awaited()

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
                {"_upstream_trace": trace,
                 "_disable_upstream_fallback": True},
            )
        self.assertEqual(result, {"route": "work"})
        remote.assert_awaited_once()
        work_options = remote.await_args.args[3]
        self.assertEqual(work_options["_trae_mode"], "work")
        self.assertEqual(work_options["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(work_options["_session_variant"], "work")
        self.assertEqual(trace["actual_endpoint"], "remote-work")
        self.assertEqual(trace["product"], "traework")
        self.assertFalse(trace["fallback_used"])
        ide.assert_not_awaited()

    async def test_public_work_preset_uses_work_tier_not_raw_or_agent(self):
        remote = AsyncMock(return_value={"route": "work"})
        with (
            patch.object(main_module, "UPSTREAM_MODE", "work"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Reply pong"}],
                "glm-5.3",
                False,
                {"_disable_upstream_fallback": True},
            )

        self.assertEqual(result, {"route": "work"})
        options = remote.await_args.args[3]
        self.assertEqual(options["_trae_mode"], "work")
        self.assertEqual(options["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(options["_session_variant"], "work")

    async def test_work_agent_fallback_switches_to_ordinary_remote_tier(self):
        """A Work failure must not issue a second Work-tier request as fallback."""

        work_failure = RuntimeError("work model unavailable")
        remote = AsyncMock(side_effect=[work_failure, {"route": "remote"}])
        trace = {}
        with (
            patch.object(main_module, "UPSTREAM_MODE", "work-agent"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(
                main_module,
                "_run_remote_with_retry",
                remote,
            ) as dispatch_remote,
        ):
            result = await main_module._dispatch_chat(
                [{"role": "user", "content": "Use the workspace tool"}],
                "auto",
                False,
                {"_upstream_trace": trace},
            )

        self.assertEqual(result, {"route": "remote"})
        self.assertEqual(dispatch_remote.await_count, 2)
        work_options = dispatch_remote.await_args_list[0].args[3]
        fallback_options = dispatch_remote.await_args_list[1].args[3]
        self.assertEqual(work_options["_remote_agent_type"], "solo_work_remote")
        self.assertEqual(work_options["_trae_mode"], "work")
        self.assertNotIn("_trae_mode", fallback_options)
        self.assertNotIn("remote_agent_type", fallback_options)
        self.assertEqual(fallback_options["_remote_agent_type"], "solo_agent_remote")
        self.assertEqual(fallback_options["_upstream_fallback_from"], "work-agent")
        self.assertEqual(trace["requested_endpoint"], "work-agent")
        self.assertEqual(trace["actual_endpoint"], "remote")
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["fallback_target"], "remote")
        self.assertEqual(trace["failed_endpoint"], "work")
        self.assertIn("work-agent: work model unavailable", trace["fallback_reason"])

    async def test_work_fallback_options_remove_work_tier_binding(self):
        fallback = main_module._remote_fallback_options(
            {
                "_trae_mode": "work",
                "_remote_agent_type": "solo_work_remote",
                "remote_agent_type": "solo_work_remote",
            },
            "work",
        )

        self.assertEqual(fallback["_upstream_mode"], "remote")
        self.assertEqual(fallback["_remote_agent_type"], "solo_agent_remote")
        self.assertNotIn("_trae_mode", fallback)
        self.assertNotIn("remote_agent_type", fallback)

    async def test_tool_request_keeps_native_error(self):
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

        self.assertEqual(result.status_code, 502)
        self.assertEqual(selected.await_args.args[3]["tools"], TOOLS)
        remote.assert_not_awaited()

    async def test_explicit_disable_fallback_keeps_native_error(self):
        selected = AsyncMock(side_effect=RuntimeError("raw unavailable"))
        remote = AsyncMock(return_value={"route": "remote"})
        options = _request_options()
        options["_upstream_mode"] = "raw"
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

    async def test_streaming_native_tool_error_does_not_downgrade(self):
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

        selected = AsyncMock(return_value=EmptyNativeResponse())
        remote = AsyncMock(side_effect=AssertionError("caller tools require a native transport"))
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
        self.assertIn('native stream disconnected', joined)
        self.assertIn("data: [DONE]", joined)
        remote.assert_not_awaited()

    async def test_streaming_ide_body_failure_preserves_remote_query_limit_error(self):
        async def broken_ide_body():
            raise RuntimeError("IDE body failed before output")
            yield "unreachable"

        messages = [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,QUJD"},
            }],
        }]
        selected = AsyncMock(
            return_value=main_module.StreamingResponse(broken_ide_body())
        )
        remote = AsyncMock(
            side_effect=main_module.RemoteQueryTooLarge(2_895_151, 480_000)
        )
        with (
            patch.object(main_module, "UPSTREAM_MODE", "ide"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
            patch.object(main_module.auth, "get_polling_status", return_value={"enabled": False}),
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(main_module, "_run_remote_with_retry", remote),
        ):
            chunks = [
                chunk
                async for chunk in main_module._deferred_dispatch_stream(
                    messages, "glm-5.2", {"_upstream_mode": "ide"}
                )
            ]

        joined = "".join(chunks)
        events = [
            json.loads(line[6:])
            for line in joined.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        errors = [event["error"] for event in events if "error" in event]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["type"], "context_length_exceeded")
        self.assertEqual(errors[0]["param"], "messages")
        self.assertIn("configured input limit", errors[0]["message"])
        self.assertNotIn("IDE body failed", errors[0]["message"])
        self.assertEqual(joined.count("data: [DONE]"), 1)
        selected.assert_awaited_once()
        remote.assert_awaited_once()
        self.assertEqual(remote.await_args.args[1], "glm-5.2")
        self.assertEqual(remote.await_args.args[3]["_upstream_fallback_from"], "ide")

    async def test_streaming_initial_query_limit_error_does_not_dispatch_fallback(self):
        response = main_module.JSONResponse(
            {
                "error": {
                    "message": "Trae remote query exceeds the configured input limit",
                    "type": "context_length_exceeded",
                    "param": "messages",
                }
            },
            status_code=413,
        )
        dispatch = AsyncMock(return_value=response)
        with patch.object(main_module, "_dispatch_chat", dispatch):
            chunks = [
                chunk
                async for chunk in main_module._deferred_dispatch_stream(
                    [{"role": "user", "content": "oversized"}],
                    "glm-5.2",
                    {"_upstream_mode": "ide"},
                )
            ]

        joined = "".join(chunks)
        events = [
            json.loads(line[6:])
            for line in joined.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        errors = [event["error"] for event in events if "error" in event]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["type"], "context_length_exceeded")
        self.assertEqual(errors[0]["param"], "messages")
        self.assertEqual(joined.count("data: [DONE]"), 1)
        dispatch.assert_awaited_once()

    async def test_work_streaming_failure_falls_back_to_remote_before_output(self):
        async def broken_work_body():
            raise RuntimeError("work stream disconnected before model output")
            yield "unreachable"

        async def remote_body():
            yield 'data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
            yield "data: [DONE]\n\n"

        work_response = main_module.StreamingResponse(broken_work_body())
        remote_response = main_module.StreamingResponse(remote_body())
        remote = AsyncMock(side_effect=[work_response, remote_response])
        options = {"_upstream_mode": "work"}

        with (
            patch.object(main_module, "UPSTREAM_MODE", "work"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
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
        self.assertEqual(remote.await_count, 2)
        self.assertEqual(
            remote.await_args_list[0].args[3]["_remote_agent_type"],
            "solo_work_remote",
        )
        self.assertEqual(
            remote.await_args_list[1].args[3]["_upstream_fallback_from"], "work"
        )

    async def test_work_streaming_failure_respects_disabled_fallback(self):
        async def broken_work_body():
            raise RuntimeError("work stream disconnected before model output")
            yield "unreachable"

        remote = AsyncMock(
            return_value=main_module.StreamingResponse(broken_work_body())
        )
        options = {
            "_upstream_mode": "work",
            "_disable_upstream_fallback": True,
        }

        with (
            patch.object(main_module, "UPSTREAM_MODE", "work"),
            patch.object(main_module, "_remote_only_models", return_value=set()),
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

        self.assertIn("work stream disconnected before model output", "".join(chunks))
        remote.assert_awaited_once()

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
                            {**_request_options(), "_upstream_mode": mode},
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
                        {},
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
                    {"_upstream_mode": "ide"},
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
                {**_request_options(), "_upstream_mode": "raw"},
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
                            {**_request_options(), "_upstream_mode": "raw"},
                        )
                        self.assertEqual(response.status_code, 502)
                        self.assertIn("repeated completed tool", response.body.decode())
                send.assert_awaited_once()
                owned.close.assert_called_once()
                ide.assert_not_awaited()
                remote.assert_not_awaited()
                usage.assert_called_once_with({"usage": exc.usage}, "auto")


class NativeEndpointRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_ide_stream_rate_limit_rotates_before_first_output(self):
        async def limited_body():
            # Keepalive is transport noise and must not prevent a retry.
            yield ": relay-keepalive\n\n"
            raise RuntimeError(
                "Trae upstream error 4011: requests have exceeded the rate limit"
            )

        async def ok_body():
            yield 'data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
            yield "data: [DONE]\n\n"

        selected = AsyncMock(
            side_effect=[
                main_module.StreamingResponse(limited_body()),
                main_module.StreamingResponse(ok_body()),
            ]
        )
        tracker = SimpleNamespace(rebind=AsyncMock())
        options = {
            "_account_id": "account-a",
            "_auth_token": "token-a",
            "_upstream_mode": "ide",
        }
        next_snapshot = (
            "account-b",
            {
                "token": "token-b",
                "provider_specific": {"bizUserId": "account-b"},
            },
        )

        with (
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": True},
            ),
            patch.object(main_module, "_polling_retry_limit", return_value=2),
            patch.object(
                main_module,
                "_next_retry_account_snapshot",
                return_value=next_snapshot,
            ) as next_account,
            patch.object(main_module, "_account_id_from_token", return_value=""),
            patch.object(main_module, "_rebind_chat_session_account") as rebind,
        ):
            response = await main_module._run_ide_with_retry(
                [{"role": "user", "content": "pong"}],
                "glm-5.2",
                True,
                options,
                tracker=tracker,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        joined = "".join(chunks)
        self.assertIn('"content":"pong"', joined)
        self.assertIn("data: [DONE]", joined)
        self.assertEqual(selected.await_count, 2)
        next_account.assert_called_once()
        rebind.assert_called_once_with(
            "",
            "account-b",
            "account-b",
            "token-b",
            {"bizUserId": "account-b"},
        )
        tracker.rebind.assert_awaited_once()

    async def test_ide_rate_limit_rotates_once_to_an_enabled_account(self):
        selected = AsyncMock(
            side_effect=[
                RuntimeError(
                    "Trae upstream error 4011: requests have exceeded the rate limit"
                ),
                {"route": "ide"},
            ]
        )
        tracker = SimpleNamespace(rebind=AsyncMock())
        options = {
            "_account_id": "account-a",
            "_auth_token": "token-a",
            "_upstream_mode": "ide",
        }
        next_snapshot = (
            "account-b",
            {
                "token": "token-b",
                "provider_specific": {"bizUserId": "account-b"},
            },
        )

        with (
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": True},
            ),
            patch.object(main_module, "_polling_retry_limit", return_value=2),
            patch.object(
                main_module,
                "_next_retry_account_snapshot",
                return_value=next_snapshot,
            ) as next_account,
            patch.object(main_module, "_account_id_from_token", return_value=""),
            patch.object(main_module, "_rebind_chat_session_account") as rebind,
        ):
            result = await main_module._run_ide_with_retry(
                [{"role": "user", "content": "pong"}],
                "glm-5.2",
                False,
                options,
                tracker=tracker,
            )

        self.assertEqual(result, {"route": "ide"})
        self.assertEqual(selected.await_count, 2)
        next_account.assert_called_once()
        rebind.assert_called_once_with(
            "",
            "account-b",
            "account-b",
            "token-b",
            {"bizUserId": "account-b"},
        )
        tracker.rebind.assert_awaited_once()
        second_options = selected.await_args_list[1].args[3]
        self.assertEqual(second_options["_account_id"], "account-b")
        self.assertEqual(second_options["_auth_token"], "token-b")

    async def test_ide_parameter_error_is_not_retried(self):
        selected = AsyncMock(
            side_effect=RuntimeError(
                "Trae upstream error 4001: the param is invalid"
            )
        )
        with (
            patch.object(main_module, "run_ide_chat", selected),
            patch.object(
                main_module.auth,
                "get_polling_status",
                return_value={"enabled": True},
            ),
            patch.object(main_module, "_polling_retry_limit", return_value=2),
            patch.object(main_module, "_next_retry_account_snapshot") as next_account,
        ):
            with self.assertRaisesRegex(RuntimeError, "4001"):
                await main_module._run_ide_with_retry(
                    [{"role": "user", "content": "pong"}],
                    "glm-5.2",
                    False,
                    {"_account_id": "account-a", "_auth_token": "token-a"},
                )

        selected.assert_awaited_once()
        next_account.assert_not_called()


class IdeRawLegacyFallbackTests(unittest.IsolatedAsyncioTestCase):
    class _ReplayResponse:
        status_code = 200

        def __init__(self, lines):
            self._lines = list(lines)

        def iter_lines(self):
            return iter(self._lines)

        def close(self):
            return None

    @staticmethod
    def _wrapper(response):
        return SimpleNamespace(response=response, close=Mock())

    def _options(self):
        return {
            "_upstream_mode": "ide",
            "_traecode_raw_chat_function": "chat_v3",
            "_ide_endpoint": "/api/agent/v3/llm_utils_chat",
            "tools": TOOLS,
            "tool_choice": "auto",
        }

    async def test_nonstream_raw_param_error_reopens_legacy_adapter(self):
        legacy = self._wrapper(
            self._ReplayResponse(
                [
                    "event: output",
                    'data: {"response":"pong"}',
                    "",
                    "event: done",
                    'data: {"finish_reason":"stop"}',
                    "",
                ]
            )
        )
        raw = AsyncMock(side_effect=RuntimeError("Trae upstream error 4001: param invalid"))
        legacy_send = AsyncMock(return_value=legacy)
        trace = {}

        with (
            patch.object(main_module.raw_client, "send_traecode_chat_request", raw),
            patch.object(main_module.trae_client, "send_chat_request", legacy_send),
        ):
            response = await main_module.run_ide_chat(
                [{"role": "user", "content": "Reply pong"}],
                "glm-5.3",
                False,
                {**self._options(), "_upstream_trace": trace},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.body.decode("utf-8")
        self.assertIn('"content":"pong"', payload)
        raw.assert_awaited_once()
        legacy_send.assert_awaited_once()
        legacy_options = legacy_send.await_args.kwargs["options"]
        self.assertNotIn("_traecode_raw_chat_function", legacy_options)
        self.assertEqual(
            legacy_options["_ide_endpoint"], "/api/agent/v3/llm_utils_chat"
        )
        self.assertTrue(trace["fallback_used"])
        self.assertEqual(trace["actual_endpoint"], "/api/agent/v3/llm_utils_chat")

    async def test_stream_raw_empty_before_output_reopens_legacy_adapter(self):
        raw = AsyncMock(
            return_value=self._wrapper(
                self._ReplayResponse(
                    [
                        # No model payload and no terminal event: the native
                        # translator raises IncompleteUpstreamResponse before
                        # anything is exposed to the caller.
                    ]
                )
            )
        )
        legacy = self._wrapper(
            self._ReplayResponse(
                [
                    "event: output",
                    'data: {"response":"pong"}',
                    "",
                    "event: done",
                    'data: {"finish_reason":"stop"}',
                    "",
                ]
            )
        )
        legacy_send = AsyncMock(return_value=legacy)

        with (
            patch.object(main_module.raw_client, "send_traecode_chat_request", raw),
            patch.object(main_module.trae_client, "send_chat_request", legacy_send),
        ):
            response = await main_module.run_ide_chat(
                [{"role": "user", "content": "Reply pong"}],
                "glm-5.3",
                True,
                self._options(),
            )
            chunks = [chunk async for chunk in response.body_iterator]

        joined = "".join(chunks)
        self.assertIn('"content": "pong"', joined)
        self.assertIn("data: [DONE]", joined)
        raw.assert_awaited_once()
        legacy_send.assert_awaited_once()

    async def test_invalid_native_tool_json_does_not_replay_through_legacy(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                native_wrapper = self._wrapper(
                    self._ReplayResponse(
                        [
                            "event: output",
                            "data: " + json.dumps(
                                {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_native",
                                            "function_call": {
                                                "name": "read_file",
                                                "arguments": '{"path":',
                                            },
                                        }
                                    ]
                                }
                            ),
                            "",
                            "event: token_usage",
                            'data: {"input_token":3,"output_token":5}',
                            "",
                            "event: done",
                            'data: {"finish_reason":"tool_calls"}',
                            "",
                        ]
                    )
                )
                legacy_wrapper = self._wrapper(
                    self._ReplayResponse(
                        [
                            "event: output",
                            'data: {"response":"unexpected legacy replay"}',
                            "",
                            "event: done",
                            'data: {"finish_reason":"stop"}',
                            "",
                        ]
                    )
                )
                raw = AsyncMock(return_value=native_wrapper)
                legacy_send = AsyncMock(return_value=legacy_wrapper)
                with (
                    patch.object(
                        main_module.raw_client, "send_traecode_chat_request", raw
                    ),
                    patch.object(
                        main_module.trae_client, "send_chat_request", legacy_send
                    ),
                ):
                    with self.assertRaises(InvalidNativeToolArguments) as error:
                        response = await main_module.run_ide_chat(
                            [{"role": "user", "content": "Read a.txt"}],
                            "glm-5.3",
                            stream,
                            self._options(),
                        )
                        if stream:
                            async for _ in response.body_iterator:
                                pass

                self.assertFalse(error.exception.retryable)
                self.assertTrue(error.exception.observed_model_event)
                self.assertEqual(error.exception.usage["total_tokens"], 8)
                raw.assert_awaited_once()
                legacy_send.assert_not_awaited()
                native_wrapper.close.assert_called_once()
                legacy_wrapper.close.assert_not_called()

    async def test_guarded_raw_open_errors_do_not_replay_through_legacy(self):
        for stream in (False, True):
            for retryable, observed in ((False, False), (True, True)):
                with self.subTest(
                    stream=stream, retryable=retryable, observed=observed
                ):
                    native_error = EmptyUpstreamResponse(
                        "native turn cannot be replayed",
                        retryable=retryable,
                        observed_model_event=observed,
                    )
                    raw = AsyncMock(side_effect=native_error)
                    legacy_send = AsyncMock()
                    with (
                        patch.object(
                            main_module.raw_client,
                            "send_traecode_chat_request",
                            raw,
                        ),
                        patch.object(
                            main_module.trae_client,
                            "send_chat_request",
                            legacy_send,
                        ),
                    ):
                        with self.assertRaises(EmptyUpstreamResponse) as error:
                            await main_module.run_ide_chat(
                                [{"role": "user", "content": "Read a.txt"}],
                                "glm-5.3",
                                stream,
                                self._options(),
                            )
                    self.assertIs(error.exception, native_error)
                    raw.assert_awaited_once()
                    legacy_send.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
