"""Routing contracts for explicit Trae endpoint modes."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from src import main as main_module


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
        remote.assert_not_awaited()

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


if __name__ == "__main__":
    unittest.main()
