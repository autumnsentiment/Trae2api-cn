"""Regression coverage for the 2026-11-09 tool request report."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from src import anthropic_api, main, raw_client, responses_api
from src.sse import UpstreamEventError

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "options": {
            "type": "object",
            "properties": {"strict": {"type": "boolean"}},
            "required": ["strict"],
            "additionalProperties": False,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}
TOOL = {
    "type": "function",
    "name": "read_file",
    "parameters": json.dumps(SCHEMA),
    "strict": True,
}
REAL_HTTPX_CLIENT = httpx.Client


class ResponsesToolSchemaReportTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()

    def test_flat_nested_and_namespace_schema_strings_survive_the_native_wire(self):
        for shape in ("flat", "nested", "namespace"):
            with self.subTest(shape=shape):
                tool = copy.deepcopy(TOOL)
                if shape == "nested":
                    tool = {"type": "function", "function": tool}
                elif shape == "namespace":
                    tool = {"type": "namespace", "name": "files", "tools": [tool]}
                original = copy.deepcopy(tool)
                _, options, _ = responses_api.normalize_request(
                    {"model": "glm-5.2", "input": "read", "tools": [tool]}
                )
                self.assertIsNone(main._validate_chat_options(options))
                body = raw_client.build_traecode_chat_body(
                    [{"role": "user", "content": "read"}],
                    "glm-5.2",
                    options={**options, "_traecode_native_tools": True},
                    session_id="report-schema",
                )

                function = body["tools"][0]["function"]
                self.assertEqual(json.loads(function["parameters"]), SCHEMA)
                self.assertNotIn("strict", function)
                self.assertEqual(tool, original)

    def test_invalid_responses_schemas_fail_before_dispatch(self):
        for parameters in ('{"type":', "[]", "null", 42, []):
            for nested in (False, True):
                with self.subTest(parameters=parameters, nested=nested):
                    tool = {**TOOL, "parameters": parameters}
                    if nested:
                        tool = {"type": "function", "function": tool}
                    dispatch = AsyncMock()
                    with (
                        patch.object(main, "API_KEYS", ["report-test-key"]),
                        patch.object(main, "_dispatch_chat", dispatch),
                    ):
                        response = self.client.post(
                            "/v1/responses",
                            json={
                                "model": "glm-5.2",
                                "input": "read",
                                "tools": [tool],
                            },
                            headers={"Authorization": "Bearer report-test-key"},
                        )

                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(
                        response.json()["error"]["param"],
                        "tools.0.function.parameters",
                    )
                    dispatch.assert_not_awaited()

    def test_messages_schema_strings_survive_and_invalid_schemas_are_rejected(self):
        tool = {"name": TOOL["name"], "input_schema": TOOL["parameters"]}
        body = {
            "model": "glm-5.2",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "read"}],
            "tools": [tool],
        }
        messages, options = anthropic_api.normalize_request(body)
        self.assertIsNone(main._validate_chat_options(options))
        native = raw_client.build_traecode_chat_body(
            messages, "glm-5.2", options={**options, "_traecode_native_tools": True}
        )
        self.assertEqual(json.loads(native["tools"][0]["function"]["parameters"]), SCHEMA)
        for parameters in ("", '{"type":', "[]", "null", False, 0, []):
            with self.subTest(parameters=parameters):
                dispatch = AsyncMock()
                with (
                    patch.object(main, "API_KEYS", ["report-test-key"]),
                    patch.object(main, "_dispatch_chat", dispatch),
                ):
                    response = self.client.post(
                        "/v1/messages",
                        json={
                            **body,
                            "tools": [{**tool, "input_schema": parameters}],
                        },
                        headers={"Authorization": "Bearer report-test-key"},
                    )

                self.assertEqual(response.status_code, 400, response.text)
                dispatch.assert_not_awaited()


class NativeToolFallbackReportTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _options():
        return {
            "_traecode_raw_chat_function": "solo_agent",
            "tools": [
                {"type": "function", "function": {"name": "read_file", "parameters": SCHEMA}}
            ],
            "tool_choice": "required",
        }

    async def test_unknown_or_account_errors_never_use_the_legacy_tool_adapter(self):
        for stream in (False, True):
            for error in (
                UpstreamEventError("authentication expired", code=401),
                UpstreamEventError("quota exhausted", code=4008),
                UpstreamEventError("unrecognized upstream failure"),
                RuntimeError("unexpected local tool serialization failure"),
            ):
                with self.subTest(stream=stream, error=str(error)):
                    raw = AsyncMock(side_effect=error)
                    legacy = AsyncMock()
                    with (
                        patch.object(raw_client, "send_traecode_chat_request", raw),
                        patch.object(main.trae_client, "send_chat_request", legacy),
                        self.assertRaises(type(error)) as caught,
                    ):
                        await main.run_ide_chat(
                            [{"role": "user", "content": "read"}],
                            "glm-5.2",
                            stream,
                            self._options(),
                        )
                    self.assertIs(caught.exception, error)
                    raw.assert_awaited_once()
                    legacy.assert_not_awaited()

    async def test_generic_compatibility_errors_with_output_or_usage_are_not_replayed(self):
        for stream in (False, True):
            for evidence in (
                {"observed_model_event": True},
                {"usage": {"prompt_tokens": 1}},
                {"usage": {}},
                {"retryable": False},
            ):
                with self.subTest(stream=stream, evidence=evidence):
                    error = RuntimeError("Trae upstream error 4001: param invalid")
                    for key, value in evidence.items():
                        setattr(error, key, value)
                    raw = AsyncMock(side_effect=error)
                    legacy = AsyncMock()
                    with (
                        patch.object(raw_client, "send_traecode_chat_request", raw),
                        patch.object(main.trae_client, "send_chat_request", legacy),
                        self.assertRaises(RuntimeError) as caught,
                    ):
                        await main.run_ide_chat(
                            [{"role": "user", "content": "read"}],
                            "glm-5.2",
                            stream,
                            self._options(),
                        )
                    self.assertIs(caught.exception, error)
                    legacy.assert_not_awaited()

    async def test_unknown_stream_error_is_closed_without_legacy_replay(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = SimpleNamespace(
                    iter_lines=lambda: iter(
                        [
                            "event: error",
                            'data: {"code":401,"message":"authentication expired"}',
                            "",
                        ]
                    )
                )
                wrapper = SimpleNamespace(response=response, close=Mock())
                legacy = AsyncMock()
                with (
                    patch.object(
                        raw_client,
                        "send_traecode_chat_request",
                        AsyncMock(return_value=wrapper),
                    ),
                    patch.object(main.trae_client, "send_chat_request", legacy),
                    self.assertRaises(UpstreamEventError),
                ):
                    result = await main.run_ide_chat(
                        [{"role": "user", "content": "read"}],
                        "glm-5.2",
                        stream,
                        self._options(),
                    )
                    if stream:
                        async for _ in result.body_iterator:
                            pass

                legacy.assert_not_awaited()
                wrapper.close.assert_called_once()

    async def test_http_compatibility_rejection_keeps_the_same_scope_and_history(self):
        messages = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_history",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"a.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_history", "content": "contents"},
        ]
        for stream in (False, True):
            for code in (4001, 1005):
                with self.subTest(stream=stream, code=code):
                    response = SimpleNamespace(
                        iter_lines=lambda: iter(
                            [
                                "event: output",
                                'data: {"response":"contents"}',
                                "",
                                "event: done",
                                "data: {}",
                                "",
                            ]
                        )
                    )
                    wrapper = SimpleNamespace(response=response, close=Mock())
                    legacy = AsyncMock(return_value=wrapper)
                    with (
                        patch.object(
                            raw_client,
                            "send_traecode_chat_request",
                            AsyncMock(
                                side_effect=raw_client.TraeCodeRequestError(
                                    400, code, "native envelope rejected"
                                )
                            ),
                        ),
                        patch.object(main.trae_client, "send_chat_request", legacy),
                    ):
                        result = await main.run_ide_chat(
                            messages,
                            "glm-5.2",
                            stream,
                            {**self._options(), "tool_choice": "auto"},
                        )
                        if stream:
                            chunks = [chunk async for chunk in result.body_iterator]
                            self.assertEqual("".join(chunks).count("data: [DONE]"), 1)
                        else:
                            self.assertEqual(result.status_code, 200)

                    self.assertEqual(legacy.await_args.args[0], messages)
                    self.assertEqual(legacy.await_args.args[1], "glm-5.2")
                    options = legacy.await_args.kwargs["options"]
                    self.assertEqual(options["_traecode_raw_chat_function"], "solo_agent")
                    self.assertEqual(options["tools"], self._options()["tools"])
                    self.assertEqual(options["tool_choice"], "auto")
                    wrapper.close.assert_called_once()

    async def test_unknown_stream_business_errors_do_not_rotate_accounts(self):
        for during_body in (False, True):
            with self.subTest(during_body=during_body):
                error = UpstreamEventError("authentication expired", code=401)

                async def body(error=error):
                    raise error
                    yield

                selected = (
                    AsyncMock(return_value=StreamingResponse(body()))
                    if during_body
                    else AsyncMock(side_effect=error)
                )
                next_account = Mock()
                with (
                    patch.object(main.auth, "get_polling_status", return_value={"enabled": True}),
                    patch.object(main, "_polling_retry_limit", return_value=2),
                    patch.object(main, "_next_retry_account_snapshot", next_account),
                    patch.object(main, "run_ide_chat", selected),
                    self.assertRaises(UpstreamEventError) as caught,
                ):
                    result = await main._run_ide_with_retry(
                        [{"role": "user", "content": "read"}],
                        "glm-5.2",
                        True,
                        {**self._options(), "_account_id": "report-account"},
                    )
                    if during_body:
                        async for _ in result.body_iterator:
                            pass

                self.assertIs(caught.exception, error)
                selected.assert_awaited_once()
                next_account.assert_not_called()

    async def test_single_endpoint_diagnostic_never_rotates_native_accounts(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                error = UpstreamEventError("busy", code=4011)
                selected = AsyncMock(side_effect=error)
                next_account = Mock()
                with (
                    patch.object(main.auth, "get_polling_status", return_value={"enabled": True}),
                    patch.object(main, "_next_retry_account_snapshot", next_account),
                    patch.object(main, "run_ide_chat", selected),
                    self.assertRaises(UpstreamEventError) as caught,
                ):
                    await main._run_ide_with_retry(
                        [{"role": "user", "content": "read"}],
                        "glm-5.2",
                        stream,
                        {**self._options(), "_disable_upstream_fallback": True},
                    )

                self.assertIs(caught.exception, error)
                selected.assert_awaited_once()
                next_account.assert_not_called()

    async def test_http_rejection_with_usage_or_output_stops_all_replays(self):
        for evidence in (
            {"usage": {"prompt_tokens": 1}},
            {"error": {"usage": {}}},
            {"token_usage": {"input_token": 1}},
            {"observed_model_event": True},
        ):
            with self.subTest(evidence=evidence):
                requests = []
                session_id = "report-http-no-replay"

                def handler(request, requests=requests, evidence=evidence):
                    requests.append(request)
                    return httpx.Response(400, json={"code": 4001, **evidence})

                def client_factory(handler=handler, **kwargs):
                    return REAL_HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

                legacy = AsyncMock()
                with (
                    patch("src.trae_client.build_headers", return_value={}),
                    patch.object(
                        raw_client,
                        "resolve_traecode_model_for_request",
                        AsyncMock(return_value=raw_client.RawModel("glm-5.2", "glm-5.2", "GLM")),
                    ),
                    patch.object(
                        raw_client,
                        "resolve_traecode_base_urls",
                        return_value=["https://primary.invalid", "https://alternate.invalid"],
                    ),
                    patch.object(raw_client.auth, "resolve_model_account_id", return_value=""),
                    patch("src.raw_client.httpx.Client", side_effect=client_factory),
                    patch.object(main.trae_client, "send_chat_request", legacy),
                    self.assertRaises(raw_client.TraeCodeRequestError) as caught,
                ):
                    await main.run_ide_chat(
                        [{"role": "user", "content": "read"}],
                        "glm-5.2",
                        False,
                        {
                            **self._options(),
                            "_auth_token": "test-token",
                            "_traecode_session_id": session_id,
                        },
                    )

                self.assertFalse(caught.exception.retryable)
                self.assertEqual(len(requests), 1)
                legacy.assert_not_awaited()
                with raw_client._RAW_SESSION_GATES_LOCK:
                    self.assertNotIn(session_id, raw_client._RAW_SESSION_GATES)
