import json
import asyncio
import unittest
import uuid
from unittest.mock import patch

from src import traework_compat


def _parse_custom_sse(frames):
    events = []
    current_event = ""
    data_lines = []

    def flush():
        nonlocal current_event, data_lines
        if data_lines:
            events.append((current_event, json.loads("\n".join(data_lines))))
        current_event = ""
        data_lines = []

    for line in "".join(frames).splitlines():
        if not line:
            flush()
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    flush()
    return events


async def _collect_async(iterator):
    return [item async for item in iterator]


class TraeWorkCompatTests(unittest.TestCase):
    def test_detector_does_not_hijack_openai_agent_metadata(self):
        self.assertFalse(
            traework_compat.is_traework_request(
                "/v1/chat/completions",
                headers={"user-agent": "openai-python/2"},
                body={
                    "model": "glm-5.3",
                    "messages": [{"role": "user", "content": "hello"}],
                    "agent_type": "coding",
                    "chat_session_id": "session-1",
                    "render_context": {"theme": "dark"},
                    "mcp_tool_list": [{"name": "read_file"}],
                },
            )
        )

    def test_detector_accepts_strong_header_or_custom_model_identity(self):
        self.assertTrue(
            traework_compat.is_traework_request(
                "/v1/chat/completions",
                headers={"x-bridge-transport": "aha"},
                body={"model": "glm-5.3"},
            )
        )
        self.assertTrue(
            traework_compat.is_traework_request(
                "/v1/chat/completions",
                body={
                    "custom_model": {
                        "config_name": "glm-5.3",
                        "is_custom_base_url": True,
                    }
                },
            )
        )

    def test_resolve_model_binding_uses_shared_raw_mapping(self):
        binding = traework_compat.resolve_model_binding("gpt-4o")

        self.assertEqual(binding.config_name, "DeepSeek-V4-Pro")
        self.assertEqual(binding.raw_model_name, "DeepSeek-V4-Pro__v2")
        self.assertEqual(binding.display_name, "DeepSeek-V4-Pro")

    def test_build_request_ids_reuses_single_seed_across_traework_ids(self):
        fixed = uuid.UUID("12345678-1234-5678-1234-567812345678")
        with patch("src.traework_compat.uuid.uuid4", return_value=fixed):
            ids = traework_compat.build_request_ids()

        expected = str(fixed)
        self.assertEqual(ids.session_id, expected)
        self.assertEqual(ids.conversation_id, expected)
        self.assertEqual(ids.agent_loop_id, expected)
        self.assertEqual(ids.user_prompt_submit_id, expected)
        self.assertEqual(ids.request_id, expected)

    def test_parse_raw_extra_header_requires_object(self):
        self.assertEqual(traework_compat.parse_raw_extra_header(""), {})
        with self.assertRaises(ValueError):
            traework_compat.parse_raw_extra_header('["not","an","object"]')

    def test_build_raw_chat_request_matches_llm_raw_chat_shape(self):
        descriptor = traework_compat.build_raw_chat_request(
            [{"role": "user", "content": "hello"}],
            "DeepSeek-V4-Flash-Official",
            token="jwt-token",
            base_url="https://bridge.example",
            options={"sessionId": "session-1"},
        )

        self.assertEqual(descriptor.method, "POST")
        self.assertEqual(descriptor.path, "/api/ide/v2/llm_raw_chat")
        self.assertTrue(descriptor.stream)
        self.assertEqual(
            set(descriptor.json_body),
            {
                "config_name",
                "conversation_id",
                "messages",
                "model_name",
                "session_id",
                "stream",
            },
        )
        self.assertEqual(descriptor.json_body["config_name"], "DeepSeek-V4-Flash-Official")
        self.assertEqual(descriptor.json_body["model_name"], "DeepSeek-V4-Flash-Official")
        self.assertEqual(descriptor.json_body["session_id"], "session-1")
        self.assertEqual(descriptor.json_body["conversation_id"], "session-1")
        self.assertEqual(descriptor.headers["Authorization"], "Cloud-IDE-JWT jwt-token")
        self.assertEqual(descriptor.headers["X-Request-Id"], "session-1")
        extra = traework_compat.parse_raw_extra_header(descriptor.headers["Extra"])
        self.assertEqual(extra["agent_loop_id"], "session-1")
        self.assertEqual(extra["user_prompt_submit_id"], "session-1")
        self.assertEqual(extra["config_name"], "DeepSeek-V4-Flash-Official")
        self.assertEqual(extra["display_name"], "DeepSeek-V4-Flash 正式版")
        self.assertEqual(extra["base_url"], "https://bridge.example/trae-cli/api/v1/llm/proxy")
        descriptor.headers["Extra"].encode("ascii")

    def test_build_raw_chat_request_can_disable_stream(self):
        descriptor = traework_compat.build_raw_chat_request(
            [{"role": "user", "content": "hello"}],
            "glm-5.3",
            stream=False,
            options={"session_id": "sync-1"},
        )

        self.assertFalse(descriptor.stream)
        self.assertFalse(descriptor.json_body["stream"])
        self.assertEqual(descriptor.headers["Accept"], "application/json")

    def test_build_remote_session_request_uses_auto_strategy_for_auto_model(self):
        descriptor = traework_compat.build_remote_session_request(
            [{"role": "user", "content": "hello"}],
            "auto",
            token="jwt-token",
            options={"provider_specific": {}},
        )

        self.assertEqual(descriptor.method, "POST")
        self.assertEqual(descriptor.path, "/api/remote/v1/chat_sessions")
        self.assertFalse(descriptor.stream)
        initial = descriptor.json_body["initial_message"]
        self.assertEqual(descriptor.json_body["mode"], "code")
        self.assertEqual(initial["agent_type"], "solo_agent_remote")
        self.assertEqual(initial["agent_id"], "solo_agent_remote")
        self.assertEqual(initial["model_selection_strategy"], "auto")
        self.assertEqual(initial["model_name"], "")
        self.assertNotIn("custom_model", initial)
        self.assertEqual(descriptor.headers["Authorization"], "Cloud-IDE-JWT jwt-token")
        self.assertEqual(descriptor.headers["X-Request-Id"], descriptor.ids.request_id)

    def test_build_remote_session_request_uses_manual_binding_for_explicit_model(self):
        descriptor = traework_compat.build_remote_session_request(
            [{"role": "user", "content": "hello"}],
            "glm-5.3",
            options={"provider_specific": {}, "_account_id": "account-1"},
        )

        initial = descriptor.json_body["initial_message"]
        self.assertEqual(initial["model_selection_strategy"], "manual")
        self.assertEqual(initial["model_name"], "glm-5.3")
        self.assertEqual(initial["model_config_source"], 1)
        self.assertTrue(initial["model_is_preset"])
        self.assertEqual(initial["model_provider"], "")
        self.assertEqual(
            initial["custom_model"],
            {
                "name": "glm-5.3",
                "config_name": "glm-5.3",
                "model_name": "glm-5.3",
                "display_name": "GLM-5.3",
                "config_source": 1,
                "is_preset": True,
                "provider": "",
            },
        )
        common = json.loads(initial["common_params"])
        self.assertEqual(common["biz_session_id"], descriptor.ids.session_id)

    def test_remote_events_path_formats_reply_target(self):
        self.assertEqual(
            traework_compat.remote_events_path("session-1", "message-1"),
            "/api/remote/v1/chat_sessions/session-1/events?reply_to_message_id=message-1",
        )

    def test_normalize_inbound_renderer_messages_preserves_tool_round_trip(self):
        request = traework_compat.normalize_inbound_request(
            {
                "config_name": "glm-5.3",
                "conversation_id": "conversation-1",
                "messages": [
                    {
                        "role": 2,
                        "content": [
                            {
                                "type": "tool_use",
                                "toolCallId": "call_write_1",
                                "name": "writeFile",
                                "parameters": {
                                    "path": "C:/workspace/output.txt",
                                    "content": "relay-ok",
                                },
                            }
                        ],
                    },
                    {
                        "role": 1,
                        "content": [
                            {
                                "type": "tool_result",
                                "toolCallId": "call_write_1",
                                "value": [{"type": "text", "value": "written"}],
                                "isError": False,
                            }
                        ],
                    },
                ],
                "stream": True,
            },
            headers={"x-bridge-transport": "aha"},
            path="/api/ide/v2/llm_raw_chat",
        )

        self.assertEqual(request.model, "glm-5.3")
        self.assertEqual(request.session_id, "conversation-1")
        self.assertTrue(request.options["_tool_protocol_requested"])
        self.assertEqual(request.messages[0]["role"], "assistant")
        self.assertEqual(request.messages[0]["content"][0]["type"], "tool_use")
        self.assertEqual(request.messages[1]["role"], "user")
        self.assertEqual(request.messages[1]["content"][0]["type"], "tool_result")

    def test_normalize_inbound_uses_extra_model_binding(self):
        request = traework_compat.normalize_inbound_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "session_id": "session-extra",
            },
            headers={
                "Extra": json.dumps(
                    {
                        "config_name": "glm-5.3",
                        "model_name": "glm-5.3",
                        "display_name": "GLM-5.3",
                        "session_id": "session-extra",
                    }
                ),
                "X-App-Id": "app",
                "X-Ide-Function": "chat",
            },
            path="/api/ide/v2/llm_raw_chat",
        )

        self.assertEqual(request.model, "glm-5.3")
        self.assertEqual(request.options["trae_raw_config_name"], "glm-5.3")
        self.assertEqual(request.options["trae_raw_model_name"], "glm-5.3")

    def test_normalize_inbound_prefers_canonical_config_over_provider_model(self):
        request = traework_compat.normalize_inbound_request(
            {
                "model": "GLM-5.2",
                "data": {
                    "config_name": "glm-5.2",
                    "model_name": "glm-5.2__dev",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                "stream": False,
            },
            path="/api/ide/v2/llm_raw_chat",
        )

        self.assertEqual(request.model, "glm-5.2")
        self.assertEqual(request.options["trae_raw_config_name"], "glm-5.2")
        self.assertEqual(request.options["trae_raw_model_name"], "glm-5.2__dev")

    def test_nested_options_keep_specific_values(self):
        request = traework_compat.normalize_inbound_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": False,
                "data": {"thinking": True, "max_tokens": 321},
            }
        )

        self.assertTrue(request.options["thinking"])
        self.assertEqual(request.options["max_tokens"], 321)

    def test_translate_openai_stream_to_traework_emits_cumulative_tool_call(self):
        async def source():
            yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_write","type":"function","function":{"name":"writeFile","arguments":"{\\"path\\":\\"C:/work/"}}]},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"out.txt\\",\\"content\\":\\"ok\\"}"}}]},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
            yield "data: [DONE]\n\n"

        async def collect():
            return [
                item
                async for item in traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-1"
                )
            ]

        frames = asyncio.run(collect())
        joined = "".join(frames)
        events = _parse_custom_sse(frames)
        outputs = [payload for name, payload in events if name == "output"]
        final_done = next(payload for name, payload in events if name == "done")
        self.assertIn("event: output", joined)
        self.assertIn('"id":"call_write"', joined)
        self.assertIn('"name":"writeFile"', joined)
        self.assertIn('C:/work/out.txt', joined)
        self.assertIn("event: token_usage", joined)
        self.assertIn("event: done", joined)
        # The TraeWork raw client treats each ``input`` as an argument delta,
        # not a cumulative snapshot. Reconstruct the client-side buffer to
        # verify the wire contract rather than only the relay accumulator.
        argument_buffer = ""
        for output in outputs:
            for call in output.get("tool_calls", []):
                argument_buffer += call.get("input", "")
        self.assertEqual(argument_buffer, '{"path":"C:/work/out.txt","content":"ok"}')
        self.assertEqual(
            final_done["tool_calls"][0]["input"],
            '{"path":"C:/work/out.txt","content":"ok"}',
        )

    def test_translate_emits_start_frame_and_compacts_reasoning(self):
        async def source():
            # This is the relay's parseable start frame while raw headers are
            # still pending; it must not be discarded by the custom adapter.
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":""},'
                '"finish_reason":null}]}\n\n'
            )
            yield (
                'data: {"choices":[{"index":0,"delta":{"reasoning_content":'
                '"Inspect files.\\nInspect files.\\nApply fix.\\nValidate.\\n"},'
                '"finish_reason":null}]}\n\n'
            )
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-reasoning"
                )
            )
        )
        events = _parse_custom_sse(frames)
        outputs = [payload for name, payload in events if name == "output"]
        done = next(payload for name, payload in events if name == "done")

        self.assertTrue(outputs)
        self.assertEqual(outputs[0]["response"], "")
        self.assertEqual(done["response"], "ok")
        summary = done["reasoning_content"]
        self.assertLessEqual(len(summary), 800)
        self.assertLessEqual(len(summary.splitlines()), 6)
        self.assertNotIn("Inspect files.\nInspect files.", summary)
        reasoning_deltas = [
            payload.get("reasoning_delta", "")
            for name, payload in events
            if name == "output"
        ]
        self.assertTrue(any(reasoning_deltas))
        self.assertEqual(
            "".join(reasoning_deltas).replace("\n", "").count("Inspect files."),
            1,
        )

    def test_intermediate_finish_frame_does_not_emit_terminal_output(self):
        """TraeWork must not see a terminal marker before the stream ends."""

        async def source():
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":"once"},'
                '"finish_reason":"stop"}]}\n\n'
            )
            yield (
                'data: {"choices":[{"index":0,"delta":{"content":" and continued"},'
                '"finish_reason":null}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-finish-later"
                )
            )
        )
        events = _parse_custom_sse(frames)
        outputs = [payload for name, payload in events if name == "output"]
        done = next(payload for name, payload in events if name == "done")

        # finish_reason belongs only to the terminal event. Some upstreams
        # emit an intermediate cumulative/finish frame and continue streaming;
        # TraeWork treats an output frame carrying it as the end of the turn.
        self.assertFalse(any("finish_reason" in payload for payload in outputs))
        self.assertEqual(done["finish_reason"], "stop")
        self.assertEqual(done["response"], "once and continued")
    def test_translate_empty_stream_emits_explicit_error(self):
        async def source():
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-empty"
                )
            )
        )
        events = _parse_custom_sse(frames)
        error = next(payload for name, payload in events if name == "error")
        done = next(payload for name, payload in events if name == "done")
        self.assertEqual(error["code"], "empty_response")
        self.assertEqual(done["status"], "error")

    def test_translate_empty_stream_emits_error_before_usage(self):
        async def source():
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":0,"total_tokens":3}}\n\n'
            )
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-empty-usage"
                )
            )
        )
        event_names = [name for name, _payload in _parse_custom_sse(frames)]

        self.assertIn("error", event_names)
        self.assertIn("token_usage", event_names)
        self.assertLess(event_names.index("error"), event_names.index("token_usage"))
        self.assertEqual(event_names[-1], "done")

    def test_translate_preserves_upstream_error_in_terminal_event(self):
        async def source():
            yield (
                'data: {"error":{"code":"rate_limited",'
                '"message":"retry later"}}\n\n'
            )
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-error"
                )
            )
        )
        events = _parse_custom_sse(frames)
        error = next(payload for name, payload in events if name == "error")
        done = next(payload for name, payload in events if name == "done")
        self.assertEqual(error["code"], "rate_limited")
        self.assertEqual(done["status"], "error")
        self.assertEqual(done["error"]["code"], "rate_limited")

    def test_translate_preserves_provider_identity_and_whitelists_usage_fields(self):
        async def source():
            yield (
                'data: {"model":"glm-5.3","provider_model_name":"glm-5.3__max",'
                '"model_provider_name":"trae","choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}],"usage":{"prompt_tokens":3,'
                '"completion_tokens":2,"total_tokens":5,"cache_read_tokens":1,'
                '"credits_consumed":0.4,"Extra":"do-not-forward"}}\n\n'
            )
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-provider"
                )
            )
        )
        events = _parse_custom_sse(frames)
        output = next(payload for name, payload in events if name == "output")
        usage = next(payload for name, payload in events if name == "token_usage")
        done = next(payload for name, payload in events if name == "done")

        for payload in (output, usage, done):
            self.assertEqual(payload["provider_model_name"], "glm-5.3__max")
            self.assertEqual(payload["config_name"], "glm-5.3")
            self.assertNotIn("Extra", payload)
        self.assertEqual(usage["cache_read_tokens"], 1)
        self.assertEqual(usage["credits_consumed"], 0.4)
        self.assertNotIn("Extra", usage)

    def test_translate_buffers_sse_json_split_across_transport_chunks(self):
        wire = (
            'data: {"provider_model_name":"glm-5.3__max","choices":['
            '{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )

        async def source():
            # Split inside both the JSON string and the SSE delimiter. Real
            # HTTP chunking is arbitrary, so neither boundary is guaranteed.
            for boundary in (43, 78, len(wire) - 1):
                nonlocal_offset[0], start = boundary, nonlocal_offset[0]
                yield wire[start:boundary]
            yield wire[nonlocal_offset[0]:]

        nonlocal_offset = [0]
        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-split"
                )
            )
        )
        events = _parse_custom_sse(frames)
        done = next(payload for name, payload in events if name == "done")

        self.assertEqual(done["response"], "hello")
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["provider_model_name"], "glm-5.3__max")
        self.assertFalse(any(name == "error" for name, _payload in events))

    def test_nonstream_conversion_preserves_nested_provider_identity(self):
        result = traework_compat.openai_completion_to_traework(
            {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "model_config": {
                    "config_name": "glm-5.3",
                    "model_name": "glm-5.3__dev",
                    "provider": "trae",
                },
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            model="glm-5.3",
        )

        self.assertEqual(result["config_name"], "glm-5.3")
        self.assertEqual(result["model_name"], "glm-5.3__dev")
        self.assertEqual(result["provider_model_name"], "glm-5.3__dev")
        self.assertEqual(result["provider"], "trae")
        self.assertEqual(result["usage"]["provider_model_name"], "glm-5.3__dev")

    def test_error_conversion_keeps_code_param_and_provider_without_extra(self):
        result = traework_compat.openai_error_to_traework(
            {
                "model_config": {
                    "config_name": "glm-5.3",
                    "model_name": "glm-5.3__max",
                    "provider": "trae",
                },
                "error": {
                    "code": "rate_limited",
                    "message": "retry later",
                    "type": "upstream_error",
                    "param": "tools",
                    "Extra": "must not leak",
                },
            },
            model="glm-5.3",
            request_id="req-error-convert",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "rate_limited")
        self.assertEqual(result["error"]["param"], "tools")
        self.assertEqual(result["error"]["type"], "upstream_error")
        self.assertEqual(result["error"]["request_id"], "req-error-convert")
        self.assertEqual(result["provider_model_name"], "glm-5.3__max")
        self.assertNotIn("Extra", result)
        self.assertNotIn("Extra", result["error"])

    def test_translate_keeps_multiple_unindexed_tool_calls_separate(self):
        async def source():
            yield (
                'data: {"choices":[{"delta":{"tool_calls":['
                '{"id":"call_a","type":"function","function":{"name":"a",'
                '"arguments":"{}"}},'
                '{"id":"call_b","type":"function","function":{"name":"b",'
                '"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-multi"
                )
            )
        )
        events = _parse_custom_sse(frames)
        done = next(payload for name, payload in events if name == "done")
        self.assertEqual(
            {call["id"] for call in done["tool_calls"]},
            {"call_a", "call_b"},
        )

    def test_translate_merges_fragmented_tool_call_info_parameters(self):
        async def source():
            yield (
                'data: {"tool_call_info":{"tool_call_id":"call_write",'
                '"name":"writeFile","parameters":"{\\"path\\":\\"C:/work/"}}}\n\n'
            )
            yield (
                'data: {"tool_call_info":{"tool_call_id":"call_write",'
                '"name":"writeFile","parameters":"out.txt\\",\\"content\\":\\"ok\\"}"}}\n\n'
            )
            yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-tool-info-fragments"
                )
            )
        )
        events = _parse_custom_sse(frames)
        outputs = [payload for name, payload in events if name == "output"]
        done = next(payload for name, payload in events if name == "done")
        self.assertEqual(
            "".join(call.get("input", "") for output in outputs for call in output.get("tool_calls", [])),
            '{"path":"C:/work/out.txt","content":"ok"}',
        )
        self.assertEqual(done["tool_calls"][0]["input"], '{"path":"C:/work/out.txt","content":"ok"}')
        self.assertEqual(done["tool_calls"][0]["name"], "writeFile")

    def test_translate_emits_reasoning_delta_from_reasoning_delta_field(self):
        async def source():
            yield 'data: {"choices":[{"delta":{"reasoning_delta":"Inspect "},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{"reasoning_delta":"files."},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        frames = asyncio.run(
            _collect_async(
                traework_compat.translate_openai_stream_to_traework(
                    source(), model="glm-5.3", request_id="req-reasoning-delta"
                )
            )
        )
        events = _parse_custom_sse(frames)
        outputs = [payload for name, payload in events if name == "output"]
        done = next(payload for name, payload in events if name == "done")
        self.assertEqual("".join(item.get("reasoning_delta", "") for item in outputs), "Inspect files.")
        self.assertEqual(done["reasoning_content"], "Inspect files.")

    def test_connectivity_response_reports_tool_and_reasoning_capabilities(self):
        response = traework_compat.connectivity_response(
            {"custom_model": {"config_name": "glm-5.3"}}
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["code"], 0)
        self.assertEqual(response["config_name"], "glm-5.3")
        self.assertFalse(response["model"]["is_preset"])
        self.assertTrue(response["model"]["is_custom_base_url"])
        self.assertTrue(response["model"]["features"]["tool_calls"]["enable"])
        self.assertTrue(response["model"]["features"]["reasoning"]["enable"])

    def test_connectivity_response_overrides_matching_upstream_preset_identity(self):
        response = traework_compat.connectivity_response(
            {"custom_model": {"config_name": "glm-5.3"}},
            models=[
                {
                    "id": "glm-5.3",
                    "name": "glm-5.3",
                    "is_preset": True,
                    "is_custom_base_url": False,
                    "upstream_marker": "preserved",
                }
            ],
        )

        selected = next(
            item for item in response["models"] if item["id"] == "glm-5.3"
        )
        self.assertFalse(selected["is_preset"])
        self.assertTrue(selected["is_custom_base_url"])
        self.assertEqual(selected["upstream_marker"], "preserved")

    def test_connectivity_response_preserves_custom_model_identity_fields(self):
        response = traework_compat.connectivity_response(
            {
                "custom_model": {
                    "config_name": "custom_openai_compatible//mimo-v2.5",
                    "model_name": "mimo-v2.5",
                    "display_name": "mimo-v2.5",
                    "provider": "custom_openai_compatible",
                    "is_preset": False,
                    "is_custom_base_url": True,
                    "custom_model_id": "1260420226",
                    "config_source": 3,
                }
            }
        )

        model = response["model"]
        self.assertFalse(model["is_preset"])
        self.assertTrue(model["is_custom_base_url"])
        self.assertEqual(model["custom_model_id"], "1260420226")
        self.assertEqual(model["config_source"], 3)
        self.assertEqual(response["config_name"], "custom_openai_compatible//mimo-v2.5")

    def test_connectivity_response_marks_preset_model_as_non_custom(self):
        response = traework_compat.connectivity_response(
            {
                "custom_model": {
                    "config_name": "glm-5.3",
                    "is_preset": True,
                    "config_source": 1,
                }
            }
        )

        model = response["model"]
        self.assertTrue(model["is_preset"])
        self.assertIsNone(model["is_custom_base_url"])
        self.assertIsNone(model["custom_model_id"])
        self.assertEqual(model["config_source"], 1)


if __name__ == "__main__":
    unittest.main()
