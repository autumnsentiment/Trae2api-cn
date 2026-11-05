"""Full upstream reasoning keeps its own channel and strips tool protocol."""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

from src import auth, cli_client, raw_client, responses_api, sse, traework_compat


class _Lines:
    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self):
        return iter(self.lines)


async def _collect(source):
    return [item async for item in source]


def _payloads(frames):
    return [
        json.loads(line[5:].strip())
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


def _chat_text(frames, field):
    return "".join(
        choice.get("delta", {}).get(field, "")
        for payload in _payloads(frames)
        for choice in payload.get("choices", [])
    )


class ReasoningOutputTests(unittest.TestCase):
    def setUp(self):
        self.setting = {"enabled": True}
        self.mock_setting = patch.object(
            auth, "get_reasoning_output_settings",
            side_effect=lambda: dict(self.setting), create=True,
        )
        self.mock_setting.start()
        self.addCleanup(self.mock_setting.stop)
        self.reasoning = (
            "The user asks for a verified result.\n"
            + "\n".join(
                "Step %02d: Inspect the deterministic input and preserve every "
                "intermediate observation before selecting the final result."
                % index
                for index in range(12)
            )
            + "\nRepeat observation.\nRepeat observation.\n"
            "```python\nif ready:\n    result = 7\n```\nConclusion: return seven."
        )
        self.assertGreater(len(self.reasoning), 800)

    def test_full_formatter_preserves_all_steps_repeats_and_code(self):
        tool = (
            '<tool_call>{"id":"call-1","name":"probe",'
            '"arguments":{"value":"hidden-protocol"}}</tool_call>'
        )
        formatted = sse.compact_reasoning_text(
            "<thinking>" + self.reasoning + "</thinking>\n" + tool
        )
        self.assertEqual(formatted, self.reasoning)
        self.assertNotIn("hidden-protocol", formatted)
        self.assertEqual(
            sse.compact_reasoning_text(self.reasoning + "\n<tool_call"),
            self.reasoning,
        )

    def test_summary_mode_retains_existing_bounds_and_fallback_is_always_bounded(self):
        for enabled, allow_full in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, allow_full=allow_full):
                self.setting["enabled"] = enabled
                result = sse.compact_reasoning_text(
                    self.reasoning, allow_full=allow_full
                )
                self.assertLessEqual(len(result), 800)
                self.assertLessEqual(len(result.splitlines()), 6)
                self.assertNotEqual(result, self.reasoning)
                self.assertNotIn("The user asks", result)

    def test_raw_prompt_uses_the_same_live_output_setting(self):
        self.assertEqual(raw_client.response_style_instruction(), "")
        self.setting["enabled"] = False
        self.assertIn("only the final", raw_client.response_style_instruction())

    def test_legacy_verbose_env_preserves_more_than_800_characters(self):
        self.mock_setting.stop()
        with patch.object(auth, "_settings", {}), patch.dict(
            os.environ, {"TRAE_VERBOSE_REASONING": "1"}, clear=False
        ):
            self.assertTrue(sse.full_reasoning_enabled())
            self.assertEqual(
                sse.compact_reasoning_text(self.reasoning), self.reasoning
            )
            self.assertEqual(raw_client.response_style_instruction(), "")

    def test_responses_visibility_defaults_and_explicit_disable_precedence(self):
        self.assertFalse(responses_api.thinking_requested({}))
        self.assertTrue(responses_api.reasoning_output_requested({}))
        self.assertTrue(responses_api.reasoning_output_requested({
            "reasoning": {"effort": "high"}
        }))
        for request in (
            {"include_reasoning": False},
            {"thinking": False},
            {"thinking": {"type": "disabled"}},
            {"thinking": True, "include_reasoning": False},
            {"thinking": {"type": "disabled"}, "include_reasoning": True},
        ):
            with self.subTest(request=request):
                self.assertFalse(
                    responses_api.reasoning_output_requested(request)
                )
        self.setting["enabled"] = False
        self.assertFalse(responses_api.reasoning_output_requested({}))
        self.assertTrue(responses_api.reasoning_output_requested({
            "thinking": {"type": "enabled"}
        }))

    def test_responses_stream_and_nonstream_preserve_full_output_but_bound_history(self):
        request = {"store": False}

        def context():
            return responses_api.ResponsesContext(
                response_id="resp-reasoning", model="m", created_at=1,
                request=request,
            )

        result = responses_api.completion_to_response({
            "choices": [{"message": {
                "role": "assistant", "content": "seven",
                "reasoning_content": self.reasoning,
            }, "finish_reason": "stop"}],
        }, context())

        async def source():
            for reasoning in (self.reasoning[:300], self.reasoning):
                yield "data: " + json.dumps({"choices": [{
                    "delta": {"reasoning_content": reasoning},
                    "finish_reason": None,
                }]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{
                "delta": {"content": "seven"}, "finish_reason": "stop",
            }]}) + "\n\n"
            yield "data: [DONE]\n\n"

        wire = asyncio.run(_collect(responses_api.translate_chat_stream(
            source(), context()
        )))
        payloads = _payloads(wire)
        streamed_reasoning = "".join(
            event.get("delta", "")
            for event in payloads
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        self.assertEqual(streamed_reasoning, self.reasoning)
        self.assertEqual(
            "".join(
                event.get("delta", "")
                for event in payloads
                if event.get("type") == "response.output_text.delta"
            ),
            "seven",
        )
        for output in (result["output"], payloads[-1]["response"]["output"]):
            reasoning_item = next(
                item for item in output if item["type"] == "reasoning"
            )
            self.assertEqual(
                reasoning_item["summary"][0]["text"], self.reasoning
            )
            assistant = responses_api._assistant_message_from_output(
                output, context()
            )
            self.assertLessEqual(len(assistant["reasoning_content"]), 800)
            self.assertLessEqual(
                len(assistant["reasoning_content"].splitlines()), 6
            )

    def test_responses_explicit_disabled_drops_reasoning_from_both_output_modes(self):
        for request in (
            {"include_reasoning": False},
            {"thinking": {"type": "disabled"}},
        ):
            with self.subTest(request=request):
                context = responses_api.ResponsesContext(
                    response_id="resp-disabled", model="m", created_at=1,
                    request={**request, "store": False},
                )
                completion = {"choices": [{"message": {
                    "content": "seven", "reasoning_content": self.reasoning,
                }}]}
                result = responses_api.completion_to_response(
                    completion, context
                )

                async def source():
                    yield "data: " + json.dumps({"choices": [{
                        "delta": completion["choices"][0]["message"],
                        "finish_reason": "stop",
                    }]}) + "\n\n"
                    yield "data: [DONE]\n\n"

                wire = asyncio.run(_collect(
                    responses_api.translate_chat_stream(source(), context)
                ))
                payloads = _payloads(wire)
                self.assertFalse(any(
                    item["type"] == "reasoning" for item in result["output"]
                ))
                self.assertFalse(any(
                    event["type"].startswith("response.reasoning")
                    for event in payloads
                ))
                self.assertNotIn(self.reasoning, json.dumps(payloads))

    def test_remote_stream_and_nonstream_keep_full_reasoning_separate(self):
        async def events():
            yield "plan_item", {
                "id": "p", "thought": "", "reasoning_content": self.reasoning[:300]
            }
            yield "plan_item", {
                "id": "p", "thought": "seven", "reasoning_content": self.reasoning
            }
            yield "done", {}

        wire = asyncio.run(_collect(sse.translate_web_events(
            events(), "m", include_reasoning=True
        )))
        result = asyncio.run(sse.collect_nonstream_web(
            events(), "m", include_reasoning=True
        ))
        self.assertEqual(_chat_text(wire, "content"), "seven")
        self.assertEqual(_chat_text(wire, "reasoning_content"), self.reasoning)
        self.assertEqual(result["choices"][0]["message"], {
            "role": "assistant", "content": "seven",
            "reasoning_content": self.reasoning,
        })

    def test_full_mode_does_not_override_disabled_request_reasoning(self):
        async def events():
            yield "plan_item", {
                "id": "p", "thought": "seven", "reasoning_content": self.reasoning
            }
            yield "done", {}

        wire = asyncio.run(_collect(sse.translate_web_events(
            events(), "m", include_reasoning=False
        )))
        result = asyncio.run(sse.collect_nonstream_web(
            events(), "m", include_reasoning=False
        ))
        self.assertEqual(_chat_text(wire, "content"), "seven")
        self.assertEqual(_chat_text(wire, "reasoning_content"), "")
        self.assertNotIn("reasoning_content", result["choices"][0]["message"])

    def test_ide_stream_and_nonstream_preserve_full_reasoning(self):
        def response():
            return _Lines([
                "data: " + json.dumps({
                    "reasoning_content": self.reasoning[:300], "response": ""
                }),
                "data: " + json.dumps({
                    "reasoning_content": self.reasoning, "response": "seven"
                }),
                "data: [DONE]",
            ])

        wire = asyncio.run(_collect(sse.translate_ide_stream(
            response(), "m", include_reasoning=True
        )))
        result = asyncio.run(sse.collect_nonstream_ide(
            response(), "m", include_reasoning=True
        ))
        self.assertEqual(_chat_text(wire, "content"), "seven")
        self.assertEqual(_chat_text(wire, "reasoning_content"), self.reasoning)
        self.assertEqual(
            result["choices"][0]["message"]["reasoning_content"], self.reasoning
        )

    def test_cli_stream_and_nonstream_preserve_full_reasoning(self):
        async def events():
            yield cli_client.CliEvent(
                type="json", data={
                    "message": {"role": "assistant", "content": "seven"},
                    "reasoning_content": self.reasoning,
                }
            )

        wire = asyncio.run(_collect(sse.translate_cli_stream(
            events(), "m", include_reasoning=True
        )))
        result = asyncio.run(sse.collect_nonstream_cli(
            events(), "m", include_reasoning=True
        ))
        self.assertEqual(_chat_text(wire, "content"), "seven")
        self.assertEqual(_chat_text(wire, "reasoning_content"), self.reasoning)
        self.assertEqual(
            result["choices"][0]["message"]["reasoning_content"], self.reasoning
        )

    def test_traework_stream_and_nonstream_do_not_recompact_full_reasoning(self):
        async def source():
            yield "data: " + json.dumps({"choices": [{
                "delta": {"reasoning_content": self.reasoning[:300]},
                "finish_reason": None,
            }]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{
                "delta": {
                    "content": "seven",
                    "reasoning_content": self.reasoning[300:],
                },
                "finish_reason": "stop",
            }]}) + "\n\n"
            yield "data: [DONE]\n\n"

        wire = asyncio.run(_collect(
            traework_compat.translate_openai_stream_to_traework(
                source(), model="m"
            )
        ))
        payloads = _payloads(wire)
        final = payloads[-1]
        self.assertEqual(final["response"], "seven")
        self.assertEqual(final["reasoning_content"], self.reasoning)
        self.assertEqual(
            "".join(item.get("reasoning_delta", "") for item in payloads),
            self.reasoning,
        )
        result = traework_compat.openai_completion_to_traework({
            "choices": [{"message": {
                "role": "assistant", "content": "seven",
                "reasoning_content": self.reasoning,
            }, "finish_reason": "stop"}]
        }, model="m")
        self.assertEqual(result["reasoning_content"], self.reasoning)
        self.assertEqual(result["response"], "seven")

    def test_traework_explicit_disabled_reasoning_is_hidden(self):
        result = traework_compat.openai_completion_to_traework(
            {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "seven",
                    "reasoning_content": self.reasoning,
                }, "finish_reason": "stop"}],
            },
            model="m",
            reasoning_mode="hidden",
        )
        self.assertNotIn("reasoning_content", result)
