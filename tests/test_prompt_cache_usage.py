"""Cache usage must survive Chat, Responses, and Messages translation."""

import json
import unittest

from src import anthropic_api, raw_client, responses_api, sse
from src.token_usage import usage_details


class PromptCacheUsageTests(unittest.TestCase):
    def test_native_details_survive_chat_and_responses(self):
        native = {
            "input_tokens": 200,
            "output_tokens": 40,
            "cache_read_tokens": 100,
            "cache_write_tokens": 20,
            "reasoning_tokens": 30,
        }
        chat = sse._map_usage(native)
        self.assertEqual(chat["prompt_tokens_details"], {
            "cached_tokens": 100, "cache_write_tokens": 20,
        })
        response = responses_api._usage(chat)
        self.assertEqual(response["input_tokens"], 200)
        self.assertEqual(response["input_tokens_details"], chat["prompt_tokens_details"])
        self.assertEqual(response["output_tokens_details"], {"reasoning_tokens": 30})

    def test_standard_nested_details_survive_chat(self):
        original = {
            "input_tokens": 300,
            "input_tokens_details": {"cached_tokens": 120, "cache_write_tokens": 40},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 25},
        }
        result = responses_api._usage(sse._map_usage(original))
        self.assertEqual(result["input_tokens_details"], original["input_tokens_details"])
        self.assertEqual(result["output_tokens_details"], original["output_tokens_details"])

    def test_messages_cache_counts_are_separate_from_uncached_input(self):
        usage = anthropic_api._usage({
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20},
        })
        self.assertEqual(usage, {
            "input_tokens": 80,
            "output_tokens": 40,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 20,
        })
        self.assertEqual(sum(usage[key] for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
        )), 200)

    def test_native_messages_usage_is_not_subtracted_twice(self):
        native = {
            "input_tokens": 80,
            "output_tokens": 40,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 20,
        }
        self.assertEqual(anthropic_api._usage(native), native)

    def test_explicit_zero_and_invalid_details(self):
        self.assertEqual(usage_details({
            "cached_tokens": 0,
            "input_tokens_details": {"cached_tokens": 100},
            "cache_write_tokens": True,
            "cache_creation_input_tokens": float("nan"),
            "output_tokens_details": {"reasoning_tokens": -2},
        }), {"cached_tokens": 0, "reasoning_tokens": 0})
        self.assertNotIn("cache_read_input_tokens", anthropic_api._usage({"prompt_tokens": 20}))

    def test_wrapped_remote_usage_and_aliases_are_unwrapped(self):
        wrapped = {
            "event": "token_usage",
            "data": {
                "token_usage": {
                    "input_tokens": 240,
                    "output_tokens": 12,
                    "prompt_cache_hit_tokens": 180,
                    "prompt_cache_write_tokens": 60,
                }
            },
        }
        mapped = sse._map_usage(wrapped)
        self.assertEqual(mapped["prompt_tokens"], 240)
        self.assertEqual(mapped["prompt_tokens_details"], {
            "cached_tokens": 180,
            "cache_write_tokens": 60,
        })


class PromptPrefixTests(unittest.TestCase):
    def test_equivalent_json_objects_keep_the_runtime_prefix(self):
        tools_a = [{"type": "function", "function": {
            "name": "read_file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string", "description": "File"}},
            },
        }}]
        tools_b = [{"function": {
            "parameters": {"properties": {"path": {"description": "File", "type": "string"}}, "type": "object"},
            "name": "read_file",
        }, "type": "function"}]
        context_a = {"system_type": "Windows", "workspace_path": "C:/app"}
        context_b = {"workspace_path": "C:/app", "system_type": "Windows"}
        self.assertEqual(
            raw_client.build_runtime_system_prompt(tools_a, context_a),
            raw_client.build_runtime_system_prompt(tools_b, context_b),
        )


class StreamingCacheUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_details_reach_completed_responses_event(self):
        _, _, context = responses_api.normalize_request({"model": "auto", "input": "hi"})
        async def chat_stream():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [], "usage": {
                "prompt_tokens": 200, "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 30},
            }}) + "\n\n"
            yield "data: [DONE]\n\n"
        events = [event async for event in responses_api.translate_chat_stream(chat_stream(), context)]
        payloads = [json.loads(line[6:]) for event in events
                    for line in event.splitlines() if line.startswith("data: ")]
        completed = next(payload for payload in payloads
                         if payload.get("type") == "response.completed")
        self.assertEqual(completed["response"]["usage"]["input_tokens_details"]["cached_tokens"], 100)
