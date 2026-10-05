"""SOLO native tool history and payload contracts, without upstream calls."""

from __future__ import annotations

import copy
import json
import unittest

from src import trae_client


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file on the caller's machine.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _body(messages: list[dict], **options: object) -> dict:
    return trae_client.build_llm_chat_body(
        messages, "glm-5.3", False, options=dict(options)
    )


def _messages(body: dict, role: str) -> list[dict]:
    return [message for message in body["messages"] if message["role"] == role]


def _system_text(body: dict) -> str:
    return "\n".join(
        block["text"]
        for message in _messages(body, "system")
        for block in message.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


class IdeNativeToolTests(unittest.TestCase):
    def test_standard_tool_history_preserves_ids_and_tool_role(self):
        history = [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Project contents"},
        ]
        original = copy.deepcopy(history)
        body = _body(history, tools=TOOLS)

        assistant = _messages(body, "assistant")[0]
        self.assertIsNone(assistant["content"])
        self.assertEqual(
            assistant["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function_call": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        )
        result = _messages(body, "tool")[0]
        self.assertEqual(result["tool_call_id"], "call_1")
        self.assertEqual(result["content"], [{"type": "text", "text": "Project contents"}])
        self.assertNotIn("Client tool history", json.dumps(body))
        self.assertEqual(history, original)
        self.assertTrue(body["stream"])

    def test_history_without_catalog_still_keeps_native_calls(self):
        body = _body(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "previous",
                            "function_call": {"name": "read_file", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "previous", "content": "done"},
            ]
        )
        assistant = _messages(body, "assistant")[0]
        self.assertNotIn("content", assistant)
        self.assertEqual(assistant["tool_calls"][0]["id"], "previous")
        self.assertEqual(
            assistant["tool_calls"][0]["function_call"],
            {"name": "read_file", "arguments": "{}"},
        )
        self.assertEqual(_messages(body, "tool")[0]["tool_call_id"], "previous")
        self.assertNotIn("tools", body)

    def test_legacy_function_call_is_not_lost(self):
        body = _body(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "read_file", "arguments": "{}"},
                },
                {"role": "function", "name": "read_file", "content": "done"},
            ]
        )
        self.assertEqual(
            _messages(body, "assistant")[0]["function_call"],
            {"name": "read_file", "arguments": "{}"},
        )
        self.assertEqual(_messages(body, "function")[0]["name"], "read_file")

    def test_renderer_calls_and_results_become_correlated_native_messages(self):
        body = _body(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "value": "Reading the file."},
                        {
                            "type": "tool_use",
                            "toolCallId": "renderer_1",
                            "name": "read_file",
                            "parameters": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "toolCallId": "renderer_1",
                            "value": [{"type": "text", "value": "File not found"}],
                            "isError": True,
                        },
                        {"type": "text", "value": "Try CONTRIBUTING.md"},
                    ],
                },
            ],
            tools=TOOLS,
        )
        assistant = _messages(body, "assistant")[0]
        self.assertEqual(
            assistant["content"], [{"type": "text", "text": "Reading the file."}]
        )
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], "renderer_1")
        self.assertEqual(json.loads(call["function_call"]["arguments"]), {"path": "README.md"})
        roles = [message["role"] for message in body["messages"]]
        self.assertEqual(roles[-3:], ["assistant", "tool", "user"])
        result = _messages(body, "tool")[0]
        self.assertEqual(result["tool_call_id"], "renderer_1")
        self.assertEqual(result["name"], "read_file")
        self.assertIn("status=failed", result["content"][0]["text"])
        self.assertIn("File not found", result["content"][0]["text"])
        self.assertEqual(
            _messages(body, "user")[-1]["content"],
            [{"type": "text", "text": "Try CONTRIBUTING.md"}],
        )

    def test_renderer_tool_wrapper_preserves_inner_id_and_failure(self):
        body = _body(
            [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "toolCallId": "actual_id",
                            "name": "read_file",
                            "value": "Permission denied",
                            "isError": True,
                        }
                    ],
                }
            ]
        )
        result = _messages(body, "tool")[0]
        self.assertEqual(result["tool_call_id"], "actual_id")
        self.assertIn("status=failed", result["content"][0]["text"])

    def test_standard_failed_result_keeps_correlation_and_failure_status(self):
        body = _body(
            [
                {
                    "role": "tool",
                    "tool_call_id": "failed_id",
                    "name": "read_file",
                    "is_error": True,
                    "content": "No access",
                }
            ]
        )
        result = _messages(body, "tool")[0]
        self.assertEqual(result["tool_call_id"], "failed_id")
        self.assertEqual(result["name"], "read_file")
        self.assertIn("status=failed", result["content"][0]["text"])

    def test_native_call_variants_and_duplicate_renderer_id(self):
        for renderer_type in ("tool_use", "tool_call", "function_call"):
            with self.subTest(renderer_type=renderer_type):
                body = _body(
                    [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "read_file", "arguments": {}},
                                },
                                {"id": "bad", "function": {"name": "  ", "arguments": "{}"}},
                            ],
                            "content": [
                                {
                                    "type": renderer_type,
                                    "toolCallId": "call_1",
                                    "name": "read_file",
                                    "parameters": {},
                                },
                                {
                                    "type": renderer_type,
                                    "toolCallId": "call_2",
                                    "name": "read_file",
                                    "parameters": {"path": "LICENSE"},
                                },
                            ],
                        }
                    ]
                )
                calls = _messages(body, "assistant")[0]["tool_calls"]
                self.assertEqual([call["id"] for call in calls], ["call_1", "call_2"])
                self.assertEqual(
                    json.loads(calls[1]["function_call"]["arguments"]), {"path": "LICENSE"}
                )

    def test_multimodal_content_and_developer_constraints_are_preserved(self):
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,fixture", "detail": "high"},
        }
        body = _body(
            [
                {"role": "system", "content": "System constraint"},
                {"role": "developer", "content": "Developer constraint"},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Describe this"}, image],
                },
            ]
        )
        self.assertIn("System constraint", _system_text(body))
        self.assertIn("Developer constraint", _system_text(body))
        content = _messages(body, "user")[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe this"})
        self.assertEqual(content[1], image)
        content[1]["image_url"]["detail"] = "low"
        self.assertEqual(image["image_url"]["detail"], "high")

    def test_runtime_requests_native_calls_but_keeps_caller_execution_constraints(self):
        catalog = copy.deepcopy(TOOLS)
        body = _body(
            [{"role": "user", "content": "Read README.md"}],
            tools=catalog,
            client_context={"workspace_path": r"C:\caller", "system_type": "Windows"},
            tool_choice="required",
            parallel_tool_calls=False,
        )
        prompt = _system_text(body)
        self.assertIn(r"C:\\caller", prompt)
        self.assertIn("same call id confirms success", prompt)
        self.assertNotIn("never invoke them through function calling", prompt)
        self.assertNotIn("opencode_tool_call", prompt)
        self.assertNotIn("input schemas (JSON)", prompt)
        self.assertEqual(body["tool_choice"], "required")
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(
            json.loads(body["tools"][0]["function"]["parameters"]),
            TOOLS[0]["function"]["parameters"],
        )
        self.assertEqual(catalog, TOOLS)

    def test_none_choice_removes_native_tools_even_with_a_catalog(self):
        for choice in ("none", " NONE ", {"type": "none"}):
            with self.subTest(choice=choice):
                body = _body(
                    [{"role": "user", "content": "Answer directly"}],
                    tools=TOOLS,
                    tool_choice=choice,
                    parallel_tool_calls=False,
                )
                self.assertNotIn("tools", body)
                self.assertNotIn("tool_choice", body)
                self.assertNotIn("parallel_tool_calls", body)
                self.assertIn("Tool choice is none", _system_text(body))
                self.assertIn("No client tools are available", _system_text(body))

    def test_multiple_renderer_results_do_not_merge_their_call_ids(self):
        body = _body(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "toolCallId": "one",
                            "value": "First result",
                        },
                        {
                            "type": "tool_result",
                            "toolCallId": "two",
                            "value": "Second result",
                        },
                    ],
                }
            ]
        )
        results = _messages(body, "tool")
        self.assertEqual([item["tool_call_id"] for item in results], ["one", "two"])
        self.assertEqual(results[0]["content"][0]["text"], "First result")
        self.assertEqual(results[1]["content"][0]["text"], "Second result")

    def test_native_tool_choice_normalization(self):
        choices = [
            ("auto", "auto"),
            ("required", "required"),
            ({"type": "auto"}, "auto"),
            ({"type": "required"}, "required"),
            ({"type": "function", "function": {"name": "read_file"}}, "read_file"),
            ({"name": "read_file"}, "read_file"),
        ]
        for supplied, expected in choices:
            with self.subTest(choice=supplied):
                body = _body(
                    [{"role": "user", "content": "Read"}],
                    tools=TOOLS,
                    tool_choice=supplied,
                    parallel_tool_calls=True,
                )
                self.assertEqual(body["tool_choice"], expected)
                self.assertTrue(body["parallel_tool_calls"])

    def test_explicit_empty_catalog_does_not_restore_inherited_tools(self):
        explicit = _body(
            [{"role": "user", "content": "Continue"}],
            tools=[],
            _inherited_tools=TOOLS,
        )
        self.assertNotIn("tools", explicit)
        inherited = _body(
            [{"role": "user", "content": "Continue"}], _inherited_tools=TOOLS
        )
        self.assertEqual(inherited["tools"][0]["function"]["name"], "read_file")

    def test_legacy_text_conversion_is_unchanged(self):
        converted = trae_client.convert_openai_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            ],
            {"tools": TOOLS},
        )
        self.assertIn("opencode_tool_call", converted[0]["content"])
        self.assertIn("Client tool history", converted[-2]["content"])
        self.assertEqual(converted[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
