import unittest

from src import raw_client


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


class RuntimeToolProtocolTests(unittest.TestCase):
    def test_native_runtime_uses_structured_calls_and_keeps_execution_constraints(self):
        prompt = raw_client.build_runtime_system_prompt(
            TOOLS, {}, "required", False, native_tools=True
        )

        self.assertIn("Use native function calling", prompt)
        self.assertIn("external API client executes", prompt)
        self.assertIn("matching client tool result", prompt)
        self.assertIn("Tool choice is required", prompt)
        self.assertIn("Request at most one tool per turn", prompt)
        self.assertNotIn("<opencode_tool_call>", prompt)
        self.assertNotIn("never invoke them through function calling", prompt)
        self.assertNotIn("Available client tool definitions", prompt)

    def test_text_runtime_keeps_its_existing_envelope(self):
        prompt = raw_client.build_runtime_system_prompt(TOOLS, {})

        self.assertIn("<opencode_tool_call>", prompt)
        self.assertIn("never invoke them through function calling", prompt)
        self.assertIn("Available client tool definitions", prompt)
        self.assertIn('"required":["path"]', prompt)
        self.assertNotIn("Use native function calling", prompt)

    def test_native_none_and_named_choice_are_explicit(self):
        none = raw_client.build_runtime_system_prompt(
            [], {}, "none", native_tools=True
        )
        named = raw_client.build_runtime_system_prompt(
            TOOLS,
            {},
            {"type": "function", "function": {"name": "write_file"}},
            native_tools=True,
        )

        self.assertIn("Tool choice is none: do not request a tool", none)
        self.assertIn("No client tools are available", none)
        self.assertIn("requires the client tool named write_file", named)
