import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import cli_client

FAKE_CMD = str(Path(__file__).resolve().parent / "fake" / "fake_cli.cmd")


async def _collect_events(agen):
    return [event async for event in agen]


class JsonBufferTests(unittest.TestCase):
    def test_split_json_buffer_complete_and_tail(self):
        buffer = '{"a":1}\r\n{"b":2} {"incomplete":'
        values, tail = cli_client.split_json_buffer(buffer)
        self.assertEqual(values, [{"a": 1}, {"b": 2}])
        self.assertEqual(tail, '{"incomplete":')

    def test_split_json_buffer_empty(self):
        self.assertEqual(cli_client.split_json_buffer(""), ([], ""))

    def test_iter_json_values_skips_log_lines(self):
        text = 'log line\n{"a":1}\ninfo: hi\n[1,2]\n'
        values = cli_client.iter_json_values(text)
        self.assertEqual(values, [{"a": 1}])


class ExtractionTests(unittest.TestCase):
    def test_extract_result_text_blocks(self):
        result = {
            "message": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "output_text", "text": " world"},
                ]
            }
        }
        self.assertEqual(cli_client.extract_result_text(result), "hello\n world")

    def test_extract_result_text_removes_think(self):
        result = {"message": {"content": "<thinking>hidden</thinking>done"}}
        self.assertEqual(cli_client.extract_result_text(result), "done")

    def test_extract_usage_variants(self):
        variants = [
            {"usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}},
            {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            {
                "message": {
                    "response_meta": {
                        "usage": {"inputTokens": 7, "outputTokens": 6, "totalTokens": 13}
                    }
                }
            },
        ]
        for result in variants:
            usage = cli_client.extract_usage(result)
            self.assertIsNotNone(usage)
            self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_extract_usage_missing_returns_none(self):
        self.assertIsNone(cli_client.extract_usage({"message": {"content": "hi"}}))

    def test_extract_text_tool_call(self):
        content = (
            '<tool_call>{"id":"call_1","name":"Bash",'
            '"arguments":{"command":"Get-ChildItem"}}</tool_call>'
        )
        calls = cli_client.extract_text_tool_calls(content)
        self.assertEqual(calls[0]["id"], "call_1")
        self.assertEqual(calls[0]["function"]["name"], "Bash")
        self.assertEqual(calls[0]["function"]["arguments"], '{"command":"Get-ChildItem"}')
        self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_recovers_opencode_call_with_arg_value_closer(self):
        content = (
            'analysis before call\n'
            '<opencode_tool_call>{"id":"call_download","name":"exec_command",'
            '"input":{"cmd":"curl.exe -L https://example.com/a.zip",'
            '"workdir":"C:\\\\workspace"}}</arg_value>'
        )

        calls = cli_client.extract_text_tool_calls(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_download")
        self.assertEqual(calls[0]["function"]["name"], "exec_command")
        self.assertIn("curl.exe", calls[0]["function"]["arguments"])
        self.assertEqual(
            cli_client.strip_tool_call_blocks(content).strip(),
            "analysis before call",
        )

    def test_does_not_recover_opencode_json_followed_by_prose(self):
        content = (
            '<opencode_tool_call>{"id":"call_1","name":"exec_command",'
            '"input":{"cmd":"Get-Date"}} this is only an example'
        )

        self.assertEqual(cli_client.extract_text_tool_calls(content), [])

    def test_extract_trae_native_function_call_argument_delta(self):
        head = cli_client.extract_tool_calls(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_native_1",
                        "type": "function",
                        "function_call": {"name": "Read", "args": ""},
                    }
                ]
            }
        )
        delta = cli_client.extract_tool_calls(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "",
                        "type": "function",
                        "function_call": {
                            "name": "",
                            "arguments": '{"filePath":"README.md"}',
                        },
                    }
                ]
            }
        )

        self.assertEqual(len(head), 1)
        self.assertEqual(head[0]["id"], "call_native_1")
        self.assertEqual(head[0]["index"], 0)
        self.assertEqual(head[0]["function"], {"name": "Read", "arguments": ""})
        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]["index"], 0)
        self.assertEqual(delta[0]["function"]["name"], "")
        self.assertEqual(
            delta[0]["function"]["arguments"], '{"filePath":"README.md"}'
        )

    def test_echoed_history_marker_is_not_executable(self):
        content = (
            'Previous client tool request(s):\n'
            '[{"id":"3","name":"read_file","input":"{\\"path\\":\\"src/main.py\\"}"}]'
        )
        calls = cli_client.extract_text_tool_calls(content)
        self.assertEqual(calls, [])
        self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_truncated_history_marker_is_hidden_without_truncating_later_tool(self):
        content = (
            'Previous client tool request(s):\n'
            '[{"id":"old","name":"read_file","input":"{\\"path\\":\\"C:\\\\work\\\\'
            '<tool_call>{"id":"new","name":"shell","arguments":{"command":"Get-Date"}}</tool_call>'
        )
        calls = cli_client.extract_text_tool_calls(content)
        self.assertEqual([call["id"] for call in calls], ["new"])
        self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_partial_history_marker_prefix_is_hidden(self):
        for content in ("Previous", "Previous client", "Previous client tool request(s):"):
            self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_partial_history_marker_without_closing_parenthesis_is_hidden_after_answer(self):
        content = "让我执行 sudo 部署脚本。\n\nPrevious client tool request(s"
        self.assertEqual(
            cli_client.strip_tool_call_blocks(content),
            "让我执行 sudo 部署脚本。",
        )

    def test_repeated_partial_history_marker_lines_are_hidden(self):
        content = (
            "让我执行 sudo 部署脚本。\n\n"
            "Previous client tool request(s\n"
            "Previous client tool request(s"
        )
        self.assertEqual(
            cli_client.strip_tool_call_blocks(content),
            "让我执行 sudo 部署脚本。",
        )

    def test_echoed_client_tool_history_is_hidden(self):
        content = (
            "Client tool calls already issued in this conversation "
            "(history only; use the matching results below and do not repeat them):\n"
            "Client tool call [check-code-status] Bash; arguments: "
            "{\"command\":\"git status\",\"description\":\"Check git status\"}\n"
            "The current result is ready."
        )
        self.assertEqual(
            cli_client.strip_tool_call_blocks(content).strip(),
            "The current result is ready.",
        )

    def test_partial_client_tool_history_marker_prefix_is_hidden(self):
        for content in (
            "Client",
            "Client tool calls",
            "Client tool calls already issued in this conversation",
        ):
            self.assertEqual(cli_client.strip_tool_call_blocks(content), "")

    def test_complete_history_marker_is_removed_without_tool_call(self):
        content = (
            'Previous client tool request(s):\n'
            '[{"id":"old","name":"list_dir","input":"{\\"path\\":\\"C:\\\\work\\"}"}]'
        )
        calls = cli_client.extract_text_tool_calls(content)
        self.assertEqual(calls, [])

    def test_protocol_text_filter_hides_history_split_across_stream_chunks(self):
        text_filter = cli_client.ProtocolTextFilter()
        chunks = (
            "Previous",
            ' client tool request(s):\n[{"id":"old","name":"shell"',
            ',"input":"{}"}]',
            "The current result is ready.",
        )
        result = "".join(text_filter.feed(chunk) for chunk in chunks)
        result += text_filter.flush()
        self.assertEqual(result, "The current result is ready.")

    def test_completed_tool_signatures_protect_only_uncontinued_calls(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "list_dir",
                            "arguments": '{"path":"C:\\\\work"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "{\"entries\":[]}",
            },
        ]
        protected = cli_client.completed_tool_signatures(messages)
        self.assertIn(
            'list_dir\x00{"path":"C:\\\\work"}',
            protected,
        )
        messages.append({"role": "user", "content": "Please run it again."})
        self.assertEqual(cli_client.completed_tool_signatures(messages), set())

    def test_failed_tool_result_does_not_protect_duplicate_signature(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command":"New-Item demo.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"status":"failed","error":"PowerShell failed"}',
            },
        ]

        self.assertEqual(cli_client.completed_tool_signatures(messages), set())

    def test_repair_tool_call_history_fills_each_missing_result(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-a",
                        "function": {"name": "Read", "arguments": "{}"},
                    },
                    {
                        "id": "call-b",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-a",
                "content": "ok",
            },
            {"role": "user", "content": "continue"},
        ]

        repaired = cli_client.repair_tool_call_history(messages)

        self.assertEqual([item["role"] for item in repaired], ["assistant", "tool", "tool", "user"])
        inserted = repaired[2]
        self.assertEqual(inserted["tool_call_id"], "call-b")
        self.assertTrue(inserted["is_error"])
        self.assertNotIn(
            cli_client.tool_call_signature(messages[0]["tool_calls"][1]),
            cli_client.completed_tool_signatures(repaired),
        )


class PromptAndArgsTests(unittest.TestCase):
    def test_build_cli_prompt_roles(self):
        messages = [
            {"role": "system", "content": "be short"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "name": "Read", "content": "file"},
        ]
        prompt = cli_client.build_cli_prompt(messages)
        self.assertIn("System:\nbe short", prompt)
        self.assertIn("User:\nhello", prompt)
        self.assertIn("Assistant:\nhi", prompt)
        self.assertIn("Tool result [call_1] Read:\nfile", prompt)

    def test_build_cli_prompt_empty_messages(self):
        self.assertEqual(cli_client.build_cli_prompt([]), "Hello")

    def test_build_cli_prompt_external_tools_and_history(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a client file",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                },
            },
        }]
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"filePath":"README.md"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "Read", "content": "hello"},
        ]
        prompt = cli_client.build_cli_prompt(
            messages,
            tools=tools,
            client_context={
                "workspace_path": r"C:\repo",
                "system_type": "Windows 11",
                "terminal_context": [{"shell": "PowerShell", "cwd": r"C:\repo"}],
            },
        )
        self.assertIn("Never execute Trae CLI internal", prompt)
        self.assertIn('"name":"Read"', prompt)
        self.assertIn(r'C:\\repo', prompt)
        self.assertIn("Windows 11", prompt)
        self.assertIn("PowerShell", prompt)
        self.assertIn("never substitute or describe the relay/Trae CLI", prompt)
        self.assertIn("Tool call [call_1] Read", prompt)
        self.assertIn("Tool result [call_1] Read", prompt)

    def test_empty_tool_policy_disables_internal_tools_in_prompt(self):
        prompt = cli_client.build_cli_prompt(
            [{"role": "user", "content": "answer directly"}],
            tools=[],
            tool_choice="none",
            client_context={},
        )
        self.assertIn("no client tools are available", prompt)
        self.assertIn("Do not execute Trae CLI internal", prompt)

    def test_resolve_model_arg_default(self):
        with patch.dict(
            os.environ,
            {"TRAE_CLI_MODEL_MODE": "default", "TRAE_CLI_MODEL": "claude-3-7"},
            clear=False,
        ):
            self.assertEqual(
                cli_client.resolve_model_arg("auto"),
                ["-c", "default_model=claude-3-7"],
            )

    def test_resolve_model_arg_config(self):
        with patch.dict(os.environ, {"TRAE_CLI_MODEL_MODE": "config"}, clear=False):
            self.assertEqual(
                cli_client.resolve_model_arg("gpt-5"), ["--config", "model.name=gpt-5"]
            )

    def test_resolve_model_arg_flag_and_none(self):
        with patch.dict(os.environ, {"TRAE_CLI_MODEL_MODE": "flag"}, clear=False):
            self.assertEqual(cli_client.resolve_model_arg("gpt-5"), ["--model", "gpt-5"])
        with patch.dict(os.environ, {"TRAE_CLI_MODEL_MODE": "none"}, clear=False):
            self.assertEqual(cli_client.resolve_model_arg("gpt-5"), [])

    def test_build_cli_args(self):
        with patch.dict(
            os.environ,
            {
                "TRAE_CLI_ARGS": "-p",
                "TRAE_CLI_DISABLE_TOOLS": "true",
                "TRAE_CLI_PROMPT_MODE": "arg",
                "TRAE_CLI_MODEL_MODE": "none",
            },
            clear=False,
        ):
            args = cli_client.build_cli_args("hello", "auto")
        self.assertIn("-p", args)
        self.assertIn("--json", args)
        self.assertIn("hello", args)
        self.assertIn("--disallowed-tool", args)

    def test_external_tools_force_disable_cli_tools(self):
        with patch.dict(
            os.environ,
            {
                "TRAE_CLI_ARGS": "-p",
                "TRAE_CLI_DISABLE_TOOLS": "false",
                "TRAE_CLI_DISALLOWED_TOOLS": "",
                "TRAE_CLI_PROMPT_MODE": "stdin",
                "TRAE_CLI_MODEL_MODE": "none",
            },
            clear=False,
        ):
            args = cli_client.build_cli_args("hello", "auto", force_disable_tools=True)
        disabled = [args[i + 1] for i, value in enumerate(args[:-1]) if value == "--disallowed-tool"]
        self.assertEqual(disabled, cli_client.DEFAULT_DISABLED_TOOLS)

    def test_external_tools_merge_configured_disallowed_tools(self):
        with patch.dict(
            os.environ,
            {
                "TRAE_CLI_ARGS": "-p",
                "TRAE_CLI_DISABLE_TOOLS": "false",
                "TRAE_CLI_DISALLOWED_TOOLS": "WebSearch,Read",
                "TRAE_CLI_PROMPT_MODE": "stdin",
                "TRAE_CLI_MODEL_MODE": "none",
            },
            clear=False,
        ):
            args = cli_client.build_cli_args("hello", "auto", force_disable_tools=True)
        disabled = [args[i + 1] for i, value in enumerate(args[:-1]) if value == "--disallowed-tool"]
        self.assertEqual(disabled, [*cli_client.DEFAULT_DISABLED_TOOLS, "WebSearch"])

    def test_resolve_cli_command_uses_env(self):
        with patch.dict(os.environ, {"TRAE_CLI_COMMAND": FAKE_CMD}, clear=False):
            resolved = cli_client.resolve_cli_command()
        self.assertEqual(os.path.normcase(resolved), os.path.normcase(FAKE_CMD))

    def test_resolve_cli_command_missing(self):
        with patch.dict(
            os.environ,
            {"TRAE_CLI_COMMAND": "", "TRAE_CLI_COMMANDS": ""},
            clear=False,
        ):
            resolved = cli_client.resolve_cli_command()
        self.assertIsNone(resolved)


class SubprocessStreamTests(unittest.TestCase):
    def test_stream_cli_chat_fake_subprocess(self):
        with tempfile.TemporaryDirectory() as workdir, patch.dict(
            os.environ,
            {
                "TRAE_CLI_COMMAND": FAKE_CMD,
                "TRAE_CLI_WORKDIR": workdir,
                "TRAE_CLI_PROMPT_MODE": "stdin",
                "TRAE_CLI_OUTPUT_MODE": "json",
                "TRAE_CLI_DISABLE_TOOLS": "false",
                "TRAE_CLI_MAX_CONCURRENCY": "2",
            },
            clear=False,
        ):
            events = asyncio.run(
                _collect_events(
                    cli_client.stream_cli_chat(
                        [{"role": "user", "content": "hello"}], "auto"
                    )
                )
            )

        json_events = [e for e in events if e.type == "json"]
        self.assertGreaterEqual(len(json_events), 2)
        texts = [cli_client.extract_result_text(e.data) for e in json_events]
        self.assertTrue(any("fake reply" in text for text in texts))
        usages = [cli_client.extract_usage(e.data) for e in json_events]
        self.assertIn({"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}, usages)

    def test_stream_cli_chat_passes_client_context_to_prompt(self):
        captured = {}
        original_build_prompt = cli_client.build_cli_prompt

        def capture_prompt(*args, **kwargs):
            captured["client_context"] = kwargs.get("client_context")
            captured["prompt"] = original_build_prompt(*args, **kwargs)
            return captured["prompt"]

        tools = [{
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        client_context = {
            "workspace_path": r"D:\work\app",
            "system_type": "Windows 11",
            "terminal_context": [{"shell": "PowerShell"}],
        }
        with tempfile.TemporaryDirectory() as workdir, patch.dict(
            os.environ,
            {
                "TRAE_CLI_COMMAND": FAKE_CMD,
                "TRAE_CLI_WORKDIR": workdir,
                "TRAE_CLI_PROMPT_MODE": "stdin",
                "TRAE_CLI_OUTPUT_MODE": "json",
                "TRAE_CLI_DISABLE_TOOLS": "false",
                "TRAE_CLI_DISALLOWED_TOOLS": "",
            },
            clear=False,
        ), patch.object(cli_client, "build_cli_prompt", side_effect=capture_prompt):
            asyncio.run(
                _collect_events(
                    cli_client.stream_cli_chat(
                        [{"role": "user", "content": "inspect"}],
                        "auto",
                        options={"tools": tools, "client_context": client_context},
                    )
                )
            )

        self.assertEqual(captured["client_context"], client_context)
        self.assertIn(r'D:\\work\\app', captured["prompt"])
        self.assertIn("PowerShell", captured["prompt"])

    def test_stream_cli_chat_forces_disable_for_empty_tool_policy(self):
        captured = {}
        original_build_args = cli_client.build_cli_args

        def capture_args(*args, **kwargs):
            captured["force_disable_tools"] = kwargs.get("force_disable_tools")
            return original_build_args(*args, **kwargs)

        with tempfile.TemporaryDirectory() as workdir, patch.dict(
            os.environ,
            {
                "TRAE_CLI_COMMAND": FAKE_CMD,
                "TRAE_CLI_WORKDIR": workdir,
                "TRAE_CLI_PROMPT_MODE": "stdin",
                "TRAE_CLI_OUTPUT_MODE": "json",
                "TRAE_CLI_DISABLE_TOOLS": "false",
            },
            clear=False,
        ), patch.object(cli_client, "build_cli_args", side_effect=capture_args):
            asyncio.run(
                _collect_events(
                    cli_client.stream_cli_chat(
                        [{"role": "user", "content": "hello"}],
                        "auto",
                        options={"tools": [], "tool_choice": "none"},
                    )
                )
            )

        self.assertTrue(captured["force_disable_tools"])

    def test_stream_cli_chat_forces_disable_for_tool_history(self):
        captured = {}
        original_build_args = cli_client.build_cli_args

        def capture_args(*args, **kwargs):
            captured["force_disable_tools"] = kwargs.get("force_disable_tools")
            return original_build_args(*args, **kwargs)

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "Read",
                "content": "done",
            },
        ]
        with tempfile.TemporaryDirectory() as workdir, patch.dict(
            os.environ,
            {
                "TRAE_CLI_COMMAND": FAKE_CMD,
                "TRAE_CLI_WORKDIR": workdir,
                "TRAE_CLI_PROMPT_MODE": "stdin",
                "TRAE_CLI_OUTPUT_MODE": "json",
                "TRAE_CLI_DISABLE_TOOLS": "false",
            },
            clear=False,
        ), patch.object(cli_client, "build_cli_args", side_effect=capture_args):
            asyncio.run(
                _collect_events(
                    cli_client.stream_cli_chat(messages, "auto", options={})
                )
            )

        self.assertTrue(captured["force_disable_tools"])


if __name__ == "__main__":
    unittest.main()
