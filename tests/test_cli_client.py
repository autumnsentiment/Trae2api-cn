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


if __name__ == "__main__":
    unittest.main()
