import asyncio
import json
import unittest

from src import sse
from src.cli_client import CliEvent


async def _cli_events():
    yield CliEvent(
        type="json",
        data={"message": {"content": [{"type": "text", "text": "hello"}]}},
    )
    yield CliEvent(
        type="json",
        data={
            "message": {"content": [{"type": "text", "text": "hello world"}]},
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        },
    )


async def _collect(chunks):
    return [chunk async for chunk in chunks]


def _parse_chunks(chunks):
    parsed = []
    done = False
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            done = True
            continue
        if chunk.startswith("data: "):
            parsed.append(json.loads(chunk[6:].strip()))
    return parsed, done


class TranslateCliStreamTests(unittest.TestCase):
    def test_translate_cli_stream(self):
        chunks = asyncio.run(_collect(sse.translate_cli_stream(_cli_events(), "m", True)))
        parsed, done = _parse_chunks(chunks)
        self.assertTrue(done)
        content = "".join(
            choice["delta"].get("content", "")
            for event in parsed
            for choice in event.get("choices", [])
        )
        self.assertEqual(content, "hello world")
        last = parsed[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            last["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    def test_translate_cli_stream_error(self):
        async def events():
            yield CliEvent(type="error", error="boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(
                _collect(sse.translate_cli_stream(events(), "m", True))
            )


class CollectNonstreamCliTests(unittest.TestCase):
    def test_collect_nonstream_cli(self):
        result = asyncio.run(sse.collect_nonstream_cli(_cli_events(), "m"))
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(
            result["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )


if __name__ == "__main__":
    unittest.main()
