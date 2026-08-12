"""Trae CLI subprocess upstream.

Runs a locally installed Trae CLI (`traecli`, `trae-cli`, `traex`) as an OpenAI
backend. JSON mode is the default because it streams structured result objects;
text mode is supported for CLI versions that only print plain text.

The CLI is deliberately run with its file/command tools disabled by default and
inside an isolated working directory, so reverse-proxied clients cannot use the
relay to execute commands on this machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_DISABLED_TOOLS = [
    "Read",
    "Bash",
    "Edit",
    "Replace",
    "Write",
    "Glob",
    "Grep",
    "Task",
]


class CliUnavailableError(RuntimeError):
    """Raised when no usable Trae CLI executable can be found."""


@dataclass
class CliEvent:
    type: str  # json | text | error
    data: Optional[dict] = None
    text: str = ""
    error: str = ""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off", "none")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def _split_args(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=(os.name != "nt"))
    except ValueError:
        return [part for part in raw.split() if part]


def _is_existing_path(candidate: str) -> bool:
    expanded = os.path.expandvars(os.path.expanduser(candidate))
    if "/" in candidate or "\\" in candidate or candidate.endswith((".exe", ".cmd", ".bat")):
        return Path(expanded).is_file()
    return False


def resolve_cli_command() -> Optional[str]:
    """Return an executable command, or None when Trae CLI is not installed."""
    configured = _env("TRAE_CLI_COMMAND") or _env("TRAE_CLI_COMMANDS")
    candidates = []
    if configured:
        candidates = [part for part in configured.split(",") if part.strip()]
    else:
        candidates = ["traecli", "trae-cli", "traex", "trae"]

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        if _is_existing_path(candidate):
            return os.path.expandvars(os.path.expanduser(candidate))
        found = shutil.which(candidate)
        if found:
            return found
    return None


def cli_available() -> bool:
    return resolve_cli_command() is not None


def get_cli_status() -> dict:
    return {
        "available": cli_available(),
        "command": resolve_cli_command() or "",
        "workdir": resolve_workdir(),
        "output_mode": output_mode(),
        "prompt_mode": prompt_mode(),
        "max_concurrency": _env_int("TRAE_CLI_MAX_CONCURRENCY", 2),
        "disable_tools": _env_bool("TRAE_CLI_DISABLE_TOOLS", True),
    }


def resolve_workdir() -> str:
    raw = _env("TRAE_CLI_WORKDIR")
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path.home() / ".trae-cn-relay" / "work"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def output_mode() -> str:
    return (_env("TRAE_CLI_OUTPUT_MODE", "json")).lower()


def prompt_mode() -> str:
    return (_env("TRAE_CLI_PROMPT_MODE", "arg")).lower()


def resolve_model_arg(model: str) -> list[str]:
    """Build the CLI model option. `auto` uses the CLI default model."""
    mode = (_env("TRAE_CLI_MODEL_MODE", "default")).lower()
    if mode == "none":
        return []

    selected = model.strip()
    if not selected or selected.lower() in ("auto", "auto-work", "solo-work"):
        selected = _env("TRAE_CLI_MODEL")
    if not selected:
        return []

    option = _env("TRAE_CLI_MODEL_ARG")
    if mode == "flag":
        option = option or "--model"
        return [option, selected]
    if mode == "config":
        option = option or "--config"
        return [option, f"model.name={selected}"]
    option = option or "-c"
    return [option, f"default_model={selected}"]


def resolve_base_args() -> list[str]:
    raw = _env("TRAE_CLI_ARGS")
    args = _split_args(raw) if raw else ["-p"]
    mode = output_mode()

    if mode != "text":
        if not any(arg == "--json" for arg in args):
            args.append("--json")
    elif not any(arg == "--output-format" for arg in args):
        args.extend(["--output-format", "text"])

    if _env_bool("TRAE_CLI_DISABLE_TOOLS", True):
        raw_tools = _env("TRAE_CLI_DISABLE_TOOLS", ",".join(DEFAULT_DISABLED_TOOLS))
        tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
        for tool in tools:
            if not any(arg == "--disallowed-tool" for arg in args):
                args.extend(["--disallowed-tool", tool])

    # Remove duplicate option/value pairs while preserving order.
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    i = 0
    while i < len(args):
        if i + 1 < len(args) and args[i] in ("-c", "--config", "--model", "--disallowed-tool", "--output-format"):
            key = (args[i], args[i + 1])
            if key not in seen:
                seen.add(key)
                out.extend([args[i], args[i + 1]])
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def build_cli_args(prompt: str, model: str) -> list[str]:
    args = resolve_base_args()
    args.extend(resolve_model_arg(model))
    if prompt_mode() != "stdin":
        args.append(prompt)
    return args


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def build_cli_prompt(messages: list[dict]) -> str:
    """Serialize OpenAI messages into the plain text prompt Trae CLI expects."""
    max_messages = _env_int("TRAE_CLI_MAX_MESSAGES", 60)
    max_chars = _env_int("TRAE_CLI_MAX_PROMPT_CHARS", 20000)

    non_system = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    if len(non_system) > max_messages:
        keep = set(non_system[-max_messages:])
        messages = [m for i, m in enumerate(messages) if m.get("role") == "system" or i in keep]

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = _content_to_text(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"System:\n{content}")
        elif role == "assistant":
            lines.append(f"Assistant:\n{content}")
        elif role == "tool":
            tool_id = message.get("tool_call_id", "") or ""
            name = message.get("name", "") or ""
            header = "Tool result"
            if tool_id:
                header += f" [{tool_id}]"
            if name:
                header += f" {name}"
            lines.append(f"{header}:\n{content}")
        else:
            lines.append(f"User:\n{content}")

    prompt = "\n\n".join(lines).strip() or "Hello"
    if len(prompt) <= max_chars:
        return prompt
    suffix = prompt[-max_chars:]
    omitted = len(prompt) - len(suffix)
    return f"[Prompt truncated: {omitted} chars omitted]\n{suffix}"


def _find_json_start(text: str, offset: int = 0) -> int:
    object_start = text.find("{", offset)
    array_start = text.find("[", offset)
    if object_start < 0:
        return array_start
    if array_start < 0:
        return object_start
    return min(object_start, array_start)


def _find_json_end(text: str, start: int) -> int:
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in ("{", "["):
            stack.append(ch)
        elif ch in ("}", "]"):
            if not stack:
                return -1
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return -1
            if not stack:
                return i + 1
    return -1


def iter_json_values(text: str) -> list[dict]:
    """Parse concatenated JSON objects/arrays, skipping log lines around them."""
    values: list[dict] = []
    cursor = 0
    while cursor < len(text):
        start = _find_json_start(text, cursor)
        if start < 0:
            break
        end = _find_json_end(text, start)
        if end < 0:
            break
        raw = text[start:end]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(parsed, dict):
            values.append(parsed)
        cursor = end
    return values


def split_json_buffer(buffer: str) -> tuple[list[dict], str]:
    """Return (complete values, incomplete tail). Used by the streaming pump."""
    values: list[dict] = []
    cursor = 0
    while cursor < len(buffer):
        start = _find_json_start(buffer, cursor)
        if start < 0:
            return values, ""
        end = _find_json_end(buffer, start)
        if end < 0:
            return values, buffer[start:]
        raw = buffer[start:end]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(parsed, dict):
            values.append(parsed)
        cursor = end
    return values, ""


def strip_think_tags(text: str) -> str:
    import re

    return (
        re.sub(
            r"<(?:think|thinking)\b[^>]*>[\s\S]*?</(?:think|thinking)>",
            "",
            text,
            flags=re.IGNORECASE,
        )
        .replace("<think>", "")
        .replace("</think>", "")
        .replace("<thinking>", "")
        .replace("</thinking>", "")
    )


def extract_result_text(result: dict) -> str:
    """Extract final text from a Trae CLI JSON result."""
    message = result.get("message")
    if not isinstance(message, dict):
        message = result
    content = message.get("content", "")
    parts: list[str] = []
    if isinstance(content, str):
        cleaned = strip_think_tags(content)
        if cleaned:
            parts.append(cleaned)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                cleaned = strip_think_tags(block)
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("text", "output_text") and isinstance(block.get("text"), str):
                    cleaned = strip_think_tags(block["text"])
                    if cleaned:
                        parts.append(cleaned)
    return "\n".join(parts)


def extract_usage(result: dict) -> Optional[dict]:
    """Map Trae CLI usage variants to OpenAI prompt/completion/total tokens."""
    usage = result.get("usage")
    if not isinstance(usage, dict):
        meta = result.get("message", {}).get("response_meta", {})
        usage = meta.get("usage") if isinstance(meta, dict) else None
    if not isinstance(usage, dict):
        return None

    def num(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        return 0

    prompt = num("input_tokens", "inputTokens", "prompt_tokens")
    completion = num("output_tokens", "outputTokens", "completion_tokens")
    total = num("total_tokens", "totalTokens")
    if not total:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


async def _pump_stream(stream, queue: asyncio.Queue, name: str) -> None:
    try:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            await queue.put((name, chunk.decode("utf-8", errors="replace")))
    finally:
        await queue.put((name, None))


async def stream_cli_chat(
    messages: list[dict],
    model: str,
    options: Optional[dict] = None,
) -> AsyncIterator[CliEvent]:
    """Stream Trae CLI JSON/text output as CliEvent objects."""
    del options
    command = resolve_cli_command()
    if not command:
        raise CliUnavailableError(
            "Trae CLI not found. Install traecli/trae-cli/traex or set TRAE_CLI_COMMAND."
        )

    prompt = build_cli_prompt(messages)
    args = build_cli_args(prompt, model)
    workdir = resolve_workdir()
    timeout = float(_env("TRAE_CLI_QUERY_TIMEOUT", "300"))
    use_stdin = prompt_mode() == "stdin"

    semaphore = _get_semaphore()
    async with semaphore:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE if use_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=os.environ.copy(),
            creationflags=creation_flags,
        )

        queue: asyncio.Queue = asyncio.Queue()
        stdout_task = asyncio.create_task(_pump_stream(proc.stdout, queue, "stdout"))
        stderr_task = asyncio.create_task(_pump_stream(proc.stderr, queue, "stderr"))

        if use_stdin:
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                proc.stdin.close()

        stderr_parts: list[str] = []
        pending = ""
        stdout_done = False
        stderr_done = False
        emitted = False

        try:
            while not (stdout_done and stderr_done):
                try:
                    name, chunk = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    raise RuntimeError(f"Trae CLI timed out after {timeout:.0f}s")
                except (asyncio.CancelledError, KeyboardInterrupt):
                    proc.kill()
                    raise

                if chunk is None:
                    if name == "stdout":
                        stdout_done = True
                    else:
                        stderr_done = True
                    continue

                if name == "stderr":
                    stderr_parts.append(chunk)
                    continue

                if output_mode() == "text":
                    if chunk:
                        emitted = True
                        yield CliEvent(type="text", text=chunk)
                    continue

                pending += chunk
                values, pending = split_json_buffer(pending)
                for value in values:
                    emitted = True
                    yield CliEvent(type="json", data=value)

            # Flush a final complete JSON value that arrived exactly at EOF.
            if pending and output_mode() != "text":
                values, _ = split_json_buffer(pending)
                for value in values:
                    emitted = True
                    yield CliEvent(type="json", data=value)

            exit_code = await asyncio.wait_for(proc.wait(), timeout=5)
            stderr = "".join(stderr_parts).strip()
            if stderr:
                logger.warning("trae-cli stderr: %s", stderr[-1200:])
            if exit_code != 0 and not emitted:
                raise RuntimeError(stderr or f"Trae CLI exited with code {exit_code}")
        finally:
            for task in (stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            if proc.returncode is None:
                proc.kill()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


_cli_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _cli_semaphore
    if _cli_semaphore is None:
        _cli_semaphore = asyncio.Semaphore(_env_int("TRAE_CLI_MAX_CONCURRENCY", 2))
    return _cli_semaphore
