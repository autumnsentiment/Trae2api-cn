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
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Optional

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


def resolve_base_args(force_disable_tools: bool = False) -> list[str]:
    raw = _env("TRAE_CLI_ARGS")
    args = _split_args(raw) if raw else ["-p"]
    mode = output_mode()

    if mode != "text":
        if not any(arg == "--json" for arg in args):
            args.append("--json")
    elif not any(arg == "--output-format" for arg in args):
        args.extend(["--output-format", "text"])

    disable_tools_raw = _env("TRAE_CLI_DISABLE_TOOLS", "true")
    if force_disable_tools or _env_bool("TRAE_CLI_DISABLE_TOOLS", True):
        if force_disable_tools:
            tools = list(DEFAULT_DISABLED_TOOLS)
            tools.extend(
                tool.strip()
                for tool in _env("TRAE_CLI_DISALLOWED_TOOLS").split(",")
                if tool.strip()
            )
            if disable_tools_raw.lower() not in (
                "1",
                "true",
                "yes",
                "on",
                "0",
                "false",
                "no",
                "off",
                "none",
            ):
                tools.extend(
                    tool.strip()
                    for tool in disable_tools_raw.split(",")
                    if tool.strip()
                )
        elif disable_tools_raw.lower() in ("1", "true", "yes", "on"):
            raw_tools = _env("TRAE_CLI_DISALLOWED_TOOLS", ",".join(DEFAULT_DISABLED_TOOLS))
            tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
        else:
            # Backward compatible: a comma-separated value can be supplied
            # directly in TRAE_CLI_DISABLE_TOOLS.
            raw_tools = disable_tools_raw
            tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
        for tool in tools:
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


def build_cli_args(prompt: str, model: str, force_disable_tools: bool = False) -> list[str]:
    args = resolve_base_args(force_disable_tools=force_disable_tools)
    args.extend(resolve_model_arg(model))
    if prompt_mode() != "stdin":
        args.append(prompt)
    return args


def renderer_block_text(block: Mapping[str, Any]) -> str:
    """Extract text from a Trae renderer content block.

    The desktop renderer serializes parts as ``{type:"text", value:...}`` and
    tool results as ``{type:"tool_result", value:[{type:"text", value:...}]}``
    (reverse-engineered NPi/$Pi converters).  Those blocks carry ``value``
    rather than ``text``/``content``, so they otherwise read as empty.
    """

    value = block.get("value")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                nested = item.get("text") or item.get("content") or item.get("value")
                if isinstance(nested, str):
                    parts.append(nested)
        return "\n".join(part for part in parts if part)
    return ""


# Backwards-compatible private alias used inside this module.
_renderer_block_text = renderer_block_text


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text in (None, ""):
                    # Trae's renderer model carries tool results as
                    # {type:"tool_result", value:[{type:"text", value:...}]}
                    # and plain text as {type:"text", value:...}.
                    text = _renderer_block_text(block)
                parts.append(text or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _iter_tool_definitions(tools: Any) -> list[dict]:
    if isinstance(tools, list):
        return [tool for tool in tools if isinstance(tool, dict)]
    if not isinstance(tools, dict):
        return []
    result = []
    for name, tool in tools.items():
        if not isinstance(tool, dict):
            continue
        item = dict(tool)
        item.setdefault("name", name)
        result.append(item)
    return result


def build_client_context(client_context: Any = None) -> dict[str, Any]:
    """Normalize the caller-owned terminal context used by CLI fallback."""

    context = dict(client_context) if isinstance(client_context, Mapping) else {}
    context["workspace_path"] = str(
        context.get("workspace_path")
        or context.get("workspacePath")
        or _env("TRAE_CLIENT_WORKSPACE_PATH", r"C:\workspace")
    )
    context["system_type"] = str(
        context.get("system_type")
        or context.get("systemType")
        or _env("TRAE_CLIENT_SYSTEM_TYPE", "Windows")
    )
    terminal_context = context.get("terminal_context", context.get("terminalContext"))
    context["terminal_context"] = terminal_context if terminal_context is not None else []
    context.pop("workspacePath", None)
    context.pop("systemType", None)
    context.pop("terminalContext", None)
    return context


def _client_context_prompt(client_context: Any = None) -> str:
    context = build_client_context(client_context)
    return (
        "The external client's environment is authoritative. "
        "Treat this context as the workspace and terminal where client tools run; "
        "never substitute or describe the relay/Trae CLI process environment. "
        "Client context JSON: "
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    )


def _external_tool_prompt(tools: Any, tool_choice: Any = None) -> str:
    definitions = []
    for tool in _iter_tool_definitions(tools):
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            name = function.get("name")
            description = function.get("description") or ""
            parameters = function.get("parameters") or {"type": "object", "properties": {}}
        else:
            name = tool.get("name")
            description = tool.get("description") or ""
            parameters = tool.get("parameters") or tool.get("inputSchema") or tool.get("input_schema")
            parameters = parameters or {"type": "object", "properties": {}}
        if not isinstance(name, str) or not name.strip():
            continue
        definitions.append({
            "name": name.strip(),
            "description": str(description),
            "parameters": parameters,
        })
    if not definitions:
        if tools is None:
            return ""
        return (
            "External client tool policy is active, but no client tools are available. "
            "Do not execute Trae CLI internal filesystem, shell, edit, write, search, "
            "or task tools. Answer directly without requesting or fabricating a tool result."
        )

    choice_instruction = "Choose a tool only when the task requires it."
    if tool_choice == "none":
        choice_instruction = "Do not call a tool for this turn."
    elif tool_choice == "required":
        choice_instruction = "You must call at least one available tool for this turn."
    elif isinstance(tool_choice, dict):
        selected = tool_choice.get("function") if tool_choice.get("type") == "function" else tool_choice
        if isinstance(selected, dict) and selected.get("name"):
            choice_instruction = f"You must call the tool named {selected['name']} for this turn."

    schema_json = json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))
    return (
        "External client tool-calling mode is active. "
        "Never execute Trae CLI internal filesystem, shell, edit, write, search, or task tools. "
        "The client will execute tools and send their results back. "
        "Treat the declared tools and their namespaces as the automatically discovered local plugin catalog; do not ask the user to list their environment or plugins first. "
        "If tool_search or another plugin discovery tool is available, call it proactively before claiming a capability is unavailable. Use a client shell or environment tool to inspect cwd, OS, runtimes, and executables when those facts matter. "
        f"{choice_instruction} "
        "When a tool is needed, return one or more blocks and no fabricated tool output: "
        '<tool_call>{"id":"call_stable_id","name":"tool_name","arguments":{...}}</tool_call>. '
        "The arguments value must match the selected JSON schema. "
        "After a Tool result message, continue using that result without repeating the same successful tool call or identical final-answer text. "
        "A Tool result is not a final answer while the user's requested checklist has pending steps; continue with the next client tool. Never print internal client-tool history or protocol markers. "
        f"Available external tools: {schema_json}"
    )


def _assistant_tool_history(message: dict) -> list[str]:
    lines: list[str] = []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return lines
    for index, call in enumerate(calls):
        normalized = normalize_tool_call(call, index=index)
        if not normalized:
            continue
        function = normalized["function"]
        lines.append(
            f"Tool call [{normalized['id']}] {function['name']}:\n{function['arguments']}"
        )
    return lines


def build_cli_prompt(
    messages: list[dict],
    tools: Any = None,
    tool_choice: Any = None,
    client_context: Any = None,
) -> str:
    """Serialize OpenAI messages into the plain text prompt Trae CLI expects."""
    max_messages = _env_int("TRAE_CLI_MAX_MESSAGES", 60)
    max_chars = _env_int("TRAE_CLI_MAX_PROMPT_CHARS", 20000)

    non_system = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    if len(non_system) > max_messages:
        keep = set(non_system[-max_messages:])
        messages = [m for i, m in enumerate(messages) if m.get("role") == "system" or i in keep]

    lines: list[str] = []
    tool_prompt = _external_tool_prompt(tools, tool_choice)
    if tool_prompt:
        lines.append(f"System:\n{tool_prompt}")
    if tool_prompt or isinstance(client_context, Mapping):
        lines.append(f"System:\n{_client_context_prompt(client_context)}")
    protected_count = len(lines)
    for message in messages:
        role = message.get("role", "user")
        content = _content_to_text(message.get("content", "")).strip()
        if role == "system":
            if content:
                lines.append(f"System:\n{content}")
        elif role == "assistant":
            if content:
                lines.append(f"Assistant:\n{content}")
            lines.extend(_assistant_tool_history(message))
        elif role == "tool":
            tool_id = message.get("tool_call_id", "") or ""
            name = message.get("name", "") or ""
            header = "Tool result"
            if tool_id:
                header += f" [{tool_id}]"
            if name:
                header += f" {name}"
            lines.append(f"{header}:\n{content or '[empty tool result]'}")
        else:
            if content:
                lines.append(f"User:\n{content}")

    prompt = "\n\n".join(lines).strip() or "Hello"
    if len(prompt) <= max_chars:
        return prompt
    if protected_count:
        protected_block = "\n\n".join(lines[:protected_count])
        # The external tool contract is a safety boundary and must not be
        # discarded when old conversation history is truncated.
        remaining = max_chars - len(protected_block) - 64
        if remaining > 256:
            conversation = "\n\n".join(lines[protected_count:])
            suffix = conversation[-remaining:]
            omitted = max(0, len(conversation) - len(suffix))
            return f"{protected_block}\n\n[Prompt truncated: {omitted} chars omitted]\n{suffix}"
        return protected_block
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


_TEXT_TOOL_BLOCK_RE = re.compile(
    r"<(?P<tag>opencode_tool_call|tool_use|tool_call|tool_cell|bash|read|write|edit|glob|grep|task)>"
    r"(?P<body>[\s\S]*?)</(?P=tag)>",
    flags=re.IGNORECASE,
)
_OPENCODE_TOOL_OPEN_RE = re.compile(r"<opencode_tool_call>", flags=re.IGNORECASE)
_OPENCODE_BAD_CLOSE_RE = re.compile(
    r"\s*</arg_value>\s*[\"'`]*\s*", flags=re.IGNORECASE
)
_NAMED_TOOL_BLOCK_RE = re.compile(
    r"<(?P<tag>tool_call|tool_cell)\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>"
    r"(?P<body>[\s\S]*?)</(?P=tag)>",
    flags=re.IGNORECASE,
)
_KIMI_TOOL_PARAMETER_RE = re.compile(
    r"<tool>\s*(?P<name>[^<]+?)\s*</tool>\s*"
    r"<parameter(?:\s+[^>]*)?>\s*(?P<body>[\s\S]*?)\s*</parameter>",
    flags=re.IGNORECASE,
)

# Some context-compaction paths escape the internal tool protocol one or more
# times before replaying it as assistant text. A projection that ignores those
# escapes and whitespace also covers CR/LF insertion and cross-block splits.
_INTERNAL_WAIT_CANONICAL_RE = re.compile(
    r"(?:<tool_call><[0-9a-f]{8,64}>wait</tool_call>"
    r"(?:<[0-9a-f]{8,64}>)?)+",
    flags=re.IGNORECASE,
)
_INTERNAL_WAIT_OPEN = "<tool_call>"
_INTERNAL_WAIT_CLOSE = "</tool_call>"
_INTERNAL_WAIT_HEX = frozenset("0123456789abcdef")
_HISTORY_TOOL_MARKER_RE = re.compile(
    r"Previous client tool request\(s\):\s*",
    flags=re.IGNORECASE,
)
_HISTORY_TOOL_MARKER_BLOCK_RE = re.compile(
    r"Previous client tool request\(s\):\s*\[[\s\S]*?\]",
    flags=re.IGNORECASE,
)
_CLIENT_TOOL_HISTORY_MARKER_RE = re.compile(
    r"Client tool calls already issued in this conversation\s*"
    r"\(history only;.*?\):",
    flags=re.IGNORECASE | re.DOTALL,
)
_CLIENT_TOOL_HISTORY_CALL_LINE_RE = re.compile(
    r"^[ \t]*Client tool call\s+\[[^\]\r\n]*\][^\r\n]*(?:\r?\n|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_TOOL_USE_TAG_RE = re.compile(r"<(?P<tag>[A-Za-z_][A-Za-z0-9_-]*)>\s*([\s\S]*?)\s*</(?P=tag)>", re.IGNORECASE)


def _project_internal_protocol(text: str) -> tuple[str, list[int]]:
    """Return protocol-significant text plus its indexes in the source."""

    projected: list[str] = []
    source_indexes: list[int] = []
    for index, char in enumerate(text):
        if char == "\\" or char.isspace():
            continue
        projected.append(char.casefold())
        source_indexes.append(index)
    return "".join(projected), source_indexes


def _consume_internal_literal(
    value: str, position: int, literal: str
) -> tuple[int, str]:
    remaining = value[position:]
    if len(remaining) < len(literal):
        if literal.startswith(remaining):
            return len(value), "partial"
        return position, "invalid"
    if value.startswith(literal, position):
        return position + len(literal), "complete"
    return position, "invalid"


def _consume_internal_sentinel(value: str, position: int) -> tuple[int, str]:
    if position >= len(value):
        return position, "partial"
    if value[position] != "<":
        return position, "invalid"
    position += 1
    digits = 0
    while (
        position < len(value)
        and value[position] in _INTERNAL_WAIT_HEX
        and digits < 64
    ):
        position += 1
        digits += 1
    if position >= len(value):
        return position, "partial"
    if digits < 8 or value[position] != ">":
        return position, "invalid"
    return position + 1, "complete"


def _internal_wait_prefix_status(value: str) -> str:
    """Classify a projected suffix as an internal wait frame or its prefix."""

    position = 0
    while True:
        position, status = _consume_internal_literal(
            value, position, _INTERNAL_WAIT_OPEN
        )
        if status != "complete":
            return status
        position, status = _consume_internal_sentinel(value, position)
        if status != "complete":
            return status
        position, status = _consume_internal_literal(value, position, "wait")
        if status != "complete":
            return status
        position, status = _consume_internal_literal(
            value, position, _INTERNAL_WAIT_CLOSE
        )
        if status != "complete":
            return status
        if position == len(value):
            return "complete"

        # The opaque trailing sentinel is optional. A following ``<t...`` is
        # the next frame; a following ``<hex...`` is the trailing sentinel.
        if value[position] != "<":
            return "invalid"
        if position + 1 == len(value):
            return "partial"
        if value[position + 1] in _INTERNAL_WAIT_HEX:
            position, status = _consume_internal_sentinel(value, position)
            if status != "complete":
                return status
            if position == len(value):
                return "complete"


def _internal_wait_residue_spans(text: str) -> list[tuple[int, int]]:
    """Locate complete frames and a trailing truncated frame in source text."""

    projected, source_indexes = _project_internal_protocol(text)
    if not projected:
        return []

    spans: list[tuple[int, int]] = []

    def source_start(projected_index: int) -> int:
        start = source_indexes[projected_index]
        while start > 0 and text[start - 1] == "\\":
            start -= 1
        return start

    for match in _INTERNAL_WAIT_CANONICAL_RE.finditer(projected):
        end = source_indexes[match.end() - 1] + 1
        if match.end() == len(projected):
            # A stream chunk can stop in the escape run immediately before an
            # optional trailing sentinel. Keep that tail with the frame until
            # the next chunk proves whether a sentinel follows.
            end = len(text)
        spans.append(
            (
                source_start(match.start()),
                end,
            )
        )

    # Hold/remove a frame cut at EOF. This is also what prevents the first
    # half of a split streaming frame from becoming an irreversible delta.
    candidate_indexes: list[int] = []
    full_open = projected.rfind(_INTERNAL_WAIT_OPEN)
    if full_open >= 0:
        candidate_indexes.append(full_open)
    for length in range(len(_INTERNAL_WAIT_OPEN) - 1, 0, -1):
        if projected.endswith(_INTERNAL_WAIT_OPEN[:length]):
            candidate_indexes.append(len(projected) - length)
            break
    for index in candidate_indexes:
        if _internal_wait_prefix_status(projected[index:]) == "partial":
            spans.append((source_start(index), len(text)))
            break

    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _remove_text_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = max(cursor, end)
    parts.append(text[cursor:])
    return "".join(parts)


def _strip_internal_wait_residue(text: str) -> str:
    """Remove complete or truncated escaped internal wait frames."""

    return _remove_text_spans(text, _internal_wait_residue_spans(text))


_ASSISTANT_TEXT_KEYS = ("content", "text", "value", "parts")


def _collect_assistant_text_leaves(content: Any, leaves: list[str]) -> None:
    if isinstance(content, str):
        leaves.append(content)
    elif isinstance(content, (list, tuple)):
        for item in content:
            _collect_assistant_text_leaves(item, leaves)
    elif isinstance(content, Mapping):
        for key in _ASSISTANT_TEXT_KEYS:
            if key in content:
                _collect_assistant_text_leaves(content[key], leaves)


def _replace_assistant_text_leaves(content: Any, replacements) -> Any:
    if isinstance(content, str):
        return next(replacements)
    if isinstance(content, list):
        return [
            _replace_assistant_text_leaves(item, replacements) for item in content
        ]
    if isinstance(content, tuple):
        return tuple(
            _replace_assistant_text_leaves(item, replacements) for item in content
        )
    if isinstance(content, Mapping):
        cleaned = dict(content)
        for key in _ASSISTANT_TEXT_KEYS:
            if key in cleaned:
                cleaned[key] = _replace_assistant_text_leaves(
                    cleaned[key], replacements
                )
        return cleaned
    return content


def sanitize_assistant_history_content(content: Any) -> Any:
    """Remove internal wait frames while preserving the caller's content shape."""

    leaves: list[str] = []
    _collect_assistant_text_leaves(content, leaves)
    if not leaves:
        return content

    combined = "".join(leaves)
    spans = _internal_wait_residue_spans(combined)
    if not spans:
        return content

    replacements: list[str] = []
    offset = 0
    for leaf in leaves:
        leaf_end = offset + len(leaf)
        local_spans = [
            (max(start, offset) - offset, min(end, leaf_end) - offset)
            for start, end in spans
            if start < leaf_end and end > offset
        ]
        replacements.append(_remove_text_spans(leaf, local_spans))
        offset = leaf_end
    return _replace_assistant_text_leaves(content, iter(replacements))


def sanitize_assistant_history_text(content: Any) -> str:
    """Return assistant text with context-compaction wait residue removed."""

    return _content_to_text(sanitize_assistant_history_content(content))


def sanitize_assistant_history_messages(messages: Any) -> list[Any]:
    """Sanitize assistant text fields without changing user or tool messages."""

    if not isinstance(messages, list):
        return []
    cleaned_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            cleaned_messages.append(message)
            continue
        cleaned = dict(message)
        for key in ("content", "parts", "text", "prompt", "message", "input"):
            if key in cleaned:
                cleaned[key] = sanitize_assistant_history_content(cleaned[key])
        cleaned_messages.append(cleaned)
    return cleaned_messages

# The raw/CLI upstream sometimes echoes the history envelope that the relay
# put in its prompt.  It is not user-visible assistant text.  In practice the
# model may truncate that envelope halfway through an escaped JSON string, so
# a regular expression that only matches a complete ``[...]`` value is not
# sufficient.  The helpers below intentionally treat an incomplete envelope
# as protocol text up to the next explicit tool block (or EOF).
_HISTORY_MARKER_LITERAL = "Previous client tool request(s):"
_HISTORY_PARTIAL_PREFIX_RE = re.compile(
    r"^\s*Previous(?:\s+client(?:\s+tool(?:\s+request(?:s|\(s\)?)?)?)?)?\s*:?\s*$",
    flags=re.IGNORECASE,
)
_CLIENT_HISTORY_PARTIAL_PREFIX_RE = re.compile(
    r"^\s*Client(?:\s+tool(?:\s+calls(?:\s+already(?:\s+issued(?:\s+in(?:\s+this(?:\s+conversation)?)?)?)?)?)?)?\s*:?\s*$",
    flags=re.IGNORECASE,
)
_HISTORY_PROTOCOL_TAG_RE = re.compile(
    r"<\s*(?:opencode_tool_call|tool_use|tool_call|tool_cell|bash|read|write|edit|glob|grep|task)\b",
    flags=re.IGNORECASE,
)

# The CLI can stop while it is echoing the history preamble.  In that case
# there is no closing ``)`` or ``:`` yet (for example ``Previous client tool
# request(s``).  Keep these as protocol residue even when ordinary answer
# text precedes the fragment.  Matching is deliberately prefix-based so a
# split marker can be held by the streaming accumulator until the next frame.
_HISTORY_MARKER_PREFIXES = tuple(
    marker.casefold()
    for marker in (
        _HISTORY_MARKER_LITERAL,
        "Previous client tool requests:",
        "Previous client tool request:",
    )
)
_HISTORY_MARKER_START_RE = re.compile(
    r"Previous\s+client\s+tool\s+request(?:s|\s*\(\s*s\s*\)?)?\s*:?",
    flags=re.IGNORECASE,
)
_CLIENT_HISTORY_MARKER_LITERAL = (
    "Client tool calls already issued in this conversation "
    "(history only; use the matching results below and do not repeat them):"
)
_CLIENT_HISTORY_MARKER_PREFIX = _CLIENT_HISTORY_MARKER_LITERAL.casefold()


def _is_history_marker_prefix(value: str) -> bool:
    """Return whether *value* is only a (possibly truncated) history marker."""

    candidate = " ".join(str(value).strip().split()).casefold()
    if not candidate:
        return False
    return any(
        marker.startswith(candidate) for marker in _HISTORY_MARKER_PREFIXES
    ) or _CLIENT_HISTORY_MARKER_PREFIX.startswith(candidate)


def _strip_partial_history_fragments(text: str) -> str:
    """Hide marker fragments that are incomplete or split across stream frames.

    ``strip_tool_call_blocks`` historically only removed a complete marker or
    a marker that occupied the whole response.  A CLI cumulative snapshot can
    instead look like ``answer\n\nPrevious client tool request(s``; exposing the
    suffix causes clients to replay the same tool request.  Remove standalone
    marker lines first, then a trailing marker attached after a line boundary
    (or an intentional two-space separator), while preserving the answer.
    """

    if not text:
        return text

    removed_line = False
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if _is_history_marker_prefix(body):
            removed_line = True
            continue
        kept.append(line)
    text = "".join(kept)
    if removed_line:
        # Removing a protocol-only final line should not leave a visible blank
        # line after the real answer.
        text = text.rstrip()

    # A fragment may share a line with the answer.  Only accept a marker at the
    # start of the response, after a newline, or after an explicit two-space
    # separator; do not strip ordinary prose such as ``See Previous``.
    for match in re.finditer(r"(?i)\b(?:previous|client)\b", text):
        before = text[: match.start()]
        if before and not (
            before.endswith(("\n", "\r", "\t", "  "))
        ):
            continue
        candidate = text[match.start() :].strip()
        if _is_history_marker_prefix(candidate):
            return before.rstrip()
    return text


def _decode_json_string_fragment(value: str) -> Optional[str]:
    """Decode a JSON string body when the model returned a closed fragment."""

    try:
        decoded = json.loads('"' + value + '"')
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, str) else None


def _recover_history_marker_calls(tail: str, index: int = 0) -> list[dict]:
    """Recover a call whose outer history array was truncated.

    We only accept a closed ``input`` string containing valid JSON arguments.
    A half-written argument is discarded rather than turned into a fabricated
    call; a native call in the same upstream payload will still be used.
    """

    recovered: list[dict] = []
    pattern = re.compile(
        r'\{\s*"id"\s*:\s*"(?P<id>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"name"\s*:\s*"(?P<name>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"input"\s*:\s*"(?P<input>(?:\\.|[^"\\])*)"',
        flags=re.IGNORECASE,
    )
    for match_index, match in enumerate(pattern.finditer(tail)):
        raw_id = _decode_json_string_fragment(match.group("id"))
        name = _decode_json_string_fragment(match.group("name"))
        input_text = _decode_json_string_fragment(match.group("input"))
        if not raw_id or not name or input_text is None:
            continue
        try:
            json.loads(input_text)
        except (TypeError, ValueError):
            continue
        call = normalize_tool_call(
            {"id": raw_id, "name": name, "input": input_text},
            index=index + match_index,
        )
        if call:
            call["_history_marker"] = True
            recovered.append(call)
    return recovered


def _history_marker_records(text: str) -> list[tuple[int, int, list[dict], bool]]:
    """Return ``(start, end, calls, complete)`` records for echoed history.

    ``end`` excludes any explicit XML tool block that follows an incomplete
    marker, allowing the normal tool-block parser to process that block.
    """

    records: list[tuple[int, int, list[dict], bool]] = []
    cursor = 0
    while True:
        match = _HISTORY_MARKER_START_RE.search(text[cursor:])
        if not match:
            break
        start = cursor + match.start()
        marker_end = cursor + match.end()
        probe = marker_end
        while probe < len(text) and text[probe].isspace():
            probe += 1

        # The serializer normally emits a JSON array.  A truncated echo can
        # omit its opening bracket and start with the first call object, which
        # is still protocol history and must not become assistant content.
        if probe >= len(text) or text[probe] not in "[{":
            line_end = text.find("\n", marker_end)
            end = len(text) if line_end < 0 else line_end + 1
            records.append((start, end, [], False))
            cursor = max(end, marker_end)
            continue

        json_end = _find_json_end(text, probe)
        if json_end >= 0:
            raw = text[probe:json_end]
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, (list, dict)):
                calls: list[dict] = []
                items = parsed if isinstance(parsed, list) else [parsed]
                for index, item in enumerate(items):
                    call = normalize_tool_call(item, index=index)
                    if call:
                        call["_history_marker"] = True
                        calls.append(call)
                record_end = json_end
                # When only the opening ``[`` was lost, consume its orphaned
                # closing bracket as part of the history envelope too.
                if isinstance(parsed, dict):
                    close_probe = json_end
                    while close_probe < len(text) and text[close_probe].isspace():
                        close_probe += 1
                    if close_probe < len(text) and text[close_probe] == "]":
                        record_end = close_probe + 1
                records.append((start, record_end, calls, True))
                cursor = record_end
                continue

        # Incomplete JSON: recover only closed call fragments, and hide the
        # remaining protocol residue.  Preserve a later explicit XML block.
        next_tag = _HISTORY_PROTOCOL_TAG_RE.search(text, probe + 1)
        end = next_tag.start() if next_tag else len(text)
        recovered = _recover_history_marker_calls(text[probe:end])
        records.append((start, end, recovered, False))
        cursor = max(end, marker_end)
    return records


def _string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _stringify_tool_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stable_tool_id(name: str, arguments: str, index: int = 0) -> str:
    digest = hashlib.sha256(f"{name}\0{arguments}\0{index}".encode("utf-8")).hexdigest()[:20]
    return f"call_{digest}"


def normalize_tool_call(raw: Any, index: int = 0, fallback_name: str = "") -> Optional[dict]:
    """Normalize Trae/native/text tool-call shapes to an OpenAI call object.

    This function only serializes a request for the API client. It never executes
    the command or touches a filesystem.
    """
    if not isinstance(raw, dict):
        return None
    function = raw.get("function") if isinstance(raw.get("function"), dict) else None
    if function is None and isinstance(raw.get("function_call"), dict):
        function = raw["function_call"]
    if function is None and isinstance(raw.get("functionCall"), dict):
        function = raw["functionCall"]
    if function is None:
        function = raw
    inherited_synthetic_id = raw.get("_synthetic_id") is True
    # Trae's renderer emits ``toolCallId`` on tool_use parts. Losing it forces
    # a synthetic id, which breaks the client's tool_call_id/tool pairing.
    raw_id = (
        _string_or_empty(raw.get("id"))
        or _string_or_empty(raw.get("tool_call_id"))
        or _string_or_empty(raw.get("toolCallId"))
    )
    if not raw_id and isinstance(function, dict):
        raw_id = _string_or_empty(function.get("id")) or _string_or_empty(
            function.get("toolCallId")
        )
    name = (
        _string_or_empty(function.get("name"))
        or _string_or_empty(raw.get("name"))
        or _string_or_empty(raw.get("tool_name"))
        or _string_or_empty(raw.get("server_name"))
        or _string_or_empty(fallback_name)
    )
    if not name and not raw_id and not isinstance(raw.get("index"), int):
        return None
    if name.lower() in {"finish", "done", "final"}:
        return None
    missing = object()
    arguments: Any = missing
    # Trae's native stream uses both ``arguments`` and ``args`` inside
    # ``function_call``. Preserve an explicitly empty first fragment so a
    # later index-only argument delta is not prefixed with a fabricated ``{}``.
    for source in (function, raw):
        for key in ("arguments", "args", "input", "params", "parameters"):
            if key in source:
                arguments = source.get(key)
                break
        if arguments is not missing:
            break
    # A tool-call record may put the actual fields at the top level. Remove
    # protocol metadata before treating that record as arguments.
    if arguments is missing:
        arguments = {
            key: value
            for key, value in raw.items()
            if key not in {
                "id", "type", "name", "tool_name", "server_name", "index",
                "function", "function_call", "functionCall",
                "tool_call_id", "toolCallId",
            }
        }
    arguments_text = _stringify_tool_arguments(arguments)
    call_id = raw_id or _stable_tool_id(name, arguments_text, index)
    result = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments_text},
        "_synthetic_id": inherited_synthetic_id or not bool(raw_id),
        "_source_index": raw.get("_source_index", index),
        "_explicit_index": raw.get("_explicit_index") is True
        or isinstance(raw.get("index"), int),
    }
    if isinstance(raw.get("index"), int):
        result["index"] = raw["index"]
    return result


def tool_call_signature(call: Any) -> str:
    """Return a stable name/arguments key for duplicate-call protection."""

    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    if not isinstance(function, dict):
        function = call
    name = _string_or_empty(function.get("name"))
    if not name:
        return ""
    arguments = function.get("arguments")
    if arguments is None:
        arguments = function.get("input")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            arguments = arguments.strip()
    try:
        encoded = json.dumps(
            arguments if arguments is not None else {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        encoded = str(arguments)
    return f"{name}\0{encoded}"


_TOOL_FAILURE_STATUSES = {
    "error",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
    "timed-out",
}


def _tool_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text")
                if text not in (None, ""):
                    parts.append(str(text))
                elif item.get("content") not in (None, ""):
                    parts.append(str(item["content"]))
                else:
                    nested = _renderer_block_text(item)
                    if nested:
                        parts.append(nested)
            elif item not in (None, ""):
                parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(value, Mapping):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(value)
    return "" if value is None else str(value)


def tool_result_is_failed(message: Mapping[str, Any]) -> bool:
    """Return whether a tool result explicitly reports a failed execution."""

    if not isinstance(message, Mapping):
        return False
    for key in ("is_error", "isError", "failed"):
        value = message.get(key)
        if value is True or (
            isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}
        ):
            return True
    success = message.get("success")
    if success is False or (
        isinstance(success, str) and success.strip().lower() in {"false", "0", "no"}
    ):
        return True
    status = str(message.get("status") or "").strip().lower()
    if status in _TOOL_FAILURE_STATUSES:
        return True

    content = message.get("content")
    # Trae's renderer marks failures on the tool_result block itself
    # ({type:"tool_result", isError:true}), not on the enclosing message.
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            for key in ("is_error", "isError"):
                flag = block.get(key)
                if flag is True or (
                    isinstance(flag, str)
                    and flag.strip().lower() in {"true", "1", "yes"}
                ):
                    return True
    text = _tool_content_text(content)
    parsed: Any = None
    if text:
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
    if isinstance(parsed, Mapping):
        parsed_status = str(parsed.get("status") or "").strip().lower()
        if parsed_status in _TOOL_FAILURE_STATUSES:
            return True
        if parsed.get("is_error") is True or parsed.get("isError") is True:
            return True
        if parsed.get("success") is False:
            return True
        if parsed.get("error") not in (None, "", False, {}, []):
            return True

    lowered = text.lower()
    return bool(
        re.match(
            r"^\s*(?:error|failed|failure|exception|traceback|command failed|"
            r"工具调用.*(?:失败|错误|技术问题)|(?:失败|错误|异常))",
            lowered,
        )
    )


def repair_tool_call_history(
    messages: Any, *, known_call_ids: Optional[set[str]] = None
) -> list[dict[str, Any]]:
    """Insert explicit failed results for assistant calls missing a response."""

    if not isinstance(messages, list):
        return []
    source = [dict(item) for item in messages if isinstance(item, Mapping)]
    repaired: list[dict[str, Any]] = []
    inserted = 0
    index = 0
    while index < len(source):
        message = source[index]
        repaired.append(message)
        index += 1
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        expected: list[tuple[str, str]] = []
        for raw in calls:
            if not isinstance(raw, Mapping):
                continue
            function = raw.get("function") if isinstance(raw.get("function"), Mapping) else raw
            call_id = str(raw.get("id") or raw.get("tool_call_id") or "").strip()
            name = str(function.get("name") or "tool").strip()
            if call_id:
                expected.append((call_id, name))
        if not expected:
            continue

        contiguous: list[dict[str, Any]] = []
        while index < len(source) and source[index].get("role") in {"tool", "function"}:
            contiguous.append(source[index])
            index += 1
        repaired.extend(contiguous)
        seen = {
            str(item.get("tool_call_id") or item.get("toolCallId") or "").strip()
            for item in contiguous
        }
        for call_id, name in expected:
            if known_call_ids is not None and call_id not in known_call_ids:
                continue
            if call_id in seen:
                continue
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": (
                        "[relay] Missing tool result for this call. The tool "
                        "execution did not complete; treat it as failed and "
                        "continue without repeating the same call blindly."
                    ),
                    "is_error": True,
                }
            )
            inserted += 1
    if inserted:
        logger.warning("repaired %d missing tool result(s) in request history", inserted)
    return repaired


def completed_tool_signatures(messages: Any) -> set[str]:
    """Find successful tool calls that should not be blindly repeated.

    A repeated call is allowed when a new user message appears after its
    result, which represents an explicit request to perform the operation
    again.  Normal assistant -> tool-result continuation turns have no such
    user message and are therefore protected.
    """

    if not isinstance(messages, list):
        return set()
    call_signatures: dict[str, str] = {}
    tool_indexes: dict[str, int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for raw in calls:
                    call = normalize_tool_call(raw, index=index)
                    if call and call.get("id"):
                        signature = tool_call_signature(call)
                        if signature:
                            call_signatures[str(call["id"])] = signature
        elif message.get("role") in {"tool", "function"}:
            call_id = _string_or_empty(
                message.get("tool_call_id")
                or message.get("toolCallId")
                or message.get("id")
            )
            if call_id and call_id in call_signatures and not tool_result_is_failed(message):
                tool_indexes[call_id] = index

    protected: set[str] = set()
    for call_id, result_index in tool_indexes.items():
        signature = call_signatures.get(call_id)
        if not signature:
            continue
        if any(
            isinstance(message, dict)
            and message.get("role") == "user"
            for message in messages[result_index + 1 :]
        ):
            continue
        protected.add(signature)
    return protected


def _parse_text_tool_body(body: str, fallback_name: str = "", index: int = 0) -> Optional[dict]:
    raw = body.strip()
    if not raw:
        return None
    # Preferred format: a JSON object with name/tool and input/arguments.
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        call = normalize_tool_call(parsed, index=index, fallback_name=fallback_name)
        if call:
            return call

    # Trae XML format: <tool_name>Read</tool_name><input>{...}</input>.
    name_match = re.search(r"<(?:tool_name|server_name|name)>\s*([^<]+?)\s*</(?:tool_name|server_name|name)>", raw, re.I)
    name = _string_or_empty(name_match.group(1)) if name_match else fallback_name
    input_match = re.search(r"<input>\s*([\s\S]*?)\s*</input>", raw, re.I)
    if input_match:
        call = normalize_tool_call(
            {"name": name, "arguments": input_match.group(1).strip()},
            index=index,
            fallback_name=name,
        )
        if call:
            return call

    # Kimi/Trae parameter forms: <tool>Bash</tool><parameter>{...}</parameter>.
    tool_match = re.search(r"<tool>\s*([^<]+?)\s*</tool>", raw, re.I)
    parameter_match = re.search(r"<parameter(?:\s+[^>]*)?>\s*([\s\S]*?)\s*</parameter>", raw, re.I)
    if tool_match and parameter_match:
        return normalize_tool_call(
            {"name": tool_match.group(1), "arguments": parameter_match.group(1).strip()},
            index=index,
        )

    # Named XML fields, e.g. <command>dir</command> or <filePath>x</filePath>.
    if name:
        fields: dict[str, str] = {}
        for field_match in _TOOL_USE_TAG_RE.finditer(raw):
            key = field_match.group("tag")
            if key.lower() in {"tool_name", "server_name", "input", "parameter", "tool"}:
                continue
            value = field_match.group(2).strip()
            if value:
                fields[key] = value
        for field_match in re.finditer(
            r"<(?:arg|parameter)\s+name=[\"']([^\"']+)[\"']\s*>\s*([\s\S]*?)\s*</(?:arg|parameter)>",
            raw,
            re.I,
        ):
            key = field_match.group(1).strip()
            value = field_match.group(2).strip()
            if key and value:
                fields[key] = value
        if fields:
            return normalize_tool_call({"name": name, "arguments": fields}, index=index)
    return None


def _recover_malformed_opencode_tool_blocks(
    text: str,
    occupied: list[tuple[int, int]],
) -> list[tuple[dict, int, int]]:
    """Recover a complete JSON call followed by Trae's wrong closing tag."""

    recovered: list[tuple[dict, int, int]] = []
    decoder = json.JSONDecoder()
    for opening in _OPENCODE_TOOL_OPEN_RE.finditer(text):
        if any(start <= opening.start() < end for start, end in occupied):
            continue
        body_start = opening.end()
        body = text[body_start:]
        leading = len(body) - len(body.lstrip())
        try:
            raw, consumed = decoder.raw_decode(body[leading:])
        except (TypeError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        json_end = body_start + leading + consumed
        trailing = text[json_end:]
        bad_close = _OPENCODE_BAD_CLOSE_RE.match(trailing)
        if bad_close is not None:
            block_end = json_end + bad_close.end()
        elif not trailing.strip():
            block_end = len(text)
        else:
            continue
        call = normalize_tool_call(raw, index=len(recovered))
        if call:
            recovered.append((call, opening.start(), block_end))
    return recovered


def extract_text_tool_calls(content: Any) -> list[dict]:
    """Extract tool calls encoded in Trae's text/XML fallback formats."""
    text = _strip_internal_wait_residue(_content_to_text(content))
    if not text:
        return []
    calls: list[dict] = []
    occupied: list[tuple[int, int]] = []
    # A model may echo the relay's serialized history marker verbatim.  That is
    # protocol residue, not a fresh client-tool request: executing it again is
    # the source of the ``Previous client`` loop.  Mark the whole envelope as
    # occupied so it is removed from visible text, but only parse real native
    # tool blocks below.
    for start, end, marker_calls, _complete in _history_marker_records(text):
        occupied.append((start, end))
    for match in _TEXT_TOOL_BLOCK_RE.finditer(text):
        tag = match.group("tag").lower()
        call = _parse_text_tool_body(match.group("body"), fallback_name=tag, index=len(calls))
        if call:
            calls.append(call)
            occupied.append((match.start(), match.end()))
    for match in _NAMED_TOOL_BLOCK_RE.finditer(text):
        call = _parse_text_tool_body(match.group("body"), fallback_name=match.group("name"), index=len(calls))
        if call:
            calls.append(call)
            occupied.append((match.start(), match.end()))
    for match in _KIMI_TOOL_PARAMETER_RE.finditer(text):
        call = _parse_text_tool_body(
            match.group("body"),
            fallback_name=match.group("name"),
            index=len(calls),
        )
        if call:
            calls.append(call)
            occupied.append((match.start(), match.end()))

    for call, start, end in _recover_malformed_opencode_tool_blocks(text, occupied):
        call["_source_index"] = len(calls)
        calls.append(call)
        occupied.append((start, end))

    # A compact <tool_call> form can be split by malformed model output. Look
    # for a JSON object containing a name/arguments pair before the closing tag.
    for match in re.finditer(r"(\{[\s\S]*?\})\s*</tool_call>", text, re.I):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        call = _parse_text_tool_body(match.group(1), index=len(calls))
        if call:
            calls.append(call)

    deduped: dict[str, dict] = {}
    for call in calls:
        # Prefer a native/XML call over the same id echoed in a history
        # envelope.  The latter is only a compatibility fallback.
        call_id = call["id"]
        previous = deduped.get(call_id)
        if previous is not None and previous.get("_history_marker") and not call.get(
            "_history_marker"
        ):
            deduped[call_id] = call
        else:
            deduped.setdefault(call_id, call)
    return list(deduped.values())


def _iter_native_tool_calls(result: dict) -> list[dict]:
    candidates: list[Any] = []
    for key in ("tool_calls", "tool_call", "function_call"):
        value = result.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, dict):
            candidates.append(value)
    message = result.get("message")
    if isinstance(message, dict):
        for key in ("tool_calls", "tool_call", "function_call"):
            value = message.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif isinstance(value, dict):
                candidates.append(value)
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {
                    "tool_use", "tool_call", "function_call", "function",
                }:
                    candidates.append(block)

    # Trae CLI often wraps the final assistant message in agent_states.
    states = result.get("agent_states")
    response_meta = message.get("response_meta") if isinstance(message, dict) else None
    finish_reason = response_meta.get("finish_reason") if isinstance(response_meta, dict) else None
    visible_message_text = ""
    if isinstance(message, dict):
        visible_message_text = strip_tool_call_blocks(message.get("content")).strip()
    use_agent_states = not visible_message_text or finish_reason in {
        "tool_calls", "tool-calls", "function_call", "tool_use",
    }
    if use_agent_states and isinstance(states, list):
        for state in reversed(states):
            if not isinstance(state, dict):
                continue
            messages = state.get("messages")
            if not isinstance(messages, list):
                continue
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    value = message.get("tool_calls")
                    if isinstance(value, list):
                        candidates.extend(value)
                    break
            if candidates:
                break

    calls: list[dict] = []
    for index, candidate in enumerate(candidates):
        call = normalize_tool_call(candidate, index=index)
        if call:
            calls.append(call)
    return calls


def extract_tool_calls(result: Any) -> list[dict]:
    """Extract native or text-encoded calls from a CLI/upstream payload."""
    if not isinstance(result, dict):
        return extract_text_tool_calls(result)
    calls = _iter_native_tool_calls(result)
    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else result.get("content")
    calls.extend(extract_text_tool_calls(content))
    if not calls:
        calls.extend(extract_text_tool_calls(result.get("response")))

    deduped: dict[str, dict] = {}
    for call in calls:
        deduped[call["id"]] = call
    return list(deduped.values())


def strip_tool_call_blocks(content: Any) -> str:
    """Remove serialized tool-call blocks from visible assistant text."""
    text = _content_to_text(content)
    if not text:
        return ""
    text = _strip_internal_wait_residue(text)
    text = _TEXT_TOOL_BLOCK_RE.sub("", text)
    text = _NAMED_TOOL_BLOCK_RE.sub("", text)
    text = _KIMI_TOOL_PARAMETER_RE.sub("", text)
    recovered = _recover_malformed_opencode_tool_blocks(text, [])
    for _call, start, end in reversed(recovered):
        text = text[:start] + text[end:]
    # Raw-chat history is serialized as a readable line so the upstream can
    # preserve the call/result relationship.  If the model echoes that line,
    # it is protocol residue, never assistant content.
    text = _CLIENT_TOOL_HISTORY_MARKER_RE.sub("", text)
    text = _CLIENT_TOOL_HISTORY_CALL_LINE_RE.sub("", text)
    # Remove complete *and truncated* echoed history envelopes.  Work from the
    # end so offsets remain valid when multiple markers occur in one response.
    for start, end, _calls, _complete in reversed(_history_marker_records(text)):
        prefix = text[:start]
        suffix = text[end:]
        text = (prefix.rstrip() if not suffix else prefix) + suffix
    text = re.sub(r"<tool_calls>\s*</tool_calls>", "", text, flags=re.I)
    # A hard upstream cutoff can leave only the first word(s) of the marker
    # (for example ``Previous`` or ``Previous client``).  When that residue is
    # the complete visible payload it is protocol noise, not an answer.
    if _HISTORY_PARTIAL_PREFIX_RE.fullmatch(text):
        return ""
    if _CLIENT_HISTORY_PARTIAL_PREFIX_RE.fullmatch(text):
        return ""
    text = re.sub(
        r"^\s*Previous(?:\s+client(?:\s+tool(?:\s+request(?:\(s\))?)?)?)?\s*:?\s*(?=<)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _strip_partial_history_fragments(text)
    return re.sub(r"^\s*---\s*$", "", text, flags=re.M)


class ProtocolTextFilter:
    """Filter echoed client-tool history across arbitrary stream chunks."""

    _MARKER = _HISTORY_MARKER_LITERAL

    def __init__(self) -> None:
        self._pending = ""
        self._suppress = False
        self._array_started = False
        self._depth = 0
        self._in_string = False
        self._escape = False

    def _prefix_length(self, value: str) -> int:
        folded = value.casefold()
        marker = self._MARKER.casefold()
        for length in range(min(len(value), len(marker)), 0, -1):
            if folded.endswith(marker[:length]):
                return length
        return 0

    def feed(self, content: Any) -> str:
        if content is None:
            return ""
        text = _content_to_text(content)
        if not text:
            return ""
        self._pending += text
        output: list[str] = []
        while self._pending:
            if not self._suppress:
                marker_index = self._pending.casefold().find(self._MARKER.casefold())
                if marker_index >= 0:
                    output.append(self._pending[:marker_index])
                    self._pending = self._pending[marker_index + len(self._MARKER):]
                    self._suppress = True
                    self._array_started = False
                    self._depth = 0
                    self._in_string = False
                    self._escape = False
                    continue
                keep = self._prefix_length(self._pending)
                if keep:
                    output.append(self._pending[:-keep])
                    self._pending = self._pending[-keep:]
                else:
                    output.append(self._pending)
                    self._pending = ""
                break

            if not self._array_started:
                array_index = self._pending.find("[")
                if array_index < 0:
                    # A marker without an array is protocol residue. Keep a
                    # bounded tail until the stream ends so split chunks can
                    # still complete the marker without growing memory.
                    self._pending = self._pending[-len(self._MARKER):]
                    break
                self._pending = self._pending[array_index:]
                self._array_started = True

            for index, char in enumerate(self._pending):
                if self._in_string:
                    if self._escape:
                        self._escape = False
                    elif char == "\\":
                        self._escape = True
                    elif char == '"':
                        self._in_string = False
                    continue
                if char == '"':
                    self._in_string = True
                elif char == "[":
                    self._depth += 1
                elif char == "]":
                    self._depth -= 1
                    if self._depth <= 0:
                        self._pending = self._pending[index + 1:]
                        self._suppress = False
                        self._array_started = False
                        self._depth = 0
                        break
            else:
                self._pending = ""
                break
        # A split marker can be emitted into ``output`` when the next frame
        # arrives before the marker has completed (for example two consecutive
        # ``Previous client tool request(s`` fragments).  Sanitize each return
        # value as well as ``flush()`` so protocol residue never reaches the
        # public stream between frames.
        return _strip_partial_history_fragments("".join(output))

    def flush(self) -> str:
        """Return safe trailing text; incomplete protocol is never exposed."""

        if self._suppress:
            self._pending = ""
            return ""
        text = self._pending
        self._pending = ""
        return strip_tool_call_blocks(text)


def extract_result_text(result: dict) -> str:
    """Extract final text from a Trae CLI JSON result."""
    message = result.get("message")
    if not isinstance(message, dict):
        message = result
    content = message.get("content", "")
    parts: list[str] = []
    if isinstance(content, str):
        cleaned = strip_tool_call_blocks(strip_think_tags(content))
        if cleaned:
            parts.append(cleaned)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                cleaned = strip_tool_call_blocks(strip_think_tags(block))
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("text", "output_text") and isinstance(block.get("text"), str):
                    cleaned = strip_tool_call_blocks(strip_think_tags(block["text"]))
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

    def optional_num(*keys: str) -> Optional[int | float]:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, value)
        return None

    prompt = num("input_tokens", "input_token", "inputTokens", "prompt_tokens")
    completion = num(
        "output_tokens", "output_token", "outputTokens", "completion_tokens"
    )
    total = num("total_tokens", "total_token", "totalTokens")
    if not total:
        total = prompt + completion
    mapped = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    credits = optional_num(
        "credits_consumed",
        "consumed_credits",
        "credit_cost",
        "credits_cost",
        "credits_float",
    )
    if credits is not None:
        mapped["credits_consumed"] = credits
    return mapped


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
    options = options or {}
    command = resolve_cli_command()
    if not command:
        raise CliUnavailableError(
            "Trae CLI not found. Install traecli/trae-cli/traex or set TRAE_CLI_COMMAND."
        )

    tool_policy = (
        bool(options.get("_tool_protocol_requested"))
        or any(
            key in options for key in ("tools", "tool_choice", "parallel_tool_calls")
        )
        or any(
            isinstance(message, dict)
            and (
                message.get("role") == "tool"
                or (
                    message.get("role") == "assistant"
                    and isinstance(message.get("tool_calls"), list)
                    and bool(message["tool_calls"])
                )
            )
            for message in messages
        )
    )
    tool_catalog = (
        options["tools"]
        if "tools" in options
        else options.get("_inherited_tools")
    )
    external_tools = _iter_tool_definitions(tool_catalog)
    client_context = options.get("client_context", options.get("clientContext"))
    prompt_context = (
        client_context
        if client_context is not None
        else ({} if tool_policy else None)
    )
    prompt = build_cli_prompt(
        messages,
        tools=external_tools if tool_policy else None,
        tool_choice=options.get("tool_choice"),
        client_context=prompt_context,
    )
    args = build_cli_args(prompt, model, force_disable_tools=tool_policy)
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
