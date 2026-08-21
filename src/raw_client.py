"""Direct Trae raw-chat transport.

This module speaks the HTTP protocol used by Trae's terminal client without
running the Trae CLI locally.  It only transports model and tool-call data;
filesystem, shell, edit, and other tools remain owned by the API caller.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx

from . import auth
from .cli_client import strip_tool_call_blocks
from .model_limits import clamp_max_completion_tokens


logger = logging.getLogger(__name__)


RAW_CHAT_ENDPOINT = "/api/agent/v3/llm_utils_chat"
RAW_APP_ID = "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8"
RAW_IDE_VERSION = "3.3.67"
RAW_IDE_VERSION_CODE = "20260401"
RAW_GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "reasoning_effort",
    "stream_options",
    "response_format",
    "service_tier",
    "user",
    "logprobs",
    "top_logprobs",
)


@dataclass(frozen=True)
class RawModel:
    config_name: str
    raw_model_name: str
    display_name: str


@dataclass
class RawChatResponse:
    """Streaming response plus the client that owns its connection."""

    response: httpx.Response
    client: httpx.Client
    auth_token: str = ""

    def close(self) -> None:
        self.response.close()
        self.client.close()

    def __enter__(self) -> "RawChatResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


_KNOWN_MODELS = {
    "auto": RawModel("glm-5.2", "glm-5.2", "GLM-5.2"),
    "coding": RawModel("glm-5.2", "glm-5.2", "GLM-5.2"),
    "glm-5.2": RawModel("glm-5.2", "glm-5.2", "GLM-5.2"),
    "glm-5.3": RawModel("glm-5.3", "glm-5.3", "GLM-5.3"),
    "glm-5.1": RawModel("glm-5.1", "glm-5.1", "GLM-5.1"),
    "kimi-k2.6": RawModel("kimi-k2.6", "kimi-k2.6", "Kimi-K2.6"),
    "kimi-k2.7-code": RawModel(
        "kimi-k2.7-code", "kimi-k2.7-code", "Kimi-K2.7-Code"
    ),
    "deepseek-v4-pro": RawModel(
        "DeepSeek-V4-Pro", "DeepSeek-V4-Pro", "DeepSeek-V4-Pro"
    ),
}


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in options and options[name] is not None:
            return options[name]
    return default


def resolve_raw_model(model: str, options: Optional[Mapping[str, Any]] = None) -> RawModel:
    """Resolve public model names to the names required by raw-chat."""

    options = options or {}
    requested = (model or "auto").strip()
    without_provider = requested[5:] if requested.lower().startswith("trae/") else requested
    known = _KNOWN_MODELS.get(without_provider.lower())
    mapped = known.config_name if known else ""
    if not mapped:
        # Keep the alias table in one place.  Import lazily because
        # ``trae_client`` imports this module for message/tool helpers.
        try:
            from . import trae_client

            mapped = trae_client.convert_model_name(without_provider) or ""
        except Exception:
            mapped = ""
    default_name = mapped or without_provider or "glm-5.2"

    config_name = _option(options, "config_name", "configName")
    if not config_name:
        config_name = default_name

    raw_model_name = _option(options, "raw_model_name", "rawModelName")
    if not raw_model_name:
        raw_model_name = known.raw_model_name if known else config_name

    display_name = _option(options, "display_name", "displayName")
    if not display_name:
        display_name = known.display_name if known else (mapped or without_provider or config_name)

    return RawModel(str(config_name), str(raw_model_name), str(display_name))


def resolve_raw_base_url(options: Optional[Mapping[str, Any]] = None) -> str:
    options = options or {}
    configured = _option(options, "base_url", "baseURL")
    base = configured or os.environ.get("TRAE_RAW_BASE_URL")
    if not base:
        api_host = os.environ.get("TRAE_API_HOST") or ""
        if not api_host:
            try:
                api_host = str(getattr(auth.get_auth(), "host", "") or "")
            except Exception:
                api_host = ""
        # www/api.trae.com.cn is the account/auth host and returns 404 for
        # llm_utils_chat. Only reuse TRAE_API_HOST when it already names the
        # model gateway used by the Trae client.
        if "mchost.guru" in api_host or "trae-api-" in api_host:
            base = api_host
    base = base or "https://trae-api-cn.mchost.guru"
    return str(base).rstrip("/")


def _mapping_value(value: Any, *names: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    for name in names:
        candidate = value.get(name)
        if candidate not in (None, ""):
            return candidate
    return None


def _header_value(headers: Any, *names: str) -> str:
    if not isinstance(headers, Mapping):
        return ""
    lowered = {str(key).lower(): value for key, value in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return str(value)
    return ""


def _infer_system_type(headers: Any) -> str:
    hint = " ".join(
        value
        for value in (
            _header_value(headers, "x-stainless-os", "x-client-os", "sec-ch-ua-platform"),
            _header_value(headers, "user-agent"),
        )
        if value
    ).lower()
    if any(value in hint for value in ("windows", "win32", "win64")):
        return "Windows"
    if any(value in hint for value in ("macos", "mac os", "darwin")):
        return "macOS"
    if "linux" in hint:
        return "Linux"
    return ""


def _infer_client_name(headers: Any) -> str:
    user_agent = _header_value(headers, "user-agent").lower()
    if "codex" in user_agent:
        return "Codex"
    if "trae" in user_agent:
        return "Trae"
    if "opencode" in user_agent:
        return "OpenCode"
    return "OpenAI-compatible client"


def build_tool_discovery(tools: Any) -> dict[str, Any]:
    """Summarize client tools and plugin namespaces without exposing schemas twice."""

    names: list[str] = []
    namespaces: list[str] = []
    for tool in _normalized_tools(tools):
        name = str(tool.get("name") or "").strip()
        if not name or name in names:
            continue
        names.append(name)
        if "__" in name:
            namespace = "__".join(name.split("__")[:-1]).strip("_")
        elif "." in name:
            namespace = name.rsplit(".", 1)[0]
        else:
            namespace = ""
        if namespace and namespace not in namespaces:
            namespaces.append(namespace)

    lowered = {name.lower(): name for name in names}
    search_tools = [
        original
        for lowered_name, original in lowered.items()
        if lowered_name in {"tool_search", "search_tools", "list_tools", "plugin_search"}
        or "tool_search" in lowered_name
    ]
    environment_tools = [
        name
        for name in names
        if any(
            marker in name.lower()
            for marker in (
                "shell",
                "exec",
                "terminal",
                "environment",
                "workspace",
                "filesystem",
                "read_file",
                "list_dir",
                "glob",
            )
        )
    ]
    return {
        "mode": "automatic",
        "declared_tools": names,
        "plugin_namespaces": namespaces,
        "tool_search_available": bool(search_tools),
        "tool_search_tools": search_tools,
        "environment_probe_tools": environment_tools,
    }


def build_client_context(
    options: Optional[Mapping[str, Any]] = None,
    *,
    request_headers: Any = None,
    metadata: Any = None,
) -> dict[str, Any]:
    """Return the external terminal context the model should treat as authoritative."""

    options = options or {}
    raw = _option(options, "client_context", "clientContext", default={})
    context = dict(raw) if isinstance(raw, Mapping) else {}
    explicit_context = bool(context)
    system_type = str(
        context.get("system_type")
        or context.get("systemType")
        or _mapping_value(metadata, "system_type", "systemType", "platform", "os")
        or _infer_system_type(request_headers)
        or os.environ.get("TRAE_CLIENT_SYSTEM_TYPE")
        or "Windows"
    )
    default_workspace = r"C:\workspace" if "windows" in system_type.lower() else "/workspace"
    context["workspace_path"] = str(
        context.get("workspace_path")
        or context.get("workspacePath")
        or _mapping_value(
            metadata,
            "workspace_path",
            "workspacePath",
            "cwd",
            "project_path",
            "projectPath",
        )
        or os.environ.get("TRAE_CLIENT_WORKSPACE_PATH")
        or default_workspace
    )
    context["system_type"] = system_type
    terminal_context = context.get("terminal_context", context.get("terminalContext"))
    if terminal_context is None:
        terminal_context = _mapping_value(metadata, "terminal_context", "terminalContext")
    if terminal_context is None:
        terminal_context = [
            {
                "shell": "PowerShell" if "windows" in system_type.lower() else "bash",
                "cwd": context["workspace_path"],
                "source": "request-inferred",
            }
        ]
    context["terminal_context"] = terminal_context
    if not explicit_context:
        context["client_name"] = _infer_client_name(request_headers)
        context["context_source"] = "request-auto-discovery"
        catalog = (
            options["tools"]
            if "tools" in options
            else options.get("_inherited_tools")
        )
        context["tool_discovery"] = build_tool_discovery(catalog)
    context.pop("workspacePath", None)
    context.pop("systemType", None)
    context.pop("terminalContext", None)
    return context


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, Mapping):
                parts.append(str(block))
                continue
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
            elif block.get("type") in ("image", "image_url", "file"):
                parts.append(f"[Unsupported {block.get('type')} input omitted]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _normalized_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            description = function.get("description", "")
            schema = function.get("parameters") or function.get("input_schema") or {}
        else:
            name = tool.get("name")
            description = tool.get("description", "")
            schema = (
                tool.get("input_schema")
                or tool.get("inputSchema")
                or tool.get("parameters")
                or {}
            )
        if not name:
            continue
        normalized.append(
            {
                "name": str(name),
                "description": str(description or ""),
                "input_schema": schema,
            }
        )
    return normalized


def build_runtime_system_prompt(
    tools: Any,
    client_context: Mapping[str, Any],
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    *,
    native_tools: bool = False,
) -> str:
    """Describe the caller-owned tool runtime and schemas to the raw model."""

    tool_defs = _normalized_tools(tools)
    lines = [
        "You are the language-model backend for an external Trae-compatible terminal client.",
        "The external client owns the authoritative workspace and executes every filesystem, shell, edit, write, browser, and sub-agent tool.",
        "Do not run or simulate tools in the upstream server environment, and do not describe that server's Linux filesystem as the user's environment.",
        "Discover the local environment and installed plugin capabilities automatically from the client context and declared tools; do not ask the user to enumerate them first.",
        "When a client tool_search or plugin discovery tool is available, call it proactively before claiming a capability is unavailable. When local OS, cwd, runtime, or executable facts are needed, use an available client shell or environment tool to inspect them.",
        "Use only the client tools listed below. Request a tool call and wait for the client to return its result; never fabricate command output.",
        "Never repeat a completed tool call with the same name and arguments after a successful result, and never repeat identical final-answer text in a loop.",
        "A tool result is not a final answer when the user's requested checklist still has pending steps; continue with the next needed client tool until the checklist is complete.",
        "Never echo relay-internal tool-history labels or protocol envelopes as an answer. If another tool is needed, emit a native tool call and wait for its result.",
        "After a tool result arrives, continue from that result. For a final answer, do not emit a tool-call block.",
        "Client context (authoritative JSON):",
        json.dumps(client_context, ensure_ascii=False, separators=(",", ":"), default=str),
    ]
    discovery = build_tool_discovery(tools)
    lines.extend(
        [
            "Auto-discovered client tool and plugin catalog (JSON):",
            json.dumps(discovery, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    if tool_defs and not native_tools:
        lines.extend(
            [
                "Available client tool definitions and input schemas (JSON):",
                json.dumps(tool_defs, ensure_ascii=False, separators=(",", ":"), default=str),
            ]
        )
    elif not tool_defs:
        lines.append("No client tools are available for this request; answer directly.")

    if tool_choice == "none":
        lines.append("Tool choice is none: do not request a tool.")
    elif tool_choice == "required":
        lines.append("Tool choice is required: request an available client tool before answering.")
    elif isinstance(tool_choice, Mapping):
        selected = tool_choice.get("function", tool_choice)
        if isinstance(selected, Mapping) and selected.get("name"):
            lines.append(f"Tool choice requires the client tool named {selected['name']}.")
    if parallel_tool_calls is False:
        lines.append("Request at most one tool per turn.")
    return "\n".join(lines)


def _serialize_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        fn = function if isinstance(function, Mapping) else call
        calls.append(
            {
                "id": call.get("id") or call.get("tool_call_id"),
                "name": fn.get("name"),
                "input": fn.get("arguments", fn.get("input", {})),
            }
        )
    if not calls:
        return ""
    return "Client tool history (already handled; do not repeat):\n" + json.dumps(
        calls, ensure_ascii=False, separators=(",", ":"), default=str
    )


def _serialize_tool_call_context(tool_calls: Any) -> str:
    """Render assistant tool history as inert conversation context.

    The raw endpoint only accepts text messages. Dropping the assistant call
    entirely leaves a later tool result orphaned, causing the upstream model to
    request the same tool again. This summary keeps the call/result relationship
    without using an executable XML block or the legacy history marker.
    """

    if not isinstance(tool_calls, list):
        return ""
    lines = [
        "Client tool calls already issued in this conversation "
        "(history only; use the matching results below and do not repeat them):"
    ]
    count = 0
    for call in tool_calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        function = function if isinstance(function, Mapping) else call
        name = str(function.get("name") or call.get("name") or "tool")
        call_id = str(call.get("id") or call.get("tool_call_id") or "unknown")
        arguments = function.get("arguments", function.get("input", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":"), default=str
            )
        lines.append(f"Client tool call [{call_id}] {name}; arguments: {arguments}")
        count += 1
    return "\n".join(lines) if count else ""


def _native_tool_definition(tool: Any) -> Optional[dict[str, Any]]:
    """Convert an OpenAI tool definition to Trae's raw native shape.

    Trae's ``FunctionDefinition.parameters`` field is a JSON *string*, unlike
    OpenAI's object-shaped schema. Keep the complete caller schema and only
    encode that one field required by the upstream protobuf/JSON adapter.
    """

    if not isinstance(tool, Mapping):
        return None
    result = copy.deepcopy(dict(tool))
    function = result.get("function")
    if not isinstance(function, Mapping):
        # Accept the compact function shape used by a few terminal adapters.
        function = {
            key: copy.deepcopy(result.get(key))
            for key in ("name", "description", "parameters", "strict")
            if key in result
        }
        result = {"type": "function", "function": function}
    else:
        result["function"] = copy.deepcopy(dict(function))
        function = result["function"]
    if not function.get("name"):
        return None
    parameters = function.get("parameters")
    if parameters is None:
        parameters = function.get("input_schema") or function.get("inputSchema") or {}
    if not isinstance(parameters, str):
        parameters = json.dumps(
            parameters,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    function["parameters"] = parameters
    function.pop("input_schema", None)
    function.pop("inputSchema", None)
    result["type"] = str(result.get("type") or "function")
    return result


def _native_tool_definitions(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    result: list[dict[str, Any]] = []
    for tool in tools:
        converted = _native_tool_definition(tool)
        if converted is not None:
            result.append(converted)
    return result


def _named_tool_choice(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    function = tool_choice.get("function")
    selected = function if isinstance(function, Mapping) else tool_choice
    name = selected.get("name") if isinstance(selected, Mapping) else None
    return str(name or "").strip()


def _native_function_call(call: Any) -> Optional[dict[str, Any]]:
    """Map an OpenAI assistant tool call to Trae's ``function_call`` field."""

    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    function = function if isinstance(function, Mapping) else call
    name = function.get("name") or call.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    if arguments is None:
        arguments = function.get("input", call.get("input", {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str)
    return {
        "name": str(name),
        "arguments": arguments,
    }


def _native_assistant_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    result: list[dict[str, Any]] = []
    for call in tool_calls:
        function_call = _native_function_call(call)
        if function_call is None:
            continue
        item: dict[str, Any] = {
            "id": str(call.get("id") or call.get("tool_call_id") or ""),
            "type": str(call.get("type") or "function"),
            "function_call": function_call,
        }
        result.append(item)
    return result


def build_raw_messages(
    messages: list[dict[str, Any]], options: Optional[Mapping[str, Any]] = None
) -> list[dict[str, Any]]:
    """Convert OpenAI messages to Trae raw-chat messages without losing tool history."""

    if not messages:
        raise ValueError("messages cannot be empty")
    options = options or {}
    tool_catalog = (
        options["tools"]
        if "tools" in options
        else options.get("_inherited_tools")
    )
    native_tools = bool(
        "tools" in options
        or "_inherited_tools" in options
        or any(
            isinstance(message, Mapping)
            and (
                message.get("role") == "tool"
                or message.get("tool_calls")
                or message.get("function_call")
            )
            for message in messages
        )
    )
    result: list[dict[str, Any]] = []
    if native_tools or "client_context" in options or "clientContext" in options:
        runtime_prompt = build_runtime_system_prompt(
            tool_catalog,
            build_client_context(options),
            options.get("tool_choice"),
            options.get("parallel_tool_calls"),
            native_tools=native_tools,
        )
        if options.get("_recover_suppressed_tool_call"):
            runtime_prompt += (
                " The previous upstream turn attempted only a tool call that "
                "the client has already completed. Use the existing tool "
                "result now. Return the final answer, or request a different "
                "necessary tool call with different arguments; do not repeat "
                "the completed call."
            )
        result.append(
            {"role": "system", "content": [{"type": "text", "text": runtime_prompt}]}
        )

    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content in (None, "", []) and not tool_calls and not message.get("function_call"):
            for key in ("parts", "text", "prompt", "message", "input"):
                candidate = message.get(key)
                if candidate not in (None, "", [], {}):
                    content = candidate
                    break
        text = _content_to_text(content)
        # ZCode may replay a truncated relay history preamble as assistant
        # text.  Forwarding that residue to Trae makes the model echo it again
        # on every continuation.  Only sanitize assistant history; user and
        # tool-result content remains authoritative and untouched.
        if role == "assistant" and text:
            text = strip_tool_call_blocks(text)
        if role == "developer":
            role = "system"
        elif role not in ("system", "user", "assistant"):
            if role == "tool":
                # Trae accepts tool results as a native role with the original
                # call id. This preserves the continuation relationship.
                role = "tool"
            else:
                role = "user"
        native_message: dict[str, Any] = {"role": role}
        if text:
            native_message["content"] = [{"type": "text", "text": text}]
        elif role in ("system", "user"):
            continue
        if role == "assistant":
            native_calls = _native_assistant_tool_calls(tool_calls)
            if not native_calls and isinstance(message.get("function_call"), Mapping):
                native_calls = _native_assistant_tool_calls([message["function_call"]])
            if native_calls:
                native_message["tool_calls"] = native_calls
            if not text and not native_calls:
                continue
        elif role == "tool":
            native_message["tool_call_id"] = str(
                message.get("tool_call_id") or message.get("toolCallId") or ""
            )
            if message.get("name"):
                native_message["name"] = str(message["name"])
            if not text:
                native_message["content"] = [{"type": "text", "text": ""}]
        result.append(native_message)
    return result


def build_raw_chat_body(
    messages: list[dict[str, Any]],
    model: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    options = options or {}
    resolved_model = resolve_raw_model(model, options)
    session_id = session_id or str(
        _option(options, "session_id", "sessionId", default=uuid.uuid4())
    )
    request_id = str(uuid.uuid4())
    body: dict[str, Any] = {
        "messages": build_raw_messages(messages, options),
        "model": resolved_model.raw_model_name,
        "function": str(
            _option(options, "raw_function", "function", default=None)
            or os.environ.get("TRAE_RAW_FUNCTION")
            or "inline_chat"
        ),
        "request_id": request_id,
        "session_id": session_id,
        "stream": True,
    }
    max_tokens = clamp_max_completion_tokens(
        _option(options, "max_tokens", "maxTokens", "max_completion_tokens"),
        resolved_model.raw_model_name,
    )
    if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool) and max_tokens > 0:
        body["max_tokens"] = int(max_tokens)
    tool_catalog = options.get("tools") if "tools" in options else options.get("_inherited_tools")
    tool_choice = options.get("tool_choice")
    selected_tool_name = _named_tool_choice(tool_choice)
    if "tools" in options or "_inherited_tools" in options:
        native_tools = _native_tool_definitions(tool_catalog)
        if selected_tool_name:
            native_tools = [
                tool
                for tool in native_tools
                if str((tool.get("function") or {}).get("name") or "")
                == selected_tool_name
            ]
        body["tools"] = native_tools
    if "tool_choice" in options:
        # Trae's protobuf/JSON adapter declares tool_choice as a string. Map
        # OpenAI's named object to a required choice and expose only that
        # definition upstream, preserving the named-choice semantics.
        body["tool_choice"] = "required" if selected_tool_name else str(tool_choice)
    if "parallel_tool_calls" in options:
        body["parallel_tool_calls"] = bool(options.get("parallel_tool_calls"))
    for key in RAW_GENERATION_FIELDS:
        if key in options and options[key] is not None:
            body[key] = copy.deepcopy(options[key])
    logger.info(
        "raw body prepared id=%s keys=%s tools=%d native_history=%s",
        str(options.get("_relay_request_id") or request_id),
        ",".join(sorted(body.keys())),
        len(body.get("tools") or []),
        any(isinstance(message, Mapping) and (message.get("tool_calls") or message.get("role") == "tool") for message in messages),
    )
    return body


def build_raw_headers(
    base_url: str,
    token: str,
    raw_model: RawModel,
    request_id: str,
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    options = options or {}
    app_id = str(
        _option(options, "app_id", "appId", default=None)
        or os.environ.get("TRAE_RAW_APP_ID")
        or RAW_APP_ID
    )
    ide_version_code = str(
        _option(options, "ide_version_code", "ideVersionCode", default=None)
        or os.environ.get("TRAE_RAW_IDE_VERSION_CODE")
        or os.environ.get("TRAE_IDE_VERSION_CODE")
        or RAW_IDE_VERSION_CODE
    )
    ide_version = str(
        _option(options, "ide_version", "ideVersion", default=None)
        or os.environ.get("TRAE_RAW_IDE_VERSION")
        or os.environ.get("TRAE_IDE_VERSION")
        or RAW_IDE_VERSION
    )
    # Reuse the same device/request headers as the normal IDE client.  The
    # upstream accepts these headers as a client fingerprint; the old raw
    # implementation derived a different fingerprint from the JWT and was
    # rejected intermittently.
    user_id = str(
        _option(options, "_auth_user_id", "auth_user_id", default=None)
        or auth.get_user_id()
        or ""
    )
    try:
        from . import trae_client

        headers = dict(
            trae_client.build_headers(
                token_override=token,
                user_id_override=user_id,
            )
        )
    except Exception:
        headers = {}

    def put(name: str, value: Any) -> None:
        for key in list(headers):
            if key.lower() == name.lower():
                del headers[key]
        headers[name] = str(value)

    put("Authorization", f"Cloud-IDE-JWT {token}")
    put("X-Cloudide-Token", token)
    put("X-Ide-Token", token)
    put("Content-Type", "application/json")
    put("Accept", "text/event-stream")
    put("Connection", "keep-alive")
    put("X-App-Id", app_id)
    put("X-Ide-Version-Code", ide_version_code)
    put("X-Ide-Version", ide_version)
    put("X-Request-Id", request_id)
    if user_id:
        put("X-Uid", user_id)
    else:
        for key in list(headers):
            if key.lower() == "x-uid":
                del headers[key]
    custom_headers = _option(options, "headers", "raw_headers", default={})
    if isinstance(custom_headers, Mapping):
        for key, value in custom_headers.items():
            put(str(key), value)
    return headers


async def send_raw_chat_request(
    messages: list[dict[str, Any]],
    model: str,
    options: Optional[dict[str, Any]] = None,
) -> RawChatResponse:
    """POST a streaming request to Trae raw-chat.

    The returned response uses the synchronous ``httpx`` streaming interface so
    the relay's existing SSE translator can consume ``response.iter_lines()``.
    Call ``RawChatResponse.close()`` after the stream is consumed.
    """

    # The main router binds a conversation to one account.  Keep the token
    # snapshot on retries/stream continuations so another request rotating the
    # global active account cannot silently switch this upstream conversation.
    token = str(options.get("_auth_token") or "").strip() if options else ""
    if not token:
        await auth.maybe_refresh()
        token = auth.get_token()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available for Trae raw chat")

    options = options or {}
    base_url = resolve_raw_base_url(options)
    raw_model = resolve_raw_model(model, options)
    body = build_raw_chat_body(messages, model, options)
    request_id = str(body["request_id"])
    headers = build_raw_headers(base_url, token, raw_model, request_id, options)
    logger.info(
        "raw upstream request id=%s url=%s body_bytes=%d messages=%d model=%s fields=%s tools=%d",
        str((options or {}).get("_relay_request_id") or request_id),
        base_url + RAW_CHAT_ENDPOINT,
        len(json.dumps(body, ensure_ascii=False, separators=(",", ":"))),
        len(body.get("messages") or []),
        raw_model.raw_model_name,
        ",".join(sorted(body.keys())),
        len(body.get("tools") or []),
    )
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))
    def open_stream() -> RawChatResponse:
        client = httpx.Client(timeout=timeout, http2=False)
        response: Optional[httpx.Response] = None
        try:
            request = client.build_request(
                "POST", base_url + RAW_CHAT_ENDPOINT, headers=headers, json=body
            )
            response = client.send(request, stream=True)
            if response.status_code != 200:
                response.read()
                detail = response.text[:800]
                raise RuntimeError(
                    f"Trae raw chat request failed with {response.status_code}"
                    + (f": {detail}" if detail else "")
                )
            return RawChatResponse(response=response, client=client, auth_token=token)
        except Exception:
            if response is not None:
                response.close()
            client.close()
            raise

    # Opening a sync httpx stream can spend several seconds waiting for Trae's
    # response headers.  Do that work off the uvicorn event loop so the public
    # SSE connection can emit its lifecycle/keepalive frames immediately.
    return await asyncio.to_thread(open_stream)
