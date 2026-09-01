"""Direct Trae raw-chat transport.

This module speaks the HTTP protocol used by Trae's terminal client without
running the Trae CLI locally.  It only transports model and tool-call data;
filesystem, shell, edit, and other tools remain owned by the API caller.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import httpx

from . import auth
from .cli_client import renderer_block_text, strip_tool_call_blocks


logger = logging.getLogger(__name__)


# ``llm_raw_chat`` is the text-only transport used by TraeWork.  Unlike the
# legacy llm_utils endpoint it binds the selected model through both the body
# and the Extra header, and it rejects OpenAI tool fields in the JSON body.
RAW_CHAT_ENDPOINT = "/api/ide/v2/llm_raw_chat"
RAW_APP_ID = "7b3f9dc2-8a4e-5c6d-2f1b-9e4a3c5b7df0"
RAW_IDE_VERSION = "3.3.67"
RAW_IDE_VERSION_CODE = "20260206"
RAW_IDE_FUNCTION = "chat"


@dataclass
class _RawSessionGate:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


_RAW_SESSION_GATES: dict[str, _RawSessionGate] = {}
_RAW_SESSION_GATES_LOCK = threading.Lock()
_RAW_SESSION_GATE_POLL_SECONDS = 0.01


def _unregister_raw_session_gate(session_id: str, gate: _RawSessionGate) -> None:
    with _RAW_SESSION_GATES_LOCK:
        if _RAW_SESSION_GATES.get(session_id) is not gate:
            return
        gate.users = max(0, gate.users - 1)
        if gate.users == 0:
            _RAW_SESSION_GATES.pop(session_id, None)


async def _acquire_raw_session_gate(session_id: str) -> Callable[[], None]:
    """Serialize streams that share Trae's model-bound raw conversation."""

    with _RAW_SESSION_GATES_LOCK:
        gate = _RAW_SESSION_GATES.get(session_id)
        if gate is None:
            gate = _RawSessionGate()
            _RAW_SESSION_GATES[session_id] = gate
        gate.users += 1

    acquired = False
    try:
        while not gate.lock.acquire(blocking=False):
            await asyncio.sleep(_RAW_SESSION_GATE_POLL_SECONDS)
        acquired = True
    except BaseException:
        if acquired:
            gate.lock.release()
        _unregister_raw_session_gate(session_id, gate)
        raise

    released = False
    release_guard = threading.Lock()

    def release() -> None:
        nonlocal released
        with release_guard:
            if released:
                return
            released = True
        try:
            gate.lock.release()
        finally:
            _unregister_raw_session_gate(session_id, gate)

    return release


@dataclass(frozen=True)
class RawModel:
    config_name: str
    raw_model_name: str
    display_name: str
    config_source: int = 1
    provider: str = ""


@dataclass
class RawChatResponse:
    """Streaming response plus the client that owns its connection."""

    response: httpx.Response
    client: httpx.Client
    auth_token: str = ""
    release_session: Optional[Callable[[], None]] = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            release_session = self.release_session
            self.release_session = None
        try:
            self.response.close()
        finally:
            try:
                self.client.close()
            finally:
                if release_session is not None:
                    release_session()

    def __enter__(self) -> "RawChatResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


_KNOWN_MODELS = {
    # These names are the exact raw ids used by TraeWork's bundled client.
    "auto": RawModel("glm-5.2", "glm-5.2", "GLM-5.2"),
    "coding": RawModel("glm-5.1", "glm-5__v2", "GLM-5.1"),
    "glm-5.2": RawModel("glm-5.2", "glm-5.2", "GLM-5.2"),
    # glm-5.3 is resolved against the account model list when available.
    "glm-5.3": RawModel("glm-5.3", "glm-5.3", "GLM-5.3"),
    "glm-5.1": RawModel("glm-5.1", "glm-5__v2", "GLM-5.1"),
    "kimi-k2.6": RawModel("kimi-k2.6", "kimi-k2.6__v2", "Kimi-K2.6"),
    "kimi-k2.7-code": RawModel(
        "kimi-k2.7-code", "kimi-k2.7-code", "Kimi-K2.7-Code"
    ),
    "deepseek-v4-pro": RawModel(
        "DeepSeek-V4-Pro", "DeepSeek-V4-Pro__v2", "DeepSeek-V4-Pro"
    ),
    "deepseek-v4-flash": RawModel(
        "DeepSeek-V4-Flash", "DeepSeek-V4-Flash__v2", "DeepSeek-V4-Flash"
    ),
    "deepseek-v4-flash-official": RawModel(
        "DeepSeek-V4-Flash-Official",
        "DeepSeek-V4-Flash-Official",
        "DeepSeek-V4-Flash \u6b63\u5f0f\u7248",
    ),
    # The Pro counterpart was missing while Flash was pinned, so a raw request
    # for the Pro official id fell back to the account default label.
    "deepseek-v4-pro-official": RawModel(
        "DeepSeek-V4-Pro-Official",
        "DeepSeek-V4-Pro-Official",
        "DeepSeek-V4-Pro \u6b63\u5f0f\u7248",
    ),
    "minimax-m3": RawModel("minimax-m3", "minimax-m3", "MiniMax M3"),
    "qwen-3.7-plus": RawModel("qwen-3.7-plus", "qwen-3.7-plus", "Qwen 3.7 Plus"),
}


_DISPLAY_MODEL_ALIASES = {
    # Trae's web UI labels are not always the config names expected by
    # llm_utils_chat. Keep an offline fallback for the current CN labels.
    "deepseek-v4-pro 正式版": "DeepSeek-V4-Pro-Official",
    "deepseek-v4-flash 正式版": "DeepSeek-V4-Flash-Official",
    "seed-2.1-pro": "Doubao-Seed-2.1-Pro",
    "seed-2.1-turbo": "Doubao-Seed-2.1-Turbo",
    "seed-code": "Doubao-Seed-Code",
    "seed-evolving": "Doubao-Seed-Evolving",
    "qwen3.7-plus": "qwen-3.7-plus",
    "qwen3.8-max": "qwen3.8-max",
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
    if known is None and mapped:
        # Aliases such as gpt-4o may resolve to a Trae config name that also
        # has an exact raw id. Re-check the mapped name so those aliases get
        # the __v2-style raw id instead of falling back to the config name.
        known = _KNOWN_MODELS.get(mapped.lower())
    default_name = mapped or without_provider or "glm-5.2"

    config_name = _option(
        options,
        "trae_raw_config_name",
        "traeRawConfigName",
        "config_name",
        "configName",
    )
    if not config_name:
        config_name = default_name

    raw_model_name = _option(
        options,
        "trae_raw_model_name",
        "traeRawModelName",
        "raw_model_name",
        "rawModelName",
    )
    if not raw_model_name:
        raw_model_name = known.raw_model_name if known else config_name

    display_name = _option(
        options, "display_name", "displayName", "model_name", "modelName"
    )
    if not display_name:
        display_name = known.display_name if known else (mapped or without_provider or config_name)

    try:
        config_source = int(
            _option(options, "config_source", "configSource", default=1) or 1
        )
    except (TypeError, ValueError):
        config_source = 1
    provider = str(_option(options, "provider", default="") or "")
    return RawModel(
        str(config_name),
        str(raw_model_name),
        str(display_name),
        config_source,
        provider,
    )


async def resolve_raw_model_for_request(
    model: str, options: Optional[Mapping[str, Any]] = None
) -> RawModel:
    """Resolve display labels before sending a raw upstream request.

    The OpenAI facade can expose Trae display labels such as
    ``DeepSeek-V4-Pro 正式版``, while the raw endpoint expects the exact
    ``config_name`` (``DeepSeek-V4-Pro-Official``). Passing the display label
    can make Trae silently use its default model.
    """

    options = options or {}
    explicit_raw = _option(
        options,
        "trae_raw_model_name",
        "traeRawModelName",
        "raw_model_name",
        "rawModelName",
    )
    resolved = resolve_raw_model(model, options)
    if explicit_raw:
        return resolved

    requested = (model or "auto").strip()
    normalized = requested[5:].lower() if requested.lower().startswith("trae/") else requested.lower()
    known = _KNOWN_MODELS.get(normalized)
    static_config = _DISPLAY_MODEL_ALIASES.get(normalized)
    if static_config:
        return RawModel(static_config, static_config, resolved.display_name)
    if known is None:
        # Aliases such as gpt-4o resolve offline to a known raw model.  Treat
        # them like the direct known mapping so the account model list cannot
        # override the pinned raw id with a missing or older value.
        try:
            from . import trae_client

            mapped = trae_client.convert_model_name(normalized)
            if mapped and mapped.lower() != normalized:
                known = _KNOWN_MODELS.get(mapped.lower())
        except Exception:
            pass
    if known is not None and normalized != "glm-5.3":
        # These models have exact offline raw ids (for example glm-5.1 maps to
        # glm-5__v2).  The account model list often omits raw_model_name for
        # them and can otherwise make the raw upstream fall back to the
        # account default model.
        return resolved

    token = str(options.get("_auth_token") or auth.get_token() or "").strip()
    try:
        from . import trae_client

        lookup_kwargs: dict[str, Any] = {"token_override": token}
        bound_user_id = str(
            options.get("_auth_user_id")
            or options.get("_billing_id")
            or options.get("_account_id")
            or ""
        ).strip()
        if bound_user_id:
            lookup_kwargs["user_id_override"] = bound_user_id
        provider_specific = options.get("provider_specific")
        if provider_specific is None and "providerSpecificData" in options:
            provider_specific = options.get("providerSpecificData")
        if isinstance(provider_specific, Mapping):
            lookup_kwargs["provider_specific"] = dict(provider_specific)
        config = await trae_client.resolve_model_config(requested, **lookup_kwargs)
        config_name = str(
            (config or {}).get("config_name")
            or (config or {}).get("name")
            or ""
        ).strip()
        if config_name:
            raw_model_name = str(
                (config or {}).get("raw_model_name")
                or (config or {}).get("rawModelName")
                or (config or {}).get("model_name")
                or (known.raw_model_name if known else "")
                or config_name
            ).strip()
            display_name = str(
                (config or {}).get("display_name")
                or (config or {}).get("display_model_name")
                or resolved.display_name
                or config_name
            ).strip()
            try:
                config_source = int((config or {}).get("config_source") or 1)
            except (TypeError, ValueError):
                config_source = 1
            return RawModel(
                config_name,
                raw_model_name,
                display_name,
                config_source,
                str((config or {}).get("provider") or ""),
            )
    except Exception as exc:
        logger.warning("raw model config lookup failed for %s: %s", requested, exc)
    return resolved


def raw_session_id(
    model: RawModel, options: Optional[Mapping[str, Any]] = None
) -> str:
    """Return a stable raw session scoped to account and model.

    Trae binds model configuration to the raw conversation id. Keep exactly one
    derived conversation per account/model pair so client-generated session ids
    cannot silently re-bootstrap that model as the upstream default (Kimi).
    Caller conversation ids and raw-session aliases are deliberately ignored.
    This keeps the model lock stable across every public API request.
    """

    options = options or {}
    account_identity = str(
        options.get("_billing_id")
        or options.get("_auth_user_id")
        or options.get("_account_id")
        or ""
    )
    if not account_identity:
        token = str(options.get("_auth_token") or "")
        account_identity = (
            "token:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
            if token
            else "default"
        )
    material_parts = [account_identity, model.config_name, model.raw_model_name]
    material = "\x1f".join(material_parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


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
            elif block.get("value") is not None:
                # Trae's renderer carries text/tool_result payloads in
                # ``value`` ({type:"text", value:...} and
                # {type:"tool_result", value:[{type:"text", value:...}]}).
                nested = renderer_block_text(block)
                if nested:
                    parts.append(nested)
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
    seen_names: set[str] = set()
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
        normalized_name = str(name)
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        normalized.append(
            {
                "name": normalized_name,
                "description": str(description or ""),
                "input_schema": schema,
            }
        )
    return normalized


def _tool_schema_budget() -> int:
    try:
        value = int(os.environ.get("TRAE_RAW_MAX_TOOL_SCHEMA_CHARS", "48000"))
    except (TypeError, ValueError):
        value = 48000
    return max(1024, value)


def _compact_tool_signature(tool: Mapping[str, Any]) -> dict[str, Any]:
    schema = tool.get("input_schema")
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    fields: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        for name, raw_hint in properties.items():
            hint: dict[str, Any] = {}
            if isinstance(raw_hint, Mapping):
                value_type = raw_hint.get("type")
                if isinstance(value_type, (str, list)):
                    hint["type"] = value_type
                enum = raw_hint.get("enum")
                if isinstance(enum, list) and len(enum) <= 20:
                    hint["enum"] = enum
                if value_type == "array" and isinstance(raw_hint.get("items"), Mapping):
                    item_type = raw_hint["items"].get("type")
                    if isinstance(item_type, str):
                        hint["items"] = item_type
            fields[str(name)] = hint or "any"
    required = schema.get("required") if isinstance(schema, Mapping) else None
    signature: dict[str, Any] = {
        "name": str(tool.get("name") or ""),
        "fields": fields,
    }
    if isinstance(required, list):
        signature["required"] = [str(value) for value in required]
    return signature


def _tool_definitions_prompt(tool_defs: list[dict[str, Any]]) -> tuple[str, bool, int]:
    """Serialize tool schemas within a bounded prompt budget.

    Normal tool sets keep their complete OpenAI schemas. Very large catalogs
    retain every tool name and as many compact field signatures as fit, with
    discovery/environment tools ordered first.
    """

    full = json.dumps(
        tool_defs, ensure_ascii=False, separators=(",", ":"), default=str
    )
    budget = _tool_schema_budget()
    if len(full) <= budget:
        return full, False, 0

    names = [str(tool.get("name") or "") for tool in tool_defs]
    discovery_names = {
        "tool_search",
        "search_tools",
        "list_tools",
        "plugin_search",
        "environment",
        "get_environment",
    }
    execution_names = {
        "shell",
        "shell_exec",
        "exec",
        "exec_command",
        "download",
        "download_file",
        "write",
        "write_file",
        "file_write",
        "edit",
        "edit_file",
        "apply_patch",
    }

    def priority(signature: Mapping[str, Any]) -> int:
        name = str(signature.get("name") or "").lower()
        # Responses namespace tools are flattened as ``namespace__tool``.
        # Rank both the complete name and its basename so functions__exec and
        # browser__download_file retain argument fields under schema pressure.
        basename = name.rsplit("__", 1)[-1]
        candidates = {name, basename}
        if candidates & discovery_names:
            return 0
        if candidates & execution_names:
            return 1
        return 2

    signatures = [_compact_tool_signature(tool) for tool in tool_defs]
    signatures.sort(key=priority)
    payload: dict[str, Any] = {
        "available_tools": names,
        "signatures": [],
        "omitted_signatures": len(signatures),
    }
    for signature in signatures:
        candidate = dict(payload)
        candidate["signatures"] = [*payload["signatures"], signature]
        candidate["omitted_signatures"] = len(signatures) - len(
            candidate["signatures"]
        )
        encoded = json.dumps(
            candidate, ensure_ascii=False, separators=(",", ":"), default=str
        )
        if len(encoded) > budget:
            continue
        payload = candidate
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return encoded, True, int(payload["omitted_signatures"])


def _concise_reasoning_enabled() -> bool:
    """Whether to ask the model for conclusions instead of a narrated chain.

    Set ``TRAE_VERBOSE_REASONING=1`` to keep the upstream's full deliberation in
    the visible answer.
    """

    return str(os.environ.get("TRAE_VERBOSE_REASONING", "")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def response_style_instruction() -> str:
    """Return the concise-answer directive, or "" when verbose mode is on."""

    if not _concise_reasoning_enabled():
        return ""
    return (
        "Answer directly: state the result and the key facts or steps that "
        "support it. Do not narrate your deliberation, restate the request, "
        "describe what you are about to do, or explain your reasoning "
        "process; give the conclusion and any code or commands it needs."
    )


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
    context = dict(client_context)
    context.setdefault("tool_discovery", build_tool_discovery(tool_defs))
    discovery = context.get("tool_discovery")
    if tool_defs and isinstance(discovery, Mapping):
        # The schema payload below already carries all tool names.
        discovery = dict(discovery)
        discovery.pop("declared_tools", None)
        context["tool_discovery"] = discovery
    lines = [
        "You are Trae Code connected to the user's local application.",
        "The external client application that started this conversation provides the client tools below.",
        "Caller client context (JSON):",
        json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str),
        "Discover the local environment through the available client tools when the request requires it.",
        "If a tool can answer the question, call it proactively and wait for the external client result.",
        "Remote or server-side tools cannot write into the caller workspace. Never claim that a client file was downloaded, created, edited, or saved unless a matching client tool result in the conversation confirms success.",
        "Never repeat a completed tool call; do not describe that server's Linux filesystem as the caller workspace.",
        "To use one, emit exactly one JSON block as your entire response and include no other text:",
        '<opencode_tool_call>{"id":"<unique-id>","name":"tool_name","input":{}}</opencode_tool_call>',
        'Replace "<unique-id>" with a new identifier for every call; never reuse an id across calls.',
        "Fill input according to the selected tool schema and wait for the client result before continuing.",
    ]
    if tool_defs and not native_tools:
        tool_payload, compacted, omitted_signatures = _tool_definitions_prompt(
            tool_defs
        )
        if compacted:
            schema_label = (
                "Available client tools and compact input signatures (JSON). "
                "The full schema catalog exceeded the relay input budget"
            )
            if omitted_signatures:
                schema_label += (
                    f"; {omitted_signatures} signature(s) are name-only. "
                    "Use an available tool_search/plugin discovery tool before "
                    "guessing arguments"
                )
            schema_label += ":"
        else:
            schema_label = "Available client tool definitions and input schemas (JSON):"
        lines.extend(
            [
                schema_label,
                tool_payload,
                "To call a tool, emit exactly one block per call in this format and no final answer in the same turn: <opencode_tool_call>{\"id\":\"<unique-id>\",\"name\":\"tool_name\",\"input\":{}}</opencode_tool_call>. Replace \"<unique-id>\" with a distinct identifier per call and fill input according to the selected tool schema.",
            "After the client returns a tool result, continue from that result until the user's request is complete.",
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
    # The remote agent returns its reasoning trace and its reply in one
    # ``thought`` field, so a narrated chain reaches the caller as the answer.
    # Ask for the conclusion instead of the deliberation.
    style = response_style_instruction()
    if style:
        lines.append(style)
    return "\n".join(lines)


def _raw_history_limit(
    options: Mapping[str, Any], names: tuple[str, ...], env_name: str, default: int
) -> int:
    value = _option(options, *names, default=None)
    if value is None:
        value = os.environ.get(env_name, str(default))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _raw_history_message_size(message: Mapping[str, Any]) -> int:
    text = _content_to_text(message.get("content"))
    if str(message.get("role") or "user") == "assistant" and text:
        text = strip_tool_call_blocks(text)
    tool_payload = message.get("tool_calls") or message.get("function_call")
    if tool_payload:
        text += json.dumps(
            tool_payload, ensure_ascii=False, separators=(",", ":"), default=str
        )
    return len(text)


def _compact_raw_history(
    messages: list[dict[str, Any]], options: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Keep unique system constraints and a bounded contiguous recent history."""

    max_messages = _raw_history_limit(
        options,
        ("trae_raw_max_messages", "traeRawMaxMessages"),
        "TRAE_RAW_MAX_MESSAGES",
        80,
    )
    max_chars = _raw_history_limit(
        options,
        ("trae_raw_max_history_chars", "traeRawMaxHistoryChars"),
        "TRAE_RAW_MAX_HISTORY_CHARS",
        120000,
    )
    valid = [dict(message) for message in messages if isinstance(message, Mapping)]
    if not valid:
        return [], 0

    system_indexes: list[int] = []
    non_system_indexes: list[int] = []
    seen_system: set[tuple[str, str]] = set()
    for index, message in enumerate(valid):
        role = str(message.get("role") or "user")
        if role in {"system", "developer"}:
            key = ("system", _content_to_text(message.get("content")))
            if key not in seen_system:
                seen_system.add(key)
                system_indexes.append(index)
        else:
            non_system_indexes.append(index)

    candidates = non_system_indexes[-max_messages:] if max_messages else non_system_indexes
    if max_chars and candidates:
        remaining = max(
            0,
            max_chars
            - sum(_raw_history_message_size(valid[index]) for index in system_indexes),
        )
        kept_reversed: list[int] = []
        used = 0
        for index in reversed(candidates):
            size = _raw_history_message_size(valid[index])
            if kept_reversed and used + size > remaining:
                break
            kept_reversed.append(index)
            used += size
        candidates = list(reversed(kept_reversed))

    # Do not begin a retained continuation with an orphaned tool result when
    # the matching assistant call is the immediately preceding message.
    if candidates and str(valid[candidates[0]].get("role") or "") == "tool":
        first_position = non_system_indexes.index(candidates[0])
        if first_position:
            previous = non_system_indexes[first_position - 1]
            previous_message = valid[previous]
            if str(previous_message.get("role") or "") == "assistant" and (
                previous_message.get("tool_calls") or previous_message.get("function_call")
            ):
                candidates.insert(0, previous)

    keep = set(system_indexes) | set(candidates)
    compacted = [message for index, message in enumerate(valid) if index in keep]
    return compacted, len(valid) - len(compacted)


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


def _renderer_tool_use_blocks(content: Any) -> list[dict[str, Any]]:
    """Collect Trae renderer ``tool_use`` content blocks as call records.

    The desktop renderer stores assistant calls as
    ``{type:"tool_use", toolCallId, name, parameters}`` blocks. Without this
    the assistant turn carries no text and gets dropped, orphaning the tool
    result that follows.
    """

    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        if str(block.get("type") or "").strip().lower() != "tool_use":
            continue
        name = block.get("name")
        if not name:
            continue
        calls.append(
            {
                "id": block.get("toolCallId")
                or block.get("tool_call_id")
                or block.get("id")
                or "unknown",
                "name": str(name),
                "input": block.get("parameters")
                if block.get("parameters") is not None
                else block.get("input", {}),
            }
        )
    return calls


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
    original_message_count = len(messages)
    messages, omitted_history = _compact_raw_history(messages, options)
    if omitted_history:
        logger.info(
            "raw history compacted input_messages=%d output_messages=%d omitted=%d",
            original_message_count,
            len(messages),
            omitted_history,
        )
    tool_catalog = (
        options["tools"]
        if "tools" in options
        else options.get("_inherited_tools")
    )
    has_tool_protocol = bool(
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
    if has_tool_protocol or "client_context" in options or "clientContext" in options:
        runtime_prompt = build_runtime_system_prompt(
            tool_catalog,
            build_client_context(options),
            options.get("tool_choice"),
            options.get("parallel_tool_calls"),
            # llm_raw_chat has no native tools field. Schemas and the output
            # envelope must therefore be explicit in the system message.
            native_tools=False,
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
        elif role == "tool":
            call_id = str(
                message.get("tool_call_id") or message.get("toolCallId") or "unknown"
            )
            tool_name = str(message.get("name") or "tool")
            text = f"Client tool result [{call_id}] {tool_name}:\n{text}"
            role = "user"
        elif role not in ("system", "user", "assistant"):
            role = "user"
        if role == "assistant":
            call_context = _serialize_tool_call_context(tool_calls)
            if not call_context and isinstance(message.get("function_call"), Mapping):
                call_context = _serialize_tool_call_context([message["function_call"]])
            if not call_context:
                # Trae's renderer keeps assistant calls as tool_use content
                # blocks instead of an OpenAI ``tool_calls`` array.
                call_context = _serialize_tool_call_context(
                    _renderer_tool_use_blocks(content)
                )
            if call_context:
                text = "\n\n".join(part for part in (text, call_context) if part)
        native_message: dict[str, Any] = {"role": role}
        if text:
            native_message["content"] = [{"type": "text", "text": text}]
        elif role in ("system", "user"):
            continue
        if role == "assistant" and not text:
            continue
        result.append(native_message)
    return result


def build_raw_chat_body(
    messages: list[dict[str, Any]],
    model: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the exact six-field payload accepted by ``llm_raw_chat``.

    TraeWork carries tool schemas and tool history in text messages.  Sending
    OpenAI ``tools`` or generation options beside these fields makes the raw
    endpoint reject the request before the model can produce any output.
    """

    options = options or {}
    resolved_model = resolve_raw_model(model, options)
    session_id = session_id or str(
        _option(options, "session_id", "sessionId", default=uuid.uuid4())
    )
    body: dict[str, Any] = {
        "config_name": resolved_model.config_name,
        "conversation_id": session_id,
        "messages": build_raw_messages(messages, options),
        "model_name": resolved_model.raw_model_name,
        "session_id": session_id,
        "stream": True,
    }
    logger.info(
        "raw body prepared id=%s keys=%s tools=%d history_tools=%s",
        str(options.get("_relay_request_id") or session_id),
        ",".join(sorted(body.keys())),
        len(
            _normalized_tools(
                options.get("tools") or options.get("_inherited_tools")
            )
        ),
        any(
            isinstance(message, Mapping)
            and (message.get("tool_calls") or message.get("role") == "tool")
            for message in messages
        ),
    )
    return body


def build_raw_headers(
    base_url: str,
    token: str,
    raw_model: RawModel,
    request_id: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    session_id: str = "",
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
    ide_function = str(
        _option(options, "ide_function", "ideFunction", default=None)
        or os.environ.get("TRAE_RAW_IDE_FUNCTION")
        or RAW_IDE_FUNCTION
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
    put("X-Ide-Function", ide_function)
    if ide_version:
        put("X-Ide-Version", ide_version)
    else:
        for key in list(headers):
            if key.lower() == "x-ide-version":
                del headers[key]
    put("X-Request-Id", request_id)
    if session_id:
        extra = {
            "agent_loop_id": session_id,
            "api_host": base_url,
            "api_key": token,
            "base_url": f"{base_url}/trae-cli/api/v1/llm/proxy",
            "config_name": raw_model.config_name,
            "config_source": raw_model.config_source,
            "display_name": raw_model.display_name,
            "model_name": raw_model.raw_model_name,
            "real_api_key": "",
            "real_base_url": "",
            "session_id": session_id,
            "user_prompt_submit_id": session_id,
        }
        put(
            "Extra",
            json.dumps(
                # HTTP header values must remain ASCII.  The JSON body can
                # carry UTF-8 directly, but model display names such as
                # ``DeepSeek-V4-Flash 正式版`` also appear in Extra.
                extra, ensure_ascii=True, separators=(",", ":"), default=str
            ),
        )
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
    raw_model = await resolve_raw_model_for_request(model, options)
    session_options = dict(options)
    session_options.setdefault("_auth_token", token)
    upstream_session_id = raw_session_id(raw_model, session_options)
    body_options = dict(options)
    body_options.setdefault("config_name", raw_model.config_name)
    body_options.setdefault("raw_model_name", raw_model.raw_model_name)
    body_options.setdefault("display_name", raw_model.display_name)
    body = build_raw_chat_body(
        messages, model, body_options, session_id=upstream_session_id
    )
    request_id = str(uuid.uuid4())
    headers = build_raw_headers(
        base_url,
        token,
        raw_model,
        request_id,
        options,
        session_id=upstream_session_id,
    )
    logger.info(
        "raw upstream request id=%s url=%s body_bytes=%d messages=%d model=%s fields=%s tools=%d",
        str((options or {}).get("_relay_request_id") or request_id),
        base_url + RAW_CHAT_ENDPOINT,
        len(json.dumps(body, ensure_ascii=False, separators=(",", ":"))),
        len(body.get("messages") or []),
        raw_model.raw_model_name,
        ",".join(sorted(body.keys())),
        len(_normalized_tools(options.get("tools") or options.get("_inherited_tools"))),
    )
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))

    release_session = await _acquire_raw_session_gate(upstream_session_id)

    def open_stream() -> RawChatResponse:
        client: Optional[httpx.Client] = None
        response: Optional[httpx.Response] = None
        try:
            client = httpx.Client(timeout=timeout, http2=False)
            request = client.build_request(
                "POST",
                base_url
                + str(os.environ.get("TRAE_RAW_CHAT_ENDPOINT") or RAW_CHAT_ENDPOINT),
                headers=headers,
                json=body,
            )
            response = client.send(request, stream=True)
            if response.status_code != 200:
                response.read()
                detail = response.text[:800]
                raise RuntimeError(
                    f"Trae raw chat request failed with {response.status_code}"
                    + (f": {detail}" if detail else "")
                )
            return RawChatResponse(
                response=response,
                client=client,
                auth_token=token,
                release_session=release_session,
            )
        except BaseException:
            try:
                if response is not None:
                    response.close()
            finally:
                try:
                    if client is not None:
                        client.close()
                finally:
                    release_session()
            raise

    # Opening a sync httpx stream can spend several seconds waiting for Trae's
    # response headers.  Do that work off the uvicorn event loop so the public
    # SSE connection can emit its lifecycle/keepalive frames immediately.
    try:
        open_task = asyncio.create_task(asyncio.to_thread(open_stream))
    except BaseException:
        release_session()
        raise

    try:
        return await asyncio.shield(open_task)
    except asyncio.CancelledError:
        # ``to_thread`` cannot cancel a synchronous httpx send already in
        # progress. Close the response as soon as it materializes so the
        # model-bound gate is still released without allowing an overlapping
        # request on the same raw conversation.
        def close_cancelled_open(task: asyncio.Task[RawChatResponse]) -> None:
            try:
                wrapped = task.result()
            except BaseException:
                release_session()
                return
            try:
                wrapped.close()
            except Exception as exc:
                logger.warning("raw cancelled response cleanup failed: %s", exc)

        open_task.add_done_callback(close_cancelled_open)
        raise
    except BaseException:
        # ``open_stream`` releases on every internal failure. This idempotent
        # fallback also covers task/executor setup failures before it starts.
        release_session()
        raise
