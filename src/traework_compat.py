"""Pure TraeWork/Fake-API compatibility helpers.

This module does not perform network I/O. It only builds and parses the
request shapes observed in TraeWork's raw ``llm_raw_chat`` and remote
``chat_sessions`` protocols so another adapter can replay or inspect them.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping, Optional

from . import raw_client, trae_client, trae_remote_client
from .sse import compact_reasoning_text


logger = logging.getLogger(__name__)


TRAEWORK_RAW_CHAT_PATH = raw_client.RAW_CHAT_ENDPOINT
TRAEWORK_REMOTE_SESSION_PATH = "/api/remote/v1/chat_sessions"
TRAEWORK_CUSTOM_CONNECTIVITY_PATH = "/api/agent/v3/custom_model_connectivity_check"
TRAEWORK_CUSTOM_RAW_ALIASES = {
    TRAEWORK_RAW_CHAT_PATH,
    "/api/agent/v3/llm_raw_chat_custom_model",
    "/api/agent/v3/custom_model_proxy/chat",
    "/v1/traework/chat",
}
TRAEWORK_CONNECTIVITY_ALIASES = {
    TRAEWORK_CUSTOM_CONNECTIVITY_PATH,
    "/api/ide/v1/custom_model_connectivity_check",
    "/v1/custom_model/connectivity",
}


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in options and options[name] is not None:
            return options[name]
    return default


def _copy_mapping(mapping: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(mapping) if isinstance(mapping, Mapping) else {}


@dataclass(frozen=True)
class TraeWorkModelBinding:
    """Model identifiers TraeWork binds into raw and remote requests."""

    config_name: str
    raw_model_name: str
    display_name: str
    config_source: int = 1
    provider: str = ""


@dataclass(frozen=True)
class TraeWorkRequestIds:
    """Identifiers TraeWork reuses inside one request payload."""

    session_id: str
    conversation_id: str
    agent_loop_id: str
    user_prompt_submit_id: str
    request_id: str


@dataclass(frozen=True)
class TraeWorkRequestDescriptor:
    """Serializable HTTP request descriptor for a compatibility adapter."""

    method: str
    path: str
    headers: dict[str, str]
    json_body: dict[str, Any]
    model: TraeWorkModelBinding
    ids: TraeWorkRequestIds
    stream: bool = True


@dataclass(frozen=True)
class TraeWorkInboundRequest:
    """Normalized request received from a TraeWork custom-model adapter."""

    messages: list[dict[str, Any]]
    model: str
    stream: bool
    options: dict[str, Any]
    session_id: str
    request_id: str
    protocol: str = "traework-raw"


def resolve_model_binding(
    model: str,
    options: Optional[Mapping[str, Any]] = None,
) -> TraeWorkModelBinding:
    """Resolve one public model id to TraeWork's config/raw/display triplet."""

    resolved = raw_client.resolve_raw_model(model, options or {})
    return TraeWorkModelBinding(
        config_name=resolved.config_name,
        raw_model_name=resolved.raw_model_name,
        display_name=resolved.display_name,
        config_source=resolved.config_source,
        provider=resolved.provider,
    )


def build_request_ids(
    options: Optional[Mapping[str, Any]] = None,
    *,
    session_id: str = "",
) -> TraeWorkRequestIds:
    """Return the per-request identifiers TraeWork keeps aligned."""

    options = options or {}
    seed = str(
        session_id
        or _option(
            options,
            "session_id",
            "sessionId",
            "connect_session_id",
            "connectSessionId",
            default="",
        )
        or uuid.uuid4()
    )
    request_id = str(
        _option(options, "request_id", "requestId", "x_request_id", default="")
        or seed
    )
    return TraeWorkRequestIds(
        session_id=seed,
        conversation_id=seed,
        agent_loop_id=seed,
        user_prompt_submit_id=seed,
        request_id=request_id,
    )


def parse_raw_extra_header(value: str) -> dict[str, Any]:
    """Parse TraeWork's JSON ``Extra`` header into a mapping."""

    if not str(value or "").strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("TraeWork Extra header must decode to a JSON object")
    return parsed


def build_raw_extra_header(
    model: TraeWorkModelBinding,
    ids: TraeWorkRequestIds,
    *,
    base_url: str,
    token: str = "",
) -> dict[str, Any]:
    """Build the JSON payload carried by TraeWork's ``Extra`` header."""

    return {
        "agent_loop_id": ids.agent_loop_id,
        "api_host": base_url.rstrip("/"),
        "api_key": token,
        "base_url": base_url.rstrip("/") + "/trae-cli/api/v1/llm/proxy",
        "config_name": model.config_name,
        "config_source": model.config_source,
        "display_name": model.display_name,
        "model_name": model.raw_model_name,
        "real_api_key": "",
        "real_base_url": "",
        "session_id": ids.session_id,
        "user_prompt_submit_id": ids.user_prompt_submit_id,
    }


def build_raw_chat_request(
    messages: list[dict[str, Any]],
    model: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
    token: str = "",
    base_url: str = "https://trae-api-cn.mchost.guru",
    stream: bool = True,
) -> TraeWorkRequestDescriptor:
    """Build one TraeWork ``llm_raw_chat`` request descriptor."""

    options_dict = _copy_mapping(options)
    binding = resolve_model_binding(model, options_dict)
    ids = build_request_ids(options_dict)

    body_options = dict(options_dict)
    body_options.setdefault("config_name", binding.config_name)
    body_options.setdefault("raw_model_name", binding.raw_model_name)
    body_options.setdefault("display_name", binding.display_name)
    body = raw_client.build_raw_chat_body(
        messages,
        model,
        body_options,
        session_id=ids.session_id,
    )
    body["stream"] = bool(stream)

    # Keep this descriptor on the same wire contract as the runtime transport:
    # duplicate token headers, X-Uid, Connection, Extra, and the generated
    # device fingerprint all come from one implementation. This helper is used
    # by probes and integrations that may send the descriptor directly.
    headers = raw_client.build_raw_headers(
        base_url,
        token,
        raw_client.RawModel(
            binding.config_name,
            binding.raw_model_name,
            binding.display_name,
            binding.config_source,
            binding.provider,
        ),
        ids.request_id,
        options_dict,
        session_id=ids.session_id,
    )
    headers["Accept"] = "text/event-stream" if stream else "application/json"
    if not token:
        for key in ("Authorization", "X-Cloudide-Token", "X-Ide-Token"):
            headers.pop(key, None)

    return TraeWorkRequestDescriptor(
        method="POST",
        path=TRAEWORK_RAW_CHAT_PATH,
        headers=headers,
        json_body=body,
        model=binding,
        ids=ids,
        stream=bool(stream),
    )


def _remote_custom_model(
    binding: TraeWorkModelBinding,
    *,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": binding.config_name,
        "config_name": binding.config_name,
        "model_name": binding.raw_model_name,
        "display_name": binding.display_name,
        "config_source": int(
            _option(options, "config_source", "configSource", default=binding.config_source)
            or binding.config_source
        ),
        "is_preset": bool(_option(options, "is_preset", "model_is_preset", default=True)),
        "provider": str(_option(options, "provider", default=binding.provider) or ""),
    }


def build_remote_session_request(
    messages: list[dict[str, Any]],
    model: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
    token: str = "",
) -> TraeWorkRequestDescriptor:
    """Build one TraeWork remote ``chat_sessions`` request descriptor."""

    options_dict = _copy_mapping(options)
    binding = resolve_model_binding(model, options_dict)
    requested_model = str(model or "").strip()
    mode, strategy, resolved_model_name = trae_remote_client.resolve_mode(model)
    agent_type = trae_remote_client.remote_agent_type(model, options_dict)
    provider_specific = _copy_mapping(
        _option(options_dict, "provider_specific", "providerSpecificData", default={})
    )
    stable_session_id = str(
        _option(
            options_dict,
            "biz_session_id",
            "bizSessionId",
            "trae_remote_session_id",
            "traeRemoteSessionId",
            default="",
        )
        or trae_remote_client.model_session_id(
            resolved_model_name or requested_model or "auto",
            options_dict,
        )
    )
    ids = build_request_ids(options_dict, session_id=stable_session_id)

    initial_message: dict[str, Any] = {
        "chat_session_id": "",
        "content": copy.deepcopy(
            _option(options_dict, "content", "initial_content", default=[])
        ),
        "query": trae_client.flatten_query(messages),
        "model_name": resolved_model_name,
        "agent_type": agent_type,
        "agent_id": agent_type,
        "model_selection_strategy": strategy,
        "common_params": trae_remote_client.common_params(
            provider_specific,
            mode,
            stable_session_id,
            options=options_dict,
        ),
    }

    if strategy == "manual":
        custom_model = _remote_custom_model(binding, options=options_dict)
        initial_message["model_name"] = binding.config_name
        initial_message["model_config_source"] = custom_model["config_source"]
        initial_message["model_is_preset"] = custom_model["is_preset"]
        initial_message["model_provider"] = custom_model["provider"]
        initial_message["custom_model"] = custom_model

    body = {
        "mode": mode,
        "environment_id": str(
            _option(options_dict, "environment_id", "environmentId", default="default")
        ),
        "initial_message": initial_message,
        "env": str(_option(options_dict, "env", default="remote")),
        "auto_create_project": bool(
            _option(options_dict, "auto_create_project", "autoCreateProject", default=False)
        ),
        "origin": str(_option(options_dict, "origin", default="web")),
    }
    headers = trae_remote_client.build_headers(
        token,
        options={"provider_specific": provider_specific, **options_dict},
        stream=False,
    )
    headers["X-Request-Id"] = ids.request_id

    return TraeWorkRequestDescriptor(
        method="POST",
        path=TRAEWORK_REMOTE_SESSION_PATH,
        headers=headers,
        json_body=body,
        model=binding,
        ids=ids,
        stream=False,
    )


def remote_events_path(session_id: str, message_id: str) -> str:
    """Return the TraeWork remote events path for one reply message."""

    return (
        f"/api/remote/v1/chat_sessions/{session_id}/events"
        f"?reply_to_message_id={message_id}"
    )


# ---------------------------------------------------------------------------
# TraeWork custom-model ingress
# ---------------------------------------------------------------------------

_NESTED_BODY_KEYS = ("data", "params", "request", "payload", "body", "input")
_MESSAGE_KEYS = (
    "messages",
    "chat_messages",
    "chatMessages",
    "message_history",
    "messageHistory",
    "history",
)
_MODEL_KEYS = (
    "model",
    "config_name",
    "configName",
    "ai_model_name",
    "aiModelName",
    "model_name",
    "modelName",
)
_SESSION_KEYS = (
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
    "chat_session_id",
    "chatSessionId",
    "agent_loop_id",
    "agentLoopId",
    "connect_session_id",
    "connectSessionId",
)


def _jsonish(value: Any) -> Any:
    """Decode a JSON string when it is clearly a serialized value."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _body_layers(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return outer-to-inner mappings without recursively duplicating fields."""

    layers: list[Mapping[str, Any]] = []
    pending: list[Mapping[str, Any]] = [body]
    seen: set[int] = set()
    while pending and len(layers) < 8:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        layers.append(current)
        for key in _NESTED_BODY_KEYS:
            nested = _jsonish(current.get(key))
            if isinstance(nested, Mapping):
                pending.append(nested)
    return layers


def _first_layer_value(layers: Iterable[Mapping[str, Any]], *keys: str) -> Any:
    for layer in layers:
        for key in keys:
            value = layer.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _specific_layer_value(layers: Iterable[Mapping[str, Any]], *keys: str) -> Any:
    """Return the deepest non-empty wrapper value for a field."""

    materialized = list(layers)
    for layer in reversed(materialized):
        for key in keys:
            value = layer.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _custom_model_mapping(layers: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    for layer in reversed(list(layers)):
        for key in ("custom_model", "customModel", "model_info", "modelInfo", "model_config", "modelConfig"):
            value = _jsonish(layer.get(key))
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _normalize_renderer_role(value: Any) -> str:
    if isinstance(value, bool):
        return "user"
    if isinstance(value, int):
        return {0: "system", 1: "user", 2: "assistant", 3: "tool"}.get(value, "user")
    text = str(value or "user").strip().lower()
    return {
        "0": "system",
        "1": "user",
        "2": "assistant",
        "3": "tool",
        "human": "user",
        "ai": "assistant",
    }.get(text, text if text in {"system", "developer", "user", "assistant", "tool", "function"} else "user")


def _normalize_inbound_messages(value: Any) -> list[dict[str, Any]]:
    value = _jsonish(value)
    if isinstance(value, Mapping):
        # A single renderer message is a valid input; a wrapper containing a
        # nested messages field is unwrapped first.
        nested = value.get("messages")
        if nested not in (None, "", [], {}):
            return _normalize_inbound_messages(nested)
        value = [value]
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        item = _jsonish(item)
        if isinstance(item, str):
            result.append({"role": "user", "content": item})
            continue
        if not isinstance(item, Mapping):
            continue
        message = dict(item)
        message["role"] = _normalize_renderer_role(
            message.get("role", message.get("speaker", message.get("role_type")))
        )
        if message.get("content") in (None, "", []):
            for key in ("parts", "text", "message", "prompt", "value", "input"):
                candidate = message.get(key)
                if candidate not in (None, "", [], {}):
                    message["content"] = candidate
                    break
        result.append(message)
    return result


def _query_to_messages(value: Any) -> list[dict[str, Any]]:
    """Convert Trae's flattened query/content wrappers to chat messages."""

    value = _jsonish(value)
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value.strip() else []
    if not isinstance(value, list):
        return _normalize_inbound_messages(value)
    parts: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            parts.append(item)
            continue
        data = item.get("data") if isinstance(item.get("data"), Mapping) else item
        event_type = str(item.get("type") or data.get("type") or "text").lower()
        if event_type in {"text", "input_text", "output_text"}:
            text = data.get("content") or data.get("text") or data.get("value")
            if text not in (None, ""):
                parts.append({"type": "text", "text": str(text)})
        elif event_type in {"tool_use", "tool_result", "thinking", "image_url", "data"}:
            parts.append(dict(data))
    return [{"role": "user", "content": parts}] if parts else []


def _header(headers: Any, *names: str) -> str:
    if not isinstance(headers, Mapping):
        return ""
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return str(value)
    return ""


def is_traework_request(
    path: str = "",
    headers: Optional[Mapping[str, Any]] = None,
    body: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Detect a TraeWork custom-model request without hijacking OpenAI calls."""

    normalized_path = "/" + str(path or "").strip().lstrip("/")
    if normalized_path.rstrip("/").lower() in {
        item.rstrip("/").lower() for item in TRAEWORK_CUSTOM_RAW_ALIASES
    }:
        return True
    if "custom_model_connectivity_check" in normalized_path.lower():
        return True
    header_blob = " ".join(
        _header(
            headers,
            "user-agent",
            "x-trae-client-type",
            "x-trae-client",
            "x-trae-protocol",
            "x-bridge-transport",
            "x-custom-model",
            "x-trae-work",
        )
        for _ in (0,)
    ).lower()
    if any(marker in header_blob for marker in ("traework", "trae-work", "aha", "custom-model")):
        return True
    # Native raw headers are a strong signal even when the User-Agent is
    # hidden by an Electron network layer.
    if _header(headers, "extra") and _header(headers, "x-app-id") and _header(headers, "x-ide-function"):
        return True
    layers = _body_layers(body) if isinstance(body, Mapping) else []
    for layer in layers:
        explicit_custom = layer.get("is_custom_model")
        if explicit_custom is None:
            explicit_custom = layer.get("isCustomModel")
        if explicit_custom is True or str(explicit_custom or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True

        # Body-only detection must require an actual custom-model identity.
        # Generic agent metadata (agent_type, render_context, mcp_tool_list,
        # chat_session_id, and similar fields) is also valid on ordinary
        # OpenAI-compatible requests and must not switch their wire protocol.
        for key in ("custom_model", "customModel"):
            custom_model = _jsonish(layer.get(key))
            if isinstance(custom_model, Mapping) and custom_model:
                return True
            if isinstance(custom_model, str) and custom_model.strip():
                return True
        for key in ("model_info", "modelInfo"):
            model_info = _jsonish(layer.get(key))
            if not isinstance(model_info, Mapping):
                continue
            if any(
                field in model_info
                for field in (
                    "custom_model_id",
                    "customModelId",
                    "is_custom_base_url",
                    "isCustomBaseUrl",
                )
            ):
                return True
    return False


def normalize_inbound_request(
    body: Mapping[str, Any],
    *,
    headers: Optional[Mapping[str, Any]] = None,
    path: str = "",
) -> TraeWorkInboundRequest:
    """Normalize raw/custom/renderer request variants into relay inputs."""

    if not isinstance(body, Mapping):
        raise ValueError("TraeWork request body must be an object")
    layers = _body_layers(body)
    custom_model = _custom_model_mapping(layers)
    extra = {}
    extra_value = _first_layer_value(layers, "Extra", "extra", "extra_info", "extraInfo")
    if extra_value in (None, ""):
        extra_value = _header(headers, "extra")
    if extra_value not in (None, ""):
        try:
            extra = parse_raw_extra_header(str(extra_value))
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.debug("ignoring malformed TraeWork Extra header")

    messages_value = _first_layer_value(layers, *_MESSAGE_KEYS)
    messages = _normalize_inbound_messages(messages_value)
    if not messages:
        for key in ("message_content", "messageContent", "query", "prompt", "text", "input_text", "content"):
            candidate = _first_layer_value(layers, key)
            if candidate not in (None, "", [], {}):
                messages = _query_to_messages(candidate) if key in {"query", "content"} else _normalize_inbound_messages(candidate)
                if messages:
                    break
    if not messages:
        raise ValueError("messages or query is required")

    # Raw requests carry both the stable catalog id (config_name) and the
    # provider-facing id (model_name). Dispatch must use the former; otherwise
    # a value such as ``glm-5.2__dev`` is treated as a public model and may be
    # rejected or silently remapped. Prefer a canonical config id whenever the
    # request supplies one, even if an outer wrapper also has a display/model
    # alias. A plain ``model`` is used only when no config id exists.
    model_value = _specific_layer_value(
        layers,
        "config_name",
        "configName",
        "ai_model_name",
        "aiModelName",
    )
    if model_value in (None, ""):
        model_value = _first_layer_value(layers, "model")
    if model_value in (None, ""):
        model_value = (
            custom_model.get("config_name")
            or custom_model.get("configName")
            or extra.get("config_name")
        )
    if model_value in (None, ""):
        model_value = (
            _first_layer_value(layers, "model_name", "modelName")
            or custom_model.get("model_name")
            or custom_model.get("modelName")
            or extra.get("model_name")
            or "auto"
        )
    model = str(model_value or "auto").strip() or "auto"

    session_value = _first_layer_value(layers, *_SESSION_KEYS)
    session_id = str(
        session_value
        or extra.get("session_id")
        or extra.get("agent_loop_id")
        or _header(headers, "x-session-id", "x-conversation-id", "x-chat-session-id")
        or uuid.uuid4()
    ).strip()
    request_id = str(
        _first_layer_value(layers, "request_id", "requestId", "x_request_id")
        or _header(headers, "x-request-id", "x-trae-request-id")
        or session_id
    ).strip()

    options: dict[str, Any] = {}
    option_keys = (
        "tools",
        "tool_choice",
        "toolChoice",
        "parallel_tool_calls",
        "parallelToolCalls",
        "thinking",
        "reasoning_effort",
        "reasoningEffort",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "response_format",
        "client_context",
        "clientContext",
        "workspace_context",
        "workspaceContext",
        "render_context",
        "renderContext",
        "environment_context",
        "environmentContext",
        "workspace_folder",
        "workspacePath",
        "agent_type",
        "agentType",
        "model_selection_strategy",
        "model_auto_selection",
        "custom_model",
        "customModel",
        "model_config_source",
        "model_is_preset",
        "model_provider",
        "provider",
        "config_source",
        "configSource",
    )
    # Inner payload values win, but copy every layer so wrappers from different
    # Trae versions remain usable.  ``layers`` is outer-to-inner; walking it in
    # reverse and using ``setdefault`` keeps the most specific value.
    for layer in reversed(layers):
        for key in option_keys:
            if key in layer and layer[key] is not None and key not in options:
                options[key] = _jsonish(layer[key])
    request_config_name = _specific_layer_value(
        layers,
        "config_name",
        "configName",
        "ai_model_name",
        "aiModelName",
    )
    request_raw_model_name = _specific_layer_value(
        layers,
        "raw_model_name",
        "rawModelName",
        "model_name",
        "modelName",
    )
    if request_config_name not in (None, ""):
        options.setdefault("trae_raw_config_name", request_config_name)
    if request_raw_model_name not in (None, ""):
        options.setdefault("trae_raw_model_name", request_raw_model_name)
    if custom_model:
        options.setdefault("custom_model", custom_model)
        options.setdefault("provider", custom_model.get("provider") or custom_model.get("provider_name") or "")
        options.setdefault("trae_raw_config_name", custom_model.get("config_name") or custom_model.get("configName"))
        options.setdefault("trae_raw_model_name", custom_model.get("raw_model_name") or custom_model.get("rawModelName") or custom_model.get("model_name") or custom_model.get("modelName"))
        options.setdefault("display_name", custom_model.get("display_name") or custom_model.get("displayName"))
        options.setdefault("config_source", custom_model.get("config_source") or custom_model.get("configSource"))
    if extra:
        options.setdefault("trae_raw_config_name", extra.get("config_name"))
        options.setdefault("trae_raw_model_name", extra.get("model_name"))
        options.setdefault("display_name", extra.get("display_name"))
        options.setdefault("config_source", extra.get("config_source"))
    context = (
        options.get("client_context")
        or options.get("clientContext")
        or options.get("workspace_context")
        or options.get("workspaceContext")
        or options.get("render_context")
        or options.get("renderContext")
        or options.get("environment_context")
        or options.get("environmentContext")
    )
    if isinstance(context, Mapping):
        options["client_context"] = dict(context)
    elif options.get("workspace_folder") or options.get("workspacePath"):
        options["client_context"] = {
            "workspace_path": options.get("workspace_folder") or options.get("workspacePath")
        }
    if options.get("toolChoice") is not None and options.get("tool_choice") is None:
        options["tool_choice"] = options["toolChoice"]
    if options.get("parallelToolCalls") is not None and options.get("parallel_tool_calls") is None:
        options["parallel_tool_calls"] = options["parallelToolCalls"]
    options["session_id"] = session_id
    options["_traework_custom_model"] = True
    options["_trae_request_path"] = path
    options["_tool_protocol_requested"] = bool(
        options.get("tools")
        or options.get("tool_choice") is not None
        or options.get("parallel_tool_calls") is not None
        or any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(block, Mapping)
                and str(block.get("type") or "").lower() in {"tool_use", "tool_result"}
                for block in message["content"]
            )
            for message in messages
            if isinstance(message, Mapping)
        )
    )
    # A Cloud-IDE-JWT header is an explicit upstream credential. A normal
    # Bearer/API key from a custom-model form must stay local to the relay.
    authorization = _header(headers, "authorization")
    if authorization.lower().startswith("cloud-ide-jwt "):
        options["_auth_token"] = authorization.split(" ", 1)[1].strip()
    stream_value = _first_layer_value(layers, "stream", "is_stream", "isStream")
    if isinstance(stream_value, str):
        stream = stream_value.strip().lower() in {"1", "true", "yes", "on"}
    else:
        stream = bool(True if stream_value is None else stream_value)
    return TraeWorkInboundRequest(
        messages=messages,
        model=model,
        stream=stream,
        options=options,
        session_id=session_id,
        request_id=request_id,
        protocol="traework-raw" if path.rstrip("/").lower() in {p.rstrip("/").lower() for p in TRAEWORK_CUSTOM_RAW_ALIASES} else "traework-custom",
    )


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("value") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, Mapping):
        return str(value.get("text") or value.get("value") or value.get("content") or "")
    return ""


def _normalize_custom_tool_call(call: Any, index: int = 0) -> Optional[dict[str, Any]]:
    if not isinstance(call, Mapping):
        return None
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name = function.get("name") or call.get("name")
    if not name:
        # OpenAI streaming tool deltas often omit id/name after the first
        # fragment and continue with only ``index`` plus argument bytes.
        has_mergeable_fragment = (
            call.get("index") is not None
            or function.get("arguments") not in (None, "")
            or function.get("input") not in (None, "")
        )
        if not has_mergeable_fragment:
            return None
        name = ""
    args = function.get("arguments")
    if args is None:
        args = function.get("input", call.get("input", call.get("parameters", {})))
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False, separators=(",", ":"), default=str)
    return {
        "id": str(call.get("id") or call.get("tool_call_id") or call.get("toolCallId") or f"call_{index}"),
        "name": str(name),
        "input": args,
    }


def custom_sse_event(event: str, payload: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(dict(payload), ensure_ascii=False, separators=(',', ':'), default=str)}\n\n"


def _scan_loose_json_string(text: str, start: int) -> tuple[str, int] | None:
    """Read a quoted JSON string even when the surrounding object is partial.

    A few upstream adapters split ``tool_call_info.parameters`` across SSE
    events.  Each fragment is therefore not a complete JSON document, even
    though the parameter value itself is still encoded as a JSON string.  The
    scanner keeps escaped quotes inside that value and stops at the first
    unescaped quote so the fragment can be merged by the normal tool delta
    logic.
    """

    if start >= len(text) or text[start] != '"':
        return None
    chars: list[str] = []
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            raw = "".join(chars)
            try:
                return str(json.loads('"' + raw + '"')), index + 1
            except (TypeError, ValueError, json.JSONDecodeError):
                # Preserve the fragment rather than dropping a tool call when
                # a provider uses a non-standard escape sequence.
                return raw.replace('\\"', '"'), index + 1
        if char == "\\" and index + 1 < len(text):
            chars.extend((char, text[index + 1]))
            index += 2
            continue
        chars.append(char)
        index += 1
    return None


def _extract_partial_tool_call_info(value: str) -> Optional[dict[str, Any]]:
    """Recover a split ``tool_call_info`` object from a malformed SSE line.

    The normal path remains strict ``json.loads``.  This fallback is narrowly
    scoped to lines that explicitly contain ``tool_call_info`` and only
    extracts its scalar fields, so unrelated malformed provider data is still
    ignored instead of being guessed at.
    """

    info_match = re.search(
        r'"(?:tool_call_info|toolCallInfo)"\s*:\s*\{', value
    )
    if not info_match:
        return None
    start = info_match.end()
    result: dict[str, Any] = {}

    def read_field(names: tuple[str, ...]) -> Any:
        for name in names:
            match = re.search(
                r'"' + re.escape(name) + r'"\s*:\s*', value[start:]
            )
            if not match:
                continue
            position = start + match.end()
            if position < len(value) and value[position] == '"':
                scanned = _scan_loose_json_string(value, position)
                if scanned is not None:
                    return scanned[0]
                continue
            tail = value[position:]
            token = re.split(r"[,}]", tail, maxsplit=1)[0].strip()
            if token:
                return _jsonish(token)
        return None

    tool_id = read_field(("tool_call_id", "toolCallId", "id"))
    name = read_field(("name",))
    parameters = read_field(("parameters", "params", "arguments", "input"))
    index = read_field(("index",))
    if tool_id is not None:
        result["tool_call_id"] = tool_id
    if name is not None:
        result["name"] = name
    if parameters is not None:
        result["parameters"] = parameters
    if index is not None:
        result["index"] = index
    return result or None


def _iter_openai_sse_payloads(chunk: Any) -> Iterable[Any]:
    if isinstance(chunk, (bytes, bytearray)):
        text = bytes(chunk).decode("utf-8", errors="replace")
    else:
        text = str(chunk or "")
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith(":") or line.lower().startswith("event:"):
            continue
        if not line.lower().startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            yield "__DONE__"
            continue
        try:
            yield json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            partial = _extract_partial_tool_call_info(value)
            if partial is not None:
                yield {"tool_call_info": partial}


async def _iter_buffered_openai_sse_payloads(
    source: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    """Parse OpenAI SSE frames across arbitrary transport chunk boundaries."""

    buffer = ""
    async for chunk in source:
        if isinstance(chunk, (bytes, bytearray)):
            text = bytes(chunk).decode("utf-8", errors="replace")
        else:
            text = str(chunk or "")
        if not text:
            continue
        buffer += text.replace("\r\n", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            for payload in _iter_openai_sse_payloads(block + "\n\n"):
                yield payload
    # A few adapters omit the final blank line when closing the stream. Flush
    # that last complete-looking block rather than silently dropping it.
    if buffer.strip():
        for payload in _iter_openai_sse_payloads(buffer + "\n\n"):
            yield payload


def _custom_output_from_openai(payload: Mapping[str, Any]) -> tuple[str, str, list[dict[str, Any]], Optional[dict[str, Any]], str]:
    """Extract text/reasoning/tool deltas from one OpenAI payload."""

    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    source = delta or message or payload
    text = _text_from_content(source.get("content"))
    reasoning = ""
    for key in ("reasoning_content", "reasoning", "thinking", "thought", "reasoning_delta"):
        candidate = source.get(key)
        if candidate not in (None, ""):
            reasoning += _text_from_content(candidate)
    calls: list[dict[str, Any]] = []

    def append_call(call: Any, index: int = 0) -> None:
        normalized = _normalize_custom_tool_call(call, index)
        if not normalized:
            return
        # Only trust an explicitly supplied stream index. When it is absent,
        # merge_call() will key by the stable call id, preventing unrelated
        # calls from collapsing into synthetic index:0.
        if isinstance(call, Mapping) and call.get("index") is not None:
            normalized["_index"] = call.get("index")
        signature = (
            normalized.get("id"),
            normalized.get("name"),
            normalized.get("input"),
        )
        if any(
            (item.get("id"), item.get("name"), item.get("input")) == signature
            for item in calls
        ):
            return
        calls.append(normalized)

    raw_calls = source.get("tool_calls")
    if isinstance(raw_calls, list):
        for index, call in enumerate(raw_calls):
            append_call(call, index)
    function_call = source.get("function_call")
    append_call(function_call, 0)
    if isinstance(payload.get("tool_calls"), list):
        for index, call in enumerate(payload["tool_calls"]):
            append_call(call, index)
    for tool_info in (
        source.get("tool_call_info"),
        payload.get("tool_call_info"),
        source.get("toolCallInfo"),
        payload.get("toolCallInfo"),
    ):
        if isinstance(tool_info, Mapping):
            append_call(tool_info, int(tool_info.get("index") or 0))
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else None
    finish = choice.get("finish_reason") or payload.get("finish_reason")
    return text, reasoning, calls, dict(usage) if usage else None, str(finish or "")


def _custom_response_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    """Extract only model/provider identity fields from an OpenAI payload."""

    pending: list[Mapping[str, Any]] = [payload]
    choices = payload.get("choices")
    if isinstance(choices, list):
        pending.extend(item for item in choices if isinstance(item, Mapping))
    layers: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    while pending and len(layers) < 16:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        layers.append(current)
        for key in (
            "delta",
            "message",
            "model_config",
            "modelConfig",
            "timing_cost",
            "timingCost",
            "data",
        ):
            nested = current.get(key)
            if isinstance(nested, Mapping):
                pending.append(nested)

    def first(*keys: str) -> str:
        for layer in layers:
            for key in keys:
                value = layer.get(key)
                if value not in (None, "", [], {}):
                    return str(value)
        return ""

    provider_model_name = first(
        "provider_model_name",
        "providerModelName",
        "model_provider_name",
        "modelProviderName",
        "actual_model_name",
        "actualModelName",
    )
    raw_model_name = first("raw_model_name", "rawModelName")
    if not provider_model_name:
        provider_model_name = raw_model_name or first("model_name", "modelName")
    identity = {
        "config_name": first("config_name", "configName"),
        "model_name": raw_model_name or provider_model_name,
        "provider_model_name": provider_model_name,
        "provider": first(
            "provider",
            "provider_name",
            "providerName",
            "model_provider",
            "modelProvider",
        ),
    }
    return {key: value for key, value in identity.items() if value}


def _apply_custom_identity(
    payload: dict[str, Any],
    identity: Mapping[str, Any],
    *,
    model: str,
) -> None:
    """Add backward-compatible, credential-free model identity fields."""

    config_name = str(identity.get("config_name") or model or "").strip()
    provider_model_name = str(
        identity.get("provider_model_name") or identity.get("model_name") or ""
    ).strip()
    if config_name:
        payload["config_name"] = config_name
    if provider_model_name:
        payload["model_name"] = str(
            identity.get("model_name") or provider_model_name
        )
        payload["provider_model_name"] = provider_model_name
    provider = str(identity.get("provider") or "").strip()
    if provider:
        payload["provider"] = provider


def _custom_usage_payload(
    usage: Mapping[str, Any],
    *,
    identity: Optional[Mapping[str, Any]] = None,
    model: str = "",
) -> dict[str, Any]:
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", usage.get("input_token", 0)))
    completion = usage.get("completion_tokens", usage.get("output_tokens", usage.get("output_token", 0)))
    total = usage.get("total_tokens", usage.get("total_token"))
    if total is None:
        try:
            total = int(prompt or 0) + int(completion or 0)
        except (TypeError, ValueError):
            total = 0
    result: dict[str, Any] = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }
    # Preserve only accounting fields understood by clients. Never copy the
    # complete upstream object because it may contain opaque provider data.
    for target, aliases in (
        ("cache_read_tokens", ("cache_read_tokens", "cacheReadTokens")),
        ("cache_write_tokens", ("cache_write_tokens", "cacheWriteTokens")),
        (
            "credits_consumed",
            (
                "credits_consumed",
                "consumed_credits",
                "credit_cost",
                "credits_cost",
            ),
        ),
    ):
        for alias in aliases:
            value = usage.get(alias)
            if value is not None:
                result[target] = value
                break
    _apply_custom_identity(result, identity or {}, model=model)
    return result


async def translate_openai_stream_to_traework(
    source: AsyncIterator[Any],
    *,
    model: str = "auto",
    request_id: str = "",
) -> AsyncIterator[str]:
    """Translate relay OpenAI SSE into TraeWork's cumulative raw SSE contract."""

    response_text = ""
    reasoning_text = ""
    calls: OrderedDict[str, dict[str, Any]] = OrderedDict()
    usage: Optional[dict[str, Any]] = None
    response_identity: dict[str, str] = {}
    finish_reason = ""
    emitted = False
    meaningful_output = False
    error_seen = False
    upstream_error: Optional[dict[str, Any]] = None
    reasoning_sent = ""
    done_seen = False

    def merge_call(call: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        """Merge one provider fragment and return the raw-client delta.

        TraeWork's raw transport calls ``tool_calls[].input``/``arguments`` an
        ``argumentsDelta`` and appends it client-side.  The relay still keeps
        a cumulative copy for the final ``done`` event, but must emit only the
        newly observed suffix in each ``output`` frame.  Providers that send a
        cumulative snapshot are handled by subtracting the previous prefix;
        providers that send a true delta are appended as-is.
        """
        index = call.get("_index")
        # OpenAI emits id/name only on the first tool delta; subsequent
        # argument fragments normally carry index alone. Key by index when it
        # is present so those fragments extend the original call instead of
        # being stranded under a synthetic call id.
        key = (
            f"index:{index}"
            if index is not None
            else str(call.get("id") or f"call:{len(calls)}")
        )
        current = calls.get(key)
        if current is None:
            current = {"id": str(call.get("id") or f"call_{len(calls)}"), "name": "", "input": ""}
            calls[key] = current
        is_first = not current.get("name") and not current.get("input")
        if call.get("name"):
            current["name"] = str(call["name"])
        delta: dict[str, Any] = {}
        if is_first and call.get("id"):
            delta["id"] = str(call["id"])
        if call.get("name"):
            delta["name"] = str(call["name"])
        argument = call.get("input")
        if argument not in (None, ""):
            argument = str(argument)
            previous = str(current.get("input") or "")
            if argument.startswith(previous):
                suffix = argument[len(previous):]
                current["input"] = argument
            elif previous.startswith(argument):
                # A shorter cumulative snapshot is a duplicate/stale frame.
                suffix = ""
            else:
                current["input"] = previous + argument
                suffix = argument
            if suffix:
                delta["input"] = suffix
        if not delta:
            return None
        delta["index"] = index if index is not None else 0
        return delta

    async for payload in _iter_buffered_openai_sse_payloads(source):
        if payload == "__DONE__":
            done_seen = True
            continue
        if not isinstance(payload, Mapping):
            continue
        identity_changed = False
        for key, value in _custom_response_identity(payload).items():
            if value and response_identity.get(key) != value:
                response_identity[key] = value
                identity_changed = True
        if isinstance(payload.get("error"), Mapping):
            error_seen = True
            upstream_error = {
                "code": payload["error"].get("code") or "upstream_error",
                "message": payload["error"].get("message") or "Trae upstream error",
                "request_id": request_id,
            }
            for key in ("type", "param"):
                value = payload["error"].get(key)
                if value not in (None, ""):
                    upstream_error[key] = value
            error_payload = dict(upstream_error)
            _apply_custom_identity(
                error_payload, response_identity, model=model
            )
            yield custom_sse_event("error", error_payload)
            done_seen = True
            continue
        text, reasoning, delta_calls, current_usage, finish = _custom_output_from_openai(payload)
        if text:
            response_text += text
        reasoning_delta = ""
        if reasoning:
            # Providers differ on whether reasoning_content is a token
            # delta or a cumulative snapshot. Merge it with the same
            # prefix/stale-frame rules used for tool arguments.
            previous_reasoning = reasoning_text
            if reasoning.startswith(previous_reasoning):
                reasoning_text = reasoning
            elif previous_reasoning.startswith(reasoning):
                reasoning = ""
            else:
                common = 0
                limit = min(len(previous_reasoning), len(reasoning))
                while common < limit and previous_reasoning[common] == reasoning[common]:
                    common += 1
                if common >= 4 or (limit and common * 2 >= limit):
                    reasoning_text = reasoning
                else:
                    reasoning_text = previous_reasoning + reasoning
            compact = compact_reasoning_text(reasoning_text)
            if compact.startswith(reasoning_sent):
                reasoning_delta = compact[len(reasoning_sent):]
            elif compact != reasoning_sent:
                previous_lines = {
                    line.casefold() for line in reasoning_sent.splitlines() if line.strip()
                }
                new_lines = [
                    line for line in compact.splitlines()
                    if line.strip() and line.casefold() not in previous_lines
                ]
                reasoning_delta = "\n".join(new_lines)
                if reasoning_delta and reasoning_sent and not reasoning_delta.startswith("\n"):
                    reasoning_delta = "\n" + reasoning_delta
            reasoning_sent = compact
        emitted_tool_deltas: list[dict[str, Any]] = []
        for call in delta_calls:
            delta_call = merge_call(call)
            if delta_call:
                emitted_tool_deltas.append(delta_call)
        if current_usage:
            usage = current_usage
        if finish:
            finish_reason = finish
        if text or reasoning or emitted_tool_deltas:
            meaningful_output = True
        # The deferred relay stream deliberately starts with an empty
        # OpenAI delta. Preserve that as a real TraeWork output frame so
        # the desktop client does not classify a slow request as an empty
        # response and cancel it before the model speaks.
        if (
            text
            or reasoning
            or emitted_tool_deltas
            or finish_reason
            or identity_changed
            or not emitted
        ):
            output: dict[str, Any] = {
                "response": response_text,
                "model": model,
            }
            _apply_custom_identity(output, response_identity, model=model)
            if reasoning_delta:
                # ``reasoning_delta`` is intentionally separate from the
                # visible response. TraeWork builds that expose a
                # thinking panel can append it incrementally, while
                # clients that do not understand the field simply ignore
                # it without polluting the answer text.
                output["reasoning_delta"] = reasoning_delta
            if emitted_tool_deltas:
                output["tool_calls"] = [dict(value) for value in emitted_tool_deltas]
                if len(emitted_tool_deltas) == 1:
                    only = emitted_tool_deltas[0]
                    snapshot = next(
                        (
                            value
                            for value in calls.values()
                            if value.get("id") == only.get("id")
                        ),
                        only,
                    )
                    output["tool_call_info"] = {
                        "tool_call_id": only.get("id") or next(iter(calls.values()))["id"],
                        "name": only.get("name") or next(iter(calls.values()))["name"],
                        # ``tool_calls[].input`` is a delta, while the
                        # Claude-style tool_call_info field is treated by
                        # some TraeWork builds as a complete call.
                        "parameters": snapshot.get("input") or "",
                    }
            # Do not expose an upstream finish_reason in an intermediate
            # output frame. TraeWork treats that as a terminal signal and can
            # stop rendering while another provider/upstream snapshot still
            # has text. The authoritative value is emitted only in ``done``.
            yield custom_sse_event("output", output)
            emitted = True
    if not meaningful_output and not error_seen:
        # A clean upstream [DONE] with no text, reasoning, or tool call is not
        # a successful answer. Emit an explicit custom error instead of a
        # silent ``done`` event that TraeWork surfaces as an empty response.
        error_seen = True
        empty_error = {
            "code": "empty_response",
            "message": "Trae upstream returned no text or tool call",
            "request_id": request_id,
        }
        _apply_custom_identity(empty_error, response_identity, model=model)
        yield custom_sse_event("error", empty_error)
    if usage:
        # Emit an empty/error terminal marker before usage. If a client closes
        # immediately after the next billable frame, the relay must retain the
        # error state instead of classifying the turn as a successful answer.
        yield custom_sse_event(
            "token_usage",
            _custom_usage_payload(
                usage, identity=response_identity, model=model
            ),
        )
    final: dict[str, Any] = {
        "status": "error" if error_seen else "completed",
        "response": response_text,
        "model": model,
    }
    _apply_custom_identity(final, response_identity, model=model)
    if reasoning_text:
        # Reasoning is a presentation-only side channel. Keep the compact
        # checkpoint summary in the final event rather than forwarding the
        # cumulative upstream trace on every token.
        final["reasoning_content"] = compact_reasoning_text(reasoning_text)
    if calls:
        final["tool_calls"] = [dict(value) for value in calls.values()]
    if finish_reason:
        final["finish_reason"] = finish_reason
    if error_seen and "error" not in final:
        final["error"] = dict(
            upstream_error
            or {
                "code": "empty_response",
                "message": "Trae upstream returned no text or tool call",
            }
        )
    yield custom_sse_event("done", final)


def openai_completion_to_traework(payload: Mapping[str, Any], *, model: str = "auto") -> dict[str, Any]:
    """Convert one non-stream OpenAI completion to the raw custom shape."""

    text, reasoning, calls, usage, finish = _custom_output_from_openai(payload)
    result: dict[str, Any] = {
        "response": text,
        "model": model,
        "finish_reason": finish or ("tool_calls" if calls else "stop"),
    }
    identity = _custom_response_identity(payload)
    _apply_custom_identity(result, identity, model=model)
    if reasoning:
        result["reasoning_content"] = compact_reasoning_text(reasoning)
    if calls:
        result["tool_calls"] = [{key: value for key, value in call.items() if not key.startswith("_")} for call in calls]
        if len(calls) == 1:
            call = result["tool_calls"][0]
            result["tool_call_info"] = {
                "tool_call_id": call["id"],
                "name": call["name"],
                "parameters": call["input"] or "{}",
            }
    if usage:
        result["usage"] = _custom_usage_payload(
            usage, identity=identity, model=model
        )
    return result


def openai_error_to_traework(
    payload: Mapping[str, Any],
    *,
    model: str = "auto",
    request_id: str = "",
) -> dict[str, Any]:
    """Convert an OpenAI-shaped error without losing safe diagnostics."""

    raw_error = payload.get("error") if isinstance(payload, Mapping) else None
    error = raw_error if isinstance(raw_error, Mapping) else {}
    message = str(error.get("message") or "Trae upstream request failed")
    safe_error: dict[str, Any] = {
        "message": message,
        "type": str(error.get("type") or "api_error"),
    }
    for key in ("code", "param"):
        value = error.get(key)
        if value not in (None, ""):
            safe_error[key] = value
    if request_id:
        safe_error["request_id"] = request_id
    result: dict[str, Any] = {
        "response": "",
        "status": "error",
        "model": model,
        "error": safe_error,
    }
    _apply_custom_identity(result, _custom_response_identity(payload), model=model)
    return result


def connectivity_response(
    body: Optional[Mapping[str, Any]] = None,
    *,
    models: Optional[list[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Return a permissive connectivity/model descriptor TraeWork accepts."""

    body = body if isinstance(body, Mapping) else {}
    layers = _body_layers(body)
    custom = _custom_model_mapping(layers)
    requested = (
        _specific_layer_value(
            layers,
            "config_name",
            "configName",
            "ai_model_name",
            "aiModelName",
        )
        or _first_layer_value(layers, "model")
        or custom.get("config_name")
        or custom.get("configName")
        or custom.get("model_name")
        or custom.get("modelName")
        or "auto"
    )
    try:
        config_source = int(
            custom.get("config_source")
            or custom.get("configSource")
            or 1
        )
    except (TypeError, ValueError):
        config_source = 1
    # This endpoint advertises the relay as a TraeWork custom Base URL.  A
    # missing identity flag must therefore default to a custom model; otherwise
    # the desktop client reads the returned model as a preset and silently
    # switches back to its native provider route.  Explicit flags are still
    # honored for callers that intentionally probe a preset model.
    explicit_preset = (
        custom.get("is_preset")
        if custom.get("is_preset") is not None
        else custom.get("model_is_preset")
    )
    is_preset = bool(explicit_preset) if explicit_preset is not None else False
    custom_base_url = custom.get("is_custom_base_url")
    if custom_base_url is None:
        custom_base_url = None if is_preset else True
    custom_model_id = (
        custom.get("custom_model_id")
        or custom.get("customModelId")
        or (None if is_preset else None)
    )
    binding = resolve_model_binding(str(requested), {
        "trae_raw_config_name": custom.get("config_name") or custom.get("configName"),
        "trae_raw_model_name": custom.get("raw_model_name") or custom.get("rawModelName") or custom.get("model_name") or custom.get("modelName"),
        "display_name": custom.get("display_name") or custom.get("displayName"),
        "provider": custom.get("provider") or "trae-cn-relay",
        "config_source": config_source,
    })
    model_item = {
        "id": binding.config_name,
        "name": binding.config_name,
        "model": binding.config_name,
        "model_name": binding.raw_model_name,
        "config_name": binding.config_name,
        "display_name": binding.display_name,
        "provider": binding.provider or "trae-cn-relay",
        "is_preset": is_preset,
        "is_custom_base_url": custom_base_url,
        "custom_model_id": custom_model_id,
        "config_source": binding.config_source,
        "context_window_size": {"default": 200000, "max": 1000000},
        "prompt_max_tokens": 936000,
        "max_tokens": 64000,
        "features": {
            "reasoning": {"enable": True},
            "tool_calls": {"enable": True},
            "context_windows": {"enable": True, "data": {"dev_context": 200000, "max_context": 1000000}},
        },
    }
    if models:
        model_list: list[dict[str, Any]] = []
        matched = False
        for item in models:
            if not isinstance(item, Mapping):
                continue
            candidate = dict(item)
            identity = str(
                candidate.get("id")
                or candidate.get("config_name")
                or candidate.get("name")
                or ""
            )
            if identity == binding.config_name:
                # Keep provider-specific capability fields while overriding
                # the identity fields that control TraeWork's route choice.
                candidate.update(model_item)
                matched = True
            model_list.append(candidate)
        if not matched:
            model_list.insert(0, model_item)
    else:
        model_list = [model_item]
    return {
        "success": True,
        "ok": True,
        "code": 0,
        "message": "ok",
        "provider": model_item["provider"],
        "config_name": binding.config_name,
        "model_name": binding.raw_model_name,
        "display_name": binding.display_name,
        "model": model_item,
        "models": model_list,
        "data": {
            "success": True,
            "code": 0,
            "model": model_item,
            "models": model_list,
        },
    }


__all__ = [
    "TRAEWORK_RAW_CHAT_PATH",
    "TRAEWORK_REMOTE_SESSION_PATH",
    "TRAEWORK_CUSTOM_CONNECTIVITY_PATH",
    "TRAEWORK_CUSTOM_RAW_ALIASES",
    "TRAEWORK_CONNECTIVITY_ALIASES",
    "TraeWorkInboundRequest",
    "TraeWorkModelBinding",
    "TraeWorkRequestDescriptor",
    "TraeWorkRequestIds",
    "build_raw_chat_request",
    "build_raw_extra_header",
    "build_remote_session_request",
    "build_request_ids",
    "parse_raw_extra_header",
    "remote_events_path",
    "resolve_model_binding",
    "is_traework_request",
    "normalize_inbound_request",
    "custom_sse_event",
    "translate_openai_stream_to_traework",
    "openai_completion_to_traework",
    "openai_error_to_traework",
    "connectivity_response",
]
