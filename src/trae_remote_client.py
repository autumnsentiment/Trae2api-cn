"""9router-style Trae remote session transport.

The implementation follows 9router's provider executor, but keeps this
relay's existing CN endpoint and account store.  It only forwards the remote
session and parses SSE; tools remain owned by the API caller.
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Mapping, Optional

import httpx

from . import auth, trae_client
from .sse import EmptyUpstreamResponse


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://trae-api-cn.mchost.guru/api/remote/v1"
DEFAULT_ORIGIN = "https://solo.trae.cn"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)

DEFAULT_MAX_CONTEXT_TOKENS = 1_000_000
DEFAULT_MAX_PROMPT_TOKENS = 936_000
DEFAULT_MAX_OUTPUT_TOKENS = 64_000
DEFAULT_MAX_MODE_TYPE = 1


class RemoteFirstEventTimeout(EmptyUpstreamResponse):
    """A created remote session ended or timed out before its first event."""

    def __init__(self, message: str):
        super().__init__(
            message,
            retryable=True,
            observed_model_event=False,
        )


class RemoteStreamReadTimeout(EmptyUpstreamResponse):
    """The remote event stream timed out after upstream activity began."""

    def __init__(self, message: str):
        super().__init__(
            message,
            retryable=False,
            observed_model_event=True,
        )


def base_url(options: Optional[Mapping[str, Any]] = None) -> str:
    options = options or {}
    configured = options.get("base_url") or options.get("baseURL")
    value = configured or os.environ.get("TRAE_WEB_BASE_URL") or DEFAULT_BASE_URL
    return str(value).rstrip("/")


def _provider_specific(options: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    if isinstance(options, Mapping):
        for key in ("provider_specific", "providerSpecificData"):
            if key not in options:
                continue
            value = options.get(key)
            if isinstance(value, Mapping):
                # Preserve an explicitly bound empty mapping. Falling back to
                # global metadata here can mix concurrently rotated accounts.
                return dict(value)
    return auth.get_psd()


def _psd_value(psd: Mapping[str, Any], key: str, default: Any = "") -> Any:
    """Read a provider field while tolerating browser-captured nested values."""
    value = psd.get(key)
    if isinstance(value, str) and value.lstrip().startswith("{"):
        # A few older account snapshots persisted a Python-dict rendering
        # instead of structured JSON.  Parse only literal mappings and never
        # evaluate arbitrary expressions.
        parsed: Any = None
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, Mapping):
            value = parsed
    if isinstance(value, Mapping):
        # Some captured CN storage snapshots keep region as
        # ``{"region":"CN", "_aiRegion":"CN"}``.
        for nested in ("region", "value", "name", "code", "_aiRegion"):
            candidate = value.get(nested)
            if candidate not in (None, "") and not isinstance(candidate, Mapping):
                return candidate
        return default
    return value if value not in (None, "") else default


def build_headers(
    token: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
    stream: bool = True,
) -> dict[str, str]:
    psd = _provider_specific(options)
    base = base_url(options).lower()
    intl = "trae.ai" in base and "trae-api-cn" not in base
    language_default = "en" if intl else "zh-CN"
    region_default = "US" if intl else "CN"
    origin_default = "https://solo.trae.ai" if intl else DEFAULT_ORIGIN
    origin = os.environ.get("TRAE_WEB_ORIGIN") or origin_default
    return {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
        "X-Trae-Client-Type": os.environ.get("TRAE_WEB_CLIENT_TYPE", "web"),
        "X-Preferenced-Language": str(
            _psd_value(psd, "appLanguage") or os.environ.get("TRAE_WEB_LANGUAGE") or language_default
        ),
        "x-user-region": str(
            _psd_value(psd, "userRegion") or os.environ.get("TRAE_WEB_USER_REGION") or region_default
        ),
        "Origin": origin,
        "Referer": origin.rstrip("/") + "/",
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/event-stream" if stream else "application/json",
    }


def resolve_mode(model: str) -> tuple[str, str, str]:
    value = (model or "").strip()
    lowered = value.lower()
    if lowered in {"work", "auto-work", "solo-work"}:
        return "work", "auto", ""
    if not value or lowered == "auto":
        return "code", "auto", ""
    return "code", "manual", trae_client.convert_model_name(value)


def common_params(
    psd: Mapping[str, Any],
    mode: str,
    session_id: str = "",
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    base = base_url(options).lower()
    intl = "trae.ai" in base and "trae-api-cn" not in base
    default_language = "en" if intl else "zh-CN"
    default_scope = "marscode-us" if intl else "marscode-cn"
    default_region = "US-East" if intl else "cn"
    params: dict[str, Any] = {
        "language": "en-us" if intl else "zh-cn",
        "app_language": _psd_value(psd, "appLanguage", default_language),
        "quality": "stable",
        "app_version": psd.get("appVersion") or "1.0.0.1229",
        "web_id": _psd_value(psd, "webId"),
        "user_identity": _psd_value(psd, "userIdentity", "Free"),
        "is_freshman": "0",
        "biz_user_id": _psd_value(psd, "bizUserId"),
        "user_unique_id": _psd_value(psd, "userUniqueId"),
        "scope": _psd_value(psd, "scope", default_scope),
        "tenant": _psd_value(psd, "tenant", "marscode"),
        "region": _psd_value(psd, "region", default_region),
        "aiRegion": _psd_value(psd, "aiRegion", _psd_value(psd, "region", default_region)),
        "is_privacy_mode": 0,
        "privacy_mode": "off",
        "solo_chat_mode": mode,
    }
    if session_id:
        params["biz_session_id"] = session_id
    return json.dumps(params, ensure_ascii=False)


def flatten_query(messages: list[dict[str, Any]]) -> str:
    return trae_client.flatten_query(messages)


def model_session_id(model: str, options: Optional[Mapping[str, Any]] = None) -> str:
    """Return a stable non-secret remote session key per account and model."""

    options = options or {}
    explicit = options.get("trae_remote_session_id") or options.get(
        "traeRemoteSessionId"
    )
    if explicit:
        return str(explicit)
    account = str(
        options.get("_billing_id")
        or options.get("_auth_user_id")
        or options.get("_account_id")
        or "default"
    )
    variant = str(options.get("_session_variant") or "")
    material = "\x1f".join((account, str(model or "auto"), variant))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_flag_default(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def remote_agent_type(
    model: str,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the upstream executor tier for one remote session.

    The desktop protocol distinguishes Agent and Work by ``agent_type``;
    keeping this decision in one place prevents a Work fallback from silently
    reusing the 1M Agent configuration.
    """

    options = options or {}
    explicit = str(
        options.get("_remote_agent_type")
        or options.get("remote_agent_type")
        or ""
    ).strip().lower()
    if explicit in {"solo_work_remote", "solo_work_lite", "work"}:
        return "solo_work_remote"
    if explicit in {"solo_agent_remote", "solo_agent_lite", "agent"}:
        return "solo_agent_remote"
    mode, _strategy, _model_name = resolve_mode(model)
    if mode != "work" and not _env_flag_default("TRAE_REMOTE_AGENT_FIRST", True):
        return "solo_work_remote"
    return "solo_work_remote" if mode == "work" else "solo_agent_remote"


def _max_mode_requested(
    model_name: str,
    custom_model: Optional[Mapping[str, Any]],
    options: Optional[Mapping[str, Any]],
) -> bool:
    """Decide whether a manual remote session should use the 1M max profile."""

    options = options or {}
    if remote_agent_type(model_name, options) != "solo_agent_remote":
        return False
    raw_flag = options.get("trae_max_mode") or options.get("max_mode")
    if isinstance(raw_flag, str):
        enabled = raw_flag.strip().lower() in ("1", "true", "yes", "on")
    else:
        enabled = bool(raw_flag)
    enabled = enabled or _env_flag("TRAE_REMOTE_MAX_MODE")
    if not enabled:
        return False
    configured = {
        item.strip().lower()
        for item in os.environ.get("TRAE_REMOTE_MAX_MODELS", "").split(",")
        if item.strip()
    }
    if configured and "*" not in configured:
        value = str(model_name or "").strip().lower()
        if value not in configured:
            return False
    # Never fabricate max limits for a model the account config does not mark.
    return bool(custom_model and custom_model.get("max_mode"))


def _max_mode_custom_model(custom_model: Mapping[str, Any]) -> dict[str, Any]:
    """Return a max-enriched custom_model mirroring the desktop config.

    The remote models endpoint returns a slim object (max_mode flag plus
    context_window_tokens) that omits the fields the upstream uses to accept a
    max session.  Fill them from the account values or the documented defaults
    so the server validates the same 1M profile the local client sends.
    """

    cws = custom_model.get("context_window_size") or {}
    raw_max = cws.get("max") or []
    if isinstance(raw_max, int):
        raw_max = [raw_max]
    raw_features = custom_model.get("features") or {}
    if isinstance(raw_features, str):
        try:
            raw_features = json.loads(raw_features)
        except (TypeError, ValueError):
            raw_features = {}
    feature_cw = (raw_features or {}).get("context_windows") or {}
    feature_data = feature_cw.get("data") or {}
    tokens = custom_model.get("context_window_tokens") or {}
    try:
        max_context = int((raw_max or [feature_data.get("max_context") or tokens.get("max") or DEFAULT_MAX_CONTEXT_TOKENS])[0])
    except (TypeError, ValueError):
        max_context = DEFAULT_MAX_CONTEXT_TOKENS
    try:
        dev_context = int(
            cws.get("default")
            or feature_data.get("dev_context")
            or tokens.get("dev")
            or 200_000
        )
    except (TypeError, ValueError):
        dev_context = 200_000
    try:
        prompt_max = int(custom_model.get("prompt_max_tokens") or DEFAULT_MAX_PROMPT_TOKENS)
    except (TypeError, ValueError):
        prompt_max = DEFAULT_MAX_PROMPT_TOKENS
    try:
        output_max = int(custom_model.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
    except (TypeError, ValueError):
        output_max = DEFAULT_MAX_OUTPUT_TOKENS
    try:
        max_turn = int(
            custom_model.get("max_turn")
            or (custom_model.get("max_turns") or {}).get("default")
            or feature_data.get("max_turns")
            or 500
        )
    except (TypeError, ValueError):
        max_turn = 500

    enriched = dict(custom_model)
    enriched["max_mode"] = True
    enriched["context_window_size"] = {"default": dev_context, "max": [max_context]}
    enriched["context_window_tokens"] = {"dev": dev_context, "max": max_context}
    enriched["prompt_max_tokens"] = prompt_max
    enriched["max_tokens"] = output_max
    enriched["max_turn"] = max_turn
    enriched["max_turns"] = {"default": max_turn, "max": max_turn}
    features = dict(raw_features)
    cw = dict(feature_cw)
    cw["enable"] = True
    cw["data"] = {
        "dev_context": dev_context,
        "max_context": max_context,
        "max_context_list": [max_context],
        "dev_turns": max_turn,
        "max_turns": max_turn,
    }
    features["context_windows"] = cw
    enriched["features"] = features
    return enriched


def _max_mode_fields(
    custom_model: Mapping[str, Any],
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the wire fields that pin a remote session to max mode."""

    cws = custom_model.get("context_window_size") or {}
    raw_max = cws.get("max") or []
    if isinstance(raw_max, int):
        raw_max = [raw_max]
    try:
        max_context = int((raw_max or [DEFAULT_MAX_CONTEXT_TOKENS])[0])
    except (TypeError, ValueError):
        max_context = DEFAULT_MAX_CONTEXT_TOKENS
    try:
        prompt_max = int(
            custom_model.get("prompt_max_tokens") or DEFAULT_MAX_PROMPT_TOKENS
        )
    except (TypeError, ValueError):
        prompt_max = DEFAULT_MAX_PROMPT_TOKENS
    try:
        output_max = int(custom_model.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
    except (TypeError, ValueError):
        output_max = DEFAULT_MAX_OUTPUT_TOKENS
    try:
        mode_type = int(
            os.environ.get("TRAE_REMOTE_MAX_MODE_TYPE", "") or DEFAULT_MAX_MODE_TYPE
        )
    except (TypeError, ValueError):
        mode_type = DEFAULT_MAX_MODE_TYPE
    return {
        "model_auto_selection": {
            "strategy": "max",
            "fallback_to_advance_model": None,
            "entitlement_id": None,
        },
        "model_selection_strategy": "max",
        "mode_type": mode_type,
        "context_window_size": max_context,
        "prompt_max_tokens": prompt_max,
        "max_tokens": output_max,
    }


async def create_session(
    client: httpx.AsyncClient,
    token: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> tuple[str, str]:
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    mode, strategy, model_name = resolve_mode(model)
    agent_type = remote_agent_type(model, options)
    provider_data = _provider_specific(options)
    custom_model: Optional[dict[str, Any]] = None
    if strategy == "manual":
        # The remote endpoint silently chooses its default model when a
        # manual request omits the complete model object.  Resolve it with
        # the bound token so a concurrent account switch cannot leak model
        # metadata between accounts.
        bound_user_id = str(
            (options or {}).get("_auth_user_id")
            or (options or {}).get("_billing_id")
            or (options or {}).get("_account_id")
            or ""
        ).strip()
        resolve_kwargs: dict[str, Any] = {
            "token_override": token,
            "user_id_override": bound_user_id,
            "provider_specific": provider_data,
        }
        if agent_type != "solo_agent_remote" or any(
            key in (options or {}) for key in ("_remote_agent_type", "remote_agent_type")
        ):
            resolve_kwargs["agent_type"] = agent_type
        custom_model = await trae_client.resolve_model_config(
            model_name,
            **resolve_kwargs,
        )
        if not custom_model:
            raise RuntimeError(
                f"Trae remote model is not available for the bound account: {model_name}"
            )
    max_requested = _max_mode_requested(model_name, custom_model, options)
    effective_strategy = "max" if max_requested else strategy
    session_options = dict(options or {})
    if max_requested:
        session_options["_session_variant"] = "max"
    stable_session_id = model_session_id(model_name or model, session_options)
    initial_message: dict[str, Any] = {
        "chat_session_id": "",
        "content": [],
        "query": flatten_query(messages),
        "model_name": model_name,
        "agent_type": agent_type,
        "model_selection_strategy": effective_strategy,
        "common_params": common_params(
            provider_data,
            mode,
            stable_session_id,
            options=options,
        ),
    }
    if strategy == "manual":
        initial_message["model_name"] = str(
            custom_model.get("model_name")
            or custom_model.get("config_name")
            or custom_model.get("name")
            or model_name
        )
        initial_message["model_config_source"] = int(
            custom_model.get("config_source") or 1
        )
        initial_message["model_is_preset"] = bool(
            custom_model.get("is_preset", True)
        )
        initial_message["model_provider"] = str(
            custom_model.get("provider") or ""
        )
        initial_message["custom_model"] = custom_model
    if max_requested:
        max_custom_model = _max_mode_custom_model(custom_model)
        initial_message["custom_model"] = max_custom_model
        initial_message.update(_max_mode_fields(max_custom_model, options))
    body = {
        "mode": mode,
        "environment_id": "default",
        "initial_message": initial_message,
        "env": "remote",
        "auto_create_project": False,
        "origin": "web",
    }
    logger.info(
        "remote upstream request model=%s url=%s body_bytes=%d messages=%d query_chars=%d",
        model,
        f"{base_url(options)}/chat_sessions",
        len(json.dumps(body, ensure_ascii=False, separators=(",", ":"))),
        len(messages),
        len(str(body["initial_message"].get("query") or "")),
    )
    response = await client.post(
        f"{base_url(options)}/chat_sessions",
        headers=build_headers(token, options=options, stream=False),
        json=body,
        timeout=float(os.environ.get("TRAE_WEB_CONNECT_TIMEOUT", "60")),
    )
    text = response.text
    if response.status_code >= 400:
        raise RuntimeError(f"Trae remote create_session [{response.status_code}]: {text[:500]}")
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Trae remote invalid create_session response: {text[:500]}") from exc
    if payload.get("code") not in (None, 0) and not payload.get("data"):
        raise RuntimeError(f"Trae remote create_session: {payload}")
    data = payload.get("data") or payload
    session_id = str(data.get("chat_session_id") or "")
    message_id = str(data.get("message_id") or "")
    if not session_id or not message_id:
        raise RuntimeError(f"Trae remote create_session missing ids: {payload}")
    return session_id, message_id


async def _stream_events_unbounded(
    client: httpx.AsyncClient,
    token: str,
    session_id: str,
    message_id: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    url = f"{base_url(options)}/chat_sessions/{session_id}/events?reply_to_message_id={message_id}"
    timeout = httpx.Timeout(
        float(os.environ.get("TRAE_WEB_STREAM_TIMEOUT", os.environ.get("STREAM_TIMEOUT", "300"))),
        connect=float(os.environ.get("TRAE_WEB_CONNECT_TIMEOUT", "60")),
    )
    async with client.stream(
        "GET", url, headers=build_headers(token, options=options, stream=True), timeout=timeout
    ) as response:
        if response.status_code >= 400:
            body = await response.aread()
            raise RuntimeError(f"Trae remote events [{response.status_code}]: {body[:500]}")
        event_name: Optional[str] = None
        buffer = ""
        async for raw in response.aiter_bytes():
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line == "":
                    event_name = None
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    yield "done", {}
                    return
                try:
                    data = json.loads(payload)
                except Exception:
                    data = {"_raw": payload}
                if not isinstance(data, dict):
                    data = {"value": data}
                yield event_name or str(data.get("event") or "message"), data
                event_name = None
        if buffer.strip().startswith("data:"):
            payload = buffer.strip()[5:].strip()
            try:
                data = json.loads(payload)
            except Exception:
                data = {"_raw": payload}
            if isinstance(data, dict):
                yield "message", data


async def stream_events(
    client: httpx.AsyncClient,
    token: str,
    session_id: str,
    message_id: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Stream events while bounding a session that is silent after creation."""

    source = _stream_events_unbounded(
        client,
        token,
        session_id,
        message_id,
        options=options,
    )
    raw_timeout = os.environ.get(
        "TRAE_REMOTE_FIRST_EVENT_TIMEOUT_SECONDS",
        os.environ.get("TRAE_WEB_FIRST_EVENT_TIMEOUT", "120"),
    )
    try:
        first_event_timeout = float(raw_timeout)
    except (TypeError, ValueError):
        first_event_timeout = 120.0
    try:
        if first_event_timeout > 0:
            try:
                first = await asyncio.wait_for(
                    source.__anext__(), timeout=first_event_timeout
                )
            except StopAsyncIteration as exc:
                raise RemoteFirstEventTimeout(
                    "Trae remote session ended before its first event"
                ) from exc
            except (asyncio.TimeoutError, httpx.ReadTimeout) as exc:
                raise RemoteFirstEventTimeout(
                    "Trae remote session emitted no event before the first-event timeout"
                ) from exc
        else:
            try:
                first = await source.__anext__()
            except StopAsyncIteration as exc:
                raise RemoteFirstEventTimeout(
                    "Trae remote session ended before its first event"
                ) from exc
            except httpx.ReadTimeout as exc:
                raise RemoteFirstEventTimeout(
                    "Trae remote session timed out before its first event"
                ) from exc
        yield first
        try:
            async for event in source:
                yield event
        except httpx.ReadTimeout as exc:
            raise RemoteStreamReadTimeout(
                "Trae remote event stream timed out after upstream activity began"
            ) from exc
    finally:
        await source.aclose()


async def stop_session(
    client: httpx.AsyncClient,
    token: str,
    session_id: str,
    message_id: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    if not session_id or not message_id or not token:
        return
    try:
        await client.post(
            f"{base_url(options)}/chat_sessions/{session_id}/stop",
            headers=build_headers(token, options=options, stream=False),
            json={"chat_session_id": session_id, "user_message_id": message_id},
            timeout=10,
        )
    except Exception:
        return
