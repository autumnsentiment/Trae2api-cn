"""9router-style Trae remote session transport.

The implementation follows 9router's provider executor, but keeps this
relay's existing CN endpoint and account store.  It only forwards the remote
session and parses SSE; tools remain owned by the API caller.
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
import json
import logging
import os
import re
import uuid
from typing import Any, AsyncIterator, Mapping, Optional

import httpx

from . import auth, trae_client
from .sse import EmptyUpstreamResponse
from .reasoning_effort import (
    apply_reasoning_effort,
    clamp_level,
    requested_level as requested_effort_level,
    supported_levels,
)


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



class RemoteStreamIncomplete(EmptyUpstreamResponse):
    """The remote event stream ended without an explicit terminal event."""

    def __init__(self, message: str):
        super().__init__(
            message,
            retryable=False,
            observed_model_event=True,
        )

class RemoteFirstEventError(EmptyUpstreamResponse):
    """The remote session failed before the selected model produced output."""

    def __init__(self, message: str):
        super().__init__(
            message,
            retryable=True,
            observed_model_event=False,
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


def _work_client_type_enabled(
    options: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether the remote executor should send the Work lite header."""

    options = options or {}
    mode = str(
        options.get("_trae_mode")
        or os.environ.get("TRAE_REMOTE_FORCE_MODE", "")
    ).strip().lower()
    agent = str(
        options.get("_remote_agent_type")
        or options.get("remote_agent_type")
        or ""
    ).strip().lower()
    if mode != "work" and "work" not in agent:
        return False
    env = os.environ.get("TRAE_REMOTE_WORK_BILLING", "").strip().lower()
    return env in ("1", "true", "yes", "on")


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
    headers = {
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
    overlay = _work_billing_overlay(options)
    if _work_client_type_enabled(options):
        headers["X-Trae-Client-Type"] = "lite"
    if overlay:
        client_type = overlay.get("client_type")
        if client_type not in (None, ""):
            headers["X-Trae-Client-Type"] = str(client_type)
        traffic_type = overlay.get("traffic_type")
        if traffic_type not in (None, ""):
            headers["request-traffic-type"] = str(traffic_type)
        if overlay.get("x_trae_work"):
            headers["x-trae-work"] = "1"
        if overlay.get("x_bridge_transport"):
            headers["x-bridge-transport"] = "aha"
    return headers


def _work_billing_overlay(
    options: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Return the experimental Work-billing field overlay, when enabled."""

    options = options or {}
    raw = options.get("_work_billing_overlay")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw not in (None, False, True):
        return None
    return {
        "entitlement_id": "335006824194",
        "available_endpoint": 1,
        "billing_source": "work",
        "product_code": "SOLO_Lite",
        "client_type": "lite",
        "traffic_type": "prod",
        "x_trae_work": True,
        "x_bridge_transport": True,
        "origin": "lite",
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
    options = options or {}
    base = base_url(options).lower()
    intl = "trae.ai" in base and "trae-api-cn" not in base
    default_language = "en" if intl else "zh-CN"
    default_scope = "marscode-us" if intl else "marscode-cn"
    default_region = "US-East" if intl else "cn"
    account_id = str(
        options.get("_auth_user_id")
        or options.get("_billing_id")
        or options.get("_account_id")
        or _psd_value(psd, "bizUserId")
        or ""
    )
    token = str(options.get("_auth_token") or options.get("auth_token") or "")
    if token:
        try:
            device_id = trae_client.checkin_device_id_for(token, account_id)
        except Exception:
            device_id = _psd_value(psd, "deviceId") or _psd_value(
                psd, "device_id", ""
            )
    else:
        device_id = _psd_value(psd, "deviceId") or _psd_value(
            psd, "device_id", ""
        )
    if not device_id:
        digest = hashlib.sha256(
            f"trae{account_id}".encode("utf-8")
        ).hexdigest()
        device_id = str(int(digest[:32], 16) % 10**16).zfill(16)
    machine_seed = "\x1f".join(
        [account_id, str(device_id), "traework-linux-probe"]
    )
    machine_id = hashlib.sha256(machine_seed.encode("utf-8")).hexdigest()
    local_device_id = "aha-" + hashlib.sha256(
        f"local-{account_id}-{device_id}".encode("utf-8")
    ).hexdigest()[:32]
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
    if str(mode).strip().lower() == "work":
        params.update(
            {
                "icube_uid": account_id,
                "user_id": account_id,
                "device_id": device_id,
                "local_device_id": local_device_id,
                "machine_id": machine_id,
                "arch": "x64",
                "system": "linux",
                "scope": "marscode",
                "tenant": "marscode",
                "region": "CN",
                "aiRegion": "CN",
                "build_version": "2.3.79946",
                "vscode_version": "1.107.1",
                "app_version": "3.3.96",
                "os_name": "linux",
                "os_version": "Linux x64",
                "os_release": "6.1",
                "platform": "electron",
                "device_model": "TraeCode Linux",
                "device_manufacturer": "TraeCode",
                "cpu": "x86_64",
                "cpu_brand": "x86_64",
                "cpu_speed": 0,
                "memory": 0,
                "chat_mode": 1,
                "identity": "0",
                "identity_str": "Free",
                "product_code": "SOLO_Lite",
                "message_source": "manual",
            }
        )
    overlay = options.get("_trae_common_params")
    if isinstance(overlay, Mapping):
        params.update(overlay)
    work_overlay = _work_billing_overlay(options)
    if work_overlay:
        common_updates = {}
        if work_overlay.get("product_code") not in (None, ""):
            common_updates["product_code"] = str(
                work_overlay.get("product_code")
            )
        if work_overlay.get("entitlement_id") not in (None, ""):
            common_updates["entitlement_id"] = str(
                work_overlay.get("entitlement_id")
            )
        if work_overlay.get("available_endpoint") not in (None, ""):
            common_updates["available_endpoint"] = int(
                work_overlay.get("available_endpoint")
            )
        if work_overlay.get("billing_source") not in (None, ""):
            common_updates["billing_source"] = str(
                work_overlay.get("billing_source")
            )
        if common_updates:
            params.update(common_updates)
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


def _callable_accepts_keyword(callback: Any, keyword: str) -> bool:
    """Return whether a resolver supports a newer optional keyword.

    Some deployments replace ``resolve_model_config`` with an older adapter.
    Keep that extension point working while using tier-aware lookup whenever
    the installed resolver supports it.
    """

    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return True
    # ``**kwargs`` alone is deliberately not treated as proof that a dynamic
    # adapter implements the newer tier argument.  This keeps compatibility
    # with old wrappers/mocks while concrete resolvers (which expose the named
    # parameter) still receive the Agent binding.
    return any(parameter.name == keyword for parameter in parameters)


async def _resolve_model_config(
    model_name: str,
    *,
    token: str,
    user_id: str,
    provider_data: Mapping[str, Any],
    agent_type: str,
    force_agent_type: bool = False,
) -> Optional[dict[str, Any]]:
    """Resolve one model against the executor tier without breaking old hooks."""

    resolver = trae_client.resolve_model_config
    kwargs: dict[str, Any] = {
        "token_override": token,
        "user_id_override": user_id,
        "provider_specific": provider_data,
    }
    tier_aware = force_agent_type or _callable_accepts_keyword(resolver, "agent_type")
    if tier_aware:
        kwargs["agent_type"] = agent_type
    try:
        return await resolver(model_name, **kwargs)
    except TypeError as exc:
        # A dynamically wrapped legacy resolver can hide its real signature.
        # Retry only the precise unsupported-keyword case; never mask a
        # TypeError raised by the resolver's own implementation.
        message = str(exc)
        if not tier_aware or "agent_type" not in message or "keyword" not in message:
            raise
        kwargs.pop("agent_type", None)
        logger.info("remote model resolver does not support agent_type; using legacy lookup")
        return await resolver(model_name, **kwargs)


async def _work_tier_reasoning_effort(
    custom_model: Any,
    model_name: str,
    options: Optional[Mapping[str, Any]],
    *,
    token: str,
    provider_data: Mapping[str, Any],
) -> tuple[Any, str]:
    """Clamp a Work-tier request against the Agent-tier effort options."""

    if not isinstance(custom_model, Mapping):
        return custom_model, ""
    user_id = str(
        (options or {}).get("_auth_user_id")
        or (options or {}).get("_billing_id")
        or (options or {}).get("_account_id")
        or ""
    ).strip()
    try:
        agent_model = await _resolve_model_config(
            model_name,
            token=token,
            user_id=user_id,
            provider_data=provider_data,
            agent_type="solo_agent_remote",
            force_agent_type=True,
        )
    except Exception as exc:  # lookup failure must not break the chat
        logger.info("work effort lookup failed model=%s err=%s", model_name, exc)
        return custom_model, ""
    level = clamp_level(
        requested_effort_level(options), supported_levels(agent_model)
    )
    if not level:
        return custom_model, ""
    updated = dict(custom_model)
    updated["reasoning_effort_config"] = dict(
        (agent_model or {}).get("reasoning_effort_config") or {}
    )
    if "reasoning_effort_level" in updated:
        updated["reasoning_effort_level"] = level
        updated.pop("reasoning_effort", None)
    else:
        updated["reasoning_effort"] = level
    return updated, level


def _model_family(value: Any) -> str:
    """Normalize config/runtime spellings for preflight model binding checks."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = str(trae_client.convert_model_name(text) or text).strip().lower()
    if text.startswith("trae/"):
        text = text[5:]
    text = text.split("__", 1)[0].replace("_", "-")
    if text.startswith("ali-deepseek-v4-"):
        text = text[4:]
    deepseek = re.fullmatch(
        r"(deepseek-v4-(?:pro|flash))(?:-(?:official|\d{4,8}))*", text
    )
    if deepseek:
        return deepseek.group(1)
    if text.endswith("-official"):
        text = text[: -len("-official")]
    return text


def _bound_model_name(
    requested_model: str,
    custom_model: Mapping[str, Any],
) -> str:
    """Return the exact config id and reject silent default-model fallback."""

    requested_family = _model_family(requested_model)
    candidates = [
        str(custom_model.get(key) or "").strip()
        for key in ("config_name", "model_name", "name")
    ]
    candidates = [candidate for candidate in candidates if candidate]
    selected = (
        next(
            (
                candidate
                for candidate in candidates
                if not requested_family
                or _model_family(candidate) == requested_family
            ),
            candidates[0],
        )
        if candidates
        else ""
    )
    resolved_families = {
        _model_family(custom_model.get(key))
        for key in (
            "config_name",
            "model_name",
            "name",
            "display_name",
            "display_model_name",
        )
        if custom_model.get(key)
    }
    if not selected or (
        requested_family
        and resolved_families
        and requested_family not in resolved_families
    ):
        raise RuntimeError(
            "Trae remote model binding mismatch: "
            f"requested {requested_model!r}, resolved {selected or '<empty>'!r}"
        )
    return selected


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
    forced_mode = str(
        (options or {}).get("_trae_mode")
        or os.environ.get("TRAE_REMOTE_FORCE_MODE", "")
    ).strip().lower()
    if forced_mode in ("work", "code"):
        mode = forced_mode
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
        custom_model = await _resolve_model_config(
            model_name,
            token=token,
            user_id=bound_user_id,
            provider_data=provider_data,
            agent_type=agent_type,
            force_agent_type=(
                agent_type != "solo_agent_remote"
                or any(
                    key in (options or {})
                    for key in ("_remote_agent_type", "remote_agent_type")
                )
            ),
        )
        if not custom_model:
            raise RuntimeError(
                f"Trae remote model is not available for the bound account: {model_name}"
            )
        bound_model_name = _bound_model_name(model_name, custom_model)
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
        # TraeWork sends both fields.  ``agent_type`` selects the executor
        # tier while ``agent_id`` pins the session to that tier; omitting the
        # latter lets the remote service silently resolve to its default
        # (often Kimi/Work), which breaks model binding and caller-tool
        # routing.
        "agent_id": agent_type,
        "model_selection_strategy": effective_strategy,
        "common_params": common_params(
            provider_data,
            mode,
            stable_session_id,
            options=options,
        ),
    }
    if strategy == "manual":
        initial_message["model_name"] = bound_model_name
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
    max_trace = (options or {}).get("_upstream_trace")
    if isinstance(max_trace, dict):
        # The last created session wins, so a Work fallback reports "off".
        max_trace["max_mode_applied"] = bool(max_requested)
        if max_requested:
            max_trace["max_context_tokens"] = initial_message.get("context_window_size")
        else:
            max_trace.pop("max_context_tokens", None)
    if isinstance(initial_message.get("custom_model"), Mapping):
        effort_model, effort_level = apply_reasoning_effort(
            initial_message["custom_model"], options
        )
        if (
            not effort_level
            and requested_effort_level(options)
            and agent_type != "solo_agent_remote"
        ):
            # Work-tier model entries omit ``reasoning_effort_config``; the
            # same model's Agent-tier entry advertises the supported levels.
            effort_model, effort_level = await _work_tier_reasoning_effort(
                effort_model,
                model_name,
                options,
                token=token,
                provider_data=provider_data,
            )
        initial_message["custom_model"] = effort_model
        if effort_level:
            logger.info(
                "remote reasoning effort model=%s level=%s", model_name, effort_level
            )
            trace = (options or {}).get("_upstream_trace")
            if isinstance(trace, dict):
                trace["reasoning_effort_applied"] = effort_level
    if mode == "work":
        initial_message["__hit_shared_artifact_and_workroom"] = True
        initial_message["__hit_my_space"] = True
    body = {
        "mode": mode,
        "environment_id": "default",
        "initial_message": initial_message,
        "env": "remote",
        "auto_create_project": False,
        "origin": "web",
    }
    work_overlay = _work_billing_overlay(options)
    if work_overlay:
        entitlement_id = str(work_overlay.get("entitlement_id") or "")
        if entitlement_id:
            initial_message["entitlement_id"] = entitlement_id
            auto_selection = initial_message.get("model_auto_selection")
            if isinstance(auto_selection, Mapping):
                auto_selection["entitlement_id"] = entitlement_id
            else:
                initial_message["model_auto_selection"] = {
                    "strategy": effective_strategy,
                    "fallback_to_advance_model": None,
                    "entitlement_id": entitlement_id,
                }
        for key, value in (
            ("available_endpoint", work_overlay.get("available_endpoint")),
            ("billing_source", work_overlay.get("billing_source")),
            ("product_code", work_overlay.get("product_code")),
        ):
            if value not in (None, ""):
                initial_message[key] = value
        if work_overlay.get("origin") not in (None, ""):
            body["origin"] = str(work_overlay.get("origin"))
        if work_overlay.get("product_code") not in (None, ""):
            body["product_code"] = str(work_overlay.get("product_code"))
        if work_overlay.get("billing_source") not in (None, ""):
            body["billing_source"] = str(work_overlay.get("billing_source"))
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
    """Stream events while bounding a session silent before model activity.

    The remote endpoint may send queue/heartbeat frames before it either
    starts the selected executor or emits an error.  Treat an early error as a
    retryable Agent-start failure so the caller can perform its explicit Work
    fallback; do not let a heartbeat consume the whole first-event timeout.
    """

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
    loop = asyncio.get_running_loop()
    deadline = loop.time() + first_event_timeout if first_event_timeout > 0 else None
    model_events = {
        "model_config",
        "plan_item",
        "message",
        "assistant_message",
        "response",
        "text",
        "output",
        "token_usage",
        "done",
    }

    def event_name(value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
        return {
            "modelconfig": "model_config",
            "planitem": "plan_item",
            "responsedone": "done",
            "response_done": "done",
            "streamdone": "done",
            "stream_done": "done",
            "tokenusage": "token_usage",
        }.get(normalized, normalized)

    try:
        observed_model_event = False
        while not observed_model_event:
            try:
                if deadline is None:
                    current = await source.__anext__()
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    current = await asyncio.wait_for(
                        source.__anext__(), timeout=remaining
                    )
            except StopAsyncIteration as exc:
                raise RemoteFirstEventTimeout(
                    "Trae remote session ended before its first model event"
                ) from exc
            except (asyncio.TimeoutError, httpx.ReadTimeout) as exc:
                raise RemoteFirstEventTimeout(
                    "Trae remote session emitted no model event before the first-event timeout"
                ) from exc
            if not isinstance(current, tuple) or len(current) != 2:
                # This should only happen with a custom test/adapter source;
                # preserve it as non-model activity instead of crashing the
                # stream wrapper.
                yield current
                continue
            name, data = current
            normalized = event_name(name)
            if normalized == "error":
                payload = data if isinstance(data, Mapping) else {}
                code = str(payload.get("code") or "").strip()
                message = str(
                    payload.get("message")
                    or payload.get("error")
                    or payload.get("detail")
                    or "remote session setup failed"
                ).strip()
                suffix = f" {code}" if code else ""
                raise RemoteFirstEventError(
                    f"Trae remote upstream error{suffix}: {message}"
                )
            yield current
            observed_model_event = normalized in model_events

        saw_terminal = False
        try:
            async for event in source:
                if isinstance(event, tuple) and len(event) == 2:
                    saw_terminal = saw_terminal or event_name(event[0]) == "done"
                yield event
        except httpx.ReadTimeout as exc:
            raise RemoteStreamReadTimeout(
                "Trae remote event stream timed out after upstream activity began"
            ) from exc
        if not saw_terminal:
            # Trae's remote endpoint can close the events connection without
            # an error after model activity (notably when the flattened query
            # exceeds its silent size limit). Treat that as an incomplete turn
            # instead of letting a partial answer masquerade as a normal EOF.
            raise RemoteStreamIncomplete(
                "Trae remote event stream ended without a done event"
            )
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
