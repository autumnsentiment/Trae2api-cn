"""Direct Trae raw-chat transport.

This module speaks the HTTP protocol used by Trae's terminal client without
running the Trae CLI locally.  It only transports model and tool-call data;
filesystem, shell, edit, and other tools remain owned by the API caller.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import json
import logging
import os
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx

from . import auth
from .cli_client import (
    renderer_block_text,
    strip_tool_call_blocks,
    tool_result_is_failed,
)
from . import images
from .reasoning_effort import apply_reasoning_effort, requested_level
logger = logging.getLogger(__name__)


# ``RAW_CHAT_ENDPOINT`` is the legacy TraeWork raw transport.  It is kept
# separate from TraeCode's CustomModel/body contract below: TraeCode's
# ``llm_raw_chat`` request does not use the old ``llm_utils_chat`` auxiliary
# path or the legacy ``Extra`` model-lock envelope by default.
RAW_CHAT_ENDPOINT = "/api/ide/v2/llm_raw_chat"
RAW_APP_ID = "7b3f9dc2-8a4e-5c6d-2f1b-9e4a3c5b7df0"
RAW_IDE_VERSION = "3.3.67"
RAW_IDE_VERSION_CODE = "20260206"
RAW_IDE_FUNCTION = "chat"

# TraeCode CN (the current desktop IDE) shares the llm_raw_chat route with
# TraeWork, but it is a different protocol contract.  In particular, the
# model/function binding is carried in the JSON body and the current IDE does
# not rely on TraeWork's ``Extra`` header.  Keep these constants separate from
# the legacy raw transport above so a Work request cannot accidentally inherit
# TraeCode's app id or function selector.
TRAECODE_RAW_CHAT_ENDPOINT = "/api/ide/v2/llm_raw_chat"
TRAECODE_RAW_CHAT_FALLBACK_ENDPOINT = "/api/ide/v1/llm_raw_chat"
TRAECODE_APP_ID = "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8"
TRAECODE_DEFAULT_GATEWAY = (
    os.environ.get("TRAE_CODE_BASE_URL")
    or "https://trae-api-cn.mchost.guru"
).rstrip("/")
TRAECODE_IDE_VERSION = os.environ.get("TRAE_CODE_IDE_VERSION", "3.4.4")
TRAECODE_IDE_VERSION_CODE = os.environ.get(
    "TRAE_CODE_IDE_VERSION_CODE", "20260212"
)
TRAECODE_DEFAULT_FUNCTION = "chat_v3"
TRAECODE_SOLO_FUNCTIONS = {
    "solo_agent",
    "solo_coder",
    "solo_builder",
    "solo_agent_lite",
    "solo_work_lite",
    "solo_agent_remote",
    "solo_work_remote",
}
TRAECODE_IDE_FUNCTIONS = {"chat_v3", "builder_v3", "builder_with_mcp_v3"}
_TRAECODE_PROTECTED_HEADERS = {
    "authorization",
    "x-cloudide-token",
    "x-ide-token",
    "x-uid",
    "x-app-id",
    "x-app-function",
    "x-ide-function",
    "extra",
}


@dataclass
class _RawSessionGate:
    slots: threading.BoundedSemaphore
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


async def _acquire_raw_session_gate(
    session_id: str,
    account_id: str = "",
    token: str = "",
    reservation_id: str = "",
) -> Callable[[], None]:
    """Acquire one model slot for a raw request.

    Persisted accounts share the entitlement-backed pool used by Web, Remote,
    and legacy IDE transports.  The per-session semaphore remains only for
    unregistered/single-token integrations, where no account entitlement is
    available.
    """

    from . import trae_client

    canonical_account = auth.resolve_model_account_id(account_id)
    if not canonical_account:
        # A few direct callers provide only the bound JWT.  Resolve its
        # immutable billing identity before falling back to the legacy gate.
        canonical_account = auth.resolve_model_account_id(
            _jwt_identity(str(token or ""))
        )
    if canonical_account:
        try:
            timeout = float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60"))
        except (TypeError, ValueError):
            timeout = 60.0
        slot_lease_token = await trae_client.acquire_web_slot(
            canonical_account,
            timeout=max(0.0, timeout),
            reservation_id=reservation_id,
        )
        released = False
        release_guard = threading.Lock()

        def release_account_slot() -> None:
            nonlocal released
            with release_guard:
                if released:
                    return
                released = True
            trae_client.release_web_slot_owned(
                canonical_account, slot_lease_token
            )

        return release_account_slot

    with _RAW_SESSION_GATES_LOCK:
        gate = _RAW_SESSION_GATES.get(session_id)
        if gate is None:
            gate = _RawSessionGate(
                threading.BoundedSemaphore(trae_client.model_request_parallel_limit())
            )
            _RAW_SESSION_GATES[session_id] = gate
        gate.users += 1

    acquired = False
    try:
        while not gate.slots.acquire(blocking=False):
            await asyncio.sleep(_RAW_SESSION_GATE_POLL_SECONDS)
        acquired = True
    except BaseException:
        if acquired:
            gate.slots.release()
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
            gate.slots.release()
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
    descriptor: Optional[Mapping[str, Any]] = field(default=None, compare=False, repr=False)


class TraeCodeRateLimitError(RuntimeError):
    """An HTTP limit that requires account rotation instead of gateway fallback."""

    def __init__(self, status_code: int, code: Any, detail: str):
        self.status_code = status_code
        self.code = code
        super().__init__(
            f"TraeCode raw chat rate limit {code} (HTTP {status_code})"
            + (f": {detail}" if detail else "")
        )


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


def _traecode_option(options: Optional[Mapping[str, Any]], *names: str, default: Any = None) -> Any:
    """Read a TraeCode-specific option without sharing legacy raw defaults."""

    options = options or {}
    for name in names:
        if name in options and options[name] is not None:
            return options[name]
    return default


def _jwt_identity(token: str) -> str:
    """Return an immutable account id from a valid JWT payload when present."""

    parts = str(token or "").strip().split(".")
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return ""

    def decode(segment: str) -> Any:
        padding = "=" * (-len(segment) % 4)
        return json.loads(base64.urlsafe_b64decode(segment + padding))

    try:
        header = decode(parts[0])
        payload = decode(parts[1])
    except (
        binascii.Error,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return ""
    if not isinstance(header, Mapping) or not isinstance(payload, Mapping):
        return ""

    candidates: list[Any] = []
    for container_name in ("data", "user", "account"):
        container = payload.get(container_name)
        if not isinstance(container, Mapping):
            continue
        candidates.extend(
            container.get(name)
            for name in (
                "id",
                "sub",
                "user_id",
                "userId",
                "uid",
                "biz_user_id",
                "bizUserId",
            )
        )
    # Match the relay's account selector: Trae's nested ``data.id`` is the
    # billing identity, while top-level ``sub`` can describe a login subject.
    candidates.extend(
        payload.get(name)
        for name in (
            "user_id",
            "userId",
            "sub",
            "uid",
            "biz_user_id",
            "bizUserId",
        )
    )
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (str, int)):
            continue
        identity = str(candidate).strip()
        if identity:
            return identity
    return ""


def _traecode_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _traecode_context_value(
    options: Optional[Mapping[str, Any]], *names: str, default: Any = None
) -> Any:
    """Read a known client value from the account-bound Trae context."""

    options = options or {}
    containers: list[Mapping[str, Any]] = [options]
    for key in (
        "client_context",
        "clientContext",
        "common_params",
        "commonParams",
        "native_common_params",
        "streamlined_common_params",
        "streamlinedCommonParams",
        "native_streamlined_common_params",
        "provider_specific",
        "providerSpecificData",
    ):
        container = _traecode_mapping(options.get(key))
        if container is not None:
            containers.append(container)
    # These are the only nested client envelopes observed in the 3.4.4
    # request chain. Keep the search bounded so arbitrary request data cannot
    # become a header by coincidence.
    for container in list(containers):
        for key in (
            "userParams",
            "user_params",
            "deviceInfo",
            "device_info",
            "clientInfo",
            "client_info",
            "commonParams",
            "common_params",
            "streamlinedCommonParams",
            "streamlined_common_params",
            "headers",
            "header",
        ):
            nested = _traecode_mapping(container.get(key))
            if nested is not None:
                containers.append(nested)
    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    return default


def _encoded_header_value(value: Any) -> str:
    """Match JavaScript's encodeURIComponent used by risk-control headers."""

    # Direct 3.4.4 clients already send these values encoded. Normalize once
    # before replay so a proxy hop never turns ``%20`` into ``%2520``.
    return quote(unquote(str(value)), safe="-_.!~*'()")


def _authorized_services_header(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple, set, frozenset)):
        services = [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (set, frozenset)):
            services.sort()
        return ",".join(services)
    if value is None:
        return ""
    return str(value).strip()


_TRAECODE_RAW_V2_OPTION_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("message", ("message",)),
    ("is_custom_model", ("is_custom_model", "isCustomModel")),
    ("provider", ("provider",)),
    ("Request", ("Request", "request")),
    (
        "encrypted_model_params",
        ("encrypted_model_params", "encryptedModelParams"),
    ),
    ("user_input", ("user_input", "userInput")),
    ("request_type", ("request_type", "requestType")),
    ("metadata", ("metadata",)),
    ("queue_id", ("queue_id", "queueId")),
    ("request_seq", ("request_seq", "requestSeq")),
    ("mode_type", ("mode_type", "modeType")),
    ("chat_mode", ("chat_mode", "chatMode")),
    ("access_type", ("access_type", "accessType")),
    ("image_config", ("image_config", "imageConfig")),
    (
        "estimate_prompt_tokens",
        ("estimate_prompt_tokens", "estimatePromptTokens"),
    ),
    ("slash_command", ("slash_command", "slashCommand")),
    ("message_origin", ("message_origin", "messageOrigin")),
    ("reasoning_effort", ("reasoning_effort", "reasoningEffort")),
    ("retry_times", ("retry_times", "retryTimes")),
    ("queue_retry", ("queue_retry", "queueRetry")),
    ("video_config", ("video_config", "videoConfig")),
    (
        "connect_check_param",
        ("connect_check_param", "connectCheckParam"),
    ),
    (
        "user_selected_model_name",
        ("user_selected_model_name", "userSelectedModelName"),
    ),
    (
        "reasoning_effort_level",
        ("reasoning_effort_level", "reasoningEffortLevel"),
    ),
    (
        "estimated_token_usage",
        ("estimated_token_usage", "estimatedTokenUsage"),
    ),
    (
        "custom_model_uniq_id",
        ("custom_model_uniq_id", "customModelUniqId"),
    ),
    (
        "application_config",
        ("application_config", "applicationConfig", "app_config", "appConfig"),
    ),
    ("provider_context", ("provider_context", "providerContext")),
    ("biz_context", ("biz_context", "bizContext")),
    ("extra_info", ("extra_info", "extraInfo")),
    ("use_fast_request", ("use_fast_request", "useFastRequest")),
)


def _normalize_traecode_function(value: Any, *, solo: bool = False) -> str:
    """Return a validated ``raw_chat_function`` for the TraeCode gateway."""

    requested = str(value or "").strip().lower()
    if not requested:
        return "solo_agent" if solo else TRAECODE_DEFAULT_FUNCTION
    if requested in TRAECODE_SOLO_FUNCTIONS or requested in TRAECODE_IDE_FUNCTIONS:
        return requested
    # A UI can send a mode name instead of the native enum.  Keep this
    # mapping local to the protocol adapter so the public mode names do not
    # leak into the wire payload.
    if requested in {"ide", "chat", "chat_v3"}:
        return "chat_v3"
    if requested in {"solo", "agent", "solo-agent", "solo_agent"}:
        return "solo_agent"
    if requested in {"builder", "builder-v3"}:
        return "builder_v3"
    if requested in {"coder", "solo-coder"}:
        return "solo_coder"
    raise ValueError(f"Unsupported TraeCode raw_chat_function: {value}")


def traecode_raw_chat_function(
    options: Optional[Mapping[str, Any]] = None,
    *,
    solo: bool = False,
) -> str:
    """Resolve the TraeCode IDE/SOLO protocol function.

    The dispatcher can pass either the exact enum in
    ``_traecode_raw_chat_function`` or a public mode via ``_traecode_mode``.
    Unlike the old ``llm_utils_chat`` path, this selector is written into the
    model request body and therefore determines the upstream product chain.
    """

    options = options or {}
    explicit = _traecode_option(
        options,
        "_traecode_raw_chat_function",
        "traecode_raw_chat_function",
        "raw_chat_function",
        "rawChatFunction",
        default=None,
    )
    if explicit is None:
        explicit = _traecode_option(
            options,
            "_traecode_mode",
            "traecode_mode",
            "mode",
            default=None,
        )
    return _normalize_traecode_function(explicit, solo=solo)


def _traecode_model_from_raw(raw_model: RawModel) -> RawModel:
    """Use TraeCode's provider model id rather than TraeWork's ``__v2`` id."""

    # TraeWork's raw catalogue uses ids such as ``glm-5__v2``.  TraeCode's
    # CustomModel model_name is the provider/config id (``glm-5.1``), while
    # callers may still explicitly provide a provider id when required.
    provider_name = raw_model.raw_model_name
    if provider_name.endswith("__v2"):
        provider_name = raw_model.config_name
    return RawModel(
        config_name=raw_model.config_name,
        raw_model_name=provider_name,
        display_name=raw_model.display_name,
        config_source=raw_model.config_source,
        provider=raw_model.provider,
        descriptor=copy.deepcopy(dict(raw_model.descriptor))
        if isinstance(raw_model.descriptor, Mapping)
        else raw_model.descriptor,
    )


def resolve_traecode_model(
    model: str, options: Optional[Mapping[str, Any]] = None
) -> RawModel:
    """Resolve a public model id to TraeCode ``CustomModel`` fields.

    This synchronous resolver intentionally uses only the local alias table.
    ``resolve_traecode_model_for_request`` below may enrich unknown models from
    the live account model list before a request is sent.
    """

    options = options or {}
    config_name = _traecode_option(
        options,
        "traecode_config_name",
        "traeCodeConfigName",
        "trae_raw_config_name",
        "traeRawConfigName",
        "config_name",
        "configName",
        default=None,
    )
    provider_name = _traecode_option(
        options,
        "traecode_model_name",
        "traeCodeModelName",
        "trae_raw_model_name",
        "traeRawModelName",
        "raw_model_name",
        "rawModelName",
        default=None,
    )
    display_name = _traecode_option(
        options,
        "display_name",
        "displayName",
        "display_model_name",
        "displayModelName",
        "model_name",
        "modelName",
        default=None,
    )
    config_source = _traecode_option(
        options,
        "traecode_config_source",
        "traeCodeConfigSource",
        "config_source",
        "configSource",
        default=None,
    )
    try:
        config_source = (
            int(config_source)
            if config_source is not None
            else None
        )
    except (TypeError, ValueError):
        config_source = None
    resolved = resolve_raw_model(model, options)
    if config_name:
        resolved = RawModel(
            str(config_name),
            str(provider_name or resolved.raw_model_name),
            str(display_name or resolved.display_name),
            config_source or resolved.config_source,
            str(_traecode_option(options, "provider", default=None) or resolved.provider),
            descriptor=copy.deepcopy(dict(resolved.descriptor))
            if isinstance(resolved.descriptor, Mapping)
            else resolved.descriptor,
        )
    else:
        resolved = _traecode_model_from_raw(resolved)
        if provider_name or display_name or config_source:
            resolved = RawModel(
                resolved.config_name,
                str(provider_name or resolved.raw_model_name),
                str(display_name or resolved.display_name),
                config_source or resolved.config_source,
                str(_traecode_option(options, "provider", default=None) or resolved.provider),
                descriptor=copy.deepcopy(dict(resolved.descriptor))
                if isinstance(resolved.descriptor, Mapping)
                else resolved.descriptor,
            )
    return resolved


async def resolve_traecode_model_for_request(
    model: str, options: Optional[Mapping[str, Any]] = None
) -> RawModel:
    """Resolve TraeCode model fields, consulting the live config when needed."""

    options = options or {}
    explicit = any(
        _traecode_option(options, key, default=None) is not None
        for key in (
            "traecode_config_name",
            "traeCodeConfigName",
            "traecode_model_name",
            "traeCodeModelName",
            "traecode_config_source",
            "traeCodeConfigSource",
            "trae_raw_config_name",
            "trae_raw_model_name",
            "config_name",
            "model_name",
        )
    )
    resolved = resolve_traecode_model(model, options)
    if explicit:
        return resolved
    # TraeCode IDE/Solo uses a different provider-id namespace from the
    # legacy TraeWork raw transport.  When the dispatcher selected a native
    # function, consult the account-bound ``get_detail_param`` catalog first;
    # it returns exact ids such as ``glm-5__dev``.  Falling through to the
    # legacy resolver here would send ``glm-5__v2`` and the gateway reports a
    # misleading 4011/4023 error.
    native_function = _traecode_option(
        options,
        "_traecode_raw_chat_function",
        "traecode_raw_chat_function",
        "raw_chat_function",
        default=None,
    )
    if native_function:
        try:
            from . import trae_client

            token = str(options.get("_auth_token") or auth.get_token() or "").strip()
            bound_user_id = str(
                options.get("_auth_user_id")
                or options.get("_billing_id")
                or options.get("_account_id")
                or ""
            ).strip()
            provider_specific = options.get("provider_specific")
            if provider_specific is None and "providerSpecificData" in options:
                provider_specific = options.get("providerSpecificData")
            max_mode = bool(
                _traecode_option(
                    options,
                    "trae_max_mode",
                    "max_mode",
                    "max",
                    default=False,
                )
            )
            live = await trae_client.resolve_traecode_model_config(
                model,
                token_override=token,
                user_id_override=bound_user_id,
                provider_specific=(
                    dict(provider_specific)
                    if isinstance(provider_specific, Mapping)
                    else None
                ),
                max_mode=max_mode,
                raw_chat_function=str(native_function),
            )
            if isinstance(live, Mapping):
                return RawModel(
                    config_name=str(
                        live.get("config_name")
                        or live.get("name")
                        or resolved.config_name
                    ),
                    raw_model_name=str(
                        live.get("raw_model_name")
                        or live.get("model_name")
                        or resolved.raw_model_name
                    ),
                    display_name=str(
                        live.get("display_name")
                        or live.get("display_model_name")
                        or resolved.display_name
                    ),
                    config_source=int(
                        live.get("config_source") or resolved.config_source or 1
                    ),
                    provider=str(live.get("provider") or resolved.provider or ""),
                    descriptor=copy.deepcopy(dict(live)),
                )
        except Exception as exc:
            logger.warning(
                "traecode live model config lookup failed for %s: %s",
                model,
                exc,
            )
    # The existing async resolver knows how to query Trae's model catalogue.
    # Reuse it for unknown/new models, then normalize the legacy ``__v2``
    # suffix away because TraeCode's CustomModel uses provider model ids.
    try:
        upstream = await resolve_raw_model_for_request(model, options)
    except Exception as exc:
        logger.warning("traecode model config lookup failed for %s: %s", model, exc)
        return resolved
    upstream = _traecode_model_from_raw(upstream)
    resolved = _traecode_model_from_raw(resolved)
    # Live account configs can reuse a public/config name while selecting a
    # different provider model id or config source. Merge their non-empty
    # fields individually instead of discarding the whole live mapping unless
    # config_name happens to differ.
    return RawModel(
        config_name=upstream.config_name or resolved.config_name,
        raw_model_name=upstream.raw_model_name or resolved.raw_model_name,
        display_name=upstream.display_name or resolved.display_name,
        config_source=upstream.config_source or resolved.config_source,
        provider=upstream.provider or resolved.provider,
        descriptor=copy.deepcopy(dict(upstream.descriptor))
        if isinstance(upstream.descriptor, Mapping)
        else (
            copy.deepcopy(dict(resolved.descriptor))
            if isinstance(resolved.descriptor, Mapping)
            else resolved.descriptor
        ),
    )


def traecode_session_id(
    model: RawModel,
    options: Optional[Mapping[str, Any]] = None,
    raw_chat_function: str = TRAECODE_DEFAULT_FUNCTION,
) -> str:
    """Return a stable account/model/function-bound TraeCode conversation id."""

    options = options or {}
    explicit = _traecode_option(
        options,
        "_traecode_session_id",
        "traecode_session_id",
        "session_id",
        "sessionId",
        default=None,
    )
    account_identity = str(
        options.get("_billing_id")
        or options.get("_auth_user_id")
        or options.get("_account_id")
        or ""
    ).strip()
    if not account_identity:
        token = str(options.get("_auth_token") or "")
        account_identity = (
            "token:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
            if token
            else "default"
        )
    cache_key = _traecode_option(
        options,
        "_prompt_cache_key_affinity",
        default=None,
    )
    if isinstance(cache_key, str) and cache_key.strip():
        # The relay may bind a random local lease id before dispatch. When the
        # caller supplied no explicit session, use the cache key for a stable
        # upstream conversation instead of allowing that local id to fragment
        # provider-side prompt cache affinity.
        material = "\x1f".join(
            (
                "prompt-cache-v1",
                account_identity,
                cache_key.strip()[:1024],
                model.config_name,
                model.raw_model_name,
                raw_chat_function,
            )
        )
        return str(uuid.UUID(hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]))
    if explicit:
        return str(explicit)
    material = "\x1f".join(
        (account_identity, model.config_name, model.raw_model_name, raw_chat_function)
    )
    return str(uuid.UUID(hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]))


def _traecode_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI/Anthropic content into TraeCode text/media blocks."""

    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        if isinstance(content, Mapping):
            return [copy.deepcopy(dict(content))]
        return [{"type": "text", "text": str(content)}]
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, Mapping):
            blocks.append({"type": "text", "text": str(part)})
            continue
        block = copy.deepcopy(dict(part))
        block_type = str(block.get("type") or "").strip().lower()
        if block_type in {"input_text", "text"}:
            text = block.get("text")
            if not isinstance(text, str):
                text = block.get("content")
            block["type"] = "text"
            block["text"] = text if isinstance(text, str) else renderer_block_text(block)
            block.pop("value", None)
        elif block_type == "image_url":
            # Keep OpenAI image_url blocks intact; the native TraeCode adapter
            # accepts these alongside text and the existing image helper can
            # still describe unsupported media at the caller boundary.
            block["type"] = "image_url"
        blocks.append(block)
    return blocks


def _traecode_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Preserve native OpenAI tool-call correlation for TraeCode history."""

    calls: list[Any] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        calls.extend(raw_calls)
    legacy = message.get("function_call")
    if isinstance(legacy, Mapping):
        calls.append(legacy)
    calls.extend(_renderer_tool_use_blocks(message.get("content")))
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or call.get("function_call")
        function = function if isinstance(function, Mapping) else call
        name = str(function.get("name") or call.get("name") or "").strip()
        if not name:
            continue
        call_id = str(
            call.get("id")
            or call.get("tool_call_id")
            or call.get("toolCallId")
            or f"call_{len(result)}"
        )
        if call_id in seen:
            continue
        seen.add(call_id)
        arguments = function.get("arguments")
        if arguments is None:
            arguments = function.get("input", function.get("parameters", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":"), default=str
            )
        result.append(
            {
                "id": call_id,
                "type": str(call.get("type") or "function"),
                "function": {"name": name, "arguments": arguments},
            }
        )
    return result


def _traecode_tool_definitions(tools: Any) -> list[dict[str, Any]]:
    """Keep OpenAI function schemas object-shaped for TraeCode."""

    if not isinstance(tools, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        function = function if isinstance(function, Mapping) else tool
        name = str(function.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parameters = function.get("parameters")
        if parameters is None:
            parameters = function.get("input_schema") or function.get("inputSchema") or {}
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except (TypeError, ValueError):
                parameters = {"type": "object"}
        fn: dict[str, Any] = {
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": copy.deepcopy(parameters),
        }
        if "strict" in function:
            fn["strict"] = bool(function.get("strict"))
        result.append({"type": "function", "function": fn})
    return result


def build_traecode_messages(
    messages: list[dict[str, Any]], options: Optional[Mapping[str, Any]] = None
) -> list[dict[str, Any]]:
    """Convert caller messages to the TraeCode ``ChatMessageData`` shape."""

    if not messages:
        raise ValueError("messages cannot be empty")
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user").strip().lower()
        if role == "developer":
            role = "system"
        elif role == "function":
            role = "tool"
        elif role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        item: dict[str, Any] = {"role": role}
        content = message.get("content")
        blocks = _traecode_content_blocks(content)
        if blocks:
            item["content"] = blocks
        if role == "assistant":
            tool_calls = _traecode_tool_calls(message)
            if tool_calls:
                item["tool_calls"] = tool_calls
            elif isinstance(message.get("function_call"), Mapping):
                legacy = _traecode_tool_calls({"tool_calls": [message["function_call"]]})
                if legacy:
                    item["function_call"] = legacy[0]["function"]
        elif role == "tool":
            if message.get("tool_call_id") is not None:
                item["tool_call_id"] = str(message["tool_call_id"])
            if message.get("name") is not None:
                item["name"] = str(message["name"])
        # Empty assistant turns are meaningful when they carry tool_calls;
        # empty system/user turns are just malformed OpenAI history.
        if not blocks and role in {"system", "user"} and not item.get("tool_calls"):
            continue
        result.append(item)
    if not result:
        raise ValueError("messages cannot be empty")
    return result


def build_traecode_chat_body(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool = True,
    options: Optional[Mapping[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build TraeCode CN's IDE/SOLO ``llm_raw_chat`` request body.

    The body intentionally lives beside, rather than inside,
    :func:`build_raw_chat_body`: TraeWork's raw protocol has six fixed fields,
    while TraeCode carries ``raw_chat_function`` and ``config_source`` in the
    ``CustomModel`` envelope and can use native OpenAI tool fields.
    """

    options = options or {}
    solo = bool(
        _traecode_option(options, "_traecode_solo", "traecode_solo", "solo", default=False)
    )
    function = traecode_raw_chat_function(options, solo=solo)
    resolved = resolve_traecode_model(model, options)
    sid = str(session_id or traecode_session_id(resolved, options, function))
    native_tools = bool(
        _traecode_option(
            options,
            "_traecode_native_tools",
            "traecode_native_tools",
            default=False,
        )
    )
    tools = options.get("tools") if "tools" in options else options.get("_inherited_tools")
    body_messages = (
        build_traecode_messages(messages, options)
        if native_tools
        else build_raw_messages(messages, options)
    )
    body: dict[str, Any] = {
        "config_name": resolved.config_name,
        "model_name": resolved.raw_model_name,
        "conversation_id": sid,
        "session_id": sid,
        # TraeCode's client always opens the SSE path; non-stream API callers
        # are aggregated by the relay after the upstream stream closes.
        "stream": bool(_traecode_option(options, "_traecode_wire_stream", default=True)),
        "raw_chat_function": function,
        "config_source": int(resolved.config_source or 1),
        "messages": body_messages,
    }
    if bool(_traecode_option(options, "_enterprise_mode", default=False)):
        body["config_source"] = 2
    # Trae 3.4.4 expanded the shared Work/Code raw-v2 DTO. Only copy fields
    # explicitly supplied by the caller: defaults for queueing, effort, and
    # client mode belong to the selected upstream model configuration.
    for wire_name, aliases in _TRAECODE_RAW_V2_OPTION_FIELDS:
        value = _traecode_option(options, *aliases, default=None)
        if value is not None:
            body[wire_name] = copy.deepcopy(value)
    descriptor = None
    for descriptor_key in (
        "_traecode_model_descriptor",
        "traecode_model_descriptor",
        "model_descriptor",
        "custom_model",
        "customModel",
    ):
        candidate = options.get(descriptor_key)
        if isinstance(candidate, Mapping):
            descriptor = candidate
            break
    if isinstance(descriptor, Mapping):
        effort_model, effort_level = apply_reasoning_effort(descriptor, options)
        body.pop("reasoning_effort", None)
        body.pop("reasoning_effort_level", None)
        if effort_level:
            effort_key = (
                "reasoning_effort_level"
                if "reasoning_effort_level" in effort_model
                else "reasoning_effort"
            )
            body[effort_key] = effort_level
        trace = options.get("_upstream_trace")
        if isinstance(trace, dict):
            if effort_level:
                trace["reasoning_effort_applied"] = effort_level
            else:
                trace.pop("reasoning_effort_applied", None)
    if native_tools and isinstance(tools, list) and tools:
        converted_tools = _traecode_tool_definitions(tools)
        if converted_tools:
            body["tools"] = converted_tools
            choice = options.get("tool_choice")
            if choice is not None:
                body["tool_choice"] = copy.deepcopy(choice)
            if options.get("parallel_tool_calls") is not None:
                body["parallel_tool_calls"] = bool(options["parallel_tool_calls"])
    return body


def build_traecode_headers(
    base_url: str,
    token: str,
    raw_model: RawModel,
    request_id: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    session_id: str = "",
    raw_chat_function: str = TRAECODE_DEFAULT_FUNCTION,
) -> dict[str, str]:
    """Build the TraeCode CN header set without leaking TraeWork ``Extra``."""

    options = options or {}
    bound_user_id = _traecode_option(
        options,
        "_auth_user_id",
        "auth_user_id",
        "_billing_id",
        "billing_id",
        "_account_id",
        "account_id",
        default=None,
    )
    # Session options can outlive a credential refresh or account rotation.
    # When the token is a real JWT its immutable identity is authoritative;
    # opaque legacy tokens retain the existing explicit-binding fallback.
    user_id = str(_jwt_identity(token) or bound_user_id or auth.get_user_id() or "")
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

    version = str(
        _traecode_option(
            options,
            "traecode_ide_version",
            "traeCodeIdeVersion",
            "ide_version",
            "ideVersion",
            default=None,
        )
        or os.environ.get("TRAE_CODE_IDE_VERSION")
        or TRAECODE_IDE_VERSION
    )
    version_code = str(
        _traecode_option(
            options,
            "traecode_ide_version_code",
            "traeCodeIdeVersionCode",
            "ide_version_code",
            "ideVersionCode",
            default=None,
        )
        or os.environ.get("TRAE_CODE_IDE_VERSION_CODE")
        or TRAECODE_IDE_VERSION_CODE
    )
    app_id = str(
        _traecode_option(
            options,
            "traecode_app_id",
            "traeCodeAppId",
            "app_id",
            "appId",
            default=None,
        )
        or os.environ.get("TRAE_CODE_APP_ID")
        or TRAECODE_APP_ID
    )
    trace_id = secrets.token_hex(16)
    put("Authorization", f"Cloud-IDE-JWT {token}")
    put("X-Cloudide-Token", token)
    put("X-Ide-Token", token)
    put("Content-Type", "application/json")
    put("Accept", "text/event-stream")
    put("Connection", "keep-alive")
    put("X-App-Id", app_id)
    put(
        "X-App-Version",
        _traecode_context_value(
            options,
            "app_version",
            "appVersion",
            default="default",
        ),
    )
    put("X-App-Version-Code", version_code)
    put("X-Ide-Version", version)
    put("X-Ide-Version-Code", version_code)
    put(
        "X-Ide-Version-Type",
        _traecode_context_value(
            options,
            "ide_version_type",
            "ideVersionType",
            "version_type",
            "versionType",
            default="stable",
        ),
    )
    put(
        "X-Plugin-Channel",
        _traecode_context_value(
            options,
            "plugin_channel",
            "pluginChannel",
            "channel",
            default="desktop",
        ),
    )
    put("X-App-Function", raw_chat_function)
    put("X-Ide-Function", raw_chat_function)
    put("X-Request-Id", request_id)
    put("X-Custom-Trace-Id", trace_id)
    put("X-Flow-Traceparent", f"04-{trace_id}-{secrets.token_hex(8)}-01")
    put(
        "request-traffic-type",
        _traecode_context_value(
            options,
            "request_traffic_type",
            "requestTrafficType",
            default="prod",
        ),
    )

    risk_headers = (
        (
            "X-Device-Cpu",
            ("cpu", "device_cpu", "deviceCpu"),
        ),
        (
            "X-Device-Id",
            (
                "guaranteedDeviceId",
                "guaranteed_device_id",
                "device_id",
                "deviceId",
            ),
        ),
        ("X-Machine-Id", ("machine_id", "machineId")),
        ("X-Device-Brand", ("device_model", "deviceModel", "device_brand")),
        ("X-Device-Type", ("os_name", "osName", "device_type", "deviceType")),
        ("X-Os-Version", ("os_version", "osVersion")),
    )
    for header_name, aliases in risk_headers:
        value = _traecode_context_value(options, *aliases, default=None)
        if value not in (None, ""):
            put(header_name, _encoded_header_value(value))

    web_id = _traecode_context_value(options, "web_id", "webId", default=None)
    if web_id not in (None, ""):
        put("X-Web-Id", web_id)

    ppe_env = _traecode_context_value(
        options,
        "ppe_env_name",
        "ppeEnvName",
        "tt_env",
        "ttEnv",
        "x_tt_env",
        default=None,
    )
    if ppe_env not in (None, ""):
        put("X-Tt-Env", ppe_env)
        put(
            "X-Use-Ppe",
            _traecode_context_value(
                options,
                "use_ppe",
                "usePpe",
                "x_use_ppe",
                default="1",
            ),
        )

    authorized_services = _authorized_services_header(
        _traecode_context_value(
            options,
            "authorized_services",
            "authorizedServices",
            "trae_authorized_services",
            default=None,
        )
    )
    if authorized_services:
        put("X-Trae-Authorized-Services", authorized_services)
    if user_id:
        put("X-Uid", user_id)
    else:
        for key in list(headers):
            if key.lower() == "x-uid":
                del headers[key]
    # TraeCode's current IDE chain models selection in CustomModel/body.  An
    # explicit compatibility switch can add the legacy Extra envelope for
    # older gateways without making it part of the default protocol.
    if session_id and _traecode_option(options, "_traecode_include_extra", default=False):
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
        put("Extra", json.dumps(extra, ensure_ascii=True, separators=(",", ":")))
    else:
        for key in list(headers):
            if key.lower() == "extra":
                del headers[key]
    custom_headers = _traecode_option(options, "headers", "traecode_headers", default={})
    if isinstance(custom_headers, Mapping):
        for key, value in custom_headers.items():
            if str(key).strip().lower() in _TRAECODE_PROTECTED_HEADERS:
                logger.warning(
                    "ignored TraeCode custom override for protected header %s",
                    key,
                )
                continue
            put(str(key), value)
    return headers


def resolve_traecode_base_url(options: Optional[Mapping[str, Any]] = None) -> str:
    """Resolve the TraeCode model gateway host, stripping stale API paths."""

    options = options or {}
    configured = _traecode_option(
        options,
        "traecode_base_url",
        "traeCodeBaseUrl",
        "model_base_url",
        "base_url",
        "baseURL",
        default=None,
    )
    base = (
        configured
        or os.environ.get("TRAE_CODE_BASE_URL")
        or os.environ.get("TRAE_MODEL_API_HOST")
    )
    if not base:
        # ``TRAE_API_HOST`` is normally the OAuth/entitlement host
        # (api.trae.cn).  Only reuse it when it is already a model gateway;
        # otherwise it would turn a valid raw request into an auth-host 404.
        candidate = str(os.environ.get("TRAE_API_HOST") or "").strip()
        if "mchost.guru" in candidate or "trae-api-" in candidate:
            base = candidate
    if not base:
        base = TRAECODE_DEFAULT_GATEWAY
    base = str(base).rstrip("/")
    try:
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(base)
        path = parsed.path.rstrip("/")
        known_prefixes = ("/api/remote", "/api/agent", "/api/ide", "/trae-cli")
        if any(path.startswith(prefix) for prefix in known_prefixes):
            base = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    except ValueError:
        pass
    return base


def resolve_traecode_base_urls(
    options: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return the TraeCode gateway candidates in request order.

    The desktop client receives ``trae-api-cn.mchost.guru`` from boot config;
    ttnet may internally dispatch that hostname to ``api5-normal``.  A relay
    cannot reproduce that client-side dispatch table, so retry the known
    alternate host when the configured gateway rejects the HTTP request.
    Custom gateways remain isolated and are never replaced by these presets.
    """

    options = options or {}
    primary = resolve_traecode_base_url(options)
    if not primary:
        return []
    configured = _traecode_option(
        options,
        "traecode_base_url",
        "traeCodeBaseUrl",
        "model_base_url",
        "base_url",
        "baseURL",
        default=None,
    )
    env_configured = (
        os.environ.get("TRAE_CODE_BASE_URL")
        or os.environ.get("TRAE_MODEL_API_HOST")
    )
    explicit = str(configured or env_configured or "").strip()
    if explicit and not _traecode_option(
        options, "_traecode_gateway_fallback", default=True
    ):
        return [primary]
    known = (
        "https://trae-api-cn.mchost.guru",
        "https://api5-normal.mchost.guru",
    )
    normalized = primary.rstrip("/")
    if normalized not in known:
        return [primary]
    return [primary] + [candidate for candidate in known if candidate != normalized]


async def send_traecode_chat_request(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool = True,
    options: Optional[dict[str, Any]] = None,
) -> RawChatResponse:
    """Send a TraeCode IDE/SOLO raw request with v1 endpoint fallback.

    ``RawChatResponse`` deliberately matches the legacy raw transport wrapper,
    so the existing SSE translator can consume both protocols while callers
    choose the protocol explicitly via ``_traecode_raw_chat_function``.
    """

    options = dict(options or {})
    token = str(options.get("_auth_token") or "").strip()
    if not token:
        await auth.maybe_refresh()
        token = str(auth.get_token() or "").strip()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available for TraeCode raw chat")

    base_urls = resolve_traecode_base_urls(options)
    if not base_urls:
        raise RuntimeError("No TraeCode model gateway configured")
    solo = bool(_traecode_option(options, "_traecode_solo", "traecode_solo", "solo", default=False))
    raw_chat_function = traecode_raw_chat_function(options, solo=solo)
    raw_model = await resolve_traecode_model_for_request(model, options)
    session_id = traecode_session_id(raw_model, options, raw_chat_function)
    body_options = dict(options)
    body_options.setdefault("_traecode_solo", solo)
    body_options.setdefault("_traecode_raw_chat_function", raw_chat_function)
    body_options.setdefault("_traecode_wire_stream", True)
    # ``build_traecode_chat_body`` is also a public synchronous helper.  Feed
    # it the result of the async/live resolver explicitly so a model catalog
    # entry such as ``config_name=deepseek-v4.1-flash`` cannot be replaced by
    # the offline alias resolver (or by the provider default, often Kimi).
    body_options.setdefault("traecode_config_name", raw_model.config_name)
    body_options.setdefault("traecode_model_name", raw_model.raw_model_name)
    body_options.setdefault("display_name", raw_model.display_name)
    body_options.setdefault("config_source", raw_model.config_source)
    if raw_model.descriptor is not None:
        body_options["_traecode_model_descriptor"] = raw_model.descriptor
    elif requested_level(body_options):
        # A live capability decision is required before forwarding effort.
        # When the bound catalog is unavailable, fail closed for the actual
        # transport instead of guessing support from the model name.
        body_options["_traecode_model_descriptor"] = {}
    body = build_traecode_chat_body(
        messages,
        model,
        stream,
        body_options,
        session_id=session_id,
    )
    request_id = str(uuid.uuid4())
    headers = build_traecode_headers(
        base_urls[0],
        token,
        raw_model,
        request_id,
        options,
        session_id=session_id,
        raw_chat_function=raw_chat_function,
    )
    logger.info(
        "traecode raw request id=%s gateways=%s endpoint=%s function=%s model=%s messages=%d tools=%d",
        str(options.get("_relay_request_id") or request_id),
        ",".join(base_urls),
        TRAECODE_RAW_CHAT_ENDPOINT,
        raw_chat_function,
        raw_model.raw_model_name,
        len(body.get("messages") or []),
        len(_traecode_tool_definitions(options.get("tools") or options.get("_inherited_tools"))),
    )

    release_session = await _acquire_raw_session_gate(
        session_id,
        str(options.get("_account_id") or ""),
        token,
        str(options.get("_account_slot_reservation_id") or ""),
    )
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))
    endpoints = [TRAECODE_RAW_CHAT_ENDPOINT]
    if bool(_traecode_option(options, "_traecode_allow_v1_fallback", default=True)):
        endpoints.append(TRAECODE_RAW_CHAT_FALLBACK_ENDPOINT)

    def open_stream() -> RawChatResponse:
        """Open v2/v1 synchronously and own all resources on the worker thread.

        ``httpx.Client.send`` cannot be cancelled safely once a socket is in
        flight. Keeping client creation, fallback, and cleanup in this worker
        means a cancelled public request never closes the client or releases
        the model gate while the worker still owns it.
        """

        client: Optional[httpx.Client] = None
        response: Optional[httpx.Response] = None
        detail = ""
        try:
            client = httpx.Client(timeout=timeout, http2=False)
            for base_url in base_urls:
                request_headers = headers
                if base_url != base_urls[0]:
                    # Keep the model/session envelope identical while
                    # updating the host-dependent compatibility headers.
                    request_headers = build_traecode_headers(
                        base_url,
                        token,
                        raw_model,
                        request_id,
                        options,
                        session_id=session_id,
                        raw_chat_function=raw_chat_function,
                    )
                for endpoint in endpoints:
                    try:
                        request = client.build_request(
                            "POST",
                            base_url + endpoint,
                            headers=request_headers,
                            json=body,
                        )
                        response = client.send(request, stream=True)
                    except httpx.RequestError as exc:
                        detail = str(exc)[:800]
                        logger.warning(
                            "traecode raw transport failed id=%s base=%s endpoint=%s error=%s",
                            str(options.get("_relay_request_id") or request_id),
                            base_url,
                            endpoint,
                            detail,
                        )
                        continue
                    if response.status_code == 200:
                        return RawChatResponse(
                            response=response,
                            client=client,
                            auth_token=token,
                            release_session=release_session,
                        )
                    status_code = response.status_code
                    detail = ""
                    rate_limit_code = 429 if status_code == 429 else None
                    try:
                        # Read the body before closing it so the connection can
                        # be reused for the next host/path candidate.
                        try:
                            response.read()
                            detail = response.text[:800]
                        except httpx.RequestError as exc:
                            detail = str(exc)[:800]
                        if rate_limit_code is None:
                            try:
                                payload = response.json()
                            except (httpx.ResponseNotRead, ValueError):
                                payload = None
                            if isinstance(payload, Mapping):
                                nested = payload.get("error")
                                sources = [payload]
                                if isinstance(nested, Mapping):
                                    sources.append(nested)
                                rate_limit_code = next(
                                    (
                                        source[key]
                                        for source in sources
                                        for key in ("code", "error_code", "errorCode")
                                        if str(source.get(key, "")).strip()
                                        == "3004"
                                    ),
                                    None,
                                )
                    finally:
                        response.close()
                        response = None
                    logger.warning(
                        "traecode raw endpoint failed id=%s base=%s endpoint=%s status=%s detail=%s",
                        str(options.get("_relay_request_id") or request_id),
                        base_url,
                        endpoint,
                        status_code,
                        detail,
                    )
                    if rate_limit_code is not None:
                        # Every candidate uses the same credential, so probing
                        # another host/path only repeats an account-level limit.
                        raise TraeCodeRateLimitError(
                            status_code, rate_limit_code, detail
                        )
            raise RuntimeError(
                "TraeCode raw chat request failed on "
                + ", ".join(
                    f"{base}{endpoint}"
                    for base in base_urls
                    for endpoint in endpoints
                )
                + (f": {detail}" if detail else "")
            )
        except BaseException:
            if response is not None:
                response.close()
            if client is not None:
                client.close()
            release_session()
            raise

    try:
        open_task = asyncio.create_task(asyncio.to_thread(open_stream))
    except BaseException:
        release_session()
        raise

    try:
        # Shield the worker just like the legacy raw path. If the public
        # request is cancelled while headers are pending, the worker keeps
        # ownership of the client and session gate until it completes.
        return await asyncio.shield(open_task)
    except asyncio.CancelledError:
        def close_cancelled_open(task: asyncio.Task[RawChatResponse]) -> None:
            try:
                pending_response = task.result()
            except BaseException:
                # ``open_stream`` releases the gate on every internal error.
                # The release closure is idempotent, so this is safe for
                # cancellation races where the worker has already failed.
                release_session()
                return
            try:
                pending_response.close()
            except Exception as exc:
                logger.warning("traecode cancelled response cleanup failed: %s", exc)

        open_task.add_done_callback(close_cancelled_open)
        raise
    except BaseException:
        # ``open_stream`` owns cleanup for worker failures. Keep this
        # idempotent fallback for task/executor setup errors.
        release_session()
        raise
_KNOWN_MODELS = {
    # These names are the exact raw ids used by TraeWork's bundled client.
    "auto": RawModel("glm-5.2", "glm-5.2", "GLM-5.2"),
    "coding": RawModel("glm-5.1", "glm-5__v2", "GLM-5.1"),
    # GLM-5.2 has its own raw provider id.  ``glm-5__v2`` belongs to the
    # GLM-5.1/coding config; sending it for GLM-5.2 lets the upstream bind the
    # session to its default provider (observed as an unexpected Kimi model).
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
    # Trae's web UI labels are not always the config names expected by the
    # raw transport. Keep an offline fallback for current CN labels.
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
    descriptor = _option(
        options,
        "_traecode_model_descriptor",
        "traecode_model_descriptor",
        "model_descriptor",
        "custom_model",
        "customModel",
        default=None,
    )
    return RawModel(
        str(config_name),
        str(raw_model_name),
        str(display_name),
        config_source,
        provider,
        descriptor=copy.deepcopy(dict(descriptor))
        if isinstance(descriptor, Mapping)
        else None,
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
    if known is not None:
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
                descriptor=copy.deepcopy(dict(config)),
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
    cache_key = _option(
        options,
        "_prompt_cache_key_affinity",
        default=None,
    )
    if isinstance(cache_key, str) and cache_key.strip():
        material_parts.extend(("prompt-cache-v1", cache_key.strip()[:1024]))
    material = "\x1f".join(material_parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def resolve_raw_base_url(options: Optional[Mapping[str, Any]] = None) -> str:
    """Resolve the TraeWork raw gateway host.

    The settings API stores the selected native gateway separately as
    ``model_base_url``.  Prefer that value when present; otherwise honor the
    Work-specific environment and retain the legacy diagnostic raw base.
    """

    options = options or {}
    mode = str(
        _option(options, "_upstream_mode", "_trae_mode", "upstream_mode", default="")
        or ""
    ).strip().lower()
    configured_model = _option(
        options,
        "model_base_url",
        "modelBaseUrl",
        "traework_base_url",
        "traeWorkBaseUrl",
    )
    configured_raw = _option(
        options,
        "raw_base_url",
        "rawBaseUrl",
        "work_base_url",
        "workBaseUrl",
        "base_url",
        "baseURL",
    )
    work_mode = mode in {"work", "work-agent", "traework", "traework-native", "native"}
    if work_mode:
        # ``model_base_url`` is the persisted model gateway for the Work
        # product. Keep it scoped to Work so a stale shared setting cannot
        # redirect a Trae raw/enterprise request to the TraeCode host.
        base = (
            configured_model
            or configured_raw
            or os.environ.get("TRAE_WORK_BASE_URL")
            or os.environ.get("TRAE_MODEL_API_HOST")
            or os.environ.get("TRAE_RAW_BASE_URL")
        )
    else:
        # Raw/diagnostic mode has its own gateway override. Explicit raw
        # options always win, while Work/TraeCode environment variables are
        # deliberately ignored to avoid cross-product host contamination.
        base = (
            configured_raw
            or os.environ.get("TRAE_RAW_BASE_URL")
        )
    if not base:
        api_host = os.environ.get("TRAE_API_HOST") or ""
        if not api_host:
            try:
                api_host = str(getattr(auth.get_auth(), "host", "") or "")
            except Exception:
                api_host = ""
        # www/api.trae.com.cn is the account/auth host and is not a model
        # gateway. Only reuse TRAE_API_HOST when it already names the model
        # gateway used by the Trae client.
        if "mchost.guru" in api_host or "trae-api-" in api_host:
            base = api_host
    base = base or "https://trae-api-cn.mchost.guru"
    return _strip_raw_gateway_path(str(base))


def _strip_raw_gateway_path(value: str) -> str:
    """Strip known API paths from a configured native gateway URL."""

    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    path = parsed.path.rstrip("/")
    known_prefixes = (
        "/api/remote",
        "/api/ide",
        "/api/agent",
        "/trae-cli",
        "/api/v1/commercial",
    )
    if any(path.startswith(prefix) for prefix in known_prefixes):
        path = ""
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


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
    # Never fabricate a workspace path: an invented absolute root makes the
    # model rewrite the caller's relative paths (``r.txt`` -> ``C:\workspace\r.txt``).
    workspace_path = str(
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
        or ""
    )
    if workspace_path:
        context["workspace_path"] = workspace_path
    else:
        context.pop("workspace_path", None)
    context["system_type"] = system_type
    terminal_context = context.get("terminal_context", context.get("terminalContext"))
    if terminal_context is None:
        terminal_context = _mapping_value(metadata, "terminal_context", "terminalContext")
    if terminal_context is None:
        terminal_context = [
            {
                "shell": "PowerShell" if "windows" in system_type.lower() else "bash",
                "cwd": workspace_path or ".",
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
            elif images.is_image_part(block) or str(block.get("type") or "") == "image":
                converted = images.to_renderer_block(block)
                parts.append(images.describe(converted) if converted else "[image omitted]")
            elif block.get("type") == "file":
                parts.append("[Unsupported file input omitted]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _without_native_images(content: Any) -> Any:
    """Drop image parts that :func:`_native_image_blocks` forwards natively."""

    if not isinstance(content, list):
        return content
    kept: list[Any] = []
    for part in content:
        if isinstance(part, Mapping):
            if images.is_image_part(part) or str(part.get("type") or "") == "image":
                continue
        kept.append(part)
    return kept


def _native_image_blocks(content: Any) -> list[dict[str, Any]]:
    """Return renderer media blocks for any images in one message's content."""

    if not isinstance(content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if not images.is_image_part(part) and str(part.get("type") or "") != "image":
            continue
        converted = images.to_renderer_block(part)
        if converted:
            blocks.append(converted)
    return blocks


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
        tool_defs, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str
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
            candidate, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str
        )
        if len(encoded) > budget:
            continue
        payload = candidate
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str
    )
    return encoded, True, int(payload["omitted_signatures"])


def _concise_reasoning_enabled() -> bool:
    """Whether to ask the model for conclusions instead of a narrated chain.

    Share the live output setting with the reasoning formatters.
    """

    return not bool(auth.get_reasoning_output_settings().get("enabled"))


def response_style_instruction() -> str:
    """Return the concise-answer directive, or "" when verbose mode is on."""

    if not _concise_reasoning_enabled():
        return ""
    return (
        "Response format: your visible reply must contain only the final "
        "answer. Start with the result itself. Never begin with a restatement "
        "of the request or a description of your plan. Do not write sentences "
        "about the user, the question, or yourself, such as \"The user wants\", "
        "\"The user is asking\", \"Let me\", \"I need to\", \"This doesn't "
        "require\", or \"No tools needed\". Keep the reasoning steps that the "
        "answer genuinely needs (calculations, key facts, code) and drop "
        "everything else. Emit each code block exactly once."
    )


def build_runtime_system_prompt(
    tools: Any,
    client_context: Mapping[str, Any],
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    *,
    native_tools: bool = False,
) -> str:
    """Describe the caller-owned runtime using the selected tool protocol."""

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
        "The external Trae-compatible terminal client application that started this conversation provides the client tools below.",
        "Caller client context (JSON):",
        json.dumps(context, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str),
        "Discover the local environment through the available client tools when the request requires it.",
        "If a tool can answer the question, call it proactively and wait for the external client result.",
        "Remote or server-side tools cannot write into the caller workspace. Never claim that a client file was downloaded, created, edited, patched, or saved unless a later matching client tool result with the same call id confirms success.",
        "A tool request, emitted tool-call block, plan, or assistant narration is not execution proof. If the matching client result reports failure, state the failure or request a corrective client tool; never report the operation as completed.",
        "Never repeat a completed tool call; do not describe that server's Linux filesystem as the caller workspace.",
    ]
    if native_tools:
        lines.extend(
            [
                "Use native function calling for the client tools supplied in the request. Their definitions and input schemas are in the tools field.",
                "Return structured tool calls, not tool-call markup or JSON tool blocks in the response text. Use an exact declared function name and arguments matching its schema.",
                "The external API client executes these calls. Wait for the matching tool result before continuing; never substitute server-side tools to access the caller's workspace.",
                "Keep file paths exactly as the user or tool result gave them.",
            ]
        )
    else:
        lines.extend(
            [
                "Client tools are NOT native/server functions: never invoke them through function calling or any built-in tool, and never fall back to server tools (LS, Glob, Read, terminal) to reach the caller's files. The only way to call a client tool is the text block below; a native call reports 'not available' because the tool lives on the client.",
                "To use tools, emit only one or more JSON blocks and include no prose or final answer in that turn:",
                '<opencode_tool_call>{"name":"tool_name","input":{}}</opencode_tool_call>',
                "Do not invent an id field; the client assigns call ids. Keep file paths exactly as the user or tool result gave them.",
                "Fill input according to the selected tool schema and wait for the client result before continuing.",
            ]
        )
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
                "To call a tool, emit one block per call in this format and no prose or final answer in the same turn: <opencode_tool_call>{\"name\":\"tool_name\",\"input\":{}}</opencode_tool_call>. Omit the id field (the client assigns it), use an exact declared tool name, and fill input according to that tool's schema.",
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
    renderer_calls = _renderer_tool_use_blocks(message.get("content"))
    if renderer_calls:
        text += json.dumps(
            renderer_calls, ensure_ascii=False, separators=(",", ":"), default=str
        )
    renderer_results = _renderer_tool_result_blocks(message.get("content"))
    if renderer_results:
        text += json.dumps(
            renderer_results, ensure_ascii=False, separators=(",", ":"), default=str
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
    if candidates and _message_has_tool_results(valid[candidates[0]]):
        first_position = non_system_indexes.index(candidates[0])
        if first_position:
            previous = non_system_indexes[first_position - 1]
            previous_message = valid[previous]
            if _message_has_assistant_tool_calls(previous_message):
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


def _renderer_tool_result_blocks(content: Any) -> list[dict[str, Any]]:
    """Collect renderer ``tool_result`` blocks without losing call identity."""

    if not isinstance(content, list):
        return []
    results: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        if str(block.get("type") or "").strip().lower() != "tool_result":
            continue
        call_id = str(
            block.get("toolCallId")
            or block.get("tool_call_id")
            or block.get("call_id")
            or block.get("id")
            or "unknown"
        )
        results.append(
            {
                "id": call_id,
                "name": str(block.get("name") or ""),
                "content": renderer_block_text(block),
                "failed": tool_result_is_failed({"content": [block]}),
            }
        )
    return results


def _message_has_assistant_tool_calls(message: Mapping[str, Any]) -> bool:
    return bool(
        str(message.get("role") or "") == "assistant"
        and (
            message.get("tool_calls")
            or message.get("function_call")
            or _renderer_tool_use_blocks(message.get("content"))
        )
    )


def _message_has_tool_results(message: Mapping[str, Any]) -> bool:
    return bool(
        str(message.get("role") or "") in {"tool", "function"}
        or _renderer_tool_result_blocks(message.get("content"))
    )


def _tool_call_name_map(messages: Any) -> dict[str, str]:
    """Map call ids to tool names across OpenAI and renderer histories."""

    names: dict[str, str] = {}
    if not isinstance(messages, list):
        return names
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        calls: list[Any] = []
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend(tool_calls)
        function_call = message.get("function_call")
        if isinstance(function_call, Mapping):
            calls.append(function_call)
        calls.extend(_renderer_tool_use_blocks(message.get("content")))
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else call
            call_id = str(
                call.get("id")
                or call.get("tool_call_id")
                or call.get("toolCallId")
                or ""
            ).strip()
            name = str(function.get("name") or call.get("name") or "").strip()
            if call_id and name:
                names[call_id] = name
    return names


def _renderer_tool_result_context(
    content: Any, tool_names: Optional[Mapping[str, str]] = None
) -> str:
    """Serialize renderer results as inert, correlated conversation history."""

    if not isinstance(content, list):
        return ""
    tool_names = tool_names or {}
    parts: list[str] = []
    found_result = False
    for block in content:
        if not isinstance(block, Mapping):
            text = _content_to_text([block])
            if text:
                parts.append(text)
            continue
        if str(block.get("type") or "").strip().lower() != "tool_result":
            text = _content_to_text([block])
            if text:
                parts.append(text)
            continue
        found_result = True
        call_id = str(
            block.get("toolCallId")
            or block.get("tool_call_id")
            or block.get("call_id")
            or block.get("id")
            or "unknown"
        )
        tool_name = str(block.get("name") or tool_names.get(call_id) or "tool")
        status = "failed" if tool_result_is_failed({"content": [block]}) else "succeeded"
        result_text = renderer_block_text(block) or "[empty tool result]"
        parts.append(
            f"Client tool result [{call_id}] {tool_name} status={status}:\n{result_text}"
        )
    return "\n\n".join(parts) if found_result else ""


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
                or _renderer_tool_use_blocks(message.get("content"))
                or _renderer_tool_result_blocks(message.get("content"))
            )
            for message in messages
        )
    )
    tool_names = _tool_call_name_map(messages)
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
        image_blocks = _native_image_blocks(content)
        # The native content channel already carries the media block, so the
        # flattened text must not repeat a placeholder for the same image.
        text = _content_to_text(
            _without_native_images(content) if image_blocks else content
        )
        renderer_results = _renderer_tool_result_blocks(content)
        # ZCode may replay a truncated relay history preamble as assistant
        # text.  Forwarding that residue to Trae makes the model echo it again
        # on every continuation.  Only sanitize assistant history; user and
        # tool-result content remains authoritative and untouched.
        if role == "assistant" and text:
            text = strip_tool_call_blocks(text)
        if renderer_results:
            # TraeWork serializes tool results as user content blocks. Preserve
            # the call id and failure state instead of flattening them to an
            # indistinguishable user sentence.
            text = _renderer_tool_result_context(content, tool_names)
            role = "user"
        elif role == "developer":
            role = "system"
        elif role == "tool":
            call_id = str(
                message.get("tool_call_id") or message.get("toolCallId") or "unknown"
            )
            tool_name = str(message.get("name") or "tool")
            status = "failed" if tool_result_is_failed(message) else "succeeded"
            text = f"Client tool result [{call_id}] {tool_name} status={status}:\n{text}"
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
            native_message["content"] = [
                {"type": "text", "text": text},
                *image_blocks,
            ]
        elif image_blocks:
            # A message whose only payload is an image still has to reach the
            # multimodal model; the raw transport carries media as renderer
            # ``image_url`` blocks beside (or instead of) text.
            native_message["content"] = image_blocks
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

    release_session = await _acquire_raw_session_gate(
        upstream_session_id,
        str(options.get("_account_id") or ""),
        token,
        str(options.get("_account_slot_reservation_id") or ""),
    )

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
        # occupied model-request slot is not made available before the worker
        # has finished using the raw conversation.
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
