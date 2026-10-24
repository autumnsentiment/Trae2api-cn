"""
main.py - Trae CN Relay 中转站
提供 OpenAI 兼容的 REST API:  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/responses
  POST /v1/chat
  POST /v1

上游模式:  UPSTREAM_MODE=cli  - 只使用本地 Trae CLI 子进程
  UPSTREAM_MODE=auto - 与 raw 相同，所有模型请求直达 Trae 原生 chat 协议
  UPSTREAM_MODE=raw  - 直连 Trae 原生 chat 协议（direct 为别名）
  UPSTREAM_MODE=remote/9router - 只用 9router 风格 remote 会话
  UPSTREAM_MODE=web  - 只用旧版 CN remote 会话（兼容保留）
  UPSTREAM_MODE=ide  - TraeCode IDE /api/agent/v3/llm_utils_chat
  UPSTREAM_MODE=solo - TraeCode SOLO /api/agent/v3/llm_utils_chat
  UPSTREAM_MODE=traework-native - Windows helper 承载 TraeWork ai-agent.dll
"""

import asyncio
import base64
import binascii
import codecs
import gzip
import hashlib
import html as html_mod
import ipaddress
import json
import logging
import os
import re
import threading
import time
import secrets
import sys
import uuid as uuid_mod
import zlib
from collections import OrderedDict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

import dotenv

# Several transport modules cache environment-derived defaults at import time.
# Load the project .env before importing them so local and Docker startup use
# the same configuration order.
dotenv.load_dotenv()

import anyio
import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import (
    anthropic_api,
    auth,
    cli_client,
    continuation,
    images,
    raw_client,
    responses_api,
    trae_client,
    trae_remote_client,
    traework_compat,
    traework_native_bridge,
)
from .auth_helper import (
    SUPPORTED_KINDS as HELPER_KINDS,
    RelayURLError,
    candidate_hosts as _helper_candidate_hosts,
    file_name as _helper_file_name,
    media_type as _helper_media_type,
    normalize_listener_port as _normalize_listener_port,
    normalize_relay_url as _normalize_relay_url,
    pick_relay_url as _pick_relay_url,
    render as _render_auth_helper,
)
from .model_limits import (
    MULTIMODAL_FALLBACK_ORDER,
    clamp_max_completion_tokens,
    configured_multimodal_model,
    descriptor_max_context,
    descriptor_multimodal,
    descriptor_supports_max_mode,
)
from .sse import (
    EmptyUpstreamResponse,
    ModelProviderMismatch,
    RepeatedCompletedToolResponse,
    collect_nonstream_cli,
    collect_nonstream_ide,
    collect_nonstream_web,
    translate_cli_stream,
    translate_ide_stream,
    translate_web_events,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("trae-cn-relay")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
API_KEYS = [k.strip() for k in os.environ.get("RELAY_API_KEYS", "").split(",") if k.strip()]
UPSTREAM_MODE = (os.environ.get("UPSTREAM_MODE", "remote") or "remote").lower()
_VALID_UPSTREAM_MODES = (
    "remote", "raw", "direct", "auto", "web", "ide", "solo", "work",
    "work-agent",
    "9router", "trae-remote", "cli", "traework-native",
    "native", "traework",
)


def _current_upstream_mode() -> str:
    """Read the upstream mode at request time so preset changes apply immediately."""
    # The module attribute is authoritative.  The startup read already merged
    # env defaults, and runtime preset changes update both env and the module
    # value.  Tests can patch ``src.main.UPSTREAM_MODE`` to isolate scenarios.
    return str(globals().get("UPSTREAM_MODE") or "remote").lower()


def _restore_persisted_relay_settings() -> dict:
    """Apply console settings that live in the account store after startup.

    The container environment is a useful bootstrap/default, but the web
    console also persists relay settings in ``data/accounts.json``.  Restore
    those values after ``auth.init_auth()`` so a container recreation does not
    silently switch the selected upstream back to the image's ``.env``.
    """

    global UPSTREAM_MODE, WEB_BASE
    settings = auth.get_settings() or {}
    # ``endpoint_id`` is the authoritative persisted identity when present;
    # URL-only restoration is retained for stores written by older releases.
    endpoint_id = str(settings.get("endpoint_id") or "").strip().lower()
    endpoint = next(
        (
            item
            for item in UPSTREAM_ENDPOINT_PRESETS
            if str(item.get("id") or "").strip().lower() == endpoint_id
        ),
        None,
    ) if endpoint_id else None
    mode = str(
        (endpoint or {}).get("mode")
        or settings.get("upstream_mode")
        or ""
    ).strip().lower()
    if mode in _VALID_UPSTREAM_MODES:
        UPSTREAM_MODE = mode
        os.environ["UPSTREAM_MODE"] = mode
    web_base = str(
        (endpoint or {}).get("base_url")
        or settings.get("web_base_url")
        or ""
    ).strip().rstrip("/")
    if web_base:
        WEB_BASE = web_base
        os.environ["TRAE_WEB_BASE_URL"] = web_base
    # Preserve an explicitly persisted custom gateway.  The preset value is
    # only a migration fallback for stores written before model_base_url was
    # introduced; preferring it would silently discard a custom gateway on
    # every restart.
    persisted_model_base = settings.get("model_base_url")
    model_base = str(
        (
            persisted_model_base
            if persisted_model_base is not None
            else (endpoint or {}).get("model_base_url")
        )
        or ""
    ).strip().rstrip("/")
    # Earlier releases made api5-normal the TraeCode preset default. The
    # desktop client gets trae-api-cn from boot config and lets its network
    # layer route that hostname to api5-normal. Migrate only that old preset
    # value; an explicit TRAE_CODE_BASE_URL remains authoritative.
    legacy_traecode_gateway = "https://api5-normal.mchost.guru"
    env_code_gateway = str(os.environ.get("TRAE_CODE_BASE_URL") or "").strip().rstrip("/")
    if (
        (endpoint_id in {"ide", "solo"} or mode in {"ide", "solo"})
        and model_base == legacy_traecode_gateway
        and DEFAULT_MODEL_GATEWAY_URL != legacy_traecode_gateway
        and (
            not env_code_gateway
            or env_code_gateway == DEFAULT_MODEL_GATEWAY_URL
        )
    ):
        model_base = DEFAULT_MODEL_GATEWAY_URL
    if model_base:
        os.environ["TRAE_MODEL_API_HOST"] = model_base
        if mode in {"work", "work-agent"}:
            os.environ["TRAE_RAW_BASE_URL"] = model_base
            os.environ["TRAE_WORK_BASE_URL"] = model_base
        elif mode in {"ide", "solo"}:
            os.environ["TRAE_CODE_BASE_URL"] = model_base
    else:
        os.environ.pop("TRAE_MODEL_API_HOST", None)
        if mode in {"work", "work-agent"}:
            os.environ.pop("TRAE_RAW_BASE_URL", None)
            os.environ.pop("TRAE_WORK_BASE_URL", None)
        elif mode in {"ide", "solo"}:
            os.environ.pop("TRAE_CODE_BASE_URL", None)
    # Repair partially migrated stores once at startup.  This also makes the
    # settings API/UI observe the same canonical mode and URL as dispatch.
    if endpoint is not None and (
        settings.get("upstream_mode") != endpoint["mode"]
        or str(settings.get("web_base_url") or "").strip().rstrip("/")
        != str(endpoint["base_url"]).rstrip("/")
        or str(settings.get("model_base_url") or "").strip().rstrip("/")
        != model_base
    ):
        try:
            auth.set_relay_settings(
                web_base_url=str(endpoint["base_url"]),
                upstream_mode=str(endpoint["mode"]),
                endpoint_id=str(endpoint["id"]),
                model_base_url=model_base,
            )
            # ``auth.set_relay_settings`` persists the canonical values, but
            # an older/custom store adapter may return a stale snapshot from
            # ``get_settings`` immediately afterwards.  Return the same
            # canonical state that was applied to the running process so the
            # settings page cannot rehydrate the shared URL as Raw/Remote.
            settings = dict(settings)
            settings.update(
                {
                    "web_base_url": str(endpoint["base_url"]).rstrip("/"),
                    "upstream_mode": str(endpoint["mode"]).strip().lower(),
                    "endpoint_id": str(endpoint["id"]).strip().lower(),
                    "model_base_url": model_base,
                }
            )
        except Exception:
            logger.debug("could not repair persisted endpoint preset", exc_info=True)
    return settings

FORWARD_USAGE = (os.environ.get("FORWARD_USAGE", "true") or "true").lower() == "true"
CHECKIN_INTERVAL = float(os.environ.get("TRAE_CHECKIN_INTERVAL_SECONDS", "60") or "60")
CHECKIN_RETRY_AFTER = float(
    os.environ.get("TRAE_CHECKIN_9074_RETRY_SECONDS", "60") or "60"
)
CHECKIN_9074_MAX_BACKOFF = float(
    os.environ.get("TRAE_CHECKIN_9074_MAX_BACKOFF_SECONDS", "3600") or "3600"
)
# Cap the doubling so a long 9074 streak settles at minutes, not the ceiling.
CHECKIN_9074_BACKOFF_EXPONENT_CAP = max(
    0,
    int(os.environ.get("TRAE_CHECKIN_9074_BACKOFF_EXPONENT_CAP", "3") or "3"),
)
CHECKIN_AUTO_RETRY_INTERVAL = float(
    os.environ.get("TRAE_CHECKIN_AUTO_RETRY_INTERVAL_SECONDS", "60") or "60"
)
DEFAULT_REMOTE_BASE_URL = "https://trae-api-cn.mchost.guru/api/remote/v1"
DEFAULT_WORK_MODEL_GATEWAY_URL = (
    os.environ.get("TRAE_WORK_BASE_URL")
    or "https://trae-api-cn.mchost.guru"
).rstrip("/")
DEFAULT_MODEL_GATEWAY_URL = (
    os.environ.get("TRAE_CODE_BASE_URL")
    or "https://trae-api-cn.mchost.guru"
).rstrip("/")
WEB_BASE = (os.environ.get("TRAE_WEB_BASE_URL", DEFAULT_REMOTE_BASE_URL)).rstrip("/")
UPSTREAM_ENDPOINT_PRESETS = (
    {
        "id": "remote",
        "label": "Remote / chat_sessions（trae remote）",
        "base_url": DEFAULT_REMOTE_BASE_URL,
        "endpoint_url": DEFAULT_REMOTE_BASE_URL + "/chat_sessions",
        "endpoint_path": "/api/remote/v1/chat_sessions",
        "mode": "remote",
        "product": "trae",
    },
    {
        "id": "work",
        "label": "TraeWork / chat_sessions（Work tier）",
        "base_url": DEFAULT_REMOTE_BASE_URL,
        "endpoint_url": DEFAULT_REMOTE_BASE_URL + "/chat_sessions",
        "endpoint_path": "/api/remote/v1/chat_sessions",
        "mode": "work",
        "product": "traework",
    },
    {
        "id": "solo",
        "label": "TraeCode Solo / llm_utils_chat（solo_work_lite）",
        "base_url": DEFAULT_REMOTE_BASE_URL,
        "model_base_url": DEFAULT_MODEL_GATEWAY_URL,
        "endpoint_url": DEFAULT_MODEL_GATEWAY_URL + "/api/agent/v3/llm_utils_chat",
        "endpoint_path": "/api/agent/v3/llm_utils_chat",
        "mode": "solo",
        "legacy_function": "solo_work_lite",
        "product": "trae",
    },
    {
        "id": "ide",
        "label": "TraeCode IDE / llm_utils_chat（solo_work_lite）",
        "base_url": DEFAULT_REMOTE_BASE_URL,
        "model_base_url": DEFAULT_MODEL_GATEWAY_URL,
        "endpoint_url": DEFAULT_MODEL_GATEWAY_URL + "/api/agent/v3/llm_utils_chat",
        "endpoint_path": "/api/agent/v3/llm_utils_chat",
        "mode": "ide",
        "legacy_function": "solo_work_lite",
        "product": "trae",
    },
    # Settings written by releases before the four-way endpoint split remain
    # readable.  The entry is intentionally hidden from the public selector;
    # new settings use ``work`` above.
    {
        "id": "agent",
        "label": "Work Agent（兼容）/ chat_sessions",
        "base_url": DEFAULT_REMOTE_BASE_URL,
        "endpoint_url": DEFAULT_REMOTE_BASE_URL + "/chat_sessions",
        "endpoint_path": "/api/remote/v1/chat_sessions",
        "mode": "work-agent",
        "product": "traework",
        "visible": False,
    },
)
CHAT_OPTION_FIELDS = (
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "client_context",
    "clientContext",
    "session_id",
    "sessionId",
    "max_tokens",
    "maxTokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    # ``thinking`` is a client-visible presentation/request hint.  It must be
    # retained through the relay even though the native raw body has no field
    # for it; the SSE/Responses translators use it to expose a compact
    # reasoning summary without leaking the full upstream trace.
    "thinking",
    "reasoning_effort",
    # Per-request 1M max opt-in; the console switch is the global default.
    "trae_max_mode",
    "max_mode",
    "stream_options",
    "response_format",
    "service_tier",
    "user",
    "logprobs",
    "top_logprobs",
    # TraeWork model-selection aliases passed through to the raw transport.
    # Raw session ids are intentionally not caller-overridable: the relay pins
    # one upstream conversation to each account/model pair.
    "traeRawConfigName",
    "traeRawModelName",
    "rawModelName",
    "configName",
    "displayName",
    "modelName",
    "provider",
    "configSource",
    # TraeWork native Ode/Gpt payload aliases. These are ignored by raw/remote
    # transports and are consumed only by the opt-in Windows helper route.
    "native_data",
    "native_user_info",
    "native_common_params",
    "native_streamlined_common_params",
    "native_client_info",
    "connect_session_id",
    "connectSessionId",
    "native_session_id",
    "native_channel_id",
    "channel_id",
    "workspace_folder",
    "workspacePath",
    "workspace_id",
    "workspaceId",
    "device_id",
    "deviceId",
    "agent_type",
    "shell_execute_strategy",
    "model_auto_selection",
    "custom_model",
    "model_config_source",
    "modelConfigSource",
    "model_is_preset",
    "modelIsPreset",
    "model_provider",
    "modelProvider",
    "ppe_env_name",
    "ppeEnvName",
    "envLane",
    "agentEnv",
    "forceSandboxType",
    "version_code",
    "versionCode",
)

# Per-request usage tracking. Records are stored separately from account data so
# a dashboard/deploy change can never rewrite the saved login cache.
_USAGE_HISTORY: list[dict] = []


def _usage_history_limit() -> int:
    """Return the persisted request-history cap.

    The daily usage buckets are the long-lived source of truth for totals, but
    the detail endpoint should not discard almost every request in a busy
    relay.  Keep the default large enough for normal deployments while
    allowing operators to tune disk usage explicitly.
    """

    raw = os.environ.get("TRAE_USAGE_MAX_HISTORY", "1000")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1000
    return max(100, min(value, 100_000))


_USAGE_MAX_HISTORY = _usage_history_limit()
_USAGE_LOCK = threading.RLock()
_USAGE_RECORDS_PATH = Path(
    os.environ.get("TRAE_USAGE_RECORDS_PATH", "")
    or (Path(__file__).resolve().parent.parent / "data" / "usage_records.json")
)
# Lifetime token totals bucketed by Asia/Shanghai day. The request history
# above is capped, so the dashboard overview reads this file instead.
_USAGE_STATS_PATH_OVERRIDE = os.environ.get("TRAE_USAGE_STATS_PATH", "")
_USAGE_STATS: dict[str, Any] = {"days": {}}
_USAGE_STATS_FIELDS = (
    "requests",
    "completed",
    "failed",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "total_tokens",
    "credits",
    # Request capability metadata is aggregated separately from token
    # accounting so the dashboard can answer "how many requests used Max or
    # tools?" without scanning the capped request history.
    "max_mode_requests",
    "max_mode_enabled_requests",
    "tool_requests",
    "tool_calls",
)
_USAGE_STATS_MAX_DAYS = 400
_PROCESS_STARTED_AT = time.time()

# Project logo: a "T" whose crossbar runs into a relay arrow, black tile with a
# green rim to match the console theme. docs/logo.svg mirrors this markup.
_LOGO_PATHS = (
    '<rect x="3" y="3" width="58" height="58" rx="14" fill="#0b100d" '
    'stroke="#22c55e" stroke-width="4"/>'
    '<g fill="none" stroke="#22c55e" stroke-width="6" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M17 22h27"/><path d="M39 14l9 8-9 8"/>'
    '<path d="M28 22v24"/></g>'
    '<circle cx="28" cy="47" r="4.5" fill="#4ade80"/>'
)
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" '
    'aria-label="Trae2api-cn"><title>Trae2api-cn</title>' + _LOGO_PATHS + "</svg>"
)
_LOGO_SVG_INLINE = (
    '<svg viewBox="0 0 64 64" aria-hidden="true" focusable="false">'
    + _LOGO_PATHS
    + "</svg>"
)
_FAVICON_LINKS = (
    '<link rel="icon" type="image/svg+xml" href="/favicon.svg">'
    '<link rel="apple-touch-icon" href="/favicon.svg">'
)
_USAGE_TRACKER: ContextVar[Any] = ContextVar("trae_usage_tracker", default=None)
_USAGE_ENRICH_TASKS: set[asyncio.Task] = set()
_USAGE_SNAPSHOT_TASKS: set[asyncio.Task] = set()
_USAGE_PENDING_TASK_IDS: set[str] = set()
_USAGE_ACTIVE_ACCOUNTS: dict[str, int] = {}
_USAGE_UNSAFE_ACCOUNTS: set[str] = set()
# Concurrent requests for one account share the same "before" snapshot.  A
# per-request snapshot makes the second request look unsafe and drops its
# credit delta, even though the account total still advances normally.
_USAGE_ACCOUNT_BEFORE_TASKS: dict[tuple[int, str], asyncio.Task] = {}
# The after-snapshot read is serialized per account and compared with the last
# observed total.  This turns two concurrent deltas (10 -> 11 -> 12) into two
# one-credit rows instead of 1 and 2 credits or an unknown row.
_USAGE_CREDIT_SETTLE_GATES: dict[tuple[int, str], asyncio.Lock] = {}
_USAGE_LAST_CREDIT_TOTALS: dict[str, float] = {}
# Credit display reconciliation is intentionally separate from request
# settlement.  The latter attributes a single turn; this overlay reconciles
# all confirmed charges that arrived after the last official balance poll.
_CREDIT_RECONCILE_TASKS: set[asyncio.Task] = set()
_CREDIT_RECONCILE_STATE: dict[str, dict[str, Any]] = {}
_CHECKIN_ACCOUNT_LOCKS: dict[str, asyncio.Lock] = {}
_CHECKIN_CLAIM_GATE: asyncio.Lock | None = None
_CHECKIN_CLAIM_GATE_LOOP: asyncio.AbstractEventLoop | None = None
_CHECKIN_NEXT_CLAIM_AT = 0.0
_CHECKIN_COOLDOWN_UNTIL: dict[str, float] = {}
# The claim endpoint and the status endpoint are throttled independently: 9074
# ("too many users, retry later") on claim does not stop status from answering
# code=0. Track the status-side window separately so a claim cooldown cannot
# freeze the dashboard on a stale checked_in value.
_CHECKIN_STATUS_COOLDOWN_UNTIL: dict[str, float] = {}
_CHECKIN_ACCEPTED_UNTIL: dict[str, float] = {}
_CHECKIN_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
# Kept as a compatibility knob for older integrations; claim no longer runs
# automatic verification probes.
_CHECKIN_VERIFY_DELAYS: tuple[float, ...] = ()

# OpenAI clients normally replay the complete conversation instead of sending
# a relay-specific session id. Tool execution can legitimately take several
# minutes, so keep the credential/session binding long enough for those
# continuations to return to the same upstream account.
_CHAT_SESSION_TTL = max(
    1.0,
    float(
        os.environ.get(
            "TRAE_SESSION_IDLE_TIMEOUT_SECONDS",
            os.environ.get("TRAE_CHAT_SESSION_TTL_SECONDS", "3600"),
        )
        or "3600"
    ),
)
_CHAT_SESSION_MAX = max(64, int(os.environ.get("TRAE_CHAT_SESSION_CACHE_SIZE", "2048") or "2048"))
_CHAT_SESSION_LOCK = threading.RLock()
_CHAT_HISTORY_SESSIONS: OrderedDict[str, tuple[str, float]] = OrderedDict()


@dataclass
class _UpstreamSessionLease:
    account_id: str
    billing_id: str
    auth_token: str
    last_client_activity: float
    active_streams: int = 0
    provider_specific: dict[str, Any] = field(default_factory=dict)


_UPSTREAM_SESSION_LEASES: OrderedDict[str, _UpstreamSessionLease] = OrderedDict()


class _ModelAccountUnavailable(RuntimeError):
    """No enabled credential can receive this model request."""


def _cancel_unclaimed_account_slot(options: Optional[Mapping[str, Any]]) -> bool:
    """Return a scheduler reservation that no transport consumed."""

    if not options or not options.get("_account_slot_reserved"):
        return False
    account_id = str(
        options.get("_account_slot_reservation_account")
        or options.get("_account_id")
        or ""
    )
    reservation_id = str(options.get("_account_slot_reservation_id") or "")
    cancelled = auth.cancel_account_slot_reservation(account_id, reservation_id)
    if cancelled and isinstance(options, dict):
        options["_account_slot_reserved"] = False
    return cancelled


def _ensure_model_account_enabled(account_id: str = "", token: str = "") -> None:
    identities = {str(account_id or ""), _account_id_from_token(token)} - {""}
    for identity in identities:
        if not auth.is_account_model_enabled(identity):
            raise _ModelAccountUnavailable(
                "The bound model account is disabled; enable it in 账号与签到 "
                "or start a new session with an enabled account"
            )
    if not identities:
        accounts = auth.get_accounts_raw()
        if accounts and all(not auth._record_model_enabled(rec) for _, rec in accounts):
            raise _ModelAccountUnavailable(
                "No enabled model account available; enable an account in 账号与签到"
            )


def _credit_settle_seconds() -> float:
    try:
        value = float(os.environ.get("TRAE_USAGE_CREDIT_SETTLE_SECONDS", "1"))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(value, 10.0))


def _session_usage_enabled() -> bool:
    return str(os.environ.get("TRAE_USAGE_SESSION_QUERY", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _session_usage_query_attempts() -> int:
    try:
        value = int(os.environ.get("TRAE_USAGE_QUERY_RETRIES", "5") or "5")
    except (TypeError, ValueError):
        value = 5
    return max(1, min(value, 8))


def _session_usage_query_delay() -> float:
    try:
        value = float(
            os.environ.get("TRAE_USAGE_QUERY_RETRY_DELAY_SECONDS", "0.5")
            or "0.5"
        )
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, min(value, 10.0))


# Web login auth
TRAE_AUTH_URL = os.environ.get("TRAE_AUTH_URL", "https://www.trae.cn/authorization")
TRAE_CLIENT_ID = os.environ.get("TRAE_CLIENT_ID") or "ono9krqynydwx5"
LOCAL_LISTENER_PORT = int(os.environ.get("WEB_LOGIN_LISTENER_PORT", "8765"))
PUBLIC_PATHS = {
    "/healthz",
    "/favicon.svg",
    "/favicon.ico",
    "/v1/status",
    "/v1/models",
    "/models",
    "/web/login",
    "/web/login/download",
    "/web/login/helper-info",
    "/authorize",
    "/api/web-auth",
    "/api/logout",
    "/api/accounts",
    "/api/accounts/switch",
    "/api/accounts/remove",
    "/api/settings",
    "/api/polling",
    "/api/polling-mode",
    "/api/max-mode",
    "/api/max-mode/models",
    "/api/auto-route",
    "/api/auto-checkin",
    "/api/auto-checkin/run",
    "/api/checkin/status",
    "/api/checkin/claim",
    "/api/checkin/accounts",
    "/api/checkin/claim-all",
    "/api/checkin/claim-credits",
    "/api/checkin/account",
    "/api/checkin/work-credits",
    "/api/usage/last",
    "/api/usage/records",
    "/api/overview",
    # TraeWork custom-model management probes cannot reliably attach the
    # relay API key before the model has been saved. Chat ingress remains
    # protected by the normal middleware; only connectivity is public.
    "/api/agent/v3/custom_model_connectivity_check",
    "/api/ide/v1/custom_model_connectivity_check",
    "/v1/custom_model/connectivity",
}
PUBLIC_PATH_PREFIXES = ("/api/checkin", "/api/accounts")
WEB_LOGIN_SCRIPT = Path(__file__).resolve().parent.parent / "web_login.py"


class _RequestBodyError(ValueError):
    """A client request body could not be decoded as an OpenAI JSON object."""

    def __init__(self, message: str, *, raw_bytes: int = 0):
        super().__init__(message)
        self.raw_bytes = max(0, int(raw_bytes))


def _request_body_charset(content_type: str) -> str:
    """Return a safe charset from Content-Type, defaulting to UTF-8."""

    match = re.search(r"(?:^|;)\s*charset\s*=\s*['\"]?([^;\"']+)", content_type, re.I)
    candidate = (match.group(1).strip() if match else "utf-8")
    try:
        return codecs.lookup(candidate).name
    except LookupError:
        return "utf-8"


def _decode_request_content(raw: bytes, content_encoding: str) -> bytes:
    """Decode common HTTP content codings before JSON parsing.

    httpx/zcode may gzip the request body while the relay is served directly
    by uvicorn. Starlette intentionally leaves Content-Encoding untouched, so
    ``Request.json()`` would reject an otherwise valid payload. Decode only
    codings available in the standard library; unknown codings are reported to
    the caller instead of silently forwarding an empty prompt.
    """

    codings = [
        item.strip().lower()
        for item in (content_encoding or "").split(",")
        if item.strip() and item.strip().lower() != "identity"
    ]
    decoded = raw
    for coding in reversed(codings):
        if coding in ("gzip", "x-gzip"):
            try:
                decoded = gzip.decompress(decoded)
            except (OSError, EOFError) as exc:
                raise _RequestBodyError(
                    "Invalid gzip request body", raw_bytes=len(raw)
                ) from exc
        elif coding == "deflate":
            try:
                decoded = zlib.decompress(decoded)
            except zlib.error:
                try:
                    # A few clients send a raw DEFLATE stream without zlib
                    # framing. Accept it when the standard form fails.
                    decoded = zlib.decompress(decoded, -zlib.MAX_WBITS)
                except zlib.error as exc:
                    raise _RequestBodyError(
                        "Invalid deflate request body", raw_bytes=len(raw)
                    ) from exc
        else:
            raise _RequestBodyError(
                f"Unsupported request Content-Encoding: {coding}",
                raw_bytes=len(raw),
            )
    return decoded


async def _read_json_body(
    request: Request,
    *,
    endpoint: str = "",
    trace_id: str = "",
) -> tuple[dict[str, Any], int]:
    """Read and normalize one incoming JSON body without losing client data.

    The raw byte count is returned for diagnostics. We deliberately do not log
    body contents or token-bearing fields. A small compatibility allowance for
    double-encoded JSON helps clients that pass a serialized request through a
    generic transport wrapper.
    """

    try:
        raw = await request.body()
    except Exception as exc:
        # A client that disconnects while still uploading must never look like
        # an accepted empty task. Report it as a request-body failure, before
        # any upstream route is selected.
        raise _RequestBodyError(
            "Request body could not be read before the upload completed"
        ) from exc
    raw_size = len(raw)
    content_encoding = request.headers.get("content-encoding", "")
    decoded = _decode_request_content(raw, content_encoding)
    if not decoded.strip():
        raise _RequestBodyError("Request body is empty", raw_bytes=raw_size)
    charset = _request_body_charset(request.headers.get("content-type", ""))
    try:
        text = decoded.decode(charset).lstrip("\ufeff").strip()
    except UnicodeDecodeError as exc:
        raise _RequestBodyError(
            f"Request body is not valid {charset} JSON", raw_bytes=raw_size
        ) from exc
    try:
        payload: Any = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _RequestBodyError(
            "Invalid JSON body", raw_bytes=raw_size
        ) from exc
    if isinstance(payload, str):
        nested = payload.lstrip("\ufeff").strip()
        if nested.startswith("{"):
            try:
                payload = json.loads(nested)
            except json.JSONDecodeError as exc:
                raise _RequestBodyError(
                    "Invalid nested JSON body", raw_bytes=raw_size
                ) from exc
    if not isinstance(payload, Mapping):
        raise _RequestBodyError(
            "JSON body must be an object", raw_bytes=raw_size
        )
    body = dict(payload)
    logger.info(
        "request body received id=%s endpoint=%s bytes=%d decoded_bytes=%d content_type=%s "
        "content_encoding=%s keys=%s",
        trace_id or "-",
        endpoint or request.url.path,
        raw_size,
        len(decoded),
        request.headers.get("content-type", ""),
        content_encoding or "identity",
        ",".join(sorted(str(key) for key in body.keys())),
    )
    return body, raw_size


def _message_content_fallback(message: Mapping[str, Any]) -> Any:
    """Extract common non-OpenAI aliases used by zcode/OpenCode adapters."""

    for key in ("parts", "text", "prompt", "message", "input"):
        value = message.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _normalize_chat_messages(value: Any) -> list[dict[str, Any]]:
    """Coerce compatible chat message wrappers while preserving tool fields."""

    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("["):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                value = [value]
        else:
            value = [value]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            normalized.append({"role": "user", "content": item})
            continue
        if not isinstance(item, Mapping):
            continue
        message = dict(item)
        role = str(message.get("role") or message.get("speaker") or "user")
        if role not in ("system", "developer", "user", "assistant", "tool", "function"):
            role = "user"
        message["role"] = role
        if (
            message.get("content") in (None, "", [])
            and not message.get("tool_calls")
        ):
            fallback = _message_content_fallback(message)
            if fallback is not None:
                message["content"] = fallback
        normalized.append(message)
    return cli_client.repair_tool_call_history(normalized)


def _json_loads_safe(value: str) -> dict:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


async def _parse_oauth_params(query: dict) -> dict:
    """解析 Trae 授权回调参数。

    Trae 网页授权页实际会走两套流程：
      1. 新流程 (code_challenge): callback 会带 authCodeInfo / code 等参数
      2. 老流程 (refreshToken): callback 直接带 refreshToken=xxx
    对于老流程，本地不会拥有 Cloud-IDE-JWT，需要拿到 refreshToken 后通过
    oauth/ExchangeToken 向 api.trae.cn 兑换 Cloud-IDE-JWT。
    """
    user_jwt = _json_loads_safe(query.get("userJwt", ""))
    token = user_jwt.get("Token") or user_jwt.get("token") or ""
    refresh = user_jwt.get("RefreshToken") or user_jwt.get("refreshToken") or query.get("refreshToken") or query.get("data") or ""
    token_exp = user_jwt.get("TokenExpireAt") or user_jwt.get("tokenExpireAt") or ""
    refresh_exp = user_jwt.get("RefreshExpireAt") or user_jwt.get("refreshExpireAt") or query.get("refreshExpireAt") or ""

    user_info = _json_loads_safe(query.get("userInfo", ""))
    user_id = user_info.get("UserID") or user_info.get("userId") or user_info.get("userID") or query.get("userId") or ""
    region = user_info.get("Region") or user_info.get("region") or query.get("region") or "CN"
    ai_region = user_info.get("AIRegion") or user_info.get("aiRegion") or region
    client_id = user_jwt.get("ClientID") or user_jwt.get("clientId") or query.get("clientID") or query.get("clientId") or query.get("client_id") or ""
    host = query.get("host") or user_info.get("Host") or user_info.get("host") or ""

    if not token and refresh:
        exchange = await _exchange_refresh_token(
            refresh_token=refresh,
            client_id=client_id or TRAE_CLIENT_ID,
            host=host,
        )
        if exchange.get("token"):
            token = exchange["token"]
            refresh = exchange.get("refresh_token") or refresh
            token_exp = exchange.get("expired_at") or token_exp
            refresh_exp = exchange.get("refresh_expired_at") or refresh_exp
            client_id = exchange.get("client_id") or client_id
            host = exchange.get("host") or host
            user_id = exchange.get("user_id") or user_id
            user_info = exchange.get("user_info") or user_info
            region = exchange.get("region") or region
            ai_region = exchange.get("ai_region") or region

    if not token:
        return {}

    uid = user_id or ""
    return {
        "token": token,
        "refresh_token": refresh,
        "user_id": uid,
        "tenant_id": user_info.get("TenantID") or user_info.get("tenantId") or "",
        "region": region,
        "ai_region": ai_region,
        "host": host,
        "expired_at": str(token_exp) if token_exp else "",
        "refresh_expired_at": str(refresh_exp) if refresh_exp else "",
        "client_id": client_id,
        "web_id": user_info.get("WebId") or user_info.get("webId") or uid,
        "biz_user_id": user_info.get("BizUserId") or user_info.get("bizUserId") or uid,
        "user_unique_id": user_info.get("UserUniqueId") or user_info.get("userUniqueId") or uid,
        "scope": query.get("scope") or user_info.get("Scope") or user_info.get("scope") or "",
        "tenant": user_info.get("Tenant") or user_info.get("tenant") or "",
        "app_language": user_info.get("AppLanguage") or user_info.get("appLanguage") or "",
        "user_region": query.get("userRegion") or user_info.get("UserRegion") or user_info.get("userRegion") or "",
        "user_identity": user_info.get("UserIdentity") or user_info.get("userIdentity") or "",
        "screen_name": user_info.get("ScreenName") or user_info.get("screenName") or "",
    }

async def _exchange_refresh_token(refresh_token: str, client_id: str, host: str = "") -> dict:
    """使用 refreshToken 向 Trae CN 兑换 Cloud-IDE-JWT。

    与 Trae 官网实现一致:
      POST https://api.trae.cn/cloudide/api/v3/trae/oauth/ExchangeToken
      {"ClientID":..., "RefreshToken":..., "ClientSecret":"-", "UserID":""}
    """
    if not refresh_token:
        return {}
    base = host or "https://api.trae.cn"
    base = base.rstrip("/")
    url = base + "/cloudide/api/v3/trae/oauth/ExchangeToken"
    payload = {
        "ClientID": client_id,
        "RefreshToken": refresh_token,
        "ClientSecret": "-",
        "UserID": "",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload)
            body = resp.text
            status = resp.status_code
    except Exception as e:
        logger.warning("ExchangeToken failed: %s", e)
        return {}

    if status != 200:
        logger.warning("ExchangeToken HTTP %s: %s", status, body[:500])
        return {}

    result = _json_loads_safe(body)
    data = result.get("Result") or result.get("result") or result
    if not isinstance(data, dict):
        logger.warning("ExchangeToken unexpected result: %s", body[:500])
        return {}

    token = data.get("Token") or data.get("token") or data.get("AccessToken") or data.get("accessToken") or ""
    if not token:
        logger.warning("ExchangeToken missing Token: %s", body[:500])
        return {}

    refresh2 = data.get("RefreshToken") or data.get("refreshToken") or refresh_token
    user_info = _json_loads_safe(str(data.get("UserInfo") or data.get("userInfo") or ""))
    return {
        "token": token,
        "refresh_token": refresh2,
        "expired_at": str(data.get("TokenExpireAt") or data.get("tokenExpireAt") or ""),
        "refresh_expired_at": str(data.get("RefreshExpireAt") or data.get("refreshExpireAt") or ""),
        "client_id": data.get("ClientID") or data.get("clientId") or client_id,
        "host": host or base,
        "user_id": user_info.get("UserID") or user_info.get("userId") or data.get("UserID") or data.get("userId") or "",
        "user_info": user_info,
        "region": user_info.get("Region") or user_info.get("region") or data.get("Region") or data.get("region") or "CN",
        "ai_region": user_info.get("AIRegion") or user_info.get("aiRegion") or data.get("AIRegion") or data.get("aiRegion") or "",
    }


def _apply_parsed_creds(p: dict) -> None:
    """将解析后的凭证写入 auth 状态。"""
    auth.apply_oauth_callback(
        token=p.get("token", ""),
        refresh_token=p.get("refresh_token", ""),
        user_id=p.get("user_id", ""),
        tenant_id=p.get("tenant_id", ""),
        region=p.get("region", ""),
        ai_region=p.get("ai_region", ""),
        host=p.get("host", ""),
        expired_at=p.get("expired_at", ""),
        refresh_expired_at=p.get("refresh_expired_at", ""),
        client_id=p.get("client_id", ""),
        web_id=p.get("web_id", ""),
        biz_user_id=p.get("biz_user_id", ""),
        user_unique_id=p.get("user_unique_id", ""),
        scope=p.get("scope", ""),
        tenant=p.get("tenant", ""),
        app_language=p.get("app_language", ""),
        user_region=p.get("user_region", ""),
        user_identity=p.get("user_identity", ""),
        screen_name=p.get("screen_name", ""),
    )


APP_VERSION = "1.0.10"

# Inline Lucide icons (ISC license) so the intranet console has no CDN dependency.
_LUCIDE_ICONS = {
    'zap': '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    'users': '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    'receipt': '<path d="M4 2v20l2-1 2 1 2-1 2 1 2-1 2 1 2-1 2 1V2l-2 1-2-1-2 1-2-1-2 1-2-1-2 1Z"/><path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8"/><path d="M12 17.5v-11"/>',
    'sliders-horizontal': '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    'activity': '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    'calendar-check': '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="m9 16 2 2 4-4"/>',
    'arrow-left-right': '<path d="M8 3 4 7l4 4"/><path d="M4 7h16"/><path d="m16 21 4-4-4-4"/><path d="M20 17H4"/>',
    'trash-2': '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
    'log-out': '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/>',
    'log-in': '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" x2="3" y1="12" y2="12"/>',
    'refresh-cw': '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    'coins': '<circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/><path d="M7 6h1v4"/><path d="m16.71 13.88.7.71-2.82 2.82"/>',
    'check-check': '<path d="M18 6 7 17l-5-5"/><path d="m22 10-7.5 7.5L13 16"/>',
    'shield-check': '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    'download': '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    'external-link': '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    'play': '<polygon points="6 3 20 12 6 21 6 3"/>',
    'save': '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
    'search': '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    'list': '<line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/><line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/><line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/>',
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    'moon': '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    'list-plus': '<path d="M11 12H3"/><path d="M16 6H3"/><path d="M16 18H3"/><path d="M18 9v6"/><path d="M21 12h-6"/>',
    'layout-dashboard': '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    'database': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    'arrow-down-to-line': '<path d="M12 17V3"/><path d="m6 11 6 6 6-6"/><path d="M19 21H5"/>',
    'arrow-up-from-line': '<path d="m18 9-6-6-6 6"/><path d="M12 3v14"/><path d="M5 21h14"/>',
    'sigma': '<path d="M18 7V5a1 1 0 0 0-1-1H6.5a.5.5 0 0 0-.4.8l4.5 6a2 2 0 0 1 0 2.4l-4.5 6a.5.5 0 0 0 .4.8H17a1 1 0 0 0 1-1v-2"/>',
    'calendar': '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    'server': '<rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/>',
    'lock': '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    'unlock': '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/>',
    'route': '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
}


def _icon(name: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{_LUCIDE_ICONS[name]}</svg>'
    )

_CONSOLE_CSS = """
:root, [data-theme="light"] {
  color-scheme: light;
  --bg: #f2f4f2;
  --surface: #ffffff;
  --surface-2: #f6f8f6;
  --frame: #1a201d;
  --line: #dde3df;
  --line-strong: #bcc6c0;
  --text: #111613;
  --muted: #4f5b55;
  --faint: #7d8983;
  --accent: #15803d;
  --accent-hover: #116a32;
  --accent-soft: #e7f5ec;
  --accent-line: #9fd5b2;
  --on-accent: #ffffff;
  --edge: #16a34a;
  --ring: rgba(22,163,74,.22);
  --row-hover: #f7faf8;
  --row-active: #eef8f1;
  --topbar: rgba(242,244,242,.92);
  --shadow: 0 10px 28px rgba(10,20,14,.14);
  --side-bg: #0e1311;
  --side-line: #1f2a24;
  --side-text: #e4ece7;
  --side-muted: #8fa198;
  --side-hover: #18211c;
  --side-active: #13301f;
  --side-accent: #4ade80;
  --ok: #15803d;
  --ok-soft: #e7f5ec;
  --ok-line: #9fd5b2;
  --warn: #a35200;
  --warn-soft: #fdf1e1;
  --warn-line: #efcf9f;
  --danger: #c0262d;
  --danger-soft: #fdebec;
  --danger-line: #f0bcbf;
  --knob: #ffffff;
  --chart-in: #16a34a;
  --chart-cache: #0b5e2e;
  --chart-out: #2563eb;
  --mono: "JetBrains Mono", "SF Mono", Consolas, "Courier New", monospace;
  --radius: 8px;
  --radius-sm: 6px;
}
[data-theme="dark"] {
  color-scheme: dark;
  --bg: #090c0b;
  --surface: #101513;
  --surface-2: #151c18;
  --frame: #1f6b40;
  --line: #222d27;
  --line-strong: #33433a;
  --text: #e3ece6;
  --muted: #9aaca2;
  --faint: #6c7d74;
  --accent: #22c55e;
  --accent-hover: #4ade80;
  --accent-soft: #10251a;
  --accent-line: #1f6b40;
  --on-accent: #04130a;
  --edge: #22c55e;
  --ring: rgba(34,197,94,.28);
  --row-hover: #131a16;
  --row-active: #102419;
  --topbar: rgba(9,12,11,.9);
  --shadow: 0 12px 32px rgba(0,0,0,.55);
  --side-bg: #060908;
  --side-line: #1a2620;
  --side-text: #e3ece6;
  --side-muted: #7f938a;
  --side-hover: #111a15;
  --side-active: #0f2a1b;
  --side-accent: #4ade80;
  --ok: #4ade80;
  --ok-soft: #0f2418;
  --ok-line: #1f6b40;
  --warn: #f5a524;
  --warn-soft: #2a1d08;
  --warn-line: #5c3f10;
  --danger: #f87171;
  --danger-soft: #2a1012;
  --danger-line: #5e2428;
  --knob: #e3ece6;
  --chart-in: #22c55e;
  --chart-cache: #137a3c;
  --chart-out: #60a5fa;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
[hidden] { display:none !important; }
html { -webkit-text-size-adjust: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Roboto, sans-serif;
  background: var(--bg); color: var(--text); min-height: 100vh;
  font-size: 14px; line-height: 1.5; letter-spacing: 0;
  transition: background .2s, color .2s;
}
code, .mono { font-family: var(--mono); font-size: 12px; }
svg { width: 16px; height: 16px; flex-shrink: 0; }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--edge); outline-offset: 2px; }
::selection { background: var(--edge); color: var(--on-accent); }

/* Shell: black sidebar with a green edge */
.app { display: grid; grid-template-columns: 224px minmax(0, 1fr); min-height: 100vh; }
.sidebar {
  background: var(--side-bg); color: var(--side-text);
  border-right: 2px solid var(--edge);
  display: flex; flex-direction: column; position: sticky; top: 0; height: 100vh; overflow-y: auto;
}
.brand { display: flex; align-items: center; gap: 10px; padding: 16px 16px 14px; border-bottom: 1px solid var(--side-line); }
.brand-mark {
  width: 34px; height: 34px; flex-shrink: 0;
  display: grid; place-items: center;
}
.brand-mark svg { width: 34px; height: 34px; display: block; }
.brand-block { display: flex; flex-direction: column; min-width: 0; }
.brand-block h1 { font-size: 15px; font-weight: 650; white-space: nowrap; color: var(--side-text); }
.brand-sub { color: var(--side-muted); font-size: 11px; font-family: var(--mono); }
.nav-list { display: flex; flex-direction: column; gap: 4px; padding: 12px 10px; }
.nav-item {
  display: flex; align-items: center; gap: 10px; width: 100%; height: 38px;
  border: 1px solid transparent; background: transparent; color: var(--side-muted);
  font: inherit; font-size: 13px; font-weight: 550; padding: 0 12px;
  border-radius: var(--radius-sm); cursor: pointer; text-align: left; position: relative;
  transition: background .12s, color .12s, border-color .12s;
}
.nav-item:hover { background: var(--side-hover); color: var(--side-text); }
.nav-item.active { background: var(--side-active); color: var(--side-accent); border-color: rgba(74,222,128,.45); }
.nav-item.active::before {
  content: ""; position: absolute; left: -11px; top: 7px; bottom: 7px; width: 3px;
  border-radius: 0 3px 3px 0; background: var(--side-accent);
}
.sidebar-foot { margin-top: auto; padding: 12px 16px 16px; border-top: 1px solid var(--side-line); font-size: 11px; color: var(--side-muted); display: flex; justify-content: space-between; gap: 8px; font-family: var(--mono); }

.main { min-width: 0; display: flex; flex-direction: column; }
.topbar {
  position: sticky; top: 0; z-index: 5; background: var(--topbar);
  backdrop-filter: blur(6px); border-bottom: 1px solid var(--frame);
  display: flex; align-items: center; justify-content: space-between; gap: 12px 20px; flex-wrap: wrap;
  padding: 12px 28px;
}
.topbar h2 { font-size: 17px; font-weight: 650; display: flex; align-items: center; gap: 10px; }
.topbar h2::before { content: ""; width: 4px; height: 18px; border-radius: 2px; background: var(--edge); }
.topbar-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.status-chips { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 9px;
  border: 1px solid var(--line-strong); background: var(--surface); border-radius: 999px;
  font-size: 12px; color: var(--muted); white-space: nowrap; max-width: 260px;
}
.chip code { font-size: 11px; color: var(--text); overflow: hidden; text-overflow: ellipsis; }
.chip-key { color: var(--faint); }
.chip-user { color: var(--accent); border-color: var(--accent-line); background: var(--accent-soft); overflow: hidden; text-overflow: ellipsis; }

/* Night mode switch */
.theme-switch {
  display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 9px;
  border: 1px solid var(--frame); border-radius: 999px; background: var(--surface); cursor: pointer;
}
.theme-icon { display: inline-grid; place-items: center; color: var(--faint); transition: color .15s; }
.theme-icon svg { width: 14px; height: 14px; }
[data-theme="light"] .theme-sun { color: #c27c00; }
[data-theme="dark"] .theme-moon { color: var(--accent); }

.content { padding: 22px 28px 48px; width: 100%; max-width: 1240px; }

/* Pages and framed sections */
.tab-page { display: none; }
.tab-page.active { display: block; animation: page-in .24s cubic-bezier(.22,1,.36,1); }
@keyframes page-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.panel-stack { display: flex; flex-direction: column; gap: 16px; }
.panel-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }
.panel-grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }
.panel-card {
  background: var(--surface); border: 1px solid var(--frame); border-radius: var(--radius);
  box-shadow: inset 0 3px 0 var(--edge);
  padding: 18px 18px 16px; min-width: 0;
}
.span-2 { grid-column: 1 / -1; }
.model-list-grid {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;
  align-items: start;
}
.model-column {
  min-width: 0; overflow: hidden;
  border: 1px solid var(--line-strong); border-radius: var(--radius-sm);
  background: var(--surface-2);
}
.model-column-head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  padding: 10px 12px; border-bottom: 1px solid var(--line-strong);
  background: var(--surface); color: var(--text);
}
.model-column-title { font-size: 13px; font-weight: 650; }
.model-column-count {
  color: var(--faint); font-family: var(--mono); font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.model-column-body {
  display: flex; flex-direction: column; gap: 6px;
  max-height: 520px; overflow: auto; padding: 10px;
}
.model-item {
  display: flex; flex-direction: column; gap: 2px; min-width: 0;
  padding: 7px 9px; border: 1px solid var(--line); border-radius: var(--radius-sm);
  background: var(--surface); color: var(--text); font-family: var(--mono); font-size: 12px;
  line-height: 1.4; overflow-wrap: anywhere;
}
.model-item:hover { border-color: var(--edge); background: var(--accent-soft); }
.model-item-name { overflow-wrap: anywhere; }
.model-item-alias { color: var(--muted); font-family: inherit; font-size: 10px; }
.model-column-empty { padding: 18px 10px; color: var(--faint); text-align: center; font-size: 12px; }
.model-fallback { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }
.model-fallback-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px; }
.model-fallback-title { color: var(--muted); font-size: 12px; font-weight: 650; }
.model-fallback-note { color: var(--faint); font-size: 11px; }
.model-fallback-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 6px; }
.section-head { display: flex; align-items: center; justify-content: space-between; gap: 8px 12px; flex-wrap: wrap; margin-bottom: 12px; }
.section-head > div { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; min-width: 0; }
.section-title { font-size: 14px; font-weight: 650; color: var(--text); }
.section-meta { color: var(--faint); font-size: 12px; }
.section-desc { color: var(--muted); font-size: 12px; margin: -6px 0 12px; }
.state-pill {
  display: inline-flex; align-items: center; gap: 6px; height: 22px; padding: 0 9px; border-radius: 999px;
  font-size: 12px; color: var(--muted); background: var(--surface-2); border: 1px solid var(--line-strong); white-space: nowrap;
}
.state-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--line-strong); }
.state-pill.on { color: var(--ok); background: var(--ok-soft); border-color: var(--ok-line); }
.state-pill.on::before { background: var(--ok); box-shadow: 0 0 0 3px var(--ring); }
.state-pill.lock { color: var(--warn); background: var(--warn-soft); border-color: var(--warn-line); }
.state-pill.lock::before { background: var(--warn); }

/* Overview stats */
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.stat {
  background: var(--surface); border: 1px solid var(--frame); border-left: 4px solid var(--edge);
  border-radius: var(--radius); padding: 12px 14px; display: flex; flex-direction: column; gap: 2px; min-width: 0;
}
.stat dt { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.stat dt svg { width: 14px; height: 14px; color: var(--accent); }
.stat dd { font-size: 22px; font-weight: 650; font-family: var(--mono); font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Buttons */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  height: 34px; padding: 0 14px; border-radius: var(--radius-sm);
  font: inherit; font-size: 13px; font-weight: 600; white-space: nowrap;
  cursor: pointer; border: 1px solid transparent; text-decoration: none;
  transition: background .12s, border-color .12s, color .12s, box-shadow .12s, transform .12s;
}
.btn-sm { height: 30px; padding: 0 11px; font-size: 12px; }
.btn-primary { background: var(--accent); border-color: var(--frame); color: var(--on-accent); }
.btn-primary:hover { background: var(--accent-hover); box-shadow: 0 0 0 3px var(--ring); }
.btn-secondary { background: var(--surface); border-color: var(--frame); color: var(--text); }
.btn-secondary:hover { background: var(--accent-soft); color: var(--accent); }
.btn-ghost { background: transparent; border-color: transparent; color: var(--muted); }
.btn-ghost:hover { background: var(--surface-2); color: var(--text); }
.btn-danger { background: transparent; border-color: var(--danger-line); color: var(--danger); }
.btn-danger:hover { background: var(--danger-soft); }
.btn:disabled, .icon-btn:disabled { opacity: .45; cursor: not-allowed; }
.icon-btn {
  width: 30px; height: 30px; display: inline-grid; place-items: center; flex-shrink: 0;
  border-radius: var(--radius-sm); border: 1px solid var(--line-strong); background: var(--surface); color: var(--muted);
  cursor: pointer; transition: background .12s, color .12s, border-color .12s, transform .12s;
}
.icon-btn:hover:not(:disabled) { color: var(--accent); border-color: var(--edge); background: var(--accent-soft); }
.icon-btn.danger:hover:not(:disabled) { color: var(--danger); border-color: var(--danger-line); background: var(--danger-soft); }
.btn-group { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.toolbar .spacer { flex: 1 1 auto; }
.busy { opacity: .6; pointer-events: none; }

/* Badges */
.badge {
  display: inline-flex; align-items: center; height: 22px; padding: 0 8px; border-radius: 999px;
  font-size: 12px; font-weight: 550; white-space: nowrap; border: 1px solid transparent;
}
.badge-ok { background: var(--ok-soft); color: var(--ok); border-color: var(--ok-line); }
.badge-expired { background: var(--warn-soft); color: var(--warn); border-color: var(--warn-line); }
.badge-none { background: var(--surface-2); color: var(--muted); border-color: var(--line-strong); }
.badge-active { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-line); }

/* Forms */
.form-group { margin-bottom: 12px; min-width: 0; }
.form-group:last-child { margin-bottom: 0; }
.form-group > label, .field-label { display: block; font-size: 12px; font-weight: 550; color: var(--muted); margin-bottom: 5px; }
.form-group input, .form-group textarea, .form-group select, .control {
  width: 100%; height: 34px; padding: 0 10px; border-radius: var(--radius-sm); border: 1px solid var(--line-strong);
  background: var(--surface-2); color: var(--text); font: inherit; font-size: 13px;
}
.form-group textarea { height: auto; min-height: 64px; padding: 8px 10px; resize: vertical; font-family: var(--mono); font-size: 12px; }
.form-group input.mono-input { font-family: var(--mono); font-size: 12px; }
.form-group input:focus, .form-group textarea:focus, .form-group select:focus, .control:focus {
  outline: none; border-color: var(--edge); box-shadow: 0 0 0 3px var(--ring);
}
.form-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.field-note { font-size: 12px; color: var(--faint); margin-top: 6px; }
.required { color: var(--danger); }

/* Auto-route lock: while auto routing is on the manual endpoint fields are
   read-only, so the effective upstream cannot contradict the switch. */
.endpoint-locked { position: relative; }
.endpoint-locked .form-group > label { color: var(--faint); }
.form-group input:disabled, .form-group select:disabled, .form-group textarea:disabled {
  background: var(--surface-2); color: var(--faint); border-color: var(--line);
  cursor: not-allowed; opacity: .72; -webkit-text-fill-color: var(--faint);
}
.endpoint-locked .form-group input:disabled, .endpoint-locked .form-group select:disabled { border-style: dashed; }
.lock-note {
  display: flex; align-items: flex-start; gap: 8px; margin: 2px 0 12px;
  padding: 10px 12px; border: 1px dashed var(--warn-line); border-radius: var(--radius-sm);
  background: var(--warn-soft); color: var(--warn); font-size: 12px; line-height: 1.5;
}
.lock-note svg { width: 15px; height: 15px; margin-top: 1px; }
.lock-note strong { font-weight: 650; }

/* Switch */
.switch-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 10px 0; }
.switch-row + .switch-row { border-top: 1px solid var(--line); }
.switch-text { min-width: 0; }
.switch-text strong { display: block; font-size: 13px; font-weight: 600; }
.switch-text span { display: block; font-size: 12px; color: var(--muted); }
.switch { position: relative; display: inline-flex; width: 36px; height: 20px; flex-shrink: 0; cursor: pointer; }
.switch input { position: absolute; inset: 0; opacity: 0; cursor: pointer; margin: 0; z-index: 1; }
.switch .track { width: 100%; height: 100%; border-radius: 999px; background: var(--line-strong); border: 1px solid var(--frame); transition: background .15s; }
.switch .track::after {
  content: ""; position: absolute; top: 3px; left: 3px; width: 14px; height: 14px; border-radius: 50%;
  background: var(--knob); box-shadow: 0 1px 2px rgba(0,0,0,.3); transition: transform .15s;
}
.switch input:checked + .track { background: var(--accent); }
.switch input:checked + .track::after { transform: translateX(16px); background: var(--on-accent); }
.switch input:focus-visible + .track { outline: 2px solid var(--edge); outline-offset: 2px; }

/* Segmented control */
.segmented { display: inline-flex; padding: 2px; border-radius: var(--radius-sm); background: var(--surface-2); border: 1px solid var(--frame); }
.segmented label { position: relative; cursor: pointer; }
.segmented input { position: absolute; opacity: 0; inset: 0; margin: 0; cursor: pointer; }
.segmented span { display: inline-flex; align-items: center; height: 28px; padding: 0 12px; border-radius: 5px; font-size: 12px; font-weight: 550; color: var(--muted); white-space: nowrap; }
.segmented input:checked + span { background: var(--accent); color: var(--on-accent); }
.segmented input:focus-visible + span { outline: 2px solid var(--edge); }

/* Check chips */
.option-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.inline-check {
  display: inline-flex; align-items: center; gap: 6px; height: 32px; padding: 0 10px;
  border: 1px solid var(--line-strong); border-radius: var(--radius-sm); background: var(--surface);
  font-size: 12px; color: var(--muted); cursor: pointer; white-space: nowrap;
}
.inline-check:has(input:checked) { border-color: var(--edge); background: var(--accent-soft); color: var(--accent); }
.inline-check input[type=checkbox] { accent-color: var(--accent); width: 14px; height: 14px; }
.inline-field select, .inline-field input { height: 26px; padding: 0 6px; border: 1px solid var(--line-strong); border-radius: 5px; font: inherit; font-size: 12px; background: var(--surface-2); color: var(--text); }
.inline-field input[type=number] { width: 70px; }

/* Tables */
.table-wrap { overflow-x: auto; border: 1px solid var(--line-strong); border-radius: var(--radius); }
.acct-table, .usage-table, .conn-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.acct-table { min-width: 860px; }
.acct-table th, .acct-table td, .usage-table th, .usage-table td, .conn-table th, .conn-table td {
  text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: middle;
}
.acct-table tbody tr:last-child td, .usage-table tbody tr:last-child td, .conn-table tbody tr:last-child td { border-bottom: none; }
.acct-table th, .usage-table th, .conn-table th {
  font-size: 12px; font-weight: 600; color: var(--muted); background: var(--surface-2);
  border-bottom: 1px solid var(--line-strong);
  position: sticky; top: 0; z-index: 1; white-space: nowrap;
}
.acct-table tbody tr:hover, .usage-table tbody tr:hover { background: var(--row-hover); }
.acct-table tbody tr.active-row { background: var(--row-active); box-shadow: inset 3px 0 var(--edge); }
.acct-table tbody tr.row-failed { background: var(--danger-soft); }
.acct-table tbody tr.checkin-row-busy { opacity: .7; }
.acct-cell { min-width: 0; }
.acct-cell strong { font-weight: 600; }
.status-cell { white-space: nowrap; }
.status-cell .badge + .badge { margin-left: 4px; }
.numeric { text-align: right !important; font-variant-numeric: tabular-nums; }
.credit-value { font-family: var(--mono); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.credit-calibration {
  display: inline-flex; align-items: center; min-height: 24px; padding: 0 7px;
  border: 1px solid var(--line-strong); border-radius: 5px; color: var(--muted);
  background: var(--surface-2); font-size: 11px; white-space: nowrap;
}
.credit-calibration.ok { color: var(--ok); border-color: var(--ok-line); background: var(--ok-soft); }
.credit-calibration.projected { color: var(--accent); border-color: var(--accent-line); background: var(--accent-soft); }
.credit-calibration.stale, .credit-calibration.error { color: var(--danger); border-color: var(--danger-line); background: var(--danger-soft); }
.credit-calibration.unknown { color: var(--faint); border-style: dashed; }
.muted-cell { color: var(--faint); font-size: 12px; white-space: nowrap; font-family: var(--mono); }
.row-subtitle { display: block; color: var(--faint); font-size: 11px; margin-top: 2px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.actions-col { width: 120px; }
.row-actions { white-space: nowrap; }
.row-actions .icon-btn + .icon-btn { margin-left: 4px; }
.model-toggle-cell { min-width: 105px; white-space: nowrap; }
.model-toggle-state { display: inline-flex; vertical-align: middle; margin-right: 8px; }
.account-model-switch { vertical-align: middle; }
.switch input:disabled { cursor: wait; }
.switch input:disabled + .track { opacity: .6; }
.empty-state { padding: 28px 12px; text-align: center; color: var(--faint); font-size: 13px; border: 1px dashed var(--frame); border-radius: var(--radius); }
.usage-records-container { max-height: calc(100vh - 220px); min-height: 160px; overflow: auto; border: 1px solid var(--line-strong); border-radius: var(--radius); }
.usage-table { min-width: 1400px; table-layout: fixed; }
.usage-table th:nth-child(1) { width: 170px; }
.usage-table th:nth-child(2) { width: 130px; }
.usage-table th:nth-child(3) { width: 145px; }
.usage-table th:nth-child(4) { width: 150px; }
.usage-table th:nth-child(5) { width: 105px; }
.usage-table th:nth-child(6) { width: 105px; }
.usage-table th:nth-child(7) { width: 145px; }
.usage-table th:nth-child(8) { width: 190px; }
.usage-table th:nth-child(9) { width: 100px; }
.usage-table th:nth-child(10) { width: 90px; }
.usage-table th:nth-child(11) { width: 90px; }
.usage-table th:nth-child(12) { width: 110px; }
.usage-table td { overflow-wrap: anywhere; }
.usage-table .usage-status { white-space: nowrap; }
.usage-table .usage-meta { white-space: nowrap; }
.usage-table .usage-tool { white-space: normal; }
.usage-table .usage-meta.unknown { color: var(--faint); }
.usage-table .usage-meta .usage-arrow { color: var(--faint); padding: 0 3px; }
.usage-table .usage-meta .usage-sub { color: var(--muted); font-size: 11px; }
.usage-route-endpoint { color: var(--muted); font-size: 11px; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.usage-model-cell { min-width: 0; }
.usage-model-cell > div:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.usage-model-flags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
.usage-flag {
  display: inline-flex; align-items: center; min-height: 18px; padding: 0 5px;
  border-radius: 4px; border: 1px solid var(--line-strong); background: var(--surface-2);
  color: var(--muted); font-size: 10px; line-height: 16px; white-space: nowrap;
}
.usage-flag.max { color: var(--accent); border-color: var(--accent-line); background: var(--accent-soft); }
.usage-flag.tool { color: var(--ok); border-color: var(--ok-line); background: var(--ok-soft); }
.usage-flag.unknown { color: var(--faint); border-style: dashed; }
.usage-empty { padding: 28px 12px; color: var(--faint); text-align: center; font-size: 13px; }
.conn-wrap { margin-top: 14px; overflow-x: auto; border: 1px solid var(--line-strong); border-radius: var(--radius); }
.conn-table { min-width: 640px; font-size: 12px; }
.conn-table td { vertical-align: top; }
.conn-table td.conn-detail { color: var(--muted); overflow-wrap: anywhere; font-family: var(--mono); font-size: 11px; }
.conn-table .conn-reasoning { margin-top: 4px; white-space: pre-wrap; font-family: inherit; }
.conn-status { white-space: nowrap; }
.conn-status.ok { color: var(--ok); font-weight: 600; }
.conn-status.fail { color: var(--danger); font-weight: 600; }
.conn-status.pending { color: var(--muted); }
.conn-time { white-space: nowrap; font-variant-numeric: tabular-nums; font-family: var(--mono); }

/* Login & schedule */
.login-steps { margin: 0 0 4px; padding-left: 18px; font-size: 12px; color: var(--muted); line-height: 1.8; }
.login-steps code { background: var(--surface-2); border: 1px solid var(--line-strong); padding: 1px 5px; border-radius: 4px; font-size: 11px; color: var(--text); }
.manual-add { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px; }
.manual-add summary { font-size: 13px; color: var(--muted); cursor: pointer; font-weight: 550; }
.manual-add summary:hover { color: var(--accent); }
.manual-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0 12px; margin-top: 12px; }
.manual-grid .span-all { grid-column: 1 / -1; }
.manual-submit { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.manual-submit .msg { margin-top: 0; }
.schedule-row { display: flex; align-items: center; gap: 12px 16px; flex-wrap: wrap; }
.time-field { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
.time-field input[type=time] { width: 112px; }
.schedule-status { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; padding: 12px 14px; background: var(--surface-2); border: 1px solid var(--line); border-left: 3px solid var(--edge); border-radius: var(--radius-sm); }
.schedule-status div { min-width: 0; }
.schedule-status dt { font-size: 11px; color: var(--faint); }
.schedule-status dd { font-size: 13px; font-family: var(--mono); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.max-models { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; font-size: 12px; color: var(--muted); }
.model-chip { border: 1px solid var(--line-strong); background: var(--surface); color: var(--text); border-radius: var(--radius-sm); height: 28px; padding: 0 9px; font: inherit; font-family: var(--mono); font-size: 12px; cursor: pointer; }
.model-chip:hover { border-color: var(--edge); color: var(--accent); background: var(--accent-soft); }
pre.code-out { margin-top: 12px; padding: 12px; background: var(--side-bg); color: #b8f5cd; border: 1px solid var(--frame); border-radius: var(--radius-sm); font-family: var(--mono); font-size: 12px; max-height: 360px; overflow: auto; white-space: pre-wrap; display: none; }

/* Feedback */
.msg { margin-top: 12px; padding: 8px 12px; border-radius: var(--radius-sm); font-size: 13px; display: none; line-height: 1.45; white-space: pre-wrap; }
.msg-ok { background: var(--ok-soft); color: var(--ok); display: block; border: 1px solid var(--ok-line); }
.msg-err { background: var(--danger-soft); color: var(--danger); display: block; border: 1px solid var(--danger-line); }
.msg-pending { background: var(--warn-soft); color: var(--warn); display: block; border: 1px solid var(--warn-line); }
.inline-msg { margin-top: 0; }
.account-msgs .msg { margin-top: 10px; }
.loading { margin-top: 12px; display: none; font-size: 13px; color: var(--muted); }
.busy-indicator { display: none; color: var(--muted); font-size: 12px; align-items: center; gap: 6px; }
.busy-indicator.visible { display: inline-flex; }
.busy-indicator::before { content: ""; width: 10px; height: 10px; border: 2px solid var(--line-strong); border-top-color: var(--edge); border-radius: 50%; animation: relay-spin .7s linear infinite; }
@keyframes relay-spin { to { transform: rotate(360deg); } }
.toast { position: fixed; top: 16px; right: 16px; z-index: 30; width: min(400px, calc(100vw - 32px)); padding: 12px 14px; border: 1px solid var(--frame); border-left: 4px solid var(--ok); border-radius: var(--radius); background: var(--surface); color: var(--text); box-shadow: var(--shadow); opacity: 0; transform: translateY(-8px); pointer-events: none; transition: opacity .18s, transform .18s; white-space: pre-wrap; line-height: 1.45; font-size: 13px; }
.toast.visible { opacity: 1; transform: translateY(0); }
.toast.error { border-left-color: var(--danger); }
.toast-title { display: block; font-size: 12px; font-weight: 650; margin-bottom: 2px; color: var(--ok); }
.toast.error .toast-title { color: var(--danger); }

/* Overview */
.stats.stats-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.stat dd.stat-sub { font-size: 11px; font-weight: 400; color: var(--faint); font-family: inherit; }
.ov-grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; align-items: start; }
.chart-readout { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; min-height: 40px; padding: 8px 12px; margin-bottom: 12px; background: var(--surface-2); border: 1px solid var(--line); border-left: 3px solid var(--edge); border-radius: var(--radius-sm); font-size: 12px; color: var(--muted); }
.chart-readout strong { color: var(--text); font-family: var(--mono); font-weight: 650; }
.readout-date { color: var(--text); font-weight: 650; font-family: var(--mono); min-width: 84px; }
.readout-item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.sw-in { background: var(--chart-in); } .sw-cache { background: var(--chart-cache); } .sw-out { background: var(--chart-out); }
.chart-wrap { display: grid; grid-template-columns: 52px minmax(0, 1fr); column-gap: 8px; }
.chart-y { display: flex; flex-direction: column; justify-content: space-between; height: 220px; font-size: 10px; color: var(--faint); font-family: var(--mono); text-align: right; }
.chart-y span { line-height: 1; transform: translateY(-50%); }
.chart-y span:last-child { transform: none; }
.chart {
  height: 220px; display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); gap: 3px; align-items: end;
  border-bottom: 1px solid var(--line-strong);
  background-image: linear-gradient(to bottom, var(--line) 1px, transparent 1px); background-size: 100% 25%;
}
.bar-col { height: 100%; display: flex; justify-content: center; cursor: pointer; border-radius: 3px 3px 0 0; outline: none; transition: background .15s; }
.bar-col:hover, .bar-col:focus-visible { background: var(--accent-soft); }
.bar-col.sel { background: var(--row-active); box-shadow: inset 0 -2px var(--edge); }
.bar-col:focus-visible { box-shadow: inset 0 0 0 2px var(--edge); }
.bar-stack { width: min(70%, 22px); height: 100%; display: flex; flex-direction: column; justify-content: flex-end; }
.bar-seg { display: block; width: 100%; height: 0; transition: height .55s var(--ease-out); }
.seg-out { background: var(--chart-out); border-radius: 3px 3px 0 0; }
.seg-in { background: var(--chart-in); }
.seg-cache { background: var(--chart-cache); }
.chart-axis { grid-column: 2; display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); gap: 3px; margin-top: 6px; font-size: 10px; color: var(--faint); font-family: var(--mono); }
.chart-axis span { text-align: center; white-space: nowrap; overflow: visible; }
.chart-empty { grid-column: 2; padding: 6px 0 0; font-size: 12px; color: var(--faint); }
.kv-list { display: flex; flex-direction: column; }
.kv-list div { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); min-width: 0; }
.kv-list div:last-child { border-bottom: none; }
.kv-list dt { font-size: 12px; color: var(--muted); display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.kv-list dd { font-size: 12px; font-family: var(--mono); color: var(--text); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.day-wrap { max-height: 440px; overflow: auto; border: 1px solid var(--line-strong); border-radius: var(--radius); }
.day-table { width: 100%; min-width: 680px; border-collapse: collapse; font-size: 12px; }
.day-table th, .day-table td { padding: 8px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.day-table th:first-child, .day-table td:first-child { text-align: left; }
.day-table th { font-weight: 600; color: var(--muted); background: var(--surface-2); border-bottom: 1px solid var(--line-strong); position: sticky; top: 0; z-index: 1; }
.day-table td { font-family: var(--mono); }
.day-table tbody tr { cursor: pointer; transition: background .15s, box-shadow .25s var(--ease-out); }
.day-table tbody tr:hover { background: var(--row-hover); }
.day-table tbody tr.sel { background: var(--row-active); box-shadow: inset 3px 0 var(--edge); }
.day-table tbody tr.zero td { color: var(--faint); }
.day-table tfoot td { font-weight: 650; background: var(--surface-2); border-top: 1px solid var(--line-strong); border-bottom: none; position: sticky; bottom: 0; }
.day-table .badge { margin-left: 6px; font-family: inherit; }
@media (max-width: 1280px) { .stats.stats-6 { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 1080px) { .ov-grid { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 820px) {
  .stats.stats-6 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .chart, .chart-y { height: 180px; }
  .chart-wrap { grid-template-columns: 40px minmax(0, 1fr); column-gap: 6px; }
  .chart, .chart-axis { gap: 2px; }
}

/* Selection motion */
:root { --ease-out: cubic-bezier(.22,1,.36,1); --ease-spring: cubic-bezier(.34,1.56,.64,1); }
.nav-list { position: relative; }
.nav-item { z-index: 1; }
.nav-item svg { transition: transform .28s var(--ease-spring); }
.nav-item:hover svg { transform: translateX(2px); }
.nav-item.active svg { transform: scale(1.12); }
.nav-indicator {
  position: absolute; left: 0; top: 0; width: 0; height: 0; z-index: 0; opacity: 0; pointer-events: none;
  border-radius: var(--radius-sm); background: var(--side-active); border: 1px solid rgba(74,222,128,.45);
}
.nav-indicator::before {
  content: ""; position: absolute; left: -11px; top: 7px; bottom: 7px; width: 3px;
  border-radius: 0 3px 3px 0; background: var(--side-accent); box-shadow: 0 0 10px rgba(74,222,128,.55);
}
.nav-indicator.ready { transition: transform .34s var(--ease-out), width .34s var(--ease-out), height .34s var(--ease-out), opacity .2s; }
.has-indicator .nav-item.active, .has-indicator .nav-item.active:hover { background: transparent; border-color: transparent; }
.has-indicator .nav-item.active::before { display: none; }

.segmented { position: relative; }
.segmented label { z-index: 1; }
.segmented span { transition: color .2s, transform .15s var(--ease-spring); }
.segmented label:active span { transform: scale(.95); }
.seg-thumb {
  position: absolute; left: 0; top: 0; width: 0; height: 0; z-index: 0; opacity: 0; pointer-events: none;
  border-radius: 5px; background: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,.25);
}
.seg-thumb.ready { transition: transform .32s var(--ease-out), width .32s var(--ease-out), opacity .2s; }
.has-thumb input:checked + span { background: transparent; }

.switch .track { transition: background .2s, box-shadow .2s; }
.switch .track::after { transition: transform .3s var(--ease-spring), width .16s ease, background .2s; }
.switch:active .track::after { width: 18px; }
.switch input:checked:active + .track::after { transform: translateX(12px); }
.switch input:checked + .track { box-shadow: 0 0 0 3px var(--ring); }

.inline-check { transition: border-color .15s, background .15s, color .15s, transform .12s; }
.inline-check:active { transform: scale(.97); }
.inline-check:has(input:checked) { animation: sel-pop .32s var(--ease-spring); }
@keyframes sel-pop { 0% { transform: scale(.95); } 60% { transform: scale(1.04); } 100% { transform: scale(1); } }

.model-chip { transition: border-color .12s, color .12s, background .12s, transform .12s; }
.btn:active:not(:disabled), .icon-btn:active:not(:disabled), .model-chip:active { transform: scale(.96); }

.acct-table tbody tr { transition: background .2s, box-shadow .25s var(--ease-out); }
.acct-table tbody tr.active-row { animation: row-select .6s var(--ease-out); }
@keyframes row-select { 0% { background: var(--accent-soft); box-shadow: inset 0 0 var(--edge); } 100% { box-shadow: inset 3px 0 var(--edge); } }

.theme-icon svg { transition: transform .45s var(--ease-spring); transform: scale(.85); }
[data-theme="light"] .theme-sun svg { transform: rotate(180deg) scale(1.1); }
[data-theme="dark"] .theme-moon svg { transform: rotate(-20deg) scale(1.1); }
html.theme-anim *, html.theme-anim *::before, html.theme-anim *::after {
  transition: background-color .3s ease, border-color .3s ease, color .3s ease, box-shadow .3s ease, transform .3s var(--ease-spring) !important;
}
@media (max-width: 820px) {
  .nav-indicator::before { left: 10px; right: 10px; top: auto; bottom: 3px; width: auto; height: 2px; border-radius: 2px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
@media (max-width: 1080px) {
  .panel-grid.cols-2 { grid-template-columns: minmax(0, 1fr); }
  .manual-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 820px) {
  .app { grid-template-columns: minmax(0, 1fr); }
  .sidebar { position: sticky; top: 0; z-index: 10; height: auto; flex-direction: row; align-items: center; border-right: none; border-bottom: 2px solid var(--edge); overflow: visible; }
  .brand { padding: 10px 12px; border-bottom: none; }
  .brand-block { display: none; }
  .nav-list { flex-direction: row; overflow-x: auto; padding: 8px 8px 8px 0; flex: 1; scrollbar-width: none; }
  .nav-item { width: auto; white-space: nowrap; height: 34px; padding: 0 10px; }
  .nav-item.active::before { display: none; }
  .sidebar-foot { display: none; }
  .topbar { position: static; padding: 12px 14px; }
  .content { padding: 14px 14px 36px; }
  .panel-card { padding: 16px 14px 14px; }
  .model-list-grid { grid-template-columns: minmax(0, 1fr); }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .stat dd { font-size: 18px; }
  .manual-grid { grid-template-columns: minmax(0, 1fr); }
  .schedule-status { grid-template-columns: minmax(0, 1fr); }
  .toolbar .btn { flex: 1 1 140px; }
  .toolbar .spacer { display: none; }
}"""


def _status_badge() -> str:
    s = auth.get_auth()
    if s.token and s.is_valid():
        return '<span class="badge badge-ok">\u5df2\u767b\u5f55</span>'
    if s.token:
        return '<span class="badge badge-expired">\u5df2\u8fc7\u671f</span>'
    return '<span class="badge badge-none">\u672a\u767b\u5f55</span>'


def _web_login_html() -> str:
    s = auth.get_auth()
    client_id = TRAE_CLIENT_ID
    listener_port = LOCAL_LISTENER_PORT
    auth_url = html_mod.escape(TRAE_AUTH_URL)
    state = s
    accounts = auth.list_accounts()
    polling = auth.get_polling_status()
    settings = auth.get_settings()
    auto_route_on = _auto_route_enabled()

    # Header status chips; the active-user chip is updated in place by JS.
    upstream_label = AUTO_ROUTE_MODE if auto_route_on else _current_upstream_mode()
    status_html = f"""
    <div class="status-chips">
      {_status_badge()}
      <span id="active-user-id" class="chip chip-user"{' hidden' if not state.user_id else ''}>{f'用户: {html_mod.escape(state.user_id)}' if state.user_id else ''}</span>
      <span class="chip" title="凭据来源"><span class="chip-key">源</span><code>{html_mod.escape(state.source)}</code></span>
      <span class="chip" title="当前上游模式"><span class="chip-key">上游</span><code>{html_mod.escape(upstream_label)}</code></span>
      <span class="chip"><span class="chip-key">轮询</span><code>{'开' if polling.get('enabled') else '关'}</code></span>
    </div>"""

    # Consumption history is rendered in its own panel below the account list.
    usage_records_html = """
    <div id="usage-records-container" class="usage-records-container">
      <table class="usage-table">
        <thead>
          <tr>
            <th>时间</th>
            <th>账号</th>
            <th>产品 / 端点</th>
            <th>模型</th>
            <th>思考强度</th>
            <th>上下文模式</th>
            <th>Max 模式</th>
            <th>Tool 参数</th>
            <th class="numeric">Tokens（入 / 出 / 总）</th>
            <th class="numeric">消耗积分</th>
            <th>状态</th>
            <th>链路</th>
          </tr>
        </thead>
        <tbody id="usage-records-body"></tbody>
      </table>
      <div id="usage-empty" class="usage-empty">暂无消费记录</div>
    </div>"""

    ic = {name: _icon(name) for name in _LUCIDE_ICONS}

    # 账号列表
    rows = ""
    stat_valid = 0
    stat_checked = 0
    stat_remaining = 0.0
    stat_known = False
    stat_unlimited = False
    for acc in accounts:
        if acc.get("is_valid"):
            st = '<span class="badge badge-ok">有效</span>'
            stat_valid += 1
        else:
            st = '<span class="badge badge-expired">无效</span>'
        act = (
            '<span class="badge badge-active" data-account-active>当前</span>'
            if acc.get("is_active")
            else '<span class="badge badge-active" data-account-active hidden>当前</span>'
        )
        active_row = "active-row" if acc.get("is_active") else ""
        switch_disabled = " disabled" if acc.get("is_active") else ""
        aid = html_mod.escape(acc.get("id") or "")
        model_enabled = bool(acc.get("model_enabled", True))
        model_checked = " checked" if model_enabled else ""
        model_badge = (
            '<span class="badge badge-ok" id="model-state-%s">已开启</span>'
            % aid
            if model_enabled
            else '<span class="badge badge-none" id="model-state-%s">已关闭</span>' % aid
        )
        label = html_mod.escape(acc.get("label") or acc.get("user_id") or acc.get("id") or "")
        uid = html_mod.escape(acc.get("user_id") or acc.get("id") or "")
        expires = html_mod.escape((acc.get("expires") or "")[:16])
        account_credits = acc.get("account_credits") or {}
        if account_credits.get("unlimited"):
            credits_text = "☆ 无限"
            stat_unlimited = True
        elif account_credits.get("remaining") is not None:
            remaining = float(account_credits["remaining"])
            credits_text = f"剩{remaining:.2f}/总{float(account_credits.get('total_limit') or 0):.2f}"
            stat_remaining += remaining
            stat_known = True
        else:
            credits_text = "-"
        checked_in = acc.get("checked_in")
        if checked_in is True:
            checkin_badge = '<span class="badge badge-ok">已签到</span>'
            stat_checked += 1
        elif checked_in is False:
            checkin_badge = '<span class="badge badge-active">未签到</span>'
        else:
            checkin_badge = '<span class="badge badge-none">未知</span>'
        valid_flag = "1" if acc.get("is_valid") else "0"
        rows += f"""<tr id="row-{aid}" class="{active_row}" data-account-id="{aid}" data-valid="{valid_flag}" data-model-enabled="{'1' if model_enabled else '0'}">
          <td><div class="acct-cell"><strong id="label-{aid}">{label}</strong><small class="row-subtitle mono">{uid}</small></div></td>
          <td class="status-cell">{st}{act}</td>
          <td class="muted-cell">{expires}</td>
          <td class="numeric"><span id="general-credits-{aid}" class="credit-value">{credits_text}</span></td>
          <td><span id="checkin-{aid}" class="checkin-state">{checkin_badge}</span><small id="checkin-detail-{aid}" class="row-subtitle"></small></td>
          <td class="model-toggle-cell"><div id="model-state-wrap-{aid}" class="model-toggle-state">{model_badge}</div><label class="switch account-model-switch" title="仅控制模型请求，不影响签到"><input type="checkbox" data-action="model-toggle" onchange="markAccountModelDirty('{aid}', this)" aria-label="启用账号模型请求"{model_checked}><span class="track"></span></label></td>
          <td class="row-actions">
            <button class="icon-btn" data-action="checkin" onclick="checkinAccount('{aid}')" title="签到" aria-label="签到">{ic['calendar-check']}</button>
            <button class="icon-btn" data-action="switch-account" onclick="switchAccount('{aid}')" title="切换为当前账号" aria-label="切换为当前账号"{switch_disabled}>{ic['arrow-left-right']}</button>
            <button class="icon-btn danger" onclick="removeAccount('{aid}')" title="删除账号" aria-label="删除账号">{ic['trash-2']}</button>
          </td>
        </tr>"""
    if accounts:
        accounts_html = f"""<div class="table-wrap">
          <table class="acct-table">
            <thead><tr><th>账号</th><th>状态</th><th>有效期</th><th class="numeric">通用积分</th><th>签到状态</th><th>模型请求</th><th class="actions-col">操作</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""
    else:
        accounts_html = '<div class="empty-state">暂无账号，请先授权登录或手动添加。</div>'
    # The toolbar stays in the DOM before the first login so the same script
    # handles an account list that becomes populated later.
    toolbar_hidden = "" if accounts else " hidden"
    stat_total = len(accounts)
    if stat_unlimited:
        stat_credits = "无限"
    elif stat_known:
        stat_credits = f"{stat_remaining:.2f}"
    else:
        stat_credits = "-"

    logout_btn = ''
    if state.token:
        logout_btn = f'<button class="icon-btn" onclick="logout()" title="登出" aria-label="登出">{ic["log-out"]}</button>'

    settings_web = (settings.get("web_base_url") or WEB_BASE).rstrip("/")
    settings_model_base = str(settings.get("model_base_url") or "").rstrip("/")
    settings_port = settings.get("relay_port") or os.environ.get("RELAY_PORT") or PORT
    # Select by mode as well as URL; URL-only matching can otherwise render a
    # different protocol after a reload.  IDE Raw is deliberately absent from
    # the public preset list because personal accounts consistently receive
    # ``2001 failed to get app config: record not found``; the raw transport
    # remains available only for internal protocol diagnostics.
    selected_endpoint_id = str(settings.get("endpoint_id") or "").strip().lower()
    selected_endpoint_id = {
        "agent": "work",
    }.get(selected_endpoint_id, selected_endpoint_id)
    selected_mode = str(
        settings.get("upstream_mode") or _current_upstream_mode() or ""
    ).strip().lower()
    selected_mode = {
        # Legacy values are kept readable in the settings store, but the
        # public UI no longer exposes the unsupported personal-account Raw
        # preset.  A stale raw selection is shown as custom until the user
        # explicitly chooses a supported endpoint.
        "direct": "raw",
        "auto": "raw",
        "9router": "remote",
        "trae-remote": "remote",
        "web": "remote",
        "work-agent": "work",
    }.get(selected_mode, selected_mode)
    def _preset_url_match(item: Mapping[str, Any]) -> bool:
        """Match the shared base, or a legacy host-only IDE/SOLO gateway.

        Releases before the IDE/SOLO gateway split persisted the bare host in
        ``web_base_url``.  Those stores must still render the IDE/SOLO preset
        (and be repaired on the next save) instead of silently becoming a
        custom URL.
        """

        shared = str(item.get("base_url") or "").rstrip("/")
        gateway = str(item.get("model_base_url") or "").rstrip("/")
        if settings_web == shared or (bool(gateway) and settings_web == gateway):
            return True
        # Stores from the first endpoint split wrote the shared host without
        # ``/api/remote/v1``.  Treat that legacy value as the same preset so
        # the page can show the selected protocol before the next save
        # canonicalizes the URL.
        try:
            parsed_web = urlsplit(settings_web)
            parsed_shared = urlsplit(shared)
            return bool(
                parsed_web.scheme.lower() == parsed_shared.scheme.lower()
                and parsed_web.netloc.lower() == parsed_shared.netloc.lower()
                and parsed_web.path.rstrip("/") in {"", "/"}
            )
        except Exception:
            return False

    url_matches = [
        item for item in UPSTREAM_ENDPOINT_PRESETS if _preset_url_match(item)
    ]
    # Prefer the persisted preset identity.  The IDE/SOLO presets keep their
    # own host-only model gateway, so URL-only matching can silently render
    # the wrong option after a reload.
    # ``endpoint_id`` is the durable protocol identity. Older releases
    # persisted a bare host instead of the canonical Remote path, so requiring
    # an exact URL match would render a valid IDE/SOLO selection as "custom"
    # after a reload. Prefer the saved id and let the settings API normalize
    # the URL on the next explicit save.
    matched_endpoint = next(
        (
            item
            for item in UPSTREAM_ENDPOINT_PRESETS
            if str(item.get("id") or "").strip().lower() == selected_endpoint_id
        ),
        None,
    )
    if matched_endpoint is None:
        matched_endpoint = next(
            (item for item in url_matches if item["mode"] == selected_mode),
            None,
        )
    if matched_endpoint is None and len(url_matches) == 1:
        matched_endpoint = url_matches[0]
    # Custom URLs that aren't in the preset list still appear as saved options.
    custom_is_saved = bool(settings.get("web_base_url")) and not matched_endpoint
    custom_selected = " selected" if not matched_endpoint else ""
    endpoint_options = [f'<option value=""{custom_selected}>自定义</option>']
    if custom_is_saved:
        endpoint_options.append(
            f'<option value="__saved" selected>{html_mod.escape(settings_web)}</option>'
        )
    for item in UPSTREAM_ENDPOINT_PRESETS:
        if not item.get("visible", True):
            continue
        selected = " selected" if matched_endpoint and matched_endpoint["id"] == item["id"] else ""
        endpoint_options.append(
            f'<option value="{html_mod.escape(item["id"])}"{selected}>{html_mod.escape(item["label"])}</option>'
        )
    endpoint_options_html = "".join(endpoint_options)
    endpoint_map_json = json.dumps(
        {
            item["id"]: {
                "base_url": item["base_url"],
                "mode": item["mode"],
                "model_base_url": item.get("model_base_url") or "",
                "endpoint_url": item.get("endpoint_url") or item["base_url"],
                "endpoint_path": item.get("endpoint_path") or "",
            }
            for item in UPSTREAM_ENDPOINT_PRESETS
        },
        ensure_ascii=False,
    )
    poll_checked = 'checked' if polling.get("enabled") else ''
    poll_mode_rr = 'checked' if polling.get('mode', 'round-robin') == 'round-robin' else ''
    poll_mode_cp = 'checked' if polling.get('mode') == 'credit-priority' else ''
    max_settings = auth.get_max_mode_settings()
    max_checked = 'checked' if max_settings.get('enabled') else ''
    max_models = html_mod.escape(max_settings.get('models') or '')
    max_state_text = '已开启' if max_settings.get('enabled') else '已关闭'
    auto_route_checked = 'checked' if auto_route_on else ''
    auto_route_state_text = '已开启' if auto_route_on else '已关闭'
    # While auto routing is enabled the dispatcher ignores the manual endpoint
    # selection, so the settings form renders those fields read-only instead of
    # letting the console show a channel that will never be used.
    endpoint_lock_pill = ' lock' if auto_route_on else ''
    endpoint_lock_text = '自动路由已接管' if auto_route_on else '手动选择'
    endpoint_fields_class = 'endpoint-locked' if auto_route_on else ''
    # Render the disabled state on the first response as well as in
    # ``applyEndpointLock``.  This prevents a brief interactive window before
    # the browser evaluates the settings script after a refresh.
    endpoint_disabled_attr = (
        ' disabled aria-disabled="true"' if auto_route_on else ' aria-disabled="false"'
    )
    auto_checkin = auth.get_auto_checkin_settings()
    auto_checkin_checked = 'checked' if auto_checkin.get('enabled') else ''
    auto_checkin_time = html_mod.escape(auto_checkin.get('time') or '08:30')
    poll_pill = ' on' if polling.get('enabled') else ''
    auto_route_pill = ' on' if auto_route_on else ''
    max_pill = ' on' if max_settings.get('enabled') else ''
    auto_checkin_pill = ' on' if auto_checkin.get('enabled') else ''
    auto_checkin_state_text = (
        f"每天 {auto_checkin_time} 自动签到" if auto_checkin.get('enabled') else '已关闭'
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Trae CN Relay 控制台</title>
{_FAVICON_LINKS}
<script>
(function(){{
  try {{
    var t = localStorage.getItem('relay-theme');
    if (t !== 'dark' && t !== 'light') t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
  }} catch (e) {{ document.documentElement.setAttribute('data-theme', 'light'); }}
}})();
</script>
<style>
{_CONSOLE_CSS}
</style>
</head>
<body>
<div id="toast" class="toast" role="alert" aria-live="assertive"><span id="toast-title" class="toast-title"></span><span id="toast-text"></span></div>
<div class="app">
<aside class="sidebar">
  <div class="brand">
    <span class="brand-mark">{_LOGO_SVG_INLINE}</span>
    <div class="brand-block">
      <h1>Trae CN Relay</h1>
      <span class="brand-sub">控制台</span>
    </div>
  </div>
  <nav class="nav-list" role="tablist" aria-label="功能导航">
    <button class="nav-item active" role="tab" data-tab="overview" data-title="概览" onclick="switchTab('overview')">{ic['layout-dashboard']}<span>概览</span></button>
    <button class="nav-item" role="tab" data-tab="accounts" data-title="账号与签到" onclick="switchTab('accounts')">{ic['users']}<span>账号与签到</span></button>
    <button class="nav-item" role="tab" data-tab="usage" data-title="消费记录" onclick="switchTab('usage')">{ic['receipt']}<span>消费记录</span></button>
    <button class="nav-item" role="tab" data-tab="settings" data-title="轮询与设置" onclick="switchTab('settings')">{ic['sliders-horizontal']}<span>轮询与设置</span></button>
    <button class="nav-item" role="tab" data-tab="models" data-title="模型测试" onclick="switchTab('models')">{ic['activity']}<span>模型测试</span></button>
    <button class="nav-item" role="tab" data-tab="model-list" data-title="模型列表" onclick="switchTab('model-list')">{ic['list']}<span>模型列表</span></button>
  </nav>
  <div class="sidebar-foot"><span>Trae2api-cn</span><span>v{APP_VERSION}</span></div>
</aside>
<main class="main">
<header class="topbar">
  <h2 id="page-title">概览</h2>
  <div class="topbar-right">
    {status_html}
    <label class="theme-switch" title="夜间模式">
      <span class="theme-icon theme-sun">{ic['sun']}</span>
      <span class="switch"><input type="checkbox" id="theme-toggle" role="switch" aria-label="夜间模式" onchange="applyTheme(this.checked?'dark':'light',true)"><span class="track"></span></span>
      <span class="theme-icon theme-moon">{ic['moon']}</span>
    </label>
    {logout_btn}
  </div>
</header>
<div class="content">

<div class="tab-page active" data-page="overview">
<div class="panel-stack">
<dl class="stats stats-6" aria-label="用量概览">
  <div class="stat"><dt>{ic['sigma']}累计 Token</dt><dd id="ov-total">--</dd><dd class="stat-sub" id="ov-total-sub">--</dd></div>
  <div class="stat"><dt>{ic['arrow-down-to-line']}累计输入</dt><dd id="ov-input">--</dd><dd class="stat-sub" id="ov-input-sub">--</dd></div>
  <div class="stat"><dt>{ic['arrow-up-from-line']}累计输出</dt><dd id="ov-output">--</dd><dd class="stat-sub" id="ov-output-sub">--</dd></div>
  <div class="stat"><dt>{ic['database']}累计缓存</dt><dd id="ov-cache">--</dd><dd class="stat-sub" id="ov-cache-sub">--</dd></div>
  <div class="stat"><dt>{ic['calendar']}今日 Token</dt><dd id="ov-today">--</dd><dd class="stat-sub" id="ov-today-sub">--</dd></div>
  <div class="stat"><dt>{ic['users']}有效账号</dt><dd id="ov-accounts">--</dd><dd class="stat-sub" id="ov-accounts-sub">--</dd></div>
</dl>
<div id="ov-msg" class="msg" role="status" aria-live="polite"></div>
<div class="ov-grid">
<section class="panel-card" aria-labelledby="ov-chart-title">
  <div class="section-head">
    <div><div class="section-title" id="ov-chart-title">每日 Token 用量</div><span id="ov-updated" class="section-meta"></span></div>
    <div class="segmented" role="radiogroup" aria-label="统计范围">
      <label><input type="radio" name="ov-range" value="7" onchange="setOverviewRange(this.value)"><span>7 天</span></label>
      <label><input type="radio" name="ov-range" value="14" onchange="setOverviewRange(this.value)" checked><span>14 天</span></label>
      <label><input type="radio" name="ov-range" value="30" onchange="setOverviewRange(this.value)"><span>30 天</span></label>
    </div>
  </div>
  <div id="ov-readout" class="chart-readout" aria-live="polite"></div>
  <div class="chart-wrap">
    <div class="chart-y" aria-hidden="true"><span id="ov-y-max">0</span><span id="ov-y-mid">0</span><span>0</span></div>
    <div id="ov-chart" class="chart" role="group" aria-label="每日 Token 柱状图"></div>
    <div id="ov-axis" class="chart-axis" aria-hidden="true"></div>
  </div>
</section>
<section class="panel-card" aria-labelledby="ov-service-title">
  <div class="section-head"><div class="section-title" id="ov-service-title">服务状态</div></div>
  <dl class="kv-list">
    <div><dt>{ic['shield-check']}登录凭据</dt><dd id="ov-token">--</dd></div>
    <div><dt>{ic['server']}版本</dt><dd id="ov-version">v{APP_VERSION}</dd></div>
    <div><dt>{ic['activity']}运行时长</dt><dd id="ov-uptime">--</dd></div>
    <div><dt>{ic['arrow-left-right']}上游端点</dt><dd id="ov-upstream">--</dd></div>
    <div><dt>{ic['refresh-cw']}多账号轮询</dt><dd id="ov-polling">--</dd></div>
    <div><dt>{ic['zap']}1M 上下文</dt><dd id="ov-max">--</dd></div>
    <div><dt>{ic['calendar-check']}自动签到</dt><dd id="ov-autocheckin">--</dd></div>
    <div><dt>{ic['list']}进行中请求</dt><dd id="ov-inflight">--</dd></div>
  <div><dt>{ic['coins']}剩余通用积分</dt><dd id="ov-credits">--</dd><dd class="stat-sub" id="ov-credits-meta">等待校准</dd></div>
  </dl>
</section>
</div>
<section class="panel-card" aria-labelledby="ov-days-title">
  <div class="section-head">
    <div><div class="section-title" id="ov-days-title">按日期汇总</div><span id="ov-days-meta" class="section-meta"></span></div>
    <label class="inline-check"><input type="checkbox" id="ov-hide-empty" onchange="renderOverviewDays()">隐藏无请求日期</label>
  </div>
  <div class="day-wrap">
    <table class="day-table">
      <thead><tr><th>日期</th><th>请求</th><th>失败</th><th>输入</th><th>输出</th><th>缓存</th><th>总计</th><th>Max 请求/生效</th><th>Tool 请求/调用</th><th>积分</th></tr></thead>
      <tbody id="ov-days-body"></tbody>
      <tfoot><tr id="ov-days-foot"></tr></tfoot>
    </table>
  </div>
</section>
</div>
</div>

<div class="tab-page" data-page="accounts">
<div class="panel-stack">
<dl class="stats" aria-label="账号概览">
  <div class="stat"><dt>{ic['users']}账号</dt><dd id="stat-total">{stat_total}</dd></div>
  <div class="stat"><dt>{ic['shield-check']}有效</dt><dd id="stat-valid">{stat_valid}</dd></div>
  <div class="stat"><dt>{ic['check-check']}今日已签到</dt><dd id="stat-checked">{stat_checked}</dd></div>
  <div class="stat"><dt>{ic['coins']}剩余通用积分</dt><dd id="stat-credits">{stat_credits}</dd><dd class="stat-sub" id="stat-credits-meta">等待校准</dd></div>
</dl>
<section class="panel-card" aria-label="账号列表">
  <div class="section-head">
    <div><div class="section-title">账号列表</div><span id="checkin-summary" class="section-meta">{stat_total} 个账号</span><span id="checkin-updated" class="section-meta"></span></div>
    <div class="toolbar account-toolbar"{toolbar_hidden} aria-live="polite">
      <span id="checkin-busy" class="busy-indicator" role="status" aria-live="polite">正在处理...</span>
      <button class="btn btn-secondary btn-sm" id="checkin-status-refresh-btn" onclick="checkinRefreshAll()">{ic['refresh-cw']}查询签到状态</button>
      <button class="btn btn-secondary btn-sm" id="credits-refresh-btn" onclick="creditsRefreshAll()">{ic['coins']}查询全部积分</button>
      <span id="credits-calibration-state" class="credit-calibration" role="status" aria-live="polite">等待校准</span>
      <button class="btn btn-primary btn-sm" id="checkin-claim-btn" onclick="checkinClaimAll()">{ic['calendar-check']}一键轮询签到</button>
      <button class="btn btn-secondary btn-sm" id="account-model-save-btn" onclick="saveAccountModelSettings()" disabled>{ic['save']}保存模型开关</button>
    </div>
  </div>
  {accounts_html}
  <div class="account-msgs">
    <div id="account-msg" class="msg" role="status" aria-live="polite"></div>
    <div id="checkin-msg" class="msg" role="status" aria-live="polite"></div>
  </div>
</section>
<div class="panel-grid cols-2">
<section class="panel-card" aria-labelledby="login-title">
  <div class="section-head"><div class="section-title" id="login-title">授权登录</div></div>
  <ol class="login-steps">
    <li>点击授权登录，自动检测本机授权助手（<code>127.0.0.1:{listener_port}</code>）。</li>
    <li>未检测到时，下载 <code>web_login.py</code> 或本机一键脚本（Windows <code>start_auth.bat</code> / macOS·Linux <code>start_auth.sh</code>）在<b>本机</b>运行后重试。</li>
    <li>浏览器需已登录 <a href="https://www.trae.cn" target="_blank" rel="noopener">trae.cn</a>，授权完成后凭据自动写入服务器。</li>
  </ol>
  <div class="btn-group">
    <button class="btn btn-primary" onclick="startAuth()" id="auth-btn">{ic['log-in']}使用 Trae 网页授权登录</button>
    <a class="btn btn-secondary" href="/web/login/download?as=py" download id="helper-py-link">{ic['download']}授权助手 web_login.py</a>
    <a class="btn btn-secondary" href="/web/login/download?as=bat" download id="bat-link">{ic['download']}start_auth.bat</a>
    <a class="btn btn-secondary" href="/web/login/download?as=sh" download id="sh-link">{ic['download']}start_auth.sh</a>
    <a class="icon-btn" href="https://www.trae.cn" target="_blank" rel="noopener" title="访问 trae.cn" aria-label="访问 trae.cn">{ic['external-link']}</a>
  </div>
  <div id="loading" class="loading">等待授权中...</div>
  <div id="auth-msg" class="msg"></div>
  <details class="manual-add">
    <summary>手动填写凭证添加账号</summary>
    <form id="manual-form" class="manual-grid">
      <div class="form-group span-all">
        <label for="manual-token">Token（Cloud-IDE-JWT）<span class="required">*</span></label>
        <textarea id="manual-token" name="token" required placeholder="eyJhbGciOiJSUzI1NiI6Ik9wZW5TU0..."></textarea>
      </div>
      <div class="form-group">
        <label for="manual-refresh">Refresh Token</label>
        <input id="manual-refresh" name="refreshToken" placeholder="可选">
      </div>
      <div class="form-group">
        <label for="manual-uid">User ID</label>
        <input id="manual-uid" name="userId" placeholder="可选">
      </div>
      <div class="form-group">
        <label for="manual-cid">Client ID</label>
        <input id="manual-cid" name="clientId" value="{html_mod.escape(client_id)}">
      </div>
      <div class="form-group">
        <label for="manual-label">备注</label>
        <input id="manual-label" name="label" placeholder="可选">
      </div>
      <div class="span-all manual-submit">
        <button type="submit" class="btn btn-primary btn-sm">添加账号</button>
        <div id="manual-msg" class="msg"></div>
      </div>
    </form>
  </details>
</section>
<section class="panel-card" id="auto-checkin-panel" aria-labelledby="auto-checkin-title">
  <div class="section-head">
    <div class="section-title" id="auto-checkin-title">自动签到</div>
    <span id="auto-checkin-state" class="state-pill{auto_checkin_pill}">{auto_checkin_state_text}</span>
  </div>
  <div class="switch-row">
    <div class="switch-text"><strong>定时签到</strong><span>每天按设定时间依次签到所有账号，已签到账号自动跳过</span></div>
    <label class="switch" title="启用定时签到">
      <input type="checkbox" id="auto-checkin-toggle" {auto_checkin_checked} onchange="markAutoCheckinDirty()" aria-label="启用定时签到"><span class="track"></span>
    </label>
  </div>
  <div class="schedule-row">
    <label class="time-field" for="auto-checkin-time">每天（北京时间）
      <input class="control" type="time" id="auto-checkin-time" value="{auto_checkin_time}" step="60" required oninput="markAutoCheckinDirty()">
    </label>
    <div class="toolbar">
      <button class="btn btn-primary btn-sm" id="auto-checkin-save-btn" onclick="saveAutoCheckin()" disabled>{ic['save']}保存</button>
      <button class="btn btn-secondary btn-sm" id="auto-checkin-run-btn" onclick="runAutoCheckinNow()" title="立即按顺序签到所有未签到账号，已签到账号自动跳过">{ic['play']}立即执行</button>
    </div>
  </div>
  <dl class="schedule-status">
    <div><dt>下次执行</dt><dd id="auto-checkin-next">-</dd></div>
    <div><dt>上次执行</dt><dd id="auto-checkin-last">-</dd></div>
    <div><dt>上次结果</dt><dd id="auto-checkin-result">-</dd></div>
  </dl>
  <div id="auto-checkin-msg" class="msg" role="status" aria-live="polite"></div>
</section>
</div>
</div>
</div>

<div class="tab-page" data-page="usage">
<div class="panel-grid">
<div class="panel-card" id="usage-panel">
  <div class="section-head"><div class="section-title">消费记录</div><span id="usage-updated" class="section-meta"></span></div>
  <div id="usage-msg" class="msg" role="status" aria-live="polite"></div>
  {usage_records_html}
</div>
</div>
</div>

<div class="tab-page" data-page="settings">
<div class="panel-grid cols-2">
<div class="panel-card">
  <div class="section-head"><div class="section-title">多账号轮询</div></div>
  <div class="switch-row">
    <div class="switch-text"><strong>启用轮询</strong><span>每次请求自动切换到下一个有效账号</span></div>
    <label class="switch" title="启用轮询">
      <input type="checkbox" id="poll-toggle" {poll_checked} onchange="markPollingDirty()" aria-label="启用轮询"><span class="track"></span>
    </label>
  </div>
  <div class="switch-row">
    <div class="switch-text"><strong>轮询模式</strong></div>
    <div class="segmented" role="radiogroup" aria-label="轮询模式">
      <label><input type="radio" name="poll-mode" value="round-robin" onchange="markPollingDirty()" {poll_mode_rr}><span>顺序轮询</span></label>
      <label><input type="radio" name="poll-mode" value="credit-priority" onchange="markPollingDirty()" {poll_mode_cp}><span>积分优先</span></label>
    </div>
  </div>
  <p id="poll-status" class="field-note">当前账号数: {polling.get('account_count', 0)}，轮询: {'开' if polling.get('enabled') else '关'}</p>
  <div class="btn-group">
    <button class="btn btn-primary btn-sm" id="poll-save-btn" onclick="savePollingSettings()" disabled>{ic['save']}保存轮询设置</button>
  </div>
  <div id="poll-msg" class="msg" role="status" aria-live="polite"></div>
</div>
<div class="panel-card">
  <div class="section-head">
    <div class="section-title">上游端点</div>
    <span id="endpoint-lock-state" class="state-pill{endpoint_lock_pill}">{endpoint_lock_text}</span>
  </div>
  <div id="endpoint-lock-note" class="lock-note" hidden>
    {ic['lock']}
    <div><strong>自动路由已开启，手动端点选择已锁定。</strong>当前由自动路由决定实际通道：工具或图片请求走 IDE Agent，纯文字聊天走 Remote，失败保底回落 Remote。如需手动指定端点，请先在下方关闭自动路由。</div>
  </div>
  <div id="endpoint-fields" class="{endpoint_fields_class}">
  <div class="form-group">
    <label for="settings-endpoint-preset">预设端点</label>
    <select id="settings-endpoint-preset"{endpoint_disabled_attr} onchange="applyEndpointPreset()">
      {endpoint_options_html}
    </select>
  </div>
  <div class="form-row">
    <div class="form-group">
      <label for="settings-web">自定义 Web Base URL</label>
    <input id="settings-web" class="mono-input" value="{html_mod.escape(settings_web)}" placeholder="https://trae-api-cn.mchost.guru/api/remote/v1"{endpoint_disabled_attr} oninput="markEndpointDirty()">
    </div>
  </div>
  <div class="form-row">
    <div class="form-group">
      <label for="settings-model-base">模型网关地址（IDE Agent / Solo）</label>
    <input id="settings-model-base" class="mono-input" value="{html_mod.escape(settings_model_base)}" placeholder="https://trae-api-cn.mchost.guru"{endpoint_disabled_attr} oninput="markEndpointDirty()">
    </div>
  </div>
  <p class="field-note" id="settings-endpoint-preview"></p>
  </div>
  <div class="btn-group">
    <button class="btn btn-primary btn-sm" id="settings-save-btn" onclick="saveSettings()" disabled>{ic['save']}保存设置</button>
  </div>
  <div id="settings-msg" class="msg"></div>
</div>
<div class="panel-card" id="relay-service-panel">
  <div class="section-head"><div class="section-title">Relay 服务</div></div>
  <div class="form-group">
    <label for="settings-port">宿主机映射端口（RELAY_PORT）</label>
      <input id="settings-port" type="number" value="{settings_port}" placeholder="8000" min="1" max="65535" oninput="markRelayServiceDirty()">
  </div>
  <p class="field-note">这是 Docker 对外发布的宿主机端口，容器内监听端口仍由 PORT 控制。修改后必须同步调整端口映射并重启，否则管理页面和 API 可能无法访问。</p>
  <div class="btn-group">
      <button class="btn btn-secondary btn-sm" id="relay-service-save-btn" onclick="saveRelayServiceSettings()" disabled>{ic['save']}保存服务设置</button>
  </div>
  <div id="relay-service-msg" class="msg"></div>
</div>
<div class="panel-card" id="auto-route-panel">
  <div class="section-head">
    <div class="section-title">自动路由</div>
    <span id="auto-route-state" class="state-pill{auto_route_pill}">{auto_route_state_text}</span>
  </div>
  <div class="switch-row">
    <div class="switch-text"><strong>启用自动路由</strong><span>工具或图片请求走 IDE Agent，纯文字聊天走 Remote，失败回落 Remote；开启后忽略预设端点的模式</span></div>
    <label class="switch" title="启用自动路由">
      <input type="checkbox" id="auto-route-toggle" {auto_route_checked} onchange="markAutoRouteDirty()" aria-label="启用自动路由"><span class="track"></span>
    </label>
  </div>
  <div class="btn-group">
    <button class="btn btn-primary btn-sm" id="auto-route-save-btn" onclick="saveAutoRoute()" disabled>{ic['save']}保存自动路由</button>
  </div>
  <div id="auto-route-msg" class="msg" role="status" aria-live="polite"></div>
</div>
<div class="panel-card" id="max-mode-panel">
  <div class="section-head">
    <div class="section-title">1M 上下文（Max 模式）</div>
    <span id="max-mode-state" class="state-pill{max_pill}">{max_state_text}</span>
  </div>
  <div class="switch-row">
    <div class="switch-text"><strong>启用 Max 模式</strong><span>Remote Agent 会话使用 1M 上下文；带调用端工具的请求走 Work，不使用 Max</span></div>
    <label class="switch" title="启用 Max 模式">
      <input type="checkbox" id="max-mode-toggle" {max_checked} onchange="markMaxModeDirty()" aria-label="启用 Max 模式"><span class="track"></span>
    </label>
  </div>
  <div class="form-group">
    <label for="max-mode-models">生效模型（逗号分隔，留空表示账号中所有支持 Max 的模型）</label>
    <input id="max-mode-models" value="{max_models}" placeholder="glm-5.3, deepseek-v4-pro" oninput="markMaxModeDirty()">
  </div>
  <div class="btn-group">
    <button class="btn btn-primary btn-sm" id="max-mode-save-btn" onclick="saveMaxMode()" disabled>{ic['save']}保存</button>
    <button class="btn btn-secondary btn-sm" id="max-mode-detect-btn" onclick="detectMaxModels()">{ic['search']}检测支持的模型</button>
  </div>
  <div id="max-mode-models-out" class="max-models" hidden></div>
  <div id="max-mode-msg" class="msg" role="status" aria-live="polite"></div>
</div>
</div>
</div>

<div class="tab-page" data-page="models">
<div class="panel-grid">
<div class="panel-card" id="conn-panel">
  <div class="section-head">
    <div class="section-title">模型连通性测试</div>
    <span id="conn-summary" class="section-meta">未运行</span>
  </div>
  <div class="form-row">
    <div class="form-group">
      <label for="conn-models">测试模型（逗号或换行分隔，留空使用常用模型）</label>
      <textarea id="conn-models" rows="2" placeholder="glm-5.3, DeepSeek-V4-Pro-Official"></textarea>
    </div>
    <div class="form-group">
      <label for="conn-endpoint">指定上游（测试期间禁止跨端点回落）</label>
    <select id="conn-endpoint">
        <option value="auto-route">自动路由（工具或图片 IDE Agent / 文字 Remote）</option>
        <option value="remote">Remote / chat_sessions</option>
        <option value="work">TraeWork / chat_sessions（Work tier）</option>
        <option value="solo">TraeCode Solo / llm_raw_chat</option>
        <option value="ide">TraeCode IDE / llm_raw_chat</option>
      </select>
    </div>
  </div>
  <div class="form-group">
    <span class="field-label">测试内容</span>
    <div class="option-row">
      <label class="inline-check"><input type="checkbox" id="conn-mode-text" checked> 文本回复</label>
      <label class="inline-check"><input type="checkbox" id="conn-mode-tool"> 工具调用</label>
      <label class="inline-check"><input type="checkbox" id="conn-thinking"> 返回思考内容</label>
      <label class="inline-check" title="请求 1M Max 上下文，结果中显示是否实际生效"><input type="checkbox" id="conn-max"> 1M Max</label>
      <label class="inline-check inline-field">思考强度
        <select id="conn-effort" title="映射到 Trae custom_model.reasoning_effort：low=light，medium/high=high，xhigh=extra_high，超出模型支持的档位会向下取">
          <option value="">默认</option>
          <option value="low">low / 轻</option>
          <option value="medium">medium / 高</option>
          <option value="high">high / 高</option>
          <option value="xhigh">xhigh / 极高</option>
          <option value="ultra">ultra / 极高</option>
        </select>
      </label>
      <label class="inline-check inline-field">超时(秒) <input type="number" id="conn-timeout" value="120" min="10" max="600"></label>
    </div>
  </div>
  <div class="btn-group">
    <button class="btn btn-primary btn-sm" id="conn-run-btn" onclick="runConnTest()">{ic['play']}开始测试</button>
    <button class="btn btn-secondary btn-sm" onclick="fillConnPreset()">{ic['list-plus']}填入常用模型</button>
  </div>
  <div class="conn-wrap" id="conn-table" hidden>
    <table class="conn-table">
      <thead><tr><th>模型</th><th>类型</th><th>结果</th><th>耗时</th><th>详情</th></tr></thead>
      <tbody id="conn-tbody"></tbody>
    </table>
  </div>
  <div id="conn-msg" class="msg"></div>
</div>
</div>
</div>

<div class="tab-page" data-page="model-list">
<div class="panel-stack">
<section id="model-list-panel" aria-labelledby="model-list-title">
  <div class="section-head">
    <div><div class="section-title" id="model-list-title">模型列表</div><span id="model-list-meta" class="section-meta">尚未加载</span></div>
    <button class="btn btn-secondary btn-sm" id="models-refresh-btn" onclick="refreshModels()">{ic['refresh-cw']}刷新模型列表</button>
  </div>
  <p class="section-desc">TraeCode：Solo / IDE · TraeWork：Work · 同一模型可出现在多个端点目录中</p>
  <div class="model-list-grid">
    <section class="model-column" aria-labelledby="models-col-remote-title">
      <div class="model-column-head"><span class="model-column-title" id="models-col-remote-title">Remote</span><span class="model-column-count" id="models-count-remote">0</span></div>
      <div class="model-column-body" id="models-col-remote"><div class="model-column-empty">尚未加载</div></div>
    </section>
    <section class="model-column" aria-labelledby="models-col-traecode-title">
      <div class="model-column-head"><span class="model-column-title" id="models-col-traecode-title">TraeCode</span><span class="model-column-count" id="models-count-traecode">0</span></div>
      <div class="model-column-body" id="models-col-traecode"><div class="model-column-empty">尚未加载</div></div>
    </section>
    <section class="model-column" aria-labelledby="models-col-traework-title">
      <div class="model-column-head"><span class="model-column-title" id="models-col-traework-title">TraeWork</span><span class="model-column-count" id="models-count-traework">0</span></div>
      <div class="model-column-body" id="models-col-traework"><div class="model-column-empty">尚未加载</div></div>
    </section>
  </div>
  <div id="models-fallback" class="model-fallback" hidden>
    <div class="model-fallback-head">
      <span class="model-fallback-title">内置回退 · 端点未确认</span>
      <span class="model-column-count" id="models-count-fallback">0</span>
      <span class="model-fallback-note">内置或未归类的候选模型，尚未验证端点可用性</span>
    </div>
    <div class="model-fallback-list" id="models-col-fallback"></div>
  </div>
  <div id="models-msg" class="msg" role="status" aria-live="polite"></div>
</section>
</div>
</div>

</div>
</main>
</div>
<script>
function applyTheme(theme,persist){{
  var t=theme==='dark'?'dark':'light';
  document.documentElement.setAttribute('data-theme',t);
  var cb=document.getElementById('theme-toggle');
  if(cb) cb.checked=(t==='dark');
  if(persist){{
    try{{ localStorage.setItem('relay-theme',t); }}catch(e){{}}
    var root=document.documentElement; root.classList.add('theme-anim');
    clearTimeout(applyTheme.timer); applyTheme.timer=setTimeout(function(){{ root.classList.remove('theme-anim'); }},360);
  }}
}}
applyTheme(document.documentElement.getAttribute('data-theme')||'light',false);
function switchTab(name){{
  document.querySelectorAll('.nav-item').forEach(function(b){{
    var on=b.getAttribute('data-tab')===name;
    b.classList.toggle('active',on);
    b.setAttribute('aria-selected',on?'true':'false');
    if(on){{ var t=document.getElementById('page-title'); if(t) t.textContent=b.getAttribute('data-title')||''; }}
  }});
  document.querySelectorAll('.tab-page').forEach(function(p){{ p.classList.toggle('active', p.getAttribute('data-page')===name); }});
  try{{ history.replaceState(null,'','#'+name); }}catch(e){{}}
  syncSelection();
  if(name==='overview') refreshOverview();
  if(name==='model-list') refreshModels();
}}
// Sliding highlights for the nav and segmented controls. The first placement
// skips the transition so elements don't fly in from the corner.
function placeSel(el,x,y,w,h){{
  if(!w||!h) return;
  var first=!el.dataset.placed;
  if(first) el.classList.remove('ready');
  el.style.width=w+'px'; el.style.height=h+'px';
  el.style.transform='translate('+x+'px,'+y+'px)'; el.style.opacity='1';
  if(first){{ el.dataset.placed='1'; void el.offsetWidth; el.classList.add('ready'); }}
}}
function syncSelection(){{
  var nav=document.querySelector('.nav-list');
  if(nav){{
    var ind=nav.querySelector('.nav-indicator');
    if(!ind){{ ind=document.createElement('span'); ind.className='nav-indicator'; ind.setAttribute('aria-hidden','true'); nav.insertBefore(ind,nav.firstChild); nav.classList.add('has-indicator'); }}
    var a=nav.querySelector('.nav-item.active');
    if(a) placeSel(ind,a.offsetLeft,a.offsetTop,a.offsetWidth,a.offsetHeight);
  }}
  document.querySelectorAll('.segmented').forEach(function(seg){{
    var th=seg.querySelector('.seg-thumb');
    if(!th){{
      th=document.createElement('span'); th.className='seg-thumb'; th.setAttribute('aria-hidden','true');
      seg.insertBefore(th,seg.firstChild); seg.classList.add('has-thumb');
      seg.addEventListener('change',syncSelection);
    }}
    var c=seg.querySelector('input:checked');
    var sp=c?c.nextElementSibling:null;
    if(sp){{ var lab=sp.parentElement; placeSel(th,lab.offsetLeft+sp.offsetLeft,lab.offsetTop+sp.offsetTop,sp.offsetWidth,sp.offsetHeight); }}
    else th.style.opacity='0';
  }});
}}
function setStatePill(id,text,on){{
  var el=document.getElementById(id);
  if(!el) return;
  el.textContent=text;
  el.classList.toggle('on',!!on);
}}
// Credit reconciliation state is intentionally kept separate from usage
// rendering.  The server's ``remaining`` field is the effective balance
// (official snapshot minus weighted, confirmed local debits); older servers
// may only expose the historical ``remaining``/``total_limit`` pair.
var creditUiState={{
  accounts:Object.create(null),
  meta:{{status:'unknown',reconciled_at:0,confidence:null,official_remaining:null,local_debit:null}},
  usageFingerprint:'',
  usageFingerprintReady:false,
  reconcileTimer:null,
  reconcileInFlight:false
}};
function creditNumber(value){{
  if(value===undefined||value===null||value==='') return null;
  var n=Number(value);
  return Number.isFinite(n)?n:null;
}}
function creditSnapshot(data){{
  if(!data||typeof data!=='object') return null;
  var raw=data.account_credits||data.credits||data;
  if(!raw||typeof raw!=='object') return null;
  var out=Object.assign({{}},raw);
  ['credits_reconciled_at','reconcile_status','reconcile_confidence','official_remaining','local_debit','credits_updated_at'].forEach(function(key){{
    if(out[key]===undefined&&data[key]!==undefined) out[key]=data[key];
  }});
  return out;
}}
function creditStatusLabel(status){{
  var value=String(status||'unknown').trim().toLowerCase();
  if(value==='ok'||value==='verified'||value==='upstream') return '已校准';
  if(value==='projected'||value==='weighted'||value==='usage_weighted') return '加权估算';
  if(value==='stale'||value==='pending'||value==='settling') return '待校准';
  if(value==='error'||value==='failed') return '校准失败';
  return '等待校准';
}}
function creditStatusClass(status){{
  var value=String(status||'unknown').trim().toLowerCase();
  if(value==='ok'||value==='verified'||value==='upstream') return 'ok';
  if(value==='projected'||value==='weighted'||value==='usage_weighted') return 'projected';
  if(value==='stale'||value==='pending'||value==='settling') return 'stale';
  if(value==='error'||value==='failed') return 'error';
  return 'unknown';
}}
function creditTime(value){{
  var n=creditNumber(value);
  if(!n) return '';
  // Upstream stores seconds; tolerate millisecond timestamps in custom
  // adapters without showing an implausible 1970 date.
  if(n>100000000000) n=n/1000;
  try{{ return new Date(n*1000).toLocaleTimeString(); }}catch(e){{ return ''; }}
}}
function renderCreditCalibrationMeta(){{
  var meta=creditUiState.meta||{{}};
  var label=creditStatusLabel(meta.status);
  var stamp=creditTime(meta.reconciled_at||meta.updated_at);
  var confidence=creditNumber(meta.confidence);
  var text=label+(stamp?' · '+stamp:'');
  if(confidence!==null) text+=' · 置信 '+Math.round(Math.max(0,Math.min(1,confidence))*100)+'%';
  var detail=[];
  var official=creditNumber(meta.official_remaining);
  var local=creditNumber(meta.local_debit);
  if(official!==null) detail.push('官方 '+official.toFixed(2));
  if(local!==null&&local>0) detail.push('待上游扣除 '+local.toFixed(2));
  ['credits-calibration-state','stat-credits-meta','ov-credits-meta'].forEach(function(id){{
    var el=document.getElementById(id);
    if(!el) return;
    el.textContent=text;
    el.title=detail.join(' · ')||'总积分由上游快照与已确认消费加权校准';
    if(el.classList){{ el.classList.remove('ok','projected','stale','error','unknown'); el.classList.add(creditStatusClass(meta.status)); }}
  }});
}}
function updateCreditCalibrationMeta(data){{
  if(!data||typeof data!=='object') return;
  var meta=creditUiState.meta||{{}};
  var source=data.accounts&&typeof data.accounts==='object'?data.accounts:data;
  var status=source.credits_reconcile_status;
  var stamp=source.credits_reconciled_at||source.credits_updated_at;
  var confidence=source.credits_reconcile_confidence;
  var official=source.credits_official_remaining;
  var local=source.credits_local_debit;
  if(status===undefined&&data.account_credits) status=data.account_credits.reconcile_status;
  if(stamp===undefined&&data.account_credits) stamp=data.account_credits.credits_reconciled_at||data.account_credits.credits_updated_at;
  if(confidence===undefined&&data.account_credits) confidence=data.account_credits.reconcile_confidence;
  if(official===undefined&&data.account_credits) official=data.account_credits.official_remaining;
  if(local===undefined&&data.account_credits) local=data.account_credits.local_debit;
  if(status!==undefined&&status!==null&&status!=='') meta.status=status;
  if(stamp!==undefined&&stamp!==null&&stamp!=='') meta.reconciled_at=stamp;
  if(confidence!==undefined&&confidence!==null&&confidence!=='') meta.confidence=confidence;
  if(official!==undefined&&official!==null&&official!=='') meta.official_remaining=official;
  if(local!==undefined&&local!==null&&local!=='') meta.local_debit=local;
  creditUiState.meta=meta;
  renderCreditCalibrationMeta();
}}
function updateCreditCalibrationFromAccounts(accounts){{
  if(!Array.isArray(accounts)||!accounts.length) return;
  var latest=0, status='', confidenceTotal=0, confidenceWeight=0, officialTotal=0, localTotal=0;
  var hasOfficial=false, hasLocal=false, successful=0;
  accounts.forEach(function(account){{
    var snap=creditSnapshot(account);
    if(!snap) return;
    var stamp=creditNumber(snap.credits_reconciled_at||account.credits_reconciled_at||snap.credits_updated_at||account.credits_updated_at);
    if(stamp!==null) latest=Math.max(latest,stamp);
    var accountStatus=snap.reconcile_status||account.credits_reconcile_status;
    if(accountStatus) status=status||accountStatus;
    var confidence=creditNumber(snap.reconcile_confidence||account.credits_reconcile_confidence);
    var remaining=creditNumber(snap.remaining);
    if(confidence!==null){{ var weight=Math.max(1,remaining===null?1:Math.abs(remaining)); confidenceTotal+=confidence*weight; confidenceWeight+=weight; }}
    var official=creditNumber(snap.official_remaining);
    if(official!==null){{ officialTotal+=official; hasOfficial=true; }}
    var local=creditNumber(snap.local_debit);
    if(local!==null){{ localTotal+=local; hasLocal=true; }}
    if(!account.error) successful++;
  }});
  if(successful===0) return;
  updateCreditCalibrationMeta({{
    credits_reconciled_at:latest||null,
    credits_reconcile_status:status||'ok',
    credits_reconcile_confidence:confidenceWeight?confidenceTotal/confidenceWeight:null,
    credits_official_remaining:hasOfficial?officialTotal:null,
    credits_local_debit:hasLocal?localTotal:null
  }});
}}
// Overview numbers are derived from the rendered rows so every partial
// refresh (single check-in, credits query, claim-all) keeps them in sync.
function refreshStats(){{
  var rows=Array.from(document.querySelectorAll('tr[data-account-id]'));
  var checked=0, sum=0, known=false, unlimited=false;
  rows.forEach(function(row){{
    var st=row.querySelector('.checkin-state');
    if(st&&st.textContent.trim()==='已签到') checked++;
    var cv=row.querySelector('.credit-value');
    var text=cv?cv.textContent:'';
    if(text.indexOf('无限')>=0) unlimited=true;
    var m=text.match(/剩(-?[0-9.]+)/);
    if(m){{ sum+=Number(m[1]); known=true; }}
  }});
  var set=function(id,v){{ var el=document.getElementById(id); if(el) el.textContent=v; }};
  set('stat-total',rows.length);
  set('stat-checked',checked);
  set('stat-credits',unlimited?'无限':(known?sum.toFixed(2):'-'));
  renderCreditCalibrationMeta();
}}
const state = {{ traceId: null, win: null }};
let currentCodeVerifier = '';
function uuid() {{ return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,function(c){{var r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16)}}) }}
function randomHex(n) {{ var a=new Uint8Array(n);crypto.getRandomValues(a);return Array.from(a,b=>b.toString(16).padStart(2,'0')).join('') }}
function randomDigits(n) {{ var s='';while(s.length<n)s+=Math.floor(Math.random()*1e10).toString();return s.slice(0,n) }}
function randomBase64Url(n) {{ var a=new Uint8Array(n); if(window.crypto&&crypto.getRandomValues){{ crypto.getRandomValues(a) }} else {{ for(var i=0;i<n;i++)a[i]=Math.floor(Math.random()*256) }} return btoa(String.fromCharCode.apply(null,a)).replace(/[+]/g,'-').split('/').join('_').replace(/=+$/,'') }}
function buildChallenge() {{ currentCodeVerifier=randomBase64Url(32); return sha256Base64Url(currentCodeVerifier) }}
async function buildAuthUrl() {{
  // Trae 授权页强制要求回调为 http://127.0.0.1:<port>/authorize，
  // 因此必须由本机 web_login.py 监听并转发凭据到服务器。
  var cb = 'http://127.0.0.1:{listener_port}/authorize';
  var mid = randomHex(32), did = randomDigits(19), tid = state.traceId;
  var p = new URLSearchParams({{
    login_version:'1',auth_from:'solo',login_channel:'native_ide',plugin_version:'2.3.24254',
    auth_type:'local',client_id:'{html_mod.escape(client_id)}',redirect:'0',login_trace_id:tid,
    auth_callback_url:cb,machine_id:mid,device_id:did,x_device_id:did,x_machine_id:mid,
    x_device_brand:'ASUS TUF Gaming A15 FA507RM_FA507RM',x_device_type:'windows',x_os_version:'Windows 10 Pro',x_env:'',
    x_app_version:'3.3.65',x_app_type:'stable',hide_saas_login:'true',
  }});
  return '{auth_url}?'+p.toString();
}}
let localHelperError = '';
async function checkLocalListener() {{
  var timer;
  localHelperError = '';
  try {{
    var ctrl = new AbortController();
    timer = setTimeout(function(){{ ctrl.abort(); }}, 2500);
    var expectedResponse = await fetch(helperEndpointPath('/web/login/helper-info') + '?relay=' + encodeURIComponent(currentDeploymentBase()), {{signal: ctrl.signal, cache: 'no-store'}});
    if (!expectedResponse.ok) {{
      localHelperError = '授权助手部署配置无效，请检查服务器配置后重新下载。';
      return false;
    }}
    var expected = await expectedResponse.json();
    var r = await fetch('http://127.0.0.1:{listener_port}/healthz', {{signal: ctrl.signal, cache: 'no-store'}});
    if (!r.ok) return false;
    var helper = await r.json();
    if (!helper.relay || normalizeHelperBase(helper.relay) !== normalizeHelperBase(expected.relay_url)) {{
      localHelperError = '本机授权助手绑定了其他部署，请关闭旧助手，重新下载并运行当前页面的助手。';
      return false;
    }}
    return true;
  }} catch(e) {{ return false; }}
  finally {{ clearTimeout(timer); }}
}}
function helperEndpointPath(path) {{
  return window.location.pathname.replace(/\\/web\\/login\\/?$/, '').replace(/\\/$/, '') + path;
}}
function currentDeploymentBase() {{
  return window.location.origin + helperEndpointPath('');
}}
function normalizeHelperBase(value) {{
  var url = new URL(value);
  return url.origin + url.pathname.replace(/\\/+$/, '');
}}
function refreshHelperDownloadLinks() {{
  var base = currentDeploymentBase();
  var suffix = '&relay=' + encodeURIComponent(base);
  [['helper-py-link', 'py'], ['bat-link', 'bat'], ['sh-link', 'sh']].forEach(function(item) {{
    var link = document.getElementById(item[0]);
    if (link) link.href = helperEndpointPath('/web/login/download') + '?as=' + item[1] + suffix;
  }});
  pickHelperPlatformScript();
}}
function pickHelperPlatformScript() {{
  // 只显示与本机系统匹配的一键脚本，避免多下一个用不了的文件。
  var probe = String(navigator.platform || '') + ' ' + String(navigator.userAgent || '');
  var isWindows = /Win/i.test(probe);
  var bat = document.getElementById('bat-link');
  var sh = document.getElementById('sh-link');
  if (bat) bat.style.display = isWindows ? 'inline-flex' : 'none';
  if (sh) sh.style.display = isWindows ? 'none' : 'inline-flex';
}}
refreshHelperDownloadLinks();
async function startAuth() {{
  try {{
  var ok = await checkLocalListener();
  if (!ok) {{
    showMsg('auth-msg', localHelperError || '未检测到本机授权助手，请先下载并运行当前页面的授权助手。', false);
    pickHelperPlatformScript();
    return;
  }}
  state.traceId = uuid();
  var url = await buildAuthUrl();
  var w = window.open(url, 'trae-relay-oauth', 'width=560,height=760');
  if (!w) {{ showMsg('auth-msg','浏览器已拦截，请允许弹窗',false); return; }}
  state.win = w;
  document.getElementById('loading').style.display='block';
  document.getElementById('auth-btn').disabled=true;
  var poll = setInterval(function(){{ if(w.closed){{ clearInterval(poll);document.getElementById('loading').style.display='none';document.getElementById('auth-btn').disabled=false; }} }},700);
  }} catch(e) {{ showMsg('auth-msg',String(e),false); document.getElementById('loading').style.display='none'; document.getElementById('auth-btn').disabled=false; }}
}}

let usageRefreshing = false;
function usageRecordsFromPayload(payload){{
  // Keep the console compatible with both the historical bare-array endpoint
  // and deployments that wrap records in an object for metadata/versioning.
  if(Array.isArray(payload)) return payload;
  if(!payload || typeof payload!=='object') return null;
  if(Array.isArray(payload.records)) return payload.records;
  if(Array.isArray(payload.data)) return payload.data;
  if(payload.data && typeof payload.data==='object' && Array.isArray(payload.data.records)) return payload.data.records;
  return null;
}}
function usageRecordField(record, keys){{
  if(!record || typeof record!=='object') return null;
  for(var i=0;i<keys.length;i++){{
    var value=record[keys[i]];
    if(value!==undefined && value!==null && value!=='') return value;
  }}
  return null;
}}
function usageBoolean(value){{
  if(value===true || value===1) return true;
  if(value===false || value===0) return false;
  if(typeof value==='string'){{
    var text=value.trim().toLowerCase();
    if(['true','1','yes','on','enabled','是','有'].indexOf(text)>=0) return true;
    if(['false','0','no','off','disabled','否','无'].indexOf(text)>=0) return false;
  }}
  return null;
}}
function usageEffortLabel(value){{
  if(value===undefined || value===null || value==='') return '';
  var text=String(value).trim().toLowerCase().replace(/[\\s-]+/g,'_');
  if(text==='none' || text==='minimal' || text==='min') return '最低';
  if(text==='low' || text==='light') return '低';
  if(text==='medium' || text==='mid' || text==='normal' || text==='default') return '中';
  if(text==='high') return '高';
  if(text==='xhigh' || text==='extra_high' || text==='very_high' || text==='max' || text==='maximum' || text==='ultra') return '极高';
  return escapeHtml(String(value));
}}
function usageReasoningMeta(record){{
  var requested=usageRecordField(record,[
    'requested_reasoning_effort','reasoning_effort_requested','thinking_effort',
    'thinking_level','reasoning_effort'
  ]);
  var thinking=record && record.thinking;
  if(requested===null && thinking && typeof thinking==='object'){{
    requested=thinking.effort || thinking.level || thinking.budget_tokens || thinking.budgetTokens || null;
  }}
  var applied=usageRecordField(record,[
    'reasoning_effort_applied','applied_reasoning_effort','native_reasoning_effort'
  ]);
  var requestedLabel=usageEffortLabel(requested);
  var appliedLabel=usageEffortLabel(applied);
  if(requestedLabel && appliedLabel && requestedLabel!==appliedLabel){{
    return {{
      text: requestedLabel+'<span class="usage-arrow">→</span>'+appliedLabel,
      title: '请求: '+String(requested)+'；生效: '+String(applied),
      unknown: false
    }};
  }}
  var label=appliedLabel || requestedLabel;
  return {{
    text: label || '--',
    title: label ? '思考强度: '+String(applied || requested) : '未记录思考强度',
    unknown: !label
  }};
}}
function usageContextMeta(record){{
  var requested=usageRecordField(record,['context_mode_requested','requested_context_mode']);
  var value=usageRecordField(record,['context_mode_actual','actual_context_mode','context_mode','contextMode','context_window_mode']);
  var raw=value;
  var boolMax=usageBoolean(usageRecordField(record,['max_mode_applied','max_mode']));
  var size=Number(usageRecordField(record,['context_window_tokens','max_context_tokens','prompt_max_tokens']) || 0);
  if(value===null && boolMax===true) value='max';
  if(value===null && size>=1000000) value='1m';
  if(value===null && boolMax===false) value='standard';
  // Older rows often persisted only the requested context.  Treat an
  // explicit standard request as the visible context label while preserving
  // the separate Max column for actual activation evidence.
  if(value===null && requested!==null) value=requested;
  if(value===null) return {{text:'--',title:'未记录上下文模式'+(requested!==null?'；请求: '+String(requested):''),unknown:true}};
  var text=String(value).trim().toLowerCase().replace(/[\\s_-]+/g,'');
  var label;
  if(['max','max1m','1m','1000000','agent','agentmax','maxmode'].indexOf(text)>=0 || size>=1000000) label='1M Max';
  else if(['standard','default','work','dev','200k','200000','normal'].indexOf(text)>=0) label='标准';
  else label=String(raw===null?value:raw);
  var requestedText='';
  if(requested!==null && String(requested)!==String(value)){{
    requestedText='请求: '+String(requested)+'；';
  }}
  return {{text:label,title:requestedText+'实际上下文模式: '+String(value)+(size?'；窗口: '+size.toLocaleString()+' tokens':''),unknown:false}};
}}
function usageMaxMeta(record){{
  var requested=usageBoolean(usageRecordField(record,['max_mode_requested','requested_max_mode']));
  var enabled=usageBoolean(usageRecordField(record,['max_mode_enabled','max_mode_applied','max_mode']));
  if(requested===null){{
    var requestedContext=usageRecordField(record,['context_mode_requested','requested_context_mode']);
    if(requestedContext!==null){{
      var normalized=String(requestedContext).trim().toLowerCase().replace(/[\\s_-]+/g,'');
      requested=['max','max1m','1m','1000000','agentmax','maxmode'].indexOf(normalized)>=0;
    }}
  }}
  if(enabled===null){{
    var actualContext=usageRecordField(record,['context_mode_actual','actual_context_mode','context_mode']);
    if(actualContext!==null){{
      var actual=String(actualContext).trim().toLowerCase().replace(/[\\s_-]+/g,'');
      enabled=['max','max1m','1m','1000000','agentmax','maxmode'].indexOf(actual)>=0;
    }}
  }}
  if(enabled===null && requested===false) enabled=false;
  if(enabled===true) return {{text:'已开启',title:'本次请求实际启用 1M Max',unknown:false,requested:requested,enabled:true}};
  if(enabled===false && requested===true) return {{text:'未生效',title:'请求开启 Max，但上游本次未实际启用',unknown:false,requested:requested,enabled:false}};
  if(enabled===false) return {{text:'未开启',title:'本次请求未启用 1M Max',unknown:false,requested:requested,enabled:false}};
  if(requested===true) return {{text:'请求 / 未知',title:'请求开启 Max，但没有上游实际生效证据',unknown:true,requested:requested,enabled:null}};
  return {{text:'--',title:'未记录 Max 模式',unknown:true,requested:requested,enabled:enabled}};
}}
function usageToolMeta(record){{
  var requested=usageBoolean(usageRecordField(record,['tools_requested','tool_protocol_requested']));
  var used=usageBoolean(usageRecordField(record,['tool_called','tool_used','tool_calls_returned','tool_call_returned','tool']));
  if(requested===null && used===true) requested=true;
  if(requested===null && record && Array.isArray(record.tools)) requested=record.tools.length>0;
  if(requested===null && record && record.tools!==undefined) requested=!!record.tools;
  if(requested===null && record && record.tool_choice!==undefined && record.tool_choice!=='none') requested=true;
  var state=requested;
  if(state===null) return {{text:'--',title:'未记录是否携带 tools 参数',unknown:true,requested:null,called:used}};
  var title=state?'请求携带 tools 参数':'请求未携带 tools 参数';
  if(used===true) title+='；模型返回了工具调用';
  else if(used===false) title+='；模型未返回工具调用';
  var text=state?'已携带':'未携带';
  if(state && used===true) text+=' / 本轮调用';
  else if(state && used===false) text+=' / 未调用';
  return {{text:text,title:title,unknown:false,requested:state,called:used}};
}}
function usageModelFlags(maxMode,tool){{
  var maxText=maxMode.enabled===true?'1M':(maxMode.enabled===false?'标准':(maxMode.requested===true?'请求':'--'));
  var maxClass=maxMode.enabled===true?' max':(maxMode.unknown?' unknown':'');
  var toolText=tool.requested===true?'是':(tool.requested===false?'否':'--');
  var toolClass=tool.requested===true?' tool':(tool.unknown?' unknown':'');
  return '<span class="usage-flag'+maxClass+'" title="'+escapeHtml(maxMode.title)+'">Max: '+maxText+'</span>'
    + '<span class="usage-flag'+toolClass+'" title="'+escapeHtml(tool.title)+'">Tool: '+toolText+'</span>';
}}
function usageTransportMeta(record){{
  // Queue position and continuation rounds are transport telemetry: they must
  // never appear in the model answer, but operators still need them to explain
  // a slow turn or a completion that took more than one upstream round.
  var position=Number(usageRecordField(record,['queue_position'])||0);
  var rounds=Number(usageRecordField(record,['auto_continue_rounds'])||0);
  var parts=[];
  var title=[];
  if(position>0){{ parts.push('排队 #'+position); title.push('上游队列位置 '+position); }}
  if(rounds>0){{ parts.push('续写 '+rounds); title.push('自动续写 '+rounds+' 轮'); }}
  if(!parts.length) return null;
  return {{text:parts.join(' · '),title:title.join('；')}};
}}
function usageRouteMeta(record){{
  var rawProduct=usageRecordField(record,['product','provider_product','upstream_product']);
  var product=rawProduct===null?'':String(rawProduct).trim().toLowerCase();
  var productLabel=product==='traework'?'TraeWork':(product==='trae'?'Trae':(rawProduct===null?'--':String(rawProduct)));
  var actual=usageRecordField(record,['actual_endpoint','actual_mode','upstream_endpoint']);
  var requested=usageRecordField(record,['requested_endpoint','requested_mode']);
  var publicEndpoint=usageRecordField(record,['endpoint']);
  var endpoint=actual||requested||publicEndpoint;
  var detail=endpoint===null?'--':String(endpoint);
  var title=[];
  if(productLabel!=='--') title.push('产品: '+productLabel);
  if(actual!==null) title.push('实际端点: '+String(actual));
  if(requested!==null && String(requested)!==String(actual||'')) title.push('请求端点: '+String(requested));
  if(publicEndpoint!==null && String(publicEndpoint)!==String(actual||requested||'')) title.push('入口: '+String(publicEndpoint));
  return {{
    text:escapeHtml(productLabel)+'<div class="usage-route-endpoint">'+escapeHtml(detail)+'</div>',
    title:title.join('；')||'未记录产品/端点',
    unknown:productLabel==='--' && endpoint===null
  }};
}}
function usageCreditFingerprint(records){{
  if(!Array.isArray(records)) return '';
  return records.map(function(record){{
    if(!record||typeof record!=='object') return '';
    var amount=creditNumber(record.credits_consumed);
    var source=String(record.credits_source||'').toLowerCase();
    var status=String(record.status||'completed').toLowerCase();
    if(amount===null||amount<=0||status==='error'||status==='cancelled'||source==='pending'||source==='unavailable'||source==='unknown') return '';
    return String(record.request_id||record.id||record.timestamp||'')+'='+amount.toFixed(4)+'@'+String(record.account_id||'');
  }}).filter(Boolean).join('|');
}}
function scheduleUsageCreditCalibration(records){{
  var fingerprint=usageCreditFingerprint(records);
  if(!fingerprint) return;
  if(!creditUiState.usageFingerprintReady){{
    creditUiState.usageFingerprint=fingerprint;
    creditUiState.usageFingerprintReady=true;
    return;
  }}
  if(fingerprint===creditUiState.usageFingerprint) return;
  creditUiState.usageFingerprint=fingerprint;
  // A settled usage row means the effective balance may have changed.  One
  // debounced bulk query refreshes all rows and the aggregate without
  // turning the five-second history poll into an upstream polling loop.
  clearTimeout(creditUiState.reconcileTimer);
  creditUiState.reconcileTimer=setTimeout(function(){{
    creditUiState.reconcileTimer=null;
    creditsRefreshAll({{silent:true,reason:'usage'}});
  }},1200);
}}
async function refreshUsage() {{
  if(usageRefreshing) return;
  usageRefreshing=true;
  var tbody=document.getElementById('usage-records-body');
  var empty=document.getElementById('usage-empty');
  var msg=document.getElementById('usage-msg');
  try {{
    var result=await requestJSON('/api/usage/records',{{method:'GET'}},15000);
    if(!result.ok) throw new Error(apiError(result.data,result.status));
    var records=usageRecordsFromPayload(result.data);
    if(records===null) throw new Error(apiError(result.data,result.status));
    scheduleUsageCreditCalibration(records);
    if(!tbody) return;
    if(records.length===0) {{
      tbody.innerHTML='';
      if(empty) empty.style.display='block';
    }} else {{
      if(empty) empty.style.display='none';
      tbody.innerHTML=records.map(function(record) {{
        var stamp=Number(record.timestamp||0);
        var when=stamp?new Date(stamp*1000).toLocaleString():'--';
        var account=record.account_id?String(record.account_id).slice(-12):'--';
        var model=record.model||'--';
         var route=usageRouteMeta(record);
         var reasoning=usageReasoningMeta(record);
         var context=usageContextMeta(record);
         var maxMode=usageMaxMeta(record);
         var tool=usageToolMeta(record);
         var transport=usageTransportMeta(record);
        var input=Number(record.input_tokens!==undefined?record.input_tokens:(record.prompt_tokens||0));
        var output=Number(record.output_tokens!==undefined?record.output_tokens:(record.completion_tokens||0));
        var total=Number(record.total_tokens!==undefined?record.total_tokens:(input+output));
        var tokenText=record.tokens_source==='unknown'?'--':(input+' / '+output+' / '+total);
        var source=record.credits_source||'unknown';
        var credits=record.credits_consumed;
        var creditText=credits===null||credits===undefined
          ?(source==='pending'?'结算中':'--')
          :Number(credits).toFixed(2);
        var status=record.status||'completed';
        var statusText=status==='completed'?'完成':(status==='cancelled'?'已取消':(status==='error'?'失败':status));
        var badge=status==='completed'?'badge-ok':(status==='error'?'badge-expired':'badge-none');
        return '<tr>'
          + '<td>'+escapeHtml(when)+'</td>'
          + '<td><code>'+escapeHtml(account)+'</code></td>'
          + '<td class="usage-meta usage-route'+(route.unknown?' unknown':'')+'" title="'+escapeHtml(route.title)+'">'+route.text+'</td>'
          + '<td class="usage-model-cell"><div title="'+escapeHtml(model)+'">'+escapeHtml(model)+'</div><div class="usage-model-flags">'+usageModelFlags(maxMode,tool)+'</div></td>'
          + '<td class="usage-meta'+(reasoning.unknown?' unknown':'')+'" title="'+escapeHtml(reasoning.title)+'">'+reasoning.text+'</td>'
          + '<td class="usage-meta'+(context.unknown?' unknown':'')+'" title="'+escapeHtml(context.title)+'">'+escapeHtml(context.text)+'</td>'
          + '<td class="usage-meta'+(maxMode.unknown?' unknown':'')+'" title="'+escapeHtml(maxMode.title)+'">'+escapeHtml(maxMode.text)+'</td>'
          + '<td class="usage-meta usage-tool'+(tool.unknown?' unknown':'')+'" title="'+escapeHtml(tool.title)+'">'+escapeHtml(tool.text)+'</td>'
          + '<td class="numeric">'+escapeHtml(tokenText)+'</td>'
          + '<td class="numeric" title="'+escapeHtml(source)+'">'+escapeHtml(creditText)+'</td>'
          + '<td class="usage-status"><span class="badge '+badge+'">'+escapeHtml(statusText)+'</span></td>'
          + '<td class="usage-meta usage-transport"'+(transport?(' title="'+escapeHtml(transport.title)+'"'):'')+'>'+(transport?escapeHtml(transport.text):'--')+'</td>'
          + '</tr>';
      }}).join('');
    }}
    var updated=document.getElementById('usage-updated');
    if(updated) updated.textContent='更新于 '+new Date().toLocaleTimeString()+' · '+records.length+' 条';
    if(msg){{ msg.textContent=''; msg.className='msg'; }}
  }} catch(e) {{
    if(msg){{ msg.textContent=String(e); msg.className='msg msg-err'; }}
  }} finally {{ usageRefreshing=false; }}
}}
setInterval(refreshUsage, 5000);
refreshUsage();
// Overview dashboard. Data comes from /api/overview (persistent daily
// buckets); the chart reuses its bar nodes so refreshes animate heights.
var ovData=null, ovRange=14, ovDay='', ovLoading=false;
function fmtInt(n){{ return Number(n||0).toLocaleString('en-US'); }}
function fmtCompact(n){{
  n=Number(n||0); var a=Math.abs(n);
  if(a>=1e9) return (n/1e9).toFixed(2)+'B';
  if(a>=1e6) return (n/1e6).toFixed(2)+'M';
  if(a>=1e4) return (n/1e3).toFixed(1)+'K';
  return fmtInt(n);
}}
function fmtPct(part,whole){{ return whole?(part/whole*100).toFixed(1)+'%':'--'; }}
function fmtUptime(s){{
  s=Math.max(0,Math.floor(s||0));
  var d=Math.floor(s/86400), h=Math.floor(s%86400/3600), m=Math.floor(s%3600/60);
  return d?(d+' 天 '+h+' 小时'):(h?(h+' 小时 '+m+' 分'):(m+' 分'));
}}
function setOv(id,value,title){{
  var el=document.getElementById(id); if(!el) return;
  el.textContent=value;
  if(title!==undefined) el.title=title;
}}
function setOvPill(id,on,label){{
  var el=document.getElementById(id); if(!el) return;
  el.innerHTML='<span class="state-pill'+(on?' on':'')+'">'+escapeHtml(label)+'</span>';
}}
function niceCeil(v){{
  if(v<=0) return 1;
  var p=Math.pow(10,Math.floor(Math.log10(v))), f=v/p;
  return (f<=1?1:f<=2?2:f<=5?5:10)*p;
}}
function ovRows(){{ return ovData?ovData.usage.daily.slice(0,ovRange).reverse():[]; }}
function ovFind(day){{
  if(!ovData) return null;
  var rows=ovData.usage.daily;
  for(var i=0;i<rows.length;i++){{ if(rows[i].date===day) return rows[i]; }}
  return null;
}}
async function refreshOverview(){{
  if(ovLoading) return;
  ovLoading=true;
  var msg=document.getElementById('ov-msg');
  try{{
    var r=await requestJSON('/api/overview?days=30',{{method:'GET'}},15000);
    if(!r.ok||!r.data||!r.data.success) throw new Error(apiError(r.data,r.status));
    ovData=r.data;
    renderOverview();
    setOv('ov-updated','更新于 '+new Date().toLocaleTimeString());
    if(msg){{ msg.textContent=''; msg.className='msg'; }}
  }}catch(e){{
    if(msg){{ msg.textContent=String(e&&e.message||e); msg.className='msg msg-err'; }}
  }}finally{{ ovLoading=false; }}
}}
function renderOverview(){{
  if(!ovData) return;
  var u=ovData.usage, t=u.totals, td=u.today, a=ovData.accounts, s=ovData.service;
  setOv('ov-total',fmtCompact(t.total_tokens),fmtInt(t.total_tokens));
  setOv('ov-total-sub',u.first_day?(fmtInt(t.requests)+' 次请求 · 自 '+u.first_day):'暂无记录');
  setOv('ov-input',fmtCompact(t.input_tokens),fmtInt(t.input_tokens));
  setOv('ov-input-sub','占总量 '+fmtPct(t.input_tokens,t.total_tokens));
  setOv('ov-output',fmtCompact(t.output_tokens),fmtInt(t.output_tokens));
  setOv('ov-output-sub','占总量 '+fmtPct(t.output_tokens,t.total_tokens));
  setOv('ov-cache',fmtCompact(t.cached_tokens),fmtInt(t.cached_tokens));
  setOv('ov-cache-sub','输入命中率 '+fmtPct(t.cached_tokens,t.input_tokens));
  setOv('ov-today',fmtCompact(td.total_tokens),fmtInt(td.total_tokens));
  setOv('ov-today-sub',fmtInt(td.requests)+' 次请求'+(td.failed?' · '+td.failed+' 失败':''));
  setOv('ov-accounts',a.valid+' / '+a.total);
  setOv('ov-accounts-sub','今日签到 '+a.checked_in+' / '+a.total);
  setOvPill('ov-token',s.token_ok,s.token_ok?'有效':'失效');
  setOv('ov-version','v'+s.version);
  setOv('ov-uptime',fmtUptime(s.uptime_seconds));
  setOv('ov-upstream',s.auto_route?'自动路由':s.upstream_mode);
  setOvPill('ov-polling',s.polling,s.polling?(s.polling_mode==='credit-priority'?'积分优先':'顺序轮询'):'关');
  setOvPill('ov-max',s.max_mode,s.max_mode?'开':'关');
  var ac=s.auto_checkin||{{}};
  setOvPill('ov-autocheckin',ac.enabled,ac.enabled?('每日 '+(ac.time||'')):'关');
  setOv('ov-inflight',String(s.in_flight||0));
  var cr=a.credits_remaining;
  setOv('ov-credits',cr==='unlimited'?'无限':(cr===null||cr===undefined?'--':Number(cr).toFixed(2)));
  updateCreditCalibrationMeta(a);
  renderOverviewChart();
  renderOverviewDays();
  showOverviewDay(ovDay);
}}
function renderOverviewChart(){{
  var chart=document.getElementById('ov-chart'), axis=document.getElementById('ov-axis');
  if(!chart||!axis) return;
  var rows=ovRows(), max=0;
  rows.forEach(function(r){{ max=Math.max(max,(r.input_tokens||0)+(r.output_tokens||0)); }});
  var scale=niceCeil(max);
  setOv('ov-y-max',max?fmtCompact(scale):'0');
  setOv('ov-y-mid',max?fmtCompact(scale/2):'0');
  var rebuild=chart.children.length!==rows.length;
  if(rebuild){{
    chart.innerHTML=rows.map(function(){{
      return '<div class="bar-col" tabindex="0" role="button"><div class="bar-stack">'
        +'<span class="bar-seg seg-out"></span><span class="bar-seg seg-in"></span><span class="bar-seg seg-cache"></span>'
        +'</div></div>';
    }}).join('');
    var step=Math.ceil(rows.length/8);
    axis.innerHTML=rows.map(function(r,i){{
      var show=(rows.length-1-i)%step===0;
      return '<span>'+(show?escapeHtml(r.date.slice(5)):'')+'</span>';
    }}).join('');
  }}
  function apply(){{
    rows.forEach(function(r,i){{
      var col=chart.children[i]; if(!col) return;
      var cache=Math.min(r.cached_tokens||0,r.input_tokens||0);
      var segs=col.firstChild.children;
      segs[0].style.height=((r.output_tokens||0)/scale*100)+'%';
      segs[1].style.height=(((r.input_tokens||0)-cache)/scale*100)+'%';
      segs[2].style.height=(cache/scale*100)+'%';
      col.setAttribute('data-day',r.date);
      col.setAttribute('aria-label',r.date+' 总计 '+fmtInt(r.total_tokens)+' Token');
      col.classList.toggle('sel',r.date===ovDay);
    }});
  }}
  // Fresh bars start at zero height and grow on the next frame.
  if(rebuild) requestAnimationFrame(function(){{ requestAnimationFrame(apply); }}); else apply();
}}
function renderOverviewDays(){{
  var body=document.getElementById('ov-days-body'), foot=document.getElementById('ov-days-foot');
  if(!body||!foot||!ovData) return;
  var hide=document.getElementById('ov-hide-empty');
  var today=ovData.usage.today.date;
  var rows=ovData.usage.daily.slice(0,ovRange);
  var sum={{requests:0,failed:0,input_tokens:0,output_tokens:0,cached_tokens:0,total_tokens:0,max_mode_requests:0,max_mode_enabled_requests:0,tool_requests:0,tool_calls:0,credits:0}};
  rows.forEach(function(r){{ for(var k in sum) sum[k]+=Number(r[k]||0); }});
  var shown=rows.filter(function(r){{ return !(hide&&hide.checked&&!r.requests); }});
  body.innerHTML=shown.length?shown.map(function(r){{
    var cls=(r.requests?'':'zero')+(r.date===ovDay?' sel':'');
    var top=(r.models||[]).slice(0,3).map(function(m){{ return m.model+' '+fmtInt(m.total_tokens); }}).join('，');
    return '<tr class="'+cls+'" data-day="'+escapeHtml(r.date)+'" title="'+escapeHtml(top)+'">'
      +'<td>'+escapeHtml(r.date)+(r.date===today?'<span class="badge badge-active">今日</span>':'')+'</td>'
      +'<td>'+fmtInt(r.requests)+'</td>'
      +'<td>'+fmtInt(r.failed)+'</td>'
      +'<td>'+fmtInt(r.input_tokens)+'</td>'
      +'<td>'+fmtInt(r.output_tokens)+'</td>'
      +'<td>'+fmtInt(r.cached_tokens)+'</td>'
      +'<td>'+fmtInt(r.total_tokens)+'</td>'
      +'<td>'+fmtInt(r.max_mode_requests||0)+' / '+fmtInt(r.max_mode_enabled_requests||0)+'</td>'
      +'<td>'+fmtInt(r.tool_requests||0)+' / '+fmtInt(r.tool_calls||0)+'</td>'
      +'<td>'+Number(r.credits||0).toFixed(2)+'</td>'
      +'</tr>';
  }}).join(''):'<tr class="zero"><td colspan="10">所选范围内没有请求</td></tr>';
  foot.innerHTML='<td>合计 '+ovRange+' 天</td>'
    +'<td>'+fmtInt(sum.requests)+'</td><td>'+fmtInt(sum.failed)+'</td>'
    +'<td>'+fmtInt(sum.input_tokens)+'</td><td>'+fmtInt(sum.output_tokens)+'</td>'
    +'<td>'+fmtInt(sum.cached_tokens)+'</td><td>'+fmtInt(sum.total_tokens)+'</td>'
    +'<td>'+fmtInt(sum.max_mode_requests)+' / '+fmtInt(sum.max_mode_enabled_requests)+'</td>'
    +'<td>'+fmtInt(sum.tool_requests)+' / '+fmtInt(sum.tool_calls)+'</td>'
    +'<td>'+sum.credits.toFixed(2)+'</td>';
  var active=rows.filter(function(r){{ return r.requests; }}).length;
  setOv('ov-days-meta',active+' / '+ovRange+' 天有请求');
}}
function showOverviewDay(day){{
  var box=document.getElementById('ov-readout');
  if(!box||!ovData) return;
  var r=ovFind(day)||ovData.usage.today;
  var label=r.date===ovData.usage.today.date?r.date+' 今日':r.date;
  function item(sw,name,value){{
    return '<span class="readout-item">'+(sw?'<span class="swatch '+sw+'"></span>':'')+name+' <strong>'+value+'</strong></span>';
  }}
  box.innerHTML='<span class="readout-date">'+escapeHtml(label)+'</span>'
    +item('sw-in','输入',fmtInt(r.input_tokens))
    +item('sw-cache','缓存',fmtInt(r.cached_tokens))
    +item('sw-out','输出',fmtInt(r.output_tokens))
    +item('','总计',fmtInt(r.total_tokens))
    +item('','请求',fmtInt(r.requests));
}}
function selectOverviewDay(day){{
  ovDay=day||'';
  document.querySelectorAll('#ov-chart .bar-col').forEach(function(c){{ c.classList.toggle('sel',c.getAttribute('data-day')===ovDay); }});
  document.querySelectorAll('#ov-days-body tr[data-day]').forEach(function(tr){{
    var on=tr.getAttribute('data-day')===ovDay;
    tr.classList.toggle('sel',on);
    if(on&&tr.scrollIntoView) tr.scrollIntoView({{block:'nearest'}});
  }});
  showOverviewDay(ovDay);
}}
function setOverviewRange(value){{
  ovRange=Number(value)||14;
  renderOverviewChart();
  renderOverviewDays();
}}
(function(){{
  var chart=document.getElementById('ov-chart');
  if(chart){{
    var pick=function(e){{ var c=e.target.closest&&e.target.closest('.bar-col'); return c?c.getAttribute('data-day'):null; }};
    chart.addEventListener('mouseover',function(e){{ var d=pick(e); if(d) showOverviewDay(d); }});
    chart.addEventListener('focusin',function(e){{ var d=pick(e); if(d) showOverviewDay(d); }});
    chart.addEventListener('mouseleave',function(){{ showOverviewDay(ovDay); }});
    chart.addEventListener('click',function(e){{ var d=pick(e); if(d) selectOverviewDay(d); }});
    chart.addEventListener('keydown',function(e){{
      if(e.key!=='Enter'&&e.key!==' ') return;
      var d=pick(e); if(d){{ e.preventDefault(); selectOverviewDay(d); }}
    }});
  }}
  var body=document.getElementById('ov-days-body');
  if(body) body.addEventListener('click',function(e){{
    var tr=e.target.closest&&e.target.closest('tr[data-day]');
    if(tr) selectOverviewDay(tr.getAttribute('data-day'));
  }});
}})();
setInterval(function(){{
  if(document.hidden) return;
  // /api/overview reads the persisted credit projection and never calls the
  // provider.  Keep the calibration badge and aggregate total fresh on every
  // page without turning this timer into an upstream credit probe.
  refreshOverview();
}},10000);
refreshOverview();
window.addEventListener('message',function(ev){{
  if (!ev.data||ev.data.type!=='trae-relay-web-login') return;
  if (state.traceId&&ev.data.loginTraceId!==state.traceId) return;
  if (ev.data.success) {{ showMsg('auth-msg','授权成功，凭证已写入服务器',true); setTimeout(function(){{ location.reload(); }},800); }}
  else {{ showMsg('auth-msg',ev.data.error||'授权失败',false); }}
  document.getElementById('loading').style.display='none';
  document.getElementById('auth-btn').disabled=false;
  if (state.win&&!state.win.closed) state.win.close();
}});
let messageTimers = {{}};
let toastTimer = null;
function showToast(title,text,ok,timeout){{
  var toast=document.getElementById('toast');
  var titleEl=document.getElementById('toast-title');
  var textEl=document.getElementById('toast-text');
  if(!toast) return;
  clearTimeout(toastTimer);
  titleEl.textContent=title||'';
  textEl.textContent=text||'';
  toast.className='toast visible '+(ok?'ok':'error');
  var delay=timeout===undefined?(ok?3000:9000):timeout;
  if(delay>0) toastTimer=setTimeout(function(){{ toast.className='toast'; }},delay);
}}
function showMsg(id,text,ok,timeout){{
  var el=document.getElementById(id);
  if(!el) return;
  clearTimeout(messageTimers[id]);
  if(!text){{ el.textContent=''; el.className='msg'; return; }}
  el.textContent=text;
  el.className='msg '+(ok?'msg-ok':'msg-err');
  var delay=timeout===undefined?(ok?3000:9000):timeout;
  if(delay>0) messageTimers[id]=setTimeout(function(){{ el.textContent=''; el.className='msg'; }},delay);
  showToast(ok?'成功':'操作失败',text,ok,delay);
}}
function endpointDiagnosticText(data){{
  if(!data||(!data.fallback_used&&!data.fallback_reason&&!data.failed_endpoint)) return '';
  var requested=String(data.requested_endpoint||data.requested_mode||'').trim();
  var actual=String(data.actual_endpoint||data.actual_mode||'').trim();
  var target=String(data.fallback_target||actual||'').trim();
  var failed=String(data.failed_endpoint||requested||'').trim();
  var reason=String(data.fallback_reason||'').trim();
  var parts=[];
  if(failed) parts.push('失败端点='+failed);
  if(target) parts.push('回落目标='+target);
  if(reason) parts.push('原因='+reason);
  return parts.join(' | ');
}}
function showEndpointDiagnostic(data,title){{
  var text=endpointDiagnosticText(data);
  if(!text) return '';
  showToast(
    title||'端点诊断',
    text,
    !data.fallback_used&&!data.fallback_reason,
    9000
  );
  return text;
}}
function escapeHtml(value){{
  return String(value===undefined||value===null?'':value).replace(/[&<>"']/g,function(ch){{ return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]; }});
}}
function responseCode(data){{
  if(!data) return '';
  var body=data.data||data;
  return body && body.code!==undefined && body.code!==null ? String(body.code) : '';
}}
function apiError(data,status){{
  var code=responseCode(data);
  var message=(data&&data.error)||(data&&data.message)||(data&&data.data&&(data.data.message||data.data.error))||('HTTP '+(status||'未知'));
  var prefix=status&&status!==200?'HTTP '+status:'';
  if(code) prefix+=(prefix?'，':'')+'业务码 '+code;
  return (prefix?prefix+'：':'')+String(message);
}}
async function requestJSON(url,options,timeoutMs){{
  var controller=new AbortController();
  var timeout=setTimeout(function(){{ controller.abort(); }},timeoutMs||45000);
  var requestOptions=Object.assign({{cache:'no-store'}},options||{{}},{{signal:controller.signal}});
  try{{
    var response=await fetch(url,requestOptions);
    var text=await response.text();
    var data={{}};
    if(text){{
      try{{ data=JSON.parse(text); }}catch(e){{ data={{error:'服务返回了无效 JSON'}}; }}
    }}
    return {{ok:response.ok,status:response.status,data:data}};
  }}catch(e){{
    if(e&&e.name==='AbortError') throw new Error('请求超时，请检查上游连接后重试');
    throw new Error('网络请求失败：'+String(e));
  }}finally{{ clearTimeout(timeout); }}
}}
async function postJSON(url,payload,timeoutMs){{
  try{{
    var result=await requestJSON(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload||{{}})}},timeoutMs||45000);
    var d=result.data;
    if(!d||typeof d!=='object'||Array.isArray(d)) d={{}};
    d._http_status=result.status; d._http_ok=result.ok;
    return d;
  }}catch(e){{ return {{success:false,error:String(e),_http_status:0,_http_ok:false}}; }}
}}
let checkinGlobalBusy=false;
let checkinAccountBusy=new Set();
function syncCheckinBusyUI(){{
  var anyBusy=checkinGlobalBusy||checkinAccountBusy.size>0;
  ['checkin-status-refresh-btn','credits-refresh-btn','checkin-claim-btn'].forEach(function(id){{
    var el=document.getElementById(id); if(el){{ el.disabled=anyBusy; el.classList.toggle('busy',anyBusy); }}
  }});
  document.querySelectorAll('[data-action="checkin"]').forEach(function(el){{
    var row=el.closest('tr[data-account-id]');
    var id=row&&row.getAttribute('data-account-id');
    var busy=checkinGlobalBusy||checkinAccountBusy.has(String(id||''));
    el.disabled=busy;
    el.classList.toggle('busy',busy);
  }});
  var indicator=document.getElementById('checkin-busy');
  if(indicator){{ indicator.classList.toggle('visible',anyBusy); indicator.textContent=anyBusy?'正在查询/签到...':''; }}
}}
function setBusy(busy){{
  checkinGlobalBusy=!!busy;
  syncCheckinBusyUI();
}}
function setAccountCheckinBusy(id,busy){{
  id=String(id);
  if(busy) checkinAccountBusy.add(id); else checkinAccountBusy.delete(id);
  var row=document.getElementById('row-'+id);
  if(row) row.classList.toggle('checkin-row-busy',busy);
  var detail=document.getElementById('checkin-detail-'+id);
  if(detail&&busy){{ detail.textContent='正在签到该账号...'; detail.style.color=''; }}
  syncCheckinBusyUI();
}}
function setCredits(id,data){{
  var value=creditSnapshot(data);
  var el=document.getElementById('general-credits-'+id);
  if(!el||!value) return;
  creditUiState.accounts[String(id)]=Object.assign({{}},creditUiState.accounts[String(id)]||{{}},value);
  var row=document.getElementById('row-'+id);
  if(row){{
    var remaining=creditNumber(value.remaining);
    var official=creditNumber(value.official_remaining);
    var local=creditNumber(value.local_debit);
    if(remaining!==null) row.dataset.creditRemaining=String(remaining);
    if(official!==null) row.dataset.creditOfficial=String(official);
    if(local!==null) row.dataset.creditLocalDebit=String(local);
    if(value.reconcile_status) row.dataset.creditStatus=String(value.reconcile_status);
    if(value.credits_reconciled_at) row.dataset.creditReconciledAt=String(value.credits_reconciled_at);
  }}
  if(value.unlimited) el.textContent='☆ 无限';
  else if(value.remaining!==undefined&&value.remaining!==null) el.textContent='剩'+Number(value.remaining).toFixed(2)+'/总'+(value.total_limit===undefined?'?':Number(value.total_limit).toFixed(2));
  else if(value.official_remaining!==undefined&&value.official_remaining!==null) el.textContent='剩'+Number(value.official_remaining).toFixed(2)+'/总'+(value.total_limit===undefined?'?':Number(value.total_limit).toFixed(2));
  else el.textContent='-';
  updateCreditCalibrationMeta(value);
}}
function setCheckinState(id,checked,detail,error){{
  var el=document.getElementById('checkin-'+id), detailEl=document.getElementById('checkin-detail-'+id);
  if(el){{
    var badge=checked===true?'<span class="badge badge-ok">已签到</span>':(checked===false?'<span class="badge badge-active">未签到</span>':'<span class="badge badge-none">未知</span>');
    el.innerHTML=badge;
  }}
  if(detailEl){{ detailEl.textContent=error||detail||''; detailEl.style.color=error?'var(--danger)':''; }}
}}
function updateAccountRow(account){{
  if(!account||!account.id) return;
  updateAccountCreditsRow(account);
  updateAccountCheckinRow(account);
}}
function updateAccountCreditsRow(account){{
  if(!account||!account.id) return;
  setCredits(account.id,account);
  refreshStats();
}}
function updateAccountCheckinRow(account){{
  if(!account||!account.id) return;
  var payload=account.data||account.checkin||{{}};
  var code=payload&&payload.code!==undefined?'业务码 '+payload.code:'';
  var detail=account.error||code||'';
  setCheckinState(account.id,account.checked_in,detail,account.error);
  var labelEl=document.getElementById('label-'+account.id);
  if(labelEl&&account.label) labelEl.textContent=account.label;
  var row=document.getElementById('row-'+account.id);
  if(row) row.classList.toggle('row-failed',!!account.error||account.success===false);
  refreshStats();
}}
function setActiveAccount(id,account){{
  document.querySelectorAll('tr[data-account-id]').forEach(function(row){{
    var active=row.getAttribute('data-account-id')===String(id);
    row.classList.toggle('active-row',active);
    var badge=row.querySelector('[data-account-active]');
    if(badge) badge.hidden=!active;
    var button=row.querySelector('[data-action="switch-account"]');
    if(button) button.disabled=active;
  }});
  var user=document.getElementById('active-user-id');
  if(user){{
    var userId=account&&(account.user_id||account.id)||id||'';
    user.textContent=userId?'用户: '+userId:'';
    user.hidden=!userId;
  }}
}}
function setSwitchBusy(busy){{
  document.querySelectorAll('[data-action="switch-account"]').forEach(function(button){{
    if(busy){{ button.disabled=true; }}
    else {{
      var row=button.closest('tr[data-account-id]');
      button.disabled=!!(row&&row.classList.contains('active-row'));
    }}
  }});
}}
function updateCheckinSummary(accounts,action){{
  var list=Array.isArray(accounts)?accounts:[];
  var ok=list.filter(function(a){{ return a.checked_in===true; }}).length;
  var failed=list.filter(function(a){{ return !!a.error||a.success===false; }}).length;
  var summary=document.getElementById('checkin-summary');
  if(summary) summary.textContent=(action||'已更新')+'：'+ok+' 已签到 / '+list.length+' 个账号'+(failed?'，'+failed+' 个异常':'');
  var updated=document.getElementById('checkin-updated');
  if(updated) updated.textContent='更新于 '+new Date().toLocaleTimeString();
  return {{ok:ok,failed:failed,total:list.length}};
}}
function updateVisibleCheckinSummary(action){{
  var rows=Array.from(document.querySelectorAll('tr[data-account-id]'));
  var ok=rows.filter(function(row){{
    var state=row.querySelector('.checkin-state');
    return !!state&&state.textContent.trim()==='已签到';
  }}).length;
  var failed=rows.filter(function(row){{ return row.classList.contains('row-failed'); }}).length;
  var summary=document.getElementById('checkin-summary');
  if(summary) summary.textContent=(action||'已更新')+'：'+ok+' 已签到 / '+rows.length+' 个账号'+(failed?'，'+failed+' 个异常':'');
  var updated=document.getElementById('checkin-updated');
  if(updated) updated.textContent='更新于 '+new Date().toLocaleTimeString();
  return {{ok:ok,failed:failed,total:rows.length}};
}}
function checkinFailureText(account){{
  var code=responseCode(account&&account.data);
  var message=(account&&account.error)||((account&&account.data&&account.data.message)||'未知错误');
  return (account&&account.label||account&&account.id||'账号')+'：'+(code?'业务码 '+code+'，':'')+message;
}}
const CONN_PRESET = ['glm-5.3','glm-5.2','kimi-k2.7-code','kimi-k3','qwen3.8-max','DeepSeek-V4-Pro-Official','DeepSeek-V4-Flash-Official'];
function fillConnPreset(){{
  document.getElementById('conn-models').value = CONN_PRESET.join(', ');
}}
function connModelList(){{
  var raw = document.getElementById('conn-models').value || '';
  // Keep the newline escape intact when this script is rendered from the
  // Python f-string.  A literal line break makes the whole inline script
  // invalid JavaScript and disables every account/usage control on the page.
  var items = raw.split(/[\\n,]+/).map(function(s){{ return s.trim(); }}).filter(Boolean);
  return items.length ? items : CONN_PRESET.slice();
}}
function connModes(){{
  var modes = [];
  if(document.getElementById('conn-mode-text').checked) modes.push('text');
  if(document.getElementById('conn-mode-tool').checked) modes.push('tool');
  return modes;
}}
function connRow(model, mode){{
  var tr = document.createElement('tr');
  var label = mode === 'tool' ? '工具调用' : '文本回复';
  [model, label, '排队中', '-', '-'].forEach(function(text, idx){{
    var td = document.createElement('td');
    td.textContent = text;
    if(idx === 2) td.className = 'conn-status pending';
    if(idx === 3) td.className = 'conn-time';
    if(idx === 4) td.className = 'conn-detail';
    tr.appendChild(td);
  }});
  return tr;
}}
async function runConnTest(){{
  var btn = document.getElementById('conn-run-btn');
  var table = document.getElementById('conn-table');
  var tbody = document.getElementById('conn-tbody');
  var summary = document.getElementById('conn-summary');
  var models = connModelList();
  var modes = connModes();
  if(!modes.length){{ showMsg('conn-msg','请至少选择一种测试内容',false); return; }}
  var timeout = parseInt(document.getElementById('conn-timeout').value,10) || 120;
  var endpoint = document.getElementById('conn-endpoint').value || 'remote';
  var effort = document.getElementById('conn-effort').value || '';
  var thinking = document.getElementById('conn-thinking').checked;
  var maxMode = document.getElementById('conn-max').checked;
  btn.disabled = true;
  table.hidden = false;
  tbody.innerHTML = '';
  showMsg('conn-msg','');
  var jobs = [];
  models.forEach(function(model){{
    modes.forEach(function(mode){{
      var tr = connRow(model, mode);
      tbody.appendChild(tr);
      jobs.push({{model: model, mode: mode, row: tr}});
    }});
  }});
  var passed = 0, done = 0, endpointWarnings = [];
  summary.textContent = '0/' + jobs.length + ' 完成';
  for(var i = 0; i < jobs.length; i++){{
    var job = jobs[i];
    var statusCell = job.row.querySelector('.conn-status');
    var timeCell = job.row.querySelector('.conn-time');
    var detailCell = job.row.querySelector('.conn-detail');
    statusCell.textContent = '测试中...';
    statusCell.className = 'conn-status pending';
    try{{
      var d = await postJSON('/api/model-test',
        {{model: job.model, mode: job.mode, timeout: timeout, endpoint: endpoint,
          reasoning_effort: effort, thinking: thinking, max_mode: maxMode,
          // This page compares one endpoint at a time.  Do not hide a native
          // endpoint failure behind the API runtime's production fallback.
          disable_fallback: true}}, (timeout + 20) * 1000);
      timeCell.textContent = (d.elapsed_ms !== undefined ? d.elapsed_ms + ' ms' : '-');
      if(d.success){{
        passed++;
        statusCell.textContent = '通过';
        statusCell.className = 'conn-status ok';
        var endpointDiagnostic=endpointDiagnosticText(d);
        if(job.mode === 'tool'){{
          var names = (d.tool_calls || []).map(function(c){{ return c.name; }}).join(', ');
          detailCell.textContent = (d.auto_route ? 'auto -> ' : '') + 'endpoint=' + (d.actual_endpoint || d.actual_mode || '-')
            + (d.fallback_used ? ' (fallback)' : '') + ' | tool=' + (names || '-')
            + (d.provider_model_name ? ' | ' + d.provider_model_name : '');
        }} else {{
          detailCell.textContent = (d.auto_route ? 'auto -> ' : '') + 'endpoint=' + (d.actual_endpoint || d.actual_mode || '-')
            + (d.fallback_used ? ' (fallback)' : '') + ' | ' + JSON.stringify(d.reply || '')
            + (d.provider_model_name ? ' | ' + d.provider_model_name : '');
        }}
        if(endpointDiagnostic) detailCell.textContent += ' | ' + endpointDiagnostic;
        if(endpointDiagnostic) endpointWarnings.push(endpointDiagnostic);
        var extra = [];
        if(d.requested_reasoning_effort){{
          extra.push('强度=' + d.requested_reasoning_effort + ' -> '
            + (d.reasoning_effort_applied || ('未生效: ' + (d.reasoning_effort_note || '模型不支持'))));
        }}
        if(maxMode){{
          extra.push('Max=' + (d.max_mode_applied ? '已生效 ' + formatContext(d.max_context_tokens) : '未生效'));
        }}
        if(d.usage && d.usage.reasoning_tokens !== undefined && d.usage.reasoning_tokens !== null){{
          extra.push('reasoning_tokens=' + d.usage.reasoning_tokens);
        }}
        if(extra.length) detailCell.textContent += ' | ' + extra.join(' | ');
        if(d.reasoning){{
          var rd = document.createElement('div');
          rd.className = 'conn-reasoning';
          rd.textContent = '思考: ' + d.reasoning;
          detailCell.appendChild(rd);
        }}
      }} else {{
        statusCell.textContent = '失败';
        statusCell.className = 'conn-status fail';
        var endpointDiagnostic=endpointDiagnosticText(d);
        detailCell.textContent = String(d.error || 'unknown')
          + (endpointDiagnostic ? ' | ' + endpointDiagnostic : '');
        if(endpointDiagnostic) endpointWarnings.push(endpointDiagnostic);
      }}
    }}catch(e){{
      statusCell.textContent = '失败';
      statusCell.className = 'conn-status fail';
      timeCell.textContent = '-';
      detailCell.textContent = String(e);
    }}
    done++;
    summary.textContent = done + '/' + jobs.length + ' 完成，通过 ' + passed;
  }}
  btn.disabled = false;
  var allOk = passed === jobs.length;
  showMsg('conn-msg', '测试结束：' + passed + '/' + jobs.length + ' 通过', allOk);
  if(endpointWarnings.length) showToast('端点回落/失败诊断',
    endpointWarnings.slice(0,3).join('\\n'),false,12000);
}}
async function refreshModels(){{
  if(refreshModels.running) return;
  refreshModels.running=true;
  var refreshButton=document.getElementById('models-refresh-btn');
  if(refreshButton) refreshButton.disabled=true;
  var meta=document.getElementById('model-list-meta');
  var groupNames=['remote','traecode','traework'];
  var columns={{}};
  groupNames.forEach(function(group){{
    columns[group]=document.getElementById('models-col-'+group);
    var count=document.getElementById('models-count-'+group);
    if(count) count.textContent='0';
    if(columns[group]) columns[group].innerHTML='<div class="model-column-empty">加载中...</div>';
  }});
  var fallbackPanel=document.getElementById('models-fallback');
  var fallbackColumn=document.getElementById('models-col-fallback');
  var fallbackCount=document.getElementById('models-count-fallback');
  if(fallbackPanel) fallbackPanel.hidden=true;
  if(fallbackColumn) fallbackColumn.innerHTML='';
  if(fallbackCount) fallbackCount.textContent='0';
  if(meta) meta.textContent='加载中...';
  try{{
    var result=await requestJSON('/v1/models?refresh=true',{{method:'GET'}},45000);
    var d=result.data;
    if(!result.ok || !d || !Array.isArray(d.data)){{
      throw new Error((d && d.error && d.error.message) || ('HTTP '+result.status));
    }}
    var groupForEndpoint=function(endpoint){{
      endpoint=String(endpoint||'').trim().toLowerCase();
      if(endpoint==='remote') return 'remote';
      if(endpoint==='work' || endpoint==='traework') return 'traework';
      if(endpoint==='solo' || endpoint==='ide' || endpoint==='traecode') return 'traecode';
      return '';
    }};
    var fallbackModels=[];
    var modelGroups=function(model){{
      var modelId=String(model.id||'').toLowerCase();
      if(modelId==='auto') return ['remote'];
      var groups=(Array.isArray(model.endpoint_groups)?model.endpoint_groups:[model.endpoint_group])
        .map(groupForEndpoint).filter(Boolean);
      if(!groups.length){{
        groups=(Array.isArray(model.endpoints)?model.endpoints:[model.endpoint])
          .map(groupForEndpoint).filter(Boolean);
      }}
      if(!groups.length && model.endpoint){{
        groups=[groupForEndpoint(model.endpoint)].filter(Boolean);
      }}
      groups=groups.filter(function(item,index){{ return item && groups.indexOf(item)===index; }});
      if(!groups.length && !fallbackModels.some(function(item){{ return item.id===model.id; }})) fallbackModels.push(model);
      return groups;
    }};
    var appendModel=function(column,model){{
      var item=document.createElement('div');
      item.className='model-item';
      var name=document.createElement('div');
      name.className='model-item-name';
      name.textContent=String(model.id||'');
      item.appendChild(name);
      if(String(model.id||'').toLowerCase()==='auto'){{
        var route=document.createElement('div');
        route.className='model-item-alias';
        route.textContent='Relay 自动路由';
        item.appendChild(route);
      }}
      if(model.alias_for){{
        var alias=document.createElement('div');
        alias.className='model-item-alias';
        alias.textContent='别名 → '+String(model.alias_for);
        item.appendChild(alias);
      }}
      column.appendChild(item);
    }};
    var grouped={{remote:[],traecode:[],traework:[]}};
    d.data.forEach(function(model){{
      modelGroups(model).forEach(function(group){{
        if(!grouped[group].some(function(existing){{ return String(existing.id||'')===String(model.id||''); }})){{
          grouped[group].push(model);
        }}
      }});
    }});
    groupNames.forEach(function(group){{
      var column=columns[group];
      var count=document.getElementById('models-count-'+group);
      if(!column) return;
      column.innerHTML='';
      var items=grouped[group]||[];
      if(count) count.textContent=String(items.length);
      if(!items.length){{
        column.innerHTML='<div class="model-column-empty">暂无模型</div>';
        return;
      }}
      items.forEach(function(model){{
        appendModel(column,model);
      }});
    }});
    if(fallbackPanel && fallbackColumn && fallbackModels.length){{
      fallbackPanel.hidden=false;
      if(fallbackCount) fallbackCount.textContent=String(fallbackModels.length);
      fallbackModels.forEach(function(model){{
        appendModel(fallbackColumn,model);
      }});
    }}
    if(meta) meta.textContent='共 '+d.data.length+' 个模型，按端点归类';
    showMsg('models-msg','成功获取 '+d.data.length+' 个模型',true);
  }}catch(e){{
    groupNames.forEach(function(group){{
      var column=columns[group];
      if(column) column.innerHTML='<div class="model-column-empty">加载失败</div>';
    }});
    if(fallbackPanel) fallbackPanel.hidden=true;
    if(fallbackColumn) fallbackColumn.innerHTML='';
    if(fallbackCount) fallbackCount.textContent='0';
    if(meta) meta.textContent='加载失败';
    showMsg('models-msg',String(e),false);
  }}finally{{
    refreshModels.running=false;
    if(refreshButton) refreshButton.disabled=false;
  }}
}}
async function logout(){{
  var d=await postJSON('/api/logout',{{}});
  showMsg('auth-msg',d.success?'已登出':(d.error||'登出失败'),d.success);
  if(d.success) setTimeout(function(){{ location.reload(); }},600);
}}
async function switchAccount(id){{
  setSwitchBusy(true);
  try{{
    var d=await postJSON('/api/accounts/switch',{{account_id:id}},30000);
    if(!d.success){{ showMsg('account-msg',d.error||'切换失败',false); return; }}
    setActiveAccount(d.active||id,d.account||{{id:id}});
    showMsg('account-msg','已切换到账号 '+String((d.account&&(d.account.label||d.account.user_id))||id),true,3000);
  }}finally{{ setSwitchBusy(false); }}
}}
var accountModelDraft={{}};
function setAccountModelVisual(id,enabled){{
  var row=document.getElementById('row-'+id);
  var input=row&&row.querySelector('input[data-action="model-toggle"]');
  if(input) input.checked=!!enabled;
  var state=document.getElementById('model-state-wrap-'+id);
  if(state){{
    state.innerHTML='<span class="badge '+(enabled?'badge-ok':'badge-none')+'">'+(enabled?'已开启':'已关闭')+'</span>';
  }}
}}
function updateAccountModelSaveButton(){{
  var btn=document.getElementById('account-model-save-btn');
  if(btn) btn.disabled=Object.keys(accountModelDraft).length===0;
}}
function toggleAccountModel(id,input){{
  if(!input) return;
  var row=document.getElementById('row-'+id);
  var baseline=row&&row.getAttribute('data-model-enabled')==='1';
  var wanted=!!input.checked;
  if(wanted===baseline) delete accountModelDraft[String(id)];
  else accountModelDraft[String(id)]=wanted;
  setAccountModelVisual(id,wanted);
  updateAccountModelSaveButton();
  showMsg('account-msg',Object.keys(accountModelDraft).length
    ? '账号模型开关已修改，请点击“保存模型开关”后生效'
    : '',true,4000);
}}
function markAccountModelDirty(id,input){{
  // Account model switches are drafts; the toolbar Save button performs the
  // actual writes in one batch so changing a switch never changes the server.
  toggleAccountModel(id,input);
}}
async function saveAccountModelSettings(){{
  var ids=Object.keys(accountModelDraft);
  if(!ids.length) return;
  var btn=document.getElementById('account-model-save-btn');
  if(btn) btn.disabled=true;
  var failed=[];
  try{{
    for(var i=0;i<ids.length;i++){{
      var id=ids[i], wanted=!!accountModelDraft[id];
      var d=await postJSON('/api/accounts/model-enabled',{{account_id:id,enabled:wanted}},30000);
      if(!d.success){{ failed.push(id); continue; }}
      setAccountModelVisual(id,!!d.model_enabled);
      var savedRow=document.getElementById('row-'+id);
      if(savedRow) savedRow.setAttribute('data-model-enabled',!!d.model_enabled?'1':'0');
      delete accountModelDraft[id];
    }}
    if(failed.length){{
      failed.forEach(function(id){{
        var row=document.getElementById('row-'+id);
        setAccountModelVisual(id,row&&row.getAttribute('data-model-enabled')==='1');
      }});
      showMsg('account-msg','部分账号保存失败：'+failed.join(', '),false,8000);
    }}else showMsg('account-msg','账号模型请求开关已保存，每日签到不受影响',true,4000);
  }}finally{{
    updateAccountModelSaveButton();
  }}
}}
async function removeAccount(id){{
  if(!confirm('确定删除该账号？')) return;
  var d=await postJSON('/api/accounts/remove',{{account_id:id}});
  if(d.success) location.reload(); else showMsg('auth-msg',d.error||'删除失败',false);
}}
function markPollingDirty(){{
  var btn=document.getElementById('poll-save-btn');
  if(btn) btn.disabled=false;
  showMsg('poll-msg','轮询设置已修改，点击“保存轮询设置”后生效',true,0);
}}
async function savePollingSettings(){{
  var on=!!document.getElementById('poll-toggle').checked;
  var mode=document.querySelector('input[name=poll-mode]:checked');
  var modeVal=mode?mode.value:'round-robin';
  var d=await postJSON('/api/polling',{{enabled:on,mode:modeVal}});
  if(d.success){{
    var btn=document.getElementById('poll-save-btn'); if(btn) btn.disabled=true;
    var status=document.getElementById('poll-status');
    var countMatch=status&&status.textContent.match(/当前账号数:\\s*(\\d+)/);
    var accountCount=countMatch?countMatch[1]:'0';
    if(status) status.textContent='当前账号数: '+(d.account_count||accountCount)+'，轮询: '+(d.enabled?'开':'关')+'（'+(d.mode||modeVal)+'）';
    showMsg('poll-msg','轮询设置已保存',true,3000);
  }}else showMsg('poll-msg',d.error||'保存失败',false);
}}
// Kept as a compatibility wrapper for older inline callers; changing a
// setting must never persist it implicitly.
function togglePolling(){{ markPollingDirty(); }}
// Endpoint changes stay in the browser draft until the explicit Save button.
// Serialize explicit writes so a slower, older request cannot finish after a
// newer save and overwrite the current selection.
let endpointSaveSerial=0;
let endpointSaveQueue=Promise.resolve();
let endpointLastGoodState=null;
let endpointDirty=false;
function markEndpointDirty(){{
  endpointDirty=true;
  var btn=document.getElementById('settings-save-btn'); if(btn) btn.disabled=false;
  var msg=document.getElementById('settings-msg');
  if(msg){{ msg.textContent='端点设置已修改，点击“保存设置”后生效'; msg.className='msg msg-pending'; }}
}}
function readEndpointSelection(){{
  var selector=document.getElementById('settings-endpoint-preset');
  var input=document.getElementById('settings-web');
  var modelInput=document.getElementById('settings-model-base');
  var preset=selector?selector.value:'';
  var map=(typeof ENDPOINT_PRESET_MAP !== 'undefined') ? ENDPOINT_PRESET_MAP : {{}};
  var entry=map[preset]||null;
  return {{
    preset:preset,
    endpoint_id:(entry&&preset)?preset:'',
    web_base_url:input?(input.value||'').trim():'',
    model_base_url:modelInput?(modelInput.value||'').trim():'',
    upstream_mode:entry?entry.mode:''
  }};
}}
function restoreEndpointSelection(state){{
  state=state||{{}};
  var selector=document.getElementById('settings-endpoint-preset');
  var input=document.getElementById('settings-web');
  var modelInput=document.getElementById('settings-model-base');
  var map=(typeof ENDPOINT_PRESET_MAP !== 'undefined') ? ENDPOINT_PRESET_MAP : {{}};
  var endpointId=String(state.endpoint_id||'').toLowerCase();
  if(selector){{
    if(endpointId&&map[endpointId]) selector.value=endpointId;
    else if(state.preset==='__saved') selector.value='__saved';
    else selector.value='';
  }}
  if(input&&state.web_base_url!==undefined&&state.web_base_url!==null){{
    input.value=String(state.web_base_url);
  }}
  if(modelInput&&state.model_base_url!==undefined&&state.model_base_url!==null){{
    modelInput.value=String(state.model_base_url);
  }}
  renderEndpointPreview();
}}
function endpointPreviewText(state){{
  state=state||readEndpointSelection();
  var map=(typeof ENDPOINT_PRESET_MAP !== 'undefined') ? ENDPOINT_PRESET_MAP : {{}};
  var entry=map[state.endpoint_id]||map[state.preset]||null;
  var web=String(state.web_base_url||'').replace(/\\/+$/,'');
  var model=String(state.model_base_url||'').replace(/\\/+$/,'');
  var parts=[];
  if(entry&&(entry.mode==='work'||entry.mode==='work-agent')){{
    parts.push('模型请求 '+(web||'https://trae-api-cn.mchost.guru/api/remote/v1')+'/chat_sessions');
    parts.push('TraeWork / solo_work_remote（Work tier）');
  }}else if(entry&&(entry.mode==='ide'||entry.mode==='solo')){{
    parts.push('模型请求 '+(model||'https://trae-api-cn.mchost.guru')+'/api/agent/v3/llm_utils_chat');
    parts.push('function=solo_work_lite');
    parts.push('raw 诊断 '+(model||'https://trae-api-cn.mchost.guru')+'/api/ide/v2/llm_raw_chat');
    parts.push('Remote 回落 '+(web||'https://trae-api-cn.mchost.guru/api/remote/v1')+'/chat_sessions');
  }}else{{
    parts.push('会话请求 '+(web||'https://trae-api-cn.mchost.guru/api/remote/v1')+'/chat_sessions');
    parts.push('模型目录 '+(web||'https://trae-api-cn.mchost.guru/api/remote/v1')+'/models');
  }}
  return parts.join(' ｜ ');
}}
function renderEndpointPreview(state){{
  var el=document.getElementById('settings-endpoint-preview');
  if(!el) return;
  el.textContent='当前请求地址：'+endpointPreviewText(state);
}}
async function saveSettings(){{
  var serial=++endpointSaveSerial;
  var state=readEndpointSelection();
  var payload={{
    web_base_url:state.web_base_url,
    model_base_url:state.model_base_url,
    upstream_mode:state.upstream_mode,
    endpoint_id:state.endpoint_id
  }};
  var task=endpointSaveQueue.then(async function(){{
    var d=await postJSON('/api/settings',payload);
    // A newer selection happened while this request was in flight.  The
    // queued newer write owns the UI.
    if(serial!==endpointSaveSerial) return d;
    if(d&&d.success){{
      var canonical=(d.settings&&typeof d.settings==='object')?d.settings:{{}};
      var canonicalId=String(canonical.endpoint_id||state.endpoint_id||'').toLowerCase();
      var canonicalUrl=canonical.web_base_url;
      var canonicalModel=canonical.model_base_url;
      var map=(typeof ENDPOINT_PRESET_MAP !== 'undefined') ? ENDPOINT_PRESET_MAP : {{}};
      if(canonicalId&&map[canonicalId]) canonicalUrl=map[canonicalId].base_url;
      if((canonicalModel===undefined||canonicalModel===null)&&canonicalId&&map[canonicalId]){{
        canonicalModel=map[canonicalId].model_base_url;
      }}
      restoreEndpointSelection({{
        preset:canonicalId?canonicalId:state.preset,
        endpoint_id:canonicalId,
        web_base_url:(canonicalUrl===undefined||canonicalUrl===null)?state.web_base_url:canonicalUrl,
        model_base_url:(canonicalModel===undefined||canonicalModel===null)?state.model_base_url:canonicalModel
      }});
      endpointLastGoodState=readEndpointSelection();
      endpointDirty=false;
      var saveButton=document.getElementById('settings-save-btn'); if(saveButton) saveButton.disabled=true;
      var endpointDiagnostic=endpointDiagnosticText(d);
      showMsg('settings-msg',(d.note||'设置已保存')
        + (endpointDiagnostic ? '\\n'+endpointDiagnostic : ''),true);
      if(endpointDiagnostic) showEndpointDiagnostic(d,'端点设置');
    }}else{{
      restoreEndpointSelection(endpointLastGoodState||state);
      var endpointDiagnostic=endpointDiagnosticText(d);
      showMsg('settings-msg',((d&&d.error)||'保存失败')
        + (endpointDiagnostic ? '\\n'+endpointDiagnostic : ''),false);
      if(endpointDiagnostic) showEndpointDiagnostic(d,'端点设置失败');
    }}
    return d;
  }}).catch(function(error){{
    if(serial===endpointSaveSerial){{
      restoreEndpointSelection(endpointLastGoodState||state);
      endpointDirty=false;
      var saveButton=document.getElementById('settings-save-btn'); if(saveButton) saveButton.disabled=true;
      showMsg('settings-msg',String(error||'保存失败'),false);
    }}
    return {{success:false,error:String(error||'保存失败')}};
  }});
  // Keep the queue alive after a failed request, while still returning this
  // request's result to the caller that initiated the save.
  endpointSaveQueue=task.catch(function(){{}});
  return task;
}}
async function saveRelayServiceSettings(){{
  var input=document.getElementById('settings-port');
  var port=(input&&input.value||'').trim();
  var value=Number(port);
  if(!/^\\d+$/.test(port)||value<1||value>65535){{
    showMsg('relay-service-msg','请输入 1-65535 的端口号',false);
    return;
  }}
  var d=await postJSON('/api/settings',{{relay_port:value}});
  if(d.success){{
    var saveButton=document.getElementById('relay-service-save-btn');
    if(saveButton) saveButton.disabled=true;
    showMsg('relay-service-msg',d.note||'服务设置已保存，重启容器后生效',true,6000);
  }}else showMsg('relay-service-msg',d.error||'保存失败',false);
}}
function markRelayServiceDirty(){{
  var button=document.getElementById('relay-service-save-btn');
  if(button) button.disabled=false;
  showMsg('relay-service-msg','服务设置已修改，点击“保存服务设置”后生效',true,0);
}}
function markAutoRouteDirty(){{
  var button=document.getElementById('auto-route-save-btn');
  if(button) button.disabled=false;
  // Mirror the switch immediately: the endpoint panel must gray out as soon as
  // auto routing is turned on, before the save round-trip completes.
  var toggle=document.getElementById('auto-route-toggle');
  applyEndpointLock(toggle&&toggle.checked);
  showMsg('auto-route-msg','自动路由设置已修改，点击“保存自动路由”后生效',true,0);
}}
// Auto routing owns the endpoint choice while it is on: the dispatcher ignores
// the preset, so the form is locked instead of accepting a value that would
// never be used.  Applied from server-rendered state on load, mirrored on every
// switch change, and re-applied after each save so the two panels can never
// disagree about which channel is actually in use.
function applyEndpointLock(enabled){{
  enabled=!!enabled;
  var fields=document.getElementById('endpoint-fields');
  var note=document.getElementById('endpoint-lock-note');
  var pill=document.getElementById('endpoint-lock-state');
  if(fields) fields.classList.toggle('endpoint-locked',enabled);
  ['settings-endpoint-preset','settings-web','settings-model-base'].forEach(function(id){{
    var el=document.getElementById(id);
    if(!el) return;
    el.disabled=enabled;
    el.setAttribute('aria-disabled',enabled?'true':'false');
    el.title=enabled?'自动路由开启时由路由自动决定端点':'';
  }});
  if(note) note.hidden=!enabled;
  if(pill){{
    pill.textContent=enabled?'自动路由已接管':'手动选择';
    pill.classList.toggle('lock',enabled);
  }}
  var saveButton=document.getElementById('settings-save-btn');
  // Keep endpoint drafts intact while the form is locked.  If the user turns
  // auto routing back off without reloading, the explicit endpoint Save button
  // must become available again instead of silently losing the draft.
  if(saveButton) saveButton.disabled=enabled || !endpointDirty;
}}
async function saveAutoRoute(){{
  var toggle=document.getElementById('auto-route-toggle');
  var d=await postJSON('/api/auto-route',{{enabled:toggle.checked}});
  if(d.success){{
    toggle.checked=!!d.enabled;
    var button=document.getElementById('auto-route-save-btn');
    if(button) button.disabled=true;
    setStatePill('auto-route-state',d.enabled?'已开启':'已关闭',d.enabled);
    applyEndpointLock(d.enabled);
    showMsg('auto-route-msg',d.enabled?'自动路由已开启，上游端点选择已锁定':'自动路由已关闭，可手动选择上游端点',true,3000);
  }} else {{
    showMsg('auto-route-msg',d.error||'保存失败',false);
  }}
}}
function markMaxModeDirty(){{
  var button=document.getElementById('max-mode-save-btn');
  if(button) button.disabled=false;
  showMsg('max-mode-msg','Max 模式设置已修改，点击“保存”后生效',true,0);
}}
async function saveMaxMode(){{
  var enabled=document.getElementById('max-mode-toggle').checked;
  var models=document.getElementById('max-mode-models').value.trim();
  var d=await postJSON('/api/max-mode',{{enabled:enabled,models:models}});
  if(d.success){{
    document.getElementById('max-mode-toggle').checked=!!d.enabled;
    document.getElementById('max-mode-models').value=d.models||'';
    var button=document.getElementById('max-mode-save-btn');
    if(button) button.disabled=true;
    setStatePill('max-mode-state',d.enabled?'已开启':'已关闭',d.enabled);
    showMsg('max-mode-msg',d.enabled?'Max 模式已开启，新会话生效':'Max 模式已关闭',true,3000);
  }} else showMsg('max-mode-msg',d.error||'保存失败',false);
}}
function formatContext(n){{
  if(!n) return '';
  return n>=1000000 ? (Math.round(n/100000)/10)+'M' : Math.round(n/1000)+'K';
}}
function addMaxModel(name){{
  var input=document.getElementById('max-mode-models');
  var items=input.value.split(',').map(function(s){{ return s.trim(); }}).filter(Boolean);
  var lower=items.map(function(s){{ return s.toLowerCase(); }});
  if(lower.indexOf(name.toLowerCase())<0){{
    items.push(name);
    input.value=items.join(', ');
    markMaxModeDirty();
  }}
}}
async function detectMaxModels(){{
  var btn=document.getElementById('max-mode-detect-btn');
  var out=document.getElementById('max-mode-models-out');
  btn.disabled=true;
  try{{
    var result=await requestJSON('/api/max-mode/models',{{method:'GET'}},45000);
    var d=result.data;
    if(!result.ok||!d||!d.success) throw new Error(apiError(d,result.status));
    out.textContent='';
    var list=d.models||[];
    if(!list.length) out.textContent='当前账号没有支持 Max 的模型';
    list.forEach(function(m){{
      var chip=document.createElement('button');
      chip.type='button';
      chip.className='model-chip';
      chip.textContent=m.name+(m.max_context?' · '+formatContext(m.max_context):'');
      chip.title='加入生效模型';
      chip.onclick=function(){{ addMaxModel(m.name); }};
      out.appendChild(chip);
    }});
    out.hidden=false;
  }}catch(e){{ showMsg('max-mode-msg',String(e),false,8000); }}
  finally{{ btn.disabled=false; }}
}}
const ENDPOINT_PRESET_MAP = {endpoint_map_json};
endpointLastGoodState=readEndpointSelection();
renderEndpointPreview(endpointLastGoodState);
// Server-rendered auto-route state decides whether the manual endpoint form
// starts locked; the DOM attributes are applied here so both panels agree on
// first paint.
applyEndpointLock({str(auto_route_on).lower()});
function applyEndpointPreset(){{
  var selector=document.getElementById('settings-endpoint-preset');
  var input=document.getElementById('settings-web');
  var modelInput=document.getElementById('settings-model-base');
  if(!selector||!input) return;
  var map=ENDPOINT_PRESET_MAP;
  var entry=map[selector.value]||null;
  var value=entry?entry.base_url:'';
  if(value) input.value=value;
  if(modelInput){{ modelInput.value=(entry&&entry.model_base_url)||''; }}
  // Selecting a preset only updates the draft URL.  The explicit Save button
  // is the sole path that persists endpoint settings.
  renderEndpointPreview();
  markEndpointDirty();
}}
async function checkinRefreshAll(){{
  setBusy(true);
  try{{
    var result=await requestJSON('/api/checkin/accounts',{{method:'GET'}},90000);
    var d=result.data;
    if(!result.ok||!d||!d.success){{ throw new Error(apiError(d,result.status)); }}
    var accounts=d.accounts||[];
    accounts.forEach(updateAccountCheckinRow);
    var summary=updateCheckinSummary(accounts,'签到状态查询完成');
    var failures=accounts.filter(function(a){{ return a.error; }});
    if(failures.length){{
      showMsg('checkin-msg','签到状态查询完成，但有 '+failures.length+' 个账号失败：\\n'+failures.slice(0,3).map(checkinFailureText).join('\\n'),false,12000);
    }}else{{
      showMsg('checkin-msg','签到状态查询成功，已刷新 '+summary.total+' 个账号',true,3000);
    }}
  }}catch(e){{ showMsg('checkin-msg',String(e),false,12000); }}
  finally{{ setBusy(false); }}
}}
async function creditsRefreshAll(){{
  var options=arguments[0]||{{}};
  var silent=!!options.silent;
  if(creditUiState.reconcileInFlight) return;
  creditUiState.reconcileInFlight=true;
  if(!silent) setBusy(true);
  try{{
    var result=await requestJSON('/api/checkin/credits/accounts',{{method:'GET'}},90000);
    var d=result.data;
    if(!result.ok||!d||!d.success){{ throw new Error(apiError(d,result.status)); }}
    var accounts=d.accounts||[];
    accounts.forEach(updateAccountCreditsRow);
    updateCreditCalibrationFromAccounts(accounts);
    var failures=accounts.filter(function(a){{ return a.error; }});
    var summary=document.getElementById('checkin-summary');
    if(summary) summary.textContent='积分查询完成：'+(accounts.length-failures.length)+' / '+accounts.length+' 个账号'+(failures.length?'，'+failures.length+' 个异常':'');
    var updated=document.getElementById('checkin-updated');
    if(updated) updated.textContent='更新于 '+new Date().toLocaleTimeString();
    if(!silent&&failures.length){{
      showMsg('checkin-msg','积分查询完成，但有 '+failures.length+' 个账号失败：\\n'+failures.slice(0,3).map(checkinFailureText).join('\\n'),false,12000);
    }}else if(!silent){{
      showMsg('checkin-msg','积分查询成功，已刷新 '+accounts.length+' 个账号',true,3000);
    }}
  }}catch(e){{
    creditUiState.meta.status='error';
    renderCreditCalibrationMeta();
    if(!silent) showMsg('checkin-msg',String(e),false,12000);
  }}finally{{
    creditUiState.reconcileInFlight=false;
    if(!silent) setBusy(false);
  }}
}}
async function checkinAccount(id){{
  setAccountCheckinBusy(id,true);
  try{{
    var result=await requestJSON('/api/checkin/account/'+encodeURIComponent(id),{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}},90000);
    var d=result.data||{{}};
    var account={{id:id,data:d.data,checked_in:d.checked_in,account_credits:d.account_credits,success:d.success,error:d.success?'':apiError(d,result.status)}};
    updateAccountRow(account);
    if(!result.ok||!d||!d.success){{ showMsg('checkin-msg','账号 '+id+' 签到失败：'+apiError(d,result.status),false,12000); return; }}
    account.checked_in=true;
    account.error='';
    updateAccountRow(account);
    updateVisibleCheckinSummary('签到完成');
    showMsg('checkin-msg',d.skipped?'账号 '+id+' 已签到，无需重复操作':'账号 '+id+' 签到成功（业务码 '+(responseCode(d)||'0')+'）',true,3000);
  }}catch(e){{ showMsg('checkin-msg',String(e),false,12000); }}
  finally{{ setAccountCheckinBusy(id,false); }}
}}
async function checkinClaimAll(){{
  if(!confirm('确定按顺序逐个对所有账号签到？')) return;
  setBusy(true);
  try{{
    var accountCount=document.querySelectorAll('tr[data-account-id]').length;
    var timeoutMs=Math.max(900000,(accountCount+1)*({CHECKIN_INTERVAL}+5)*1000);
    var result=await requestJSON('/api/checkin/claim-all',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}},timeoutMs);
    var d=result.data;
    if(!result.ok||!d||!d.success){{ throw new Error(apiError(d,result.status)); }}
    var accounts=d.accounts||[];
    accounts.forEach(updateAccountRow);
    var summary=updateCheckinSummary(accounts,'轮询完成');
    if(summary.failed){{
      showMsg('checkin-msg','轮询完成：'+summary.failed+' 个账号失败\\n'+accounts.filter(function(a){{return a.error||a.success===false;}}).slice(0,5).map(checkinFailureText).join('\\n'),false,12000);
    }}else{{
      showMsg('checkin-msg','轮询签到成功，'+summary.ok+' 个账号已签到',true,3000);
    }}
  }}catch(e){{ showMsg('checkin-msg',String(e),false,12000); }}
  finally{{ setBusy(false); }}
}}
function formatCheckinTime(iso){{
  if(!iso) return '-';
  var m=String(iso).match(/^(\\d{{4}})-(\\d{{2}})-(\\d{{2}})T(\\d{{2}}):(\\d{{2}})/);
  return m ? (m[2]+'-'+m[3]+' '+m[4]+':'+m[5]) : String(iso);
}}
let autoCheckinDirty=false;
function markAutoCheckinDirty(){{
  autoCheckinDirty=true;
  var button=document.getElementById('auto-checkin-save-btn');
  if(button) button.disabled=false;
  showMsg('auto-checkin-msg','自动签到设置已修改，点击“保存”后生效',true,0);
}}
function renderAutoCheckin(d){{
  if(!d||!d.success) return;
  var toggle=document.getElementById('auto-checkin-toggle');
  var timeInput=document.getElementById('auto-checkin-time');
  // A background status refresh must not overwrite edits that have not been
  // saved yet.  It may still update the read-only execution status below.
  if(!autoCheckinDirty){{
    if(toggle) toggle.checked=!!d.enabled;
    if(timeInput&&d.time) timeInput.value=d.time;
  }}
  setStatePill('auto-checkin-state',d.running?'正在执行...':(d.enabled?('每天 '+d.time+' 自动签到'):'已关闭'),d.enabled||d.running);
  document.getElementById('auto-checkin-next').textContent=d.enabled?formatCheckinTime(d.next_run):'未启用';
  document.getElementById('auto-checkin-last').textContent=d.last_run_at
    ? formatCheckinTime(d.last_run_at)+(d.last_trigger==='manual'?'（手动）':'（定时）') : '-';
  var s=d.summary;
  document.getElementById('auto-checkin-result').textContent=s
    ? ('成功 '+s.ok+' / 跳过 '+s.skipped+' / 失败 '+s.failed+(s.no_token?(' / 无凭证 '+s.no_token):'')) : '-';
  var runBtn=document.getElementById('auto-checkin-run-btn');
  if(runBtn) runBtn.disabled=!!d.running;
}}
async function loadAutoCheckin(){{
  try{{
    var result=await requestJSON('/api/auto-checkin',{{method:'GET'}},15000);
    renderAutoCheckin(result.data);
    return result.data;
  }}catch(e){{ return null; }}
}}
async function saveAutoCheckin(){{
  var toggle=document.getElementById('auto-checkin-toggle');
  var timeValue=document.getElementById('auto-checkin-time').value;
  if(!/^\\d{{2}}:\\d{{2}}$/.test(timeValue||'')){{ showMsg('auto-checkin-msg','请填写有效的签到时间（HH:MM）',false); return; }}
  var d=await postJSON('/api/auto-checkin',{{enabled:toggle.checked,time:timeValue}});
  if(d.success){{
    autoCheckinDirty=false;
    var button=document.getElementById('auto-checkin-save-btn');
    if(button) button.disabled=true;
    renderAutoCheckin(d);
    showMsg('auto-checkin-msg',d.enabled?('已开启，每天 '+d.time+' 自动签到所有账号'):'自动签到已关闭',true,3000);
  }} else showMsg('auto-checkin-msg',d.error||'保存失败',false);
}}
var autoCheckinPoll=null;
async function runAutoCheckinNow(){{
  if(!confirm('立即按顺序签到所有未签到账号？已签到账号会自动跳过。')) return;
  var d=await postJSON('/api/auto-checkin/run',{{}});
  if(!d.success){{ showMsg('auto-checkin-msg',d.error||'启动失败',false); return; }}
  renderAutoCheckin(d);
  showMsg('auto-checkin-msg',d.started?'已开始执行，账号之间按签到间隔依次处理':'已有签到任务在执行',true,4000);
  clearInterval(autoCheckinPoll);
  autoCheckinPoll=setInterval(async function(){{
    var latest=await loadAutoCheckin();
    if(latest&&!latest.running){{
      clearInterval(autoCheckinPoll);
      if(typeof checkinRefreshAll==='function'&&latest.summary&&latest.summary.ok) showMsg('auto-checkin-msg','执行完成，可点击“查询签到状态”刷新列表',true,5000);
    }}
  }},5000);
}}
loadAutoCheckin();
setInterval(loadAutoCheckin,60000);
(function(){{
  var initial=(location.hash||'').replace('#','');
  if(initial&&document.querySelector('.tab-page[data-page="'+initial+'"]')) switchTab(initial);
}})();
syncSelection();
window.addEventListener('resize',syncSelection);
if(document.fonts&&document.fonts.ready) document.fonts.ready.then(syncSelection);
var manualForm=document.getElementById('manual-form');
if(manualForm) manualForm.addEventListener('submit',async function(e){{
  e.preventDefault();var fd=new FormData(e.target);
  var payload={{}};
  for(var[k,v]of fd.entries())if(v)payload[k]=v;
  var d=await postJSON('/api/web-auth',payload);
  showMsg('manual-msg',d.success?'账号已添加':d.error||'提交失败',d.success);
  if(d.success) setTimeout(function(){{ location.reload(); }},600);
}});
</script>
</body>
</html>"""


def _oauth_result_html(success: bool, message: str, login_trace_id: str = "") -> str:
    safe_msg = html_mod.escape(message)
    safe_trace = html_mod.escape(login_trace_id)
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Trae 授权</title>{_FAVICON_LINKS}
<style>
body {{ font:16px -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#eef0f4;color:#1d2331;padding:40px; }}
.msg {{ padding:20px;border-radius:10px;margin-bottom:16px;border:1px solid; }}
.ok {{ background:#e2f3eb;color:#0a7451;border-color:#bfe3d3; }}
.err {{ background:#fdecec;color:#dc2626;border-color:#f3c6c6; }}
</style>
</head>
<body>
<div class="msg {'ok' if success else 'err'}">
  <h2 style="margin:0 0 8px">{'成功' if success else '失败'}</h2>
  <p>{safe_msg}</p>
</div>
<script>
(function(){{
  // 授权回调由 Trae 授权页直接跳转到服务器 /authorize。
  // 成功后回到控制台自动刷新，失败则停留展示错误。
  if ({str(success).lower()}) {{
    setTimeout(function(){{ window.location.href = '/web/login'; }}, 1200);
  }}
}})();
</script>
</body>
</html>"""


async def _peek_async(ait):
    try:
        first = await ait.__anext__()
    except StopAsyncIteration:
        return None, ait

    async def chain():
        yield first
        async for item in ait:
            yield item

    return first, chain()


async def _empty_cli_events():
    if False:
        yield None


def _sse_headers() -> dict:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Content-Type-Options": "nosniff",
    }


def _stream_heartbeat_seconds() -> float:
    try:
        value = float(os.environ.get("SSE_HEARTBEAT_SECONDS", "1"))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, value)


def _stream_error_event(response) -> str:
    """Turn a late upstream failure into an OpenAI-compatible SSE error."""
    raw = getattr(response, "body", b"")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        error = {
            "message": str(payload or "Upstream stream failed"),
            "type": "api_error",
        }
    return "data: " + json.dumps({"error": error}, ensure_ascii=False) + "\n\n"


def _stream_start_event(model: str) -> str:
    """Send a parseable SSE frame while the first Trae frame is pending.

    A comment-only keepalive is legal SSE, but a few terminal clients treat it
    as an empty response and close/retry before the upstream request finishes.
    An empty OpenAI delta keeps those clients attached without exposing text or
    inventing a completion.

    The delta is deliberately empty: emitting ``role`` here would duplicate the
    translator's own opening frame, and the OpenAI stream contract carries the
    assistant role exactly once.
    """
    return (
        "data: "
        + json.dumps(
            {
                "id": "",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )


async def _deferred_dispatch_stream(
    messages: list[dict], model: str, options: Optional[dict] = None
):
    """Open the public SSE stream before waiting for Trae's upstream headers.

    This keeps fallback routing intact because `_dispatch_chat` still performs
    the complete route selection in one task.  The task itself is awaited with
    keepalives, so a slow upstream cannot leave the client staring at a blank
    connection or freeze the event loop.
    """
    # Start routing before emitting any keepalive.  Some terminal clients
    # (notably zcode) treat a comment-only first frame as an empty cached
    # response and close the HTTP stream before asking for the next frame. The
    # old ordering created the upstream task *after* that first yield, so the
    # request could be cancelled without ever reaching Trae.
    options = _apply_auto_route(messages, options)
    requested_mode = str(
        options.get("_upstream_mode") or _current_upstream_mode()
    ).strip().lower()
    fallback_allowed = (
        not bool(options.get("_disable_upstream_fallback"))
        and requested_mode
        in {
            "raw",
            "direct",
            "auto",
            "ide",
            "solo",
            "work",
            "work-agent",
            "traework-native",
            "native",
            "traework",
        }
        and _remote_fallback_allowed_for_request(messages, options)
    )
    fallback_attempted = False
    fallback_input_error_event = None
    task = asyncio.create_task(_dispatch_chat(messages, model, True, options))
    dispatch_claimed = False
    response = None
    iterator = None
    request_id = str((options or {}).get("_relay_request_id") or "")
    started_at = time.monotonic()
    upstream_chunks = 0
    upstream_payload_chunks = 0
    upstream_observed_model_event = False
    saw_done = False
    stream_status = "opening"
    sent_start_event = False

    async def close_current() -> None:
        """Close the currently selected response/iterator before a retry."""

        nonlocal iterator, response
        if iterator is None:
            iterator = getattr(response, "body_iterator", None)
        if iterator is not None:
            close_iterator = getattr(iterator, "aclose", None)
            if close_iterator is not None:
                try:
                    await close_iterator()
                except Exception:
                    pass
            iterator = None
        close_response = getattr(response, "close", None)
        if close_response is not None:
            try:
                close_response()
            except Exception:
                pass
        response = None

    async def try_remote_fallback(reason: str):
        """Open one Remote stream after a native path fails before output."""

        nonlocal fallback_attempted, response, fallback_input_error_event
        if not fallback_allowed or fallback_attempted:
            return None
        fallback_attempted = True
        trace = options.get("_upstream_trace")
        _trace_append_failure(trace, requested_mode, reason)
        if isinstance(trace, dict):
            trace.update(
                fallback_used=True,
                fallback_target="remote",
            )
        fallback_options = _remote_fallback_options(options, requested_mode)
        logger.warning(
            "stream endpoint failed before output; falling back to remote "
            "id=%s requested_mode=%s reason=%s",
            request_id,
            requested_mode,
            reason,
        )
        try:
            fallback = await _dispatch_chat(
                messages, model, True, fallback_options
            )
        except Exception as exc:
            logger.warning(
                "stream remote fallback failed id=%s requested_mode=%s error=%s",
                request_id,
                requested_mode,
                exc,
            )
            return None
        if _upstream_response_error(fallback):
            logger.warning(
                "stream remote fallback returned error id=%s requested_mode=%s detail=%s",
                request_id,
                requested_mode,
                _upstream_response_error(fallback),
            )
            if getattr(fallback, "status_code", 200) == 413:
                fallback_input_error_event = _stream_error_event(fallback)
            response = fallback
            await close_current()
            return None
        if isinstance(trace, dict):
            trace.update(
                actual_mode="remote",
                actual_endpoint="remote",
                fallback_used=True,
                fallback_target="remote",
            )
        return fallback

    async def forward_response(current_response):
        """Forward one StreamingResponse and propagate pre-output failures."""

        nonlocal iterator, upstream_chunks, upstream_payload_chunks
        nonlocal saw_done, stream_status, sent_start_event
        iterator = _response_body_iterator(current_response)
        buffer = ""
        saw_model_output = False
        observed_model_event = False
        saw_error = False
        pending_terminal_chunks: list[str] = []

        def public_chunks(chunk: str):
            nonlocal upstream_chunks, upstream_payload_chunks
            nonlocal upstream_observed_model_event
            nonlocal saw_done, stream_status, sent_start_event
            nonlocal saw_model_output, observed_model_event, saw_error
            text = _chunk_text_delta(chunk)
            reasoning = _chunk_reasoning_delta(chunk)
            has_tools = _chunk_has_tool_calls(chunk)
            has_output = bool(text.strip() or reasoning.strip() or has_tools)
            saw_model_output = saw_model_output or has_output
            observed_model_event = observed_model_event or bool(
                text or reasoning or has_tools
            )
            payloads = list(_iter_sse_payloads(chunk))
            for value in payloads:
                if (
                    isinstance(value.get("usage"), Mapping)
                    or value.get("provider_model_name")
                ):
                    observed_model_event = True
            if not payloads and any(
                line.lstrip().startswith("data:")
                and line.split(":", 1)[1].strip() not in {"", "[DONE]"}
                for line in chunk.splitlines()
            ):
                # An invalid model frame cannot prove that replay is safe.
                observed_model_event = True
            state = _chunk_stream_state(chunk)
            saw_error = saw_error or state == "error"
            upstream_observed_model_event = bool(
                upstream_observed_model_event or observed_model_event or saw_error
            )
            if state == "completed" and not saw_model_output and not saw_error:
                # Do not let a bare finish/[DONE] turn an empty upstream into
                # success before the EOF check can surface its real failure.
                pending_terminal_chunks.append(chunk)
                return
            if chunk:
                upstream_chunks += 1
                if has_output:
                    upstream_payload_chunks += 1
                if "data: [DONE]" in chunk:
                    saw_done = True
                    stream_status = "completed"
                if not sent_start_event and chunk.lstrip().startswith(":"):
                    yield _stream_start_event(model)
                    sent_start_event = True
                yield chunk

        async for chunk in iterator:
            if isinstance(chunk, (bytes, bytearray)):
                chunk = bytes(chunk).decode("utf-8", errors="replace")
            buffer += str(chunk or "").replace("\r\n", "\n")
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                for public_chunk in public_chunks(block + "\n\n"):
                    yield public_chunk
        if buffer.strip():
            for public_chunk in public_chunks(buffer):
                yield public_chunk
        if not saw_model_output and not saw_error:
            raise EmptyUpstreamResponse(
                "Trae upstream stream ended without text, reasoning, or tool calls",
                retryable=not observed_model_event,
                observed_model_event=observed_model_event,
            )
        for chunk in pending_terminal_chunks:
            upstream_chunks += 1
            if "data: [DONE]" in chunk:
                saw_done = True
            yield chunk

    async def close_dispatch() -> None:
        nonlocal response, iterator
        if not task.done():
            task.cancel()
        try:
            dispatched_response = await task
        except (asyncio.CancelledError, Exception):
            pass
        else:
            # Dispatch may have finished while the public stream was paused
            # at its initial frame, before it could take response ownership.
            if not dispatch_claimed:
                response = dispatched_response
        if iterator is None and response is not None:
            iterator = getattr(response, "body_iterator", None)
        await close_current()

    try:
        # Enter the selected transport before emitting the first keepalive,
        # while retaining ownership if cancellation occurs in this handoff.
        await asyncio.sleep(0)
        # The task may still be establishing the Trae request.  Emit one real
        # data frame before comment heartbeats so zcode/OpenCode does not treat
        # the stream as an empty cached response and cancel the task.
        if not task.done():
            yield _stream_start_event(model)
            sent_start_event = True
        interval = _stream_heartbeat_seconds()
        while True:
            try:
                if interval > 0:
                    response = await asyncio.wait_for(
                        asyncio.shield(task), interval
                    )
                else:
                    response = await task
                dispatch_claimed = True
                break
            except asyncio.TimeoutError:
                yield ": relay-keepalive\n\n"

        if getattr(response, "status_code", 200) >= 400:
            input_too_large = getattr(response, "status_code", 200) == 413
            upstream_error = _upstream_response_error(response) or "http error"
            error_event = _stream_error_event(response)
            await close_current()
            fallback = (
                None
                if input_too_large
                else await try_remote_fallback(upstream_error)
            )
            if fallback is None:
                stream_status = "upstream_error"
                yield fallback_input_error_event or error_event
                yield "data: [DONE]\n\n"
                saw_done = True
                return
            response = fallback

        try:
            async for chunk in forward_response(response):
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            if isinstance(exc, EmptyUpstreamResponse):
                _track_usage_from_exception(exc, model)
                # Native Trae endpoints may emit ``model_config`` and a
                # usage-turn id before returning account-scoped 4011/429.
                # The translator preserves those metadata fields on the
                # exception, but no model output was sent.  Let the endpoint
                # fallback proceed for this capacity error; otherwise the
                # public streaming request becomes a 502 while the same
                # non-stream probe can still rotate/fallback successfully.
                native_capacity_error = _is_native_rate_limit_error(exc)
                if (
                    (not exc.retryable or exc.observed_model_event)
                    and not native_capacity_error
                ):
                    raise
            # A StreamingResponse can be returned with HTTP 200 and fail only
            # when its body iterator starts (empty native SSE, invalid headers,
            # or a provider disconnect).  Retry Remote only before any native
            # payload has reached the caller; once partial output is visible we
            # preserve that stream and surface the original failure.
            if (
                upstream_payload_chunks
                or upstream_observed_model_event
                or fallback_attempted
            ):
                raise
            await close_current()
            fallback = await try_remote_fallback(str(exc))
            if fallback is None:
                if fallback_input_error_event is None:
                    raise
                stream_status = "upstream_error"
                yield fallback_input_error_event
                yield "data: [DONE]\n\n"
                saw_done = True
                return
            response = fallback
            async for chunk in forward_response(response):
                yield chunk
        stream_status = "completed"
    except asyncio.CancelledError:
        if saw_done:
            # The upstream already emitted [DONE]; a client disconnecting
            # during final teardown is not an aborted model turn.
            stream_status = "completed"
            logger.info(
                "public stream cancelled after done id=%s chunks=%d elapsed_ms=%d",
                request_id,
                upstream_chunks,
                int((time.monotonic() - started_at) * 1000),
            )
        else:
            stream_status = "client_cancelled"
            logger.warning(
                "public stream cancelled id=%s upstream_ready=%s task_done=%s chunks=%d elapsed_ms=%d",
                request_id,
                response is not None,
                task.done(),
                upstream_chunks,
                int((time.monotonic() - started_at) * 1000),
            )
        raise
    except GeneratorExit:
        stream_status = "client_closed" if not saw_done else "completed"
        raise
    except Exception as exc:
        stream_status = "error"
        _track_usage_from_exception(exc, model)
        logger.warning("deferred stream dispatch failed: %s", exc)
        yield "data: " + json.dumps(
            {"error": {"message": str(exc), "type": "api_error"}},
            ensure_ascii=False,
        ) + "\n\n"
        yield "data: [DONE]\n\n"
        saw_done = True
    finally:
        try:
            await _await_session_cleanup(asyncio.create_task(close_dispatch()))
        finally:
            _cancel_unclaimed_account_slot(options)
            logger.info(
                "public stream closed id=%s status=%s chunks=%d done=%s elapsed_ms=%d",
                request_id,
                stream_status,
                upstream_chunks,
                saw_done,
                int((time.monotonic() - started_at) * 1000),
            )


def _continuation_allowed(options: Optional[Mapping[str, Any]]) -> bool:
    """Whether this request may transparently continue a truncated answer."""

    if not continuation.enabled():
        return False
    options = options or {}
    if options.get("_disable_auto_continue"):
        return False
    # TraeWork's custom-model ingress already owns its own turn loop; a relay
    # continuation there would double-bill a client-managed conversation.
    if options.get("_traework_custom_model"):
        return False
    return True


async def _stream_with_continuation(
    messages: list[dict],
    model: str,
    options: dict,
    first_response,
):
    """Forward one streamed turn, continuing it while the answer is cut off.

    The public response is a single OpenAI SSE stream: continued rounds append
    deltas, and only the last round emits the finish frame and ``[DONE]``.  Any
    tool call, an explicit ``finish_reason=length`` on the client's request
    budget, or a repeated short answer stops the loop so a caller never sees a
    duplicated or unbounded answer.
    """

    limit = continuation.max_continues()
    if not _continuation_allowed(options) or limit <= 0:
        async for chunk in first_response:
            yield chunk
        return

    threshold = continuation.text_threshold()
    similarity_limit = continuation.similarity_threshold()
    current_messages = list(messages)
    current_response = _response_body_iterator(first_response)
    text_total = ""
    last_short_text = ""
    finish_chunks: list[str] = []
    saw_tool_call = False
    finish_reason = "stop"
    reasoning_total = ""
    request_id = str((options or {}).get("_relay_request_id") or "")
    # Upstream/proxies may coalesce several SSE blocks into one network
    # chunk. Keep an incomplete block between reads so a terminal block never
    # causes the preceding text block to be held back or dropped.
    sse_buffer = ""

    def complete_blocks(raw: Any, *, flush: bool = False) -> list[str]:
        nonlocal sse_buffer
        if isinstance(raw, (bytes, bytearray)):
            text = bytes(raw).decode("utf-8", errors="replace")
        else:
            text = str(raw or "")
        if text:
            sse_buffer += text.replace("\r\n", "\n")
        blocks: list[str] = []
        while "\n\n" in sse_buffer:
            block, sse_buffer = sse_buffer.split("\n\n", 1)
            if block.strip():
                blocks.append(block + "\n\n")
        if flush and sse_buffer.strip():
            blocks.append(sse_buffer)
            sse_buffer = ""
        return blocks

    for attempt in range(limit + 1):
        finish_chunks = []
        round_text = ""
        round_reasoning = ""
        saw_tool_call = False
        finish_reason = "stop"
        async for raw_chunk in current_response:
            for chunk in complete_blocks(raw_chunk):
                parsed = _chunk_finish_reason(chunk)
                if parsed:
                    finish_reason = parsed
                if _chunk_has_tool_calls(chunk):
                    saw_tool_call = True
                state = _chunk_stream_state(chunk)
                if state is not None:
                    # Hold only the terminal SSE block. If a proxy coalesced
                    # text + finish + [DONE], the text block was already
                    # yielded above instead of being lost with the finish.
                    finish_chunks.append(chunk)
                    continue
                round_text += _chunk_text_delta(chunk)
                round_reasoning += _chunk_reasoning_delta(chunk)
                yield chunk

        # Flush a final non-blank fragment for upstreams that close without a
        # blank-line delimiter. A terminal [DONE] block is always delimited.
        for chunk in complete_blocks("", flush=True):
            parsed = _chunk_finish_reason(chunk)
            if parsed:
                finish_reason = parsed
            if _chunk_has_tool_calls(chunk):
                saw_tool_call = True
            state = _chunk_stream_state(chunk)
            if state is not None:
                finish_chunks.append(chunk)
                continue
            round_text += _chunk_text_delta(chunk)
            round_reasoning += _chunk_reasoning_delta(chunk)
            yield chunk

        text_total += round_text
        reasoning_total += round_reasoning
        truncated = continuation.looks_truncated(
            text_total,
            finish_reason,
            has_tool_calls=saw_tool_call,
            reasoning=round_reasoning,
        )
        reasoning_only = continuation.looks_reasoning_only(
            round_text,
            has_tool_calls=saw_tool_call,
            reasoning=round_reasoning,
        )
        if not truncated and not reasoning_only:
            break
        if attempt >= limit:
            logger.info(
                "auto-continue limit reached id=%s rounds=%d finish=%s",
                request_id,
                attempt + 1,
                finish_reason,
            )
            break

        # A short round that repeats the previous round verbatim means the
        # model is looping; continuing again would only duplicate it.  Compare
        # per-round text, never the accumulated answer, so a legitimate
        # continuation that shares a prefix is not mistaken for a loop.
        current_text = round_text.strip()
        if len(current_text) < threshold and last_short_text:
            if continuation.similarity(last_short_text, current_text) > similarity_limit:
                logger.warning(
                    "auto-continue stopped on repeated short answer id=%s", request_id
                )
                break
        if len(current_text) < threshold:
            last_short_text = current_text

        logger.info(
            "auto-continue round=%d id=%s model=%s reason=%s finish=%s chars=%d",
            attempt + 1,
            request_id,
            model,
            "reasoning-only" if reasoning_only else "truncated",
            finish_reason,
            len(round_text),
        )
        active_tracker = _USAGE_TRACKER.get()
        bind_rounds = getattr(active_tracker, "bind_auto_continue", None)
        if callable(bind_rounds):
            bind_rounds(attempt + 1)
        current_messages = continuation.continued_messages(
            messages, text_total, reasoning_only=reasoning_only
        )
        # Re-enter the deferred dispatcher so a continuation round keeps the
        # same endpoint fallback and pre-output retry guarantees as the first
        # round.  It is an async generator, so a mid-stream failure surfaces as
        # a stream error frame instead of an exception at this call site.
        current_response = _deferred_dispatch_stream(
            current_messages, model, dict(options)
        )

    if not finish_chunks:
        # A continued round that closed without its own terminal frame must not
        # leave the caller waiting on a half-open stream.
        finish_chunks = [
            _stream_error_event_free_finish(model),
            "data: [DONE]\n\n",
        ]
    for chunk in finish_chunks:
        yield chunk


def _stream_error_event_free_finish(model: str) -> str:
    """A terminal Chat Completions frame for a stream that ended silently."""

    return (
        "data: "
        + json.dumps(
            {
                "id": "",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )


async def _nonstream_with_continuation(
    messages: list[dict],
    model: str,
    options: dict,
    tracker: "_UsageTracker",
):
    """Non-streaming twin of :func:`_stream_with_continuation`.

    Completion objects are merged in place: the continued text is appended to
    the first answer's message, usage is summed, and the terminal finish reason
    comes from the final round.
    """

    limit = continuation.max_continues()
    threshold = continuation.text_threshold()
    similarity_limit = continuation.similarity_threshold()
    request_id = str((options or {}).get("_relay_request_id") or "")
    current_messages = list(messages)
    merged: Optional[dict[str, Any]] = None
    text_total = ""
    reasoning_total = ""
    last_short_text = ""
    # One tracker owns the whole multi-round turn: begin once, fold every
    # round's usage into it, and finish once so the console still records a
    # single billable request with the summed token counts.
    context_token = _USAGE_TRACKER.set(tracker)
    status = "completed"
    try:
        await tracker.begin()
        for attempt in range(limit + 1):
            response = await _dispatch_chat(current_messages, model, False, options)
            if getattr(response, "status_code", 200) >= 400:
                status = "error"
                return response
            try:
                completion = json.loads(getattr(response, "body", b"{}"))
            except Exception:
                return response
            if not isinstance(completion, dict):
                return response
            _track_usage_from_result(completion, model)
            choices = completion.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else {}
            choice = choice if isinstance(choice, Mapping) else {}
            message = choice.get("message")
            message = message if isinstance(message, Mapping) else {}
            tool_calls = message.get("tool_calls")
            tool_calls = tool_calls if isinstance(tool_calls, list) else []
            round_text = str(message.get("content") or "")
            round_reasoning = str(
                message.get("reasoning_content") or message.get("reasoning") or ""
            )
            finish_reason = str(choice.get("finish_reason") or "stop")

            if merged is None:
                merged = completion
            else:
                merged_choices = merged.setdefault("choices", [{}])
                merged_choice = (
                    merged_choices[0] if isinstance(merged_choices, list) and merged_choices else {}
                )
                if not isinstance(merged_choice, dict):
                    merged_choice = {}
                    merged_choices.insert(0, merged_choice)
                merged_message = merged_choice.setdefault("message", {})
                if not isinstance(merged_message, dict):
                    merged_message = {}
                    merged_choice["message"] = merged_message
                combined = str(merged_message.get("content") or "") + round_text
                merged_message["content"] = combined or None
                if round_reasoning:
                    merged_message["reasoning_content"] = (
                        str(merged_message.get("reasoning_content") or "")
                        + round_reasoning
                    )
                if tool_calls:
                    merged_message["tool_calls"] = tool_calls
                merged_choice["finish_reason"] = finish_reason
                merged_usage = merged.get("usage")
                round_usage = completion.get("usage")
                if isinstance(merged_usage, dict) and isinstance(round_usage, Mapping):
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        merged_usage[key] = int(merged_usage.get(key) or 0) + int(
                            round_usage.get(key) or 0
                        )

            text_total += round_text
            reasoning_total += round_reasoning
            truncated = continuation.looks_truncated(
                text_total,
                finish_reason,
                has_tool_calls=bool(tool_calls),
                reasoning=round_reasoning,
            )
            reasoning_only = continuation.looks_reasoning_only(
                round_text,
                has_tool_calls=bool(tool_calls),
                reasoning=round_reasoning,
            )
            if not truncated and not reasoning_only:
                break
            if attempt >= limit:
                break
            current_text = round_text.strip()
            if len(current_text) < threshold and last_short_text:
                if (
                    continuation.similarity(last_short_text, current_text)
                    > similarity_limit
                ):
                    logger.warning(
                        "auto-continue stopped on repeated short answer id=%s",
                        request_id,
                    )
                    break
            if len(current_text) < threshold:
                last_short_text = current_text
            logger.info(
                "auto-continue round=%d id=%s model=%s nonstream reason=%s chars=%d",
                attempt + 1,
                request_id,
                model,
                "reasoning-only" if reasoning_only else "truncated",
                len(round_text),
            )
            bind_auto_continue = getattr(tracker, "bind_auto_continue", None)
            if callable(bind_auto_continue):
                bind_auto_continue(attempt + 1)
            current_messages = continuation.continued_messages(
                messages, text_total, reasoning_only=reasoning_only
            )
    except Exception:
        status = "error"
        raise
    finally:
        try:
            _USAGE_TRACKER.reset(context_token)
            await tracker.finish(status)
        finally:
            _cancel_unclaimed_account_slot(options)

    if merged is None:
        return _openai_error(502, "Upstream returned no completion", "api_error")
    return JSONResponse(content=merged)


def _response_body_iterator(response):
    if hasattr(response, "__aiter__"):
        return response
    if hasattr(response, "__iter__") and not isinstance(response, (str, bytes, Mapping)):
        return _as_async(response)
    iterator = getattr(response, "body_iterator", None)
    if iterator is not None:
        return iterator
    body = getattr(response, "body", b"") or b""
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)

    async def single():
        if text:
            yield text

    return single()


async def _as_async(iterable):
    """Adapt a plain (sync) iterator to an async one for the stream helpers."""

    for item in iterable:
        yield item


def _chunk_text_delta(chunk: Any) -> str:
    return _chunk_delta_field(chunk, "content")


def _chunk_reasoning_delta(chunk: Any) -> str:
    text = _chunk_delta_field(chunk, "reasoning_content")
    return text or _chunk_delta_field(chunk, "reasoning")


def _chunk_delta_field(chunk: Any, field: str) -> str:
    for value in _iter_sse_payloads(chunk):
        choices = value.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping):
                    text = delta.get(field)
                    if isinstance(text, str) and text:
                        return text
        text = value.get(field)
        if isinstance(text, str) and text:
            return text
    return ""


def _chunk_finish_reason(chunk: Any) -> str:
    for value in _iter_sse_payloads(chunk):
        choices = value.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            reason = choice.get("finish_reason")
            if reason not in (None, ""):
                return str(reason)
    return ""


def _chunk_has_tool_calls(chunk: Any) -> bool:
    for value in _iter_sse_payloads(chunk):
        if _tool_call_count(value) > 0:
            return True
        choices = value.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping) and delta.get("tool_calls"):
                return True
    return False


def _iter_sse_payloads(chunk: Any):
    if isinstance(chunk, (bytes, bytearray)):
        text = bytes(chunk).decode("utf-8", errors="replace")
    else:
        text = str(chunk or "")
    if not text:
        return
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        data_lines = [
            line[5:].lstrip()
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload = "\n".join(data_lines).strip()
        if payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except Exception:
            continue
        if isinstance(value, Mapping):
            yield value


def _tool_translation_options(
    options: Optional[dict], messages: Optional[list[dict]] = None
) -> dict:
    options = options or {}
    has_explicit_catalog = "tools" in options or "_inherited_tools" in options
    tool_catalog = (
        options["tools"]
        if "tools" in options
        else options.get("_inherited_tools", [])
    )
    # TraeWork's custom-model adapter owns the local toolhost and its raw
    # six-field request normally has no OpenAI ``tools`` array.  Passing an
    # empty list to the OpenAI translator means "suppress every tool call";
    # for this ingress that would turn a valid model tool request into plain
    # text and the desktop client could never execute it.  ``None`` preserves
    # the caller-owned protocol and lets the custom SSE adapter forward calls
    # parsed from native fields or the XML marker in the model response.
    if options.get("_traework_custom_model") and not has_explicit_catalog:
        tool_catalog = None
    return {
        # API callers execute tools. With no tools field, suppress any internal
        # Trae tool event instead of exposing a call the client cannot handle.
        "allowed_tools": tool_catalog,
        "tool_choice": options.get("tool_choice"),
        "parallel_tool_calls": options.get("parallel_tool_calls"),
        # Keep reasoning presentation separate from the raw upstream request.
        # ``thinking`` is intentionally consumed only by the translators.
        "include_reasoning": responses_api.thinking_requested(options),
        # Protect the continuation turn from an upstream model that echoes an
        # already completed call with a fresh id. A new user message after the
        # result clears this set in cli_client.completed_tool_signatures().
        "completed_tool_signatures": cli_client.completed_tool_signatures(
            messages or []
        ),
    }


def _tool_protocol_requested(
    options: Optional[dict], messages: Optional[list[dict]] = None
) -> bool:
    options = options or {}
    if options.get("_tool_protocol_requested") or any(
        key in options for key in ("tools", "tool_choice", "parallel_tool_calls")
    ):
        return True
    return any(
        isinstance(message, dict) and _message_contains_tool_protocol(message)
        for message in (messages or [])
    )


def _message_contains_tool_protocol(message: Mapping[str, Any]) -> bool:
    """Detect OpenAI and TraeWork renderer tool messages in request history.

    TraeWork does not always use an OpenAI ``role=tool`` message. Its renderer
    serializes calls/results as ``tool_use``/``tool_result`` content blocks,
    with results commonly carried by a ``role=user`` message. Those turns
    still require the tool-aware runtime prompt and the raw-safe route.
    """

    role = str(message.get("role") or "").strip().lower()
    if role in {"tool", "function"}:
        return True
    if role == "assistant" and (
        (isinstance(message.get("tool_calls"), list) and bool(message["tool_calls"]))
        or isinstance(message.get("function_call"), Mapping)
    ):
        return True
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, Mapping)
        and str(block.get("type") or "").strip().lower()
        in {"tool_use", "tool_call", "function_call", "tool_result"}
        for block in content
    )


def _with_auto_client_context(
    req: Request,
    body: Mapping[str, Any],
    messages: list[dict],
    options: dict,
) -> dict:
    """Infer caller environment and plugin catalog when the client omitted it."""

    if not _tool_protocol_requested(options, messages):
        return options
    if "client_context" in options or "clientContext" in options:
        return options
    # OpenAI clients commonly send `metadata`, while new-api's Responses DTO
    # preserves the equivalent caller hints under `client_metadata`.
    metadata = body.get("client_metadata") or body.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    enriched = dict(options)
    enriched["client_context"] = raw_client.build_client_context(
        enriched,
        request_headers=dict(req.headers),
        metadata=metadata,
    )
    return enriched


def _request_session_hint(req: Request, body: Mapping[str, Any]) -> str:
    """Read common conversation-id aliases without exposing them downstream."""
    for key in ("session_id", "sessionId", "conversation_id", "conversationId"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = body.get("client_metadata") or body.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("session_id", "sessionId", "conversation_id", "conversationId"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in (
        "x-session-id",
        "x-conversation-id",
        "x-chat-session-id",
        "conversation-id",
    ):
        value = req.headers.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def _apply_tool_header_hints(req: Request, options: dict) -> dict:
    """Accept optional tool-policy hints used by nonstandard terminal clients.

    OpenAI places tool definitions in JSON, not headers. Some adapters still
    send a small ``Tool``/``X-Tools`` hint; treat it as a routing signal and,
    when it contains JSON, restore the same options that would have appeared in
    the request body. Arbitrary client headers are never forwarded upstream.
    """

    enriched = dict(options)
    saw_hint = False
    for header in ("tools", "tool", "x-tools", "x-tool"):
        value = req.headers.get(header)
        if not value:
            continue
        saw_hint = True
        if "tools" in enriched:
            continue
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            enriched["tools"] = parsed
    for header in ("tool-choice", "x-tool-choice"):
        value = req.headers.get(header)
        if not value:
            continue
        saw_hint = True
        if "tool_choice" in enriched:
            continue
        try:
            enriched["tool_choice"] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            enriched["tool_choice"] = value
    for header in ("parallel-tool-calls", "x-parallel-tool-calls"):
        value = req.headers.get(header)
        if not value:
            continue
        saw_hint = True
        if "parallel_tool_calls" not in enriched:
            enriched["parallel_tool_calls"] = value.strip().lower() in {
                "1", "true", "yes", "on",
            }
    if saw_hint:
        enriched["_tool_protocol_requested"] = True
    return enriched


def _chat_history_key(messages: list[dict], length: int | None = None) -> str:
    selected = messages if length is None else messages[:length]
    encoded = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _chat_history_prefix_keys(messages: list[dict]) -> list[str]:
    """Hash all non-empty prefixes while serializing each message once."""

    digest = hashlib.sha256(b"[")
    keys: list[str] = []
    for index, message in enumerate(messages):
        if index:
            digest.update(b",")
        encoded = json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest.update(encoded.encode("utf-8"))
        # Closing only the snapshot preserves the canonical JSON array hash
        # while the running digest remains ready for the next message.
        prefix = digest.copy()
        prefix.update(b"]")
        keys.append(prefix.hexdigest())
    return keys


def _chat_has_prior_turn(messages: list[dict]) -> bool:
    return any(
        isinstance(message, dict)
        and message.get("role") in {"assistant", "tool", "function"}
        for message in messages
    )


def _prune_chat_sessions(now: float) -> None:
    cutoff = now - _CHAT_SESSION_TTL
    while _CHAT_HISTORY_SESSIONS:
        key, (_session_id, touched) = next(iter(_CHAT_HISTORY_SESSIONS.items()))
        if touched >= cutoff and len(_CHAT_HISTORY_SESSIONS) <= _CHAT_SESSION_MAX:
            break
        _CHAT_HISTORY_SESSIONS.pop(key, None)

    for session_id, lease in list(_UPSTREAM_SESSION_LEASES.items()):
        expired = lease.last_client_activity < cutoff
        oversized = len(_UPSTREAM_SESSION_LEASES) > _CHAT_SESSION_MAX
        if lease.active_streams or (not expired and not oversized):
            continue
        _UPSTREAM_SESSION_LEASES.pop(session_id, None)
        for key, (history_session_id, _touched) in list(
            _CHAT_HISTORY_SESSIONS.items()
        ):
            if history_session_id == session_id:
                _CHAT_HISTORY_SESSIONS.pop(key, None)


def _touch_chat_session(session_id: str, now: Optional[float] = None) -> None:
    """Record client activity without looking up or refreshing authentication."""

    if not session_id:
        return
    now = time.monotonic() if now is None else now
    with _CHAT_SESSION_LOCK:
        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        if lease is None:
            return
        lease.last_client_activity = now
        _UPSTREAM_SESSION_LEASES.move_to_end(session_id)
        for key, (history_session_id, _touched) in list(
            _CHAT_HISTORY_SESSIONS.items()
        ):
            if history_session_id == session_id:
                _CHAT_HISTORY_SESSIONS[key] = (session_id, now)
                _CHAT_HISTORY_SESSIONS.move_to_end(key)


def _capture_chat_session_auth(session_id: str, token: str) -> None:
    """Persist the token obtained on a first turn for later continuations."""

    if not session_id or not token:
        return
    with _CHAT_SESSION_LOCK:
        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        if lease is not None and not lease.auth_token:
            lease.auth_token = token
    _touch_chat_session(session_id)


def _rebind_chat_session_account(
    session_id: str,
    account_id: str,
    billing_id: str,
    token: str,
    provider_specific: Optional[Mapping[str, Any]] = None,
) -> None:
    """Keep the relay session lease aligned with a retry account."""

    if not session_id:
        return
    with _CHAT_SESSION_LOCK:
        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        if lease is None:
            return
        lease.account_id = str(account_id or lease.account_id)
        lease.billing_id = str(billing_id or lease.billing_id or lease.account_id)
        lease.auth_token = str(token or lease.auth_token)
        if provider_specific is not None:
            lease.provider_specific = dict(provider_specific)
        lease.last_client_activity = time.monotonic()
        _UPSTREAM_SESSION_LEASES.move_to_end(session_id)


def _begin_chat_stream(session_id: str) -> None:
    if not session_id:
        return
    with _CHAT_SESSION_LOCK:
        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        if lease is None:
            return
        lease.active_streams += 1
    _touch_chat_session(session_id)


def _end_chat_stream(session_id: str) -> None:
    if not session_id:
        return
    with _CHAT_SESSION_LOCK:
        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        if lease is None:
            return
        lease.active_streams = max(0, lease.active_streams - 1)
    _touch_chat_session(session_id)


async def _lease_stream(source, session_id: str):
    """Keep the session lease alive while the API client consumes an SSE body."""

    _begin_chat_stream(session_id)
    try:
        async for chunk in source:
            _touch_chat_session(session_id)
            yield chunk
    finally:
        _end_chat_stream(session_id)


async def _reap_idle_chat_sessions() -> int:
    now = time.monotonic()
    with _CHAT_SESSION_LOCK:
        before = len(_UPSTREAM_SESSION_LEASES)
        _prune_chat_sessions(now)
        return before - len(_UPSTREAM_SESSION_LEASES)


def _bind_chat_session(
    messages: list[dict],
    options: dict,
    *,
    requested_session_id: str = "",
    rotate_for_new: bool = True,
) -> dict:
    """Attach a stable upstream session and account to one API request."""
    history_keys = _chat_history_prefix_keys(messages)
    history_key = history_keys[-1] if history_keys else _chat_history_key(messages)
    now = time.monotonic()
    with _CHAT_SESSION_LOCK:
        _prune_chat_sessions(now)
        polling_enabled = bool(auth.get_polling_status().get("enabled"))
        session_id = requested_session_id.strip()
        inferred_account_snapshot: tuple[str, dict] | None = None
        if not session_id and _chat_has_prior_turn(messages):
            # Capture the currently selected account once for an inferred
            # replay. This both makes the account comparison atomic and lets a
            # manual account switch take effect without consulting the mutable
            # legacy getters separately.
            inferred_account_snapshot = auth.get_active_account_snapshot()
            # Prefer the most specific known prefix, then fall back to the full
            # replay for idempotent retries of the same request.
            for prefix_key in reversed(history_keys):
                found = _CHAT_HISTORY_SESSIONS.get(prefix_key)
                if found is not None:
                    candidate_id = found[0]
                    candidate_lease = _UPSTREAM_SESSION_LEASES.get(candidate_id)
                    # A client that switched accounts and then replayed a full
                    # OpenAI history without a relay session id must start a
                    # fresh upstream conversation. Explicit session ids (used
                    # by Responses continuation) bypass this branch and keep
                    # their original credential by design.
                    active_account = str(
                        (inferred_account_snapshot or ("", {}))[0] or ""
                    )
                    if (
                        not polling_enabled
                        and active_account
                        and candidate_lease is not None
                        and candidate_lease.account_id
                        and candidate_lease.account_id != active_account
                    ):
                        continue
                    session_id = candidate_id
                    _CHAT_HISTORY_SESSIONS.move_to_end(prefix_key)
                    break
        if not session_id:
            session_id = uuid_mod.uuid4().hex

        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        # A reservation belongs to this API request, not to the logical chat
        # lease.  It is consumed by the selected transport (or cancelled by
        # the dispatch wrapper if routing fails before a transport opens).
        slot_reserved = False
        slot_reservation_account = ""
        slot_reservation_id = ""
        if lease is None:
            upstream_mode = _current_upstream_mode()
            slot_mode = upstream_mode not in ("", "cli")
            selected_snapshot: tuple[str, dict] | None = None
            account_rows = auth.get_accounts_raw() if slot_mode else []
            if rotate_for_new and slot_mode and account_rows:
                # Selection and reservation are one operation.  This is what
                # lets credit-priority polling fill every entitlement slot on
                # the richest account before moving to the next account.
                selected_snapshot = auth.reserve_model_account(
                    require_polling=polling_enabled
                )
                if selected_snapshot[0]:
                    slot_reserved = True
                    slot_reservation_account = selected_snapshot[0]
                    slot_reservation_id = str(
                        selected_snapshot[1].get("_slot_reservation_id") or ""
                    )
            if rotate_for_new and selected_snapshot is None and upstream_mode in (
                "raw",
                "direct",
                "web",
                "remote",
                "9router",
                "trae-remote",
                "auto",
                "ide",
                "solo",
                "work",
                "work-agent",
            ):
                # Keep the legacy rotation hook for single-token/test
                # deployments that have no persisted account rows.  Known
                # accounts are selected by ``reserve_model_account`` above.
                auth.next_polling_account()
            # Read the selected id and its credential record as one snapshot.
            # Separate getter calls can race an account rotation and bind an
            # account id to a different account's token.
            if selected_snapshot is not None and selected_snapshot[0]:
                account_id, record = selected_snapshot
            elif inferred_account_snapshot is not None and not polling_enabled:
                account_id, record = inferred_account_snapshot
            else:
                account_id, record = auth.get_active_account_snapshot()
            # A disabled account may remain selected for check-in management.
            # New sessions can select another enabled account; continuations
            # keep their credential and are refused while it is disabled.
            if not account_id or not auth._record_model_enabled(record):
                account_id, record = auth.get_model_account_snapshot()
            # Keep compatibility with integrations/tests that replace the
            # legacy getters while still preferring the atomic snapshot in
            # normal operation.
            if not account_id:
                # An account store with no eligible rows must not fall back to
                # the management-selected (possibly disabled) credential.
                if auth.get_accounts_raw():
                    raise _ModelAccountUnavailable(
                        "No enabled model account available; enable an account in "
                        "账号与签到"
                    )
                account_id = auth.get_active_account_id() or ""
            if account_id and not record:
                record = auth.get_account_record(account_id)
            try:
                _ensure_model_account_enabled(
                    account_id, str(record.get("token") or "")
                )
            except BaseException:
                if slot_reserved:
                    auth.cancel_account_slot_reservation(
                        slot_reservation_account or account_id,
                        slot_reservation_id,
                    )
                raise
            token = str(record.get("token") or "")
            if not token and not account_id:
                # Preserve the CLI/env single-credential path when the
                # account store is empty.  Once accounts are stored, an
                # all-disabled state must fail explicitly instead of silently
                # bypassing the per-account switch.
                if auth.get_accounts_raw():
                    raise _ModelAccountUnavailable(
                        "No enabled model account available; enable an account in "
                        "账号与签到"
                    )
                token = str(auth.get_token() or "")
            billing_id = _account_id_from_token(token) or account_id
            lease = _UpstreamSessionLease(
                account_id=account_id,
                billing_id=billing_id,
                auth_token=token,
                last_client_activity=now,
                provider_specific=dict(
                    record.get("provider_specific")
                    or record.get("providerSpecificData")
                    or {}
                ),
            )
            _UPSTREAM_SESSION_LEASES[session_id] = lease
        else:
            # A continuation must stay on the credential captured for its first
            # turn. Do not rotate accounts, call refresh, or mutate global auth.
            _ensure_model_account_enabled(lease.account_id, lease.auth_token)
            if (
                _current_upstream_mode() not in ("", "cli")
                and auth.get_accounts_raw()
            ):
                # A logical session can have multiple simultaneous turns.  Each
                # turn therefore reserves its own account slot while retaining
                # the session's credential affinity.
                selected_snapshot = auth.reserve_model_account(
                    preferred_account_id=lease.account_id
                )
                if selected_snapshot[0]:
                    slot_reserved = True
                    slot_reservation_account = selected_snapshot[0]
                    slot_reservation_id = str(
                        selected_snapshot[1].get("_slot_reservation_id") or ""
                    )
            lease.last_client_activity = now
            _UPSTREAM_SESSION_LEASES.move_to_end(session_id)

        _CHAT_HISTORY_SESSIONS[history_key] = (session_id, now)
        _CHAT_HISTORY_SESSIONS.move_to_end(history_key)

    bound = dict(options)
    bound["session_id"] = session_id
    if slot_reserved:
        bound["_account_slot_reserved"] = True
        bound["_account_slot_reservation_account"] = (
            slot_reservation_account or lease.account_id
        )
        bound["_account_slot_reservation_id"] = slot_reservation_id
    if lease.account_id:
        bound["_account_id"] = lease.account_id
        if _current_upstream_mode() in ("raw", "direct", "auto"):
            bound["_auth_user_id"] = lease.billing_id or lease.account_id
    if lease.billing_id:
        bound["_billing_id"] = lease.billing_id
    if lease.auth_token:
        bound["_auth_token"] = lease.auth_token
    if _current_upstream_mode() != "cli":
        # Keep account metadata pinned with the credential. The global auth
        # state may switch before the remote request has built its headers.
        # An explicit empty mapping prevents fallback to another account's
        # mutable global metadata.
        bound["provider_specific"] = dict(lease.provider_specific)
    return bound


def _validate_chat_options(options: dict) -> Optional[JSONResponse]:
    """Validate the OpenAI tool surface before choosing an upstream route."""

    tool_names: set[str] = set()
    if "tools" in options:
        tools = options["tools"]
        if not isinstance(tools, list):
            return _openai_error(
                400, "tools must be an array", "invalid_request_error", "tools"
            )
        normalized_tools = []
        for index, tool in enumerate(tools):
            param = f"tools.{index}"
            if not isinstance(tool, dict) or tool.get("type") != "function":
                return _openai_error(
                    400,
                    f"{param} must be an OpenAI function tool",
                    "invalid_request_error",
                    param,
                )
            function = tool.get("function")
            if not isinstance(function, dict):
                # Accept the compact Responses-style flat tool shape and
                # normalize it to the Chat-completions nested function shape.
                function = {
                    key: tool.get(key)
                    for key in ("name", "description", "parameters", "strict")
                    if key in tool
                }
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                return _openai_error(
                    400,
                    f"{param}.function.name must be a non-empty string",
                    "invalid_request_error",
                    f"{param}.function.name",
                )
            if name in tool_names:
                return _openai_error(
                    400,
                    f"Duplicate tool name: {name}",
                    "invalid_request_error",
                    f"{param}.function.name",
                )
            parameters = function.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                return _openai_error(
                    400,
                    f"{param}.function.parameters must be an object",
                    "invalid_request_error",
                    f"{param}.function.parameters",
                )
            tool_names.add(name)
            normalized_tools.append({"type": "function", "function": dict(function)})
        if normalized_tools:
            options["tools"] = normalized_tools

    if "tool_choice" in options:
        tool_choice = options["tool_choice"]
        if isinstance(tool_choice, str):
            if tool_choice not in ("none", "auto", "required"):
                return _openai_error(
                    400,
                    "tool_choice must be none, auto, required, or a named function",
                    "invalid_request_error",
                    "tool_choice",
                )
            if tool_choice == "required" and not tool_names:
                return _openai_error(
                    400,
                    "tool_choice=required requires at least one tool",
                    "invalid_request_error",
                    "tool_choice",
                )
        elif isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if not isinstance(function, dict):
                # Accept the compact Responses-style flat tool_choice shape too.
                function = tool_choice
            name = function.get("name") if isinstance(function, dict) else None
            if tool_choice.get("type") != "function" or not isinstance(name, str):
                return _openai_error(
                    400,
                    "tool_choice must select a named function",
                    "invalid_request_error",
                    "tool_choice",
                )
            if name not in tool_names:
                return _openai_error(
                    400,
                    f"tool_choice references undeclared tool: {name}",
                    "invalid_request_error",
                    "tool_choice",
                )
            options["tool_choice"] = {"type": "function", "function": {"name": name}}
        else:
            return _openai_error(
                400,
                "tool_choice must be a string or object",
                "invalid_request_error",
                "tool_choice",
            )

    if "parallel_tool_calls" in options and not isinstance(
        options["parallel_tool_calls"], bool
    ):
        return _openai_error(
            400,
            "parallel_tool_calls must be a boolean",
            "invalid_request_error",
            "parallel_tool_calls",
        )

    for key in ("client_context", "clientContext"):
        if key in options and not isinstance(options[key], dict):
            return _openai_error(
                400,
                f"{key} must be an object",
                "invalid_request_error",
                key,
            )

    alias_pairs = (
        ("client_context", "clientContext"),
        ("session_id", "sessionId"),
        ("max_tokens", "maxTokens"),
    )
    for canonical, alias in alias_pairs:
        if canonical in options and alias in options and options[canonical] != options[alias]:
            return _openai_error(
                400,
                f"{canonical} and {alias} must not conflict",
                "invalid_request_error",
                canonical,
            )

    for key in ("session_id", "sessionId"):
        if key in options:
            value = options[key]
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or "\x00" in value
            ):
                return _openai_error(
                    400,
                    f"{key} must be a non-empty string of at most 256 characters",
                    "invalid_request_error",
                    key,
                )

    for key in ("max_tokens", "maxTokens", "max_completion_tokens"):
        if key in options:
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return _openai_error(
                    400,
                    f"{key} must be a positive integer",
                    "invalid_request_error",
                    key,
                )
    return None


def _number_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, (int, float)):
        return None
    return max(0, value)



def _credit_round(value: int | float | None) -> int | float | None:
    """Round credit values to 2 decimal places for consistent display."""
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None

def _first_number(data: Mapping[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = _number_value(data.get(key))
        if value is not None:
            return value
        # A few native adapters serialize accounting numbers as strings.
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = float(raw.strip())
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and parsed >= 0:
                return parsed
    return None


def _nested_credit_value(value: Any, depth: int = 0) -> int | float | None:
    """Find an explicit credit field without mistaking tokens or money for it."""

    if depth > 6:
        return None
    keys = (
        "credits_consumed",
        "consumed_credits",
        "credit_cost",
        "credits_cost",
        "credits_float",
        "creditsFloat",
        "amount_float",
        "amountFloat",
    )
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                candidate = _first_number(value, key)
                if candidate is not None:
                    return candidate
        preferred = (
            "user_usage_group_by_session",
            "userUsageGroupBySession",
            "usage",
            "billing",
            "cost",
            "response_meta",
            "responseMeta",
            "data",
        )
        for key in preferred:
            if key in value:
                candidate = _nested_credit_value(value.get(key), depth + 1)
                if candidate is not None:
                    return candidate
        for child in value.values():
            candidate = _nested_credit_value(child, depth + 1)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = _nested_credit_value(child, depth + 1)
            if candidate is not None:
                return candidate
    return None


def _usage_values(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "credits_consumed": None,
        }
    prompt = _first_number(
        usage, "prompt_tokens", "input_tokens", "input_token", "inputTokens"
    ) or 0
    completion = _first_number(
        usage,
        "completion_tokens",
        "output_tokens",
        "output_token",
        "outputTokens",
    ) or 0
    total = _first_number(usage, "total_tokens", "total_token", "totalTokens")
    if total is None:
        total = prompt + completion
    credits = _nested_credit_value(usage)
    cached = _first_number(
        usage,
        "cached_tokens",
        "cache_read_tokens",
        "cacheReadTokens",
        "cachedTokens",
        "cache_read_input_tokens",
    )
    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(details_key)
        if cached is None and isinstance(details, Mapping):
            cached = _first_number(details, "cached_tokens", "cache_read_tokens")
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "cached_tokens": int(cached or 0),
        "credits_consumed": credits,
    }


def _optional_bool(value: Any) -> bool | None:
    """Normalize persisted/request metadata without guessing missing values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled", "是", "有"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled", "否", "无"}:
            return False
    return None


def _normalize_usage_thinking(value: Any) -> dict[str, Any] | str | None:
    """Keep a compact, non-sensitive copy of the caller's thinking hint.

    Request options can contain the full Anthropic/OpenAI ``thinking`` object.
    Usage history only needs the small routing/display fields; persisting the
    whole object would make records noisy and could retain adapter-specific
    payloads indefinitely.
    """

    if value in (None, "", False):
        return None
    if isinstance(value, Mapping):
        allowed = (
            "type",
            "enabled",
            "enable",
            "effort",
            "level",
            "budget_tokens",
            "budgetTokens",
        )
        compact = {
            key: value[key]
            for key in allowed
            if key in value and value[key] not in (None, "")
        }
        return compact or None
    if isinstance(value, bool):
        return {"type": "enabled"} if value else None
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    return text or None


def _normalize_usage_product(value: Any) -> str | None:
    """Normalize route products to the public ``trae``/``traework`` labels."""

    if value in (None, ""):
        return None
    product = str(value).strip().lower().replace("_", "-")
    if product in {
        "work",
        "work-agent",
        "remote-work",
        "trae-work",
        "traework",
        "traework-native",
        "native",
        "solo-work",
    }:
        return "traework"
    if product in {"trae", "trae-cn", "remote", "ide", "solo"}:
        return "trae"
    # Do not mistake an HTTP path (the legacy ``endpoint`` field) for a
    # product when normalizing old rows. Unknown products remain unknown;
    # callers can still inspect the raw route fields separately.
    return None


def _context_mode_bool(value: Any) -> bool | None:
    """Map known context labels to a Max-enabled boolean.

    Unknown labels stay ``None`` instead of being guessed as standard.  This
    matters when normalizing old records written by a custom adapter.
    """

    if value in (None, ""):
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "")
    if normalized in {"max", "max_1m", "max1m", "1m", "1000000", "agent_max", "max_mode"}:
        return True
    if normalized in {
        "standard",
        "default",
        "work",
        "dev",
        "200k",
        "200000",
        "normal",
    }:
        return False
    return None


def _option_flag(value: Any) -> bool:
    """Return a strict truth value for request flags."""

    return _optional_bool(value) is True


def _tool_catalog_has_entries(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return value not in (None, "", False)


def _tools_requested_from_options(options: Optional[Mapping[str, Any]]) -> bool:
    """Whether this request carries a caller tool protocol.

    ``_tool_protocol_requested`` can also be set for a continuation that only
    carries tool history.  An explicit ``tools=[]`` remains false unless an
    inherited catalog or a tool choice says otherwise.
    """

    options = options or {}
    # An explicit catalog, including ``tools=[]``, owns this turn.  Do not
    # revive a previous Responses catalog from ``_inherited_tools`` when the
    # caller intentionally supplied an empty tools array.
    if "tools" in options:
        return _tool_catalog_has_entries(options.get("tools"))
    if _tool_catalog_has_entries(options.get("_inherited_tools")):
        return True
    if "_inherited_tools" not in options:
        if _option_flag(options.get("_tool_protocol_requested")):
            return True
    choice = options.get("tool_choice")
    if choice not in (None, "", "none", False):
        return True
    return False


def _requested_reasoning_effort(options: Optional[Mapping[str, Any]]) -> str | None:
    """Extract the caller-facing thinking strength without normalizing labels."""

    options = options or {}
    for key in ("reasoning_effort", "reasoningEffort", "thinking_effort"):
        value = options.get(key)
        if value not in (None, ""):
            return str(value).strip() or None
    reasoning = options.get("reasoning")
    if isinstance(reasoning, Mapping):
        value = reasoning.get("effort") or reasoning.get("level")
        if value not in (None, ""):
            return str(value).strip() or None
    thinking = options.get("thinking")
    if isinstance(thinking, Mapping):
        value = (
            thinking.get("effort")
            or thinking.get("level")
            or thinking.get("budget_tokens")
            or thinking.get("budgetTokens")
        )
        if value not in (None, ""):
            return str(value).strip() or None
        if _option_flag(thinking.get("enabled")) or _option_flag(thinking.get("enable")):
            return "enabled"
    elif thinking not in (None, "", False):
        return "enabled"
    return None


def _context_mode_metadata(
    options: Optional[Mapping[str, Any]],
) -> tuple[str | None, str | None, int | float | None, bool | None]:
    """Return requested/actual context mode and its known token window."""

    options = options or {}
    trace = options.get("_upstream_trace")
    trace = trace if isinstance(trace, Mapping) else {}
    if "trae_max_mode" in options:
        requested_max = _option_flag(options["trae_max_mode"])
    elif "max_mode" in options:
        requested_max = _option_flag(options["max_mode"])
    else:
        requested_max = bool(auth.get_max_mode_settings().get("enabled"))
    applied_marker = trace.get("max_mode_applied") if "max_mode_applied" in trace else None
    applied_max = _optional_bool(applied_marker)
    requested = "max_1m" if requested_max else "standard"
    actual = (
        "max_1m"
        if applied_max is True
        else "standard"
        if applied_max is False
        # A request that explicitly/defaults to standard has no separate
        # upstream max acknowledgement to wait for.  Keep ordinary requests
        # useful in the console while preserving "unknown" for a requested
        # Max turn whose upstream trace never confirmed the mode.
        else "standard"
        if not requested_max
        else None
    )
    # A max acknowledgement does not itself prove the descriptor's context
    # window.  Keep the value unknown unless the bound model descriptor/trace
    # supplied an explicit token count; never synthesize a 1M capability.
    window = _number_value(trace.get("max_context_tokens"))
    return requested, actual, window, applied_max


def _tool_call_count(value: Any) -> int:
    """Count structured tool calls in one translated model payload."""

    if not isinstance(value, Mapping):
        return 0
    value_type = str(value.get("type") or "").strip().lower()
    if value_type in {
        "function_call",
        "custom_tool_call",
        "tool_call",
        "tool_use",
    }:
        return 1
    count = 0
    calls = value.get("tool_calls")
    if isinstance(calls, list):
        count += sum(1 for item in calls if isinstance(item, Mapping))
    elif isinstance(calls, Mapping):
        count += 1
    if isinstance(value.get("function_call"), Mapping):
        count += 1
    for key in ("message", "delta", "output", "response", "item", "items", "content"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            count += _tool_call_count(nested)
        elif isinstance(nested, list):
            count += sum(_tool_call_count(item) for item in nested)
    choices = value.get("choices")
    if isinstance(choices, list):
        count += sum(_tool_call_count(item) for item in choices)
    return count


def _request_account_identity() -> tuple[str, str]:
    token = auth.get_token() or ""
    token_identity = _account_id_from_token(token)
    if token_identity:
        return token_identity, token
    account_id = (
        auth.get_active_account_id()
        or auth.get_user_id()
        or token[:16]
        or "default"
    )
    return str(account_id), token


def _account_id_from_token(token: str) -> str:
    """Extract the immutable Trae account id carried by a JWT, if present."""

    raw_token = str(token or "").strip()
    if not raw_token:
        return ""
    try:
        parts = raw_token.split(".")
        if len(parts) < 2:
            return ""
        encoded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    data = payload.get("data")
    if isinstance(data, Mapping):
        for key in ("id", "user_id", "userId", "sub"):
            value = data.get(key)
            if value not in (None, ""):
                return str(value)
    for key in ("user_id", "userId", "sub"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


async def _fetch_used_credits(token: str) -> int | float | None:
    if not token:
        return None
    try:
        raw = await trae_client.fetch_account_total_credits(token)
        parsed = trae_client.parse_account_credits(raw)
        return _number_value(parsed.get("used"))
    except Exception as exc:
        logger.debug("usage credit snapshot unavailable: %s", exc)
        return None


def _credit_reconcile_interval_seconds() -> float:
    """Return the cadence for official total-credit reconciliation."""

    try:
        value = float(
            os.environ.get("TRAE_CREDIT_RECONCILE_INTERVAL_SECONDS", "120")
            or "120"
        )
    except (TypeError, ValueError):
        value = 120.0
    return max(15.0, min(value, 86_400.0))


def _credit_settled_timestamp(record: Mapping[str, Any]) -> float:
    value = record.get("credits_settled_at")
    if value in (None, ""):
        value = record.get("timestamp")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _credit_source_weight(record: Mapping[str, Any]) -> float:
    """Return confidence for a confirmed charge source.

    Session usage and explicit upstream usage are authoritative.  Account
    total deltas are reliable but can race another request, so they receive a
    slightly lower weight.  Pending/unknown rows never reduce the displayed
    balance.
    """

    source = str(record.get("credits_source") or "").strip().lower()
    if source in {"session_usage", "upstream", "manual"}:
        return 1.0
    if source in {"snapshot_delta", "account_total"}:
        return 0.85
    return 0.0


def _settled_credit_overlay(
    account_id: str, anchor_at: float
) -> tuple[float, float, float, int, float]:
    """Return weighted post-anchor charges for one account.

    The tuple is ``(weighted, raw, confidence, count, latest_timestamp)``.
    Rows are read under the usage lock, but account persistence happens after
    the lock is released so usage updates and account refreshes cannot deadlock.
    """

    if not account_id or anchor_at <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0
    weighted = raw = confidence_total = 0.0
    count = 0
    latest = 0.0
    with _USAGE_LOCK:
        records = tuple(_USAGE_HISTORY)
    for record in records:
        if str(record.get("account_id") or "") != str(account_id):
            continue
        amount = _number_value(record.get("credits_consumed"))
        settled_at = _credit_settled_timestamp(record)
        # Include the boundary instant.  The provider and local settlement can
        # legitimately share the same timestamp (especially in tests or when
        # the billing response is handled in the same event-loop tick); using
        # ``<=`` would then hide a confirmed charge from the projection.
        if amount is None or amount < 0 or settled_at < anchor_at:
            continue
        weight = _credit_source_weight(record)
        if weight <= 0:
            continue
        weighted += float(amount) * weight
        raw += float(amount)
        confidence_total += weight
        count += 1
        latest = max(latest, settled_at)
    confidence = (confidence_total / count) if count else 0.0
    return (
        _credit_round(weighted) or 0.0,
        _credit_round(raw) or 0.0,
        round(min(1.0, confidence), 4),
        count,
        latest,
    )


def _credit_projection_for_account(
    account_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a local balance projection from the latest official anchor."""

    credits = auth.get_account_credit_state(account_id)
    if not credits:
        return {}, {}
    official = _number_value(
        credits.get("official_remaining")
        if "official_remaining" in credits
        else credits.get("remaining")
    )
    if official is None or credits.get("unlimited"):
        return credits, {
            "remaining": credits.get("remaining"),
            "local_debit": 0.0,
            "confidence": 1.0,
            "status": "unlimited" if credits.get("unlimited") else "unknown",
            "source": "upstream",
            "latest": 0.0,
        }
    anchor = credits.get("credits_anchor_at") or credits.get(
        "credits_reconciled_at"
    )
    try:
        anchor_at = float(anchor or 0)
    except (TypeError, ValueError):
        anchor_at = 0.0
    weighted, raw, confidence, count, latest = _settled_credit_overlay(
        account_id, anchor_at
    )
    effective = max(0.0, float(official) - float(weighted))
    projection = {
        "remaining": _credit_round(effective),
        "local_debit": _credit_round(weighted) or 0.0,
        "confidence": confidence if count else 1.0,
        "status": "projected" if count else "ok",
        "source": "usage_weighted" if count else "upstream",
        "latest": latest,
        "raw_debit": _credit_round(raw) or 0.0,
    }
    return credits, projection


def _refresh_credit_projection(account_id: str) -> dict[str, Any]:
    """Persist the current post-settlement balance projection."""

    credits, projection = _credit_projection_for_account(account_id)
    if not credits or not projection:
        return credits
    return auth.update_account_credit_projection(
        account_id,
        effective_remaining=projection.get("remaining"),
        local_debit=projection.get("local_debit", 0),
        status=projection.get("status", "projected"),
        source=projection.get("source", "usage_weighted"),
        confidence=projection.get("confidence", 0),
        settled_at=projection.get("latest"),
    ) or credits


def _credit_snapshot_projection(
    account_id: str, parsed: Mapping[str, Any], observed_at: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combine an upstream snapshot with charges not yet reflected upstream."""

    previous = auth.get_account_credit_state(account_id)
    previous_used = _number_value(
        previous.get("official_used")
        if "official_used" in previous
        else previous.get("used")
    )
    incoming_used = _number_value(parsed.get("used"))
    upstream_delta = 0.0
    if previous_used is not None and incoming_used is not None:
        upstream_delta = max(0.0, float(incoming_used) - float(previous_used))
    # Charges after the previous anchor may already be included in the new
    # upstream ``used`` counter.  Subtract only the weighted remainder.
    anchor = previous.get("credits_anchor_at") or previous.get(
        "credits_reconciled_at"
    )
    try:
        anchor_at = float(anchor or 0)
    except (TypeError, ValueError):
        anchor_at = 0.0
    weighted, raw, confidence, count, latest = _settled_credit_overlay(
        account_id, anchor_at
    )
    unmatched = max(0.0, float(weighted) - upstream_delta)
    official_remaining = _number_value(parsed.get("remaining"))
    if official_remaining is None:
        effective = parsed.get("remaining")
    else:
        effective = _credit_round(max(0.0, float(official_remaining) - unmatched))
    state = {
        "local_debit": _credit_round(unmatched) or 0.0,
        "confidence": confidence if count else 1.0,
        "status": "projected" if unmatched > 0 else "ok",
        "source": "upstream_weighted" if unmatched > 0 else "upstream",
        "raw_debit": _credit_round(raw) or 0.0,
        "upstream_delta": _credit_round(upstream_delta) or 0.0,
        "latest": latest,
    }
    return state, {"effective_remaining": effective}


def _begin_credit_snapshot(account_id: str, token: str) -> bool:
    if not account_id or account_id == "default" or not token:
        return False
    with _USAGE_LOCK:
        active = _USAGE_ACTIVE_ACCOUNTS.get(account_id, 0)
        if active == 0:
            # A new in-flight batch starts a fresh attribution window.  Do
            # not let a previous test/request's terminal total become the
            # baseline for this batch.
            _USAGE_LAST_CREDIT_TOTALS.pop(account_id, None)
        if active:
            _USAGE_UNSAFE_ACCOUNTS.add(account_id)
        _USAGE_ACTIVE_ACCOUNTS[account_id] = active + 1
        # A delta is only attributable when this account has one request in
        # flight. Concurrent calls share the same upstream counter.
        return active == 0


def _end_credit_snapshot(account_id: str) -> bool:
    """Release one live snapshot and return whether it was the last owner."""

    if not account_id:
        return False
    with _USAGE_LOCK:
        active = _USAGE_ACTIVE_ACCOUNTS.get(account_id, 0)
        if active <= 1:
            _USAGE_ACTIVE_ACCOUNTS.pop(account_id, None)
            _USAGE_UNSAFE_ACCOUNTS.discard(account_id)
            # The shared before task is no longer useful once every request
            # in the batch has settled.  Remove only the map entry here; the
            # caller decides whether it is safe to cancel the task.
            for key in tuple(_USAGE_ACCOUNT_BEFORE_TASKS):
                if key[1] == account_id:
                    _USAGE_ACCOUNT_BEFORE_TASKS.pop(key, None)
            for key in tuple(_USAGE_CREDIT_SETTLE_GATES):
                if key[1] == account_id:
                    _USAGE_CREDIT_SETTLE_GATES.pop(key, None)
            return True
        else:
            _USAGE_ACTIVE_ACCOUNTS[account_id] = active - 1
            return False


def _credit_snapshot_is_safe(account_id: str) -> bool:
    with _USAGE_LOCK:
        return account_id not in _USAGE_UNSAFE_ACCOUNTS


def _shared_credit_before_task(account_id: str, token: str) -> asyncio.Task | None:
    """Return one before-snapshot task shared by concurrent requests."""

    if not account_id or account_id == "default" or not token:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    key = (id(loop), account_id)
    with _USAGE_LOCK:
        existing = _USAGE_ACCOUNT_BEFORE_TASKS.get(key)
        if existing is not None:
            return existing
        task = _spawn_usage_task(
            _fetch_used_credits(token),
            _USAGE_SNAPSHOT_TASKS,
        )
        if task is not None:
            _USAGE_ACCOUNT_BEFORE_TASKS[key] = task
        return task


def _credit_settle_gate(account_id: str) -> asyncio.Lock | None:
    """Get a loop-local gate for account after-snapshot attribution."""

    if not account_id or account_id == "default":
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    key = (id(loop), account_id)
    with _USAGE_LOCK:
        gate = _USAGE_CREDIT_SETTLE_GATES.get(key)
        if gate is None:
            gate = asyncio.Lock()
            _USAGE_CREDIT_SETTLE_GATES[key] = gate
        return gate


def _credit_delta_from_total(
    account_id: str,
    before: int | float | None,
    after: int | float | None,
) -> tuple[float, float] | None:
    """Attribute one monotonic account-total increment.

    The first request in a concurrent batch uses its before snapshot. Later
    requests use the last observed after total as their baseline, preventing
    the same upstream increment from being counted twice.
    """

    before_value = _number_value(before)
    after_value = _number_value(after)
    if after_value is None:
        return None
    with _USAGE_LOCK:
        previous = _number_value(_USAGE_LAST_CREDIT_TOTALS.get(account_id))
        candidates = [value for value in (before_value, previous) if value is not None]
        if not candidates:
            return None
        baseline = max(candidates)
        if after_value < baseline:
            return None
        _USAGE_LAST_CREDIT_TOTALS[account_id] = max(
            after_value,
            previous if previous is not None else after_value,
        )
    return _credit_round(after_value - baseline), _credit_round(baseline)


def _usage_pending_max_age_seconds() -> float:
    try:
        # Commercial usage can lag the model stream by several polling
        # intervals, especially when two requests finish on the same account.
        # Keep a row recoverable long enough for a periodic reconciliation to
        # observe the eventual bill instead of freezing it after one miss.
        value = float(os.environ.get("TRAE_USAGE_PENDING_MAX_AGE_SECONDS", "300"))
    except (TypeError, ValueError):
        value = 300.0
    return max(1.0, min(value, 86_400.0))


def _usage_enrichment_interval_seconds() -> float:
    """Return the cadence for retrying unresolved usage rows."""

    try:
        value = float(
            os.environ.get("TRAE_USAGE_ENRICH_INTERVAL_SECONDS", "5") or "5"
        )
    except (TypeError, ValueError):
        value = 5.0
    return max(1.0, min(value, 60.0))


def _usage_account_token(account_id: str) -> str:
    """Resolve a persisted row to the credential that owns its bill.

    The account store may be keyed by a local user id while usage rows carry
    the immutable JWT subject (or the reverse), especially after polling or
    token refresh. Match both forms before abandoning a recoverable row.
    """

    identity = str(account_id or "").strip()
    if not identity:
        return ""
    direct = auth.get_account_record(identity)
    token = str((direct or {}).get("token") or "").strip()
    if token:
        return token
    for stored_id, record in auth.get_accounts_raw():
        record = record if isinstance(record, Mapping) else {}
        candidate = str(record.get("token") or "").strip()
        if not candidate:
            continue
        if (
            str(stored_id or "") == identity
            or str(record.get("user_id") or "") == identity
            or _account_id_from_token(candidate) == identity
        ):
            return candidate
    return ""


def _usage_enrich_concurrency() -> int:
    try:
        value = int(os.environ.get("TRAE_USAGE_ENRICH_CONCURRENCY", "8") or "8")
    except (TypeError, ValueError):
        value = 8
    return max(1, min(value, 64))


def _spawn_usage_task(
    coro,
    registry: set[asyncio.Task] | None = None,
) -> asyncio.Task | None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # A few management/test paths construct a tracker synchronously.
        # Close the coroutine we were handed so it cannot leak a
        # ``was never awaited`` warning; real request paths always have a
        # running loop and continue to schedule normally.
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        return None
    registry = registry if registry is not None else _USAGE_ENRICH_TASKS
    task = asyncio.create_task(coro)
    registry.add(task)

    def done(completed: asyncio.Task) -> None:
        registry.discard(completed)
        try:
            completed.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("usage enrichment failed: %s", exc)

    task.add_done_callback(done)
    return task


def _schedule_pending_usage_enrichment() -> int:
    """Resume bill lookups for persisted rows after a relay restart."""

    async def enrich_pending(
        record: Mapping[str, Any], token: str, semaphore: asyncio.Semaphore
    ) -> None:
        # A restart-resumed lookup never acquired a live account snapshot.
        # Keep it bounded so a large history cannot burst the usage endpoint.
        request_id = str(record.get("request_id") or "")
        try:
            async with semaphore:
                await _enrich_usage_credits(
                    request_id,
                    str(record.get("account_id") or ""),
                    token,
                    None,
                    usage_turn_id=str(record.get("usage_turn_id") or ""),
                    credit_safe=False,
                    release_snapshot=False,
                )
        finally:
            # Cancellation while waiting for the semaphore bypasses the
            # enrichment coroutine's own cleanup.
            _USAGE_PENDING_TASK_IDS.discard(request_id)

    now = time.time()
    stale_request_ids: list[str] = []
    unrecoverable_request_ids: list[str] = []
    with _USAGE_LOCK:
        pending = []
        for record in _USAGE_HISTORY:
            request_id = str(record.get("request_id") or "")
            # A live request may already have a settlement task. Do not mark
            # that row unavailable from the periodic restart/reconciliation
            # scan while the in-process lookup is still running.
            if request_id and request_id in _USAGE_PENDING_TASK_IDS:
                continue
            source = str(record.get("credits_source") or "").lower()
            if (
                _number_value(record.get("credits_consumed")) is not None
                # ``pending`` is the normal post-stream state.  Keep
                # ``unknown`` rows eligible too because older adapters may
                # have written a terminal row before the usage turn id was
                # discovered by the SSE parser.
                or source not in {"", "unknown", "pending"}
                or not record.get("request_id")
            ):
                continue
            age = max(0.0, now - float(record.get("timestamp") or now))
            if age > _usage_pending_max_age_seconds():
                stale_request_ids.append(str(record.get("request_id") or ""))
                continue
            # A restart cannot reconstruct a missing before-snapshot.  Turn
            # usage ids are recoverable from the commercial endpoint; rows
            # without one are only safe to enrich while the live request is
            # still in this process.
            if record.get("usage_turn_id"):
                pending.append(dict(record))
            else:
                unrecoverable_request_ids.append(str(record.get("request_id") or ""))
    for request_id in stale_request_ids:
        _update_usage_record(
            request_id,
            credits_source="unavailable",
            credits_resolution="expired",
        )
    for request_id in unrecoverable_request_ids:
        _update_usage_record(
            request_id,
            credits_source="unavailable",
            credits_resolution="no_usage_turn",
        )
    scheduled = 0
    concurrency_limit = _usage_enrich_concurrency()
    semaphore = asyncio.Semaphore(concurrency_limit)
    for record in pending:
        request_id = str(record.get("request_id") or "")
        if not request_id or request_id in _USAGE_PENDING_TASK_IDS:
            continue
        account_id = str(record.get("account_id") or "")
        token = _usage_account_token(account_id)
        if not token:
            _update_usage_record(
                request_id,
                credits_source="unavailable",
                credits_resolution="account_token_missing",
            )
            continue
        _USAGE_PENDING_TASK_IDS.add(request_id)
        _spawn_usage_task(enrich_pending(record, token, semaphore))
        scheduled += 1
    if scheduled:
        logger.info(
            "scheduled %d pending usage credit lookup(s) concurrency=%d",
            scheduled,
            concurrency_limit,
        )
    return scheduled


async def _cancel_usage_task(
    task: asyncio.Task | None,
    registry: set[asyncio.Task] | None = None,
) -> None:
    """Cancel and drain one background usage task without leaking exceptions."""
    if task is None:
        return
    try:
        if not task.done():
            task.cancel()
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.debug("usage background task stopped with error: %s", exc)
    finally:
        if registry is not None:
            registry.discard(task)


async def _cancel_usage_tasks() -> None:
    """Cancel and await all usage enrichment/snapshot tasks during shutdown."""
    tasks = set(_USAGE_ENRICH_TASKS) | set(_USAGE_SNAPSHOT_TASKS)
    if not tasks:
        _USAGE_PENDING_TASK_IDS.clear()
        with _USAGE_LOCK:
            _USAGE_ACCOUNT_BEFORE_TASKS.clear()
            _USAGE_CREDIT_SETTLE_GATES.clear()
            _USAGE_LAST_CREDIT_TOTALS.clear()
        return
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _USAGE_ENRICH_TASKS.difference_update(tasks)
    _USAGE_SNAPSHOT_TASKS.difference_update(tasks)
    _USAGE_PENDING_TASK_IDS.clear()
    with _USAGE_LOCK:
        _USAGE_ACCOUNT_BEFORE_TASKS.clear()
        _USAGE_CREDIT_SETTLE_GATES.clear()
        _USAGE_LAST_CREDIT_TOTALS.clear()


def _usage_credit_retry_delays() -> tuple[float, ...]:
    """Return short delays used while a just-finished turn settles upstream."""

    explicit = os.environ.get("TRAE_USAGE_CREDIT_RETRY_DELAYS")
    if explicit:
        values: list[float] = []
        for item in str(explicit).split(","):
            try:
                values.append(max(0.0, min(float(item.strip()), 30.0)))
            except (TypeError, ValueError):
                continue
        if values:
            return tuple(values)
    attempts = _session_usage_query_attempts()
    base = _session_usage_query_delay()
    return tuple(
        0.0 if index == 0 else min(30.0, base * (2 ** (index - 1)))
        for index in range(attempts)
    )


async def _enrich_usage_credits(
    request_id: str,
    account_id: str,
    token: str,
    before_task: asyncio.Task | None,
    *,
    usage_turn_id: str = "",
    credit_safe: bool = True,
    release_snapshot: bool = True,
    allow_shared_snapshot: bool = False,
) -> None:
    if request_id:
        _USAGE_PENDING_TASK_IDS.add(request_id)
    try:
        settle = _credit_settle_seconds()
        if settle:
            await asyncio.sleep(settle)
        if usage_turn_id and _session_usage_enabled():
            for attempt, delay in enumerate(_usage_credit_retry_delays()):
                if attempt and delay:
                    await asyncio.sleep(delay)
                try:
                    session_usage = await trae_client.fetch_session_usage(
                        usage_turn_id,
                        token,
                    )
                    credits = _number_value(
                        session_usage.get("credits_consumed")
                        if isinstance(session_usage, Mapping)
                        else None
                    )
                    # Zero is a valid settled charge. Only None means that
                    # the commercial endpoint has not produced a bill yet.
                    if credits is not None and credits >= 0:
                        _update_usage_record(
                            request_id,
                            credits_consumed=_credit_round(credits),
                            credits_source="session_usage",
                        )
                        return
                    logger.debug(
                        "session usage not settled request=%s attempt=%d",
                        request_id,
                        attempt + 1,
                    )
                except Exception as exc:
                    logger.debug(
                        "session usage enrichment unavailable request=%s attempt=%d: %s",
                        request_id,
                        attempt + 1,
                        exc,
                    )
            # A real upstream turn id is preferred.  For a live request that
            # also owns a before snapshot, however, the account-total delta is
            # a useful bounded fallback when the commercial turn endpoint is
            # temporarily unavailable.  This prevents every other concurrent
            # request from remaining in "settling" forever.  Restart-resumed
            # rows have no before snapshot and must remain pending so they are
            # never charged from an unrelated account delta.
            if request_id and usage_turn_id and (
                not credit_safe or before_task is None
            ):
                _update_usage_record(
                    request_id,
                    credits_source="pending",
                    credits_resolution="session_usage_pending",
                )
                return
        if (
            (not credit_safe and not allow_shared_snapshot)
            or not account_id
            or account_id == "default"
        ):
            if request_id:
                _update_usage_record(
                    request_id,
                    credits_source=(
                        "pending"
                        if usage_turn_id
                        else "unavailable"
                    ),
                    credits_resolution=(
                        "concurrent_pending"
                        if usage_turn_id
                        else "concurrent_or_unbound"
                    ),
                )
            return
        # Snapshot deltas are only safe for a single in-flight account turn.
        # ``allow_shared_snapshot`` is retained for compatibility with older
        # callers, but it must not opt concurrent requests into false
        # per-turn attribution.
        if not credit_safe and not allow_shared_snapshot:
            if request_id:
                _update_usage_record(
                    request_id,
                    credits_source="unavailable",
                    credits_resolution="concurrent_snapshot_unattributable",
                )
            return
        before = None
        if before_task is not None:
            try:
                before = await before_task
            except Exception:
                before = None
        if before is None:
            if request_id:
                _update_usage_record(
                    request_id,
                    # Keep a bound turn recoverable.  The account entitlement
                    # endpoint can transiently return no data while the
                    # commercial bill is being committed; the background
                    # reconciler will retry this row.
                    credits_source=(
                        "pending" if usage_turn_id else "unavailable"
                    ),
                    credits_resolution=(
                        "before_snapshot_pending"
                        if usage_turn_id
                        else "before_snapshot_unavailable"
                    ),
                )
            return
        gate = _credit_settle_gate(account_id)
        if gate is None:
            after = await _fetch_used_credits(token)
            delta = _credit_delta_from_total(account_id, before, after)
            if delta is None:
                _update_usage_record(
                    request_id,
                    credits_source=(
                        "pending" if usage_turn_id else "unavailable"
                    ),
                    credits_resolution=(
                        "after_snapshot_pending"
                        if usage_turn_id
                        else "after_snapshot_unavailable"
                    ),
                )
                return
            credits, baseline = delta
            _update_usage_record(
                request_id,
                credits_consumed=credits,
                credits_source="snapshot_delta",
                credits_before=baseline,
                credits_after=_credit_round(after),
            )
            return
        async with gate:
            after = await _fetch_used_credits(token)
            delta = _credit_delta_from_total(account_id, before, after)
            if delta is None:
                _update_usage_record(
                    request_id,
                    credits_source=(
                        "pending" if usage_turn_id else "unavailable"
                    ),
                    credits_resolution=(
                        "after_snapshot_pending"
                        if usage_turn_id
                        else "after_snapshot_unavailable"
                    ),
                )
                return
            credits, baseline = delta
            _update_usage_record(
                request_id,
                credits_consumed=credits,
                credits_source="snapshot_delta",
                credits_before=baseline,
                credits_after=_credit_round(after),
            )
    finally:
        if request_id:
            _USAGE_PENDING_TASK_IDS.discard(request_id)
        if release_snapshot:
            last_owner = _end_credit_snapshot(account_id)
            # A shared before snapshot must survive until the last request in
            # this account batch has finished its enrichment.
            if last_owner:
                await _cancel_usage_task(before_task, _USAGE_SNAPSHOT_TASKS)


class _UsageTracker:
    def __init__(
        self,
        model: str,
        endpoint: str,
        stream: bool,
        options: Optional[Mapping[str, Any]] = None,
        messages: Optional[list[dict[str, Any]]] = None,
    ):
        self.request_id = "req-" + uuid_mod.uuid4().hex
        options = options if isinstance(options, dict) else dict(options or {})
        if not isinstance(options.get("_upstream_trace"), dict):
            options["_upstream_trace"] = {}
        self.options = options
        self.messages = messages or []
        self.account_id = str(options.get("_account_id") or "")
        self.billing_id = str(options.get("_billing_id") or "")
        self.token = str(options.get("_auth_token") or "")
        if self.account_id and not self.token:
            # An explicitly bound account owns the lookup.  Never fill its
            # token from the mutable global auth state.
            record = auth.get_account_record(self.account_id)
            self.token = str(record.get("token") or "")
        token_identity = _account_id_from_token(self.token)
        if token_identity:
            # The JWT is the credential that the upstream actually bills. It
            # is authoritative over a stale UI-selected account id.
            if self.account_id and self.account_id != token_identity:
                logger.warning(
                    "usage account corrected from bound id=%s to token id=%s",
                    self.account_id,
                    token_identity,
                )
            self.account_id = token_identity
        if self.billing_id:
            # billing_id overrides account_id for usage records so the
            # deducted credits are attributed to the token owner.
            self.account_id = self.billing_id
        if not self.account_id or not self.token:
            fallback_account_id, fallback_token = _request_account_identity()
            self.account_id = self.account_id or fallback_account_id
            self.token = self.token or fallback_token
        self.model = model
        self.endpoint = endpoint
        self.stream = bool(stream)
        self.started = time.perf_counter()
        self.usage = _usage_values({})
        self.usage_turn_id = ""
        self.saw_usage = False
        # Request metadata is captured before dispatch; the trace and tool
        # return state are updated later as the selected upstream responds.
        self.reasoning_effort = _requested_reasoning_effort(options)
        (
            self.context_mode_requested,
            self.context_mode_actual,
            self.context_window_tokens,
            self.max_mode_applied,
        ) = _context_mode_metadata(options)
        # Keep request capability and model behaviour separate.  A non-empty
        # ``tools`` catalog means the caller offered tools; it does not mean
        # that this response actually returned a tool call.
        self.tools_requested = _tools_requested_from_options(options)
        self.thinking = _normalize_usage_thinking(options.get("thinking"))
        self.tool_choice = options.get("tool_choice")
        self.tool_used: bool | None = None
        self.tool_calls_returned: bool | None = None
        self.queue_position: int | None = None
        self.auto_continue_rounds: int | None = None
        self.status = "in_progress"
        self._finished = False
        self._credit_snapshot_started = bool(
            self.account_id and self.account_id != "default" and self.token
        )
        self._credit_safe = _begin_credit_snapshot(self.account_id, self.token)
        self._before_task = (
            _shared_credit_before_task(self.account_id, self.token)
            if self._credit_safe
            else None
        )
        # Concurrent requests join the first request's before snapshot.  The
        # old unsafe request path had no before value at all and consequently
        # produced a permanently missing credit row.
        if self._before_task is None and self._credit_snapshot_started:
            self._before_task = _shared_credit_before_task(
                self.account_id, self.token
            )

    def update(self, usage: Any) -> None:
        self.saw_usage = True
        values = _usage_values(usage)
        # Upstream streams may send cumulative usage more than once. Keep the
        # latest complete token snapshot instead of creating duplicate records.
        # A later token-only frame must not erase explicit credit evidence
        # reported by an earlier frame.
        if values["total_tokens"] >= self.usage["total_tokens"]:
            explicit_credits = self.usage.get("credits_consumed")
            self.usage.update(values)
            if (
                values.get("credits_consumed") is None
                and explicit_credits is not None
            ):
                self.usage["credits_consumed"] = explicit_credits
        elif values.get("credits_consumed") is not None:
            self.usage["credits_consumed"] = values["credits_consumed"]

    def mark_tool_calls(self, value: Any) -> None:
        """Record whether the model actually returned a structured tool call."""

        if value is None:
            return
        has_calls = (
            _tool_call_count(value) > 0
            if isinstance(value, Mapping)
            else bool(value)
        )
        if has_calls:
            self.tool_calls_returned = True

    def bind_usage_turn(self, usage_turn_id: Any, *, replace: bool = False) -> None:
        value = str(usage_turn_id or "").strip()
        if value and (replace or not self.usage_turn_id):
            self.usage_turn_id = value

    def bind_queue_position(self, position: Any) -> None:
        """Record the deepest queue position observed for this request."""

        if isinstance(position, bool) or not isinstance(position, (int, float)):
            return
        value = int(position)
        if self.queue_position is None or value > self.queue_position:
            self.queue_position = value

    def bind_auto_continue(self, rounds: Any) -> None:
        """Record how many automatic continuation rounds completed the answer."""

        if isinstance(rounds, bool) or not isinstance(rounds, (int, float)):
            return
        value = int(rounds)
        if value > 0:
            self.auto_continue_rounds = value

    async def rebind(self, options: Optional[Mapping[str, Any]] = None) -> None:
        """Move billing/credit tracking to the account used by a retry.

        Remote account rotation can happen after the tracker has started its
        before-credit snapshot.  Cancel that snapshot and restart it for the
        new JWT so the eventual usage row and credit delta follow the account
        that actually handled the request.
        """
        options = options or {}
        token = str(options.get("_auth_token") or "").strip()
        account_id = str(options.get("_account_id") or "").strip()
        billing_id = str(options.get("_billing_id") or "").strip()
        token_identity = _account_id_from_token(token)
        if token_identity:
            billing_id = token_identity
            account_id = token_identity
        elif billing_id:
            account_id = billing_id
        if not account_id and token:
            account_id = str(auth.get_active_account_id() or "default")
        if not token and account_id:
            token = str((auth.get_account_record(account_id) or {}).get("token") or "").strip()
        if not account_id and not token:
            return
        if account_id == self.account_id and token == self.token:
            return

        old_account = self.account_id
        old_before_task = self._before_task
        self._before_task = None
        if self._credit_snapshot_started:
            last_owner = _end_credit_snapshot(old_account)
            if last_owner:
                await _cancel_usage_task(old_before_task, _USAGE_SNAPSHOT_TASKS)

        self.account_id = account_id or self.account_id
        self.billing_id = billing_id or self.account_id
        self.token = token or self.token
        self.usage_turn_id = ""
        self._credit_snapshot_started = bool(
            self.account_id and self.account_id != "default" and self.token
        )
        self._credit_safe = _begin_credit_snapshot(self.account_id, self.token)
        if self._credit_snapshot_started:
            self._before_task = _shared_credit_before_task(
                self.account_id, self.token
            )

    async def finish(self, status: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        final_status = status or self.status or "completed"
        values = self.usage
        trace = self.options.get("_upstream_trace")
        trace = trace if isinstance(trace, Mapping) else {}
        requested_effort = _requested_reasoning_effort(self.options)
        if requested_effort:
            self.reasoning_effort = requested_effort
        applied_effort = (
            trace.get("reasoning_effort_applied")
            or self.options.get("_reasoning_effort_applied")
        )
        product = _normalize_usage_product(
            trace.get("product")
            or self.options.get("product")
            or self.options.get("provider_product")
            or (
                "traework"
                if self.options.get("_traework_custom_model")
                or str(self.options.get("_trae_mode") or "").lower() == "work"
                else None
            )
        )
        requested_endpoint = (
            trace.get("requested_endpoint") or trace.get("requested_mode")
        )
        actual_endpoint = trace.get("actual_endpoint") or trace.get("actual_mode")
        requested_mode = trace.get("requested_mode")
        actual_mode = trace.get("actual_mode")
        (
            context_requested,
            context_actual,
            context_window,
            max_applied,
        ) = _context_mode_metadata(self.options)
        self.context_mode_requested = context_requested or self.context_mode_requested
        self.context_mode_actual = context_actual or self.context_mode_actual
        self.context_window_tokens = context_window or self.context_window_tokens
        self.max_mode_applied = (
            max_applied
            if max_applied is not None
            else self.max_mode_applied
        )
        if self.tool_calls_returned is None and final_status == "completed":
            # A completed turn with an offered tool catalog and no structured
            # call is an explicit negative result.  Failed/cancelled turns
            # remain unknown because no terminal model decision was observed.
            self.tool_calls_returned = False
        if final_status == "completed":
            self.tool_used = self.tool_calls_returned is True
        else:
            self.tool_used = self.tool_calls_returned
        context_mode = self.context_mode_actual
        explicit_credits = values.get("credits_consumed")
        # A failed/cancelled request with no usage frame has no evidence of a
        # billable upstream turn.  Do not render it as "settling" forever.
        no_billable_activity = (
            final_status in {"error", "cancelled"}
            and not self.saw_usage
            and not self.usage_turn_id
        )
        credits_source = (
            "upstream"
            if explicit_credits is not None
            else "unavailable"
            if no_billable_activity
            else "pending"
            if self.usage_turn_id or self._credit_snapshot_started
            else "unknown"
        )
        _record_usage(
            self.account_id,
            self.model,
            values["prompt_tokens"],
            values["completion_tokens"],
            credits_consumed=explicit_credits,
            credits_source=credits_source,
            usage_turn_id=self.usage_turn_id or None,
            request_id=self.request_id,
            endpoint=self.endpoint,
            product=product,
            requested_endpoint=requested_endpoint,
            actual_endpoint=actual_endpoint,
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            stream=self.stream,
            status=final_status,
            duration_ms=round((time.perf_counter() - self.started) * 1000, 1),
            tokens_source="upstream" if self.saw_usage else "unknown",
            cached_tokens=values.get("cached_tokens", 0),
            reasoning_effort=self.reasoning_effort,
            requested_reasoning_effort=self.reasoning_effort,
            reasoning_effort_applied=(
                str(applied_effort).strip()
                if applied_effort not in (None, "")
                else None
            ),
            thinking=self.thinking,
            tool_choice=self.tool_choice,
            context_mode_requested=self.context_mode_requested,
            context_mode=context_mode,
            context_mode_actual=self.context_mode_actual,
            context_window_tokens=self.context_window_tokens,
            max_mode_applied=self.max_mode_applied,
            max_mode_requested=self.context_mode_requested == "max_1m",
            max_mode_enabled=self.max_mode_applied,
            tools_requested=self.tools_requested,
            tool_used=self.tool_used,
            tool_calls_returned=self.tool_calls_returned,
            tool_called=self.tool_used,
            queue_position=self.queue_position,
            auto_continue_rounds=self.auto_continue_rounds,
        )
        if explicit_credits is not None:
            try:
                last_owner = (
                    _end_credit_snapshot(self.account_id)
                    if self._credit_snapshot_started
                    else False
                )
                if last_owner:
                    await _cancel_usage_task(
                        self._before_task, _USAGE_SNAPSHOT_TASKS
                    )
            finally:
                self._before_task = None
        elif not no_billable_activity and (
            self.usage_turn_id or self._credit_snapshot_started
        ):
            _spawn_usage_task(
                _enrich_usage_credits(
                    self.request_id,
                    self.account_id,
                    self.token,
                    self._before_task,
                    usage_turn_id=self.usage_turn_id,
                    credit_safe=self._credit_safe,
                    release_snapshot=self._credit_snapshot_started,
                    allow_shared_snapshot=True,
                )
            )
        else:
            try:
                last_owner = (
                    _end_credit_snapshot(self.account_id)
                    if self._credit_snapshot_started
                    else False
                )
                if last_owner:
                    await _cancel_usage_task(
                        self._before_task, _USAGE_SNAPSHOT_TASKS
                    )
            finally:
                self._before_task = None

    async def begin(self) -> None:
        if self._credit_snapshot_started and self._before_task is None:
            self._before_task = _shared_credit_before_task(
                self.account_id, self.token
            )
        if self._before_task is not None:
            # Let the before-snapshot request start without delaying the first
            # model frame on the result of a separate billing endpoint.
            await asyncio.sleep(0)


def _track_usage_from_result(result: dict, model: str) -> None:
    """Pass non-stream usage to the request tracker, or keep legacy fallback."""
    usage = result.get("usage") or {}
    tracker = _USAGE_TRACKER.get()
    if tracker is not None:
        mark_tool_calls = getattr(tracker, "mark_tool_calls", None)
        if callable(mark_tool_calls):
            mark_tool_calls(result)
        tracker.update(usage)
        return
    values = _usage_values(usage)
    account_id, _ = _request_account_identity()
    _record_usage(
        account_id,
        model,
        values["prompt_tokens"],
        values["completion_tokens"],
        credits_consumed=values.get("credits_consumed"),
        credits_source="upstream" if values.get("credits_consumed") is not None else "unknown",
        cached_tokens=values.get("cached_tokens", 0),
    )


def _track_usage_from_exception(exc: BaseException, model: str) -> None:
    usage = getattr(exc, "usage", None)
    if usage is not None and not getattr(exc, "_relay_usage_recorded", False):
        _track_usage_from_result({"usage": usage}, model)
        exc._relay_usage_recorded = True


def _track_usage_from_chunk(chunk: str, model: str) -> None:
    """Pass OpenAI and TraeWork SSE usage to the request tracker.

    Ordinary relay streams put usage on a ``data:`` JSON frame. The TraeWork
    custom-model adapter wraps the same payload as ``event: token_usage``.
    Parse complete SSE blocks instead of requiring a chunk to start with
    ``data:``; an ASGI chunk may contain several events.
    """
    if isinstance(chunk, (bytes, bytearray)):
        text = bytes(chunk).decode("utf-8", errors="replace")
    else:
        text = str(chunk or "")
    if not text:
        return

    tracker = _USAGE_TRACKER.get()
    normalized = text.replace("\r\n", "\n")
    for block in normalized.split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("event:"):
                event_name = stripped.split(":", 1)[1].strip().lower()
            elif stripped.lower().startswith("data:"):
                data_lines.append(stripped.split(":", 1)[1].lstrip())
        if not data_lines:
            continue
        payload_text = "\n".join(data_lines).strip()
        if payload_text == "[DONE]":
            continue
        try:
            data = json.loads(payload_text)
        except Exception:
            continue
        if not isinstance(data, Mapping):
            continue
        if tracker is not None:
            mark_tool_calls = getattr(tracker, "mark_tool_calls", None)
            if callable(mark_tool_calls):
                mark_tool_calls(data)
        usage = data.get("usage")
        if not usage and event_name == "token_usage":
            usage = data
        if not isinstance(usage, Mapping):
            continue
        if tracker is not None:
            update = getattr(tracker, "update", None)
            if callable(update):
                update(usage)
            # Test/dummy trackers may only expose ``saw_usage``; they already
            # own their terminal bookkeeping and should not create a second
            # fallback usage row here.
            continue
        values = _usage_values(usage)
        account_id, _ = _request_account_identity()
        _record_usage(
            account_id,
            model,
            values["prompt_tokens"],
            values["completion_tokens"],
            credits_consumed=values.get("credits_consumed"),
            credits_source="upstream" if values.get("credits_consumed") is not None else "unknown",
            cached_tokens=values.get("cached_tokens", 0),
        )


def _bind_usage_turn(usage_turn_id: Any, *, replace: bool = False) -> None:
    tracker = _USAGE_TRACKER.get()
    if tracker is not None:
        tracker.bind_usage_turn(usage_turn_id, replace=replace)


def _bind_usage_turn_from_metadata(metadata: Any) -> None:
    if isinstance(metadata, Mapping):
        _bind_usage_turn(metadata.get("usage_turn_id"))
        position = metadata.get("queue_position")
        tracker = _USAGE_TRACKER.get()
        bind_position = getattr(tracker, "bind_queue_position", None)
        if callable(bind_position) and isinstance(position, (int, float)):
            bind_position(position)


def _remote_only_models() -> set[str]:
    """Return explicit overrides for remote manual selection."""
    raw = os.environ.get("TRAE_REMOTE_ONLY_MODELS", "")
    configured: set[str] = set()
    for item in raw.split(","):
        value = item.strip().lower()
        if value.startswith("trae/"):
            value = value[5:]
        if value:
            configured.add(value)
    return configured


def _raw_history_limit_int(
    options: Mapping[str, Any],
    names: tuple[str, ...],
    env_name: str,
    default: int,
) -> int:
    value = None
    for name in names:
        value = options.get(name)
        if value is not None:
            break
    if value is None:
        value = os.environ.get(env_name, str(default))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


class RemoteQueryTooLarge(ValueError):
    """The compacted query still cannot fit Trae Remote's hard input cap."""

    def __init__(self, query_chars: int, limit: int):
        self.query_chars = max(0, int(query_chars))
        self.limit = max(0, int(limit))
        super().__init__(
            "Trae remote query exceeds the configured input limit "
            f"({self.query_chars} > {self.limit} characters after compaction). "
            "Attached images count toward this limit; reduce the input size "
            "or use the IDE/Solo endpoint for larger image payloads."
        )


def _bounded_remote_query(
    messages: list[Mapping[str, Any]],
    options: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Truncate oldest non-system messages so the flattened query fits upstream.

    Trae's remote session silently ends the event stream (no text, no ``done``)
    when initial_message.query exceeds roughly 500k chars.  Keep the system
    prompt and the newest turns intact and drop the oldest conversation from
    the front until the complete wire query, including images, fits the cap.
    """

    raw_limit = _raw_history_limit_int(
        options,
        ("trae_remote_query_max_chars", "traeRemoteQueryMaxChars"),
        "TRAE_REMOTE_QUERY_MAX_CHARS",
        480_000,
    )
    if raw_limit <= 0:
        return [dict(m) for m in messages], 0
    original = [dict(m) for m in messages]
    query = trae_client.flatten_query_parts(original)
    if len(query) <= raw_limit:
        return original, 0
    non_system = [
        index
        for index, message in enumerate(original)
        if str(message.get("role") or "user") not in {"system", "developer"}
    ]
    boundaries = [0]
    offset = 0
    while offset + 1 < len(non_system):
        head = non_system[offset]
        message = original[head]
        paired = (
            str(message.get("role") or "") == "assistant"
            and bool(message.get("tool_calls") or message.get("function_call"))
            and head + 1 == non_system[offset + 1]
            and str(original[head + 1].get("role") or "") == "tool"
        )
        offset += 2 if paired else 1
        boundaries.append(offset)

    non_system_set = set(non_system)

    def suffix(removed_count: int) -> list[dict[str, Any]]:
        first = non_system[removed_count] if removed_count < len(non_system) else len(original)
        return [
            message
            for index, message in enumerate(original)
            if index >= first or index not in non_system_set
        ]

    # Ordinary histories only lose serialized text/media when a prefix is
    # removed. Find the same first fitting suffix without re-encoding a large
    # image once per history message on the event loop. Renderer tool results
    # can inherit names from earlier calls, so keep their exact sequential
    # behavior instead of assuming their serialized size is monotonic.
    inherited_tool_names = any(
        raw_client._renderer_tool_result_blocks(message.get("content"))
        for message in original
    )
    removed = 0
    bounded = original
    if inherited_tool_names:
        for removed in boundaries[1:]:
            bounded = suffix(removed)
            query = trae_client.flatten_query_parts(bounded)
            if len(query) <= raw_limit:
                return bounded, removed
    elif len(boundaries) > 1:
        removed = boundaries[-1]
        bounded = suffix(removed)
        query = trae_client.flatten_query_parts(bounded)
        if len(query) <= raw_limit:
            low, high = 1, len(boundaries) - 1
            while low < high:
                middle = (low + high) // 2
                candidate = suffix(boundaries[middle])
                candidate_query = trae_client.flatten_query_parts(candidate)
                if len(candidate_query) <= raw_limit:
                    high = middle
                    bounded, query = candidate, candidate_query
                else:
                    low = middle + 1
            return bounded, boundaries[low]

    # Media cannot be shortened by the text-fitting pass. Reject its fixed
    # lower bound before repeatedly normalizing an attachment that cannot fit.
    query_parts = json.loads(query)
    media = [
        part
        for part in query_parts
        if isinstance(part, Mapping) and part.get("type") == "image_url"
    ]
    if media and len(json.dumps(media, ensure_ascii=False)) > raw_limit:
        raise RemoteQueryTooLarge(len(query), raw_limit)

    # The loop above can only remove complete turns.  A request containing one
    # enormous user/tool-result message has no removable non-system turn, so
    # the old implementation returned the oversized query unchanged.  Trae
    # accepts the create-session request but then closes the events stream
    # before the first model event; callers consequently rotate through every
    # account and end with a burst of 502 responses.  Clamp message content as
    # a final step so a single large turn cannot bypass the upstream limit.
    bounded = _fit_remote_query_contents(bounded, raw_limit)
    final_query_chars = len(trae_client.flatten_query_parts(bounded))
    if final_query_chars > raw_limit:
        # Never send a request that the provider is known to accept only
        # partially. A non-text tool payload or an intentionally tiny custom
        # limit can leave fixed JSON/tool metadata larger than the budget even
        # after text truncation. Failing before ``create_session`` keeps this
        # deterministic client error from being replayed on every account.
        raise RemoteQueryTooLarge(final_query_chars, raw_limit)
    return bounded, removed


def _remote_fallback_allowed_for_request(
    messages: Optional[list[Mapping[str, Any]]],
    options: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether a native failure may safely retry the request on Remote.

    Remote can compact ordinary conversation history, but an image attachment
    is intentionally kept intact.  When that attachment alone leaves the
    flattened query above Remote's hard input cap, retrying the same payload
    would only replace a useful IDE error with a deterministic 413 (and may
    replay it across polling accounts).  Text, tool requests, and images that
    fit after compaction retain the normal fallback behavior.
    """

    if not images.messages_have_images(messages):
        return True
    try:
        _bounded_remote_query(messages or [], options or {})
    except RemoteQueryTooLarge:
        return False
    return True


_REMOTE_QUERY_TRUNCATION_MARKER = (
    "\n\n[relay: older context omitted to fit the Trae remote query limit]\n\n"
)


def _truncate_remote_text(text: str, limit: int) -> str:
    """Keep the useful head and tail of one message within ``limit`` chars."""

    value = str(text or "")
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= len(_REMOTE_QUERY_TRUNCATION_MARKER):
        return value[:limit]
    available = limit - len(_REMOTE_QUERY_TRUNCATION_MARKER)
    head = max(1, int(available * 0.65))
    tail = max(0, available - head)
    if not tail:
        return value[:head] + _REMOTE_QUERY_TRUNCATION_MARKER
    return (
        value[:head]
        + _REMOTE_QUERY_TRUNCATION_MARKER
        + value[-tail:]
    )


def _remote_message_text(message: Mapping[str, Any]) -> str:
    """Return the trimmable text, excluding intact image attachments."""

    content = message.get("content")
    if content in (None, "", []) and not message.get("tool_calls"):
        for key in ("parts", "text", "prompt", "message", "input"):
            candidate = message.get(key)
            if candidate not in (None, "", [], {}):
                content = candidate
                break
    return raw_client._content_to_text(raw_client._without_native_images(content))


def _truncate_remote_message(
    message: Mapping[str, Any],
    content_limit: int,
) -> dict[str, Any]:
    """Copy one message while bounding its text contribution.

    Image attachments and tool call metadata remain intact. Only text may be
    shortened; an attachment that cannot fit must produce an explicit error.
    """

    updated = dict(message)
    text = _remote_message_text(message)
    truncated = _truncate_remote_text(text, content_limit)
    content = message.get("content")
    attachments = (
        [
            part
            for part in content
            if isinstance(part, Mapping) and images.is_image_part(part)
        ]
        if isinstance(content, list)
        else []
    )
    updated["content"] = (
        [{"type": "text", "text": truncated}, *attachments]
        if attachments
        else truncated
    )
    for key in ("parts", "text", "prompt", "message", "input"):
        if key in updated and key != "content":
            updated.pop(key, None)
    return updated


def _fit_remote_query_contents(
    messages: list[dict[str, Any]],
    raw_limit: int,
) -> list[dict[str, Any]]:
    """Trim oversized individual messages until the flattened query fits."""

    if raw_limit <= 0:
        return messages
    bounded = [dict(message) for message in messages]
    while len(trae_client.flatten_query_parts(bounded)) > raw_limit:
        before_query_chars = len(trae_client.flatten_query_parts(bounded))
        candidates = sorted(
            (
                (index, len(_remote_message_text(message)))
                for index, message in enumerate(bounded)
                if _remote_message_text(message)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not candidates:
            break
        index, text_length = candidates[0]
        current = bounded[index]
        low, high = 0, text_length
        best: Optional[dict[str, Any]] = None
        # Find the largest prefix/suffix slice that makes the whole query fit.
        while low <= high:
            middle = (low + high) // 2
            candidate = _truncate_remote_message(current, middle)
            probe = list(bounded)
            probe[index] = candidate
            if len(trae_client.flatten_query_parts(probe)) <= raw_limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            # Keep a small marker rather than returning an invalid oversized
            # request when fixed JSON framing consumes the entire budget.
            best = _truncate_remote_message(
                current,
                max(1, min(text_length, len(_REMOTE_QUERY_TRUNCATION_MARKER))),
            )
        bounded[index] = best
        if len(trae_client.flatten_query_parts(bounded)) >= before_query_chars:
            # Avoid spinning forever when a non-text tool payload alone is
            # larger than the configured budget.  The normal cap leaves ample
            # room for tool metadata; this is only a defensive escape hatch
            # for malformed or extremely small custom limits.
            break
    return bounded


_CONTINUATION_ONLY_RE = re.compile(
    r"^(?:继续(?:执行|完成|处理|下载|安装|操作|进行|做)?|接着(?:做|执行|处理)?|"
    r"往下继续|下一步|j继续|continue(?:\s+(?:it|this|working))?|go\s+on|"
    r"proceed|keep\s+going|resume)[\s。.!！?？]*$",
    re.IGNORECASE,
)


def _remote_client_tool_task_anchor(
    messages: list[Mapping[str, Any]], options: Mapping[str, Any]
) -> dict[str, str] | None:
    """Keep the active client-tool task when the newest turn only says continue.

    Very large Codex sessions can contain more than a thousand messages. The
    remote query cap must discard old turns, but a short continuation such as
    ``继续`` has no task semantics by itself. Preserve the nearest preceding
    user request as a system constraint so URLs and destination paths survive
    compaction and the model still emits a caller-owned tool call.
    """
    if not _tool_protocol_requested(options, list(messages)):
        return None
    user_messages: list[tuple[int, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "user") != "user":
            continue
        text = raw_client._content_to_text(message.get("content")).strip()
        if text:
            user_messages.append((index, text))
    if len(user_messages) < 2:
        return None
    latest_index, latest_text = user_messages[-1]
    if not _CONTINUATION_ONLY_RE.fullmatch(latest_text):
        return None
    for index, text in reversed(user_messages[:-1]):
        if index >= latest_index or _CONTINUATION_ONLY_RE.fullmatch(text):
            continue
        max_chars = 16_000
        if len(text) > max_chars:
            text = text[:8_000] + "\n...[task middle omitted]...\n" + text[-8_000:]
        return {
            "role": "system",
            "content": (
                "Active caller task retained during history compaction. The "
                "latest user message is only a continuation request. Continue "
                "the task below. For any caller-side download or file change, "
                "emit the matching client tool call and wait for its result; "
                "do not claim success from a remote/internal tool.\n\n"
                + text
            ),
        }
    return None


def _requires_remote_model(model: str) -> bool:
    """Return whether an explicit operator override forces remote routing."""

    configured = _remote_only_models()
    if not configured:
        return False
    if "*" in configured:
        return True
    value = str(model or "").strip()
    if value.lower().startswith("trae/"):
        value = value[5:]
    candidates = {value.lower()} if value else set()
    try:
        mapped = str(trae_client.convert_model_name(value) or "").strip().lower()
    except Exception:
        mapped = ""
    if mapped:
        candidates.add(mapped)
    return bool(candidates & configured)


def _chunk_stream_state(chunk: Any) -> str | None:
    """Return the public stream state proven by one SSE chunk.

    Clients commonly close immediately after the final Chat Completions
    frame (the one carrying ``finish_reason``), before they consume the
    protocol ``[DONE]`` sentinel.  Treating that normal close as a cancelled
    turn makes already-billed streaming requests look unpaid in the console.
    Parse both OpenAI data frames and Responses event names here so usage
    tracking has one protocol-aware terminal check.
    """
    if isinstance(chunk, (bytes, bytearray)):
        try:
            text = bytes(chunk).decode("utf-8", errors="replace")
        except Exception:
            text = ""
    elif isinstance(chunk, str):
        text = chunk
    else:
        text = str(chunk)
    normalized = text.replace("\r\n", "\n")
    # Parse one SSE block at a time so a custom ``event: output`` line stays
    # associated with its data payload. This also handles a proxy coalescing
    # ``token_usage`` and ``done`` events in one chunk.
    for block in normalized.split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("event:"):
                event_name = stripped.split(":", 1)[1].strip().lower()
            elif stripped.lower().startswith("data:"):
                data_lines.append(stripped.split(":", 1)[1].lstrip())
        if event_name in {"response.failed", "error"}:
            return "error"
        if event_name in {"response.completed", "response.incomplete"}:
            return "completed"
        if not data_lines:
            if event_name == "done":
                return "completed"
            continue
        payload = "\n".join(data_lines).strip()
        if payload == "[DONE]":
            return "completed"
        try:
            value = json.loads(payload)
        except Exception:
            continue
        if not isinstance(value, Mapping):
            if event_name == "done":
                return "completed"
            continue
        value_type = str(value.get("type") or "").strip().lower()
        if value_type in {"response.failed", "error"}:
            return "error"
        if value_type in {"response.completed", "response.incomplete"}:
            return "completed"
        if isinstance(value.get("error"), Mapping):
            return "error"
        if event_name == "done":
            status = str(value.get("status") or "").strip().lower()
            if status in {"error", "failed", "cancelled", "canceled"}:
                return "error"
            return "completed"
        # TraeWork's output event carries the terminal finish reason in its
        # own payload. A client may close immediately after consuming this
        # frame, before the adapter's later ``event: done`` sentinel arrives.
        if event_name == "output" and value.get("finish_reason") not in (None, ""):
            return "completed"
        # The public translators put finish_reason inside ``choices`` only on
        # the final OpenAI frame. Upstream cumulative snapshots are consumed
        # before translation, so this is safe to use when a client disconnects
        # just before the following [DONE].
        choices = value.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                if choice.get("finish_reason") in (None, ""):
                    continue
                # A terminal translated frame has an empty delta. Do not
                # classify a cumulative frame that carries both text/tool data
                # and a provisional finish_reason as complete.
                delta = choice.get("delta")
                if isinstance(delta, Mapping) and not any(
                    item not in (None, "", [], {}) for item in delta.values()
                ):
                    return "completed"
    return None


def _chunk_marks_terminal(chunk: Any) -> bool:
    """Whether an SSE chunk proves the response reached its end."""
    return _chunk_stream_state(chunk) is not None


async def _tracked_stream(source, tracker: _UsageTracker):
    context_token = _USAGE_TRACKER.set(tracker)
    status = "cancelled"
    saw_terminal = False
    terminal_status: str | None = None
    try:
        await tracker.begin()
        async for chunk in source:
            # TraeWork custom streams expose usage as ``event: token_usage``;
            # feed every frame through the shared parser so streaming and
            # non-stream requests settle the same usage record.
            _track_usage_from_chunk(chunk, getattr(tracker, "model", ""))
            chunk_status = _chunk_stream_state(chunk)
            if chunk_status is not None:
                saw_terminal = True
                # Keep an error terminal state sticky if a cleanup sentinel is
                # emitted after it.
                if chunk_status == "error" or terminal_status is None:
                    terminal_status = chunk_status
            yield chunk
        usage_seen = bool(getattr(tracker, "saw_usage", False))
        status = terminal_status or ("completed" if usage_seen else "cancelled")
    except asyncio.CancelledError:
        # A final usage frame is emitted immediately before [DONE] by the
        # OpenAI translators.  Some clients close after consuming that frame
        # but before reading [DONE]; preserve the billable turn as completed.
        usage_seen = bool(getattr(tracker, "saw_usage", False))
        status = terminal_status or ("completed" if saw_terminal or usage_seen else "cancelled")
        raise
    except GeneratorExit:
        usage_seen = bool(getattr(tracker, "saw_usage", False))
        status = terminal_status or ("completed" if saw_terminal or usage_seen else "cancelled")
        raise
    except Exception:
        status = "error"
        raise
    finally:
        _USAGE_TRACKER.reset(context_token)
        await tracker.finish(status)


async def _tracked_dispatch(
    messages: list[dict],
    model: str,
    options: dict,
    tracker: _UsageTracker,
):
    context_token = _USAGE_TRACKER.set(tracker)
    status = "completed"
    try:
        await tracker.begin()
        response = await _dispatch_chat(messages, model, False, options)
        if getattr(response, "status_code", 200) >= 400:
            status = "error"
        return response
    except Exception:
        status = "error"
        raise
    finally:
        try:
            _USAGE_TRACKER.reset(context_token)
            await tracker.finish(status)
        finally:
            _cancel_unclaimed_account_slot(options)


async def run_cli_chat(messages, model, stream: bool, options: Optional[dict] = None):
    """本地 Trae CLI 子进程上游。"""
    event_iter = cli_client.stream_cli_chat(messages, model, options=options)
    translation_options = _tool_translation_options(options, messages)
    if not stream:
        result = await collect_nonstream_cli(event_iter, model, **translation_options)
        _track_usage_from_result(result, model)
        return JSONResponse(content=result)

    first, rest = await _peek_async(event_iter)
    if first is None:
        async def empty_gen():
            async for chunk in translate_cli_stream(
                _empty_cli_events(), model, FORWARD_USAGE, **translation_options
            ):
                yield chunk
        return StreamingResponse(empty_gen(), media_type="text/event-stream", headers=_sse_headers())
    if first.type == "error":
        raise RuntimeError(first.error or "Trae CLI failed before output")

    async def gen():
        async def chain():
            yield first
            async for item in rest:
                yield item
        async for chunk in translate_cli_stream(
            chain(), model, FORWARD_USAGE, **translation_options
        ):
            _track_usage_from_chunk(chunk, model)
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_sse_headers())


async def _await_session_cleanup(task: asyncio.Task) -> None:
    """Finish owned cleanup before propagating repeated client cancellation."""
    pending_exception = sys.exception()
    cancellation = (
        pending_exception if isinstance(pending_exception, asyncio.CancelledError) else None
    )
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    raise
                cancellation = cancellation or exc
            except Exception:
                break
        try:
            task.result()
        except BaseException:
            if cancellation is None:
                raise
            logger.warning("session cleanup failed while client cancelled", exc_info=True)
    if cancellation is not None:
        raise cancellation


class _SessionStreamIterator:
    """Keep cleanup ownership even when a response is closed before iteration."""

    def __init__(self, source, close):
        self._source = source
        self._close = close

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await anext(self._source)

    async def aclose(self):
        try:
            await self._source.aclose()
        finally:
            await self._close()


def _chat_response_closer(response, upstream_metadata: dict[str, Any]):
    cleanup_task = None

    async def cleanup() -> None:
        try:
            _bind_usage_turn_from_metadata(upstream_metadata)
        finally:
            response.close()

    async def close() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(cleanup())
        await _await_session_cleanup(cleanup_task)

    return close


async def run_web_session(messages, model, stream: bool, options: Optional[dict] = None):
    """OmniRoute 风格网页版 remote 会话，带账号并发槽和空闲回收。"""
    options = dict(options or {})
    token = str(options.get("_auth_token") or "").strip()
    account_id = str(options.get("_account_id") or "").strip()
    record = auth.get_account_record(account_id) if account_id else {}
    if not account_id and not token:
        account_id, record = auth.get_model_account_snapshot()
    if not token and account_id:
        token = str((record or {}).get("token") or "").strip()
    if not token and not account_id:
        if auth.get_accounts_raw():
            raise _ModelAccountUnavailable(
                "No enabled model account available; enable an account in 账号与签到"
            )
        token = str(auth.get_token() or "").strip()
    _ensure_model_account_enabled(account_id, token)
    token_identity = _account_id_from_token(token)
    if token_identity:
        account_id = token_identity
    if not account_id:
        account_id = str(
            auth.get_active_account_id() or auth.get_user_id() or token[:16] or "default"
        )
    if not token:
        token = str((auth.get_account_record(account_id) or {}).get("token") or "").strip()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    translation_options = _tool_translation_options(options, messages)
    bound_options = {**options, "_auth_token": token, "_account_id": account_id}
    provider_specific = bound_options.get("provider_specific")
    if provider_specific is None and "providerSpecificData" in bound_options:
        provider_specific = bound_options.get("providerSpecificData")
    if not isinstance(provider_specific, Mapping):
        provider_specific = (record or {}).get("provider_specific") or (
            record or {}
        ).get("providerSpecificData")
    bound_options["provider_specific"] = (
        dict(provider_specific) if isinstance(provider_specific, Mapping) else {}
    )
    slot_lease_token = await trae_client.acquire_web_slot(
        account_id,
        timeout=float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")),
        reservation_id=str(
            bound_options.get("_account_slot_reservation_id") or ""
        ),
    )
    client = None
    session_id = ""
    message_id = ""
    lease_registered = False
    cleanup_task = None

    async def cleanup_web_session() -> None:
        # Claim the registered lease before any await so the idle reaper and
        # response teardown cannot both close and release the same resources.
        if lease_registered and not trae_client.unregister_web_lease(session_id):
            return
        try:
            if session_id and client is not None:
                await trae_client.stop_web_session(
                    client, session_id, message_id, options=bound_options
                )
        finally:
            try:
                if client is not None:
                    await client.aclose()
            finally:
                trae_client.release_web_slot_owned(account_id, slot_lease_token)

    async def close_web_session() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(cleanup_web_session())
        await _await_session_cleanup(cleanup_task)

    try:
        client = httpx.AsyncClient(timeout=60)
        session_id, message_id = await trae_client.create_web_session(
            client,
            model,
            messages,
            options=bound_options,
        )
        _bind_usage_turn(message_id)
        trae_client.register_web_lease(
            account_id,
            session_id,
            message_id,
            client,
            token=token,
            provider_specific=bound_options.get("provider_specific"),
            slot_lease_token=slot_lease_token,
        )
        lease_registered = True
        event_iter = trae_client.stream_web_events(
            client, session_id, message_id, options=bound_options
        )
        if stream:
            async def gen():
                try:
                    async for chunk in translate_web_events(
                        event_iter, model, FORWARD_USAGE, **translation_options
                    ):
                        _track_usage_from_chunk(chunk, model)
                        yield chunk
                finally:
                    await close_web_session()
            return StreamingResponse(
                _SessionStreamIterator(gen(), close_web_session),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        try:
            result = await collect_nonstream_web(
                event_iter, model, **translation_options
            )
            _track_usage_from_result(result, model)
            return JSONResponse(content=result)
        finally:
            await close_web_session()
    except BaseException:
        await close_web_session()
        raise


def _is_remote_account_capacity_error(error: BaseException) -> bool:
    """Recognize account capacity errors that cannot be repaired through Work."""

    if bool(getattr(error, "account_quota_exhausted", False)):
        return True
    if str(getattr(error, "code", "") or "").strip() in {
        "3004", "4008", "4011", "429"
    }:
        return True
    text = str(error or "").lower()
    return bool(
        re.search(r"(?:\[|\bhttp\s+)(?:3004|4008|4011|429)\b", text)
        or any(
            marker in text
            for marker in (
                "parallel",
                "rate limit",
                "rate-limit",
                "too many requests",
                "quota exceeded",
                "exceeded the quota",
            )
        )
    )


async def run_remote_session(messages, model, stream: bool, options: Optional[dict] = None):
    """9router-style Trae remote session using the current account snapshot.

    Unlike the legacy web helper, this path never reads a mutable global token
    after dispatch.  The account-bound token and provider metadata captured by
    ``_bind_chat_session`` are used for both create and events requests.
    """
    options = dict(options or {})
    account_id = str(options.get("_account_id") or "").strip()
    token = str(options.get("_auth_token") or "").strip()
    model_record: dict = {}
    if not account_id and not token:
        account_id, model_record = auth.get_model_account_snapshot()
    if not token and account_id:
        # A bound account owns its credential.  Do not fall back to the
        # mutable global token, which may belong to a concurrently selected
        # account.
        token = str((auth.get_account_record(account_id) or {}).get("token") or "").strip()
    if not token and not account_id:
        if auth.get_accounts_raw():
            raise _ModelAccountUnavailable(
                "No enabled model account available; enable an account in 账号与签到"
            )
        token = str(auth.get_token() or "").strip()
    _ensure_model_account_enabled(account_id, token)
    token_identity = _account_id_from_token(token)
    if token_identity:
        # The JWT is the identity Trae bills.  It is authoritative if an old
        # account-store key or UI selection is stale.
        account_id = token_identity
    if not account_id:
        account_id = str(auth.get_active_account_id() or "default")
    record = auth.get_account_record(account_id) if account_id else model_record
    if not token:
        token = str(record.get("token") or "").strip()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    remote_options = dict(options)
    # Agent is the default executor for every remote request, including turns
    # that advertise caller-owned tools.  This matches the desktop Agent
    # contract and is required for the Agent-tier model binding/1M profile.
    # Operators that need local-workspace tool execution can explicitly opt
    # those turns into Work with ``TRAE_REMOTE_CALLER_TOOLS_USE_WORK=1``.
    caller_tools_use_work = (
        os.environ.get("TRAE_REMOTE_CALLER_TOOLS_USE_WORK", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if (
        caller_tools_use_work
        and _tool_protocol_requested(options, messages)
        and not remote_options.get("_remote_agent_type")
        and not remote_options.get("remote_agent_type")
    ):
        # Agent sessions own a remote workspace and may consume their internal
        # shell/file tools, then report success without emitting a caller tool
        # call. Work mode has no such ownership ambiguity: caller-advertised
        # tools remain executable only by Codex/the API client.
        remote_options["_remote_agent_type"] = "solo_work_remote"
        remote_options["_session_variant"] = "caller-tools-work"
    # A session lease may carry provider metadata captured from the account
    # store before the JWT identity is normalized. Prefer that bound snapshot;
    # only consult the mutable account lookup as a fallback.
    provider_specific = remote_options.get("provider_specific") or remote_options.get(
        "providerSpecificData"
    )
    if not isinstance(provider_specific, Mapping):
        provider_specific = record.get("provider_specific") or record.get(
            "providerSpecificData"
        )
    remote_options["provider_specific"] = (
        dict(provider_specific) if isinstance(provider_specific, Mapping) else {}
    )
    slot_released = False
    slot_lease_token = ""
    cleanup_task = None

    def release_slot_once() -> None:
        """Release the account slot exactly once across all exit paths."""

        nonlocal slot_released
        if slot_released:
            return
        slot_released = True
        trae_client.release_web_slot_owned(account_id, slot_lease_token)

    client = None
    session_id = ""
    message_id = ""

    async def cleanup_remote_session() -> None:
        """Stop and close one remote attempt without leaking its account slot."""
        try:
            if session_id and client is not None:
                await trae_remote_client.stop_session(
                    client,
                    token,
                    session_id,
                    message_id,
                    options=remote_options,
                )
        finally:
            try:
                if client is not None:
                    await client.aclose()
            finally:
                release_slot_once()

    async def close_remote_session() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(cleanup_remote_session())
        await _await_session_cleanup(cleanup_task)

    translation_options = _tool_translation_options(options, messages)
    explicit_remote_type = str(
        remote_options.get("_remote_agent_type")
        or remote_options.get("remote_agent_type")
        or ""
    ).strip().lower()
    explicit_work = explicit_remote_type in {
        "solo_work_remote",
        "solo_work_lite",
        "work",
    } or str(model or "").strip().lower() in {"work", "auto-work", "solo-work"}
    work_fallback_enabled = (
        os.environ.get("TRAE_REMOTE_WORK_FALLBACK", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    can_work_fallback = work_fallback_enabled and not explicit_work
    can_work_fallback = can_work_fallback and (
        trae_remote_client.remote_agent_type(model, remote_options)
        == "solo_agent_remote"
    )
    work_fallback_used = False

    def work_fallback_options(base: Mapping[str, Any]) -> dict[str, Any]:
        fallback = dict(base)
        fallback["_remote_agent_type"] = "solo_work_remote"
        fallback["_session_variant"] = "work-fallback"
        return fallback

    def mark_work_fallback_trace() -> None:
        """Expose an Agent -> Work fallback as the actual billed product."""

        trace = options.get("_upstream_trace")
        if isinstance(trace, dict):
            trace.update(
                actual_mode="work",
                actual_endpoint="work",
                product="traework",
                fallback_used=True,
                fallback_target="work",
            )

    def agent_create_may_fallback(error: BaseException) -> bool:
        """Only replay transient Agent setup failures through Work.

        A deterministic account/config/auth error cannot be repaired by
        sending the same model to another executor. Replaying those errors
        creates a second request, obscures the real diagnosis, and can charge
        the wrong product.
        """

        text = str(error or "").lower()
        if _is_remote_account_capacity_error(error):
            return False
        if any(
            marker in text
            for marker in (
                "model binding mismatch",
                "model is not available",
                "failed to get app config",
                "app config",
                "record not found",
                "unauthorized",
                "authentication",
                "invalid token",
                "no cloud-ide-jwt",
                "http 400",
                "http 401",
                "http 403",
                "http 404",
                "http 422",
            )
        ):
            return False
        return True

    # Inject caller-owned tool definitions as a system prompt so the
    # remote upstream sees the available tools even though its transport
    # only accepts text messages.
    prepared_messages = trae_client._messages_with_client_runtime(messages, options)
    task_anchor = _remote_client_tool_task_anchor(messages, options)
    if task_anchor is not None:
        prepared_messages = [prepared_messages[0], task_anchor, *prepared_messages[1:]]
        logger.info(
            "remote retained active client-tool task id=%s anchor_chars=%d",
            str(options.get("_relay_request_id") or ""),
            len(task_anchor["content"]),
        )
    compact_options = dict(options)
    # The agent-remote session advertises a 1M-token window, but the upstream
    # silently drops sessions whose flattened query exceeds ~500k chars.
    # Bound history by message count and content size first, then trim the
    # flattened query to the hard cap so large tool sessions still respond.
    compact_options.setdefault(
        "trae_raw_max_messages",
        os.environ.get("TRAE_REMOTE_MAX_MESSAGES", "500"),
    )
    compact_options.setdefault(
        "trae_raw_max_history_chars",
        os.environ.get("TRAE_REMOTE_MAX_HISTORY_CHARS", "480000"),
    )
    compact_options.setdefault(
        "trae_remote_query_max_chars",
        os.environ.get("TRAE_REMOTE_QUERY_MAX_CHARS", "480000"),
    )
    original_message_count = len(prepared_messages)
    prepared_messages, omitted_history = raw_client._compact_raw_history(
        prepared_messages, compact_options
    )
    query_chars_before_bound = len(trae_client.flatten_query_parts(prepared_messages))
    prepared_messages, query_trimmed = _bounded_remote_query(
        prepared_messages, compact_options
    )
    query_chars_after_bound = len(trae_client.flatten_query_parts(prepared_messages))
    if omitted_history:
        logger.info(
            "remote history compacted input_messages=%d output_messages=%d omitted=%d",
            original_message_count,
            len(prepared_messages),
            omitted_history,
        )
    if query_trimmed:
        logger.warning(
            "remote query trimmed for upstream size input_messages=%d output_messages=%d dropped=%d query_chars=%d",
            original_message_count,
            len(prepared_messages),
            query_trimmed,
            query_chars_after_bound,
        )
    elif query_chars_after_bound < query_chars_before_bound:
        logger.warning(
            "remote query content trimmed for upstream size "
            "input_query_chars=%d output_query_chars=%d",
            query_chars_before_bound,
            query_chars_after_bound,
        )
    slot_lease_token = await trae_client.acquire_web_slot(
        account_id,
        timeout=float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")),
        reservation_id=str(
            remote_options.get("_account_slot_reservation_id") or ""
        ),
    )
    try:
        client = httpx.AsyncClient(timeout=None)
        logger.info(
            "remote create start id=%s account=%s model=%s messages=%d "
            "input_last_chars=%d query_chars=%d",
            str(options.get("_relay_request_id") or ""),
            account_id,
            model,
            len(messages),
            len(str(messages[-1].get("content") or "")) if messages else 0,
            query_chars_after_bound,
        )
        try:
            session_id, message_id = await trae_remote_client.create_session(
                client,
                token,
                model,
                prepared_messages,
                options=remote_options,
            )
        except Exception as create_exc:
            if not can_work_fallback or not agent_create_may_fallback(create_exc):
                raise
            logger.warning(
                "remote Agent create failed; falling back to Work id=%s model=%s error=%s",
                str(options.get("_relay_request_id") or ""),
                model,
                create_exc,
            )
            # No session id was returned, so the failed create cannot be
            # stopped. Reuse the acquired account slot with a fresh client.
            failed_client = client
            client = None
            await _await_session_cleanup(asyncio.create_task(failed_client.aclose()))
            client = httpx.AsyncClient(timeout=None)
            remote_options = work_fallback_options(remote_options)
            work_fallback_used = True
            mark_work_fallback_trace()
            session_id, message_id = await trae_remote_client.create_session(
                client,
                token,
                model,
                prepared_messages,
                options=remote_options,
            )
        _bind_usage_turn(message_id)
        logger.info(
            "remote create ok id=%s chat_session=%s message=%s",
            str(options.get("_relay_request_id") or ""),
            session_id,
            message_id,
        )
        event_iter = trae_remote_client.stream_events(
            client,
            token,
            session_id,
            message_id,
            options=remote_options,
        )
        if stream:
            async def gen():
                nonlocal work_fallback_used
                request_id = str(options.get("_relay_request_id") or "")
                started_at = time.monotonic()
                chunk_count = 0
                tool_chunk_count = 0
                saw_done = False
                stream_status = "running"
                try:
                    try:
                        async for chunk in translate_web_events(
                            event_iter,
                            model,
                            FORWARD_USAGE,
                            fail_on_empty=True,
                            **translation_options,
                        ):
                            _track_usage_from_chunk(chunk, model)
                            chunk_count += 1
                            if '"tool_calls"' in chunk:
                                tool_chunk_count += 1
                            if "data: [DONE]" in chunk:
                                saw_done = True
                                stream_status = "completed"
                            yield chunk
                        stream_status = "completed"
                    except (
                        EmptyUpstreamResponse,
                        trae_remote_client.RemoteFirstEventTimeout,
                    ) as exc:
                        exc_usage = getattr(exc, "usage", None)
                        exc_retryable = bool(getattr(exc, "retryable", True))
                        exc_observed_model_event = bool(
                            getattr(exc, "observed_model_event", False)
                        )
                        quota_exhausted = bool(
                            getattr(exc, "account_quota_exhausted", False)
                        )
                        if exc_usage is not None:
                            _track_usage_from_exception(exc, model)
                        polling_retry_enabled = bool(
                            auth.get_polling_status().get("enabled")
                            and not options.get(
                                "_disable_remote_stream_account_retry"
                            )
                        )
                        if (
                            not exc_retryable
                            or exc_observed_model_event
                            or exc_usage is not None
                            or work_fallback_used
                            or not (can_work_fallback or polling_retry_enabled)
                            or (quota_exhausted and not polling_retry_enabled)
                        ):
                            logger.warning(
                                "remote empty response is not safe to retry "
                                "id=%s model=%s observed_model_event=%s "
                                "fallback_used=%s quota_exhausted=%s",
                                request_id,
                                model,
                                exc_observed_model_event,
                                work_fallback_used,
                                quota_exhausted,
                            )
                            raise
                        logger.warning(
                            "remote upstream ended before any model event; "
                            "%s once id=%s model=%s",
                            (
                                "rotating account after quota exhaustion"
                                if quota_exhausted
                                else (
                                    "falling back to Work"
                                    if can_work_fallback and not work_fallback_used
                                    else "retrying"
                                )
                            ),
                            request_id,
                            model,
                        )
                        await close_remote_session()
                        slot_reacquired = False
                        retry_slot_lease_token = ""
                        retry_client = None
                        retry_session_id = ""
                        retry_message_id = ""
                        retry_account_id = account_id

                        async def cleanup_retry_session() -> None:
                            try:
                                if retry_session_id and retry_client is not None:
                                    try:
                                        await trae_remote_client.stop_session(
                                            retry_client,
                                            retry_token,
                                            retry_session_id,
                                            retry_message_id,
                                            options=retry_options,
                                        )
                                    except Exception:
                                        pass
                            finally:
                                try:
                                    if retry_client is not None:
                                        await retry_client.aclose()
                                finally:
                                    if slot_reacquired:
                                        trae_client.release_web_slot_owned(
                                            retry_account_id,
                                            retry_slot_lease_token,
                                        )

                        try:
                            retry_options = dict(remote_options)
                            retry_model = model
                            if (
                                can_work_fallback
                                and not work_fallback_used
                                and not quota_exhausted
                            ):
                                retry_options = work_fallback_options(retry_options)
                                work_fallback_used = True
                                mark_work_fallback_trace()
                            retry_token = token
                            # A 4008 is account-scoped. Never retry it through
                            # the same account's Work executor; rotate to a
                            # different enabled credential instead. Ordinary
                            # empty Agent responses retain the historical
                            # same-account Work fallback before rotation.
                            if polling_retry_enabled and (
                                quota_exhausted
                                or not (can_work_fallback and work_fallback_used)
                            ):
                                _cancel_unclaimed_account_slot(retry_options)
                                next_snapshot = _next_retry_account_snapshot(
                                    {_retry_account_key(options, 0)}, 1
                                )
                                if next_snapshot is not None:
                                    next_account_id, record = next_snapshot
                                    retry_account_id = next_account_id
                                    (
                                        retry_options,
                                        retry_token,
                                        retry_billing_id,
                                        retry_provider,
                                    ) = _retry_options_for_account(
                                        retry_options, next_account_id, record
                                    )
                                    try:
                                        retry_tracker = _USAGE_TRACKER.get()
                                        if retry_tracker is not None:
                                            await retry_tracker.rebind(retry_options)
                                        _rebind_chat_session_account(
                                            str(
                                                retry_options.get("session_id")
                                                or retry_options.get("sessionId")
                                                or ""
                                            ),
                                            next_account_id,
                                            retry_billing_id,
                                            retry_token,
                                            retry_provider,
                                        )
                                    except BaseException:
                                        _cancel_unclaimed_account_slot(retry_options)
                                        raise
                                elif quota_exhausted:
                                    # Do not reacquire the exhausted account
                                    # and replay the turn. The caller receives
                                    # the original quota error instead.
                                    raise
                            retry_slot_lease_token = await trae_client.acquire_web_slot(
                                retry_account_id,
                                timeout=float(
                                    os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")
                                ),
                                reservation_id=str(
                                    retry_options.get(
                                        "_account_slot_reservation_id"
                                    )
                                    or ""
                                ),
                            )
                            slot_reacquired = True
                            retry_client = httpx.AsyncClient(timeout=None)
                            logger.info(
                                "remote empty retry start id=%s model=%s account=%s",
                                request_id,
                                model,
                                str(retry_options.get("_account_id") or ""),
                            )
                            retry_session_id, retry_message_id = (
                                await trae_remote_client.create_session(
                                    retry_client,
                                    retry_token,
                                    retry_model,
                                    prepared_messages,
                                    options=retry_options,
                                )
                            )
                            # The first Agent attempt ended before any model
                            # event. Credit enrichment must follow the Work (or
                            # rotated-account) attempt that actually completed.
                            _bind_usage_turn(retry_message_id, replace=True)
                            retry_event_iter = trae_remote_client.stream_events(
                                retry_client,
                                retry_token,
                                retry_session_id,
                                retry_message_id,
                                options=retry_options,
                            )
                            async for chunk in translate_web_events(
                                retry_event_iter,
                                model,
                                FORWARD_USAGE,
                                fail_on_empty=True,
                                **translation_options,
                            ):
                                _track_usage_from_chunk(chunk, model)
                                chunk_count += 1
                                if '"tool_calls"' in chunk:
                                    tool_chunk_count += 1
                                if "data: [DONE]" in chunk:
                                    saw_done = True
                                    stream_status = "completed"
                                yield chunk
                            stream_status = "completed"
                        finally:
                            try:
                                await _await_session_cleanup(
                                    asyncio.create_task(cleanup_retry_session())
                                )
                            finally:
                                _cancel_unclaimed_account_slot(retry_options)
                except asyncio.CancelledError:
                    stream_status = "client_cancelled"
                    raise
                except GeneratorExit:
                    stream_status = "client_closed" if not saw_done else "completed"
                    raise
                except Exception:
                    stream_status = "error"
                    raise
                finally:
                    logger.info(
                        "public stream closed id=%s status=%s chunks=%d "
                        "tool_chunks=%d done=%s elapsed_ms=%d",
                        request_id,
                        stream_status,
                        chunk_count,
                        tool_chunk_count,
                        saw_done,
                        int((time.monotonic() - started_at) * 1000),
                    )
                    await close_remote_session()

            return StreamingResponse(
                _SessionStreamIterator(gen(), close_remote_session),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        try:
            result = await collect_nonstream_web(
                event_iter,
                model,
                fail_on_empty=True,
                **translation_options,
            )
            _track_usage_from_result(result, model)
            return JSONResponse(content=result)
        except EmptyUpstreamResponse as exc:
            _track_usage_from_exception(exc, model)
            if not (
                can_work_fallback
                and not work_fallback_used
                and exc.retryable
                and not exc.observed_model_event
                and exc.usage is None
                and not _is_remote_account_capacity_error(exc)
            ):
                raise
            logger.warning(
                "remote upstream ended before any model event; falling back to Work "
                "id=%s model=%s",
                str(options.get("_relay_request_id") or ""),
                model,
            )
            work_fallback_used = True
            stopped_session_id = session_id
            stopped_message_id = message_id
            session_id = ""
            message_id = ""
            try:
                if stopped_session_id:
                    await _await_session_cleanup(
                        asyncio.create_task(
                            trae_remote_client.stop_session(
                                client,
                                token,
                                stopped_session_id,
                                stopped_message_id,
                                options=remote_options,
                            )
                        )
                    )
            except Exception:
                pass
            remote_options = work_fallback_options(remote_options)
            mark_work_fallback_trace()
            session_id, message_id = await trae_remote_client.create_session(
                client,
                token,
                model,
                prepared_messages,
                options=remote_options,
            )
            # Replace the empty Agent attempt's message id with the Work turn
            # that produced the response and incurred the final charge.
            _bind_usage_turn(message_id, replace=True)
            fallback_events = trae_remote_client.stream_events(
                client,
                token,
                session_id,
                message_id,
                options=remote_options,
            )
            result = await collect_nonstream_web(
                fallback_events,
                model,
                fail_on_empty=True,
                **translation_options,
            )
            _track_usage_from_result(result, model)
            return JSONResponse(content=result)
        finally:
            await close_remote_session()
    except BaseException:
        await close_remote_session()
        raise


def _effort_note(effort: str, trace: Mapping[str, Any]) -> Optional[str]:
    """Explain why a requested thinking strength was not applied."""

    if not effort or trace.get("reasoning_effort_applied"):
        return None
    mode = str(trace.get("actual_mode") or trace.get("actual_endpoint") or "")
    if mode in {"ide", "solo"}:
        # TraeCode's raw endpoint carries model selection in CustomModel.  The
        # relay keeps the caller's requested level in usage metadata even
        # when a particular native config does not expose a writable effort
        # field, so the diagnostic must name the actual protocol.
        return "当前 TraeCode IDE/SOLO raw 配置未声明可写思考档位"
    if mode == "raw":
        return "raw 端点需要企业 PAT"
    return "该模型未声明 reasoning_effort_config"


async def run_ide_chat(messages, model, stream: bool, options: Optional[dict] = None):
    """Run the IDE/SOLO protocol with a bounded raw -> legacy fallback.

    TraeCode raw v2 is the preferred protocol because it preserves the
    desktop model binding. Personal CN accounts, however, can reject a valid
    catalog model with ``4001 param invalid`` or close an empty stream before
    the first model event. The same account is accepted by the legacy
    ``/api/agent/v3/llm_utils_chat`` adapter. Try raw first, then reopen the
    legacy request only when raw failed before any model output. This keeps
    existing raw replay tests valid while making the public IDE/SOLO presets
    usable on accounts that do not have the raw app-config record.
    """

    options = dict(options or {})
    translation_options = _tool_translation_options(options, messages)
    raw_requested = bool(options.get("_traecode_raw_chat_function"))
    legacy_fallback_allowed = not bool(
        options.get("_disable_upstream_fallback")
        or options.get("_disable_traecode_legacy_fallback")
    )
    upstream_metadata: dict[str, Any] = {}

    def _can_fallback(error: BaseException) -> bool:
        # Buffered native tools may not have reached the wire yet, but an
        # observed model turn is still billable and cannot be safely replayed.
        if isinstance(error, EmptyUpstreamResponse) and (
            not error.retryable
            or error.observed_model_event
            or error.usage is not None
        ):
            return False
        # Account throttling affects both native protocols. Let the outer
        # retry helper rotate credentials instead of replaying this account.
        return not _is_native_rate_limit_error(error)

    def _legacy_options() -> dict:
        legacy = dict(options)
        legacy.pop("_traecode_raw_chat_function", None)
        legacy.pop("_traecode_solo", None)
        legacy["_ide_endpoint"] = str(
            legacy.get("_legacy_ide_endpoint")
            or legacy.get("_ide_endpoint")
            or "/api/agent/v3/llm_utils_chat"
        )
        return legacy

    def _mark_raw_fallback(error: BaseException) -> None:
        trace = options.get("_upstream_trace")
        if not isinstance(trace, dict):
            return
        requested = str(
            trace.get("requested_endpoint")
            or trace.get("requested_mode")
            or options.get("_upstream_mode")
            or "ide"
        )
        reason = f"{requested}: raw -> legacy llm_utils_chat: {str(error)[:800]}"
        previous = str(trace.get("fallback_reason") or "").strip()
        trace.update(
            fallback_used=True,
            fallback_target="legacy-ide",
            fallback_reason="; ".join(
                part for part in (previous, reason) if part
            )[:1200],
            actual_endpoint="/api/agent/v3/llm_utils_chat",
        )

    async def _open(kind: str):
        if kind == "raw":
            code_options = dict(options)
            function = str(
                code_options.get("_traecode_raw_chat_function") or "chat_v3"
            )
            code_options["_traecode_solo"] = function.startswith("solo_")
            return await raw_client.send_traecode_chat_request(
                messages,
                model,
                stream=stream,
                options=code_options,
            )
        return await trae_client.send_chat_request(
            messages,
            model,
            stream,
            options=_legacy_options(),
        )

    if not raw_requested:
        raw_kind = "legacy"
    else:
        raw_kind = "raw"

    if stream:
        current_kind = raw_kind
        current_wrapper = None
        current_close = None
        try:
            try:
                current_wrapper = await _open(current_kind)
            except Exception as exc:
                if (
                    current_kind == "raw"
                    and legacy_fallback_allowed
                    and _can_fallback(exc)
                ):
                    _mark_raw_fallback(exc)
                    current_kind = "legacy"
                    current_wrapper = await _open(current_kind)
                else:
                    raise
            current_close = _chat_response_closer(
                current_wrapper, upstream_metadata
            )

            async def close_current():
                if current_close is not None:
                    await current_close()

        except BaseException:
            if current_close is not None:
                await current_close()
            raise

        async def gen():
            nonlocal current_kind, current_wrapper, current_close
            emitted_model_output = False
            while True:
                try:
                    response = current_wrapper.response
                    async for chunk in translate_ide_stream(
                        response,
                        model,
                        FORWARD_USAGE,
                        fail_on_empty=True,
                        upstream_metadata=upstream_metadata,
                        **translation_options,
                    ):
                        if (
                            _chunk_text_delta(chunk)
                            or _chunk_reasoning_delta(chunk)
                            or _chunk_has_tool_calls(chunk)
                        ):
                            emitted_model_output = True
                        _bind_usage_turn_from_metadata(upstream_metadata)
                        _track_usage_from_chunk(chunk, model)
                        yield chunk
                    await close_current()
                    return
                except (asyncio.CancelledError, GeneratorExit):
                    await close_current()
                    raise
                except Exception as exc:
                    if (
                        current_kind == "raw"
                        and legacy_fallback_allowed
                        and not emitted_model_output
                        and _can_fallback(exc)
                    ):
                        await close_current()
                        _mark_raw_fallback(exc)
                        current_kind = "legacy"
                        current_wrapper = await _open(current_kind)
                        current_close = _chat_response_closer(
                            current_wrapper, upstream_metadata
                        )
                        continue
                    await close_current()
                    raise

        try:
            return StreamingResponse(
                _SessionStreamIterator(gen(), close_current),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        except BaseException:
            await close_current()
            raise

    current_kind = raw_kind
    while True:
        response_wrapper = None
        close_response = None
        try:
            response_wrapper = await _open(current_kind)
            response = response_wrapper.response
            close_response = _chat_response_closer(
                response_wrapper, upstream_metadata
            )
            result = await collect_nonstream_ide(
                response,
                model,
                fail_on_empty=True,
                upstream_metadata=upstream_metadata,
                **translation_options,
            )
            _bind_usage_turn_from_metadata(upstream_metadata)
            _track_usage_from_result(result, model)
            return JSONResponse(content=result)
        except Exception as exc:
            if (
                current_kind == "raw"
                and legacy_fallback_allowed
                and _can_fallback(exc)
            ):
                _mark_raw_fallback(exc)
                current_kind = "legacy"
                continue
            raise
        finally:
            if close_response is not None:
                await close_response()


async def run_traework_native_chat(
    messages, model, stream: bool, options: Optional[dict] = None
):
    """TraeWork native AHA bridge backed by an external Windows helper.

    The helper owns ai_agent.dll and sscronet.dll and returns the native SSE
    event stream. Linux deployments receive a clear 502 instead of attempting
    to load a Windows PE DLL.
    """

    options = dict(options or {})
    translation_options = _tool_translation_options(options, messages)
    upstream_metadata: dict[str, Any] = {}
    requested_slot_account = str(options.get("_account_id") or "").strip()
    slot_account_id = auth.resolve_model_account_id(requested_slot_account)
    if not slot_account_id:
        slot_account_id = requested_slot_account
    if not slot_account_id and options.get("_auth_token"):
        slot_account_id = "token:" + hashlib.sha256(
            str(options.get("_auth_token")).encode("utf-8")
        ).hexdigest()
    slot_acquired = False
    slot_released = False
    slot_lease_token = ""
    slot_release_lock = threading.Lock()

    def release_slot_once() -> None:
        nonlocal slot_released
        with slot_release_lock:
            if slot_released:
                return
            slot_released = True
        if slot_acquired:
            trae_client.release_web_slot_owned(
                slot_account_id, slot_lease_token
            )

    if slot_account_id:
        try:
            slot_timeout = float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60"))
        except (TypeError, ValueError):
            slot_timeout = 60.0
        slot_lease_token = await trae_client.acquire_web_slot(
            slot_account_id,
            timeout=max(0.0, slot_timeout),
            reservation_id=str(
                options.get("_account_slot_reservation_id") or ""
            ),
        )
        slot_acquired = True
    try:
        native_resp = await traework_native_bridge.send_native_chat_request(
            messages,
            model,
            stream=stream,
            options=options,
        )
    except BaseException:
        release_slot_once()
        raise
    close_native_response = _chat_response_closer(
        native_resp, upstream_metadata
    )
    close_task = None

    async def close_response() -> None:
        nonlocal close_task
        if close_task is None:
            async def cleanup() -> None:
                try:
                    await close_native_response()
                finally:
                    release_slot_once()

            close_task = asyncio.create_task(cleanup())
        await _await_session_cleanup(close_task)

    if stream:
        async def gen():
            try:
                async for chunk in translate_ide_stream(
                    native_resp.response,
                    model,
                    FORWARD_USAGE,
                    fail_on_empty=True,
                    upstream_metadata=upstream_metadata,
                    **translation_options,
                ):
                    _bind_usage_turn_from_metadata(upstream_metadata)
                    _track_usage_from_chunk(chunk, model)
                    yield chunk
            finally:
                await close_response()

        try:
            return StreamingResponse(
                _SessionStreamIterator(gen(), close_response),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        except BaseException:
            await close_response()
            raise
    try:
        result = await collect_nonstream_ide(
            native_resp.response,
            model,
            fail_on_empty=True,
            upstream_metadata=upstream_metadata,
            **translation_options,
        )
        _bind_usage_turn_from_metadata(upstream_metadata)
        _track_usage_from_result(result, model)
        return JSONResponse(content=result)
    finally:
        await close_response()


async def run_raw_chat(messages, model, stream: bool, options: Optional[dict] = None):
    """直连 Trae 原生 chat 协议，响应暂复用 IDE SSE 翻译器。"""
    translation_options = _tool_translation_options(options, messages)
    upstream_metadata: dict[str, Any] = {}
    logger.info(
        "raw send start id=%s model=%s messages=%d last_chars=%d",
        str((options or {}).get("_relay_request_id") or ""),
        model,
        len(messages),
        len(str(messages[-1].get("content") or "")) if messages else 0,
    )
    raw_resp = await raw_client.send_raw_chat_request(messages, model, options)
    current = raw_resp
    close_current = _chat_response_closer(current, upstream_metadata)

    async def close_response() -> None:
        await close_current()

    try:
        logger.info(
            "raw send ok id=%s status=%s",
            str((options or {}).get("_relay_request_id") or ""),
            getattr(raw_resp.response, "status_code", 0),
        )
        _capture_chat_session_auth(
            str((options or {}).get("session_id") or ""),
            str(getattr(raw_resp, "auth_token", "") or ""),
        )
    except BaseException:
        await close_response()
        raise
    if stream:
        async def gen():
            nonlocal current, close_current
            request_id = str((options or {}).get("_relay_request_id") or "")
            started_at = time.monotonic()
            chunk_count = 0
            tool_chunk_count = 0
            saw_done = False
            stream_status = "running"
            try:
                retry_options = dict(options or {})
                for attempt in range(2):
                    try:
                        async for chunk in translate_ide_stream(
                            current.response,
                            model,
                            FORWARD_USAGE,
                            fail_on_empty=True,
                            require_terminal=False,
                            upstream_metadata=upstream_metadata,
                            **translation_options,
                        ):
                            _bind_usage_turn_from_metadata(upstream_metadata)
                            chunk_count += 1
                            if '"tool_calls"' in chunk:
                                tool_chunk_count += 1
                            if "data: [DONE]" in chunk:
                                saw_done = True
                                stream_status = "completed"
                            _track_usage_from_chunk(chunk, model)
                            yield chunk
                        stream_status = "completed"
                        return
                    except RepeatedCompletedToolResponse as exc:
                        _track_usage_from_exception(exc, model)
                        logger.warning(
                            "raw upstream repeated an already completed tool call; "
                            "automatic replay is disabled to avoid a second billed turn"
                        )
                        raise
                    except EmptyUpstreamResponse as exc:
                        _bind_usage_turn_from_metadata(upstream_metadata)
                        _track_usage_from_exception(exc, model)
                        if attempt or not exc.retryable:
                            logger.warning(
                                "raw upstream response is not safe to retry "
                                "attempt=%d observed_model_event=%s",
                                attempt + 1,
                                exc.observed_model_event,
                            )
                            raise
                        logger.warning(
                            "raw upstream ended before any model event; retrying once"
                        )
                        retry_options = dict(options or {})
                    finally:
                        await close_response()

                    try:
                        current = await raw_client.send_raw_chat_request(
                            messages, model, retry_options
                        )
                        close_current = _chat_response_closer(current, upstream_metadata)
                        _capture_chat_session_auth(
                            str((options or {}).get("session_id") or ""),
                            str(getattr(current, "auth_token", "") or ""),
                        )
                    except Exception as exc:
                        logger.warning("raw empty-response retry failed: %s", exc)
                        stream_status = "empty_retry_failed"
                        raise
            except asyncio.CancelledError:
                stream_status = "client_cancelled"
                raise
            except GeneratorExit:
                stream_status = "client_closed" if not saw_done else "completed"
                raise
            except Exception:
                stream_status = "error"
                raise
            finally:
                await close_response()
                logger.info(
                    "raw stream closed id=%s status=%s chunks=%d tool_chunks=%d done=%s elapsed_ms=%d",
                    request_id,
                    stream_status,
                    chunk_count,
                    tool_chunk_count,
                    saw_done,
                    int((time.monotonic() - started_at) * 1000),
                )
        try:
            return StreamingResponse(
                _SessionStreamIterator(gen(), close_response),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        except BaseException:
            await close_response()
            raise

    retry_options = dict(options or {})
    for attempt in range(2):
        try:
            result = await collect_nonstream_ide(
                current.response,
                model,
                fail_on_empty=True,
                require_terminal=False,
                upstream_metadata=upstream_metadata,
                **translation_options,
            )
            _bind_usage_turn_from_metadata(upstream_metadata)
            _track_usage_from_result(result, model)
            return JSONResponse(content=result)
        except RepeatedCompletedToolResponse as exc:
            _bind_usage_turn_from_metadata(upstream_metadata)
            _track_usage_from_exception(exc, model)
            logger.warning(
                "raw upstream repeated an already completed tool call; "
                "automatic replay is disabled to avoid a second billed turn"
            )
            raise
        except EmptyUpstreamResponse as exc:
            _bind_usage_turn_from_metadata(upstream_metadata)
            _track_usage_from_exception(exc, model)
            if attempt or not exc.retryable:
                logger.warning(
                    "raw upstream response is not safe to retry "
                    "attempt=%d observed_model_event=%s",
                    attempt + 1,
                    exc.observed_model_event,
                )
                raise
            logger.warning(
                "raw upstream ended before any model event; retrying once"
            )
            retry_options = dict(options or {})
        finally:
            await close_response()
        try:
            current = await raw_client.send_raw_chat_request(
                messages, model, retry_options
            )
            close_current = _chat_response_closer(current, upstream_metadata)
            _capture_chat_session_auth(
                str((options or {}).get("session_id") or ""),
                str(getattr(current, "auth_token", "") or ""),
            )
        except BaseException as exc:
            await close_response()
            logger.warning("raw empty-response retry failed: %s", exc)
            raise

    raise RuntimeError("raw empty-response retry ended unexpectedly")


def _openai_error(status: int, message: str, error_type: str, param: Optional[str] = None) -> JSONResponse:
    body = {"error": {"message": message, "type": error_type}}
    if param:
        body["error"]["param"] = param
    return JSONResponse(body, status_code=status)


def _retry_account_key(options: Optional[Mapping[str, Any]], fallback_index: int) -> str:
    options = options or {}
    token = str(options.get("_auth_token") or "")
    token_identity = _account_id_from_token(token)
    if token_identity:
        # The JWT owner is the account Trae actually bills. Two stale account
        # rows that reference the same JWT must not receive this request twice.
        return token_identity
    if token:
        return "token:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    account_id = str(
        options.get("_account_id")
        or options.get("_billing_id")
        or auth.get_active_account_id()
        or ""
    )
    if account_id:
        return account_id
    return f"attempt-{fallback_index}"


def _polling_retry_limit(options: Optional[Mapping[str, Any]]) -> int:
    """Count usable polling accounts without granting an extra retry."""

    account_ids: set[str] = set()
    anonymous_accounts = 0
    for account in auth.list_accounts():
        if isinstance(account, Mapping):
            if (
                account.get("is_valid") is False
                or account.get("model_enabled") is False
                or account.get("model_eligible") is False
            ):
                continue
            account_id = str(account.get("id") or "")
            if account_id:
                account_ids.add(account_id)
            else:
                anonymous_accounts += 1
        else:
            anonymous_accounts += 1
    bound_account = str((options or {}).get("_account_id") or "")
    if bound_account and auth._record_model_enabled(auth.get_account_record(bound_account)):
        account_ids.add(bound_account)
    return max(1, len(account_ids) + anonymous_accounts)


def _next_retry_account_snapshot(
    attempted_accounts: set[str], max_rotations: int
) -> tuple[str, dict] | None:
    """Atomically reserve an untried account for one retry attempt.

    Credit-priority selection normally keeps returning the richest account
    while it has capacity.  That is correct for new traffic but not for an
    account-scoped retry: once that account failed, selecting it again would
    prevent lower-credit accounts from ever being tried.  Translate every
    attempted billing identity back to all matching store rows, exclude them,
    and let the auth scheduler choose and reserve the best remaining slot.
    """

    if max_rotations <= 0:
        return None
    excluded_account_ids: set[str] = set()
    stored_accounts = auth.get_accounts_raw()
    # ``list_accounts`` is the public account inventory used by the retry
    # budget and by a few integrations that provide credentials dynamically.
    # Use the atomic reservation ledger only when that inventory is backed by
    # the same persisted rows.  If an integration exposes a temporary/mock
    # inventory while the process still has unrelated persisted rows, forcing
    # the ledger path would silently ignore the advertised accounts and make a
    # retry look exhausted.  In the normal deployment the ids are identical
    # (apart from filtered invalid rows), so this remains the atomic path.
    listed_accounts = auth.list_accounts()
    listed_ids = {
        str(item.get("id") or "")
        for item in listed_accounts
        if isinstance(item, Mapping) and str(item.get("id") or "")
    }
    stored_ids = {str(account_id or "") for account_id, _ in stored_accounts}
    inventory_matches_store = listed_ids == stored_ids
    for rotation_index, (account_id, record) in enumerate(stored_accounts):
        safe_record = dict(record) if isinstance(record, Mapping) else {}
        candidate_key = _retry_account_key(
            {
                "_account_id": str(account_id or ""),
                "_auth_token": str(safe_record.get("token") or ""),
            },
            rotation_index,
        )
        if (
            candidate_key in attempted_accounts
            or str(account_id or "") in attempted_accounts
        ):
            excluded_account_ids.add(str(account_id or ""))

    if stored_accounts and inventory_matches_store:
        account_id, record = auth.reserve_model_account(
            require_polling=True,
            excluded_account_ids=excluded_account_ids,
        )
        if not account_id:
            return None
        return str(account_id), dict(record)

    # Legacy/single-token integrations and tests may expose polling accounts
    # through the compatibility getters without persisting them in auth.
    # Retain that path only when the atomic store is genuinely empty.
    for rotation_index in range(max(1, max_rotations)):
        auth.next_polling_account()
        account_id, record = auth.get_active_account_snapshot()
        safe_record = dict(record) if isinstance(record, Mapping) else {}
        if not auth._record_model_enabled(safe_record):
            continue
        candidate_options = {
            "_account_id": str(account_id or ""),
            "_auth_token": str(safe_record.get("token") or ""),
        }
        account_key = _retry_account_key(candidate_options, rotation_index)
        if account_key not in attempted_accounts:
            return str(account_id or ""), safe_record
    return None


def _retry_options_for_account(
    options: Optional[Mapping[str, Any]], account_id: str, record: Mapping[str, Any]
) -> tuple[dict, str, str, Mapping[str, Any]]:
    """Bind a reserved retry slot and its immutable credential snapshot."""

    rebound = dict(options or {})
    for key in (
        "_account_slot_reserved",
        "_account_slot_reservation_account",
        "_account_slot_reservation_id",
    ):
        rebound.pop(key, None)
    token = str(record.get("token") or "")
    billing_id = _account_id_from_token(token) or account_id
    provider_specific = (
        record.get("provider_specific")
        or record.get("providerSpecificData")
        or {}
    )
    rebound["_account_id"] = account_id
    rebound["_billing_id"] = billing_id
    rebound["_auth_token"] = token
    rebound["_auth_user_id"] = billing_id
    if isinstance(provider_specific, Mapping):
        rebound["provider_specific"] = dict(provider_specific)
    reservation_id = str(record.get("_slot_reservation_id") or "")
    if reservation_id:
        rebound["_account_slot_reserved"] = True
        rebound["_account_slot_reservation_account"] = account_id
        rebound["_account_slot_reservation_id"] = reservation_id
    return rebound, token, billing_id, provider_specific


async def _run_web_with_retry(
    messages,
    model,
    stream: bool,
    options: Optional[dict] = None,
    *,
    tracker: Optional[_UsageTracker] = None,
):
    """web 上游 429 并发限制时轮询切换账号重试。"""
    tracker = tracker or _USAGE_TRACKER.get()
    if auth.get_polling_status().get("enabled"):
        attempts = _polling_retry_limit(options)
        attempted_accounts: set[str] = set()
        for attempt_index in range(attempts):
            account_key = _retry_account_key(options, attempt_index)
            if account_key in attempted_accounts:
                break
            attempted_accounts.add(account_key)
            attempt_options = options
            try:
                return await run_web_session(
                    messages, model, stream, attempt_options
                )
            except RuntimeError as e:
                err = str(e)
                if "solo_agent_parallel_limit" in err or "429" in err:
                    if attempt_index + 1 >= attempts:
                        break
                    logger.warning("web 429 parallel limit, rotating account: %s", err)
                    _cancel_unclaimed_account_slot(options)
                    next_snapshot = _next_retry_account_snapshot(
                        attempted_accounts, attempts
                    )
                    if next_snapshot is None:
                        break
                    # Rebind options from the newly rotated account so the
                    # retry uses the correct token and billing identity.
                    account_id, record = next_snapshot
                    (
                        options,
                        token,
                        billing_id,
                        provider_specific,
                    ) = _retry_options_for_account(
                        options, account_id, record
                    )
                    try:
                        if tracker is not None:
                            await tracker.rebind(options)
                        _rebind_chat_session_account(
                            str(
                                options.get("session_id")
                                or options.get("sessionId")
                                or ""
                            ),
                            account_id,
                            billing_id,
                            token,
                            provider_specific,
                        )
                    except BaseException:
                        _cancel_unclaimed_account_slot(options)
                        raise
                    continue
                raise
            finally:
                _cancel_unclaimed_account_slot(attempt_options)
        raise RuntimeError("All web accounts busy: Trae parallel limit reached")
    return await run_web_session(messages, model, stream, options)


async def _run_remote_stream_with_retry(
    messages,
    model,
    options: Optional[dict] = None,
    *,
    tracker: Optional[_UsageTracker] = None,
):
    """Keep a streaming Remote request inside the polling-account retry loop.

    ``run_remote_session(..., stream=True)`` returns before its body iterator
    is consumed.  Consequently the ordinary ``_run_remote_with_retry`` loop
    cannot observe a first-event timeout or an early Remote error: by then it
    has already returned the ``StreamingResponse`` to Starlette.  This wrapper
    owns the body iterator and rotates credentials when an attempt ends before
    model output.  Each nested attempt disables its own account rotation so
    there is one bounded retry loop and no duplicate account/slot ownership.
    """

    options = dict(options or {})
    if not auth.get_polling_status().get("enabled"):
        return await run_remote_session(messages, model, True, options)

    attempts = _polling_retry_limit(options)
    attempted_accounts: set[str] = set()
    current_options = dict(options)

    def _account_key(attempt_index: int) -> str:
        return _retry_account_key(current_options, attempt_index)

    def _runtime_retry_reason(error: BaseException) -> Optional[str]:
        lowered = str(error).lower()
        if (
            "remote model is not available for the bound account" in lowered
            or "model binding mismatch" in lowered
        ):
            return "model unavailable"
        if _is_remote_account_capacity_error(error) or "429" in lowered:
            return "parallel limit"
        return None

    async def rotate_account(reason: str, attempt_index: int) -> bool:
        nonlocal current_options
        _cancel_unclaimed_account_slot(current_options)
        next_snapshot = _next_retry_account_snapshot(
            attempted_accounts, max(1, attempts)
        )
        if next_snapshot is None:
            return False
        account_id, record = next_snapshot
        (
            current_options,
            token,
            billing_id,
            provider_specific,
        ) = _retry_options_for_account(current_options, account_id, record)
        try:
            if tracker is not None:
                await tracker.rebind(current_options)
            _rebind_chat_session_account(
                str(
                    current_options.get("session_id")
                    or current_options.get("sessionId")
                    or ""
                ),
                account_id,
                billing_id,
                token,
                provider_specific,
            )
        except BaseException:
            _cancel_unclaimed_account_slot(current_options)
            raise
        logger.warning(
            "remote stream %s; rotating account attempt=%d/%d account=%s",
            reason,
            attempt_index + 1,
            attempts,
            account_id,
        )
        return True

    async def close_response(response) -> None:
        iterator = getattr(response, "body_iterator", None)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    # Open the first session before returning so deterministic request errors
    # (notably RemoteQueryTooLarge -> HTTP 413) keep their status code instead
    # of being deferred into a 200 streaming wrapper.  Creation failures are
    # handled here too because the response body has not reached Starlette yet.
    initial_response = None
    initial_response_started = False
    initial_response_closed = False
    initial_attempt_index = 0
    for initial_attempt_index in range(max(1, attempts)):
        account_key = _account_key(initial_attempt_index)
        if account_key in attempted_accounts:
            break
        attempted_accounts.add(account_key)
        initial_options = dict(current_options)
        initial_options["_disable_remote_stream_account_retry"] = True
        try:
            initial_response = await run_remote_session(
                messages, model, True, initial_options
            )
            break
        except RemoteQueryTooLarge:
            raise
        except (
            EmptyUpstreamResponse,
            trae_remote_client.RemoteFirstEventTimeout,
        ) as exc:
            if (
                not bool(getattr(exc, "retryable", True))
                or bool(getattr(exc, "observed_model_event", False))
                or initial_attempt_index + 1 >= max(1, attempts)
                or not await rotate_account(
                    "empty upstream response", initial_attempt_index
                )
            ):
                raise
        except RuntimeError as exc:
            reason = _runtime_retry_reason(exc)
            if (
                bool(getattr(exc, "observed_model_event", False))
                or reason is None
                or initial_attempt_index + 1 >= max(1, attempts)
                or not await rotate_account(reason, initial_attempt_index)
            ):
                raise
        finally:
            _cancel_unclaimed_account_slot(initial_options)
    if initial_response is None:
        raise RuntimeError("remote stream retry ended without a response")

    async def gen():
        nonlocal current_options, initial_response_started
        attempt_index = initial_attempt_index
        response = initial_response
        while True:
            account_key = _account_key(attempt_index)
            if account_key not in attempted_accounts:
                attempted_accounts.add(account_key)
            elif attempt_index != initial_attempt_index:
                raise RuntimeError("remote stream account already attempted")
            local_options = dict(current_options)
            # The outer wrapper is the sole owner of account rotation.  The
            # inner session may still perform its same-account Agent -> Work
            # fallback, but must not start a second account loop.
            local_options["_disable_remote_stream_account_retry"] = True
            observed_model_event = False
            try:
                if attempt_index == initial_attempt_index:
                    initial_response_started = True
                if attempt_index != initial_attempt_index:
                    response = await run_remote_session(
                        messages, model, True, local_options
                    )
                iterator = getattr(response, "body_iterator", None)
                if iterator is None:
                    body = getattr(response, "body", b"")
                    body_text = (
                        body.decode("utf-8", errors="replace")
                        if isinstance(body, bytes)
                        else str(body or "")
                    )
                    if not body_text.strip():
                        raise EmptyUpstreamResponse(
                            "Trae remote upstream returned an empty response",
                            retryable=True,
                            observed_model_event=False,
                        )
                    observed_model_event = bool(
                        _chunk_text_delta(body_text)
                        or _chunk_reasoning_delta(body_text)
                        or _chunk_has_tool_calls(body_text)
                        or _chunk_marks_terminal(body_text)
                        or any(
                            isinstance(value.get("usage"), Mapping)
                            for value in _iter_sse_payloads(body_text)
                        )
                    )
                    yield body_text
                    return
                async for chunk in iterator:
                    if (
                        _chunk_text_delta(chunk)
                        or _chunk_reasoning_delta(chunk)
                        or _chunk_has_tool_calls(chunk)
                        or _chunk_marks_terminal(chunk)
                        or any(
                            isinstance(value.get("usage"), Mapping)
                            for value in _iter_sse_payloads(chunk)
                        )
                    ):
                        observed_model_event = True
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except RemoteQueryTooLarge:
                # Request-size errors are deterministic and cannot be repaired
                # by another account.
                raise
            except (
                EmptyUpstreamResponse,
                trae_remote_client.RemoteFirstEventTimeout,
            ) as exc:
                if (
                    observed_model_event
                    or not bool(getattr(exc, "retryable", True))
                    or bool(getattr(exc, "observed_model_event", False))
                ):
                    raise
                if attempt_index + 1 >= max(1, attempts):
                    raise
                if not await rotate_account("empty upstream response", attempt_index):
                    raise
                attempt_index += 1
                continue
            except RuntimeError as exc:
                reason = _runtime_retry_reason(exc)
                if (
                    observed_model_event
                    or bool(getattr(exc, "observed_model_event", False))
                    or reason is None
                ):
                    raise
                if attempt_index + 1 >= max(1, attempts):
                    raise
                if not await rotate_account(
                    reason,
                    attempt_index,
                ):
                    raise
                attempt_index += 1
                continue
            finally:
                # A failed nested stream normally runs its own generator
                # cleanup.  Explicitly closing the wrapper prevents a
                # transport-specific iterator from retaining the slot.
                try:
                    if response is not None:
                        await close_response(response)
                finally:
                    _cancel_unclaimed_account_slot(local_options)

    async def close_unconsumed_initial_response() -> None:
        nonlocal initial_response_closed
        if initial_response_started or initial_response_closed:
            return
        initial_response_closed = True
        await close_response(initial_response)

    return StreamingResponse(
        _SessionStreamIterator(gen(), close_unconsumed_initial_response),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


async def _run_remote_with_retry(
    messages,
    model,
    stream,
    options: Optional[dict] = None,
    *,
    tracker: Optional[_UsageTracker] = None,
):
    """Retry 9router-style remote sessions on the provider's parallel limit."""
    tracker = tracker or _USAGE_TRACKER.get()
    if stream and auth.get_polling_status().get("enabled"):
        return await _run_remote_stream_with_retry(
            messages, model, options, tracker=tracker
        )
    if auth.get_polling_status().get("enabled"):
        attempts = _polling_retry_limit(options)
        attempted_accounts: set[str] = set()

        async def rotate_remote_account(reason: str) -> bool:
            nonlocal options
            logger.warning("remote %s, rotating account", reason)
            _cancel_unclaimed_account_slot(options)
            next_snapshot = _next_retry_account_snapshot(
                attempted_accounts, attempts
            )
            if next_snapshot is None:
                return False
            account_id, record = next_snapshot
            (
                options,
                token,
                billing_id,
                provider_specific,
            ) = _retry_options_for_account(
                options, account_id, record
            )
            try:
                if tracker is not None:
                    await tracker.rebind(options)
                _rebind_chat_session_account(
                    str(
                        options.get("session_id")
                        or options.get("sessionId")
                        or ""
                    ),
                    account_id,
                    billing_id,
                    token,
                    provider_specific,
                )
            except BaseException:
                _cancel_unclaimed_account_slot(options)
                raise
            return True

        for attempt_index in range(attempts):
            account_key = _retry_account_key(options, attempt_index)
            if account_key in attempted_accounts:
                break
            attempted_accounts.add(account_key)
            attempt_options = options
            try:
                return await run_remote_session(
                    messages, model, stream, attempt_options
                )
            except RemoteQueryTooLarge:
                # This is a deterministic request-size failure. Preserve it
                # for the dispatcher so it can return 413 without replaying
                # the same payload on another account.
                raise
            except EmptyUpstreamResponse as exc:
                if (
                    not exc.retryable
                    or exc.observed_model_event
                    or exc.usage is not None
                    or attempt_index + 1 >= attempts
                ):
                    raise
                if not await rotate_remote_account(
                    "upstream returned an empty response"
                ):
                    raise
                continue
            except RuntimeError as exc:
                if (
                    bool(getattr(exc, "observed_model_event", False))
                    or getattr(exc, "usage", None) is not None
                ):
                    raise
                message = str(exc)
                lowered = message.lower()
                unavailable = (
                    "remote model is not available for the bound account" in lowered
                    or "model binding mismatch" in lowered
                )
                if unavailable:
                    # Model entitlements differ per account. A bound account
                    # missing the requested model is not a request failure;
                    # rotate to the next polling account that can serve it.
                    reason = "model unavailable on bound account"
                elif _is_remote_account_capacity_error(exc) or "429" in lowered:
                    reason = "parallel limit"
                else:
                    raise
                if attempt_index + 1 >= attempts:
                    if unavailable:
                        raise
                    break
                if not await rotate_remote_account(reason):
                    if unavailable:
                        raise
                    break
                continue
            finally:
                _cancel_unclaimed_account_slot(attempt_options)
        raise RuntimeError("All remote accounts busy: Trae parallel limit reached")
    return await run_remote_session(messages, model, stream, options)


def _is_native_rate_limit_error(error: Any) -> bool:
    """Return whether a native IDE/SOLO failure is a pre-output rate limit.

    TraeCode reports account-level throttling as business codes ``3004`` and
    ``4011`` while some gateway variants expose it as HTTP ``429`` or a
    human-readable rate-limit message.  Parameter errors such as ``4001`` are
    deliberately excluded: retrying those with another account cannot repair
    an invalid request envelope.
    """

    if isinstance(error, EmptyUpstreamResponse) and (
        not error.retryable
        or error.observed_model_event
        or error.usage is not None
    ):
        return False
    code = str(getattr(error, "code", "") or "").strip()
    if code in {"3004", "4011", "429"}:
        return True
    if code == "4001":
        return False
    detail = str(error or "").strip().lower()
    if not detail:
        return False
    if "4001" in detail or "param is invalid" in detail:
        return False
    return any(
        marker in detail
        for marker in (
            "4011",
            "429",
            "rate limit",
            "rate-limit",
            "too many requests",
            "exceeded the rate",
        )
    )


async def _run_ide_stream_with_retry(
    messages,
    model,
    options: Optional[dict] = None,
    *,
    tracker: Optional[_UsageTracker] = None,
):
    """Open a native IDE/SOLO stream with pre-output account rotation.

    ``run_ide_chat`` returns a ``StreamingResponse`` before the upstream SSE
    body is iterated, so its 4011/429 business error is raised after the
    dispatcher has already returned.  The old retry helper intentionally
    skipped all streaming requests and consequently left the public stream
    with a 502/empty response.  Own the body iterator here, retrying only
    before the first model data frame and preserving cancellation after output.
    """

    options = dict(options or {})
    tracker = tracker or _USAGE_TRACKER.get()
    polling = auth.get_polling_status()
    if not polling.get("enabled"):
        return await run_ide_chat(messages, model, True, options)

    # A rate-limit response is account-scoped.  The old hard cap of two
    # attempts made a large polling pool effectively useless: after the first
    # account and one fallback account both returned 4011/429, the request was
    # surfaced as a 502 even when other enabled accounts were idle.  Rotate
    # through every usable account once, while retaining the pre-output-only
    # replay guard below.
    attempts = _polling_retry_limit(options)
    attempted_accounts: set[str] = set()

    async def rotate_account() -> bool:
        nonlocal options
        _cancel_unclaimed_account_slot(options)
        next_snapshot = _next_retry_account_snapshot(
            attempted_accounts, attempts
        )
        if next_snapshot is None:
            return False
        account_id, record = next_snapshot
        (
            options,
            token,
            billing_id,
            provider_specific,
        ) = _retry_options_for_account(
            options, account_id, record
        )
        try:
            if tracker is not None:
                await tracker.rebind(options)
            _rebind_chat_session_account(
                str(
                    options.get("session_id")
                    or options.get("sessionId")
                    or ""
                ),
                account_id,
                billing_id,
                token,
                provider_specific,
            )
        except BaseException:
            _cancel_unclaimed_account_slot(options)
            raise
        logger.warning(
            "native IDE/SOLO stream rate limit, retrying on account=%s",
            account_id,
        )
        return True

    async def close_response(response: Any) -> None:
        if response is None:
            return
        iterator = getattr(response, "body_iterator", None)
        close_iterator = getattr(iterator, "aclose", None)
        if close_iterator is not None:
            try:
                await close_iterator()
            except Exception:
                pass
        close = getattr(response, "close", None)
        if close is not None:
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def has_public_data(chunk: Any) -> bool:
        if isinstance(chunk, (bytes, bytearray)):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = str(chunk or "")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("data:"):
                continue
            payload = stripped.split(":", 1)[1].strip()
            if payload and payload != "[DONE]":
                return True
        return False

    def retryable_before_output(exc: Exception, saw_output: bool) -> bool:
        if saw_output or (
            isinstance(exc, EmptyUpstreamResponse)
            and (
                not exc.retryable
                or exc.observed_model_event
                or exc.usage is not None
            )
        ):
            return False
        if _is_native_rate_limit_error(exc):
            return True
        return (
            isinstance(exc, EmptyUpstreamResponse)
            and bool(exc.retryable)
            and not bool(exc.observed_model_event)
        )

    async def open_for_attempt(attempt_index: int):
        account_key = _retry_account_key(options, attempt_index)
        if account_key in attempted_accounts:
            raise RuntimeError("native IDE/SOLO account already attempted")
        attempted_accounts.add(account_key)
        attempt_options = dict(options)
        try:
            return await run_ide_chat(messages, model, True, attempt_options)
        finally:
            _cancel_unclaimed_account_slot(attempt_options)

    current = None
    attempt_index = 0
    while True:
        try:
            current = await open_for_attempt(attempt_index)
            break
        except Exception as exc:
            if (
                attempt_index + 1 >= attempts
                or not retryable_before_output(exc, False)
                or not await rotate_account()
            ):
                raise
            attempt_index += 1

    async def body():
        nonlocal current, attempt_index
        try:
            while True:
                saw_output = False
                try:
                    async for chunk in _response_body_iterator(current):
                        saw_output = saw_output or has_public_data(chunk)
                        yield chunk
                    return
                except Exception as exc:
                    if (
                        attempt_index + 1 >= attempts
                        or not retryable_before_output(exc, saw_output)
                    ):
                        raise
                    await close_response(current)
                    if not await rotate_account():
                        raise
                    attempt_index += 1
                    current = await open_for_attempt(attempt_index)
        finally:
            await close_response(current)

    async def close_current() -> None:
        await close_response(current)

    return StreamingResponse(
        _SessionStreamIterator(body(), close_current),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


async def _run_ide_with_retry(
    messages,
    model,
    stream: bool,
    options: Optional[dict] = None,
    *,
    tracker: Optional[_UsageTracker] = None,
):
    """Retry a native IDE/SOLO request once on an account rate limit.

    The native endpoint can return a business error before emitting any model
    event.  In that case the request is safe to replay on another enabled
    account.  This helper intentionally handles only the request/collection
    phase; once a streaming response has been handed to the caller, the
    existing stream cancellation and usage guards remain authoritative.
    """

    options = dict(options or {})
    tracker = tracker or _USAGE_TRACKER.get()
    polling = auth.get_polling_status()
    if stream:
        return await _run_ide_stream_with_retry(
            messages,
            model,
            options,
            tracker=tracker,
        )
    if not polling.get("enabled"):
        return await run_ide_chat(messages, model, stream, options)

    # Keep non-stream and stream native requests consistent.  Both paths may
    # safely replay only before model output, but neither should stop after an
    # arbitrary two-account prefix when polling has more usable credentials.
    attempts = _polling_retry_limit(options)
    attempted_accounts: set[str] = set()
    for attempt_index in range(max(1, attempts)):
        account_key = _retry_account_key(options, attempt_index)
        if account_key in attempted_accounts:
            break
        attempted_accounts.add(account_key)
        attempt_options = options
        try:
            return await run_ide_chat(messages, model, stream, attempt_options)
        except Exception as exc:
            if (
                not _is_native_rate_limit_error(exc)
                or attempt_index + 1 >= attempts
            ):
                raise
            next_snapshot = _next_retry_account_snapshot(
                attempted_accounts, attempts
            )
            if next_snapshot is None:
                raise
            account_id, record = next_snapshot
            (
                options,
                token,
                billing_id,
                provider_specific,
            ) = _retry_options_for_account(
                options, account_id, record
            )
            try:
                if tracker is not None:
                    await tracker.rebind(options)
                _rebind_chat_session_account(
                    str(
                        options.get("session_id")
                        or options.get("sessionId")
                        or ""
                    ),
                    account_id,
                    billing_id,
                    token,
                    provider_specific,
                )
            except BaseException:
                _cancel_unclaimed_account_slot(options)
                raise
            logger.warning(
                "native IDE/SOLO rate limit, retrying on account=%s",
                account_id,
            )
        finally:
            _cancel_unclaimed_account_slot(attempt_options)
    raise RuntimeError("native IDE/SOLO rate-limit retry ended unexpectedly")


def _normalize_usage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    values = _usage_values(record)
    prompt = values["prompt_tokens"]
    completion = values["completion_tokens"]
    total = values["total_tokens"] or prompt + completion
    credits = values.get("credits_consumed")
    requested_effort = (
        record.get("reasoning_effort")
        or record.get("requested_reasoning_effort")
        or record.get("reasoning_effort_requested")
        or record.get("thinking_effort")
    )
    applied_effort = (
        record.get("reasoning_effort_applied")
        or record.get("applied_reasoning_effort")
        or record.get("native_reasoning_effort")
    )
    requested_context = (
        record.get("context_mode_requested")
        or record.get("requested_context_mode")
    )
    context_mode = (
        record.get("context_mode")
        or record.get("contextMode")
        or record.get("context_window_mode")
    )
    actual_context_mode = (
        record.get("context_mode_actual")
        or record.get("actual_context_mode")
    )
    # Older rows recorded the requested standard profile but did not persist
    # an explicit actual/max result.  Standard is deterministic in that case:
    # expose it as an actual non-Max request instead of rendering ``--`` in
    # the console.  Unknown/custom profiles remain unknown.
    requested_context_bool = _context_mode_bool(requested_context)
    if context_mode in (None, "") and requested_context_bool is False:
        context_mode = "standard"
    if actual_context_mode in (None, "") and requested_context_bool is False:
        actual_context_mode = "standard"
    max_applied = _optional_bool(record.get("max_mode_applied"))
    if max_applied is None:
        max_applied = _optional_bool(record.get("max_mode_enabled"))
    if max_applied is None:
        max_applied = _optional_bool(record.get("max_mode"))
    if max_applied is None:
        max_applied = _optional_bool(record.get("max"))
    if context_mode in (None, "") and max_applied is not None:
        context_mode = "max_1m" if max_applied else "standard"
    if context_mode in (None, "") and actual_context_mode not in (None, ""):
        context_mode = actual_context_mode
    if actual_context_mode in (None, "") and max_applied is not None:
        actual_context_mode = "max_1m" if max_applied else "standard"
    if actual_context_mode in (None, "") and context_mode not in (None, ""):
        actual_context_mode = context_mode
    # Keep explicit boolean metadata alongside the human-readable context
    # fields.  The booleans are easier for API clients and daily aggregation
    # to consume, while the legacy fields remain available for old consoles.
    max_mode_requested = _optional_bool(
        record.get("max_mode_requested")
        if "max_mode_requested" in record
        else None
    )
    if max_mode_requested is None and requested_context not in (None, ""):
        max_mode_requested = _context_mode_bool(requested_context)
    if max_mode_requested is None and "max_mode" in record:
        max_mode_requested = _optional_bool(record.get("max_mode"))
    if max_mode_requested is None and "max" in record:
        max_mode_requested = _optional_bool(record.get("max"))
    max_mode_enabled = _optional_bool(
        record.get("max_mode_enabled")
        if "max_mode_enabled" in record
        else None
    )
    if max_mode_enabled is None:
        max_mode_enabled = max_applied
    if max_mode_enabled is None and actual_context_mode not in (None, ""):
        max_mode_enabled = _context_mode_bool(actual_context_mode)
    tool_used = _optional_bool(record.get("tool_used"))
    tools_requested = _optional_bool(
        record.get("tools_requested")
        if "tools_requested" in record
        else record.get("tool_protocol_requested")
    )
    tool_calls_returned = _optional_bool(
        record.get("tool_calls_returned")
        if "tool_calls_returned" in record
        else record.get("tool_call_returned")
    )
    tool_called = _optional_bool(
        record.get("tool_called") if "tool_called" in record else None
    )
    if tool_called is None and "tool" in record:
        tool_called = _optional_bool(record.get("tool"))
    if tool_called is None:
        # ``tool_used`` is the canonical actual-result marker in v1.0.2;
        # ``tool_calls_returned`` is retained for compatibility with rows
        # written by the stream parser.
        tool_called = tool_used
    if tool_called is None:
        tool_called = tool_calls_returned
    context_window_tokens = _number_value(
        record.get("context_window_tokens")
        or record.get("max_context_tokens")
    )
    usage_turn_id = str(
        record.get("usage_turn_id")
        or record.get("billing_turn_id")
        or record.get("_usage_turn_id")
        or ""
    ).strip() or None
    credits_settled_at = _number_value(record.get("credits_settled_at"))
    queue_position = _number_value(record.get("queue_position"))
    auto_continue_rounds = _number_value(record.get("auto_continue_rounds"))
    trace = record.get("_upstream_trace") or record.get("upstream_trace")
    trace = trace if isinstance(trace, Mapping) else {}
    product = _normalize_usage_product(
        record.get("product")
        or record.get("provider_product")
        or record.get("upstream_product")
        or trace.get("product")
    )
    requested_endpoint = (
        record.get("requested_endpoint")
        or record.get("requested_mode")
        or trace.get("requested_endpoint")
        or trace.get("requested_mode")
    )
    actual_endpoint = (
        record.get("actual_endpoint")
        or record.get("actual_mode")
        or trace.get("actual_endpoint")
        or trace.get("actual_mode")
    )
    if product is None:
        product = _normalize_usage_product(actual_endpoint or requested_endpoint)
    requested_mode = record.get("requested_mode") or trace.get("requested_mode")
    actual_mode = record.get("actual_mode") or trace.get("actual_mode")
    thinking = _normalize_usage_thinking(
        record.get("thinking")
        if "thinking" in record
        else record.get("thinking_config")
    )
    tool_choice = record.get("tool_choice")
    if isinstance(tool_choice, Mapping):
        # Keep only the public selection name; function schemas belong to the
        # request payload and are not useful in a billing history row.
        function = tool_choice.get("function")
        tool_choice = (
            function.get("name")
            if isinstance(function, Mapping) and function.get("name")
            else tool_choice.get("type") or "specified"
        )
    elif tool_choice not in (None, ""):
        tool_choice = str(tool_choice).strip() or None
    normalized.update(
        {
            "account_id": str(record.get("account_id") or "default"),
            "model": str(record.get("model") or "auto"),
            "product": product,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": total,
            "cached_tokens": int(values.get("cached_tokens") or 0),
            "tokens_source": str(
                record.get("tokens_source")
                or (
                    "upstream"
                    if any(
                        key in record
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                        )
                    )
                    else "unknown"
                )
            ),
            "credits_consumed": _credit_round(credits),
            "credits_source": str(
                record.get("credits_source")
                or ("upstream" if credits is not None else "unknown")
            ),
            "credits_settled_at": (
                credits_settled_at
                if credits_settled_at is not None
                else None
            ),
            # Keep only the opaque upstream turn id needed for delayed billing
            # enrichment. Credentials and token payloads are never persisted.
            "usage_turn_id": usage_turn_id,
            "request_id": str(record.get("request_id") or ""),
            "endpoint": record.get("endpoint") or None,
            "requested_endpoint": (
                str(requested_endpoint).strip()
                if requested_endpoint not in (None, "")
                else None
            ),
            "actual_endpoint": (
                str(actual_endpoint).strip()
                if actual_endpoint not in (None, "")
                else None
            ),
            "requested_mode": (
                str(requested_mode).strip()
                if requested_mode not in (None, "")
                else None
            ),
            "actual_mode": (
                str(actual_mode).strip()
                if actual_mode not in (None, "")
                else None
            ),
            "stream": record.get("stream") if "stream" in record else None,
            "status": str(record.get("status") or "completed"),
            "duration_ms": _number_value(record.get("duration_ms")),
            "timestamp": _number_value(record.get("timestamp")) or 0,
            # Request metadata is kept separate from token accounting so old
            # rows can still be rendered with an explicit unknown marker.
            "reasoning_effort": (
                str(requested_effort).strip()
                if requested_effort not in (None, "")
                else None
            ),
            "requested_reasoning_effort": (
                str(requested_effort).strip()
                if requested_effort not in (None, "")
                else None
            ),
            "reasoning_effort_applied": (
                str(applied_effort).strip()
                if applied_effort not in (None, "")
                else None
            ),
            "thinking": thinking,
            "tool_choice": tool_choice,
            "context_mode_requested": (
                str(requested_context).strip()
                if requested_context not in (None, "")
                else None
            ),
            "context_mode": (
                str(context_mode).strip()
                if context_mode not in (None, "")
                else None
            ),
            "context_mode_actual": (
                str(actual_context_mode).strip()
                if actual_context_mode not in (None, "")
                else None
            ),
            "context_window_tokens": context_window_tokens,
            "max_mode_applied": max_applied,
            "max_mode_requested": max_mode_requested,
            "max_mode_enabled": max_mode_enabled,
            # Stable aliases for API clients and older frontends.  The
            # canonical fields above retain the requested-vs-actual
            # distinction; these aliases represent the actual observed
            # capability when it is known.
            "max_mode": max_mode_enabled,
            "max": max_mode_enabled,
            "tools_requested": tools_requested,
            "tool_used": tool_used,
            "tool_calls_returned": tool_calls_returned,
            "tool_called": tool_called,
            "tool": tool_called,
            # Transport telemetry: how long the request waited in Trae's queue
            # and how many automatic continuation rounds completed the answer.
            "queue_position": queue_position,
            "auto_continue_rounds": auto_continue_rounds,
        }
    )
    return normalized


def _public_usage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a usage row without exposing the upstream billing correlation id."""

    normalized = _normalize_usage_record(record)
    normalized.pop("usage_turn_id", None)
    normalized.pop("billing_turn_id", None)
    normalized.pop("_usage_turn_id", None)
    return normalized


def _save_usage_history_locked() -> None:
    try:
        _USAGE_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": _USAGE_HISTORY[:_USAGE_MAX_HISTORY],
        }
        temporary = _USAGE_RECORDS_PATH.with_name(
            _USAGE_RECORDS_PATH.name + ".tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )
        os.replace(temporary, _USAGE_RECORDS_PATH)
    except Exception as exc:
        logger.warning("usage records could not be saved: %s", exc)


def _load_usage_history() -> None:
    global _USAGE_HISTORY
    try:
        if not _USAGE_RECORDS_PATH.exists():
            return
        payload = json.loads(_USAGE_RECORDS_PATH.read_text("utf-8"))
        records = payload.get("records", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("usage records must be a list")
        normalized = [
            _normalize_usage_record(record)
            for record in records
            if isinstance(record, Mapping)
        ][:_USAGE_MAX_HISTORY]
        with _USAGE_LOCK:
            _USAGE_HISTORY = normalized
    except Exception as exc:
        logger.warning("usage records could not be loaded: %s", exc)


def _usage_day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, _CHECKIN_TIMEZONE).strftime("%Y-%m-%d")


def _empty_usage_bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {key: 0 for key in _USAGE_STATS_FIELDS}
    bucket["credits"] = 0.0
    bucket["models"] = {}
    return bucket


def _usage_contribution(record: Mapping[str, Any]) -> dict[str, Any]:
    status = str(record.get("status") or "completed")
    credits = _number_value(record.get("credits_consumed"))
    return {
        "model": str(record.get("model") or "auto"),
        "requests": 1,
        "completed": 1 if status == "completed" else 0,
        "failed": 1 if status == "error" else 0,
        "input_tokens": int(record.get("input_tokens") or 0),
        "output_tokens": int(record.get("output_tokens") or 0),
        "cached_tokens": int(record.get("cached_tokens") or 0),
        "total_tokens": int(record.get("total_tokens") or 0),
        "credits": float(credits or 0),
        "max_mode_requests": 1
        if _optional_bool(record.get("max_mode_requested")) is True
        else 0,
        "max_mode_enabled_requests": 1
        if _optional_bool(record.get("max_mode_enabled")) is True
        else 0,
        "tool_requests": 1
        if _optional_bool(record.get("tools_requested")) is True
        else 0,
        "tool_calls": 1
        if _optional_bool(record.get("tool_called")) is True
        else 0,
    }


def _apply_usage_contribution_locked(
    day: str, contribution: Mapping[str, Any], sign: int
) -> None:
    days = _USAGE_STATS.setdefault("days", {})
    bucket = days.setdefault(day, _empty_usage_bucket())
    for key in _USAGE_STATS_FIELDS:
        value = bucket.get(key, 0) + sign * contribution.get(key, 0)
        bucket[key] = max(0, round(value, 4) if key == "credits" else int(value))
    models = bucket.setdefault("models", {})
    model = contribution.get("model") or "auto"
    entry = models.setdefault(
        model, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    for key in ("requests", "input_tokens", "output_tokens", "total_tokens"):
        entry[key] = max(0, int(entry.get(key, 0) + sign * contribution.get(key, 0)))
    if entry["requests"] == 0:
        models.pop(model, None)
    if len(days) > _USAGE_STATS_MAX_DAYS:
        for stale in sorted(days)[: len(days) - _USAGE_STATS_MAX_DAYS]:
            days.pop(stale, None)


def _usage_stats_replace_locked(
    previous: Mapping[str, Any] | None, record: Mapping[str, Any]
) -> None:
    """Move one request's contribution from its previous form to ``record``.

    Records are rewritten in place (same request id) when retries settle or
    credits are enriched later; subtracting the old row keeps totals exact.
    """
    if previous is not None:
        day = _usage_day(float(previous.get("timestamp") or 0) or time.time())
        _apply_usage_contribution_locked(day, _usage_contribution(previous), -1)
    day = _usage_day(float(record.get("timestamp") or 0) or time.time())
    _apply_usage_contribution_locked(day, _usage_contribution(record), 1)
    _save_usage_stats_locked()


def _usage_stats_path() -> Path:
    # Sits next to usage_records.json unless explicitly overridden, so a
    # relocated records path (tests, alternate data dirs) moves both files.
    if _USAGE_STATS_PATH_OVERRIDE:
        return Path(_USAGE_STATS_PATH_OVERRIDE)
    return _USAGE_RECORDS_PATH.with_name("usage_stats.json")


def _save_usage_stats_locked() -> None:
    try:
        path = _usage_stats_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "days": _USAGE_STATS.get("days", {})}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", "utf-8")
        os.replace(temporary, path)
    except Exception as exc:
        logger.warning("usage stats could not be saved: %s", exc)


def _load_usage_stats() -> None:
    """Load daily totals, seeding them once from the request history."""
    global _USAGE_STATS
    try:
        path = _usage_stats_path()
        if path.exists():
            payload = json.loads(path.read_text("utf-8"))
            days = payload.get("days") if isinstance(payload, dict) else None
            if not isinstance(days, dict):
                raise ValueError("usage stats must contain a days mapping")
            with _USAGE_LOCK:
                _USAGE_STATS = {"days": days}
            return
        with _USAGE_LOCK:
            _USAGE_STATS = {"days": {}}
            for record in _USAGE_HISTORY:
                day = _usage_day(float(record.get("timestamp") or 0) or time.time())
                _apply_usage_contribution_locked(day, _usage_contribution(record), 1)
            if _USAGE_HISTORY:
                _save_usage_stats_locked()
    except Exception as exc:
        logger.warning("usage stats could not be loaded: %s", exc)


def _usage_overview(days: int = 30) -> dict[str, Any]:
    days = max(1, min(int(days or 30), 366))
    now = time.time()
    today = _usage_day(now)
    with _USAGE_LOCK:
        stored = json.loads(json.dumps(_USAGE_STATS.get("days", {})))
    totals = _empty_usage_bucket()
    totals.pop("models")
    for bucket in stored.values():
        for key in _USAGE_STATS_FIELDS:
            totals[key] += bucket.get(key, 0)
    totals["credits"] = round(totals["credits"], 2)
    start = datetime.fromtimestamp(now, _CHECKIN_TIMEZONE).date()
    daily = []
    for offset in range(days):
        day = (start - timedelta(days=offset)).strftime("%Y-%m-%d")
        bucket = stored.get(day) or _empty_usage_bucket()
        models = sorted(
            (
                {"model": name, **values}
                for name, values in (bucket.get("models") or {}).items()
            ),
            key=lambda item: item.get("total_tokens", 0),
            reverse=True,
        )
        daily.append(
            {
                "date": day,
                **{key: bucket.get(key, 0) for key in _USAGE_STATS_FIELDS},
                "credits": round(float(bucket.get("credits") or 0), 2),
                "models": models[:6],
            }
        )
    return {
        "today": next((row for row in daily if row["date"] == today), daily[0]),
        "totals": totals,
        "active_days": sum(1 for bucket in stored.values() if bucket.get("requests")),
        "first_day": min(stored) if stored else None,
        "daily": daily,
    }


def _record_usage(
    account_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    credits_consumed: int | float | None = None,
    credits_source: str = "unknown",
    usage_turn_id: str | None = None,
    request_id: str = "",
    endpoint: str | None = None,
    product: str | None = None,
    requested_endpoint: str | None = None,
    actual_endpoint: str | None = None,
    requested_mode: str | None = None,
    actual_mode: str | None = None,
    stream: bool | None = None,
    status: str = "completed",
    duration_ms: int | float | None = None,
    tokens_source: str = "upstream",
    cached_tokens: int = 0,
    reasoning_effort: str | None = None,
    requested_reasoning_effort: str | None = None,
    reasoning_effort_applied: str | None = None,
    thinking: Any = None,
    context_mode_requested: str | None = None,
    context_mode: str | None = None,
    context_mode_actual: str | None = None,
    context_window_tokens: int | float | None = None,
    max_mode_applied: bool | None = None,
    max_mode_requested: bool | None = None,
    max_mode_enabled: bool | None = None,
    tools_requested: bool | None = None,
    tool_choice: Any = None,
    tool_used: bool | None = None,
    tool_calls_returned: bool | None = None,
    tool_called: bool | None = None,
    queue_position: int | None = None,
    auto_continue_rounds: int | None = None,
) -> dict[str, Any]:
    """Record one API request (newest first) and persist it independently."""
    global _USAGE_HISTORY
    record = _normalize_usage_record(
        {
            "account_id": account_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": cached_tokens,
            "tokens_source": tokens_source,
            "credits_consumed": credits_consumed,
            "credits_source": credits_source,
            "credits_settled_at": time.time() if credits_consumed is not None else None,
            "usage_turn_id": usage_turn_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "product": product,
            "requested_endpoint": requested_endpoint,
            "actual_endpoint": actual_endpoint,
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "stream": stream,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            "reasoning_effort": reasoning_effort,
            "requested_reasoning_effort": requested_reasoning_effort,
            "reasoning_effort_applied": reasoning_effort_applied,
            "thinking": thinking,
            "context_mode_requested": context_mode_requested,
            "context_mode": context_mode,
            "context_mode_actual": context_mode_actual,
            "context_window_tokens": context_window_tokens,
            "max_mode_applied": max_mode_applied,
            "max_mode_requested": max_mode_requested,
            "max_mode_enabled": max_mode_enabled,
            "tools_requested": tools_requested,
            "tool_choice": tool_choice,
            "tool_used": tool_used,
            "tool_calls_returned": tool_calls_returned,
            "tool_called": tool_called,
            "queue_position": queue_position,
            "auto_continue_rounds": auto_continue_rounds,
        }
    )
    with _USAGE_LOCK:
        previous = None
        if request_id:
            for index, existing in enumerate(_USAGE_HISTORY):
                if existing.get("request_id") == request_id:
                    previous = existing
                    # Keep the original day bucket for a rewritten request.
                    record["timestamp"] = existing.get("timestamp") or record["timestamp"]
                    _USAGE_HISTORY[index] = record
                    break
            else:
                _USAGE_HISTORY.insert(0, record)
        else:
            _USAGE_HISTORY.insert(0, record)
        _usage_stats_replace_locked(previous, record)
        if len(_USAGE_HISTORY) > _USAGE_MAX_HISTORY:
            _USAGE_HISTORY = _USAGE_HISTORY[:_USAGE_MAX_HISTORY]
        _save_usage_history_locked()
    if request_id and credits_consumed is not None:
        credits = _refresh_credit_projection(str(account_id or ""))
        projected = _number_value(credits.get("remaining"))
        if projected is not None:
            _update_usage_record(
                request_id,
                credits_remaining_after=_credit_round(projected),
                credits_reconcile_status=credits.get("reconcile_status"),
                credits_reconcile_confidence=credits.get(
                    "reconcile_confidence"
                ),
            )
    return record


def _upstream_response_error(response: Any) -> str:
    """Return a concise error for an upstream response, or ``""``.

    Adapters normally raise on non-2xx responses, but a few native bridges
    return a JSON error response instead.  Treat both forms uniformly so the
    dispatcher can continue to the configured Remote fallback.
    """

    status = int(getattr(response, "status_code", 200) or 200)
    if status < 400:
        return ""
    payload = _response_json_payload(response)
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping):
        message = str(error.get("message") or "").strip()
    else:
        message = str(error or "").strip()
    return f"HTTP {status}: {message or 'upstream returned an error'}"


def _remote_fallback_options(
    options: Optional[Mapping[str, Any]], requested_mode: str
) -> dict[str, Any]:
    """Build options for a native-endpoint -> Remote fallback attempt."""

    fallback = dict(options or {})
    fallback["_upstream_mode"] = "remote"
    fallback["_upstream_fallback_from"] = requested_mode
    # Work's primary route uses Remote chat_sessions with an explicit Work
    # executor. Its fallback must clear those bindings and use the ordinary
    # Agent tier; otherwise a request labelled Remote can still be billed and
    # executed as Work.
    if requested_mode in {"work", "work-agent"}:
        fallback.pop("_trae_mode", None)
        fallback.pop("_remote_agent_type", None)
        fallback.pop("remote_agent_type", None)
        fallback["_remote_agent_type"] = "solo_agent_remote"
    elif not (
        fallback.get("_remote_agent_type")
        or fallback.get("remote_agent_type")
    ):
        fallback_agent_type = (
            "solo_work_remote"
            if requested_mode in {
                "traework-native",
                "native",
                "traework",
            }
            else "solo_agent_remote"
        )
        fallback["_remote_agent_type"] = fallback_agent_type
    fallback.setdefault("_session_variant", f"{requested_mode}-fallback")
    return fallback


def _update_usage_record(request_id: str, **updates: Any) -> None:
    if not request_id:
        return
    account_id = ""
    should_refresh_projection = False
    normalized_record: dict[str, Any] | None = None
    with _USAGE_LOCK:
        for index, existing in enumerate(_USAGE_HISTORY):
            if existing.get("request_id") != request_id:
                continue
            merged = dict(existing)
            merged.update(updates)
            if (
                "credits_consumed" in updates
                and _number_value(updates.get("credits_consumed")) is not None
            ):
                # The request timestamp can precede a delayed commercial bill.
                # Keep a settlement timestamp so the overlay is anchored to
                # when the charge became observable, not when the model began.
                merged["credits_settled_at"] = time.time()
                should_refresh_projection = True
            # Round credit fields before normalizing
            for key in ("credits_consumed", "credits_before", "credits_after"):
                if key in merged:
                    merged[key] = _credit_round(merged[key])
            normalized_record = _normalize_usage_record(merged)
            _USAGE_HISTORY[index] = normalized_record
            account_id = str(normalized_record.get("account_id") or "")
            _usage_stats_replace_locked(existing, normalized_record)
            _save_usage_history_locked()
            break
    if normalized_record is None:
        return
    if should_refresh_projection and account_id:
        credits = _refresh_credit_projection(account_id)
        projected = _number_value(credits.get("remaining"))
        if projected is not None:
            # Store the balance visible immediately after this charge.  This
            # is metadata on the row only; the account cache remains the
            # single source used by the dashboard.
            with _USAGE_LOCK:
                for index, current in enumerate(_USAGE_HISTORY):
                    if current.get("request_id") != request_id:
                        continue
                    updated = dict(current)
                    updated["credits_remaining_after"] = _credit_round(projected)
                    updated["credits_reconcile_status"] = credits.get(
                        "reconcile_status"
                    )
                    updated["credits_reconcile_confidence"] = credits.get(
                        "reconcile_confidence"
                    )
                    normalized = _normalize_usage_record(updated)
                    _USAGE_HISTORY[index] = normalized
                    _usage_stats_replace_locked(current, normalized)
                    _save_usage_history_locked()
                    break


# Raw v2 answers ``2001 failed to get app config: record not found`` when the
# account has no enterprise app config.  That is deterministic, so remember it
# briefly and route straight to the next candidate instead of paying the
# round-trip on every request.
_RAW_APP_CONFIG_MISSING: dict[str, float] = {}
_RAW_APP_CONFIG_TTL_SECONDS = 600.0


def _raw_app_config_missing_error(error: Any) -> bool:
    text = str(error or "").lower()
    return "app config" in text and "record not found" in text


def _raw_app_config_key(model: Any, options: Mapping) -> str:
    return f"{options.get('_account_id') or 'default'}|{str(model or '').lower()}"


def _raw_app_config_known_missing(model: Any, options: Mapping) -> bool:
    if not options.get("_account_id"):
        return False
    key = _raw_app_config_key(model, options)
    expires = _RAW_APP_CONFIG_MISSING.get(key)
    if not expires:
        return False
    if expires < time.monotonic():
        _RAW_APP_CONFIG_MISSING.pop(key, None)
        return False
    return True


AUTO_ROUTE_MODE = "auto-route"
# Endpoints chosen by auto routing. IDE Agent passes the full tool suite
# (including parallel calls); Remote is the general chat path and the final
# fallback for everything.
AUTO_ROUTE_TOOL_ENDPOINT = "ide"
AUTO_ROUTE_CHAT_ENDPOINT = "remote"


def _auto_route_enabled() -> bool:
    try:
        return bool(auth.get_auto_route_settings().get("enabled"))
    except Exception:
        return False


def _task_model_routing_enabled() -> bool:
    """Whether ``model=auto`` should be assigned from task complexity.

    Automatic task routing is the safe default for this relay: explicit model
    names are never rewritten, while ``auto`` is assigned to the configured
    simple/complex candidates before endpoint dispatch. Operators that need
    the legacy provider-owned ``auto`` selection can disable it explicitly.
    """

    value = str(os.environ.get("TRAE_TASK_MODEL_ROUTING", "1") or "").strip().lower()
    return value not in {"", "0", "false", "no", "off", "disabled"}


def _task_model_is_automatic(model: Any) -> bool:
    """Return whether the caller delegated model selection to the relay."""

    return str(model or "").strip().lower() in {"", "auto"}


def _task_message_text(message: Any) -> str:
    """Extract bounded plain text for deterministic complexity classification."""

    if not isinstance(message, Mapping):
        return str(message or "")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, Mapping):
        return " ".join(
            _task_message_text({"content": value}) for value in content.values()
        )
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                value = (
                    item.get("text")
                    or item.get("content")
                    or item.get("value")
                    or item.get("input")
                    or ""
                )
                if isinstance(value, (Mapping, list)):
                    parts.append(_task_message_text({"content": value}))
                else:
                    parts.append(str(value or ""))
            elif item not in (None, ""):
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


_TASK_COMPLEXITY_MARKERS = re.compile(
    r"""
    \b(?:implement|implementation|modify|change|create|write|edit|patch|refactor|
       debug|diagnose|fix|test|review|analy[sz]e|research|deploy|download|install|
       command|shell|script|file|files|project|repository|repo|code|coding|
       multi[- ]?step|step[- ]?by[- ]?step|batch|automate|automation)\b
    |(?:实现|修改|创建|编写|编辑|补丁|重构|调试|排查|修复|测试|审查|分析|研究|
       部署|下载|安装|命令|脚本|文件|项目|仓库|代码|多步|批量|自动化|完整链路)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _task_is_complex(
    messages: Optional[list[dict[str, Any]]], options: Optional[Mapping[str, Any]]
) -> tuple[bool, str]:
    """Classify a request without contacting the upstream model catalog."""

    messages = messages or []
    options = options or {}
    if _tool_protocol_requested(dict(options), messages):
        return True, "tools"
    if images.messages_have_images(messages):
        return True, "image"
    if any(
        isinstance(message, Mapping)
        and str(message.get("role") or "").strip().lower()
        in {"tool", "function"}
        for message in messages
    ):
        return True, "tool_history"

    text = "\n".join(_task_message_text(message) for message in messages)
    # Long history and large prompts are much more likely to need the
    # high-capability model even when the latest user turn is short.
    if len(messages) >= 12 or len(text) >= 16_000:
        return True, "long_context"
    if _TASK_COMPLEXITY_MARKERS.search(text):
        return True, "action_or_code"
    return False, "simple"


def _task_model_endpoint(options: Optional[Mapping[str, Any]] = None) -> str:
    """Return the canonical endpoint identity used to scope model evidence."""

    options = options or {}
    mode = str(
        options.get("_upstream_mode") or _current_upstream_mode() or ""
    ).strip().lower()
    aliases = {
        "direct": "raw",
        "auto": "raw",
        "work-agent": "work",
        "9router": "remote",
        "trae-remote": "remote",
    }
    return aliases.get(mode, mode)


def _task_model_account_key(options: Optional[Mapping[str, Any]] = None) -> str:
    """Return the identity used by the account-bound model catalogue cache."""

    options = options or {}
    token = str(options.get("_auth_token") or "").strip()
    user_id = str(
        options.get("_auth_user_id")
        or options.get("_billing_id")
        or options.get("_account_id")
        or ""
    ).strip()
    if not token and not user_id:
        return ""
    try:
        return str(trae_client._web_model_cache_key(token, user_id) or "")
    except Exception:
        return user_id


def _task_catalog_model_names(
    options: Optional[Mapping[str, Any]] = None,
    *,
    endpoint: str = "",
) -> set[str]:
    """Read only model evidence scoped to this endpoint and bound account."""

    options = options or {}
    endpoint = endpoint or _task_model_endpoint(options)
    account_key = _task_model_account_key(options)
    names: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, Mapping):
            descriptor_keys = {
                "id",
                "name",
                "config_name",
                "model_name",
                "display_name",
                "display_model_name",
            }
            if not descriptor_keys.intersection(value):
                for key in value:
                    add(key)
            for key in (
                "id",
                "name",
                "config_name",
                "model_name",
                "display_name",
                "display_model_name",
            ):
                add(value.get(key))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        text = str(value or "").strip().lower()
        if text:
            names.add(text)

    # Internal callers may provide a request-local catalog, but only when its
    # endpoint and account labels match the already-bound request identity.
    catalog_endpoint = str(options.get("_available_models_endpoint") or "").strip().lower()
    catalog_account = str(options.get("_available_models_account_key") or "").strip()
    if (
        catalog_endpoint == endpoint
        and account_key
        and catalog_account == account_key
    ):
        add(options.get("_available_models"))

    # The remote model cache is partitioned by account and executor tier. Do
    # not use the merged /v1/models cache or a different account/tier as proof
    # that a candidate can be used on this request.
    if endpoint in {"remote", "web", "work"} and account_key:
        requested_agent_type = str(
            options.get("_remote_agent_type")
            or options.get("remote_agent_type")
            or ""
        ).strip().lower()
        if endpoint == "work" or requested_agent_type in {
            "solo_work_remote",
            "solo_work_lite",
            "work",
        }:
            agent_types = ("solo_work_remote",)
        elif requested_agent_type in {"solo_agent_remote", "solo_agent_lite", "agent"}:
            agent_types = ("solo_agent_remote",)
        else:
            agent_types = (
                "solo_work_remote",
                "solo_agent_remote",
            ) if os.environ.get("TRAE_REMOTE_AGENT_FIRST", "1").strip().lower() in {
                "0", "false", "no", "off"
            } else ("solo_agent_remote",)
        cache = getattr(trae_client, "_WEB_MODEL_CACHE", {})
        ttl = float(getattr(trae_client, "_WEB_MODEL_CACHE_TTL", 300) or 300)
        now = time.time()
        for agent_type in agent_types:
            if not isinstance(cache, Mapping):
                continue
            prefix = f"{account_key}:{agent_type}:"
            for cache_key, cached in cache.items():
                if str(cache_key) != prefix[:-1] and not str(cache_key).startswith(prefix):
                    continue
                if (
                    isinstance(cached, tuple)
                    and len(cached) > 1
                    and now - float(cached[0]) <= ttl
                    and isinstance(cached[1], Mapping)
                ):
                    add(cached[1])
                    break
    return names


def _task_model_available(
    model: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    endpoint: str = "",
) -> bool:
    """Require a current-endpoint, current-account catalogue match."""

    candidate = str(model or "").strip().lower()
    if not candidate:
        return False
    endpoint = endpoint or _task_model_endpoint(options)
    names = _task_catalog_model_names(options, endpoint=endpoint)
    if candidate in names:
        return True
    try:
        mapped = str(trae_client.convert_model_name(candidate) or "").strip().lower()
    except Exception:
        mapped = ""
    return bool(mapped and mapped in names)


def _apply_task_model_routing(
    messages: Optional[list[dict[str, Any]]],
    model: Any,
    options: Optional[dict],
) -> tuple[str, dict]:
    """Assign configured simple/complex models only for automatic requests.

    The complex candidate is deliberately selected only when it is present in
    a catalog that was already loaded by this process.  A missing ``6.1sol``
    must remain an explicit ``auto`` request rather than silently becoming
    Kimi or another static fallback.
    """

    options = dict(options or {})
    requested = str(model or "").strip()
    if not _task_model_routing_enabled() or not _task_model_is_automatic(model):
        return requested or str(model or ""), options

    endpoint = _task_model_endpoint(options)
    account_key = _task_model_account_key(options)
    existing = str(options.get("_task_model_effective") or "").strip()
    if (
        options.get("_task_model_routed")
        and existing
        and str(options.get("_task_model_routed_endpoint") or "") == endpoint
        and str(options.get("_task_model_routed_account_key") or "") == account_key
    ):
        return existing, options

    complex_task, reason = _task_is_complex(messages, options)
    simple_model = (
        str(os.environ.get("TRAE_SIMPLE_MODEL", "deepseek-v4.1-flash") or "")
        .strip()
        or "deepseek-v4.1-flash"
    )
    complex_model = (
        str(os.environ.get("TRAE_COMPLEX_MODEL", "6.1sol") or "").strip()
        or "6.1sol"
    )
    candidate = complex_model if complex_task else simple_model
    effective = candidate
    fallback_reason = ""
    if not _task_model_available(candidate, options, endpoint=endpoint):
        effective = "auto"
        fallback_reason = (
            f"{candidate} is not confirmed for endpoint={endpoint or 'unknown'} "
            f"account={account_key or 'unbound'}"
        )

    options.update(
        {
            "_task_model_routed": True,
            "_task_model_routed_endpoint": endpoint,
            "_task_model_routed_account_key": account_key,
            "_task_model_requested": requested or "auto",
            "_task_model_class": "complex" if complex_task else "simple",
            "_task_model_reason": reason,
            "_task_model_candidate": candidate,
            "_task_model_effective": effective,
        }
    )
    if _requested_reasoning_effort(options) is None:
        options["reasoning_effort"] = "ultra"
        options["_task_model_reasoning_effort"] = "ultra"

    trace = options.get("_upstream_trace")
    if not isinstance(trace, dict):
        trace = {}
        options["_upstream_trace"] = trace
    trace.update(
        task_model_routing=True,
        task_model_requested=requested or "auto",
        task_model_class=options["_task_model_class"],
        task_model_reason=reason,
        task_model_candidate=candidate,
        task_model_effective=effective,
    )
    if fallback_reason:
        trace.update(
            task_model_fallback="auto",
            task_model_fallback_reason=fallback_reason,
        )
        logger.warning(
            "task model candidate unavailable requested=%s candidate=%s; keeping auto",
            requested or "auto",
            candidate,
        )
    else:
        trace.pop("task_model_fallback", None)
        trace.pop("task_model_fallback_reason", None)
    logger.info(
        "task model route requested=%s class=%s candidate=%s effective=%s reason=%s",
        requested or "auto",
        options["_task_model_class"],
        candidate,
        effective,
        reason,
    )
    return effective, options


def _multimodal_catalog_match(
    configs: Mapping[str, Any], model: Any
) -> tuple[str, Optional[Mapping[str, Any]]]:
    requested = str(model or "").strip().casefold()
    if not requested:
        return "", None
    for name, config in configs.items():
        if not isinstance(config, Mapping):
            continue
        values = trae_client._model_descriptor_values(name, config)
        if any(value.casefold() == requested for value in values):
            return str(name), config
    return "", None


def _multimodal_catalog_scope(
    options: Optional[Mapping[str, Any]],
) -> tuple[str, str]:
    options = options or {}
    endpoint = _task_model_endpoint(options)
    explicit_function = str(
        options.get("_traecode_raw_chat_function")
        or options.get("raw_chat_function")
        or ""
    ).strip().lower()
    if explicit_function:
        return "", explicit_function
    requested_agent = str(
        options.get("_remote_agent_type")
        or options.get("remote_agent_type")
        or ""
    ).strip().lower()
    if endpoint == "work" or requested_agent in {
        "solo_work_remote",
        "solo_work_lite",
        "work",
    }:
        return "solo_work_remote", "solo_work_remote"
    if endpoint in {"remote", "web"} or requested_agent:
        agent = (
            "solo_agent_remote"
            if requested_agent not in {"solo_work_remote", "solo_work_lite", "work"}
            else "solo_work_remote"
        )
        return agent, agent
    if endpoint == "solo":
        return "", "solo_agent"
    if endpoint in {"ide", "raw"}:
        return "", "chat_v3"
    return "", ""


async def _load_multimodal_catalog(
    messages: Optional[list[dict]], options: dict
) -> None:
    """Attach current-account/function model evidence to one request."""

    if not images.messages_have_images(messages):
        return
    requested_model = str(options.get("_requested_model") or "").strip()
    explicit_request = requested_model and not _task_model_is_automatic(requested_model)
    explicit_switch = str(
        os.environ.get("TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if explicit_request and not explicit_switch:
        return
    token = str(options.get("_auth_token") or "").strip()
    user_id = str(
        options.get("_auth_user_id")
        or options.get("_billing_id")
        or options.get("_account_id")
        or ""
    ).strip()
    if not token:
        return
    provider = options.get("provider_specific")
    if provider is None:
        provider = options.get("providerSpecificData")
    agent_type, raw_function = _multimodal_catalog_scope(options)
    try:
        configs = await trae_client.get_scoped_model_configs(
            token_override=token,
            user_id_override=user_id,
            provider_specific=(dict(provider) if isinstance(provider, Mapping) else None),
            agent_type=agent_type,
            raw_chat_function=raw_function,
        )
    except Exception as exc:
        logger.info("multimodal catalog lookup failed: %s", exc)
        return
    if configs:
        options["_trae_model_capability_catalog"] = configs


def _multimodal_fallback_model(
    model: str, options: Optional[Mapping[str, Any]] = None
) -> str:
    """Pick a vision-capable model for an image-bearing request.

    Trae's text-only configs cannot read an attached image; forwarding the
    request unchanged makes the model answer as if no image existed.  Prefer
    the operator's explicit override, then the account's live catalog, then the
    built-in preference order.
    """

    requested = str(model or "").strip()
    catalog = (options or {}).get("_trae_model_capability_catalog")
    if not isinstance(catalog, Mapping):
        return ""
    _current_name, current = _multimodal_catalog_match(catalog, requested)
    if descriptor_multimodal(current) is True:
        return ""
    available: dict[str, str] = {}
    for name, config in catalog.items():
        if isinstance(config, Mapping) and descriptor_multimodal(config) is True:
            available[str(name).strip().casefold()] = str(name).strip()
    candidates: list[str] = []
    override = configured_multimodal_model()
    if override:
        override_name, override_config = _multimodal_catalog_match(catalog, override)
        if override_name and descriptor_multimodal(override_config) is True:
            candidates.append(override_name)
    candidates.extend(
        available[name.casefold()]
        for name in MULTIMODAL_FALLBACK_ORDER
        if name.casefold() in available
    )
    candidates.extend(
        name for _key, name in sorted(available.items()) if name not in candidates
    )
    for candidate in candidates:
        name = str(candidate or "").strip()
        if name and name.lower() != requested.lower():
            return name
    return ""


def _apply_multimodal_switch(
    messages: Optional[list[dict]], model: str, options: Optional[dict]
) -> str:
    """Re-point an image request at a vision model; returns the effective model."""

    if not images.messages_have_images(messages):
        return model
    catalog = options.get("_trae_model_capability_catalog") if isinstance(options, Mapping) else None
    if isinstance(catalog, Mapping):
        _name, descriptor = _multimodal_catalog_match(catalog, model)
        if descriptor_multimodal(descriptor) is True:
            return model
    # An explicitly selected model is part of the API contract.  Re-pointing
    # it to a vision model silently changes both the provider fingerprint and
    # the billable model (for example glm-5.2 -> kimi-k2.6).  Automatic model
    # selection may still choose a vision-capable candidate, but an explicit
    # request must stay pinned unless the operator opts into the legacy
    # behaviour with TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT=1.
    options = options if isinstance(options, dict) else {}
    requested_model = str(options.get("_requested_model") or "").strip()
    explicit_request = requested_model and not _task_model_is_automatic(requested_model)
    explicit_switch = str(
        os.environ.get("TRAE_MULTIMODAL_AUTO_SWITCH_EXPLICIT", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if explicit_request and not explicit_switch:
        trace = options.get("_upstream_trace")
        if isinstance(trace, dict):
            trace["multimodal_switch_skipped"] = {
                "requested": requested_model,
                "effective": model,
                "reason": "explicit_model_pinned",
            }
        logger.info(
            "image request keeps explicitly selected model=%s",
            model,
        )
        return model
    if str(os.environ.get("TRAE_MULTIMODAL_AUTO_SWITCH", "1")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return model
    fallback = _multimodal_fallback_model(model, options)
    if not fallback:
        logger.warning(
            "image request kept on text-only model=%s (no multimodal model available)",
            model,
        )
        return model
    logger.info(
        "image request re-pointed model=%s -> %s", model, fallback
    )
    trace = (options or {}).get("_upstream_trace")
    if isinstance(trace, dict):
        trace["multimodal_switch"] = {"from": model, "to": fallback}
    return fallback


def _apply_auto_route(
    messages: Optional[list[dict]], options: Optional[dict]
) -> dict:
    """Resolve the endpoint for one request when auto routing applies.

    Auto routing runs when the caller asked for ``auto-route`` explicitly, or
    when the console switch is on and the request carries no explicit
    endpoint. Tool and image requests go to IDE Agent, where media travels
    natively instead of inflating Remote's flattened query. Plain text chat
    goes to Remote; the existing dispatcher owns endpoint fallback.
    """

    options = dict(options or {})
    explicit = str(options.get("_upstream_mode") or "").strip().lower()
    if explicit and explicit != AUTO_ROUTE_MODE:
        return options
    if options.get("_traework_custom_model"):
        return options
    if not explicit:
        if not _auto_route_enabled():
            return options
        # Local CLI / native helper modes are host-specific; leave them alone.
        if _current_upstream_mode() in (
            "cli", "traework-native", "native", "traework"
        ):
            return options
    uses_tools = _tool_protocol_requested(options, messages)
    has_images = images.messages_have_images(messages)
    endpoint = (
        AUTO_ROUTE_TOOL_ENDPOINT
        if uses_tools or has_images
        else AUTO_ROUTE_CHAT_ENDPOINT
    )
    options["_upstream_mode"] = endpoint
    options["_auto_route"] = "tools" if uses_tools else "images" if has_images else "chat"
    trace = options.get("_upstream_trace")
    if isinstance(trace, dict):
        trace.update(auto_route=True, auto_route_reason=options["_auto_route"])
    logger.info(
        "auto route id=%s model_request=%s endpoint=%s",
        str(options.get("_relay_request_id") or ""),
        options["_auto_route"],
        endpoint,
    )
    return options


def _trace_failure_reason(endpoint: str, error: Any) -> str:
    """Return a bounded, readable endpoint failure diagnostic."""

    detail = str(error or "upstream request failed").strip().replace("\r", " ").replace("\n", " ")
    if len(detail) > 400:
        detail = detail[:397] + "..."
    return f"{endpoint}: {detail}" if endpoint else detail


def _trace_append_failure(trace: Any, endpoint: str, error: Any) -> None:
    """Append one endpoint failure to the request trace without duplication."""

    if not isinstance(trace, dict):
        return
    item = _trace_failure_reason(endpoint, error)
    previous = str(trace.get("fallback_reason") or "").strip()
    parts = [part.strip() for part in previous.split(";") if part.strip()]
    if item not in parts:
        parts.append(item)
    trace["fallback_reason"] = "; ".join(parts)[:1200]
    if endpoint:
        failed = [part.strip() for part in str(trace.get("failed_endpoint") or "").split(",") if part.strip()]
        if endpoint not in failed:
            failed.append(endpoint)
        trace["failed_endpoint"] = ",".join(failed)


async def _dispatch_chat(messages, model, stream: bool, options: Optional[dict] = None):
    requested_model = model
    options = dict(options or {})
    options.setdefault("_requested_model", str(requested_model or "").strip())
    options = _apply_auto_route(messages, options)
    # Resolve the concrete endpoint before selecting an automatic task model.
    # Model availability is product/account scoped; choosing from a global
    # alias table before this point can send an otherwise valid ``auto``
    # request to a model unsupported by the selected executor.
    model, options = _apply_task_model_routing(messages, model, options)
    trace = options.get("_upstream_trace")
    active_tracker = _USAGE_TRACKER.get()
    if active_tracker is not None:
        # Automatic endpoint/model selection happens at dispatch time. Keep
        # both the usage row's model and its metadata aligned even when the
        # selected model stays unchanged (for example, auto-route adds the
        # resolved endpoint or task strength to a copied options mapping).
        if getattr(active_tracker, "model", model) != model:
            active_tracker.model = model
        active_tracker.options = options
    # A text-only config cannot read an attached image, so an image request is
    # re-pointed at a multimodal model before any routing decision is made.
    # Load evidence for this bound account/function first; the switch helper
    # intentionally refuses to use static model-name capability tables.
    await _load_multimodal_catalog(messages, options)
    model = _apply_multimodal_switch(messages, model, options)
    trace = options.get("_upstream_trace")
    if active_tracker is not None:
        # Keep billing/history metadata aligned with the model that will
        # actually be sent upstream.  This matters when automatic multimodal
        # selection is enabled; otherwise the row can say glm-5.2 while the
        # provider fingerprint is kimi-k2.6.
        active_tracker.model = model
        active_tracker.options = options
    if isinstance(trace, dict):
        trace["requested_model"] = str(
            options.get("_requested_model") or requested_model or ""
        ).strip() or "auto"
        trace["effective_model"] = str(model or "").strip()
    try:
        _ensure_model_account_enabled(
            str(options.get("_account_id") or ""),
            str(options.get("_auth_token") or ""),
        )
    except _ModelAccountUnavailable as exc:
        return _openai_error(503, str(exc), "model_account_unavailable")
    active_mode = str(options.get("_upstream_mode") or _current_upstream_mode()).lower()
    fallback_from = str(options.get("_upstream_fallback_from") or "").strip().lower()
    if isinstance(trace, dict):
        if fallback_from:
            # A streaming/native fallback re-enters this dispatcher with the
            # same trace object. Preserve the original requested endpoint and
            # reason while updating the current target.
            trace.setdefault("requested_mode", fallback_from)
            trace.setdefault("requested_endpoint", fallback_from)
            trace.update(
                actual_mode="",
                actual_endpoint="",
                fallback_used=True,
                fallback_target=active_mode,
            )
        else:
            trace.update(
                requested_mode=active_mode,
                requested_endpoint=active_mode,
                actual_mode="",
                actual_endpoint="",
                fallback_used=False,
                fallback_reason="",
                fallback_target="",
                failed_endpoint="",
            )
    logger.info(
        "dispatch start model=%s stream=%s messages=%d last_role=%s last_chars=%d "
        "tools=%s session=%s",
        model,
        stream,
        len(messages or []),
        (messages[-1].get("role") if messages and isinstance(messages[-1], Mapping) else ""),
        (len(str(messages[-1].get("content") or "")) if messages and isinstance(messages[-1], Mapping) else 0),
        _tool_protocol_requested(options, messages),
        str(options.get("session_id") or options.get("sessionId") or "")[:16],
    )
    if not str(model or "").strip():
        return _openai_error(
            400,
            "model is required and cannot be blank",
            "invalid_request_error",
            "model",
        )
    if not trae_client.is_model_supported(model):
        return _openai_error(400, f"Unsupported model: {model}", "invalid_request_error", "model")

    # TraeWork's custom-model adapter is a separate ingress contract while the
    # actual file/shell tools remain owned by the Windows client. The official
    # raw endpoint currently returns a business authentication error for these
    # account credentials, so the verified remote transport is the default.
    # Keep raw/direct as explicit diagnostics instead of advertising a fallback
    # that cannot catch errors raised later by a StreamingResponse iterator.
    if options.get("_traework_custom_model"):
        # Trae 3.4.4 Work/Code clients share the raw-v2 endpoint and declare
        # the native function in the body/header. Route that explicit chain
        # through the TraeCode adapter so its extended DTO, client headers,
        # and JWT identity binding are preserved. Older custom-model calls do
        # not set this marker and retain the Remote compatibility default.
        if options.get("_traecode_raw_chat_function"):
            if isinstance(trace, dict):
                trace.update(
                    product="trae",
                    requested_mode="traecode",
                    requested_endpoint="/api/ide/v2/llm_raw_chat",
                    actual_mode="ide",
                    actual_endpoint="/api/ide/v2/llm_raw_chat",
                )
            logger.info(
                "dispatch TraeCode raw-v2 ingress id=%s model=%s function=%s",
                str(options.get("_relay_request_id") or ""),
                model,
                str(options.get("_traecode_raw_chat_function") or ""),
            )
            return await run_ide_chat(messages, model, stream, options)
        custom_mode = str(
            options.get("_traework_upstream_mode")
            or os.environ.get("TRAEWORK_CUSTOM_UPSTREAM_MODE", "remote")
        ).strip().lower()
        if isinstance(trace, dict):
            trace.update(
                product="traework",
                requested_mode="traework",
                requested_endpoint="traework",
                actual_mode=custom_mode,
                actual_endpoint=(
                    "remote-work" if custom_mode in {"remote", "web"} else custom_mode
                ),
            )
        if custom_mode in {"raw", "direct", "auto"}:
            logger.info(
                "dispatch TraeWork custom-model ingress id=%s mode=raw model=%s",
                str(options.get("_relay_request_id") or ""),
                model,
            )
            return await run_raw_chat(messages, model, stream, options)
        if custom_mode in {"remote", "web"}:
            logger.info(
                "dispatch TraeWork custom-model ingress id=%s mode=%s model=%s",
                str(options.get("_relay_request_id") or ""),
                custom_mode,
                model,
            )
            if custom_mode == "web":
                return await _run_web_with_retry(messages, model, stream, options)
            return await _run_remote_with_retry(messages, model, stream, options)
        raise RuntimeError(
            "Unsupported TRAEWORK_CUSTOM_UPSTREAM_MODE: " + custom_mode
        )

    # External tools always execute on the API caller.  Web/IDE agent routes
    # may execute their own tools on the relay host, so they are never valid
    # fallbacks for a request that advertises caller-owned tools.
    tool_protocol_requested = _tool_protocol_requested(options, messages)
    # Caller-owned tools are now supported on all upstream paths.
    # Remote/web routes inject tool definitions as a system prompt via
    # `_messages_with_client_runtime` and filter tool calls from the
    # upstream text response with `_filter_tool_calls`.
    if tool_protocol_requested:
        logger.info(
            "dispatch tool protocol id=%s mode=%s model=%s",
            str(options.get("_relay_request_id") or ""),
            active_mode,
            model,
        )

    # Raw v2 is the default for every model. Operators can opt specific models
    # (or ``*``) into the account-bound remote executor for diagnostics.
    if not options.get("_disable_upstream_fallback") and _requires_remote_model(model) and active_mode in (
        "raw",
        "direct",
        "auto",
        "ide",
        "solo",
    ):
        logger.info(
            "dispatch remote-only model id=%s model=%s account=%s",
            str(options.get("_relay_request_id") or ""),
            model,
            str(options.get("_account_id") or "default"),
        )
        try:
            if isinstance(trace, dict):
                trace.update(
                    actual_mode="remote",
                    actual_endpoint="remote",
                    fallback_used=True,
                    fallback_target="remote",
                )
                _trace_append_failure(
                    trace,
                    active_mode,
                    "model is configured for the Remote executor",
                )
            return await _run_remote_with_retry(messages, model, stream, options)
        except ModelProviderMismatch as exc:
            _trace_append_failure(trace, "remote", exc)
            logger.error(
                "remote provider model mismatch id=%s requested=%s error=%s",
                str(options.get("_relay_request_id") or ""),
                model,
                exc,
            )
            return _openai_error(502, str(exc), "upstream_model_mismatch", "model")

    # ``auto`` is the direct-proxy mode: every model request reaches Trae's
    # native raw transport. Legacy compatibility modes remain explicit
    # diagnostics and are never silent fallbacks for API traffic.
    errors = []
    fallback_reasons: list[str] = []
    modes = []
    if active_mode == "cli":
        modes = ["cli"]
    elif active_mode in ("raw", "direct"):
        modes = ["raw"]
    elif active_mode in ("remote", "9router", "trae-remote"):
        modes = ["remote"]
    elif active_mode == "web":
        modes = ["web"]
    elif active_mode in ("ide", "solo"):
        modes = [active_mode]
    elif active_mode in ("work", "work-agent"):
        modes = ["work"]
    elif active_mode in ("traework-native", "native", "traework"):
        modes = ["traework-native"]
    else:
        modes = ["raw"]

    # Native IDE/Work/Raw paths are useful diagnostics, but the current CN
    # gateway does not expose all three paths for every account.  Keep the
    # selected path first and automatically try the verified Remote transport
    # for both plain and caller-owned-tool requests.  The old implementation
    # excluded tool requests here, which turned a transient native rejection
    # into a guaranteed 502 for Codex clients.
    fallback_enabled = not bool(options.get("_disable_upstream_fallback"))
    fallback_modes = {
        "raw",
        "direct",
        "auto",
        "ide",
        "solo",
        "work", "work-agent",
        "traework-native",
        "native",
        "traework",
    }
    remote_fallback_allowed = (
        fallback_enabled
        and _remote_fallback_allowed_for_request(messages, options)
    )
    if fallback_enabled and active_mode in ("raw", "direct"):
        modes.append("ide")
    if (
        remote_fallback_allowed
        and active_mode in fallback_modes
        and "remote" not in modes
    ):
        modes.append("remote")
    elif (
        fallback_enabled
        and active_mode in fallback_modes
        and "remote" not in modes
        and images.messages_have_images(messages)
    ):
        # Preserve the native endpoint error when the image cannot fit Remote
        # even after history compaction; retrying would only produce a
        # deterministic 413 and may rotate the same payload through accounts.
        logger.info(
            "skip remote fallback for oversized image request id=%s mode=%s",
            str(options.get("_relay_request_id") or ""),
            active_mode,
        )
        if isinstance(trace, dict):
            trace["fallback_skipped"] = "remote_query_limit"
    route_initial_mode = modes[0] if modes else active_mode
    raw_skipped = False
    if (
        fallback_enabled
        and modes[:1] == ["raw"]
        and len(modes) > 1
        and _raw_app_config_known_missing(model, options)
    ):
        skip_detail = "skipped (app config record not found, cached)"
        skip_reason = f"raw: {skip_detail}"
        errors.append(skip_reason)
        fallback_reasons.append(skip_reason)
        _trace_append_failure(trace, "raw", skip_detail)
        raw_skipped = True
        modes = modes[1:]

    logger.info(
        "dispatch route id=%s mode=%s candidates=%s tool_protocol=%s",
        str(options.get("_relay_request_id") or ""),
        active_mode,
        ",".join(modes),
        tool_protocol_requested,
    )

    for mode in modes:
        try:
            mode_options = dict(options)
            if mode == "remote" and modes[0] != "remote":
                mode_options = _remote_fallback_options(
                    mode_options, active_mode
                )
            mode_model = model
            if mode != active_mode:
                # A fallback is a different executor and may have a different
                # model catalogue. Re-evaluate from the caller's original
                # ``auto`` instead of carrying a candidate chosen for the
                # failed endpoint into the fallback request.
                mode_model, mode_options = _apply_task_model_routing(
                    messages, requested_model, mode_options
                )
                if active_tracker is not None:
                    if getattr(active_tracker, "model", mode_model) != mode_model:
                        active_tracker.model = mode_model
                    active_tracker.options = mode_options
            if isinstance(trace, dict):
                route_fallback = (
                    bool(fallback_from)
                    or bool(fallback_reasons)
                    or raw_skipped
                    or mode != route_initial_mode
                )
                trace.update(
                    actual_mode=mode,
                    actual_endpoint=mode,
                    fallback_used=route_fallback,
                    product=(
                        "traework"
                        if mode in {"work", "work-agent", "traework-native", "native", "traework"}
                        else "trae"
                    ),
                )
                if route_fallback:
                    trace["fallback_target"] = (
                        "remote-work" if mode == "work" else mode
                    )
                    if fallback_reasons:
                        trace["fallback_reason"] = "; ".join(fallback_reasons)[:1200]
                trace.pop("reasoning_effort_applied", None)
                trace.pop("max_context_tokens", None)
                if mode in {"raw", "ide", "solo", "cli", "traework-native", "web"}:
                    trace["max_mode_applied"] = False
                else:
                    trace.pop("max_mode_applied", None)
            if mode == "raw":
                result = await run_raw_chat(messages, mode_model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"raw returned {error}")
                if isinstance(trace, dict) and (fallback_reasons or raw_skipped):
                    trace["fallback_target"] = "raw"
                return result
            # Web and remote routes must receive the lease-bound credential;
            # otherwise a concurrent account switch can make the upstream bill
            # one token while the usage tracker records another.
            if mode not in (
                "remote",
                "web",
                "ide",
                "solo",
                "work",
                "work-agent",
                "traework-native",
            ):
                mode_options.pop("_auth_token", None)
                mode_options.pop("_account_id", None)
            if mode == "cli":
                mode_options.pop("_upstream_trace", None)
                result = await run_cli_chat(messages, mode_model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"cli returned {error}")
                return result
            if mode == "web":
                result = await _run_web_with_retry(messages, mode_model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"web returned {error}")
                return result
            if mode == "remote":
                result = await _run_remote_with_retry(messages, mode_model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"remote returned {error}")
                return result
            if mode in ("work", "work-agent"):
                # TraeWork uses the Remote chat_sessions transport, pinned to
                # the Work executor so model binding and billing stay aligned.
                mode_options["_trae_mode"] = "work"
                mode_options["_remote_agent_type"] = "solo_work_remote"
                mode_options["_session_variant"] = "work"
                if isinstance(trace, dict):
                    trace.update(
                        actual_mode="work",
                        actual_endpoint="remote-work",
                        product="traework",
                    )
                result = await _run_remote_with_retry(
                    messages, mode_model, stream, mode_options
                )
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"work returned {error}")
                return result
            if mode == "traework-native":
                result = await run_traework_native_chat(
                    messages, mode_model, stream, mode_options
                )
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"traework-native returned {error}")
                return result
            if mode in ("ide", "solo"):
                # Prefer TraeCode's native raw contract.  Personal CN
                # accounts can lack the raw app-config record, in which case
                # run_ide_chat reopens the verified solo_work_lite adapter.
                # Keep the legacy endpoint visible in options so diagnostics
                # and older callers retain the exact fallback path.
                mode_options["_ide_endpoint"] = "/api/agent/v3/llm_utils_chat"
                mode_options["_legacy_ide_endpoint"] = (
                    "/api/agent/v3/llm_utils_chat"
                )
                mode_options["_traecode_raw_chat_function"] = (
                    "solo_agent" if mode == "solo" else "chat_v3"
                )
                if isinstance(trace, dict):
                    trace["product"] = "trae"
            result = await _run_ide_with_retry(
                messages,
                mode_model,
                stream,
                mode_options,
                tracker=active_tracker,
            )
            if getattr(result, "status_code", 200) >= 400:
                payload = _response_json_payload(result)
                detail = payload.get("error") if isinstance(payload, Mapping) else None
                detail = (
                    detail.get("message")
                    if isinstance(detail, Mapping)
                    else str(detail or "upstream returned an error")
                )
                raise RuntimeError(
                    f"{mode} returned HTTP {getattr(result, 'status_code', 0)}: {detail}"
                )
            return result
        except Exception as e:
            logger.warning("upstream %s failed: %s", mode, e)
            # ``work-agent`` is the public endpoint identity while ``work``
            # is the internal dispatch branch. Keep the public label in
            # diagnostics so a failed Work request is not misreported as a
            # different endpoint.
            diagnostic_mode = (
                active_mode
                if mode == "work" and active_mode in {"work-agent", "work"}
                else mode
            )
            reason = _trace_failure_reason(diagnostic_mode, e)
            fallback_reasons.append(reason)
            # Keep ``failed_endpoint`` tied to the actual dispatch branch
            # (``work``) while the human-facing reason retains the requested
            # public tier (``work-agent``).
            _trace_append_failure(trace, mode, e)
            if diagnostic_mode != mode and isinstance(trace, dict):
                previous = str(trace.get("fallback_reason") or "").strip()
                internal = _trace_failure_reason(mode, e)
                parts = [part.strip() for part in previous.split(";") if part.strip()]
                if internal in parts:
                    parts.remove(internal)
                if reason not in parts:
                    parts.append(reason)
                trace["fallback_reason"] = "; ".join(parts)[:1200]
            if isinstance(e, RemoteQueryTooLarge):
                # The provider rejects oversized flattened queries
                # deterministically. Do not fall through to another executor
                # or replay the same payload on every polling account.
                return _openai_error(
                    413,
                    str(e),
                    "context_length_exceeded",
                    "messages",
                )
            if isinstance(e, _ModelAccountUnavailable):
                return _openai_error(503, str(e), "model_account_unavailable")
            if isinstance(e, EmptyUpstreamResponse):
                _track_usage_from_exception(e, model)
                if not e.retryable or e.observed_model_event:
                    if stream:
                        raise
                    return _openai_error(502, str(e), "api_error")
            errors.append(f"{mode}: {e}")
            if (
                mode == "raw"
                and options.get("_account_id")
                and _raw_app_config_missing_error(e)
            ):
                _RAW_APP_CONFIG_MISSING[_raw_app_config_key(model, options)] = (
                    time.monotonic() + _RAW_APP_CONFIG_TTL_SECONDS
                )

    return _openai_error(502, "All upstream paths failed: " + "; ".join(errors), "api_error")


def _response_json_payload(response: Any) -> dict[str, Any]:
    """Decode a Starlette JSON response without assuming a concrete class."""

    body = getattr(response, "body", b"")
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            value = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}
    return dict(body) if isinstance(body, Mapping) else {}


async def _handle_traework_custom(
    req: Request,
    body: Optional[Mapping[str, Any]] = None,
    *,
    request_id: str = "",
):
    """Serve TraeWork's custom-model/raw ingress and preserve client tools.

    The desktop client owns the toolhost. This endpoint only converts its raw
    request into the existing relay dispatch and converts the OpenAI-shaped
    result back to TraeWork's cumulative ``event: output`` stream. No tool is
    executed in the relay container.
    """

    request_id = request_id or ("req-" + uuid_mod.uuid4().hex)
    if body is None:
        try:
            body, _ = await _read_json_body(
                req, endpoint="traework-custom", trace_id=request_id
            )
        except _RequestBodyError as exc:
            return _openai_error(400, str(exc), "invalid_request_error")
    try:
        descriptor = traework_compat.normalize_inbound_request(
            body,
            headers=dict(req.headers),
            path=req.url.path,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _openai_error(400, str(exc), "invalid_request_error")

    messages = cli_client.sanitize_assistant_history_messages(
        _normalize_chat_messages(descriptor.messages)
    )
    options = dict(descriptor.options)
    options = _apply_tool_header_hints(req, options)
    options = _with_auto_client_context(req, body, messages, options)
    options = _bind_chat_session(
        messages,
        options,
        requested_session_id=descriptor.session_id,
    )
    # These values are deliberately set after session binding. The lease owns
    # the account/JWT; the custom model's Bearer key must never replace it.
    options["_traework_custom_model"] = True
    options["_relay_request_id"] = request_id
    tracker = _UsageTracker(
        descriptor.model,
        req.url.path,
        descriptor.stream,
        options,
        messages,
    )
    tracker.request_id = request_id
    logger.info(
        "traework custom ingress id=%s path=%s model=%s stream=%s messages=%d tools=%s session=%s",
        request_id,
        req.url.path,
        descriptor.model,
        descriptor.stream,
        len(messages),
        _tool_protocol_requested(options, messages),
        descriptor.session_id[:32],
    )

    if descriptor.stream:
        async def openai_source():
            async for chunk in _deferred_dispatch_stream(
                messages, descriptor.model, options
            ):
                yield chunk

        translated = traework_compat.translate_openai_stream_to_traework(
            openai_source(), model=descriptor.model, request_id=request_id
        )
        session_id = str(options.get("session_id") or descriptor.session_id)
        return StreamingResponse(
            _tracked_stream(_lease_stream(translated, session_id), tracker),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    response = await _tracked_dispatch(messages, descriptor.model, options, tracker)
    if getattr(response, "status_code", 200) >= 400:
        payload = _response_json_payload(response)
        return JSONResponse(
            traework_compat.openai_error_to_traework(
                payload,
                model=descriptor.model,
                request_id=request_id,
            ),
            status_code=getattr(response, "status_code", 502),
        )
    payload = _response_json_payload(response)
    return JSONResponse(
        traework_compat.openai_completion_to_traework(
            payload, model=descriptor.model
        )
    )


async def handle_chat(req: Request):
    request_id = "req-" + uuid_mod.uuid4().hex
    try:
        body, _body_bytes = await _read_json_body(
            req,
            endpoint="chat",
            trace_id=request_id,
        )
    except _RequestBodyError as exc:
        logger.warning(
            "request body rejected id=%s endpoint=chat bytes=%d error=%s",
            request_id,
            exc.raw_bytes,
            exc,
        )
        return _openai_error(400, str(exc), "invalid_request_error")

    # A TraeWork custom model may use the ordinary /v1/chat/completions URL,
    # but native raw paths and the client's identifying headers are distinct
    # enough to route it without changing OpenAI callers.
    if traework_compat.is_traework_request(
        req.url.path, dict(req.headers), body
    ):
        return await _handle_traework_custom(
            req, body=body, request_id=request_id
        )

    messages = _normalize_chat_messages(body.get("messages"))
    if not isinstance(messages, list) or not messages:
        # Accept the lightweight aliases emitted by a few terminal adapters.
        # This keeps a valid user prompt from being mistaken for an empty
        # request when the adapter does not use the Chat Completions field name.
        for alias in ("prompt", "query", "input_text", "content"):
            candidate = body.get(alias)
            if candidate not in (None, "", [], {}):
                messages = [{"role": "user", "content": candidate}]
                break
        if messages:
            options_from_response = {}
        else:
            # Some OpenAI-compatible clients accidentally send a Responses-shaped
            # payload to /chat/completions. Normalize it instead of dropping the
            # user prompt before the request can reach Trae.
            if "input" in body:
                try:
                    messages, response_options, _response_context = responses_api.normalize_request(body)
                    options_from_response = dict(response_options)
                except responses_api.ResponsesRequestError as exc:
                    return _openai_error(400, str(exc), "invalid_request_error", exc.param)
            else:
                return _openai_error(400, "messages is required", "invalid_request_error")
    else:
        options_from_response = {}

    messages = cli_client.sanitize_assistant_history_messages(messages)

    model = body.get("model") or "auto"
    stream = bool(body.get("stream", False))
    options = {
        key: body[key]
        for key in CHAT_OPTION_FIELDS
        if key in body
    }
    options.update(options_from_response)
    options = _apply_tool_header_hints(req, options)
    requested_session_id = _request_session_hint(req, body)
    if requested_session_id and not options.get("session_id") and not options.get("sessionId"):
        options["session_id"] = requested_session_id

    option_error = _validate_chat_options(options)
    if option_error is not None:
        return option_error
    for key in ("max_tokens", "maxTokens", "max_completion_tokens"):
        if key in options:
            options[key] = clamp_max_completion_tokens(options[key], model)
    options = _with_auto_client_context(req, body, messages, options)
    options = _bind_chat_session(
        messages,
        options,
        requested_session_id=requested_session_id,
    )
    tracker = _UsageTracker(model, req.url.path, stream, options, messages)
    tracker.request_id = request_id
    options["_relay_request_id"] = request_id
    logger.info(
        "request received id=%s path=%s keys=%s messages=%d input_chars=%d stream=%s",
        tracker.request_id,
        req.url.path,
        ",".join(sorted(str(key) for key in body.keys())),
        len(messages),
        sum(len(str(item.get("content") or "")) for item in messages if isinstance(item, Mapping)),
        stream,
    )
    logger.info(
        "request binding id=%s account=%s billing=%s session=%s",
        tracker.request_id,
        str(options.get("_account_id") or "default"),
        str(options.get("_billing_id") or options.get("_account_id") or "default"),
        str(options.get("session_id") or options.get("sessionId") or "")[:32],
    )
    if stream:
        session_id = str(options.get("session_id") or options.get("sessionId") or "")
        return StreamingResponse(
            _tracked_stream(
                _lease_stream(
                    _deferred_dispatch_stream(messages, model, options),
                    session_id,
                ),
                tracker,
            ),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )
    return await _tracked_dispatch(messages, model, options, tracker)


async def handle_responses(req: Request):
    request_id = "req-" + uuid_mod.uuid4().hex
    try:
        body, _body_bytes = await _read_json_body(
            req,
            endpoint="responses",
            trace_id=request_id,
        )
    except _RequestBodyError as exc:
        logger.warning(
            "request body rejected id=%s endpoint=responses bytes=%d error=%s",
            request_id,
            exc.raw_bytes,
            exc,
        )
        return _openai_error(400, str(exc), "invalid_request_error")

    try:
        messages, options, context = responses_api.normalize_request(body)
    except responses_api.ResponsesRequestError as exc:
        return _openai_error(
            400, str(exc), "invalid_request_error", exc.param
        )

    messages = cli_client.sanitize_assistant_history_messages(messages)

    options = _apply_tool_header_hints(req, options)
    option_error = _validate_chat_options(options)
    if option_error is not None:
        return option_error
    if "max_tokens" in options:
        options["max_tokens"] = clamp_max_completion_tokens(
            options["max_tokens"], context.model
        )
    options = _with_auto_client_context(req, body, messages, options)
    # Responses response ids change on every turn.  The context retains the
    # first raw session id so multi-tool continuations stay in one upstream
    # Trae conversation even when the caller sends only function_call_output.
    if not options.get("session_id") and not options.get("sessionId"):
        options["session_id"] = context.upstream_session_id or context.response_id
    options = _bind_chat_session(
        messages,
        options,
        requested_session_id=str(options.get("session_id") or options.get("sessionId") or ""),
    )
    stream = bool(body.get("stream", False))
    tracker = _UsageTracker(context.model, req.url.path, stream, options, messages)
    tracker.request_id = request_id
    options["_relay_request_id"] = request_id
    logger.info(
        "request received id=%s path=%s keys=%s messages=%d input_chars=%d stream=%s",
        tracker.request_id,
        req.url.path,
        ",".join(sorted(str(key) for key in body.keys())),
        len(messages),
        sum(len(str(item.get("content") or "")) for item in messages if isinstance(item, Mapping)),
        stream,
    )
    logger.info(
        "request binding id=%s account=%s billing=%s session=%s",
        tracker.request_id,
        str(options.get("_account_id") or "default"),
        str(options.get("_billing_id") or options.get("_account_id") or "default"),
        str(options.get("session_id") or options.get("sessionId") or "")[:32],
    )
    if stream:
        session_id = str(options.get("session_id") or options.get("sessionId") or "")
        return StreamingResponse(
            _tracked_stream(
                _lease_stream(
                    responses_api.translate_chat_stream(
                        _deferred_dispatch_stream(messages, context.model, options),
                        context,
                    ),
                    session_id,
                ),
                tracker,
            ),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )
    chat_response = await _tracked_dispatch(
        messages, context.model, options, tracker
    )
    if getattr(chat_response, "status_code", 200) >= 400:
        return chat_response
    try:
        completion = json.loads(chat_response.body)
    except Exception:
        return _openai_error(
            502, "Invalid Chat Completions response", "api_error"
        )
    return JSONResponse(
        content=responses_api.completion_to_response(completion, context)
    )


async def handle_anthropic_messages(req: Request):
    """Anthropic Messages API ingress for Claude Code and compatible clients."""

    request_id = "req-" + uuid_mod.uuid4().hex
    try:
        body, _body_bytes = await _read_json_body(
            req,
            endpoint="messages",
            trace_id=request_id,
        )
    except _RequestBodyError as exc:
        logger.warning(
            "request body rejected id=%s endpoint=messages bytes=%d error=%s",
            request_id,
            exc.raw_bytes,
            exc,
        )
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )
    try:
        messages, options = anthropic_api.normalize_request(body)
    except anthropic_api.AnthropicRequestError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    messages = cli_client.sanitize_assistant_history_messages(messages)
    model = str(body.get("model") or "auto")
    options = _apply_tool_header_hints(req, options)
    option_error = _validate_chat_options(options)
    if option_error is not None:
        return option_error
    for key in ("max_tokens", "maxTokens", "max_completion_tokens"):
        if key in options:
            options[key] = clamp_max_completion_tokens(options[key], model)
    options = _with_auto_client_context(req, body, messages, options)
    options = _bind_chat_session(
        messages,
        options,
        requested_session_id=_request_session_hint(req, body),
    )
    stream = bool(body.get("stream", False))
    tracker = _UsageTracker(model, req.url.path, stream, options, messages)
    tracker.request_id = request_id
    options["_relay_request_id"] = request_id
    input_tokens = anthropic_api.estimate_input_tokens(
        messages, body.get("system"), options.get("tools")
    )
    logger.info(
        "request received id=%s path=%s keys=%s messages=%d input_chars=%d stream=%s",
        request_id,
        req.url.path,
        ",".join(sorted(str(key) for key in body.keys())),
        len(messages),
        sum(
            len(str(item.get("content") or ""))
            for item in messages
            if isinstance(item, Mapping)
        ),
        stream,
    )
    logger.info(
        "request binding id=%s account=%s billing=%s session=%s",
        request_id,
        str(options.get("_account_id") or "default"),
        str(options.get("_billing_id") or options.get("_account_id") or "default"),
        str(options.get("session_id") or options.get("sessionId") or "")[:32],
    )
    if stream:
        session_id = str(options.get("session_id") or options.get("sessionId") or "")
        return StreamingResponse(
            _tracked_stream(
                _lease_stream(
                    anthropic_api.translate_chat_stream(
                        _deferred_dispatch_stream(messages, model, options),
                        model,
                        request_input_tokens=input_tokens,
                    ),
                    session_id,
                ),
                tracker,
            ),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )
    chat_response = await _tracked_dispatch(messages, model, options, tracker)
    if getattr(chat_response, "status_code", 200) >= 400:
        detail = _upstream_response_error(chat_response) or "Upstream request failed"
        return JSONResponse(
            status_code=getattr(chat_response, "status_code", 502),
            content={
                "type": "error",
                "error": {"type": "api_error", "message": detail},
            },
        )
    try:
        completion = json.loads(chat_response.body)
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Invalid Chat Completions response",
                },
            },
        )
    try:
        message = anthropic_api.completion_to_message(
            completion, model, request_input_tokens=input_tokens
        )
    except anthropic_api.AnthropicRequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            },
        )
    return JSONResponse(content=message)


async def handle_anthropic_count_tokens(req: Request):
    """Anthropic-compatible token count estimate; this endpoint does not bill."""

    request_id = "req-" + uuid_mod.uuid4().hex
    try:
        body, _body_bytes = await _read_json_body(
            req,
            endpoint="messages/count_tokens",
            trace_id=request_id,
        )
    except _RequestBodyError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )
    try:
        messages, options = anthropic_api.normalize_request(body)
    except anthropic_api.AnthropicRequestError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )
    return JSONResponse(
        content={
            "input_tokens": anthropic_api.estimate_input_tokens(
                messages, body.get("system"), options.get("tools")
            )
        }
    )


# Keys that ship as examples or are trivially guessable.  ``.env.example``
# ships ``your-master-key``; a plain ``cp .env.example .env`` must not turn
# into a reachable relay protected by a value anyone can read from the repo.
_PLACEHOLDER_API_KEYS = {
    "",
    "your-master-key",
    "your-key",
    "changeme",
    "change-me",
    "master-key",
    "key",
    "key1",
    "test",
    "test-key",
    "sk-xxx",
}


def _placeholder_api_keys() -> list[str]:
    return [key for key in API_KEYS if key.strip().lower() in _PLACEHOLDER_API_KEYS]


def _is_lan_reachable() -> bool:
    """True when the configured host accepts connections beyond loopback."""

    host = str(HOST or "").strip().lower()
    if host in ("0.0.0.0", "::", "*"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not (address.is_loopback or address.is_unspecified)


def _warn_insecure_lan_key() -> None:
    """Warn loudly when a non-loopback bind has no real API key.

    The relay spends the signed-in account's own upstream quota and, by
    design, leaves the management endpoints public to private networks.  That
    is acceptable on a trusted LAN only if the model endpoints are protected,
    so a blank or placeholder key on a reachable host is a deployment mistake
    worth surfacing at startup rather than discovering after the quota is gone.
    """

    if not _is_lan_reachable():
        return
    placeholders = _placeholder_api_keys()
    if not API_KEYS:
        logger.warning(
            "LAN mode without RELAY_API_KEYS: %s is reachable from the network "
            "and /v1 is unprotected. Set RELAY_API_KEYS (or start with "
            "./start_lan.sh, which mints one) before exposing this relay.",
            HOST,
        )
        return
    if placeholders:
        logger.warning(
            "LAN mode with placeholder API key(s) %s: /v1 on %s is protected by "
            "a guessable value. Replace RELAY_API_KEYS with a random key "
            "(./start_lan.sh generates and persists one).",
            ", ".join(repr(key) for key in placeholders),
            HOST,
        )


async def init_app():
    _load_usage_history()
    _load_usage_stats()
    auth.init_auth()
    _restore_persisted_relay_settings()
    auth.apply_max_mode_settings()
    _schedule_pending_usage_enrichment()
    _warn_insecure_lan_key()
    logger.info(
        "Trae CN relay initialized (auth source=%s edition=%s cli=%s)",
        auth.get_auth().source,
        auth.get_auth().edition,
        cli_client.resolve_cli_command() or "not-found",
    )


async def _usage_enrichment_loop():
    """Retry bill lookups that were still settling after a model response.

    The model response must stay independent from the commercial accounting
    endpoint.  A short background pass keeps the API fast while preventing a
    transient empty usage response from becoming a permanently missing credit
    value in the console.
    """

    interval = _usage_enrichment_interval_seconds()
    while True:
        await asyncio.sleep(interval)
        try:
            _schedule_pending_usage_enrichment()
        except Exception as exc:
            logger.debug("usage enrichment scheduler error: %s", exc)


async def _credit_reconcile_cycle(trigger: str = "schedule") -> dict[str, Any]:
    """Refresh total credits for every account and keep the local overlay.

    Account-level check-in locks serialize this with manual credit refreshes.
    A failed upstream query leaves the last known balance intact and only marks
    the in-memory status stale.
    """

    summary = {"total": 0, "ok": 0, "failed": 0, "skipped": 0}
    for account_id, record in list(auth.get_accounts_raw()):
        summary["total"] += 1
        if not (record.get("token") or ""):
            summary["skipped"] += 1
            continue
        try:
            async with _checkin_account_lock(account_id):
                await _fetch_credit_account_snapshot(account_id, record)
            _CREDIT_RECONCILE_STATE[account_id] = {
                "status": "ok",
                "at": time.time(),
                "trigger": trigger,
            }
            summary["ok"] += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Do not overwrite the cached balance with a zero/empty response.
            auth.mark_account_credit_reconcile_stale(account_id)
            _CREDIT_RECONCILE_STATE[account_id] = {
                "status": "stale",
                "at": time.time(),
                "trigger": trigger,
                "error": str(exc)[:160],
            }
            summary["failed"] += 1
            logger.debug("credit reconciliation failed account=%s: %s", account_id, exc)
    return summary


async def _credit_reconcile_loop():
    """Periodically calibrate the dashboard's aggregate total credits."""

    try:
        initial_delay = float(
            os.environ.get("TRAE_CREDIT_RECONCILE_INITIAL_DELAY_SECONDS", "2")
            or "2"
        )
    except (TypeError, ValueError):
        initial_delay = 2.0
    if initial_delay > 0:
        await asyncio.sleep(min(initial_delay, 60.0))
    while True:
        try:
            await _credit_reconcile_cycle("schedule")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("credit reconciliation cycle error: %s", exc)
        await asyncio.sleep(_credit_reconcile_interval_seconds())


async def _web_reaper_loop():
    interval = float(os.environ.get("TRAE_WEB_REAP_INTERVAL", "10"))
    while True:
        await asyncio.sleep(interval)
        try:
            await trae_client.reap_idle_web_sessions()
        except Exception as e:
            logger.warning("web reaper error: %s", e)


async def _terminal_session_reaper_loop():
    try:
        interval = float(os.environ.get("TRAE_SESSION_REAP_INTERVAL_SECONDS", "5"))
    except (TypeError, ValueError):
        interval = 5.0
    interval = max(1.0, min(interval, 60.0))
    while True:
        await asyncio.sleep(interval)
        try:
            reaped = await _reap_idle_chat_sessions()
            if reaped:
                logger.info("reaped %d idle terminal session lease(s)", reaped)
        except Exception as e:
            logger.warning("terminal session reaper error: %s", e)


async def _checkin_auto_retry_cycle() -> None:
    """Retry accounts whose persisted 9074 backoff has expired."""
    try:
        for account_id, record in auth.get_accounts_raw():
            checkin = record.get("checkin") or {}
            if checkin.get("checked_in") is True and _checkin_cache_is_today(record):
                if float(checkin.get("retry_backoff") or 0) > 0:
                    _checkin_clear_retry_state(account_id)
                continue
            if not (record.get("token") or ""):
                continue
            # Only accounts that actually hit 9074 belong here; this loop must
            # not turn into a general claim poller. ``device_generation``
            # survives a retry-state reset, so an account stays visible after
            # its backoff is cleared but before it has claimed.
            pending_9074 = (
                float(checkin.get("retry_backoff") or 0) > 0
                or int(checkin.get("retry_9074_count") or 0) > 0
                or int(checkin.get("device_generation") or 0) > 0
            )
            if not pending_9074:
                continue
            cooldown = _checkin_cooldown_remaining(account_id)
            if cooldown:
                continue
            try:
                result = await _claim_checkin_account(account_id)
            except Exception as exc:
                logger.warning(
                    "checkin auto retry account=%s error: %s",
                    account_id,
                    exc,
                )
                continue
            if result.get("success"):
                logger.info(
                    "checkin auto retry ok account=%s skipped=%s",
                    account_id,
                    result.get("skipped"),
                )
            else:
                logger.info(
                    "checkin auto retry pending account=%s retry_after=%s",
                    account_id,
                    result.get("retry_after_seconds"),
                )
    except Exception as exc:
        logger.warning("checkin auto retry cycle error: %s", exc)


async def _checkin_auto_retry_loop():
    """Periodically retry accounts stuck in an upstream 9074 window."""
    interval = max(15.0, float(CHECKIN_AUTO_RETRY_INTERVAL))
    while True:
        await asyncio.sleep(interval)
        await _checkin_auto_retry_cycle()


# Scheduled daily check-in.  The loop wakes every few seconds, re-reads the
# console settings and fires once per business day (UTC+8) after the chosen
# time.  A relay that starts late still claims that same day.
AUTO_CHECKIN_POLL_SECONDS = 30.0
_AUTO_CHECKIN_STATE: dict = {
    "running": False,
    "last_run_at": 0.0,
    "last_run_date": "",
    "last_trigger": "",
    "summary": None,
}
_AUTO_CHECKIN_LOCK = asyncio.Lock()


def _auto_checkin_target(now: Optional[datetime] = None, time_text: str = "08:30") -> datetime:
    """Return today's scheduled run time in the check-in timezone."""
    current = now or datetime.now(_CHECKIN_TIMEZONE)
    hour, minute = (int(part) for part in time_text.split(":", 1))
    return current.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _auto_checkin_next_run(settings: dict, now: Optional[datetime] = None) -> Optional[datetime]:
    if not settings.get("enabled"):
        return None
    current = now or datetime.now(_CHECKIN_TIMEZONE)
    target = _auto_checkin_target(current, settings.get("time") or "08:30")
    if _AUTO_CHECKIN_STATE.get("last_run_date") == current.date().isoformat():
        return target + timedelta(days=1)
    # Past today's time but not run yet: the loop fires on its next tick.
    return max(target, current)


def _auto_checkin_due(settings: dict, now: Optional[datetime] = None) -> bool:
    if not settings.get("enabled"):
        return False
    current = now or datetime.now(_CHECKIN_TIMEZONE)
    if _AUTO_CHECKIN_STATE.get("last_run_date") == current.date().isoformat():
        return False
    return current >= _auto_checkin_target(current, settings.get("time") or "08:30")


async def _auto_checkin_cycle(trigger: str = "schedule") -> dict:
    """Claim check-in for every account with a token, strictly in order."""
    summary = {"total": 0, "ok": 0, "skipped": 0, "failed": 0, "no_token": 0}
    if _AUTO_CHECKIN_LOCK.locked():
        return {**summary, "busy": True}
    async with _AUTO_CHECKIN_LOCK:
        _AUTO_CHECKIN_STATE["running"] = True
        now = datetime.now(_CHECKIN_TIMEZONE)
        try:
            for account_id, record in list(auth.get_accounts_raw()):
                summary["total"] += 1
                if not (record.get("token") or ""):
                    summary["no_token"] += 1
                    continue
                checkin = record.get("checkin") or {}
                if checkin.get("checked_in") is True and _checkin_cache_is_today(record):
                    summary["skipped"] += 1
                    continue
                try:
                    result = await _claim_checkin_account(account_id)
                except Exception as exc:
                    logger.warning("auto checkin account=%s error: %s", account_id, exc)
                    summary["failed"] += 1
                    continue
                if result.get("success"):
                    summary["skipped" if result.get("skipped") else "ok"] += 1
                else:
                    summary["failed"] += 1
        finally:
            _AUTO_CHECKIN_STATE.update(
                running=False,
                last_run_at=time.time(),
                last_trigger=trigger,
                summary=dict(summary),
            )
            # Only the scheduler consumes the daily slot; a manual run leaves
            # the scheduled pass in place (already claimed accounts are skipped).
            if trigger == "schedule":
                _AUTO_CHECKIN_STATE["last_run_date"] = now.date().isoformat()
        logger.info(
            "auto checkin done trigger=%s total=%d ok=%d skipped=%d failed=%d no_token=%d",
            trigger,
            summary["total"],
            summary["ok"],
            summary["skipped"],
            summary["failed"],
            summary["no_token"],
        )
        return summary


async def _auto_checkin_loop():
    while True:
        await asyncio.sleep(AUTO_CHECKIN_POLL_SECONDS)
        try:
            if _auto_checkin_due(auth.get_auto_checkin_settings()):
                await _auto_checkin_cycle("schedule")
        except Exception as exc:
            logger.warning("auto checkin loop error: %s", exc)


def _auto_checkin_payload() -> dict:
    settings = auth.get_auto_checkin_settings()
    next_run = _auto_checkin_next_run(settings)
    last_at = float(_AUTO_CHECKIN_STATE.get("last_run_at") or 0)
    return {
        "success": True,
        **settings,
        "timezone": "Asia/Shanghai",
        "next_run": next_run.isoformat(timespec="minutes") if next_run else None,
        "running": bool(_AUTO_CHECKIN_STATE.get("running")),
        "last_run_at": (
            datetime.fromtimestamp(last_at, _CHECKIN_TIMEZONE).isoformat(timespec="seconds")
            if last_at > 0
            else None
        ),
        "last_trigger": _AUTO_CHECKIN_STATE.get("last_trigger") or None,
        "summary": _AUTO_CHECKIN_STATE.get("summary"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_app()
    web_reaper = asyncio.create_task(_web_reaper_loop())
    terminal_reaper = asyncio.create_task(_terminal_session_reaper_loop())
    checkin_retry = asyncio.create_task(_checkin_auto_retry_loop())
    auto_checkin = asyncio.create_task(_auto_checkin_loop())
    usage_enrichment = asyncio.create_task(_usage_enrichment_loop())
    credit_reconcile = asyncio.create_task(_credit_reconcile_loop())
    try:
        yield
    finally:
        for reaper in (
            web_reaper,
            terminal_reaper,
            checkin_retry,
            auto_checkin,
            usage_enrichment,
            credit_reconcile,
        ):
            reaper.cancel()
        await asyncio.gather(
            web_reaper,
            terminal_reaper,
            checkin_retry,
            auto_checkin,
            usage_enrichment,
            credit_reconcile,
            return_exceptions=True,
        )
        await _cancel_usage_tasks()


app = FastAPI(title="Trae CN Relay", version=APP_VERSION, lifespan=lifespan)


@app.exception_handler(_ModelAccountUnavailable)
async def model_account_unavailable_handler(_request: Request, exc: _ModelAccountUnavailable):
    return _openai_error(503, str(exc), "model_account_unavailable")


@app.get("/api/overview")
async def api_overview(days: int = Query(30, ge=1, le=366)):
    """Dashboard home: token totals by day plus account and service state."""
    accounts = auth.list_accounts()
    remaining = 0.0
    official_remaining = 0.0
    local_debit = 0.0
    credits_known = False
    unlimited = False
    for account in accounts:
        credits = account.get("account_credits") or {}
        if credits.get("unlimited"):
            unlimited = True
        elif credits.get("remaining") is not None:
            try:
                remaining += float(credits["remaining"])
                credits_known = True
            except (TypeError, ValueError):
                pass
        try:
            official = credits.get(
                "official_remaining", credits.get("remaining")
            )
            if official is not None:
                official_remaining += float(official)
            local_debit += float(credits.get("local_debit") or 0)
        except (TypeError, ValueError):
            pass
    polling = auth.get_polling_status()
    reconcile_statuses = {
        str(account.get("credits_reconcile_status") or "").strip().lower()
        for account in accounts
    }
    if reconcile_statuses.intersection({"stale", "error"}):
        aggregate_reconcile_status = "stale"
    elif "projected" in reconcile_statuses:
        aggregate_reconcile_status = "projected"
    else:
        aggregate_reconcile_status = "ok"
    with _USAGE_LOCK:
        in_flight = sum(_USAGE_ACTIVE_ACCOUNTS.values())
        recent = [_public_usage_record(record) for record in _USAGE_HISTORY[:5]]
    return JSONResponse(
        {
            "success": True,
            "generated_at": time.time(),
            "timezone": "Asia/Shanghai",
            "usage": _usage_overview(days),
            "accounts": {
                "total": len(accounts),
                "valid": sum(1 for account in accounts if account.get("is_valid")),
                "checked_in": sum(1 for account in accounts if account.get("checked_in") is True),
                "credits_remaining": (
                    "unlimited" if unlimited else round(remaining, 2) if credits_known else None
                ),
                "credits_official_remaining": (
                    "unlimited"
                    if unlimited
                    else round(official_remaining, 2) if credits_known else None
                ),
                "credits_local_debit": round(local_debit, 4),
                "credits_reconcile_status": aggregate_reconcile_status,
                "credits_reconciled_at": max(
                    (
                        float(account.get("credits_reconciled_at") or 0)
                        for account in accounts
                    ),
                    default=0,
                ),
                "credits_reconcile_confidence": min(
                    (
                        float(account.get("credits_reconcile_confidence"))
                        for account in accounts
                        if account.get("credits_reconcile_confidence") is not None
                    ),
                    default=None,
                ),
                "active": auth.get_active_account_id() or None,
            },
            "service": {
                "version": APP_VERSION,
                "uptime_seconds": round(time.time() - _PROCESS_STARTED_AT),
                "upstream_mode": _current_upstream_mode(),
                "auto_route": _auto_route_enabled(),
                "polling": bool(polling.get("enabled")),
                "polling_mode": polling.get("mode"),
                "max_mode": bool(auth.get_max_mode_settings().get("enabled")),
                "auto_checkin": auth.get_auto_checkin_settings(),
                "in_flight": in_flight,
                "token_ok": auth.get_auth().is_valid(),
            },
            "recent": recent,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/usage/records")
async def get_usage_records():
    """Return the usage history list (newest first)."""
    with _USAGE_LOCK:
        records = [_public_usage_record(record) for record in _USAGE_HISTORY]
    return JSONResponse(records, headers={"Cache-Control": "no-store"})

@app.get("/api/usage/last")
async def api_usage_last():
    """Backward-compatible: return only the latest record."""
    with _USAGE_LOCK:
        records = [_public_usage_record(record) for record in _USAGE_HISTORY[:1]]
    return JSONResponse(records, headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not API_KEYS:
        return await call_next(request)
    if request.url.path == "/api/model-test":
        # Keep the dashboard probe usable on loopback/RFC1918 management
        # networks without exposing a billable model endpoint to the public.
        client_host = str(getattr(request.client, "host", "") or "").strip()
        try:
            client_ip = ipaddress.ip_address(client_host)
        except ValueError:
            client_ip = None
        if client_ip is not None and (
            client_ip.is_loopback
            or client_ip.is_private
            or client_ip.is_link_local
        ):
            return await call_next(request)
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith(PUBLIC_PATH_PREFIXES):
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
    else:
        token = header.strip()
    if not token:
        # Anthropic clients (Claude Code) authenticate with ``x-api-key``
        # instead of a bearer Authorization header.
        token = request.headers.get("x-api-key", "").strip()
    if token not in API_KEYS:
        return _openai_error(401, "Invalid API key", "authentication_error")
    return await call_next(request)


@app.get("/")
async def root():
    return {"service": "trae-cn-relay", "status": "ok"}


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(
        LOGO_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/status")
async def status():
    state = auth.get_auth()
    return {
        "status": "ok",
        "edition": state.edition,
        "source": state.source,
        "base_url": state.host,
        "web_base": auth.get_settings().get("web_base_url") or WEB_BASE,
        "upstream_mode": _current_upstream_mode(),
        "auto_route": _auto_route_enabled(),
        "build_revision": os.environ.get("RELAY_BUILD_REVISION", "local"),
        "traework_native": {
            "enabled": traework_native_bridge.NativeBridgeConfig.from_env().enabled,
            "platform_supported": traework_native_bridge.NativeBridgeConfig.from_env().enabled_for_platform,
            "install_dir_configured": bool(
                traework_native_bridge.NativeBridgeConfig.from_env().install_dir
            ),
            "helper_url": traework_native_bridge.NativeBridgeConfig.from_env().bridge_url,
        },
        "tool_execution": "client",
        "capabilities": {
            "openai_tool_calls": True,
            "openai_responses": True,
            "responses_custom_tools": True,
            "responses_namespaces": True,
            "client_context": True,
            "tool_result_continuation": True,
            "traework_custom_model": True,
            "traework_raw_ingress": True,
            "traework_connectivity_check": True,
            "parallel_tool_calls": True,
            "server_executes_caller_tools": False,
            "tool_upstreams": (
                ["raw"]
                if _current_upstream_mode() in ("raw", "direct", "auto")
                else ["cli"]
                if _current_upstream_mode() == "cli"
                else ["traework-native"]
                if _current_upstream_mode() in ("traework-native", "native", "traework")
                else []
            ),
            "terminal_session_leases": True,
        },
        "has_token": bool(state.token),
        "token_ok": state.is_valid(),
        "session_idle_timeout_seconds": _CHAT_SESSION_TTL,
        "port": PORT,
        "cli": cli_client.get_cli_status(),
    }


@app.get("/v1/models")
async def models(request: Request):
    force = request.query_params.get("refresh", "").lower() in ("1", "true", "yes")
    try:
        items = await trae_client.get_models(force=force)
    except Exception as e:
        return _openai_error(502, str(e), "api_error")
    return {"object": "list", "data": items}


@app.get("/models")
async def models_compat(request: Request):
    """Unversioned discovery alias used by some TraeWork model forms."""

    return await models(request)


@app.post("/v1")
async def chat_v1(req: Request):
    return await handle_chat(req)


@app.post("/v1/chat")
async def chat_v1_chat(req: Request):
    return await handle_chat(req)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    return await handle_chat(req)


@app.post("/chat/completions")
async def chat_completions_compat(req: Request):
    """OpenAI-compatible alias for custom-model base URLs without ``/v1``."""

    return await handle_chat(req)


@app.post("/api/ide/v2/llm_raw_chat")
@app.post("/api/agent/v3/llm_raw_chat_custom_model")
@app.post("/api/agent/v3/custom_model_proxy/chat")
@app.post("/v1/traework/chat")
async def traework_custom_chat(req: Request):
    return await _handle_traework_custom(req)


@app.post("/api/agent/v3/custom_model_connectivity_check")
@app.post("/api/ide/v1/custom_model_connectivity_check")
@app.post("/v1/custom_model/connectivity")
async def traework_custom_connectivity(req: Request):
    request_id = "req-" + uuid_mod.uuid4().hex
    try:
        body, _ = await _read_json_body(
            req, endpoint="traework-connectivity", trace_id=request_id
        )
    except _RequestBodyError:
        # Several TraeWork builds send an empty connectivity POST and expect
        # the backend to report its default model. Keep that probe useful.
        body = {}
    try:
        upstream_models = await trae_client.get_models(force=False)
    except Exception as exc:
        logger.warning("traework connectivity model list failed: %s", exc)
        upstream_models = []
    return JSONResponse(
        traework_compat.connectivity_response(body, models=upstream_models)
    )


@app.post("/v1/responses")
async def responses(req: Request):
    return await handle_responses(req)


@app.post("/v1/messages")
async def anthropic_messages(req: Request):
    return await handle_anthropic_messages(req)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(req: Request):
    return await handle_anthropic_count_tokens(req)


@app.post("/api/model-test")
async def api_model_test(req: Request):
    """Probe one model end to end through the real dispatch path.

    Runs the same routing a client request would, so the result reflects actual
    connectivity rather than a config lookup. ``mode`` selects a plain text
    probe or a tool-call probe.
    """

    try:
        body = await req.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    model = str(body.get("model") or "").strip()
    if not model:
        return JSONResponse(
            {"success": False, "error": "model is required"}, status_code=400
        )
    mode = str(body.get("mode") or "text").strip().lower()
    if mode not in {"text", "tool"}:
        return JSONResponse(
            {"success": False, "error": "mode must be text or tool"}, status_code=400
        )
    try:
        timeout = float(body.get("timeout") or 120)
    except (TypeError, ValueError):
        timeout = 120.0
    timeout = max(10.0, min(timeout, 600.0))
    endpoint = str(body.get("endpoint") or _current_upstream_mode()).strip().lower()
    if endpoint not in set(_VALID_UPSTREAM_MODES) | {AUTO_ROUTE_MODE}:
        return JSONResponse(
            {"success": False, "error": f"unsupported endpoint: {endpoint}"},
            status_code=400,
        )

    probe_tools = [
        {
            "type": "function",
            "function": {
                "name": "relay_probe",
                "description": "Echo a probe token back to the relay",
                "parameters": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
            },
        }
    ]
    if mode == "tool":
        messages = [
            {
                "role": "user",
                "content": (
                    "Call relay_probe with token=\"pong\". Tool call only, "
                    "no explanation."
                ),
            }
        ]
        options: dict[str, Any] = {
            "tools": probe_tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
    else:
        messages = [{"role": "user", "content": "Reply with exactly: pong"}]
        options = {}
    options["max_tokens"] = 64
    effort = str(body.get("reasoning_effort") or "").strip().lower()
    if effort:
        if effort not in {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            return JSONResponse(
                {"success": False, "error": f"unsupported reasoning_effort: {effort}"},
                status_code=400,
            )
        options["reasoning_effort"] = effort
        # Effort only matters when the model actually thinks; give it room.
        options["max_tokens"] = 4096
    if bool(body.get("thinking")):
        options["thinking"] = {"type": "enabled"}
    max_mode_requested = bool(body.get("max_mode"))
    if max_mode_requested:
        options["trae_max_mode"] = True
    trace: dict[str, Any] = {}
    options["_upstream_mode"] = endpoint
    # Connectivity checks should exercise the same resilient route as a real
    # API request.  Operators can still pass ``disable_fallback=true`` when
    # diagnosing one native endpoint in isolation.
    options["_disable_upstream_fallback"] = bool(body.get("disable_fallback", False))
    options["_upstream_trace"] = trace

    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            _dispatch_chat(messages, model, False, options),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            {
                "success": False,
                "model": model,
                "mode": mode,
                "requested_endpoint": endpoint,
                **trace,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "error": f"timed out after {int(timeout)}s",
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "success": False,
                "model": model,
                "mode": mode,
                "requested_endpoint": endpoint,
                **trace,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "error": str(exc)[:400],
            }
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    payload: dict[str, Any] = {}
    http_status = 200
    if isinstance(result, JSONResponse):
        http_status = result.status_code
        try:
            payload = json.loads(bytes(result.body).decode("utf-8"))
        except Exception:
            payload = {}
    elif isinstance(result, Mapping):
        payload = dict(result)
    if not isinstance(payload, dict):
        payload = {}
    upstream_error = payload.get("error")
    if http_status >= 400 or isinstance(upstream_error, Mapping):
        message = ""
        if isinstance(upstream_error, Mapping):
            message = str(upstream_error.get("message") or "")
        return JSONResponse(
            {
                "success": False,
                "model": model,
                "mode": mode,
                "requested_endpoint": endpoint,
                **trace,
                "elapsed_ms": elapsed_ms,
                "error": (message or f"upstream returned HTTP {http_status}")[:400],
            }
        )
    result = payload
    choice = ((result or {}).get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    usage = (result or {}).get("usage") or {}
    empty_marker = "trae upstream returned an empty response" in content
    if mode == "tool":
        ok = bool(tool_calls)
    else:
        ok = bool(content.strip()) and not empty_marker
    return JSONResponse(
        {
            "success": ok,
            "model": model,
            "mode": mode,
            "requested_endpoint": endpoint,
            **trace,
            "elapsed_ms": elapsed_ms,
            "finish_reason": choice.get("finish_reason"),
            "provider_model_name": (result or {}).get("provider_model_name") or "",
            "requested_reasoning_effort": effort or None,
            "reasoning_effort_applied": trace.get("reasoning_effort_applied") or None,
            "reasoning_effort_note": _effort_note(effort, trace),
            "requested_max_mode": max_mode_requested,
            "reasoning": str(message.get("reasoning_content") or "")[:600] or None,
            "reply": content[:200],
            "tool_calls": [
                {
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": ((call.get("function") or {}).get("arguments") or "")[
                        :120
                    ],
                }
                for call in tool_calls
                if isinstance(call, dict)
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reasoning_tokens": (
                    usage["completion_tokens_details"].get("reasoning_tokens")
                    if isinstance(usage.get("completion_tokens_details"), Mapping)
                    else usage.get("reasoning_tokens")
                ),
            },
            "error": None if ok else (
                "upstream returned an empty response"
                if empty_marker or not content.strip()
                else "no tool call was returned"
            ),
        }
    )


@app.get("/v1")
async def v1_index(request: Request):
    return await models(request)


# ---- Web login endpoints ----

@app.get("/web/login", response_class=HTMLResponse)
async def web_login():
    return _web_login_html()


def _helper_generated_at() -> str:
    return datetime.now(_CHECKIN_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _helper_relay_url(request: Request, relay_param: str = "") -> str:
    """Resolve the deployment address to bake into a downloaded helper.

    Prefer an explicit public URL, then the browser address, proxy headers and
    Host. Never substitute the container's listening port for a mapped port.
    """
    try:
        scheme = str(request.url.scheme or "http").lower()
    except (AttributeError, ValueError):
        scheme = "http"
    host_header = str(request.headers.get("host") or "").strip()
    # 直连部署可以让 X-Forwarded-* 完全不参与判定，避免代理头被伪造后写进下载脚本。
    trust_proxy = str(
        os.environ.get("TRAE_TRUST_PROXY_HEADERS", "1")
    ).strip().lower() not in ("0", "false", "no", "off")
    forwarded_proto = str(request.headers.get("x-forwarded-proto") or "") if trust_proxy else ""
    forwarded_host = str(request.headers.get("x-forwarded-host") or "") if trust_proxy else ""
    if forwarded_proto.split(",")[0].strip().lower() in ("http", "https"):
        scheme = forwarded_proto.split(",")[0].strip().lower()
    public_base = str(
        os.environ.get("TRAE_PUBLIC_BASE_URL") or ""
    ).strip()
    if public_base:
        # A typo in an explicit deployment URL should be reported rather than
        # silently producing a helper bound to a different server.
        return _normalize_relay_url(public_base)
    prefix = str(request.scope.get("root_path") or "").rstrip("/")
    candidates = _helper_candidate_hosts(
        public_base=public_base,
        forwarded_proto=forwarded_proto,
        forwarded_host=forwarded_host + prefix if forwarded_host else "",
        scheme=scheme,
        host=host_header + prefix if host_header else "",
    )
    browser_origin = str(relay_param or "").strip()
    if browser_origin:
        try:
            browser_base = _normalize_relay_url(browser_origin)
        except RelayURLError:
            browser_base = ""
        if browser_base:
            def origin(base: str) -> tuple[str, str, int]:
                parsed = urlsplit(_normalize_relay_url(base))
                return (
                    parsed.scheme, parsed.hostname or "",
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                )

            allowed_origins = set()
            # Do not authorize an arbitrary target using the loopback fallback.
            for candidate in candidates[:-1]:
                try:
                    allowed_origins.add(origin(candidate))
                except RelayURLError:
                    continue
            if origin(browser_base) not in allowed_origins:
                raise RelayURLError("relay URL must match the deployment host or trusted proxy")
            candidates.insert(0, browser_base)
    return _pick_relay_url(candidates)


def _helper_payload(
    request: Request, kind: str, relay_param: str = ""
) -> tuple[str, str, str, str]:
    kind = (kind or "py").strip().lower()
    if kind not in HELPER_KINDS:
        raise RelayURLError(f"unsupported helper kind: {kind}")
    relay_url = _helper_relay_url(request, relay_param)
    listener_port = _normalize_listener_port(
        os.environ.get("WEB_LOGIN_LISTENER_PORT") or LOCAL_LISTENER_PORT
    )
    template = ""
    if not WEB_LOGIN_SCRIPT.exists():
        raise FileNotFoundError("web_login.py 未随镜像打包")
    if kind == "py":
        template = WEB_LOGIN_SCRIPT.read_text(encoding="utf-8")
    text = _render_auth_helper(
        kind,
        template=template,
        relay_url=relay_url,
        listener_port=listener_port,
        client_id=TRAE_CLIENT_ID,
        auth_url=TRAE_AUTH_URL,
        generated_at=_helper_generated_at(),
    )
    filename = _helper_file_name(kind)
    return text, filename, _helper_media_type(kind), relay_url


@app.get("/web/login/download")
async def web_login_download(
    request: Request,
    as_param: str = Query("py", alias="as"),
    relay_param: str = Query("", alias="relay"),
):
    """Serve the local authorization helper with this deployment's address baked in."""
    try:
        payload, filename, media, relay_url = _helper_payload(request, as_param, relay_param)
    except FileNotFoundError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)
    except RelayURLError as exc:
        return JSONResponse({"success": False, "error": f"授权助手生成参数无效: {exc}"}, status_code=400)
    except OSError as exc:
        return JSONResponse({"success": False, "error": f"读取模板失败: {exc}"}, status_code=500)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "X-Relay-Base-Url": relay_url,
    }
    return Response(content=payload.encode("utf-8"), media_type=media, headers=headers)


@app.get("/web/login/helper-info")
async def web_login_helper_info(request: Request, relay_param: str = Query("", alias="relay")):
    """Report the address that generated downloads will embed."""
    try:
        relay_url = _helper_relay_url(request, relay_param)
        listener_port = _normalize_listener_port(
            os.environ.get("WEB_LOGIN_LISTENER_PORT") or LOCAL_LISTENER_PORT
        )
    except RelayURLError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)
    return JSONResponse(
        {
            "success": True,
            "relay_url": relay_url,
            "listener_port": listener_port,
            "generated_py": "/web/login/download?as=py",
            "generated_bat": "/web/login/download?as=bat",
            "generated_sh": "/web/login/download?as=sh",
            "relay_version": APP_VERSION,
            "revision": str(os.environ.get("RELAY_BUILD_REVISION") or "local"),
        }
    )


@app.get("/authorize", response_class=HTMLResponse)
async def oauth_callback(request: Request):
    parsed = await _parse_oauth_params(dict(request.query_params))
    trace_id = request.query_params.get("loginTraceID") or request.query_params.get("login_trace_id") or ""
    if not parsed.get("token"):
        return HTMLResponse(
            _oauth_result_html(False, "未收到有效的 userJwt，请确认已登录 trae.cn", trace_id),
            status_code=400,
        )
    try:
        auth.add_account(parsed)
    except ValueError as e:
        return HTMLResponse(_oauth_result_html(False, str(e), trace_id), status_code=400)
    return HTMLResponse(_oauth_result_html(True, "登录成功，凭证已写入服务器", trace_id))


@app.post("/api/web-auth")
async def web_auth(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)

    token = body.get("token") or body.get("accessToken") or ""
    if not token:
        return JSONResponse({"success": False, "error": "token is required"}, status_code=400)

    parsed = {
        "token": token,
        "refresh_token": body.get("refreshToken") or body.get("refresh_token") or "",
        "user_id": body.get("userId") or body.get("user_id") or "",
        "tenant_id": body.get("tenantId") or body.get("tenant_id") or "",
        "region": body.get("region") or "",
        "ai_region": body.get("aiRegion") or body.get("ai_region") or "",
        "host": body.get("host") or "",
        "expired_at": body.get("expiredAt") or body.get("expired_at") or body.get("tokenExpires") or "",
        "refresh_expired_at": body.get("refreshExpiredAt") or body.get("refresh_expired_at") or body.get("refreshExpires") or "",
        "client_id": body.get("clientId") or body.get("client_id") or body.get("clientID") or "",
        "web_id": body.get("webId") or body.get("web_id") or "",
        "biz_user_id": body.get("bizUserId") or body.get("biz_user_id") or "",
        "user_unique_id": body.get("userUniqueId") or body.get("user_unique_id") or "",
        "scope": body.get("scope") or "",
        "tenant": body.get("tenant") or "",
        "app_language": body.get("appLanguage") or body.get("app_language") or "",
        "user_region": body.get("userRegion") or body.get("user_region") or "",
        "user_identity": body.get("userIdentity") or body.get("user_identity") or "",
        "screen_name": body.get("screenName") or body.get("screen_name") or "",
    }
    label = body.get("label") or ""
    try:
        auth.add_account(parsed, label=label)
    except ValueError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)

    s = auth.get_auth()
    return JSONResponse({
        "success": True,
        "has_token": bool(s.token),
        "user_id": s.user_id or "",
    })


@app.get("/api/checkin/status")
async def api_checkin_status():
    try:
        data = await trae_client.fetch_checkin_credits_status()
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=502)
    return JSONResponse({"success": True, "data": data})

@app.get("/api/checkin/accounts")
async def api_checkin_accounts():
    """Refresh daily checkin state only for every stored account.

    Entitlement credits use a different upstream API and are intentionally
    refreshed through ``/api/checkin/credits/accounts``. Keeping these probes
    apart stops an ordinary dashboard refresh from creating extra checkin
    traffic while a user is recovering from code 9074.
    """
    raw_accounts = auth.get_accounts_raw()

    async def _query_one(aid: str, rec: dict) -> dict:
        row = _cached_checkin_account_snapshot(aid, rec)
        if not (rec.get("token") or ""):
            row["error"] = "No token"
            return row
        try:
            async with _checkin_account_lock(aid):
                row.update(
                    await _fetch_checkin_status_snapshot(
                        aid, rec, use_cached_on_cooldown=True
                    )
                )
        except Exception as e:
            row["error"] = str(e)
        # Username check rides on the status refresh, not on credits queries.
        try:
            row.update(await _sync_account_user_name(aid))
        except Exception as e:
            row["user_name_error"] = str(e)[:200]
        return row

    results = await asyncio.gather(
        *[_query_one(aid, rec) for aid, rec in raw_accounts],
        return_exceptions=True,
    )
    for idx, item in enumerate(results):
        if isinstance(item, BaseException):
            aid, rec = raw_accounts[idx]
            results[idx] = {
                **_cached_checkin_account_snapshot(aid, rec),
                "error": str(item),
            }
    return JSONResponse(
        {"success": True, "active": auth.get_active_account_id(), "accounts": results}
    )


@app.get("/api/checkin/credits/accounts")
async def api_checkin_credits_accounts():
    """Refresh entitlement credits only for every stored account.

    This endpoint never calls the daily-checkin status API. It is deliberately
    separate from ``/api/checkin/accounts`` so credits refreshes cannot create
    extra checkin probes or interfere with claim cooldown handling.
    """
    raw_accounts = auth.get_accounts_raw()

    async def _query_one(aid: str, rec: dict) -> dict:
        row = _cached_checkin_account_snapshot(aid, rec)
        if not (rec.get("token") or ""):
            row["error"] = "No token"
            return row
        try:
            async with _checkin_account_lock(aid):
                row.update(await _fetch_credit_account_snapshot(aid, rec))
        except Exception as e:
            row["error"] = str(e)
        return row

    results = await asyncio.gather(
        *[_query_one(aid, rec) for aid, rec in raw_accounts],
        return_exceptions=True,
    )
    for idx, item in enumerate(results):
        if isinstance(item, BaseException):
            aid, rec = raw_accounts[idx]
            results[idx] = {
                **_cached_checkin_account_snapshot(aid, rec),
                "error": str(item),
            }
    return JSONResponse({"success": True, "accounts": results})


@app.get("/api/checkin/credits/{account_id}")
async def api_checkin_credits(account_id: str):
    """Fetch general account credits for one account without querying checkin.

    Keep this dynamic route after ``/api/checkin/credits/accounts``. Starlette
    resolves routes in declaration order, so placing it first makes the bulk
    path look like an account whose id is literally ``accounts``.
    """
    rec = auth.get_account_record(account_id)
    if not (rec.get("token") or ""):
        return JSONResponse(
            {"success": False, "error": "account not found or token missing"},
            status_code=404,
        )
    try:
        # Keep the single-account endpoint on the same reconciliation path as
        # bulk and scheduled refreshes.  Directly merging the provider result
        # here used to erase the weighted local debit until the next poll.
        async with _checkin_account_lock(account_id):
            snapshot = await _fetch_credit_account_snapshot(account_id, rec)
        parsed = snapshot.get("account_credits") or {}
        return JSONResponse(
            {
                "success": True,
                "id": account_id,
                "credits": parsed,
                # ``raw`` remains for older clients.  It is returned only
                # from the single-account diagnostic endpoint; the persisted
                # cache still stores the parsed, credential-free projection.
                "raw": snapshot.get("_raw", parsed),
            }
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=502)


def _credits_user_name(raw: dict) -> str:
    """Extract the upstream 用户名 from an entitlement-usage response.

    The pay API sometimes carries the profile name (user_name / userName /
    screenName) next to the pack list; use whichever key shows up first.
    """
    if not isinstance(raw, dict):
        return ""
    for key in ("user_name", "userName", "screenName", "screen_name", "userNickName"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    user = raw.get("user") or raw.get("user_info")
    if isinstance(user, dict):
        for key in ("user_name", "userName", "screenName", "name"):
            value = user.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


async def _fetch_full_credits(token: str) -> dict:
    """Fetch the merged general (通用) credits for one token.

    Upstream merged every pack into a single 通用积分 pool, so the relay only
    queries req_source=1 and no longer derives work/total splits.
    """
    raw_ac = await trae_client.fetch_account_credits(token)
    return {
        "account_credits": trae_client.parse_account_credits(raw_ac),
        "user_name": _credits_user_name(raw_ac),
        # Keep the original payload available to the legacy single-account
        # endpoint without making callers perform a second upstream request.
        "_raw": raw_ac,
    }


def _cached_checkin_account_snapshot(
    account_id: str, rec: dict | None = None
) -> dict:
    """Build one dashboard row from the persisted cache without an upstream call."""
    record = rec if rec is not None else auth.get_account_record(account_id)
    cached = dict(record.get("checkin") or {})
    return {
        "success": True,
        "id": account_id,
        "user_id": record.get("user_id", account_id),
        "label": record.get("label") or record.get("user_id") or account_id,
        "source": record.get("source", ""),
        "has_token": bool(record.get("token")),
        "is_active": account_id == auth.get_active_account_id(),
        "is_valid": bool(record.get("token")),
        "expires": record.get("expired_at", ""),
        "checked_in": cached.get("checked_in"),
        "credits": cached.get("credits"),
        "checkin_enable": cached.get("enable"),
        "account_credits": cached.get("account_credits"),
        "credits_official_remaining": (
            cached.get("account_credits", {}).get(
                "official_remaining",
                cached.get("account_credits", {}).get("remaining"),
            )
            if isinstance(cached.get("account_credits"), Mapping)
            else None
        ),
        "credits_official_used": (
            cached.get("account_credits", {}).get("official_used")
            if isinstance(cached.get("account_credits"), Mapping)
            else None
        ),
        "credits_local_debit": (
            cached.get("account_credits", {}).get("local_debit", 0)
            if isinstance(cached.get("account_credits"), Mapping)
            else 0
        ),
        "credits_reconciled_at": (
            cached.get("account_credits", {}).get("credits_reconciled_at", 0)
            if isinstance(cached.get("account_credits"), Mapping)
            else 0
        ),
        "credits_reconcile_status": (
            cached.get("account_credits", {}).get("reconcile_status", "unknown")
            if isinstance(cached.get("account_credits"), Mapping)
            else "unknown"
        ),
        "credits_reconcile_confidence": (
            cached.get("account_credits", {}).get("reconcile_confidence")
            if isinstance(cached.get("account_credits"), Mapping)
            else None
        ),
        "checkin_updated_at": record.get(
            "checkin_status_updated_at", record.get("checkin_updated_at", 0)
        ),
        "credits_updated_at": record.get("credits_updated_at", 0),
        "checkin": cached,
    }


def _checkin_cache_is_today(record: dict) -> bool:
    """Only trust a cached checked-in flag for today's Trae CN business day."""
    try:
        updated_at = float(record.get("checkin_status_updated_at") or 0)
    except (TypeError, ValueError):
        return False
    if updated_at <= 0:
        return False
    updated_date = datetime.fromtimestamp(updated_at, _CHECKIN_TIMEZONE).date()
    current_date = datetime.fromtimestamp(time.time(), _CHECKIN_TIMEZONE).date()
    return updated_date == current_date


def _checkin_response_is_rate_limited(data: dict) -> bool:
    return isinstance(data, dict) and (
        data.get("code") == 9074 or data.get("_error_code") == 9074
    )


def _checkin_auth_failed(data: dict) -> bool:
    """Detect a server-side token invalidation inside a business-code response."""
    if not isinstance(data, dict):
        return False
    if data.get("code") == 1001:
        return True
    text = str(data.get("message") or data.get("error") or "")
    return "not able to authenticate" in text


def _credits_auth_failed(exc: BaseException) -> bool:
    """Detect a server-side token invalidation inside a credits fetch error."""
    text = str(exc)
    return "[401]" in text or "not able to authenticate" in text


async def _fetch_credit_account_snapshot(
    account_id: str, rec: dict | None = None
) -> dict:
    """Refresh entitlement credits only, preserving cached daily-checkin state."""
    record = rec if rec is not None else auth.get_account_record(account_id)
    token = record.get("token") or ""
    if not token:
        raise KeyError("account not found or token missing")

    try:
        full = await _fetch_full_credits(token)
    except Exception as exc:
        if _credits_auth_failed(exc) and await auth.refresh_account(account_id):
            fresh = auth.get_account_record(account_id)
            full = await _fetch_full_credits(fresh.get("token") or "")
        else:
            raise
    patch = {key: value for key, value in full.items() if value is not None}
    if patch and isinstance(patch.get("account_credits"), Mapping):
        observed_at = time.time()
        reconcile_state, effective = _credit_snapshot_projection(
            account_id,
            patch["account_credits"],
            observed_at,
        )
        merged = auth.reconcile_account_credits(
            account_id,
            patch,
            effective_remaining=effective.get("effective_remaining"),
            local_debit=reconcile_state.get("local_debit", 0),
            status=reconcile_state.get("status", "ok"),
            source=reconcile_state.get("source", "upstream"),
            confidence=reconcile_state.get("confidence", 1.0),
            observed_at=observed_at,
        )
        # A test harness or a legacy storage adapter may expose a record
        # through ``get_account_record`` without registering it in the
        # persistence map used by ``reconcile_account_credits``.  Preserve the
        # fetched projection for that adapter; normal auth storage returns the
        # reconciled mapping above.
        if not merged:
            merged = dict(record.get("checkin") or {})
            merged["account_credits"] = dict(patch["account_credits"])
            merged["account_credits"].update(
                {
                    "remaining": effective.get("effective_remaining"),
                    "local_debit": reconcile_state.get("local_debit", 0),
                    "reconcile_status": reconcile_state.get("status", "ok"),
                    "reconcile_source": reconcile_state.get("source", "upstream"),
                    "reconcile_confidence": reconcile_state.get("confidence", 1.0),
                }
            )
    else:
        merged = dict(record.get("checkin") or {})
    row = _cached_checkin_account_snapshot(account_id, record)
    row.update(
        {
            "account_credits": merged.get("account_credits"),
            "checkin": merged,
        }
    )
    if "_raw" in full:
        row["_raw"] = full.get("_raw")
    return row


async def _sync_account_user_name(account_id: str) -> dict:
    """Fetch GetUserInfo once and sync the ScreenName into the account label."""
    record = auth.get_account_record(account_id)
    token = record.get("token") or ""
    if not token:
        raise KeyError("account not found or token missing")
    try:
        info = await trae_client.fetch_user_info(token)
    except Exception as exc:
        if _credits_auth_failed(exc) and await auth.refresh_account(account_id):
            fresh = auth.get_account_record(account_id)
            info = await trae_client.fetch_user_info(fresh.get("token") or "")
        else:
            raise
    name = str(info.get("ScreenName") or "").strip()
    label, changed = auth.sync_account_screen_name(
        account_id, name, str(info.get("UserID") or "")
    )
    out = {"label_synced": changed}
    if label:
        out["label"] = label
    if name:
        out["user_name"] = name
    return out


async def _fetch_checkin_status_snapshot(
    account_id: str,
    rec: dict | None = None,
    *,
    respect_pending: bool = True,
    use_cached_on_cooldown: bool = False,
) -> dict:
    """Refresh daily-checkin state only, never entitlement credits."""
    record = rec if rec is not None else auth.get_account_record(account_id)
    token = record.get("token") or ""
    if not token:
        raise KeyError("account not found or token missing")

    cached_row = _cached_checkin_account_snapshot(account_id, record)
    # The 9074 cooldown belongs to the *claim* endpoint only. Probing showed the
    # status endpoint keeps answering code=0 during that window, so skipping it
    # here only pinned the dashboard to a stale checked_in=false.
    cooldown_remaining = (
        _checkin_cooldown_remaining(account_id) if use_cached_on_cooldown else 0
    )
    if use_cached_on_cooldown:
        status_cooldown = _checkin_status_cooldown_remaining(account_id)
        if status_cooldown:
            # The status endpoint itself answered 9074 recently; probing again
            # only extends that window.
            return {
                **cached_row,
                "success": False,
                "stale": True,
                "rate_limited": True,
                "retryable": True,
                "retry_after_seconds": status_cooldown,
                "error": (
                    "Trae checkin status skipped [9074]: local cooldown active; "
                    f"retry after at least {status_cooldown}s"
                ),
            }

    status = await trae_client.fetch_checkin_credits_status(token, account_id)
    if _checkin_auth_failed(status):
        if await auth.refresh_account(account_id):
            fresh = auth.get_account_record(account_id)
            status = await trae_client.fetch_checkin_credits_status(
                fresh.get("token") or "", account_id
            )
    if _checkin_response_is_rate_limited(status):
        retry_after = _checkin_start_cooldown(account_id)
        _CHECKIN_STATUS_COOLDOWN_UNTIL[account_id] = time.monotonic() + retry_after
        return {
            **cached_row,
            "success": False,
            "rate_limited": True,
            "data": status,
            "retryable": True,
            "retry_after_seconds": retry_after,
            "error": _checkin_claim_error(status),
        }

    # A successful claim can be visible at the status endpoint a little later.
    # Keep accepted state monotonic during the grace window without issuing
    # extra automatic verification probes.
    now = time.monotonic()
    accepted_until = _CHECKIN_ACCEPTED_UNTIL.get(account_id, 0.0)
    if status.get("checked_in") is True:
        _CHECKIN_ACCEPTED_UNTIL.pop(account_id, None)
        status = {**status, "verification_pending": False}
        # The account is done for today, so a leftover 9074 backoff would only
        # block tomorrow's first claim. Retire it together with the cooldown.
        _CHECKIN_COOLDOWN_UNTIL.pop(account_id, None)
        _CHECKIN_STATUS_COOLDOWN_UNTIL.pop(account_id, None)
        if _checkin_retry_state(account_id)[0]:
            _checkin_clear_retry_state(account_id)
    elif respect_pending and accepted_until > now:
        status = {
            **status,
            "checked_in": True,
            "verification_pending": True,
        }
    elif accepted_until:
        _CHECKIN_ACCEPTED_UNTIL.pop(account_id, None)

    merged = auth.merge_account_checkin(account_id, status)
    row = {
        **cached_row,
        "checked_in": status.get("checked_in"),
        "credits": status.get("credits"),
        "checkin_enable": status.get("enable"),
        "checkin": status,
        "account_credits": merged.get("account_credits"),
        "checkin_updated_at": auth.get_account_record(account_id).get(
            "checkin_status_updated_at",
            auth.get_account_record(account_id).get(
                "checkin_updated_at", cached_row.get("checkin_updated_at", 0)
            ),
        ),
    }
    if cooldown_remaining and status.get("checked_in") is not True:
        # Report the live status and keep the claim window visible so the UI
        # does not invite a claim the upstream will reject with 9074.
        row.update(
            {
                "claim_rate_limited": True,
                "retryable": True,
                "retry_after_seconds": cooldown_remaining,
            }
        )
    return row


async def _fetch_checkin_account_snapshot(
    account_id: str,
    rec: dict | None = None,
    *,
    respect_pending: bool = True,
) -> dict:
    """Backward-compatible name for a checkin-only account refresh."""
    return await _fetch_checkin_status_snapshot(
        account_id,
        rec,
        respect_pending=respect_pending,
        use_cached_on_cooldown=True,
    )

def _checkin_claim_error(data: dict) -> str:
    """Format an upstream business-code failure without hiding its cause."""
    data = data if isinstance(data, dict) else {}
    code = data.get("code")
    message = data.get("message") or data.get("error") or "unknown upstream error"
    if code == 9074:
        retry_after = max(1, int(CHECKIN_RETRY_AFTER))
        message = f"{message}; upstream rate limit, retry after at least {retry_after}s"
    return f"Trae checkin failed [{code}]: {message}"


def _checkin_claim_ok(data: dict) -> bool:
    """Only code=0 is a successful claim; HTTP 200 alone is insufficient."""
    return isinstance(data, dict) and data.get("code") == 0


def _checkin_account_lock(account_id: str) -> asyncio.Lock:
    """Return an account lock that is safe across test event loops/reloads."""
    lock = _CHECKIN_ACCOUNT_LOCKS.get(account_id)
    current_loop = asyncio.get_running_loop()
    bound_loop = getattr(lock, "_loop", None) if lock is not None else None
    if lock is None or (bound_loop is not None and bound_loop is not current_loop):
        lock = asyncio.Lock()
        _CHECKIN_ACCOUNT_LOCKS[account_id] = lock
    return lock


def _checkin_claim_gate() -> asyncio.Lock:
    """Serialize actual upstream claims and enforce the configured interval."""
    global _CHECKIN_CLAIM_GATE, _CHECKIN_CLAIM_GATE_LOOP
    current_loop = asyncio.get_running_loop()
    if _CHECKIN_CLAIM_GATE is None or _CHECKIN_CLAIM_GATE_LOOP is not current_loop:
        _CHECKIN_CLAIM_GATE = asyncio.Lock()
        _CHECKIN_CLAIM_GATE_LOOP = current_loop
    return _CHECKIN_CLAIM_GATE


def _checkin_status_cooldown_remaining(account_id: str) -> int:
    """Return the remaining window after the *status* endpoint returned 9074.

    This is deliberately in-memory only: a claim-side 9074 is common and
    persisted, while a status-side 9074 is rare and must not survive a restart
    as a permanent read block.
    """

    until = _CHECKIN_STATUS_COOLDOWN_UNTIL.get(account_id, 0.0)
    remaining = until - time.monotonic()
    if remaining <= 0:
        _CHECKIN_STATUS_COOLDOWN_UNTIL.pop(account_id, None)
        return 0
    return max(1, int(remaining))


def _checkin_cooldown_remaining(account_id: str) -> int:
    until = _CHECKIN_COOLDOWN_UNTIL.get(account_id, 0.0)
    remaining = until - time.monotonic()
    if remaining <= 0:
        _CHECKIN_COOLDOWN_UNTIL.pop(account_id, None)
        # Fall back to persisted wall-clock retry deadline so a container
        # restart does not immediately re-hammer a still-limited upstream.
        rec = auth.get_account_record(account_id)
        checkin = rec.get("checkin") or {}
        backoff = float(checkin.get("retry_backoff") or 0)
        updated = float(checkin.get("retry_updated_at") or 0)
        if backoff > 0 and updated > 0:
            persisted_remaining = (updated + backoff) - time.time()
            if persisted_remaining > 0:
                remaining = persisted_remaining
    if remaining <= 0:
        return 0
    return max(1, int(remaining + 0.999))


def _checkin_retry_state(account_id: str) -> tuple[float, int]:
    """Return persisted (retry_after_backoff, consecutive_9074_count)."""
    rec = auth.get_account_record(account_id)
    checkin = rec.get("checkin") or {}
    backoff = float(checkin.get("retry_backoff") or 0)
    count = int(checkin.get("retry_9074_count") or 0)
    return backoff, count


def _checkin_persist_retry_state(
    account_id: str, backoff: float, count: int
) -> None:
    auth.merge_account_retry(
        account_id,
        {
            "retry_backoff": backoff,
            "retry_9074_count": count,
            "retry_updated_at": time.time(),
        },
    )


def _checkin_clear_retry_state(account_id: str) -> None:
    auth.merge_account_retry(
        account_id,
        {
            "retry_backoff": 0,
            "retry_9074_count": 0,
        },
    )


def _checkin_next_backoff(account_id: str) -> float:
    """Exponential backoff for one account's 9074 streak."""
    _, count = _checkin_retry_state(account_id)
    base = max(1.0, float(CHECKIN_RETRY_AFTER))
    # 9074 is an upstream capacity signal ("too many users"), not a penalty for
    # this account, and each retry now also rotates the device id. Escalate
    # gently so a still-unclaimed account keeps getting attempts within the
    # day instead of parking on the hour-long ceiling.
    backoff = base * (2 ** min(count, CHECKIN_9074_BACKOFF_EXPONENT_CAP))
    max_backoff = max(base, float(CHECKIN_9074_MAX_BACKOFF))
    return max(base, min(backoff, max_backoff))


def _checkin_start_cooldown(account_id: str, retry_after: float | None = None) -> int:
    """Start one account's local cooldown after an upstream 9074 response."""
    if retry_after is None:
        retry_after = _checkin_next_backoff(account_id)
    retry_after = max(1, int(retry_after))
    _CHECKIN_COOLDOWN_UNTIL[account_id] = time.monotonic() + retry_after
    _, count = _checkin_retry_state(account_id)
    _checkin_persist_retry_state(account_id, float(retry_after), count + 1)
    return retry_after


def _checkin_mark_accepted(account_id: str, snapshot: dict, *, pending: bool) -> None:
    """Persist a successful claim without allowing missing fields to erase cache."""
    cached = auth.get_account_record(account_id).get("checkin") or {}
    checkin = dict(cached)
    checkin.update(snapshot.get("checkin") or {})
    checkin["checked_in"] = True
    if pending:
        checkin["verification_pending"] = True
    else:
        checkin.pop("verification_pending", None)
    for key in ("account_credits",):
        value = snapshot.get(key)
        if value is not None:
            checkin[key] = value
    auth.merge_account_checkin(account_id, checkin)


def _checkin_cooldown_payload(snapshot: dict, retry_after: int) -> dict:
    data = {
        "code": 9074,
        "message": "local cooldown active; upstream claim was not sent",
    }
    return {
        **snapshot,
        "success": False,
        "skipped": True,
        "claim_sent": False,
        "data": data,
        "retryable": True,
        "retry_after_seconds": retry_after,
        "error": f"Trae checkin skipped [9074]: local cooldown active; retry after at least {retry_after}s",
    }


def _checkin_device_rotation_enabled() -> bool:
    """Whether a 9074 claim may rotate to a fresh device id.

    Set ``TRAE_CHECKIN_NO_DEVICE_ROTATION=1`` to keep one fixed id per account.
    """

    return str(
        os.environ.get("TRAE_CHECKIN_NO_DEVICE_ROTATION", "")
    ).strip().lower() not in {"1", "true", "yes", "on"}


async def _claim_checkin_throttled(account_id: str, token: str) -> tuple[dict | None, int]:
    """Send at most one claim after account cooldown and global spacing checks."""
    remaining = _checkin_cooldown_remaining(account_id)
    if remaining:
        return None, remaining

    global _CHECKIN_NEXT_CLAIM_AT
    async with _checkin_claim_gate():
        remaining = _checkin_cooldown_remaining(account_id)
        if remaining:
            return None, remaining
        wait_for = max(0.0, _CHECKIN_NEXT_CLAIM_AT - time.monotonic())
        if wait_for:
            await asyncio.sleep(wait_for)
        try:
            data = await trae_client.claim_checkin_credits(token, account_id)
        finally:
            _CHECKIN_NEXT_CLAIM_AT = time.monotonic() + max(0.0, CHECKIN_INTERVAL)

        if data.get("code") == 9074:
            # 9074 is scoped to the device id, not the account: the same token
            # claims successfully on a freshly derived id. Rotate once and retry
            # before falling back to a timed cooldown.
            if _checkin_device_rotation_enabled() and trae_client.rotate_checkin_device_id(
                token, account_id
            ):
                logger.info(
                    "checkin 9074 account=%s: rotated device id, retrying once",
                    account_id,
                )
                try:
                    data = await trae_client.claim_checkin_credits(token, account_id)
                finally:
                    _CHECKIN_NEXT_CLAIM_AT = time.monotonic() + max(
                        0.0, CHECKIN_INTERVAL
                    )
        if data.get("code") == 9074:
            retry_after = _checkin_start_cooldown(account_id)
            until = time.monotonic() + retry_after
            _CHECKIN_NEXT_CLAIM_AT = max(_CHECKIN_NEXT_CLAIM_AT, until)
            return data, retry_after
        if _checkin_claim_ok(data):
            _CHECKIN_COOLDOWN_UNTIL.pop(account_id, None)
            _checkin_clear_retry_state(account_id)
        return data, 0


async def _claim_checkin_account(account_id: str) -> dict:
    """Lock, use today's cache when available, and send at most one claim."""
    lock = _checkin_account_lock(account_id)
    async with lock:
        record = auth.get_account_record(account_id)
        token = record.get("token") or ""
        if not token:
            return {
                "success": False,
                "id": account_id,
                "error": "account not found or token missing",
            }

        before = _cached_checkin_account_snapshot(account_id, record)
        accepted_recently = _CHECKIN_ACCEPTED_UNTIL.get(account_id, 0.0) > time.monotonic()
        if accepted_recently:
            before["checked_in"] = True
            before["verification_pending"] = True
        if before.get("checked_in") is True and (
            accepted_recently or _checkin_cache_is_today(record)
        ):
            _CHECKIN_COOLDOWN_UNTIL.pop(account_id, None)
            _checkin_clear_retry_state(account_id)
            return {
                **before,
                "success": True,
                "skipped": True,
                "claim_sent": False,
                "data": {"code": 0, "message": "already checked in, skipped"},
            }

        cooldown = _checkin_cooldown_remaining(account_id)
        if cooldown:
            return _checkin_cooldown_payload(before, cooldown)

        try:
            data, retry_after = await _claim_checkin_throttled(account_id, token)
        except Exception as exc:
            return {**before, "success": False, "claim_sent": False, "error": str(exc)}

        if data is None:
            return _checkin_cooldown_payload(before, retry_after)

        if _checkin_auth_failed(data) and await auth.refresh_account(account_id):
            # Server-side token invalidation: rotate this account and retry the
            # claim once instead of surfacing a transient auth error.
            fresh = auth.get_account_record(account_id)
            data, retry_after = await _claim_checkin_throttled(
                account_id, fresh.get("token") or ""
            )
            if data is None:
                return _checkin_cooldown_payload(before, retry_after)

        if data.get("code") == 9074:
            # A 9074 response is already a rate-limit signal.  Querying status
            # immediately after it only compounds the upstream frequency limit.
            payload = _checkin_cooldown_payload(before, retry_after)
            payload["skipped"] = False
            payload["claim_sent"] = True
            payload["data"] = data
            payload["error"] = _checkin_claim_error(data)
            return payload

        if not _checkin_claim_ok(data):
            return {
                **before,
                "success": False,
                "skipped": False,
                "claim_sent": True,
                "data": data,
                "error": _checkin_claim_error(data),
            }

        # Code 0 is an accepted claim.  Persist it immediately; the explicit
        # status button can verify later without making the claim path noisy.
        _CHECKIN_ACCEPTED_UNTIL[account_id] = time.monotonic() + max(
            10.0, float(CHECKIN_RETRY_AFTER)
        )
        _CHECKIN_COOLDOWN_UNTIL.pop(account_id, None)
        _checkin_clear_retry_state(account_id)
        latest = {**before, "checked_in": True, "verification_pending": True}
        _checkin_mark_accepted(account_id, latest, pending=True)

        payload = {
            **latest,
            "success": True,
            "skipped": False,
            "claim_sent": True,
            "data": data,
            "checked_in": True,
        }
        payload["verification_pending"] = True
        return payload


@app.post("/api/checkin/claim-all")
async def api_checkin_claim_all():
    """One-click polling checkin for every stored account.

    Claims are strictly ordered, and this endpoint does not perform a status or
    credit refresh.  The separate dashboard query buttons own those reads;
    keeping them out of this path prevents a claim burst from becoming a 9074
    burst as well.
    """
    raw_accounts = list(auth.get_accounts_raw())
    active_id = auth.get_active_account_id()
    results = []
    for aid, rec in raw_accounts:
        token = rec.get("token") or ""
        row = _cached_checkin_account_snapshot(aid, rec)
        row["is_active"] = aid == active_id
        if not token:
            row["success"] = False
            row["skipped"] = False
            row["error"] = "No token"
            results.append(row)
            continue
        claimed = await _claim_checkin_account(row["id"])
        results.append({**row, **claimed})
    return JSONResponse({"success": True, "accounts": results, "interval": CHECKIN_INTERVAL})

@app.post("/api/checkin/claim-credits")
async def api_checkin_claim_credits():
    """Credit-ordered polling checkin.

    First fetches total entitlement credits for every account, sorts
    accounts by remaining credits descending (higher credits first),
    then processes them one by one with the standard interval.
    """
    results = []
    raw_accounts = list(auth.get_accounts_raw())
    records_by_id = {account_id: record for account_id, record in raw_accounts}
    enriched = []

    for aid, rec in raw_accounts:
        token = rec.get("token") or ""
        label = rec.get("label") or rec.get("user_id") or aid
        row = {
            "id": aid,
            "label": label,
            "user_id": rec.get("user_id", ""),
            "is_active": aid == auth.get_active_account_id(),
        }
        if not token:
            row["success"] = False
            row["error"] = "No token"
            row["credits_sort"] = -1
            enriched.append(row)
            continue

        cached = rec.get("checkin") or {}
        row["checked_in"] = cached.get("checked_in")
        row["checkin"] = cached

        credits_sort = 0
        try:
            # Reconcile against the last official anchor before sorting.  This
            # preserves any confirmed local debit that the provider has not
            # reflected yet and serializes the refresh with other credit reads.
            async with _checkin_account_lock(aid):
                snapshot = await _fetch_credit_account_snapshot(aid, rec)
            parsed = snapshot.get("account_credits") or {}
            row["account_credits"] = parsed
            row["checkin"] = snapshot.get("checkin") or row["checkin"]
            # Keep compatibility with callers that expect the full checkin
            # mapping in the claim response.  The mapping is already the
            # reconciled projection; this does not re-merge raw provider data.
            if row["checkin"]:
                row["checkin"] = auth.merge_account_credits(aid, row["checkin"])
            # Every pack lives in the merged 通用积分 pool, so the single
            # account_credits value drives the credit-priority ordering.
            credits_sort = parsed.get("remaining") or parsed.get("total_limit") or 0
            if parsed.get("unlimited"):
                credits_sort = 999999999
        except Exception as exc:
            row["account_credits"] = None
            row["credit_error"] = str(exc)
            credits_sort = 0

        row["credits_sort"] = credits_sort
        enriched.append(row)

    # Sort by credits descending (richer accounts first)
    enriched.sort(key=lambda x: x.get("credits_sort", 0), reverse=True)

    for row in enriched:
        if row.get("error"):
            results.append(row)
            continue
        if row.get("checked_in") is True and _checkin_cache_is_today(
            records_by_id.get(row["id"], {})
        ):
            row["success"] = True
            row["skipped"] = True
            row["data"] = {"code": 0, "message": "already checked in, skipped"}
            results.append(row)
            continue
        claimed = await _claim_checkin_account(row["id"])
        results.append({**row, **claimed})

    for r in results:
        if "credits_sort" in r:
            del r["credits_sort"]

    return JSONResponse({"success": True, "accounts": results, "interval": CHECKIN_INTERVAL})

@app.get("/api/checkin/account/{account_id}")
async def api_checkin_account_status(account_id: str):
    """Refresh only the requested account without sending a claim."""
    try:
        return JSONResponse(await _fetch_checkin_account_snapshot(account_id))
    except KeyError as exc:
        return JSONResponse({"success": False, "error": str(exc.args[0])}, status_code=404)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=502)


@app.post("/api/checkin/account/{account_id}")
async def api_checkin_account(account_id: str):
    """Check in one account, avoiding duplicate or parallel claim requests."""
    if not (auth.get_account_record(account_id).get("token") or ""):
        return JSONResponse(
            {"success": False, "error": "account not found or token missing"},
            status_code=404,
        )
    return JSONResponse(await _claim_checkin_account(account_id))

@app.post("/api/checkin/claim")
async def api_checkin_claim():
    account_id = auth.get_active_account_id()
    if not account_id:
        return JSONResponse(
            {"success": False, "error": "no active account"}, status_code=404
        )
    return JSONResponse(await _claim_checkin_account(account_id))

# ---- Account management & settings endpoints ----

@app.post("/api/logout")
async def api_logout():
    auth.logout_active()
    return JSONResponse({"success": True})


@app.get("/api/accounts")
async def api_accounts():
    accounts = auth.list_accounts()
    polling = auth.get_polling_status()
    return JSONResponse({"success": True, "accounts": accounts, "polling": polling, "active": polling.get("active_account", "")})


@app.post("/api/accounts/switch")
async def api_accounts_switch(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    account_id = body.get("account_id") or body.get("id") or ""
    if not account_id:
        return JSONResponse({"success": False, "error": "account_id is required"}, status_code=400)
    ok = auth.switch_account(account_id)
    if not ok:
        return JSONResponse({"success": False, "error": "account not found"}, status_code=404)
    accounts = auth.list_accounts()
    active = auth.get_active_account_id()
    account = next((item for item in accounts if item.get("id") == active), None)
    return JSONResponse(
        {
            "success": True,
            "active": active,
            "account": account,
            "accounts": accounts,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/accounts/model-enabled")
async def api_accounts_model_enabled(request: Request):
    """Toggle whether one account receives new model requests.

    This endpoint intentionally does not touch check-in state.  Disabled
    accounts remain visible and continue to participate in all daily
    check-in/status/credits operations.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"success": False, "error": "Invalid JSON body"}, status_code=400
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {"success": False, "error": "JSON body must be an object"}, status_code=400
        )
    account_id = body.get("account_id") or body.get("id") or ""
    if not isinstance(account_id, str) or not account_id.strip():
        return JSONResponse(
            {"success": False, "error": "account_id is required"}, status_code=400
        )
    if "enabled" not in body and "model_enabled" not in body:
        return JSONResponse(
            {"success": False, "error": "enabled is required"}, status_code=400
        )
    value = body.get("enabled", body.get("model_enabled"))
    if isinstance(value, bool):
        enabled = value
    elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        enabled = value.strip().lower() == "true"
    else:
        return JSONResponse(
            {"success": False, "error": "enabled must be a boolean"}, status_code=400
        )
    account_id = account_id.strip()
    if not auth.set_account_model_enabled(account_id, enabled):
        return JSONResponse(
            {"success": False, "error": "account not found"}, status_code=404
        )
    account = next(
        (item for item in auth.list_accounts() if item.get("id") == account_id),
        None,
    )
    return JSONResponse(
        {
            "success": True,
            "account": account,
            "account_id": account_id,
            "model_enabled": bool(account and account.get("model_enabled", enabled)),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/accounts/remove")
async def api_accounts_remove(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    account_id = body.get("account_id") or body.get("id") or ""
    if not account_id:
        return JSONResponse({"success": False, "error": "account_id is required"}, status_code=400)
    ok = auth.remove_account(account_id)
    if not ok:
        return JSONResponse({"success": False, "error": "account not found"}, status_code=404)
    return JSONResponse({"success": True})


@app.post("/api/settings")
async def api_settings(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"success": False, "error": "JSON body must be an object"}, status_code=400)
    web_base_url = body.get("web_base_url") or ""
    upstream_mode = body.get("upstream_mode") or ""
    endpoint_id = body.get("endpoint_id")
    # The IDE/SOLO model gateway is stored separately from the Web/Remote base
    # so selecting an IDE/SOLO preset cannot strip ``/api/remote/v1`` from the
    # Remote transport.  ``None`` means the field was absent (port-only call).
    body_model_base_url = body.get("model_base_url")
    # Older clients only know URL/mode and do not send the newer preset
    # identity.  Keep ``None`` for a port-only update, where the existing
    # endpoint must remain untouched; endpoint-field updates are migrated below.
    endpoint_fields_present = (
        "web_base_url" in body or "upstream_mode" in body or "endpoint_id" in body
    )
    legacy_endpoint_update = endpoint_id is None and endpoint_fields_present
    if not isinstance(web_base_url, str) or not isinstance(upstream_mode, str):
        return JSONResponse({"success": False, "error": "URL and mode must be strings"}, status_code=400)
    if endpoint_id is not None and not isinstance(endpoint_id, str):
        return JSONResponse({"success": False, "error": "endpoint_id must be a string"}, status_code=400)
    if body_model_base_url is not None and not isinstance(body_model_base_url, str):
        return JSONResponse({"success": False, "error": "model_base_url must be a string"}, status_code=400)
    web_base_url = web_base_url.strip().rstrip("/")
    upstream_mode = upstream_mode.strip().lower()
    endpoint_id = endpoint_id.strip().lower() if isinstance(endpoint_id, str) else None
    explicit_model_base_url = (
        body_model_base_url.strip().rstrip("/")
        if isinstance(body_model_base_url, str)
        else None
    )
    model_base_url: Optional[str] = None
    if endpoint_id:
        endpoint = next(
            (
                item
                for item in UPSTREAM_ENDPOINT_PRESETS
                if str(item.get("id") or "").strip().lower() == endpoint_id
            ),
            None,
        )
        if endpoint is None:
            return JSONResponse(
                {"success": False, "error": f"Unsupported endpoint preset: {endpoint_id}"},
                status_code=400,
            )
        # The preset id is authoritative.  The shared Web/Remote base URL is
        # always the preset's canonical value; the IDE/SOLO model gateway is a
        # separate host-only base so selecting an IDE/SOLO preset can no longer
        # strip ``/api/remote/v1`` from the Remote transport.
        if not web_base_url:
            web_base_url = str(endpoint["base_url"]).rstrip("/")
        if not upstream_mode:
            upstream_mode = str(endpoint["mode"]).strip().lower()
        # An explicit body value wins so a custom gateway can be paired with a
        # preset, but the preset default is used when the field is absent.
        model_base_url = (
            explicit_model_base_url
            if explicit_model_base_url is not None
            else str(endpoint.get("model_base_url") or "").rstrip("/")
        )
        preset_web_base = str(endpoint["base_url"]).rstrip("/")
        legacy_gateway = str(endpoint.get("model_base_url") or "").rstrip("/")
        if upstream_mode != str(endpoint["mode"]).strip().lower():
            return JSONResponse(
                {"success": False, "error": "endpoint_id does not match URL/mode"},
                status_code=400,
            )
        # Older consoles persisted only the shared host (for example
        # ``https://trae-api-cn.mchost.guru``) when selecting IDE/SOLO.  The
        # endpoint id is authoritative, so accept that host-only form and
        # normalize it to the concrete Remote path instead of rejecting a
        # harmless migration request.
        try:
            submitted = urlsplit(web_base_url)
            canonical = urlsplit(preset_web_base)
            submitted_is_host_only = (
                bool(submitted.scheme)
                and bool(submitted.netloc)
                and not submitted.path.rstrip("/")
                and submitted.netloc.lower() == canonical.netloc.lower()
            )
        except ValueError:
            submitted_is_host_only = False
        if submitted_is_host_only:
            web_base_url = preset_web_base
        if web_base_url not in (preset_web_base, legacy_gateway) and legacy_gateway:
            return JSONResponse(
                {"success": False, "error": "endpoint_id does not match URL"},
                status_code=400,
            )
        # Old IDE/SOLO stores may have persisted the host-only gateway into
        # the shared web base.  Normalize it back to the Remote base so a
        # subsequent fallback keeps the ``/api/remote/v1`` prefix.
        if web_base_url == legacy_gateway:
            web_base_url = preset_web_base
    elif endpoint_id == "":
        # The custom entry represents a caller-supplied Web/Remote base.  Do
        # not leave the previously selected IDE/Work mode active when the UI
        # submits an empty preset id; that would make the preview and dispatch
        # disagree after the next refresh or restart.
        if not web_base_url:
            return JSONResponse(
                {"success": False, "error": "自定义端点需要 web_base_url"},
                status_code=400,
            )
        upstream_mode = "remote"
        model_base_url = explicit_model_base_url or ""
    elif explicit_model_base_url is not None:
        # No preset identity: the caller manages the model gateway directly.
        model_base_url = explicit_model_base_url
    elif legacy_endpoint_update and (web_base_url or upstream_mode):
        # Older clients only sent URL/mode.  Migrate those updates to the
        # unique preset when possible, and clear a stale shared-URL identity
        # when the update is custom or ambiguous.  Without this step a prior
        # ``raw``/``remote`` id can survive a legacy update and overwrite the
        # newly selected mode after the next restart.
        matches = [
            item
            for item in UPSTREAM_ENDPOINT_PRESETS
            if (
                not web_base_url
                or web_base_url == str(item.get("base_url") or "").rstrip("/")
                or (
                    item.get("model_base_url")
                    and web_base_url
                    == str(item.get("model_base_url") or "").rstrip("/")
                )
            )
            and (
                not upstream_mode
                or upstream_mode == str(item.get("mode") or "").strip().lower()
            )
        ]
        endpoint_id = (
            str(matches[0].get("id") or "").strip().lower()
            if len(matches) == 1
            else ""
        )
        if endpoint_id:
            matched = next(
                (
                    item
                    for item in UPSTREAM_ENDPOINT_PRESETS
                    if str(item.get("id") or "").strip().lower() == endpoint_id
                ),
                None,
            )
            if matched is not None and model_base_url is None:
                model_base_url = str(matched.get("model_base_url") or "").rstrip("/")
        elif web_base_url and model_base_url is None:
            # A custom URL is an explicit Web/Remote base; clear the preset
            # model gateway so it cannot leak into the next selection.
            model_base_url = ""
    relay_port = body.get("relay_port", body.get("port", 0))
    port_requested = "relay_port" in body or "port" in body
    if upstream_mode and upstream_mode not in _VALID_UPSTREAM_MODES:
        return JSONResponse(
            {"success": False, "error": f"Unsupported upstream mode: {upstream_mode}"},
            status_code=400,
        )
    if port_requested:
        if (
            isinstance(relay_port, bool)
            or not isinstance(relay_port, (int, str))
            or not str(relay_port).strip().isdigit()
        ):
            return JSONResponse({"success": False, "error": "relay_port must be an integer between 1 and 65535"}, status_code=400)
        relay_port = int(relay_port)
        if not 1 <= relay_port <= 65535:
            return JSONResponse({"success": False, "error": "relay_port must be between 1 and 65535"}, status_code=400)
    if not web_base_url and not relay_port and not upstream_mode and endpoint_id is None:
        return JSONResponse({"success": False, "error": "nothing to update"}, status_code=400)
    relay_settings = {
        "web_base_url": web_base_url,
        "port": relay_port,
        "upstream_mode": upstream_mode,
    }
    # A port-only call leaves ``model_base_url`` as ``None`` so the adapter
    # keeps the current gateway; endpoint updates always send the resolved
    # value (possibly empty for a custom URL).
    if model_base_url is not None:
        relay_settings["model_base_url"] = model_base_url
    # Keep the new keyword out of port-only calls for compatibility with
    # older store adapters.  Passing ``None`` is semantically equivalent in
    # the current adapter, but legacy wrappers may treat it as an overwrite.
    if endpoint_id is not None:
        relay_settings["endpoint_id"] = endpoint_id
    auth.set_relay_settings(**relay_settings)
    if upstream_mode:
        # Synchronise the running module so _current_upstream_mode() reflects
        # the new preset immediately without a container restart.
        global UPSTREAM_MODE
        UPSTREAM_MODE = upstream_mode
        os.environ["UPSTREAM_MODE"] = upstream_mode
    if web_base_url:
        # Remote clients read this environment value directly. Keep it aligned
        # with the saved preset so endpoint changes take effect immediately.
        global WEB_BASE
        WEB_BASE = web_base_url
        os.environ["TRAE_WEB_BASE_URL"] = web_base_url
    if model_base_url is not None:
        if upstream_mode in {"work", "work-agent"}:
            if model_base_url:
                os.environ["TRAE_RAW_BASE_URL"] = model_base_url
                os.environ["TRAE_WORK_BASE_URL"] = model_base_url
            else:
                os.environ.pop("TRAE_RAW_BASE_URL", None)
                os.environ.pop("TRAE_WORK_BASE_URL", None)
        elif upstream_mode in {"ide", "solo"}:
            if model_base_url:
                os.environ["TRAE_CODE_BASE_URL"] = model_base_url
            else:
                os.environ.pop("TRAE_CODE_BASE_URL", None)
    note = (
        "Relay 端口已保存；需同步 Docker 端口映射并重启容器生效"
        if relay_port
        else "上游端点设置已保存"
    )
    return JSONResponse(
        {
            "success": True,
            "note": note,
            "settings": auth.get_settings(),
        }
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@app.get("/api/auto-route")
async def api_get_auto_route():
    return JSONResponse(
        {
            "success": True,
            **auth.get_auto_route_settings(),
            "tool_endpoint": AUTO_ROUTE_TOOL_ENDPOINT,
            "chat_endpoint": AUTO_ROUTE_CHAT_ENDPOINT,
        }
    )


@app.post("/api/auto-route")
async def api_set_auto_route(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict) or "enabled" not in body:
        return JSONResponse({"success": False, "error": "enabled is required"}, status_code=400)
    settings = auth.set_auto_route_settings(_truthy(body.get("enabled")))
    return JSONResponse({"success": True, **settings})


@app.get("/api/auto-checkin")
async def api_get_auto_checkin():
    return JSONResponse(_auto_checkin_payload(), headers={"Cache-Control": "no-store"})


@app.post("/api/auto-checkin")
async def api_set_auto_checkin(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict) or "enabled" not in body:
        return JSONResponse({"success": False, "error": "enabled is required"}, status_code=400)
    try:
        auth.set_auto_checkin_settings(_truthy(body.get("enabled")), body.get("time"))
    except ValueError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)
    return JSONResponse(_auto_checkin_payload())


@app.post("/api/auto-checkin/run")
async def api_run_auto_checkin():
    if _AUTO_CHECKIN_LOCK.locked():
        return JSONResponse({**_auto_checkin_payload(), "started": False, "error": "already running"})
    asyncio.create_task(_auto_checkin_cycle("manual"))
    await asyncio.sleep(0)
    return JSONResponse({**_auto_checkin_payload(), "started": True})


@app.get("/api/max-mode")
async def api_get_max_mode():
    return JSONResponse({"success": True, **auth.get_max_mode_settings()})


@app.post("/api/max-mode")
async def api_set_max_mode(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict) or "enabled" not in body:
        return JSONResponse({"success": False, "error": "enabled is required"}, status_code=400)
    models = body.get("models", "")
    if models is None:
        models = ""
    if not isinstance(models, (str, list)):
        return JSONResponse(
            {"success": False, "error": "models must be a string or list"}, status_code=400
        )
    settings = auth.set_max_mode_settings(_truthy(body.get("enabled")), models)
    return JSONResponse({"success": True, **settings})


def _max_context_of(config: Mapping[str, Any]) -> int:
    return descriptor_max_context(config) or 0


@app.get("/api/max-mode/models")
async def api_max_mode_models():
    """List Agent-tier models the active account marks with ``max_mode``."""

    account_id, record = auth.get_active_account_snapshot()
    token = str((record or {}).get("token") or "") or str(auth.get_token() or "")
    if not token:
        return JSONResponse({"success": False, "error": "没有可用账号"}, status_code=400)
    provider = (record or {}).get("provider_specific") or (record or {}).get(
        "providerSpecificData"
    )
    configs = await trae_client._fetch_web_model_configs(
        token_override=token,
        provider_specific=dict(provider) if isinstance(provider, Mapping) else None,
        agent_type="solo_agent_remote",
    )
    models = [
        {
            "name": name,
            "display_name": str(cfg.get("display_name") or name),
            "max_context": _max_context_of(cfg),
        }
        for name, cfg in configs.items()
        if isinstance(cfg, Mapping) and descriptor_supports_max_mode(cfg)
    ]
    if not configs:
        return JSONResponse(
            {"success": False, "error": "上游模型列表获取失败"}, status_code=502
        )
    return JSONResponse({"success": True, "account_id": account_id, "models": models})


@app.get("/api/polling")
async def api_get_polling():
    polling = auth.get_polling_status()
    return JSONResponse({"success": True, "enabled": polling.get("enabled", False), "mode": polling.get("mode", "round-robin")})


@app.post("/api/polling-mode")
async def api_polling_mode(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    mode = (body.get("mode") or "").strip()
    if not mode:
        return JSONResponse({"success": False, "error": "mode is required"}, status_code=400)
    if mode not in ("round-robin", "credit-priority"):
        return JSONResponse({"success": False, "error": "mode must be round-robin or credit-priority"}, status_code=400)
    auth.set_polling_mode(mode)
    return JSONResponse({"success": True, "mode": mode})
@app.post("/api/polling")
async def api_polling(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    enabled = bool(body.get("enabled", False))
    auth.set_polling(enabled)
    mode = (body.get("mode") or "").strip()
    if mode in ("round-robin", "credit-priority"):
        auth.set_polling_mode(mode)
    polling = auth.get_polling_status()
    return JSONResponse({"success": True, "enabled": enabled, "mode": polling.get("mode", "round-robin")})
