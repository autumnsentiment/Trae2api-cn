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
  UPSTREAM_MODE=ide  - 只用 trae2api 风格 /api/ide/v1/chat
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
import uuid as uuid_mod
import zlib
from collections import OrderedDict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import dotenv

# Several transport modules cache environment-derived defaults at import time.
# Load the project .env before importing them so local and Docker startup use
# the same configuration order.
dotenv.load_dotenv()

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.responses import FileResponse

from . import (
    auth,
    cli_client,
    raw_client,
    responses_api,
    trae_client,
    trae_remote_client,
    traework_compat,
    traework_native_bridge,
)
from .model_limits import clamp_max_completion_tokens
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
    "remote", "raw", "direct", "auto", "web", "ide", "work-agent",
    "9router", "trae-remote", "cli", "traework-native",
    "native", "traework",
)


def _current_upstream_mode() -> str:
    """Read the upstream mode at request time so preset changes apply immediately."""
    # The module attribute is authoritative.  The startup read already merged
    # env defaults, and runtime preset changes update both env and the module
    # value.  Tests can patch ``src.main.UPSTREAM_MODE`` to isolate scenarios.
    return str(globals().get("UPSTREAM_MODE") or "remote").lower()

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
WEB_BASE = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
UPSTREAM_ENDPOINT_PRESETS = (
    {
        "id": "remote",
        "label": "Remote / chat_sessions",
        "base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
        "mode": "remote",
    },
    {
        "id": "raw",
        "label": "IDE Raw / llm_raw_chat",
        "base_url": "https://trae-api-cn.mchost.guru",
        "mode": "raw",
    },
    {
        "id": "ide",
        "label": "IDE Agent / llm_utils_chat",
        "base_url": "https://trae-api-cn.mchost.guru",
        "mode": "ide",
    },
    {
        "id": "agent",
        "label": "Work Agent / solo_work_remote",
        "base_url": "https://trae-api-cn.mchost.guru/api/remote/v1",
        "mode": "work-agent",
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
_USAGE_MAX_HISTORY = 100
_USAGE_LOCK = threading.RLock()
_USAGE_RECORDS_PATH = Path(
    os.environ.get("TRAE_USAGE_RECORDS_PATH", "")
    or (Path(__file__).resolve().parent.parent / "data" / "usage_records.json")
)
_USAGE_TRACKER: ContextVar[Any] = ContextVar("trae_usage_tracker", default=None)
_USAGE_ENRICH_TASKS: set[asyncio.Task] = set()
_USAGE_SNAPSHOT_TASKS: set[asyncio.Task] = set()
_USAGE_ACTIVE_ACCOUNTS: dict[str, int] = {}
_USAGE_UNSAFE_ACCOUNTS: set[str] = set()
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

# Web login auth
TRAE_AUTH_URL = os.environ.get("TRAE_AUTH_URL", "https://www.trae.cn/authorization")
TRAE_CLIENT_ID = os.environ.get("TRAE_CLIENT_ID") or "ono9krqynydwx5"
LOCAL_LISTENER_PORT = int(os.environ.get("WEB_LOGIN_LISTENER_PORT", "8765"))
PUBLIC_PATHS = {
    "/healthz",
    "/v1/status",
    "/v1/models",
    "/models",
    "/web/login",
    "/web/login/download",
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

    # 根据当前状态显示不同文案
    status_html = f"""
    <div class="status-row">
      <span class="label">状态</span>
      {_status_badge()}
      <span id="active-user-id" class="user-id"{' hidden' if not state.user_id else ''}>{f'用户: {html_mod.escape(state.user_id)}' if state.user_id else ''}</span>
    </div>
    <div class="status-row">
      <span class="label">源</span>
      <code>{html_mod.escape(state.source)}</code>
      <span class="label separator">上游</span>
      <code>{html_mod.escape(AUTO_ROUTE_MODE if auto_route_on else _current_upstream_mode())}</code>
    </div>
    <div class="status-row">
      <span class="label">轮询</span>
      <code>{'开' if polling.get('enabled') else '关'}</code>
      <span class="label separator">账号数</span>
      <code>{polling.get('account_count', 0)}</code>
    </div>"""

    # Consumption history is rendered in its own panel below the account list.
    usage_records_html = """
    <div id="usage-records-container" class="usage-records-container">
      <table class="usage-table">
        <thead>
          <tr>
            <th>时间</th>
            <th>账号</th>
            <th>模型</th>
            <th class="numeric">Tokens（入 / 出 / 总）</th>
            <th class="numeric">消耗积分</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody id="usage-records-body"></tbody>
      </table>
      <div id="usage-empty" class="usage-empty">暂无消费记录</div>
    </div>"""

    # 账号列表
    rows = ""
    for acc in accounts:
        if acc.get("is_valid"):
            st = '<span class="badge badge-ok">有效</span>'
        else:
            st = '<span class="badge badge-expired">无效</span>'
        act = (
            '<span class="badge badge-active" data-account-active>当前</span>'
            if acc.get("is_active")
            else '<span class="badge badge-active" data-account-active hidden>当前</span>'
        )
        active_row = " active-row" if acc.get("is_active") else ""
        switch_disabled = " disabled" if acc.get("is_active") else ""
        aid = acc.get("id") or ""
        label = acc.get("label") or acc.get("user_id") or aid
        uid = acc.get("user_id") or aid
        expires = (acc.get("expires") or "")[:16]
        account_credits = acc.get("account_credits") or {}
        if account_credits.get("unlimited"):
            credits_text = "☆ 无限"
        elif account_credits.get("remaining") is not None:
            credits_text = f"剩{float(account_credits['remaining']):.2f}/总{float(account_credits.get('total_limit') or 0):.2f}"
        else:
            credits_text = "-"
        credits = acc.get("credits")
        checked_in = acc.get("checked_in")
        if checked_in is True:
            checkin_badge = '<span class="badge badge-ok">已签到</span>'
        elif checked_in is False:
            checkin_badge = '<span class="badge badge-active">未签到</span>'
        else:
            checkin_badge = '<span class="badge badge-none">未知</span>'
        rows += f"""<tr id="row-{html_mod.escape(aid)}" class="{active_row.strip()}" data-account-id="{html_mod.escape(aid)}">
          <td><strong id="label-{html_mod.escape(aid)}">{html_mod.escape(label)}</strong><small class="row-subtitle">{html_mod.escape(uid)}</small></td>
          <td><code>{html_mod.escape(uid)}</code></td>
          <td>{st} {act}</td>
          <td class="muted-cell">{html_mod.escape(expires)}</td>
          <td><span id="general-credits-{html_mod.escape(aid)}" class="credit-value">{credits_text}</span></td>
          <td><span id="checkin-{html_mod.escape(aid)}" class="checkin-state">{checkin_badge}</span><small id="checkin-detail-{html_mod.escape(aid)}" class="row-subtitle"></small></td>
          <td class="row-actions">
            <button class="btn btn-ghost btn-sm" data-action="checkin" onclick="checkinAccount('{html_mod.escape(aid)}')" title="签到">签到</button>
            <button class="btn btn-ghost btn-sm" data-action="switch-account" onclick="switchAccount('{html_mod.escape(aid)}')" title="切换当前账号"{switch_disabled}>切换</button>
            <button class="btn btn-danger btn-sm" onclick="removeAccount('{html_mod.escape(aid)}')" title="删除账号">删除</button>
          </td>
        </tr>"""
    if accounts:
        accounts_html = f"""<div class="form-group">
          <div class="section-head"><div><label>账号列表（{len(accounts)}）</label><span id="checkin-summary" class="section-meta">等待查询</span></div><span id="checkin-updated" class="section-meta"></span></div>
          <div class="btn-group account-toolbar" aria-live="polite">
            <button class="btn btn-secondary btn-sm" id="checkin-status-refresh-btn" onclick="checkinRefreshAll()">查询签到状态</button>
            <button class="btn btn-secondary btn-sm" id="credits-refresh-btn" onclick="creditsRefreshAll()">查询全部积分</button>
            <button class="btn btn-primary btn-sm" id="checkin-claim-btn" onclick="checkinClaimAll()">一键轮询签到</button>
            <span id="account-msg" class="msg inline-msg" role="status" aria-live="polite"></span>
            <span id="checkin-msg" class="msg inline-msg" role="status" aria-live="polite"></span>
            <span id="checkin-busy" class="busy-indicator" role="status" aria-live="polite">正在处理...</span>
          </div>
          <table class="acct-table">
            <thead><tr><th>账号</th><th>用户ID</th><th>状态</th><th>有效期</th><th>通用积分</th><th>签到状态</th><th>操作</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""
    else:
        # Keep the account actions in the DOM even before the first login.
        # This gives the console a stable control surface and lets the same
        # frontend code handle an account list that becomes populated after a
        # login without requiring a page-specific script branch.
        accounts_html = '''
        <p class="card-hint">暂无账号，请先登录或手动添加。</p>
        <div class="account-toolbar" hidden>
          <button class="btn btn-secondary btn-sm" id="checkin-status-refresh-btn" onclick="checkinRefreshAll()">查询签到状态</button>
          <button class="btn btn-secondary btn-sm" id="credits-refresh-btn" onclick="creditsRefreshAll()">查询全部积分</button>
          <button class="btn btn-primary btn-sm" id="checkin-claim-btn" onclick="checkinClaimAll()">一键轮询签到</button>
        </div>'''

    logout_btn = ''
    if state.token:
        logout_btn = '<button class="btn btn-ghost btn-sm" onclick="logout()">登出</button>'

    settings_web = settings.get("web_base_url") or WEB_BASE
    settings_port = settings.get("relay_port") or PORT
    matched_endpoint = next(
        (
            item
            for item in UPSTREAM_ENDPOINT_PRESETS
            if settings_web == item["base_url"]
        ),
        None,
    )
    # Custom URLs that aren't in the preset list still appear as saved options.
    custom_is_saved = bool(settings.get("web_base_url")) and not matched_endpoint
    custom_selected = " selected" if not matched_endpoint else ""
    endpoint_options = [f'<option value=""{custom_selected}>自定义</option>']
    if custom_is_saved:
        endpoint_options.append(
            f'<option value="__saved" selected>{html_mod.escape(settings_web)}</option>'
        )
    for item in UPSTREAM_ENDPOINT_PRESETS:
        selected = " selected" if matched_endpoint and matched_endpoint["id"] == item["id"] else ""
        endpoint_options.append(
            f'<option value="{html_mod.escape(item["id"])}"{selected}>{html_mod.escape(item["label"])}</option>'
        )
    endpoint_options_html = "".join(endpoint_options)
    endpoint_map_json = json.dumps(
        {
            item["id"]: {"base_url": item["base_url"], "mode": item["mode"]}
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
    auto_checkin = auth.get_auto_checkin_settings()
    auto_checkin_checked = 'checked' if auto_checkin.get('enabled') else ''
    auto_checkin_time = html_mod.escape(auto_checkin.get('time') or '08:30')
    auto_checkin_state_text = (
        f"每天 {auto_checkin_time} 自动签到" if auto_checkin.get('enabled') else '已关闭'
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trae CN Relay 控制台</title>
<style>
:root {{
  --bg: #eef0f4;
  --panel: #ffffff;
  --panel2: #f6f7fa;
  --border: #e3e6ec;
  --border-strong: #c9cfdb;
  --text: #1d2331;
  --muted: #5b6474;
  --faint: #8b93a5;
  --accent: #0d8a5f;
  --accent-strong: #0a7451;
  --accent-soft: #e2f3eb;
  --accent-border: #bfe3d3;
  --info: #2563eb;
  --info-soft: #e9effd;
  --info-border: #c8d8f8;
  --warn: #b45309;
  --warn-soft: #fdf1e2;
  --warn-border: #f3dcb8;
  --danger: #dc2626;
  --danger-soft: #fdecec;
  --danger-border: #f3c6c6;
  --sidebar-bg: #181b25;
  --sidebar-hover: #232838;
  --sidebar-active: #272d40;
  --sidebar-text: #a7aebf;
  --sidebar-border: #262b3a;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--text); min-height: 100vh;
  font-size: 14px; line-height: 1.5;
}}
.app {{ display: flex; min-height: 100vh; }}
.sidebar {{
  width: 232px; flex-shrink: 0; background: var(--sidebar-bg); color: var(--sidebar-text);
  display: flex; flex-direction: column; position: sticky; top: 0; height: 100vh;
  overflow-y: auto;
}}
.brand {{ display: flex; align-items: center; gap: 10px; padding: 18px 16px 16px; border-bottom: 1px solid var(--sidebar-border); }}
.brand-mark {{
  width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
  background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; letter-spacing: 0;
}}
.brand-block {{ display: flex; flex-direction: column; min-width: 0; }}
.brand-block h1 {{ font-size: 15px; font-weight: 650; color: #f2f4f8; letter-spacing: 0; white-space: nowrap; }}
.brand-sub {{ color: #79839a; font-size: 11px; margin-top: 1px; }}
.nav-list {{ display: flex; flex-direction: column; gap: 2px; padding: 12px; }}
.nav-item {{
  display: flex; align-items: center; gap: 10px; width: 100%;
  border: none; background: transparent; color: var(--sidebar-text);
  font-size: 13px; font-weight: 550; padding: 9px 12px; border-radius: 8px; cursor: pointer;
  transition: background .15s, color .15s; font-family: inherit; text-align: left;
}}
.nav-item svg {{ width: 16px; height: 16px; flex-shrink: 0; opacity: .9; }}
.nav-item:hover {{ color: #e8ebf2; background: var(--sidebar-hover); }}
.nav-item.active {{ color: #fff; background: var(--sidebar-active); box-shadow: inset 2px 0 0 var(--accent); }}
.sidebar-status {{ padding: 12px 16px; border-top: 1px solid var(--sidebar-border); margin-top: auto; }}
.sidebar-foot {{ padding: 12px 16px 16px; border-top: 1px solid var(--sidebar-border); }}
.main {{ flex: 1; min-width: 0; padding: 24px 28px 48px; }}
.main-inner {{ max-width: 1180px; margin: 0 auto; }}
.status-strip {{ display: flex; flex-direction: column; gap: 7px; font-size: 12px; color: var(--sidebar-text); }}
.status-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.status-row .label {{ color: #6d7690; min-width: 30px; }}
.status-row .separator {{ margin-left: 4px; }}
.status-row code {{ background: #232838; padding: 1px 6px; border-radius: 4px; font-size: 11px; color: #c3c9d8; }}
.user-id {{ color: #7db3f5; font-size: 12px; }}
.tab-page {{ display: none; }}
.tab-page.active {{ display: block; }}
.panel-grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
.panel-grid.cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: stretch; }}
.panel-stack {{ display: flex; flex-direction: column; gap: 14px; }}
.login-layout {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px 24px; align-items: start; }}
.login-steps {{ margin: 0; padding-left: 18px; font-size: 12px; color: var(--muted); line-height: 1.7; }}
.login-steps code {{ background: var(--panel2); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; font-size: 11px; color:#39415a; }}
.login-steps a {{ color: var(--info); }}
.login-actions {{ display: flex; flex-direction: column; align-items: flex-end; }}
.login-actions .btn-group {{ margin-top: 0; justify-content: flex-end; }}
.login-actions .btn-group + .btn-group {{ margin-top: 8px; }}
.manual-add {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }}
.manual-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0 12px; margin-top: 12px; }}
.manual-grid .span-all {{ grid-column: 1 / -1; }}
.manual-submit {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.manual-submit .msg {{ margin-top: 0; }}
.panel-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px 20px;
  min-width: 0;
  overflow-x: auto;
  box-shadow: 0 1px 2px rgba(20,26,40,.05);
}}
.section-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
.section-head > div {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.section-meta {{ color:var(--faint); font-size:12px; font-weight:400; }}
.badge {{ display:inline-flex; align-items:center; gap:4px; font-size: 12px; padding: 3px 9px; border-radius: 999px; font-weight: 600; white-space:nowrap; }}
[hidden] {{ display:none !important; }}
.badge-ok {{ background: var(--accent-soft); color: var(--accent-strong); border:1px solid var(--accent-border); }}
.badge-expired {{ background: var(--warn-soft); color: var(--warn); border:1px solid var(--warn-border); }}
.badge-none {{ background: var(--panel2); color: var(--muted); border:1px solid var(--border); }}
.badge-active {{ background: var(--info-soft); color: var(--info); border:1px solid var(--info-border); }}
hr {{ border: none; border-top: 1px solid var(--border); margin: 16px 0; }}
.btn {{
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  padding: 9px 18px; border-radius: 8px; font-size: 13px; font-weight: 550;
  cursor: pointer; border: 1px solid transparent; transition: all .15s;
  text-decoration: none; min-height: 32px;
}}
.btn-sm {{ padding: 5px 10px; font-size: 12px; min-height: 26px; }}
.btn-primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
.btn-primary:hover {{ background: var(--accent-strong); border-color: var(--accent-strong); }}
.btn-primary:disabled {{ opacity: .5; cursor: not-allowed; }}
.btn-secondary {{ background: var(--panel); border-color: var(--border-strong); color: var(--text); }}
.btn-secondary:hover {{ background: var(--panel2); }}
.btn-ghost {{ background: transparent; border-color: var(--border-strong); color: var(--muted); }}
.btn-ghost:hover {{ border-color: #9aa5b5; color: var(--text); }}
.btn-danger {{ background: transparent; border-color: var(--danger-border); color: var(--danger); }}
.btn-danger:hover {{ background: var(--danger-soft); border-color: #e59a9e; }}
.btn-group {{ display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }}
.account-toolbar {{ align-items:center; margin: 10px 0 14px; }}
.account-toolbar .inline-msg {{ margin:0; flex:1 1 260px; }}
.form-group {{ margin-bottom: 10px; }}
.form-group label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
.form-group input, .form-group textarea, .form-group select {{
  width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border-strong);
  background: var(--panel); color: var(--text); font-size: 13px; font-family: "SF Mono", Consolas, monospace;
}}
.form-group select {{ font-family: inherit; }}
.form-group textarea {{ resize: vertical; min-height: 60px; }}
.form-group input:focus, .form-group textarea:focus, .form-group select:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(13,138,95,.12); }}
.form-inline {{ display:flex; gap:12px; flex-wrap:wrap; }}
.form-inline .form-group {{ flex:1 1 220px; }}
.acct-table {{ width: 100%; min-width: 920px; border-collapse: collapse; font-size: 13px; }}
.acct-table th, .acct-table td {{ text-align: left; padding: 9px 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
.acct-table tbody tr {{ transition: background .15s ease; }}
.acct-table tbody tr:hover {{ background: var(--panel2); }}
.acct-table tbody tr.active-row {{ background: var(--accent-soft); box-shadow:inset 3px 0 var(--accent); }}
.acct-table tbody tr.active-row:hover {{ background:#d5ecdf; }}
.acct-table tbody tr.row-failed {{ background: var(--danger-soft); }}
.acct-table tbody tr.row-failed:hover {{ background:#fbdddd; }}
.acct-table th {{ color: var(--muted); font-weight: 600; font-size: 12px; position:sticky; top:0; background:var(--panel); z-index:1; }}
.acct-table th:nth-child(5), .acct-table td:nth-child(5) {{ text-align:right; }}
.credit-value {{ font-variant-numeric: tabular-nums; white-space:nowrap; color:#39415a; }}
.muted-cell {{ color:var(--faint); font-size:12px; white-space:nowrap; }}
.row-subtitle {{ display:block; color:var(--faint); font-size:11px; margin-top:3px; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.row-actions {{ white-space:nowrap; }}
.row-actions .btn {{ margin:2px 0; }}
.usage-records-container {{ max-height: 320px; overflow: auto; }}
.usage-table {{ width: 100%; min-width: 760px; border-collapse: collapse; font-size: 13px; table-layout: fixed; }}
.usage-table th, .usage-table td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); overflow-wrap: anywhere; }}
.usage-table th {{ color: var(--muted); font-weight: 500; font-size: 12px; position: sticky; top: 0; background: var(--panel); }}
.usage-table .numeric {{ text-align: right; font-variant-numeric:tabular-nums; }}
.usage-table .usage-status {{ white-space:nowrap; }}
.usage-empty {{ padding: 18px 12px; color: var(--faint); text-align: center; font-size: 13px; }}
.msg {{ margin-top: 12px; padding: 8px 12px; border-radius: 6px; font-size: 13px; display: none; line-height:1.45; }}
.msg-ok {{ background: var(--accent-soft); color: var(--accent-strong); display: block; border:1px solid var(--accent-border); }}
.msg-err {{ background: var(--danger-soft); color: var(--danger); display: block; border:1px solid var(--danger-border); }}
.busy-indicator {{ display:none; color:var(--muted); font-size:12px; align-items:center; gap:6px; }}
.busy-indicator.visible {{ display:inline-flex; }}
.busy-indicator::before {{ content:""; width:10px; height:10px; border:2px solid var(--border-strong); border-top-color:var(--accent); border-radius:50%; animation:relay-spin .7s linear infinite; }}
@keyframes relay-spin {{ to {{ transform:rotate(360deg); }} }}
.toast {{ position:fixed; top:20px; right:20px; z-index:20; width:min(420px,calc(100vw - 40px)); padding:12px 14px; border:1px solid var(--border); border-radius:8px; background:var(--panel); color:var(--text); box-shadow:0 12px 32px rgba(20,26,40,.16); opacity:0; transform:translateY(-8px); pointer-events:none; transition:opacity .18s ease, transform .18s ease; white-space:pre-wrap; line-height:1.45; }}
.toast.visible {{ opacity:1; transform:translateY(0); }}
.toast.ok {{ border-color:var(--accent-border); }}
.toast.error {{ border-color:var(--danger-border); color:var(--danger); background:var(--danger-soft); }}
.toast-title {{ display:block; font-size:12px; font-weight:700; margin-bottom:3px; color:var(--accent-strong); }}
.toast.error .toast-title {{ color:var(--danger); }}
.busy {{ opacity:.65; pointer-events:none; }}
.loading {{ margin-top: 12px; display: none; font-size: 13px; color: var(--muted); }}
.section-title {{ font-size: 13px; font-weight: 650; color: var(--text); margin: 0 0 10px; letter-spacing: 0; }}
.section-title-block {{ font-size: 13px; font-weight: 650; color: var(--text); margin: 18px 0 8px; padding-top: 14px; border-top: 1px solid var(--border); }}
.card-hint {{ font-size:12px; color:var(--muted); line-height:1.6; margin-bottom:10px; }}
.card-hint code {{ background: var(--panel2); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; font-size: 11px; color:#39415a; }}
.card-hint a {{ color:var(--info); }}
.check-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }}
.check-row .label {{ color:var(--faint); }}
.check-row + .check-row {{ margin-top: 8px; }}
.check-row label {{ cursor: pointer; }}
.check-row input[type=checkbox], .check-row input[type=radio] {{ accent-color: var(--accent); width: 15px; height: 15px; flex-shrink: 0; }}
.radio-group {{ display: inline-flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.inline-check {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); font-weight: 400; cursor: pointer; white-space: nowrap; }}
.inline-check input[type=checkbox] {{ accent-color: var(--accent); width: 15px; height: 15px; }}
.form-group .inline-check {{ display: inline-flex; margin-bottom: 0; font-size: 13px; }}
.form-group .inline-field select, .form-group .inline-field input {{ width: auto; padding: 4px 8px; font-family: inherit; }}
.form-group .inline-field input[type=number] {{ width: 76px; }}
.option-row {{ display: flex; gap: 10px 18px; align-items: center; flex-wrap: wrap; }}
.field-spaced {{ margin-top: 12px; }}
.field-note {{ font-size: 12px; color: var(--faint); margin-top: 8px; }}
.schedule-row {{ display: flex; align-items: center; gap: 10px 16px; flex-wrap: wrap; }}
.schedule-row .time-field {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }}
.schedule-row input[type=time] {{ padding: 5px 8px; border-radius: 6px; border: 1px solid var(--border-strong); background: var(--panel); color: var(--text); font-size: 13px; font-family: inherit; min-width: 108px; }}
.schedule-row input[type=time]:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(13,138,95,.12); }}
.schedule-row .btn-group {{ margin-top: 0; }}
.schedule-status {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }}
.schedule-status div {{ min-width: 0; }}
.schedule-status dt {{ font-size: 11px; color: var(--faint); }}
.schedule-status dd {{ font-size: 13px; color: var(--text); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
.conn-table {{ width: 100%; margin-top: 14px; border-collapse: collapse; font-size: 12px; }}
.conn-table th {{ text-align: left; color: var(--muted); font-weight: 600; padding: 7px 6px; border-bottom: 1px solid var(--border); }}
.conn-table td {{ padding: 7px 6px; border-bottom: 1px solid var(--border); vertical-align: top; }}
.conn-table td.conn-detail {{ color: var(--muted); overflow-wrap: anywhere; }}
.conn-table .conn-reasoning {{ margin-top: 4px; color: var(--muted); white-space: pre-wrap; }}
.conn-status.ok {{ color: var(--accent); font-weight: 600; }}
.conn-status.fail {{ color: var(--danger); font-weight: 600; }}
.conn-status.pending {{ color: var(--muted); }}
.max-models {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; font-size:12px; color:var(--muted); }}
.max-models[hidden] {{ display:none; }}
.model-chip {{ border:1px solid var(--border-strong); background:var(--panel); color:var(--text); border-radius:6px; padding:4px 8px; font-size:12px; cursor:pointer; font-family:inherit; }}
.model-chip:hover {{ border-color:var(--accent); color:var(--accent); }}
pre.code-out {{ margin-top:12px; padding:12px; background:var(--panel2); border:1px solid var(--border); border-radius:6px; font-size:12px; max-height:220px; overflow:auto; white-space:pre-wrap; color:#39415a; display:none; }}
details summary {{ font-size:13px; color:var(--muted); cursor:pointer; }}
@media (max-width: 960px) {{
  .panel-grid.cols-2 {{ grid-template-columns: 1fr; }}
  .login-layout {{ grid-template-columns: 1fr; }}
  .login-actions {{ align-items: stretch; }}
  .login-actions .btn-group {{ justify-content: flex-start; }}
  .manual-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .schedule-status {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 820px) {{
  .app {{ flex-direction: column; }}
  .sidebar {{ width: 100%; height: auto; position: static; }}
  .nav-list {{ flex-direction: row; overflow-x: auto; padding: 8px 12px; }}
  .nav-item {{ width: auto; white-space: nowrap; }}
  .sidebar-status {{ margin-top: 0; border-top: 1px solid var(--sidebar-border); }}
  .sidebar-foot {{ border-top: none; }}
  .main {{ padding: 16px 14px 36px; }}
  .panel-card {{ padding: 14px 12px; }}
  .btn {{ padding:8px 12px; }}
  .btn-sm {{ padding:6px 9px; }}
  .account-toolbar {{ align-items:stretch; }}
  .account-toolbar .btn {{ flex:1 1 150px; }}
  .account-toolbar .inline-msg {{ flex-basis:100%; }}
  .usage-table {{ min-width:560px; }}
  .manual-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div id="toast" class="toast" role="alert" aria-live="assertive"><span id="toast-title" class="toast-title"></span><span id="toast-text"></span></div>
<div class="app">
<aside class="sidebar">
  <div class="brand">
    <span class="brand-mark">TR</span>
    <div class="brand-block">
      <h1>Trae CN Relay</h1>
      <span class="brand-sub">控制台</span>
    </div>
  </div>
  <nav class="nav-list" role="tablist">
    <button class="nav-item active" data-tab="accounts" onclick="switchTab('accounts')">账号与签到</button>
    <button class="nav-item" data-tab="usage" onclick="switchTab('usage')">消费记录</button>
    <button class="nav-item" data-tab="settings" onclick="switchTab('settings')">轮询与设置</button>
    <button class="nav-item" data-tab="models" onclick="switchTab('models')">模型测试</button>
  </nav>
  <div class="sidebar-status">
    <div class="status-strip">{status_html}</div>
  </div>
  <div class="sidebar-foot">
    {logout_btn}
  </div>
</aside>
<main class="main">
<div class="main-inner">
<div class="tab-page active" data-page="accounts">
<div class="panel-stack">
<section class="panel-card" aria-labelledby="login-title">
<div class="section-head"><div class="section-title" id="login-title">授权登录</div></div>
<div class="login-layout">
  <ol class="login-steps">
    <li>点击授权登录，自动检测本机授权助手（<code>127.0.0.1:{listener_port}</code>）。</li>
    <li>未检测到时，下载 <code>web_login.py</code> 或 <code>start_auth.bat</code> 在<b>本机</b>运行后重试。</li>
    <li>确保浏览器已登录 <a href="https://www.trae.cn" target="_blank" rel="noopener">trae.cn</a>，授权完成后凭据自动写入服务器。</li>
  </ol>
  <div class="login-actions">
    <div class="btn-group">
      <button class="btn btn-primary" onclick="startAuth()" id="auth-btn">使用 Trae 网页授权登录</button>
      <a class="btn btn-ghost" href="https://www.trae.cn" target="_blank" rel="noopener">访问 trae.cn</a>
    </div>
    <div class="btn-group">
      <a class="btn btn-secondary" href="/web/login/download" download>下载授权助手 web_login.py</a>
      <a class="btn btn-ghost" href="/web/login/download?as=bat" download id="bat-link" style="display:none">下载 start_auth.bat</a>
    </div>
  </div>
</div>
<div id="loading" class="loading">等待授权中...</div>
<div id="auth-msg" class="msg"></div>
<details class="manual-add">
  <summary>手动填写凭证添加账号</summary>
  <form id="manual-form" class="manual-grid">
    <div class="form-group span-all">
      <label for="manual-token">Token（Cloud-IDE-JWT）<span style="color:var(--danger)">*</span></label>
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
      <label for="manual-label">备注（标签）</label>
      <input id="manual-label" name="label" placeholder="可选">
    </div>
    <div class="span-all manual-submit">
      <button type="submit" class="btn btn-primary">添加账号</button>
      <div id="manual-msg" class="msg"></div>
    </div>
  </form>
</details>
</section>
<section class="panel-card" id="auto-checkin-panel" aria-labelledby="auto-checkin-title">
  <div class="section-head">
    <div class="section-title" id="auto-checkin-title">自动签到</div>
    <span id="auto-checkin-state" class="section-meta">{auto_checkin_state_text}</span>
  </div>
  <div class="schedule-row">
    <label class="inline-check" for="auto-checkin-toggle">
      <input type="checkbox" id="auto-checkin-toggle" {auto_checkin_checked}> 启用定时签到
    </label>
    <label class="time-field" for="auto-checkin-time">每天（北京时间）
      <input type="time" id="auto-checkin-time" value="{auto_checkin_time}" step="60" required>
    </label>
    <div class="btn-group">
      <button class="btn btn-primary btn-sm" id="auto-checkin-save-btn" onclick="saveAutoCheckin()">保存</button>
      <button class="btn btn-secondary btn-sm" id="auto-checkin-run-btn" onclick="runAutoCheckinNow()" title="立即按顺序签到所有未签到账号，已签到账号自动跳过">立即执行</button>
    </div>
  </div>
  <dl class="schedule-status">
    <div><dt>下次执行</dt><dd id="auto-checkin-next">-</dd></div>
    <div><dt>上次执行</dt><dd id="auto-checkin-last">-</dd></div>
    <div><dt>上次结果</dt><dd id="auto-checkin-result">-</dd></div>
  </dl>
  <div id="auto-checkin-msg" class="msg" role="status" aria-live="polite"></div>
</section>
<section class="panel-card" aria-label="账号列表">
<div class="section-title">账号列表</div>
{accounts_html}
</section>
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
<div class="section-title">多账号轮询</div>
<div class="check-row">
  <input type="checkbox" id="poll-toggle" {poll_checked} onchange="togglePolling()">
  <label for="poll-toggle">启用轮询（每次请求自动切换下一个有效账号）</label>
</div>
<div class="check-row">
  <span class="label">轮询模式</span>
  <span class="radio-group">
    <label class="inline-check"><input type="radio" name="poll-mode" value="round-robin" onchange="togglePolling()" {poll_mode_rr}> 顺序轮询</label>
    <label class="inline-check"><input type="radio" name="poll-mode" value="credit-priority" onchange="togglePolling()" {poll_mode_cp}> 积分优先</label>
  </span>
</div>
<p id="poll-status" class="field-note">当前账号数: {polling.get('account_count', 0)}，轮询: {'开' if polling.get('enabled') else '关'}</p>
</div>
<div class="panel-card">
  <div class="section-title">上游端点</div>
<div class="form-group">
  <label>预设端点</label>
  <select id="settings-endpoint-preset" onchange="applyEndpointPreset()">
    <option value="">自定义</option>
    {endpoint_options_html}
  </select>
</div>
<div class="form-group">
  <label>自定义 Web Base URL</label>
  <input id="settings-web" value="{html_mod.escape(settings_web)}" placeholder="https://trae-api-cn.mchost.guru/api/remote/v1">
</div>
<div class="form-group">
  <label>Relay 端口（需重启容器生效）</label>
  <input id="settings-port" type="number" value="{settings_port}" placeholder="8000">
</div>
<div class="btn-group">
  <button class="btn btn-secondary" onclick="saveSettings()">保存设置</button>
</div>
<div id="settings-msg" class="msg"></div>
</div>
<div class="panel-card" id="auto-route-panel">
  <div class="section-head">
    <div class="section-title">自动路由</div>
    <span id="auto-route-state" class="section-meta">{auto_route_state_text}</span>
  </div>
  <div class="check-row">
    <input type="checkbox" id="auto-route-toggle" {auto_route_checked} onchange="saveAutoRoute()">
    <label for="auto-route-toggle" title="开启后忽略上方预设端点的模式选择；端点失败时回落 Remote">启用自动路由（工具调用走 IDE Agent，纯聊天走 Remote，失败回落 Remote）</label>
  </div>
  <div id="auto-route-msg" class="msg" role="status" aria-live="polite"></div>
</div>
<div class="panel-card" id="max-mode-panel">
  <div class="section-head">
    <div class="section-title">1M 上下文（Max 模式）</div>
    <span id="max-mode-state" class="section-meta">{max_state_text}</span>
  </div>
  <div class="check-row">
    <input type="checkbox" id="max-mode-toggle" {max_checked}>
    <label for="max-mode-toggle" title="只对 Remote 的 Agent 会话生效；带调用端工具的请求默认走 Work，不使用 Max">启用 Max 模式（Remote Agent 会话使用 1M 上下文）</label>
  </div>
  <div class="form-group field-spaced">
    <label for="max-mode-models">生效模型（逗号分隔，留空表示账号中所有支持 Max 的模型）</label>
    <input id="max-mode-models" value="{max_models}" placeholder="glm-5.3, deepseek-v4-pro">
  </div>
  <div class="btn-group">
    <button class="btn btn-secondary" onclick="saveMaxMode()">保存</button>
    <button class="btn btn-ghost" id="max-mode-detect-btn" onclick="detectMaxModels()">检测支持的模型</button>
  </div>
  <div id="max-mode-models-out" class="max-models" hidden></div>
  <div id="max-mode-msg" class="msg" role="status" aria-live="polite"></div>
</div>
</div>
</div>
<div class="tab-page" data-page="models">
<div class="panel-grid">
<div class="panel-card">
<div class="section-title">模型列表</div>
<div class="form-group">
  <label>刷新 /v1/models（TRAE_FETCH_MODEL_LIST=true 时从上游拉取，否则返回内置列表）</label>
</div>
<div class="btn-group">
  <button class="btn btn-secondary" onclick="refreshModels()">获取模型列表</button>
</div>
<pre id="models-out" class="code-out"></pre>
<div id="models-msg" class="msg"></div>
</div>
<div class="panel-card" id="conn-panel">
  <div class="section-head">
    <div class="section-title">模型连通性测试</div>
    <span id="conn-summary" class="section-meta">未运行</span>
  </div>
  <div class="form-group">
    <label for="conn-models">测试模型（逗号或换行分隔，留空使用下方常用模型）</label>
    <textarea id="conn-models" rows="2" placeholder="glm-5.3, DeepSeek-V4-Pro-Official"></textarea>
  </div>
  <div class="form-group">
    <label>测试内容</label>
    <div class="option-row">
      <label class="inline-check">
        <input type="checkbox" id="conn-mode-text" checked> 文本回复
      </label>
      <label class="inline-check">
        <input type="checkbox" id="conn-mode-tool"> 工具调用
      </label>
      <label class="inline-check inline-field">
        思考强度
        <select id="conn-effort" title="映射到 Trae custom_model.reasoning_effort：low=light，medium/high=high，xhigh=extra_high，超出模型支持的档位会向下取">
          <option value="">默认（不传）</option>
          <option value="low">low / 轻</option>
          <option value="medium">medium / 高</option>
          <option value="high">high / 高</option>
          <option value="xhigh">xhigh / 极高</option>
        </select>
      </label>
      <label class="inline-check">
        <input type="checkbox" id="conn-thinking"> 返回思考内容
      </label>
      <label class="inline-check" title="请求 1M Max 上下文，结果中显示是否实际生效">
        <input type="checkbox" id="conn-max"> 1M Max
      </label>
      <label class="inline-check inline-field">
        超时(秒) <input type="number" id="conn-timeout" value="120" min="10" max="600">
      </label>
    </div>
  </div>
  <div class="form-group">
    <label for="conn-endpoint">指定上游（测试期间禁止跨端点回落）</label>
    <select id="conn-endpoint">
      <option value="auto-route">自动路由（工具 IDE Agent / 聊天 Remote）</option>
      <option value="remote">Remote / chat_sessions</option>
      <option value="raw">IDE Raw / llm_raw_chat</option>
      <option value="ide">IDE Agent / llm_utils_chat</option>
      <option value="work-agent">Work Agent / solo_work_remote</option>
    </select>
  </div>
  <div class="btn-group">
    <button class="btn btn-secondary" id="conn-run-btn" onclick="runConnTest()">开始测试</button>
    <button class="btn btn-secondary" onclick="fillConnPreset()">填入常用模型</button>
  </div>
  <table id="conn-table" class="conn-table" hidden>
    <thead><tr><th>模型</th><th>类型</th><th>结果</th><th>耗时</th><th>详情</th></tr></thead>
    <tbody id="conn-tbody"></tbody>
  </table>
  <div id="conn-msg" class="msg"></div>
</div>
</div>
</div>
</div>
</main>
</div>
<script>
function switchTab(name){{
  document.querySelectorAll('.nav-item').forEach(function(b){{ b.classList.toggle('active', b.getAttribute('data-tab')===name); }});
  document.querySelectorAll('.tab-page').forEach(function(p){{ p.classList.toggle('active', p.getAttribute('data-page')===name); }});
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
async function checkLocalListener() {{
  try {{
    var ctrl = new AbortController();
    var timer = setTimeout(function(){{ ctrl.abort(); }}, 1200);
    var r = await fetch('http://127.0.0.1:{listener_port}/healthz', {{signal: ctrl.signal, cache: 'no-store'}});
    clearTimeout(timer);
    return r.ok;
  }} catch(e) {{ return false; }}
}}
async function startAuth() {{
  try {{
  var ok = await checkLocalListener();
  if (!ok) {{
    showMsg('auth-msg', '未检测到本机授权助手，请先下载并双击运行 web_login.py（需 Python）或 start_auth.bat：', false);
    document.getElementById('bat-link').style.display='inline-flex';
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
        var input=Number(record.input_tokens!==undefined?record.input_tokens:(record.prompt_tokens||0));
        var output=Number(record.output_tokens!==undefined?record.output_tokens:(record.completion_tokens||0));
        var total=Number(record.total_tokens!==undefined?record.total_tokens:(input+output));
        var tokenText=record.tokens_source==='unknown'?'--':(input+' / '+output+' / '+total);
        var credits=record.credits_consumed;
        var creditText=credits===null||credits===undefined?'--':Number(credits).toFixed(2);
        var source=record.credits_source||'unknown';
        var status=record.status||'completed';
        var statusText=status==='completed'?'完成':(status==='cancelled'?'已取消':(status==='error'?'失败':status));
        var badge=status==='completed'?'badge-ok':(status==='error'?'badge-expired':'badge-none');
        return '<tr>'
          + '<td>'+escapeHtml(when)+'</td>'
          + '<td><code>'+escapeHtml(account)+'</code></td>'
          + '<td>'+escapeHtml(model)+'</td>'
          + '<td class="numeric">'+escapeHtml(tokenText)+'</td>'
          + '<td class="numeric" title="'+escapeHtml(source)+'">'+escapeHtml(creditText)+'</td>'
          + '<td class="usage-status"><span class="badge '+badge+'">'+escapeHtml(statusText)+'</span></td>'
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
  var values=[['general-credits-',data&&data.account_credits]];
  values.forEach(function(pair){{
    var el=document.getElementById(pair[0]+id), value=pair[1];
    if(!el) return;
    if(value===undefined||value===null) return;
    if(value.unlimited) el.textContent='☆ 无限';
    else if(value.remaining!==undefined&&value.remaining!==null) el.textContent='剩'+Number(value.remaining).toFixed(2)+'/总'+(value.total_limit===undefined?'?':Number(value.total_limit).toFixed(2));
    else el.textContent='-';
  }});
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
  var passed = 0, done = 0;
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
          reasoning_effort: effort, thinking: thinking, max_mode: maxMode}}, (timeout + 20) * 1000);
      timeCell.textContent = (d.elapsed_ms !== undefined ? d.elapsed_ms + ' ms' : '-');
      if(d.success){{
        passed++;
        statusCell.textContent = '通过';
        statusCell.className = 'conn-status ok';
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
        detailCell.textContent = String(d.error || 'unknown');
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
}}
async function refreshModels(){{
  var el=document.getElementById('models-out');
  var msg=document.getElementById('models-msg');
  el.style.display='block';
  el.textContent='加载中...';
  try{{
    var result=await requestJSON('/v1/models?refresh=true',{{method:'GET'}},45000);
    var d=result.data;
    if(!result.ok || !d || !Array.isArray(d.data)){{
      throw new Error((d && d.error && d.error.message) || ('HTTP '+result.status));
    }}
    el.textContent=d.data.map(function(model,index){{ return String(index+1).padStart(2,'0')+'  '+String(model.id||''); }}).join('\\n');
    showMsg('models-msg','成功获取 '+d.data.length+' 个模型',true);
  }}catch(e){{
    el.textContent=String(e);
    showMsg('models-msg',String(e),false);
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
async function removeAccount(id){{
  if(!confirm('确定删除该账号？')) return;
  var d=await postJSON('/api/accounts/remove',{{account_id:id}});
  if(d.success) location.reload(); else showMsg('auth-msg',d.error||'删除失败',false);
}}
async function togglePolling(){{
  var on=document.getElementById('poll-toggle').checked;
  var mode=document.querySelector('input[name=poll-mode]:checked');
  var modeVal=mode?mode.value:'round-robin';
  var d=await postJSON('/api/polling',{{enabled:on,mode:modeVal}});
  if(d.success) location.reload();
  else showMsg('auth-msg',d.error||'操作失败',false);
}}
async function saveSettings(){{
  var web=document.getElementById('settings-web').value.trim();
  var port=document.getElementById('settings-port').value.trim();
  var mode=(typeof ENDPOINT_PRESET_MAP !== 'undefined' && ENDPOINT_PRESET_MAP[document.getElementById('settings-endpoint-preset')?.value || '']) ? ENDPOINT_PRESET_MAP[document.getElementById('settings-endpoint-preset').value].mode : '';
  var d=await postJSON('/api/settings',{{web_base_url:web,relay_port:port,upstream_mode:mode}});
  if(d.success){{ showMsg('settings-msg',d.note||'设置已保存',true); setTimeout(function(){{ location.reload(); }},800); }}
  else showMsg('settings-msg',d.error||'保存失败',false);
}}
async function saveAutoRoute(){{
  var toggle=document.getElementById('auto-route-toggle');
  var d=await postJSON('/api/auto-route',{{enabled:toggle.checked}});
  if(d.success){{
    toggle.checked=!!d.enabled;
    document.getElementById('auto-route-state').textContent=d.enabled?'已开启':'已关闭';
    showMsg('auto-route-msg',d.enabled?'自动路由已开启，下一个请求生效':'自动路由已关闭，使用预设端点',true,3000);
  }} else {{
    toggle.checked=!toggle.checked;
    showMsg('auto-route-msg',d.error||'保存失败',false);
  }}
}}
async function saveMaxMode(){{
  var enabled=document.getElementById('max-mode-toggle').checked;
  var models=document.getElementById('max-mode-models').value.trim();
  var d=await postJSON('/api/max-mode',{{enabled:enabled,models:models}});
  if(d.success){{
    document.getElementById('max-mode-toggle').checked=!!d.enabled;
    document.getElementById('max-mode-models').value=d.models||'';
    document.getElementById('max-mode-state').textContent=d.enabled?'已开启':'已关闭';
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
  if(lower.indexOf(name.toLowerCase())<0) items.push(name);
  input.value=items.join(', ');
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
async function applyEndpointPreset(){{
  var selector=document.getElementById('settings-endpoint-preset');
  var input=document.getElementById('settings-web');
  if(!selector||!input) return;
  var map=ENDPOINT_PRESET_MAP;
  var entry=map[selector.value]||null;
  var value=entry?entry.base_url:'';
  if(value) input.value=value;
  // Custom saved URL just selects, doesn't need save.
  if (selector.value === '__saved') return;
  // Auto-save on preset change (unless switching to custom).
  if (selector.value) {{
    await saveSettings();
  }}
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
  setBusy(true);
  try{{
    var result=await requestJSON('/api/checkin/credits/accounts',{{method:'GET'}},90000);
    var d=result.data;
    if(!result.ok||!d||!d.success){{ throw new Error(apiError(d,result.status)); }}
    var accounts=d.accounts||[];
    accounts.forEach(updateAccountCreditsRow);
    var failures=accounts.filter(function(a){{ return a.error; }});
    var summary=document.getElementById('checkin-summary');
    if(summary) summary.textContent='积分查询完成：'+(accounts.length-failures.length)+' / '+accounts.length+' 个账号'+(failures.length?'，'+failures.length+' 个异常':'');
    var updated=document.getElementById('checkin-updated');
    if(updated) updated.textContent='更新于 '+new Date().toLocaleTimeString();
    if(failures.length){{
      showMsg('checkin-msg','积分查询完成，但有 '+failures.length+' 个账号失败：\\n'+failures.slice(0,3).map(checkinFailureText).join('\\n'),false,12000);
    }}else{{
      showMsg('checkin-msg','积分查询成功，已刷新 '+accounts.length+' 个账号',true,3000);
    }}
  }}catch(e){{ showMsg('checkin-msg',String(e),false,12000); }}
  finally{{ setBusy(false); }}
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
function renderAutoCheckin(d){{
  if(!d||!d.success) return;
  var toggle=document.getElementById('auto-checkin-toggle');
  var timeInput=document.getElementById('auto-checkin-time');
  if(toggle) toggle.checked=!!d.enabled;
  if(timeInput&&d.time) timeInput.value=d.time;
  var stateEl=document.getElementById('auto-checkin-state');
  if(stateEl) stateEl.textContent=d.running?'正在执行...':(d.enabled?('每天 '+d.time+' 自动签到'):'已关闭');
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
<head><meta charset="utf-8"><title>Trae 授权</title>
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
            "work-agent",
            "traework-native",
            "native",
            "traework",
        }
    )
    fallback_attempted = False
    task = asyncio.create_task(_dispatch_chat(messages, model, True, options))
    # Give the task one event-loop turn to enter the selected upstream path
    # (and, for raw/remote transports, begin opening the provider request).
    await asyncio.sleep(0)
    response = None
    iterator = None
    request_id = str((options or {}).get("_relay_request_id") or "")
    started_at = time.monotonic()
    upstream_chunks = 0
    upstream_payload_chunks = 0
    saw_done = False
    stream_status = "opening"
    sent_start_event = False

    async def close_current() -> None:
        """Close the currently selected response/iterator before a retry."""

        nonlocal iterator, response
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

        nonlocal fallback_attempted
        if not fallback_allowed or fallback_attempted:
            return None
        fallback_attempted = True
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
            close = getattr(fallback, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
            return None
        trace = options.get("_upstream_trace")
        if isinstance(trace, dict):
            trace.update(
                actual_mode="remote",
                actual_endpoint="remote",
                fallback_used=True,
            )
        return fallback

    async def forward_response(current_response):
        """Forward one StreamingResponse and propagate pre-output failures."""

        nonlocal iterator, upstream_chunks, upstream_payload_chunks
        nonlocal saw_done, stream_status, sent_start_event
        iterator = getattr(current_response, "body_iterator", None)
        if iterator is None:
            body = getattr(current_response, "body", b"")
            if body:
                yield body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
            return
        async for chunk in iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            if chunk:
                upstream_chunks += 1
                if any(
                    line.lstrip().startswith("data:")
                    for line in chunk.splitlines()
                ):
                    upstream_payload_chunks += 1
                if "data: [DONE]" in chunk:
                    saw_done = True
                    stream_status = "completed"
                if not sent_start_event and chunk.lstrip().startswith(":"):
                    yield _stream_start_event(model)
                    sent_start_event = True
                yield chunk

    try:
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
                break
            except asyncio.TimeoutError:
                yield ": relay-keepalive\n\n"

        if getattr(response, "status_code", 200) >= 400:
            fallback = await try_remote_fallback(
                _upstream_response_error(response) or "http error"
            )
            if fallback is None:
                stream_status = "upstream_error"
                yield _stream_error_event(response)
                yield "data: [DONE]\n\n"
                saw_done = True
                return
            await close_current()
            response = fallback

        try:
            async for chunk in forward_response(response):
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            # A StreamingResponse can be returned with HTTP 200 and fail only
            # when its body iterator starts (empty native SSE, invalid headers,
            # or a provider disconnect).  Retry Remote only before any native
            # payload has reached the caller; once partial output is visible we
            # preserve that stream and surface the original failure.
            if upstream_payload_chunks or fallback_attempted:
                raise
            await close_current()
            fallback = await try_remote_fallback(str(exc))
            if fallback is None:
                raise
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
        if not task.done():
            task.cancel()
        raise
    except GeneratorExit:
        stream_status = "client_closed" if not saw_done else "completed"
        raise
    except Exception as exc:
        stream_status = "error"
        logger.warning("deferred stream dispatch failed: %s", exc)
        yield "data: " + json.dumps(
            {"error": {"message": str(exc), "type": "api_error"}},
            ensure_ascii=False,
        ) + "\n\n"
        yield "data: [DONE]\n\n"
        saw_done = True
    finally:
        await close_current()
        logger.info(
            "public stream closed id=%s status=%s chunks=%d done=%s elapsed_ms=%d",
            request_id,
            stream_status,
            upstream_chunks,
            saw_done,
            int((time.monotonic() - started_at) * 1000),
        )


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
    now = time.monotonic()
    with _CHAT_SESSION_LOCK:
        _prune_chat_sessions(now)
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
            for length in range(len(messages), 0, -1):
                found = _CHAT_HISTORY_SESSIONS.get(_chat_history_key(messages, length))
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
                        active_account
                        and candidate_lease is not None
                        and candidate_lease.account_id
                        and candidate_lease.account_id != active_account
                    ):
                        continue
                    session_id = candidate_id
                    _CHAT_HISTORY_SESSIONS.move_to_end(_chat_history_key(messages, length))
                    break
        if not session_id:
            session_id = uuid_mod.uuid4().hex

        lease = _UPSTREAM_SESSION_LEASES.get(session_id)
        if lease is None:
            if rotate_for_new and _current_upstream_mode() in (
                "raw",
                "direct",
                "web",
                "remote",
                "9router",
                "trae-remote",
                "auto",
            ):
                auth.next_polling_account()
            # Read the selected id and its credential record as one snapshot.
            # Separate getter calls can race an account rotation and bind an
            # account id to a different account's token.
            if inferred_account_snapshot is not None and not bool(
                auth.get_polling_status().get("enabled")
            ):
                account_id, record = inferred_account_snapshot
            else:
                account_id, record = auth.get_active_account_snapshot()
            # Keep compatibility with integrations/tests that replace the
            # legacy getters while still preferring the atomic snapshot in
            # normal operation.
            if not account_id:
                account_id = auth.get_active_account_id() or ""
            if account_id and not record:
                record = auth.get_account_record(account_id)
            token = str(record.get("token") or "")
            if not token and not account_id:
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
            lease.last_client_activity = now
            _UPSTREAM_SESSION_LEASES.move_to_end(session_id)

        _CHAT_HISTORY_SESSIONS[_chat_history_key(messages)] = (session_id, now)
        _CHAT_HISTORY_SESSIONS.move_to_end(_chat_history_key(messages))

    bound = dict(options)
    bound["session_id"] = session_id
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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
    return None


def _usage_values(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
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
    credits = _first_number(
        usage,
        "credits_consumed",
        "consumed_credits",
        "credit_cost",
        "credits_cost",
        "credits_float",
    )
    billing = usage.get("billing") or usage.get("cost")
    if credits is None and isinstance(billing, Mapping):
        credits = _first_number(
            billing,
            "credits_consumed",
            "consumed_credits",
            "credit_cost",
            "credits_cost",
            "credits_float",
        )
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "credits_consumed": credits,
    }


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


def _begin_credit_snapshot(account_id: str, token: str) -> bool:
    if not account_id or account_id == "default" or not token:
        return False
    with _USAGE_LOCK:
        active = _USAGE_ACTIVE_ACCOUNTS.get(account_id, 0)
        if active:
            _USAGE_UNSAFE_ACCOUNTS.add(account_id)
        _USAGE_ACTIVE_ACCOUNTS[account_id] = active + 1
        # A delta is only attributable when this account has one request in
        # flight. Concurrent calls share the same upstream counter.
        return active == 0


def _end_credit_snapshot(account_id: str) -> None:
    if not account_id:
        return
    with _USAGE_LOCK:
        active = _USAGE_ACTIVE_ACCOUNTS.get(account_id, 0)
        if active <= 1:
            _USAGE_ACTIVE_ACCOUNTS.pop(account_id, None)
            _USAGE_UNSAFE_ACCOUNTS.discard(account_id)
        else:
            _USAGE_ACTIVE_ACCOUNTS[account_id] = active - 1


def _credit_snapshot_is_safe(account_id: str) -> bool:
    with _USAGE_LOCK:
        return account_id not in _USAGE_UNSAFE_ACCOUNTS


def _spawn_usage_task(
    coro,
    registry: set[asyncio.Task] | None = None,
) -> asyncio.Task:
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
        return
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _USAGE_ENRICH_TASKS.difference_update(tasks)
    _USAGE_SNAPSHOT_TASKS.difference_update(tasks)


async def _enrich_usage_credits(
    request_id: str,
    account_id: str,
    token: str,
    before_task: asyncio.Task | None,
    *,
    usage_turn_id: str = "",
    credit_safe: bool = True,
) -> None:
    try:
        settle = _credit_settle_seconds()
        if settle:
            await asyncio.sleep(settle)
        if usage_turn_id and _session_usage_enabled():
            try:
                session_usage = await trae_client.fetch_session_usage(
                    usage_turn_id,
                    token,
                )
                credits = _number_value(session_usage.get("credits_consumed"))
                if credits is not None and credits >= 0:
                    await _cancel_usage_task(
                        before_task,
                        _USAGE_SNAPSHOT_TASKS,
                    )
                    before_task = None
                    _update_usage_record(
                        request_id,
                        credits_consumed=_credit_round(credits),
                        credits_source="session_usage",
                    )
                    return
            except Exception as exc:
                logger.debug("session usage enrichment unavailable: %s", exc)
        if not credit_safe or not _credit_snapshot_is_safe(account_id):
            return
        before = None
        if before_task is not None:
            try:
                before = await before_task
            except Exception:
                before = None
        after = await _fetch_used_credits(token)
        if before is None or after is None or after < before:
            return
        _update_usage_record(
            request_id,
            credits_consumed=_credit_round(after - before),
            credits_source="snapshot_delta",
            credits_before=_credit_round(before),
            credits_after=_credit_round(after),
        )
    finally:
        await _cancel_usage_task(before_task, _USAGE_SNAPSHOT_TASKS)
        _end_credit_snapshot(account_id)


class _UsageTracker:
    def __init__(
        self,
        model: str,
        endpoint: str,
        stream: bool,
        options: Optional[Mapping[str, Any]] = None,
    ):
        self.request_id = "req-" + uuid_mod.uuid4().hex
        options = options or {}
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
        self.status = "in_progress"
        self._finished = False
        self._credit_snapshot_started = bool(
            self.account_id and self.account_id != "default" and self.token
        )
        self._credit_safe = _begin_credit_snapshot(self.account_id, self.token)
        self._before_task = (
            _spawn_usage_task(
                _fetch_used_credits(self.token),
                _USAGE_SNAPSHOT_TASKS,
            )
            if self._credit_safe
            else None
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

    def bind_usage_turn(self, usage_turn_id: Any, *, replace: bool = False) -> None:
        value = str(usage_turn_id or "").strip()
        if value and (replace or not self.usage_turn_id):
            self.usage_turn_id = value

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
        await _cancel_usage_task(self._before_task, _USAGE_SNAPSHOT_TASKS)
        self._before_task = None
        if self._credit_snapshot_started:
            _end_credit_snapshot(old_account)

        self.account_id = account_id or self.account_id
        self.billing_id = billing_id or self.account_id
        self.token = token or self.token
        self.usage_turn_id = ""
        self._credit_snapshot_started = bool(
            self.account_id and self.account_id != "default" and self.token
        )
        self._credit_safe = _begin_credit_snapshot(self.account_id, self.token)
        if self._credit_safe:
            self._before_task = _spawn_usage_task(
                _fetch_used_credits(self.token),
                _USAGE_SNAPSHOT_TASKS,
            )

    async def finish(self, status: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        final_status = status or self.status or "completed"
        values = self.usage
        explicit_credits = values.get("credits_consumed")
        credits_source = "upstream" if explicit_credits is not None else "unknown"
        _record_usage(
            self.account_id,
            self.model,
            values["prompt_tokens"],
            values["completion_tokens"],
            credits_consumed=explicit_credits,
            credits_source=credits_source,
            request_id=self.request_id,
            endpoint=self.endpoint,
            stream=self.stream,
            status=final_status,
            duration_ms=round((time.perf_counter() - self.started) * 1000, 1),
            tokens_source="upstream" if self.saw_usage else "unknown",
        )
        if explicit_credits is not None:
            try:
                await _cancel_usage_task(self._before_task, _USAGE_SNAPSHOT_TASKS)
            finally:
                self._before_task = None
                if self._credit_snapshot_started:
                    _end_credit_snapshot(self.account_id)
        elif self.usage_turn_id or self._credit_safe:
            _spawn_usage_task(
                _enrich_usage_credits(
                    self.request_id,
                    self.account_id,
                    self.token,
                    self._before_task,
                    usage_turn_id=self.usage_turn_id,
                    credit_safe=self._credit_safe,
                )
            )
        else:
            try:
                await _cancel_usage_task(self._before_task, _USAGE_SNAPSHOT_TASKS)
            finally:
                self._before_task = None
                if self._credit_snapshot_started:
                    _end_credit_snapshot(self.account_id)

    async def begin(self) -> None:
        if self._before_task is not None:
            # Let the before-snapshot request start without delaying the first
            # model frame on the result of a separate billing endpoint.
            await asyncio.sleep(0)


def _track_usage_from_result(result: dict, model: str) -> None:
    """Pass non-stream usage to the request tracker, or keep legacy fallback."""
    usage = result.get("usage") or {}
    tracker = _USAGE_TRACKER.get()
    if tracker is not None:
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
    )


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
        )


def _bind_usage_turn(usage_turn_id: Any, *, replace: bool = False) -> None:
    tracker = _USAGE_TRACKER.get()
    if tracker is not None:
        tracker.bind_usage_turn(usage_turn_id, replace=replace)


def _bind_usage_turn_from_metadata(metadata: Any) -> None:
    if isinstance(metadata, Mapping):
        _bind_usage_turn(metadata.get("usage_turn_id"))


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


def _bounded_remote_query(
    messages: list[Mapping[str, Any]],
    options: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Truncate oldest non-system messages so the flattened query fits upstream.

    Trae's remote session silently ends the event stream (no text, no ``done``)
    when initial_message.query exceeds roughly 500k chars.  Keep the system
    prompt and the newest turns intact and drop the oldest conversation from
    the front until ``flatten_query`` is under the configured cap.
    """

    raw_limit = _raw_history_limit_int(
        options,
        ("trae_remote_query_max_chars", "traeRemoteQueryMaxChars"),
        "TRAE_REMOTE_QUERY_MAX_CHARS",
        480_000,
    )
    if raw_limit <= 0:
        return [dict(m) for m in messages], 0
    bounded = [dict(m) for m in messages]
    removed = 0
    while True:
        query = trae_client.flatten_query(bounded)
        if len(query) <= raw_limit:
            return bounded, removed
        non_system = [
            index
            for index, message in enumerate(bounded)
            if str(message.get("role") or "user") not in {"system", "developer"}
        ]
        if len(non_system) <= 1:
            return bounded, removed
        head = non_system[0]
        drop = {head}
        head_message = bounded[head]
        if str(head_message.get("role") or "") == "assistant" and (
            head_message.get("tool_calls") or head_message.get("function_call")
        ):
            if head + 1 < len(bounded):
                following = bounded[head + 1]
                if str(following.get("role") or "") == "tool":
                    drop.add(head + 1)
        bounded = [m for index, m in enumerate(bounded) if index not in drop]
        removed += len(drop)


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
        _USAGE_TRACKER.reset(context_token)
        await tracker.finish(status)


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


async def run_web_session(messages, model, stream: bool, options: Optional[dict] = None):
    """OmniRoute 风格网页版 remote 会话，带账号并发槽和空闲回收。"""
    options = dict(options or {})
    token = str(options.get("_auth_token") or "").strip()
    account_id = str(options.get("_account_id") or "").strip()
    record = auth.get_account_record(account_id) if account_id else {}
    if not token and account_id:
        token = str((record or {}).get("token") or "").strip()
    if not token and not account_id:
        token = str(auth.get_token() or "").strip()
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
    await trae_client.acquire_web_slot(account_id, timeout=float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")))
    client = httpx.AsyncClient(timeout=60)
    session_id = ""
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
    try:
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
        )
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
                    # Actively interrupt the upstream session so it stops
                    # occupying a running slot, then close local resources.
                    await trae_client.stop_web_session(
                        client, session_id, message_id, options=bound_options
                    )
                    await client.aclose()
                    if trae_client.unregister_web_lease(session_id):
                        trae_client.release_web_slot(account_id)
            return StreamingResponse(
                gen(),
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
            await trae_client.stop_web_session(
                client, session_id, message_id, options=bound_options
            )
            await client.aclose()
            if trae_client.unregister_web_lease(session_id):
                trae_client.release_web_slot(account_id)
    except Exception:
        if session_id:
            try:
                await trae_client.stop_web_session(
                    client, session_id, message_id, options=bound_options
                )
            except Exception:
                pass
        await client.aclose()
        if session_id:
            if trae_client.unregister_web_lease(session_id):
                trae_client.release_web_slot(account_id)
        else:
            trae_client.release_web_slot(account_id)
        raise


async def run_remote_session(messages, model, stream: bool, options: Optional[dict] = None):
    """9router-style Trae remote session using the current account snapshot.

    Unlike the legacy web helper, this path never reads a mutable global token
    after dispatch.  The account-bound token and provider metadata captured by
    ``_bind_chat_session`` are used for both create and events requests.
    """
    options = dict(options or {})
    account_id = str(options.get("_account_id") or "").strip()
    token = str(options.get("_auth_token") or "").strip()
    if not token and account_id:
        # A bound account owns its credential.  Do not fall back to the
        # mutable global token, which may belong to a concurrently selected
        # account.
        token = str((auth.get_account_record(account_id) or {}).get("token") or "").strip()
    if not token and not account_id:
        token = str(auth.get_token() or "").strip()
    token_identity = _account_id_from_token(token)
    if token_identity:
        # The JWT is the identity Trae bills.  It is authoritative if an old
        # account-store key or UI selection is stale.
        account_id = token_identity
    if not account_id:
        account_id = str(auth.get_active_account_id() or "default")
    record = auth.get_account_record(account_id) if account_id else {}
    if not token:
        token = str(record.get("token") or "").strip()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    remote_options = dict(options)
    # Caller-owned tools must run in the API client's workspace.  Agent owns a
    # separate remote workspace and can consume file/shell work internally,
    # then return only a success sentence; Work keeps the advertised tools as
    # calls for Codex/the API client to execute.  Plain text requests remain
    # Agent-first (including the 1M/max profile), and operators can explicitly
    # opt tool requests back into Agent with ``...USE_WORK=0`` for diagnostics.
    caller_tools_use_work = (
        os.environ.get("TRAE_REMOTE_CALLER_TOOLS_USE_WORK", "1").strip().lower()
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
    await trae_client.acquire_web_slot(
        account_id,
        timeout=float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")),
    )
    slot_released = False
    cleanup_started = False

    def release_slot_once() -> None:
        """Release the account slot exactly once across all exit paths."""

        nonlocal slot_released
        if slot_released:
            return
        slot_released = True
        trae_client.release_web_slot(account_id)

    client = httpx.AsyncClient(timeout=None)
    session_id = ""
    message_id = ""

    async def close_remote_session() -> None:
        """Stop and close one remote attempt without leaking its account slot."""

        nonlocal cleanup_started
        if cleanup_started:
            return
        cleanup_started = True
        try:
            if session_id:
                await trae_remote_client.stop_session(
                    client,
                    token,
                    session_id,
                    message_id,
                    options=remote_options,
                )
        finally:
            try:
                await client.aclose()
            finally:
                release_slot_once()

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
    prepared_messages, query_trimmed = _bounded_remote_query(
        prepared_messages, compact_options
    )
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
            len(trae_client.flatten_query(prepared_messages)),
        )
    try:
        logger.info(
            "remote create start id=%s account=%s model=%s messages=%d last_chars=%d",
            str(options.get("_relay_request_id") or ""),
            account_id,
            model,
            len(messages),
            len(str(messages[-1].get("content") or "")) if messages else 0,
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
            if not can_work_fallback:
                raise
            logger.warning(
                "remote Agent create failed; falling back to Work id=%s model=%s error=%s",
                str(options.get("_relay_request_id") or ""),
                model,
                create_exc,
            )
            # No session id was returned, so the failed create cannot be
            # stopped. Reuse the acquired account slot with a fresh client.
            await client.aclose()
            client = httpx.AsyncClient(timeout=None)
            remote_options = work_fallback_options(remote_options)
            work_fallback_used = True
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
                        if exc_usage is not None:
                            _track_usage_from_result({"usage": exc_usage}, model)
                        polling_retry_enabled = bool(
                            auth.get_polling_status().get("enabled")
                        )
                        if (
                            not exc_retryable
                            or work_fallback_used
                            or not (can_work_fallback or polling_retry_enabled)
                        ):
                            logger.warning(
                                "remote empty response is not safe to retry "
                                "id=%s model=%s observed_model_event=%s fallback_used=%s",
                                request_id,
                                model,
                                exc_observed_model_event,
                                work_fallback_used,
                            )
                            raise
                        logger.warning(
                            "remote upstream ended before any model event; "
                            "%s once id=%s model=%s",
                            "falling back to Work" if can_work_fallback and not work_fallback_used else "retrying",
                            request_id,
                            model,
                        )
                        await close_remote_session()
                        slot_reacquired = False
                        retry_client = None
                        retry_session_id = ""
                        retry_message_id = ""
                        retry_account_id = account_id
                        try:
                            retry_options = dict(remote_options)
                            retry_model = model
                            if can_work_fallback and not work_fallback_used:
                                retry_options = work_fallback_options(retry_options)
                                work_fallback_used = True
                            retry_token = token
                            # Prefer the same-account Work fallback.  Account
                            # rotation remains the outer retry policy after a
                            # Work attempt is exhausted.
                            if auth.get_polling_status().get("enabled") and not (
                                can_work_fallback and work_fallback_used
                            ):
                                next_snapshot = _next_retry_account_snapshot(
                                    {_retry_account_key(options, 0)}, 1
                                )
                                if next_snapshot is not None:
                                    next_account_id, record = next_snapshot
                                    retry_account_id = next_account_id
                                    retry_token = (
                                        str(record.get("token") or "") or retry_token
                                    )
                                    retry_billing_id = (
                                        _account_id_from_token(retry_token)
                                        or next_account_id
                                    )
                                    retry_options = dict(retry_options)
                                    retry_options["_account_id"] = next_account_id
                                    retry_options["_billing_id"] = retry_billing_id
                                    retry_options["_auth_token"] = retry_token
                                    retry_options["_auth_user_id"] = retry_billing_id
                                    retry_provider = (
                                        record.get("provider_specific")
                                        or record.get("providerSpecificData")
                                        or {}
                                    )
                                    if isinstance(retry_provider, Mapping):
                                        retry_options["provider_specific"] = dict(
                                            retry_provider
                                        )
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
                            await trae_client.acquire_web_slot(
                                retry_account_id,
                                timeout=float(
                                    os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")
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
                            try:
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
                                if retry_session_id:
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
                            if slot_reacquired:
                                trae_client.release_web_slot(retry_account_id)
                            if retry_client is not None:
                                await retry_client.aclose()
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
                gen(), media_type="text/event-stream", headers=_sse_headers()
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
            if exc.usage is not None:
                _track_usage_from_result({"usage": exc.usage}, model)
            if not (can_work_fallback and not work_fallback_used and exc.retryable):
                raise
            logger.warning(
                "remote upstream ended before any model event; falling back to Work "
                "id=%s model=%s",
                str(options.get("_relay_request_id") or ""),
                model,
            )
            work_fallback_used = True
            try:
                if session_id:
                    await trae_remote_client.stop_session(
                        client,
                        token,
                        session_id,
                        message_id,
                        options=remote_options,
                    )
            except Exception:
                pass
            session_id = ""
            message_id = ""
            remote_options = work_fallback_options(remote_options)
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
    except Exception:
        await close_remote_session()
        raise


def _effort_note(effort: str, trace: Mapping[str, Any]) -> Optional[str]:
    """Explain why a requested thinking strength was not applied."""

    if not effort or trace.get("reasoning_effort_applied"):
        return None
    mode = str(trace.get("actual_mode") or trace.get("actual_endpoint") or "")
    if mode == "ide":
        # llm_utils_chat is the IDE utility route (titles/prompt tools); the
        # real IDE sends strength only via the ai-agent task, so the field is
        # ignored here.
        return "llm_utils_chat 端点不接受强度字段"
    if mode == "raw":
        return "raw 端点需要企业 PAT"
    return "该模型未声明 reasoning_effort_config"


async def run_ide_chat(messages, model, stream: bool, options: Optional[dict] = None):
    """trae2api 风格 IDE chat，流式响应消费完成后关闭 response 和 client。"""
    ide_resp = await trae_client.send_chat_request(messages, model, stream, options=options)
    response = ide_resp.response
    translation_options = _tool_translation_options(options, messages)
    upstream_metadata: dict[str, Any] = {}
    if stream:
        async def gen():
            try:
                async for chunk in translate_ide_stream(
                    response,
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
                _bind_usage_turn_from_metadata(upstream_metadata)
                ide_resp.close()
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )
    try:
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
    finally:
        _bind_usage_turn_from_metadata(upstream_metadata)
        ide_resp.close()


async def run_traework_native_chat(
    messages, model, stream: bool, options: Optional[dict] = None
):
    """TraeWork native AHA bridge backed by an external Windows helper.

    The helper owns ai_agent.dll and sscronet.dll and returns the native SSE
    event stream. Linux deployments receive a clear 502 instead of attempting
    to load a Windows PE DLL.
    """

    native_resp = await traework_native_bridge.send_native_chat_request(
        messages,
        model,
        stream=stream,
        options=options,
    )
    translation_options = _tool_translation_options(options, messages)
    upstream_metadata: dict[str, Any] = {}
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
                _bind_usage_turn_from_metadata(upstream_metadata)
                native_resp.close()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )
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
        _bind_usage_turn_from_metadata(upstream_metadata)
        native_resp.close()


async def run_raw_chat(messages, model, stream: bool, options: Optional[dict] = None):
    """直连 Trae 原生 chat 协议，响应暂复用 IDE SSE 翻译器。"""
    logger.info(
        "raw send start id=%s model=%s messages=%d last_chars=%d",
        str((options or {}).get("_relay_request_id") or ""),
        model,
        len(messages),
        len(str(messages[-1].get("content") or "")) if messages else 0,
    )
    raw_resp = await raw_client.send_raw_chat_request(messages, model, options)
    logger.info(
        "raw send ok id=%s status=%s",
        str((options or {}).get("_relay_request_id") or ""),
        getattr(raw_resp.response, "status_code", 0),
    )
    _capture_chat_session_auth(
        str((options or {}).get("session_id") or ""),
        str(getattr(raw_resp, "auth_token", "") or ""),
    )
    translation_options = _tool_translation_options(options, messages)
    upstream_metadata: dict[str, Any] = {}
    if stream:
        async def gen():
            current = raw_resp
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
                        if exc.usage is not None:
                            _track_usage_from_result({"usage": exc.usage}, model)
                        logger.warning(
                            "raw upstream repeated an already completed tool call; "
                            "automatic replay is disabled to avoid a second billed turn"
                        )
                        raise RuntimeError(str(exc)) from exc
                    except EmptyUpstreamResponse as exc:
                        _bind_usage_turn_from_metadata(upstream_metadata)
                        if exc.usage is not None:
                            _track_usage_from_result({"usage": exc.usage}, model)
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
                        _bind_usage_turn_from_metadata(upstream_metadata)
                        current.close()

                    try:
                        current = await raw_client.send_raw_chat_request(
                            messages, model, retry_options
                        )
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
                logger.info(
                    "raw stream closed id=%s status=%s chunks=%d tool_chunks=%d done=%s elapsed_ms=%d",
                    request_id,
                    stream_status,
                    chunk_count,
                    tool_chunk_count,
                    saw_done,
                    int((time.monotonic() - started_at) * 1000),
                )
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    current = raw_resp
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
            if exc.usage is not None:
                _track_usage_from_result({"usage": exc.usage}, model)
            logger.warning(
                "raw upstream repeated an already completed tool call; "
                "automatic replay is disabled to avoid a second billed turn"
            )
            raise RuntimeError(str(exc)) from exc
        except EmptyUpstreamResponse as exc:
            _bind_usage_turn_from_metadata(upstream_metadata)
            if exc.usage is not None:
                _track_usage_from_result({"usage": exc.usage}, model)
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
            _bind_usage_turn_from_metadata(upstream_metadata)
            current.close()
        try:
            current = await raw_client.send_raw_chat_request(
                messages, model, retry_options
            )
            _capture_chat_session_auth(
                str((options or {}).get("session_id") or ""),
                str(getattr(current, "auth_token", "") or ""),
            )
        except Exception as exc:
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
            if account.get("is_valid") is False:
                continue
            account_id = str(account.get("id") or "")
            if account_id:
                account_ids.add(account_id)
            else:
                anonymous_accounts += 1
        else:
            anonymous_accounts += 1
    bound_account = str((options or {}).get("_account_id") or "")
    if bound_account:
        account_ids.add(bound_account)
    return max(1, len(account_ids) + anonymous_accounts)


def _next_retry_account_snapshot(
    attempted_accounts: set[str], max_rotations: int
) -> tuple[str, dict] | None:
    """Rotate to an account that has not already received this request."""

    for rotation_index in range(max(1, max_rotations)):
        auth.next_polling_account()
        account_id, record = auth.get_active_account_snapshot()
        safe_record = dict(record) if isinstance(record, Mapping) else {}
        candidate_options = {
            "_account_id": str(account_id or ""),
            "_auth_token": str(safe_record.get("token") or ""),
        }
        account_key = _retry_account_key(candidate_options, rotation_index)
        if account_key not in attempted_accounts:
            return str(account_id or ""), safe_record
    return None


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
            try:
                return await run_web_session(messages, model, stream, options)
            except RuntimeError as e:
                err = str(e)
                if "solo_agent_parallel_limit" in err or "429" in err:
                    if attempt_index + 1 >= attempts:
                        break
                    logger.warning("web 429 parallel limit, rotating account: %s", err)
                    next_snapshot = _next_retry_account_snapshot(
                        attempted_accounts, attempts
                    )
                    if next_snapshot is None:
                        break
                    # Rebind options from the newly rotated account so the
                    # retry uses the correct token and billing identity.
                    account_id, record = next_snapshot
                    token = str(record.get("token") or "")
                    billing_id = _account_id_from_token(token) or account_id
                    provider_specific = (
                        record.get("provider_specific")
                        or record.get("providerSpecificData")
                        or {}
                    )
                    options = dict(options or {})
                    options["_account_id"] = account_id
                    options["_billing_id"] = billing_id
                    options["_auth_token"] = token
                    options["_auth_user_id"] = billing_id
                    if isinstance(provider_specific, Mapping):
                        options["provider_specific"] = dict(provider_specific)
                    if tracker is not None:
                        await tracker.rebind(options)
                    _rebind_chat_session_account(
                        str(options.get("session_id") or options.get("sessionId") or ""),
                        account_id,
                        billing_id,
                        token,
                        provider_specific,
                    )
                    continue
                raise
        raise RuntimeError("All web accounts busy: Trae parallel limit reached")
    return await run_web_session(messages, model, stream, options)


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
    if auth.get_polling_status().get("enabled"):
        attempts = _polling_retry_limit(options)
        attempted_accounts: set[str] = set()

        async def rotate_remote_account(reason: str) -> bool:
            nonlocal options
            logger.warning("remote %s, rotating account", reason)
            next_snapshot = _next_retry_account_snapshot(
                attempted_accounts, attempts
            )
            if next_snapshot is None:
                return False
            account_id, record = next_snapshot
            token = str(record.get("token") or "")
            billing_id = _account_id_from_token(token) or account_id
            provider_specific = (
                record.get("provider_specific")
                or record.get("providerSpecificData")
                or {}
            )
            options = dict(options or {})
            options["_account_id"] = account_id
            options["_billing_id"] = billing_id
            options["_auth_token"] = token
            options["_auth_user_id"] = billing_id
            if isinstance(provider_specific, Mapping):
                options["provider_specific"] = dict(provider_specific)
            if tracker is not None:
                await tracker.rebind(options)
            _rebind_chat_session_account(
                str(options.get("session_id") or options.get("sessionId") or ""),
                account_id,
                billing_id,
                token,
                provider_specific,
            )
            return True

        for attempt_index in range(attempts):
            account_key = _retry_account_key(options, attempt_index)
            if account_key in attempted_accounts:
                break
            attempted_accounts.add(account_key)
            try:
                return await run_remote_session(messages, model, stream, options)
            except EmptyUpstreamResponse as exc:
                if not exc.retryable or attempt_index + 1 >= attempts:
                    raise
                if not await rotate_remote_account(
                    "upstream returned an empty response"
                ):
                    raise
                continue
            except RuntimeError as exc:
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
                elif "parallel" in lowered or "429" in lowered:
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
        raise RuntimeError("All remote accounts busy: Trae parallel limit reached")
    return await run_remote_session(messages, model, stream, options)


def _normalize_usage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    values = _usage_values(record)
    prompt = values["prompt_tokens"]
    completion = values["completion_tokens"]
    total = values["total_tokens"] or prompt + completion
    credits = values.get("credits_consumed")
    normalized.update(
        {
            "account_id": str(record.get("account_id") or "default"),
            "model": str(record.get("model") or "auto"),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": total,
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
            "request_id": str(record.get("request_id") or ""),
            "endpoint": record.get("endpoint") or None,
            "stream": record.get("stream") if "stream" in record else None,
            "status": str(record.get("status") or "completed"),
            "duration_ms": _number_value(record.get("duration_ms")),
            "timestamp": _number_value(record.get("timestamp")) or 0,
        }
    )
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


def _record_usage(
    account_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    credits_consumed: int | float | None = None,
    credits_source: str = "unknown",
    request_id: str = "",
    endpoint: str | None = None,
    stream: bool | None = None,
    status: str = "completed",
    duration_ms: int | float | None = None,
    tokens_source: str = "upstream",
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
            "tokens_source": tokens_source,
            "credits_consumed": credits_consumed,
            "credits_source": credits_source,
            "request_id": request_id,
            "endpoint": endpoint,
            "stream": stream,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
    )
    with _USAGE_LOCK:
        if request_id:
            for index, existing in enumerate(_USAGE_HISTORY):
                if existing.get("request_id") == request_id:
                    _USAGE_HISTORY[index] = record
                    break
            else:
                _USAGE_HISTORY.insert(0, record)
        else:
            _USAGE_HISTORY.insert(0, record)
        if len(_USAGE_HISTORY) > _USAGE_MAX_HISTORY:
            _USAGE_HISTORY = _USAGE_HISTORY[:_USAGE_MAX_HISTORY]
        _save_usage_history_locked()
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
    fallback_agent_type = (
        "solo_work_remote"
        if requested_mode in {
            "work-agent",
            "traework-native",
            "native",
            "traework",
        }
        else "solo_agent_remote"
    )
    if not (
        fallback.get("_remote_agent_type")
        or fallback.get("remote_agent_type")
    ):
        fallback["_remote_agent_type"] = fallback_agent_type
    fallback.setdefault("_session_variant", f"{requested_mode}-fallback")
    fallback.pop("_upstream_trace", None)
    return fallback


def _update_usage_record(request_id: str, **updates: Any) -> None:
    if not request_id:
        return
    with _USAGE_LOCK:
        for index, existing in enumerate(_USAGE_HISTORY):
            if existing.get("request_id") != request_id:
                continue
            merged = dict(existing)
            merged.update(updates)
            # Round credit fields before normalizing
            for key in ("credits_consumed", "credits_before", "credits_after"):
                if key in merged:
                    merged[key] = _credit_round(merged[key])
            _USAGE_HISTORY[index] = _normalize_usage_record(merged)
            _save_usage_history_locked()
            return


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


def _apply_auto_route(
    messages: Optional[list[dict]], options: Optional[dict]
) -> dict:
    """Resolve the endpoint for one request when auto routing applies.

    Auto routing runs when the caller asked for ``auto-route`` explicitly, or
    when the console switch is on and the request carries no explicit
    endpoint.  Tool requests (tool definitions or tool history) go to IDE
    Agent, plain chat goes to Remote; the dispatcher's existing fallback then
    lands on Remote if the chosen endpoint fails.
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
    endpoint = AUTO_ROUTE_TOOL_ENDPOINT if uses_tools else AUTO_ROUTE_CHAT_ENDPOINT
    options["_upstream_mode"] = endpoint
    options["_auto_route"] = "tools" if uses_tools else "chat"
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


async def _dispatch_chat(messages, model, stream: bool, options: Optional[dict] = None):
    options = _apply_auto_route(messages, options)
    active_mode = str(options.get("_upstream_mode") or _current_upstream_mode()).lower()
    trace = options.get("_upstream_trace")
    if isinstance(trace, dict):
        trace.update(
            requested_mode=active_mode,
            requested_endpoint=active_mode,
            actual_mode="",
            actual_endpoint="",
            fallback_used=False,
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
        custom_mode = str(
            options.get("_traework_upstream_mode")
            or os.environ.get("TRAEWORK_CUSTOM_UPSTREAM_MODE", "remote")
        ).strip().lower()
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
                )
            return await _run_remote_with_retry(messages, model, stream, options)
        except ModelProviderMismatch as exc:
            logger.error(
                "remote provider model mismatch id=%s requested=%s error=%s",
                str(options.get("_relay_request_id") or ""),
                model,
                exc,
            )
            return _openai_error(502, str(exc), "upstream_model_mismatch", "model")

    # ``auto`` is the direct-proxy mode: every model request reaches Trae's
    # native llm_utils_chat endpoint. Legacy modes remain explicit opt-ins for
    # diagnostics, but they are never silent fallbacks for API traffic.
    errors = []
    modes = []
    if active_mode == "cli":
        modes = ["cli"]
    elif active_mode in ("raw", "direct"):
        modes = ["raw"]
    elif active_mode in ("remote", "9router", "trae-remote"):
        modes = ["remote"]
    elif active_mode == "web":
        modes = ["web"]
    elif active_mode == "ide":
        modes = ["ide"]
    elif active_mode == "work-agent":
        modes = ["work-agent"]
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
        "work-agent",
        "traework-native",
        "native",
        "traework",
    }
    if fallback_enabled and active_mode in ("raw", "direct"):
        modes.append("ide")
    if fallback_enabled and active_mode in fallback_modes and "remote" not in modes:
        modes.append("remote")
    if (
        fallback_enabled
        and modes[:1] == ["raw"]
        and len(modes) > 1
        and _raw_app_config_known_missing(model, options)
    ):
        errors.append("raw: skipped (app config record not found, cached)")
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
            if isinstance(trace, dict):
                trace.update(
                    actual_mode=mode,
                    actual_endpoint=mode,
                    fallback_used=mode != modes[0],
                )
            mode_options = dict(options)
            if mode == "raw":
                result = await run_raw_chat(messages, model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"raw returned {error}")
                return result
            # Web and remote routes must receive the lease-bound credential;
            # otherwise a concurrent account switch can make the upstream bill
            # one token while the usage tracker records another.
            if mode not in ("remote", "web", "ide", "work-agent", "traework-native"):
                mode_options.pop("_auth_token", None)
                mode_options.pop("_account_id", None)
            if mode == "cli":
                result = await run_cli_chat(messages, model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"cli returned {error}")
                return result
            if mode == "web":
                result = await _run_web_with_retry(messages, model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"web returned {error}")
                return result
            if mode == "remote":
                if modes[0] != "remote":
                    mode_options = _remote_fallback_options(
                        mode_options, active_mode
                    )
                result = await _run_remote_with_retry(messages, model, stream, mode_options)
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"remote returned {error}")
                return result
            if mode == "work-agent":
                mode_options["_trae_mode"] = "work"
                mode_options["_remote_agent_type"] = "solo_work_remote"
                mode_options.setdefault("_session_variant", "work-agent")
                if isinstance(trace, dict):
                    trace.update(actual_endpoint="remote-work")
                result = await _run_remote_with_retry(
                    messages, model, stream, mode_options
                )
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"work-agent returned {error}")
                return result
            if mode == "traework-native":
                result = await run_traework_native_chat(
                    messages, model, stream, mode_options
                )
                error = _upstream_response_error(result)
                if error:
                    raise RuntimeError(f"traework-native returned {error}")
                return result
            if mode == "ide":
                mode_options["_ide_endpoint"] = "/api/agent/v3/llm_utils_chat"
            result = await run_ide_chat(messages, model, stream, mode_options)
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
    tracker = _UsageTracker(descriptor.model, req.url.path, descriptor.stream, options)
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
    tracker = _UsageTracker(model, req.url.path, stream, options)
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
                    _deferred_dispatch_stream(messages, model, options), session_id
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
    tracker = _UsageTracker(context.model, req.url.path, stream, options)
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


async def init_app():
    _load_usage_history()
    auth.init_auth()
    auth.apply_max_mode_settings()
    logger.info(
        "Trae CN relay initialized (auth source=%s edition=%s cli=%s)",
        auth.get_auth().source,
        auth.get_auth().edition,
        cli_client.resolve_cli_command() or "not-found",
    )


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
    try:
        yield
    finally:
        for reaper in (web_reaper, terminal_reaper, checkin_retry, auto_checkin):
            reaper.cancel()
        await asyncio.gather(web_reaper, terminal_reaper, return_exceptions=True)
        await asyncio.gather(checkin_retry, auto_checkin, return_exceptions=True)
        await _cancel_usage_tasks()


app = FastAPI(title="Trae CN Relay", version="1.0.0", lifespan=lifespan)


@app.get("/api/usage/records")
async def get_usage_records():
    """Return the usage history list (newest first)."""
    with _USAGE_LOCK:
        records = [_normalize_usage_record(record) for record in _USAGE_HISTORY]
    return JSONResponse(records, headers={"Cache-Control": "no-store"})

@app.get("/api/usage/last")
async def api_usage_last():
    """Backward-compatible: return only the latest record."""
    with _USAGE_LOCK:
        records = [_normalize_usage_record(record) for record in _USAGE_HISTORY[:1]]
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
    if token not in API_KEYS:
        return _openai_error(401, "Invalid API key", "authentication_error")
    return await call_next(request)


@app.get("/")
async def root():
    return {"service": "trae-cn-relay", "status": "ok"}


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
        if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
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


START_AUTH_BAT = Path(__file__).resolve().parent.parent / "start_auth.bat"

@app.get("/web/login/download", response_class=FileResponse)
async def web_login_download(as_param: str = Query("", alias="as")):
    if as_param == "bat":
        if not START_AUTH_BAT.exists():
            return JSONResponse({"success": False, "error": "start_auth.bat not found"}, status_code=404)
        return FileResponse(
            START_AUTH_BAT,
            media_type="application/octet-stream",
            filename="start_auth.bat",
        )
    if not WEB_LOGIN_SCRIPT.exists():
        return JSONResponse({"success": False, "error": "web_login.py not found"}, status_code=404)
    return FileResponse(
        WEB_LOGIN_SCRIPT,
        media_type="text/plain; charset=utf-8",
        filename="web_login.py",
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
    token = rec.get("token") or ""
    if not token:
        return JSONResponse(
            {"success": False, "error": "account not found or token missing"},
            status_code=404,
        )
    try:
        raw = await trae_client.fetch_account_credits(token)
        parsed = trae_client.parse_account_credits(raw)
        auth.merge_account_credits(account_id, {"account_credits": parsed})
        return JSONResponse(
            {
                "success": True,
                "id": account_id,
                "credits": parsed,
                "raw": raw,
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
    merged = (
        auth.merge_account_credits(account_id, patch)
        if patch
        else dict(record.get("checkin") or {})
    )
    row = _cached_checkin_account_snapshot(account_id, record)
    row.update(
        {
            "account_credits": merged.get("account_credits"),
            "checkin": merged,
        }
    )
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
            full = await _fetch_full_credits(token)
            row.update(full)
            parsed = full.get("account_credits") or {}
            # Every pack lives in the merged 通用积分 pool, so the single
            # account_credits value drives the credit-priority ordering.
            credits_sort = parsed.get("remaining") or parsed.get("total_limit") or 0
            if parsed.get("unlimited"):
                credits_sort = 999999999
            fresh_checkin = dict(rec.get("checkin") or {})
            fresh_checkin.update({key: value for key, value in full.items() if value is not None})
            row["checkin"] = auth.merge_account_credits(aid, fresh_checkin)
        except Exception:
            row["account_credits"] = None
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
    web_base_url = (body.get("web_base_url") or "").strip()
    relay_port = body.get("relay_port") or body.get("port") or 0
    upstream_mode = (body.get("upstream_mode") or "").strip().lower()
    if upstream_mode and upstream_mode not in _VALID_UPSTREAM_MODES:
        return JSONResponse(
            {"success": False, "error": f"Unsupported upstream mode: {upstream_mode}"},
            status_code=400,
        )
    try:
        relay_port = int(relay_port)
    except (TypeError, ValueError):
        relay_port = 0
    if not web_base_url and not relay_port and not upstream_mode:
        return JSONResponse({"success": False, "error": "nothing to update"}, status_code=400)
    auth.set_relay_settings(web_base_url=web_base_url, port=relay_port, upstream_mode=upstream_mode)
    if upstream_mode:
        # Synchronise the running module so _current_upstream_mode() reflects
        # the new preset immediately without a container restart.
        global UPSTREAM_MODE
        UPSTREAM_MODE = upstream_mode
        os.environ["UPSTREAM_MODE"] = upstream_mode
    return JSONResponse({"success": True, "note": "端口变更需重启容器生效"})


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
    size = config.get("context_window_size")
    if isinstance(size, Mapping):
        raw = size.get("max")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if raw:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    tokens = config.get("context_window_tokens")
    if isinstance(tokens, Mapping) and tokens.get("max"):
        try:
            return int(tokens.get("max"))
        except (TypeError, ValueError):
            pass
    return 0


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
        if isinstance(cfg, Mapping) and cfg.get("max_mode")
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
