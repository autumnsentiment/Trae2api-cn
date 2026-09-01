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

FORWARD_USAGE = (os.environ.get("FORWARD_USAGE", "true") or "true").lower() == "true"
CHECKIN_INTERVAL = float(os.environ.get("TRAE_CHECKIN_INTERVAL_SECONDS", "60") or "60")
CHECKIN_RETRY_AFTER = float(
    os.environ.get("TRAE_CHECKIN_9074_RETRY_SECONDS", "60") or "60"
)
CHECKIN_9074_MAX_BACKOFF = float(
    os.environ.get("TRAE_CHECKIN_9074_MAX_BACKOFF_SECONDS", "3600") or "3600"
)
CHECKIN_AUTO_RETRY_INTERVAL = float(
    os.environ.get("TRAE_CHECKIN_AUTO_RETRY_INTERVAL_SECONDS", "60") or "60"
)
WEB_BASE = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
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
    "reasoning_effort",
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
    "/api/checkin/status",
    "/api/checkin/claim",
    "/api/checkin/accounts",
    "/api/checkin/claim-all",
    "/api/checkin/claim-credits",
    "/api/checkin/account",
    "/api/checkin/work-credits",
    "/api/usage/last",
    "/api/usage/records",
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
      <code>{html_mod.escape(UPSTREAM_MODE)}</code>
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
        acct_credits = acc.get("account_credits") or {}
        if acct_credits.get("unlimited"):
            credits_text = "\u2606 \u65e0\u9650"
        elif acct_credits.get("remaining") is not None:
            credits_text = f"\u5269{acct_credits['remaining']}/\u603b{acct_credits.get('total_limit','?')}"
        else:
            credits_text = "-"
        work_acct = acc.get("work_credits") or {}
        if work_acct.get("unlimited"):
            work_credits_text = "☆ 无限"
        elif work_acct.get("remaining") is not None:
            work_credits_text = f"剩{float(work_acct['remaining']):.2f}/总{float(work_acct.get('total_limit') or 0):.2f}"
        else:
            work_credits_text = "-"
        total = acc.get("total_credits") or {}
        if total.get("unlimited"):
            total_credits_text = "☆ 无限"
        elif total.get("remaining") is not None:
            total_credits_text = f"剩{float(total['remaining']):.2f}/总{float(total.get('total_limit') or 0):.2f}"
        else:
            total_credits_text = "-"
        credits = acc.get("credits")
        checked_in = acc.get("checked_in")
        if checked_in is True:
            checkin_badge = '<span class="badge badge-ok">已签到</span>'
        elif checked_in is False:
            checkin_badge = '<span class="badge badge-active">未签到</span>'
        else:
            checkin_badge = '<span class="badge badge-none">未知</span>'
        rows += f"""<tr id="row-{html_mod.escape(aid)}" class="{active_row.strip()}" data-account-id="{html_mod.escape(aid)}">
          <td><strong>{html_mod.escape(label)}</strong><small class="row-subtitle">{html_mod.escape(uid)}</small></td>
          <td><code>{html_mod.escape(uid)}</code></td>
          <td>{st} {act}</td>
          <td class="muted-cell">{html_mod.escape(expires)}</td>
          <td><span id="credits-{html_mod.escape(aid)}" class="credit-value">{credits_text}</span></td>
          <td><span id="work-credits-{html_mod.escape(aid)}" class="credit-value">{work_credits_text}</span></td>
          <td><span id="total-credits-{html_mod.escape(aid)}" class="credit-value">{total_credits_text}</span></td>
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
            <thead><tr><th>账号</th><th>用户ID</th><th>状态</th><th>有效期</th><th>通用积分</th><th>Work积分</th><th>总积分</th><th>签到状态</th><th>操作</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""
    else:
        # Keep the account actions in the DOM even before the first login.
        # This gives the console a stable control surface and lets the same
        # frontend code handle an account list that becomes populated after a
        # login without requiring a page-specific script branch.
        accounts_html = '''
        <p style="font-size:13px;color:#9aa0b0;margin-bottom:8px">暂无账号，请先登录或手动添加。</p>
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
    poll_checked = 'checked' if polling.get("enabled") else ''
    poll_mode_rr = 'checked' if polling.get('mode', 'round-robin') == 'round-robin' else ''
    poll_mode_cp = 'checked' if polling.get('mode') == 'credit-priority' else ''

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trae CN Relay 控制台</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #0f1117; color: #e8eaed; min-height: 100vh; display: flex; align-items: flex-start; justify-content: center;
  padding: 24px;
}}
.panel {{
  background: #1a1d28; border-radius: 8px; max-width: 1500px; width: 100%;
  padding: 24px; border: 1px solid #2d3140; box-shadow: 0 18px 45px rgba(0,0,0,.18);
}}
.panel-grid {{
  display: block;
  width: 100%;
}}
.panel-card {{
  background: #151823;
  border: 1px solid #2d3140;
  border-radius: 8px;
  padding: 18px 16px;
  min-width: 0;
  overflow-x: auto;
  margin-bottom: 16px;
}}
h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
.section-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
.section-head > div {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.section-meta {{ color:#7f8799; font-size:12px; font-weight:400; }}
.status-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; font-size: 13px; }}
.status-row .label {{ color: #9aa0b0; }}
.status-row .separator {{ margin-left: 8px; }}
.status-row code {{ background: #252836; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
.badge {{ display:inline-flex; align-items:center; gap:4px; font-size: 12px; padding: 3px 9px; border-radius: 999px; font-weight: 600; white-space:nowrap; }}
[hidden] {{ display:none !important; }}
.badge-ok {{ background: #1f6c3a; color: #a8e6b8; }}
.badge-expired {{ background: #6c3a1f; color: #e6b8a8; }}
.badge-none {{ background: #2d3140; color: #9aa0b0; }}
.badge-active {{ background: #1a3a6c; color: #a8c6e6; }}
.user-id {{ color: #8ab4f8; font-size: 12px; }}
hr {{ border: none; border-top: 1px solid #2d3140; margin: 16px 0; }}
.btn {{
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 500;
  cursor: pointer; border: 1px solid transparent; transition: all .15s;
  text-decoration: none; color: #fff; min-height: 32px;
}}
.btn-sm {{ padding: 5px 10px; font-size: 12px; }}
.btn-primary {{ background: #1a8c5c; border-color: #1a8c5c; }}
.btn-primary:hover {{ background: #14a06a; }}
.btn-primary:disabled {{ opacity: .5; cursor: not-allowed; }}
.btn-secondary {{ background: #2d3140; border-color: #3a3f54; }}
.btn-secondary:hover {{ background: #3a3f54; }}
.btn-danger {{ background: #8c1f1f; border-color: #8c1f1f; }}
.btn-danger:hover {{ background: #a02020; }}
.btn-ghost {{ background: transparent; border-color: #3a3f54; color: #9aa0b0; }}
.btn-ghost:hover {{ border-color: #5a5f74; color: #e8eaed; }}
.btn-danger {{ background: transparent; border-color:#67323a; color:#e9a8ae; }}
.btn-danger:hover {{ background:#55252d; border-color:#914550; color:#ffd9dd; }}
.btn-group {{ display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }}
.account-toolbar {{ align-items:center; margin: 10px 0 14px; }}
.account-toolbar .inline-msg {{ margin:0; flex:1 1 260px; }}
.form-group {{ margin-bottom: 12px; }}
.form-group label {{ display: block; font-size: 12px; color: #9aa0b0; margin-bottom: 4px; }}
.form-group input, .form-group textarea {{
  width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid #3a3f54;
  background: #252836; color: #e8eaed; font-size: 13px; font-family: "SF Mono", Consolas, monospace;
}}
.form-group textarea {{ resize: vertical; min-height: 60px; }}
.form-group input:focus, .form-group textarea:focus {{ outline: none; border-color: #1a8c5c; }}
.acct-table {{ width: 100%; min-width: 980px; border-collapse: collapse; font-size: 13px; }}
.acct-table th, .acct-table td {{ text-align: left; padding: 9px 8px; border-bottom: 1px solid #2d3140; vertical-align: middle; }}
.acct-table tbody tr {{ transition: background .15s ease; }}
.acct-table tbody tr:hover {{ background:#1d2130; }}
.acct-table tbody tr.active-row {{ background:rgba(26,140,92,.12); box-shadow:inset 3px 0 #1a8c5c; }}
.acct-table tbody tr.active-row:hover {{ background:rgba(26,140,92,.18); }}
.acct-table tbody tr.row-failed {{ background:rgba(140,31,31,.12); }}
.acct-table tbody tr.row-failed:hover {{ background:rgba(140,31,31,.2); }}
.acct-table th {{ color: #9aa0b0; font-weight: 600; font-size: 12px; position:sticky; top:0; background:#151823; z-index:1; }}
.acct-table th:nth-child(n+5):nth-child(-n+7), .acct-table td:nth-child(n+5):nth-child(-n+7) {{ text-align:right; }}
.credit-value {{ font-variant-numeric: tabular-nums; white-space:nowrap; color:#cbd0dc; }}
.muted-cell {{ color:#8c93a4; font-size:12px; white-space:nowrap; }}
.row-subtitle {{ display:block; color:#7f8799; font-size:11px; margin-top:3px; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.row-actions {{ white-space:nowrap; }}
.row-actions .btn {{ margin:2px 0; }}
.usage-records-container {{ max-height: 320px; overflow: auto; }}
.usage-table {{ width: 100%; min-width: 760px; border-collapse: collapse; font-size: 13px; table-layout: fixed; }}
.usage-table th, .usage-table td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid #2d3140; overflow-wrap: anywhere; }}
.usage-table th {{ color: #9aa0b0; font-weight: 500; font-size: 12px; position: sticky; top: 0; background: #151823; }}
.usage-table .numeric {{ text-align: right; font-variant-numeric:tabular-nums; }}
.usage-table .usage-status {{ white-space:nowrap; }}
.usage-empty {{ padding: 18px 12px; color: #888; text-align: center; font-size: 13px; }}
.msg {{ margin-top: 12px; padding: 8px 12px; border-radius: 6px; font-size: 13px; display: none; line-height:1.45; }}
.msg-ok {{ background: #1f6c3a; color: #a8e6b8; display: block; }}
.msg-err {{ background: #6c1f1f; color: #e6a8a8; display: block; }}
.busy-indicator {{ display:none; color:#9aa0b0; font-size:12px; align-items:center; gap:6px; }}
.busy-indicator.visible {{ display:inline-flex; }}
.busy-indicator::before {{ content:""; width:10px; height:10px; border:2px solid #4a5064; border-top-color:#76d5a5; border-radius:50%; animation:relay-spin .7s linear infinite; }}
@keyframes relay-spin {{ to {{ transform:rotate(360deg); }} }}
.toast {{ position:fixed; top:20px; right:20px; z-index:20; width:min(420px,calc(100vw - 40px)); padding:12px 14px; border:1px solid #3a3f54; border-radius:8px; background:#1d2130; color:#e8eaed; box-shadow:0 12px 32px rgba(0,0,0,.35); opacity:0; transform:translateY(-8px); pointer-events:none; transition:opacity .18s ease, transform .18s ease; white-space:pre-wrap; line-height:1.45; }}
.toast.visible {{ opacity:1; transform:translateY(0); }}
.toast.ok {{ border-color:#2c8150; }}
.toast.error {{ border-color:#9a3d47; color:#ffd9dd; background:#331d25; }}
.toast-title {{ display:block; font-size:12px; font-weight:700; margin-bottom:3px; color:#a8e6b8; }}
.toast.error .toast-title {{ color:#ffb7bf; }}
.busy {{ opacity:.65; pointer-events:none; }}
.loading {{ margin-top: 12px; display: none; font-size: 13px; color: #9aa0b0; }}
.section-title {{ font-size: 14px; font-weight: 600; color: #c8cbd6; margin: 14px 0 8px; }}
.check-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #9aa0b0; }}
@media (max-width: 720px) {{
  body {{ padding:10px; }}
  .panel {{ padding:14px 10px; }}
  .panel-card {{ padding:14px 10px; }}
  h1 {{ font-size:18px; }}
  .status-row {{ font-size:12px; }}
  .btn {{ padding:8px 12px; }}
  .btn-sm {{ padding:6px 9px; }}
  .account-toolbar {{ align-items:stretch; }}
  .account-toolbar .btn {{ flex:1 1 150px; }}
  .account-toolbar .inline-msg {{ flex-basis:100%; }}
  .usage-table {{ min-width:560px; }}
}}
</style>
</head>
<body>
<div id="toast" class="toast" role="alert" aria-live="assertive"><span id="toast-title" class="toast-title"></span><span id="toast-text"></span></div>
<div class="panel">
<h1>Trae CN Relay 控制台</h1>
<div class="status-row" style="justify-content:space-between;align-items:center">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">{status_html}</div>
  {logout_btn}
</div>
<hr>
<div class="panel-grid">
<div class="panel-card">
<div class="section-title">授权登录</div>
<p style="font-size:13px;color:#9aa0b0;margin-bottom:8px;line-height:1.5">
  1. 点击下方按钮后会自动检测本机授权助手（<code>127.0.0.1:{listener_port}</code>）。<br>
  2. 若未检测到，请先下载 <code>web_login.py</code>（或一键 <code>start_auth.bat</code>）并在<b>本机</b>运行，再重试。<br>
  3. 确保浏览器已登录 <a href="https://www.trae.cn" target="_blank" rel="noopener" style="color:#8ab4f8">trae.cn</a>，授权完成后凭据自动写入服务器。
</p>
<div class="btn-group">
  <button class="btn btn-primary" onclick="startAuth()" id="auth-btn">使用 Trae 网页授权登录</button>
  <a class="btn btn-ghost" href="https://www.trae.cn" target="_blank" rel="noopener">访问 trae.cn</a>
</div>
<div class="btn-group">
  <a class="btn btn-secondary" href="/web/login/download" download>下载本机授权助手 web_login.py</a>
  <a class="btn btn-ghost" href="/web/login/download?as=bat" download id="bat-link" style="display:none">下载一键启动 start_auth.bat</a>
</div>
<div id="loading" class="loading">等待授权中...</div>
<div id="auth-msg" class="msg"></div>
<hr>
</div>
<div class="panel-card">
<div class="section-title">账号列表</div>
{accounts_html}
<details>
  <summary style="font-size:13px;color:#9aa0b0;cursor:pointer">手动填写凭证添加账号</summary>
  <form id="manual-form" style="margin-top:12px">
    <div class="form-group">
      <label>Token（Cloud-IDE-JWT）<span style="color:#e6a8a8">*</span></label>
      <textarea name="token" required placeholder="eyJhbGciOiJSUzI1NiI6Ik9wZW5TU0..."></textarea>
    </div>
    <div class="form-group">
      <label>Refresh Token</label>
      <input name="refreshToken" placeholder="可选">
    </div>
    <div class="form-group">
      <label>User ID</label>
      <input name="userId" placeholder="可选">
    </div>
    <div class="form-group">
      <label>Client ID</label>
      <input name="clientId" value="{html_mod.escape(client_id)}">
    </div>
    <div class="form-group">
      <label>备注（标签）</label>
      <input name="label" placeholder="可选">
    </div>
    <button type="submit" class="btn btn-primary">添加账号</button>
    <div id="manual-msg" class="msg"></div>
  </form>
</details>
</div>
<div class="panel-card" id="usage-panel">
<div class="section-head"><div class="section-title">消费记录</div><span id="usage-updated" class="section-meta"></span></div>
<div id="usage-msg" class="msg" role="status" aria-live="polite"></div>
{usage_records_html}
</div>
<div class="panel-card">
<div class="section-title">多账号轮询</div>
<div class="check-row">
  <input type="checkbox" id="poll-toggle" {poll_checked} onchange="togglePolling()">
  <label for="poll-toggle" style="cursor:pointer">启用轮询（每次请求自动切换下一个有效账号）</label>
</div>
<div class="check-row" style="margin-top:4px">
  <span class="label" style="margin-right:8px">轮询模式:</span>
  <label style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;cursor:pointer">
    <input type="radio" name="poll-mode" value="round-robin" onchange="togglePolling()" {poll_mode_rr}> 顺序轮询
  </label>
  <label style="display:inline-flex;align-items:center;gap:4px;cursor:pointer">
    <input type="radio" name="poll-mode" value="credit-priority" onchange="togglePolling()" {poll_mode_cp}> 积分优先
  </label>
</div>
<p id="poll-status" style="font-size:12px;color:#9aa0b0;margin-top:4px">当前账号数: {polling.get('account_count', 0)}，轮询: {'开' if polling.get('enabled') else '关'}</p>
<hr>
<div class="section-title">自定义 URL / 端口</div>
<div class="form-group">
  <label>Web Base URL</label>
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
<hr>
<div class="section-title">模型列表</div>
<div class="form-group">
  <label>刷新 /v1/models（TRAE_FETCH_MODEL_LIST=true 时从上游拉取，否则返回内置列表）</label>
</div>
<div class="btn-group">
  <button class="btn btn-secondary" onclick="refreshModels()">获取模型列表</button>
</div>
<pre id="models-out" style="margin-top:12px;padding:12px;background:#252836;border-radius:6px;font-size:12px;max-height:220px;overflow:auto;white-space:pre-wrap;color:#c8cbd6;display:none"></pre>
<div id="models-msg" class="msg"></div>
</div>
</div>
<script>
const state = {{ traceId: null, win: null }};
let currentCodeVerifier = '';
function uuid() {{ return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,function(c){{var r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16)}}) }}
function randomHex(n) {{ var a=new Uint8Array(n);crypto.getRandomValues(a);return Array.from(a,b=>b.toString(16).padStart(2,'0')).join('') }}
function randomDigits(n) {{ var s='';while(s.length<n)s+=Math.floor(Math.random()*1e10).toString();return s.slice(0,n) }}
function randomBase64Url(n) {{ var a=new Uint8Array(n); if(window.crypto&&crypto.getRandomValues){{ crypto.getRandomValues(a) }} else {{ for(var i=0;i<n;i++)a[i]=Math.floor(Math.random()*256) }} return btoa(String.fromCharCode.apply(null,a)).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'') }}
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
async function refreshUsage() {{
  if(usageRefreshing) return;
  usageRefreshing=true;
  var tbody=document.getElementById('usage-records-body');
  var empty=document.getElementById('usage-empty');
  var msg=document.getElementById('usage-msg');
  try {{
    var result=await requestJSON('/api/usage/records',{{method:'GET'}},15000);
    if(!result.ok) throw new Error(apiError(result.data,result.status));
    var records=Array.isArray(result.data)?result.data:[];
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
  var values=[['credits-',data&&data.account_credits],['work-credits-',data&&data.work_credits],['total-credits-',data&&data.total_credits]];
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
  if(detailEl){{ detailEl.textContent=error||detail||''; detailEl.style.color=error?'#f0a2aa':''; }}
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
  var d=await postJSON('/api/settings',{{web_base_url:web,relay_port:port}});
  if(d.success){{ showMsg('settings-msg',d.note||'设置已保存',true); setTimeout(function(){{ location.reload(); }},800); }}
  else showMsg('settings-msg',d.error||'保存失败',false);
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
    var account={{id:id,data:d.data,checked_in:d.checked_in,account_credits:d.account_credits,work_credits:d.work_credits,total_credits:d.total_credits,success:d.success,error:d.success?'':apiError(d,result.status)}};
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
document.getElementById('manual-form').addEventListener('submit',async function(e){{
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
body {{ font:16px -apple-system,sans-serif;background:#0f1117;color:#e8eaed;padding:40px; }}
.msg {{ padding:20px;border-radius:8px;margin-bottom:16px; }}
.ok {{ background:#1f6c3a; }}
.err {{ background:#6c1f1f; }}
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
                        "delta": {"role": "assistant", "content": ""},
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
    task = asyncio.create_task(_dispatch_chat(messages, model, True, options))
    # Give the task one event-loop turn to enter the selected upstream path
    # (and, for raw/remote transports, begin opening the provider request).
    await asyncio.sleep(0)
    response = None
    iterator = None
    request_id = str((options or {}).get("_relay_request_id") or "")
    started_at = time.monotonic()
    upstream_chunks = 0
    saw_done = False
    stream_status = "opening"
    sent_start_event = False
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
            stream_status = "upstream_error"
            yield _stream_error_event(response)
            yield "data: [DONE]\n\n"
            saw_done = True
            return

        iterator = getattr(response, "body_iterator", None)
        if iterator is None:
            body = getattr(response, "body", b"")
            if body:
                yield body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
            return
        async for chunk in iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            if chunk:
                upstream_chunks += 1
                if "data: [DONE]" in chunk:
                    saw_done = True
                    stream_status = "completed"
                # When headers arrived quickly but the model has not produced
                # its first data frame, the raw translator emits a comment
                # heartbeat. Put one parseable OpenAI start event before that
                # first comment so zcode does not classify the stream as empty.
                if not sent_start_event and chunk.lstrip().startswith(":"):
                    yield _stream_start_event(model)
                    sent_start_event = True
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
        if iterator is not None:
            close_iterator = getattr(iterator, "aclose", None)
            if close_iterator is not None:
                try:
                    await close_iterator()
                except Exception:
                    pass
        close_response = getattr(response, "close", None)
        if close_response is not None:
            try:
                close_response()
            except Exception:
                pass
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
    tool_catalog = (
        options["tools"]
        if "tools" in options
        else options.get("_inherited_tools", [])
    )
    return {
        # API callers execute tools. With no tools field, suppress any internal
        # Trae tool event instead of exposing a call the client cannot handle.
        "allowed_tools": tool_catalog,
        "tool_choice": options.get("tool_choice"),
        "parallel_tool_calls": options.get("parallel_tool_calls"),
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
        isinstance(message, dict)
        and (
            message.get("role") == "tool"
            or (
                message.get("role") == "assistant"
                and isinstance(message.get("tool_calls"), list)
                and bool(message["tool_calls"])
            )
        )
        for message in (messages or [])
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
            if rotate_for_new and UPSTREAM_MODE in (
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
        if UPSTREAM_MODE in ("raw", "direct", "auto"):
            bound["_auth_user_id"] = lease.billing_id or lease.account_id
    if lease.billing_id:
        bound["_billing_id"] = lease.billing_id
    if lease.auth_token:
        bound["_auth_token"] = lease.auth_token
    if UPSTREAM_MODE != "cli":
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

    def bind_usage_turn(self, usage_turn_id: Any) -> None:
        value = str(usage_turn_id or "").strip()
        if value and not self.usage_turn_id:
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
    """Pass SSE usage to the request tracker without duplicating rows."""
    if not chunk.startswith("data: "):
        return
    try:
        data = json.loads(chunk[len("data: "):].strip())
    except Exception:
        return
    usage = data.get("usage")
    if not usage:
        return
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


def _bind_usage_turn(usage_turn_id: Any) -> None:
    tracker = _USAGE_TRACKER.get()
    if tracker is not None:
        tracker.bind_usage_turn(usage_turn_id)


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


def _chunk_marks_terminal(chunk: Any) -> bool:
    """Whether an SSE chunk proves the response reached its end.

    Chat Completions ends with ``data: [DONE]``; the Responses translation
    ends with a ``response.completed``/``response.incomplete`` event.  Both
    mean the upstream finished and billed the turn even if the API client
    disconnects a moment later.
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
    return (
        "data: [DONE]" in text
        or "event: response.completed" in text
        or "event: response.incomplete" in text
    )


async def _tracked_stream(source, tracker: _UsageTracker):
    context_token = _USAGE_TRACKER.set(tracker)
    status = "completed"
    saw_terminal = False
    try:
        await tracker.begin()
        async for chunk in source:
            if not saw_terminal and _chunk_marks_terminal(chunk):
                saw_terminal = True
            yield chunk
    except asyncio.CancelledError:
        status = "completed" if saw_terminal else "cancelled"
        raise
    except GeneratorExit:
        status = "completed" if saw_terminal else "cancelled"
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
                            _bind_usage_turn(retry_message_id)
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
            _bind_usage_turn(message_id)
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
                if "parallel" not in message.lower() and "429" not in message:
                    raise
                if attempt_index + 1 >= attempts:
                    break
                if not await rotate_remote_account("parallel limit"):
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


async def _dispatch_chat(messages, model, stream: bool, options: Optional[dict] = None):
    options = options or {}
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
            UPSTREAM_MODE,
            model,
        )

    # Raw v2 is the default for every model. Operators can opt specific models
    # (or ``*``) into the account-bound remote executor for diagnostics.
    if _requires_remote_model(model) and UPSTREAM_MODE in (
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
    if UPSTREAM_MODE == "cli":
        modes = ["cli"]
    elif UPSTREAM_MODE in ("raw", "direct"):
        modes = ["raw"]
    elif UPSTREAM_MODE in ("remote", "9router", "trae-remote"):
        modes = ["remote"]
    elif UPSTREAM_MODE == "web":
        modes = ["web"]
    elif UPSTREAM_MODE == "ide":
        modes = ["ide"]
    elif UPSTREAM_MODE in ("traework-native", "native", "traework"):
        modes = ["traework-native"]
    else:
        modes = ["raw"]

    logger.info(
        "dispatch route id=%s mode=%s candidates=%s tool_protocol=%s",
        str(options.get("_relay_request_id") or ""),
        UPSTREAM_MODE,
        ",".join(modes),
        tool_protocol_requested,
    )

    for mode in modes:
        try:
            if mode == "raw":
                return await run_raw_chat(messages, model, stream, options)
            mode_options = dict(options)
            # Web and remote routes must receive the lease-bound credential;
            # otherwise a concurrent account switch can make the upstream bill
            # one token while the usage tracker records another.
            if mode not in ("remote", "web", "ide", "traework-native"):
                mode_options.pop("_auth_token", None)
                mode_options.pop("_account_id", None)
            if mode == "cli":
                return await run_cli_chat(messages, model, stream, mode_options)
            if mode == "web":
                return await _run_web_with_retry(messages, model, stream, mode_options)
            if mode == "remote":
                return await _run_remote_with_retry(messages, model, stream, mode_options)
            if mode == "traework-native":
                return await run_traework_native_chat(
                    messages, model, stream, mode_options
                )
            return await run_ide_chat(messages, model, stream, mode_options)
        except Exception as e:
            logger.warning("upstream %s failed: %s", mode, e)
            errors.append(f"{mode}: {e}")

    return _openai_error(502, "All upstream paths failed: " + "; ".join(errors), "api_error")


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
            if float(checkin.get("retry_backoff") or 0) <= 0:
                continue
            if checkin.get("checked_in") is True and _checkin_cache_is_today(record):
                _checkin_clear_retry_state(account_id)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_app()
    web_reaper = asyncio.create_task(_web_reaper_loop())
    terminal_reaper = asyncio.create_task(_terminal_session_reaper_loop())
    checkin_retry = asyncio.create_task(_checkin_auto_retry_loop())
    try:
        yield
    finally:
        for reaper in (web_reaper, terminal_reaper, checkin_retry):
            reaper.cancel()
        await asyncio.gather(web_reaper, terminal_reaper, return_exceptions=True)
        await asyncio.gather(checkin_retry, return_exceptions=True)
        await _cancel_usage_tasks()


app = FastAPI(title="Trae CN Relay", version="1.2.0", lifespan=lifespan)


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


@app.get("/api/checkin/work-credits/{account_id}")
async def api_checkin_work_credits(account_id: str):
    """Fetch work-specific account credits for one account.

    Work credits = total (all packs) - general (req_source=1).
    """
    rec = auth.get_account_record(account_id)
    token = rec.get("token") or ""
    if not token:
        return JSONResponse({"success": False, "error": "account not found or token missing"}, status_code=404)
    try:
        # Fetch general and total
        raw_ac = await trae_client.fetch_account_credits(token)
        general = trae_client.parse_account_credits(raw_ac)
        raw_total = await trae_client.fetch_account_total_credits(token)
        total = trae_client.parse_account_credits(raw_total)
        work = _sub_credits(total, general)
        auth.merge_account_credits(
            account_id,
            {
                "work_credits": work,
                "total_credits": total,
                "account_credits": general,
            },
        )
        return JSONResponse({"success": True, "id": account_id, "credits": work, "raw_total": raw_total, "raw_general": raw_ac})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=502)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not API_KEYS:
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
        "web_base": WEB_BASE,
        "upstream_mode": UPSTREAM_MODE,
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
            "parallel_tool_calls": True,
            "server_executes_caller_tools": False,
            "tool_upstreams": (
                ["raw"]
                if UPSTREAM_MODE in ("raw", "direct", "auto")
                else ["cli"]
                if UPSTREAM_MODE == "cli"
                else ["traework-native"]
                if UPSTREAM_MODE in ("traework-native", "native", "traework")
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


@app.post("/v1")
async def chat_v1(req: Request):
    return await handle_chat(req)


@app.post("/v1/chat")
async def chat_v1_chat(req: Request):
    return await handle_chat(req)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    return await handle_chat(req)


@app.post("/v1/responses")
async def responses(req: Request):
    return await handle_responses(req)


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

def _sub_credits(total: dict | None, general: dict | None) -> dict | None:
    """Calculate work-specific credits = total - general."""
    if not total or not general:
        return None
    if total.get("unlimited") or general.get("unlimited"):
        return {"total_limit": -1, "used": 0, "remaining": None, "unlimited": True}
    t_limit = total.get("total_limit") or 0
    g_limit = general.get("total_limit") or 0
    t_used = total.get("used") or 0
    g_used = general.get("used") or 0
    wt = max(t_limit - g_limit, 0)
    wu = max(t_used - g_used, 0)
    return {
        "total_limit": wt,
        "used": wu,
        "remaining": max(wt - wu, 0),
        "unlimited": False,
    }

async def _fetch_full_credits(token: str) -> dict:
    """Fetch general credits and the combined (all-packs) credits for one token.

    Upstream semantics:
      - req_source=1 returns only general IDE packs (account_credits).
      - req_source=0/2 returns all packs, i.e. the actual TOTAL credits.
    So total_credits = req_source=0 result, and work_credits = total - general.
    """
    raw_ac, raw_total = await asyncio.gather(
        trae_client.fetch_account_credits(token),
        trae_client.fetch_account_total_credits(token),
        return_exceptions=True,
    )
    if isinstance(raw_ac, BaseException) and isinstance(raw_total, BaseException):
        raise RuntimeError(
            "Trae credits query failed: general and total credits are unavailable"
        ) from raw_ac
    account_credits = (
        None
        if isinstance(raw_ac, BaseException)
        else trae_client.parse_account_credits(raw_ac)
    )
    total_credits = (
        None
        if isinstance(raw_total, BaseException)
        else trae_client.parse_account_credits(raw_total)
    )
    work_credits = _sub_credits(total_credits, account_credits)
    return {
        "account_credits": account_credits,
        "work_credits": work_credits,
        "total_credits": total_credits,
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
        "work_credits": cached.get("work_credits"),
        "total_credits": cached.get("total_credits"),
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
            "work_credits": merged.get("work_credits"),
            "total_credits": merged.get("total_credits"),
            "checkin": merged,
        }
    )
    return row


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
        "work_credits": merged.get("work_credits"),
        "total_credits": merged.get("total_credits"),
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
    backoff = base * (2 ** min(count, 10))
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
    for key in ("account_credits", "work_credits", "total_credits"):
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
            # Prefer total credits for the "credit-priority" sort key so work
            # and general entitlements both count toward account priority.
            total_parsed = full.get("total_credits") or {}
            credits_sort = total_parsed.get("remaining") or total_parsed.get("total_limit") or parsed.get("remaining") or parsed.get("total_limit") or 0
            if total_parsed.get("unlimited") or parsed.get("unlimited"):
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
    try:
        relay_port = int(relay_port)
    except (TypeError, ValueError):
        relay_port = 0
    if not web_base_url and not relay_port:
        return JSONResponse({"success": False, "error": "nothing to update"}, status_code=400)
    auth.set_relay_settings(web_base_url=web_base_url, port=relay_port)
    return JSONResponse({"success": True, "note": "端口变更需重启容器生效"})


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
