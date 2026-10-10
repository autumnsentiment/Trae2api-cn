"""
trae_client.py - Trae API 客户端

支持两条上游路径：
1. IDE/桌面版：先走 /api/agent/v3/llm_utils_chat，失败后依次回退
   /api/ide/v1/chat 与 /api/agent/v3/create_agent_task
2. 网页版 remote 会话（OmniRoute 风格）：POST /chat_sessions +
   GET /chat_sessions/{id}/events
"""

import asyncio
import base64
import binascii
import copy
import hashlib
import json
import logging
import math
import os
import random
import string
import time
import uuid
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator, Mapping, Optional

import anyio
import httpx

from . import auth, raw_client
from .cli_client import (
    renderer_block_text,
    sanitize_assistant_history_messages,
    tool_result_is_failed,
)
from .model_limits import clamp_max_completion_tokens

logger = logging.getLogger(__name__)

IDE_VERSION = os.environ.get("TRAE_IDE_VERSION", "3.3.67")
IDE_VERSION_CODE = os.environ.get("TRAE_IDE_VERSION_CODE", "20260401")
IDE_VERSION_CODE_NUM = int(os.environ.get("TRAE_IDE_VERSION_CODE_NUM", "20260401") or 20260401)
IDE_APP_ID = os.environ.get("TRAE_APP_ID", "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8")
TRAE_CLIENT_ID = os.environ.get("TRAE_CLIENT_ID", "ono9krqynydwx5")
TRAE_UG_API_HOST = os.environ.get("TRAE_UG_API_HOST", "https://api.trae.cn")
TRAE_PAY_API_HOST = os.environ.get("TRAE_PAY_API_HOST", "https://api.trae.cn")
# The commercial usage endpoint is separate from the entitlement/credits API.
# Keep it independently configurable because TraeWork reports session charges
# through the api5-normal host rather than api.trae.cn.
TRAE_USAGE_API_HOST = os.environ.get(
    "TRAE_USAGE_API_HOST", "https://api5-normal.mchost.guru"
)

# 按优先级依次尝试的 IDE 版上游端点（参考 laojichao/trae-local-api）
IDE_ENDPOINTS = [
    "/api/agent/v3/llm_utils_chat",
    "/api/ide/v1/chat",
    "/api/agent/v3/create_agent_task",
]
DEFAULT_MODEL_GATEWAY = "https://trae-api-cn.mchost.guru"
# Legacy Work ``create_agent_task`` is served from the Work gateway.  The
# native Work Lite ``workflow/start`` path remains on trae-api-cn.mchost.guru;
# keeping this host separate prevents an IDE request from being sent to the
# OAuth/API host and mirrors the captured TraeWork client routing.
WORK_AGENT_HOST = os.environ.get("TRAE_WORK_AGENT_HOST", "https://api5-normal.mchost.guru").rstrip("/")
WORK_AGENT_APP_ID = os.environ.get("TRAE_WORK_APP_ID", "931506")
# create_agent_task is sent with the chat application id in TraeWork's native
# client; 931506 is the Work function/app id used by the config probe itself.
WORK_AGENT_APP_ID_CHAT = os.environ.get("TRAE_WORK_APP_ID_CHAT", IDE_APP_ID)
WORK_AGENT_VERSION = os.environ.get("TRAE_WORK_IDE_VERSION", "0.1.63")
WORK_AGENT_VERSION_CODE = os.environ.get("TRAE_WORK_IDE_VERSION_CODE", "20260901")
# SOLO/llm_utils_chat uses a different client version family from the legacy
# IDE payload builder.  Keep it independently configurable so old desktop
# request values do not make the gateway reject otherwise valid parameters.
SOLO_GATEWAY_VERSION = os.environ.get("TRAE_SOLO_GATEWAY_VERSION", "0.1.52")
SOLO_GATEWAY_VERSION_CODE = os.environ.get(
    "TRAE_SOLO_GATEWAY_VERSION_CODE", "20260811"
)
SOLO_DEVICE_BRAND = os.environ.get("TRAE_SOLO_DEVICE_BRAND", "83DG")
SOLO_OS_VERSION = os.environ.get("TRAE_SOLO_OS_VERSION", "Windows 11 Pro")
SOLO_DEVICE_TYPE = os.environ.get("TRAE_SOLO_DEVICE_TYPE", "windows")
SOLO_PACKAGE_TYPE = os.environ.get("TRAE_SOLO_PACKAGE_TYPE", "stable_cn")
SOLO_PLATFORM = os.environ.get("TRAE_SOLO_PLATFORM", "windows")
SOLO_LSCBD_AID = os.environ.get("TRAE_SOLO_LSCBD_AID", "787976")

# 外部模型名 -> Trae CN 内部模型名（基于最新 Trae CN 模型映射）
MODEL_ALIASES = {
    # OpenAI / Claude 外部名 -> Trae CN 内部模型名
    "auto": "glm-5.2",
    "glm-5.2": "glm-5.2",
    "glm-5.3": "glm-5.3",
    "claude-opus-4-7": "glm-5.2",
    "claude-opus-4-6": "glm-5.2",
    "claude-opus-4-5": "glm-5.2",
    "claude-sonnet-4-6": "glm-5.2",
    "claude-sonnet-4-5": "glm-5.2",
    "claude-sonnet-4": "glm-5.2",
    "claude-3.7-sonnet": "glm-5.2",
    "claude-3-7-sonnet": "glm-5.2",
    "claude-3.5-sonnet": "glm-5.2",
    "claude3.5": "glm-5.2",
    "aws_sdk_claude37_sonnet": "glm-5.2",
    "claude-haiku-4-5": "glm-5.1",
    "glm-5.1": "glm-5.1",
    "glm-5": "glm-5",
    "glm-4.7": "glm-4.7",
    "glm-4.6": "glm-4.6",
    # Trae 真实模型名直通（保持原始大小写，web 上游按 config_name 精确匹配）
    "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash-Official": "DeepSeek-V4-Flash-Official",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-v4-pro 正式版": "DeepSeek-V4-Pro-Official",
    "deepseek-v4-flash 正式版": "DeepSeek-V4-Flash-Official",
    "seed-2.1-pro": "Doubao-Seed-2.1-Pro",
    "seed-2.1-turbo": "Doubao-Seed-2.1-Turbo",
    "seed-code": "Doubao-Seed-Code",
    "seed-evolving": "Doubao-Seed-Evolving",
    "qwen3.7-plus": "qwen-3.7-plus",
    "kimi-k2.6": "kimi-k2.6",
    "kimi-k3": "kimi-k3",
    "kimi-k2.7-code": "kimi-k2.7-code",
    "kimi-k2": "kimi-k2",
    "kimi-k2.5": "kimi-k2.5",
    "qwen-3.7-plus": "qwen-3.7-plus",
    "qwen-3.6-plus": "qwen-3.6-plus",
    "qwen3.8-max": "qwen3.8-max",
    "qwen-3.5": "qwen-3.5",
    "qwen3-coder": "qwen3-coder",
    "minimax-m3": "minimax-m3",
    "minimax-m2.7": "minimax-m2.7",
    "minimax-m2.1": "minimax-m2.1",
    "mimo-v2.5-pro": "mimo-v2.5-pro",
    "mimo-v2.5": "mimo-v2.5",
    "minimax-m25": "minimax-m25",
    "qwen36-35b": "qwen36-35b",
    "Doubao-Seed-2.1-Pro": "Doubao-Seed-2.1-Pro",
    "Doubao-Seed-2.1-Turbo": "Doubao-Seed-2.1-Turbo",
    "Doubao-Seed-Code": "Doubao-Seed-Code",
    "doubao-seed-2.1-pro": "Doubao-Seed-2.1-Pro",
    "doubao-seed-2.1-turbo": "Doubao-Seed-2.1-Turbo",
    "doubao-seed-code": "Doubao-Seed-Code",
    "gpt-4o": "DeepSeek-V4-Pro",
    "gpt-4o-latest": "DeepSeek-V4-Pro",
    "gpt-4.1": "DeepSeek-V4-Pro",
    "deepseek-v3": "DeepSeek-V4-Pro",
    "deepseek-r1": "DeepSeek-V4-Pro",
    "gpt-4o-mini": "DeepSeek-V4-Flash",
    "deepseek-v4-flash-official": "DeepSeek-V4-Flash-Official",
    "deepseek-v4-pro-official": "DeepSeek-V4-Pro-Official",
    "doubao-seed-evolving": "Doubao-Seed-Evolving",
    "work": "work",
}

_ALIAS_LOOKUP = {k.lower(): v for k, v in MODEL_ALIASES.items()}

# /v1/models 返回的模型 ID
SUPPORTED_MODELS = set(_ALIAS_LOOKUP.keys()) | set(_ALIAS_LOOKUP.values())

# 设备信息（参考 trae2api config/device.go）
_DEVICE_BRANDS = ["92L3", "91C9", "814S", "8P15V", "35G4", "65G4", "55G4"]
_WEB_SLOTS: dict[str, asyncio.Semaphore] = {}
_WEB_LEASES: dict[str, dict] = {}
_WEB_MODEL_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
_WEB_PARALLEL_LIMIT = int(os.environ.get("TRAE_WEB_PARALLEL_LIMIT", "2"))
_WEB_IDLE_TIMEOUT = float(os.environ.get("TRAE_WEB_IDLE_TIMEOUT", "60"))
_WEB_MODEL_CACHE_TTL = float(os.environ.get("TRAE_WEB_MODEL_CACHE_TTL", "300"))
_WORKSPACE_PREFIXES = ["User", "home", "workspace", "data"]
_WORKSPACE_DIRS = ["projects", "workspace", "dev", "code", "work"]
_TOOL_OPTION_KEYS = ("tools", "tool_choice", "parallel_tool_calls")


def _web_model_cache_key(token: str = "", user_id_override: str = "") -> str:
    """Return a stable, account-bound key without retaining raw credentials.

    A bound request can carry a token for an account different from the
    process-wide active account. Prefer the explicit billing identity, then
    the immutable JWT identity, and only hash the token as a final fallback.
    """
    token = str(token or "").strip()
    explicit_id = str(user_id_override or "").strip()
    if explicit_id:
        return explicit_id

    if token:
        identity = _checkin_identity(token)
        if identity and identity != token:
            return identity
        return "token:" + hashlib.sha256(token.encode("utf-8")).hexdigest()

    return str(auth.get_user_id() or "default")


def _tool_protocol_requested(
    options: Optional[dict], messages: Optional[list[dict]] = None
) -> bool:
    options = options or {}
    if bool(options.get("_tool_protocol_requested")) or any(
        key in options for key in _TOOL_OPTION_KEYS
    ):
        return True
    return any(
        isinstance(message, dict) and _message_contains_tool_protocol(message)
        for message in (messages or [])
    )


def _message_contains_tool_protocol(message: Mapping[str, Any]) -> bool:
    """Detect OpenAI and TraeWork renderer tool blocks in message history."""

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


def _requested_session_id(options: Optional[dict]) -> str:
    options = options or {}
    return str(options.get("session_id") or options.get("sessionId") or "").strip()


@dataclass
class DeviceInfo:
    machine_id: str
    device_id: str
    brand: str
    use_count: int = 0
    max_uses: int = 3


@dataclass
class IdeChatResponse:
    """IDE 端点的响应与所属客户端，调用方消费完流式内容后负责关闭。"""
    response: httpx.Response
    client: httpx.Client

    def close(self) -> None:
        self.response.close()
        self.client.close()


_current_device: Optional[DeviceInfo] = None


def get_current_device() -> DeviceInfo:
    """获取/轮换设备指纹。trae2api 在几次请求后换一次。"""
    global _current_device
    if _current_device is None or _current_device.use_count >= _current_device.max_uses:
        device = DeviceInfo(
            machine_id="".join(random.choices("0123456789abcdef", k=32)),
            device_id=str(random.randint(10**17, 10**18 - 1)),
            brand=random.choice(_DEVICE_BRANDS),
            use_count=0,
            max_uses=3 + random.randint(0, 2),
        )
        _current_device = device
    _current_device.use_count += 1
    return _current_device


def convert_model_name(model: str) -> str:
    """Map OpenAI/Claude-style names to Trae CN internal names.

    Preserve exact case for known upstream model names (the web upstream
    matches custom_model.config_name precisely)."""
    if not model:
        return model
    m = model.strip().lower()
    return _ALIAS_LOOKUP.get(m) or _UPSTREAM_MODEL_CASE.get(m) or model


def is_model_supported(model: str) -> bool:
    """中转站保持透传：未知模型也交给上游判断，避免新模型上线后需要改代码。"""
    return True


def generate_session_id_from_messages(messages: list[dict]) -> str:
    """Return a fresh stateless chat session id.

    OpenAI callers send the full history on every turn. Reusing an id derived
    from prompt text can merge unrelated callers that happen to start alike.
    Callers that need a stable upstream id should pass ``session_id``.
    """
    del messages
    return str(uuid.uuid4())


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                text = c.get("text") or c.get("content")
                if text in (None, ""):
                    # Trae's renderer stores text/tool_result payloads under
                    # ``value`` instead of ``text``/``content``.
                    text = renderer_block_text(c)
                parts.append(text or "")
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "value", "data"):
            candidate = content.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return str(content)


def _random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _random_username(length: int) -> str:
    if length < 2:
        length = 2
    return random.choice(string.ascii_lowercase) + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=length - 1)
    )


def generate_random_workspace_path() -> str:
    root = random.choice(_WORKSPACE_PREFIXES)
    username = _random_username(4 + random.randint(0, 3))
    project = _random_string(6 + random.randint(0, 4))
    return f"/{root}/{username}/Documents/{random.choice(_WORKSPACE_DIRS)}/project-{project}"


def _rfc3339_zh_time() -> str:
    """当前时间，格式参考 trae2api：2025-03-25 12:00:00，星期X"""
    now = time.localtime()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return time.strftime("%Y-%m-%d %H:%M:%S", now) + "，" + weekdays[now.tm_wday]


def build_headers(
    token_override: str = "", user_id_override: str = ""
) -> dict[str, str]:
    """构造与 trae2api/laojichao 兼容的 IDE 请求头。

    A relay session may pin its credential for a complete tool chain.  The
    optional overrides keep that request independent from global account
    rotation while retaining the historic call signature for other callers.
    """
    device = get_current_device()
    token = token_override or auth.get_token()
    user_id = user_id_override or auth.get_user_id()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Cloud-IDE-JWT {token}",
        "X-Cloudide-Token": token,
        "x-uid": user_id,
        "x-app-id": IDE_APP_ID,
        "x-device-id": device.device_id,
        "x-machine-id": device.machine_id,
        "x-request-id": str(uuid.uuid4()),
        "x-ide-version": IDE_VERSION,
        "x-ide-version-code": IDE_VERSION_CODE,
        "x-ide-version-type": "stable",
        "x-device-cpu": "AMD",
        "x-device-brand": device.brand,
        "x-device-type": "windows",
        "x-ide-token": token,
        "x-os-version": "Windows 10",
        "x-system-type": "Windows",
        "Accept": "text/event-stream",
        "Connection": "keep-alive",
        "User-Agent": "",
    }


def _model_gateway_base(*candidates: Any) -> str:
    """Choose the model gateway, keeping OAuth/auth hosts out of model calls.

    Trae stores ``api.trae.cn``/``api.trae.com.cn`` as account or OAuth hosts
    in some login records.  Those hosts intentionally do not expose the IDE
    model routes.  Explicit non-Trae hosts remain usable for a custom gateway.
    """

    configured = str(os.environ.get("TRAE_MODEL_API_HOST") or "").strip()
    values = [configured, *(str(item or "").strip() for item in candidates)]
    for value in values:
        if not value:
            continue
        value = value.rstrip("/")
        lowered = value.lower()
        if any(host in lowered for host in (
            "api.trae.cn",
            "api.trae.com.cn",
            "trae-api.com.cn",
            "oauth",
        )) and "mchost.guru" not in lowered and "trae-api-" not in lowered:
            continue
        # Account snapshots sometimes persist the Remote API URL as the
        # account host (``.../api/remote/v1``).  IDE/SOLO routes are rooted at
        # the same gateway host, so retaining that path would generate
        # ``/api/remote/v1/api/agent/...`` and yield a misleading 404/502.
        # Keep an explicit custom path only when it is not a known Trae API
        # prefix; this preserves custom gateways while repairing stale account
        # records.
        try:
            parsed = urlsplit(value)
        except ValueError:
            parsed = None
        if parsed and parsed.scheme and parsed.netloc:
            path = (parsed.path or "").rstrip("/")
            if path.startswith(("/api/remote/", "/api/agent/", "/api/ide/")):
                value = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        return value
    return DEFAULT_MODEL_GATEWAY


# Cache of checkin device ids per account. The upstream checkin API is
# device-scoped and rate limits the claim endpoint (code 9074). A JWT is
# refreshed periodically, so the raw token is not a stable device identity;
# derive the id from the immutable user id carried in the JWT instead.
_CHECKIN_DEVICE_IDS: dict[str, str] = {}

CHECKIN_DEVICE_BRAND = os.environ.get(
    "TRAE_CHECKIN_DEVICE_BRAND", "ASUS TUF Gaming A15 FA507RM_FA507RM"
).strip()
CHECKIN_DEVICE_TYPE = os.environ.get("TRAE_CHECKIN_DEVICE_TYPE", "windows").strip()
CHECKIN_OS_VERSION = os.environ.get("TRAE_CHECKIN_OS_VERSION", "Windows 10 Pro").strip()
CHECKIN_APP_VERSION = os.environ.get("TRAE_CHECKIN_APP_VERSION", "3.3.65").strip()


def _configured_checkin_device_id(identity: str, account_id: str = "") -> str:
    """Return a TraeWork device id configured for this account, if any."""
    raw_mapping = os.environ.get("TRAE_CHECKIN_DEVICE_IDS_JSON", "").strip()
    if raw_mapping:
        try:
            mapping = json.loads(raw_mapping)
        except (TypeError, ValueError, json.JSONDecodeError):
            mapping = None
        if isinstance(mapping, dict):
            for key in (identity, account_id):
                value = mapping.get(str(key)) if key else None
                if value is not None and str(value).strip():
                    return str(value).strip()

    return os.environ.get("TRAE_CHECKIN_DEVICE_ID", "").strip()


def _checkin_identity(token: str, account_id: str = "") -> str:
    """Return an account-stable identity without retaining or logging a token."""
    if token:
        try:
            parts = token.split(".")
            if len(parts) >= 2:
                encoded = parts[1] + "=" * (-len(parts[1]) % 4)
                payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
                data = payload.get("data")
                if isinstance(data, dict) and data.get("id"):
                    return str(data["id"])
                for key in ("user_id", "userId", "sub"):
                    if payload.get(key):
                        return str(payload[key])
        except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError):
            pass
    if account_id:
        return str(account_id)
    # Non-JWT credentials are uncommon, but remain deterministic for the
    # lifetime of that credential instead of producing an empty header.
    return token


# Device-id generation per account identity. Advancing it moves an account off
# a device id the upstream has rate limited (business code 9074).
_CHECKIN_DEVICE_GENERATIONS: dict[str, int] = {}


def checkin_device_generation(identity: str) -> int:
    """Return the current device-id generation for an account identity."""

    if not identity:
        return 0
    if identity in _CHECKIN_DEVICE_GENERATIONS:
        return _CHECKIN_DEVICE_GENERATIONS[identity]
    stored = 0
    try:
        record = auth.get_account_record(identity) or {}
        checkin = record.get("checkin") or {}
        stored = int(checkin.get("device_generation") or 0)
    except Exception:
        stored = 0
    if stored < 0:
        stored = 0
    _CHECKIN_DEVICE_GENERATIONS[identity] = stored
    return stored


def rotate_checkin_device_id(token: str = "", account_id: str = "") -> str:
    """Advance to the next device id after an upstream 9074 on the current one.

    Probing showed 9074 is device-scoped: the same account claims successfully
    on a freshly derived id. Returns the new id, or "" when an explicit id is
    configured (rotating past an operator-pinned device id is not our call).
    """

    identity = _checkin_identity(token, account_id)
    if not identity:
        return ""
    if _configured_checkin_device_id(identity, account_id):
        return ""
    generation = checkin_device_generation(identity) + 1
    _CHECKIN_DEVICE_GENERATIONS[identity] = generation
    try:
        auth.merge_account_retry(identity, {"device_generation": generation})
    except Exception:
        logger.warning("checkin device rotation was not persisted")
    return checkin_device_id_for(token, account_id)


def checkin_device_id_for(token: str, account_id: str = "") -> str:
    """Return a stable, account-bound 16-digit device id.

    Business code 9074 ("too many users, retry later") is scoped to the device
    id, not the account: probing showed a claim that returns 9074 on one id
    succeeds immediately on a freshly derived id for the same account. The id
    therefore stays stable across token refreshes, but a caller that keeps
    hitting 9074 can advance ``generation`` to move off an exhausted id.
    """
    identity = _checkin_identity(token, account_id)
    configured = _configured_checkin_device_id(identity, account_id)
    if configured:
        return configured
    if not token and not account_id:
        return ""
    generation = checkin_device_generation(identity)
    cache_key = f"{identity}#{generation}"
    if cache_key in _CHECKIN_DEVICE_IDS:
        return _CHECKIN_DEVICE_IDS[cache_key]
    material = identity if generation <= 0 else f"{identity}#gen{generation}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    did = str(int(digest, 16) % 10**16).zfill(16)
    _CHECKIN_DEVICE_IDS[cache_key] = did
    return did

def build_checkin_headers(token: str, account_id: str = "") -> dict[str, str]:
    """Build headers for the client daily-checkin API."""
    headers = {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
        "x-device-id": checkin_device_id_for(token, account_id),
    }
    if CHECKIN_DEVICE_BRAND and os.environ.get("TRAE_CHECKIN_NO_BRAND") != "1":
        headers["x-device-brand"] = CHECKIN_DEVICE_BRAND
    if CHECKIN_DEVICE_TYPE and os.environ.get("TRAE_CHECKIN_NO_TYPE") != "1":
        headers["x-device-type"] = CHECKIN_DEVICE_TYPE
    if os.environ.get("TRAE_CHECKIN_EXTRA_VERSION_HEADERS") == "1":
        headers["x-os-version"] = CHECKIN_OS_VERSION
        headers["x-app-version"] = CHECKIN_APP_VERSION
    return headers

async def fetch_checkin_credits_status(token: str = "", account_id: str = "") -> dict:
    """Query daily checkin credits status, returns the original JSON data."""
    return await _post_checkin(
        "/trae/api/v2/ug/checkin_credits/status", token, account_id=account_id
    )

async def claim_checkin_credits(token: str = "", account_id: str = "") -> dict:
    """Claim today's credits once and preserve upstream business errors.

    In particular, do not immediately retry code 9074. A second claim within
    seconds extends the upstream rate-limit window and was the reason one UI
    click produced multiple failing upstream requests.
    """
    if not token:
        token = auth.get_token()
    return await _post_checkin(
        "/trae/api/v2/ug/checkin_credits/claim", token, account_id=account_id
    )

async def _post_checkin(
    path: str, token: str = "", *, account_id: str = ""
) -> dict:
    if not token:
        token = auth.get_token()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    base = os.environ.get("TRAE_UG_API_HOST", TRAE_UG_API_HOST).rstrip("/")
    url = f"{base}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url, json={}, headers=build_checkin_headers(token, account_id)
        )
        text = resp.text
        if resp.status_code != 200:
            raise RuntimeError(f"Trae checkin [{resp.status_code}]: {text[:500]}")
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Trae checkin invalid json: {e}") from e
        if not isinstance(data, dict):
            raise RuntimeError("Trae checkin invalid json: top-level value is not an object")

        if path.endswith("/status"):
            if not isinstance(data.get("enable"), bool) or not isinstance(
                data.get("checked_in"), bool
            ):
                raise RuntimeError(
                    "Trae checkin invalid status: enable/checked_in must be boolean"
                )
            credits = data.get("credits")
            if isinstance(credits, bool) or not isinstance(credits, (int, float)):
                data.pop("credits", None)
            elif not math.isfinite(credits) or credits <= 0:
                data.pop("credits", None)

        # Upstream uses business codes in a 200 response. Do not turn 9095 into
        # ``checked_in=True``: its message is device-scoped (the same device
        # may already have checked in for another account), so only the account
        # status endpoint can establish whether this account is checked in.
        code = data.get("code")
        if code == 9074:
            # Preserve the upstream code so retry paths can react to rate
            # limiting; everything else is passed through as-is.
            data = dict(data)
            data["_error_code"] = code
        return data

async def fetch_account_credits(token: str = "", req_source: int = 1) -> dict:
    """Fetch account credits/entitlement usage from the IDE pay API.

    req_source=1 returns general IDE credits, req_source=2 returns work-specific credits.
    This returns the same data the Trae CN client shows for total account
    credits (not the daily-checkin bonus credits field).
    """
    if not token:
        token = auth.get_token()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    base = os.environ.get("TRAE_PAY_API_HOST", TRAE_PAY_API_HOST).rstrip("/")
    url = f"{base}/trae/api/v2/pay/ide_user_ent_usage"
    payload = {"require_usage": True, "req_source": req_source}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=build_checkin_headers(token))
        text = resp.text
        if resp.status_code != 200:
            raise RuntimeError(f"Trae pay credits [{resp.status_code}]: {text[:500]}")
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Trae pay credits invalid json: {e}") from e
        if data.get("code") is not None and data.get("code") != 0:
            raise RuntimeError(f"Trae pay credits: {data.get('message') or data}")
        return data

async def fetch_user_info(token: str = "") -> dict:
    """Fetch the upstream profile (ScreenName/UserID) like the Trae client.

    The client posts to ``/cloudide/api/v3/trae/GetUserInfo`` with the JWT in
    ``x-cloudide-token``; ``Authorization: Cloud-IDE-JWT`` is rejected (401).
    """
    if not token:
        token = auth.get_token()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    base = os.environ.get("TRAE_UG_API_HOST", TRAE_UG_API_HOST).rstrip("/")
    url = f"{base}/cloudide/api/v3/trae/GetUserInfo"
    headers = {"Content-Type": "application/json", "x-cloudide-token": token}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json={"ReqSource": "IDE"}, headers=headers)
        text = resp.text
        if resp.status_code != 200:
            raise RuntimeError(f"Trae user info [{resp.status_code}]: {text[:300]}")
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Trae user info invalid json: {e}") from e
    result = data.get("Result") if isinstance(data, dict) else None
    if not isinstance(result, dict) or not result.get("UserID"):
        raise RuntimeError("Trae user info: empty Result")
    return result


def _credit_decimal(value: Any) -> Decimal | None:
    """Parse an upstream credit value without losing fractional precision."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _json_credit_number(value: Decimal) -> int | float:
    """Return a regular JSON-compatible number while retaining fractions."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def parse_account_credits(data: dict, available_endpoint_filter: int = None) -> dict:
    """Parse total account credits from the entitlement pack list.

    Only counts packs with product_type=2 (actual credits), skipping
    product_type=0 (feature flag packs) that have no credits_limit.

    When available_endpoint_filter is set, only packs with that value
    for entitlement_base_info.available_endpoint are counted:
      - 0 = general IDE credits
      - 1 = work-specific credits
    This is needed because req_source=2 returns ALL packs (same as req_source=0),
    not just work-specific packs.
    """
    packs = data.get("user_entitlement_pack_list") or []
    total_limit = Decimal(0)
    used = Decimal(0)
    unlimited = False
    for pack in packs:
        bi = pack.get("entitlement_base_info") or {}
        ae = bi.get("available_endpoint", -1)
        # If filtering by endpoint, skip packs that don't match
        if available_endpoint_filter is not None and ae != available_endpoint_filter:
            continue
        usage = pack.get("usage") or {}
        quota = bi.get("quota") or {}
        limit = _credit_decimal(quota.get("credits_limit"))
        if limit is None:
            continue  # skip feature-only packs (product_type=0)
        amount = _credit_decimal(usage.get("credits_amount")) or Decimal(0)
        if limit == Decimal(-1):
            unlimited = True
        elif limit >= 0:
            total_limit += limit
        if amount >= 0:
            used += amount
    remaining = None if unlimited else max(total_limit - used, Decimal(0))
    return {
        "total_limit": -1 if unlimited else _json_credit_number(total_limit),
        "used": _json_credit_number(used),
        "remaining": None if remaining is None else _json_credit_number(remaining),
        "unlimited": unlimited,
        "is_credits_billing": bool(data.get("is_credits_billing")),
        "packs": len(packs),
    }


async def fetch_account_work_credits(token: str = "") -> dict:
    """Fetch work-specific account credits.

    req_source=2 returns ALL packs (same as req_source=0), not just work.
    So we fetch req_source=0 and filter by available_endpoint=1 in parse.
    Returns the raw API response for downstream parse_account_credits(data, ae_filter=1).
    """
    return await fetch_account_credits(token, req_source=0)

async def fetch_account_total_credits(token: str = "") -> dict:
    """Fetch all account credits (all packs, both general and work)."""
    return await fetch_account_credits(token, req_source=0)


async def fetch_session_usage(
    session_id: str,
    token: str = "",
    *,
    base_url: str = "",
) -> dict:
    """Read the billable usage for one completed TraeWork turn.

    ``session_id`` is the upstream *user-message/turn id* (usually
    ``reply_to_message_id``), not the relay's fixed chat conversation UUID.
    This endpoint is read-only and is used only as asynchronous billing
    enrichment after the model response has already been returned.
    """

    session_id = str(session_id or "").strip()
    if not session_id:
        raise ValueError("session_id is required")
    token = str(token or auth.get_token() or "").strip()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    base = str(base_url or os.environ.get("TRAE_USAGE_API_HOST", TRAE_USAGE_API_HOST)).rstrip("/")
    url = f"{base}/api/v1/commercial/get_session_usage"
    headers = {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
    }
    timeout = float(os.environ.get("TRAE_USAGE_QUERY_TIMEOUT_SECONDS", "15") or "15")
    timeout = max(1.0, min(timeout, 60.0))
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            json={"session_id": session_id},
            headers=headers,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"Trae session usage [{response.status_code}]")
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Trae session usage invalid json: {exc}") from exc

    if not isinstance(payload, Mapping):
        return {}
    code = payload.get("code")
    if code not in (None, 0, "0"):
        raise RuntimeError(
            f"Trae session usage: {payload.get('message') or payload.get('msg') or code}"
        )
    data = payload.get("data")
    if not isinstance(data, Mapping):
        data = payload

    # The commercial endpoint has shipped several envelopes.  The current
    # client normally returns ``data.user_usage_group_by_session`` but older
    # gateways return that object at the top level, wrap it in ``usage`` or
    # use camelCase keys.  Walk only billing-shaped fields and return the
    # first explicit numeric value; do not infer a charge from token counts or
    # money amounts.
    credit_keys = (
        "credits_float",
        "creditsFloat",
        "credits_consumed",
        "consumed_credits",
        "credit_cost",
        "credits_cost",
        "amount_float",
        "amountFloat",
    )

    def find_credit(value: Any, depth: int = 0) -> Decimal | None:
        if depth > 6:
            return None
        if isinstance(value, Mapping):
            for key in credit_keys:
                if key in value:
                    parsed = _credit_decimal(value.get(key))
                    if parsed is not None:
                        return parsed
            # Prefer the known billing containers before a generic recursive
            # walk so unrelated metadata cannot mask the session charge.
            preferred = (
                "user_usage_group_by_session",
                "userUsageGroupBySession",
                "usage",
                "billing",
                "cost",
                "result",
                "data",
            )
            for key in preferred:
                if key in value:
                    parsed = find_credit(value.get(key), depth + 1)
                    if parsed is not None:
                        return parsed
            for child in value.values():
                parsed = find_credit(child, depth + 1)
                if parsed is not None:
                    return parsed
        elif isinstance(value, list):
            for child in value:
                parsed = find_credit(child, depth + 1)
                if parsed is not None:
                    return parsed
        return None

    # Return only billing-safe scalar fields.  Do not persist or log
    # ``extra_info`` (which contains token details) or any upstream envelope.
    credits = find_credit(data)
    if credits is None or credits < 0:
        return {}
    return {
        "credits_consumed": _json_credit_number(credits),
        "credits_source": "session_usage",
    }

def _provider_specific(options: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    if isinstance(options, Mapping):
        for key in ("provider_specific", "providerSpecificData"):
            if key not in options:
                continue
            value = options.get(key)
            if isinstance(value, Mapping):
                return dict(value)
    return auth.get_psd()


def build_web_headers(
    token: str, options: Optional[Mapping[str, Any]] = None
) -> dict[str, str]:
    """构造参考 OmniRoute 网页版 remote 会话的请求头。"""
    psd = _provider_specific(options)
    return {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
        "X-Trae-Client-Type": "web",
        "X-Preferenced-Language": psd.get("appLanguage")
        or os.environ.get("TRAE_WEB_LANGUAGE")
        or "zh-CN",
        "x-user-region": psd.get("userRegion")
        or os.environ.get("TRAE_WEB_USER_REGION")
        or "CN",
        "Origin": web_origin(),
        "Referer": web_origin() + "/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
        ),
    }


def web_origin() -> str:
    """Return the browser Origin/Referer matching the current web upstream."""
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).lower()
    if "core-normal.trae.cn" in base or "trae-api-cn" in base or "mchost.guru" in base:
        return "https://solo.trae.cn"
    return "https://solo.trae.ai"


def _web_common_params(psd: dict, mode: str, session_id: str = "") -> str:
    params = {
        "language": "zh-cn",
        "app_language": psd.get("appLanguage") or "zh-CN",
        "quality": "stable",
        "app_version": psd.get("appVersion") or "1.0.0.1229",
        "web_id": psd.get("webId") or "",
        "user_identity": psd.get("userIdentity") or "Free",
        "is_freshman": "0",
        "biz_user_id": psd.get("bizUserId") or "",
        "user_unique_id": psd.get("userUniqueId") or "",
        "scope": psd.get("scope") or "marscode-cn",
        "tenant": psd.get("tenant") or "marscode",
        "region": psd.get("region") or "cn",
        "aiRegion": psd.get("aiRegion") or psd.get("region") or "cn",
        "is_privacy_mode": 0,
        "privacy_mode": "off",
        "solo_chat_mode": mode,
    }
    if session_id:
        params["biz_session_id"] = session_id
    return json.dumps(params, ensure_ascii=False)


def flatten_query(messages: list[dict]) -> str:
    """把 OpenAI 消息扁平化为 web remote 端点的 query JSON 字符串。"""
    parts: list[str] = []
    for m in sanitize_assistant_history_messages(messages):
        content_value = m.get("content", "")
        if content_value in (None, "", []) and not m.get("tool_calls"):
            for key in ("parts", "text", "prompt", "message", "input"):
                candidate = m.get(key)
                if candidate not in (None, "", [], {}):
                    content_value = candidate
                    break
        content = _content_to_text(content_value)
        renderer_results = raw_client._renderer_tool_result_blocks(content_value)
        role = m.get("role", "user")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            assistant_parts = [content] if content else []
            tool_history = raw_client._serialize_tool_calls(m.get("tool_calls"))
            if not tool_history:
                # Renderer histories carry calls as tool_use content blocks.
                tool_history = raw_client._serialize_tool_calls(
                    raw_client._renderer_tool_use_blocks(content_value)
                )
            if tool_history:
                assistant_parts.append(tool_history)
            if assistant_parts:
                parts.append("[Assistant]\n" + "\n\n".join(assistant_parts))
        elif renderer_results:
            result_context = raw_client._renderer_tool_result_context(
                content_value,
                raw_client._tool_call_name_map(messages),
            )
            parts.append(f"[Client Tool Result]\n{result_context}")
        elif role == "tool":
            tool_id = m.get("tool_call_id") or m.get("toolCallId") or "unknown"
            tool_name = m.get("name") or "tool"
            status = "failed" if tool_result_is_failed(m) else "succeeded"
            parts.append(
                f"[Client Tool Result: {tool_id} {tool_name} status={status}]\n{content}"
            )
        else:
            parts.append(content)
    text = "\n\n".join(parts)
    return json.dumps([{"type": "text", "data": {"content": text}}], ensure_ascii=False)


def build_web_content(messages: list[dict]) -> list[dict]:
    """Build a structured content array from OpenAI messages for the web remote endpoint.

    Tool calls and results are formatted as readable text blocks so the upstream
    agent understands the full conversation context including previous tool usage.
    """
    items: list[dict] = []
    tool_names = raw_client._tool_call_name_map(messages)
    for m in sanitize_assistant_history_messages(messages):
        role = m.get("role", "user")
        content = m.get("content", "")
        if content in (None, "", []) and not m.get("tool_calls"):
            for key in ("parts", "text", "prompt", "message", "input"):
                candidate = m.get(key)
                if candidate not in (None, "", [], {}):
                    content = candidate
                    break
        tool_calls = m.get("tool_calls")
        renderer_results = raw_client._renderer_tool_result_blocks(content)
        if role == "assistant" and not tool_calls:
            # Renderer histories carry calls as tool_use content blocks.
            renderer_calls = raw_client._renderer_tool_use_blocks(content)
            if renderer_calls:
                tool_calls = [
                    {
                        "id": call.get("id"),
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(
                                call.get("input") or {},
                                ensure_ascii=False,
                                separators=(",", ":"),
                                default=str,
                            ),
                        },
                    }
                    for call in renderer_calls
                ]

        if role == "system":
            items.append({"type": "text", "data": {"content": f"[System]\n{_content_to_text(content)}"}})
        elif role == "assistant" and tool_calls:
            text = _content_to_text(content)
            if text:
                items.append({"type": "text", "data": {"content": text}})
            for tc in tool_calls:
                func = tc.get("function", {})
                tc_id = tc.get("id") or tc.get("tool_call_id") or "unknown"
                tc_text = (
                    f"\n[Client Tool Call: {tc_id} {func.get('name', 'unknown')}]"
                    f"\nArguments: {func.get('arguments', '{}')}"
                )
                items.append({"type": "text", "data": {"content": tc_text}})
        elif renderer_results:
            items.append(
                {
                    "type": "text",
                    "data": {
                        "content": (
                            "\n[Client Tool Result]\n"
                            + raw_client._renderer_tool_result_context(content, tool_names)
                        )
                    },
                }
            )
        elif role == "tool":
            tc_id = m.get("tool_call_id") or m.get("toolCallId") or ""
            tool_name = m.get("name") or "tool"
            status = "failed" if tool_result_is_failed(m) else "succeeded"
            items.append(
                {
                    "type": "text",
                    "data": {
                        "content": (
                            f"\n[Client Tool Result: {tc_id} {tool_name} status={status}]\n"
                            f"{_content_to_text(content)}"
                        )
                    },
                }
            )
        else:
            items.append({"type": "text", "data": {"content": _content_to_text(content)}})
    return items


def _messages_with_client_runtime(
    messages: list[dict],
    options: Optional[dict] = None,
    *,
    native_tools: bool = False,
) -> list[dict]:
    options = options or {}
    if (
        not _tool_protocol_requested(options, messages)
        and "client_context" not in options
        and "clientContext" not in options
    ):
        # No tool runtime to describe, but the response-style constraint still
        # applies: the remote agent otherwise returns its full reasoning trace
        # as the visible answer.
        style = raw_client.response_style_instruction()
        if not style:
            return messages
        return [{"role": "system", "content": style}, *messages]
    tool_catalog = (
        options.get("tools")
        if "tools" in options
        else options.get("_inherited_tools")
    )
    tool_choice = options.get("tool_choice")
    context_options = options
    if native_tools:
        normalized_choice = _ide_native_tool_choice(tool_choice)
        if normalized_choice in {"none", "auto", "required"}:
            tool_choice = normalized_choice
        if normalized_choice == "none":
            tool_catalog = []
            context_options = {**options, "tools": []}
    prompt = raw_client.build_runtime_system_prompt(
        tool_catalog,
        raw_client.build_client_context(context_options),
        tool_choice,
        options.get("parallel_tool_calls"),
        native_tools=native_tools,
    )
    return [{"role": "system", "content": prompt}, *messages]


def _resolve_mode(model: str) -> tuple[str, str, str]:
    m = (model or "").strip().lower()
    if m in ("work", "auto-work", "solo-work"):
        return ("work", "auto", "")
    auto = not m or m == "auto"
    if auto:
        return ("code", "auto", "")
    return ("code", "manual", convert_model_name(model))


def _web_account_key() -> str:
    uid = auth.get_user_id()
    return uid or (auth.get_token()[:16] or "default")


def _web_slot(account_id: str = "") -> asyncio.Semaphore:
    key = account_id or _web_account_key()
    sem = _WEB_SLOTS.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_WEB_PARALLEL_LIMIT)
        _WEB_SLOTS[key] = sem
    return sem


def web_slot_available(account_id: str = "") -> bool:
    return not _web_slot(account_id).locked()


async def acquire_web_slot(account_id: str = "", timeout: float = 60.0) -> None:
    sem = _web_slot(account_id)
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError("Trae web session slot busy: parallel limit reached")


def release_web_slot(account_id: str = "") -> None:
    _web_slot(account_id).release()


def register_web_lease(
    account_id: str,
    session_id: str,
    message_id: str,
    client: httpx.AsyncClient,
    token: str = "",
    provider_specific: Optional[Mapping[str, Any]] = None,
) -> None:
    _WEB_LEASES[session_id] = {
        "account_id": account_id,
        "session_id": session_id,
        "message_id": message_id,
        "client": client,
        "token": token,
        "provider_specific": dict(provider_specific or {}),
        "last_activity": time.monotonic(),
    }


def unregister_web_lease(session_id: str) -> bool:
    return _WEB_LEASES.pop(session_id, None) is not None


def touch_web_lease(session_id: str) -> None:
    lease = _WEB_LEASES.get(session_id)
    if lease:
        lease["last_activity"] = time.monotonic()


def idle_web_leases() -> list[dict]:
    now = time.monotonic()
    return [lease for lease in _WEB_LEASES.values() if now - lease["last_activity"] >= _WEB_IDLE_TIMEOUT]



async def stop_web_session(
    client: httpx.AsyncClient,
    session_id: str,
    message_id: str,
    options: Optional[dict] = None,
) -> None:
    """Actively interrupt an upstream web session so it stops occupying a running slot."""
    if not session_id or not message_id:
        return
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    options = options or {}
    token = str(options.get("_auth_token") or auth.get_token() or "")
    if not token:
        return
    try:
        resp = await client.post(
            f"{base}/chat_sessions/{session_id}/stop",
            headers=build_web_headers(token, options),
            json={"chat_session_id": session_id, "user_message_id": message_id},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("trae-client: stop web session %s returned %s", session_id, resp.status_code)
    except Exception as e:
        logger.warning("trae-client: stop web session %s failed: %s", session_id, e)


async def reap_idle_web_sessions() -> int:
    async def cleanup_lease(lease: dict) -> None:
        client = lease.get("client")
        try:
            if client is not None:
                try:
                    await stop_web_session(
                        client,
                        lease["session_id"],
                        lease.get("message_id") or "",
                        options={
                            "_auth_token": lease.get("token") or "",
                            "provider_specific": dict(
                                lease.get("provider_specific") or {}
                            ),
                        },
                    )
                except Exception:
                    pass
        finally:
            try:
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
            finally:
                release_web_slot(lease["account_id"])

    count = 0
    for lease in idle_web_leases():
        session_id = lease["session_id"]
        account_id = lease["account_id"]
        logger.warning("trae-client: reaping idle web session %s (account %s)", session_id, account_id)
        if unregister_web_lease(session_id):
            # Lease ownership has moved to the reaper. The response's close
            # path will now skip it, so cancellation must drain this cleanup.
            cleanup_task = asyncio.create_task(cleanup_lease(lease))
            cancellation = None
            with anyio.CancelScope(shield=True):
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError as exc:
                        if cleanup_task.cancelled():
                            raise
                        cancellation = cancellation or exc
                    except Exception:
                        break
                try:
                    cleanup_task.result()
                except BaseException:
                    if cancellation is None:
                        raise
                    logger.warning(
                        "trae-client: idle session cleanup failed while reaper cancelled",
                        exc_info=True,
                    )
            if cancellation is not None:
                raise cancellation
            await anyio.lowlevel.checkpoint_if_cancelled()
            count += 1
    return count



def _build_web_model_config(raw: dict) -> dict:
    """Keep the model object the web frontend would send as custom_model."""
    cfg = dict(raw)
    features = cfg.get("features")
    if isinstance(features, str):
        try:
            cfg["features"] = json.loads(features)
        except Exception:
            cfg["features"] = {}
    name = cfg.get("name") or ""
    cfg["config_name"] = cfg.get("config_name") or name
    # The CN model-list response uses ``name`` as the config identifier.  The
    # remote executor also accepts/records ``model_name``; keeping both makes
    # the selected model explicit instead of allowing a default fallback.
    cfg["model_name"] = cfg.get("model_name") or name
    cfg["config_source"] = cfg.get("config_source") or 1
    cfg["provider"] = cfg.get("provider") or ""
    cfg["multimodal"] = bool(cfg.get("multimodal"))
    cfg["ak"] = cfg.get("ak") or ""
    cfg["sk"] = cfg.get("sk") or ""
    cfg["base_url"] = cfg.get("base_url") or ""
    cfg["auth_type"] = cfg.get("auth_type") or 0
    cfg["use_remote_service"] = not bool(cfg.get("client_connect"))
    return cfg


async def _fetch_web_model_configs(
    token_override: str = "",
    provider_specific: Optional[Mapping[str, Any]] = None,
    agent_type: str = "",
) -> dict[str, dict]:
    """Fetch the same model list used by the Trae web client."""
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    token = token_override or auth.get_token()
    if not token:
        return {}
    url = f"{base}/models?functions=solo_agent_remote%2Csolo_work_remote%2Csolo_design_remote&show_custom_model=true"
    try:
        header_options = (
            {"provider_specific": dict(provider_specific)}
            if provider_specific is not None
            else None
        )
        async with httpx.AsyncClient(
            headers=build_web_headers(token, header_options), timeout=30
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning("trae-client: web model list returned %s", resp.status_code)
                return {}
            data = resp.json()
    except Exception as e:
        logger.warning("trae-client: web model list failed: %s", e)
        return {}
    out: dict[str, dict] = {}

    def _ingest_models(group: Mapping[str, Any]) -> None:
        if not isinstance(group, Mapping):
            return
        for raw in group.get("models", []):
            if not isinstance(raw, Mapping):
                continue
            name = (raw.get("name") or "").strip()
            if name and name not in out:
                out[name] = _build_web_model_config(raw)

    def _group_function(group: Mapping[str, Any]) -> str:
        if not isinstance(group, Mapping):
            return ""
        value = group.get("function") or group.get("agent_type") or ""
        return str(value).strip().lower()

    groups = data.get("data", {}).get("list", [])
    requested = str(agent_type or "").strip().lower()
    # Keep each agent tier separate.  Agent carries the 1M profile while Work
    # remains the 200K fallback; selecting by tier must not reuse the Agent
    # config from the account cache.
    preferred = []
    if requested:
        preferred = [g for g in groups if _group_function(g) == requested]
    if not preferred and requested:
        # A requested tier may be represented by its lite sibling in older
        # model lists, but must never silently fall back from Work to Agent.
        sibling = (
            requested[:-7] + "_lite"
            if requested.endswith("_remote")
            else requested[:-5] + "_remote"
            if requested.endswith("_lite")
            else ""
        )
        if sibling:
            preferred = [g for g in groups if _group_function(g) == sibling]
    if not preferred and requested:
        return out
    if not preferred:
        agent_remote = [g for g in groups if _group_function(g) == "solo_agent_remote"]
        agent_lite = [g for g in groups if _group_function(g) == "solo_agent_lite"]
        preferred = agent_remote + agent_lite
    for group in preferred:
        _ingest_models(group)
    # For the default Agent lookup, preserve the historical fallback to Work;
    # an explicit Work lookup must stay in the Work tier.
    if not requested:
        for group in groups:
            _ingest_models(group)
    return out


async def _fetch_solo_model_configs(
    token_override: str = "",
    provider_specific: Optional[Mapping[str, Any]] = None,
) -> dict[str, dict]:
    """Fetch the native Solo catalog from ``get_detail_param``.

    The web/Remote catalog and the native Solo catalog are not identical:
    Solo can expose newly enabled configs before they appear in the Remote
    groups. Keep this lookup independent from the Remote parser and merge it
    at public model-list time.
    """

    token = token_override or auth.get_token()
    if not token:
        return {}
    base = _model_gateway_base()
    url = f"{base}/api/ide/v1/get_detail_param"
    body = {
        "function": "solo_work_lite",
        "config_names": None,
        "need_prompt": False,
        "current_config_info": None,
        "poly_prompt": True,
        "mode_type": None,
        "agent_type": None,
    }
    headers = build_headers(token_override=token)
    headers.update(
        {
            "Accept": "application/json",
            "Origin": web_origin(),
            "Referer": web_origin() + "/",
            "x-app-version": "default",
            "x-app-version-code": SOLO_GATEWAY_VERSION_CODE,
            "x-ide-version": SOLO_GATEWAY_VERSION,
            "x-ide-version-code": SOLO_GATEWAY_VERSION_CODE,
            "x-ide-version-type": "stable",
            "x-device-brand": SOLO_DEVICE_BRAND,
            "x-device-type": SOLO_DEVICE_TYPE,
            "x-device-platform": SOLO_PLATFORM,
            "x-version-code": SOLO_GATEWAY_VERSION_CODE,
            "x-os-version": SOLO_OS_VERSION,
            "package-type": SOLO_PACKAGE_TYPE,
            "x-lscbd-aid": SOLO_LSCBD_AID,
            "x-lscbd-platform": SOLO_PLATFORM,
            "x-ss-dp": SOLO_LSCBD_AID,
            "x-plugin-channel": "icube-ai",
            "app-version": SOLO_GATEWAY_VERSION,
            "request-traffic-type": "prod",
            "User-Agent": f"Trae/{SOLO_GATEWAY_VERSION}",
        }
    )
    try:
        timeout = float(
            os.environ.get("TRAE_SOLO_MODEL_LIST_TIMEOUT_SECONDS", "8") or "8"
        )
        timeout = max(1.0, min(timeout, 20.0))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                logger.warning(
                    "trae-client: solo model list returned %s", resp.status_code
                )
                return {}
            data = resp.json()
    except Exception as exc:
        logger.warning("trae-client: solo model list failed: %s", exc)
        return {}

    if not isinstance(data, Mapping):
        return {}
    raw_list = data.get("config_info_list")
    if not isinstance(raw_list, list):
        nested = data.get("data")
        raw_list = (
            nested.get("config_info_list")
            if isinstance(nested, Mapping)
            else []
        )
    if not isinstance(raw_list, list):
        return {}

    out: dict[str, dict] = {}
    for raw in raw_list:
        if not isinstance(raw, Mapping):
            continue
        # Profiles such as custom_model_1M_text are routing configurations,
        # not selectable chat models.
        if str(raw.get("usage") or "").strip().lower() not in ("", "chat_completion"):
            continue
        if raw.get("custom_models"):
            continue
        config_name = str(raw.get("config_name") or raw.get("name") or "").strip()
        if not config_name:
            continue
        display_config = raw.get("display_config")
        display_config = display_config if isinstance(display_config, Mapping) else {}
        normalized = dict(raw)
        normalized["name"] = config_name
        normalized["config_name"] = config_name
        normalized["model_name"] = normalized.get("model_name") or config_name
        normalized["display_name"] = (
            normalized.get("display_name")
            or display_config.get("display_name")
            or config_name
        )
        normalized["max_mode"] = bool(
            normalized.get("max_mode") or display_config.get("max_mode")
        )
        out[config_name] = _build_web_model_config(normalized)
    return out


async def _get_web_custom_model(
    model_name: str,
    *,
    token_override: str = "",
    user_id_override: str = "",
    provider_specific: Optional[Mapping[str, Any]] = None,
    agent_type: str = "",
) -> Optional[dict]:
    """Return custom_model for a manual web model selection.

    Lookup priority:
      1. exact config name (e.g. DeepSeek-V4-Flash-Official)
      2. case-insensitive config/display name (e.g. "DeepSeek-V4-Flash 正式版")
      3. legacy synthetic fallbacks for custom-openai models
    """
    if not model_name:
        return None
    token = token_override or auth.get_token()
    account_key = _web_model_cache_key(token, user_id_override) + ":" + str(agent_type or "default")
    now = time.time()
    cached = _WEB_MODEL_CACHE.get(account_key)
    if not cached or now - cached[0] > _WEB_MODEL_CACHE_TTL:
        fetch_kwargs: dict[str, Any] = {}
        if provider_specific is not None:
            fetch_kwargs["provider_specific"] = provider_specific
        if agent_type:
            fetch_kwargs["agent_type"] = agent_type
        configs = await _fetch_web_model_configs(token, **fetch_kwargs)
        # Only pay the extra native-Solo lookup when the requested config is
        # absent from Remote. This keeps normal model resolution on the fast
        # path while allowing Solo-only models to be used through Remote.
        lowered_model = str(model_name).strip().lower()
        found = any(
            str(name).strip().lower() == lowered_model
            or str(
                (candidate or {}).get("display_name")
                or (candidate or {}).get("display_model_name")
                or ""
            ).strip().lower()
            == lowered_model
            for name, candidate in configs.items()
        )
        if not found and not agent_type:
            solo_configs = await _fetch_solo_model_configs(token, **fetch_kwargs)
            if isinstance(solo_configs, Mapping):
                for name, config in solo_configs.items():
                    configs.setdefault(name, config)
        cached = (now, configs)
        _WEB_MODEL_CACHE[account_key] = cached

    configs = cached[1]
    cfg = configs.get(model_name)
    if cfg is not None:
        return cfg

    lowered = model_name.lower()
    for name, candidate in configs.items():
        if name.lower() == lowered:
            return candidate
    for name, candidate in configs.items():
        display = (candidate.get("display_name") or candidate.get("display_model_name") or "").strip()
        if display and (display.lower() == lowered or display == model_name):
            return candidate
    for name, candidate in configs.items():
        display = (candidate.get("display_name") or "").strip()
        if display and lowered in (display.lower(), display.replace(" ", "-").lower()):
            return candidate

    if lowered in ("mimo-v2.5", "mimo-v2.5-pro", "minimax-m25", "qwen36-35b"):
        return {
            "name": model_name,
            "model_name": model_name,
            "config_name": model_name,
            "display_model_name": model_name,
            "display_name": model_name,
            "config_source": 3,
            "provider": "custom_openai_compatible",
            "multimodal": lowered.startswith("mimo"),
            "is_preset": False,
            "use_remote_service": True,
            "ak": "", "sk": "", "base_url": "", "region": None, "auth_type": 0,
            "features": {},
        }
    return None


async def resolve_model_config(
    model_name: str,
    *,
    token_override: str = "",
    user_id_override: str = "",
    provider_specific: Optional[Mapping[str, Any]] = None,
    agent_type: str = "",
) -> Optional[dict]:
    """Resolve a public/display model label to Trae's exact config object."""

    kwargs: dict[str, Any] = {
        "token_override": token_override,
        "user_id_override": user_id_override,
    }
    if provider_specific is not None:
        kwargs["provider_specific"] = provider_specific
    if agent_type:
        kwargs["agent_type"] = agent_type
    return await _get_web_custom_model(model_name, **kwargs)


async def create_web_session(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    options: Optional[dict] = None,
) -> tuple[str, str]:
    """POST /chat_sessions（OmniRoute 网页版方案）。"""
    if _tool_protocol_requested(options, messages):
        raise RuntimeError(
            "Trae web remote cannot safely proxy caller-owned tool policy"
        )
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    options = options or {}
    token = str(options.get("_auth_token") or auth.get_token() or "")
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")

    mode, strategy, model_name = _resolve_mode(model)
    requested_agent_type = str(
        options.get("_remote_agent_type")
        or options.get("remote_agent_type")
        or ""
    ).strip().lower()
    agent_type = (
        "solo_work_remote"
        if requested_agent_type in {"solo_work_remote", "solo_work_lite", "work"}
        or mode == "work"
        else "solo_agent_remote"
    )
    prepared_messages = _messages_with_client_runtime(messages, options)
    psd = _provider_specific(options)
    initial_message = {
        "chat_session_id": "",
        "content": build_web_content(prepared_messages),
        "query": flatten_query(prepared_messages),
        "model_name": model_name,
        "agent_type": agent_type,
        "model_selection_strategy": strategy,
        "common_params": _web_common_params(psd, mode),
    }
    if strategy != "auto":
        bound_user_id = str(
            options.get("_auth_user_id")
            or options.get("_billing_id")
            or options.get("_account_id")
            or ""
        ).strip()
        lookup_kwargs: dict[str, Any] = {
            "token_override": token,
            "user_id_override": bound_user_id,
            "provider_specific": psd,
        }
        if agent_type != "solo_agent_remote" or requested_agent_type:
            lookup_kwargs["agent_type"] = agent_type
        custom_model = await _get_web_custom_model(model_name, **lookup_kwargs)
        if custom_model:
            initial_message["custom_model"] = custom_model
    body = {
        "mode": mode,
        "environment_id": "default",
        "initial_message": initial_message,
        "env": "remote",
        "auto_create_project": False,
        "origin": "web",
    }
    resp = await client.post(
        f"{base}/chat_sessions",
        headers=build_web_headers(token, options),
        json=body,
        timeout=60,
    )
    text = resp.text
    if resp.status_code != 200:
        raise RuntimeError(f"Trae web create_session [{resp.status_code}]: {text[:500]}")
    data = resp.json()
    if data.get("code") not in (0, None) and not data.get("data"):
        raise RuntimeError(f"Trae web create_session: {data}")
    payload = data.get("data") or data
    session_id = payload.get("chat_session_id") or ""
    message_id = payload.get("message_id") or ""
    if not session_id or not message_id:
        raise RuntimeError(f"Trae web create_session missing ids: {data}")
    return session_id, message_id


async def stream_web_events(
    client: httpx.AsyncClient,
    session_id: str,
    message_id: str,
    options: Optional[dict] = None,
) -> AsyncIterator[tuple[str, dict]]:
    """GET /chat_sessions/{id}/events?reply_to_message_id=...（SSE）。"""
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    url = f"{base}/chat_sessions/{session_id}/events?reply_to_message_id={message_id}"
    options = options or {}
    token = str(options.get("_auth_token") or auth.get_token() or "")
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))
    async with client.stream(
        "GET", url, headers=build_web_headers(token, options), timeout=timeout
    ) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            raise RuntimeError(f"Trae web events stream [{resp.status_code}]: {body[:500]}")
        buffer = ""
        event = None
        async for raw in resp.aiter_raw():
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    payload = line[5:].strip()
                    try:
                        data = json.loads(payload)
                    except Exception:
                        data = {"_raw": payload}
                    touch_web_lease(session_id)
                    yield (event, data)
                    event = None
                elif line == "":
                    event = None


def _ide_native_content(content: Any) -> Any:
    """Keep SOLO content blocks intact, normalizing only renderer text."""

    if content is None:
        return None
    if not isinstance(content, list):
        if isinstance(content, Mapping):
            return [copy.deepcopy(dict(content))]
        return [{"type": "text", "text": str(content)}]
    blocks: list[Any] = []
    for block in content:
        if isinstance(block, str):
            blocks.append({"type": "text", "text": block})
        elif isinstance(block, Mapping):
            converted = copy.deepcopy(dict(block))
            if str(block.get("type") or "").lower() in {"text", "input_text"}:
                converted["type"] = "text"
                text = block.get("text")
                if not isinstance(text, str):
                    text = block.get("content")
                converted["text"] = (
                    text if isinstance(text, str) else renderer_block_text(block)
                )
                converted.pop("value", None)
            blocks.append(converted)
        else:
            blocks.append(copy.deepcopy(block))
    return blocks


def _ide_native_tool_calls(message: Mapping[str, Any]) -> list[dict]:
    """Map OpenAI and renderer assistant calls to SOLO's function_call."""

    calls = message.get("tool_calls")
    calls = list(calls) if isinstance(calls, list) else []
    content = message.get("content")
    if isinstance(content, list):
        calls.extend(
            block
            for block in content
            if isinstance(block, Mapping)
            and str(block.get("type") or "").lower()
            in {"tool_use", "tool_call", "function_call"}
        )
    prepared = []
    seen_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or call.get("function_call")
        function = function if isinstance(function, Mapping) else call
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        call_id = str(
            call.get("id") or call.get("tool_call_id") or call.get("toolCallId") or ""
        )
        if call_id and call_id in seen_ids:
            continue
        if call_id:
            seen_ids.add(call_id)
        arguments = function.get("arguments")
        if arguments is None:
            arguments = function.get("input", function.get("parameters", {}))
        prepared.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return raw_client._native_assistant_tool_calls(prepared)


def _convert_ide_native_messages(
    messages: list[dict], options: Optional[dict] = None
) -> list[dict]:
    """Preserve caller-owned tool correlation on llm_utils_chat only."""

    result: list[dict] = []
    prepared = sanitize_assistant_history_messages(
        _messages_with_client_runtime(messages, options, native_tools=True)
    )
    tool_names = raw_client._tool_call_name_map(messages)
    for message in prepared:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user").strip().lower()
        if role == "developer":
            role = "system"
        elif role not in {"system", "user", "assistant", "tool", "function"}:
            role = "user"
        item: dict[str, Any] = {"role": role}
        content = message.get("content")
        if role == "assistant":
            calls = _ide_native_tool_calls(message)
            if calls:
                item["tool_calls"] = calls
            legacy_call = message.get("function_call")
            if isinstance(legacy_call, Mapping):
                converted_call = raw_client._native_function_call(legacy_call)
                if converted_call:
                    item["function_call"] = converted_call
            if isinstance(content, list):
                content = [
                    block
                    for block in content
                    if not (
                        isinstance(block, Mapping)
                        and str(block.get("type") or "").lower()
                        in {"tool_use", "tool_call", "function_call"}
                    )
                ]
        renderer_results = raw_client._renderer_tool_result_blocks(content)
        if role in {"tool", "function"}:
            for key in ("tool_call_id", "name"):
                if message.get(key) is not None:
                    item[key] = str(message[key])
            if not renderer_results and tool_result_is_failed(message):
                content = "Client tool result status=failed:\n" + _content_to_text(content)

        if renderer_results:
            # Split renderer user turns at result blocks so a following user
            # instruction does not hide or precede the native tool response.
            pending: list[Any] = []
            for block in content:
                block_results = raw_client._renderer_tool_result_blocks([block])
                if not block_results:
                    pending.append(block)
                    continue
                if pending:
                    result.append({**item, "content": _ide_native_content(pending)})
                    pending = []
                tool_result = block_results[0]
                text = tool_result["content"]
                if tool_result["failed"]:
                    text = "Client tool result status=failed:\n" + text
                native_result = {
                    "role": "tool",
                    "tool_call_id": tool_result["id"],
                    "content": _ide_native_content(text),
                }
                name = tool_result["name"] or tool_names.get(tool_result["id"])
                if name:
                    native_result["name"] = name
                result.append(native_result)
            if pending:
                result.append({**item, "content": _ide_native_content(pending)})
            continue
        if "content" in message:
            item["content"] = _ide_native_content(content)
        result.append(item)
    return result


def _ide_native_tool_choice(choice: Any) -> Optional[str]:
    if isinstance(choice, str):
        choice = choice.strip()
        return choice.lower() if choice.lower() in {"none", "auto", "required"} else choice
    if not isinstance(choice, Mapping):
        return None
    choice_type = str(choice.get("type") or "").strip().lower()
    if choice_type in {"none", "auto", "required"}:
        return choice_type
    name = raw_client._named_tool_choice(choice)
    return name or ("auto" if choice_type == "function" else None)


def build_llm_chat_body(
    messages: list[dict],
    model: str,
    stream: bool,
    max_tokens: Optional[int] = None,
    options: Optional[dict] = None,
) -> dict:
    """构造 Trae SOLO ``llm_utils_chat`` 的原生请求体.

    The upstream endpoint is not an OpenAI-compatible ``inline_chat`` route.
    TraeWork rewrites the request to ``solo_work_lite`` and pins the selected
    model through ``config_name``.  It also keeps the transport streaming;
    non-stream callers aggregate the SSE response in the relay.
    """
    options = options or {}
    converted = _convert_ide_native_messages(messages, options)
    session_id = _requested_session_id(options) or str(uuid.uuid4())
    config_name = convert_model_name(model) or model or "glm-5.2"
    body = {
        "messages": converted,
        "config_name": config_name,
        "model": config_name,
        "function": "solo_work_lite",
        "stream": True,
        "request_id": session_id,
        "session_id": session_id,
    }
    max_tokens = clamp_max_completion_tokens(max_tokens, model)
    if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool) and max_tokens > 0:
        body["max_tokens"] = int(max_tokens)
    tools = options["tools"] if "tools" in options else options.get("_inherited_tools")
    choice = _ide_native_tool_choice(options.get("tool_choice"))
    if isinstance(tools, list) and tools and str(choice or "").lower() != "none":
        native_tools = raw_client._native_tool_definitions(tools)
        if native_tools:
            body["tools"] = native_tools
            if choice is not None:
                body["tool_choice"] = choice
            if options.get("parallel_tool_calls") is not None:
                body["parallel_tool_calls"] = bool(options["parallel_tool_calls"])
    return body


def build_work_agent_body(
    messages: list[dict],
    model: str,
    stream: bool,
    options: Optional[dict] = None,
) -> dict:
    """Build the legacy Work ``create_agent_task`` envelope.

    Work does not use the plaintext ``llm_utils_chat`` body.  The public
    client still owns tools, so the schemas are represented both in the
    runtime system message (for compatibility with older Work builds) and in
    the native ``tools`` field when supplied.
    """

    options = options or {}
    converted = convert_openai_messages(messages, options)
    session_id = _requested_session_id(options) or str(uuid.uuid4())
    conversation_id = str(
        options.get("conversation_id")
        or options.get("conversationId")
        or session_id
    )
    prompt = ""
    for item in reversed(converted):
        if str(item.get("role") or "") == "user":
            prompt = _content_to_text(item.get("content"))
            if prompt:
                break
    config_name = convert_model_name(model) or model or "glm-5.2"
    raw_model = config_name if config_name.endswith("__dev") else config_name + "__dev"
    query = json.dumps(
        [{"type": "text", "data": {"content": prompt}}],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    user_messages = [
        {
            "role": str(item.get("role") or "user"),
            "content": _content_to_text(item.get("content")),
        }
        for item in converted
        if _content_to_text(item.get("content"))
    ]
    body: dict[str, Any] = {
        "conversation_id": conversation_id,
        "session_id": session_id,
        "user_id": str(options.get("_auth_user_id") or options.get("_account_id") or auth.get_user_id() or ""),
        "device_id": str(options.get("device_id") or ""),
        "agent_type": "solo_work_lite",
        "model_name": raw_model,
        "config_name": config_name.removesuffix("__dev"),
        "ide_version": WORK_AGENT_VERSION,
        "version_code": int(WORK_AGENT_VERSION_CODE or "20260901"),
        "mode_type": 1,
        "plugin_channel": "stable",
        "history_id_list": [],
        "user_input": {
            "id": str(uuid.uuid4()),
            "query": query,
            "messages": user_messages or [{"role": "user", "content": prompt}],
        },
    }
    tools = options.get("tools") or options.get("_inherited_tools")
    if isinstance(tools, list) and tools:
        body["tools"] = raw_client._native_tool_definitions(tools)
        if options.get("tool_choice") is not None:
            choice = options["tool_choice"]
            if isinstance(choice, Mapping):
                function = choice.get("function")
                selected = function if isinstance(function, Mapping) else choice
                choice = str((selected.get("name") if isinstance(selected, Mapping) else "") or "auto")
            if choice != "none":
                body["tool_choice"] = choice
        if options.get("parallel_tool_calls") is not None:
            body["parallel_tool_calls"] = options["parallel_tool_calls"]
    if not stream:
        body["stream"] = False
    return body


async def build_trae_ide_request(
    messages: list[dict],
    model: str,
    max_tokens: Optional[int] = None,
    options: Optional[dict] = None,
) -> tuple[str, dict]:
    """构造与 trae2api 一致的 /api/ide/v1/chat 请求体。"""
    converted = convert_openai_messages(messages, options)
    session_id = _requested_session_id(options) or generate_session_id_from_messages(
        messages
    )
    trae_model = convert_model_name(model)

    messages_len = len(converted)
    if messages_len == 0:
        raise ValueError("messages cannot be empty")
    last_content = _content_to_text(converted[-1].get("content", ""))

    client_context = raw_client.build_client_context(options)
    terminal_context = client_context.get("terminal_context")
    if not isinstance(terminal_context, list):
        terminal_context = []
    context_resolvers = [
        {"resolver_id": "project-labels", "variables": '{"labels":"- go\n- go.mod"}'},
        {
            "resolver_id": "terminal_context",
            "variables": json.dumps(
                {"terminal_context": terminal_context}, ensure_ascii=False
            ),
        },
    ]

    variables = {
        "language": "",
        "locale": "zh-cn",
        "input": last_content,
        "version_code": IDE_VERSION_CODE_NUM,
        "is_inline_chat": False,
        "is_command": False,
        "raw_input": last_content,
        "problem": "",
        "current_filename": "",
        "is_select_code_before_chat": False,
        "last_select_time": int(time.time() * 1000),
        "last_turn_session": "",
        "hash_workspace": False,
        "hash_file": 0,
        "hash_code": 0,
        "use_filepath": True,
        "current_time": _rfc3339_zh_time(),
        "badge_clickable": True,
        "workspace_path": str(client_context.get("workspace_path") or generate_random_workspace_path()),
        "brand": "Trae",
        "system_type": str(client_context.get("system_type") or "Windows"),
    }

    chat_history: list[dict] = []
    last_llm_response_info = None
    for msg in converted[:-1]:
        item = {
            "role": msg.get("role", "user"),
            "session_id": session_id,
            "locale": "zh-cn" if msg.get("role") == "assistant" else "",
            "content": _content_to_text(msg.get("content", "")),
            "status": "success",
        }
        chat_history.append(item)
        if msg.get("role") == "assistant":
            last_llm_response_info = {
                "turn": len(chat_history) - 1,
                "is_error": False,
                "response": item["content"],
            }
            variables["last_turn_session"] = session_id

    valid_turns = list(range(len(chat_history)))

    trae_req = {
        "user_input": last_content,
        "intent_name": "general_qa_intent",
        "variables": json.dumps(variables, ensure_ascii=False),
        "context_resolvers": context_resolvers,
        "generate_suggested_questions": False,
        "chat_history": chat_history,
        "session_id": session_id,
        "conversation_id": session_id,
        "current_turn": messages_len - 1,
        "valid_turns": valid_turns,
        "multi_media": [],
        "model_name": trae_model,
        "last_llm_response_info": last_llm_response_info,
        "is_preset": True,
        "provider": "",
    }
    max_tokens = clamp_max_completion_tokens(max_tokens, trae_model)
    if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool) and max_tokens > 0:
        trae_req["max_output_tokens"] = int(max_tokens)
    return trae_model, trae_req


def convert_openai_messages(
    messages: list[dict], options: Optional[dict] = None
) -> list[dict]:
    """Convert OpenAI messages without losing external tool-call history."""
    result: list[dict] = []
    prepared_messages = sanitize_assistant_history_messages(
        _messages_with_client_runtime(messages, options)
    )
    tool_names = raw_client._tool_call_name_map(messages)
    for m in prepared_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        renderer_results = raw_client._renderer_tool_result_blocks(content)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("text", "input_text"):
                        parts.append(block.get("text") or block.get("content") or "")
                    elif block.get("text"):
                        parts.append(block["text"])
            content = "\n".join(parts)
            if renderer_results:
                content = raw_client._renderer_tool_result_context(
                    m.get("content"), tool_names
                )
                # Renderer results are already fully correlated above. Treat
                # them as inert user history so an outer ``role=tool`` wrapper
                # does not prepend a second, less precise result label.
                role = "user"
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        if role == "assistant":
            tool_history = raw_client._serialize_tool_calls(m.get("tool_calls"))
            if not tool_history:
                tool_history = raw_client._serialize_tool_calls(
                    raw_client._renderer_tool_use_blocks(m.get("content"))
                )
            if tool_history:
                content = "\n\n".join(part for part in (content, tool_history) if part)
        elif role == "tool":
            tool_id = m.get("tool_call_id") or "unknown"
            tool_name = m.get("name") or "tool"
            status = "failed" if tool_result_is_failed(m) else "succeeded"
            content = f"Client tool result [{tool_id}] {tool_name} status={status}:\n{content}"
            role = "user"
        elif role == "developer":
            role = "system"
        elif role not in ("system", "user", "assistant"):
            role = "user"
        result.append({"role": role, "content": content})
    return result


async def send_chat_request(messages: list[dict], model: str, stream: bool, options: Optional[dict] = None) -> IdeChatResponse:
    """发送 IDE 版聊天请求，按 IDE_ENDPOINTS 顺序自动回退。

    返回 IdeChatResponse 包装对象，调用方消费完流式内容后需调用 .close()。
    """
    from . import auth as auth_module

    # A relay session captures its credential before dispatch.  Keep that
    # snapshot independent from the mutable global auth state: account polling
    # or a console switch may happen while this request is waiting for headers.
    options = dict(options or {})
    bound_account_id = str(options.get("_account_id") or "").strip()
    bound_token = str(options.get("_auth_token") or "").strip()
    bound_record = (
        auth_module.get_account_record(bound_account_id)
        if bound_account_id
        else {}
    )
    if not bound_token and not bound_account_id:
        # The legacy unbound path may refresh the selected account.  Never
        # refresh here when a caller supplied a lease account: refresh_token()
        # operates on global auth and could replace a different account's
        # credential mid-request.
        await auth_module.maybe_refresh()
        bound_token = str(auth.get_token() or "").strip()
    if not bound_token:
        bound_token = str(bound_record.get("token") or "").strip()
    if not bound_token:
        raise RuntimeError("No Cloud-IDE-JWT token available")

    bound_user_id = str(
        options.get("_auth_user_id")
        or bound_record.get("user_id")
        or bound_account_id
        or auth.get_user_id()
        or ""
    ).strip()
    base = _model_gateway_base(
        options.get("_auth_host"),
        bound_record.get("host"),
        os.environ.get("TRAE_API_HOST", ""),
        getattr(auth.get_auth(), "host", ""),
    )

    model_name = convert_model_name(model)
    max_tokens = options.get("maxTokens") or options.get("max_tokens")
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))
    errors: list[str] = []

    requested_endpoint = str(options.get("_ide_endpoint") or "").strip()
    if requested_endpoint and requested_endpoint not in IDE_ENDPOINTS:
        raise ValueError(f"Unsupported IDE endpoint: {requested_endpoint}")
    endpoints = [requested_endpoint] if requested_endpoint else IDE_ENDPOINTS
    for endpoint in endpoints:
        endpoint_headers = build_headers(
            token_override=bound_token,
            user_id_override=bound_user_id,
        )
        # Match TraeWork's SOLO headers for llm_utils_chat/ide-v1.  The
        # account OAuth host and the old IDE version headers are accepted by
        # some endpoints but produce a generic "param is invalid" response on
        # the current gateway.
        endpoint_headers.update(
            {
                "x-app-version": "default",
                "x-app-version-code": SOLO_GATEWAY_VERSION_CODE,
                "x-ide-version": SOLO_GATEWAY_VERSION,
                "x-ide-version-code": SOLO_GATEWAY_VERSION_CODE,
                "x-ide-version-type": "stable",
                "x-device-brand": SOLO_DEVICE_BRAND,
                "x-device-type": SOLO_DEVICE_TYPE,
                "x-device-platform": SOLO_PLATFORM,
                "x-version-code": SOLO_GATEWAY_VERSION_CODE,
                "x-os-version": SOLO_OS_VERSION,
                # These CN/SOLO routing headers are present on the native
                # Trae client requests.  They are deliberately kept
                # configurable because account snapshots may come from a
                # different desktop build, while the defaults match the
                # current Trae CN stable client.
                "package-type": SOLO_PACKAGE_TYPE,
                "x-lscbd-aid": SOLO_LSCBD_AID,
                "x-lscbd-platform": SOLO_PLATFORM,
                "x-ss-dp": SOLO_LSCBD_AID,
                "x-plugin-channel": "icube-ai",
                "app-version": SOLO_GATEWAY_VERSION,
                "x-bridge-transport": "aha",
                "x-ahanet-timeout": "86400",
                "x-lgw-req-sdk-type": "3",
                "x-net-sdk-domain-dispatch": "1",
                "x-ttnet-bypass-decompression": "1",
                "x-ttnet-bypass-cookie": "0",
                "request-traffic-type": "prod",
                "User-Agent": f"Trae/{SOLO_GATEWAY_VERSION}",
            }
        )
        if endpoint == "/api/agent/v3/create_agent_task":
            endpoint_headers.update(
                {
                    "x-app-id": WORK_AGENT_APP_ID_CHAT,
                    "x-device-type": "macos",
                    "x-device-platform": "darwin",
                    "x-platform": "darwin",
                    "x-os": "darwin",
                    "x-ostype": "darwin",
                    "x-system": "darwin",
                    "x-ide-version": WORK_AGENT_VERSION,
                    "x-ide-version-code": WORK_AGENT_VERSION_CODE,
                    "x-app-version-code": WORK_AGENT_VERSION_CODE,
                    "x-version-code": WORK_AGENT_VERSION_CODE,
                    "request-traffic-type": "prod",
                    "User-Agent": f"Trae/{WORK_AGENT_VERSION}",
                }
            )
        client = httpx.Client(
            headers=endpoint_headers,
            timeout=timeout,
            http2=False,
        )
        try:
            if endpoint == "/api/ide/v1/chat":
                _, trae_req = await build_trae_ide_request(
                    messages,
                    model,
                    max_tokens,
                    options,
                )
            elif endpoint == "/api/agent/v3/create_agent_task":
                trae_req = build_work_agent_body(messages, model_name, stream, options)
            else:
                trae_req = build_llm_chat_body(
                    messages,
                    model_name,
                    stream,
                    max_tokens,
                    options,
                )
            request_base = WORK_AGENT_HOST if endpoint == "/api/agent/v3/create_agent_task" else base
            # The native client sends a route-specific referer.  Supplying it
            # also makes account records that carry ``/api/remote/v1`` behave
            # like a host-only gateway after _model_gateway_base() normalizes
            # the stale path.
            endpoint_headers["Referer"] = request_base + endpoint
            logger.info("trae-client: POST %s%s model=%s tools=%s", request_base, endpoint, model_name, _tool_protocol_requested(options, messages))
            request = client.build_request("POST", request_base + endpoint, json=trae_req)
            resp = await asyncio.to_thread(client.send, request, stream=True)
            if resp.status_code == 200:
                return IdeChatResponse(response=resp, client=client)

            await asyncio.to_thread(resp.read)
            body = resp.text[:800]
            logger.warning("trae-client: %s returned %s: %s", endpoint, resp.status_code, body)
            errors.append(f"{endpoint} [{resp.status_code}]: {body}")
        except Exception as e:
            logger.warning("trae-client: %s error: %s", endpoint, e)
            errors.append(f"{endpoint}: {e}")
        client.close()

    raise RuntimeError("Trae ide chat failed: " + " | ".join(errors))


_MODEL_LIST_CACHE: dict[str, tuple[float, list[dict]]] = {}
_MODEL_LIST_CACHE_TTL = float(os.environ.get("TRAE_MODEL_LIST_CACHE_TTL", "300"))
# Lowercase public id -> exact upstream config name, learned from the live
# model list (for example ``doubao-seed-code`` -> ``Doubao-Seed-Code``).
_UPSTREAM_MODEL_CASE: dict[str, str] = {}
# Routing ids that are not upstream configs but are always accepted.
_ROUTING_MODEL_IDS = ("auto", "work")


def _model_item(model_id: str, created: int) -> dict:
    return {"id": model_id, "object": "model", "created": created, "owned_by": "trae"}


def _static_model_list(created: int) -> list[dict]:
    """Offline fallback: lowercase built-in ids only, one entry per spelling."""
    ids = {mid for mid in SUPPORTED_MODELS if mid and mid == mid.lower()}
    ids.update(_ROUTING_MODEL_IDS)
    return [_model_item(mid, created) for mid in sorted(ids)]


async def get_models(force: bool = False) -> list[dict]:
    """Return the model list for /v1/models.

    Source is the active account's live Trae model lists: the Remote web
    catalog (merged across Agent / Work / Design tiers) plus native Solo's
    ``/api/ide/v1/get_detail_param`` catalog for Solo-only models.
    Only config names are listed; display labels such as ``GLM-5.3`` are not.
    Ids are exposed lowercase: config names that carry capitals appear once in
    lowercase and ``convert_model_name`` maps them back to the exact config
    name. The built-in table is used only when the upstream list is empty.
    """
    created = int(time.time())
    account_id, record = auth.get_active_account_snapshot()
    token = str((record or {}).get("token") or "") or auth.get_token()
    cache_key = account_id or auth.get_user_id() or token[:16] or "default"

    cached = _MODEL_LIST_CACHE.get(cache_key)
    if not force and cached and time.time() - cached[0] < _MODEL_LIST_CACHE_TTL:
        return [dict(item) for item in cached[1]]

    configs: dict[str, dict] = {}
    if token:
        kwargs: dict[str, Any] = {"token_override": token}
        provider = (record or {}).get("provider_specific") or (record or {}).get(
            "providerSpecificData"
        )
        if isinstance(provider, Mapping):
            kwargs["provider_specific"] = dict(provider)
        # Remote is the primary catalog because it carries the richer
        # Agent/Work metadata.  Solo fills only models absent from Remote;
        # this keeps discovery resilient when a newly enabled Solo model has
        # not propagated to the web catalog yet.
        remote_task = _fetch_web_model_configs(**kwargs)
        solo_task = _fetch_solo_model_configs(**kwargs)
        remote_configs, solo_configs = await asyncio.gather(
            remote_task,
            solo_task,
            return_exceptions=True,
        )
        if isinstance(remote_configs, Mapping):
            configs.update(remote_configs)
        elif isinstance(remote_configs, Exception):
            logger.warning("trae-client: web model list fetch failed: %s", remote_configs)
        if isinstance(solo_configs, Mapping):
            for name, config in solo_configs.items():
                configs.setdefault(name, config)
        elif isinstance(solo_configs, Exception):
            logger.warning("trae-client: solo model list fetch failed: %s", solo_configs)

    ids: set[str] = set()
    for name in configs:
        exact = str(name or "").strip()
        if not exact:
            continue
        lowered = exact.lower()
        ids.add(lowered)
        if lowered != exact:
            _UPSTREAM_MODEL_CASE[lowered] = exact
    if not ids:
        # Not cached, so the next request retries the upstream list.
        logger.warning("trae-client: upstream model list empty, using built-in list")
        return _static_model_list(created)

    ids.update(_ROUTING_MODEL_IDS)
    items = [_model_item(mid, created) for mid in sorted(ids)]
    _MODEL_LIST_CACHE[cache_key] = (time.time(), items)
    return [dict(item) for item in items]
