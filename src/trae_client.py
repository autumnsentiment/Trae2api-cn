"""
trae_client.py - Trae API 客户端

支持两条上游路径：
1. IDE/桌面版：先走 /api/agent/v3/llm_utils_chat，失败后依次回退
   /api/ide/v1/chat 与 /api/agent/v3/create_agent_task
2. 网页版 remote 会话（OmniRoute 风格）：POST /chat_sessions +
   GET /chat_sessions/{id}/events
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import string
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

from . import auth

logger = logging.getLogger(__name__)

IDE_VERSION = os.environ.get("TRAE_IDE_VERSION", "3.3.67")
IDE_VERSION_CODE = os.environ.get("TRAE_IDE_VERSION_CODE", "20260401")
IDE_VERSION_CODE_NUM = int(os.environ.get("TRAE_IDE_VERSION_CODE_NUM", "20260401") or 20260401)
IDE_APP_ID = os.environ.get("TRAE_APP_ID", "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8")
TRAE_CLIENT_ID = os.environ.get("TRAE_CLIENT_ID", "ono9krqynydwx5")

# 按优先级依次尝试的 IDE 版上游端点（参考 laojichao/trae-local-api）
IDE_ENDPOINTS = [
    "/api/agent/v3/llm_utils_chat",
    "/api/ide/v1/chat",
    "/api/agent/v3/create_agent_task",
]

# 外部模型名 -> Trae CN 内部模型名（基于最新 Trae CN 模型映射）
MODEL_ALIASES = {
    "auto": "glm-5.2",
    "glm-5.2": "glm-5.2",
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
    "mimo-v2.5-pro": "mimo-v2.5-pro",
    "mimo-v2.5": "mimo-v2.5",
    "claude-haiku-4-5": "glm-5.1",
    "glm-5.1": "glm-5.1",
    "qwen-3.7-plus": "qwen-3.7-plus",
    "kimi-k2.6": "kimi-k2.6",
    "gpt-4o": "DeepSeek-V4-Pro",
    "gpt-4o-latest": "DeepSeek-V4-Pro",
    "gpt-4.1": "DeepSeek-V4-Pro",
    "deepseek-v3": "DeepSeek-V4-Pro",
    "deepseek-r1": "DeepSeek-V4-Pro",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "glm-5": "glm-5",
    "qwen-3.6-plus": "qwen-3.6-plus",
    "minimax-m3": "minimax-m3",
    "gpt-4o-mini": "DeepSeek-V4-Flash",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "glm-4.7": "glm-4.7",
    "kimi-k2": "kimi-k2",
    "qwen3-coder": "qwen3-coder",
    "minimax-m2.7": "minimax-m2.7",
    "glm-4.6": "glm-4.6",
    "minimax-m2.1": "minimax-m2.1",
    "work": "work",
}

_ALIAS_LOOKUP = {k.lower(): v for k, v in MODEL_ALIASES.items()}

# /v1/models 返回的模型 ID
SUPPORTED_MODELS = set(_ALIAS_LOOKUP.keys()) | set(_ALIAS_LOOKUP.values())

# 设备信息（参考 trae2api config/device.go）
_DEVICE_BRANDS = ["92L3", "91C9", "814S", "8P15V", "35G4", "65G4", "55G4"]
_SESSION_CACHE: dict[str, str] = {}

_WEB_SLOTS: dict[str, asyncio.Semaphore] = {}
_WEB_LEASES: dict[str, dict] = {}
_WEB_MODEL_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
_WEB_PARALLEL_LIMIT = int(os.environ.get("TRAE_WEB_PARALLEL_LIMIT", "2"))
_WEB_IDLE_TIMEOUT = float(os.environ.get("TRAE_WEB_IDLE_TIMEOUT", "60"))
_WEB_MODEL_CACHE_TTL = float(os.environ.get("TRAE_WEB_MODEL_CACHE_TTL", "300"))
_WORKSPACE_PREFIXES = ["User", "home", "workspace", "data"]
_WORKSPACE_DIRS = ["projects", "workspace", "dev", "code", "work"]


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
    """把 OpenAI/Claude 风格模型名转成 Trae CN 内部模型名。"""
    m = (model or "").strip().lower()
    return _ALIAS_LOOKUP.get(m, model)


def is_model_supported(model: str) -> bool:
    """中转站保持透传：未知模型也交给上游判断，避免新模型上线后需要改代码。"""
    return True


def generate_session_id_from_messages(messages: list[dict]) -> str:
    """根据首条消息内容生成稳定会话 ID（和 trae2api 一致）。"""
    if not messages:
        return str(uuid.uuid4())
    first = messages[0]
    key = f"{first.get('role','')}: {_content_to_text(first.get('content',''))}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    if digest in _SESSION_CACHE:
        return _SESSION_CACHE[digest]
    sid = str(uuid.uuid4())
    _SESSION_CACHE[digest] = sid
    return sid


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or c.get("content") or "")
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
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


def build_headers() -> dict[str, str]:
    """构造与 trae2api/laojichao 兼容的 IDE 请求头。"""
    device = get_current_device()
    token = auth.get_token()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Cloud-IDE-JWT {token}",
        "X-Cloudide-Token": token,
        "x-uid": auth.get_user_id(),
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


def build_web_headers(token: str) -> dict[str, str]:
    """构造参考 OmniRoute 网页版 remote 会话的请求头。"""
    psd = auth.get_psd()
    return {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
        "X-Trae-Client-Type": "web",
        "X-Preferenced-Language": psd.get("appLanguage") or os.environ.get("TRAE_WEB_LANGUAGE", "zh-CN"),
        "x-user-region": psd.get("userRegion") or os.environ.get("TRAE_WEB_USER_REGION", "CN"),
        "Referer": "https://solo.trae.cn/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
        ),
    }


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
    for m in messages:
        content = _content_to_text(m.get("content", ""))
        role = m.get("role", "user")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    text = "\n\n".join(parts)
    return json.dumps([{"type": "text", "data": {"content": text}}], ensure_ascii=False)


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


def register_web_lease(account_id: str, session_id: str, message_id: str, client: httpx.AsyncClient) -> None:
    _WEB_LEASES[session_id] = {
        "account_id": account_id,
        "session_id": session_id,
        "message_id": message_id,
        "client": client,
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



async def stop_web_session(client: httpx.AsyncClient, session_id: str, message_id: str) -> None:
    """Actively interrupt an upstream web session so it stops occupying a running slot."""
    if not session_id or not message_id:
        return
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    token = auth.get_token()
    if not token:
        return
    try:
        resp = await client.post(
            f"{base}/chat_sessions/{session_id}/stop",
            headers=build_web_headers(token),
            json={"chat_session_id": session_id, "user_message_id": message_id},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("trae-client: stop web session %s returned %s", session_id, resp.status_code)
    except Exception as e:
        logger.warning("trae-client: stop web session %s failed: %s", session_id, e)


async def reap_idle_web_sessions() -> int:
    count = 0
    for lease in idle_web_leases():
        session_id = lease["session_id"]
        account_id = lease["account_id"]
        client = lease.get("client")
        message_id = lease.get("message_id") or ""
        logger.warning("trae-client: reaping idle web session %s (account %s)", session_id, account_id)
        if unregister_web_lease(session_id):
            if client is not None:
                try:
                    await stop_web_session(client, session_id, message_id)
                except Exception:
                    pass
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass
            release_web_slot(account_id)
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
    cfg["config_source"] = cfg.get("config_source") or 1
    cfg["provider"] = cfg.get("provider") or ""
    cfg["multimodal"] = bool(cfg.get("multimodal"))
    cfg["ak"] = cfg.get("ak") or ""
    cfg["sk"] = cfg.get("sk") or ""
    cfg["base_url"] = cfg.get("base_url") or ""
    cfg["auth_type"] = cfg.get("auth_type") or 0
    cfg["use_remote_service"] = not bool(cfg.get("client_connect"))
    return cfg


async def _fetch_web_model_configs() -> dict[str, dict]:
    """Fetch the same model list used by the Trae web client."""
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    token = auth.get_token()
    if not token:
        return {}
    url = f"{base}/models?functions=solo_agent_remote%2Csolo_work_remote%2Csolo_design_remote&show_custom_model=true"
    try:
        async with httpx.AsyncClient(headers=build_web_headers(token), timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning("trae-client: web model list returned %s", resp.status_code)
                return {}
            data = resp.json()
    except Exception as e:
        logger.warning("trae-client: web model list failed: %s", e)
        return {}
    out: dict[str, dict] = {}
    for group in data.get("data", {}).get("list", []):
        for raw in group.get("models", []):
            name = (raw.get("name") or "").strip()
            if name:
                out[name] = _build_web_model_config(raw)
    return out


async def _get_web_custom_model(model_name: str) -> Optional[dict]:
    """Return custom_model for a manual web model selection."""
    if not model_name:
        return None
    account_key = auth.get_user_id() or auth.get_token()[:16] or "default"
    now = time.time()
    cached = _WEB_MODEL_CACHE.get(account_key)
    if not cached or now - cached[0] > _WEB_MODEL_CACHE_TTL:
        configs = await _fetch_web_model_configs()
        cached = (now, configs)
        _WEB_MODEL_CACHE[account_key] = cached
    cfg = cached[1].get(model_name)
    if cfg is not None:
        return cfg
    lowered = model_name.lower()
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


async def create_web_session(
    client: httpx.AsyncClient,
    model: str,
    query: str,
) -> tuple[str, str]:
    """POST /chat_sessions（OmniRoute 网页版方案）。"""
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    token = auth.get_token()
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")

    mode, strategy, model_name = _resolve_mode(model)
    psd = auth.get_psd()
    initial_message = {
        "chat_session_id": "",
        "content": [],
        "query": query,
        "model_name": model_name,
        "agent_type": "solo_agent_remote",
        "model_selection_strategy": strategy,
        "common_params": _web_common_params(psd, mode),
    }
    if strategy != "auto":
        custom_model = await _get_web_custom_model(model_name)
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
        headers=build_web_headers(token),
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
) -> AsyncIterator[tuple[str, dict]]:
    """GET /chat_sessions/{id}/events?reply_to_message_id=...（SSE）。"""
    base = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")
    url = f"{base}/chat_sessions/{session_id}/events?reply_to_message_id={message_id}"
    token = auth.get_token()
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))
    async with client.stream("GET", url, headers=build_web_headers(token), timeout=timeout) as resp:
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


def build_llm_chat_body(
    messages: list[dict],
    model: str,
    stream: bool,
    max_tokens: Optional[int] = None,
) -> dict:
    """构造 /api/agent/v3/llm_utils_chat 与 create_agent_task 的请求体。"""
    converted = convert_openai_messages(messages)
    session_id = str(uuid.uuid4())
    body = {
        "messages": [
            {
                "role": m.get("role", "user"),
                "content": [{"type": "text", "text": m.get("content", "")}],
            }
            for m in converted
        ],
        "model": model,
        "function": "inline_chat",
        "stream": stream,
        "request_id": session_id,
        "session_id": session_id,
    }
    if max_tokens and max_tokens > 0:
        body["max_tokens"] = int(max_tokens)
    return body


async def build_trae_ide_request(
    messages: list[dict],
    model: str,
    max_tokens: Optional[int] = None,
) -> tuple[str, dict]:
    """构造与 trae2api 一致的 /api/ide/v1/chat 请求体。"""
    converted = convert_openai_messages(messages)
    session_id = generate_session_id_from_messages(converted)
    trae_model = convert_model_name(model)

    messages_len = len(converted)
    if messages_len == 0:
        raise ValueError("messages cannot be empty")
    last_content = _content_to_text(converted[-1].get("content", ""))

    context_resolvers = [
        {"resolver_id": "project-labels", "variables": '{"labels":"- go\n- go.mod"}'},
        {"resolver_id": "terminal_context", "variables": '{"terminal_context":[]}'},
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
        "workspace_path": generate_random_workspace_path(),
        "brand": "Trae",
        "system_type": "Windows",
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
    if max_tokens:
        trae_req["max_output_tokens"] = max_tokens

    return trae_model, trae_req


def convert_openai_messages(messages: list[dict]) -> list[dict]:
    """把 OpenAI 消息数组转成 Trae 需要的纯文本消息。"""
    result: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("text", "input_text"):
                        parts.append(block.get("text") or block.get("content") or "")
                    elif block.get("text"):
                        parts.append(block["text"])
            content = "\n".join(parts)
        elif not isinstance(content, str):
            content = str(content)
        result.append({"role": role, "content": content})
    return result


async def send_chat_request(messages: list[dict], model: str, stream: bool, options: Optional[dict] = None) -> IdeChatResponse:
    """发送 IDE 版聊天请求，按 IDE_ENDPOINTS 顺序自动回退。

    返回 IdeChatResponse 包装对象，调用方消费完流式内容后需调用 .close()。
    """
    from . import auth as auth_module

    await auth_module.maybe_refresh()

    base = os.environ.get("TRAE_API_HOST", "") or auth.get_auth().host or "https://trae-api-cn.mchost.guru"
    base = base.rstrip("/")

    model_name = convert_model_name(model)
    options = options or {}
    max_tokens = options.get("maxTokens") or options.get("max_tokens")
    timeout = float(os.environ.get("STREAM_TIMEOUT", "300"))
    errors: list[str] = []

    for endpoint in IDE_ENDPOINTS:
        client = httpx.Client(headers=build_headers(), timeout=timeout, http2=False)
        try:
            if endpoint == "/api/ide/v1/chat":
                _, trae_req = await build_trae_ide_request(messages, model, max_tokens)
            else:
                trae_req = build_llm_chat_body(messages, model_name, stream, max_tokens)
            logger.info("trae-client: POST %s%s model=%s", base, endpoint, model_name)
            request = client.build_request("POST", base + endpoint, json=trae_req)
            resp = client.send(request, stream=True)
            if resp.status_code == 200:
                return IdeChatResponse(response=resp, client=client)

            resp.read()
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


def _static_model_list(created: int) -> list[dict]:
    """内置模型表；无配置时兜底返回 auto。"""
    items = [
        {"id": mid, "object": "model", "created": created, "owned_by": "trae"}
        for mid in sorted(SUPPORTED_MODELS)
    ]
    return items or [{"id": "auto", "object": "model", "created": created, "owned_by": "trae"}]


async def get_models(force: bool = False) -> list[dict]:
    """返回模型列表。

    默认使用内置模型表。TRAE_FETCH_MODEL_LIST=true 时尝试从上游获取，
    结果缓存 TRAE_MODEL_LIST_CACHE_TTL 秒；force=True 时跳过缓存强制刷新。
    """
    created = int(time.time())
    cache_key = auth.get_user_id() or auth.get_token()[:16] or "default"

    if os.environ.get("TRAE_FETCH_MODEL_LIST", "").lower() != "true":
        return _static_model_list(created)

    cached = _MODEL_LIST_CACHE.get(cache_key)
    if not force and cached and time.time() - cached[0] < _MODEL_LIST_CACHE_TTL:
        return cached[1]

    await auth.maybe_refresh()
    base = os.environ.get("TRAE_API_HOST", "") or auth.get_auth().host or "https://trae-api-cn.mchost.guru"
    base = base.rstrip("/")
    url = f"{base}/api/ide/v1/model_list?type=chat"
    try:
        async with httpx.AsyncClient(headers=build_headers(), timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"model_list [{resp.status_code}]: {resp.text[:500]}")
            data = resp.json()
        out = []
        for item in data.get("model_configs", []):
            name = item.get("name", "")
            if name == "aws_sdk_claude37_sonnet":
                name = "claude-3-7-sonnet"
            elif name == "claude3.5":
                name = "claude-3-5-sonnet"
            if name:
                out.append({"id": name, "object": "model", "created": created, "owned_by": "trae"})
        items = out or _static_model_list(created)
        _MODEL_LIST_CACHE[cache_key] = (time.time(), items)
        return items
    except Exception as e:
        logger.warning("trae-client: model_list failed, using static list: %s", e)
        items = _static_model_list(created)
        _MODEL_LIST_CACHE[cache_key] = (time.time(), items)
        return items
