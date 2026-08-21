"""
auth.py - 认证管理

支持四种凭证来源：
- auto: 自动从 Trae CN 桌面客户端 storage.json 解密；失败时退回 .env，最后退回本机 Trae CLI
- env:  从 .env 读取 TRAE_TOKEN / TRAE_REFRESH_TOKEN / TRAE_USER_*
- manual: 用户粘贴 Cloud-IDE-JWT（网页版抓包）
- cli: 使用本机 Trae CLI 子进程，不要求 Cloud-IDE-JWT
"""

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from . import trae_decrypt

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = APP_DIR / ".env"
ACCOUNTS_PATH = APP_DIR / "data" / "accounts.json"

DEFAULT_BASE_URLS = {
    "cn": "https://trae-api-cn.mchost.guru",
    "solo": "https://trae-api-cn.mchost.guru",
    "sg": "https://a0ai-api-sg.byteintlapi.com",
    "solo-sg": "https://a0ai-api-sg.byteintlapi.com",
    "web-cn": "https://trae-api-cn.mchost.guru/api/remote/v1",
    "web-sg": "https://core-normal.trae.ai/api/remote/v1",
}


@dataclass
class AuthState:
    edition: str = "cn"
    source: str = "auto"
    token: str = ""
    refresh_token: Optional[str] = None
    user_id: str = ""
    host: str = ""
    client_id: str = ""
    expired_at: Optional[str] = None
    refresh_expired_at: Optional[str] = None
    provider_specific: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_valid(self) -> bool:
        if not self.token:
            return self.source == "cli"
        ts = self.expires_ts()
        if ts is None:
            return True
        return ts - time.time() > 300  # 5 分钟缓冲

    def expires_ts(self) -> Optional[float]:
        if not self.expired_at:
            return None
        raw = str(self.expired_at).strip()
        try:
            value = float(raw)
            if value > 1e12:
                value /= 1000.0  # 毫秒时间戳
            return value
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    def log_summary(self) -> None:
        if self.token:
            logger.info("auth: token=%s...", self.token[:36])
        if self.user_id:
            logger.info("auth: user_id=%s", self.user_id)
        if self.expired_at:
            logger.info("auth: expires=%s", self.expired_at)
        if self.source == "cli":
            logger.info("auth: using Trae CLI subprocess, no JWT required")


_auth = AuthState()
_refresh_lock = threading.Lock()
_STORE_LOCK = threading.RLock()
_accounts: dict[str, dict] = {}
_active_account: str = ""
_poll_enabled: bool = False
_polling_mode: str = "round-robin"  # "round-robin" | "credit-priority"
_rotation_cursor: int = 0
_settings: dict = {}


def _safe_read_env_value(key: str) -> str:
    try:
        if not ENV_PATH.exists():
            return ""
        content = ENV_PATH.read_text("utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
        return ""
    except Exception:
        return ""


def init_auth() -> AuthState:
    """初始化认证状态。每次启动时调用。"""
    source = os.environ.get("TRAE_AUTH_SOURCE", "auto").lower()
    state = AuthState(source=source)
    global _auth

    if source == "manual":
        token = os.environ.get("TRAE_MANUAL_TOKEN", "").strip()
        if not token:
            token = _safe_read_env_value("TRAE_MANUAL_TOKEN")
        state.token = token
        state.user_id = os.environ.get("TRAE_USER_ID", "") or _safe_read_env_value("TRAE_USER_ID")
        state.refresh_token = os.environ.get("TRAE_REFRESH_TOKEN", "") or _safe_read_env_value("TRAE_REFRESH_TOKEN") or None
        state.expired_at = os.environ.get("TRAE_TOKEN_EXPIRES", "") or _safe_read_env_value("TRAE_TOKEN_EXPIRES") or None
        state.host = os.environ.get("TRAE_API_HOST", "") or _safe_read_env_value("TRAE_API_HOST")
        _load_web_provider_specific(state)
        _load_env_overrides(state)
        if not state.token:
            raise RuntimeError("TRAE_AUTH_SOURCE=manual but TRAE_MANUAL_TOKEN is empty")
        state.log_summary()

    elif source == "env":
        state.token = os.environ.get("TRAE_TOKEN", "") or _safe_read_env_value("TRAE_TOKEN")
        state.refresh_token = os.environ.get("TRAE_REFRESH_TOKEN", "") or _safe_read_env_value("TRAE_REFRESH_TOKEN") or None
        state.user_id = os.environ.get("TRAE_USER_ID", "") or _safe_read_env_value("TRAE_USER_ID")
        state.expired_at = os.environ.get("TRAE_TOKEN_EXPIRES", "") or _safe_read_env_value("TRAE_TOKEN_EXPIRES") or None
        state.host = os.environ.get("TRAE_API_HOST", "") or _safe_read_env_value("TRAE_API_HOST")
        _load_web_provider_specific(state)
        _load_env_overrides(state)
        if not state.token:
            raise RuntimeError("TRAE_AUTH_SOURCE=env but TRAE_TOKEN is empty")
        state.log_summary()

    elif source == "web-login":
        # 浏览器授权登录：允许启动时无 token，由 /api/web-auth 写入。
        state.token = os.environ.get("TRAE_TOKEN", "") or _safe_read_env_value("TRAE_TOKEN")
        state.refresh_token = os.environ.get("TRAE_REFRESH_TOKEN", "") or _safe_read_env_value("TRAE_REFRESH_TOKEN") or None
        state.user_id = os.environ.get("TRAE_USER_ID", "") or _safe_read_env_value("TRAE_USER_ID")
        state.expired_at = os.environ.get("TRAE_TOKEN_EXPIRES", "") or _safe_read_env_value("TRAE_TOKEN_EXPIRES") or None
        state.host = os.environ.get("TRAE_API_HOST", "") or _safe_read_env_value("TRAE_API_HOST")
        _load_web_provider_specific(state)
        _load_env_overrides(state)
        if state.token:
            state.log_summary()
    elif source == "cli":
        from . import cli_client

        command = cli_client.resolve_cli_command()
        if not command:
            raise RuntimeError(
                "TRAE_AUTH_SOURCE=cli but Trae CLI executable not found; "
                "install traecli/trae-cli/traex or set TRAE_CLI_COMMAND"
            )
        state.source = "cli"
        state.edition = "cli"
        state.host = os.environ.get("TRAE_API_HOST", "") or _safe_read_env_value("TRAE_API_HOST")
        _load_web_provider_specific(state)
        _load_env_overrides(state)
        state.log_summary()

    else:  # auto
        auth_data, edition = trae_decrypt.try_auto_discover()
        if not auth_data:
            # 也尝试从环境变量兜底
            token = os.environ.get("TRAE_TOKEN", "") or _safe_read_env_value("TRAE_TOKEN")
            if token:
                state.edition = "env"
                state.token = token
                state.refresh_token = os.environ.get("TRAE_REFRESH_TOKEN", "") or _safe_read_env_value("TRAE_REFRESH_TOKEN") or None
                state.user_id = os.environ.get("TRAE_USER_ID", "") or _safe_read_env_value("TRAE_USER_ID")
                state.expired_at = os.environ.get("TRAE_TOKEN_EXPIRES", "") or _safe_read_env_value("TRAE_TOKEN_EXPIRES") or None
                state.host = os.environ.get("TRAE_API_HOST", "") or _safe_read_env_value("TRAE_API_HOST")
                state.log_summary()
                state.source = "env"
                _auth = state
                _save_env_snapshot()
                return state

            # 最后尝试本机 Trae CLI，不需要 JWT
            from . import cli_client

            if cli_client.resolve_cli_command():
                state.source = "cli"
                state.edition = "cli"
                state.host = os.environ.get("TRAE_API_HOST", "") or _safe_read_env_value("TRAE_API_HOST")
                _load_web_provider_specific(state)
                _load_env_overrides(state)
                _auth = state
                state.log_summary()
                return state

            raise RuntimeError(
                "Auto auth failed. Configure TRAE_AUTH_SOURCE=manual/env/cli, "
                "install Trae CLI, or ensure Trae CN is installed and logged in."
            )

        state.edition = edition
        state.source = "auto"
        state.token = auth_data.get("token") or ""
        state.refresh_token = auth_data.get("refreshToken") or None
        state.user_id = auth_data.get("userId") or auth_data.get("UserID") or ""
        state.expired_at = auth_data.get("expiredAt") or auth_data.get("TokenExpireAt") or ""
        state.refresh_expired_at = auth_data.get("refreshExpiredAt") or auth_data.get("RefreshExpireAt") or ""
        state.host = auth_data.get("host") or os.environ.get("TRAE_API_HOST", "") or DEFAULT_BASE_URLS.get(edition, "")
        if edition == "cn" and (not state.host or "mchost.guru" in state.host):
            state.host = "https://api.trae.cn"
        state.client_id = auth_data.get("clientID") or auth_data.get("ClientID") or auth_data.get("clientId") or os.environ.get("TRAE_CLIENT_ID", "") or _safe_read_env_value("TRAE_CLIENT_ID")
        _extract_provider_specific_from_auth(state, auth_data)
        _load_env_overrides(state)
        state.log_summary()

    _auth = state
    _bootstrap_account_store()
    if state.source != "cli" or state.token:
        _save_env_snapshot()
    return state


def _load_web_provider_specific(state: AuthState) -> None:
    """从环境变量加载网页版身份字段"""
    psd = {
        "webId": os.environ.get("TRAE_WEB_ID", "") or _safe_read_env_value("TRAE_WEB_ID"),
        "bizUserId": os.environ.get("TRAE_BIZ_USER_ID", "") or _safe_read_env_value("TRAE_BIZ_USER_ID"),
        "userUniqueId": os.environ.get("TRAE_USER_UNIQUE_ID", "") or _safe_read_env_value("TRAE_USER_UNIQUE_ID"),
        "scope": os.environ.get("TRAE_WEB_SCOPE", "marscode-cn") or _safe_read_env_value("TRAE_WEB_SCOPE") or "marscode-cn",
        "tenant": os.environ.get("TRAE_WEB_TENANT", "marscode") or _safe_read_env_value("TRAE_WEB_TENANT") or "marscode",
        "region": os.environ.get("TRAE_WEB_REGION", "cn") or _safe_read_env_value("TRAE_WEB_REGION") or "cn",
        "aiRegion": os.environ.get("TRAE_WEB_AI_REGION", "cn") or _safe_read_env_value("TRAE_WEB_AI_REGION") or "cn",
        "appLanguage": os.environ.get("TRAE_WEB_APP_LANGUAGE", "zh-CN") or _safe_read_env_value("TRAE_WEB_APP_LANGUAGE") or "zh-CN",
        "appVersion": os.environ.get("TRAE_WEB_APP_VERSION", "1.0.0.1229") or _safe_read_env_value("TRAE_WEB_APP_VERSION") or "1.0.0.1229",
        "userRegion": os.environ.get("TRAE_WEB_USER_REGION", "CN") or _safe_read_env_value("TRAE_WEB_USER_REGION") or "CN",
        "userIdentity": os.environ.get("TRAE_WEB_USER_IDENTITY", "Free") or _safe_read_env_value("TRAE_WEB_USER_IDENTITY") or "Free",
    }
    state.provider_specific.update(psd)


def _extract_provider_specific_from_auth(state: AuthState, auth_data: dict) -> None:
    """从 storage.json 解密结果中提取网页版身份字段"""
    psd = state.provider_specific

    # 常见字段名映射
    field_map = {
        "webId": ["webId", "web_id", "WebId"],
        "bizUserId": ["bizUserId", "biz_user_id", "BizUserId"],
        "userUniqueId": ["userUniqueId", "user_unique_id", "UserUniqueId"],
        "scope": ["scope", "Scope"],
        "tenant": ["tenant", "Tenant"],
        "region": ["region", "Region"],
        "aiRegion": ["aiRegion", "ai_region"],
        "appLanguage": ["appLanguage", "app_language"],
        "appVersion": ["appVersion", "app_version"],
        "userRegion": ["userRegion", "user_region"],
        "userIdentity": ["userIdentity", "user_identity"],
    }
    for key, names in field_map.items():
        if not psd.get(key):
            for name in names:
                if auth_data.get(name):
                    psd[key] = auth_data[name]
                    break

    # 如果 auth_data 包含嵌套 providerSpecificData 或 common_params
    for nested_key in ("providerSpecificData", "commonParams", "common_params"):
        nested = auth_data.get(nested_key)
        if isinstance(nested, dict):
            for key in field_map:
                if not psd.get(key) and nested.get(key):
                    psd[key] = nested[key]


def _load_env_overrides(state: AuthState) -> None:
    """允许用户用环境变量覆盖 WEB 端身份字段"""
    overrides = {
        "webId": "TRAE_WEB_ID",
        "bizUserId": "TRAE_BIZ_USER_ID",
        "userUniqueId": "TRAE_USER_UNIQUE_ID",
        "scope": "TRAE_WEB_SCOPE",
        "tenant": "TRAE_WEB_TENANT",
        "region": "TRAE_WEB_REGION",
        "aiRegion": "TRAE_WEB_AI_REGION",
        "appLanguage": "TRAE_WEB_APP_LANGUAGE",
        "appVersion": "TRAE_WEB_APP_VERSION",
        "userRegion": "TRAE_WEB_USER_REGION",
        "userIdentity": "TRAE_WEB_USER_IDENTITY",
    }
    for key, env in overrides.items():
        value = os.environ.get(env) or _safe_read_env_value(env)
        if value:
            state.provider_specific[key] = value


def get_auth() -> AuthState:
    return _auth


def get_token() -> str:
    with _auth._lock:
        return _auth.token or ""


def get_user_id() -> str:
    with _auth._lock:
        return _auth.user_id or ""


def get_psd() -> dict:
    with _auth._lock:
        return dict(_auth.provider_specific)


def needs_refresh() -> bool:
    state = _auth
    if not state.refresh_token:
        return False
    if not state.expired_at:
        return False
    ts = state.expires_ts()
    if ts is None:
        return False
    return time.time() >= ts - 1800  # 提前 30 分钟刷新


async def refresh_token() -> bool:
    """调用 ExchangeToken 刷新 Cloud-IDE-JWT"""
    global _auth
    if not _refresh_lock.acquire(blocking=False):
        return False
    try:
        # Capture the account and its refresh credentials together. Account
        # polling or a console switch may replace ``_auth`` while the network
        # request is in flight, so the response must never be written through
        # the mutable global state.
        with _STORE_LOCK:
            captured_account_id = _active_account
            captured_record = dict(_accounts.get(captured_account_id) or {})
            captured_state = _auth
            if captured_record:
                rt = captured_record.get("refresh_token") or ""
                host = captured_record.get("host") or DEFAULT_BASE_URLS.get(
                    captured_record.get("edition", "cn"),
                    "https://trae-api-cn.mchost.guru",
                )
                client_id = captured_record.get("client_id") or "ono9krqynydwx5"
                user_id = captured_record.get("user_id") or ""
            else:
                with captured_state._lock:
                    rt = captured_state.refresh_token or ""
                    host = captured_state.host or DEFAULT_BASE_URLS.get(
                        captured_state.edition,
                        "https://trae-api-cn.mchost.guru",
                    )
                    client_id = captured_state.client_id or "ono9krqynydwx5"
                    user_id = captured_state.user_id or ""
        if not rt:
            logger.error("auth: no refresh token, cannot refresh")
            return False

        url = f"{host}/cloudide/api/v3/trae/oauth/ExchangeToken"
        payload = {
            "ClientID": client_id,
            "RefreshToken": rt,
            "ClientSecret": "-",
            "UserID": user_id,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        result = data.get("Result") or data.get("result") or {}
        new_token = result.get("Token") or result.get("token") or ""
        new_refresh = result.get("RefreshToken") or result.get("refreshToken") or rt
        new_expire = result.get("TokenExpireAt") or result.get("tokenExpireAt") or ""

        if not new_token:
            error = data.get("ResponseMetadata", {}).get("Error")
            logger.error("auth: exchange token failed: %s", error)
            return False

        normalized_expire = _normalize_expire(new_expire)
        update_active_state = False
        with _STORE_LOCK:
            if captured_account_id:
                current_record = _accounts.get(captured_account_id)
                if current_record is None:
                    logger.warning(
                        "auth: refreshed account %s was removed before update",
                        captured_account_id,
                    )
                    return False
                current_refresh = current_record.get("refresh_token") or ""
                if current_refresh and current_refresh != rt:
                    logger.warning(
                        "auth: discarded stale refresh response for account %s",
                        captured_account_id,
                    )
                    return False
                updated_record = dict(current_record)
                updated_record.update(
                    {
                        "token": new_token,
                        "refresh_token": new_refresh,
                        "expired_at": normalized_expire,
                        "client_id": client_id,
                    }
                )
                _accounts[captured_account_id] = updated_record
                if _active_account == captured_account_id:
                    _switch_record(updated_record)
                    update_active_state = True
                _save_accounts()
            else:
                # Preserve the legacy single-account path, but only while the
                # exact state captured before the request is still selected.
                if _active_account or _auth is not captured_state:
                    logger.warning(
                        "auth: discarded refresh response after account switch"
                    )
                    return False
                with captured_state._lock:
                    captured_state.token = new_token
                    captured_state.refresh_token = new_refresh
                    captured_state.expired_at = normalized_expire
                    captured_state.client_id = client_id
                update_active_state = True

        logger.info(
            "auth: token refreshed%s",
            f" for account {captured_account_id}" if captured_account_id else "",
        )
        if update_active_state:
            _save_env_snapshot()
        return True
    except Exception as e:
        logger.error("auth: refresh error: %s", e)
        return False
    finally:
        _refresh_lock.release()


async def maybe_refresh() -> bool:
    if needs_refresh():
        logger.info("auth: token near expiry, refreshing")
        return await refresh_token()
    return False


def apply_oauth_callback(
    token: str,
    refresh_token: str = "",
    user_id: str = "",
    tenant_id: str = "",
    region: str = "",
    ai_region: str = "",
    host: str = "",
    expired_at: str = "",
    refresh_expired_at: str = "",
    client_id: str = "",
    web_id: str = "",
    biz_user_id: str = "",
    user_unique_id: str = "",
    scope: str = "",
    tenant: str = "",
    app_language: str = "",
    user_region: str = "",
    user_identity: str = "",
    screen_name: str = "",
) -> None:
    """Save credentials captured by the browser OAuth callback."""
    global _auth
    state = _auth
    with state._lock:
        if token:
            state.token = token
        if refresh_token:
            state.refresh_token = refresh_token
        if user_id:
            state.user_id = user_id
        if host:
            state.host = host.rstrip("/")
        if client_id:
            state.client_id = client_id
        if expired_at:
            state.expired_at = _normalize_expire(expired_at)
        if refresh_expired_at:
            state.refresh_expired_at = _normalize_expire(refresh_expired_at)

        psd = state.provider_specific
        if web_id:
            psd["webId"] = web_id
        if biz_user_id:
            psd["bizUserId"] = biz_user_id
        if user_unique_id:
            psd["userUniqueId"] = user_unique_id
        if scope:
            psd["scope"] = scope
        if tenant:
            psd["tenant"] = tenant
        if region:
            psd["region"] = region.lower()
        if ai_region:
            psd["aiRegion"] = ai_region.lower()
        if app_language:
            psd["appLanguage"] = app_language
        if user_region:
            psd["userRegion"] = user_region
        if user_identity:
            psd["userIdentity"] = user_identity
        if tenant_id:
            psd["tenantId"] = tenant_id
        if screen_name:
            psd["screenName"] = screen_name
        state.source = "env"
    _save_env_snapshot()
    _persist_active_account()
    logger.info("auth: web login credentials saved (user=%s)", state.user_id or "unknown")


def _normalize_expire(value) -> str:
    """把秒/毫秒时间戳或 ISO 字符串统一成 ISO 8601 UTC 字符串。"""
    if value in (None, ""):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        num = float(raw)
        if num > 1e12:
            num /= 1000.0  # 毫秒时间戳
        return datetime.fromtimestamp(num, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return raw


def _save_env_snapshot() -> None:
    """把当前认证状态写回 .env；已有键原地更新，未出现的键追加，保留其他用户配置。"""
    state = _auth
    if state.source == "cli" and not state.token:
        return
    try:
        values = {
            "TRAE_AUTH_SOURCE": state.source,
            "TRAE_EDITION": state.edition,
            "TRAE_TOKEN": state.token,
            "TRAE_REFRESH_TOKEN": state.refresh_token or "",
            "TRAE_USER_ID": state.user_id or "",
            "TRAE_API_HOST": state.host or "",
            "TRAE_TOKEN_EXPIRES": state.expired_at or "",
            "TRAE_REFRESH_EXPIRES": state.refresh_expired_at or "",
            "TRAE_CLIENT_ID": state.client_id or "",
            "TRAE_WEB_ID": state.provider_specific.get("webId", ""),
            "TRAE_BIZ_USER_ID": state.provider_specific.get("bizUserId", ""),
            "TRAE_USER_UNIQUE_ID": state.provider_specific.get("userUniqueId", ""),
            "TRAE_WEB_SCOPE": state.provider_specific.get("scope", ""),
            "TRAE_WEB_TENANT": state.provider_specific.get("tenant", ""),
            "TRAE_WEB_REGION": state.provider_specific.get("region", ""),
            "TRAE_WEB_AI_REGION": state.provider_specific.get("aiRegion", ""),
            "TRAE_WEB_APP_LANGUAGE": state.provider_specific.get("appLanguage", ""),
            "TRAE_WEB_APP_VERSION": state.provider_specific.get("appVersion", ""),
            "TRAE_WEB_USER_REGION": state.provider_specific.get("userRegion", ""),
            "TRAE_WEB_USER_IDENTITY": state.provider_specific.get("userIdentity", ""),
        }
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text("utf-8").splitlines()
        else:
            lines = []

        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            key = ""
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
            if key in values:
                if key not in seen:
                    out.append(f"{key}={values[key]}")
                    seen.add(key)
            else:
                out.append(line)

        for key, value in values.items():
            if key not in seen:
                out.append(f"{key}={value}")
                seen.add(key)

        ENV_PATH.write_text("\n".join(out) + "\n", "utf-8")
    except Exception as e:
        logger.warning("auth: could not save .env snapshot: %s", e)


# ===== Account store & polling =====


def _record_to_state(record: dict) -> AuthState:
    return AuthState(
        edition=record.get('edition', 'cn'),
        source=record.get('source', 'web-login'),
        token=record.get('token', ''),
        refresh_token=record.get('refresh_token') or None,
        user_id=record.get('user_id', ''),
        host=record.get('host', ''),
        client_id=record.get('client_id', ''),
        expired_at=record.get('expired_at', ''),
        refresh_expired_at=record.get('refresh_expired_at', ''),
        provider_specific=dict(record.get('provider_specific') or {}),
    )


def _state_to_record(state: AuthState) -> dict:
    return {
        'user_id': state.user_id,
        'label': state.provider_specific.get('screenName') or state.user_id or '',
        'token': state.token,
        'refresh_token': state.refresh_token or '',
        'expired_at': state.expired_at or '',
        'refresh_expired_at': state.refresh_expired_at or '',
        'host': state.host or '',
        'client_id': state.client_id or '',
        'source': state.source,
        'edition': state.edition,
        'provider_specific': dict(state.provider_specific),
    }


def _merge_state_record(state: AuthState, previous: Optional[dict] = None) -> dict:
    """Refresh auth fields without discarding cached account metadata."""
    old = dict(previous or {})
    record = dict(old)
    record.update(_state_to_record(state))
    record['label'] = old.get('label') or record['label']
    return record


def _record_valid(record: dict) -> bool:
    if not record.get('token'):
        return False
    return _record_to_state(record).is_valid()


def _save_accounts() -> None:
    try:
        ACCOUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'accounts': _accounts,
            'active': _active_account,
            'poll_enabled': _poll_enabled,
            'polling_mode': _polling_mode,
            'rotation_cursor': _rotation_cursor,
            'settings': _settings,
        }
        ACCOUNTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
    except Exception as e:
        logger.warning('auth: could not save account store: %s', e)


def _bootstrap_account_store() -> None:
    global _accounts, _active_account, _poll_enabled, _settings, _aut, _polling_mode, _rotation_cursor
    try:
        if ACCOUNTS_PATH.exists():
            data = json.loads(ACCOUNTS_PATH.read_text('utf-8'))
            _accounts = data.get('accounts', {}) or {}
            _active_account = data.get('active', '') or ''
            _poll_enabled = bool(data.get('poll_enabled', False))
            _settings = data.get('settings', {}) or {}
            _polling_mode = data.get('polling_mode', 'round-robin') or 'round-robin'
            _rotation_cursor = data.get('rotation_cursor', 0) or 0
    except Exception as e:
        logger.warning('auth: could not load account store: %s', e)

    if _auth.source == 'cli':
        return

    current = _auth
    if current.token:
        aid = current.user_id or current.token[:24]
        old = _accounts.get(aid, {})
        _accounts[aid] = _merge_state_record(current, old)
        _active_account = aid
    if current.token:
        # Store the valid env/authorize token in memory even if persistence failed.
        pass
    elif _active_account and _active_account in _accounts:
        # Always load the selected account; treat as usable as long as it has a token.
        _switch_record(_accounts[_active_account])
    else:
        # Prefer a valid account, but fall back to any account with a token so a
        # fresh deploy with an expired-at field still becomes usable.
        for aid, rec in _accounts.items():
            if _record_valid(rec):
                _active_account = aid
                _switch_record(rec)
                break
        else:
            for aid, rec in _accounts.items():
                if rec.get('token'):
                    _active_account = aid
                    _switch_record(rec)
                    break
            else:
                _active_account = next(iter(_accounts), '')
                if _active_account:
                    _switch_record(_accounts[_active_account])
    _save_accounts()


def _persist_active_account() -> None:
    global _active_account
    state = _auth
    if not state.token:
        return
    aid = state.user_id or state.token[:24]
    old = _accounts.get(aid, {})
    _accounts[aid] = _merge_state_record(state, old)
    _active_account = aid
    _save_accounts()


def _switch_record(record: dict) -> None:
    global _auth
    _auth = _record_to_state(record)


def get_accounts_raw() -> list[tuple[str, dict]]:
    """Return (account_id, record) pairs without exposing secrets in the API layer."""
    with _STORE_LOCK:
        return [(aid, dict(rec)) for aid, rec in _accounts.items()]


def get_active_account_id() -> str:
    with _STORE_LOCK:
        return _active_account


def get_active_account_snapshot() -> tuple[str, dict]:
    """Return the selected account id and its credential record atomically.

    Account polling and the web console can switch the global account while a
    request is being prepared.  Callers that need a token plus its owning id
    must read both values under the same store lock; otherwise an id from one
    account can be paired with a token from another account.
    """

    with _STORE_LOCK:
        account_id = _active_account
        return account_id, dict(_accounts.get(account_id) or {})


def set_account_checkin(account_id: str, data: dict) -> None:
    """Replace one account's daily-checkin state and mark it freshly queried."""
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        if not rec:
            return
        rec['checkin'] = dict(data or {})
        now = time.time()
        rec['checkin_status_updated_at'] = now
        rec['checkin_updated_at'] = now
        _save_accounts()


def merge_account_checkin(account_id: str, data: dict) -> dict:
    """Merge daily-checkin state and mark the status snapshot freshly queried."""
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        if not rec:
            return {}
        checkin = dict(rec.get('checkin') or {})
        checkin.update(dict(data or {}))
        rec['checkin'] = checkin
        now = time.time()
        rec['checkin_status_updated_at'] = now
        rec['checkin_updated_at'] = now
        _save_accounts()
        return dict(checkin)


def merge_account_credits(account_id: str, data: dict) -> dict:
    """Merge entitlement credits without refreshing the daily-checkin date."""
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        if not rec:
            return {}
        checkin = dict(rec.get('checkin') or {})
        checkin.update(dict(data or {}))
        rec['checkin'] = checkin
        rec['credits_updated_at'] = time.time()
        _save_accounts()
        return dict(checkin)


def get_account_record(account_id: str) -> dict:
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        return dict(rec) if rec else {}


def list_accounts() -> list[dict]:
    with _STORE_LOCK:
        result = []
        for aid, rec in _accounts.items():
            checkin = rec.get('checkin') or {}
            result.append({
                'id': aid,
                'user_id': rec.get('user_id', aid),
                'label': rec.get('label', ''),
                'source': rec.get('source', ''),
                'has_token': bool(rec.get('token')),
                'is_active': aid == _active_account,
                'is_valid': _record_valid(rec),
                'expires': rec.get('expired_at', ''),
                'credits': checkin.get('credits'),
                'checked_in': checkin.get('checked_in'),
                'checkin_enable': checkin.get('enable'),
                'checkin_updated_at': rec.get(
                    'checkin_status_updated_at', rec.get('checkin_updated_at', 0)
                ),
                'credits_updated_at': rec.get('credits_updated_at', 0),
                'account_credits': checkin.get('account_credits'),
                'work_credits': checkin.get('work_credits'),
                'total_credits': checkin.get('total_credits'),
            })
        return result


def add_account(creds: dict, label: str = '') -> str:
    global _active_account
    token = creds.get('token') or ''
    if not token:
        raise ValueError('token is required')
    user_id = creds.get('user_id') or creds.get('userId') or creds.get('web_id') or ''
    aid = user_id or token[:24]
    psd = dict(creds.get('provider_specific') or {})
    for k, v in {
        'webId': creds.get('web_id') or creds.get('webId'),
        'bizUserId': creds.get('biz_user_id') or creds.get('bizUserId'),
        'userUniqueId': creds.get('user_unique_id') or creds.get('userUniqueId'),
        'scope': creds.get('scope'),
        'tenant': creds.get('tenant'),
        'region': creds.get('region'),
        'aiRegion': creds.get('ai_region') or creds.get('aiRegion'),
        'appLanguage': creds.get('app_language') or creds.get('appLanguage'),
        'userRegion': creds.get('user_region') or creds.get('userRegion'),
        'userIdentity': creds.get('user_identity') or creds.get('userIdentity'),
        'screenName': creds.get('screen_name') or creds.get('screenName'),
    }.items():
        if v:
            psd[k] = v
    record = {
        'user_id': user_id,
        'label': label or creds.get('label') or psd.get('screenName') or user_id or aid,
        'token': token,
        'refresh_token': creds.get('refresh_token') or creds.get('refreshToken') or '',
        'expired_at': creds.get('expired_at') or creds.get('expiredAt') or '',
        'refresh_expired_at': creds.get('refresh_expired_at') or creds.get('refreshExpiredAt') or '',
        'host': creds.get('host') or '',
        'client_id': creds.get('client_id') or creds.get('clientId') or '',
        'source': creds.get('source') or 'web-login',
        'edition': creds.get('edition') or 'cn',
        'provider_specific': psd,
    }
    with _STORE_LOCK:
        _accounts[aid] = record
        _active_account = aid
        _switch_record(record)
        _save_accounts()
    _save_env_snapshot()
    return aid


def remove_account(account_id: str) -> bool:
    global _active_account
    with _STORE_LOCK:
        if account_id not in _accounts:
            return False
        del _accounts[account_id]
        next_id = ''
        if _active_account == account_id:
            _active_account = ''
            for aid, rec in _accounts.items():
                if _record_valid(rec):
                    next_id = aid
                    break
            if next_id:
                _active_account = next_id
                _switch_record(_accounts[next_id])
            else:
                clear_active_auth()
        _save_accounts()
        return True


def switch_account(account_id: str) -> bool:
    global _active_account
    with _STORE_LOCK:
        if account_id not in _accounts:
            return False
        _active_account = account_id
        _switch_record(_accounts[account_id])
        _save_accounts()
    _save_env_snapshot()
    return True


def clear_active_auth() -> None:
    state = _auth
    with state._lock:
        state.token = ''
        state.refresh_token = None
        state.user_id = ''
        state.expired_at = ''
        state.refresh_expired_at = ''
        state.provider_specific.clear()
    _save_env_snapshot()


def logout_active() -> bool:
    global _active_account
    with _STORE_LOCK:
        aid = _active_account
        if aid in _accounts:
            del _accounts[aid]
        _active_account = ''
        next_id = ''
        for account_id, rec in _accounts.items():
            if _record_valid(rec):
                next_id = account_id
                break
        if next_id:
            _active_account = next_id
            _switch_record(_accounts[next_id])
        else:
            clear_active_auth()
        _save_accounts()
    if next_id:
        _save_env_snapshot()
    return True


def set_polling(enabled: bool) -> None:
    global _poll_enabled
    with _STORE_LOCK:
        _poll_enabled = enabled
        _save_accounts()


def get_polling_status() -> dict:
    with _STORE_LOCK:
        return {
            'enabled': _poll_enabled,
            'active_account': _active_account,
            'account_count': len(_accounts),
            'mode': _polling_mode,
        }


def next_polling_account() -> None:
    global _active_account, _rotation_cursor
    if not _poll_enabled:
        return
    with _STORE_LOCK:
        ids = [aid for aid, rec in _accounts.items() if _record_valid(rec)]
        if not ids:
            return

        # Credit-priority: sort by remaining credits descending (more credits = higher priority)
        if _polling_mode == "credit-priority":
            def _credits_sort_key(aid: str) -> tuple:
                rec = _accounts.get(aid, {})
                ac = (rec.get("checkin") or {}).get("account_credits") or {}
                if ac.get("unlimited"):
                    return (-999999999, aid)
                remaining = ac.get("remaining") or 0
                return (-remaining, aid)
            ids.sort(key=_credits_sort_key)

        n = len(ids)
        for i in range(n):
            idx = (_rotation_cursor + i) % n
            aid = ids[idx]
            if aid != _active_account:
                _rotation_cursor = (idx + 1) % n
                _active_account = aid
                _switch_record(_accounts[aid])
                logger.info("auth: polling switched to account %s (mode=%s)", aid, _polling_mode)
                return
        _rotation_cursor = (_rotation_cursor + 1) % n
def get_settings() -> dict:
    with _STORE_LOCK:
        return {
            'web_base_url': _settings.get('web_base_url', ''),
            'relay_port': _settings.get('relay_port', 0),
            'poll_enabled': _poll_enabled,
        }


def set_relay_settings(web_base_url: str = '', port: int = 0) -> None:
    with _STORE_LOCK:
        if web_base_url:
            _settings['web_base_url'] = web_base_url
        if port and port > 0:
            _settings['relay_port'] = port
        _save_accounts()

    try:
        lines = ENV_PATH.read_text('utf-8').splitlines() if ENV_PATH.exists() else []
        out = []
        seen = set()
        for line in lines:
            stripped = line.strip()
            key = ''
            if stripped and not stripped.startswith('#') and '=' in stripped:
                key = stripped.split('=', 1)[0].strip()
            if key in ('TRAE_WEB_BASE_URL', 'RELAY_PORT'):
                if key not in seen:
                    seen.add(key)
                    if key == 'TRAE_WEB_BASE_URL' and web_base_url:
                        out.append(f'TRAE_WEB_BASE_URL={web_base_url}')
                    elif key == 'RELAY_PORT' and port and port > 0:
                        out.append(f'RELAY_PORT={port}')
            else:
                out.append(line)
        if 'TRAE_WEB_BASE_URL' not in seen and web_base_url:
            out.append(f'TRAE_WEB_BASE_URL={web_base_url}')
        if 'RELAY_PORT' not in seen and port and port > 0:
            out.append(f'RELAY_PORT={port}')
        ENV_PATH.write_text('\n'.join(out) + '\n', 'utf-8')
    except Exception as e:
        logger.warning('auth: could not save relay settings: %s', e)



def set_polling_mode(mode: str) -> None:
    """Set the polling rotation strategy."""
    global _polling_mode
    if mode not in ("round-robin", "credit-priority"):
        raise ValueError(f"Invalid polling mode: {mode}")
    with _STORE_LOCK:
        _polling_mode = mode
        _save_accounts()
    logger.info("auth: polling mode set to %s", mode)


def get_polling_mode() -> str:
    with _STORE_LOCK:
        return _polling_mode
