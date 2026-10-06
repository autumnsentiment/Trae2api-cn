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
import re
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
_account_refresh_locks: dict[str, threading.Lock] = {}
_account_refresh_locks_guard = threading.Lock()
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


def _account_refresh_lock(account_id: str) -> threading.Lock:
    """Return the per-account refresh lock, creating it on first use."""
    with _account_refresh_locks_guard:
        lock = _account_refresh_locks.get(account_id)
        if lock is None:
            lock = threading.Lock()
            _account_refresh_locks[account_id] = lock
        return lock


async def refresh_account(account_id: str) -> bool:
    """Exchange a fresh Cloud-IDE-JWT for one stored account and persist it.

    Unlike :func:`refresh_token`, this targets a specific stored account so the
    polling/checkin/credits paths can recover from a server-side token
    invalidation (HTTP 401 or business code 1001) that ``expired_at`` alone
    cannot predict.  Concurrent refreshes for the same account are serialized
    with a non-blocking lock so a burst of failed calls results in one request.
    """
    lock = _account_refresh_lock(account_id)
    if not lock.acquire(blocking=False):
        return False
    try:
        with _STORE_LOCK:
            record = dict(_accounts.get(account_id) or {})
        if not record:
            logger.warning("auth: refresh_account %s not found", account_id)
            return False
        rt = str(record.get("refresh_token") or "")
        if not rt:
            logger.warning("auth: refresh_account %s has no refresh token", account_id)
            return False
        host = str(
            record.get("host")
            or DEFAULT_BASE_URLS.get(
                record.get("edition", "cn"), "https://trae-api-cn.mchost.guru"
            )
        ).rstrip("/")
        client_id = str(record.get("client_id") or "ono9krqynydwx5")
        user_id = str(record.get("user_id") or "")
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
            logger.error("auth: exchange failed for %s: %s", account_id, data)
            return False
        normalized_expire = _normalize_expire(new_expire)
        with _STORE_LOCK:
            current = _accounts.get(account_id)
            if current is None:
                logger.warning("auth: refreshed account %s removed before update", account_id)
                return False
            current_refresh = current.get("refresh_token") or ""
            if current_refresh and current_refresh != rt:
                logger.warning(
                    "auth: discarded stale refresh response for account %s", account_id
                )
                return False
            current = dict(current)
            current.update(
                {
                    "token": new_token,
                    "refresh_token": new_refresh,
                    "expired_at": normalized_expire,
                    "client_id": client_id,
                }
            )
            _accounts[account_id] = current
            if _active_account == account_id:
                _switch_record(current)
            _save_accounts()
        logger.info("auth: token refreshed for account %s", account_id)
        return True
    except Exception as exc:
        logger.error("auth: refresh_account %s error: %s", account_id, exc)
        return False
    finally:
        lock.release()


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


def _looks_like_cloudide_token(token: str) -> bool:
    """Return whether a token has the shape used by Cloud-IDE-JWT.

    The desktop/web login flow can also leave a short opaque OAuth/session
    value in its account store.  That value may carry a far-future expiry but
    cannot authenticate the model gateways.  Keep the generic ``is_valid``
    semantics for callers/tests that only need expiry bookkeeping, while
    using this stricter predicate when choosing a model account.
    """
    value = str(token or '').strip()
    parts = value.split('.')
    return len(value) >= 128 and len(parts) == 3 and all(parts)


def _record_model_valid(record: dict) -> bool:
    """Return whether an account is suitable for model-gateway requests."""
    if not _record_valid(record):
        return False
    source = str(record.get('source') or '').strip().lower()
    if source == 'cli':
        return True
    return _looks_like_cloudide_token(record.get('token') or '')


def _record_model_enabled(record: dict) -> bool:
    """Return whether an account may receive new model-gateway requests.

    ``model_enabled`` is intentionally separate from daily check-in state.
    Older account stores do not contain the field and remain enabled for
    backwards compatibility.  A legacy alias is accepted when operators
    have already written it by hand, but the nested ``checkin.enable`` value
    is never consulted here.
    """
    if not isinstance(record, dict):
        return True
    value = record.get('model_enabled')
    if value is None:
        value = record.get('model_request_enabled')
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off'}
    return bool(value)


def _record_model_eligible(record: dict) -> bool:
    """Return whether an account is valid and enabled for model requests."""
    return _record_model_valid(record) and _record_model_enabled(record)


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
    # An auto-discovered credential is authoritative only when it is both
    # unexpired and shaped like a Cloud-IDE-JWT.  A stale/opaque desktop
    # session must not overwrite a healthy persisted account (the old code
    # always promoted ``current`` and caused every endpoint to use uid-1).
    current_model_valid = bool(current.token) and (
        current.source == 'cli' or (
            current.is_valid() and _looks_like_cloudide_token(current.token)
        )
    )
    if current_model_valid:
        aid = current.user_id or current.token[:24]
        old = _accounts.get(aid, {})
        _accounts[aid] = _merge_state_record(current, old)
        _active_account = aid
    if current.token and not current_model_valid:
        logger.warning(
            'auth: ignoring non-model auto credential during account bootstrap '
            '(source=%s user=%s)',
            current.source,
            current.user_id or 'unknown',
        )
    if current_model_valid:
        # Store the valid env/authorize token in memory even if persistence failed.
        pass
    else:
        # Prefer a model-usable account.  Do not blindly load the persisted
        # active row: it may be an opaque OAuth value or an expired JWT.
        selected = ''
        if _active_account and _active_account in _accounts and _record_model_valid(
            _accounts[_active_account]
        ):
            selected = _active_account
        if not selected:
            for aid, rec in _accounts.items():
                if _record_model_valid(rec):
                    selected = aid
                    break
        if selected:
            _active_account = selected
            _switch_record(_accounts[selected])
        else:
            # Preserve the old fallback only when no model-shaped credential
            # exists at all.  This keeps management/UI startup usable while
            # making the eventual upstream error explicit.
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


def get_model_account_snapshot() -> tuple[str, dict]:
    """Return one enabled, model-valid account for a new model request.

    The UI may keep a disabled account selected for check-in management.  New
    model requests must not silently use that credential, so select the first
    enabled account when the selected account is disabled.  Returning an
    empty snapshot when every stored account is disabled lets the caller
    surface a useful error instead of accidentally falling back to the
    process-global token.
    """
    with _STORE_LOCK:
        if (
            _active_account
            and _active_account in _accounts
            and _record_model_eligible(_accounts[_active_account])
        ):
            return _active_account, dict(_accounts[_active_account])
        for aid, rec in _accounts.items():
            if _record_model_eligible(rec):
                return aid, dict(rec)
        return "", {}


def is_account_model_enabled(account_id: str) -> bool:
    """Read the routing flag without retrieving or replacing credentials."""
    with _STORE_LOCK:
        return _record_model_enabled(_accounts.get(str(account_id or "")) or {})


def set_account_model_enabled(account_id: str, enabled: bool) -> bool:
    """Enable or disable one account for model requests only.

    Check-in data, tokens, active selection, and daily scheduling state are
    deliberately untouched.  The enabled flag is persisted independently so
    disabling an account never prevents its check-in endpoints from running.
    """
    with _STORE_LOCK:
        rec = _accounts.get(str(account_id or ""))
        if rec is None:
            return False
        rec["model_enabled"] = bool(enabled)
        _save_accounts()
        return True


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


def sync_account_label_from_credits(account_id: str, user_name: str) -> str:
    """Fill the account label with the upstream 用户名 when none was chosen.

    A manually assigned label is never overwritten; only empty labels or
    labels that still mirror the raw account/user id get the synced name.
    """
    name = (user_name or '').strip()
    if not name:
        return ''
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        if not rec:
            return ''
        existing = (rec.get('label') or '').strip()
        user_id = (rec.get('user_id') or '').strip()
        if existing and existing not in (account_id, user_id):
            return existing
        rec['label'] = name
        _save_accounts()
        return name


def sync_account_screen_name(
    account_id: str, screen_name: str, upstream_user_id: str = ""
) -> tuple[str, bool]:
    """Sync the upstream ScreenName into the account label.

    The label follows the upstream name only while it is still auto-managed:
    empty, equal to the raw account/user id, or equal to the previously synced
    ``provider_specific.screenName``. A manually edited label is kept. Returns
    ``(effective_label, changed)``.
    """
    name = (screen_name or '').strip()
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        if not rec:
            return '', False
        existing = (rec.get('label') or '').strip()
        user_id = str(rec.get('user_id') or '').strip()
        upstream = str(upstream_user_id or '').strip()
        if not name or (upstream and user_id and upstream != user_id):
            return existing or user_id or account_id, False
        provider = rec.get('provider_specific')
        if not isinstance(provider, dict):
            provider = {}
        previous = str(provider.get('screenName') or '').strip()
        changed = False
        if previous != name:
            provider['screenName'] = name
            rec['provider_specific'] = provider
            changed = True
        if existing != name and (
            not existing or existing in (account_id, user_id, previous)
        ):
            rec['label'] = name
            existing = name
            changed = True
        if changed:
            _save_accounts()
        return existing or user_id or account_id, changed


def merge_account_retry(account_id: str, data: dict) -> dict:
    """Persist 9074 retry bookkeeping without touching checkin timestamps.

    The daily-checkin date decides whether an account is "checked in today",
    so retry state must never bump checkin_status_updated_at/checkin_updated_at.
    """
    with _STORE_LOCK:
        rec = _accounts.get(account_id)
        if not rec:
            return {}
        checkin = dict(rec.get('checkin') or {})
        checkin.update(dict(data or {}))
        rec['checkin'] = checkin
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
                'model_enabled': _record_model_enabled(rec),
                'model_eligible': _record_model_eligible(rec),
                'expires': rec.get('expired_at', ''),
                'credits': checkin.get('credits'),
                'checked_in': checkin.get('checked_in'),
                'checkin_enable': checkin.get('enable'),
                'checkin_updated_at': rec.get(
                    'checkin_status_updated_at', rec.get('checkin_updated_at', 0)
                ),
                'credits_updated_at': rec.get('credits_updated_at', 0),
                'account_credits': checkin.get('account_credits'),
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
        previous = _accounts.get(aid) or {}
        record['model_enabled'] = _record_model_enabled(previous)
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
                if _record_model_eligible(rec):
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
            if _record_model_eligible(rec):
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
        ids = [
            aid for aid, rec in _accounts.items()
            if _record_model_eligible(rec)
        ]
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
            'upstream_mode': _settings.get('upstream_mode', ''),
            # ``web_base_url`` is shared by the IDE Raw/IDE Agent pair and by
            # Remote/Work Agent.  Keep the selected preset id separately so a
            # restart cannot reconstruct the first URL match and silently
            # switch the endpoint tier.
            'endpoint_id': _settings.get('endpoint_id', ''),
            'relay_port': _settings.get('relay_port', 0),
            'poll_enabled': _poll_enabled,
        }


def set_relay_settings(
    web_base_url: str = '',
    port: int = 0,
    upstream_mode: str = '',
    endpoint_id: str | None = None,
) -> None:
    with _STORE_LOCK:
        if web_base_url:
            _settings['web_base_url'] = web_base_url
            # Update the running process so the change takes effect without a restart.
            os.environ['TRAE_WEB_BASE_URL'] = web_base_url
        if upstream_mode:
            _settings['upstream_mode'] = upstream_mode
            os.environ['UPSTREAM_MODE'] = upstream_mode
        # ``None`` means this is a partial update (for example the relay port
        # form) and the existing preset identity must be retained.  An empty
        # string explicitly selects the custom URL entry and clears a stale
        # preset id.
        if endpoint_id is not None:
            _settings['endpoint_id'] = str(endpoint_id).strip()
        if port and port > 0:
            _settings['relay_port'] = port
        _save_accounts()

    # Only rewrite keys explicitly changed by this call.  The previous
    # hand-rolled writer dropped unrelated relay keys whenever the UI
    # submitted an empty mode or port, making the next restart fall back to a
    # different endpoint.
    values = {}
    if web_base_url:
        values['TRAE_WEB_BASE_URL'] = web_base_url
    if upstream_mode:
        values['UPSTREAM_MODE'] = upstream_mode
    if port and port > 0:
        values['RELAY_PORT'] = port
    if values:
        _write_env_values(values)


_MAX_MODE_ENV = 'TRAE_REMOTE_MAX_MODE'
_MAX_MODELS_ENV = 'TRAE_REMOTE_MAX_MODELS'


def _normalize_max_models(models) -> str:
    if isinstance(models, (list, tuple, set)):
        items = list(models)
    else:
        items = str(models or '').replace('\n', ',').split(',')
    out: list[str] = []
    for item in items:
        value = str(item or '').strip()
        if value and value.lower() not in {v.lower() for v in out}:
            out.append(value)
    return ','.join(out)


def _write_env_values(values: dict) -> None:
    """Update or append KEY=value lines in .env without touching other keys."""
    try:
        lines = ENV_PATH.read_text('utf-8').splitlines() if ENV_PATH.exists() else []
        out, seen = [], set()
        for line in lines:
            stripped = line.strip()
            key = ''
            if stripped and not stripped.startswith('#') and '=' in stripped:
                key = stripped.split('=', 1)[0].strip()
            if key in values:
                if key not in seen:
                    seen.add(key)
                    out.append(f'{key}={values[key]}')
                continue
            out.append(line)
        for key, value in values.items():
            if key not in seen:
                out.append(f'{key}={value}')
        ENV_PATH.write_text('\n'.join(out) + '\n', 'utf-8')
    except Exception as e:
        logger.warning('auth: could not save env values: %s', e)


def get_max_mode_settings() -> dict:
    """Return the effective 1M max-mode switch; the console value wins over env."""
    with _STORE_LOCK:
        saved = _settings.get('max_mode')
    if isinstance(saved, dict):
        return {
            'enabled': bool(saved.get('enabled')),
            'models': _normalize_max_models(saved.get('models')),
            'source': 'console',
        }
    enabled = os.environ.get(_MAX_MODE_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')
    return {
        'enabled': enabled,
        'models': _normalize_max_models(os.environ.get(_MAX_MODELS_ENV, '')),
        'source': 'env',
    }


def apply_max_mode_settings() -> None:
    """Push the persisted console switch into the process environment.

    The remote client reads these variables per request, so the change applies
    to the next session without a restart.  Container env_file values are only
    a default: a saved console choice survives ``docker restart``.
    """
    with _STORE_LOCK:
        saved = _settings.get('max_mode')
    if not isinstance(saved, dict):
        return
    os.environ[_MAX_MODE_ENV] = '1' if saved.get('enabled') else '0'
    os.environ[_MAX_MODELS_ENV] = _normalize_max_models(saved.get('models'))


def set_max_mode_settings(enabled: bool, models='') -> dict:
    normalized = _normalize_max_models(models)
    with _STORE_LOCK:
        _settings['max_mode'] = {'enabled': bool(enabled), 'models': normalized}
        _save_accounts()
    apply_max_mode_settings()
    _write_env_values({
        _MAX_MODE_ENV: '1' if enabled else '0',
        _MAX_MODELS_ENV: normalized,
    })
    return get_max_mode_settings()



_AUTO_ROUTE_ENV = 'TRAE_AUTO_ROUTE'


def get_auto_route_settings() -> dict:
    """Return the auto endpoint routing switch; the console value wins over env."""
    with _STORE_LOCK:
        saved = _settings.get('auto_route')
    if isinstance(saved, dict):
        return {'enabled': bool(saved.get('enabled')), 'source': 'console'}
    enabled = os.environ.get(_AUTO_ROUTE_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')
    return {'enabled': enabled, 'source': 'env'}


def set_auto_route_settings(enabled: bool) -> dict:
    with _STORE_LOCK:
        _settings['auto_route'] = {'enabled': bool(enabled)}
        _save_accounts()
    os.environ[_AUTO_ROUTE_ENV] = '1' if enabled else '0'
    _write_env_values({_AUTO_ROUTE_ENV: '1' if enabled else '0'})
    return get_auto_route_settings()


_AUTO_CHECKIN_ENV = 'TRAE_AUTO_CHECKIN'
_AUTO_CHECKIN_TIME_ENV = 'TRAE_AUTO_CHECKIN_TIME'
_AUTO_CHECKIN_DEFAULT_TIME = '08:30'


def normalize_checkin_time(value) -> str:
    """Validate an HH:MM (24h) string; raise ValueError when malformed."""
    text = str(value or '').strip()
    m = re.fullmatch(r'(\d{1,2}):(\d{2})', text)
    if not m:
        raise ValueError(f'Invalid time: {text!r}, expected HH:MM')
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        raise ValueError(f'Invalid time: {text!r}, expected HH:MM')
    return f'{hour:02d}:{minute:02d}'


def get_auto_checkin_settings() -> dict:
    """Return the scheduled check-in switch and daily time (UTC+8)."""
    with _STORE_LOCK:
        saved = _settings.get('auto_checkin')
    if isinstance(saved, dict):
        try:
            time_text = normalize_checkin_time(saved.get('time'))
        except ValueError:
            time_text = _AUTO_CHECKIN_DEFAULT_TIME
        return {'enabled': bool(saved.get('enabled')), 'time': time_text, 'source': 'console'}
    enabled = os.environ.get(_AUTO_CHECKIN_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        time_text = normalize_checkin_time(os.environ.get(_AUTO_CHECKIN_TIME_ENV) or _AUTO_CHECKIN_DEFAULT_TIME)
    except ValueError:
        time_text = _AUTO_CHECKIN_DEFAULT_TIME
    return {'enabled': enabled, 'time': time_text, 'source': 'env'}


def set_auto_checkin_settings(enabled: bool, time_text=None) -> dict:
    current = get_auto_checkin_settings()
    normalized = normalize_checkin_time(time_text) if time_text not in (None, '') else current['time']
    with _STORE_LOCK:
        _settings['auto_checkin'] = {'enabled': bool(enabled), 'time': normalized}
        _save_accounts()
    os.environ[_AUTO_CHECKIN_ENV] = '1' if enabled else '0'
    os.environ[_AUTO_CHECKIN_TIME_ENV] = normalized
    _write_env_values({_AUTO_CHECKIN_ENV: '1' if enabled else '0', _AUTO_CHECKIN_TIME_ENV: normalized})
    return get_auto_checkin_settings()


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
