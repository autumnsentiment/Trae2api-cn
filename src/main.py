"""
main.py - Trae CN Relay 中转站
提供 OpenAI 兼容的 REST API:  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/chat
  POST /v1

上游模式:  UPSTREAM_MODE=cli  - 只使用本地 Trae CLI 子进程
  UPSTREAM_MODE=auto - 优先 CLI，其次 web remote，最后 ide chat
  UPSTREAM_MODE=web  - 只用 OmniRoute 风格网页版 remote 会话
  UPSTREAM_MODE=ide  - 只用 trae2api 风格 /api/ide/v1/chat
"""

import asyncio
import html as html_mod
import json
import logging
import os
import secrets
import uuid as uuid_mod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import dotenv
import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.responses import FileResponse

from . import auth, cli_client, trae_client
from .sse import (
    collect_nonstream_cli,
    collect_nonstream_ide,
    collect_nonstream_web,
    translate_cli_stream,
    translate_ide_stream,
    translate_web_events,
)

dotenv.load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("trae-cn-relay")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
API_KEYS = [k.strip() for k in os.environ.get("RELAY_API_KEYS", "").split(",") if k.strip()]
UPSTREAM_MODE = (os.environ.get("UPSTREAM_MODE", "auto") or "auto").lower()
FORWARD_USAGE = (os.environ.get("FORWARD_USAGE", "true") or "true").lower() == "true"
WEB_BASE = (os.environ.get("TRAE_WEB_BASE_URL", "https://trae-api-cn.mchost.guru/api/remote/v1")).rstrip("/")

# Web login auth
TRAE_AUTH_URL = os.environ.get("TRAE_AUTH_URL", "https://www.trae.cn/authorization")
TRAE_CLIENT_ID = os.environ.get("TRAE_CLIENT_ID") or "ono9krqynydwx5"
LOCAL_LISTENER_PORT = int(os.environ.get("WEB_LOGIN_LISTENER_PORT", "8765"))
PUBLIC_PATHS = {"/healthz", "/v1/status", "/web/login", "/authorize", "/api/web-auth"}
PUBLIC_PATHS = {"/healthz", "/v1/status", "/v1/models", "/web/login", "/web/login/download", "/authorize", "/api/web-auth", "/api/logout", "/api/accounts", "/api/accounts/switch", "/api/accounts/remove", "/api/settings", "/api/polling"}
WEB_LOGIN_SCRIPT = Path(__file__).resolve().parent.parent / "web_login.py"


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
      {f'<span class="user-id">用户: {html_mod.escape(state.user_id)}</span>' if state.user_id else ''}
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

    # 账号列表
    rows = ""
    for acc in accounts:
        if acc.get("is_valid"):
            st = '<span class="badge badge-ok">有效</span>'
        else:
            st = '<span class="badge badge-expired">无效</span>'
        act = '<span class="badge badge-active">当前</span>' if acc.get("is_active") else ""
        aid = acc.get("id") or ""
        label = acc.get("label") or acc.get("user_id") or aid
        uid = acc.get("user_id") or aid
        expires = (acc.get("expires") or "")[:16]
        rows += f"""<tr>
          <td>{html_mod.escape(label)}</td>
          <td><code>{html_mod.escape(uid)}</code></td>
          <td>{st} {act}</td>
          <td style="font-size:12px;color:#9aa0b0">{html_mod.escape(expires)}</td>
          <td>
            <button class="btn btn-ghost btn-sm" onclick="switchAccount('{html_mod.escape(aid)}')">切换</button>
            <button class="btn btn-ghost btn-sm" onclick="removeAccount('{html_mod.escape(aid)}')">删除</button>
          </td>
        </tr>"""
    if accounts:
        accounts_html = f"""<div class="form-group">
          <label>账号列表（{len(accounts)}）</label>
          <table class="acct-table">
            <thead><tr><th>标签</th><th>用户ID</th><th>状态</th><th>有效期</th><th>操作</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""
    else:
        accounts_html = '<p style="font-size:13px;color:#9aa0b0;margin-bottom:8px">暂无账号，请先登录或手动添加。</p>'

    logout_btn = ''
    if state.token:
        logout_btn = '<button class="btn btn-ghost btn-sm" onclick="logout()">登出</button>'

    settings_web = settings.get("web_base_url") or WEB_BASE
    settings_port = settings.get("relay_port") or PORT
    poll_checked = 'checked' if polling.get("enabled") else ''

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
  padding: 20px;
}}
.panel {{
  background: #1a1d28; border-radius: 8px; max-width: 680px; width: 100%;
  padding: 28px 24px; border: 1px solid #2d3140;
}}
h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
.status-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; font-size: 13px; }}
.status-row .label {{ color: #9aa0b0; }}
.status-row .separator {{ margin-left: 8px; }}
.status-row code {{ background: #252836; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
.badge {{ font-size: 12px; padding: 2px 10px; border-radius: 10px; font-weight: 500; }}
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
  text-decoration: none; color: #fff;
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
.btn-group {{ display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }}
.form-group {{ margin-bottom: 12px; }}
.form-group label {{ display: block; font-size: 12px; color: #9aa0b0; margin-bottom: 4px; }}
.form-group input, .form-group textarea {{
  width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid #3a3f54;
  background: #252836; color: #e8eaed; font-size: 13px; font-family: "SF Mono", Consolas, monospace;
}}
.form-group textarea {{ resize: vertical; min-height: 60px; }}
.form-group input:focus, .form-group textarea:focus {{ outline: none; border-color: #1a8c5c; }}
.acct-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.acct-table th, .acct-table td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #2d3140; vertical-align: middle; }}
.acct-table th {{ color: #9aa0b0; font-weight: 500; font-size: 12px; }}
.msg {{ margin-top: 12px; padding: 8px 12px; border-radius: 6px; font-size: 13px; display: none; }}
.msg-ok {{ background: #1f6c3a; color: #a8e6b8; display: block; }}
.msg-err {{ background: #6c1f1f; color: #e6a8a8; display: block; }}
.loading {{ margin-top: 12px; display: none; font-size: 13px; color: #9aa0b0; }}
.section-title {{ font-size: 14px; font-weight: 600; color: #c8cbd6; margin: 14px 0 8px; }}
.check-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #9aa0b0; }}
</style>
</head>
<body>
<div class="panel">
<h1>Trae CN Relay 控制台</h1>
<div class="status-row" style="justify-content:space-between;align-items:center">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">{status_html}</div>
  {logout_btn}
</div>
<hr>
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
<hr>
<div class="section-title">多账号轮询</div>
<div class="check-row">
  <input type="checkbox" id="poll-toggle" {poll_checked} onchange="togglePolling()">
  <label for="poll-toggle" style="cursor:pointer">启用轮询（每次请求自动切换下一个有效账号）</label>
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
<script>
const state = {{ traceId: null, win: null }};
let currentCodeVerifier = '';
function uuid() {{ return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,function(c){{var r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16)}}) }}
function randomHex(n) {{ var a=new Uint8Array(n);crypto.getRandomValues(a);return Array.from(a,b=>b.toString(16).padStart(2,'0')).join('') }}
function randomDigits(n) {{ var s='';while(s.length<n)s+=Math.floor(Math.random()*1e10).toString();return s.slice(0,n) }}
function randomBase64Url(n) {{ var a=new Uint8Array(n); if(window.crypto&&crypto.getRandomValues){{ crypto.getRandomValues(a) }} else {{ for(var i=0;i<n;i++)a[i]=Math.floor(Math.random()*256) }} return btoa(String.fromCharCode.apply(null,a)).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'') }}
function _sha256Bytes(ascii) {{ var K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]; var i; var h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a,h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19; ascii=unescape(encodeURIComponent(ascii)); var msg=[]; for(i=0;i<ascii.length;i++)msg.push(ascii.charCodeAt(i)); msg.push(0x80); while(msg.length%64!==56)msg.push(0); var bitlen=ascii.length*8; var hi=Math.floor(bitlen/4294967296), lo=bitlen>>>0; for(i=3;i>=0;i--)msg.push((hi>>>(i*8))&255); for(i=3;i>=0;i--)msg.push((lo>>>(i*8))&255); for(var o=0;o<msg.length;o+=64) {{ var w=new Array(64); for(i=0;i<16;i++)w[i]=(msg[o+i*4]<<24)|(msg[o+i*4+1]<<16)|(msg[o+i*4+2]<<8)|(msg[o+i*4+3]); for(i=16;i<64;i++) {{ var s0=((w[i-15]>>>7)|(w[i-15]<<25))^((w[i-15]>>>18)|(w[i-15]<<14))^(w[i-15]>>>3); var s1=((w[i-2]>>>17)|(w[i-2]<<15))^((w[i-2]>>>19)|(w[i-2]<<13))^(w[i-2]>>>10); w[i]=(w[i-16]+s0+w[i-7]+s1)>>>0 }} var a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7; for(i=0;i<64;i++) {{ var S1=((e>>>6)|(e<<26))^((e>>>11)|(e<<21))^((e>>>25)|(e<<7)); var ch=(e&f)^((~e)&g); var t1=(h+S1+ch+K[i]+w[i])>>>0; var S0=((a>>>2)|(a<<30))^((a>>>13)|(a<<19))^((a>>>22)|(a<<10)); var maj=(a&b)^(a&c)^(b&c); var t2=(S0+maj)>>>0; h=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0 }} h0=(h0+a)>>>0; h1=(h1+b)>>>0; h2=(h2+c)>>>0; h3=(h3+d)>>>0; h4=(h4+e)>>>0; h5=(h5+f)>>>0; h6=(h6+g)>>>0; h7=(h7+h)>>>0 }} return [h0,h1,h2,h3,h4,h5,h6,h7] }}
function sha256Hex(str) {{ var hb=_sha256Bytes(str); var out=''; for(var i=0;i<8;i++) {{ var v=hb[i]; for(var b=3;b>=0;b--) {{ var by=(v>>>(b*8))&255; out+=((by<16)?'0':'')+by.toString(16) }} }} return out }}
function sha256Base64Url(str) {{ var hb=_sha256Bytes(str); var bytes=new Uint8Array(32); for(var i=0;i<8;i++) {{ bytes[i*4]=(hb[i]>>>24)&255; bytes[i*4+1]=(hb[i]>>>16)&255; bytes[i*4+2]=(hb[i]>>>8)&255; bytes[i*4+3]=hb[i]&255 }} var s=''; for(var i=0;i<32;i++)s+=String.fromCharCode(bytes[i]); return btoa(s).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'') }}

function buildChallenge() {{ currentCodeVerifier=randomBase64Url(32); return sha256Base64Url(currentCodeVerifier) }}
async function buildAuthUrl() {{
  // Trae 授权页强制要求回调为 http://127.0.0.1:<port>/authorize，
  // 因此必须由本机 web_login.py 监听并转发凭据到服务器。
  var cb = 'http://127.0.0.1:{listener_port}/authorize';
  var challenge = await buildChallenge();
  var mid = randomHex(32), did = randomDigits(19), tid = state.traceId;
  var p = new URLSearchParams({{
    login_version:'1',auth_from:'solo',login_channel:'native_ide',plugin_version:'2.3.24254',
    auth_type:'local',client_id:'{html_mod.escape(client_id)}',redirect:'0',login_trace_id:tid,
    auth_callback_url:cb,machine_id:mid,device_id:did,x_device_id:did,x_machine_id:mid,
    x_device_brand:'ASUS TUF Gaming A15 FA507RM_FA507RM',x_device_type:'windows',x_os_version:'Windows 10 Pro',x_env:'',
    x_app_version:'3.3.65',x_app_type:'stable',hide_saas_login:'true',
    code_challenge:challenge,code_challenge_method:'S256'
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

window.addEventListener('message',function(ev){{
  if (!ev.data||ev.data.type!=='trae-relay-web-login') return;
  if (state.traceId&&ev.data.loginTraceId!==state.traceId) return;
  if (ev.data.success) {{ showMsg('auth-msg','授权成功，凭证已写入服务器',true); setTimeout(function(){{ location.reload(); }},800); }}
  else {{ showMsg('auth-msg',ev.data.error||'授权失败',false); }}
  document.getElementById('loading').style.display='none';
  document.getElementById('auth-btn').disabled=false;
  if (state.win&&!state.win.closed) state.win.close();
}});
function showMsg(id,text,ok){{ var el=document.getElementById(id);el.textContent=text;el.className='msg'+(ok?' msg-ok':' msg-err'); }}
async function postJSON(url,payload){{
  try{{
    var r=await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload||{{}})}});
    var d=await r.json();
    return d;
  }}catch(e){{ return {{success:false,error:String(e)}}; }}
}}
async function refreshModels(){{
  var el=document.getElementById('models-out');
  var msg=document.getElementById('models-msg');
  el.style.display='block';
  el.textContent='加载中...';
  try{{
    var r=await fetch('/v1/models?refresh=true');
    var d=await r.json();
    if(!r.ok || !d || !Array.isArray(d.data)){{
      throw new Error((d && d.error && d.error.message) || ('HTTP '+r.status));
    }}
    el.textContent=JSON.stringify(d.data.map(m=>m.id),null,2);
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
  var d=await postJSON('/api/accounts/switch',{{account_id:id}});
  if(d.success) location.reload(); else showMsg('auth-msg',d.error||'切换失败',false);
}}
async function removeAccount(id){{
  if(!confirm('确定删除该账号？')) return;
  var d=await postJSON('/api/accounts/remove',{{account_id:id}});
  if(d.success) location.reload(); else showMsg('auth-msg',d.error||'删除失败',false);
}}
async function togglePolling(){{
  var on=document.getElementById('poll-toggle').checked;
  var d=await postJSON('/api/polling',{{enabled:on}});
  if(d.success) location.reload();
  else showMsg('auth-msg',d.error||'设置失败',false);
}}
async function saveSettings(){{
  var web=document.getElementById('settings-web').value.trim();
  var port=document.getElementById('settings-port').value.trim();
  var d=await postJSON('/api/settings',{{web_base_url:web,relay_port:port}});
  if(d.success){{ showMsg('settings-msg',d.note||'设置已保存',true); setTimeout(function(){{ location.reload(); }},800); }}
  else showMsg('settings-msg',d.error||'保存失败',false);
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
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


async def run_cli_chat(messages, model, stream: bool):
    """本地 Trae CLI 子进程上游。"""
    event_iter = cli_client.stream_cli_chat(messages, model)
    if not stream:
        result = await collect_nonstream_cli(event_iter, model)
        return JSONResponse(content=result)

    first, rest = await _peek_async(event_iter)
    if first is None:
        async def empty_gen():
            async for chunk in translate_cli_stream(_empty_cli_events(), model, FORWARD_USAGE):
                yield chunk
        return StreamingResponse(empty_gen(), media_type="text/event-stream", headers=_sse_headers())
    if first.type == "error":
        raise RuntimeError(first.error or "Trae CLI failed before output")

    async def gen():
        async def chain():
            yield first
            async for item in rest:
                yield item
        async for chunk in translate_cli_stream(chain(), model, FORWARD_USAGE):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_sse_headers())


async def run_web_session(messages, model, stream: bool):
    """OmniRoute 风格网页版 remote 会话，带账号并发槽和空闲回收。"""
    account_id = auth.get_user_id() or auth.get_token()[:16] or "default"
    await trae_client.acquire_web_slot(account_id, timeout=float(os.environ.get("TRAE_WEB_SLOT_TIMEOUT", "60")))
    client = httpx.AsyncClient(timeout=60)
    session_id = ""
    try:
        session_id, message_id = await trae_client.create_web_session(client, model, trae_client.flatten_query(messages))
        trae_client.register_web_lease(account_id, session_id, message_id, client)
        event_iter = trae_client.stream_web_events(client, session_id, message_id)
        if stream:
            async def gen():
                try:
                    async for chunk in translate_web_events(event_iter, model, FORWARD_USAGE):
                        yield chunk
                finally:
                    # Actively interrupt the upstream session so it stops
                    # occupying a running slot, then close local resources.
                    await trae_client.stop_web_session(client, session_id, message_id)
                    await client.aclose()
                    if trae_client.unregister_web_lease(session_id):
                        trae_client.release_web_slot(account_id)
            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        try:
            result = await collect_nonstream_web(event_iter, model)
            return JSONResponse(content=result)
        finally:
            await trae_client.stop_web_session(client, session_id, message_id)
            await client.aclose()
            if trae_client.unregister_web_lease(session_id):
                trae_client.release_web_slot(account_id)
    except Exception:
        if session_id:
            try:
                await trae_client.stop_web_session(client, session_id, message_id)
            except Exception:
                pass
        await client.aclose()
        if session_id:
            if trae_client.unregister_web_lease(session_id):
                trae_client.release_web_slot(account_id)
        else:
            trae_client.release_web_slot(account_id)
        raise


async def run_ide_chat(messages, model, stream: bool):
    """trae2api 风格 IDE chat，流式响应消费完成后关闭 response 和 client。"""
    ide_resp = await trae_client.send_chat_request(messages, model, stream)
    response = ide_resp.response
    if stream:
        async def gen():
            try:
                async for chunk in translate_ide_stream(response, model, FORWARD_USAGE):
                    yield chunk
            finally:
                ide_resp.close()
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )
    try:
        result = await collect_nonstream_ide(response, model)
        return JSONResponse(content=result)
    finally:
        ide_resp.close()


def _openai_error(status: int, message: str, error_type: str, param: Optional[str] = None) -> JSONResponse:
    body = {"error": {"message": message, "type": error_type}}
    if param:
        body["error"]["param"] = param
    return JSONResponse(body, status_code=status)


async def _run_web_with_retry(messages, model, stream: bool):
    """web 上游 429 并发限制时轮询切换账号重试。"""
    if auth.get_polling_status().get("enabled"):
        attempts = max(1, len(auth.list_accounts()))
        for _ in range(attempts + 1):
            try:
                return await run_web_session(messages, model, stream)
            except RuntimeError as e:
                err = str(e)
                if "solo_agent_parallel_limit" in err or "429" in err:
                    logger.warning("web 429 parallel limit, rotating account: %s", err)
                    auth.next_polling_account()
                    continue
                raise
        raise RuntimeError("All web accounts busy: Trae parallel limit reached")
    return await run_web_session(messages, model, stream)


async def handle_chat(req: Request):
    # 多账号轮询：web/auto 模式下每个请求前切换到下一个合法账号
    if UPSTREAM_MODE in ("web", "auto"):
        auth.next_polling_account()
    try:
        body = await req.json()
    except Exception:
        return _openai_error(400, "Invalid JSON body", "invalid_request_error")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _openai_error(400, "messages is required", "invalid_request_error")

    model = body.get("model") or "auto"
    stream = bool(body.get("stream", False))

    if not trae_client.is_model_supported(model):
        return _openai_error(400, f"Unsupported model: {model}", "invalid_request_error", "model")

    # 路由选择：cli 优先，web/ide 按配置
    errors = []
    modes = []
    if UPSTREAM_MODE == "cli":
        modes = ["cli"]
    elif UPSTREAM_MODE == "web":
        modes = ["web"]
    elif UPSTREAM_MODE == "ide":
        modes = ["ide"]
    else:
        modes = ["cli", "web", "ide"]

    for mode in modes:
        try:
            if mode == "cli":
                return await run_cli_chat(messages, model, stream)
            if mode == "web":
                return await _run_web_with_retry(messages, model, stream)
            return await run_ide_chat(messages, model, stream)
        except Exception as e:
            logger.warning("upstream %s failed: %s", mode, e)
            errors.append(f"{mode}: {e}")

    return _openai_error(502, "All upstream paths failed: " + "; ".join(errors), "api_error")


async def init_app():
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_app()
    reaper = asyncio.create_task(_web_reaper_loop())
    try:
        yield
    finally:
        reaper.cancel()
        try:
            await reaper
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Trae CN Relay", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not API_KEYS:
        return await call_next(request)
    if request.url.path in PUBLIC_PATHS:
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
        "has_token": bool(state.token),
        "token_ok": state.is_valid(),
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
    return JSONResponse({"success": True})


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


@app.post("/api/polling")
async def api_polling(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)
    enabled = bool(body.get("enabled", False))
    auth.set_polling(enabled)
    return JSONResponse({"success": True, "enabled": enabled})
