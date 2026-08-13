#!/usr/bin/env python3
"""web_login.py - Trae CN Relay 本地授权回调监听器

使用方式:
  python web_login.py [--relay http://192.168.5.246:8000] [--port 8765]

依赖: 仅用 Python 标准库。
"""

import argparse
import http.server
import json
import sys
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Trae CN Relay 本地授权回调监听器")
    parser.add_argument("--relay", default="http://192.168.5.246:8000", help="Relay 地址")
    parser.add_argument("--port", type=int, default=8765, help="本地监听端口")
    parser.add_argument("--client-id", default="ono9krqynydwx5", help="Trae Client ID")
    parser.add_argument("--auth-url", default="https://www.trae.cn/authorization", help="Trae 授权 URL")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    return parser.parse_args()


def json_loads_safe(value: str) -> dict:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_oauth_params(query: dict) -> dict:
    user_jwt = json_loads_safe(query.get("userJwt", ""))
    token = user_jwt.get("Token") or user_jwt.get("token") or ""
    if not token:
        return {}
    refresh = user_jwt.get("RefreshToken") or user_jwt.get("refreshToken") or query.get("refreshToken") or ""
    token_exp = user_jwt.get("TokenExpireAt") or user_jwt.get("tokenExpireAt") or ""
    refresh_exp = user_jwt.get("RefreshExpireAt") or user_jwt.get("refreshExpireAt") or query.get("refreshExpireAt") or ""
    user_info = json_loads_safe(query.get("userInfo", ""))
    user_id = user_info.get("UserID") or user_info.get("userId") or user_info.get("userID") or ""
    region = user_info.get("Region") or user_info.get("region") or "CN"
    ai_region = user_info.get("AIRegion") or user_info.get("aiRegion") or region
    client_id = user_jwt.get("ClientID") or user_jwt.get("clientId") or ""
    uid = user_id or ""
    return {
        "token": token,
        "refreshToken": refresh,
        "userId": uid,
        "tenantId": user_info.get("TenantID") or user_info.get("tenantId") or "",
        "region": region,
        "aiRegion": ai_region,
        "host": query.get("host", ""),
        "expiredAt": str(token_exp) if token_exp else "",
        "refreshExpiredAt": str(refresh_exp) if refresh_exp else "",
        "clientId": client_id,
        "webId": user_info.get("WebId") or user_info.get("webId") or uid,
        "bizUserId": user_info.get("BizUserId") or user_info.get("bizUserId") or uid,
        "userUniqueId": user_info.get("UserUniqueId") or user_info.get("userUniqueId") or uid,
        "scope": query.get("scope") or user_info.get("Scope") or user_info.get("scope") or "",
        "tenant": user_info.get("Tenant") or user_info.get("tenant") or "",
        "appLanguage": user_info.get("AppLanguage") or user_info.get("appLanguage") or "",
        "userRegion": query.get("userRegion") or user_info.get("UserRegion") or user_info.get("userRegion") or "",
        "userIdentity": user_info.get("UserIdentity") or user_info.get("userIdentity") or "",
        "screenName": user_info.get("ScreenName") or user_info.get("screenName") or "",
    }


def forward_to_relay(relay_url: str, creds: dict) -> tuple[bool, str]:
    """POST 凭证到 relay /api/web-auth，返回 (success, message)"""
    url = relay_url.rstrip("/") + "/api/web-auth"
    data = json.dumps(creds).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            if result.get("success"):
                return True, "凭证已写入服务器"
            else:
                return False, result.get("error", "写入失败")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
            msg = json.loads(body).get("error", str(e))
        except Exception:
            msg = str(e)
        return False, msg
    except Exception as e:
        return False, str(e)


class RelayAuthHandler(http.server.BaseHTTPRequestHandler):
    relay_url = "http://192.168.5.246:8000"
    client_id = "ono9krqynydwx5"
    auth_url = "https://www.trae.cn/authorization"

    def log_message(self, fmt, *args):
        pass  # 不输出请求日志

    def _html(self, content: str) -> str:
        return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Trae CN Relay 授权</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body {{font:15px -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;background:#0f1117;color:#e8eaed;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}}
.panel {{background:#1a1d28;border-radius:8px;max-width:460px;width:100%;padding:28px 24px;border:1px solid #2d3140}}
h1 {{font-size:18px;font-weight:600;margin-bottom:12px}}
p {{color:#9aa0b0;font-size:13px;line-height:1.5;margin-bottom:12px}}
.btn {{display:inline-flex;align-items:center;justify-content:center;padding:10px 20px;border-radius:6px;font-size:14px;font-weight:500;cursor:pointer;border:1px solid transparent;text-decoration:none;color:#fff}}
.btn-primary {{background:#1a8c5c;border-color:#1a8c5c}}
.btn-primary:hover {{background:#14a06a}}
.btn-primary:disabled {{opacity:.5;cursor:not-allowed}}
.btn-ghost {{background:transparent;border-color:#3a3f54;color:#9aa0b0}}
.btn-ghost:hover {{border-color:#5a5f74;color:#e8eaed}}
.btn-group {{display:flex;gap:8px;margin-top:12px}}
.msg {{margin-top:12px;padding:8px 12px;border-radius:6px;font-size:13px;display:none}}
.msg-ok {{background:#1f6c3a;color:#a8e6b8;display:block}}
.msg-err {{background:#6c1f1f;color:#e6a8a8;display:block}}
.loading {{margin-top:12px;display:none;font-size:13px;color:#9aa0b0}}
code {{background:#252836;padding:1px 6px;border-radius:4px;font-size:12px}}
</style></head>
<body><div class="panel">{content}</div></body></html>"""

    def _index_page(self) -> str:
        port = self.server.server_address[1]
        relay = self.relay_url
        return self._html(f"""
<h1>Trae CN Relay 授权</h1>
<p>
  监听端口: <code>{port}</code><br>
  中转站: <code>{relay}</code>
</p>
<p>
  1. 确保浏览器已登录 <a href="https://www.trae.cn" target="_blank" rel="noopener" style="color:#8ab4f8">trae.cn</a><br>
  2. 点击下方按钮
</p>
<div class="btn-group">
  <button class="btn btn-primary" onclick="startAuth()" id="auth-btn">使用 Trae 网页授权登录</button>
  <a class="btn btn-ghost" href="https://www.trae.cn" target="_blank" rel="noopener">打开 trae.cn</a>
</div>
<div id="loading" class="loading">等待授权中…</div>
<div id="auth-msg" class="msg"></div>
<script>
var state = {{ traceId: null, win: null }};
function uuid(){{return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,function(c){{var r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16)}})}}
function randomHex(n){{var a=new Uint8Array(n);crypto.getRandomValues(a);return Array.from(a,b=>b.toString(16).padStart(2,'0')).join('')}}
function randomDigits(n){{var s='';while(s.length<n)s+=Math.floor(Math.random()*1e10).toString();return s.slice(0,n)}}
function buildAuthUrl(){{
  var cb = 'http://127.0.0.1:{port}/authorize';
  var mid = randomHex(32), did = randomDigits(19), tid = state.traceId;
  var p = new URLSearchParams({{
    login_version:'1',auth_from:'solo',login_channel:'native_ide',plugin_version:'2.3.24254',
    auth_type:'local',client_id:'{self.client_id}',redirect:'0',login_trace_id:tid,
    auth_callback_url:cb,machine_id:mid,device_id:did,x_device_id:did,x_machine_id:mid,
    x_device_brand:'Mac14,7',x_device_type:'mac',x_os_version:'macOS 26.4.1',x_env:'',
    x_app_version:'0.1.7',x_app_type:'stable',hide_saas_login:'true'
  }});
  return '{self.auth_url}?'+p.toString();
}}
function startAuth(){{
  state.traceId = uuid();
  var url = buildAuthUrl();
  var w = window.open(url, 'trae-relay-oauth', 'width=560,height=760');
  if (!w){{ showMsg('auth-msg','弹出窗口被拦截',false); return; }}
  state.win = w;
  document.getElementById('loading').style.display='block';
  document.getElementById('auth-btn').disabled=true;
  var poll = setInterval(function(){{ if(w.closed){{ clearInterval(poll);document.getElementById('loading').style.display='none';document.getElementById('auth-btn').disabled=false; }} }},700);
}}
window.addEventListener('message',function(ev){{
  if (!ev.data||ev.data.type!=='trae-relay-web-login') return;
  if (state.traceId&&ev.data.loginTraceId!==state.traceId) return;
  if (ev.data.success){{ showMsg('auth-msg','授权成功',true); }}
  else{{ showMsg('auth-msg',ev.data.error||'授权失败',false); }}
  document.getElementById('loading').style.display='none';
  document.getElementById('auth-btn').disabled=false;
  if (state.win&&!state.win.closed) state.win.close();
}});
function showMsg(id,text,ok){{ var el=document.getElementById(id);el.textContent=text;el.className='msg'+(ok?' msg-ok':' msg-err'); }}
</script>""")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if parsed.path == "/healthz":
            self._send_json({"status": "ok", "relay": self.relay_url, "port": self.server.server_address[1]})
        elif parsed.path == "/relay-url":
            self._send_json({"relay": self.relay_url})
        elif parsed.path == "/authorize":
            self._handle_authorize(params)
        elif parsed.path in ("/", "/index.html", ""):
            self._send_html(self._index_page())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def _send_json(self, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _handle_authorize(self, query: dict):
        creds = parse_oauth_params(query)
        trace_id = query.get("loginTraceID") or query.get("login_trace_id") or ""
        if not creds.get("token"):
            html = self._oauth_result_page(False, "未收到有效的 userJwt，请确认已登录 trae.cn", trace_id)
            self._send_html(html)
            return

        # 转发到 relay
        success, msg = forward_to_relay(self.relay_url, creds)
        html = self._oauth_result_page(success, msg, trace_id)
        self._send_html(html)

    def _oauth_result_page(self, success: bool, message: str, login_trace_id: str = "") -> str:
        safe_msg = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        safe_msg = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
        safe_trace = login_trace_id.replace("&", "&amp;").replace("<", "&lt;")
        return self._html(f"""
<div class="msg {'msg-ok' if success else 'msg-err'}">
  <h2 style="margin:0 0 8px;font-size:16px">{"成功" if success else "失败"}</h2>
  <p>{safe_msg}</p>
</div>
<script>
(function(){{
  try {{
    if (!window.opener) return;
    var msg = {{ type:'trae-relay-web-login', success:{"true" if success else "false"}, error:null, loginTraceId:'{safe_trace}' }};
    if (!{str(success).lower()}) msg.error = '{safe_msg}';
    window.opener.postMessage(msg, '*');
  }} catch(e){{}}
  setTimeout(function(){{ window.close(); }}, {800 if success else 4000});
}})();
</script>
""")

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    args = parse_args()
    relay = args.relay.rstrip("/")
    port = args.port
    client_id = args.client_id
    auth_url = args.auth_url

    handler = RelayAuthHandler
    handler.relay_url = relay
    handler.client_id = client_id
    handler.auth_url = auth_url

    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    print(f"Trae CN Relay 授权回调监听器已启动")
    print(f"  本地监听: http://127.0.0.1:{port}")
    print(f"  中转站:   {relay}")
    print(f"  授权 URL: {auth_url}")
    print(f"  ClientID: {client_id}")
    print()
    print(f"请在浏览器中打开 http://127.0.0.1:{port}")
    print(f"或直接在中转站页面点击授权按钮")
    print()

    if not args.no_open:
        try:
            # Trae 授权完成后回调到本机 /authorize；打开 relay 控制台方便用户点授权。
            webbrowser.open(f"{relay}/web/login")
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
