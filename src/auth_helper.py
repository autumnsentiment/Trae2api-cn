"""auth_helper.py - 授权助手脚本按部署环境自动生成

relay 控制台的“授权助手”下载按钮不再返回仓库静态文件，而是由本模块按当前部署
环境实时生成脚本：中转站地址与监听端口直接写进文件，用户下载后双击或
``python web_login.py`` 即可运行，不需要手动填 IP / 端口，也不需要去 GitHub
Releases 下载后再编辑。

生成内容只包含部署地址、监听端口与公开的 client_id，不包含任何账号凭据。
写入的地址只允许 ``scheme://host[:port][/子路径]``；引号、反斜杠、``$``、``%``、
``&``、``;``、空格等在 Python 字符串字面量、batch 变量和 shell 引号里都有特殊
含义，一律判定为非法，因此调用方传入什么都不会改变生成脚本的语义。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping, Optional

from web_login import normalize_relay_url as _normalize_helper_url

KIND_PY = "py"
KIND_BAT = "bat"
KIND_SH = "sh"
SUPPORTED_KINDS = (KIND_PY, KIND_BAT, KIND_SH)

DEFAULT_RELAY_URL = "http://127.0.0.1:8000"
DEFAULT_LISTENER_PORT = 8765
DEFAULT_CLIENT_ID = "ono9krqynydwx5"
DEFAULT_AUTH_URL = "https://www.trae.cn/authorization"

MAX_RELAY_URL_LENGTH = 200

# 模板里可被改写的占位符：仓库通用版使用占位，下载版使用常量。
RELAY_BASE_SENTINEL = "__TRAE_RELAY_BASE_URL__"
RELAY_DOC_PLACEHOLDER = "http://<relay-host>:<port>"
DEFAULTS_MARKER = "# ---- 部署默认值"

class RelayURLError(ValueError):
    """中转站地址不可用或存在注入风险。"""


def normalize_relay_url(raw: object) -> str:
    """校验并归一化中转站地址，返回不带结尾斜杠的 http(s) 地址。

    This value is copied into Python, batch and shell source.  Parse it as a
    URL first and reject characters with special meaning in any of those
    languages.  Paths are supported for reverse proxies mounted below a
    prefix; query strings and credentials are intentionally not supported.
    """
    value = str(raw or "").strip()
    if not value:
        raise RelayURLError("relay url is empty")
    if len(value) > MAX_RELAY_URL_LENGTH:
        raise RelayURLError("relay url is too long")
    if not value.lower().startswith(("http://", "https://")):
        raise RelayURLError("relay url must start with http:// or https://")
    normalized = _normalize_helper_url(value)
    if not normalized:
        raise RelayURLError("relay url has an invalid host, port, path or unsafe characters")
    return normalized


def is_safe_relay_url(value: object) -> bool:
    try:
        normalize_relay_url(value)
    except RelayURLError:
        return False
    return True


def normalize_relay_base_url(value: object) -> str:
    """宽松版本：非法或为空时返回空串，供调用方继续回落下一级来源。"""
    try:
        return normalize_relay_url(value)
    except RelayURLError:
        return ""


def normalize_listener_port(raw: object, default: int = DEFAULT_LISTENER_PORT) -> int:
    if raw is None or raw == "":
        raw = default
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise RelayURLError("listener port must be an integer") from exc
    if port < 1 or port > 65535:
        raise RelayURLError("listener port must be between 1 and 65535")
    return port


def _first_header(headers: Mapping[str, str], name: str) -> str:
    for key, value in (headers or {}).items():
        if str(key).lower() == name:
            text = str(value or "").split(",")[0].strip()
            if text:
                return text
    return ""


def candidate_hosts(
    *,
    public_base: str = "",
    forwarded_proto: str = "",
    forwarded_host: str = "",
    scheme: str = "",
    host: str = "",
    server_port: int = 0,
) -> list[str]:
    """按优先级列出候选中转站地址。

    显式环境变量最优先，其次反向代理透传的 X-Forwarded-*，然后是浏览器实际访问
    的 scheme/host（它就是用户能连通的正确地址），最后是容器端口推导的本机地址。
    """
    candidates: list[str] = []
    if public_base:
        candidates.append(public_base)
    forwarded_host_value = str(forwarded_host or "").split(",")[0].strip()
    if forwarded_host_value:
        proto = str(forwarded_proto or "").split(",")[0].strip()
        if proto not in ("http", "https"):
            proto = "https" if str(scheme).lower() == "https" else "http"
        candidates.append(f"{proto}://{forwarded_host_value}")
    if host:
        proto = str(scheme or "http").lower()
        if proto not in ("http", "https"):
            proto = "http"
        candidates.append(f"{proto}://{host.strip()}")
    try:
        port = int(server_port or 0)
    except (TypeError, ValueError):
        port = 0
    if port:
        candidates.append(f"http://127.0.0.1:{port}")
    candidates.append(DEFAULT_RELAY_URL)
    return candidates


def candidate_relay_base_urls(
    *,
    relay_param: str = "",
    headers: Optional[Mapping[str, str]] = None,
    server_port: int = 0,
    env: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """从请求头/环境变量推导候选地址（供直接拿 headers 的调用方使用）。"""
    environ = env if env is not None else os.environ
    headers = headers or {}
    public_base = str(
        environ.get("TRAE_RELAY_URL")
        or environ.get("WEB_LOGIN_RELAY_URL")
        or environ.get("TRAE_PUBLIC_BASE_URL")
        or ""
    )
    candidates: list[str] = []
    if relay_param:
        # 浏览器 origin 优先于服务端推导：用户地址栏里的 host:port 一定可用。
        candidates.append(str(relay_param))
    if public_base:
        candidates.append(public_base)
    candidates.extend(
        candidate_hosts(
            forwarded_proto=_first_header(headers, "x-forwarded-proto"),
            forwarded_host=_first_header(headers, "x-forwarded-host"),
            scheme=_first_header(headers, "x-forwarded-proto") or "http",
            host=_first_header(headers, "host"),
            server_port=server_port,
        )
    )
    return candidates


def pick_relay_url(candidates: list[str]) -> str:
    """取第一个合法地址；全部非法时回落到默认地址，保证下载永不失败。"""
    for candidate in candidates:
        normalized = normalize_relay_base_url(candidate)
        if normalized:
            return normalized
    return normalize_relay_url(DEFAULT_RELAY_URL)


def resolve_relay_base_url(
    *,
    relay_param: str = "",
    headers: Optional[Mapping[str, str]] = None,
    server_port: int = 0,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    return pick_relay_url(
        candidate_relay_base_urls(
            relay_param=relay_param,
            headers=headers,
            server_port=server_port,
            env=env,
        )
    )


def _set_constant(text: str, name: str, value: str) -> str:
    pattern = re.compile(
        rf"""^({re.escape(name)}\s*=\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[0-9]+)(.*)$""",
        re.MULTILINE,
    )
    text, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
    if count != 1:
        raise RelayURLError(f"authorization helper template is missing {name}")
    return text


def _set_string_constant(text: str, name: str, value: str) -> str:
    # JSON string literals are also valid Python literals and correctly escape
    # quotes, backslashes, newlines and non-ASCII values from environment vars.
    return _set_constant(text, name, json.dumps(str(value), ensure_ascii=False))


def inject_relay_base_url(template: str, base_url: str) -> str:
    """把模板里的占位/默认地址改写成本次部署的实际中转站地址。"""
    relay_url = normalize_relay_url(base_url)
    text = (template or "").replace("\r\n", "\n")

    text = text.replace(RELAY_BASE_SENTINEL, relay_url)
    text = text.replace(RELAY_DOC_PLACEHOLDER, relay_url)

    text = _set_string_constant(text, "DEFAULT_RELAY_URL", relay_url)

    if DEFAULTS_MARKER not in text:
        # 兼容尚未引入常量的旧模板，历史镜像也能正常生成。
        text = re.sub(
            r'(--relay",\s*default=")[^"]*(")',
            lambda match: f"{match.group(1)}{relay_url}{match.group(2)}",
            text,
            count=1,
        )
        text = re.sub(
            r'^(?P<indent>[ \t]*relay_url[ \t]*=[ \t]*")[^"]*(?P<tail>".*)$',
            lambda match: f"{match.group('indent')}{relay_url}{match.group('tail')}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    return text


def set_listener_port(text: str, listener_port: int) -> str:
    port = normalize_listener_port(listener_port)
    body = (text or "").replace("\r\n", "\n")
    body = _set_constant(body, "DEFAULT_LISTENER_PORT", str(port))
    body = re.sub(
        r"^(  python web_login\.py \[--relay [^\]]+\] )\[--port \d+\]$",
        rf"\g<1>[--port {port}]",
        body,
        count=1,
        flags=re.MULTILINE,
    )
    body = re.sub(r"(?m)^set (\"?)PORT=\d+\1$", f"set \\1PORT={port}\\1", body)
    return body


def _stamp(
    relay_url: str,
    listener_port: int,
    generated_at: str,
    revision: str = "",
    generic: bool = False,
) -> str:
    if generic:
        lines = [
            "# 本副本为通用版授权助手，未写死任何部署地址。",
            f"# 默认回环地址: {relay_url}（仅当中转站就在本机同端口时可直接使用）",
            f"# 监听端口: {listener_port}",
            "# 其他部署请从中转站控制台下载已配置的助手。",
            "# 也可显式指定: python web_login.py --relay http://<host>:<port>",
        ]
        if revision:
            lines.append(f"# 构建 {revision}")
        return "\n".join(lines) + "\n"
    lines = [
        f"# 本副本由中转站 {relay_url} 于 {generated_at} 自动生成。",
        f"# 中转站:   {relay_url}",
        f"# 监听端口: {listener_port}",
        f"# 部署地址与监听端口(:{listener_port})已写入下方常量，直接运行即可，无需修改 IP/端口。",
    ]
    if revision:
        lines.append(f"# 构建 {revision}；如需指向其它中转站，使用 python web_login.py --relay http://<host>:<port>")
    else:
        lines.append("# 如需指向其它中转站，使用: python web_login.py --relay http://<host>:<port>")
    return "\n".join(lines) + "\n"


def banner_lines(base_url: str, *, revision: str = "", version: str = "") -> tuple[str, str]:
    first = f"由 Trae CN Relay v{version or 'dev'} 自动生成，部署地址 {base_url}。"
    second = f"构建 {revision or 'local'}；下载后直接运行即可，IP 与端口已写入，无需手动修改。"
    return first, second


def render_web_login_py(
    template: str,
    *,
    relay_url: str,
    listener_port: int = DEFAULT_LISTENER_PORT,
    client_id: str = DEFAULT_CLIENT_ID,
    auth_url: str = DEFAULT_AUTH_URL,
    generated_at: str = "",
    revision: str = "",
    generic: bool = False,
) -> str:
    """把仓库模板里的默认常量改写成本次部署的实际参数。"""
    relay_url = normalize_relay_url(relay_url)
    listener_port = normalize_listener_port(listener_port)

    text = inject_relay_base_url(template, relay_url)
    text = _set_string_constant(text, "DEFAULT_CLIENT_ID", str(client_id or DEFAULT_CLIENT_ID))
    text = _set_string_constant(text, "DEFAULT_AUTH_URL", str(auth_url or DEFAULT_AUTH_URL))
    text = set_listener_port(text, listener_port)
    text = re.sub(
        r"(?m)^(\s*python web_login\.py \[--relay ).*(\] \[--port )\d+(\].*)$",
        rf"\g<1>{relay_url}\g<2>{listener_port}\g<3>",
        text,
        count=1,
    )

    if generated_at:
        banner = _stamp(relay_url, listener_port, generated_at, revision, generic)
        if DEFAULTS_MARKER in text:
            head, separator, tail = text.partition(DEFAULTS_MARKER)
            text = head + banner + separator + tail
        else:
            lines = text.split("\n")
            anchor = 1 if lines and lines[0].startswith("#!") else 0
            lines[anchor:anchor] = [
                f"# {banner_line}" for banner_line in banner.strip().split("\n")
            ]
            text = "\n".join(lines)
    try:
        compile(text, "web_login.py", "exec")
    except SyntaxError as exc:
        raise RelayURLError("authorization helper template is invalid Python") from exc
    return text


def render_start_auth_bat(
    *,
    relay_url: str,
    listener_port: int = DEFAULT_LISTENER_PORT,
    generated_at: str = "",
    generic: bool = False,
) -> str:
    """Windows 一键授权脚本；每次启动先从中转站刷新 web_login.py。

    ``generic=True`` is an optional local-deployment fallback. The console
    download always generates a deployment-specific launcher.
    """
    relay_url = normalize_relay_url(relay_url)
    port = normalize_listener_port(listener_port)
    lines = [
        "@echo off",
        "rem ============================================================",
        ("rem  Trae2api-cn 授权助手（通用版）" if generic
         else "rem  Trae2api-cn 授权助手（按部署环境自动生成）"),
        f"rem  中转站:   {relay_url}",
        f"rem  本机回调: http://127.0.0.1:{port}/authorize",
        f"rem  生成时间: {generated_at or '自动生成'}",
        ("rem  本机默认部署的通用版；其他部署请从控制台下载助手。" if generic
         else "rem  双击运行即可，无需修改 IP 和端口。"),
        "rem ============================================================",
        "setlocal enableextensions disabledelayedexpansion",
        "chcp 65001 >nul",
        f'set "RELAY={relay_url}"',
        f'set "PORT={port}"',
        'set "HELPER=%~dp0web_login.py"',
        'set "TMP_HELPER=%HELPER%.download-%RANDOM%"',
        "",
        'if exist "%HELPER%\\*" (',
        "  echo web_login.py 目标不能是文件夹，请更换授权助手目录后重试。",
        "  pause",
        "  exit /b 1",
        ")",
        "echo [1/2] 正在从中转站刷新授权助手脚本 web_login.py ...",
        'del /q "%TMP_HELPER%" >nul 2>nul',
        'curl -fsSL --max-time 20 -o "%TMP_HELPER%" "%RELAY%/web/login/download?as=py"',
        "if errorlevel 1 (",
        '  del /q "%TMP_HELPER%" >nul 2>nul',
        "  echo 授权助手脚本下载失败，请确认本机可以访问 %RELAY%",
        "  pause",
        "  exit /b 1",
        ")",
        'move /y "%TMP_HELPER%" "%HELPER%" >nul',
        "if errorlevel 1 (",
        '  del /q "%TMP_HELPER%" >nul 2>nul',
        "  echo 无法更新授权助手脚本，请检查本目录写入权限。",
        "  pause",
        "  exit /b 1",
        ")",
        "",
        "where python >nul 2>nul",
        "if errorlevel 1 (",
        "  where py >nul 2>nul",
        "  if errorlevel 1 (",
        "    echo 未检测到 Python 3，请先安装 Python 并勾选 Add to PATH。",
        "    pause",
        "    exit /b 1",
        "  )",
        '  set "PYCMD=py -3"',
        ") else (",
        '  set "PYCMD=python"',
        ")",
        "",
        'echo [2/2] 启动本机授权监听器: http://127.0.0.1:%PORT%',
        "echo 浏览器会自动打开中转站授权页面，在该页面点击授权登录即可。",
        '%PYCMD% "%HELPER%" --relay "%RELAY%" --port "%PORT%"',
        "if errorlevel 1 (",
        "  echo 启动失败，请确认已安装 Python 3 并加入 PATH。",
        "  pause",
        "  exit /b 1",
        ")",
    ]
    return "\r\n".join(lines) + "\r\n"


def render_start_auth_sh(
    *,
    relay_url: str,
    listener_port: int = DEFAULT_LISTENER_PORT,
    generated_at: str = "",
    generic: bool = False,
) -> str:
    """macOS / Linux 一键授权脚本；generic 模式生成仓库 / Releases 通用版。"""
    relay_url = normalize_relay_url(relay_url)
    port = normalize_listener_port(listener_port)
    lines = [
        "#!/bin/sh",
        "# ============================================================",
        ("#  Trae2api-cn 授权助手（通用版）" if generic
         else "#  Trae2api-cn 授权助手（按部署环境自动生成）"),
        f"#  中转站:   {relay_url}",
        f"#  本机回调: http://127.0.0.1:{port}/authorize",
        f"#  生成时间: {generated_at or '自动生成'}",
        "#  运行: sh start_auth.sh  （或 chmod +x 后直接执行）",
        "# ============================================================",
        "set -e",
        "",
        'DIR="$(cd "$(dirname "$0")" && pwd)"',
        f'RELAY="{relay_url}"',
        f'PORT="{port}"',
        'HELPER="$DIR/web_login.py"',
        "",
        'if [ -d "$HELPER" ]; then',
        '  echo "web_login.py 目标不能是文件夹，请更换授权助手目录后重试。"',
        "  exit 1",
        "fi",
        'echo "[1/2] 正在从中转站刷新授权助手脚本 web_login.py ..."',
        'TMP_HELPER="$HELPER.download.$$"',
        'rm -f "$TMP_HELPER"',
        'if ! curl -fsSL --max-time 20 -o "$TMP_HELPER" "$RELAY/web/login/download?as=py"; then',
        '  rm -f "$TMP_HELPER"',
        '  echo "授权助手脚本下载失败，请确认本机可以访问 $RELAY"',
        "  exit 1",
        "fi",
        'if ! mv "$TMP_HELPER" "$HELPER"; then',
        '  rm -f "$TMP_HELPER"',
        '  echo "无法更新授权助手脚本，请检查本目录写入权限。"',
        "  exit 1",
        "fi",
        "",
        "if command -v python3 >/dev/null 2>&1; then",
        '  PYCMD="python3"',
        "elif command -v python >/dev/null 2>&1; then",
        '  PYCMD="python"',
        "else",
        '  echo "未检测到 Python 3，请先安装。"',
        "  exit 1",
        "fi",
        "",
        'echo "[2/2] 启动本机授权监听器: http://127.0.0.1:$PORT"',
        'exec "$PYCMD" "$HELPER" --relay "$RELAY" --port "$PORT"',
    ]
    return "\n".join(lines) + "\n"


def render(
    kind: str,
    *,
    template: str = "",
    relay_url: str,
    listener_port: int = DEFAULT_LISTENER_PORT,
    client_id: str = DEFAULT_CLIENT_ID,
    auth_url: str = DEFAULT_AUTH_URL,
    generated_at: str = "",
    revision: str = "",
    generic: bool = False,
) -> str:
    normalized_kind = _normalize_kind(kind)
    if normalized_kind == KIND_PY:
        return render_web_login_py(
            template,
            relay_url=relay_url,
            listener_port=listener_port,
            client_id=client_id,
            auth_url=auth_url,
            generated_at=generated_at,
            revision=revision,
            generic=generic,
        )
    if normalized_kind == KIND_BAT:
        return render_start_auth_bat(
            relay_url=relay_url,
            listener_port=listener_port,
            generated_at=generated_at,
            generic=generic,
        )
    return render_start_auth_sh(
        relay_url=relay_url,
        listener_port=listener_port,
        generated_at=generated_at,
        generic=generic,
    )


def build_helper_text(
    kind: str,
    template: str = "",
    *,
    base_url: str = "",
    listener_port: int = DEFAULT_LISTENER_PORT,
    client_id: str = DEFAULT_CLIENT_ID,
    auth_url: str = DEFAULT_AUTH_URL,
    generated_at: str = "",
    revision: str = "",
    version: str = "",
) -> str:
    """下载接口与发布打包共用的入口；base_url 为空时生成通用版。"""
    generic = not normalize_relay_base_url(base_url)
    relay_url = normalize_relay_base_url(base_url) or normalize_relay_url(DEFAULT_RELAY_URL)
    normalized_kind = _normalize_kind(kind)
    extra = f" v{version}" if version else ""
    return render(
        normalized_kind,
        template=template,
        relay_url=relay_url,
        listener_port=listener_port,
        client_id=client_id,
        auth_url=auth_url,
        generated_at=(generated_at + extra).strip(),
        revision=revision,
        generic=generic,
    )


def _normalize_kind(kind: object) -> str:
    value = str(kind or KIND_PY).strip().lower().lstrip(".")
    if value in ("python", "py", ""):
        return KIND_PY
    if value in ("bat", "cmd", "ps1"):
        return KIND_BAT
    if value in ("sh", "bash", "zsh"):
        return KIND_SH
    raise RelayURLError(f"unsupported helper kind: {kind}")


def file_name(kind: object, *, version: Optional[str] = None) -> str:
    base = {
        KIND_PY: "web_login.py",
        KIND_BAT: "start_auth.bat",
        KIND_SH: "start_auth.sh",
    }[_normalize_kind(kind)]
    if not version:
        return base
    stem, _, suffix = base.partition(".")
    return f"{stem}-{version}.{suffix}"


def media_type(kind: object) -> str:
    normalized = _normalize_kind(kind)
    if normalized == KIND_BAT:
        return "application/octet-stream"
    return "text/plain; charset=utf-8"


def bat_text(text: str) -> str:
    """batch 使用 CRLF，避免某些 Windows 版本对裸 LF 解析异常。"""
    body = py_text(text)
    return body.replace("\n", "\r\n")


def py_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def build_bundled_launchers() -> None:
    """Validate the Python template and build credential-free image assets."""
    root = Path(__file__).resolve().parent.parent
    render(
        "py",
        template=(root / "web_login.py").read_text(encoding="utf-8"),
        relay_url=DEFAULT_RELAY_URL,
    )
    for kind in ("bat", "sh"):
        (root / file_name(kind)).write_bytes(
            render(kind, relay_url=DEFAULT_RELAY_URL, generic=True).encode("utf-8")
        )


if __name__ == "__main__":
    build_bundled_launchers()
