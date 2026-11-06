#!/usr/bin/env bash
# ===========================================================
#  Trae CN Relay - 局域网（LAN）模式启动器 - Linux / macOS
#
#  监听所有网卡，让同一局域网内的手机、平板和其它电脑都能
#  通过 http://<本机IP>:<端口> 使用中继站。
#
#  这个模式强制使用 API Key：没有配置或只配置了占位符时，
#  会自动生成一个随机 key 并写入 .env，重启后继续复用。
#
#  用法：
#     ./start_lan.sh                     # 端口取 .env 的 PORT，缺省 8000
#     ./start_lan.sh 8000                # 指定端口
#     ./start_lan.sh 8000 MyApiKey       # 指定端口 + 自定义 key
#     ./start_lan.sh --dry-run           # 只打印将要执行的命令，不启动
#
#  端口优先级：命令行参数 > .env 的 PORT > 8000。
#  LAN_PROBE_ADDR 可覆盖探测本机出口地址用的目标（默认 8.8.8.8:80，
#  内网/无外网环境可改成任一可达地址）。
#
#  这是 start_auth.sh 的“服务器端”对照脚本：start_auth.sh 负责
#  在浏览器里完成授权，start_lan.sh 负责把中继站开放给局域网。
# ===========================================================

set -u

PORT=""
KEY=""
DRY_RUN=0
PORT_FROM_ARG=0

# 位置参数按顺序取值（端口、key），带 - 开头的视为选项，不占位置。
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1; continue ;;
    -h|--help)
      echo "用法: ./start_lan.sh [端口] [API Key] [--dry-run]"
      exit 0
      ;;
    -*) echo "[错误] 未知选项：$arg"; exit 1 ;;
  esac
  if [ -z "$PORT" ]; then
    PORT="$arg"
    PORT_FROM_ARG=1
  elif [ -z "$KEY" ]; then
    KEY="$arg"
  else
    echo "[错误] 多余的参数：$arg"
    echo "       用法: ./start_lan.sh [端口] [API Key]"
    exit 1
  fi
done

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  case "$SOURCE" in
    /*) ;;
    *) SOURCE="$DIR/$SOURCE" ;;
  esac
done
HERE="$(cd -P "$(dirname "$SOURCE")" && pwd)"
ENV_FILE="$HERE/.env"
LAN_HELPER="$HERE/scripts/lan_env.py"

if [ ! -f "$HERE/src/main.py" ]; then
  echo "[错误] 未找到 src/main.py，请在项目根目录运行本脚本。"
  echo "       期望路径：$HERE/src/main.py"
  exit 1
fi

# ---------------------------------------------------------------
# 1) 选择可用的 Python 3.11+
# ---------------------------------------------------------------
PYEXE=""

py_ok() {
  [ -n "$1" ] || return 1
  if [ -x "$1" ]; then
    :
  elif ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

try_py() {
  [ -n "$PYEXE" ] && return 0
  py_ok "$1" || return 1
  PYEXE="$1"
  return 0
}

for cand in "$HERE/.venv/bin/python" "$HERE/venv/bin/python" python3 python; do
  [ -n "$PYEXE" ] && break
  try_py "$cand"
done

if [ -z "$PYEXE" ]; then
  echo "[错误] 未找到可用的 Python 3.11+（项目使用 3.11 语法，3.9/3.10 无法运行）。"
  echo
  echo "请先安装 Python 3，例如："
  echo "  Debian/Ubuntu : sudo apt install python3 python3-venv"
  echo "  macOS         : brew install python"
  echo "  或者直接编辑本脚本，把 PYEXE 指向完整的解释器路径。"
  echo
  exit 1
fi

PYREAL="$("$PYEXE" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PYEXE")"

# ---------------------------------------------------------------
# 2) 解析端口：命令行参数 > .env 的 PORT > 8000
# ---------------------------------------------------------------
if [ "$PORT_FROM_ARG" -eq 0 ]; then
  ENV_PORT=""
  if [ -f "$LAN_HELPER" ] && [ -f "$ENV_FILE" ]; then
    ENV_PORT="$("$PYEXE" "$LAN_HELPER" get --env "$ENV_FILE" --key PORT 2>/dev/null || true)"
  fi
  case "$ENV_PORT" in
    '') PORT="8000" ;;
    *[!0-9]*)
      echo "[错误] .env 中的 PORT 不是有效端口：$ENV_PORT"
      echo "       请修正 .env，或用 ./start_lan.sh <端口> 显式指定。"
      exit 1
      ;;
    *) PORT="$ENV_PORT" ;;
  esac
fi

case "$PORT" in
  ''|*[!0-9]*)
    echo "[错误] 无效端口：$PORT"
    echo "       用法: ./start_lan.sh [端口] [API Key]"
    exit 1
    ;;
esac

# ---------------------------------------------------------------
# 3) 依赖检查（缺少 FastAPI/uvicorn 时给出明确指引）
# ---------------------------------------------------------------
DEPS_OK=0
if "$PYEXE" -c 'import fastapi, uvicorn, httpx, dotenv' >/dev/null 2>&1; then
  DEPS_OK=1
fi

if [ "$DEPS_OK" -ne 1 ]; then
  echo "[提示] 缺少运行依赖（fastapi / uvicorn / httpx / python-dotenv）。"
  echo "       请先执行："
  echo "         $PYEXE -m pip install -r \"$HERE/requirements.txt\""
  echo "       或使用项目虚拟环境："
  echo "         python3 -m venv \"$HERE/.venv\" && \"$HERE/.venv/bin/pip\" install -r \"$HERE/requirements.txt\""
  echo
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] 忽略依赖缺失，继续演示。"
  else
    exit 1
  fi
fi

# ---------------------------------------------------------------
# 4) 解析 / 生成 LAN API Key
# ---------------------------------------------------------------
KEY_SOURCE="existing"
if [ -f "$LAN_HELPER" ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    # 演示时同样解析（必要时生成）key，但不写回 .env。
    KEY_OUT="$("$PYEXE" "$LAN_HELPER" resolve --env "$ENV_FILE" --key "$KEY" --json 2>/dev/null || true)"
  else
    KEY_OUT="$("$PYEXE" "$LAN_HELPER" resolve --env "$ENV_FILE" --key "$KEY" --save --json 2>/dev/null || true)"
  fi
  if [ -n "$KEY_OUT" ]; then
    PARSED="$("$PYEXE" -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("key","")); print(d.get("source","existing"))' "$KEY_OUT" 2>/dev/null || true)"
    KEY="$(printf '%s\n' "$PARSED" | sed -n '1p')"
    KEY_SOURCE="$(printf '%s\n' "$PARSED" | sed -n '2p')"
    [ -z "$KEY_SOURCE" ] && KEY_SOURCE="existing"
  fi
else
  echo "[警告] 未找到 scripts/lan_env.py，跳过自动生成 API Key。"
fi

if [ -z "$KEY" ]; then
  echo "[错误] 未配置可用的 LAN API Key。"
  echo "       请显式传入，例如：./start_lan.sh $PORT MyApiKey"
  exit 1
fi

# ---------------------------------------------------------------
# 5) 检查端口是否被占用
# ---------------------------------------------------------------
port_in_use() {
  "$PYEXE" - "$1" <<'PY' 2>/dev/null
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    raise SystemExit(0)
finally:
    sock.close()
raise SystemExit(1)
PY
}

if [ "$DRY_RUN" -ne 1 ] && port_in_use "$PORT"; then
  echo "[错误] 端口 $PORT 已被占用，无法启动。"
  echo "       换一个端口：./start_lan.sh 8001"
  echo "       或查看占用进程："
  echo "         Linux : ss -ltnp 'sport = :$PORT'"
  echo "         macOS : lsof -nP -iTCP:$PORT -sTCP:LISTEN"
  exit 1
fi

# ---------------------------------------------------------------
# 6) 收集本机局域网地址
# ---------------------------------------------------------------
# 探测目标可通过 LAN_PROBE_ADDR 覆盖，内网/无外网环境不必依赖公网 DNS。
# UDP connect 不发包，只让内核选出默认路由对应的本机地址。
LAN_PROBE_ADDR="${LAN_PROBE_ADDR:-8.8.8.8:80}"
export LAN_PROBE_ADDR

LOCAL_IPS="$("$PYEXE" - <<'PY' 2>/dev/null || true
import os
import socket

found = []
try:
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ip = info[4][0]
        if ip not in found and not ip.startswith("127."):
            found.append(ip)
except Exception:
    pass
if not found:
    target = os.environ.get("LAN_PROBE_ADDR", "8.8.8.8:80")
    host, _, port = target.rpartition(":")
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect((host or target, int(port or 80)))
        found.append(probe.getsockname()[0])
        probe.close()
    except Exception:
        pass
print("\n".join(found))
PY
)"

echo "==========================================================="
echo "  Trae CN Relay - 局域网 (LAN) 模式"
echo
echo "  端口    : $PORT"
echo "  Python  : $PYREAL"
echo "  .env    : $ENV_FILE"
case "$KEY_SOURCE" in
  generated)
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "  API Key : $KEY  （将生成并写入 .env）"
    else
      echo "  API Key : $KEY  （新生成，已写入 .env）"
    fi
    ;;
  explicit)  echo "  API Key : $KEY  （命令行指定）" ;;
  *)         echo "  API Key : $KEY  （沿用 .env 中的配置）" ;;
esac
echo
echo "  其它设备访问地址："
if [ -n "$LOCAL_IPS" ]; then
  while IFS= read -r ip; do
    [ -n "$ip" ] && echo "    http://$ip:$PORT/"
  done <<<"$LOCAL_IPS"
else
  echo "    http://<本机IP>:$PORT/"
fi
echo
echo "  其它设备上的使用方式："
echo "    - 控制台（授权登录 / 账号 / 签到 / 设置）："
if [ -n "$LOCAL_IPS" ]; then
  while IFS= read -r ip; do
    [ -n "$ip" ] && echo "        http://$ip:$PORT/web/login"
  done <<<"$LOCAL_IPS"
else
  echo "        http://<本机IP>:$PORT/web/login"
fi
echo "    - OpenAI 兼容接口 Base URL："
if [ -n "$LOCAL_IPS" ]; then
  while IFS= read -r ip; do
    [ -n "$ip" ] && echo "        http://$ip:$PORT/v1   （API Key 填上面的值）"
  done <<<"$LOCAL_IPS"
else
  echo "        http://<本机IP>:$PORT/v1   （API Key 填上面的值）"
fi
echo
echo "  如果其它设备连不上："
echo "    - Linux  : 放行防火墙端口，例如 sudo ufw allow $PORT/tcp"
echo "    - macOS  : 首次监听时系统会询问“允许传入连接”，选择允许；"
echo "               macOS 15+ 还需在 系统设置 > 隐私与安全性 中允许终端访问本地网络。"
echo "    - 确认所有设备处于同一网段，且路由器没有开启 AP/客户端隔离。"
echo
echo "  说明：Trae 授权回调固定为 127.0.0.1，手机/平板无法完成网页授权。"
echo "        请在服务器或另一台电脑上运行 ./start_auth.sh，或先在服务器上"
echo "        登录一次账号；之后所有设备共用同一账号池。"
echo
echo "  保持本窗口开启，关闭即停止服务；按 Ctrl+C 退出。"
echo "==========================================================="
echo

export HOST="0.0.0.0"
export PORT="$PORT"
# 环境变量优先于 .env，确保本次启动一定使用上面解析出的 key。
export RELAY_API_KEYS="$KEY"

# 清理会破坏 httpx 的 socks 代理变量。httpx 只支持 http / https / socks5，
# 不认识 socks:// 这个 scheme（常见于 Clash 等代理软件把 http 端口写成了
# socks），一旦命中会导致所有上游请求报 "Unknown scheme for proxy URL"。
# 只清除 socks:// 开头的项，保留 http/https 代理以便本机访问外网。
for _pv in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
  _val="${!_pv:-}"
  case "$_val" in
    socks://*) unset "$_pv" ;;
  esac
done

CMD=("$PYEXE" -m uvicorn src.main:app --host "$HOST" --port "$PORT")

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] 将执行："
  echo "  HOST=$HOST PORT=$PORT RELAY_API_KEYS=<已解析> \\"
  echo "    ${CMD[*]}"
  exit 0
fi

cd "$HERE" || exit 1
"${CMD[@]}"
STATUS=$?

echo
echo "[服务已退出] 退出码 $STATUS"
if [ -t 0 ]; then
  read -r -p "按回车键关闭..." _
fi
exit $STATUS
