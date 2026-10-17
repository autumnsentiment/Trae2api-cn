#!/bin/bash
# ===========================================================
#  start_lan.command - start_lan.sh 的 macOS 双击包装脚本
#
#  以局域网模式启动中继站（同一网络内的手机、平板和其它
#  电脑都能访问）。
#
#  首次运行若被 Gatekeeper 拦截，请右键 > 打开。
#  双击无反应时先执行：
#      chmod +x start_lan.command start_lan.sh
# ===========================================================

set -u

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

if [ ! -f "$HERE/start_lan.sh" ]; then
  echo "[错误] 未找到同目录下的 start_lan.sh。"
  echo "       期望路径：$HERE/start_lan.sh"
  if [ -t 0 ]; then
    echo
    read -n 1 -s -r -p "按任意键关闭..."
  fi
  exit 1
fi

bash "$HERE/start_lan.sh" "$@"
STATUS=$?

if [ -t 0 ]; then
  echo
  read -n 1 -s -r -p "按任意键关闭..."
  echo
fi
exit $STATUS
