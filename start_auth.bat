@echo off
rem Trae CN Relay - 一键网页授权（Windows）
rem 在浏览器打开的 relay 页面上点授权后，凭据会自动写入服务器。
chcp 65001 >nul
set RELAY=http://192.168.5.246:8000
set PORT=8765
echo ================================================
echo  Trae CN Relay 本机授权监听器
echo  中转站: %RELAY%
echo  本机回调: http://127.0.0.1:%PORT%/authorize
echo ================================================
echo.
python web_login.py --relay %RELAY% --port %PORT%
if errorlevel 1 (
  echo.
  echo 启动失败，请确认已安装 Python 3 并加入 PATH。
  pause
)
