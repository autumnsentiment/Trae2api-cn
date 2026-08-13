# Trae2api-cn

Trae CN / Trae Solo CN 模型反代。把 Trae 的模型能力包装成 OpenAI 兼容 API，支持多账号轮询、网页 OAuth 登录、账号并发控制、空闲会话回收。

**仅供学习使用。请勿用于商业用途。**

## 特点

- OpenAI 兼容接口：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/chat`、`POST /v1`
- 流式与非流式输出
- 三种上游模式：Trae CLI 本地子进程、OmniRoute 网页版 remote 会话、IDE chat 端点
- 自动按优先级回退到可用上游端点
- 多账号管理：网页 UI 添加/切换/删除账号，round-robin 轮询
- 网页 OAuth 登录：在浏览器中直接授权，无需手动抓取 JWT
- 多账号轮询：每次请求自动切换到下一个有效账号，突破单账号并发限制
- 账号并发控制：每账号最大 2 个并行会话（与上游一致），超限排队等待
- 空闲会话回收：60 秒无活动的会话自动中断，释放并发槽位
- 自动刷新 Cloud-IDE-JWT 令牌
- 自动轮换设备指纹，降低 IDE 端点风控
- 未知模型透传上游，无需为每个新模型改代码
- 网页授权自动检测本机助手（方案 A：下载一次、双击即用）
- Docker 一键部署

## 认证方式

| 方式 | 说明 |
|---|---|
| `auto` | 自动解密本地 Trae CN / SOLO CN 的 `storage.json` |
| `env` | 从 `.env` 读取 TRAE_TOKEN / TRAE_REFRESH_TOKEN / TRAE_USER_ID |
| `manual` | 网页抓包得到的 Cloud-IDE-JWT |
| `cli` | 本地 Trae CLI 子进程，不需要 JWT |
| `web-login` | 浏览器授权登录，打开 `http://服务器:8000/web/login` 后在页面完成授权 |

## 快速开始

```bash
cp .env.example .env
# 编辑 .env 配置 TRAE_AUTH_SOURCE 和上游模式
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Docker 部署

```bash
cp .env.example .env
docker compose up -d --build
```

容器默认监听 `8000` 端口，可通过 `RELAY_PORT` 环境变量修改。

## 网页授权（推荐：web-login 模式）

> **为什么需要一个本机助手？**
> Trae 授权页强制要求授权回调地址为 `http://127.0.0.1:<端口>/authorize`，
> 且浏览器网页出于安全沙箱无法监听本机 TCP 端口，因此必须由本机运行
> 一个轻量监听器来接收回调并转发给服务器。**这是 Trae 的限制，不是本项目的缺陷。**

### 首次使用（只需一次）

1. 打开 `http://服务器:8000/web/login`
2. 点击「使用 Trae 网页授权登录」，页面会自动检测本机助手是否在线
3. 若提示「未检测到本机授权助手」，点击页面上的「下载一键启动 start_auth.bat」
4. 双击运行 `start_auth.bat`（Windows，无需安装 Python 时会提示安装），保持窗口开启
5. 回到页面重新点击授权，浏览器弹出 Trae 授权页，确认登录即可
6. 凭据自动写入服务器，页面自动刷新显示「已登录」

> 也可手动运行：`python web_login.py --relay http://服务器:8000`

### 之后每次授权

本机助手已在线时，直接点授权即可，无需重复下载和启动。

### 助手端点

- `GET http://127.0.0.1:8765/healthz` — 健康检查（供网页检测）
- `GET http://127.0.0.1:8765/relay-url` — 返回配置的 relay 地址

## 配置

详见 `.env.example` 中的注释。

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TRAE_AUTH_SOURCE` | `auto` | 认证来源 |
| `UPSTREAM_MODE` | `auto` | 上游模式：`cli` / `web` / `ide` / `auto` |
| `TRAE_WEB_BASE_URL` | `https://core-normal.trae.cn/api/remote/v1` | 网页版上游端点 |
| `TRAE_WEB_PARALLEL_LIMIT` | `2` | 每账号最大并行会话数 |
| `TRAE_WEB_IDLE_TIMEOUT` | `60` | 空闲会话回收超时（秒） |
| `TRAE_FETCH_MODEL_LIST` | `false` | `/v1/models` 是否从上游拉取真实模型列表 |
| `RELAY_API_KEYS` | 空 | API 密钥鉴权（逗号分隔多个） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 网页管理界面

启动后访问 `http://服务器:8000/web/login`：

- 状态查看
- 账号列表管理（切换 / 删除）
- 多账号轮询开关
- 自定义上游 URL 和端口
- 手动添加凭证
- 登出
- 获取模型列表：一键刷新 `/v1/models`（`TRAE_FETCH_MODEL_LIST=true` 时从上游拉取）

## 开源协议

MIT License

Copyright (c) 2026 autumnsentiment

本项目仅用于学习和研究目的。使用本项目产生的任何后果由使用者自行承担。
