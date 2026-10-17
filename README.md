<p align="center"><img src="docs/logo.svg" width="96" height="96" alt="Trae2api-cn"></p>

# Trae2api-cn

Trae CN / Trae Solo CN 模型反代。把 Trae 的 Remote / IDE / Work 模型通道包装成 OpenAI 兼容 API（Chat Completions + Responses），支持调用方工具调用、thinking 分离输出、思考强度映射、多账号并发轮询、每日签到和积分消费记录，可直接接入 Codex、new-api 和各类 OpenAI SDK。

**仅供学习使用。请勿用于商业用途。**

## 特点

- OpenAI 兼容接口：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/responses`、`POST /v1/chat`、`POST /v1`
- Anthropic Messages 接口：`POST /v1/messages`、`POST /v1/messages/count_tokens`（Claude Code 直连），支持 `system`、`tool_use` / `tool_result`、`thinking` 与 `content_block` 流式事件，鉴权可用 `x-api-key`
- 流式与非流式输出，SSE 心跳防止网关空闲断开
- 原始流兼容：保留上游排队状态和终止帧顺序，不隐式重放被截断的模型请求
- 图片输入：`image_url` / Anthropic `image` 块会转成 Trae 渲染层媒体块；所选模型不支持视觉时自动改用多模态模型
- 完整工具调用：`tools` / `tool_choice`（auto / none / required / 指定函数）/ `parallel_tool_calls`，流式输出标准 `delta.tool_calls`
- Codex Responses API：文本流、`function_call`、`custom_tool_call`、namespace 工具、`*_call_output` 续轮和 `previous_response_id`
- thinking 与正文分离：思考内容只进 `reasoning_content` / Responses reasoning 事件，不混入正文，且默认压缩为关键结论
- 思考强度：`reasoning_effort` / `thinking.budget_tokens` 映射到 Trae 原生 `light` / `high` / `extra_high`，按模型声明的档位自动钳制
- 四个正式可切换上游端点：Remote、Work、Solo、IDE，原生端点失败时可自动回落 Remote
- 自动路由开关：按请求是否带工具自动选端点（工具走 IDE Agent，纯聊天走 Remote），失败保底回落 Remote
- 多账号：网页 OAuth 登录、手动添加凭证、账号模型请求开关（不影响签到）、顺序轮询 / 积分优先轮询、每账号并发槽位与排队
- 每日签到：一键签到、定时自动签到（控制台开关 + 每日时间，北京时间）、9074 风控自动换设备 ID 并退避重试、后台错峰自动重试
- 积分与消费记录：统一显示通用积分，按请求记录 tokens、单次积分、状态、思考强度、上下文模式及工具调用
- 自动任务模型路由：`model=auto` 时，简单请求使用 `deepseek-v4.1-flash`，工具/代码/文件/图片/长上下文请求尝试使用 `6.1sol`，两者默认思考强度为 `ultra`；显式模型请求保持不变
- 网页控制台：账号与签到、消费记录、轮询与设置、模型连通性测试（可选端点、工具探针和思考强度）
- 1M Max 上下文模式（可选）、自动刷新 Cloud-IDE-JWT、未知模型透传
- Docker 一键部署，镜像只包含运行所需源码

## 端点与能力

控制台「轮询与设置 → 上游端点」可在运行时切换，也可用 `UPSTREAM_MODE` 固定。以下为 glm-5.3 / deepseek-v4-pro 实测结果（7 项工具测试：流式与非流式的工具调用、两轮工具续写、并行调用、强制 `tool_choice`、`tool_choice=none`）：

| 端点 | `UPSTREAM_MODE` | 上游路径 | 工具调用 | 思考强度 | 说明 |
|---|---|---|---|---|---|
| Remote（默认） | `remote` | `/api/remote/v1/chat_sessions` | glm-5.3 6/7（不支持并行调用） | 生效 | 最稳定的通用通道；Agent 优先，失败回退一次 Work |
| Solo | `solo` | 默认 `https://trae-api-cn.mchost.guru` + `/api/ide/v2/llm_raw_chat`；请求失败时依次尝试 v1 和已知调度主机 `api5-normal.mchost.guru` | 由 TraeCode `raw_chat_function=solo_agent` 承接；工具字段按原生协议/提示桥接 | 由模型 `CustomModel` 声明决定 | TraeCode SOLO 本地链路；Remote 沙箱仍使用 `solo_agent_remote` |
| IDE Agent | `ide` | 默认 `https://trae-api-cn.mchost.guru` + `/api/ide/v2/llm_raw_chat`；请求失败时依次尝试 v1 和已知调度主机 `api5-normal.mchost.guru` | 由 `raw_chat_function=chat_v3` / `builder_v3` 承接 | 由模型 `CustomModel` 声明决定 | TraeCode IDE 主聊天链路；与 Remote 基址分离，回落 Remote 仍保留 `/api/remote/v1` |
| Work | `work`（`work-agent` 为旧兼容别名） | `/api/remote/v1/chat_sessions`，请求固定 `agent_type=agent_id=solo_work_remote` | 由 Work 档模型配置决定 | Work 不启用 1M Max | TraeWork 产品链路；Work 失败时清除 Work 绑定并回落普通 Remote Agent |

部分模型在 Remote / Work 上会出现上游 `4028` / 502（例如 deepseek-v4-pro），此时请切换到 IDE Agent。需要完整工具能力（含并行调用）时推荐 IDE Agent；只聊天或需要思考强度时用 Remote。

> **IDE Raw 暂不作为正式端点发布。** 逆向报告确认其协议是

> **基址分离**：Web/Remote 使用 `TRAE_WEB_BASE_URL`（含 `/api/remote/v1`），TraeCode IDE / Solo 使用 `TRAE_CODE_BASE_URL`（仅主机名；旧变量 `TRAE_MODEL_API_HOST` 兼容）。两者分开保存，选择 IDE/SOLO 预设不会再破坏 Remote 回落（#9）。
> `POST /api/ide/v2/llm_raw_chat`，但个人账号的 Cloud-IDE-JWT 没有企业
> app config 时，上游在鉴权/配置绑定阶段固定返回
> `2001 failed to get app config: record not found`。这不是请求体、模型名或
> SSE 重试可以修复的错误，必须使用企业版 app config/PAT。relay 仍保留
> `raw_client.py` 和显式 `UPSTREAM_MODE=raw` 供内部协议诊断，但正式控制台、
> 模型测试页和预设列表不再暴露该端点。

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

开发环境使用 `pip install -r requirements-dev.txt`，Windows TraeWork native
helper 使用 `pip install -r requirements-native.txt`。Linux relay 镜像只安装核心依赖。

## 局域网（LAN）模式

默认 `HOST=0.0.0.0`，服务本身就会监听所有网卡；`start_lan.sh` 在此基础上
把“能连”做成“能用且安全”：自动生成并持久化 API Key、打印各设备可用的
访问地址，并在缺少依赖或端口被占用时给出明确提示。

```bash
./start_lan.sh                    # 端口取 .env 的 PORT，缺省 8000
./start_lan.sh 8000               # 指定端口
./start_lan.sh 8000 MyApiKey      # 指定端口 + 自定义 key
./start_lan.sh --dry-run          # 只打印将要执行的命令，不启动
```

端口优先级为 **命令行参数 > `.env` 的 `PORT` > 8000**，与直接用
`uvicorn` 启动时的取值顺序一致。需要 Python 3.11+（项目使用了 3.10+
的联合类型注解，3.9/3.10 无法导入）。

macOS 可双击 `start_lan.command`（首次被 Gatekeeper 拦截时右键 > 打开）。

脚本通过 UDP connect 探测本机出口地址，默认目标为 `8.8.8.8:80`（不会真的
发包）。纯内网或无外网环境可用 `LAN_PROBE_ADDR=<host>:<port>` 覆盖探测目标，
探测失败时会回退为打印 `http://<本机IP>:<端口>`。

启动后脚本会打印：

- **控制台**：`http://<本机IP>:8000/web/login`，用于授权登录、账号与签到管理
- **OpenAI 兼容接口**：`http://<本机IP>:8000/v1`，API Key 填脚本打印的值

### API Key 与安全

- 局域网可达时 `/v1` 必须用真实 Key 保护：`.env.example` 里的
  `your-master-key` 只是占位符。`start_lan.sh` 检测到空值或占位符时会生成
  一个随机 key（`trae-lan-` 前缀）写入 `.env`，重启后继续复用。
- 已经配置了真实 key 时脚本不会改写 `.env`，多个 key（逗号分隔）会原样保留。
  解析规则与 `python-dotenv` 一致：支持 `export` 前缀、单/双引号和行尾注释。
- 直接用 `uvicorn` 启动、且 `HOST` 非回环时，若 `RELAY_API_KEYS` 为空或仍是
  占位符，服务会在启动日志中给出告警，提示先设置真实 key。
- 管理接口和模型测试接口对内网地址免鉴权（见「使用注意事项 · 网络边界」），
  因此不要把该端口直接暴露到公网；公网访问请加 TLS 反代。

### 其它设备连不上时

- **Linux**：放行防火墙端口，例如 `sudo ufw allow 8000/tcp`。
- **macOS**：首次监听时系统会询问“允许传入连接”，选择允许；macOS 15+ 还需在
  「系统设置 > 隐私与安全性」中允许终端访问本地网络。
- 确认所有设备处于同一网段，且路由器没有开启 AP / 客户端隔离。

> **授权只能在服务器本机完成**：Trae 授权页强制回调 `http://127.0.0.1:<端口>/authorize`，
> 手机 / 平板无法运行本机助手。请在服务器或另一台电脑上执行 `./start_auth.sh`
> （或先在服务器上登录一次账号），之后所有设备共用同一账号池。

## Docker 部署

### 使用 GitHub 发布镜像（推荐）

每个 `v*.*.*` 标签会由 GitHub Actions 自动完成测试、源码包发布和
GHCR 镜像构建。也可以在仓库的 **Actions → Release and container image →
Run workflow** 手动执行，默认会构建 `v1.0.5`。部署机器不需要安装 Python、
Node 或本项目源码，只需要 Docker、Docker Compose 和一个 `.env` 文件：

```bash
cp .env.example .env
docker pull ghcr.io/autumnsentiment/trae2api-cn:latest
docker compose -f docker-compose.image.yml up -d
```

镜像以公开 GHCR 包发布，支持 `linux/amd64` 和 `linux/arm64`，无需登录即可
拉取。需要固定版本时，把 `docker-compose.image.yml` 中的 `:latest` 改为发布页的
版本标签，例如 `:1.0.5`。更新已有部署时执行
`docker compose -f docker-compose.image.yml pull`，再执行
`docker compose -f docker-compose.image.yml up -d`。

预构建镜像的 Compose 文件不依赖外部 `new-api` 网络。如果 new-api 也在 Docker
中，可通过宿主地址访问 relay，或显式把两者加入同一网络。宿主端口由
`RELAY_PORT` 控制；容器监听端口由 `.env` 的 `PORT` 控制，两者默认都是 `8000`。

### 从源码构建（开发或离线场景）

源码构建仍可用于开发和没有外网的部署环境：

```bash
cp .env.example .env
docker compose up -d --build
```

容器默认监听 `8000` 端口。`RELAY_PORT` 修改宿主机发布端口，不修改容器内
监听端口；调整端口映射后需执行 `docker compose up -d --force-recreate`，
仅重启旧容器不会改变映射。
Actions 发布的镜像会把提交 SHA 写入 `RELAY_BUILD_REVISION` 镜像 label，并由
`/v1/status` 返回，便于确认运行容器与源码版本一致。手动源码构建可设置
`RELAY_BUILD_REVISION=$(git rev-parse --short HEAD)`。

## 网页授权（推荐：web-login 模式）

> **为什么需要一个本机助手？**
> Trae 授权页强制要求授权回调地址为 `http://127.0.0.1:<端口>/authorize`，
> 且浏览器网页出于安全沙箱无法监听本机 TCP 端口，因此必须由本机运行
> 一个轻量监听器来接收回调并转发给服务器。**这是 Trae 的限制，不是本项目的缺陷。**

### 首次使用（只需一次）

1. 打开 `http://服务器:8000/web/login`
2. 点击「使用 Trae 网页授权登录」，页面会自动检测本机助手是否在线
3. 若提示「未检测到本机授权助手」，点击页面上的下载按钮获取
   `web_login.py` / `start_auth.bat` / `start_auth.sh`
   （三个按钮都由服务端按当前部署实时生成，**IP 与端口已写好，无需手动修改**）
4. 双击运行 `start_auth.bat`（Windows）或 `sh start_auth.sh`（macOS/Linux），保持窗口开启
5. 回到页面重新点击授权，浏览器弹出 Trae 授权页，确认登录即可
6. 凭据自动写入服务器，页面自动刷新显示「已登录」

> 助手模板随 Docker 镜像打包，下载由当前部署直接提供，不需要访问 GitHub。
> 一键脚本每次启动都会从同一部署刷新 `web_login.py`，下载失败不执行旧副本；
> 本机需要 Python 3。
> 下载版可直接运行 `python web_login.py`，无需指定 IP 或端口。

### 之后每次授权

本机助手已在线并绑定当前部署时，直接点授权即可。服务器地址、映射端口或助手
监听端口变更后，关闭旧助手并从当前控制台重新下载，避免凭据送到旧地址。

### 助手端点

- `GET http://127.0.0.1:8765/healthz` — 健康检查（供网页检测）
- `GET http://127.0.0.1:8765/relay-url` — 返回配置的 relay 地址

### 中转站侧的下载与生成接口

| 接口 | 说明 |
| --- | --- |
| `GET /web/login/download?as=py\|bat\|sh` | 按当前部署实时生成授权助手，地址与端口已写入文件 |
| `GET /web/login/helper-info` | 返回本次生成会写入的中转站地址、监听端口与本地下载入口 |

生成顺序为：`TRAE_PUBLIC_BASE_URL` → 浏览器实际访问地址（`?relay=`）→
`X-Forwarded-Proto/Host` → 请求 `Host` 头 → 默认回环地址。反向代理部署可显式设置
`TRAE_PUBLIC_BASE_URL`，例如 `TRAE_PUBLIC_BASE_URL=http://gateway.internal:8000`。
脚本保留 Docker 映射端口、HTTPS 与反向代理子路径，不使用容器内部端口替换外部
端口。生成内容只包含部署地址、端口和公开 `client_id`，不含任何账号凭据。
直接部署时可设置 `TRAE_TRUST_PROXY_HEADERS=0`，忽略未受信任的代理地址头。

## 配置

详见 `.env.example` 中的注释。

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TRAE_AUTH_SOURCE` | `auto` | 认证来源 |
| `UPSTREAM_MODE` | `remote` | 上游模式：`remote` 为 Trae Remote Agent；`work`/`work-agent` 为 TraeWork Work-tier Remote；`solo`/`ide` 为 TraeCode raw；另有 `cli` / `web`；`raw` 仅供内部诊断 |
| `TRAE_WEB_SLOT_TIMEOUT` | `60` | 等待账号并发槽位的超时（秒）；所有账号槽位占满时请求排队等待 |
| `TRAE_VERBOSE_REASONING` | 空 | 设为 `1` 时输出完整思考链；默认只保留压缩后的关键结论 |
| `TRAE_MAX_COMPLETION_TOKENS` | `64000` | 单次输出 token 上限；Agent 模型最高 64K |
| `TRAEWORK_CUSTOM_UPSTREAM_MODE` | `remote` | TraeWork custom-model/raw 入站使用的上游；默认走已验证的 remote 会话，`raw` 仅保留作原生协议诊断 |
| `TRAE_CHECKIN_DEVICE_ID` | 空 | 可选的 TraeWork 原生 `guaranteedDeviceId`；配置后签到优先使用真实客户端设备身份 |
| `TRAE_CHECKIN_DEVICE_IDS_JSON` | 空 | 可选的账号到 TraeWork 设备 ID 的 JSON 映射，优先级高于全局设备 ID |
| `TRAE_CHECKIN_INTERVAL_SECONDS` | `60` | 多账号轮询时相邻实际签到请求的间隔 |
| `TRAE_CHECKIN_9074_RETRY_SECONDS` | `60` | 上游返回业务码 9074 后提示的最短重试等待时间；relay 不会立即重复 claim |
| `TRAE_CHECKIN_9074_MAX_BACKOFF_SECONDS` | `3600` | 连续 9074 指数退避的等待上限（秒） |
| `TRAE_CHECKIN_AUTO_RETRY_INTERVAL_SECONDS` | `60` | 后台自动错峰重试的扫描间隔（秒），到点自动 claim 冷却到期的账号 |
| `TRAE_RAW_BASE_URL` | `https://trae-api-cn.mchost.guru` | 内部 Raw v2 诊断地址；个人账号缺企业 app config 时会返回 `2001 record not found`，不属于正式能力 |
| `TRAE_RAW_MAX_MESSAGES` | `80` | raw 请求保留的非系统历史消息上限；会保留最近连续历史及边界工具调用配对 |
| `TRAE_RAW_MAX_HISTORY_CHARS` | `120000` | raw 历史文本字符上限，避免重复工具 schema 和过长历史造成额外消费 |
| `TRAE_RAW_MAX_TOOL_SCHEMA_CHARS` | `48000` | raw 系统提示中的工具 schema 字符预算；超限时保留全部工具名并压缩为字段签名，减少输入积分消耗 |
| `TRAE_REMOTE_ONLY_MODELS` | 空 | 仅将列出的显式模型强制送往 remote；逗号分隔，`*` 表示全部强制 remote |
| `TRAE_REMOTE_MAX_MESSAGES` | `500` | remote 会话保留的非系统历史消息上限 |
| `TRAE_REMOTE_MAX_HISTORY_CHARS` | `480000` | remote 历史文本字符上限（压缩阶段） |
| `TRAE_REMOTE_QUERY_MAX_CHARS` | `480000` | remote 扁平化 query 的硬上限；上游超过约 500K 字符会静默结束事件流，超限时从最早的非系统消息开始裁剪 |
| `TRAE_AUTO_ROUTE` | `0` | 自动路由默认值，控制台「自动路由」开关保存后以控制台为准；开启后带 `tools` / `tool_choice` 或工具历史的请求走 IDE Agent，其余走 Remote，失败回落 Remote |
| `TRAE_AUTO_CHECKIN` | `0` | 定时自动签到默认值，控制台「自动签到」开关保存后以控制台为准 |
| `TRAE_AUTO_CHECKIN_TIME` | `08:30` | 每日自动签到时间（`HH:MM`，北京时间）；relay 晚于该时间启动时当天会补签一次 |
| `TRAE_REMOTE_MAX_MODE` | `0` | 默认值，控制台「1M 上下文」开关保存后以控制台为准（写入 `data/accounts.json`，重启后保留）；remote 会话启用 1M Max 模式；对账号配置 `max_mode=true` 的模型注入 `strategy=max` 与 1M/936K/64K 参数，并使用独立的 max 会话 ID |
| `TRAE_REMOTE_MAX_MODELS` | 空 | Max 模型白名单，逗号分隔；留空表示所有 `max_mode=true` 模型生效 |
| `TRAE_REMOTE_MAX_MODE_TYPE` | `1` | 服务端 `get_model_selection_modes` 的模式枚举；`1` 已实测生效 |
| `TRAE_REMOTE_AGENT_FIRST` | `1` | remote 是否默认锁定 Agent 执行器；关闭后普通请求直接使用 Work |
| `TRAE_REMOTE_WORK_FALLBACK` | `1` | Agent 创建失败或可重试空响应时，是否同账号回退一次 Work |
| `TRAE_REMOTE_CALLER_TOOLS_USE_WORK` | `0` | remote 请求默认统一使用 Agent（包括带 `tools` 的请求）；设为 `1` 才将调用端工具请求固定到 Work，以便工具在调用端/本地工作区执行 |
| `TRAE_TASK_MODEL_ROUTING` | `1` | 是否对 `model=auto` 按任务复杂度选模型；显式模型不受影响，设为 `0` 可恢复 provider-owned auto |
| `TRAE_SIMPLE_MODEL` | `deepseek-v4.1-flash` | 自动路由的简单任务模型 |
| `TRAE_COMPLEX_MODEL` | `6.1sol` | 自动路由的复杂任务模型；仅在当前目录确认可用时发送 |
| `TRAE_CLIENT_WORKSPACE_PATH` | `C:\workspace` | 未传 `client_context` 时使用的调用方工作区 |
| `TRAE_CLIENT_SYSTEM_TYPE` | `Windows` | 未传 `client_context` 时使用的调用方系统 |
| `TRAE_CLI_DISALLOWED_TOOLS` | `Read,Bash,Edit,Replace,Write,Glob,Grep,Task` | CLI 模式额外禁用的 relay 本机工具；外部工具请求会与默认项合并 |
| `TRAE_WEB_BASE_URL` | `https://trae-api-cn.mchost.guru/api/remote/v1` | remote 上游端点；`remote` 模式按 9router 的 `chat_sessions` + `events` 协议转发 |
| `TRAE_CODE_BASE_URL` | `https://trae-api-cn.mchost.guru` | TraeCode IDE / Solo 模型网关主机地址；仅填写主机名，relay 自动追加 `/api/ide/v2/llm_raw_chat`。默认失败会在 TraeCode v1 与已知调度主机间重试；显式自定义主机不会被覆盖。旧变量 `TRAE_MODEL_API_HOST` 仍作为兼容别名 |
| `TRAE_WORK_BASE_URL` | 空 | 旧版 TraeWork raw 诊断链路的可选网关；正常 `work` 请求使用 `TRAE_WEB_BASE_URL` 的 Remote `chat_sessions` |
| `TRAE_WEB_PARALLEL_LIMIT` | `2` | 每账号最大并行会话数 |
| `TRAE_WEB_IDLE_TIMEOUT` | `60` | 空闲会话回收超时（秒） |
| `TRAE_REMOTE_FIRST_EVENT_TIMEOUT_SECONDS` | `120` | remote 会话创建成功但没有首个 SSE 事件时的重试等待；首事件前 EOF/读超时同样按可重试空响应处理，`0` 表示关闭独立首事件期限 |
| `TRAE_MODEL_LIST_CACHE_TTL` | `300` | `/v1/models` 上游模型列表缓存秒数。列表合并 Remote 的 Agent / Work / Design 目录与 Solo 原生 `get_detail_param` 目录，保留上游精确大小写并为混合大小写配置提供小写别名，条目带 `endpoint`/`endpoints` 并按端点排序；控制类内部模型会隐藏，保留 `auto`；上游拉取失败时回落内置列表 |
| `TRAE_SOLO_MODEL_LIST_TIMEOUT_SECONDS` | `8` | Solo 原生模型目录补齐请求的超时秒数；仅用于 Remote 目录缺失模型的发现与列表合并 |
| `SSE_HEARTBEAT_SECONDS` | `1` | Chat/Responses 上游空窗时发送标准 SSE 注释心跳；`0` 为关闭 |
| `TRAE_USAGE_RECORDS_PATH` | `data/usage_records.json` | 消费记录独立持久化文件，不改写 `data/accounts.json` |
| `TRAE_USAGE_SESSION_QUERY` | `true` | 有上游回合 ID 时异步查询精确积分；失败自动回退账号快照差值 |
| `TRAE_USAGE_API_HOST` | `https://api5-normal.mchost.guru` | TraeWork 商业用量查询主机，不与 entitlement API 混用 |
| `TRAE_USAGE_QUERY_TIMEOUT_SECONDS` | `15` | 回合积分查询超时时间；只影响后台 enrichment |
| `TRAE_USAGE_CREDIT_SETTLE_SECONDS` | `1` | 请求完成后等待上游积分账单落库再计算单次积分差值 |
| `TRAE_USAGE_MAX_HISTORY` | `1000` | 消费明细保留上限；每日汇总不受影响 |
| `TRAE_USAGE_QUERY_RETRIES` | `5` | 回合积分账单为空或暂不可用时的查询次数 |
| `TRAE_USAGE_QUERY_RETRY_DELAY_SECONDS` | `0.5` | 回合积分查询的指数退避基准秒数 |
| `TRAE_USAGE_ENRICH_CONCURRENCY` | `8` | 服务重启恢复历史待结算记录时的最大并发查询数（限制为 1-64） |
| `RESPONSES_SESSION_TTL_SECONDS` | `3600` | `previous_response_id` 会话缓存有效期（秒） |
| `RESPONSES_SESSION_MAX_ENTRIES` | `1024` | Responses 进程内会话缓存最大条数 |
| `RELAY_API_KEYS` | 空 | API 密钥鉴权（逗号分隔多个）；公网部署必须设置并配合 TLS |
| `LOG_LEVEL` | `INFO` | 日志级别 |

控制台的“消费记录”按请求保存一行，包含输入/输出/总 tokens、单次消耗积分、请求状态、模型、思考强度、上下文模式和 Tool 参数。工具定义与模型实际返回的工具调用分别记录，Max 请求值与上游应用值分别保存，没有上游证据时不标为已启用，历史记录缺失字段显示 `--`。积分优先级为：上游显式 usage、TraeWork 回合级 `credits_float`、同一账号请求前后的累计积分差值；账单尚未落库时显示“结算中”，后台会按退避策略重试，无法安全归属时才显示 `--`，不会把未知值伪装成 0。回合级查询使用上游 `reply_to_message_id/userMessageId`，并把 opaque 回合 ID 仅保存在本地记录中用于重启补偿，不会把固定 raw 会话 UUID 当作计费键，也不会阻塞模型首帧。记录保存在独立的 `usage_records.json`，账号凭据仍只在 `accounts.json` 中维护。

### 账号模型请求开关

「账号与签到」中每个账号都有独立的“模型请求”开关，旧账号默认开启，
开关保存在 `data/accounts.json` 中。关闭后，该账号不会接收新模型请求或
轮询重试；手动签到、每日定时签到、状态刷新与积分查询照常运行。
已经输出中的请求不会被强制中断，绑定该账号的会话下一轮请求返回
`503 model_account_unavailable`，可重新开启账号后继续。全部账号关闭时，
Chat Completions / Responses 返回明确的 `503`，不会绕回当前管理账号。

也可通过管理 API 设置：

```bash
curl -X POST http://127.0.0.1:8000/api/accounts/model-enabled \
  -H 'Content-Type: application/json' \
  -d '{"account_id":"你的账号ID","enabled":false}'
```

端点模式与自定义 URL 同样持久化在账号存储中，启动后恢复控制台保存的选择，
优先于容器环境默认值。Relay 服务端口单独保存，不会被端点切换顺带修改。

## 工具调用

工具调用使用标准 OpenAI 多轮协议。首次请求把工具 schema 和调用方环境放进请求：

```json
{
  "model": "auto",
  "stream": false,
  "session_id": "terminal-session-1",
  "messages": [{"role": "user", "content": "读取 README.md"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "读取调用方工作区中的文件",
      "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  }],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "client_context": {
    "workspace_path": "C:\\Users\\me\\project",
    "system_type": "Windows 11",
    "terminal_context": [{"shell": "PowerShell", "cwd": "C:\\Users\\me\\project"}]
  }
}
```

非流式响应使用 `message.content: null`、`message.tool_calls` 和 `finish_reason: "tool_calls"`；`function.arguments` 是 JSON 字符串。流式响应在 `delta.tool_calls` 中返回增量，客户端需要按 `index` / `id` 拼接，直到收到 `finish_reason: "tool_calls"`。

客户端收到 assistant 的 `tool_calls` 后，在自己的终端执行工具，再把原 assistant 消息和 `role: "tool"`、匹配的 `tool_call_id`（建议同时带 `name`）及执行结果一起发起下一轮请求。relay 本身不会执行请求中声明的外部工具，也不会自动拥有调用方文件系统；`client_context` 只是告诉模型真实环境，不能替代客户端工具实现。

IDE Agent 使用原生工具协议：schema 放入上游 `tools`，`parameters` 按 SOLO 协议编码为 JSON 字符串；历史中的 `assistant.tool_calls`、调用 ID、`role: "tool"` 和 `tool_call_id` 保持关联。Remote / Work Agent 与 Raw 使用文本工具桥接，不代表上游原生支持 OpenAI `tools`。2026-10-03 的独立 IDE 实测中，`glm-5.3` 的流式和非流式两轮调用均完成 Windows 客户端本地读文件、结果回传及内容校验，实际 provider 为 `glm-5.3`，未启用回落。

第二轮请求需带回完整历史，例如：

```json
{
  "model": "auto",
  "session_id": "terminal-session-1",
  "messages": [
    {"role": "user", "content": "读取 README.md"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_read_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_read_1",
      "name": "read_file",
      "content": "README.md 的实际内容"
    }
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "read_file",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
    }
  }]
}
```

路由规则：选中的端点总是第一个尝试。`ide` / `solo` / `work`（以及旧别名 `work-agent`）失败时回落 `remote`；Raw 仅在显式内部诊断模式下按兼容逻辑处理。空响应仅在允许重试且未观察到模型活动时可跨端点回落。已收到输出、provider、usage 或工具事件，或返回不可重试错误时，不重放模型请求，避免重复消费。公开流的初始空 delta / 心跳不是模型输出。`TRAE_REMOTE_ONLY_MODELS` 可把指定模型强制送往 remote，`*` 表示全部。

自动路由：在「轮询与设置 → 自动路由」打开开关（或 `POST /api/auto-route {"enabled":true}`）后，每个请求按内容选端点：请求带 `tools` / `tool_choice` / `parallel_tool_calls`，或历史中有工具调用 / 工具结果时走 IDE Agent，否则走 Remote；IDE Agent 失败时按上面的规则回落 Remote。开关开启时预设端点的模式选择不再生效，关闭后恢复预设端点。本地 CLI / TraeWork native 模式不受影响。模型测试页可选「自动路由」单独验证，不需要打开全局开关。

内部 Raw 诊断端点的 HTTP body 固定为 `config_name`、`conversation_id`、`messages`、`model_name`、`session_id`、`stream` 六个字段，OpenAI 工具字段由 relay 转成系统提示，再把模型文本中的工具调用解析回 OpenAI 事件；个人账号因缺少企业 app config 仍无法通过上游绑定校验。

### 传输实现说明

`remote` 模式复用了 9router Trae executor 的两步会话协议：先 `POST /chat_sessions` 创建回合，再 `GET /chat_sessions/{id}/events?reply_to_message_id=...` 读取 SSE。`plan_item.thought` 按累计快照计算增量，`token_usage`、`done` 和 `error` 会转换为现有 OpenAI Chat/Responses 输出。它只复用 9router 的转发逻辑，不切换到国际版；默认仍使用当前 CN remote 地址。

`solo` 和 `ide` 模式都调用 TraeCode 的 `/api/ide/v2/llm_raw_chat`（失败回退 `/api/ide/v1/llm_raw_chat`），使用 `config_name`、`model_name`、`raw_chat_function`、`config_source`、结构化 `messages` 和 Cloud-IDE-JWT 请求头。`solo` 使用 `raw_chat_function=solo_agent`，`ide` 使用 `chat_v3`；两者共用账号绑定、SSE 心跳、消费记录和 Responses 会话缓存。旧 `llm_utils_chat` 仅保留给内部辅助/兼容调用。

`work`（旧兼容名 `work-agent`）走 Remote `chat_sessions`，但在模型配置解析、会话 `agent_type` 和 `agent_id` 中都固定为 `solo_work_remote`，避免账号默认 provider 接管模型。Work 失败时会清除 Work 标记，再以普通 Agent 身份回落 Remote，不会表面回落、实际仍请求 Work。请求记录仍标识为 TraeWork；TraeCode 的 `solo`/`ide` 使用独立 raw v2 链路。

raw 模式不会向上游发送其不接受的 OpenAI 顶层工具字段，也不会在空响应后伪造占位正文。只有在首个模型事件出现前允许一次空响应重试；已有输出、provider、usage 或工具事件后不会重放请求，避免重复消费。`GET /v1/status` 的 `tool_execution` 固定为 `client`，并列出当前工具桥接能力。

`UPSTREAM_MODE=cli` 仅作为显式兼容模式保留，会禁用默认工具并合并 `TRAE_CLI_DISALLOWED_TOOLS`；`auto` 不会进入该路径。

工具调用属于不可信模型输出。客户端应校验工具名 allowlist 和 JSON schema，并对路径、命令、权限、超时及输出大小做限制。提示词、`client_context`、工具 schema 和工具结果都会发送给 relay/Trae 上游，敏感内容仍需在调用端裁剪。

## 1M 上下文（Max 模式）

在控制台「轮询与设置 → 1M 上下文（Max 模式）」打开开关即可，新会话立即生效，无需重启。「生效模型」留空表示账号中所有标记 `max_mode` 的模型；点「检测支持的模型」可列出当前账号支持 Max 的模型并一键加入。单次请求也可带 `"trae_max_mode": true` 临时开启。

Max 只作用于 Remote 端点的 Agent 会话。Remote 请求默认先走 Agent，包含带调用端工具的请求；若需要让调用端工具固定在本地工作区执行，可设置 `TRAE_REMOTE_CALLER_TOOLS_USE_WORK=1`，此时该回合使用 Work 且不会使用 Max。IDE / Work 端点也不使用 Max。模型测试页勾选「1M Max」可确认是否实际生效。

## 思考内容与思考强度

思考内容默认不对外输出。请求带 `thinking: {"type": "enabled"}`（或 `include_reasoning: true`）时，Chat 返回 `reasoning_content`，Responses 返回独立的 reasoning 事件，正文 `content` 中不会出现思考文本。思考内容默认被压缩为关键结论，设置 `TRAE_VERBOSE_REASONING=1` 可恢复完整输出。

思考强度支持以下写法，统一映射到 Trae `custom_model.reasoning_effort`：

| 请求参数 | Trae 档位 |
|---|---|
| `reasoning_effort: minimal / low` | `light` |
| `reasoning_effort: medium / high` | `high` |
| `reasoning_effort: xhigh / max / ultra` | `extra_high` |
| `reasoning: {"effort": ...}`（Responses） | 同上 |
| `thinking.budget_tokens` < 4096 / < 16384 / 更大 | `light` / `high` / `extra_high` |

档位会被钳制到模型 `reasoning_effort_config` 声明的选项内。模型不支持思考档位或端点不接受该字段时（IDE Agent、内部 Raw 诊断），请求照常完成，强度字段被忽略；模型测试页会显示未生效原因。

## Codex Responses API

Codex 使用 `POST /v1/responses`，不能只把 `wire_api` 改成 Responses 后继续返回 Chat Completions SSE。relay 会把 Responses 的 `input`、扁平 function、自定义工具和 namespace 工具转换到统一工具管线，再按选定端点使用 IDE 原生协议或文本桥接，返回带类型的 Responses 事件。

流式工具轮次会依次包含完整的 `response.output_item.done` 和 `response.completed`。Codex 收到当前响应完成后，才会在调用方终端执行工具，并用相同 `call_id` 加入 `function_call_output` 或 `custom_tool_call_output`，自动发起下一次 `/v1/responses` 请求。Responses 流不使用 Chat Completions 的 `data: [DONE]` 作为完成信号。

Codex 自定义 provider 示例：

```toml
model_provider = "trae-relay"
model = "glm-5.3"

[model_providers.trae-relay]
base_url = "http://服务器:8000/v1"
wire_api = "responses"
requires_openai_auth = true
```

若前面还有 new-api，应确保其 Responses 渠道把 `/v1/responses` 原样转发到本 relay；relay 地址自身则是 `http://服务器:8000/v1`。

relay 支持两种连续会话方式：客户端可以在每轮重放完整 `input` 历史，也可以只发送新的输入或工具结果并携带上一轮返回的 `previous_response_id`。后一种方式会从有 TTL 和容量上限的进程内缓存恢复用户消息、assistant 输出、工具调用及绑定；客户端即使同时重放完整历史也会进行重叠去重。`store: false` 不写入缓存，未知、过期、容器重启后失效或超过缓存上限的 response id 会返回明确的 `400 previous_response_id` 错误。

## Anthropic Messages API

Claude Code 等 Anthropic 客户端使用 `POST /v1/messages`：

```json
{
  "model": "glm-5.3",
  "max_tokens": 32000,
  "system": "be concise",
  "tools": [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}],
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "读一下 a.txt"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "文件内容"}]}
  ]
}
```

- 鉴权：`x-api-key: <RELAY_API_KEYS 中的值>` 或 `Authorization: Bearer <key>`。
- `POST /v1/messages/count_tokens` 只做本地输入 token 估算，不创建上游会话，也不会产生积分消费记录。
- 流式返回标准 `message_start` / `content_block_start` / `content_block_delta` / `message_delta` / `message_stop`，`thinking` 走 `thinking_delta`，工具调用走 `tool_use` + `input_json_delta`。
- `tool_choice` 的 `auto` / `any` / `tool` / `none` 分别映射到 `auto` / `required` / 指定函数 / `none`；`thinking.budget_tokens` 映射到原生思考档位。
- 工具仍然只在调用端执行，relay 只负责协议翻译。

Claude Code 环境变量示例：

```bash
export ANTHROPIC_BASE_URL=http://服务器:8000
export ANTHROPIC_API_KEY=<RELAY_API_KEYS 中的值>
export ANTHROPIC_MODEL=glm-5.3
```

## 图片输入

- **图片输入**：`image_url`（OpenAI）与 `image`（Anthropic，base64 或 URL）都会转成 Trae 渲染层媒体块随请求发送。若所选模型不支持视觉，relay 会按账号实际可用的多模态模型自动切换，并在请求追踪中记录 `multimodal_switch`。用 `TRAE_MULTIMODAL_AUTO_SWITCH=0` 关闭，或用 `TRAE_MULTIMODAL_MODEL` 指定目标模型。远程图片 URL 会原样转发给上游，relay 不会代下载；需要确定性输入时请使用 base64 data URL。
- **排队位置**：Trae raw 上游的排队事件按原始兼容协议作为 assistant 状态文本转发，并同时记录到请求元数据；不会额外生成 synthetic SSE keepalive。

## 网页管理界面

启动后访问 `http://服务器:8000/web/login`，左侧纵向导航五个页面，右上角可切换浅色 / 夜间主题：

- **概览**（默认首页）：总 tokens、输入 / 输出 / 缓存 tokens、今日用量、有效账号数；按日期堆叠柱状图（7 / 14 / 30 天）、服务状态（版本、运行时长、上游端点、自动路由、轮询、1M、自动签到、并发中请求）和按日期汇总表。数据来自 `GET /api/overview?days=N`，每日汇总持久化在 `data/usage_stats.json`，不受消费明细保留上限影响
- **账号与签到**：上方是授权登录（网页授权、本机助手下载、手动填写凭证），中间是自动签到（启用开关、每日时间、立即执行，以及下次执行 / 上次执行 / 上次结果），下方是整宽账号列表（账号、用户 ID、状态、有效期、通用积分、签到状态，单账号签到 / 切换 / 删除，以及查询签到状态、查询全部积分、一键轮询签到）
- **消费记录**：按请求显示模型、tokens、单次积分、状态、思考强度、上下文模式，以及是否携带工具参数、本轮是否返回工具调用和排队链路
- **轮询与设置**：多账号轮询开关、顺序 / 积分优先模式、上游端点预设与自定义 URL、Relay 端口、1M 上下文（Max 模式）开关与生效模型（可一键检测账号支持 Max 的模型）
- **模型测试**：批量测试模型连通性，可选文本 / 工具探针、思考强度、thinking 输出和 1M Max，并可指定端点（测试期间禁止跨端点回落）

## 使用注意事项

- **网络边界**：本项目设计为内网部署。不要把 8000 端口直接暴露到公网；如必须公网访问，请设置 `RELAY_API_KEYS` 并在前面加 TLS 反代。管理接口和模型测试接口对内网地址免鉴权。局域网共享给手机等设备时建议用 `./start_lan.sh`，它会生成并持久化一个随机 Key，避免 `your-master-key` 这类占位符被直接用于可达端口。
- **敏感数据**：`.env` 与 `data/`（`accounts.json`、`usage_records.json`）包含 JWT、刷新令牌和账号信息，已在 `.gitignore` 和 `.dockerignore` 中排除，切勿提交或分享。
- **工具执行在调用端**：relay 只转发工具调用和结果，不会替客户端执行命令或写文件。模型声称“已下载 / 已写入”但本地没有文件，说明客户端没有真正执行工具，请检查客户端是否把 `tools` 发给 relay、是否回传了 `role: "tool"` 结果。
- **端点选择**：不想手动切换时打开「自动路由」；需要工具调用时优先用 IDE Agent 或 Remote。IDE Raw 已从正式控制台和打包预设中移除，个人账号不要选择 `UPSTREAM_MODE=raw`。切换端点或开关后无需重启，立即生效。
- **截断处理**：relay 不会自动续写或隐式重放模型请求；收到 `finish_reason=length` 或上游提前断流时，由调用端决定是否发起下一轮，避免重复消费和工具回合错乱。
- **上下文长度**：remote 扁平化 query 约 500K 字符时上游会静默断流，relay 默认在 480K 字符处裁剪最早的历史。长会话建议客户端自行压缩上下文。
- **并发**：每账号默认 2 个并行会话（`TRAE_WEB_PARALLEL_LIMIT`），多个 bot 同时请求时开启多账号轮询，请求会分散到不同账号；槽位全满时排队，超过 `TRAE_WEB_SLOT_TIMEOUT` 返回错误。
- **签到 9074**：9074 是上游风控码。relay 会自动轮换签到设备 ID 并指数退避重试，无需手动反复点击；若长期失败，可在 `TRAE_CHECKIN_DEVICE_IDS_JSON` 中填入真实客户端的设备 ID。
- **自动签到**：定时签到按顺序逐个账号执行，已签到和无凭证的账号会跳过，账号之间沿用 `TRAE_CHECKIN_INTERVAL_SECONDS` 间隔，遇到 9074 的账号交给后台退避重试。时间以北京时间计算，每天只触发一次定时任务；「立即执行」不会占用当天的定时任务。
- **用量统计**：缓存 tokens 仅在上游返回时统计，旧记录按 0 计；日期按北京时间划分，最多保留 400 天。
- **积分**：控制台只显示合并后的通用积分。单次积分需要等上游账单落库后才能计算，个别记录短时间显示 `--` 属于正常现象。
- **502 / 空响应**：首个模型事件前的空响应会自动重试一次，仍失败时返回 502 并在错误信息中列出每个端点的失败原因。常见原因有模型未绑定到账号和上游 `4028`。如果日志出现 `raw: ... 2001 failed to get app config: record not found`，这是个人账号不具备企业 app config 的固定权限错误；正式版本已移除 Raw 预设，请改用 IDE Agent、Solo 或 Remote。
- **更新部署**：更新前建议先 `docker commit` 备份当前容器镜像；更新后浏览器按 Ctrl+F5 强制刷新控制台。
- **合规**：使用本项目需自行承担账号风险，请遵守 Trae 服务条款。

## 项目结构与运维

正式可部署源码只有以下边界：

```text
src/                    relay 服务与协议转换
  main.py               路由、端点调度、网页控制台、签到与消费记录
  trae_remote_client.py Remote / Work Agent 会话协议
  trae_client.py        IDE 端点、账号槽位、模型列表
  raw_client.py         raw v2 协议与工具提示词桥接
  responses_api.py      Responses API 转换与会话缓存
  sse.py                SSE 转换、thinking 分离与工具调用解析
  reasoning_effort.py   思考强度映射
  traework_compat.py    TraeWork custom-model 入站兼容
tests/                  离线协议、流式和路由回归测试
native/                 TraeWork native 文件清单与说明，不包含 DLL
tools/                  Windows native helper
scripts/backup_runtime.sh  Docker 运行态一致性备份
```

探针、抓包、逆向材料、账号状态、旧发布包和备份不进入 Docker 构建上下文，
也不应提交到仓库。`trae-cn-relay` 是唯一产品源码目录；研究目录只用于生成报告或补丁。

发布前运行：

```bash
python -m pytest -q
docker compose build
```

在 Docker 主机上备份当前运行容器、容器内源码、宿主配置和数据：

```bash
sudo bash scripts/backup_runtime.sh
```

脚本会短暂停止并自动重启容器，备份写入 `backups/<timestamp>-runtime/`，
同时生成镜像归档和 SHA-256 校验清单。`.env` 与 `data` 含敏感状态，备份目录权限为 `0700`，文件为 `0600`。

## 开源协议

MIT License

Copyright (c) 2026 autumnsentiment

本项目仅用于学习和研究目的。使用本项目产生的任何后果由使用者自行承担。
