# Trae2api-cn

Trae CN / Trae Solo CN 模型与工具调用反代。把 Trae 终端协议包装成 OpenAI 兼容 API，支持调用方终端执行工具、多账号轮询、网页 OAuth 登录、账号并发控制和空闲会话回收。

**仅供学习使用。请勿用于商业用途。**

## 特点

- OpenAI 兼容接口：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/responses`、`POST /v1/chat`、`POST /v1`
- 流式与非流式输出
- OpenAI `tools` / `tool_choice` / `parallel_tool_calls` 兼容，流式输出标准 `delta.tool_calls`
- Codex Responses API 兼容：支持文本流、`function_call`、`custom_tool_call`、namespace 工具及 `*_call_output` 续轮
- 调用方工具桥接：请求中声明的 Windows 工作区、Shell、读写和编辑工具由 API 调用方执行，relay 只转发调用与结果
- 六种上游模式：Trae raw chat、9router 风格 remote 会话、Trae CLI 子进程、旧版网页 remote、IDE chat 端点、可选 TraeWork native bridge
- `auto` 与 `raw` 都只直连 Trae raw v2 `llm_raw_chat`，不会把请求留在 relay 缓存，也不会回退到 CLI/remote/web/IDE 模拟路径
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

开发环境使用 `pip install -r requirements-dev.txt`，Windows TraeWork native
helper 使用 `pip install -r requirements-native.txt`。Linux relay 镜像只安装核心依赖。

## Docker 部署

```bash
cp .env.example .env
docker compose up -d --build
```

容器默认监听 `8000` 端口，可通过 `RELAY_PORT` 环境变量修改。
发布构建可设置 `RELAY_BUILD_REVISION=$(git rev-parse --short HEAD)`；该值会写入
镜像 label，并由 `/v1/status` 返回，便于确认运行容器与源码版本一致。

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
| `UPSTREAM_MODE` | `remote` | 上游模式：remote 默认先用 `solo_agent_remote`，创建失败或首个模型事件前空响应时最多回退一次 `solo_work_remote`；其他兼容模式可显式选择 `raw` / `cli` / `web` / `ide` |
| `TRAE_CHECKIN_INTERVAL_SECONDS` | `60` | 多账号轮询时相邻实际签到请求的间隔 |
| `TRAE_CHECKIN_9074_RETRY_SECONDS` | `60` | 上游返回业务码 9074 后提示的最短重试等待时间；relay 不会立即重复 claim |
| `TRAE_RAW_BASE_URL` | `https://trae-api-cn.mchost.guru` | Trae raw v2 `llm_raw_chat` 网关；账号站 `api.trae.com.cn` 不提供此模型端点 |
| `TRAE_RAW_MAX_MESSAGES` | `80` | raw 请求保留的非系统历史消息上限；会保留最近连续历史及边界工具调用配对 |
| `TRAE_RAW_MAX_HISTORY_CHARS` | `120000` | raw 历史文本字符上限，避免重复工具 schema 和过长历史造成额外消费 |
| `TRAE_RAW_MAX_TOOL_SCHEMA_CHARS` | `48000` | raw 系统提示中的工具 schema 字符预算；超限时保留全部工具名并压缩为字段签名，减少输入积分消耗 |
| `TRAE_REMOTE_ONLY_MODELS` | 空 | 仅将列出的显式模型强制送往 remote；逗号分隔，`*` 表示全部强制 remote |
| `TRAE_REMOTE_MAX_MESSAGES` | `500` | remote 会话保留的非系统历史消息上限 |
| `TRAE_REMOTE_MAX_HISTORY_CHARS` | `480000` | remote 历史文本字符上限（压缩阶段） |
| `TRAE_REMOTE_QUERY_MAX_CHARS` | `480000` | remote 扁平化 query 的硬上限；上游超过约 500K 字符会静默结束事件流，超限时从最早的非系统消息开始裁剪 |
| `TRAE_REMOTE_MAX_MODE` | `0` | remote 会话启用 1M Max 模式；对账号配置 `max_mode=true` 的模型注入 `strategy=max` 与 1M/936K/64K 参数，并使用独立的 max 会话 ID |
| `TRAE_REMOTE_MAX_MODELS` | 空 | Max 模型白名单，逗号分隔；留空表示所有 `max_mode=true` 模型生效 |
| `TRAE_REMOTE_MAX_MODE_TYPE` | `1` | 服务端 `get_model_selection_modes` 的模式枚举；`1` 已实测生效 |
| `TRAE_REMOTE_AGENT_FIRST` | `1` | remote 是否默认锁定 Agent 执行器；关闭后普通请求直接使用 Work |
| `TRAE_REMOTE_WORK_FALLBACK` | `1` | Agent 创建失败或可重试空响应时，是否同账号回退一次 Work |
| `TRAE_REMOTE_CALLER_TOOLS_USE_WORK` | `1` | 带调用端工具的 remote 请求固定使用 Work，避免 Agent 内部远端工具把“已下载/已写入”误报为调用端本地文件操作；普通无工具请求仍默认 Agent |
| `TRAE_CLIENT_WORKSPACE_PATH` | `C:\workspace` | 未传 `client_context` 时使用的调用方工作区 |
| `TRAE_CLIENT_SYSTEM_TYPE` | `Windows` | 未传 `client_context` 时使用的调用方系统 |
| `TRAE_CLI_DISALLOWED_TOOLS` | `Read,Bash,Edit,Replace,Write,Glob,Grep,Task` | CLI 模式额外禁用的 relay 本机工具；外部工具请求会与默认项合并 |
| `TRAE_WEB_BASE_URL` | `https://trae-api-cn.mchost.guru/api/remote/v1` | remote 上游端点；`remote` 模式按 9router 的 `chat_sessions` + `events` 协议转发 |
| `TRAE_WEB_PARALLEL_LIMIT` | `2` | 每账号最大并行会话数 |
| `TRAE_WEB_IDLE_TIMEOUT` | `60` | 空闲会话回收超时（秒） |
| `TRAE_REMOTE_FIRST_EVENT_TIMEOUT_SECONDS` | `120` | remote 会话创建成功但没有首个 SSE 事件时的重试等待；首事件前 EOF/读超时同样按可重试空响应处理，`0` 表示关闭独立首事件期限 |
| `TRAE_FETCH_MODEL_LIST` | `false` | `/v1/models` 是否从上游拉取真实模型列表 |
| `SSE_HEARTBEAT_SECONDS` | `1` | Chat/Responses 上游空窗时发送标准 SSE 注释心跳；`0` 为关闭 |
| `TRAE_USAGE_RECORDS_PATH` | `data/usage_records.json` | 消费记录独立持久化文件，不改写 `data/accounts.json` |
| `TRAE_USAGE_SESSION_QUERY` | `true` | 有上游回合 ID 时异步查询精确积分；失败自动回退账号快照差值 |
| `TRAE_USAGE_API_HOST` | `https://api5-normal.mchost.guru` | TraeWork 商业用量查询主机，不与 entitlement API 混用 |
| `TRAE_USAGE_QUERY_TIMEOUT_SECONDS` | `15` | 回合积分查询超时时间；只影响后台 enrichment |
| `TRAE_USAGE_CREDIT_SETTLE_SECONDS` | `1` | 请求完成后等待上游积分账单落库再计算单次积分差值 |
| `RESPONSES_SESSION_TTL_SECONDS` | `3600` | `previous_response_id` 会话缓存有效期（秒） |
| `RESPONSES_SESSION_MAX_ENTRIES` | `1024` | Responses 进程内会话缓存最大条数 |
| `RELAY_API_KEYS` | 空 | API 密钥鉴权（逗号分隔多个）；公网部署必须设置并配合 TLS |
| `LOG_LEVEL` | `INFO` | 日志级别 |

控制台的“消费记录”按请求保存一行，包含输入/输出/总 tokens、单次消耗积分、请求状态和模型。积分优先级为：上游显式 usage、TraeWork 回合级 `credits_float`、同一账号请求前后的累计积分差值；无法安全归属时显示 `--`，不会把未知值伪装成 0。回合级查询使用上游 `reply_to_message_id/userMessageId`，不会把固定 raw 会话 UUID 当作计费键，也不会阻塞模型首帧。记录保存在独立的 `usage_records.json`，账号凭据仍只在 `accounts.json` 中维护。

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

`UPSTREAM_MODE=auto` 与 `UPSTREAM_MODE=raw` 的路由完全相同：Chat 与 Responses 请求都只发送到 `/api/ide/v2/llm_raw_chat`。raw HTTP body 固定为 `config_name`、`conversation_id`、`messages`、`model_name`、`session_id`、`stream` 六个字段；OpenAI `tools`、`tool_choice`、`parallel_tool_calls` 和工具历史不会作为顶层 raw 字段发送，而是由 relay 转成稳定的系统提示和不可执行历史，再把模型文本中的工具调用解析回 OpenAI 事件。

每个账号与模型使用确定性的独立 raw 会话，并同时在 body 与 `Extra` 中绑定 `config_name` / `model_name`，避免缺失模型选择时回落到默认 provider。显式模型默认走 raw；只有 `TRAE_REMOTE_ONLY_MODELS` 中列出的模型才改走 remote，`*` 可作为诊断时的全量强制开关。

### 传输实现说明

`remote` 模式复用了 9router Trae executor 的两步会话协议：先 `POST /chat_sessions` 创建回合，再 `GET /chat_sessions/{id}/events?reply_to_message_id=...` 读取 SSE。`plan_item.thought` 按累计快照计算增量，`token_usage`、`done` 和 `error` 会转换为现有 OpenAI Chat/Responses 输出。它只复用 9router 的转发逻辑，不切换到国际版；默认仍使用当前 CN remote 地址。

`ide` 模式保留 trae2api 的 `/api/ide/v1/chat` 请求结构：稳定的 `session_id` / `conversation_id`、`chat_history`、`last_llm_response_info`、设备指纹和 Cloud-IDE-JWT 请求头。两种模式共用现有账号切换、token 快照、SSE 心跳、消费记录和 Responses 会话缓存。

raw 模式不会向上游发送其不接受的 OpenAI 顶层工具字段，也不会在空响应后伪造占位正文。只有在首个模型事件出现前允许一次空响应重试；已有输出、provider、usage 或工具事件后不会重放请求，避免重复消费。`GET /v1/status` 的 `tool_execution` 固定为 `client`，并列出当前工具桥接能力。

`UPSTREAM_MODE=cli` 仅作为显式兼容模式保留，会禁用默认工具并合并 `TRAE_CLI_DISALLOWED_TOOLS`；`auto` 不会进入该路径。

工具调用属于不可信模型输出。客户端应校验工具名 allowlist 和 JSON schema，并对路径、命令、权限、超时及输出大小做限制。提示词、`client_context`、工具 schema 和工具结果都会发送给 relay/Trae 上游，敏感内容仍需在调用端裁剪。

## Codex Responses API

Codex 使用 `POST /v1/responses`，不能只把 `wire_api` 改成 Responses 后继续返回 Chat Completions SSE。relay 会把 Responses 的 `input`、扁平 function、自定义工具和 namespace 工具转换到现有 raw/CLI 工具管线，再返回带类型的 Responses 事件。

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

## 网页管理界面

启动后访问 `http://服务器:8000/web/login`：

- 状态查看
- 账号列表管理（切换 / 删除）
- 多账号轮询开关
- 自定义上游 URL 和端口
- 手动添加凭证
- 登出
- 获取模型列表：一键刷新 `/v1/models`（`TRAE_FETCH_MODEL_LIST=true` 时从上游拉取）

## 项目结构与运维

正式可部署源码只有以下边界：

```text
src/                    relay 服务与协议转换
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
