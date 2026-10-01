# Trae2api-cn

Trae CN / Trae Solo CN 模型反代。把 Trae 的 Remote / IDE / Work 模型通道包装成 OpenAI 兼容 API（Chat Completions + Responses），支持调用方工具调用、thinking 分离输出、思考强度映射、多账号并发轮询、每日签到和积分消费记录，可直接接入 Codex、new-api 和各类 OpenAI SDK。

**仅供学习使用。请勿用于商业用途。**

## 特点

- OpenAI 兼容接口：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/responses`、`POST /v1/chat`、`POST /v1`
- 流式与非流式输出，SSE 心跳防止网关空闲断开
- 完整工具调用：`tools` / `tool_choice`（auto / none / required / 指定函数）/ `parallel_tool_calls`，流式输出标准 `delta.tool_calls`
- Codex Responses API：文本流、`function_call`、`custom_tool_call`、namespace 工具、`*_call_output` 续轮和 `previous_response_id`
- thinking 与正文分离：思考内容只进 `reasoning_content` / Responses reasoning 事件，不混入正文，且默认压缩为关键结论
- 思考强度：`reasoning_effort` / `thinking.budget_tokens` 映射到 Trae 原生 `light` / `high` / `extra_high`，按模型声明的档位自动钳制
- 四个可切换上游端点：Remote、IDE Agent、Work Agent、IDE Raw，原生端点失败时可自动回落 Remote
- 自动路由开关：按请求是否带工具自动选端点（工具走 IDE Agent，纯聊天走 Remote），失败保底回落 Remote
- 多账号：网页 OAuth 登录、手动添加凭证、顺序轮询 / 积分优先轮询、每账号并发槽位与排队
- 每日签到：一键签到、定时自动签到（控制台开关 + 每日时间，北京时间）、9074 风控自动换设备 ID 并退避重试、后台错峰自动重试
- 积分与消费记录：统一显示通用积分，按请求记录 tokens、单次积分和状态
- 网页控制台：账号与签到、消费记录、轮询与设置、模型连通性测试（可选端点、工具探针和思考强度）
- 1M Max 上下文模式（可选）、自动刷新 Cloud-IDE-JWT、未知模型透传
- Docker 一键部署，镜像只包含运行所需源码

## 端点与能力

控制台「轮询与设置 → 上游端点」可在运行时切换，也可用 `UPSTREAM_MODE` 固定。以下为 glm-5.3 / deepseek-v4-pro 实测结果（7 项工具测试：流式与非流式的工具调用、两轮工具续写、并行调用、强制 `tool_choice`、`tool_choice=none`）：

| 端点 | `UPSTREAM_MODE` | 上游路径 | 工具调用 | 思考强度 | 说明 |
|---|---|---|---|---|---|
| Remote（默认） | `remote` | `/api/remote/v1/chat_sessions` | glm-5.3 6/7（不支持并行调用） | 生效 | 最稳定的通用通道；Agent 优先，失败回退一次 Work |
| IDE Agent | `ide` | `/api/agent/v3/llm_utils_chat` | glm-5.3 7/7，deepseek-v4-pro 7/7 | 不生效（端点忽略该字段） | 工具能力最完整，适合 Codex / Agent 类客户端 |
| Work Agent | `work-agent` | Remote `solo_work_remote` | glm-5.3 6/7 | 生效（无 Work 档位时借用 Agent 档位） | 走 Work 执行器，不会被默认 provider 接管 |
| IDE Raw | `raw` | `/api/ide/v2/llm_raw_chat` | 不可用 | 不可用 | 个人账号返回 `2001 app config record not found`，需要企业 PAT；保留作协议诊断 |

部分模型在 Remote / Work 上会出现上游 `4028` / 502（例如 deepseek-v4-pro），此时请切换到 IDE Agent。需要完整工具能力（含并行调用）时推荐 IDE Agent；只聊天或需要思考强度时用 Remote。

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
| `TRAE_RAW_BASE_URL` | `https://trae-api-cn.mchost.guru` | Trae raw v2 `llm_raw_chat` 网关；账号站 `api.trae.com.cn` 不提供此模型端点 |
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

路由规则：选中的端点总是第一个尝试。`raw` 失败时按 `IDE Agent -> Remote` 回落，`ide` / `work-agent` 失败时回落 `remote`；请求进入公开流之后不会再跨端点重放，避免重复消费。`TRAE_REMOTE_ONLY_MODELS` 可把指定模型强制送往 remote，`*` 表示全部。

自动路由：在「轮询与设置 → 自动路由」打开开关（或 `POST /api/auto-route {"enabled":true}`）后，每个请求按内容选端点：请求带 `tools` / `tool_choice` / `parallel_tool_calls`，或历史中有工具调用 / 工具结果时走 IDE Agent，否则走 Remote；IDE Agent 失败时按上面的规则回落 Remote。开关开启时预设端点的模式选择不再生效，关闭后恢复预设端点。本地 CLI / TraeWork native 模式不受影响。模型测试页可选「自动路由」单独验证，不需要打开全局开关。

raw 端点的 HTTP body 固定为 `config_name`、`conversation_id`、`messages`、`model_name`、`session_id`、`stream` 六个字段，OpenAI 工具字段由 relay 转成系统提示，再把模型文本中的工具调用解析回 OpenAI 事件。

### 传输实现说明

`remote` 模式复用了 9router Trae executor 的两步会话协议：先 `POST /chat_sessions` 创建回合，再 `GET /chat_sessions/{id}/events?reply_to_message_id=...` 读取 SSE。`plan_item.thought` 按累计快照计算增量，`token_usage`、`done` 和 `error` 会转换为现有 OpenAI Chat/Responses 输出。它只复用 9router 的转发逻辑，不切换到国际版；默认仍使用当前 CN remote 地址。

`ide` 模式保留 trae2api 的 `/api/ide/v1/chat` 请求结构：稳定的 `session_id` / `conversation_id`、`chat_history`、`last_llm_response_info`、设备指纹和 Cloud-IDE-JWT 请求头。两种模式共用现有账号切换、token 快照、SSE 心跳、消费记录和 Responses 会话缓存。

`work-agent` 模式固定使用 Remote 的 `solo_work_remote` 执行器，并设置 `_trae_mode=work`，避免 Work 请求被默认 provider（例如 Kimi/Agent）接管。它保留调用端 `tools` 定义和工具历史；若 Work Remote 创建或首事件失败，则按配置回退到 Remote 通用路径。

raw 模式不会向上游发送其不接受的 OpenAI 顶层工具字段，也不会在空响应后伪造占位正文。只有在首个模型事件出现前允许一次空响应重试；已有输出、provider、usage 或工具事件后不会重放请求，避免重复消费。`GET /v1/status` 的 `tool_execution` 固定为 `client`，并列出当前工具桥接能力。

`UPSTREAM_MODE=cli` 仅作为显式兼容模式保留，会禁用默认工具并合并 `TRAE_CLI_DISALLOWED_TOOLS`；`auto` 不会进入该路径。

工具调用属于不可信模型输出。客户端应校验工具名 allowlist 和 JSON schema，并对路径、命令、权限、超时及输出大小做限制。提示词、`client_context`、工具 schema 和工具结果都会发送给 relay/Trae 上游，敏感内容仍需在调用端裁剪。

## 1M 上下文（Max 模式）

在控制台「轮询与设置 → 1M 上下文（Max 模式）」打开开关即可，新会话立即生效，无需重启。「生效模型」留空表示账号中所有标记 `max_mode` 的模型；点「检测支持的模型」可列出当前账号支持 Max 的模型并一键加入。单次请求也可带 `"trae_max_mode": true` 临时开启。

Max 只作用于 Remote 端点的 Agent 会话。带调用端工具的请求默认走 Work（`TRAE_REMOTE_CALLER_TOOLS_USE_WORK=1`），不会使用 Max；IDE Agent / Work Agent / IDE Raw 端点也不使用 Max。模型测试页勾选「1M Max」可确认是否实际生效。

## 思考内容与思考强度

思考内容默认不对外输出。请求带 `thinking: {"type": "enabled"}`（或 `include_reasoning: true`）时，Chat 返回 `reasoning_content`，Responses 返回独立的 reasoning 事件，正文 `content` 中不会出现思考文本。思考内容默认被压缩为关键结论，设置 `TRAE_VERBOSE_REASONING=1` 可恢复完整输出。

思考强度支持以下写法，统一映射到 Trae `custom_model.reasoning_effort`：

| 请求参数 | Trae 档位 |
|---|---|
| `reasoning_effort: minimal / low` | `light` |
| `reasoning_effort: medium / high` | `high` |
| `reasoning_effort: xhigh / max` | `extra_high` |
| `reasoning: {"effort": ...}`（Responses） | 同上 |
| `thinking.budget_tokens` < 4096 / < 16384 / 更大 | `light` / `high` / `extra_high` |

档位会被钳制到模型 `reasoning_effort_config` 声明的选项内。模型不支持思考档位或端点不接受该字段时（IDE Agent、IDE Raw），请求照常完成，强度字段被忽略；模型测试页会显示未生效原因。

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

启动后访问 `http://服务器:8000/web/login`，左侧纵向导航四个页面：

- **账号与签到**：上方是授权登录（网页授权、本机助手下载、手动填写凭证），中间是自动签到（启用开关、每日时间、立即执行，以及下次执行 / 上次执行 / 上次结果），下方是整宽账号列表（账号、用户 ID、状态、有效期、通用积分、签到状态，单账号签到 / 切换 / 删除，以及查询签到状态、查询全部积分、一键轮询签到）
- **消费记录**：按请求显示模型、tokens、单次积分和状态
- **轮询与设置**：多账号轮询开关、顺序 / 积分优先模式、上游端点预设与自定义 URL、Relay 端口、1M 上下文（Max 模式）开关与生效模型（可一键检测账号支持 Max 的模型）
- **模型测试**：批量测试模型连通性，可选文本 / 工具探针、思考强度、thinking 输出和 1M Max，并可指定端点（测试期间禁止跨端点回落）

## 使用注意事项

- **网络边界**：本项目设计为内网部署。不要把 8000 端口直接暴露到公网；如必须公网访问，请设置 `RELAY_API_KEYS` 并在前面加 TLS 反代。管理接口和模型测试接口对内网地址免鉴权。
- **敏感数据**：`.env` 与 `data/`（`accounts.json`、`usage_records.json`）包含 JWT、刷新令牌和账号信息，已在 `.gitignore` 和 `.dockerignore` 中排除，切勿提交或分享。
- **工具执行在调用端**：relay 只转发工具调用和结果，不会替客户端执行命令或写文件。模型声称“已下载 / 已写入”但本地没有文件，说明客户端没有真正执行工具，请检查客户端是否把 `tools` 发给 relay、是否回传了 `role: "tool"` 结果。
- **端点选择**：不想手动切换时打开「自动路由」；需要工具调用时优先用 IDE Agent 或 Remote；IDE Raw 对个人账号不可用。切换端点或开关后无需重启，立即生效。
- **上下文长度**：remote 扁平化 query 约 500K 字符时上游会静默断流，relay 默认在 480K 字符处裁剪最早的历史。长会话建议客户端自行压缩上下文。
- **并发**：每账号默认 2 个并行会话（`TRAE_WEB_PARALLEL_LIMIT`），多个 bot 同时请求时开启多账号轮询，请求会分散到不同账号；槽位全满时排队，超过 `TRAE_WEB_SLOT_TIMEOUT` 返回错误。
- **签到 9074**：9074 是上游风控码。relay 会自动轮换签到设备 ID 并指数退避重试，无需手动反复点击；若长期失败，可在 `TRAE_CHECKIN_DEVICE_IDS_JSON` 中填入真实客户端的设备 ID。
- **自动签到**：定时签到按顺序逐个账号执行，已签到和无凭证的账号会跳过，账号之间沿用 `TRAE_CHECKIN_INTERVAL_SECONDS` 间隔，遇到 9074 的账号交给后台退避重试。时间以北京时间计算，每天只触发一次定时任务；「立即执行」不会占用当天的定时任务。
- **积分**：控制台只显示合并后的通用积分。单次积分需要等上游账单落库后才能计算，个别记录短时间显示 `--` 属于正常现象。
- **502 / 空响应**：首个模型事件前的空响应会自动重试一次，仍失败时返回 502 并在错误信息中列出每个端点的失败原因。常见原因有模型未绑定到账号、上游 `4028` 和 raw 的 `2001`，可在模型测试页逐个端点排查。
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
