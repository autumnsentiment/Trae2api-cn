# Issue #8 修复说明

## 结论

已完成真实端点复测。Remote、Solo、IDE Agent 和 Work Agent 可以作为正式
端点使用；IDE Raw 对当前个人账号不可用。

## IDE Raw 原因

逆向报告确认客户端的 Raw 请求是：

```text
POST /api/ide/v2/llm_raw_chat
```

该网关会先按企业 app config 绑定 `x-ide-token`。个人账号没有对应企业
配置时，无论模型名、`Extra` 头或请求体如何调整，都会在模型处理前返回：

```text
2001 failed to get app config: record not found
```

这是账号权限/配置缺失，不是 relay 的 SSE、工具参数或模型映射问题。该
路径需要企业版 app config/PAT，因此本版本已从正式控制台端点预设和模型
测试选择器中移除。`src/raw_client.py` 和显式 `UPSTREAM_MODE=raw` 仍保留
给内部协议诊断使用，但不建议个人账号启用。

## Solo 修复

Solo 端点固定使用：

```text
POST /api/agent/v3/llm_utils_chat
```

请求体使用 Trae SOLO 原生字段 `function=solo_work_lite`，同时写入
`config_name`、`model`、`session_id`、`request_id` 和结构化消息；请求头
使用 Cloud-IDE-JWT、设备指纹和生产流量标识。这样不会回退到 Raw，也不会
把 Solo 请求误路由到旧 `/api/ide/v1/chat` 封装。

## 使用建议

个人账号请选择 Remote、Solo 或 IDE Agent；需要调用端工具时优先使用
IDE Agent 或 Remote，或者打开自动路由。
