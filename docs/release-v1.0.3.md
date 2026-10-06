# Trae2api-cn v1.0.3

## 更新内容

- 修复 Solo 端点请求链路，固定使用
  `/api/agent/v3/llm_utils_chat` 与 `function=solo_work_lite`。
- 修复账号地址带 `/api/remote/v1` 时 Agent/Solo 路径重复拼接的问题。
- 补齐 Solo/CN 设备指纹、路由和生产流量请求头。
- 修复 Tool 调用增量分片、调用 ID/参数拼接和截断 JSON 的处理。
- IDE Raw 已从正式控制台、模型测试页和端点预设移除。
  个人账号调用 `/api/ide/v2/llm_raw_chat` 会返回
  `2001 failed to get app config: record not found`，该路径需要企业版
  app config/PAT；`UPSTREAM_MODE=raw` 仍仅用于内部协议诊断。
- 真实端点验证：Remote、Solo、IDE Agent、Work Agent 的文本和 Tool 请求
  均可用。

## 验证

- `705 passed`
- `318 subtests passed`
- `python -m compileall -q src tests`

## 直接拉取镜像

```bash
docker pull ghcr.io/autumnsentiment/trae2api-cn:1.0.3
docker compose -f docker-compose.image.yml up -d
```

也可以使用 `latest` 跟随正式版本。更新已有部署前建议先备份当前容器和
`data/` 目录。

## 使用注意

- 个人账号请选择 Remote、Solo 或 IDE Agent；需要 Tool 时优先使用 IDE Agent
  或开启自动路由。
- IDE Raw 不属于个人账号可用的正式端点。
- Relay 只转发 Tool 调用，工具必须由调用端执行并回传结果。
