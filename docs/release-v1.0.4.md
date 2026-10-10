# Trae2api-cn v1.0.4

## 更新内容

- 合并 Remote 与 Solo 原生模型目录，补齐 `glm-5.3-flash`、
  `glm-5.3-flashx` 等 Solo-only 模型。
- 修复原生 Tool 调用增量分片、重复帧和跨 plan/message 文本重复输出。
- 改进流式空响应与账号配额错误的识别，避免把上游元数据误判为模型输出。
- 改进消费记录的回合级积分结算、重试和服务重启后的待结算恢复。
- 增加 Solo 模型目录超时、消费明细保留上限和结算重试等配置项。

## 验证

- 通过模型目录、Solo 请求解析、Tool 调用与消费记录相关测试。
- GitHub Actions 会在本标签上运行完整测试，并构建 `linux/amd64`、
  `linux/arm64` 的 GHCR 镜像。

## 使用

```bash
docker pull ghcr.io/autumnsentiment/trae2api-cn:1.0.4
docker compose -f docker-compose.image.yml up -d
```

升级已有部署前请备份 `.env` 与 `data/`，然后执行：

```bash
docker compose -f docker-compose.image.yml pull
docker compose -f docker-compose.image.yml up -d
```
