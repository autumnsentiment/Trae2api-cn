# Trae2api-cn v1.0.2

## 更新内容

- 账号与签到页新增每账号“模型请求”开关，默认开启并持久化。关闭只影响模型请求和轮询重试，不影响手动或每日签到。
- 全部模型账号关闭时返回明确的 503；已关闭账号绑定的会话不再允许下一轮请求，当前输出不强制取消。
- 修复 Issue #6：共享 URL 的 IDE Agent / IDE Raw 和 Remote / Work Agent 预设按模式区分，刷新与再次保存不再切错端点。
- 启动时恢复持久化的上游模式和 URL，容器重建不再覆盖控制台选择；局部保存不删除其它环境设置。
- Relay 服务端口与上游端点分开保存，明确宿主机映射端口和容器监听端口区别。
- 消费记录新增思考强度、上下文模式和工具参数状态，区分请求值与上游实际应用值、工具定义与实际工具调用。
- 更新实时模型列表，规范化小写模型名并去重。
- IDE Agent 保留原生工具定义、调用 ID 和结果关联；修复工具增量拼装、空响应回退边界及流式取消时的会话资源回收。
- 新增 GitHub Actions 发布流水线：自动测试、源码包与 SHA256 校验文件，以及 linux/amd64、linux/arm64 的 GHCR 镜像。
- 根据 Issue #8 的实测结论，IDE Raw 不再作为正式控制台/模型测试预设发布。
  个人账号调用 `/api/ide/v2/llm_raw_chat` 会固定返回
  `2001 failed to get app config: record not found`；该路径需要企业版
  app config/PAT。内部 `UPSTREAM_MODE=raw` 与 `raw_client.py` 仍保留用于协议诊断。

## 直接拉取镜像

```bash
docker pull ghcr.io/autumnsentiment/trae2api-cn:1.0.2
docker compose -f docker-compose.image.yml up -d
```

预构建 Compose 不要求源码构建，也不要求现有 new-api 外部网络。可把
`docker-compose.image.yml` 的镜像标签固定为 `1.0.2`，或用 `latest` 跟随正式版。
保留原有 `.env` 和 `data/` 挂载，更新前备份账号存储。

## 使用注意

- 模型输出工具调用后，必须由调用端执行并回传工具结果。Relay 不会访问调用端的本地文件系统。
- 个人账号的 IDE Raw 可能返回 `2001 app config record not found`；正式控制台已移除该预设，请选择 IDE Agent / Solo / Remote 或打开自动路由。
- 端口映射变化需重新创建容器，单纯 `docker restart` 不会改变 Docker 映射。
- 仅建议内网部署；源码包不包含 `.env`、账号数据、诊断文件或备份。
