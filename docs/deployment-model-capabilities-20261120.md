# 2026-11-20 模型能力与 1M Max 部署记录

## 变更

- 对照 Magpie Trae 插件的 live descriptor 规则，区分 `__dev` 与 `__max`。
- 只有 live catalog 明确提供更大的 `context_window_tokens.max` 和真实
  `__max` detail 时，才暴露 `<model>-max`。
- `/v1/models` 增加 live `limit`、`capabilities`、`variant_of` 等兼容性扩展；
  普通模型使用 dev context/output，Max 变体使用真实 Max profile。
- Remote 请求对 `<model>-max` 自动绑定基础 config、`__max` provider detail、
  `model_selection_strategy=max`、context/output/prompt 预算。
- TraeCode 原生解析同样识别 `-max`，按基础模型查目录并选择 `__max`。
- 不从模型名称或静态列表合成 1M、936K、64K；目录暂时为空时仍安全回落。

## 验证

- 本地全量：`1359 passed, 6 skipped, 908 subtests`
- 部署容器：`healthy`
- revision：`model-capabilities-20261120-222853-model-max`
- 容器内 `compileall`：通过
- 容器内 live profile 解析示例：
  - context: `1048576`
  - output: `64000`
  - prompt: `984576`
  - base context: `200000`

## 回滚

- 部署前镜像备份：`trae-cn-relay:pre-20261120-222853-model-max`
- 生产 `.env`、`data/` 和账号数据未改动。

首次尝试的普通 Docker build 因远端基础镜像 registry 返回 `401 Unauthorized`
未完成，脚本已自动回滚；随后使用当前运行镜像复制已测试 Python 源码并提交候选镜像，
避免重新拉取基础镜像。
