# 心流插件（Heartflow）

基于双模型架构的 AstrBot 群聊主动回复插件。判断模型负责评估当前消息是否值得参与，大模型继续使用 AstrBot 当前会话的提供商和人格生成回复。判断引擎可选：小参数 LLM（默认）或 TypeSafe Jev（结构化判断，v2.4.0 新增）。

## 功能

- 使用独立的小模型或 Jev 从内容、意愿、社交、时机和连贯性五个维度评分
- 按群隔离上下文、精力、冷却和统计状态
- 保存普通群聊和直接 @ 机器人的对话，自动排除命令消息
- 使用每群异步锁，避免并发消息同时突破冷却限制
- 将判断规则放在系统提示词中，并严格校验模型返回的 JSON
- 支持群聊白名单、评分权重、上下文数量、判断超时和最短回复间隔
- 限制群聊状态与人格摘要缓存，避免长期运行后无界占用内存

## Jev 判断引擎（judge_mode=jev）

[TypeSafe Jev](https://typesafe.ai/) 是判断专用模型：一次请求并行评估五维 Score（0-4 档）加一道总判断 Noul，返回概率加权分、概率分布和每题置信度，输出结构化、无需 JSON 重试。按输入 token 计费（$0.042/Mtok），输出免费。

配置方式：

1. 将 `judge_mode` 设为 `jev`。
2. 填写 `jev_api_key`（TypeSafe 控制台获取），或设置环境变量 `TYPESAFE_API_KEY`。
3. 其余配置（阈值、权重、超时）与 LLM 模式完全通用。

可选加固项：

- `jev_use_noul_gate` + `jev_noul_gate_threshold`：五维加权过阈值之外，总判断概率也须过门槛（双保险）
- `jev_min_confidence`：任一维度置信度低于此值时不触发（0 关闭）

注意：

- 判断指令使用英文（Jev 官方英语准确率最佳），群聊内容保持中文原文。
- `judge_max_retries` 对 Jev 模式无效——Jev 返回结构化答案，没有 JSON 解析失败这回事；仅 429/529 限流会在超时预算内退避重试。
- 人格摘要（`judge_provider_name` 配置的 LLM 压缩人格）在 Jev 模式下仍复用；未配置时自动截断原始人格前 400 字作为判断参考。
- 官方模型别名 `jev-latest` 会漂移，默认锁定 `jev-1.13.0` 保证判断可复现；判断日志会记录实际响应的模型版本号。
- Jev 官方提示中文准确率目前低于英语，建议上线后观察判断日志中的各维度分数与置信度，必要时调整 `reply_threshold` 或开启 Noul 门槛。
- 国内服务器需确认到 `api.typesafe.ai` 的网络可达性；不可达时为 AstrBot 进程配置 `https_proxy`。

## 兼容性

- AstrBot `>=4.27,<5`
- 插件本身没有额外的第三方依赖

AstrBot 4.27 已经提供概率式群聊主动回复。如果使用本插件，请关闭 AstrBot 的 `provider_ltm_settings.active_reply.enable`，避免两套触发策略同时生效；插件检测到冲突时也会在日志中警告一次。

## 安装

推荐直接在 AstrBot WebUI 的插件管理页面通过仓库地址安装：

```text
https://github.com/advent259141/Astrbot_plugin_Heartflow
```

手动安装时，将仓库克隆到 AstrBot 的 `data/plugins/`：

```bash
cd AstrBot/data/plugins
git clone https://github.com/advent259141/Astrbot_plugin_Heartflow.git
```

重启 AstrBot 或在插件管理页面重载插件。

## 必要配置

1. 选择判断引擎：`judge_mode=llm` 时在 AstrBot 配置一个成本较低、响应较快的聊天模型提供商，并将 `judge_provider_name` 设置为该提供商的 ID；`judge_mode=jev` 时填 `jev_api_key`（此时 `judge_provider_name` 可选，仅用于人格摘要）。
2. 打开 `enable_heartflow`。
3. 建议先设置 `min_reply_interval_seconds`，再逐步调整回复阈值。

## 主要配置

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `enable_heartflow` | `false` | 是否启用心流主动回复 |
| `judge_provider_name` | 空 | 用于评分的小模型提供商 ID |
| `reply_threshold` | `0.6` | 综合评分达到该值才触发回复 |
| `energy_decay_rate` | `0.1` | 每次成功生成主动回复后消耗的精力 |
| `energy_recovery_rate` | `0.02` | 未回复以及每经过 5 分钟恢复的精力 |
| `context_messages_count` | `5` | 本地上下文缓冲基数 |
| `judge_context_count` | `10` | 实际传给判断模型的最近消息数 |
| `judge_timeout_seconds` | `30` | 整轮判断（含格式重试）或人格总结的超时秒数，范围 `5-120` |
| `min_reply_interval_seconds` | `0` | 两次主动触发的最短间隔；`0` 表示不限制 |
| `whitelist_enabled` | `false` | 是否仅处理白名单群聊 |
| `chat_whitelist` | `[]` | 通过 `/sid` 获取并填写完整会话 ID |
| `judge_max_retries` | `3` | 判断结果格式错误时的重试次数，范围 `0-5` |
| `max_tracked_chats` | `1000` | 内存中最多保留的群聊状态与消息缓冲数 |
| `max_persona_cache` | `100` | 内存中最多保留的人格摘要数 |
| `judge_mode` | `llm` | 判断引擎：`llm` 小参数模型 / `jev` TypeSafe Jev |
| `jev_api_key` | 空 | Jev API key；留空读环境变量 `TYPESAFE_API_KEY` |
| `jev_model` | `jev-1.13.0` | Jev 模型版本（锁版本可复现，`jev-latest` 跟随官方） |
| `jev_use_noul_gate` | `false` | 总判断 Noul 作为第二道硬门槛 |
| `jev_noul_gate_threshold` | `0.5` | Noul 门槛值（开启门槛时生效） |
| `jev_min_confidence` | `0` | 任一维度置信度下限，0 关闭 |

五项评分权重默认分别为：内容相关度 25%、回复意愿 20%、社交适宜性 20%、时机 15%、连贯性 20%。权重不必手动保证总和为 1，插件会自动归一化；全部为 0 时会回退到默认权重。

## 管理命令

以下命令仅限 AstrBot 管理员：

- `/heartflow`：查看当前群聊的精力、回复统计和配置
- `/heartflow_reset`：重置当前群聊状态和本地消息缓冲
- `/heartflow_cache`：查看人格提示词摘要缓存
- `/heartflow_cache_clear`：清空人格提示词摘要缓存

## 工作方式

1. 插件记录当前群聊消息；斜杠命令不会进入上下文。
2. 同一群聊的消息按顺序交给判断模型评分。
3. 综合评分超过阈值且不在冷却期时，插件唤醒 AstrBot 的正常大模型回复流程。
4. 只有大模型成功返回内容后，插件才扣除精力并增加回复次数；生成失败会释放本次冷却预留。
5. AstrBot 发出的普通或流式 LLM 最终回复会写回本地上下文，命令和其他非 LLM 输出不会污染记录。

人格设定、历史消息和当前消息均作为不可信数据传给判断模型；评分规则只放在系统提示词中。模型返回的五项分数必须是 JSON 数字且位于 `0-10`，否则会在总超时预算内重试。

所有内存状态会在插件重载或 AstrBot 重启后清空。

## 本地检查

```bash
python -m unittest discover -s tests -v
python -m ruff check main.py tests
python -m ruff format --check main.py tests
```

在 AstrBot 4.27 源码目录中，可使用 uv 隔离环境执行真实类型和事件钩子集成测试：

```bash
uv run --isolated --project . python ../Astrbot_plugin_Heartflow/tests/astrbot_e2e.py --plugin-root ../Astrbot_plugin_Heartflow
```

## 许可证

本插件遵循仓库中的许可证文件。
