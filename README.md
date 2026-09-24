# 心流插件（Heartflow）

基于双模型架构的 AstrBot 群聊主动回复插件。判断模型负责评估当前消息是否值得参与，大模型继续使用 AstrBot 当前会话的提供商和人格生成回复。判断引擎可选：小参数 LLM（默认）或 TypeSafe Jev（结构化判断，v2.4.0 新增）。

## 功能

- 使用独立的小模型或 Jev 从内容、意愿、社交、时机和连贯性五个维度评分
- 按群隔离上下文、精力、冷却和统计状态
- 保存普通群聊和直接 @ 机器人的对话，自动排除命令消息
- 使用每群异步锁，避免并发消息同时突破冷却限制；判断期间到达的消息合并进下一批，不会逐条积压
- 触发时把判断看到的群聊记录作为本轮临时内容块（`extra_user_content_parts`，不写历史）交给主 LLM，系统提示词保持不变、不破坏 KV 缓存
- 将判断规则放在系统提示词中，并严格校验模型返回的 JSON
- 支持群聊白名单、评分权重、上下文数量、判断超时和最短回复间隔
- 可选的判断防抖：刷屏时把一个窗口内的消息合成一次判断，省调用也不打断别人说话
- 限制群聊状态数量，避免长期运行后无界占用内存
- 判断器默认拿人格全文；开启 `compress_persona` 后改用压缩人格，只压缩一次并写入配置 `compressed_persona`，每次判断看到的人格都一样，可查看和修改（两种引擎共用）

## Jev 判断引擎（judge_mode=jev）

[TypeSafe Jev](https://typesafe.ai/) 是判断专用模型，按输入 token 计费（$0.042/Mtok），输出免费。`jev_decision_mode` 二选一，每种模式只发自己要用的题：

- `noul`（默认）：只问一道总判断 Noul，Jev 返回校准过的"该不该回复"概率，直接和 `reply_threshold` 比较。
- `score`：只问五维 Score（内容、意愿、社交、时机、连贯，各 0-4 档），折算成 0-10 分后按评分权重加权成 0-1 综合分，和 `reply_threshold` 比较。

配置方式：

1. 将 `judge_mode` 设为 `jev`。
2. 填写 `jev_api_key`（TypeSafe 控制台获取），或设置环境变量 `TYPESAFE_API_KEY`。
3. 选择 `jev_decision_mode`；`reply_threshold` 在 noul 模式下是概率阈值，在 score 模式下是加权综合分阈值，建议从 0.6 起步观察日志调整。

注意：

- 判断指令使用英文（Jev 官方英语准确率最佳），群聊内容保持中文原文。
- `judge_max_retries` 两种模式共用：`llm` 模式重试的是无效 JSON，`jev` 模式重试的是 429/529 限流（在超时预算内退避）。Jev 返回结构化答案，不存在 JSON 解析失败。
- 六道判断题的题干和档位标准在配置 `jev_prompts` 里可改（题目键名固定，Score 题必须 5 档）。
- 日志级别为 DEBUG 时，每次判断都会打印 `Jev 请求体: {...}`，即发给 Jev 的完整 JSON（不含 API key）。
- 官方模型别名 `jev-latest` 会漂移，默认锁定 `jev-1.13.0` 保证判断可复现；判断日志会记录实际响应的模型版本号。
- Jev 官方提示中文准确率目前低于英语，建议上线后观察判断日志（noul 模式含 `noul=`，score 模式含五维分数与 `min_conf=`），必要时调整 `reply_threshold`。
- 判断结果里的 `overall_score` 统一表示"实际参与阈值比较的那个数"：`llm` 模式是五维加权综合分，`jev` 的 noul 模式是 noul 概率、score 模式是五维加权综合分，所以触发日志的 `评分:` 在两种模式下都能直接和阈值对照。
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

1. 选择判断引擎：`judge_mode=llm` 时在 AstrBot 配置一个成本较低、响应较快的聊天模型提供商，并将 `judge_provider_name` 设置为该提供商的 ID；`judge_mode=jev` 时填 `jev_api_key`（此时 `judge_provider_name` 可选，仅在开启 `compress_persona` 且 `compressed_persona` 为空时用于压缩人格）。
2. 打开 `enable_heartflow`。
3. 建议先设置 `min_reply_interval_seconds`，再逐步调整回复阈值。

## 主要配置

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `enable_heartflow` | `false` | 是否启用心流主动回复 |
| `judge_provider_name` | 空 | 用于评分的小模型提供商 ID |
| `compress_persona` | `false` | 判断器是否使用压缩人格；关闭时拿全文（两种引擎共用） |
| `compressed_persona` | 空 | 压缩人格；开启压缩且为空时自动压缩一次并写回，之后固定使用 |
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
| `debounce_seconds` | `0` | 判断防抖窗口秒数，`0` 关闭（逐条判断），范围 `0-60` |
| `judge_mode` | `llm` | 判断引擎：`llm` 小参数模型 / `jev` TypeSafe Jev |
| `jev_api_key` | 空 | Jev API key；留空读环境变量 `TYPESAFE_API_KEY` |
| `jev_model` | `jev-1.13.0` | Jev 模型版本（锁版本可复现，`jev-latest` 跟随官方） |
| `jev_decision_mode` | `noul` | Jev 判断模式：`noul` 总判断概率 / `score` 五维加权 |
| `jev_prompts` | 内置六题 | Jev 判断题的题干与档位标准 |

五项评分权重默认分别为：内容相关度 25%、回复意愿 20%、社交适宜性 20%、时机 15%、连贯性 20%。权重不必手动保证总和为 1，插件会自动归一化；全部为 0 时会回退到默认权重。权重在 `judge_mode=llm` 和 Jev 的 score 模式下参与触发决策；Jev 的 noul 模式不使用权重。

## 判断防抖（debounce_seconds）

默认 `0`：来一条判断一条；判断进行中又到达的消息会攒成一批，等这轮结束后由最新那条带整批判断一次，不逐条补判。设成正数后：

1. 第一条新消息开一个窗口，窗口长度就是 `debounce_seconds`，**从开窗那刻算起，不会被后续消息续期**。
2. 窗口内到达的消息依次攒进同一批。
3. 窗口结束时，整批消息一起交给判断模型，放在 `current_messages` 字段里（按时间顺序）；本批之前的历史消息仍在 `chat_log` 里当背景。

所以一个窗口只产生一次判断调用，而且模型看到的是**这一整段话**，不是只看最后一条。

举例：窗口 15 秒，期间来了 1~5 号消息，五条一起进 `current_messages`。哪怕点名机器人的是 2 号、5 号只是「哈哈」，判断依然能看到 2 号。

注意：

- 机器人发言会比开窗那条晚至多 `debounce_seconds` 秒，这是防抖的固有代价。建议从 `3`~`8` 秒起步。
- 直接 @ 机器人的消息由 AstrBot 核心唤醒接管，根本不进心流判断，因此**不受防抖影响**，@ 了仍是秒回。
- 批次由窗口长度决定，不受 `judge_context_count` 限制；后者只管 `chat_log` 里的背景消息条数。
- 触发时由窗口内最后那条消息把事件标记成唤醒，所以主 LLM 回复的是最新那条的语境——判断用整批，回复贴最新，两边都合理。
- 等待发生在群聊锁之外，每条消息在 AstrBot 里本就是独立任务，因此不会阻塞其他群或其他消息。

## 管理命令

以下命令仅限 AstrBot 管理员：

- `/heartflow`：查看当前群聊的精力、回复统计和配置
- `/heartflow_reset`：重置当前群聊状态和本地消息缓冲
- `/heartflow_cache`：查看当前的压缩人格（`compressed_persona`）
- `/heartflow_cache_clear`：清空压缩人格，下次判断重新压缩一次

## 工作方式

1. 插件记录当前群聊消息；斜杠命令不会进入上下文。
2. 同一群聊的消息按顺序交给判断模型评分。
3. 综合评分超过阈值且不在冷却期时，插件唤醒 AstrBot 的正常大模型回复流程。
4. 只有大模型成功返回内容后，插件才扣除精力并增加回复次数；生成失败会释放本次冷却预留，并丢弃要发进群的报错消息（流式输出的报错走流式通道，拦不到）。
5. AstrBot 发出的普通或流式 LLM 最终回复会写回本地上下文，命令和其他非 LLM 输出不会污染记录。被 @ 后的回复同样计入「距上次回复」和最短回复间隔。

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
