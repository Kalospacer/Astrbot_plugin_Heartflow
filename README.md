# Heartflow

AstrBot 群聊主动回复插件：判断器决定要不要插话，命中后把事件标记为唤醒，交给 AstrBot 主 LLM 按当前会话的提供商和人格回复。

## 安装

- AstrBot `>=4.27,<5`，依赖 `httpx`
- WebUI 插件管理页 → 通过仓库地址安装：`https://github.com/Kalospacer/Astrbot_plugin_Heartflow`
- 手动：`cd AstrBot/data/plugins && git clone https://github.com/Kalospacer/Astrbot_plugin_Heartflow.git`，然后在插件管理页重载
- 关闭 AstrBot 内置主动回复 `provider_ltm_settings.active_reply.enable`，否则两套策略会同时触发（插件检测到冲突时会警告一次）

## 最小配置

- LLM 判断：`enable_heartflow=true`，`judge_mode=llm`，`judge_provider_name=<低成本聊天模型提供商 ID>`
- Jev 判断：`enable_heartflow=true`，`judge_mode=jev`，`jev_api_key=<key>`（或环境变量 `TYPESAFE_API_KEY`）

## 判断引擎

| | `llm` | `jev` + `noul` | `jev` + `score` |
| --- | --- | --- | --- |
| 发出的内容 | 系统提示词 + JSON 用户消息，要求返回五维 0-10 分 | 1 道 Noul 题 `should_reply` | 5 道 Score 题（0-4 档） |
| 和 `reply_threshold` 比较的数 | 五维加权综合分 | Noul 概率 | 五维折算 0-10 后的加权综合分 |
| 评分权重 | 生效 | 不使用 | 生效 |
| `judge_max_retries` 重试的是 | 无效 JSON | 429/529 限流、网络错误 | 同左 |
| 判断失败 | 本条不触发 | 本条不触发，打 warning | 同左 |

- 判断器看到的输入：人格、精力、距上次回复分钟数、群聊摘要（近 10 分钟活跃度、历史回复率、当前时间）、`chat_log`（本批之前的最近 `judge_context_count` 条）、机器人上次回复、`current_messages`（本批新消息）
- Jev 请求：`POST {jev_base_url}/v1/systemone`，body 为 `{"model", "state", "questions"}`；日志级别为 DEBUG 时打印整个请求体 `Jev 请求体: {...}`（不含 API key）
- Jev 判断题的题干与档位标准在 `jev_prompts` 里编辑；题目键名固定，Score 题必须正好 5 条标准

## 配置

### 通用

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `enable_heartflow` | `false` | 总开关 |
| `judge_mode` | `llm` | `llm` / `jev` |
| `reply_threshold` | `0.6` | 触发阈值，`0-1`；比较对象见上表 |
| `min_reply_interval_seconds` | `0` | 距上次触发或上次回复（含 @ 回复）的最短秒数，`0` 不限 |
| `judge_timeout_seconds` | `30` | 整轮判断（含重试）或单次人格压缩的超时，`5-120` |
| `judge_max_retries` | `3` | `0-5`，重试内容见上表 |
| `judge_context_count` | `10` | `chat_log` 条数，`1-100` |
| `context_messages_count` | `5` | 本地消息缓冲大小 = `max(本项, judge_context_count) × 4` |
| `debounce_seconds` | `0` | 防抖窗口秒数，`0-60`，见「批次与防抖」 |
| `energy_decay_rate` | `0.1` | 每次主动回复成功后扣除的精力，精力下限 `0.1` |
| `energy_recovery_rate` | `0.02` | 每次判断不通过时恢复的精力，同时按每 5 分钟恢复这么多；每天首次访问额外 +0.2 |
| `whitelist_enabled` | `false` | 只处理白名单群 |
| `chat_whitelist` | `[]` | 完整会话 ID（`/sid` 获取）；开启白名单但列表为空时不处理任何群 |
| `max_tracked_chats` | `1000` | 内存中保留的群状态数，超出后淘汰最久未用、且不在处理中的群 |

### 人格

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `compress_persona` | `false` | 关闭：判断器拿人格全文；开启：用 `compressed_persona` |
| `compressed_persona` | 空 | 压缩人格，可手动编辑；开启压缩且本项为空时，用 `judge_provider_name` 压缩一次并写回本项 |
| `judge_provider_name` | 空 | LLM 判断的模型；也是自动压缩人格所用的模型 |

### LLM 引擎

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `judge_relevance` | `0.25` | 权重：内容相关度 |
| `judge_willingness` | `0.2` | 权重：回复意愿 |
| `judge_social` | `0.2` | 权重：社交适宜性 |
| `judge_timing` | `0.15` | 权重：时机 |
| `judge_continuity` | `0.2` | 权重：连贯性 |
| `judge_include_reasoning` | `true` | 要求判断模型附带 `reasoning` |

权重同样作用于 Jev 的 score 模式。总和不为 1 时自动归一化，全部为 0 时回退默认值。

### Jev 引擎

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `jev_api_key` | 空 | 为空时读环境变量 `TYPESAFE_API_KEY` |
| `jev_base_url` | `https://api.typesafe.ai` | 国内不可达时可换成反代地址 |
| `jev_model` | `jev-1.13.0` | 锁定版本，保证判断可复现；`jev-latest` 会漂移 |
| `jev_decision_mode` | `noul` | `noul` / `score`，每种模式只发自己的题 |
| `jev_prompts` | 内置英文题目 | 各题的 `instructions`、`criteria` / `criteria_true` / `criteria_false` |

## 批次与防抖

- 每个群两段持锁：第一段记录消息并放进待判断批次，第二段只由批次里最新的那条消息去判断整批
- `debounce_seconds=0`：判断进行中到达的消息合成下一批，积压最多一轮
- `debounce_seconds>0`：窗口从第一条消息开始计时，后来的消息不会让窗口续期，窗口内的消息都进同一批
- 整批放进 `current_messages`，不受 `judge_context_count` 限制
- 防抖等待不占锁；判断期间持有群锁，同一个群的新消息要等判断结束才能被记录

## 管理命令

仅管理员可用：

- `/heartflow`：当前群的精力、统计、引擎、判断模式、人格压缩状态、权重
- `/heartflow_reset`：清空当前群的状态和消息缓冲
- `/heartflow_cache`：查看 `compressed_persona`
- `/heartflow_cache_clear`：清空 `compressed_persona`，下次判断重新压缩

## 插件互通

心流触发的事件会带上这些 extras：

`event.get_extra("heartflow_triggered")  # bool`

`event.get_extra("heartflow_judge_result")  # JudgeResult: relevance/willingness/social/timing/continuity(0-10), overall_score, reasoning, should_reply`

`event.get_extra("heartflow_batch")  # list[RawMessage]: sender_name, sender_id, content, timestamp, is_bot`

`event.get_extra("heartflow_trigger_time")  # float，本次触发占用的冷却时间戳`

## 注意事项

- 精力不会直接拦截回复，只作为判断器的输入；noul 模式的题目不看精力
- @ 机器人或唤醒词触发的消息不经过心流判断，也不受防抖影响；它们仍会写入缓冲，其回复计入冷却和「距上次回复」
- 斜杠命令既不判断也不写入缓冲
- 触发时，群聊记录和本批消息以临时内容块 `extra_user_content_parts`（`mark_as_temp`）交给主 LLM：不改 `system_prompt`，也不写入会话历史
- 主动回复失败时回滚冷却，并丢弃要发进群的报错；流式输出的报错走流式通道，拦不到
- 只有主 LLM 成功返回后，才扣精力、计回复数
- `compressed_persona` 只有一份，所有人格共用；人格原文改了不会自动更新，需要清空后重新压缩
- 群状态和消息缓冲在重载后清空；`compressed_persona` 保存在配置里，重载后保留

## 本地检查

```bash
python -m unittest discover -s tests -v
python -m ruff check main.py tests
python -m ruff format --check main.py tests
```

AstrBot 源码目录下的集成测试：

```bash
uv run --isolated --project . python ../Astrbot_plugin_Heartflow/tests/astrbot_e2e.py --plugin-root ../Astrbot_plugin_Heartflow
```

## 许可证

AGPL-3.0
