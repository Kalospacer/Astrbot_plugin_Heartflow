# Heartflow 插件使用文档

本插件是“接管版群聊记忆 + Heartflow 主动回复”单插件方案。

它做两件事：
1. 接管群聊上下文记忆（含图片记忆策略）。
2. 用小模型判断是否主动回复，并触发主模型回复。

## 核心行为

1. 首次出现的图片：必须原生进入 `req.image_urls`（多模态直读）。
2. 第二次及以后：走文本记忆，格式为 `[Image]` 或 `[Image: xxx]`。
3. 主动回复：只由 Heartflow 评分触发，不走随机兜底。

## 重要前置设置

为了避免原版和插件双重注入，建议关闭 AstrBot 内置 LTM 两个开关：

1. `provider_ltm_settings.group_icl_enable = false`
2. `provider_ltm_settings.active_reply.enable = false`

如果你没关，本插件会 `WARN` 提示，但不会阻断运行。

## 安装与启用

1. 将插件放到 AstrBot 插件目录并加载。
2. 在插件配置里至少设置：
   - `enable_heartflow = true`
   - `judge_provider_name = <你的判断模型提供商ID>`
3. 如需接管群聊记忆，确保：
   - `enable_group_context = true`（默认即 true）
4. 重启 AstrBot。

## 配置说明

### 群聊记忆接管配置

- `enable_group_context`：启用群聊上下文接管（默认 `true`）
- `group_message_max_cnt`：群聊记忆最大消息数（默认 `300`）
- `history_message_window`：图片候选扫描窗口（默认 `5`）
- `max_native_images_per_round`：每轮最多原生注入图片数（默认 `2`）
- `pending_max_wait_rounds`：待注入图片最大等待轮数（默认 `2`）
- `image_caption`：复见图片是否转述成 `[Image: xxx]`（默认 `false`）
- `image_caption_provider_id`：图片转述模型提供商 ID（默认空）
- `image_caption_prompt`：图片转述提示词

### Heartflow 主动回复配置

- `enable_heartflow`：启用主动回复（默认 `false`）
- `judge_provider_name`：判断模型提供商 ID（必填）
- `reply_threshold`：主动回复阈值（默认 `0.6`）
- `min_reply_interval_seconds`：最短主动回复间隔秒数（默认 `0`）
- `context_messages_count`：判断模型使用的历史条数（默认 `5`）
- `judge_context_count`：传给判断模型的上下文条数（默认 `10`）
- `judge_max_retries`：判断 JSON 解析失败重试次数（默认 `3`）
- `judge_include_reasoning`：是否输出判断理由（默认 `true`）

### 白名单配置

- `whitelist_enabled`：启用白名单（默认 `false`）
- `chat_whitelist`：允许触发主动回复的会话 SID 列表

## 管理命令

- `/heartflow`：查看当前会话状态（含记忆接管状态）
- `/heartflow_reset`：清空当前会话状态（主动回复状态 + 图片记忆状态）
- `/heartflow_cache`：查看系统提示词缓存
- `/heartflow_cache_clear`：清除系统提示词缓存

## 工作机制（图片相关）

1. `on_group_message`
   - 记录群聊结构化消息（文本/At/图片）。
   - 再进行 Heartflow 主动回复判断。

2. `on_llm_request`
   - 注入群聊历史文本。
   - 从最近 `history_message_window` 条消息提取图片。
   - 执行注入策略：
     - 每轮最多注入 `max_native_images_per_round` 张未见图片。
     - `pending` 与 `new` 各占配额，奇数额外名额给 `pending`。
     - 超过 `pending_max_wait_rounds` 的待注入图片淘汰。

3. `on_llm_response`
   - 仅在响应后将本轮注入图片标记为 `seen`。
   - 若启用 `image_caption`，异步生成摘要。

## 快速验证

按以下顺序验证最关键路径：

1. 群里先发图片，再发唤醒词：
   - 预期：首轮图片进入 `image_urls`。
2. 下一轮继续聊同图：
   - 预期：不再原生注入，历史显示 `[Image]` 或 `[Image: xxx]`。
3. 同一窗口多图：
   - 预期：遵循每轮上限和 pending 淘汰规则。

建议关注日志：

- `Heartflow judge | ...`：主动回复评分结果
- `Heartflow LTM | chat=... | inject=... pending=... dropped=...`：图片注入决策
- `Heartflow LTM 接管模式检测到原版开关仍开启...`：原版冲突告警

## 常见问题

1. 群聊看不到图，但私聊能看
   - 检查是否命中了“首轮原生注入”路径（看 `Heartflow LTM` 日志中的 `inject`）。
   - 检查 `max_native_images_per_round` 是否过小。

2. 图片一直是 `[Image]`，没有 `[Image: xxx]`
   - 确认 `image_caption = true`
   - 确认 `image_caption_provider_id` 已配置且可用

3. 主动回复不触发
   - 确认 `enable_heartflow = true`
   - 确认 `judge_provider_name` 有效
   - 检查白名单是否限制了当前会话
   - 检查 `reply_threshold` 是否过高

4. 出现 handler 参数不匹配报错
   - 当前版本的 `on_llm_request/on_llm_response/on_group_message` 已使用兼容签名（含 `*args, **kwargs`）。
   - 若仍报错，确认运行的是最新插件代码。
