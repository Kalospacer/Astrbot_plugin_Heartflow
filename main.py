import asyncio
import datetime
import hashlib
import json
import re
import time
import traceback
import uuid
from collections import deque
from typing import Deque, Dict
from dataclasses import dataclass, field

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger
from astrbot.api.message_components import At, Image, Plain
from astrbot.api.platform import MessageType


@dataclass
class JudgeResult:
    """判断结果数据类"""

    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0  # 新增：与上次回复的连贯性
    reasoning: str = ""
    should_reply: bool = False
    confidence: float = 0.0
    overall_score: float = 0.0
    related_messages: list = None

    def __post_init__(self):
        if self.related_messages is None:
            self.related_messages = []


@dataclass
class RawMessage:
    """原始群聊消息条目"""

    sender_name: str
    sender_id: str
    content: str
    timestamp: float
    is_bot: bool = False


@dataclass
class ChatState:
    """群聊状态数据类"""

    energy: float = 1.0
    last_reply_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0
    last_energy_update: float = 0.0


@dataclass
class GroupPart:
    """群聊结构化消息片段"""

    kind: str  # text | at | image
    text: str = ""
    image_key: str = ""
    image_url: str = ""


@dataclass
class GroupMessageRecord:
    """群聊结构化消息"""

    sender_name: str
    timestamp: str
    parts: list[GroupPart] = field(default_factory=list)


@dataclass
class SeenImageState:
    """图片已见状态"""

    url: str
    seen: bool = False
    seen_round: int = 0
    summary_status: str = "none"  # none | pending | ready | failed
    summary_text: str = ""


@dataclass
class PendingImage:
    """待注入的未见图片"""

    image_key: str
    image_url: str
    wait_rounds: int = 0


def _extract_json(text: str) -> dict:
    """从模型返回的文本中稳健地提取 JSON 对象。

    依次尝试：
    1. 直接解析
    2. 去除 markdown 代码块后解析
    3. 正则提取第一个 {...} 子串后解析
    """
    text = text.strip()

    # 1. 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 去除 markdown 代码块
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 正则提取最外层 {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"无法从文本中提取有效 JSON: {text[:200]}")


def _clamp_score(v) -> float:
    """将模型返回的分数值钉位到 [0, 10]。"""
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


class GroupMemoryEngine:
    """替代版群聊记忆引擎：记录消息、注入上下文、首见图片原生注入、响应后记账。"""

    def __init__(self, context: star.Context, plugin_config: dict):
        self.context = context
        self.plugin_config = plugin_config

        self.session_records: Dict[str, Deque[GroupMessageRecord]] = {}
        self.seen_images: Dict[str, Dict[str, SeenImageState]] = {}
        self.pending_queue: Dict[str, Deque[PendingImage]] = {}
        self.inflight_rounds: Dict[str, tuple[str, int, list[str]]] = {}
        self.session_round_counter: Dict[str, int] = {}
        self.warned_builtin_ltm_at: Dict[str, float] = {}

    def _cfg_bool(self, key: str, default: bool) -> bool:
        return bool(self.plugin_config.get(key, default))

    def _cfg_int(self, key: str, default: int, minimum: int = 0) -> int:
        try:
            value = int(self.plugin_config.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    def _cfg_str(self, key: str, default: str) -> str:
        val = self.plugin_config.get(key, default)
        return str(val) if val is not None else default

    def _enabled(self) -> bool:
        return self._cfg_bool("enable_group_context", True)

    def _group_message_max_cnt(self) -> int:
        return self._cfg_int("group_message_max_cnt", 300, minimum=1)

    def _history_window(self) -> int:
        return self._cfg_int("history_message_window", 5, minimum=1)

    def _max_native_images(self) -> int:
        return self._cfg_int("max_native_images_per_round", 2, minimum=0)

    def _pending_max_wait_rounds(self) -> int:
        return self._cfg_int("pending_max_wait_rounds", 2, minimum=0)

    def _image_caption_enabled(self) -> bool:
        return self._cfg_bool("image_caption", False) and bool(
            self._cfg_str("image_caption_provider_id", "").strip()
        )

    def _image_caption_provider_id(self) -> str:
        return self._cfg_str("image_caption_provider_id", "").strip()

    def _image_caption_prompt(self) -> str:
        return self._cfg_str(
            "image_caption_prompt",
            "Please describe the image using Chinese.",
        )

    def warn_builtin_ltm_enabled(
        self,
        event: AstrMessageEvent | None = None,
        force: bool = False,
    ) -> None:
        """若原版 LTM 开关未关闭，仅告警不阻断。"""
        try:
            if event:
                cfg = self.context.get_config(umo=event.unified_msg_origin)
                key = event.unified_msg_origin
            else:
                cfg = self.context.get_config()
                key = "__global__"
        except Exception:
            return

        ltm_cfg = cfg.get("provider_ltm_settings", {})
        group_icl_enabled = bool(ltm_cfg.get("group_icl_enable", False))
        active_reply_enabled = bool(
            ltm_cfg.get("active_reply", {}).get("enable", False)
        )
        if not (group_icl_enabled or active_reply_enabled):
            return

        now = time.time()
        if not force:
            last_warn = self.warned_builtin_ltm_at.get(key, 0.0)
            if now - last_warn < 300:
                return
        self.warned_builtin_ltm_at[key] = now
        logger.warning(
            "Heartflow LTM 接管模式检测到原版开关仍开启: group_icl_enable=%s active_reply.enable=%s。建议关闭内置 LTM，避免双重注入。",
            group_icl_enabled,
            active_reply_enabled,
        )

    def _get_or_create_records(self, umo: str) -> Deque[GroupMessageRecord]:
        max_cnt = self._group_message_max_cnt()
        records = self.session_records.get(umo)
        if records is None:
            records = deque(maxlen=max_cnt)
            self.session_records[umo] = records
            return records
        if records.maxlen != max_cnt:
            records = deque(list(records)[-max_cnt:], maxlen=max_cnt)
            self.session_records[umo] = records
        return records

    def _image_key(self, image_url: str) -> str:
        normalized = (image_url or "").strip()
        return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()

    def record_user_message(self, event: AstrMessageEvent) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        parts: list[GroupPart] = []
        for comp in event.get_messages():
            if isinstance(comp, Plain):
                if comp.text:
                    parts.append(GroupPart(kind="text", text=comp.text))
            elif isinstance(comp, At):
                parts.append(GroupPart(kind="at", text=comp.name or ""))
            elif isinstance(comp, Image):
                image_url = comp.url or comp.file or ""
                if image_url:
                    parts.append(
                        GroupPart(
                            kind="image",
                            image_key=self._image_key(image_url),
                            image_url=image_url,
                        )
                    )
        if not parts and event.message_str:
            parts.append(GroupPart(kind="text", text=event.message_str))
        if not parts:
            return

        self._get_or_create_records(event.unified_msg_origin).append(
            GroupMessageRecord(
                sender_name=event.get_sender_name() or "Unknown",
                timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
                parts=parts,
            )
        )

    def record_bot_reply(self, event: AstrMessageEvent, reply_text: str) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        reply_text = (reply_text or "").strip()
        if not reply_text:
            return
        self._get_or_create_records(event.unified_msg_origin).append(
            GroupMessageRecord(
                sender_name="You",
                timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
                parts=[GroupPart(kind="text", text=reply_text)],
            )
        )

    def clear_session(self, umo: str) -> dict:
        info = {
            "records": len(self.session_records.get(umo, [])),
            "seen_images": len(self.seen_images.get(umo, {})),
            "pending": len(self.pending_queue.get(umo, [])),
        }
        self.session_records.pop(umo, None)
        self.seen_images.pop(umo, None)
        self.pending_queue.pop(umo, None)
        self.session_round_counter.pop(umo, None)
        deleting = [
            rid for rid, (sid, _, _) in self.inflight_rounds.items() if sid == umo
        ]
        for rid in deleting:
            self.inflight_rounds.pop(rid, None)
        return info

    def _render_seen_image_token(self, state: SeenImageState) -> str:
        if not self._image_caption_enabled():
            return "[Image]"
        if state.summary_status == "ready" and state.summary_text.strip():
            return f"[Image: {state.summary_text.strip()}]"
        return "[Image]"

    def _build_chat_history_text(self, event: AstrMessageEvent) -> str:
        umo = event.unified_msg_origin
        records = self.session_records.get(umo)
        if not records:
            return ""
        seen_map = self.seen_images.setdefault(umo, {})
        lines: list[str] = []
        for rec in records:
            parts = [f"[{rec.sender_name}/{rec.timestamp}]:"]
            for part in rec.parts:
                if part.kind == "text":
                    parts.append(f" {part.text}")
                elif part.kind == "at":
                    parts.append(f" [At: {part.text}]")
                elif part.kind == "image":
                    state = seen_map.get(part.image_key)
                    if not state or not state.seen:
                        continue
                    parts.append(f" {self._render_seen_image_token(state)}")
            line = "".join(parts).strip()
            if line:
                lines.append(line)
        return "\n---\n".join(lines)

    def _collect_candidate_images(self, umo: str) -> list[tuple[str, str]]:
        records = list(self.session_records.get(umo, []))
        records = records[-self._history_window() :]
        candidates: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        for rec in reversed(records):
            for part in reversed(rec.parts):
                if part.kind != "image" or not part.image_key or not part.image_url:
                    continue
                if part.image_key in seen_keys:
                    continue
                seen_keys.add(part.image_key)
                candidates.append((part.image_key, part.image_url))
        return candidates

    def _select_images_for_round(
        self,
        umo: str,
        candidates: list[tuple[str, str]],
    ) -> tuple[list[str], list[str], list[str]]:
        max_native = self._max_native_images()
        if max_native <= 0:
            return [], [], []

        seen_map = self.seen_images.setdefault(umo, {})
        queue = self.pending_queue.setdefault(umo, deque())
        existing_pending = {item.image_key for item in queue}

        candidate_map: dict[str, str] = {}
        unseen_keys: list[str] = []
        unseen_set: set[str] = set()
        for key, url in candidates:
            candidate_map.setdefault(key, url)
            state = seen_map.get(key)
            if state and state.seen:
                continue
            if key not in unseen_set:
                unseen_set.add(key)
                unseen_keys.append(key)
        for key in unseen_keys:
            if key not in existing_pending:
                queue.append(
                    PendingImage(image_key=key, image_url=candidate_map.get(key, ""))
                )

        pending_candidates = [
            item.image_key for item in queue if item.image_key in unseen_set
        ]
        new_candidates = [key for key in unseen_keys if key not in existing_pending]

        pending_quota = (max_native + 1) // 2  # 奇数额外名额给 pending
        new_quota = max_native // 2

        selected: list[str] = []
        selected.extend(pending_candidates[:pending_quota])
        for key in new_candidates:
            if len([k for k in selected if k in new_candidates]) >= new_quota:
                break
            if key not in selected:
                selected.append(key)

        remain = max_native - len(selected)
        if remain > 0:
            for key in pending_candidates:
                if key in selected:
                    continue
                selected.append(key)
                remain -= 1
                if remain <= 0:
                    break
        if remain > 0:
            for key in new_candidates:
                if key in selected:
                    continue
                selected.append(key)
                remain -= 1
                if remain <= 0:
                    break

        selected_set = set(selected)
        max_wait = self._pending_max_wait_rounds()
        new_queue: Deque[PendingImage] = deque()
        dropped: list[str] = []
        for item in queue:
            if item.image_key in selected_set:
                continue
            state = seen_map.get(item.image_key)
            if state and state.seen:
                continue
            item.wait_rounds += 1
            if item.wait_rounds > max_wait:
                dropped.append(item.image_key)
                continue
            new_queue.append(item)
        self.pending_queue[umo] = new_queue

        selected_urls: list[str] = []
        for key in selected:
            url = candidate_map.get(key, "")
            if not url:
                continue
            selected_urls.append(url)
            seen_map.setdefault(key, SeenImageState(url=url)).url = url
        return selected, selected_urls, dropped

    async def apply_on_request(
        self, event: AstrMessageEvent, req, heartflow_triggered: bool
    ) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        self.warn_builtin_ltm_enabled(event)
        umo = event.unified_msg_origin
        history_text = self._build_chat_history_text(event)

        if history_text:
            if heartflow_triggered and hasattr(req, "prompt"):
                prompt = req.prompt or ""
                req.prompt = (
                    "You are now in a chatroom. The chat history is as follows:\n"
                    f"{history_text}"
                    f"\nNow, a new message is coming: `{prompt}`. "
                    "Please react to it. Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                )
                if hasattr(req, "contexts"):
                    req.contexts = []
            elif hasattr(req, "system_prompt"):
                req.system_prompt = (req.system_prompt or "") + (
                    "You are now in a chatroom. The chat history is as follows: \n"
                    f"{history_text}"
                )
            elif hasattr(req, "prompt"):
                req.prompt = (
                    (req.prompt or "")
                    + "\n\nYou are now in a chatroom. The chat history is as follows: \n"
                    + history_text
                )

        selected_keys, selected_urls, dropped = self._select_images_for_round(
            umo,
            self._collect_candidate_images(umo),
        )

        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []
        if not isinstance(req.image_urls, list):
            req.image_urls = list(req.image_urls)

        existed = set(req.image_urls)
        injected = 0
        for url in selected_urls:
            if url in existed:
                continue
            req.image_urls.append(url)
            existed.add(url)
            injected += 1

        round_idx = self.session_round_counter.get(umo, 0) + 1
        self.session_round_counter[umo] = round_idx
        round_id = uuid.uuid4().hex
        self.inflight_rounds[round_id] = (umo, round_idx, selected_keys)
        event.set_extra("_heartflow_ltm_round_id", round_id)

        logger.info(
            "Heartflow LTM | chat=%s | inject=%d pending=%d dropped=%d",
            umo,
            injected,
            len(self.pending_queue.get(umo, [])),
            len(dropped),
        )

    async def _get_image_caption(self, umo: str, image_url: str) -> str:
        provider_id = self._image_caption_provider_id()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
        else:
            provider = self.context.get_using_provider(umo)
        if not provider:
            raise RuntimeError("未找到可用的图片转述提供商")
        response = await provider.text_chat(
            prompt=self._image_caption_prompt(),
            session_id=uuid.uuid4().hex,
            image_urls=[image_url],
            persist=False,
        )
        return (response.completion_text or "").strip()

    async def _caption_image_task(
        self, umo: str, image_key: str, image_url: str
    ) -> None:
        try:
            caption = await self._get_image_caption(umo, image_url)
            state = self.seen_images.setdefault(umo, {}).get(image_key)
            if not state:
                return
            if caption:
                state.summary_text = caption
                state.summary_status = "ready"
            else:
                state.summary_status = "failed"
        except Exception as e:
            state = self.seen_images.setdefault(umo, {}).get(image_key)
            if state:
                state.summary_status = "failed"
            logger.warning(
                "Heartflow LTM 图片摘要失败 | chat=%s key=%s err=%s",
                umo,
                image_key[:8],
                e,
            )

    async def apply_on_response(self, event: AstrMessageEvent, resp) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        round_id = event.get_extra("_heartflow_ltm_round_id")
        if not round_id:
            return
        inflight = self.inflight_rounds.pop(round_id, None)
        if not inflight:
            return

        umo, round_idx, image_keys = inflight
        if resp is None:
            return
        seen_map = self.seen_images.setdefault(umo, {})
        for image_key in image_keys:
            state = seen_map.get(image_key)
            if not state:
                continue
            state.seen = True
            state.seen_round = round_idx
            if self._image_caption_enabled() and state.summary_status in (
                "none",
                "failed",
            ):
                state.summary_status = "pending"
                asyncio.create_task(self._caption_image_task(umo, image_key, state.url))

        completion_text = (getattr(resp, "completion_text", "") or "").strip()
        if completion_text:
            self.record_bot_reply(event, completion_text)


class HeartflowPlugin(star.Star):
    def __init__(self, context: star.Context, config):
        super().__init__(context)
        self.config = config
        self.group_memory = GroupMemoryEngine(self.context, self.config)
        self.group_memory.warn_builtin_ltm_enabled(force=True)

        # 判断模型配置
        self.judge_provider_name = self.config.get("judge_provider_name", "")

        # 心流参数配置
        self.reply_threshold = self.config.get("reply_threshold", 0.6)
        self.energy_decay_rate = self.config.get("energy_decay_rate", 0.1)
        self.energy_recovery_rate = self.config.get("energy_recovery_rate", 0.02)
        self.context_messages_count = self.config.get("context_messages_count", 5)
        self.judge_context_count = self.config.get(
            "judge_context_count", self.context_messages_count
        )
        self.min_reply_interval = self.config.get("min_reply_interval_seconds", 0)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}

        # 原始群聊消息缓冲区：{unified_msg_origin: deque[RawMessage]}
        # 记录所有群聊原始消息（无论是否触发 LLM），用于判断上下文
        self._raw_msg_buffer: Dict[str, deque] = {}
        self._raw_msg_buffer_size = (
            max(self.context_messages_count, self.judge_context_count) * 4
        )  # 缓冲区保留更多条以备用

        # 系统提示词缓存：{conversation_id: {"original": str, "summarized": str, "persona_id": str}}
        self.system_prompt_cache: Dict[str, Dict[str, str]] = {}

        # 判断配置
        self.judge_include_reasoning = self.config.get("judge_include_reasoning", True)
        self.judge_max_retries = max(
            0, self.config.get("judge_max_retries", 3)
        )  # 确保最小为0

        # 判断权重配置
        self.weights = {
            "relevance": self.config.get("judge_relevance", 0.25),
            "willingness": self.config.get("judge_willingness", 0.2),
            "social": self.config.get("judge_social", 0.2),
            "timing": self.config.get("judge_timing", 0.15),
            "continuity": self.config.get("judge_continuity", 0.2),
        }
        # 检查权重和
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"判断权重和不为1，当前和为{weight_sum}")
            # 进行归一化处理
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
            logger.info(f"判断权重和已归一化，当前配置为: {self.weights}")

        logger.info("心流插件已初始化（接管群聊记忆 + 主动回复）")

    async def _get_or_create_summarized_system_prompt(
        self, event: AstrMessageEvent, original_prompt: str
    ) -> str:
        """获取或创建精简版系统提示词"""
        try:
            # 获取当前会话ID
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                event.unified_msg_origin
            )
            if not curr_cid:
                return original_prompt

            # 获取当前人格ID作为缓存键（仅用 persona_id，不包含 cid）
            # cid 随对话切换会变，但提示词是按人格存的，缓存键不应包含 cid
            conversation = await self.context.conversation_manager.get_conversation(
                event.unified_msg_origin, curr_cid
            )
            persona_id = (
                conversation.persona_id if conversation else None
            ) or "default"

            # 构建缓存键
            cache_key = persona_id

            # 检查缓存
            if cache_key in self.system_prompt_cache:
                cached = self.system_prompt_cache[cache_key]
                # 如果原始提示词没有变化，返回缓存的总结
                if cached.get("original") == original_prompt:
                    logger.debug(f"使用缓存的精简系统提示词: {cache_key}")
                    return cached.get("summarized", original_prompt)

            # 如果没有缓存或原始提示词发生变化，进行总结
            if not original_prompt or len(original_prompt.strip()) < 50:
                # 如果原始提示词太短，直接返回
                return original_prompt

            summarized_prompt = await self._summarize_system_prompt(original_prompt)

            # 更新缓存
            self.system_prompt_cache[cache_key] = {
                "original": original_prompt,
                "summarized": summarized_prompt,
                "persona_id": persona_id,
            }

            logger.info(
                f"创建新的精简系统提示词: [{cache_key}] | 原长度:{len(original_prompt)} -> 新长度:{len(summarized_prompt)}"
            )
            return summarized_prompt

        except Exception as e:
            logger.error(f"获取精简系统提示词失败: {e}")
            return original_prompt

    async def _summarize_system_prompt(self, original_prompt: str) -> str:
        """使用小模型对系统提示词进行总结"""
        try:
            if not self.judge_provider_name:
                return original_prompt

            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                return original_prompt

            summarize_prompt = f"""请将以下机器人角色设定总结为简洁的核心要点，保留关键的性格特征、行为方式和角色定位。
总结后的内容应该在100-200字以内，突出最重要的角色特点。

原始角色设定：
{original_prompt}

请以JSON格式回复：
{{
    "summarized_persona": "精简后的角色设定，保留核心特征和行为方式"
}}

**重要：你的回复必须是完整的JSON对象，不要包含任何其他内容！**"""

            llm_response = await judge_provider.text_chat(
                prompt=summarize_prompt,
                contexts=[],  # 不需要上下文
            )

            content = llm_response.completion_text.strip()

            # 尝试提取JSON
            try:
                result_data = _extract_json(content)
                summarized = result_data.get("summarized_persona", "")

                if summarized and len(summarized.strip()) > 10:
                    return summarized.strip()
                else:
                    logger.warning("小模型返回的总结内容为空或过短")
                    return original_prompt

            except (json.JSONDecodeError, ValueError):
                logger.error(f"小模型总结系统提示词返回非有效JSON: {content}")
                return original_prompt

        except Exception as e:
            logger.error(f"总结系统提示词异常: {e}")
            return original_prompt

    async def judge_with_tiny_model(self, event: AstrMessageEvent) -> JudgeResult:
        """使用小模型进行智能判断"""

        if not self.judge_provider_name:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return JudgeResult(should_reply=False, reasoning="提供商未配置")

        # 获取指定的 provider
        try:
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                logger.warning(f"未找到提供商: {self.judge_provider_name}")
                return JudgeResult(
                    should_reply=False,
                    reasoning=f"提供商不存在: {self.judge_provider_name}",
                )
        except Exception as e:
            logger.error(f"获取提供商失败: {e}")
            return JudgeResult(
                should_reply=False, reasoning=f"获取提供商失败: {str(e)}"
            )

        # 获取群聊状态
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 获取当前对话的人格系统提示词，让模型了解大参数LLM的角色设定
        original_persona_prompt = await self._get_persona_system_prompt(event)
        logger.debug(
            f"小参数模型获取原始人格提示词: {'有' if original_persona_prompt else '无'} | 长度: {len(original_persona_prompt) if original_persona_prompt else 0}"
        )

        # 获取或创建精简版系统提示词
        persona_system_prompt = await self._get_or_create_summarized_system_prompt(
            event, original_persona_prompt
        )
        logger.debug(
            f"小参数模型使用精简人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}"
        )

        # 构建判断上下文
        chat_context = self._build_chat_context(event)
        recent_messages = self._get_recent_messages(event)
        last_bot_reply = self._get_last_bot_reply(event)

        reasoning_part = ""
        if self.judge_include_reasoning:
            reasoning_part = ',\n    "reasoning": "详细分析原因，说明为什么应该或不应该回复，需要结合机器人角色特点进行分析，特别说明与上次回复的关联性"'

        judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

## 机器人角色设定
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

## 当前群聊情况
- 群聊ID: {event.unified_msg_origin}
- 我的精力水平: {chat_state.energy:.1f}/1.0
- 上次发言: {self._get_minutes_since_last_reply(event.unified_msg_origin)}分钟前

## 群聊基本信息
{chat_context}

## 最近{self.context_messages_count}条对话历史
{recent_messages}

## 上次机器人回复
{last_bot_reply if last_bot_reply else "暂无上次回复记录"}

## 待判断消息
发送者: {event.get_sender_name()}
内容: {event.message_str}
时间: {datetime.datetime.now().strftime("%H:%M:%S")}

## 评估要求
请从以下5个维度评估（0-10分），**重要提醒：基于上述机器人角色设定来判断是否适合回复**：

1. **内容相关度**(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - **结合机器人角色特点，判断是否符合角色定位**

2. **回复意愿**(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和心情状态
   - 考虑今日回复频率控制
   - **基于机器人角色设定，判断是否应该主动参与此话题**

3. **社交适宜性**(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - **考虑机器人角色在群中的定位和表现方式**

4. **时机恰当性**(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. **对话连贯性**(0-10)：当前消息与上次机器人回复的关联程度
   - 如果当前消息是对上次回复的回应或延续，应给高分
   - 如果当前消息与上次回复完全无关，给中等分数
   - 如果没有上次回复记录，给默认分数5分

**回复阈值**: {self.reply_threshold} (综合评分达到此分数才回复)

**重要！！！请严格按照以下JSON格式回复，不要添加任何其他内容：**

请以JSON格式回复：
{{
    "relevance": 分数,
    "willingness": 分数,
    "social": 分数,
    "timing": 分数,
    "continuity": 分数{reasoning_part}
}}

**注意：你的回复必须是完整的JSON对象，不要包含任何解释性文字或其他内容！**
"""

        try:
            # 构建完整的判断提示词，将系统提示直接整合到prompt中
            complete_judge_prompt = (
                "你是一个专业的群聊回复决策系统，能够准确判断消息价值和回复时机。"
            )
            if persona_system_prompt:
                complete_judge_prompt += (
                    f"\n\n你正在为以下角色的机器人做决策：\n{persona_system_prompt}"
                )
            complete_judge_prompt += "\n\n**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**\n\n"
            complete_judge_prompt += judge_prompt

            # 提前计算对话历史上下文（循环外只算一次）
            recent_contexts = self._get_buffered_history(
                event.unified_msg_origin,
                n=self.judge_context_count,
                exclude_last_content=event.message_str,
                as_dict=True,
            )

            # 重试机制：使用配置的重试次数
            max_retries = self.judge_max_retries + 1
            if self.judge_max_retries == 0:
                max_retries = 1

            for attempt in range(max_retries):
                try:
                    llm_response = await judge_provider.text_chat(
                        prompt=complete_judge_prompt,
                        contexts=recent_contexts,
                        image_urls=[],
                    )

                    content = llm_response.completion_text.strip()
                    logger.debug(f"小参数模型原始返回内容: {content[:200]}...")

                    judge_data = _extract_json(content)

                    # 直接从 JSON 根对象获取分数，并钉位到 [0, 10]
                    relevance = _clamp_score(judge_data.get("relevance", 0))
                    willingness = _clamp_score(judge_data.get("willingness", 0))
                    social = _clamp_score(judge_data.get("social", 0))
                    timing = _clamp_score(judge_data.get("timing", 0))
                    continuity = _clamp_score(judge_data.get("continuity", 0))

                    # 计算综合评分
                    overall_score = (
                        relevance * self.weights["relevance"]
                        + willingness * self.weights["willingness"]
                        + social * self.weights["social"]
                        + timing * self.weights["timing"]
                        + continuity * self.weights["continuity"]
                    ) / 10.0

                    # 根据综合评分判断是否应该回复
                    should_reply = overall_score >= self.reply_threshold

                    logger.debug(
                        f"小参数模型判断成功，综合评分: {overall_score:.3f}, 是否回复: {should_reply}"
                    )

                    reasoning_text = (
                        judge_data.get("reasoning", "")
                        if self.judge_include_reasoning
                        else ""
                    )
                    logger.info(
                        "Heartflow judge | chat=%s | rel=%.1f will=%.1f soc=%.1f time=%.1f cont=%.1f | overall=%.3f threshold=%.3f | reply=%s | reason=%s",
                        event.unified_msg_origin,
                        relevance,
                        willingness,
                        social,
                        timing,
                        continuity,
                        overall_score,
                        self.reply_threshold,
                        should_reply,
                        (reasoning_text[:200] + "...")
                        if reasoning_text and len(reasoning_text) > 200
                        else reasoning_text,
                    )

                    return JudgeResult(
                        relevance=relevance,
                        willingness=willingness,
                        social=social,
                        timing=timing,
                        continuity=continuity,
                        reasoning=reasoning_text,
                        should_reply=should_reply,
                        confidence=overall_score,  # 使用综合评分作为置信度
                        overall_score=overall_score,
                        related_messages=[],  # 不再使用关联消息功能
                    )

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"小参数模型返回JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    logger.warning(f"无法解析的内容: {content[:500]}...")

                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，返回失败结果
                        logger.error(
                            f"小参数模型重试{self.judge_max_retries}次后仍然返回无效JSON，放弃处理"
                        )
                        return JudgeResult(
                            should_reply=False,
                            reasoning=f"JSON解析失败，重试{self.judge_max_retries}次",
                        )
                    else:
                        # 还有重试机会，添加更强的提示
                        complete_judge_prompt = complete_judge_prompt.replace(
                            "**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**",
                            f"**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！这是第{attempt + 2}次尝试，请确保返回有效的JSON格式！**",
                        )
                        continue

        except Exception as e:
            logger.error(f"小参数模型判断异常: {e}")
            return JudgeResult(should_reply=False, reasoning=f"异常: {str(e)}")

    def _record_raw_message(
        self, event: AstrMessageEvent, is_bot: bool = False
    ) -> None:
        """将消息写入原始消息缓冲区"""
        umo = event.unified_msg_origin
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        self._raw_msg_buffer[umo].append(
            RawMessage(
                sender_name=event.get_sender_name(),
                sender_id=str(event.get_sender_id()),
                content=event.message_str,
                timestamp=time.time(),
                is_bot=is_bot,
            )
        )

    def _get_raw_buffer(self, umo: str) -> list[RawMessage]:
        """获取缓冲区中的消息列表（时间顺序）"""
        return list(self._raw_msg_buffer.get(umo, []))

    def _get_buffered_history(
        self,
        umo: str,
        n: int = 10,
        exclude_last_content: str | None = None,
        as_dict: bool = False,
    ):
        """Compatibility shim for historical calls using local raw-message buffer."""
        msgs = self._get_raw_buffer(umo)

        if exclude_last_content and msgs and msgs[-1].content == exclude_last_content:
            msgs = msgs[:-1]

        if n > 0 and len(msgs) > n:
            msgs = msgs[-n:]

        if not as_dict:
            return msgs

        return [
            {"role": ("assistant" if m.is_bot else "user"), "content": m.content}
            for m in msgs
        ]

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent, *args, **kwargs):
        """群聊消息入口：先记忆，再做 Heartflow 主动回复判断。"""

        self.group_memory.warn_builtin_ltm_enabled(event)

        if event.get_sender_id() != event.get_self_id():
            self.group_memory.record_user_message(event)
            if event.message_str and event.message_str.strip():
                self._record_raw_message(event, is_bot=False)

        if not self._should_process_message(event):
            return

        try:
            judge_result = await self.judge_with_tiny_model(event)

            if judge_result.should_reply:
                logger.info(
                    "Heartflow 主动回复触发 | chat=%s | score=%.3f",
                    event.unified_msg_origin,
                    judge_result.overall_score,
                )
                event.is_at_or_wake_command = True
                event.set_extra("heartflow_triggered", True)
                self._update_active_state(event, judge_result)
                return

            self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error("心流插件处理消息异常: %s", e)
            logger.error(traceback.format_exc())

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """在消息发送后将机器人的回复写入原始消息缓冲区，以便后续判断参考"""
        clean_session = event.get_extra("_clean_ltm_session", False)
        if clean_session:
            chat_id = event.unified_msg_origin
            if chat_id in self.chat_states:
                del self.chat_states[chat_id]
            if chat_id in self._raw_msg_buffer:
                del self._raw_msg_buffer[chat_id]
            cleared = self.group_memory.clear_session(chat_id)
            logger.info(
                "[%s] 已清空 Heartflow 会话上下文（/reset 或 /new）: records=%d, seen_images=%d, pending=%d",
                chat_id,
                cleared["records"],
                cleared["seen_images"],
                cleared["pending"],
            )
            return

        if not self.config.get("enable_heartflow", False):
            return

        result = event.get_result()
        if result is None or not result.chain:
            return

        # 提取回复的纯文本内容
        reply_text = "".join(
            comp.text for comp in result.chain if isinstance(comp, Plain)
        ).strip()
        if not reply_text:
            return

        umo = event.unified_msg_origin
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        self._raw_msg_buffer[umo].append(
            RawMessage(
                sender_name="bot",
                sender_id="bot",
                content=reply_text,
                timestamp=time.time(),
                is_bot=True,
            )
        )
        logger.debug(f"机器人回复已写入缓冲区: {umo[:20]}... | {reply_text[:40]}...")

    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req, *args, **kwargs):
        """请求 LLM 前：注入接管版群聊记忆 + 心流触发提示。"""
        if not req:
            return

        await self.group_memory.apply_on_request(
            event,
            req,
            heartflow_triggered=bool(event.get_extra("heartflow_triggered")),
        )

        if not event.get_extra("heartflow_triggered"):
            return
        if not hasattr(req, "system_prompt"):
            return
        note = "（注意：本次是你主动参与群聊的，不是用户叫你。回复应自然随意，像普通群成员一样加入话题。）"
        req.system_prompt = (req.system_prompt or "") + "\n" + note

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp, *args, **kwargs):
        """LLM 响应后：响应后记 seen、异步摘要、写入机器人回复。"""
        await self.group_memory.apply_on_response(event, resp)

    def _should_process_message(self, event: AstrMessageEvent) -> bool:
        """检查是否应该处理这条消息"""

        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False

        # 检查插件是否启用
        if not self.config.get("enable_heartflow", False):
            return False

        # 跳过已经被其他插件或系统标记为唤醒的消息
        if event.is_at_or_wake_command:
            logger.debug(f"跳过已被标记为唤醒的消息: {event.message_str}")
            return False

        # 检查白名单
        if self.whitelist_enabled:
            if not self.chat_whitelist:
                logger.debug(f"白名单为空，跳过处理: {event.unified_msg_origin}")
                return False

            if event.unified_msg_origin not in self.chat_whitelist:
                logger.debug(f"群聊不在白名单中，跳过处理: {event.unified_msg_origin}")
                return False

        # 跳过机器人自己的消息
        if event.get_sender_id() == event.get_self_id():
            return False

        # 跳过空消息
        if not event.message_str or not event.message_str.strip():
            return False

        # 冷却时间校验：防止短时间内连续触发
        if self.min_reply_interval > 0:
            minutes = self._get_minutes_since_last_reply(event.unified_msg_origin)
            elapsed_seconds = minutes * 60
            if elapsed_seconds < self.min_reply_interval:
                logger.debug(
                    f"冷却中，距上次回复还有 {self.min_reply_interval - elapsed_seconds:.0f}s"
                )
                return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取群聊状态"""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState()

        state = self.chat_states[chat_id]
        now = time.time()

        # 检查日期重置
        today = datetime.date.today().isoformat()
        if state.last_reset_date != today:
            state.last_reset_date = today
            state.energy = min(1.0, state.energy + 0.2)

        # 基于时间流逝自然恢复精力，不污染 last_reply_time
        if state.last_energy_update <= 0:
            state.last_energy_update = now
        elif state.last_reply_time > 0:
            elapsed_minutes = (now - state.last_energy_update) / 60.0
            if elapsed_minutes > 0:
                time_recovery = elapsed_minutes * (self.energy_recovery_rate * 5)
                state.energy = min(1.0, state.energy + time_recovery)
            state.last_energy_update = now

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """从原始消息缓冲区获取最近对话上下文（用于传递给小参数模型）。

        使用本地缓冲区而非 conversation_manager，以便包含所有群聊消息，
        而不仅仅是触发过 LLM 的消息。
        """
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        # 排除当前这条消息（已被 _record_raw_message 写入），取之前的若干条
        if msgs and msgs[-1].content == event.message_str:
            msgs = msgs[:-1]
        recent = (
            msgs[-self.context_messages_count :]
            if len(msgs) > self.context_messages_count
            else msgs
        )

        contexts = []
        for m in recent:
            role = "assistant" if m.is_bot else "user"
            contexts.append({"role": role, "content": m.content})
        return contexts

    def _get_recent_messages(self, event: AstrMessageEvent) -> str:
        """从原始消息缓冲区获取最近的消息历史（用于小参数模型判断）。

        包含所有群聊成员的消息，而非仅 LLM 处理过的消息。
        """
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        # 排除当前这条消息（已被 _record_raw_message 写入），取之前的若干条
        if msgs and msgs[-1].content == event.message_str:
            msgs = msgs[:-1]
        recent = (
            msgs[-self.context_messages_count :]
            if len(msgs) > self.context_messages_count
            else msgs
        )

        if not recent:
            return "暂无对话历史"

        lines = []
        for m in recent:
            prefix = "[机器人]" if m.is_bot else f"[{m.sender_name}]"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def _get_last_bot_reply(self, event: AstrMessageEvent) -> str | None:
        """从原始消息缓冲区获取上次机器人的回复内容。"""
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        for m in reversed(msgs):
            if m.is_bot and m.content.strip():
                return m.content
        return None

    def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文摘要信息。"""
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 检查上次机器人回复后群里有没有人接话（评估回复质量）
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        post_reply_engagement = ""
        found_bot = False
        user_msgs_after_bot = 0
        for m in reversed(msgs):
            if m.is_bot:
                found_bot = True
                break
            user_msgs_after_bot += 1
        if found_bot:
            if user_msgs_after_bot >= 3:
                post_reply_engagement = "（上次回复后群里进行了热烈讨论）"
            elif user_msgs_after_bot == 0:
                post_reply_engagement = "（上次回复后无人接话）"

        if chat_state.total_messages > 100:
            activity_level = "高"
        elif chat_state.total_messages > 20:
            activity_level = "中"
        else:
            activity_level = "低"

        context_info = f"最近活跃度: {activity_level}\n"
        context_info += f"历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%\n"
        context_info += f"当前时间: {datetime.datetime.now().strftime('%H:%M')}"

        if post_reply_engagement:
            context_info += f"\n回复效果: {post_reply_engagement}"

        return context_info

    def _update_active_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新主动回复状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新回复相关状态
        now = time.time()
        chat_state.last_reply_time = now
        chat_state.last_energy_update = now
        chat_state.total_replies += 1
        chat_state.total_messages += 1

        # 精力消耗（回复后精力下降）
        chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)

        logger.debug(f"更新主动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f}")

    def _update_passive_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新被动状态（未回复）"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新消息计数
        chat_state.total_messages += 1

        # 精力恢复（不回复时精力缓慢恢复）
        chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

        logger.debug(
            f"更新被动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f} | 原因: {judge_result.reasoning[:30]}..."
        )

    # 管理员命令：查看心流状态
    @filter.command("heartflow")
    async def heartflow_status(self, event: AstrMessageEvent):
        """查看心流状态"""

        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        status_info = f"""
🔮 心流状态报告

📊 **当前状态**
- 群聊ID: {event.unified_msg_origin}
- 精力水平: {chat_state.energy:.2f}/1.0 {"🟢" if chat_state.energy > 0.7 else "🟡" if chat_state.energy > 0.3 else "🔴"}
- 上次回复: {self._get_minutes_since_last_reply(chat_id)}分钟前

📈 **历史统计**
- 总消息数: {chat_state.total_messages}
- 总回复数: {chat_state.total_replies}
- 回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%

⚙️ **配置参数**
- 回复阈值: {self.reply_threshold}
- 判断提供商: {self.judge_provider_name}
- 最大重试次数: {self.judge_max_retries}
- 白名单模式: {"✅ 开启" if self.whitelist_enabled else "❌ 关闭"}
- 白名单群聊数: {len(self.chat_whitelist) if self.whitelist_enabled else 0}

🧠 **智能缓存**
- 系统提示词缓存: {len(self.system_prompt_cache)} 个

🎯 **评分权重**
- 内容相关度: {self.weights["relevance"]:.0%}
- 回复意愿: {self.weights["willingness"]:.0%}
- 社交适宜性: {self.weights["social"]:.0%}
- 时机恰当性: {self.weights["timing"]:.0%}
- 对话连贯性: {self.weights["continuity"]:.0%}

🎯 **插件状态**: {"✅ 已启用" if self.config.get("enable_heartflow", False) else "❌ 已禁用"}
"""

        event.set_result(event.plain_result(status_info))

    # 管理员命令：重置心流状态
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_reset")
    async def heartflow_reset(self, event: AstrMessageEvent):
        """重置心流状态"""

        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            del self.chat_states[chat_id]
        if chat_id in self._raw_msg_buffer:
            del self._raw_msg_buffer[chat_id]
        cleared = self.group_memory.clear_session(chat_id)

        event.set_result(
            event.plain_result(
                "✅ 心流状态已重置\n"
                f"- 清理记忆记录: {cleared['records']}\n"
                f"- 清理图片状态: {cleared['seen_images']}\n"
                f"- 清理待注入队列: {cleared['pending']}"
            )
        )
        logger.info(f"心流状态已重置: {chat_id}")

    # 管理员命令：查看系统提示词缓存
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_cache")
    async def heartflow_cache_status(self, event: AstrMessageEvent):
        """查看系统提示词缓存状态"""

        cache_info = "🧠 系统提示词缓存状态\n\n"

        if not self.system_prompt_cache:
            cache_info += "📭 当前无缓存记录"
        else:
            cache_info += f"📝 总缓存数量: {len(self.system_prompt_cache)}\n\n"

            for cache_key, cache_data in self.system_prompt_cache.items():
                original_len = len(cache_data.get("original", ""))
                summarized_len = len(cache_data.get("summarized", ""))
                persona_id = cache_data.get("persona_id", "unknown")

                cache_info += f"🔑 **缓存键**: {cache_key}\n"
                cache_info += f"👤 **人格ID**: {persona_id}\n"
                cache_info += f"📏 **压缩率**: {original_len} -> {summarized_len} ({(1 - summarized_len / max(1, original_len)) * 100:.1f}% 压缩)\n"
                cache_info += (
                    f"📄 **精简内容**: {cache_data.get('summarized', '')[:100]}...\n\n"
                )

        event.set_result(event.plain_result(cache_info))

    # 管理员命令：清除系统提示词缓存
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_cache_clear")
    async def heartflow_cache_clear(self, event: AstrMessageEvent):
        """清除系统提示词缓存"""

        cache_count = len(self.system_prompt_cache)
        self.system_prompt_cache.clear()

        event.set_result(
            event.plain_result(f"✅ 已清除 {cache_count} 个系统提示词缓存")
        )
        logger.info(f"系统提示词缓存已清除，共清除 {cache_count} 个缓存")

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前对话的人格系统提示词"""
        try:
            persona_mgr = self.context.persona_manager

            # 获取当前对话，尝试拿到会话绑定的 persona_id
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                event.unified_msg_origin
            )
            persona_id: str | None = None
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(
                    event.unified_msg_origin, curr_cid
                )
                if conversation:
                    persona_id = conversation.persona_id

            # 用户显式取消人格
            if persona_id == "[%None]":
                return ""

            if persona_id:
                # 直接通过 PersonaManager 查询数据库
                try:
                    persona = await persona_mgr.get_persona(persona_id)
                    return persona.system_prompt or ""
                except ValueError:
                    logger.debug(f"未找到人格 {persona_id}，回退到默认人格")

            # 无 persona_id 或查询失败，使用默认人格
            default_persona = await persona_mgr.get_default_persona_v3(
                event.unified_msg_origin
            )
            return default_persona.get("prompt", "")

        except Exception as e:
            logger.debug(f"获取人格系统提示词失败: {e}")
            return ""
