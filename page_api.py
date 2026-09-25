"""Heartflow 管理面板的 Web API，前端通过 AstrBot 插件页 bridge 的 apiGet/apiPost 调用。"""

import time
from types import SimpleNamespace

from astrbot.api.web import request

PLUGIN_NAME = "astrbot_plugin_Heartflow"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"

# 面板「判断配置」页可以修改的配置项
JUDGE_CONFIG_KEYS = (
    "judge_mode",
    "jev_decision_mode",
    "reply_threshold",
    "judge_include_reasoning",
    "llm_judge_preamble",
    "score_dimensions",
    "jev_noul_question",
)


def _ok(data=None) -> dict:
    return {"success": True, "data": data}


def _error(message: str) -> dict:
    return {"success": False, "error": message}


class HeartflowPageApi:
    def __init__(self, plugin) -> None:
        self.plugin = plugin

    def register_routes(self) -> None:
        routes = [
            ("/overview", self.overview, ["GET"], "Heartflow 概览"),
            ("/chat/reset", self.reset_chat, ["POST"], "Heartflow 重置群状态"),
            ("/judgements", self.judgements, ["GET"], "Heartflow 判断日志"),
            ("/last_request", self.last_request, ["GET"], "Heartflow 最近判断请求"),
            ("/persona", self.persona, ["GET"], "Heartflow 人格"),
            ("/persona/update", self.update_persona, ["POST"], "Heartflow 保存人格"),
            (
                "/persona/regenerate",
                self.regenerate_persona,
                ["POST"],
                "Heartflow 重新压缩人格",
            ),
            ("/judge_config", self.judge_config, ["GET"], "Heartflow 判断配置"),
            (
                "/judge_config/preview",
                self.preview_judge_config,
                ["POST"],
                "Heartflow 预览判断配置",
            ),
            (
                "/judge_config/update",
                self.update_judge_config,
                ["POST"],
                "Heartflow 保存判断配置",
            ),
        ]
        for path, handler, methods, desc in routes:
            self.plugin.context.register_web_api(
                f"{PAGE_API_PREFIX}{path}", handler, methods, desc
            )

    async def overview(self) -> dict:
        p = self.plugin
        now = time.time()
        last_judge = {item["umo"]: item for item in p.judge_log}
        chats = []
        for umo, state in p.chat_states.items():
            # 按判断时的算法补上距上次更新以来的自然恢复，只读不写回状态
            idle_minutes = max(0.0, now - state.last_energy_update_time) / 60.0
            energy = min(
                1.0, state.energy + idle_minutes / 5.0 * p.energy_recovery_rate
            )
            last_activity = max(state.last_reply_time, state.last_trigger_time)
            cooldown = (
                max(0.0, p.min_reply_interval - (now - last_activity))
                if p.min_reply_interval and last_activity
                else 0.0
            )
            last = last_judge.get(umo)
            chats.append(
                {
                    "umo": umo,
                    "energy": energy,
                    "total_messages": state.total_messages,
                    "total_replies": state.total_replies,
                    "last_reply_time": state.last_reply_time,
                    "cooldown_seconds": cooldown,
                    "pending": len(state.debounce_batch or []),
                    "last_active": state.last_access_time,
                    "last_signal": last["signal"] if last else None,
                    "last_triggered": last["triggered"] if last else None,
                }
            )
        chats.sort(key=lambda c: c["last_active"], reverse=True)
        return _ok(
            {
                "now": now,
                "enabled": bool(p.config.get("enable_heartflow", False)),
                "judge_mode": p.judge_mode,
                "jev_decision_mode": p.jev_decision_mode,
                "jev_model": p.jev_model,
                "judge_provider_name": p.judge_provider_name,
                "reply_threshold": p.reply_threshold,
                "compress_persona": p.compress_persona,
                "debounce_seconds": p.debounce_seconds,
                "min_reply_interval": p.min_reply_interval,
                "whitelist_enabled": bool(p.whitelist_enabled),
                "chat_whitelist": sorted(p.chat_whitelist),
                "judge_count": len(p.judge_log),
                "trigger_count": sum(1 for item in p.judge_log if item["triggered"]),
                "chats": chats,
            }
        )

    async def reset_chat(self) -> dict:
        body = await request.json({})
        umo = str(body.get("umo") or "")
        async with self.plugin._get_chat_lock(umo):
            self.plugin.chat_states.pop(umo, None)
            self.plugin._raw_msg_buffer.pop(umo, None)
        return _ok({"umo": umo})

    async def judgements(self) -> dict:
        umo = request.query.get("umo", "")
        triggered_only = request.query.get("triggered", "") == "1"
        limit = request.query.get("limit", 100, type=int)
        items = [
            item
            for item in reversed(self.plugin.judge_log)
            if (not umo or item["umo"] == umo)
            and (not triggered_only or item["triggered"])
        ]
        return _ok({"items": items[:limit], "total": len(self.plugin.judge_log)})

    async def last_request(self) -> dict:
        return _ok(self.plugin.last_judge_request)

    async def _original_persona(self, umo: str | None) -> str:
        event = SimpleNamespace(unified_msg_origin=umo)
        persona_id = await self.plugin._resolve_persona_id(event)
        return await self.plugin._get_persona_system_prompt(event, persona_id)

    async def persona(self) -> dict:
        umo = request.query.get("umo", "") or None
        p = self.plugin
        return _ok(
            {
                "umo": umo,
                "compress_persona": p.compress_persona,
                "compressed_persona": str(p.config.get("compressed_persona") or ""),
                "judge_provider_name": p.judge_provider_name,
                "original": await self._original_persona(umo),
            }
        )

    async def update_persona(self) -> dict:
        body = await request.json({})
        p = self.plugin
        if "compress_persona" in body:
            p.config["compress_persona"] = bool(body["compress_persona"])
        if "compressed_persona" in body:
            p.config["compressed_persona"] = str(body["compressed_persona"]).strip()
        p.config.save_config()
        p._apply_judge_config()
        return _ok(
            {
                "compress_persona": p.compress_persona,
                "compressed_persona": p.config["compressed_persona"],
            }
        )

    async def regenerate_persona(self) -> dict:
        body = await request.json({})
        p = self.plugin
        if not p.judge_provider_name:
            return _error("未配置判断模型提供商 judge_provider_name，无法压缩")
        original = await self._original_persona(str(body.get("umo") or "") or None)
        async with p._compress_lock:
            compressed = await p._summarize_system_prompt(original)
            if compressed is None:
                return _error("压缩失败，原压缩人格保持不变，详情见 AstrBot 日志")
            p.config["compressed_persona"] = compressed
            p.config.save_config()
        return _ok({"compressed_persona": compressed, "original_length": len(original)})

    def _judge_draft(self, body: dict) -> dict:
        """当前判断配置叠加面板提交的改动。"""
        draft = {key: self.plugin.config.get(key) for key in JUDGE_CONFIG_KEYS}
        draft.update({key: body[key] for key in JUDGE_CONFIG_KEYS if key in body})
        for dimension in draft["score_dimensions"]:
            dimension.setdefault("__template_key", "dimension")
        return draft

    async def judge_config(self) -> dict:
        draft = self._judge_draft({})
        return _ok(
            {"config": draft, "preview": self.plugin.render_judge_preview(draft)}
        )

    async def preview_judge_config(self) -> dict:
        draft = self._judge_draft(await request.json({}))
        try:
            return _ok(self.plugin.render_judge_preview(draft))
        except (KeyError, TypeError, ValueError) as e:
            return _error(f"配置格式有误: {e}")

    async def update_judge_config(self) -> dict:
        draft = self._judge_draft(await request.json({}))
        try:
            preview = self.plugin.render_judge_preview(draft)
        except (KeyError, TypeError, ValueError) as e:
            return _error(f"配置格式有误，未保存: {e}")
        for key, value in draft.items():
            self.plugin.config[key] = value
        self.plugin.config.save_config()
        self.plugin._apply_judge_config()
        return _ok({"config": draft, "preview": preview})
