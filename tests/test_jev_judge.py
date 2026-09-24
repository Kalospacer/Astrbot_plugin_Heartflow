import os
import unittest
from unittest.mock import AsyncMock, patch

from test_main import _config, _Context, _Event, heartflow

_SCORE_NAMES = heartflow._SCORE_NAMES


class _JevResp:
    def __init__(self, status_code=200, data=None, text="", headers=None):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data


def _jev_payload(score_map=None, noul=None, conf_map=None):
    score_map = score_map or {}
    conf_map = conf_map or {}
    answers = {}
    for name in _SCORE_NAMES:
        answers[name] = {
            "type": "score",
            "score": score_map.get(name, 3.0),
            "confidence": conf_map.get(name, 0.9),
            "probabilities": {"3": 1.0},
            "legend": {},
        }
    if noul is not None:
        answers["should_reply"] = {"type": "noul", "noul": noul}
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 1234, "output_tokens": 56},
    }


class JevJudgeTests(unittest.IsolatedAsyncioTestCase):
    def _plugin(self, **cfg):
        base = {"judge_mode": "jev", "judge_provider_name": "", "jev_api_key": "k"}
        base.update(cfg)
        return heartflow.HeartflowPlugin(_Context(), _config(**base))

    async def _run_judge(self, plugin, resp_seq):
        client = AsyncMock()
        client.is_closed = False
        client.post = AsyncMock(side_effect=resp_seq)
        plugin._jev_client = client
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await plugin._judge_with_jev(_Event())
        return result, client

    async def test_noul_mode_decides_by_noul_and_sends_only_noul(self):
        plugin = self._plugin()
        result, client = await self._run_judge(
            plugin, [_JevResp(200, _jev_payload(noul=0.9))]
        )
        self.assertTrue(result.should_reply)
        self.assertAlmostEqual(result.overall_score, 0.9)
        self.assertIn("by=noul", result.reasoning)
        sent = client.post.await_args.kwargs["json"]["questions"]
        self.assertEqual(list(sent), ["should_reply"])

        plugin = self._plugin()
        result, _ = await self._run_judge(
            plugin, [_JevResp(200, _jev_payload(noul=0.3))]
        )
        self.assertAlmostEqual(result.overall_score, 0.3)
        self.assertFalse(result.should_reply)

    async def test_noul_mode_missing_noul_fails(self):
        plugin = self._plugin()
        result, _ = await self._run_judge(
            plugin, [_JevResp(200, _jev_payload(noul=None))]
        )
        self.assertFalse(result.should_reply)
        self.assertIn("noul 非法", result.reasoning)

    async def test_score_mode_decides_by_weighted_scores(self):
        plugin = self._plugin(jev_decision_mode="score")
        high = {n: 3.2 for n in _SCORE_NAMES}
        conf = {n: 0.95 for n in _SCORE_NAMES}
        conf["social"] = 0.0
        result, client = await self._run_judge(
            plugin,
            [_JevResp(200, _jev_payload(score_map=high, conf_map=conf))],
        )
        self.assertAlmostEqual(result.overall_score, 0.8)
        self.assertTrue(result.should_reply)
        self.assertIn("by=score", result.reasoning)
        self.assertIn("min_conf=0.00", result.reasoning)
        sent = client.post.await_args.kwargs["json"]["questions"]
        self.assertEqual(list(sent), list(_SCORE_NAMES))

    async def test_429_backoff_then_success(self):
        plugin = self._plugin()
        ok = _JevResp(200, _jev_payload(noul=0.9))
        limited = _JevResp(429, text="rate limited", headers={"retry-after": "1"})
        result, client = await self._run_judge(plugin, [limited, ok])
        self.assertTrue(result.should_reply)
        self.assertEqual(client.post.await_count, 2)

    async def test_401_and_422_do_not_retry(self):
        for code in (401, 422):
            plugin = self._plugin()
            result, client = await self._run_judge(plugin, [_JevResp(code, text="bad")])
            self.assertFalse(result.should_reply)
            self.assertIn(str(code), result.reasoning)
            self.assertEqual(client.post.await_count, 1)

    async def test_missing_api_key_fails_without_request(self):
        os.environ.pop("TYPESAFE_API_KEY", None)
        plugin = self._plugin(jev_api_key="")
        result, client = await self._run_judge(plugin, [])
        self.assertFalse(result.should_reply)
        self.assertIn("API key", result.reasoning)
        client.post.assert_not_awaited()

    async def test_missing_question_in_answers_fails(self):
        plugin = self._plugin(jev_decision_mode="score")
        payload = _jev_payload(noul=0.9)
        del payload["answers"]["continuity"]
        result, _ = await self._run_judge(plugin, [_JevResp(200, payload)])
        self.assertFalse(result.should_reply)
        self.assertIn("缺题", result.reasoning)

    async def test_score_out_of_range_fails(self):
        plugin = self._plugin(jev_decision_mode="score")
        bad = {n: 3.0 for n in _SCORE_NAMES}
        bad["timing"] = 5.0  # 超出 0-4 档
        result, _ = await self._run_judge(
            plugin, [_JevResp(200, _jev_payload(score_map=bad))]
        )
        self.assertFalse(result.should_reply)
        self.assertIn("越界", result.reasoning)

    async def test_state_excludes_current_message_and_structure(self):
        plugin = self._plugin()
        event = _Event(message="当前消息")
        plugin._record_raw_message(_Event(message="上一条消息"))
        plugin._record_bot_message(event.unified_msg_origin, "bot 上次回复")
        plugin._record_raw_message(event)
        state = await plugin._build_jev_state(
            event, plugin._get_chat_state(event.unified_msg_origin)
        )
        for key in (
            "persona",
            "bot_energy",
            "minutes_since_last_reply",
            "chat_activity",
            "chat_log",
            "bot_last_reply",
            "current_messages",
        ):
            self.assertIn(key, state)
        texts = [m["text"] for m in state["chat_log"]]
        self.assertIn("上一条消息", texts)
        self.assertNotIn("当前消息", texts)
        self.assertEqual(state["bot_last_reply"], "bot 上次回复")
        self.assertEqual([m["text"] for m in state["current_messages"]], ["当前消息"])
        self.assertEqual(state["persona"], "no persona set")

    async def test_full_persona_sent_when_compression_off(self):
        plugin = self._plugin(judge_provider_name="judge")
        plugin._get_persona_system_prompt = AsyncMock(return_value="x" * 1000)
        state = await plugin._build_jev_state(
            _Event(), plugin._get_chat_state("test:GroupMessage:10001")
        )
        self.assertEqual(state["persona"], "x" * 1000)

    async def test_llm_mode_still_routes_to_llm(self):
        plugin = self._plugin(judge_mode="llm")
        called = []

        async def fake_llm(event):
            called.append(True)
            return heartflow.JudgeResult(should_reply=False, reasoning="llm")

        plugin._judge_with_llm = fake_llm
        result = await plugin.judge_with_tiny_model(_Event())
        self.assertTrue(called)
        self.assertEqual(result.reasoning, "llm")

    async def test_terminate_closes_jev_client(self):
        plugin = self._plugin()
        client = AsyncMock()
        client.is_closed = False
        plugin._jev_client = client
        await plugin.terminate()
        client.aclose.assert_awaited_once()
        self.assertIsNone(plugin._jev_client)


if __name__ == "__main__":
    unittest.main()
