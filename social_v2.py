"""二代仿真模式（social v2）的会话状态存储。

移植自 qq-bridge 二代仿真模式（reserved2）的核心机制：
- 每条消息入库 ``unread``（未读）+ ``recent_messages``（最近，含 AI 自己发的）
- AI 不被动接收历史上下文，而是通过工具主动看未读/最近消息
- ``wake_config`` 唤醒配置：活跃/潜水 + 触发条件（@/名字/提问/关键词/概率/指定成员）
- 防永眠：无限期潜水必须至少保留一个可触发条件
- 会话 key 为 unified_msg_origin，群/私聊统一管理

状态持久化为 JSON，结构与 qq-bridge 的 socialV2 state 对齐。
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

# 默认作息概率：普通消息概率（未被 @ 时的小概率唤醒）。
_DEFAULT_PROBABILITY = 0.05

# 无限期潜水时默认推荐的触发条件。
_DEFAULT_TRIGGERS = {
    "at_mention": True,
    "name_mention": True,
    "question": True,
    "poke": True,
    "any_message": False,
    "probability": _DEFAULT_PROBABILITY,
    "keywords": [],
    "speaker_ids": [],
}

# 对方明确结束对话的结束语特征（命中后设置潜水不再需要沉睡前观察）。
EXPLICIT_END_RE = re.compile(
    r"(?:不聊了|不说了|晚安|睡了|先睡了|下了|先下了|拜拜|再见|走了|先走|撤了|"
    r"去忙|忙了|下次再聊|下次聊|散了吧|就到这|先这样|就这样吧|886|睡觉了|"
    r"下班了|去洗澡|去吃饭了)",
    re.IGNORECASE,
)

# 常见“话说一半/没说完”的结尾特征。
UNFINISHED_TAIL_RE = re.compile(
    r"(?:你知道|等一下|我跟你讲|其实吧|但是|所以说|然后|那个|就是|我想说|对了|"
    r"等我|等等|我看看|还有|再说|主要是|毕竟|因为|所以|但是吧|回头|回头说|"
    r"待会|晚点|再说吧)$"
)

# 以中文逗号/顿号/分号/冒号等“非终止标点”结尾，也常表示还没说完。
UNFINISHED_PUNCT_RE = re.compile(r"[，、；：,;:]$")

# 明确以终止标点/省略号结尾的消息视为说完。
FINISHED_TAIL_RE = re.compile(r"[。！？!?…～~]+$")

# AI 等不到下文时可以用的“催话”短句。
UNFINISHED_PROMPT_REPLIES = ("什么", "啥", "你说啊", "然后呢", "？")


def looks_like_unfinished(text: str) -> bool:
    """Heuristically decide whether a message looks cut off mid-sentence.

    Args:
        text: The message text.

    Returns:
        True when the tail suggests the speaker is not finished.
    """
    s = str(text or "").strip()
    if not s:
        return False
    if FINISHED_TAIL_RE.search(s):
        return False
    if UNFINISHED_TAIL_RE.search(s):
        return True
    if UNFINISHED_PUNCT_RE.search(s):
        return True
    return False


def default_wake_config() -> dict:
    """Build the default wake configuration.

    Returns:
        A wake config dict (diving mode with the recommended triggers).
    """
    return {
        "mode": "diving",
        "infinite": True,
        "sleep_until": None,
        "triggers": dict(_DEFAULT_TRIGGERS, keywords=[], speaker_ids=[]),
        "batch_window_ms": 8000,
        "last_wake_at": 0,
        "wake_count": 0,
        "confirmed_at": 0,
        "confirmed_by": "",
    }


def normalize_speaker_ids(raw) -> list[str]:
    """Normalize a speaker-id list: keep numeric ids, dedupe, bound count.

    Args:
        raw: The raw value from a wake config.

    Returns:
        A clean list of numeric user id strings (max 50).
    """
    if not isinstance(raw, list):
        return []
    seen: list[str] = []
    for item in raw:
        s = str(item or "").strip()
        if s.isdigit() and s not in seen:
            seen.append(s)
        if len(seen) >= 50:
            break
    return seen


def normalize_trigger_bool(name: str, raw_triggers: dict, fallback: bool) -> bool:
    """Read a boolean trigger, accepting only real booleans.

    Args:
        name: Trigger name (snake_case).
        raw_triggers: The incoming trigger dict.
        fallback: The current value to keep when the key is absent.

    Returns:
        The normalized boolean.
    """
    if name in raw_triggers:
        return raw_triggers[name] is True
    return fallback


class SocialV2Store:
    """Persistent per-conversation social v2 state.

    Args:
        path: Path of the JSON persistence file.
        recent_limit: Max recent messages kept per conversation.
        unread_limit: Max unread messages kept per conversation.
    """

    def __init__(
        self,
        path: str | Path,
        recent_limit: int = 100,
        unread_limit: int = 30,
    ) -> None:
        self.path = Path(path)
        self.recent_limit = int(recent_limit)
        self.unread_limit = int(unread_limit)
        self._conversations: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            convs = data.get("conversations", {})
            if isinstance(convs, dict):
                self._conversations = {
                    key: value
                    for key, value in convs.items()
                    if isinstance(value, dict)
                }
        except (OSError, json.JSONDecodeError):
            self._conversations = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"conversations": self._conversations}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError:
            pass

    def get_state(self, conv_id: str) -> dict:
        """Get (or create) a conversation's state.

        Args:
            conv_id: Conversation identifier (unified_msg_origin).

        Returns:
            The state dict with wake_config / recent_messages / unread.
        """
        st = self._conversations.get(conv_id)
        if st is None:
            st = {
                "wake_config": default_wake_config(),
                "recent_messages": [],
                "unread": [],
                "last_unread_seq": 0,
                "last_ai_reply_at": 0,
                "last_incoming_at": 0,
                "last_wake_reason": "",
                "pending_wake_reasons": [],
                "pre_sleep_satisfied_at": 0,
                "pre_sleep_observed_at": 0,
                "wake_times": [],
                "send_times": [],
                "bootstrap_sent": False,
                "agent_token": uuid.uuid4().hex,
            }
            self._conversations[conv_id] = st
            self._save()
        return st

    def append_message(
        self,
        conv_id: str,
        sender: str,
        user_id: str,
        text: str,
        message_id: str = "",
        quote_target_is_self: bool = False,
        is_self: bool = False,
        kind: str = "",
        images: list[str] | None = None,
    ) -> dict:
        """Append an incoming (or self-sent) message to unread + recent.

        A new incoming message also invalidates the pre-sleep wait markers,
        mirroring qq-bridge: after a reply the AI must re-observe before it
        may dive again.

        Args:
            conv_id: Conversation identifier.
            sender: Sender display name.
            user_id: Sender platform id.
            text: Message text (truncated to 200 chars).
            message_id: Platform message id.
            quote_target_is_self: Whether a quote targets the bot itself.
            is_self: Whether this message was sent by the bot.
            kind: Optional kind tag (``poke`` for poke events).
            images: Optional image URLs contained in the message.

        Returns:
            The appended message dict.
        """
        st = self.get_state(conv_id)
        text = str(text or "").strip()[:200]
        msg = {
            "seq": int(st.get("last_unread_seq", 0)) + 1,
            "message_id": str(message_id or ""),
            "sender": sender or "未知",
            "user_id": str(user_id or ""),
            "text": text,
            "quote_target_is_self": bool(quote_target_is_self),
            "is_self": bool(is_self),
            "kind": kind,
            "images": list(images or [])[:5],
            "time": time.time(),
        }
        st["last_unread_seq"] = msg["seq"]
        st["recent_messages"] = (st.get("recent_messages") or [])[-(self.recent_limit - 1):] + [msg]
        if not is_self:
            st["unread"] = (st.get("unread") or [])[-(self.unread_limit - 1):] + [msg]
            st["last_incoming_at"] = time.time()
            st["pre_sleep_satisfied_at"] = 0
            st["pre_sleep_observed_at"] = 0
        self._save()
        return msg

    def record_sent(self, conv_id: str, messages: list[str]) -> None:
        """Record the agent's own sent messages into recent (sender='我').

        Args:
            conv_id: Conversation identifier.
            messages: The sent text segments.
        """
        st = self.get_state(conv_id)
        for text in messages:
            self.append_message(
                conv_id, "我", "", str(text), is_self=True
            )
        st["last_ai_reply_at"] = time.time()
        self._save()

    def record_wake(self, conv_id: str) -> None:
        """Record a wake event for rate limiting.

        Keeps the last 200 wake timestamps per conversation, mirroring
        qq-bridge's ``st.wakeTimes``.

        Args:
            conv_id: Conversation identifier.
        """
        st = self.get_state(conv_id)
        times = st.get("wake_times") or []
        times.append(time.time())
        st["wake_times"] = times[-200:]
        st["last_wake_at"] = times[-1]
        self._save()

    def wake_rate_exceeded(self, conv_id: str, per_minute: int, per_hour: int) -> bool:
        """Whether waking this conversation now would exceed the rate limit.

        Args:
            conv_id: Conversation identifier.
            per_minute: Max wakes per minute (0 = unlimited).
            per_hour: Max wakes per hour (0 = unlimited).

        Returns:
            True when the limit is reached and the wake should be skipped.
        """
        st = self._conversations.get(conv_id)
        if not st:
            return False
        now = time.time()
        times = [t for t in (st.get("wake_times") or []) if now - t < 3600]
        if per_minute > 0 and sum(1 for t in times if now - t < 60) >= per_minute:
            return True
        if per_hour > 0 and len(times) >= per_hour:
            return True
        return False

    def bump_no_action(self, conv_id: str, acted: bool, limit: int = 3) -> bool:
        """Track no-action turns and reset the wake config at the limit.

        Mirrors qq-bridge: a wake turn with neither a send nor mark_read /
        set_wake_config increments ``no_action_count``; reaching the limit
        resets the wake config to the default so the agent cannot get stuck.

        Args:
            conv_id: Conversation identifier.
            acted: Whether the turn took a real action (sent or tool call).
            limit: Consecutive no-action turns allowed.

        Returns:
            True when the wake config was reset.
        """
        st = self.get_state(conv_id)
        wc = st.get("wake_config") or {}
        if acted:
            wc["no_action_count"] = 0
            st["wake_config"] = wc
            self._save()
            return False
        count = int(wc.get("no_action_count", 0)) + 1
        if count >= max(1, limit):
            wc = default_wake_config()
            wc["no_action_count"] = 0
            st["wake_config"] = wc
            self._save()
            return True
        wc["no_action_count"] = count
        st["wake_config"] = wc
        self._save()
        return False

    def mark_read(self, conv_id: str) -> int:
        """Clear the unread queue (the agent has seen the messages).

        Args:
            conv_id: Conversation identifier.

        Returns:
            The number of messages marked read.
        """
        st = self.get_state(conv_id)
        count = len(st.get("unread") or [])
        st["unread"] = []
        wc = st.get("wake_config") or {}
        wc["no_action_count"] = 0
        if not wc.get("infinite") and not wc.get("sleep_until"):
            # mark_read 收尾：确认当前唤醒配置继续作为下一次唤醒（无限期）。
            wc["infinite"] = True
        wc["confirmed_at"] = time.time()
        wc["confirmed_by"] = "mark_read"
        st["wake_config"] = wc
        st["pre_sleep_satisfied_at"] = 0
        st["pre_sleep_observed_at"] = 0
        self._save()
        return count

    def set_wake_config(self, conv_id: str, config: dict) -> dict:
        """Update a conversation's wake config.

        Mirrors qq-bridge: active mode forces any_message; infinite diving
        requires at least one trigger (anti-permanent-sleep); finite sleep
        needs a sleep_until timestamp.

        Args:
            conv_id: Conversation identifier.
            config: The incoming wake config (partial update).

        Returns:
            The updated wake config.

        Raises:
            ValueError: When the config would leave the agent un-wakeable.
        """
        st = self.get_state(conv_id)
        current = st.get("wake_config") or default_wake_config()
        cfg = config if isinstance(config, dict) else {}
        incoming_triggers = (
            cfg.get("triggers") if isinstance(cfg.get("triggers"), dict) else {}
        )
        cur_triggers = current.get("triggers") or {}

        mode = cfg.get("mode") or current.get("mode") or "diving"
        if mode not in ("active", "diving"):
            mode = "diving"
        infinite = cfg.get("infinite")
        if not isinstance(infinite, bool):
            infinite = current.get("infinite", True)
        # 从 active 切回 diving 时未显式保留 any_message 则清除，
        # 避免“潜水=每条都唤醒”的语义矛盾。
        any_message = normalize_trigger_bool(
            "any_message", incoming_triggers, cur_triggers.get("any_message", False)
        )
        if mode == "diving" and "any_message" not in incoming_triggers:
            any_message = False
        if mode == "active":
            any_message = True

        keywords = incoming_triggers.get("keywords")
        if keywords is None:
            keywords = cur_triggers.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k or "").strip()[:100] for k in keywords if str(k or "").strip()][:50]

        probability = incoming_triggers.get("probability")
        if probability is None:
            probability = cur_triggers.get("probability", 0)
        try:
            probability = min(1.0, max(0.0, float(probability)))
        except (TypeError, ValueError):
            probability = 0.0

        next_triggers = {
            "at_mention": normalize_trigger_bool(
                "at_mention", incoming_triggers, cur_triggers.get("at_mention", True)
            ),
            "name_mention": normalize_trigger_bool(
                "name_mention", incoming_triggers, cur_triggers.get("name_mention", True)
            ),
            "question": normalize_trigger_bool(
                "question", incoming_triggers, cur_triggers.get("question", True)
            ),
            "poke": normalize_trigger_bool(
                "poke", incoming_triggers, cur_triggers.get("poke", True)
            ),
            "any_message": any_message,
            "probability": probability,
            "keywords": keywords,
            "speaker_ids": normalize_speaker_ids(
                incoming_triggers.get("speaker_ids", cur_triggers.get("speaker_ids"))
            ),
        }
        if conv_id.startswith("private:"):
            # 私聊不适用“指定成员发言醒来”。
            next_triggers["speaker_ids"] = []

        sleep_until = current.get("sleep_until")
        if infinite:
            sleep_until = None
        elif cfg.get("sleep_until"):
            try:
                sleep_until = float(cfg.get("sleep_until"))
            except (TypeError, ValueError):
                sleep_until = None
        elif cfg.get("sleep_minutes"):
            try:
                sleep_until = time.time() + float(cfg.get("sleep_minutes")) * 60
            except (TypeError, ValueError):
                sleep_until = None
        elif not sleep_until:
            # 既没有无限也没有时间：使用推荐的有限时长（30 分钟 ~ 2 小时）。
            sleep_until = time.time() + 1800 * 2

        # 防永眠：无限期潜水必须至少有一个可触发条件。
        has_trigger = (
            next_triggers["at_mention"]
            or next_triggers["name_mention"]
            or next_triggers["poke"]
            or next_triggers["question"]
            or next_triggers["any_message"]
            or next_triggers["probability"] > 0
            or bool(next_triggers["keywords"])
            or bool(next_triggers["speaker_ids"])
        )
        if infinite and not has_trigger:
            raise ValueError(
                "无限期潜水必须至少保留一个唤醒条件（@/名字/拍一拍/提问/关键词/概率>0），"
                "否则会永眠"
            )

        wc = {
            **current,
            "mode": mode,
            "infinite": infinite,
            "sleep_until": sleep_until,
            "triggers": next_triggers,
            "confirmed_at": time.time(),
            "confirmed_by": "ai",
        }
        st["wake_config"] = wc
        self._save()
        return wc

    def ensure_wakeable(self, conv_id: str) -> bool:
        """Anti-permanent-sleep guard: reset to defaults when un-wakeable.

        Args:
            conv_id: Conversation identifier.

        Returns:
            True when the config was reset.
        """
        st = self._conversations.get(conv_id)
        if not st:
            return False
        wc = st.get("wake_config") or {}
        tr = wc.get("triggers") or {}
        tr["speaker_ids"] = normalize_speaker_ids(tr.get("speaker_ids"))
        timed = (
            not wc.get("infinite")
            and wc.get("sleep_until")
            and float(wc.get("sleep_until", 0)) > time.time()
        )
        wakeable = (
            wc.get("mode") == "active"
            or tr.get("any_message")
            or tr.get("at_mention")
            or tr.get("name_mention")
            or tr.get("poke")
            or tr.get("question")
            or float(tr.get("probability", 0)) > 0
            or bool(tr.get("keywords"))
            or bool(tr.get("speaker_ids"))
            or timed
        )
        if not wakeable:
            st["wake_config"] = default_wake_config()
            self._save()
            return True
        return False

    def is_sleeping(self, conv_id: str) -> bool:
        """Whether the conversation's wake config is a sleeping one.

        Args:
            conv_id: Conversation identifier.

        Returns:
            True when diving without any_message.
        """
        st = self._conversations.get(conv_id)
        if not st:
            return True
        wc = st.get("wake_config") or {}
        if wc.get("mode") == "active" or (wc.get("triggers") or {}).get("any_message"):
            return False
        return True

    def should_wake(self, conv_id: str) -> bool:
        """Whether a sleeping conversation's wake time has arrived.

        Args:
            conv_id: Conversation identifier.

        Returns:
            True when a finite sleep_until has passed.
        """
        st = self._conversations.get(conv_id)
        if not st:
            return False
        wc = st.get("wake_config") or {}
        if wc.get("mode") == "active" or (wc.get("triggers") or {}).get("any_message"):
            return True
        if not wc.get("infinite") and wc.get("sleep_until"):
            return float(wc.get("sleep_until", 0)) <= time.time()
        return False

    def pre_sleep_blocked(self, conv_id: str) -> tuple[bool, int]:
        """Pre-sleep observation guard: forbid diving right after a chat.

        The agent must either have had no new messages for the observation
        window, or have completed one full wait (via the wait tool), or the
        other side said an explicit goodbye.

        Args:
            conv_id: Conversation identifier.

        Returns:
            ``(blocked, remaining_ms)`` tuple.
        """
        st = self._conversations.get(conv_id)
        if not st:
            return False, 0
        wait_ms = 300_000
        last = st.get("last_recent_non_self_text") or self._last_non_self_text(st)
        if last and EXPLICIT_END_RE.search(last):
            return False, 0
        now = time.time()
        if st.get("last_incoming_at") and now - st["last_incoming_at"] >= wait_ms:
            return False, 0
        if st.get("pre_sleep_satisfied_at") and (
            not st.get("last_incoming_at")
            or st["last_incoming_at"] <= st["pre_sleep_satisfied_at"]
        ):
            return False, 0
        if st.get("pre_sleep_observed_at") and (
            not st.get("last_incoming_at")
            or st["last_incoming_at"] <= st["pre_sleep_observed_at"]
        ):
            return False, 0
        base = st.get("last_incoming_at") or 0
        remaining = max(0, int((wait_ms - (now - base) * 1000) if base else wait_ms))
        return True, remaining

    @staticmethod
    def _last_non_self_text(st: dict) -> str:
        """Last non-self message text from recent, for the explicit-end check.

        Args:
            st: The conversation state.

        Returns:
            The text or an empty string.
        """
        for msg in reversed(st.get("recent_messages") or []):
            if msg and not msg.get("is_self"):
                return str(msg.get("text") or "")
        return ""

    def set_pre_sleep(self, conv_id: str, satisfied: bool, observed: bool) -> None:
        """Set the pre-sleep wait markers after a wait tool call.

        Args:
            conv_id: Conversation identifier.
            satisfied: Whether a full observation window elapsed with no new
                messages.
            observed: Whether an observation was made (possibly with new
                messages returned to the agent).
        """
        st = self.get_state(conv_id)
        now = time.time()
        if satisfied:
            st["pre_sleep_satisfied_at"] = now
        if observed:
            st["pre_sleep_observed_at"] = now
        self._save()

    def snapshot(self) -> dict:
        """Export all conversations for the WebUI / admin commands.

        Returns:
            A summary dict per conversation.
        """
        return {
            key: {
                "mode": (st.get("wake_config") or {}).get("mode"),
                "unread": len(st.get("unread") or []),
                "recent": len(st.get("recent_messages") or []),
                "last_wake_reason": st.get("last_wake_reason", ""),
            }
            for key, st in self._conversations.items()
        }
