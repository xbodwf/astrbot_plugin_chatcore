"""Per-conversation affinity (好感度) tracking.

A single closeness score per conversation (0-100) that grows with
interaction, decays over idle time, and shapes how the AI talks to the user
(from distant/formal at low affinity to familiar/casual at high affinity).
Persisted as JSON.
"""

import json
import os
import time
from pathlib import Path

_INITIAL = 50.0
_DECAY_PER_DAY = 2.0

_TIERS = (
    (20, "冷淡", "关系疏远，保持客气与距离"),
    (40, "疏离", "不太熟悉，礼貌但少闲聊"),
    (60, "普通", "普通朋友，自然交流"),
    (80, "熟络", "熟识的朋友，语气放松、可以开玩笑"),
    (101, "亲密", "非常亲近，可以亲密称呼、随性说话"),
)


class AffinityManager:
    """Persistent per-conversation affinity scores.

    Args:
        path: Path of the JSON persistence file.
        initial: Starting affinity for a new conversation.
        decay_per_day: Affinity lost per day without interaction.
    """

    def __init__(
        self,
        path: str | Path,
        initial: float = _INITIAL,
        decay_per_day: float = _DECAY_PER_DAY,
    ) -> None:
        self.path = Path(path)
        self.initial = float(initial)
        self.decay_per_day = float(decay_per_day)
        self._values: dict[str, float] = {}
        self._last_ts: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            values = raw.get("values", {}) if isinstance(raw, dict) else {}
            last = raw.get("last_ts", {}) if isinstance(raw, dict) else {}
            self._values = {str(k): float(v) for k, v in values.items()}
            self._last_ts = {str(k): float(v) for k, v in last.items()}
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            self._values = {}
            self._last_ts = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"values": self._values, "last_ts": self._last_ts},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError:
            pass

    def interact(self, conv_id: str, amount: float = 1.0) -> None:
        """Bump a conversation's affinity and refresh its last-interaction time.

        Args:
            conv_id: Conversation identifier.
            amount: Affinity delta (positive for friendly interaction).
        """
        current = self.get(conv_id)
        self._values[conv_id] = max(0.0, min(100.0, current + amount))
        self._last_ts[conv_id] = time.time()
        self._save()

    def get(self, conv_id: str) -> float:
        """Current affinity for a conversation, applying idle decay.

        Args:
            conv_id: Conversation identifier.

        Returns:
            The affinity in ``[0, 100]``.
        """
        value = self._values.get(conv_id, self.initial)
        last_ts = self._last_ts.get(conv_id, 0.0)
        if last_ts > 0 and self.decay_per_day > 0:
            idle_days = max(0.0, (time.time() - last_ts) / 86400.0)
            value = max(0.0, value - idle_days * self.decay_per_day)
        return value

    def tier(self, conv_id: str) -> tuple[str, str]:
        """Resolve the affinity tier of a conversation.

        Args:
            conv_id: Conversation identifier.

        Returns:
            A ``(tier_name, description)`` tuple.
        """
        value = self.get(conv_id)
        for threshold, name, desc in _TIERS:
            if value < threshold:
                return name, desc
        return _TIERS[-1][1], _TIERS[-1][2]

    def inject_text(self, conv_id: str) -> str:
        """Build the system-prompt fragment for the current affinity.

        Args:
            conv_id: Conversation identifier.

        Returns:
            A short block describing the closeness to the other party.
        """
        name, desc = self.tier(conv_id)
        value = round(self.get(conv_id))
        return (
            f"\n\n【当前关系】你与对方现在是{name}的关系（好感度 {value}），"
            f"{desc}。按这个亲疏程度自然把握语气和称呼，不要提及这条说明。"
        )

    def snapshot(self) -> list[dict]:
        """Export all affinity states for the WebUI.

        Returns:
            One summary dict per conversation.
        """
        return [
            {
                "conv_id": conv_id,
                "value": round(self.get(conv_id), 1),
                "tier": self.tier(conv_id)[0],
            }
            for conv_id in self._values
        ]

    def reset(self, conv_id: str) -> None:
        """Drop a conversation's affinity back to the initial value.

        Args:
            conv_id: Conversation identifier.
        """
        self._values.pop(conv_id, None)
        self._last_ts.pop(conv_id, None)
        self._save()
