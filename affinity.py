"""Per-user affinity (好感度) tracking.

A single closeness score per user (0-100) that grows with interaction,
decays over idle time, and shapes how the AI talks to that person (from
distant/formal at low affinity to familiar/casual at high affinity). Keys
are the platform user ids, so group and private chats of the same person
share one affinity. Persisted as JSON.
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

    def interact(self, user_id: str, amount: float = 1.0) -> None:
        """Bump a user's affinity and refresh the last-interaction time.

        Args:
            user_id: Platform user id.
            amount: Affinity delta (positive for friendly interaction).
        """
        current = self.get(user_id)
        self._values[user_id] = max(0.0, min(100.0, current + amount))
        self._last_ts[user_id] = time.time()
        self._save()

    def get(self, user_id: str) -> float:
        """Current affinity for a user, applying idle decay.

        Args:
            user_id: Platform user id.

        Returns:
            The affinity in ``[0, 100]``.
        """
        value = self._values.get(user_id, self.initial)
        last_ts = self._last_ts.get(user_id, 0.0)
        if last_ts > 0 and self.decay_per_day > 0:
            idle_days = max(0.0, (time.time() - last_ts) / 86400.0)
            value = max(0.0, value - idle_days * self.decay_per_day)
        return value

    def tier(self, user_id: str) -> tuple[str, str]:
        """Resolve the affinity tier of a user.

        Args:
            user_id: Platform user id.

        Returns:
            A ``(tier_name, description)`` tuple.
        """
        value = self.get(user_id)
        for threshold, name, desc in _TIERS:
            if value < threshold:
                return name, desc
        return _TIERS[-1][1], _TIERS[-1][2]

    def inject_text(self, user_id: str) -> str:
        """Build the system-prompt fragment for a user's affinity.

        Args:
            user_id: Platform user id.

        Returns:
            A short block describing the closeness to that user.
        """
        name, desc = self.tier(user_id)
        value = round(self.get(user_id))
        return (
            f"\n\n【当前关系】你与对方现在是{name}的关系（好感度 {value}），"
            f"{desc}。按这个亲疏程度自然把握语气和称呼，不要提及这条说明。"
        )

    def snapshot(self) -> list[dict]:
        """Export all affinity states for the WebUI.

        Returns:
            One summary dict per user.
        """
        return [
            {
                "user_id": user_id,
                "value": round(self.get(user_id), 1),
                "tier": self.tier(user_id)[0],
            }
            for user_id in self._values
        ]

    def reset(self, user_id: str) -> None:
        """Drop a user's affinity back to the initial value.

        Args:
            user_id: Platform user id.
        """
        self._values.pop(user_id, None)
        self._last_ts.pop(user_id, None)
        self._save()

    def set(self, user_id: str, value: float) -> None:
        """Set a user's affinity to an absolute value.

        Args:
            user_id: Platform user id.
            value: Target affinity in ``[0, 100]``.
        """
        self._values[user_id] = max(0.0, min(100.0, value))
        self._last_ts[user_id] = time.time()
        self._save()
