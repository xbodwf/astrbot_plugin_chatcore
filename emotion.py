"""Emotion / mood state machine.

MaiBot-style personas often talk in a single flat tone. ChatCore keeps a
per-conversation emotion state whose tone shifts with the conversation's
atmosphere: after a reply the state may switch to another configured state
(probability-weighted), and after a period of silence it decays back to the
persona's baseline.

The state is only ever a hint injected into the system prompt; the model still
writes its own words.
"""

import random
import time

_TRAIT_BASELINES = {
    "rational_calm": {
        "key": "理性平静",
        "description": "措辞客观、克制，不长篇大论",
    },
    "sentimental": {
        "key": "感性细腻",
        "description": "温和、细腻，愿意共情",
    },
    "neutral": {
        "key": "中性",
        "description": "语气平和，不刻意加长或缩短回复",
    },
}

DEFAULT_STATES = [
    {
        "key": "兴奋",
        "description": "热情、话变多、可以多用感叹号",
        "switch_prob": 0.20,
    },
    {
        "key": "慵懒",
        "description": "话少、简短、慢悠悠",
        "switch_prob": 0.20,
    },
    {
        "key": "严肃",
        "description": "认真、正经、尽量客观严谨",
        "switch_prob": 0.15,
    },
    {
        "key": "调侃",
        "description": "带点玩笑、损友风格，但不过分",
        "switch_prob": 0.15,
    },
]

_EXCITED_MARKERS = ("!", "！", "哈哈", "笑死", "hhh", "绷不住", "卧槽", "666", "牛")
_SERIOUS_MARKERS = ("烦", "气", "滚", "怒", "正经", "说正事", "别闹")


class EmotionManager:
    """Per-conversation mood state machine.

    Args:
        trait: Baseline personality trait; one of ``rational_calm``,
            ``sentimental``, ``neutral``.
        states: List of switchable mood states, each a dict with ``key``,
            ``description`` and ``switch_prob``.
        switch_probability: Probability (0~1) that an atmosphere change
            actually switches the current state.
        decay_seconds: Idle time after which the mood reverts to baseline.
    """

    def __init__(
        self,
        trait: str = "neutral",
        states: list[dict] | None = None,
        switch_probability: float = 0.5,
        decay_seconds: float = 1800.0,
    ) -> None:
        self.trait = trait if trait in _TRAIT_BASELINES else "neutral"
        self.states = states if states else DEFAULT_STATES
        self.switch_probability = max(0.0, min(switch_probability, 1.0))
        self.decay_seconds = max(60.0, decay_seconds)
        self._states: dict[str, dict] = {}

    def _baseline(self) -> dict:
        return dict(_TRAIT_BASELINES[self.trait])

    def _get_state(self, conv_id: str) -> dict:
        state = self._states.get(conv_id)
        if state is None:
            state = {
                "key": self._baseline()["key"],
                "updated_at": time.time(),
            }
            self._states[conv_id] = state
        return state

    def current(self, conv_id: str) -> dict:
        """Return the effective mood, decaying to baseline when stale.

        Args:
            conv_id: Conversation identifier.

        Returns:
            A dict with ``key``, ``description`` and ``updated_at``.
        """
        state = self._get_state(conv_id)
        if state["key"] != self._baseline()["key"]:
            if time.time() - state["updated_at"] > self.decay_seconds:
                state["key"] = self._baseline()["key"]
                state["updated_at"] = time.time()
        key = state["key"]
        for s in self.states:
            if s["key"] == key:
                return {"key": key, "description": s["description"]}
        base = self._baseline()
        return {"key": key, "description": base["description"]}

    @staticmethod
    def _hint_state(user_text: str) -> str | None:
        """Heuristically guess the mood suggested by a user's message.

        Args:
            user_text: The triggering user message text.

        Returns:
            A state key hint, or None to keep the current mood.
        """
        text = user_text or ""
        if any(m in text for m in _EXCITED_MARKERS):
            return "兴奋"
        if any(m in text for m in _SERIOUS_MARKERS):
            return "严肃"
        return None

    def update_after_reply(
        self,
        conv_id: str,
        user_text: str,
        assistant_text: str = "",
    ) -> str:
        """Maybe switch mood after a reply based on the user's message.

        A hint is derived from the user's text; the switch only happens with
        probability ``switch_probability``, and when the hint is unavailable a
        random state is chosen weighted by each state's ``switch_prob``.

        Args:
            conv_id: Conversation identifier.
            user_text: The user message that triggered the reply.
            assistant_text: The assistant's reply (unused but kept for parity).

        Returns:
            The (possibly new) active state key.
        """
        state = self._get_state(conv_id)
        if random.random() >= self.switch_probability:
            return state["key"]
        hint = self._hint_state(user_text)
        if hint and any(s["key"] == hint for s in self.states):
            state["key"] = hint
            state["updated_at"] = time.time()
            return hint
        weights = [max(0.0, s.get("switch_prob", 0.1)) for s in self.states]
        total = sum(weights)
        if total <= 0:
            return state["key"]
        pick = random.random() * total
        for s, w in zip(self.states, weights):
            pick -= w
            if pick <= 0:
                state["key"] = s["key"]
                state["updated_at"] = time.time()
                return s["key"]
        return state["key"]

    def reset(self, conv_id: str) -> None:
        """Drop a conversation's mood back to its baseline.

        Args:
            conv_id: Conversation identifier.
        """
        self._states.pop(conv_id, None)

    def snapshot(self) -> list[dict]:
        """Export a management-friendly view of all moods (WebUI).

        Returns:
            One summary dict per tracked conversation.
        """
        return [
            {
                "conv_id": conv_id,
                "mood": self.current(conv_id)["key"],
                "trait": self.trait,
            }
            for conv_id in self._states
        ]

    def inject_text(self, conv_id: str) -> str:
        """Build the system-prompt fragment for the current mood.

        Args:
            conv_id: Conversation identifier.

        Returns:
            A text block to append, or an empty string when neutral-baseline.
        """
        mood = self.current(conv_id)
        return (
            "\n\n【当前状态】当前情绪: "
            + mood["key"]
            + "（"
            + mood["description"]
            + "）。按此情绪基调自然措辞，不要生硬提及本条说明。"
        )
