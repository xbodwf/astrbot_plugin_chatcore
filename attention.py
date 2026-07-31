"""Attention state machine for group chat.

ChatCore decides whether the AI should spontaneously join a group
conversation. Each group keeps a lightweight state; probabilities are
computed lazily on read (no background timers).
"""

import math
import random
import time


class AttentionManager:
    """Per-group probability state machine.

    Probability model::

        idle_min = (now - last_interaction_ts) / 60
        active_contribution = (active_cap - bubble_base) * exp(-k * idle_min / decay_minutes)
        boost_remaining = sum(boost_i * exp(-k * (now - ts_i) / decay_minutes))
        prob = clamp(bubble_base + active_contribution + boost_remaining, 0, active_cap)

    Args:
        bubble_base: Base probability to chime in with no interaction (1%~3%).
        active_cap: Upper probability bound while engaged in conversation.
        decay_minutes: After this many idle minutes the probability decays back
            to the baseline.
        hard_trigger_boost: Extra probability added each time the AI is
            @-mentioned / replied to.
    """

    _DECAY_FACTOR = 3.0

    def __init__(
        self,
        bubble_base: float = 0.02,
        active_cap: float = 0.30,
        decay_minutes: float = 10.0,
        hard_trigger_boost: float = 0.10,
    ) -> None:
        self.bubble_base = max(0.0, min(bubble_base, 1.0))
        self.active_cap = max(self.bubble_base, min(active_cap, 1.0))
        self.decay_minutes = max(0.1, decay_minutes)
        self.hard_trigger_boost = max(0.0, hard_trigger_boost)
        self._states: dict[str, dict] = {}

    def _get_state(self, group_id: str) -> dict:
        state = self._states.get(group_id)
        if state is None:
            state = {
                "last_interaction_ts": time.time(),
                "boosts": [],
            }
            self._states[group_id] = state
        return state

    def _decay(self, elapsed_min: float) -> float:
        return math.exp(
            -self._DECAY_FACTOR * elapsed_min / self.decay_minutes,
        )

    def record_interaction(self, group_id: str) -> None:
        """Record that the AI actively interacted with the group.

        Args:
            group_id: Group identifier.
        """
        state = self._get_state(group_id)
        state["last_interaction_ts"] = time.time()

    def record_hard_trigger(self, group_id: str) -> None:
        """Record an @-mention / reply: refreshes activity and adds a boost.

        Args:
            group_id: Group identifier.
        """
        state = self._get_state(group_id)
        now = time.time()
        state["last_interaction_ts"] = now
        state["boosts"].append({"ts": now, "value": self.hard_trigger_boost})
        # Keep only boosts that still contribute something.
        state["boosts"] = [
            b
            for b in state["boosts"]
            if b["value"] * self._decay((now - b["ts"]) / 60) > 1e-4
        ]

    def current_probability(self, group_id: str) -> float:
        """Compute the current reply probability for a group.

        Args:
            group_id: Group identifier.

        Returns:
            Probability in ``[0, active_cap]``.
        """
        state = self._get_state(group_id)
        now = time.time()
        idle_min = (now - state["last_interaction_ts"]) / 60.0

        active_contribution = (
            self.active_cap - self.bubble_base
        ) * self._decay(idle_min)

        boost_remaining = sum(
            b["value"] * self._decay((now - b["ts"]) / 60.0)
            for b in state["boosts"]
        )

        return max(0.0, min(self.active_cap, self.bubble_base + active_contribution + boost_remaining))

    def should_respond(self, group_id: str) -> bool:
        """Roll the dice for a soft-trigger reply.

        Does not mutate state; call ``record_interaction`` when the AI actually
        engages so only real interactions maintain the activity level.

        Args:
            group_id: Group identifier.

        Returns:
            True if the AI should reply this time.
        """
        prob = self.current_probability(group_id)
        return random.random() < prob

    def bump_probability(self, group_id: str, amount: float) -> None:
        """Raise the group's activity by a fixed amount (e.g. implicit analysis).

        Args:
            group_id: Group identifier.
            amount: Probability to add.
        """
        state = self._get_state(group_id)
        now = time.time()
        state["boosts"].append({"ts": now, "value": max(0.0, amount)})

    def reset(self, group_id: str) -> None:
        """Drop all state for a group.

        Args:
            group_id: Group identifier.
        """
        self._states.pop(group_id, None)
