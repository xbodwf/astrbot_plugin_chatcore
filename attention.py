"""Attention state machine for group chat.

ChatCore decides whether the AI should spontaneously join a group
conversation. Each group keeps a lightweight state; probabilities are
computed lazily on read (no background timers).

Beyond the base probability model this module implements the "read the room"
backoff strategy: a cooldown right after the bot replies, a temporary
probability drop after consecutive soft-trigger misses, per-time-window
probability rules, and a density-aware suppression when other members are
dominating the conversation without addressing the bot.
"""

import math
import random
import time

_TONIGHT = "23:59"
_MIDNIGHT = "00:00"


class AttentionManager:
    """Per-group probability state machine.

    Probability model::

        idle_min = (now - last_interaction_ts) / 60
        active_contribution = (active_cap - bubble_base) * exp(-k * idle_min / decay_minutes)
        boost_remaining = sum(boost_i * exp(-k * (now - ts_i) / decay_minutes))
        prob = clamp(bubble_base + active_contribution + boost_remaining, 0, active_cap)

    ``should_respond`` returns False while the group is in its post-reply
    cooldown. Otherwise the base probability is scaled by the no-action
    backoff multiplier, the current time-window rule and the "read the room"
    density factor, then clamped to ``active_cap``.

    Args:
        bubble_base: Base probability to chime in with no interaction (1%~3%).
        active_cap: Upper probability bound while engaged in conversation.
        decay_minutes: After this many idle minutes the probability decays back
            to the baseline.
        hard_trigger_boost: Extra probability added each time the AI is
            @-mentioned / replied to.
        cool_down_seconds: Suppress soft triggers for this long right after the
            AI replied (hard triggers still respond).
        no_action_backoff: Multiplier applied per consecutive soft-trigger
            miss (``0 < value <= 1``).
        backoff_floor: Minimum multiplier the no-action backoff can reach.
        time_rules: Per-window probability multipliers; each item is a dict
            with ``start``/``end`` as ``HH:MM`` and a ``multiplier``.
        read_air_factor: Extra suppression (0~1) when other members dominate
            the conversation without addressing the bot.
        others_window_seconds: Window used to judge others' message density.
        others_density_threshold: Others' messages within the window at or
            above which they are considered to be dominating.
        followup_boost: Probability bonus when the topic continues right after
            the bot's last reply.
        followup_window_seconds: Window after a bot reply in which other
            messages are treated as continuing the topic.
    """

    _DECAY_FACTOR = 3.0

    def __init__(
        self,
        bubble_base: float = 0.02,
        active_cap: float = 0.30,
        decay_minutes: float = 10.0,
        hard_trigger_boost: float = 0.10,
        cool_down_seconds: float = 120.0,
        no_action_backoff: float = 0.6,
        backoff_floor: float = 0.25,
        time_rules: list[dict] | None = None,
        read_air_factor: float = 0.5,
        others_window_seconds: float = 180.0,
        others_density_threshold: int = 3,
        followup_boost: float = 0.05,
        followup_window_seconds: float = 180.0,
        poke_decay_seconds: float = 300.0,
    ) -> None:
        self.bubble_base = max(0.0, min(bubble_base, 1.0))
        self.active_cap = max(self.bubble_base, min(active_cap, 1.0))
        self.decay_minutes = max(0.1, decay_minutes)
        self.hard_trigger_boost = max(0.0, hard_trigger_boost)
        self.cool_down_seconds = max(0.0, cool_down_seconds)
        self.no_action_backoff = max(0.0, min(no_action_backoff, 1.0))
        self.backoff_floor = max(0.0, min(backoff_floor, 1.0))
        self.time_rules = [r for r in (time_rules or []) if isinstance(r, dict)]
        self.read_air_factor = max(0.0, min(read_air_factor, 1.0))
        self.others_window_seconds = max(1.0, others_window_seconds)
        self.others_density_threshold = max(1, int(others_density_threshold))
        self.followup_boost = max(0.0, followup_boost)
        self.followup_window_seconds = max(1.0, followup_window_seconds)
        self.poke_decay_seconds = max(1.0, poke_decay_seconds)
        self._states: dict[str, dict] = {}

    def _get_state(self, group_id: str) -> dict:
        state = self._states.get(group_id)
        if state is None:
            state = {
                "last_interaction_ts": time.time(),
                "boosts": [],
                "last_reply_ts": 0.0,
                "soft_misses": 0,
                "others_ts": [],
                "last_poke_ts": 0.0,
                "poke_total": 0.0,
                "poke_count": 0,
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

    def record_reply(self, group_id: str) -> None:
        """Record that the AI just sent a reply (starts the cooldown).

        Args:
            group_id: Group identifier.
        """
        self._get_state(group_id)["last_reply_ts"] = time.time()

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

    def record_soft_miss(self, group_id: str) -> None:
        """Record a soft-trigger chance the AI chose not to take.

        Consecutive misses reduce the next soft-trigger probability (backoff).

        Args:
            group_id: Group identifier.
        """
        self._get_state(group_id)["soft_misses"] += 1

    def record_soft_hit(self, group_id: str) -> None:
        """Record that a soft-trigger chance was taken (resets backoff).

        Args:
            group_id: Group identifier.
        """
        self._get_state(group_id)["soft_misses"] = 0

    def record_others_message(self, group_id: str) -> None:
        """Record that another member posted in the group.

        Used to judge whether members are dominating the conversation.

        Args:
            group_id: Group identifier.
        """
        state = self._get_state(group_id)
        now = time.time()
        state["others_ts"].append(now)
        state["others_ts"] = [
            t for t in state["others_ts"] if now - t <= self.others_window_seconds
        ]

    def _time_multiplier(self, now: float) -> float:
        """Resolve the current time-window probability multiplier.

        Args:
            now: Current unix time.

        Returns:
            The multiplier of the first matching rule, or 1.0 when none match.
        """
        if not self.time_rules:
            return 1.0
        struct = time.localtime(now)
        cur = f"{struct.tm_hour:02d}:{struct.tm_min:02d}"
        for rule in self.time_rules:
            start = str(rule.get("start", "")).strip() or _MIDNIGHT
            end = str(rule.get("end", "")).strip() or _TONIGHT
            in_rule = (
                start <= cur <= end if start <= end else (cur >= start or cur <= end)
            )
            if in_rule:
                try:
                    return float(rule.get("multiplier", 1.0))
                except (TypeError, ValueError):
                    return 1.0
        return 1.0

    def _read_air_multiplier(self, state: dict, now: float) -> float:
        """Compute the density-based read-the-room factor.

        Members dominating the conversation (without addressing the bot)
        suppress the bubble probability; a topic that keeps going right after
        the bot's last reply instead nudges the probability up.

        Args:
            state: The group's state dict.
            now: Current unix time.

        Returns:
            A multiplier for the base probability.
        """
        state["others_ts"] = [
            t for t in state["others_ts"] if now - t <= self.others_window_seconds
        ]
        dense = len(state["others_ts"]) >= self.others_density_threshold
        last_reply = state.get("last_reply_ts", 0.0)
        followup = last_reply > 0 and now - last_reply <= self.followup_window_seconds
        if followup:
            return 1.0 + self.followup_boost
        if dense:
            return self.read_air_factor
        return 1.0

    def current_probability(self, group_id: str) -> float:
        """Compute the current reply probability for a group.

        Returns 0.0 while the group is in the post-reply cooldown. Otherwise
        the base probability is scaled by the no-action backoff, the
        time-window rule and the read-the-room factor.

        Args:
            group_id: Group identifier.

        Returns:
            Probability in ``[0, active_cap]``.
        """
        state = self._get_state(group_id)
        now = time.time()
        if now - state["last_reply_ts"] < self.cool_down_seconds:
            return 0.0
        idle_min = (now - state["last_interaction_ts"]) / 60.0

        base = self.bubble_base + (self.active_cap - self.bubble_base) * self._decay(
            idle_min
        )
        base += sum(
            b["value"] * self._decay((now - b["ts"]) / 60.0) for b in state["boosts"]
        )

        backoff = max(
            self.backoff_floor,
            self.no_action_backoff ** state["soft_misses"],
        )
        prob = (
            base
            * backoff
            * self._time_multiplier(now)
            * self._read_air_multiplier(state, now)
        )
        return max(0.0, min(self.active_cap, prob))

    def record_poke(self, group_id: str) -> None:
        """Record a poke and update the poke-only reply probability.

        The first poke lifts the poke probability to 50%; each further poke
        adds 20% scaled by the poke cadence (dense pokes add more, sparse
        ones less). Three consecutive dense pokes force a guaranteed reply.
        Poke state never touches the normal chat probability.

        Args:
            group_id: Group identifier.
        """
        state = self._get_state(group_id)
        now = time.time()
        last = state.get("last_poke_ts", 0.0)
        if last <= 0:
            boost = 0.5
        else:
            gap = now - last
            if gap <= 15:
                factor = 1.0
            elif gap <= 120:
                factor = 0.5
            else:
                factor = 0.15
            boost = state.get("poke_total", 0.0) + 0.2 * factor
        state["poke_total"] = min(1.0, boost)
        state["last_poke_ts"] = now
        state["poke_count"] = state.get("poke_count", 0) + 1
        if state["poke_count"] >= 3 and now - last <= 30:
            state["poke_total"] = 1.0

    def effective_poke_probability(self, group_id: str) -> float:
        """Effective poke reply probability with natural decay.

        Before any poke this equals the normal chat probability; after pokes
        it is the max of that and the poke-accumulated target, which decays
        linearly back to the chat baseline over ``poke_decay_seconds``.

        Args:
            group_id: Group identifier.

        Returns:
            Probability in ``[0, 1]``.
        """
        state = self._get_state(group_id)
        now = time.time()
        poke_total = state.get("poke_total", 0.0)
        last = state.get("last_poke_ts", 0.0)
        if last > 0 and poke_total > 0:
            poke_total = max(
                0.0,
                poke_total - (now - last) / self.poke_decay_seconds,
            )
        base = self.current_probability(group_id)
        return max(0.0, min(1.0, max(base, poke_total)))

    def should_respond_poke(self, group_id: str) -> bool:
        """Roll the dice for a poke-triggered reply.

        Args:
            group_id: Group identifier.

        Returns:
            True when the AI should reply to this poke.
        """
        return random.random() < self.effective_poke_probability(group_id)

    def in_cooldown(self, group_id: str) -> bool:
        """Whether the group is inside the post-reply cooldown window.

        Args:
            group_id: Group identifier.

        Returns:
            True when the AI replied recently and hard triggers should be
            suppressed to avoid spamming the group.
        """
        state = self._get_state(group_id)
        return time.time() - state["last_reply_ts"] < self.cool_down_seconds

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

    def snapshot(self) -> list[dict]:
        """Export a management-friendly view of all group states (WebUI).

        Returns:
            One summary dict per tracked group.
        """
        now = time.time()
        rows = []
        for group_id, state in self._states.items():
            rows.append(
                {
                    "group_id": group_id,
                    "probability": round(self.current_probability(group_id), 4),
                    "in_cooldown": (
                        now - state["last_reply_ts"] < self.cool_down_seconds
                    ),
                    "soft_misses": state["soft_misses"],
                    "last_interaction_ts": round(state["last_interaction_ts"], 2),
                    "others_recent": len(state["others_ts"]),
                }
            )
        return rows
