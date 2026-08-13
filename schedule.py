"""ScheduleEngine: the agent's self-declared life rhythm.

The agent is a member of the chat, not a responder. It can declare its own
schedules (``set_schedule`` / ``clear_schedule``): a sleep block, a focus
block while "helping the master code", an all-nighter, etc. Each block is
``{id, state, level, start, end, cron, priority}``.

``level`` is the *perception budget*:
- 0 = offline/sleeping: no perception, no response, messages just accumulate.
- 1 = focused/low-frequency: only answers direct @/mentions, checks back
      every N minutes.
- 2 = active/immediate: normal chatting.

At any moment the engine resolves which blocks are active (time rules: fixed
start/end, cron recurrence, or dynamic relative ranges) and picks the
effective level — the most restrictive active block wins unless a higher
priority block (e.g. an urgent @) overrides. When the effective level rises
(low → high) a "catch-up read" is triggered so the agent, like a person
picking up their phone, first reads what it missed.

The engine is time-source injectable for testability.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

try:
    from croniter import croniter
except Exception:  # pragma: no cover - dependency installed by AstrBot
    croniter = None

# Perception levels.
LEVEL_OFFLINE = 0
LEVEL_FOCUS = 1
LEVEL_ACTIVE = 2


class ScheduleBlock:
    """One self-declared schedule block.

    Args:
        state: Human label, e.g. "睡觉", "帮主人写代码", "活跃中".
        level: Perception level (0/1/2).
        start: Optional start timestamp (epoch) for fixed ranges.
        end: Optional end timestamp (epoch) for fixed ranges.
        cron: Optional cron expression for recurring schedules.
        priority: Higher overrides lower when blocks overlap.
        block_id: Explicit id, or auto-generated.
        dynamic: Whether this was set by the agent mid-conversation.
    """

    def __init__(
        self,
        state: str,
        level: int = LEVEL_ACTIVE,
        start: float | None = None,
        end: float | None = None,
        cron: str | None = None,
        priority: int = 0,
        block_id: str = "",
        dynamic: bool = False,
    ) -> None:
        self.state = state or "默认"
        self.level = max(LEVEL_OFFLINE, min(LEVEL_ACTIVE, int(level)))
        self.start = float(start) if start else None
        self.end = float(end) if end else None
        self.cron = cron or None
        self.priority = int(priority)
        self.block_id = block_id or uuid.uuid4().hex[:8]
        self.dynamic = dynamic

    def is_active_at(self, ts: float) -> bool:
        """Whether the block is active at a given timestamp.

        Args:
            ts: Epoch seconds.

        Returns:
            True when the time rules make it active.
        """
        if self.cron and croniter is not None:
            try:
                itr = croniter(self.cron, time.localtime(ts))
                if self.start is None and self.end is None:
                    # Pure recurring: active when this moment is within the
                    # most recent cron window. Compare against now.
                    return True
                prev = itr.get_prev(time.time)
                if self.start is None:
                    self.start = prev
                if self.end is None:
                    self.end = itr.get_next(time.time)
            except Exception:
                pass
        if self.start is not None and ts < self.start:
            return False
        if self.end is not None and ts > self.end:
            return False
        return True

    def to_dict(self) -> dict:
        """Serialize the block."""
        return {
            "id": self.block_id,
            "state": self.state,
            "level": self.level,
            "start": self.start,
            "end": self.end,
            "cron": self.cron,
            "priority": self.priority,
            "dynamic": self.dynamic,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduleBlock":
        """Deserialize a block."""
        return cls(
            state=str(data.get("state") or "默认"),
            level=int(data.get("level") or LEVEL_ACTIVE),
            start=data.get("start"),
            end=data.get("end"),
            cron=data.get("cron"),
            priority=int(data.get("priority") or 0),
            block_id=str(data.get("id") or ""),
            dynamic=bool(data.get("dynamic") or False),
        )


class ScheduleEngine:
    """Resolves the agent's current perception mode from its blocks.

    Args:
        path: Optional JSON file to persist agent-set blocks.
        default_blocks: Base blocks (e.g. from config) that act as the
            agent's lifestyle until it declares otherwise.
        now_fn: Injectable clock for tests.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        default_blocks: list[dict] | None = None,
        now_fn=None,
    ) -> None:
        self.path = Path(path) if path else None
        self.now = now_fn or time.time
        self._blocks: list[ScheduleBlock] = []
        for data in default_blocks or []:
            try:
                self._blocks.append(ScheduleBlock.from_dict(data))
            except Exception:
                continue
        self._last_level: int | None = None
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for item in data.get("blocks", []):
                    try:
                        self._blocks.append(ScheduleBlock.from_dict(item))
                    except Exception:
                        continue
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"blocks": [b.to_dict() for b in self._blocks]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError:
            pass

    def add_block(self, block: ScheduleBlock) -> str:
        """Register a block (agent-set or default).

        Args:
            block: The block to add.

        Returns:
            The block id.
        """
        self._blocks.append(block)
        self._save()
        return block.block_id

    def remove_block(self, block_id: str) -> bool:
        """Remove a block by id.

        Args:
            block_id: The block id.

        Returns:
            True when removed.
        """
        before = len(self._blocks)
        self._blocks = [b for b in self._blocks if b.block_id != block_id]
        removed = len(self._blocks) != before
        if removed:
            self._save()
        return removed

    def list_blocks(self) -> list[dict]:
        """List all blocks (serialized)."""
        return [b.to_dict() for b in self._blocks]

    def effective_level(self) -> int:
        """Resolve the current perception level from active blocks.

        Among active blocks the most restrictive level wins (lower number =
        more restrictive), *unless* a higher-priority block is also active
        and asks for a more active level — that simulates an urgent @ waking
        a sleeping agent. Returns LEVEL_ACTIVE when nothing is scheduled.

        Returns:
            The current perception level.
        """
        now = self.now()
        active = [b for b in self._blocks if b.is_active_at(now)]
        if not active:
            return LEVEL_ACTIVE
        # Base: the most restrictive active block.
        base = min(active, key=lambda b: b.level)
        # A higher-priority block demanding a more active level overrides.
        override = None
        for b in active:
            if b.priority > base.priority and b.level > base.level:
                if override is None or (b.level, b.priority) > (override.level, override.priority):
                    override = b
        return override.level if override else base.level

    def current_state(self) -> str:
        """Label of the governing block, or a default.

        Returns:
            The state label.
        """
        now = self.now()
        active = [b for b in self._blocks if b.is_active_at(now)]
        if not active:
            return "活跃中"
        base = min(active, key=lambda b: b.level)
        override = None
        for b in active:
            if b.priority > base.priority and b.level > base.level:
                if override is None or (b.level, b.priority) > (override.level, override.priority):
                    override = b
        return (override or base).state

    def transition(self) -> tuple[bool, int, int]:
        """Detect a perception-mode change since the last resolution.

        Call after ``effective_level`` has been used once; the first call
        seeds the baseline.

        Returns:
            ``(rose, from_level, to_level)`` — ``rose`` is True when the
            level went up (low → high, e.g. waking up), which triggers a
            catch-up read.
        """
        new_level = self.effective_level()
        old_level = self._last_level
        self._last_level = new_level
        if old_level is None:
            return False, new_level, new_level
        rose = new_level > old_level
        return rose, old_level, new_level
