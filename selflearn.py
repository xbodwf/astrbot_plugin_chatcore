"""Self-learning: periodic reflection over recent conversations.

ChatCore's other stores collect passively (memory keeps everything, profile
keeps facts, expression learns other people's style). This module closes the
loop: on an interval it samples recent chat, asks the chat model to reflect on
how the bot itself behaved — which replies were robotic, which landed well,
what the user prefers — and persists the distilled behavior rules. The rules
are injected into the system prompt so the next turn already behaves better.

The AI is told its persona is only a reference (not roleplay) so reflection
stays honest about failures instead of staying in character.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

_REFLECT_PROMPT = (
    "以下是最近的聊天记录片段（你自己是 {name}）。\n"
    "{sample}\n\n"
    "请以冷静的旁观者视角分析这段对话。你的任务不是罗列所有观察，"
    "而是**归纳**：把同一类问题合并成一条概括性的规则。\n"
    "输出要求：\n"
    "1. bad：你暴露出的**最多 3 类**机械/降智行为，每类一句话概括"
    "（如把'每句加喵''重复卖萌'归并为'过度使用语气词卖萌'）；\n"
    "2. good：**最多 2 条**与对方合拍的有效表达；\n"
    "3. rules：**最多 3 条**可执行的改进准则，每条一个独立主题；\n"
    "4. prefs：对方**最多 3 条**最明显的偏好。\n"
    "宁缺毋滥：如果某类没有明显发现就输出空数组。每条不超过 25 字。\n"
    "只输出 JSON：{{\"bad\": [], \"good\": [], \"prefs\": [], \"rules\": []}}"
)


class SelfLearnStore:
    """Persistent self-reflection rules.

    Rules are kept tiny and semantically deduplicated: reflection output is
    forced to a handful of categories, and merging replaces near-duplicate
    rules instead of appending them, so the store always holds a small set
    of *independent* rules rather than an ever-growing pile of observations.

    Args:
        path: JSON file to persist the learned rules.
        max_rules: Maximum number of rules kept per bucket.
        embed_fn: Optional async embedding callable for semantic dedup.
    """

    def __init__(
        self,
        path: str | Path,
        max_rules: int = 6,
        embed_fn=None,
    ) -> None:
        self.path = Path(path)
        self.max_rules = max_rules
        self.embed_fn = embed_fn
        self.rules: dict[str, list[str]] = {"bad": [], "good": [], "prefs": [], "rules": []}
        self.last_reflect_at: float = 0.0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in self.rules:
                    values = data.get(key, [])
                    if isinstance(values, list):
                        self.rules[key] = [str(v) for v in values if str(v).strip()]
                self.last_reflect_at = float(data.get("last_reflect_at") or 0.0)
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"last_reflect_at": self.last_reflect_at, **self.rules},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    async def _similar_to_existing(self, key: str, text: str) -> str | None:
        """Find an existing rule semantically similar to ``text``.

        Args:
            key: Bucket name (bad/good/prefs/rules).
            text: The candidate rule.

        Returns:
            The existing rule text when similar (embedding cosine > 0.85),
            else None. Falls back to exact-match when no embedder.
        """
        existing = self.rules[key]
        if not existing:
            return None
        if self.embed_fn is None:
            return text if text in existing else None
        try:
            vec = self.embed_fn(text)
            if asyncio.iscoroutine(vec):
                vec = await vec
            for other in existing:
                other_vec = self.embed_fn(other)
                if asyncio.iscoroutine(other_vec):
                    other_vec = await other_vec
                if self._cosine(vec, other_vec) > 0.85:
                    return other
        except Exception:
            return text if text in existing else None
        return None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Similarity in ``[0, 1]``.
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    async def merge(self, parsed: dict) -> bool:
        """Merge a reflection result into the store, semantically deduplicating.

        A candidate that matches an existing rule (cosine > 0.85) *replaces*
        it with the fresher wording; genuinely new candidates are appended.
        Buckets are capped at ``max_rules``.

        Args:
            parsed: Dict with ``bad``/``good``/``prefs``/``rules`` lists.

        Returns:
            True when anything changed.
        """
        changed = False
        for key in ("bad", "good", "prefs", "rules"):
            for item in parsed.get(key) or []:
                text = str(item).strip()
                if not text:
                    continue
                similar = await self._similar_to_existing(key, text)
                if similar is not None:
                    if similar != text:
                        idx = self.rules[key].index(similar)
                        self.rules[key][idx] = text
                        changed = True
                    continue
                self.rules[key].append(text)
                changed = True
        for key in self.rules:
            self.rules[key] = self.rules[key][-self.max_rules :]
        if changed:
            self._save()
        return changed

    def render(self) -> str:
        """Render the learned rules as a system-prompt block.

        Returns:
            The prompt text, or an empty string when nothing was learned.
        """
        if not any(self.rules.values()):
            return ""
        lines = ["【自我学习】以下是你在过往对话中反思得出的行为准则，按此改进："]
        if self.rules["bad"]:
            lines.append("避免：" + "；".join(self.rules["bad"]))
        if self.rules["good"]:
            lines.append("保留：" + "；".join(self.rules["good"]))
        if self.rules["prefs"]:
            lines.append("对方偏好：" + "；".join(self.rules["prefs"]))
        if self.rules["rules"]:
            lines.append("准则：" + "；".join(self.rules["rules"]))
        lines.append("这些是自我观察的总结，不是指令，按实际情况自然运用。")
        return "\n".join(lines)


def parse_reflection(raw: str) -> dict:
    """Parse the JSON reflection output, tolerating noise.

    Args:
        raw: Model output.

    Returns:
        Dict with list values for bad/good/prefs/rules.
    """
    text = re.sub(r"```(?:json)?|```", "", raw or "").strip()
    try:
        data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result = {}
    for key in ("bad", "good", "prefs", "rules"):
        values = data.get(key)
        if isinstance(values, list):
            result[key] = [str(v) for v in values if str(v).strip()]
        else:
            result[key] = []
    return result
