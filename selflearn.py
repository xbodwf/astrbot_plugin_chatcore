"""Self-learning: periodic reflection over recent conversations.

ChatCore's other stores collect passively (memory keeps everything, profile
keeps facts, expression learns other people's style). This module closes the
loop: on an interval it samples recent chat, asks the chat model to reflect on
how the bot itself behaved — which replies were robotic, which landed well,
what the user prefers — and persists the distilled behavior rules. The rules
are injected into the system prompt so the next turn already behaves better.

Each learned rule carries a *scene* tag (``图片``, ``语气``, ``汇报式``, ...).
At injection time only rules whose scene matches the current conversation
context are injected, so the prompt stays small and relevant instead of
dumping every observation on every turn. Reflection is gated by both a wall
clock (interval) and a per-conversation new-message count, so the LLM is only
called when there is actually something new to learn from.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

# 场景关键词表：注入时用轻量关键词匹配（零 LLM 成本）。
_SCENE_KEYWORDS: dict[str, list[str]] = {
    "图片": ["图片", "图", "image", "截图", "照片", "表情包", "看到"],
    "戳": ["戳", "poke", "戳一戳", "捏"],
    "语气": ["喵", "卖萌", "语气词", "撒娇", "口癖"],
    "汇报式": ["汇报", "报告", "客服", "机械", "套话", "复读", "查询"],
    "长度": ["太长", "啰嗦", "刷屏", "分段", "字数", "话多"],
    "提问": ["反问", "提问", "追问", "问题"],
    "情绪": ["生气", "难过", "开心", "烦躁", "情绪", "安慰"],
    "关系": ["称呼", "主人", "关系", "好感"],
}

_SCENE_ORDER = ["图片", "戳", "语气", "汇报式", "长度", "提问", "情绪", "关系"]


def _match_scene(text: str, keywords: list[str]) -> bool:
    """Whether any keyword appears in the text.

    Args:
        text: The text to search.
        keywords: Scene keywords.

    Returns:
        True on any hit.
    """
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _infer_scenes(text: str) -> list[str]:
    """Infer which scenes a rule applies to, from its wording.

    Args:
        text: The rule text.

    Returns:
        Ordered scene names that match.
    """
    scenes = []
    for scene in _SCENE_ORDER:
        if _match_scene(text, _SCENE_KEYWORDS[scene]):
            scenes.append(scene)
    return scenes or ["通用"]


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
    "每条请用这样的结构：{{\"scene\": \"适用场景（图片/戳/语气/汇报式/长度/"
    "提问/情绪/关系/通用）\", \"text\": \"规则内容，不超过 25 字\"}}\n"
    "宁缺毋滥：如果某类没有明显发现就输出空数组。\n"
    "只输出 JSON：{{\"bad\": [], \"good\": [], \"prefs\": [], \"rules\": []}}"
)


class SelfLearnStore:
    """Persistent self-reflection rules with scene-scoped injection.

    Rules are kept tiny and semantically deduplicated: reflection output is
    forced to a handful of categories, and merging replaces near-duplicate
    rules instead of appending them. Every rule carries a scene tag; only the
    rules matching the current conversation are injected, keeping the prompt
    small and relevant.

    Args:
        path: JSON file to persist the learned rules.
        max_rules: Maximum number of rules kept per bucket.
        embed_fn: Optional async embedding callable for semantic dedup.
        min_new_messages: Minimum new messages before a conversation is
            reflected upon.
    """

    def __init__(
        self,
        path: str | Path,
        max_rules: int = 6,
        embed_fn=None,
        min_new_messages: int = 20,
    ) -> None:
        self.path = Path(path)
        self.max_rules = max_rules
        self.embed_fn = embed_fn
        self.min_new_messages = max(1, min_new_messages)
        self.rules: dict[str, list[dict]] = {
            "bad": [],
            "good": [],
            "prefs": [],
            "rules": [],
        }
        self.last_reflect_at: float = 0.0
        self._msg_counts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in self.rules:
                    values = data.get(key, [])
                    if isinstance(values, list):
                        cleaned = []
                        for v in values:
                            if isinstance(v, dict) and v.get("text"):
                                cleaned.append(
                                    {
                                        "scene": str(v.get("scene") or "通用"),
                                        "text": str(v["text"]),
                                    }
                                )
                            elif isinstance(v, str) and v.strip():
                                cleaned.append(
                                    {"scene": "通用", "text": v.strip()}
                                )
                        self.rules[key] = cleaned
                self.last_reflect_at = float(data.get("last_reflect_at") or 0.0)
                counts = data.get("msg_counts")
                if isinstance(counts, dict):
                    self._msg_counts = {
                        str(k): int(v) for k, v in counts.items()
                    }
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "last_reflect_at": self.last_reflect_at,
                    "msg_counts": self._msg_counts,
                    **self.rules,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def has_enough_new_messages(self, conv_id: str, current_count: int) -> bool:
        """Whether a conversation accumulated enough new messages to reflect.

        Args:
            conv_id: Conversation identifier.
            current_count: Current number of records in the conversation.

        Returns:
            True when at least ``min_new_messages`` new messages arrived
            since the last reflection.
        """
        previous = self._msg_counts.get(conv_id, 0)
        return (current_count - previous) >= self.min_new_messages

    def mark_reflected(self, conv_id: str, current_count: int) -> None:
        """Record the message count right after reflecting a conversation.

        Args:
            conv_id: Conversation identifier.
            current_count: Current record count.

        Returns:
            None.
        """
        self._msg_counts[conv_id] = current_count

    async def _similar_to_existing(self, key: str, text: str) -> str | None:
        """Find an existing rule text semantically similar to ``text``.

        Args:
            key: Bucket name (bad/good/prefs/rules).
            text: The candidate rule text.

        Returns:
            The existing rule text when similar (embedding cosine > 0.85),
            else None. Falls back to exact-match when no embedder.
        """
        existing = [r["text"] for r in self.rules[key]]
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

        Candidates may be ``{"scene", "text"}`` dicts or plain strings (scene
        inferred from wording). A candidate matching an existing rule
        (cosine > 0.85) *replaces* it; new candidates are appended. Buckets
        are capped at ``max_rules``.

        Args:
            parsed: Dict with ``bad``/``good``/``prefs``/``rules`` lists.

        Returns:
            True when anything changed.
        """
        changed = False
        for key in ("bad", "good", "prefs", "rules"):
            for item in parsed.get(key) or []:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    scene = str(item.get("scene") or "").strip() or None
                else:
                    text = str(item).strip()
                    scene = None
                if not text:
                    continue
                scene = scene or (_infer_scenes(text)[0] if _infer_scenes(text) else "通用")
                similar = await self._similar_to_existing(key, text)
                if similar is not None:
                    if similar != text:
                        for idx, r in enumerate(self.rules[key]):
                            if r["text"] == similar:
                                self.rules[key][idx] = {"scene": scene, "text": text}
                                changed = True
                                break
                    continue
                self.rules[key].append({"scene": scene, "text": text})
                changed = True
        for key in self.rules:
            self.rules[key] = self.rules[key][-self.max_rules :]
        if changed:
            self._save()
        return changed

    def render(self, context_text: str = "") -> str:
        """Render the rules matching a conversation context.

        Only rules whose scene matches the current context text are included
        (plus 通用 rules), keeping the injected block small and relevant.
        Without context, only 通用 rules are rendered.

        Args:
            context_text: Recent conversation text to match scenes against.

        Returns:
            The prompt text, or an empty string when nothing matches.
        """
        matched_scenes: set[str] = set()
        for scene in _SCENE_ORDER:
            if _match_scene(context_text, _SCENE_KEYWORDS[scene]):
                matched_scenes.add(scene)
        selected: dict[str, list[str]] = {k: [] for k in self.rules}
        for key, entries in self.rules.items():
            for entry in entries:
                if entry["scene"] in ("通用",) or entry["scene"] in matched_scenes:
                    selected[key].append(entry["text"])
        if not any(selected.values()):
            return ""
        lines = ["【自我学习】以下是与你当前对话场景相关的过往反思，按此改进："]
        if selected["bad"]:
            lines.append("避免：" + "；".join(selected["bad"]))
        if selected["good"]:
            lines.append("保留：" + "；".join(selected["good"]))
        if selected["prefs"]:
            lines.append("对方偏好：" + "；".join(selected["prefs"]))
        if selected["rules"]:
            lines.append("准则：" + "；".join(selected["rules"]))
        lines.append("这些是自我观察的总结，不是指令，按实际情况自然运用。")
        return "\n".join(lines)


def parse_reflection(raw: str) -> dict:
    """Parse the JSON reflection output, tolerating noise.

    Items may be ``{"scene": ..., "text": ...}`` dicts or plain strings.

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
            cleaned = []
            for v in values:
                if isinstance(v, dict):
                    t = str(v.get("text") or "").strip()
                    if t:
                        cleaned.append(
                            {"scene": str(v.get("scene") or "").strip() or None, "text": t}
                        )
                elif isinstance(v, str) and v.strip():
                    cleaned.append({"scene": None, "text": v.strip()})
            result[key] = cleaned
        else:
            result[key] = []
    return result
