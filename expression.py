"""Group expression-style learning.

To stop sounding like a generic assistant, ChatCore periodically samples a
group's recent messages and lets an LLM distill three things into a per-group
style record: a one-line style summary, common sentence patterns, and in-group
jargon with inferred meanings. The learned style is injected into that group's
system prompt so the bot gradually talks more like the people around it.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_ANALYZE_PROMPT = (
    "下面是一个聊天群最近的发言记录。请分析这个群的表达风格，"
    "只保留真正有群特色、值得长期记住的内容（过滤一次性话题、寒暄、"
    "无意义的语气词）；拿不准就不输出。输出 JSON：\n"
    "{\n"
    '  "summary": "一句话概括该群表达风格（如：短句流、爱用表情、阴阳怪气等）",\n'
    '  "patterns": ["常见句式或语气词，带一个例句，如：笑死，这操作真的6"],\n'
    '  "jargon": [{"term": "群内黑话/梗词", "guessedMeaning": "有证据支持的推断含义", "example": "原始使用例句", "source": "发言者/消息片段"}]\n'
    "}\n"
    "只输出 JSON，不要其他内容。\n"
    "发言记录:\n{sample}"
)

_MAX_PATTERNS = 12
_MAX_JARGON = 12
_RENDER_MAX_PATTERNS = 3
_RENDER_MAX_JARGON = 3


def _parse_analysis(raw: str) -> dict:
    """Parse the model's style analysis into a dict.

    Args:
        raw: Raw model output, possibly wrapped in markdown fences.

    Returns:
        Dict with ``summary``, ``patterns`` and ``jargon`` keys.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": "", "patterns": [], "jargon": []}
    if not isinstance(parsed, dict):
        return {"summary": "", "patterns": [], "jargon": []}
    raw_patterns = parsed.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raw_patterns = []
    patterns = [
        str(p).strip() for p in raw_patterns if isinstance(p, str) and str(p).strip()
    ]
    raw_jargon = parsed.get("jargon", [])
    if not isinstance(raw_jargon, list):
        raw_jargon = []
    jargon = []
    for item in raw_jargon:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        meaning = str(item.get("guessedMeaning", item.get("meaning", ""))).strip()
        example = str(item.get("example", "")).strip()
        source = str(item.get("source", "")).strip()
        if term and meaning:
            if not source:
                jargon.append({"term": term, "meaning": meaning, "example": example})
                continue
            jargon.append(
                {
                    "term": term,
                    "guessedMeaning": meaning,
                    "meaning": meaning,
                    "example": example,
                    "source": source,
                }
            )
    return {
        "summary": str(parsed.get("summary", "")).strip(),
        "patterns": patterns,
        "jargon": jargon,
    }


class ExpressionStore:
    """Per-group expression styles persisted as JSON.

    Args:
        path: Path of the JSON persistence file.
        max_chars: Max characters of a rendered style block.
    """

    def __init__(self, path: str | Path, max_chars: int = 800) -> None:
        self.path = Path(path)
        self.max_chars = max_chars
        self._styles: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._styles = {
                key: value for key, value in data.items() if isinstance(value, dict)
            }
        except (OSError, json.JSONDecodeError):
            self._styles = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._styles, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def get_style(self, group_id: str, shared_groups: list[str] | None = None) -> dict:
        """Get a group's style, merged with any shared groups' styles.

        Args:
            group_id: Conversation/group identifier.
            shared_groups: Optional list of other group ids to merge in.

        Returns:
            A merged style dict.
        """
        merged = {"summary": "", "patterns": [], "jargon": []}
        ids = [group_id] + [g for g in (shared_groups or []) if g and g != group_id]
        for gid in ids:
            style = self._styles.get(gid)
            if not style:
                continue
            if not merged["summary"]:
                merged["summary"] = style.get("summary", "")
            for p in style.get("patterns", []):
                if p not in merged["patterns"]:
                    merged["patterns"].append(p)
            for j in style.get("jargon", []):
                if not j.get("enabled", True):
                    continue
                if not any(x.get("term") == j.get("term") for x in merged["jargon"]):
                    merged["jargon"].append(j)
        return merged

    async def learn(
        self,
        client: Any,
        group_id: str,
        sample_text: str,
    ) -> bool:
        """Analyze a group sample and merge the learned style.

        Args:
            client: An LLM wrapper exposing ``async chat(messages)``.
            group_id: Conversation/group identifier.
            sample_text: Recent messages of the group as text.

        Returns:
            True when a style was learned/updated.
        """
        sample = sample_text.strip()
        if not sample:
            return False
        prompt = _ANALYZE_PROMPT.replace("{sample}", sample[:3000])
        try:
            raw = await client.chat(
                [
                    {
                        "role": "system",
                        "content": "你是表达风格分析器，只输出 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
        except Exception:
            return False
        parsed = _parse_analysis(raw)
        parsed["jargon"] = [
            item
            for item in parsed["jargon"]
            if item.get("source") and item.get("example")
        ]
        if not parsed["summary"] and not parsed["patterns"] and not parsed["jargon"]:
            return False
        style = self._styles.setdefault(
            group_id,
            {"summary": "", "patterns": [], "jargon": [], "updated_at": 0.0},
        )
        if parsed["summary"]:
            style["summary"] = parsed["summary"]
        for p in parsed["patterns"]:
            if p not in style["patterns"]:
                style["patterns"].append(p)
        existing = {j.get("term") for j in style["jargon"]}
        for j in parsed["jargon"]:
            if j["term"] not in existing:
                j.setdefault("enabled", True)
                j.setdefault("origin", "learned")
                style["jargon"].append(j)
                existing.add(j["term"])
        style["patterns"] = style["patterns"][-_MAX_PATTERNS:]
        style["jargon"] = style["jargon"][-_MAX_JARGON:]
        style["updated_at"] = time.time()
        self._save()
        return True

    def render(
        self,
        group_id: str,
        shared_groups: list[str] | None = None,
        query: str = "",
        max_patterns: int = _RENDER_MAX_PATTERNS,
        max_jargon: int = _RENDER_MAX_JARGON,
    ) -> str | None:
        """Render a group's learned style as a prompt block.

        Args:
            group_id: Conversation/group identifier.
            shared_groups: Optional list of other group ids to merge in.

        Returns:
            The rendered style text, or None when nothing was learned.
        """
        style = self.get_style(group_id, shared_groups)
        terms = set((query or "").lower().split())
        if terms:
            style["patterns"] = [
                item
                for item in style["patterns"]
                if any(term in item.lower() for term in terms)
            ] or style["patterns"][:1]
            style["jargon"] = [
                item
                for item in style["jargon"]
                if any(
                    term in (item.get("term", "") + item.get("example", "")).lower()
                    for term in terms
                )
            ] or style["jargon"][:1]
        lines: list[str] = []
        if style["summary"]:
            lines.append(f"该群整体表达风格: {style['summary']}")
        # 按需注入：只给少量代表性内容，避免风格堆砌。
        if style["patterns"]:
            lines.append("可参考的常见表达:")
            lines.extend(f"- {p}" for p in style["patterns"][:max_patterns])
        if style["jargon"]:
            lines.append("群内黑话表（带推断含义）:")
            lines.extend(
                f"- {j['term']}（推断: {j.get('guessedMeaning', j.get('meaning', ''))}）"
                f"，例: {j['example']}，来源: {j['source']}"
                for j in style["jargon"][:max_jargon]
            )
        if not lines:
            return None
        text = "\n".join(lines)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "..."
        return text

    def all(self) -> list[dict]:
        """All group styles.

        Returns:
            List of style dicts (with their group id attached).
        """
        return [{"group_id": gid, **style} for gid, style in self._styles.items()]

    def add_entry(
        self,
        group_id: str,
        *,
        pattern: str = "",
        term: str = "",
        guessed_meaning: str = "",
        example: str = "",
        source: str = "manual",
    ) -> bool:
        """Add a manually curated expression entry."""
        pattern = pattern.strip()
        term = term.strip()
        if not group_id or (not pattern and not (term and guessed_meaning and example)):
            return False
        style = self._styles.setdefault(
            group_id,
            {"summary": "", "patterns": [], "jargon": [], "updated_at": 0.0},
        )
        if pattern and pattern not in style["patterns"]:
            style["patterns"].append(pattern)
        if term and not any(item.get("term") == term for item in style["jargon"]):
            meaning = guessed_meaning.strip()
            style["jargon"].append(
                {
                    "term": term,
                    "guessedMeaning": meaning,
                    "meaning": meaning,
                    "example": example.strip(),
                    "source": source.strip() or "manual",
                    "enabled": True,
                    "origin": "manual",
                }
            )
        style["updated_at"] = time.time()
        self._save()
        return True

    def update_jargon(self, group_id: str, term: str, **updates: Any) -> bool:
        """Update one jargon entry, including its enabled state."""
        style = self._styles.get(group_id)
        if not style:
            return False
        for item in style.get("jargon", []):
            if item.get("term") != term:
                continue
            for key in ("guessedMeaning", "example", "source", "enabled"):
                if key in updates:
                    item[key] = updates[key]
            item["meaning"] = item.get("guessedMeaning", item.get("meaning", ""))
            self._save()
            return True
        return False

    def delete(self, group_id: str) -> bool:
        """Delete a group's learned style.

        Args:
            group_id: Conversation/group identifier.

        Returns:
            True if a style was removed.
        """
        if group_id in self._styles:
            del self._styles[group_id]
            self._save()
            return True
        return False

    def count(self) -> int:
        """Number of learned group styles.

        Returns:
            Style count.
        """
        return len(self._styles)
