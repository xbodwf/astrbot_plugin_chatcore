"""Structured person profiles with LLM memory writeback.

Unlike the vector-text memories in ``memory.py``, a profile is a compact,
structured understanding of a single person (stable facts, preferences and
interaction traits). After each reply, an LLM extracts new stable facts from
the exchange and merges them back into the profile, so ChatCore gradually
"gets to know" the people it talks to. Persisted as JSON.
"""

import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

_EXTRACT_PROMPT = (
    "以下是关于说话人 {name} 的发言记录（可能包含多条）。请从中提取关于 {name} "
    "的稳定、有价值的人物事实，例如身份、职业、称呼偏好、喜好、习惯、性格特点、"
    "与其他人/群的关系等。忽略一次性的话题内容（如某件事的讨论、临时求助）。"
    "没有有价值信息就输出空数组。\n"
    '只输出 JSON 对象数组，例如 [{{"fact":"喜欢喝奶茶","evidence":"我每天都喝奶茶","confidence":0.9,"action":"add"}}].\n'
    "已有画像（只能基于证据修订，不要盲目保留）:\n{existing}\n"
    "本次发言记录:\n{text}"
)

_MAX_FACTS = 30


def _parse_fact_list(raw: str) -> list[str]:
    """Parse a JSON string array out of a model reply.

    Args:
        raw: Raw model output, possibly wrapped in markdown fences.

    Returns:
        The parsed fact strings.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = []
    if not isinstance(parsed, list):
        return []
    return [
        str(item).strip()
        for item in parsed
        if isinstance(item, str) and str(item).strip()
    ]


def _parse_fact_items(raw: str) -> list[dict]:
    """Parse evidence-backed profile updates from model output."""
    text = raw.strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    result = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        action = str(item.get("action", "add")).strip().lower()
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        if (
            fact
            and evidence
            and confidence >= 0.75
            and action in {"add", "replace", "remove"}
        ):
            result.append(
                {
                    "fact": fact,
                    "evidence": evidence,
                    "confidence": confidence,
                    "action": action,
                }
            )
    return result


class ProfileStore:
    """Persistent per-person structured profiles.

    Args:
        path: Path of the JSON persistence file.
        max_chars: Max characters of a rendered profile block.
    """

    def __init__(self, path: str | Path, max_chars: int = 600) -> None:
        self.path = Path(path)
        self.max_chars = max_chars
        self._profiles: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._profiles = {
                key: value for key, value in data.items() if isinstance(value, dict)
            }
        except (OSError, json.JSONDecodeError):
            self._profiles = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._profiles, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def get(self, person_id: str) -> dict | None:
        """Get a person's profile dict, or None.

        Args:
            person_id: Stable platform id of the person.

        Returns:
            The profile dict or None.
        """
        return self._profiles.get(person_id)

    def all(self) -> list[dict]:
        """All profiles.

        Returns:
            List of profile dicts (with their id attached).
        """
        return [
            {"person_id": pid, **profile} for pid, profile in self._profiles.items()
        ]

    def delete(self, person_id: str) -> bool:
        """Delete a person's profile.

        Args:
            person_id: Stable platform id of the person.

        Returns:
            True if a profile was removed.
        """
        if person_id in self._profiles:
            del self._profiles[person_id]
            self._save()
            return True
        return False

    def merge(
        self,
        person_id: str,
        nickname: str,
        facts: list[str | dict],
    ) -> None:
        """Merge new facts into a person's profile.

        Args:
            person_id: Stable platform id of the person.
            nickname: Latest observed nickname.
            facts: Newly extracted stable facts.
        """
        now = time.time()
        profile = self._profiles.get(person_id)
        if profile is None:
            profile = {
                "person_id": person_id,
                "nickname": nickname or "",
                "facts": [],
                "preferences": [],
                "interaction": [],
                "first_seen": now,
                "updated_at": now,
            }
            self._profiles[person_id] = profile
        profile["nickname"] = nickname or profile.get("nickname", "")
        existing = profile.get("facts", []) or []
        added = []
        for item in facts:
            fact = item.get("fact", "") if isinstance(item, dict) else str(item)
            if not fact:
                continue
            if isinstance(item, dict) and item.get("action") == "remove":
                existing = [
                    old
                    for old in existing
                    if (old.get("fact", "") if isinstance(old, dict) else old) != fact
                ]
                continue
            if isinstance(item, dict) and item.get("action") == "replace":
                existing = [
                    old
                    for old in existing
                    if (old.get("fact", "") if isinstance(old, dict) else old) != fact
                ]
            # 近似去重: LLM 措辞变体（"小明是大学生" vs "小明在读大学"）不重复累积
            if any(
                difflib.SequenceMatcher(
                    None,
                    fact,
                    old.get("fact", "") if isinstance(old, dict) else old,
                ).ratio()
                > 0.65
                for old in existing + added
            ):
                continue
            added.append(item if isinstance(item, dict) else fact)
        if added:
            profile["facts"] = (existing + added)[-_MAX_FACTS:]
            profile["updated_at"] = now
            self._save()

    def render(self, person_id: str) -> str | None:
        """Render a person's profile as an injection block.

        Args:
            person_id: Stable platform id of the person.

        Returns:
            A compact profile text, or None when absent.
        """
        profile = self._profiles.get(person_id)
        if not profile:
            return None
        facts = profile.get("facts", []) or []
        if not facts:
            return None
        nickname = profile.get("nickname", "") or person_id
        parts = [f"该用户昵称: {nickname}"]
        for item in facts:
            fact = item.get("fact", "") if isinstance(item, dict) else str(item)
            evidence = item.get("evidence", "") if isinstance(item, dict) else ""
            parts.append(f"- {fact}" + (f"（依据: {evidence}）" if evidence else ""))
        text = "\n".join(parts)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "..."
        return text

    async def extract_facts(
        self,
        client: Any,
        name: str,
        text: str,
        existing_facts: list[str] | None = None,
    ) -> list[dict] | list[str]:
        """Extract stable facts about a speaker from one message.

        Args:
            client: An LLM wrapper exposing ``async chat(messages)``.
            name: The speaker's display name.
            text: The speaker's message text.

        Returns:
            The extracted facts, or an empty list on any failure.
        """
        message = text.strip()
        if not message:
            return []
        prompt = _EXTRACT_PROMPT.format(
            name=name,
            existing="\n".join(
                f"- {item.get('fact', '') if isinstance(item, dict) else item}"
                for item in (existing_facts or [])[-30:]
            )
            or "（无）",
            text=message[:1500],
        )
        try:
            raw = await client.chat(
                [
                    {"role": "system", "content": "你是信息提取助手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                log_name="profile",
            )
        except Exception:
            return []
        return _parse_fact_items(raw)

    def count(self) -> int:
        """Number of profiles.

        Returns:
            Profile count.
        """
        return len(self._profiles)
