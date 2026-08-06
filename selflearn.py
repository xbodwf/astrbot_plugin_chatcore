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

import json
import re
import time
from pathlib import Path

_REFLECT_PROMPT = (
    "以下是最近的聊天记录片段（你自己是 {name}）。\n"
    "{sample}\n\n"
    "请以冷静的旁观者视角分析这段对话，找出：\n"
    "1. 你自己的哪些回复显得机械、降智、不像真人（如汇报式语气、重复套话、过度礼貌）；\n"
    "2. 哪些表达方式与对方合拍、效果好，值得保留；\n"
    "3. 对方的明显偏好（话题、语气、称呼习惯）。\n"
    "只输出 JSON：{{\"bad\": [\"具体降智行为\"], \"good\": [\"有效表达\"], "
    "\"prefs\": [\"对方偏好\"], \"rules\": [\"以后要遵守的短规则\"]}}。"
    "每条最多 20 字，宁缺毋滥，没有就不写。"
)


class SelfLearnStore:
    """Persistent self-reflection rules.

    Args:
        path: JSON file to persist the learned rules.
        max_rules: Maximum number of rules kept per bucket.
    """

    def __init__(self, path: str | Path, max_rules: int = 20) -> None:
        self.path = Path(path)
        self.max_rules = max_rules
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

    def merge(self, parsed: dict) -> bool:
        """Merge a reflection result into the store, deduplicating.

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
                if text not in self.rules[key]:
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
