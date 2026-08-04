"""Parse AI proactive-action markers in generated segments.

The AI can reply to or @ specific people by emitting markers inside a segment,
e.g. ``[[reply:小明]]这是对你的回复 [[at:小红]]一起来聊。`` Markers are
resolved against the recent conversation by display name.

``[[poke:userId]]`` is a poke marker: it must stand alone on its own line or
segment (only one per segment) and is rendered as a real poke action, never as
text.

Keeping this module free of AstrBot imports makes the parsing logic unit
testable in isolation.
"""

from __future__ import annotations

import json
import re

_MARKER_RE = re.compile(
    r"\[\[(at|reply|poke):([^\]]+)\]\]|\[@([^:\]]+):[^\]]*\]"
)

Token = tuple[str, str]


def parse_reply_decision(text: str) -> dict[str, str]:
    """Parse the small JSON response returned by the reply decision model.

    Args:
        text: Model output containing a JSON object.

    Returns:
        A dict containing optional ``reply`` and ``at`` display names.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("reply", "at"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    return result


def parse_actions(text: str) -> list[Token]:
    """Parse markers out of text, preserving order.

    ``[[at:昵称]]`` / ``[[reply:昵称]]`` are the canonical markers; the
    ``[@昵称: ...]`` legacy shape is also treated as a reply so models that
    picked it up from context still trigger a real quote.

    ``[[poke:userId]]`` marks a poke: the value is a raw platform user id (or
    ``yourself`` for the bot itself), never a display name.

    A marker prefixed with ``\\`` is escaped and stays literal text (the
    backslash is dropped), so the AI can talk about the syntax itself, e.g.
    ``\\[[at:小明]]`` renders as the plain text ``[[at:小明]]``.

    Args:
        text: The generated segment text.

    Returns:
        A list of ``("text", content)`` / ``("at", name)`` /
        ``("reply", name)`` / ``("poke", userId)`` tokens in order.
    """
    tokens: list[Token] = []
    pos = 0
    for m in _MARKER_RE.finditer(text):
        if m.start() > pos:
            seg = text[pos : m.start()]
            if seg.endswith("\\"):
                seg = seg[:-1]
            if seg:
                tokens.append(("text", seg))
        if m.start() > 0 and text[m.start() - 1] == "\\":
            tokens.append(("text", text[m.start() : m.end()]))
            pos = m.end()
            continue
        if m.group(1):
            tokens.append((m.group(1), m.group(2).strip()))
        else:
            tokens.append(("reply", m.group(3).strip()))
        pos = m.end()
    if pos < len(text):
        tokens.append(("text", text[pos:]))
    return tokens
