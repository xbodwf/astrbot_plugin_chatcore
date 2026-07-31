"""Parse AI proactive-action markers in generated segments.

The AI can reply to or @ specific people by emitting markers inside a segment,
e.g. ``[[reply:小明]]这是对你的回复 [[at:小红]]一起来聊。`` Markers are
resolved against the recent conversation by display name.

Keeping this module free of AstrBot imports makes the parsing logic unit
testable in isolation.
"""

from __future__ import annotations

import re

_MARKER_RE = re.compile(r"\[\[(at|reply):([^\]]+)\]\]")

Token = tuple[str, str]


def parse_actions(text: str) -> list[Token]:
    """Parse markers out of text, preserving order.

    Args:
        text: The generated segment text.

    Returns:
        A list of ``("text", content)`` / ``("at", name)`` /
        ``("reply", name)`` tokens in order.
    """
    tokens: list[Token] = []
    pos = 0
    for m in _MARKER_RE.finditer(text):
        if m.start() > pos:
            tokens.append(("text", text[pos : m.start()]))
        tokens.append((m.group(1), m.group(2).strip()))
        pos = m.end()
    if pos < len(text):
        tokens.append(("text", text[pos:]))
    return tokens
