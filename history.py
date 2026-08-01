"""Access AstrBot's persisted chat history.

ChatCore's own in-memory history only covers the current process lifetime and
its embedding memory only stores hand-picked facts. This module reads the
authoritative chat records AstrBot persists per session (through
``conversation_manager``), so the bot can recall what was said in other
conversations or before the plugin started.
"""

import json
import logging
import re

logger = logging.getLogger("astrbot")

_MARKER_ESCAPE_RE = re.compile(
    r"\[\[|\[引用了|\[引用消息|\[图片|\[媒体|\[转发|\[@|\[At|\[表情"
)

_ONE_BOT_FORWARD_RE = re.compile(
    r"查看\s*\d+\s*条转发消息|群聊的聊天记录|合并转发|转发消息"
)
_ONE_BOT_QUOTE_RE = re.compile(r"\[引用消息:?\s+([^\]\n:]+)[:\n]\s*([^\]]+?)\]")


def escape_user_markers(text: str) -> str:
    """Escape system marker syntax inside user-authored text.

    Users could type ``[引用消息: ...]``, ``[[at:...]]`` / ``[[reply:...]]``
    etc. to spoof system markers and trick the AI. Replacing the opening half
    width ``[`` with its full width ``［`` makes those a distinctive format
    the AI cannot mistake for, or imitate as, system markers.

    Args:
        text: The user-authored text.

    Returns:
        The text with marker prefixes escaped.
    """
    return _MARKER_ESCAPE_RE.sub(lambda m: "［" + m.group(0)[1:], text)


def clean_placeholder_text(text: str) -> str:
    """Clean adapter placeholder text out of a message before the AI sees it.

    OneBot clients replace forwarded messages and failed media downloads with
    plain-text placeholders (``查看 1 条转发消息``, ``群聊的聊天记录``,
    ``此项媒体下载失败``) that pollute the context and confuse the AI.
    They are rewritten into stable, descriptive markers.

    OneBot-style quote fragments (``[引用消息 昵称\n内容]``) are normalized
    into the same ``[引用了昵称的消息: 内容]`` shape ChatCore renders.

    Args:
        text: The raw message text.

    Returns:
        The cleaned text.
    """
    text = _ONE_BOT_FORWARD_RE.sub("[转发消息]", text)
    text = text.replace("此项媒体下载失败", "[媒体(下载失败)]")
    text = _ONE_BOT_QUOTE_RE.sub(
        lambda m: f"[引用了{m.group(1).strip()}的消息: {m.group(2).strip()}]",
        text,
    )
    return text


def build_umo(platform: str, message_type: str, session_id: str) -> str:
    """Compose a Unified Message Origin string.

    Args:
        platform: Platform adapter id (e.g. ``aiocqhttp``).
        message_type: ``FriendMessage`` or ``GroupMessage``.
        session_id: Peer user/group id.

    Returns:
        The UMO string ``{platform}:{message_type}:{session_id}``.
    """
    return f"{platform}:{message_type}:{session_id}"


def build_friend_umo(unified_msg_origin: str, sender_id: str) -> str:
    """Build the private-chat UMO of a user from a group event's UMO.

    Args:
        unified_msg_origin: A group event's UMO (``platform:...:group_id``).
        sender_id: The user's id.

    Returns:
        The user's private-chat UMO.
    """
    platform = unified_msg_origin.split(":")[0] if unified_msg_origin else "default"
    return build_umo(platform, "FriendMessage", sender_id)


def extract_text_history(history: list) -> list[dict]:
    """Extract plain-text user/assistant messages from OpenAI-format history.

    Filters out system/tool roles and non-text content segments (images).

    Args:
        history: The raw message list from ``conversation.history``.

    Returns:
        A list of ``{"role", "content"}`` dicts with non-empty text.
    """
    result = []
    for msg in history:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                seg.get("text", "")
                for seg in content
                if isinstance(seg, dict) and seg.get("type") == "text"
            )
        else:
            continue
        text = text.strip()
        if text:
            result.append({"role": msg["role"], "content": text})
    return result


def render_history_block(
    messages: list[dict],
    *,
    max_messages: int,
    max_chars: int,
    header: str,
) -> str:
    """Render extracted history as an injectable background block.

    Args:
        messages: Plain-text ``{role, content}`` messages.
        max_messages: How many most-recent messages to keep.
        max_chars: Total character budget (keeps the tail).
        header: A lead-in line explaining what the block is.

    Returns:
        The rendered block, or an empty string when there is nothing.
    """
    if not messages:
        return ""
    recent = messages[-max_messages:]
    lines = [
        f"{'用户' if m['role'] == 'user' else 'bot'}: {escape_user_markers(clean_placeholder_text(m['content']))}"
        for m in recent
    ]
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[-max_chars:]
    return header + "\n" + block


class HistoryReader:
    """Reads AstrBot's persisted conversation history.

    Args:
        conversation_manager: The AstrBot conversation manager.
    """

    def __init__(self, conversation_manager) -> None:
        self.conversation_manager = conversation_manager

    async def read_session(
        self,
        umo: str,
        *,
        max_messages: int = 10,
        max_chars: int = 1200,
        header: str = "【历史记录】以下是该会话的最近聊天记录，仅作背景参考:",
    ) -> str:
        """Read the current conversation history of a session as a block.

        Args:
            umo: The session's Unified Message Origin.
            max_messages: How many most-recent messages to include.
            max_chars: Character budget for the block.
            header: Lead-in line for the block.

        Returns:
            The rendered history block, or an empty string when unavailable.
        """
        try:
            cid = await self.conversation_manager.get_curr_conversation_id(umo)
            if not cid:
                return ""
            conversation = await self.conversation_manager.get_conversation(umo, cid)
            if not conversation or not conversation.history:
                return ""
            history = json.loads(conversation.history)
            return render_history_block(
                extract_text_history(history),
                max_messages=max_messages,
                max_chars=max_chars,
                header=header,
            )
        except Exception as e:
            logger.warning(f"ChatCore: read chat history failed ({umo}): {e}")
            return ""
