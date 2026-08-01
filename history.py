"""Access AstrBot's persisted chat history.

ChatCore's own in-memory history only covers the current process lifetime and
its embedding memory only stores hand-picked facts. This module reads the
authoritative chat records AstrBot persists per session (through
``conversation_manager``), so the bot can recall what was said in other
conversations or before the plugin started.
"""

import json
import logging

logger = logging.getLogger("astrbot")


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
        f"{'用户' if m['role'] == 'user' else '你'}: {m['content']}" for m in recent
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
