"""Smart context building.

AstrBot's context is a flat dump of everything; ChatCore instead builds a
layered context: recent messages kept verbatim, older history compressed per
message, plus recalled global memories and image descriptions.
"""

import time
from dataclasses import dataclass, field


@dataclass
class MessageRecord:
    """A single chat message kept in a conversation.

    Args:
        role: ``user`` for humans, ``assistant`` for the bot itself.
        sender_name: Display name of the sender.
        text: The plain text content.
        sender_id: Platform id of the sender (for proactive @).
        message_id: Platform message id (for proactive reply).
        ts: Unix timestamp.
    """

    role: str
    sender_name: str
    text: str
    sender_id: str = ""
    message_id: str = ""
    ts: float = field(default_factory=time.time)


class ContextManager:
    """Per-conversation history and OpenAI message building.

    Args:
        recent_count: How many recent messages stay verbatim.
        history_count: How many older messages are kept (compressed).
        old_msg_chars: Per-message char cap for compressed history.
    """

    def __init__(
        self,
        recent_count: int = 10,
        history_count: int = 30,
        old_msg_chars: int = 40,
    ) -> None:
        self.recent_count = max(1, recent_count)
        self.history_count = max(0, history_count)
        self.old_msg_chars = max(1, old_msg_chars)
        self._histories: dict[str, list[MessageRecord]] = {}

    def _history(self, conv_id: str) -> list[MessageRecord]:
        history = self._histories.get(conv_id)
        if history is None:
            history = []
            self._histories[conv_id] = history
        return history

    def record(
        self,
        conv_id: str,
        role: str,
        sender_name: str,
        text: str,
        sender_id: str = "",
        message_id: str = "",
    ) -> None:
        """Append a message to a conversation, trimming the oldest.

        Args:
            conv_id: Conversation identifier (unified_msg_origin).
            role: Message role (``user`` / ``assistant``).
            sender_name: Sender display name.
            text: Message text.
            sender_id: Sender platform id.
            message_id: Platform message id.
        """
        history = self._history(conv_id)
        history.append(
            MessageRecord(
                role=role,
                sender_name=sender_name,
                text=text,
                sender_id=sender_id,
                message_id=message_id,
            )
        )
        cap = self.recent_count + self.history_count
        del history[: max(0, len(history) - cap)]

    def active_conversations(self) -> list[str]:
        """List conversation ids that have recorded history.

        Returns:
            Conversation id list.
        """
        return list(self._histories.keys())

    def resolve_target(self, conv_id: str, name: str) -> dict | None:
        """Resolve a display name to the most recent matching user message.

        Used to build proactive ``[[at:name]]`` / ``[[reply:name]]`` actions.

        Args:
            conv_id: Conversation identifier.
            name: Display name to look up.

        Returns:
            A dict with ``sender_id`` and ``message_id``, or None if not found.
        """
        name = name.strip()
        for record in reversed(self._history(conv_id)):
            if record.role != "user" or not record.sender_id:
                continue
            if (
                record.sender_name == name
                or name in record.sender_name
                or record.sender_name in name
            ):
                return {
                    "sender_id": record.sender_id,
                    "message_id": record.message_id,
                }
        return None

    def summary_text(self, conv_id: str, max_chars: int = 400) -> str:
        """Compact recent conversation text, for implicit intent analysis.

        Args:
            conv_id: Conversation identifier.
            max_chars: Maximum characters to return.

        Returns:
            The trailing conversation text.
        """
        records = self._history(conv_id)[-10:]
        text = "\n".join(self._format_record(r) for r in records)
        return text[-max_chars:]

    def clear(self, conv_id: str) -> None:
        """Drop a conversation's history.

        Args:
            conv_id: Conversation identifier.
        """
        self._histories.pop(conv_id, None)

    def remove_message(self, conv_id: str, message_id: str) -> None:
        """Remove a user message from a conversation's history.

        Used when a message is recalled so it stops polluting the context.

        Args:
            conv_id: Conversation identifier.
            message_id: Platform id of the message to remove.
        """
        history = self._histories.get(conv_id)
        if not history:
            return
        self._histories[conv_id] = [
            record
            for record in history
            if not (record.role == "user" and record.message_id == message_id)
        ]

    def _format_record(self, record: MessageRecord) -> str:
        if record.role == "assistant":
            return record.text
        return f"{record.sender_name}: {record.text}"

    def build_messages(
        self,
        conv_id: str,
        *,
        system_prompt: str,
        memory_texts: list[str] | None = None,
        image_descriptions: list[str] | None = None,
    ) -> list[dict]:
        """Build the OpenAI-style message list from a conversation's history.

        Cache-stable layout: ``system`` first, then the verbatim recent
        messages as a stable, growing prefix. Everything short-lived (image
        descriptions, recalled memories, compressed older history) goes into a
        single trailing ``user`` block at the end, so it never shifts the
        shared prefix and provider-side prompt caching stays effective between
        window slides.

        Args:
            conv_id: Conversation identifier.
            system_prompt: The AI persona / system prompt.
            memory_texts: Recalled global memories to inject.
            image_descriptions: Descriptions of images in the current message.

        Returns:
            OpenAI-style message dict list.
        """
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        all_records = self._history(conv_id)
        recent = all_records[-self.recent_count :]
        older = all_records[: -self.recent_count]

        messages.extend(
            {
                "role": record.role,
                "content": self._format_record(record),
            }
            for record in recent
        )

        background: list[str] = []
        if image_descriptions:
            background.append("当前消息中的图片内容:")
            background.extend(f"- {desc}" for desc in image_descriptions)
        if memory_texts:
            background.append(
                "以下是你回忆起的过往对话片段（可能来自其他对话或其他人，"
                "不代表你本人执行过任何操作，不要当成你做过的事）:"
            )
            background.extend(f"- {text}" for text in memory_texts)
        if older:
            background.append("更早的对话（已压缩）:")
            background.extend(
                f"- {self._format_record(r)[: self.old_msg_chars]}"
                for r in older[-self.history_count :]
            )

        if background:
            messages.append({"role": "user", "content": "\n".join(background)})

        return messages
