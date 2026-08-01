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
        images: Image URLs attached to the message (empty when none).
        description: Generated image description, ``[图片描述: ...]`` marker.
        sender_id: Platform id of the sender (for proactive @).
        message_id: Platform message id (for proactive reply).
        ts: Unix timestamp.
    """

    role: str
    sender_name: str
    text: str
    images: list[str] = field(default_factory=list)
    description: str = ""
    sender_id: str = ""
    message_id: str = ""
    quote: str = ""
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
        self._summaries: dict[str, dict] = {}

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
        images: list[str] | None = None,
        quote: str = "",
    ) -> None:
        """Append a message to a conversation, trimming the oldest.

        Args:
            conv_id: Conversation identifier (unified_msg_origin).
            role: Message role (``user`` / ``assistant``).
            sender_name: Sender display name.
            text: Message text.
            sender_id: Sender platform id.
            message_id: Platform message id.
            images: Image URLs attached to the message.
            quote: Quoted-message content this message replies to.
        """
        history = self._history(conv_id)
        history.append(
            MessageRecord(
                role=role,
                sender_name=sender_name,
                text=text,
                images=list(images or []),
                sender_id=sender_id,
                message_id=message_id,
                quote=quote,
            )
        )
        cap = self.recent_count + self.history_count
        del history[: max(0, len(history) - cap)]

    def set_image_description(
        self, conv_id: str, message_id: str, description: str
    ) -> None:
        """Attach a generated image description to a user message.

        The description is rendered as a ``[图片描述: ...]`` marker so the
        model references real content instead of a bare ``[图片]`` placeholder.

        Args:
            conv_id: Conversation identifier.
            message_id: Platform id of the message to update.
            description: Generated image description text.
        """
        if not message_id or not description:
            return
        for record in reversed(self._history(conv_id)):
            if record.role == "user" and record.message_id == message_id:
                record.description = description
                return

    def set_summary(self, conv_id: str, text: str, covered_count: int) -> None:
        """Store an LLM-generated summary of the older history.

        ``covered_count`` is the number of older records the summary already
        folded in, so stale summaries can be refreshed only when new messages
        have slid out of the recent window.

        Args:
            conv_id: Conversation identifier.
            text: The summarized text.
            covered_count: How many older records the summary covers.
        """
        self._summaries[conv_id] = {"text": text, "count": covered_count}

    def get_summary(self, conv_id: str) -> str:
        """Return the cached LLM summary of a conversation, if any.

        Args:
            conv_id: Conversation identifier.

        Returns:
            The summary text, or an empty string.
        """
        entry = self._summaries.get(conv_id)
        return entry["text"] if entry else ""

    def summary_stale(self, conv_id: str, threshold: int = 1) -> bool:
        """Whether older history needs a fresh LLM summary.

        Args:
            conv_id: Conversation identifier.
            threshold: How many unsized older records trigger a refresh.

        Returns:
            True when messages have slid out that aren't covered yet.
        """
        older_count = self.older_count(conv_id)
        if older_count <= 0:
            return False
        entry = self._summaries.get(conv_id)
        covered = entry["count"] if entry else 0
        return (older_count - covered) >= threshold

    def older_count(self, conv_id: str) -> int:
        """Number of records that have slid out of the recent window.

        Args:
            conv_id: Conversation identifier.

        Returns:
            The count of older (compressible) records.
        """
        return len(self._history(conv_id)) - self.recent_count

    def summary_payload(self, conv_id: str, max_records: int = 50) -> str:
        """Format older records as the input for an LLM summarizer.

        Args:
            conv_id: Conversation identifier.
            max_records: How many older records to include.

        Returns:
            Formatted older-history text.
        """
        older = self._history(conv_id)[: -self.recent_count][-max_records:]
        return "\n".join(self._format_record(r) for r in older)

    def active_conversations(self) -> list[str]:
        """List conversation ids that have recorded history.

        Returns:
            Conversation id list.
        """
        return list(self._histories.keys())

    def conversation_stats(self) -> list[dict]:
        """Summarize per-conversation state for monitoring (WebUI).

        Returns:
            One summary dict per active conversation.
        """
        rows = []
        for conv_id in self._histories:
            history = self._history(conv_id)
            summary = self._summaries.get(conv_id)
            rows.append(
                {
                    "conv_id": conv_id,
                    "messages": len(history),
                    "older_count": self.older_count(conv_id),
                    "has_summary": bool(summary and summary[0]),
                    "summary_len": len(summary[0]) if summary and summary[0] else 0,
                }
            )
        return rows

    def find_message(self, conv_id: str, message_id: str) -> MessageRecord | None:
        """Find the most recent user message with the given platform id.

        Args:
            conv_id: Conversation identifier.
            message_id: Platform message id to look up.

        Returns:
            The matching record, or None.
        """
        if not message_id:
            return None
        for record in reversed(self._history(conv_id)):
            if record.role == "user" and record.message_id == message_id:
                return record
        return None

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
        prefix = f"{record.sender_name}: "
        body = ""
        if record.quote:
            body += f"[引用了消息: {record.quote}] "
        if record.description:
            desc = f"[图片描述: {record.description}]"
            body += f"{record.text} {desc}" if record.text else desc
        elif record.text:
            body += record.text
        if body:
            return prefix + body
        return f"{prefix}[图片]"

    def _compress_record(self, record: MessageRecord) -> str | None:
        """Format a record for the compressed older-history block.

        Image placeholders (``[图片]`` with no description) carry no real
        content and are dropped entirely so they don't pollute the summary;
        a stored description participates as real text. Assistant lines get a
        ``bot:`` prefix so the speaker is unambiguous.

        Args:
            record: The record to compress.

        Returns:
            The compressed line, or None to skip this record.
        """
        if record.role == "assistant":
            text = record.text or ""
            return f"bot: {text}" if text else None
        body = ""
        if record.quote:
            body += f"[引用了消息: {record.quote}] "
        if record.description:
            desc = f"[图片描述: {record.description}]"
            body += f"{record.text} {desc}" if record.text else desc
        elif record.text:
            body += record.text
        if record.images and not body:
            return None
        return f"{record.sender_name}: {body}" if body else None

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text at a boundary, appending an ellipsis.

        Prefers to cut after sentence punctuation (。！？，,. ) so Chinese
        text isn't split mid-sentence; falls back to a hard cut.

        Args:
            text: Text to truncate.
            limit: Maximum characters.

        Returns:
            The truncated text.
        """
        if len(text) <= limit:
            return text
        head = text[:limit]
        for sep in ("。", "！", "？", "，", ".", "！", ",", " "):
            idx = head.rfind(sep)
            if idx >= 0:
                return head[: idx + 1] + "…"
        return head + "…"

    def build_messages(
        self,
        conv_id: str,
        *,
        system_prompt: str,
        memory_texts: list[str] | None = None,
        history_texts: list[str] | None = None,
        profile_texts: list[str] | None = None,
    ) -> list[dict]:
        """Build the OpenAI-style message list from a conversation's history.

        Cache-stable layout: ``system`` first, then the verbatim recent
        messages as a stable, growing prefix. Everything short-lived (recalled
        memories, injected chat history, compressed older history) goes into a
        single trailing ``user`` block at the end, so it never shifts the
        shared prefix and provider-side prompt caching stays effective between
        window slides. Image descriptions are stored on the message records
        themselves and rendered inline (``[图片描述: ...]``) rather than as a
        separate block.

        Args:
            conv_id: Conversation identifier.
            system_prompt: The AI persona / system prompt.
            memory_texts: Recalled global memories to inject.
            history_texts: Persisted chat-history blocks to inject.
            profile_texts: Structured person-profile blocks to inject.

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
        for block in history_texts or []:
            background.append(block)
        if memory_texts:
            background.append(
                "以下是你回忆起的过往对话片段（可能来自其他对话或其他人，"
                "不代表你本人执行过任何操作，不要当成你做过的事）:"
            )
            background.extend(f"- {text}" for text in memory_texts)
        if profile_texts:
            background.append("以下是你对该用户的了解（人物画像，可能随时间更新）:")
            background.extend(profile_texts)
        if older:
            summary = self.get_summary(conv_id)
            if summary:
                background.append("更早的对话（已压缩摘要）:")
                background.append(summary)
            else:
                compressed = [
                    self._truncate(line, self.old_msg_chars)
                    for record in older[-self.history_count :]
                    if (line := self._compress_record(record))
                ]
                if compressed:
                    background.append("更早的对话（已压缩）:")
                    background.extend(f"- {line}" for line in compressed)

        if background:
            messages.append({"role": "user", "content": "\n".join(background)})

        return messages
