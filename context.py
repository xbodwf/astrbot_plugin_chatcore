"""Smart context building.

AstrBot's context is a flat dump of everything; ChatCore instead builds a
layered context: recent messages kept verbatim, older history compressed per
message, plus recalled global memories and image descriptions.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .history import escape_user_markers


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

    History is optionally persisted to a JSON file so a plugin reload keeps
    the verbatim per-conversation context (``persist_path``). Without a path
    everything stays in memory (tests / backwards compatibility).

    Args:
        recent_count: How many recent messages stay verbatim.
        history_count: How many older messages are kept (compressed).
        old_msg_chars: Per-message char cap for compressed history.
        persist_path: Optional JSON file to load/save conversation history.
    """

    def __init__(
        self,
        recent_count: int = 10,
        history_count: int = 30,
        old_msg_chars: int = 40,
        persist_path: str | Path | None = None,
    ) -> None:
        self.recent_count = max(1, recent_count)
        self.history_count = max(0, history_count)
        self.old_msg_chars = max(1, old_msg_chars)
        self._persist_path = Path(persist_path) if persist_path else None
        self._histories: dict[str, list[MessageRecord]] = {}
        self._summaries: dict[str, dict] = {}
        if self._persist_path:
            self._load()

    @staticmethod
    def _record_to_dict(record: MessageRecord) -> dict:
        """Serialize one message record.

        Args:
            record: The record to serialize.

        Returns:
            A JSON-able dict.
        """
        return {
            "role": record.role,
            "sender_name": record.sender_name,
            "text": record.text,
            "images": record.images,
            "description": record.description,
            "sender_id": record.sender_id,
            "message_id": record.message_id,
            "quote": record.quote,
            "ts": record.ts,
        }

    @staticmethod
    def _record_from_dict(data: dict) -> MessageRecord | None:
        """Deserialize one message record, tolerating corrupt entries.

        Args:
            data: The serialized dict.

        Returns:
            The record, or None when the dict is invalid.
        """
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            return None
        return MessageRecord(
            role=str(data.get("role") or "user"),
            sender_name=str(data.get("sender_name") or ""),
            text=data["text"],
            images=[str(i) for i in data.get("images") or [] if isinstance(i, str)],
            description=str(data.get("description") or ""),
            sender_id=str(data.get("sender_id") or ""),
            message_id=str(data.get("message_id") or ""),
            quote=str(data.get("quote") or ""),
            ts=float(data.get("ts") or time.time()),
        )

    def _load(self) -> None:
        """Load persisted conversation history from disk."""
        path = self._persist_path
        if path is None:
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        for conv_id, records in raw.items():
            if not isinstance(conv_id, str) or not isinstance(records, list):
                continue
            loaded = [
                r
                for r in (self._record_from_dict(item) for item in records)
                if r is not None
            ]
            if loaded:
                self._histories[conv_id] = loaded

    def _persist(self) -> None:
        """Atomically persist conversation history to disk.

        Failures are silent (history is best-effort cache).
        """
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        conv_id: [self._record_to_dict(r) for r in records]
                        for conv_id, records in self._histories.items()
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._persist_path)
        except OSError:
            pass

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
        self._persist()

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
                self._persist()
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
            summary_text = summary.get("text") if summary else ""
            summary_text = summary_text or ""
            rows.append(
                {
                    "conv_id": conv_id,
                    "messages": len(history),
                    "older_count": self.older_count(conv_id),
                    "has_summary": bool(summary_text),
                    "summary_len": len(summary_text),
                }
            )
        return rows

    def recent_user_texts(
        self,
        conv_id: str,
        sender_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Recent plain texts of one sender, newest first.

        Used as extraction material for profile writeback so the profile is
        built from more than the single triggering message.

        Args:
            conv_id: Conversation identifier.
            sender_id: The sender's platform id.
            limit: How many recent messages to return.

        Returns:
            The recent texts, empty when the sender has none.
        """
        texts: list[str] = []
        for record in reversed(self._history(conv_id)):
            if record.role != "user" or record.sender_id != sender_id:
                continue
            text = (record.text or "").strip()
            if text and text not in ("[图片]", "[媒体(下载失败)]", "[转发消息]"):
                texts.append(text)
            if len(texts) >= limit:
                break
        return texts

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
        self._persist()

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
        self._persist()

    @staticmethod
    def _sender_label(record: MessageRecord) -> str:
        """Render a user record's sender as ``昵称`` or ``昵称(QQ号)``.

        The platform id is appended so the AI can tell same-nickname people
        apart and address them precisely.

        Args:
            record: The message record.

        Returns:
            The display label of the sender.
        """
        name = record.sender_name or "未知用户"
        if record.sender_id:
            return f"{name}({record.sender_id})"
        return name

    def _format_record(self, record: MessageRecord) -> str:
        if record.role == "assistant":
            return record.text
        prefix = f"{self._sender_label(record)}: "
        body = ""
        if record.quote:
            body += f"[引用了消息: {escape_user_markers(record.quote)}] "
        if record.description:
            desc = f"[图片描述: {record.description}]"
            body += (
                f"{escape_user_markers(record.text)} {desc}" if record.text else desc
            )
        elif record.text:
            body += escape_user_markers(record.text)
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
            body += f"[引用了消息: {escape_user_markers(record.quote)}] "
        if record.description:
            desc = f"[图片描述: {record.description}]"
            body += (
                f"{escape_user_markers(record.text)} {desc}" if record.text else desc
            )
        elif record.text:
            body += escape_user_markers(record.text)
        if record.images and not body:
            return None
        return f"{self._sender_label(record)}: {body}" if body else None

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
                "以下是你回忆起的过往片段（可能来自其他群聊、私聊或很久以前，"
                "不代表当前发生的事，也不代表你本人执行过任何操作）。它们只是"
                "背景知识：除非用户明确问起，不要在当前对话中主动提起或当作"
                "当前聊天的话题或依据:"
            )
            background.extend(f"- {escape_user_markers(text)}" for text in memory_texts)
        if profile_texts:
            background.append(
                "以下是你对该用户的了解（人物画像，可能随时间更新）。自然融入"
                "即可，不要主动复述画像内容，也不要把它当作当前对话的依据:"
            )
            background.extend(escape_user_markers(text) for text in profile_texts)
        if older:
            summary = self.get_summary(conv_id)
            if summary:
                background.append("更早的对话（已压缩摘要）:")
                background.append(escape_user_markers(summary))
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
