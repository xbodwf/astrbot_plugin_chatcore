"""Emoji/sticker library with full provenance.

MaiBot-style emoji libraries are blind: the AI only sees a categorized list
and cannot tell where an emoji came from or what it meant in its original
conversation. ChatCore stores every emoji together with its provenance
(source group, sender, original message text and context window), and the
category/tags are derived at collection time from the image description PLUS
that source context. When the AI wants to send an emoji it searches by intent,
reads each candidate's original context, and only then picks one.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

_CLASSIFY_PROMPT = (
    "这是一条群聊表情包。结合【图片描述】和【来源语境】判断它的使用含义。\n"
    "图片描述: {desc}\n"
    "来源语境: {context}\n"
    '只输出 JSON：{{"category": "分类", "tags": ["标签1", "标签2"]}}。\n'
    "分类从：开心/嘲讽/敷衍/震惊/生气/可爱/疑问/委屈/无语/其他 中选一个。"
)

_CATEGORY_HINT = "开心 嘲讽 敷衍 震惊 生气 可爱 疑问 委屈 无语 其他"


class EmojiStore:
    """Persistent emoji library with provenance.

    The image files live under ``data_dir`` and the metadata index under
    ``index_path``. Each record carries the full source context so the AI can
    read what an emoji originally meant before using it.

    Args:
        data_dir: Directory holding the emoji image files.
        index_path: Path of the JSON metadata index.
        max_entries: Maximum number of stored emoji (oldest evicted).
    """

    def __init__(
        self,
        data_dir: str | Path,
        index_path: str | Path,
        max_entries: int = 500,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path)
        self.max_entries = max_entries
        self._records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._records = {
                key: value
                for key, value in data.items()
                if isinstance(value, dict) and value.get("emoji_id")
            }
        except (OSError, json.JSONDecodeError):
            self._records = {}

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._records, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.index_path)

    def collect(
        self,
        source_file: str | Path,
        *,
        source_group: str,
        source_sender: str,
        source_message_id: str,
        source_text: str,
        source_context: str,
    ) -> str | None:
        """Copy an image into the library and record its provenance.

        Args:
            source_file: Local path of the source image.
            source_group: Group id where it was collected.
            source_sender: Sender who posted it.
            source_message_id: Original message id.
            source_text: Text of the original message.
            source_context: Surrounding conversation window.

        Returns:
            The new emoji id, or None when the image is missing.
        """
        src = Path(source_file)
        if not src.is_file():
            return None
        emoji_id, dest = self._new_destination(src.suffix or ".image")
        try:
            shutil.copy2(src, dest)
        except OSError:
            return None
        return self._register(
            emoji_id,
            dest,
            source_group=source_group,
            source_sender=source_sender,
            source_message_id=source_message_id,
            source_text=source_text,
            source_context=source_context,
        )

    async def collect_from_url(
        self,
        url: str,
        *,
        source_group: str,
        source_sender: str,
        source_message_id: str,
        source_text: str,
        source_context: str,
    ) -> str | None:
        """Download an image URL into the library and record its provenance.

        OneBot image components usually carry only a network URL, so
        collection must fetch the bytes itself.

        Args:
            url: The image URL.
            source_group: Group id where it was collected.
            source_sender: Sender who posted it.
            source_message_id: Original message id.
            source_text: Text of the original message.
            source_context: Surrounding conversation window.

        Returns:
            The new emoji id, or None on any failure.
        """
        import aiohttp
        from urllib.parse import urlparse

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.read()
        except Exception:
            return None
        if not data:
            return None
        ext = Path(urlparse(url).path).suffix or ".image"
        emoji_id, dest = self._new_destination(ext)
        try:
            dest.write_bytes(data)
        except OSError:
            return None
        return self._register(
            emoji_id,
            dest,
            source_group=source_group,
            source_sender=source_sender,
            source_message_id=source_message_id,
            source_text=source_text,
            source_context=source_context,
        )

    def _new_destination(self, ext: str) -> tuple[str, Path]:
        """Allocate a fresh emoji id and its destination file path.

        Args:
            ext: File extension for the stored image.

        Returns:
            A ``(emoji_id, dest_path)`` tuple.
        """
        now = time.time()
        emoji_id = f"emoji_{int(now * 1000)}"
        while emoji_id in self._records:
            now += 1.0
            emoji_id = f"emoji_{int(now * 1000)}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return emoji_id, self.data_dir / f"{emoji_id}{ext}"

    def _register(
        self,
        emoji_id: str,
        dest: Path,
        *,
        source_group: str,
        source_sender: str,
        source_message_id: str,
        source_text: str,
        source_context: str,
    ) -> str:
        """Record an emoji entry, evicting the oldest when over capacity.

        Args:
            emoji_id: The allocated emoji id.
            dest: The stored image file.
            source_group: Group id where it was collected.
            source_sender: Sender who posted it.
            source_message_id: Original message id.
            source_text: Text of the original message.
            source_context: Surrounding conversation window.

        Returns:
            The emoji id.
        """
        self._records[emoji_id] = {
            "emoji_id": emoji_id,
            "file": str(dest),
            "source_group": source_group or "",
            "source_sender": source_sender or "",
            "source_message_id": source_message_id or "",
            "source_text": (source_text or "")[:200],
            "source_context": (source_context or "")[:500],
            "collected_at": time.time(),
            "category": "",
            "tags": [],
            "usage_count": 0,
        }
        if len(self._records) > self.max_entries:
            oldest = sorted(
                self._records,
                key=lambda rid: self._records[rid]["collected_at"],
            )
            for rid in oldest[: len(self._records) - self.max_entries]:
                self.delete(rid, remove_file=True)
        self._save()
        return emoji_id

    def set_meta(self, emoji_id: str, category: str, tags: list[str]) -> None:
        """Set an emoji's category and tags after classification.

        Args:
            emoji_id: The emoji id.
            category: Classified usage category.
            tags: Classified usage tags.
        """
        record = self._records.get(emoji_id)
        if not record:
            return
        if category:
            record["category"] = category
        if tags:
            record["tags"] = record.get("tags", []) + list(tags)
            seen = []
            for t in record["tags"]:
                if t not in seen:
                    seen.append(t)
            record["tags"] = seen[:12]
        self._save()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search emoji by intent across category, tags and source text.

        Args:
            query: The AI's intent text.
            top_k: Max results.

        Returns:
            Matching records (including provenance).
        """
        if not self._records:
            return []
        q = (query or "").strip().lower()
        if not q:
            return sorted(
                self._records.values(),
                key=lambda r: r.get("usage_count", 0),
                reverse=True,
            )[:top_k]
        scored = []
        for record in self._records.values():
            haystack = " ".join(
                [
                    record.get("category", ""),
                    " ".join(record.get("tags", [])),
                    record.get("source_text", ""),
                    record.get("source_context", ""),
                ]
            ).lower()
            score = sum(1 for token in q.split() if token in haystack)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def get(self, emoji_id: str) -> dict | None:
        """Get a single record.

        Args:
            emoji_id: The emoji id.

        Returns:
            The record dict or None.
        """
        return self._records.get(emoji_id)

    def file_path(self, emoji_id: str) -> str | None:
        """Local file path of an emoji.

        Args:
            emoji_id: The emoji id.

        Returns:
            The file path or None.
        """
        record = self._records.get(emoji_id)
        return record.get("file") if record else None

    def mark_used(self, emoji_id: str) -> None:
        """Increment usage count.

        Args:
            emoji_id: The emoji id.
        """
        record = self._records.get(emoji_id)
        if record:
            record["usage_count"] = record.get("usage_count", 0) + 1
            self._save()

    def delete(self, emoji_id: str, remove_file: bool = False) -> bool:
        """Delete an emoji.

        Args:
            emoji_id: The emoji id.
            remove_file: Also delete the stored image file.

        Returns:
            True if a record was removed.
        """
        record = self._records.pop(emoji_id, None)
        if record is None:
            return False
        if remove_file:
            try:
                Path(record.get("file", "")).unlink(missing_ok=True)
            except OSError:
                pass
        self._save()
        return True

    def all(self) -> list[dict]:
        """All records.

        Returns:
            List of records.
        """
        return list(self._records.values())

    def count(self) -> int:
        """Number of stored emoji.

        Returns:
            Emoji count.
        """
        return len(self._records)

    @staticmethod
    def render_candidates(records: list[dict]) -> str:
        """Render search candidates for the AI to read and choose from.

        Args:
            records: Search results.

        Returns:
            A text block listing each candidate with its source context.
        """
        if not records:
            return "没有匹配的表情包。"
        lines = [
            "表情包候选（请结合来源语境判断含义后选择最合适的一个，用 [[emoji:编号]] 表示）:"
        ]
        for i, record in enumerate(records, start=1):
            lines.append(
                f"[{i}] 编号={record['emoji_id']}，分类={record.get('category', '未分类')}，"
                f"标签={','.join(record.get('tags', [])) or '无'}"
            )
            if record.get("source_text"):
                lines.append(f"    原消息: {record['source_text']}")
            if record.get("source_context"):
                lines.append(f"    来源语境: {record['source_context']}")
            lines.append(
                f"    来源: {record.get('source_group', '未知群')} / {record.get('source_sender', '?')}"
            )
        return "\n".join(lines)


def parse_classify_response(raw: str) -> tuple[str, list[str]]:
    """Parse the classification JSON out of a model reply.

    Args:
        raw: Raw model output.

    Returns:
        A ``(category, tags)`` tuple; empty category on failure.
    """
    text = raw.strip()
    import re

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return "", []
    if not isinstance(parsed, dict):
        return "", []
    category = str(parsed.get("category", "")).strip()
    tags = parsed.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if isinstance(t, str) and str(t).strip()]
    return category, tags


async def classify_emoji(
    vision_client: Any,
    llm_client: Any,
    image_path: str,
    source_context: str,
) -> tuple[str, list[str]]:
    """Classify an emoji from its image description plus source context.

    Args:
        vision_client: An LLM wrapper exposing ``async describe_image(url)``.
        llm_client: An LLM wrapper exposing ``async chat(messages)``.
        image_path: Local path or URL of the image.
        source_context: The original conversation context of the emoji.

    Returns:
        A ``(category, tags)`` tuple; empty category on any failure.
    """
    try:
        desc = await vision_client.describe_image(image_path)
    except Exception:
        desc = ""
    if not desc:
        return "", []
    prompt = _CLASSIFY_PROMPT.replace("{desc}", desc[:200]).replace(
        "{context}", (source_context or "")[:300]
    )
    try:
        raw = await llm_client.chat(
            [
                {
                    "role": "system",
                    "content": "你是表情包分类助手，只输出 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception:
        return "", []
    return parse_classify_response(raw)
