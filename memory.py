"""Global memory store for self-learning.

ChatCore keeps its own persistent memory shared across all group chats. It is a
simple vector store (embedding + cosine similarity) persisted as JSON under
AstrBot's plugin data directory, so no external database is required.
"""

import asyncio
import json
import math
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryStore:
    """Persistent embedding-based memory store.

    Args:
        embed_fn: Async callable mapping a text to an embedding vector.
        path: Path of the JSON persistence file.
        max_entries: Maximum number of stored memory entries.
        min_score: Minimum cosine similarity for a memory to be recalled.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        path: str | Path,
        max_entries: int = 2000,
        min_score: float = 0.4,
    ) -> None:
        self.embed_fn = embed_fn
        self.path = Path(path)
        self.max_entries = max_entries
        self.min_score = min_score
        self._entries: list[dict] = []
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = [e for e in data if isinstance(e, dict) and e.get("vec")]
        except (OSError, json.JSONDecodeError):
            self._entries = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._entries, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    async def add(self, text: str, tags: list[str] | None = None) -> None:
        """Embed and store a memory entry.

        Args:
            text: The memory text to remember.
            tags: Optional tags (e.g. group id) for future filtering.
        """
        text = text.strip()
        if not text:
            return
        async with self._lock:
            vec = await self.embed_fn(text)
            if not vec:
                return
            self._entries.append(
                {
                    "text": text,
                    "tags": tags or [],
                    "ts": time.time(),
                    "vec": vec,
                }
            )
            if len(self._entries) > self.max_entries:
                del self._entries[: len(self._entries) - self.max_entries]
            self._save()

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Retrieve the most relevant memory texts for a query.

        Args:
            query: The text to match against stored memories.
            top_k: Maximum number of memories to return.
            tags: If given, only memories carrying any of these tags match.

        Returns:
            Recalled memory texts sorted by relevance.
        """
        if not self._entries:
            return []
        qvec = await self.embed_fn(query)
        if not qvec:
            return []
        scored = []
        for entry in self._entries:
            if tags is not None and not (set(entry.get("tags", [])) & set(tags)):
                continue
            score = _cosine(qvec, entry["vec"])
            if score >= self.min_score:
                scored.append((score, entry["text"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def count(self) -> int:
        """Number of stored memory entries.

        Returns:
            Entry count.
        """
        return len(self._entries)

    def list_entries(self) -> list[dict]:
        """List all stored memories for management (WebUI).

        Returns:
            A list of entry summaries (without embedding vectors).
        """
        return [
            {
                "index": i,
                "text": e.get("text", ""),
                "tags": e.get("tags", []),
                "ts": e.get("ts", 0),
            }
            for i, e in enumerate(self._entries)
        ]

    def delete_entry(self, index: int) -> bool:
        """Delete a memory entry by its list index.

        Args:
            index: Index returned by ``list_entries``.

        Returns:
            True if an entry was removed.
        """
        if not 0 <= index < len(self._entries):
            return False
        self._entries.pop(index)
        self._save()
        return True
