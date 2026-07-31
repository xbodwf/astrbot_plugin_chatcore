"""AI self-segmentation and streaming orchestration.

AstrBot's regex / machine segmentation is replaced by letting the AI mark its
own segment boundaries with a configurable delimiter. The delimiter can be
escaped so the AI can output it literally when needed.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

_SENTENCE_BOUNDARIES = {".", "!", "?", "。", "！", "？", "\n", "…"}


class StreamSegmenter:
    """Escape-aware stream segmenter.

    Feeds text chunks; completed segments are returned as a list. A trailing
    escape char is kept literal on flush.

    Args:
        delimiter: The delimiter string that separates segments.
        escape_char: Prefix char that escapes the next character literally.
        max_segment_chars: Hard cap per segment (0 disables).
    """

    def __init__(
        self,
        delimiter: str,
        escape_char: str = "\\",
        max_segment_chars: int = 0,
    ) -> None:
        self.delimiter = delimiter
        self.escape_char = escape_char
        self.max_segment_chars = max(0, max_segment_chars)
        self._chars: list[str] = []
        self._escaped = False
        self._delim_pos = 0

    def _emit(self) -> str | None:
        text = "".join(self._chars).strip()
        self._chars = []
        self._delim_pos = 0
        return text or None

    def feed(self, text: str) -> list[str]:
        """Feed a chunk of streamed text.

        Args:
            text: The chunk of text.

        Returns:
            Completed segments split out by the delimiter.
        """
        segments: list[str] = []
        for ch in text:
            if self._escaped:
                self._chars.append(ch)
                self._escaped = False
                self._delim_pos = 0
                continue
            if self.escape_char and ch == self.escape_char:
                self._escaped = True
                continue
            self._chars.append(ch)
            if self.delimiter:
                if ch == self.delimiter[self._delim_pos]:
                    self._delim_pos += 1
                    if self._delim_pos == len(self.delimiter):
                        del self._chars[-len(self.delimiter) :]
                        seg = self._emit()
                        if seg:
                            segments.append(seg)
                else:
                    self._delim_pos = 0
            if self.max_segment_chars and len(self._chars) >= self.max_segment_chars:
                seg = self._emit()
                if seg:
                    segments.append(seg)
        return segments

    def has_boundary(self) -> bool:
        """Whether the pending buffer ends on a sentence boundary.

        Returns:
            True if the last buffered char is a natural sentence end.
        """
        return bool(self._chars and self._chars[-1] in _SENTENCE_BOUNDARIES)

    def flush(self) -> str | None:
        """Flush the remaining buffer as the final segment.

        Returns:
            The trailing text, or None if empty.
        """
        if self._escaped:
            self._chars.append(self.escape_char)
            self._escaped = False
        return self._emit()


async def _wait_for_boundary(
    segmenter: StreamSegmenter,
    timeout: float,
) -> str | None:
    """Wait briefly for a natural sentence boundary, then flush.

    Args:
        segmenter: The active segmenter.
        timeout: Max seconds to wait for a boundary.

    Returns:
        The flushed trailing text, or None if empty.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if segmenter.has_boundary():
            break
        await asyncio.sleep(0.05)
    return segmenter.flush()


async def _finish_interrupted(
    stream_gen: AsyncGenerator[str, None],
    segmenter: StreamSegmenter,
    send_fn: Callable[[str], Awaitable[None]],
    timeout: float,
    segment_token: Callable[[], Any] | None = None,
) -> tuple[str | None, Any]:
    """Keep consuming the stream so the AI can finish its current sentence.

    Consumes until a sentence boundary or a completed delimiter segment is
    reached, or until ``timeout`` elapses. Completed segments are sent via
    ``send_fn``; the final flushed remainder is returned so the caller decides
    how to deliver it (e.g. suppressed when the reply was recalled).

    At every completed segment, ``segment_token`` is polled; when it returns a
    truthy value (a newer debounced message or a cancel), consumption stops
    immediately, the buffered remainder is discarded, and that value is
    returned so the caller can act on it.

    Args:
        stream_gen: The active text stream.
        segmenter: The active segmenter.
        send_fn: Async callback to send a finished segment.
        timeout: Max seconds to wait for the sentence to finish.
        segment_token: Optional callback polled at segment boundaries.

    Returns:
        A ``(trailing_text, stop_token)`` tuple. ``trailing_text`` is the
        flushed remainder to deliver (None when stopped early); ``stop_token``
        is the truthy ``segment_token`` value when consumption should stop.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        while True:
            if segmenter.has_boundary():
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(
                    stream_gen.__anext__(),
                    timeout=remaining,
                )
            except (StopAsyncIteration, asyncio.TimeoutError):
                break
            for seg in segmenter.feed(chunk):
                token = segment_token() if segment_token else None
                if token:
                    return None, token
                await send_fn(seg)
    except asyncio.CancelledError:
        raise
    return segmenter.flush(), None


async def stream_respond(
    stream_gen: AsyncGenerator[str, None],
    send_fn: Callable[[str], Awaitable[None]],
    *,
    delimiter: str,
    escape_char: str = "\\",
    interval: float = 1.0,
    max_segment_chars: int = 0,
    interrupt_check: Callable[[], Any] | None = None,
    finish_timeout: float = 2.0,
) -> tuple[str | None, Any] | None:
    """Consume a text stream, segment it and send each segment.

    Segments are sent with a small ``interval`` pause in between so the user
    perceives natural typing while the AI keeps generating. If
    ``interrupt_check`` returns a truthy value (e.g. a debounced new message or
    a recall cancel), the current sentence is allowed to finish (bounded by
    ``finish_timeout``); its trailing remainder is returned along with the
    interrupt value so the caller decides how to proceed.

    Args:
        stream_gen: Async generator yielding text deltas.
        send_fn: Async callback to send a finished segment.
        delimiter: Segment delimiter used by the AI.
        escape_char: Escape char for the delimiter.
        interval: Pause between segments, in seconds.
        max_segment_chars: Hard cap per segment (0 disables).
        interrupt_check: Optional callable returning ``(text, cancelled)`` when
            interrupted, else None.
        finish_timeout: Max seconds to wait for a sentence boundary on interrupt.

    Returns:
        A ``(trailing_text, (text, cancelled))`` tuple when interrupted, else
        None. ``trailing_text`` is the current sentence's remainder to deliver.
    """
    segmenter = StreamSegmenter(delimiter, escape_char, max_segment_chars)
    try:
        async for chunk in stream_gen:
            signal = interrupt_check() if interrupt_check else None
            if signal:
                for seg in segmenter.feed(chunk):
                    await send_fn(seg)
                trailing, newer = await _finish_interrupted(
                    stream_gen,
                    segmenter,
                    send_fn,
                    finish_timeout,
                    segment_token=interrupt_check,
                )
                await stream_gen.aclose()
                return trailing, newer or signal
            for seg in segmenter.feed(chunk):
                await send_fn(seg)
                if interval > 0:
                    await asyncio.sleep(interval)
        final = segmenter.flush()
        if final:
            await send_fn(final)
    except asyncio.CancelledError:
        raise
    return None
