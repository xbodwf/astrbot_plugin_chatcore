"""Model access through AstrBot's provider system.

ChatCore no longer manages its own model endpoints. The user picks the chat,
vision and implicit-analysis providers inside AstrBot (``_special``
``select_provider`` fields in the plugin config), and ChatCore resolves them by
provider id through the provider manager at call time.
"""

from collections.abc import AsyncGenerator
import base64
import mimetypes
from pathlib import Path

from astrbot.api.provider import Provider
from astrbot.api.star import Context
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import EmbeddingProvider

from .request_log import write_latest

_VISION_PROMPT = "请用一两句话简洁描述这张图片的内容。"

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ThinkStripper:
    """Removes ``<think>...</think>`` reasoning blocks from a text stream.

    Tags may be split across arbitrary chunk boundaries, so the stripper holds
    back characters that could be a tag prefix and only emits text that is
    outside a think block.

    Args:
        open_tag: The opening tag.
        close_tag: The closing tag.
    """

    def __init__(
        self,
        open_tag: str = _THINK_OPEN,
        close_tag: str = _THINK_CLOSE,
    ) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self._buf: list[str] = []
        self._in_think = False

    def feed(self, text: str) -> list[str]:
        """Feed a text chunk.

        Args:
            text: The chunk of text.

        Returns:
            Emitted non-think text fragments.
        """
        out: list[str] = []
        for ch in text:
            if self._in_think:
                if self._buf or ch == self.close_tag[0]:
                    self._buf.append(ch)
                    candidate = "".join(self._buf)
                    if candidate == self.close_tag:
                        self._in_think = False
                        self._buf = []
                    elif not self.close_tag.startswith(candidate):
                        self._buf = []
                continue
            if ch == self.open_tag[0]:
                self._buf.append(ch)
            elif self._buf:
                self._buf.append(ch)
                candidate = "".join(self._buf)
                if candidate == self.open_tag:
                    self._in_think = True
                    self._buf = []
                elif not self.open_tag.startswith(candidate):
                    out.append(candidate)
                    self._buf = []
            else:
                out.append(ch)
        return out

    def flush(self) -> str:
        """Release any remaining held-back text.

        Returns:
            Buffered non-think text (empty inside an open think block).
        """
        if self._in_think:
            self._buf = []
            return ""
        text = "".join(self._buf)
        self._buf = []
        return text


def strip_think(text: str) -> str:
    """Strip ``<think>...</think>`` blocks from a complete text.

    Args:
        text: The complete text.

    Returns:
        The text without think blocks.
    """
    stripper = ThinkStripper()
    return "".join(stripper.feed(text)) + stripper.flush()


class LLMProvider:
    """Resolves an AstrBot chat provider by id and wraps chat/stream calls.

    Args:
        context: AstrBot plugin context (holds the provider manager).
        provider_id: Provider id selected in the plugin config.
    """

    def __init__(self, context: Context, provider_id: str) -> None:
        self.context = context
        self.provider_id = provider_id

    async def _get(self) -> Provider:
        """Resolve the provider instance by id.

        Returns:
            The AstrBot chat provider.

        Raises:
            ValueError: When the provider is unconfigured or missing.
        """
        if not self.provider_id:
            raise ValueError("未配置模型提供商（provider_id 为空）")
        provider = await self.context.provider_manager.get_provider_by_id(
            self.provider_id
        )
        if not provider:
            raise ValueError(
                f"提供商 {self.provider_id} 不存在，请在 AstrBot 提供商页面检查配置"
            )
        return provider

    @staticmethod
    def _to_text(resp: LLMResponse) -> str:
        """Extract plain text from an LLMResponse.

        Args:
            resp: The provider response.

        Returns:
            The plain text, possibly empty for tool-call-only chunks.
        """
        if resp.result_chain is not None:
            text = resp.result_chain.get_plain_text()
            if text:
                return text
        return (resp.completion_text or "").strip()

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        images: list[str] | None = None,
        func_tool=None,
        log_name: str = "chat",
    ) -> str:
        """Get a full (non-streaming) chat completion text.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            images: Image URLs to attach to the request.
            func_tool: Optional ToolSet for function calling.

        Returns:
            The assistant's reply text.
        """
        provider = await self._get()
        write_latest(
            log_name,
            {
                "provider_id": self.provider_id,
                "messages": messages,
                "temperature": temperature,
                "images": images or [],
                "func_tool": repr(func_tool) if func_tool is not None else None,
            },
        )
        resp = await provider.text_chat(
            contexts=messages,
            image_urls=images or None,
            temperature=temperature,
            func_tool=func_tool,
        )
        return strip_think(self._to_text(resp))

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        images: list[str] | None = None,
        func_tool=None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion, yielding text deltas.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            images: Image URLs to attach to the request.
            func_tool: Optional ToolSet for function calling.

        Yields:
            Text deltas as they arrive. The trailing full-completion response
            is skipped so deltas are not accumulated twice.
        """
        provider = await self._get()
        stripper = ThinkStripper()
        async for resp in provider.text_chat_stream(
            contexts=messages,
            image_urls=images or None,
            temperature=temperature,
            func_tool=func_tool,
        ):
            if not resp.is_chunk:
                continue
            text = self._to_text(resp)
            if not text:
                continue
            for delta in stripper.feed(text):
                if delta:
                    yield delta

    async def chat_stream_raw(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        images: list[str] | None = None,
        func_tool=None,
        log_name: str = "chat",
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream raw LLM responses (chunks and the final completion).

        The trailing full completion carries the accumulated function-call
        fields (``tools_call_name`` etc.), which the text-level
        ``chat_stream`` drops. Used by the tool-call loop.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            images: Image URLs to attach to the request.
            func_tool: Optional ToolSet for function calling.

        Yields:
            Every LLMResponse, chunks included.
        """
        provider = await self._get()
        write_latest(
            log_name,
            {
                "provider_id": self.provider_id,
                "messages": messages,
                "temperature": temperature,
                "images": images or [],
                "func_tool": repr(func_tool) if func_tool is not None else None,
            },
        )
        async for resp in provider.text_chat_stream(
            contexts=messages,
            image_urls=images or None,
            temperature=temperature,
            func_tool=func_tool,
        ):
            yield resp

    async def describe_image(self, image_url: str, *, log_name: str = "vision") -> str:
        """Describe a single image with the vision provider.

        Args:
            image_url: URL or base64 data URI of the image.

        Returns:
            A short text description of the image.
        """
        image_input = image_url
        image_path = Path(image_url)
        if image_path.is_file():
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            image_input = (
                f"data:{mime_type};base64,"
                + base64.b64encode(image_path.read_bytes()).decode("ascii")
            )
        messages = [{"role": "user", "content": _VISION_PROMPT}]
        return await self.chat(
            messages,
            temperature=0.0,
            images=[image_input],
            log_name=log_name,
        )


class EmbeddingAdapter:
    """Resolves an AstrBot embedding provider by id.

    Args:
        context: AstrBot plugin context.
        provider_id: Provider id of an embedding provider.
    """

    def __init__(self, context: Context, provider_id: str) -> None:
        self.context = context
        self.provider_id = provider_id

    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            ValueError: When the provider is missing or not an embedding provider.
        """
        if not self.provider_id:
            raise ValueError("未配置 Embedding 提供商（provider_id 为空）")
        provider = await self.context.provider_manager.get_provider_by_id(
            self.provider_id
        )
        if not provider or not isinstance(provider, EmbeddingProvider):
            raise ValueError(f"提供商 {self.provider_id} 不是有效的 Embedding 提供商")
        return await provider.get_embedding(text)
