"""Model access through AstrBot's provider system.

ChatCore no longer manages its own model endpoints. The user picks the chat,
vision and implicit-analysis providers inside AstrBot (``_special``
``select_provider`` fields in the plugin config), and ChatCore resolves them by
provider id through the provider manager at call time.
"""

from collections.abc import AsyncGenerator

from astrbot.api.provider import Provider
from astrbot.api.star import Context
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import EmbeddingProvider

_VISION_PROMPT = "请用一两句话简洁描述这张图片的内容。"


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
    ) -> str:
        """Get a full (non-streaming) chat completion text.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            images: Image URLs to attach to the request.

        Returns:
            The assistant's reply text.
        """
        provider = await self._get()
        resp = await provider.text_chat(
            contexts=messages,
            image_urls=images or None,
            temperature=temperature,
        )
        return self._to_text(resp)

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        images: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion, yielding text deltas.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            images: Image URLs to attach to the request.

        Yields:
            Text deltas as they arrive. The trailing full-completion response
            is skipped so deltas are not accumulated twice.
        """
        provider = await self._get()
        async for resp in provider.text_chat_stream(
            contexts=messages,
            image_urls=images or None,
            temperature=temperature,
        ):
            if not resp.is_chunk:
                continue
            text = self._to_text(resp)
            if text:
                yield text

    async def describe_image(self, image_url: str) -> str:
        """Describe a single image with the vision provider.

        Args:
            image_url: URL or base64 data URI of the image.

        Returns:
            A short text description of the image.
        """
        messages = [{"role": "user", "content": _VISION_PROMPT}]
        return await self.chat(messages, temperature=0.0, images=[image_url])


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
            raise ValueError(
                f"提供商 {self.provider_id} 不是有效的 Embedding 提供商"
            )
        return await provider.get_embedding(text)
