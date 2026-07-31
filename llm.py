"""Independent model provider clients (OpenAI-compatible APIs).

ChatCore keeps its own model configuration, independent from AstrBot's
provider system, so it always talks to the exact model the user configured.
All HTTP is done through aiohttp.
"""

import json
from collections.abc import AsyncGenerator

import aiohttp


class ChatClient:
    """OpenAI-compatible chat / streaming / vision / embedding client.

    Args:
        base_url: OpenAI-compatible API base URL, e.g. ``https://api.openai.com/v1``.
        api_key: API key.
        model: Default chat model name.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._timeout = aiohttp.ClientTimeout(total=120)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post_json(self, path: str, payload: dict) -> dict:
        async with aiohttp.ClientSession(
            timeout=self._timeout,
            headers=self._headers(),
        ) as session:
            async with session.post(
                f"{self.base_url}/{path}",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    @staticmethod
    def _attach_images(
        messages: list[dict],
        images: list[str],
    ) -> list[dict]:
        """Merge image URLs into the last message as multimodal content.

        Args:
            messages: OpenAI-style message list.
            images: Image URLs to attach to the final user message.

        Returns:
            A new message list with the images attached.
        """
        if not messages or not images:
            return messages
        last = messages[-1]
        content: list[dict] = [{"type": "text", "text": last.get("content", "")}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in images
        )
        return [*messages[:-1], {**last, "content": content}]

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.8,
        images: list[str] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion, yielding text deltas.

        Args:
            messages: OpenAI-style message list.
            model: Override model name; defaults to the client default.
            temperature: Sampling temperature.
            images: Image URLs to attach to the final user message.
            extra_body: Extra payload fields to merge in.

        Yields:
            Text deltas as they arrive.
        """
        if images:
            messages = self._attach_images(messages, images)
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)

        async with aiohttp.ClientSession(
            timeout=self._timeout,
            headers=self._headers(),
        ) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    line = line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:") :].strip()
                    if not data or data == b"[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.8,
        images: list[str] | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """Get a full (non-streaming) chat completion text.

        Args:
            messages: OpenAI-style message list.
            model: Override model name; defaults to the client default.
            temperature: Sampling temperature.
            images: Image URLs to attach to the final user message.
            extra_body: Extra payload fields to merge in.

        Returns:
            The assistant's reply text.
        """
        if images:
            messages = self._attach_images(messages, images)
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)
        data = await self._post_json("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    async def describe_image(self, image_url: str) -> str:
        """Describe an image using a vision-capable model.

        Args:
            image_url: URL or base64 data URI of the image.

        Returns:
            A short text description of the image.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": "请用一两句话简洁描述这张图片的内容。",
                    },
                ],
            }
        ]
        return await self.chat(messages)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            RuntimeError: If the embedding request failed.
        """
        payload = {"model": self.model, "input": text}
        data = await self._post_json("embeddings", payload)
        embeds = data.get("data") or []
        if not embeds:
            raise RuntimeError("Embedding response contains no data.")
        return embeds[0].get("embedding", [])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        payload = {"model": self.model, "input": texts}
        data = await self._post_json("embeddings", payload)
        by_index = {
            item.get("index"): item.get("embedding", [])
            for item in data.get("data") or []
        }
        return [by_index.get(i, []) for i in range(len(texts))]
