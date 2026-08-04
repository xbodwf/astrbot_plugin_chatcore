"""Overwrite-on-request logs for inspecting the latest LLM context."""

import json
import os
import time
from pathlib import Path
from typing import Any


class RequestLogger:
    """Write the latest request for each model purpose to a data directory.

    Args:
        directory: Directory containing the ``latest_*.log`` files.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def write(
        self,
        name: str,
        *,
        provider_id: str,
        messages: list[dict],
        temperature: float,
        images: list[str] | None = None,
        func_tool: Any = None,
    ) -> None:
        """Overwrite one latest-request log.

        Args:
            name: Log filename stem, for example ``latest_chat``.
            provider_id: Selected AstrBot provider id.
            messages: Complete provider context.
            temperature: Sampling temperature.
            images: Image URLs or data URIs attached to the request.
            func_tool: Tool set, if one is attached to the request.
        """
        safe_name = "".join(
            char for char in str(name) if char.isalnum() or char in ("_", "-")
        )
        if not safe_name:
            return
        payload = {
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider_id": provider_id,
            "temperature": temperature,
            "images": images or [],
            "tools": repr(func_tool) if func_tool is not None else None,
            "messages": messages,
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / f"{safe_name}.log"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        except OSError:
            pass
