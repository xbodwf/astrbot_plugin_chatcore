"""Persist the latest complete prompt for each ChatCore request kind."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path


def write_latest(name: str, payload: dict[str, Any]) -> None:
    """Overwrite the latest request log for one request kind.

    Args:
        name: Stable request kind used in the filename.
        payload: Complete request data to serialize.
    """
    log_dir = Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_chatcore" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    data = {"logged_at": datetime.now(timezone.utc).isoformat(), **payload}
    (log_dir / f"latest_{name}.log").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
