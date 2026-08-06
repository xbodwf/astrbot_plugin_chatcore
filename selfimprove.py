"""Self-improvement: AI proposes source changes, admin approves.

The improvement loop periodically hands the plugin source (read-only) plus
recent chat samples to the chat model in a special session. The model can
only write into a staging directory (``selfimprove/staging``); every proposal
must pass ``ruff`` before it becomes a pending patch. The admin reviews with
``chatcore view`` and applies with ``chatcore approve <id>``; applying copies
the changed files back over the plugin source and reloads the plugin.
"""

from __future__ import annotations

import difflib
import json
import shutil
import time
import uuid
from pathlib import Path

_SYSTEM_PROMPT = (
    "你现在不是在进行角色扮演，人格仅作为依据。"
    "你是一个在改进自己源代码的 AI 工程师。"
    "你的目标是找出自己在日常角色扮演中的降智行为、缺失的真人化能力、"
    "插件代码中可以改进的地方，并直接动手改进。\n"
    "规则：\n"
    "1. 只能读取插件源码目录（read-only）与聊天记录片段文件；\n"
    "2. 只能通过写工具修改 staging 目录下的文件（与源码目录同结构）；\n"
    "3. 每次修改后必须运行 `ruff check <文件>` 并通过，才能提交；\n"
    "4. 提交时给出简明说明（问题、改动、理由）。\n"
    "不要改动与目标无关的文件，不要提交无法通过 ruff 的代码。"
)


class SelfImprove:
    """Pending-improvement store with staging management.

    Args:
        source_dir: Plugin source directory (read-only for the model).
        work_dir: Directory holding ``staging/`` and ``pending.json``.
    """

    def __init__(self, source_dir: str | Path, work_dir: str | Path) -> None:
        self.source_dir = Path(source_dir)
        self.work_dir = Path(work_dir)
        self.staging_dir = self.work_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.work_dir / "pending.json"
        self._pending: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.pending_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._pending = {
                    k: v for k, v in data.items() if isinstance(v, dict)
                }
        except (OSError, json.JSONDecodeError):
            self._pending = {}

    def _save(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.pending_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._pending, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.pending_path)

    def new_staging_root(self) -> str:
        """Allocate a fresh staging root for one improvement session.

        Returns:
            Absolute path of the new staging root.
        """
        root = self.staging_dir / f"session_{int(time.time())}"
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    def submit(
        self, session_id: str, note: str, files: list[str]
    ) -> str:
        """Register a pending improvement from a session's staging files.

        Args:
            session_id: Staging root folder name.
            note: Model's explanation of the change.
            files: Relative file paths changed within the session.

        Returns:
            The new pending id.
        """
        pid = uuid.uuid4().hex[:10]
        self._pending[pid] = {
            "id": pid,
            "session": session_id,
            "note": note,
            "files": files,
            "created_at": time.time(),
        }
        self._save()
        return pid

    def list_pending(self) -> list[dict]:
        """List pending improvements (newest first)."""
        return sorted(
            self._pending.values(),
            key=lambda p: p.get("created_at", 0),
            reverse=True,
        )

    def get(self, pid: str) -> dict | None:
        """Get one pending improvement."""
        return self._pending.get(pid)

    def diff(self, pid: str) -> str:
        """Render the unified diff of a pending improvement.

        Args:
            pid: Pending id.

        Returns:
            The unified diff text, or a message when missing.
        """
        pending = self._pending.get(pid)
        if not pending:
            return f"未找到审批 {pid}"
        session = self.staging_dir / str(pending["session"])
        if not session.is_dir():
            return "staging 目录已不存在"
        parts: list[str] = []
        for rel in pending.get("files", []):
            staged = session / rel
            original = self.source_dir / rel
            if not staged.is_file():
                continue
            new_text = staged.read_text(encoding="utf-8", errors="replace")
            old_text = (
                original.read_text(encoding="utf-8", errors="replace")
                if original.is_file()
                else ""
            )
            parts.append(
                "".join(
                    difflib.unified_diff(
                        old_text.splitlines(keepends=True),
                        new_text.splitlines(keepends=True),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                    )
                )
            )
        return "\n".join(parts) or "(空 diff)"

    def apply(self, pid: str) -> tuple[bool, str]:
        """Apply a pending improvement onto the source directory.

        Args:
            pid: Pending id.

        Returns:
            ``(ok, message)``.
        """
        pending = self._pending.get(pid)
        if not pending:
            return False, f"未找到审批 {pid}"
        session = self.staging_dir / str(pending["session"])
        if not session.is_dir():
            return False, "staging 目录已不存在"
        applied = []
        for rel in pending.get("files", []):
            staged = session / rel
            if not staged.is_file():
                continue
            target = self.source_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, target)
            applied.append(rel)
        self._pending.pop(pid, None)
        self._save()
        return True, f"已应用 {len(applied)} 个文件: {', '.join(applied)}"

    def reject(self, pid: str) -> bool:
        """Reject (remove) a pending improvement.

        Args:
            pid: Pending id.

        Returns:
            True when removed.
        """
        if pid in self._pending:
            self._pending.pop(pid, None)
            self._save()
            return True
        return False


async def ruff_check(paths: list[str]) -> tuple[bool, str]:
    """Run ``ruff check`` on the given paths.

    Args:
        paths: Absolute file paths to check.

    Returns:
        ``(ok, message)``.
    """
    import asyncio

    if not paths:
        return True, ""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ruff",
            "check",
            *paths,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except FileNotFoundError:
        return False, "ruff 未安装（pip install ruff）"
    except asyncio.TimeoutError:
        return False, "ruff 超时"
    out = (stdout + stderr).decode("utf-8", errors="replace")
    return proc.returncode == 0, out
