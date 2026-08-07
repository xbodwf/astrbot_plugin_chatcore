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
import subprocess
import time
import uuid
from pathlib import Path

_SYSTEM_PROMPT = (
    "你现在不是在进行角色扮演，人格仅作为依据。"
    "你是一个在改进自己源代码的 AI 工程师。"
    "你的目标是找出自己在日常角色扮演中的降智行为、缺失的真人化能力、"
    "插件代码中可以改进的地方，并直接动手改进。\n"
    "工作流程：\n"
    "1. 先 list_source 了解源码结构，再 read_source 阅读相关文件"
    "（大文件用 offset/limit 分片读）；\n"
    "2. 找出问题后用 edit_staging 做精确片段替换（推荐，只传改动部分即可），"
    "追加或新建文件用 add_staging，只有整体重写才用 write_staging；\n"
    "3. 每改完一个文件用 run_ruff 校验，不通过就修正；\n"
    "4. 全部改完且 ruff 通过后，调用 submit_improvement 提交（note 说明改动，"
    "files 列出所有改动的相对路径）。\n"
    "5. 当你认为已经完成（提交了改进，或确认无需改动）时，调用 stop_improvement "
    "主动结束会话。\n"
    "规则：只能读取插件源码目录（read-only）；只能写 staging 目录；"
    "不要改动与目标无关的文件，不要提交无法通过 ruff 的代码。\n"
    "重要：不要只分析不行动。哪怕只是小改进（注释、边界处理、提示词措辞），"
    "也至少完成一项 write_staging + run_ruff + submit_improvement 并提交；"
    "只有在你确认代码完全无需改动时才可以不提交。"
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
        self._git_ensure()

    def _git(self, *args: str) -> tuple[int, str]:
        """Run a git command inside the source directory.

        Args:
            *args: Git arguments.

        Returns:
            ``(returncode, combined_output)``.
        """
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.source_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except Exception as e:
            return 1, str(e)

    def _git_ensure(self) -> None:
        """Initialize a git repository on the plugin source, if missing.

        The repo makes every improvement session auditable: each session
        commits a baseline, and proposals are reviewed as git diffs against
        that baseline. Failures are non-fatal (diffing falls back to the
        plain-text comparison).

        Every plugin load snapshots the source state unconditionally: the
        current on-disk files (including automatic-update replacements) are
        committed so history always reflects what was actually running.

        Returns:
            None.
        """
        git_dir = self.source_dir / ".git"
        if not git_dir.exists():
            self._git("init", "-q")
        self._git("add", "-A")
        self._git(
            "commit",
            "-q",
            "-m",
            f"chatcore sync baseline {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "--allow-empty",
        )

    def baseline_commit(self) -> str:
        """Create (or reuse) the baseline commit for a new session.

        Staging the current source and committing it gives the session a
        stable reference point; proposals diff against it.

        Returns:
            The baseline commit hash (or empty on failure).
        """
        self._git("add", "-A")
        code, out = self._git(
            "commit", "-q", "-m", "chatcore session baseline", "--allow-empty"
        )
        if code != 0:
            return ""
        return out.splitlines()[-1] if out else ""

    def commit_apply(self, pid: str, note: str) -> None:
        """Commit an applied improvement into the source git history.

        Args:
            pid: The applied pending id.
            note: The improvement note.

        Returns:
            None.
        """
        self._git("add", "-A")
        self._git(
            "commit", "-q", "-m", f"chatcore apply {pid}: {note[:60]}", "--allow-empty"
        )

    def file_changed_since_baseline(self, rel: str) -> bool:
        """Whether a source file differs from the last committed baseline.

        Used as a conflict hint before applying: if the file already changed
        after the session's baseline (e.g. by a previous apply), applying an
        older snapshot may clobber newer work.

        Args:
            rel: Relative file path.

        Returns:
            True when git reports the file as modified vs HEAD.
        """
        code, _ = self._git("diff", "--quiet", "HEAD", "--", rel)
        return code != 0

    def _load(self) -> None:
        try:
            data = json.loads(self.pending_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._pending = {
                    k: v for k, v in data.items() if isinstance(v, dict)
                }
        except (OSError, json.JSONDecodeError):
            self._pending = {}

    def register_orphan_sessions(self) -> list[str]:
        """Register staging sessions that have files but no pending record.

        Sessions whose AI wrote files but never called ``submit_improvement``
        (e.g. after a crash or a missed submission) are registered so the
        admin can review them. Called once at startup.

        Returns:
            List of newly registered pending ids.
        """
        registered: list[str] = []
        known = {p.get("session") for p in self._pending.values()}
        if not self.staging_dir.is_dir():
            return registered
        sessions = [
            d
            for d in self.staging_dir.iterdir()
            if d.is_dir() and d.name not in known
        ]
        # 只登记最新一个有文件的孤儿 session：更早的是失败残留，
        # 全部登记会制造互相冲突的 pending。
        sessions.sort(
            key=lambda d: d.stat().st_mtime, reverse=True
        )
        for session_dir in sessions:
            files = sorted(
                p.name
                for p in session_dir.iterdir()
                if p.is_file()
                and p.name != "chat_samples.txt"
                and _is_legit_source_file(self.source_dir, p.name)
            )
            if not files:
                continue
            pid = self.submit(
                session_dir.name,
                "（孤儿 staging：AI 未提交，启动时自动登记）",
                files,
            )
            if pid:
                registered.append(pid)
            break
        return registered

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

        Files that are not legitimate source files (probes, temp files,
        path traversal) are filtered out; if nothing remains, no pending
        record is created.

        Args:
            session_id: Staging root folder name.
            note: Model's explanation of the change.
            files: Relative file paths changed within the session.

        Returns:
            The new pending id, or an empty string when nothing valid.
        """
        valid = [
            rel
            for rel in files
            if _is_legit_source_file(self.source_dir, rel)
        ]
        if not valid:
            return ""
        pid = uuid.uuid4().hex[:10]
        self._pending[pid] = {
            "id": pid,
            "session": session_id,
            "note": note,
            "files": valid,
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
        """Render the diff of a pending improvement against its git baseline.

        The baseline is the source as committed when the session started
        (``HEAD`` at session time). Staging copies are compared to that
        version, so the diff shows exactly what the AI changed even if the
        live source has since moved on. Falls back to comparing against the
        live source when git is unavailable.

        Args:
            pid: Pending id.

        Returns:
            The diff text, or a message when missing.
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
            if not staged.is_file():
                continue
            new_text = staged.read_text(encoding="utf-8", errors="replace")
            code, out = self._git("show", f"HEAD:{rel}")
            old_text = out if code == 0 else ""
            if code != 0:
                # 基线里没有该文件（可能是源码目录无 git 或新文件）。
                original = self.source_dir / rel
                if original.is_file():
                    old_text = original.read_text(
                        encoding="utf-8", errors="replace"
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

        A file can only be applied from its newest pending record: if another
        pending (newer than this one) touches the same file, applying this
        record would overwrite newer work with an older snapshot, so it is
        rejected with a hint.

        Args:
            pid: Pending id.

        Returns:
            ``(ok, message)``.
        """
        pending = self._pending.get(pid)
        if not pending:
            return False, f"未找到审批 {pid}"
        pending_files = set(pending.get("files", []))
        newer_conflicts = []
        for other in self._pending.values():
            if other["id"] == pid:
                continue
            if other.get("created_at", 0) > pending.get("created_at", 0):
                overlap = pending_files & set(other.get("files", []))
                if overlap:
                    newer_conflicts.append((other["id"], sorted(overlap)))
        if newer_conflicts:
            detail = "；".join(
                f"{oid}（文件: {', '.join(files)}）" for oid, files in newer_conflicts
            )
            return (
                False,
                f"存在更新的待审批改动包含相同文件，先处理它们以避免覆盖：{detail}",
            )
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
        # 应用结果提交进 git 历史，保证每次变更可审计、可回滚。
        self.commit_apply(pid, str(pending.get("note", "")))
        # 清理 session 目录，避免重载后 register_orphan_sessions 重复登记。
        shutil.rmtree(session, ignore_errors=True)
        return True, f"已应用 {len(applied)} 个文件: {', '.join(applied)}"

    def reject(self, pid: str) -> bool:
        """Reject (remove) a pending improvement.

        Args:
            pid: Pending id.

        Returns:
            True when removed.
        """
        pending = self._pending.get(pid)
        if not pending:
            return False
        session = self.staging_dir / str(pending.get("session", ""))
        self._pending.pop(pid, None)
        self._save()
        shutil.rmtree(session, ignore_errors=True)
        return True


def _is_legit_source_file(source_dir: Path, rel: str) -> bool:
    """Whether a relative path is a legitimate plugin source file.

    Existing source files are always fine. New files must be plausible
    plugin files (python modules, schema, metadata) and must not look like
    temporary probes (``ruff_probe``, ``test_`` probes, ``tmp_`` etc.).

    Args:
        source_dir: Plugin source directory.
        rel: Relative file path.

    Returns:
        True when the file is legitimate.
    """
    name = Path(str(rel)).name
    if not str(rel).strip() or ".." in str(rel):
        return False
    if (source_dir / rel).is_file():
        return True
    lowered = name.lower()
    if lowered.startswith(("ruff_", "probe", "test_", "tmp_", "._")):
        return False
    if name.endswith((".py", ".json", ".yaml", ".md", ".txt")):
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
