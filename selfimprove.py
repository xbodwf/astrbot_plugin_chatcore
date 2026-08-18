"""Self-improvement: the AI improves its own source in a git-backed workdir.

The plugin keeps a *work directory* (under plugin data) that mirrors the
plugin source. The improvement AI reads and writes that workdir directly; a
git repository there records every change it makes, so successive sessions
build on each other without conflicts (a later commit contains the earlier
ones). Approving a commit applies it onto the live plugin directory with
``git diff <base>..<commit> | git apply``, then the plugin reloads.

The workdir is re-synced from the live plugin source on every plugin load /
update, so it always starts from the currently running code.
"""

from __future__ import annotations

import json
import re
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
    "工作目录：你当前的工作区就是插件源码，直接在其中读取和修改。\n"
    "工作流程：\n"
    "1. 先 list_source 了解结构，用 bash_source 的 grep/ls 快速定位，"
    "**不要从头到尾盲读文件**——只读与目标相关的部分（read_source 支持 "
    "offset/limit 分片）；\n"
    "2. 找出问题后用 edit_staging 做精确片段替换（推荐，只传改动部分即可），"
    "追加或新建文件用 add_staging，只有整体重写才用 write_staging；\n"
    "3. 每改完一个文件用 run_ruff 校验，不通过就修正；\n"
    "4. 全部改完且 ruff 通过后，调用 submit_improvement 提交（note 说明改动）。"
    "提交后你的改动会成为一个待审批的 commit，管理员审批后才会应用到线上；\n"
    "5. 当你认为已经完成（提交了改进，或确认无需改动）时，调用 stop_improvement "
    "主动结束会话。\n"
    "重要约束：\n"
    "- 你**最多只有 12 轮工具调用**，不要浪费在无休止的探索上；"
    "尽早定位、尽早动手。\n"
    "- 不要只分析不行动。哪怕只是小改进（注释、边界处理、提示词措辞），"
    "也至少完成一项修改并提交；只有在你确认代码完全无需改动时才可以不提交。\n"
    "- 定位问题时优先 bash_source 的 grep（一次能搜全项目），"
    "而非逐个文件 read_source。"
)

# 允许出现在源码工作区的文件（与插件源码一致）。
_LEGIT_NAMES = (
    ".py", ".json", ".yaml", ".yml", ".md", ".txt", ".gitignore",
    ".gitattributes", "requirements.txt", "metadata.yaml", "LICENSE",
)


def _is_legit_source_file(rel: str) -> bool:
    """Whether a relative path is a legitimate plugin source file.

    Args:
        rel: Relative file path.

    Returns:
        True when the file belongs to the plugin source tree.
    """
    name = Path(str(rel)).name
    if not str(rel).strip() or ".." in str(rel):
        return False
    lowered = name.lower()
    if lowered.startswith(("ruff_", "probe", "test_", "tmp_", "._", ".git")):
        return False
    return name.endswith(_LEGIT_NAMES)


class SelfImprove:
    """Git-backed self-improvement workspace.

    Args:
        source_dir: Live plugin source directory.
        work_dir: Workspace directory that mirrors the plugin source and
            holds the improvement git repository.
    """

    def __init__(self, source_dir: str | Path, work_dir: str | Path) -> None:
        self.source_dir = Path(source_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.work_dir / "pending.json"
        self.summary_path = self.work_dir / "improve_summaries.json"
        self._pending: dict[str, dict] = {}
        self._summaries: list[dict] = []
        self._sync_head: str = ""
        self._load()
        self._git_ensure()

    # ---- git helpers ----

    def _git(self, *args: str, cwd: Path | None = None) -> tuple[int, str]:
        """Run a git command inside the workspace.

        Args:
            *args: Git arguments.
            cwd: Working directory override (defaults to work_dir).

        Returns:
            ``(returncode, combined_output)``.
        """
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(cwd or self.work_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except Exception as e:
            return 1, str(e)

    def _git_ensure(self) -> None:
        """Ensure the workspace is a git repository with an initial commit.

        Sets repo-local author identity if the environment lacks one, so
        commits never fail on machines without global git config.

        Returns:
            None.
        """
        if not (self.work_dir / ".git").exists():
            self._git("init", "-q")
        self._git("config", "user.name", "ChatCore")
        self._git("config", "user.email", "chatcore@local")
        self._git("add", "-A")
        self._git(
            "commit",
            "-q",
            "-m",
            f"chatcore workspace baseline {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "--allow-empty",
        )

    # ---- workspace sync ----

    def sync_from_source(self) -> None:
        """Copy the live plugin source into the workspace and commit.

        Called on plugin load and after plugin updates, so the workspace
        always starts from the currently running code. Existing workspace
        files are overwritten; the pre-sync state is committed first so no
        history is lost.

        Returns:
            None.
        """
        # 先把当前工作区状态留痕（保留未审批的改动在历史里）。
        self._git("add", "-A")
        self._git(
            "commit",
            "-q",
            "-m",
            f"chatcore pre-sync {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "--allow-empty",
        )
        for item in self.source_dir.iterdir():
            if item.name in (".git", "__pycache__"):
                continue
            dest = self.work_dir / item.name
            if item.is_dir():
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(item, dest)
        # 清理工作区里已不存在的文件（保持与插件目录一致）。
        for item in self.work_dir.iterdir():
            if item.name in (".git", "pending.json", "improve_summaries.json"):
                continue
            if not (self.source_dir / item.name).exists():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        self._git("add", "-A")
        self._git(
            "commit",
            "-q",
            "-m",
            f"chatcore sync from source {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "--allow-empty",
        )
        code, out = self._git("rev-parse", "HEAD")
        self._sync_head = out.strip() if code == 0 else ""

    # ---- session / commits ----

    def new_session(self) -> str:
        """Start a new improvement session: record the current baseline.

        Returns:
            The current HEAD commit hash (baseline for this session).
        """
        self._git("add", "-A")
        self._git(
            "commit",
            "-q",
            "-m",
            f"chatcore session baseline {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "--allow-empty",
        )
        code, out = self._git("rev-parse", "HEAD")
        return out.strip() if code == 0 else ""

    def commit_session(self, note: str) -> str:
        """Commit the AI's current workspace changes as a pending improvement.

        The pending's baseline is the *sync baseline* (the live code as of the
        last ``sync_from_source``), so approving it applies the cumulative
        diff from the online code to this commit — a later improvement
        already contains the earlier ones.

        Args:
            note: The AI's explanation of the change.

        Returns:
            The commit hash (or "" when nothing changed).
        """
        base = self._sync_head or ""
        self._git("add", "-A")
        code, _ = self._git(
            "commit",
            "-q",
            "-m",
            f"chatcore improve: {note[:80]}",
            "--allow-empty",
        )
        if code != 0:
            return ""
        code, out = self._git("rev-parse", "HEAD")
        commit = out.strip() if code == 0 else ""
        if commit:
            pid = uuid.uuid4().hex[:10]
            self._pending[pid] = {
                "id": pid,
                "commit": commit,
                "base": base,
                "note": note,
                "created_at": time.time(),
            }
            self._save()
        return commit

    def list_pending(self) -> list[dict]:
        """List pending improvements (newest first)."""
        return sorted(
            self._pending.values(),
            key=lambda p: p.get("created_at", 0),
            reverse=True,
        )

    def get(self, pid: str) -> dict | None:
        """Get one pending improvement by its id.

        Args:
            pid: Pending id (or full/short commit hash).

        Returns:
            The pending dict, or None.
        """
        if pid in self._pending:
            return self._pending[pid]
        for p in self._pending.values():
            commit = str(p.get("commit") or "")
            if commit.startswith(pid):
                return p
        return None

    def diff(self, pid: str) -> str:
        """Render the diff of a pending commit against its parent.

        Args:
            pid: Pending id (or commit hash).

        Returns:
            The unified diff text, or a message when missing.
        """
        pending = self.get(pid)
        if not pending:
            return f"未找到审批 {pid}"
        commit = str(pending.get("commit") or "")
        if not commit:
            return "缺少 commit 信息"
        code, out = self._git("show", "--stat", "--format=fuller", commit)
        if code != 0:
            return f"commit 不可用: {out}"
        code2, diff = self._git("show", "--format=", commit)
        return (out + "\n" + diff) if code2 == 0 else out

    def apply(self, pid: str) -> tuple[bool, str]:
        """Apply a pending commit onto the live plugin source.

        Uses ``git diff <commit>^ <commit>`` (or the cumulative diff from the
        baseline when the commit is a session's latest) and applies it with
        ``git apply`` in the plugin directory.

        Args:
            pid: Pending id (or commit hash).

        Returns:
            ``(ok, message)``.
        """
        pending = self.get(pid)
        if not pending:
            return False, f"未找到审批 {pid}"
        commit = str(pending.get("commit") or "")
        if not commit:
            return False, "缺少 commit 信息"
        # 生成从基线到该 commit 的累积 diff：包含此前所有改进。
        base = str(pending.get("base") or f"{commit}^")
        code, diff = self._git("diff", base, commit)
        if code != 0 or not diff.strip():
            return False, f"无法生成 diff: {diff[:200]}"
        # _git 会 strip 输出，补回 diff 末尾换行，否则 git apply 报损坏。
        if not diff.endswith("\n"):
            diff += "\n"
        # 应用到插件目录。
        try:
            proc = subprocess.run(
                ["git", "apply", "-"],
                cwd=str(self.source_dir),
                input=diff,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            return False, "git 不可用"
        if proc.returncode != 0:
            return False, f"应用失败: {(proc.stderr or proc.stdout)[:300]}"
        # 删除不再存在的文件（diff 里的删除操作 git apply 已处理）。
        self._pending.pop(pending["id"], None)
        self._save()
        return True, f"已应用 {pending['id']} ({commit[:10]}): {pending.get('note', '')[:60]}"

    def reject(self, pid: str) -> bool:
        """Reject (remove) a pending improvement.

        Args:
            pid: Pending id.

        Returns:
            True when removed.
        """
        pending = self.get(pid)
        if not pending:
            return False
        self._pending.pop(pending["id"], None)
        self._save()
        return True

    # ---- summaries (chat-shared reflections) ----

    def add_session_summary(self, summary: str) -> None:
        """Store a self-improve session summary for later chat injection.

        Args:
            summary: The detailed session summary text.
        """
        if not (summary or "").strip():
            return
        self._summaries.append({"ts": time.time(), "text": summary.strip()})
        self._summaries = self._summaries[-5:]
        try:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.summary_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._summaries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.summary_path)
        except OSError:
            pass

    def latest_session_summary(self) -> str:
        """The most recent session summary, if any.

        Returns:
            The summary text, or "".
        """
        if self._summaries:
            return self._summaries[-1]["text"]
        return ""

    # ---- persistence ----

    def _save(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.pending_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._pending, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.pending_path)

    def _load(self) -> None:
        try:
            data = json.loads(self.pending_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._pending = {
                    k: v for k, v in data.items() if isinstance(v, dict)
                }
        except (OSError, json.JSONDecodeError):
            self._pending = {}
        try:
            summaries = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if isinstance(summaries, list):
                self._summaries = [
                    s for s in summaries if isinstance(s, dict) and s.get("text")
                ][-5:]
        except (OSError, json.JSONDecodeError):
            self._summaries = []
