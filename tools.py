"""Sandboxed agent tools: files, bash, fetch, forward records.

All file tools operate inside a chroot (the plugin data directory by
default). Paths are resolved and validated so ``../`` escapes are rejected.
Handlers follow AstrBot's ``@filter.llm_tool`` signature ``(event, **kwargs)``
so ``FunctionToolExecutor`` can call them directly.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from pathlib import Path

_FETCH_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")
_FETCH_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB
_BASH_DEFAULT_TIMEOUT = 30.0
_MAX_FILE_BYTES = 512 * 1024  # 512 KiB per file for read/write


def _resolve_chroot_path(chroot: str | Path, target: str) -> Path:
    """Resolve ``target`` inside ``chroot``, rejecting escapes.

    Args:
        chroot: The sandbox root directory.
        target: The requested path (absolute or relative).

    Returns:
        The absolute path inside the chroot.

    Raises:
        ValueError: When the path escapes the chroot.
    """
    root = Path(chroot).resolve()
    raw = Path(str(target or ""))
    if not raw.is_absolute():
        raw = root / raw
    resolved = raw.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"路径 {target} 越过了沙箱目录，禁止访问")
    return resolved


class SandboxTools:
    """Filesystem / bash / fetch tools confined to a sandbox root.

    Reads are allowed anywhere inside ``chroot`` (the AstrBot project root by
    default, so the model can inspect framework/plugin code). Writes are
    confined to ``write_root`` (the plugin's staging directory) — every write
    path is resolved against it and rejected otherwise.

    Args:
        chroot: Read-only sandbox root.
        write_root: The only directory where writes are permitted.
        bash_timeout: Default timeout for bash commands, in seconds.
        fetch_max_bytes: Maximum response body size for fetch, in bytes.
    """

    def __init__(
        self,
        chroot: str | Path,
        write_root: str | Path | None = None,
        bash_timeout: float = _BASH_DEFAULT_TIMEOUT,
        fetch_max_bytes: int = _FETCH_MAX_BYTES,
    ) -> None:
        self.chroot = Path(chroot)
        self.chroot.mkdir(parents=True, exist_ok=True)
        self.write_root = (
            Path(write_root) if write_root is not None else self.chroot
        )
        self.write_root.mkdir(parents=True, exist_ok=True)
        self.bash_timeout = max(1.0, float(bash_timeout))
        self.fetch_max_bytes = max(1024, int(fetch_max_bytes))

    def _resolve_write_path(self, target: str) -> Path:
        """Resolve a write target inside ``write_root``, rejecting escapes.

        Args:
            target: The requested path (absolute or relative to write_root).

        Returns:
            The absolute path inside the write root.

        Raises:
            ValueError: When the path escapes the write root.
        """
        return _resolve_chroot_path(self.write_root, target)

    async def read_files(
        self,
        event,
        path: str,
        offset: int = 0,
        limit: int = 0,
        max_bytes: int = 0,
    ) -> dict:
        """读取沙箱内的文件内容（文本）。

        Args:
            event: The message event.
            path: 文件路径（相对沙箱根或绝对路径）。
            offset: 起始行号（从 0 开始），只读该行起的部分。
            limit: 最多读取的行数，0 表示读到末尾。
            max_bytes: 最多读取的字符数，0 表示使用默认上限。

        Returns:
            dict: ``{"content": ..., "lines": N}`` 或 ``{"error": ...}``。
        """
        try:
            target = _resolve_chroot_path(self.chroot, path)
        except ValueError as e:
            return {"error": str(e)}
        if not target.is_file():
            return {"error": f"文件不存在: {path}"}
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return {"error": f"读取失败: {e}"}
        start = max(0, int(offset or 0))
        end = None if not limit else start + max(0, int(limit))
        selected = lines[start:end]
        text = "\n".join(selected)
        limit_chars = max_bytes or _MAX_FILE_BYTES
        if len(text) > limit_chars:
            text = text[:limit_chars]
        return {
            "content": text,
            "lines": len(selected),
            "offset": start,
            "total_lines": len(lines),
        }

    async def list_files(self, event, path: str = ".") -> dict:
        """列出沙箱内目录的内容。

        Args:
            event: The message event.
            path: 目录路径，默认沙箱根。

        Returns:
            dict: ``{"entries": [...]}`` 或 ``{"error": ...}``。
        """
        try:
            target = _resolve_chroot_path(self.chroot, path)
        except ValueError as e:
            return {"error": str(e)}
        if not target.is_dir():
            return {"error": f"目录不存在: {path}"}
        entries = []
        for child in sorted(target.iterdir()):
            try:
                rel = child.relative_to(self.chroot)
            except ValueError:
                rel = child
            entries.append(
                {
                    "name": str(rel),
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else 0,
                }
            )
        return {"entries": entries}

    async def write_files(self, event, path: str, content: str) -> dict:
        """写入（覆盖）沙箱内的文件。

        Args:
            event: The message event.
            path: 目标文件路径。
            content: 要写入的内容。

        Returns:
            dict: ``{"ok": True}`` 或 ``{"error": ...}``。
        """
        try:
            target = self._resolve_write_path(path)
        except ValueError as e:
            return {"error": str(e)}
        if len(content or "") > _MAX_FILE_BYTES:
            return {"error": f"内容超过上限 {_MAX_FILE_BYTES} 字符"}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
            return {"ok": True, "path": str(target.relative_to(self.write_root))}
        except OSError as e:
            return {"error": f"写入失败: {e}"}

    async def edit_files(
        self, event, path: str, old: str, new: str
    ) -> dict:
        """替换文件中的一段文本（单一替换）。

        Args:
            event: The message event.
            path: 目标文件路径。
            old: 要被替换的原文（必须精确匹配一次）。
            new: 替换后的文本。

        Returns:
            dict: ``{"ok": True}`` 或 ``{"error": ...}``。
        """
        try:
            target = self._resolve_write_path(path)
        except ValueError as e:
            return {"error": str(e)}
        if not target.is_file():
            return {"error": f"文件不存在: {path}"}
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as e:
            return {"error": f"读取失败: {e}"}
        count = text.count(old)
        if count != 1:
            return {"error": f"old 文本在文件中出现 {count} 次，要求恰好 1 次"}
        target.write_text(text.replace(old, new), encoding="utf-8")
        return {"ok": True, "path": str(target.relative_to(self.write_root))}

    async def bash(self, event, command: str, timeout: float = 0) -> dict:
        """在沙箱根目录执行终端命令。

        Args:
            event: The message event.
            command: 要执行的 shell 命令。
            timeout: 超时秒数，0 使用默认值。

        Returns:
            dict: ``{"stdout": ..., "stderr": ..., "code": ...}`` 或 ``{"error": ...}``。
        """
        if not (command or "").strip():
            return {"error": "命令为空"}
        limit = self.bash_timeout if not timeout else max(0.1, float(timeout))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.chroot),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            return {"error": f"启动命令失败: {e}"}
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=limit
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"error": f"命令超时（>{limit:.0f}s），已终止"}
        return {
            "stdout": stdout.decode("utf-8", errors="replace")[:4000],
            "stderr": stderr.decode("utf-8", errors="replace")[:2000],
            "code": proc.returncode,
        }

    async def fetch(
        self,
        event,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        body: str = "",
        timeout: float = 15.0,
    ) -> dict:
        """发起 HTTP 请求（curl 风格），抓取远程内容。

        Args:
            event: The message event.
            url: 完整 URL。
            method: HTTP 方法，仅允许 GET/POST/PUT/PATCH/DELETE/HEAD。
            headers: 自定义请求头（dict）。
            body: 请求体（对 GET/HEAD 忽略）。
            timeout: 超时秒数。

        Returns:
            dict: ``{"status": ..., "headers": ..., "body": ...}`` 或 ``{"error": ...}``。
        """
        import aiohttp

        if not (url or "").strip():
            return {"error": "URL 为空"}
        if not str(url).startswith(("http://", "https://")):
            return {"error": "仅支持 http/https URL"}
        method = (method or "GET").upper()
        if method not in _FETCH_METHODS:
            return {"error": f"不支持的方法 {method}，允许: {', '.join(_FETCH_METHODS)}"}
        headers = headers or {}
        if not isinstance(headers, dict):
            return {"error": "headers 必须是对象"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    data=body if method not in ("GET", "HEAD") else None,
                    timeout=aiohttp.ClientTimeout(total=max(1.0, float(timeout))),
                ) as resp:
                    data = await resp.content.read(self.fetch_max_bytes + 1)
                    truncated = len(data) > self.fetch_max_bytes
                    return {
                        "status": resp.status,
                        "headers": {k: v for k, v in resp.headers.items()},
                        "body": data[: self.fetch_max_bytes].decode(
                            "utf-8", errors="replace"
                        ),
                        "truncated": truncated,
                    }
        except asyncio.TimeoutError:
            return {"error": f"请求超时（>{timeout}s）"}
        except Exception as e:
            return {"error": f"请求失败: {e}"}

    async def screenshot_bash(
        self, event, command: str, timeout: float = 0
    ) -> dict:
        """执行终端命令并将输出渲染为 PNG 图片（ANSI 风格）。

        Args:
            event: The message event.
            command: 要执行的命令。
            timeout: 超时秒数，0 使用默认值。

        Returns:
            dict: ``{"image": "<path>"}`` 或 ``{"error": ...}``。
        """
        result = await self.bash(event, command, timeout)
        if "error" in result:
            return result
        text = (result.get("stdout") or "") + (result.get("stderr") or "")
        text = text or "(无输出)"
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = self._render_text_to_image(text)
            out_dir = self.chroot / "screenshots"
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"term_{int(time.time() * 1000)}.png"
            img.save(dest)
            return {"image": str(dest)}
        except Exception as e:
            return {"error": f"截图渲染失败: {e}"}

    @staticmethod
    def _render_text_to_image(text: str, width: int = 100) -> object:
        """Render plain text to a PIL image with monospace font.

        Args:
            text: The text to render.
            width: Max characters per line.

        Returns:
            A PIL image.
        """
        from PIL import Image, ImageDraw, ImageFont

        lines = text.splitlines() or [""]
        font = ImageFont.load_default()
        char_w = 9
        line_h = 16
        img = Image.new(
            "RGB",
            (max(200, width * char_w), max(40, len(lines) * line_h + 16)),
            (18, 18, 18),
        )
        draw = ImageDraw.Draw(img)
        y = 8
        for line in lines[:400]:
            draw.text((8, y), line[:width], fill=(220, 220, 220), font=font)
            y += line_h
        return img
