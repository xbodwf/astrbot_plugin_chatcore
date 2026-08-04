"""PluginPage WebUI backend (AstrBot Dashboard Pages).

Registers REST routes under ``/astrbot_plugin_chatcore/<route>`` through
``context.register_web_api``. The frontend lives in ``pages/dashboard/`` and
talks to these routes through the ``AstrBotPluginPage`` bridge (the bridge
endpoint omits the plugin-name prefix, e.g. ``stats`` maps to
``/astrbot_plugin_chatcore/stats``).

Two kinds of pages are served: management (profiles / memories / emoji / styles)
and live monitoring (attention, context windows, emotion, backoff cooldowns).
"""

import base64
import mimetypes
import time
from pathlib import Path

from astrbot.api.web import error_response, file_response, json_response, request
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .emoji import classify_emoji

PLUGIN_NAME = "astrbot_plugin_chatcore"
_PREFIX = f"/{PLUGIN_NAME}"


class ChatCoreWebUI:
    """Registers and serves ChatCore's Dashboard Pages API.

    Args:
        plugin: The plugin instance, exposing the stores and managers.
    """

    def __init__(self, plugin) -> None:
        self.plugin = plugin

    def register_routes(self) -> None:
        """Register all Web API routes on the plugin context."""
        register = self.plugin.context.register_web_api
        routes = [
            (f"{_PREFIX}/stats", self.get_stats, ["GET"], "ChatCore overview stats"),
            (f"{_PREFIX}/profiles", self.list_profiles, ["GET"], "ChatCore profiles"),
            (
                f"{_PREFIX}/profiles/delete",
                self.delete_profile,
                ["POST"],
                "ChatCore delete profile",
            ),
            (f"{_PREFIX}/emojis", self.list_emojis, ["GET"], "ChatCore emoji list"),
            (
                f"{_PREFIX}/emojis/delete",
                self.delete_emoji,
                ["POST"],
                "ChatCore delete emoji",
            ),
            (
                f"{_PREFIX}/emojis/update",
                self.update_emoji,
                ["POST"],
                "ChatCore update emoji meta",
            ),
            (
                f"{_PREFIX}/emojis/import",
                self.import_emoji,
                ["POST"],
                "ChatCore import emoji",
            ),
            (
                f"{_PREFIX}/emojis/<emoji_id>/image",
                self.get_emoji_image,
                ["GET"],
                "ChatCore emoji image",
            ),
            (
                f"{_PREFIX}/emojis/<emoji_id>/image/data",
                self.get_emoji_image_data,
                ["GET"],
                "ChatCore emoji image as base64",
            ),
            (f"{_PREFIX}/memories", self.list_memories, ["GET"], "ChatCore memories"),
            (
                f"{_PREFIX}/memories/delete",
                self.delete_memory,
                ["POST"],
                "ChatCore delete memory",
            ),
            (
                f"{_PREFIX}/expressions",
                self.list_expressions,
                ["GET"],
                "ChatCore expression styles",
            ),
            (
                f"{_PREFIX}/expressions/delete",
                self.delete_expression,
                ["POST"],
                "ChatCore delete expression style",
            ),
            (
                f"{_PREFIX}/expressions/add",
                self.add_expression,
                ["POST"],
                "ChatCore add expression entry",
            ),
            (
                f"{_PREFIX}/expressions/update",
                self.update_expression,
                ["POST"],
                "ChatCore update expression entry",
            ),
        ]
        for route, handler, methods, desc in routes:
            register(route, handler, methods, desc)

    async def get_stats(self) -> dict:
        """Overview of counts plus live monitoring state.

        Returns:
            A stats payload for the monitoring tab.
        """
        plugin = self.plugin
        emoji_count = plugin.emoji_store.count() if plugin.emoji_store else 0
        return json_response(
            {
                "profile_count": plugin.profile_store.count()
                if plugin.profile_store
                else 0,
                "memory_count": plugin.memory.count() if plugin.memory else 0,
                "emoji_count": emoji_count,
                "expression_count": plugin.expression_store.count()
                if plugin.expression_store
                else 0,
                "recalls_cancelled": plugin.recalls_cancelled,
                "attention": plugin.attention.snapshot() if plugin.attention else [],
                "context": plugin.context_mgr.conversation_stats()
                if plugin.context_mgr
                else [],
                "emotion": plugin.emotion_mgr.snapshot() if plugin.emotion_mgr else [],
                "affinity": plugin.affinity_mgr.snapshot()
                if getattr(plugin, "affinity_mgr", None)
                else [],
            }
        )

    async def list_profiles(self) -> dict:
        """List all person profiles.

        Returns:
            A profiles payload.
        """
        store = self.plugin.profile_store
        if not store:
            return json_response({"profiles": []})
        return json_response({"profiles": store.all()})

    async def delete_profile(self) -> dict:
        """Delete a person profile.

        Returns:
            Success or error payload.
        """
        payload = await request.json(default={})
        person_id = payload.get("person_id")
        store = self.plugin.profile_store
        if not store or not person_id:
            return error_response("person_id is required")
        if not store.delete(str(person_id)):
            return error_response("profile not found", status_code=404)
        return json_response({"deleted": person_id})

    async def list_emojis(self) -> dict:
        """List all emoji records with provenance.

        Returns:
            An emoji payload.
        """
        store = self.plugin.emoji_store
        if not store:
            return json_response({"emojis": []})
        return json_response({"emojis": store.all()})

    async def delete_emoji(self) -> dict:
        """Delete an emoji and its stored image file.

        Returns:
            Success or error payload.
        """
        payload = await request.json(default={})
        emoji_id = payload.get("emoji_id")
        store = self.plugin.emoji_store
        if not store or not emoji_id:
            return error_response("emoji_id is required")
        if not store.delete(str(emoji_id), remove_file=True):
            return error_response("emoji not found", status_code=404)
        return json_response({"deleted": emoji_id})

    async def update_emoji(self) -> dict:
        """Update an emoji's category and tags.

        Returns:
            Success or error payload.
        """
        payload = await request.json(default={})
        emoji_id = payload.get("emoji_id")
        store = self.plugin.emoji_store
        if not store or not emoji_id:
            return error_response("emoji_id is required")
        if not store.get(str(emoji_id)):
            return error_response("emoji not found", status_code=404)
        category = str(payload.get("category", "")).strip()
        tags = payload.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        await store.set_meta(str(emoji_id), category, [str(t) for t in tags])
        return json_response({"updated": emoji_id})

    async def import_emoji(self) -> dict:
        """Import an uploaded image into the emoji library.

        The uploaded file is saved into the store, then classified by the
        vision model (source context is empty for manual imports).

        Returns:
            A payload with the imported emoji id.
        """
        store = self.plugin.emoji_store
        if not store:
            return error_response("emoji store disabled", status_code=404)
        files = await request.files()
        upload = files.get("file")
        if not upload:
            return error_response("file is required")
        tmp_dir = (
            Path(get_astrbot_plugin_data_path())
            / "astrbot_plugin_chatcore"
            / "emoji"
            / "tmp"
        )
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(upload.filename or "img").suffix or ".image"
        tmp = tmp_dir / f"import_{int(time.time() * 1000)}{ext}"
        try:
            await upload.save(tmp)
            emoji_id = store.collect(
                tmp,
                source_group="手动导入",
                source_sender="admin",
                source_message_id="",
                source_text="",
                source_context="",
            )
            if emoji_id:
                stored = store.file_path(emoji_id)
                plugin = self.plugin
                vision = plugin.emoji_vision_client or plugin.vision_client
                category, tags = await classify_emoji(
                    vision,
                    plugin.summary_client,
                    stored or str(tmp),
                    "",
                )
                if category or tags:
                    await store.set_meta(emoji_id, category, tags)
        except Exception as e:
            return error_response(f"import failed: {e}", status_code=500)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        if not emoji_id:
            return error_response("import failed", status_code=500)
        return json_response({"imported": emoji_id})

    async def get_emoji_image(self, emoji_id: str) -> dict:
        """Serve an emoji's stored image file.

        Args:
            emoji_id: Route parameter, the emoji id.

        Returns:
            A file response, or an error payload.
        """
        store = self.plugin.emoji_store
        if not store:
            return error_response("emoji store disabled", status_code=404)
        path = store.file_path(emoji_id)
        if not path or not Path(path).is_file():
            return error_response("emoji image not found", status_code=404)
        return file_response(path)

    async def get_emoji_image_data(self, emoji_id: str) -> dict:
        """Serve an emoji image as a base64 data URI for inline preview.

        Args:
            emoji_id: Route parameter, the emoji id.

        Returns:
            A JSON payload with a ``data`` data-URI string.
        """
        store = self.plugin.emoji_store
        if not store:
            return error_response("emoji store disabled", status_code=404)
        path = store.file_path(emoji_id)
        if not path or not Path(path).is_file():
            return error_response("emoji image not found", status_code=404)
        try:
            raw = Path(path).read_bytes()
        except OSError:
            return error_response("emoji image unreadable", status_code=500)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        return json_response({"data": data_uri})

    async def list_memories(self) -> dict:
        """List memory entries, optionally filtered by session tag.

        Returns:
            A memories payload.
        """
        memory = self.plugin.memory
        if not memory:
            return json_response({"memories": []})
        entries = memory.list_entries()
        group_id = str(request.query.get("group_id", "") or "").strip()
        if group_id:
            entries = [
                e for e in entries if group_id in [str(t) for t in e.get("tags", [])]
            ]
        return json_response({"memories": entries})

    async def delete_memory(self) -> dict:
        """Delete a memory entry by list index.

        Returns:
            Success or error payload.
        """
        payload = await request.json(default={})
        index = payload.get("index")
        memory = self.plugin.memory
        if not memory or not isinstance(index, int):
            return error_response("index is required")
        if not memory.delete_entry(index):
            return error_response("memory not found", status_code=404)
        return json_response({"deleted": index})

    async def list_expressions(self) -> dict:
        """List all learned expression styles.

        Returns:
            An expressions payload.
        """
        store = self.plugin.expression_store
        if not store:
            return json_response({"expressions": []})
        return json_response({"expressions": store.all()})

    async def delete_expression(self) -> dict:
        """Delete a group's learned expression style.

        Returns:
            Success or error payload.
        """
        payload = await request.json(default={})
        group_id = payload.get("group_id")
        store = self.plugin.expression_store
        if not store or not group_id:
            return error_response("group_id is required")
        if not store.delete(str(group_id)):
            return error_response("style not found", status_code=404)
        return json_response({"deleted": group_id})

    async def add_expression(self) -> dict:
        """Add a manually curated pattern or jargon entry."""
        payload = await request.json(default={})
        store = self.plugin.expression_store
        group_id = str(payload.get("group_id", "")).strip()
        if not store or not group_id:
            return error_response("group_id is required")
        if not store.add_entry(
            group_id,
            pattern=str(payload.get("pattern", "")),
            term=str(payload.get("term", "")),
            guessed_meaning=str(payload.get("guessedMeaning", "")),
            example=str(payload.get("example", "")),
            source=str(payload.get("source", "manual")),
        ):
            return error_response("pattern or complete jargon fields are required")
        return json_response({"added": group_id})

    async def update_expression(self) -> dict:
        """Update or disable one manually learned jargon entry."""
        payload = await request.json(default={})
        store = self.plugin.expression_store
        group_id = str(payload.get("group_id", "")).strip()
        term = str(payload.get("term", "")).strip()
        if not store or not group_id or not term:
            return error_response("group_id and term are required")
        updates = {
            key: payload[key]
            for key in ("guessedMeaning", "example", "source", "enabled")
            if key in payload
        }
        if not store.update_jargon(group_id, term, **updates):
            return error_response("expression entry not found", status_code=404)
        return json_response({"updated": term})
