"""ChatCore: a better chat core for AstrBot (OneBot V11).

Takes over AstrBot's vanilla chat pipeline: smart context, attention-based
reply probability, AI self-segmentation over streaming, and debounce when the
user sends a new message mid-reply.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.star import Context, Star
from astrbot.core.message.message_event_result import (
    MessageEventResult,
    ResultContentType,
)
from astrbot.core.platform.message_type import MessageType
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .actions import parse_actions
from .attention import AttentionManager
from .context import ContextManager
from .llm import EmbeddingAdapter, LLMProvider
from .memory import MemoryStore
from .segmentation import stream_respond

DEFAULT_IMPLICIT_PROMPT = (
    "你是聊天活跃度分析器。判断群聊最近记录中是否有人隐性提及机器人、"
    "在邀请机器人参与话题、或有机器人值得参与的话题。只回答“是”或“否”。"
)

FALLBACK_SYSTEM_PROMPT = "你是一个友善、自然的聊天机器人，请像真人一样聊天，回复不要机械化。"


class GenerationTask:
    """Tracks an in-flight streaming conversation and debounced messages.

    Args:
        conv_id: Conversation identifier (unified_msg_origin).
        trigger_message_id: Platform id of the message the current round answers.
    """

    def __init__(self, conv_id: str, trigger_message_id: str = "") -> None:
        self.conv_id = conv_id
        self.trigger_message_id = trigger_message_id
        self.cancel_requested = False
        self.suppress_record = False
        self._pending: str | None = None

    def enqueue(self, text: str, message_id: str = "") -> None:
        """Queue a new user message arriving mid-reply.

        The queued message becomes the trigger of the next round.

        Args:
            text: The new message text.
            message_id: Platform id of the new message.
        """
        self._pending = text
        self.trigger_message_id = message_id

    def pop_pending(self) -> str | None:
        """Take the pending message, if any.

        Returns:
            The pending text, or None.
        """
        text = self._pending
        self._pending = None
        return text

    def request_cancel(self) -> None:
        """Ask the running stream to stop after the current segment."""
        self.cancel_requested = True

    def signal(self) -> tuple[str | None, bool] | None:
        """Debounce / cancel signal polled at each segment boundary.

        Returns:
            A ``(pending_text, cancelled)`` tuple when the running stream should
            stop, else None.
        """
        if self.cancel_requested:
            self.cancel_requested = False
            self.suppress_record = True
            return self.pop_pending(), True
        if self._pending:
            return self.pop_pending(), False
        return None


class Main(Star):
    """ChatCore plugin entry point.

    Args:
        context: AstrBot star context.
        config: Plugin config parsed from ``_conf_schema.json``.
    """

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        if not hasattr(self, "logger"):
            # Older AstrBot builds do not inject a plugin logger.
            self.logger = logging.getLogger("astrbot")
        self._config = config
        self._init_from_config(config)
        self.active_tasks: dict[str, GenerationTask] = {}
        self._analysis_task: asyncio.Task | None = None

    def _init_from_config(self, config: AstrBotConfig) -> None:
        """Build all runtime components from the plugin config.

        Args:
            config: Plugin config.
        """
        providers = config.get("providers", {})
        chat_cfg = providers.get("chat", {})
        self.chat_provider_id = chat_cfg.get("provider_id", "")
        self.chat_client = LLMProvider(self.context, self.chat_provider_id)
        self.chat_multimodal = chat_cfg.get("multimodal", False)
        self.markers_enabled = config.get("chat", {}).get("markers_enabled", True)

        vision_cfg = providers.get("vision", {})
        self.vision_client = LLMProvider(
            self.context,
            vision_cfg.get("provider_id", "") or self.chat_provider_id,
        )

        attn = config.get("attention", {})
        self.attention = AttentionManager(
            bubble_base=attn.get("bubble_base_prob", 0.02),
            active_cap=attn.get("active_max_prob", 0.30),
            decay_minutes=attn.get("decay_minutes", 10.0),
            hard_trigger_boost=attn.get("hard_trigger_boost", 0.10),
        )
        self.hard_trigger_force = attn.get("hard_trigger_force", True)
        self.wake_prefix = [str(w).lower() for w in attn.get("wake_prefix", [])]

        ctx = config.get("context", {})
        self.context_mgr = ContextManager(
            recent_count=ctx.get("recent_count", 10),
            history_count=ctx.get("history_count", 30),
            old_msg_chars=ctx.get("old_msg_chars", 40),
        )

        seg = config.get("segment", {})
        self.segment_delimiter = seg.get("delimiter", "\n---\n")
        self.segment_escape = seg.get("escape_char", "\\")
        self.segment_interval = seg.get("interval", 1.0)
        self.max_segment_chars = seg.get("max_segment_chars", 600)

        memory_cfg = config.get("memory", {})
        self.memory_shared = memory_cfg.get("shared_across_groups", True)
        self.memory_top_k = memory_cfg.get("max_recall", 5)
        self.memory = None
        emb_cfg = providers.get("embedding", {})
        if memory_cfg.get("enabled", True) and emb_cfg.get("provider_id"):
            embed_adapter = EmbeddingAdapter(self.context, emb_cfg["provider_id"])
            path = (
                Path(get_astrbot_plugin_data_path())
                / "astrbot_plugin_chatcore"
                / "memory.json"
            )
            self.memory = MemoryStore(embed_adapter.embed, path)

        implicit_cfg = config.get("implicit", {})
        self.implicit_enabled = implicit_cfg.get("enabled", True)
        self.implicit_interval = max(
            1,
            int(implicit_cfg.get("interval_minutes", 30)),
        )
        self.implicit_boost = implicit_cfg.get("prob_boost", 0.05)
        self.implicit_prompt = (
            str(implicit_cfg.get("prompt", "")).strip() or DEFAULT_IMPLICIT_PROMPT
        )
        self.analysis_client = None
        if implicit_cfg.get("provider_id"):
            self.analysis_client = LLMProvider(
                self.context,
                implicit_cfg["provider_id"],
            )

        self.recall_cancel_enabled = config.get("recall_cancel", {}).get(
            "enabled", True
        )
        self.recalls_cancelled = 0

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def on_message(self, event: AstrMessageEvent) -> None:
        """Intercept chat messages and drive the ChatCore pipeline.

        Args:
            event: Current platform message event.
        """
        if event.get_platform_name() != "aiocqhttp":
            return
        chat_cfg = self._config.get("chat", {})
        if not chat_cfg.get("enabled", True):
            return
        if not self.chat_provider_id:
            return

        text = event.get_message_str().strip()
        if text.startswith("/"):
            # Leave plugin commands untouched.
            return

        components = event.get_messages()
        images = [comp for comp in components if isinstance(comp, Image)]
        if not text and not images:
            return

        is_private = event.get_message_type() == MessageType.FRIEND_MESSAGE
        conv_id = event.unified_msg_origin
        msg_id = str(getattr(event.message_obj, "message_id", "") or "")

        self.logger.info(
            f"ChatCore recv | {'私聊' if is_private else '群聊'} | {conv_id}"
            f" | {event.get_sender_name()}: {text or '[图片]'}"
        )

        self.context_mgr.record(
            conv_id,
            "user",
            event.get_sender_name(),
            text or "[图片]",
            sender_id=event.get_sender_id(),
            message_id=msg_id,
        )

        hard = self._is_hard_trigger(event, text)

        should_reply = False
        if is_private:
            should_reply = chat_cfg.get("private_force_reply", True)
        elif chat_cfg.get("group_enabled", True):
            if hard:
                self.attention.record_hard_trigger(conv_id)
                should_reply = self.hard_trigger_force or self.attention.should_respond(conv_id)
            elif self._addresses_other_user(event):
                # Directed at someone else; don't chime in on a soft trigger.
                should_reply = False
            else:
                should_reply = self.attention.should_respond(conv_id)
                if should_reply:
                    self.attention.record_interaction(conv_id)

        if self.memory:
            asyncio.create_task(
                self._remember(conv_id, event.get_sender_name(), text),
            )

        takeover = chat_cfg.get("takeover", True)
        if not should_reply:
            if takeover:
                event.stop_event()
            return

        task = self.active_tasks.get(conv_id)
        if task:
            # Smart debounce: queue the new message; the running stream will
            # finish its current sentence and restart with the new context.
            task.enqueue(text, msg_id)
            event.stop_event()
            return

        task = GenerationTask(conv_id, msg_id)
        self.active_tasks[conv_id] = task
        event.stop_event()
        asyncio.create_task(
            self._run_conversation(task, event, conv_id, text, images),
        )

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_recall_notice(self, event: AstrMessageEvent) -> None:
        """Cancel an in-flight reply when its triggering message is recalled.

        OneBot V11 recall notices arrive as ``notice`` events carrying the
        recalled message id. When that id is the trigger of an active
        generation, the stream is asked to stop after the current segment and
        the recalled message is dropped from the conversation history.

        Args:
            event: The recall notice event.
        """
        raw = getattr(event.message_obj, "raw_message", None)
        notice_type = self._raw_get(raw, "notice_type")
        if notice_type not in ("group_recall", "friend_recall"):
            return
        recalled_msg_id = str(self._raw_get(raw, "message_id") or "")
        if not recalled_msg_id:
            return

        conv_id = event.unified_msg_origin
        cancelled = False
        if self.recall_cancel_enabled:
            task = self.active_tasks.get(conv_id)
            if task and task.trigger_message_id == recalled_msg_id:
                task.request_cancel()
                cancelled = True
                self.recalls_cancelled += 1
                self.logger.info(
                    "ChatCore: recall cancelled in-flight reply "
                    f"| msg_id={recalled_msg_id}"
                )
        self.context_mgr.remove_message(conv_id, recalled_msg_id)
        if cancelled:
            # Leave a marker so the model understands the gap in history.
            self.context_mgr.record(conv_id, "user", "系统", "「已撤回」")

    @staticmethod
    def _raw_get(raw: Any, key: str, default: Any = None) -> Any:
        """Read a field from a OneBot raw message payload.

        The payload is usually an aiocqhttp ``Event`` (dict-like) but may also
        be a plain dict.

        Args:
            raw: The raw message payload.
            key: Field name.
            default: Value returned when the field is absent.

        Returns:
            The field value, or default.
        """
        if raw is None:
            return default
        if isinstance(raw, dict):
            return raw.get(key, default)
        value = getattr(raw, key, None)
        if value is not None:
            return value
        getter = getattr(raw, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                return default
        return default

    def _is_hard_trigger(self, event: AstrMessageEvent, text: str) -> bool:
        """Whether the message directly addresses the bot.

        Args:
            event: Current platform message event.
            text: Plain message text.

        Returns:
            True for @-mention, reply-to-bot or wake prefix.
        """
        self_id = str(event.get_self_id())
        for comp in event.get_messages():
            if isinstance(comp, At) and str(comp.qq) == self_id:
                return True
            if isinstance(comp, Reply):
                if str(getattr(comp, "sender_id", "")) == self_id:
                    return True
                quoted = comp.message_str or ""
                if self_id and self_id in quoted:
                    return True
        lowered = text.lower()
        return any(w and lowered.startswith(w) for w in self.wake_prefix)

    def _addresses_other_user(self, event: AstrMessageEvent) -> bool:
        """Whether the message @-mentions a specific user other than the bot.

        Such messages are directed at someone else, so the AI should not chime
        in via soft-trigger. @all (``AtAll``) and @-ing the bot itself do not
        count.

        Args:
            event: Current platform message event.

        Returns:
            True if the message @'s a non-bot, non-"all" user.
        """
        self_id = str(event.get_self_id())
        for comp in event.get_messages():
            if isinstance(comp, At) and str(comp.qq) not in ("all", self_id):
                return True
        return False

    async def _run_conversation(
        self,
        task: GenerationTask,
        event: AstrMessageEvent,
        conv_id: str,
        first_text: str,
        images: list[Image],
    ) -> None:
        """Run the streaming conversation loop, restarting on debounce.

        Args:
            task: The generation task (holds pending debounced messages).
            event: The message event that triggered the conversation.
            conv_id: Conversation identifier.
            first_text: Text of the triggering message.
            images: Images attached to the triggering message.
        """
        try:
            image_urls: list[str] = []
            image_descs: list[str] = []
            if self.chat_multimodal:
                image_urls = [
                    img.url or img.file for img in images if img.url or img.file
                ]
            else:
                image_descs = await self._describe_images(images)

            current_text = first_text
            while True:
                task.suppress_record = False
                messages = self.context_mgr.build_messages(
                    conv_id,
                    system_prompt=await self._build_system_prompt(conv_id),
                    memory_texts=await self._recall(conv_id, current_text),
                    image_descriptions=image_descs,
                )
                image_descs = []

                stream_gen = self.chat_client.chat_stream(
                    messages,
                    images=image_urls,
                )
                image_urls = []

                async def send_fn(segment: str) -> None:
                    chain = await self._decorate_segment(event, segment)
                    if not chain:
                        return
                    if not task.suppress_record:
                        self.context_mgr.record(conv_id, "assistant", "bot", segment)
                    self.logger.info(
                        f"ChatCore send | {conv_id} | bot: {segment}"
                    )
                    await self.context.send_message(
                        conv_id,
                        MessageChain(chain=chain),
                    )

                pending = await stream_respond(
                    stream_gen,
                    send_fn,
                    delimiter=self.segment_delimiter,
                    escape_char=self.segment_escape,
                    interval=self.segment_interval,
                    max_segment_chars=self.max_segment_chars,
                    interrupt_check=task.signal,
                )
                if pending is None:
                    break
                trailing, (next_text, cancelled) = pending
                if trailing:
                    # Deliver the finished current sentence; on recall it is
                    # sent but not recorded so it stops polluting the context.
                    chain = await self._decorate_segment(event, trailing)
                    if chain:
                        if not cancelled:
                            self.context_mgr.record(
                                conv_id, "assistant", "bot", trailing
                            )
                        self.logger.info(
                            f"ChatCore send | {conv_id} | bot: {trailing}"
                        )
                        await self.context.send_message(
                            conv_id,
                            MessageChain(chain=chain),
                        )
                if cancelled:
                    # The triggering message was recalled: stop after the
                    # current segment instead of restarting.
                    break
                current_text = next_text
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"ChatCore conversation failed: {e}")
            try:
                await self.context.send_message(
                    conv_id,
                    MessageChain().message("抱歉，我这边出了点问题。"),
                )
            except Exception:
                pass
        finally:
            self.active_tasks.pop(conv_id, None)

    async def _build_system_prompt(self, conv_id: str) -> str:
        """Build the system prompt for a conversation.

        The persona (人格) is taken from AstrBot's own persona manager, so it
        follows the persona the user selected in AstrBot settings. The
        segmentation and action-marker rules are appended on top.

        Args:
            conv_id: Conversation identifier (unified_msg_origin).

        Returns:
            The full system prompt.
        """
        persona = (await self._resolve_persona_prompt(conv_id)) or FALLBACK_SYSTEM_PROMPT
        delim = self.segment_delimiter.replace("\n", "\\n")
        rules = (
            "\n\n回复规则：你可以自行分段，段与段之间用分段符 `"
            + delim
            + "` 分隔，每一段会作为一条独立消息发送。"
            f"如果你确实需要输出分段符本身，请在其前加 `{self.segment_escape}` 转义。"
        )
        if self.markers_enabled:
            rules += (
                "\n如需 @ 某人，在回复中写 `[[at:昵称]]`；如需回复某人的消息，"
                "写 `[[reply:昵称]]`（昵称用最近对话里对方的名字）。"
            )
        rules += (
            "\n当前消息附带的图片可以直接查看；历史消息里的“[图片]”表示你"
            "看不到该图片的实际内容，严禁编造或猜测图片内容。"
        )
        return persona + rules

    async def _resolve_persona_prompt(self, conv_id: str) -> str:
        """Resolve the persona prompt AstrBot applies to this conversation.

        Mirrors AstrBot's own resolution order: session rule -> conversation
        persona -> provider default (``astr_main_agent._ensure_persona_and_skills``).

        Args:
            conv_id: Conversation identifier (unified_msg_origin).

        Returns:
            The persona system prompt, or an empty string when unresolved.
        """
        try:
            provider_settings = self.context.get_config(conv_id).get(
                "provider_settings", {}
            )
        except Exception:
            provider_settings = {}
        conversation_persona_id = None
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(
                conv_id
            )
            if cid:
                conv = await self.context.conversation_manager.get_conversation(
                    conv_id, cid
                )
                conversation_persona_id = getattr(conv, "persona_id", None)
        except Exception:
            pass
        try:
            _, persona, _, _ = (
                await self.context.persona_manager.resolve_selected_persona(
                    umo=conv_id,
                    conversation_persona_id=conversation_persona_id,
                    platform_name="aiocqhttp",
                    provider_settings=provider_settings,
                )
            )
        except Exception as e:
            self.logger.warning(f"Persona resolution failed: {e}")
            return ""
        return str((persona or {}).get("prompt", "")).strip()

    def _segment_to_chain(self, conv_id: str, segment: str) -> list:
        """Convert a segment to a message chain, resolving action markers.

        Args:
            conv_id: Conversation identifier.
            segment: The generated segment text.

        Returns:
            A list of message components (Plain / At / Reply).
        """
        if not self.markers_enabled:
            return [Plain(segment)]
        chain: list = []
        for kind, value in parse_actions(segment):
            if kind == "text":
                if value:
                    chain.append(Plain(value))
                continue
            info = self.context_mgr.resolve_target(conv_id, value)
            if kind == "at":
                if info:
                    chain.append(At(qq=info["sender_id"], name=value))
                else:
                    chain.append(Plain(f"@{value}"))
            elif info and info["message_id"]:
                chain.append(Reply(id=info["message_id"]))
            else:
                chain.append(Plain(f"(回复 {value})"))
        for i, comp in enumerate(chain):
            if isinstance(comp, Reply) and i != 0:
                chain.insert(0, chain.pop(i))
                break
        return chain

    async def _decorate_segment(self, event: AstrMessageEvent, segment: str) -> list:
        """Replay AstrBot's pre-send decoration hooks on one streamed segment.

        ChatCore bypasses the vanilla pipeline, so the
        ``OnDecoratingResultEvent`` hooks registered by other plugins
        (``on_decorating_result``) never fire as they normally would in
        ``ResultDecorateStage``. Re-run them so plugins can still modify each
        reply before it is sent.

        Args:
            event: The message event that triggered this conversation.
            segment: The generated segment text.

        Returns:
            The final message chain to send, or an empty list when suppressed.
        """
        result = MessageEventResult(
            chain=self._segment_to_chain(event.unified_msg_origin, segment)
        )
        result.set_result_content_type(ResultContentType.STREAMING_FINISH)
        event.set_result(result)
        try:
            handlers = star_handlers_registry.get_handlers_by_event_type(
                EventType.OnDecoratingResultEvent,
                plugins_name=event.plugins_name,
            )
            for handler in handlers:
                try:
                    await handler.handler(event)
                except BaseException as e:
                    self.logger.error(
                        f"on_decorating_result hook {handler.handler_name} failed: {e}"
                    )
                    continue
                result = event.get_result()
                if result is None or not result.chain:
                    break
            result = event.get_result()
            return list(result.chain) if result and result.chain else []
        finally:
            event.clear_result()

    async def _describe_images(self, images: list[Image]) -> list[str]:
        """Describe attached images with the vision model.

        Args:
            images: Image components of the triggering message.

        Returns:
            Descriptions, one per readable image.
        """
        if not self.vision_client or not images:
            return []
        descriptions: list[str] = []
        for img in images:
            url = img.url or img.file
            if not url:
                continue
            try:
                descriptions.append(await self.vision_client.describe_image(url))
            except Exception as e:
                self.logger.warning(f"Image description failed: {e}")
        return descriptions

    async def _recall(self, conv_id: str, text: str) -> list[str] | None:
        """Recall relevant global memories for a message.

        Args:
            conv_id: Conversation identifier.
            text: Query text.

        Returns:
            Recalled memory texts, or None when memory is disabled.
        """
        if not self.memory:
            return None
        try:
            tags = None if self.memory_shared else [conv_id]
            return await self.memory.recall(text, top_k=self.memory_top_k, tags=tags)
        except Exception as e:
            self.logger.warning(f"Memory recall failed: {e}")
            return None

    async def _remember(self, conv_id: str, sender_name: str, text: str) -> None:
        """Store a message into global memory (fire and forget).

        Args:
            conv_id: Conversation identifier.
            sender_name: Sender display name.
            text: Message text.
        """
        if not self.memory or not text:
            return
        try:
            await self.memory.add(
                f"{sender_name} 说: {text}",
                tags=[conv_id] if not self.memory_shared else None,
            )
        except Exception as e:
            self.logger.debug(f"Memory add failed: {e}")

    @filter.command("chatcore")
    async def chatcore(self, event: AstrMessageEvent):
        """Show ChatCore runtime status.

        Args:
            event: Current platform message event.
        """
        lines = [
            "ChatCore 运行状态：",
            f"- 聊天模型: {self.chat_provider_id or '(未配置)'}",
            f"- 多模态: {'原生识图' if self.chat_multimodal else '走识图模型'}"
            + (
                f" ({self.vision_client.provider_id})"
                if not self.chat_multimodal
                else ""
            ),
            f"- 全局记忆: {'启用' if self.memory else '未启用'}",
            f"- 隐性分析: "
            + (
                f"启用 ({self.analysis_client.provider_id})"
                if self.implicit_enabled and self.analysis_client
                else "关闭（未配置分析模型）"
            ),
            f"- 主动动作: {'启用' if self.markers_enabled else '关闭'}",
            f"- 撤回取消: {'启用' if self.recall_cancel_enabled else '关闭'} "
            f"(已取消 {self.recalls_cancelled} 次)",
            f"- 分段符: {self.segment_delimiter!r}",
            f"- 分段间隔: {self.segment_interval}s",
            f"- 正在进行的对话: {len(self.active_tasks)}",
        ]
        yield event.plain_result("\n".join(lines))

    async def initialize(self) -> None:
        """Start background tasks when the plugin is activated."""
        if self.implicit_enabled and self.analysis_client:
            self._analysis_task = asyncio.create_task(self._implicit_analysis_loop())

    async def _implicit_analysis_loop(self) -> None:
        """Periodically analyze group topics for implicit AI relevance.

        Runs infrequently (interval + random jitter); a single failed analysis
        must not stop the loop.
        """
        while True:
            await asyncio.sleep(
                self.implicit_interval * 60 + random.uniform(0, 600),
            )
            try:
                await self._run_implicit_analysis_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"Implicit analysis failed: {e}")

    async def _run_implicit_analysis_once(self) -> None:
        """Judge each active group topic and bump activity on relevance.

        Groups with an in-flight generation are skipped to avoid competing
        with an ongoing reply.
        """
        if not self.analysis_client:
            return
        for conv_id in self.context_mgr.active_conversations():
            if conv_id in self.active_tasks:
                continue
            context_text = self.context_mgr.summary_text(conv_id)
            if not context_text:
                continue
            try:
                verdict = await self.analysis_client.chat(
                    [
                        {"role": "system", "content": self.implicit_prompt},
                        {
                            "role": "user",
                            "content": f"群聊最近记录：\n{context_text}",
                        },
                    ],
                    temperature=0.0,
                )
            except Exception as e:
                self.logger.warning(f"Implicit analysis call failed: {e}")
                continue
            if "是" in verdict:
                self.attention.bump_probability(conv_id, self.implicit_boost)

    async def terminate(self) -> None:
        """Clean up on plugin disable / reload."""
        if self._analysis_task:
            self._analysis_task.cancel()
            self._analysis_task = None
        self.active_tasks.clear()
