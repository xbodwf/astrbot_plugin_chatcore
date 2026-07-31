# ChatCore 架构设计

> 本文档是 `astrbot_plugin_chatcore` 的活文档，所有设计决策与用户补充要点必须持续更新到此文档。

## 1. 背景与目标

AstrBot 原生聊天能力差，具体体现在：上下文爆塞、分段不智能、流式跳过装饰阶段、无法主动 reply/@、群聊回复机制死板。ChatCore 的目标是**接管 AstrBot 的 vanilla 聊天链路**，把 AI 聊天变得像真人一样：看得懂、回得快、分得准、懂时机、有记忆。

## 2. 核心问题

| 问题 | 现状 | 后果 |
|---|---|---|
| 上下文 | 历史无差别爆塞 | AI 稀疏注意力只捞部分信息，漏信息、变笨 |
| 分段 | 正则/机器分段 | AI 想输出符号被误切；关流式=回复卡顿，开流式=插件装饰失效 |
| 回复机制 | 群聊固定概率 | 死板，不像真人 |
| 主动动作 | 不能 reply/@ 他人 | 即便有插件也会 @ 错消息 |
| 消息插队 | 回复中途用户发新消息 | 信息差，AI 继续发过时回复 |

## 3. 总体架构

```
消息事件(aiocqhttp)
   │
   ▼
┌─────────────────────────────────────────────┐
│ main.py 调度层                               │
│  - 触发判定(硬/软触发)                       │
│  - 智能防抖(消息插队)                        │
│  - 消息记录 → context / memory               │
└─────────────────────────────────────────────┘
   │                        │
   ▼                        ▼
attention.py            context.py
注意力状态机(每群)      智能上下文构建
- 冒泡基线 1~3%        - 近 N 条完整
- 活跃封顶 30%         - 更早历史压缩
- 指数衰减 10min       - 全局记忆召回
- @/reply 加成          - 图片转描述(视觉模型)
   │                        │
   ▼                        ▼
┌─────────────────────────────────────────────┐
│ llm.py AstrBot 提供商访问层                  │
│  - 按 provider_id 解析提供商                │
│  - chat / chat_stream                       │
│  - vision 识图                             │
│  - embedding 嵌入                          │
└─────────────────────────────────────────────┘
   │
   ▼
segmentation.py
智能分段(流式消费)
- AI 自我分段(转义分段符)
- 每段独立发送+间隔
- 防抖中断点
```

**核心原则**：插件不管理任何模型接口，完全复用 AstrBot 自身提供商体系。用户在 AstrBot「模型提供商」页面配置后，插件用 `_special: "select_provider"` 面板选择，运行期经 `provider_manager.get_provider_by_id()` 解析调用；所有异步任务（流式、隐性分析、记忆）基于 asyncio。

## 4. 模块设计

### 4.1 注意力机制（attention.py）

双通道触发，每群一个状态机：

- **硬触发（必回复）**：私聊 / `@bot` / `reply(bot)` / 唤醒前缀。
- **软触发（概率回复）**：
  - **冒泡基线** 1%~3%：群聊无人搭理 AI 时，AI 也有小概率自发插话。
  - **活跃态封顶 30%**：有人在跟 AI 对话 → 概率爬升。
  - **指数衰减**：最后一次交互后，若 `decay_minutes`（默认 10 分钟）内无人维持，概率沿指数曲线降回基线。
  - **@/reply 加成**：硬触发除了必回，还显著拉高活跃度。
  - **维持**：任何"有人回应"都刷新 `last_interaction_ts`。

概率模型（惰性计算，不靠定时器）：

```
idle_min = (now - last_interaction_ts) / 60
active_contribution = (active_cap - bubble_base) * exp(-k * idle_min / decay_minutes)
boost_remaining = sum(每次硬触发的加成 * exp(-k * (now - ts_i) / decay_minutes))
prob = clamp(bubble_base + active_contribution + boost_remaining, 0, active_cap)
```

`k=3`（在 `decay_minutes` 时刻衰减到约 5%，视为降回基线）。命中后再真正发起 LLM 请求，**"决定回复"与"生成内容"分离**。

### 4.2 智能上下文（context.py）

告别爆塞，分层构建：

1. **近 N 条完整原文**（`recent_count`，默认 10）：AI 直接可用的最新对话。
2. **更早历史压缩**（`history_count` 内，超出 recent 的部分逐条截断到 `old_msg_chars`）：保留关键信息、去掉噪声，适配稀疏注意力。
3. **全局记忆召回**：用 embedding 检索跨群共享记忆，注入相关片段（见 4.6）。
4. **图片转描述**：消息含图片 → 交给视觉模型描述 → 以文本形式进上下文。

注入路径：不用 `system_prompt +=`（会打爆缓存）。消息布局采用**缓存友好前缀**：`system`（人格，首位且字节稳定）→ `recent` 近 N 条逐字对话（只增不改，作为稳定前缀）→ 末尾单个 `user` 背景块（图片描述 / 记忆召回 / 压缩的更早历史，全部是短命内容）。这样动态内容永远不挤占共享前缀，窗口未滑动时命中率近乎满；窗口滑动瞬间（有界窗口的固有代价）整条已累积前缀作废，只能靠加大 `recent_count` 减少滑动频率。

### 4.3 智能分段（segmentation.py）

接管 vanilla LLM，使用流式输出，**AI 自己分段**：

- Prompt 约定分段符（默认 `\n---\n`，可配置），并声明**转义规则**：AI 真想输出该符号时用转义符（默认 `\`）前缀。
- 流式消费 token，按分段符切成段；**转义感知**：`\` 后一个字符按字面处理，不触发切分。
- 每段独立发送 → 每段是一条消息，**发送等待与流式生成并行**（解决"很久不回+断断续续"）。
- **手动补装饰阶段**：绕过 vanilla 流式后，原 `on_decorating_result` 等插件链失效。ChatCore 在 `main.py._decorate_segment` 里**重放装饰钩子**：把每段构造成 `MessageEventResult`（`result_content_type=STREAMING_FINISH`，与 vanilla 流式语义一致），按 `EventType.OnDecoratingResultEvent` 从 `star_handlers_registry` 取钩子逐个调用，装饰后再发送。

### 4.4 智能防抖（main.py / segmentation.py）

AI 回复中途用户发新消息（且命中触发）时：

1. 等**当前段**凑满/收尾 → 立即结束本次流式。
2. 新消息并入历史上下文 → 以新上下文发起新一轮请求。
3. 消除"AI 继续发过时回复"的信息差。

实现：每会话一个 `GenerationTask`，持有流式任务 + 新消息队列 + 中断标记；分段循环在段边界检查中断。

### 4.5 AstrBot 提供商访问层（llm.py）

插件不管理任何模型接口，复用 AstrBot 提供商体系：

- **配置**：`providers.chat.provider_id`、`providers.vision.provider_id`、`implicit.provider_id` 使用 `_special: "select_provider"`，在插件配置面板直接下拉选择 AstrBot 已配置的提供商；`providers.embedding.provider_id` 为手填 ID（AstrBot 无 embedding 专用选择器）。
- **解析**：运行期 `context.provider_manager.get_provider_by_id(id)` 取实例（查不到或为空抛中文 `ValueError`）；`LLMProvider` 包 `Provider.text_chat` / `text_chat_stream`，流式时跳过非 chunk 的尾部汇总响应，文本优先取 `result_chain.get_plain_text()` 再回落 `completion_text`。
- **多模态**：若聊天模型原生支持识图（`providers.chat.multimodal=true`），图片直接以 `image_url` 内容块传给聊天模型；否则走独立 vision 模型转文字描述。
- **vision 识图模型**：消息带图时转描述（`describe_image`）。
- **embedding 模型**：`EmbeddingAdapter` 校验实例为 `EmbeddingProvider` 后调用 `get_embedding`，用于全局记忆向量召回。

无 aiohttp、无独立 SSE 解析，全部交给 AstrBot 提供商实现。

### 4.6 记忆与自我学习（memory）

- **全局记忆**：跨群共享。群聊数据互通 → AI 学习各群的人设/话题/用户画像，更懂人。
- **存储**：不使用独立数据库，也不依赖 AstrBot 的知识库体系；插件**自建向量库**（embedding + 余弦相似度）持久化为 JSON 文件（`astrbot.core.utils.astrbot_path` 提供的插件数据目录），无外部依赖。
- **召回时机**：请求构建上下文时，用 embedding 检索相关记忆注入。

> 注：记忆模块首版先落「写入+向量召回」骨架，隐性意图分析见 4.7。

### 4.7 隐性意图分析（异步任务）

- 单独的 AI 异步任务，**不定时、不频繁**地分析群上下文，判断有无隐性提及/相关话题。
- 命中则提升该群活跃度（走 4.1 同一概率模型）。
- 也服务于"冒泡"质量：冒泡命中时先判断话题接得上再开口。
- asyncio 承载，任务多，单任务失败不拖垮主流程。

### 4.8 主动动作

- 构建回复消息链时支持 `Comp.At`（指向目标用户）与 `Comp.Reply`（基于原始消息 ID），修掉"@ 错消息"。
- AI 需要时可在回复中用 `[[at:昵称]]` / `[[reply:昵称]]` 主动 @ / 回复（由 `actions.py` 解析并降级处理）。

## 5. 配置项（_conf_schema.json）

所有参数用户可调，分组：

| 分组 | 关键项 |
|---|---|
| chat | 总开关、群/私聊开关、接管模式、多模态开关、主动动作标记开关（人格提示词取自 AstrBot，不在此配置；流式+智能分段为固定设计） |
| attention | 冒泡基线 1~3%、活跃封顶 30%、衰减分钟数、@/reply 加成、唤醒前缀 |
| segment | 分段符、转义符、分段发送间隔(秒)、单段最大字符数 |
| context | 近 N 条完整数、历史压缩条数、每条压缩字符数 |
| providers | chat / vision / implicit 用 `select_provider` 下拉选 AstrBot 提供商；embedding 手填 provider_id；chat 多模态开关 |
| memory | 记忆开关、跨群共享开关、召回条数 |
| implicit | 隐性分析开关、分析最小间隔(分钟)、概率提升、独立分析模型（`select_provider` 选提供商）、分析提示词（留空用默认） |
| recall_cancel | 撤回取消开关 |

## 6. 变更记录

- `2026-07-31` 初始架构：智能上下文、智能分段、注意力机制、智能防抖、接管 vanilla LLM。
- `2026-07-31` 补充：注意力机制更正为双通道 + 活跃度模型（冒泡基线 1~3%、活跃封顶 30%、10 分钟指数衰减、@/reply 必回且加活跃度）。
- `2026-07-31` 补充：AI 无对话也可能自发"冒泡"（= 软触发基线态，闭环）。
- `2026-07-31` 补充：独立模型提供商（chat/vision/embedding 自配）；AI 自我学习、群聊数据互通、插件自建记忆库（复用 AstrBot 嵌入向量库）+ AI 全局记忆。
- `2026-07-31` 约定：后续用户补充要点必须同步更新本文档。
- `2026-07-31` 补充：多模态——聊天模型原生支持识图时（配置开关），图片直接传给聊天模型，不再强制需要独立识图模型。
- `2026-07-31` 补充：提示词——聊天人格不再自配 system_prompt，改为调用 AstrBot API（`context.persona_manager`）直接取用户在 AstrBot 配置中选择的人格（含会话级/对话级覆盖）；唯一需要用户自写的是隐性分析器的提示词（插件配置 `implicit.prompt`，留空用默认），且隐性分析模型配置（`implicit.model`）与聊天模型相互独立。
- `2026-07-31` 补充：主动动作——AI 可用 `[[at:昵称]]` / `[[reply:昵称]]` 标记主动 @ 或回复群成员，插件把昵称解析为最近对话中的发送者 id/消息 id 后转为 `Comp.At`/`Comp.Reply` 组件发送（解析不出时降级为纯文本）。
- `2026-07-31` 补充：手动补装饰阶段——接管 vanilla 流式会跳过 `on_decorating_result` 装饰钩子，现于 `_decorate_segment` 重放 `OnDecoratingResultEvent` 钩子（每段构造 `MessageEventResult` 后逐个调用注册的钩子，装饰结果即为最终发送内容）。
- `2026-07-31` 补充：撤回取消——用户撤回触发 LLM 的消息后，等当前段输出完立即停止 LLM 回复；该撤回回复发送但不计入上下文，并从历史中移除该消息（留下「已撤回」占位）。实现：独立 `on_recall_notice` handler（aiocqhttp、priority=100，从 `raw_message` 读 `notice_type`/`message_id` 匹配当前任务的触发消息 id）；`GenerationTask` 增 cancel 信号，`stream_respond` 在段边界重查中断信号（`segment_token`），取消时丢弃后续缓冲。
- `2026-07-31` 调整：上下文缓存友好化——`build_messages` 把图片/记忆/压缩历史等动态内容从「system 之后、recent 之前」移到消息列表末尾，保证 `system + recent 逐字对话` 作为稳定前缀，窗口未滑动时共享前缀近乎 100% 命中（详见 4.2）。
- `2026-07-31` 调整：模型访问改走 AstrBot 提供商体系——移除插件自带 aiohttp `ChatClient` 与 base_url/api_key/model 配置；chat/vision/implicit 改用 `_special: "select_provider"` 面板选择、embedding 手填 provider_id，运行期 `provider_manager.get_provider_by_id()` 解析（llm.py 重写为 `LLMProvider` / `EmbeddingAdapter`，见 4.5）。
- `2026-07-31` 约定：`_conf_schema.json` 中 `description` 为配置项标题、`hint` 为说明文字，说明性内容一律进 `hint`。
- `2026-07-31` 补充：推理内容过滤——流式/非流式输出统一剥离 `<think>...</think>` 推理块（`llm.py.ThinkStripper`，跨 chunk 边界与未闭合标签均可处理），防止思考内容泄漏给用户。
