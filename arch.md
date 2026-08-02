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
5. **AstrBot 持久化聊天记录注入**（`history.*`，见 4.9）。

注入路径：不用 `system_prompt +=`（会打爆缓存）。消息布局采用**缓存友好前缀**：`system`（人格，首位且字节稳定）→ `recent` 近 N 条逐字对话（只增不改，作为稳定前缀）→ 末尾单个 `user` 背景块（图片描述 / 记忆召回 / 压缩的更早历史 / 持久化历史，全部是短命内容）。这样动态内容永远不挤占共享前缀，窗口未滑动时命中率近乎满；窗口滑动瞬间（有界窗口的固有代价）整条已累积前缀作废，只能靠加大 `recent_count` 减少滑动频率。

**压缩与滚动摘要**：

- **逐条截断**：超出 `recent_count` 的旧消息逐条截断到 `old_msg_chars` 字符。图片占位符 `[图片]` 不参与压缩（描述参与），assistant 消息带 `bot:` 前缀区分说话人，`_truncate` 优先在标点/空格断句，避免裁出半句话。
- **LLM 滚动摘要**：启用 `context.llm_summarize`（默认 true）时，历史超过阈值触发 `_schedule_summary` → `_summarize_history` 异步调用摘要模型，把更早对话提炼为摘要缓存（`_summaries`，按 conv_id 存 `(摘要, 断点, 模型, 版本)`）。`build_messages` 有摘要就用摘要代替该段原文（减少 token），无摘要则回落逐条截断。
- **摘要独立模型**：`providers.summary.provider_id` 可单独指定摘要用的提供商（留空复用聊天模型）。`_init_from_config` 构造独立 `summary_client`（`LLMProvider`），压缩任务不占用聊天模型配额。

**引用回溯（quote）**：用户引用消息时（如 QQ 回复某条消息），`on_message` 用 `_extract_quote_chain` 沿 `Reply.chain` 递归解析引用链，把被引内容（含逐级回溯的原始消息）以 `[引用了消息: ...]` 前缀注入当前用户消息（`MessageRecord.quote`），最大嵌套深度 `context.quote_max_depth`（默认 15）。解析不到则用 `find_message` 查内存历史兜底；系统提示告知 AI 可据此回看对方引用的内容。

### 4.3 智能分段（segmentation.py）

接管 vanilla LLM，使用流式输出，**AI 自己分段**：

- Prompt 约定分段符（默认 `---` 独占一行，可配置），并声明**转义规则**：AI 真想输出该符号时用转义符（默认 `\`）前缀。
- **行级识别**：`StreamSegmenter` 用 `_try_exact`（内联子串匹配，如 `。`）与 `_try_standalone_line`（独占一行匹配，容忍 `\n\n---\n\n` 与结尾无换行的 `---`）两种模式切分；转义行（`\---`）按字面处理不触发切分，`a---b` 行内不会误伤，`flush()` 丢弃结尾独立的未闭合分隔符行，彻底杜绝分隔符泄漏到输出。
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

### 4.9 持久化聊天记录注入（history.py）

复用 AstrBot 的 `conversation_manager` 与 `platform_message_history_mgr` 读取已持久化的聊天记录，补足插件内存上下文的空窗：

- **当前会话历史**：`HistoryReader.read_session(umo, max_messages, max_chars)` 读当前会话（`build_umo`）最近消息，`_inject_history_blocks` 在构建请求时以 `【历史参考】...` 头部注入。
- **群聊 + 私聊交叉**：群聊中用户发言时，`build_friend_umo(conv_id, sender_id)` 定位该发送者对应的私聊会话，自动注入其最近私聊历史（`【记忆参考】...`），让 AI 记得与该用户的私聊语境。
- 文本化：`extract_text_history` / `render_history_block` 把 `MessageChain` 序列化为纯文本（图片→`[图片]`），丢弃非文本组件。
- **配置**：`history.inject_enabled`（默认 true）、`history.max_messages`（默认 10）、`history.max_chars`（默认 1200）。

### 4.10 引用回溯（quote）

用户引用某条消息（如 QQ 回复）时，AI 需要读到被引用的内容：

- `on_message` 解析事件组件中的 `Reply`，沿 `Reply.chain` 递归回溯整条引用链（`_extract_quote_chain` / `_resolve_quote_node`）。
- 被引内容（逐级含"又引用了"前缀）以 `MessageRecord.quote` 字段随消息存入，渲染为 `[引用了消息: ...]` 前缀进入上下文。
- 内存历史兜底：`find_message(conv_id, message_id)` 查不到 adapter 解析结果时，从插件内存历史取原文。
- 深度上限 `context.quote_max_depth`（默认 15），防止循环引用与超深链拖垮性能。

### 4.11 人物画像 + 记忆写回（profile.py）

为每个用户维护**结构化画像**（区别于 4.6 的文本片段向量记忆），让 AI "越来越懂这个人"：

- **画像字段**：`person_id`（platform+user_id）、昵称、稳定事实（`facts`）、偏好（`preferences`）、互动特点（`interaction`）、最近活跃时间。
- **写回**：每次回复后异步 `_extract_person_facts`——用摘要模型从本轮对话提取**稳定的人物事实**（不是流水账），去重合并进该用户画像。
- **注入**：`build_messages` 时把当前发送者的画像压缩成背景块注入（与记忆召回同级），`_truncate` 限长。
- **持久化**：JSON 文件（插件数据目录），重启不丢。

与 4.6 全局记忆的分工：记忆 = 向量召回的"片段"，画像 = 结构化的"对人的理解"；两者互不重叠。

### 4.12 表达风格学习（expression.py）

解决"永远 AI 腔"的最大短板：**从群聊学习表达方式，模仿群友**。

- **收集**：异步定时任务（复用隐性分析的调度节奏）从各群抽取最近消息样本。
- **分析**：LLM 从样本中提炼三类产出并落 JSON：
  - 常用句式/语气（如 `hhh`、`绷不住了`、短句流、方言词）；
  - 黑话词条及 LLM 推断的含义（`?` 有歧义时保留来源例句）；
  - 该群表达风格的一句话总结。
- **注入**：对应群的 system prompt 追加「表达风格」块，包含句式偏好 + 黑话表（带例句）。
- **共享组**：`expression.shared_groups` 允许跨群互通表达风格（可选，默认仅本群生效）。

### 4.13 表情包系统（带溯源）（emoji.py）

吸收 MaiBot 教训（偷包与使用脱节、AI 只看到分类好的库里有什么、无法溯源、无法结合原语境理解含义）。ChatCore 设计为**带溯源的表情包对象**：

- **数据模型**：每条表情包携带完整来源——
  `emoji_id`、本地 `file_path`、`source_group`、`source_sender`、`source_message_id`、`source_text`（偷包时该消息的文字）、`source_context`（原上下文窗口）、`collected_at`、`category`（VLM 分类：开心/嘲讽/敷衍…）、`tags`、`usage_count`。
- **收集（偷包）**：群消息带图且开启收集时，落库图片**并记录来源语境**（发送者/群/原消息文字/上下文窗口）；异步用视觉模型打分类与标签。**不是无意识偷图，每张都带出处**。
- **检索使用**：AI 通过工具 `search_emoji(意图)` 检索，返回的条目**附来源语境原文**（如 `[表情: 草.jpg] 来源语境: "笑死，你这头像跟哈批一样"）——AI 先读到"这张图当初是配什么话用的"，再决定用不用，语义精准。
- **使用回写**：AI 选用后 `usage_count+1`；WebUI 可查看、分类、删除、编辑标签。

### 4.14 情绪/状态系统（emotion.py）

让人格随聊天氛围流动，避免一个调子说话到底：

- 每会话维护 `EmotionState(mood, energy, state_key, updated_at)`。
- **情绪特质**：`emotion.trait`（rational_calm / neutral / sentimental）决定基线。
- **状态列表**：`emotion.states` 可配置一组状态（每项含描述 + 替换概率），聊天氛围变化时按概率切换；回复后由 LLM 或启发式更新状态。
- **注入**：当前状态描述进 system prompt（`当前情绪: 慵懒`），影响措辞长度与语气。

### 4.15 退避策略（读空气）（attention.py 扩展）

在 4.1 概率模型上补"该闭嘴时闭嘴"：

- **冷却**：bot 刚回复过 → `attention.cool_down_seconds` 内抑制软触发（硬触发仍必回）。
- **退避**：连续 soft-trigger 落空（没轮到说话）→ 临时降低后续概率（no_action 退避）。
- **动态发言频率**：`attention.time_rules` 按时间段配置发言概率调整（如凌晨低、晚间高）。
- **读空气**：他人高密度对话（无 @bot）时进一步压低冒泡概率；话题在 bot 上次发言后仍在延续则抬高。

### 4.16 WebUI（PluginPage）

AstrBot 4.23.4+ 插件页机制（`pages/<页面>/index.html` 由 dashboard iframe 托管，自动注入 `AstrBotPluginPage` 桥：`ready/apiGet/apiPost/upload/subscribeSSE`，相对资源自动加鉴权 token 重写；后端用 `context.register_web_api(route, handler, methods, desc)` 注册，暴露于 `/api/plug/...`）。

- **页面结构**：
  ```
  pages/
    dashboard/                 # 管理面板
      index.html
      app.js / styles.css
      vendor/                  # MUI MD3 等第三方前端资源，提前下载备好
      assets/
    LICENSE                    # vendor 资源版权/许可声明（协议名+作者+出处）
  .gitattributes               # vendor/ 打 linguist-vendored，避免计入代码占用分析
  ```
- **前端技术**：无构建步骤的纯静态页 + MUI Material Design 3（JS/CSS 提前下载进 `vendor/`，离线可用）；主题走 `data-theme` 自适应深浅色。
- **页面内容**（两者都要）：
  1. **记忆/画像/表情包管理**：查看/编辑/删除人物画像、记忆片段、表情包库（含来源溯源）；
  2. **运行状态监控**：注意力活跃度曲线、上下文窗口占用、分段统计、退避冷却、情绪状态。
- **后端 API**：`register_web_api` 注册 `/chatcore/{profiles|memories|emoji|stats}` 等 REST 路由，页内 `apiGet/apiPost` 调用。
- **资源合规**：`vendor/` 内每个第三方资源在 `pages/LICENSE` 注明协议（如 MIT/Apache-2.0）+ 作者 + 来源 URL；`.gitattributes` 标记 vendored 以免污染语言统计。

### 4.17 工具调用（按需下发）（main.py / llm.py）

让 AI 具备 function calling（查资料、定时提醒等），但**普通闲聊零工具开销**：

- **按需下发（模型自主声明）**：常规轮次**不携带工具 schema**（回复快、token 省）。system prompt 告知模型：完成请求必须使用工具时，在回复最开头独占一行输出 `[[tools]]`。插件在分段发送时拦截该标记（该段不发送），下一轮以**带工具**的请求重试。是否需要工具由模型自己判断，无硬编码关键词。
- **工具集**：`_build_tool_set()` 懒构建缓存——AstrBot 全局注册表 `llm_tools.func_list`（其他插件注册的工具）+ 内置 `FutureTaskTool`（复用 AstrBot 定时任务）+ 自研 `schedule_task`（`create/list/delete` 一次性提醒）。`chat.tools_enabled` 总开关，`chat.max_tool_rounds` 上限。
- **执行**：`_execute_tool` 复用 AstrBot `FunctionToolExecutor`（构造最小 `AstrAgentContext` wrapper），内置/插件工具行为与 vanilla 一致；结果以 `role:"tool"` 回传。
- **回传协议**：OpenAI 要求 tool 结果跟在带 `tool_calls` 声明的 assistant 消息之后，否则 AstrBot 的 `_sanitize_assistant_messages` 当作孤儿消息丢弃（表现为模型反复调用同一工具）。工具轮先 append assistant 声明（`tool_calls` 含 id/name/arguments JSON），再 append 各 tool 结果。
- **定时任务调度**：`_scheduler_loop`（20s 轮询，JSON 持久化 `scheduled_jobs.json`）触发到期任务，构造 `CronMessageEvent` 走 `_run_conversation` 完整链路（人格/情绪/记忆/分段/表情）投递到目标会话。AstrBot 原生 cron 触发固定走 vanilla agent，无法注入，故调度器自研。

### 4.19 好感度系统（affinity.py）

MaiBot 风格的关系亲密度：每会话（conv_id，群/私聊天然隔离）维护好感度 0~100：

- **增减**：私聊互动 +3、硬触发（@/reply） +2、普通互动 +1；封顶 100、下限 0。
- **衰减**：无互动时按天惰性衰减（`decay_per_day`，默认 2），`get()` 时按距上次互动时间计算，不依赖定时器。
- **档位**：冷淡(<20) / 疏离(<40) / 普通(<60) / 熟络(<80) / 亲密(>=80)，`inject_text` 生成「当前关系」块注入 system prompt（自然措辞、禁止模型提及该说明），影响语气与称呼。
- **持久化**：JSON（插件数据目录 `affinity.json`），重载不丢；WebUI 监控 tab 展示好感度与档位。
- **配置**：`affinity.*`（开关、初始值、每日衰减）。

### 4.18 回复中断续聊

生成被中断（异常/取消/插件重载）时标记该会话（`_interrupted`，1 小时内有效）；下次该会话互动时在背景块注入「你上一条回复因故中断了，如果合适请接着把没说完的话补完」，让 AI 自然续上（详见 4.2 背景块注入路径）。

## 5. 配置项（_conf_schema.json）

所有参数用户可调，分组：

| 分组 | 关键项 |
|---|---|
| chat | 总开关、群/私聊开关、接管模式、多模态开关、主动动作标记开关、工具调用开关、单次回复最大工具轮数（人格提示词取自 AstrBot，不在此配置；流式+智能分段为固定设计） |
| attention | 冒泡基线 1~3%、活跃封顶 30%、衰减分钟数、@/reply 加成、唤醒前缀 |
| segment | 分段符、转义符、分段发送间隔(秒)、单段最大字符数 |
| context | 近 N 条完整数、历史压缩条数、每条压缩字符数 |
| providers | chat / vision / implicit 用 `select_provider` 下拉选 AstrBot 提供商；embedding 手填 provider_id；chat 多模态开关 |
| memory | 记忆开关、跨群共享开关、召回条数 |
| implicit | 隐性分析开关、分析最小间隔(分钟)、概率提升、独立分析模型（`select_provider` 选提供商）、分析提示词（留空用默认） |
| recall_cancel | 撤回取消开关 |
| profile | 画像开关、每会话注入条数/字符上限 |
| expression | 表达学习开关、采样间隔、注入字符上限、跨群共享组 |
| emoji | 表情包收集开关、存储上限、分类模型（`select_provider`）、收集白/黑名单 |
| emotion | 情绪系统开关、情绪特质、状态列表与替换概率 |
| backoff | 冷却秒数、退避衰减、时间段发言频率规则 |

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
- `2026-08-01` 调整：上下文防幻觉——系统提示明确「历史消息里的 [图片] 表示看不到内容、严禁编造图片内容」（当前消息附带图片可直接查看）；记忆召回背景块改为「过往对话片段（可能来自其他对话或其他人，不代表你本人执行过任何操作）」。
- `2026-08-01` 调整：触发防抢话——群消息若 @ 的是别的用户（非 bot、非 @all），不参与软触发（`_addresses_other_user`）。
- `2026-08-01` 补充：图片重构——多模态模型直接读图输出 `[图片描述: ...]` 标记存入历史；`MessageRecord` 增 `images`/`description` 字段，`set_image_description` 按消息 id 回填描述；压缩时裸 `[图片]` 占位不参与压缩、描述参与；assistant 压缩行加 `bot:` 前缀；`_truncate` 优先断句（详见 4.2）。
- `2026-08-01` 补充：上下文压缩 LLM 摘要化——`ContextManager._summaries` 缓存摘要（set_summary/get_summary/summary_stale/older_count/summary_payload）；`build_messages` 有摘要用摘要、无则逐条截断兜底；`main.py` 异步 `_schedule_summary`→`_summarize_history`；配置 `context.llm_summarize`（默认 true）。
- `2026-08-01` 补充：压缩独立模型——`providers.summary.provider_id` 单独指定摘要提供商（留空复用聊天模型），`_init_from_config` 构造独立 `summary_client`，压缩任务不占聊天配额。
- `2026-08-01` 补充：引用回溯——用户引用消息时沿 `Reply.chain` 递归解析引用链（`_extract_quote_chain`/`_resolve_quote_node`），被引内容以 `[引用了消息: ...]` 前缀注入；`MessageRecord.quote` 字段持久化；`find_message` 查内存历史兜底；深度上限 `context.quote_max_depth`（默认 15）。
- `2026-08-01` 调整：分段器重写——`StreamSegmenter` 改用独占行匹配（`_try_exact`/`_try_standalone_line`），容忍 `\n\n---\n\n` 与结尾无换行 `---`，转义行不被匹配，`flush()` 丢弃结尾独立分隔符行，杜绝分隔符泄漏（见 4.3）；系统提示措辞改为「单独写一行」。
- `2026-08-01` 补充：持久化聊天记录注入——新增 `history.py`（`build_umo`/`build_friend_umo`/`extract_text_history`/`render_history_block`/`HistoryReader.read_session`）；`main.py._inject_history_blocks` 自动注入当前会话历史 + 群聊注入同发送者私聊历史（`【记忆参考】`），配置 `history.inject_enabled`/`max_messages`/`max_chars`（见 4.9）。
- `2026-08-01` 调整：分段符改用真实换行——默认 `delimiter` 由字面 `\n---\n` 改为真实换行 `\n---\n`（JSON 里直接回车），`main.py` 加载时把存量的字面 `\n` 归一化为真实换行；提示词明示「换行是真实换行符、不含反斜杠、不要写 \n 字面量、分隔符行前后不留空格不加反引号」；`segmentation._delimiter_core` 对字面 `\n`/`\r` 与真实换行一并剥离，两种存量配置都能命中独占行（见 4.3）。
- `2026-08-01` 设计：追赶 MaiBot 聊天能力（写入本文档，逐步开工）——人物画像+记忆写回（4.11）、表达风格学习（4.12）、带溯源的表情包系统（4.13）、情绪/状态系统（4.14）、退避策略读空气（4.15）、PluginPage WebUI（4.16，MUI MD3 + vendor 资源 + `.gitattributes`/LICENSE 合规）。
- `2026-08-01` 实现：PluginPage WebUI 前端落地——MUI v9（MD3）经 esbuild 预构建为自包含 ESM `pages/dashboard/vendor/mui.full.js`（内联 React 19/ReactDOM/Emotion，离线可跑、页面无构建步骤）；`app.js` 用 `React.createElement` 免 JSX/Babel；双 Tab：监控（数据统计 + 注意力/上下文/情绪实况）+ 管理（画像/记忆/表情包/表达风格 CRUD）；`styles.css` 提供页面骨架样式；主题经 MutationObserver 监听宿主 `data-theme` 驱动 `useColorScheme().setMode` 深浅色自适应（见 4.16）。
- `2026-08-01` 补充：表情包 WebUI 内联预览——iframe sandbox 无 `allow-same-origin`，不可直读二进制，新增 `GET .../emojis/<emoji_id>/image/data` 返回 base64 data URI JSON；父桥 `apiGet` 会解包 `response.data?.data`，故该路由返回 `{"data": "data:image/<mime>;base64,..."}` 后前端直接拿到 URI 字符串（`EmojiThumb` 处理）。
- `2026-08-02` 补充：工具调用——按需下发（模型自主声明 `[[tools]]`，常规轮不携带工具 schema）、`_build_tool_set`（插件工具 + `FutureTaskTool` + `schedule_task`）、`FunctionToolExecutor` 复用、tool 结果回传协议（assistant tool_calls 声明先行，防孤儿丢弃）、自研定时任务调度器（`_scheduler_loop` + `CronMessageEvent` 复用完整回复链路）、回复中断续聊标记（见 4.17/4.18）。
- `2026-08-02` 调整：工具请求标记 `[[tools]]` 在无工具轮被分段层拦截丢弃，模型声明后插件升级为带工具请求重试（仅允许一次升级，防循环）。
- `2026-08-02` 补充：对话历史持久化——`ContextManager` 可选 `persist_path`（插件数据目录 `context_history.json`），`record`/`set_image_description`/`remove_message`/`clear` 原子写盘（tmp+rename，失败静默）；插件重载后恢复每会话逐字历史（含图片描述/引用/发送者 ID），WebUI「活跃会话」不再因重载清空；无路径时保持纯内存（测试兼容）。
- `2026-08-02` 补充：好感度系统——`affinity.py` 每会话好感度（互动增减、按天惰性衰减、五档位注入 system prompt），JSON 持久化，WebUI 展示，配置 `affinity.*`（见 4.19）。
- `2026-08-02` 调整：system prompt 规则精简——分段/标记/防伪/工具/图片说明由约 700 字压缩为约 250 字编号短句（自然语气「一些约定，记住即可」），降低工程指令密度，恢复人格主导的角色扮演味道。
- `2026-08-02` 调整：好感度按用户存储——键从 conv_id 改为 sender_id（群/私聊共用同一份好感），`chatcore affinity` 指令（无指令组，在 `chatcore` handler 内手动匹配子命令）：查询自己；管理员可 `chatcore affinity <用户ID>` 查询指定用户；非管理员查询他人返回权限提示。
- `2026-08-02` 补充：LLM 黑名单——`chat.llm_blacklist`（会话 UMO 列表，如 `Sylvia:GroupMessage:384128966`）中的会话完全禁用 LLM：消息照常记录但不回复（含硬触发），接管模式下 stop_event 阻止 vanilla 回复。
- `2026-08-02` 调整：防无中生有——system prompt 新增规则⑦（只依据当前聊天记录实际发生的内容回应、不脑补未发生动作/人名/话题）；记忆召回/画像/私聊历史注入措辞统一为「背景知识，除非用户问起不要主动提起」，阻止跨会话话题被当成当前聊天依据（如把其他群的记忆直接引出）。
- `2026-08-02` 补充：`chatcore reset` 子命令——对齐 vanilla `/reset` 权限（群聊需管理员、私聊成员可用）：停止进行中的回复、清空 ChatCore 内存+持久化历史与摘要缓存、清空 AstrBot 该会话的持久化对话历史。
- `2026-08-02` 调整：自然度与分段——system prompt ① 强化分段（两三句以上就拆多条消息、每段只讲一件事、短小口语）；新增 ⑧ 真人感规则（短句口语、别把话说满、可用语气词、宁短勿长）；`segment.max_segment_chars` 默认 600→300（AI 忘分段时兜底更勤）。
- `2026-08-02` 补充：防复读——`_last_reply` 记录每会话最后一条回复（5 分钟内有效），重建上下文时注入「你刚刚才说过…不要重复类似的话」，打破模型被单一话题（如"早呀"）锁住导致的刷屏复读；`AttentionManager.in_cooldown` 供硬触发冷却使用。
- `2026-08-02` 调整：消息格式 metadata 化（借鉴 MaiBot）——recent 用户消息渲染为 `<message user="昵称(QQ)" msg_id=".." time="..">内容</message>`（元信息与正文分离）；压缩历史用紧凑 `[昵称]内容`（省 token、截断保留说话者）；AstrBot 持久化历史注入同用 `<message user="用户">` / `<message self="bot">`；背景块（记忆/画像/摘要/压缩历史/私聊历史）统一 `[参考消息]` 前缀，与实时对话明确分离，降低"串"。
- `2026-08-02` 补充：XML 标签防注入——`escape_user_markers` 把用户内容里形如 `<字母...>` 的标签转义为全角 `＜...＞`（`>` 全转），防止插件/机器人产生的 `<refuse>` 等标签嵌套进 `<message>` 渲染或被模型当成系统指令。
- `2026-08-02` 调整：空行分段——`StreamSegmenter` 把连续两个换行（空行）作为分段边界（与 `---` 分隔符等效），符合真人聊天用空行分段的习惯；prompt ① 改为「分段就在两段之间空一行」，模型不再依赖写 `---`。
- `2026-08-02` 补充：兼容与诊断——每轮生成结束后触发 `OnLLMResponseEvent`（让 input_state 等插件停止"正在输入"循环）；`OnLLMRequestEvent`/`OnLLMResponseEvent` hook 调用加 5s 超时；`ChatCore timing` 日志输出每轮 ctx/首 token/首发送/总耗时；`chatcore` 指令别名 `c2c`（chat-to-core，对齐 llm2api 的数字缩写方式）/`ctc`。
- `2026-08-02` 调整：动态分段间隔——`segment.interval` 支持公式（`length` 为段字数，可用 `log`/`sqrt`/`abs`/`min`/`max`/`floor`/`ceil` 等），如 `1+length*0.07`（长段停顿更长，模拟打字）；纯数字仍为固定间隔；公式在受限命名空间求值（无 builtins，防注入），非法回退 1 秒。
- `2026-08-02` 修复+集成：表情包收集——根因：OneBot 图片组件只有网络 URL（`image.url`），旧代码按本地文件路径判断导致收集恒 0 个。现 `EmojiStore.collect_from_url` 用 aiohttp 下载入库；`_collect_emoji` 优先 URL、概率筛选（`emoji.collect_probability` 默认 0.6），入库后视觉模型分类。使用：`[[emoji:id]]` 精确 id（也支持按意图搜索），标记触发分段（`a\n[[emoji:01]]bcd` → a、表情、bcd 三段），表情段计时按 length=5。
- `2026-08-02` 补充：引用图片可见——`_resolve_quote_node` 在被引用消息无文本时检查消息链组件，图片标记 `[图片]`、文件标记 `[文件: 名]`，Bot 不再对引用的图片/文件一无所知。
- `2026-08-02` 补充：表情包手动导入——WebUI `emojis/import` 路由（前端「导入表情包」按钮上传图片 → 入库 → 视觉模型自动分类），导入源标记为「手动导入」。
- `2026-08-02` 补充：戳一戳接管——`on_poke` handler（notify/poke 事件）接管 OneBot 戳一戳，走 ChatCore 完整链路（人格/上下文/防复读），stop_event 阻止 vanilla 与 llm_poke 等插件的出戏复读；poke 记为 `（xxx 戳了戳你）` 用户消息并加好感。
- `2026-08-02` 调整：表达风格注入强化——风格块措辞改为「说话时模仿它的句式和长度（短句、口语、像群里的人），但不要复述原文」，缓解群友吐槽的「捧读感」。
- `2026-08-02` 调整：表达风格——学习阶段（入库）由 AI 自主筛选（只保留真正有群特色、值得长期记住的内容，拿不准就不输出）；注入阶段按需提供（summary + 最多 3 条句式 + 3 条黑话，不全量堆砌），且为参考性质、由 AI 自主判断怎么用。
- `2026-08-02` 调整：戳一戳概率触发——`attention.poke_probability`（默认 0.3）基础概率；每次戳记一次硬触发累积活跃度，短时间内多次戳提高概率（封顶 active_max_prob），跳过回复冷却。

