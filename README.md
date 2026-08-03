# astrbot_plugin_chatcore

一个更聪明的 AstrBot 聊天核心插件，改善原生日聊体验：注意力驱动的群聊活跃回复、智能分段流式输出、全局记忆、隐性意图分析、撤回取消等。

> 本插件不管理任何模型接口。模型全部复用 AstrBot 自身已配置的提供商：先在 AstrBot「模型提供商」页面配置好聊天/识图/Embedding 提供商，再到插件配置里选择即可。

## 功能特性

- **注意力机制**：群聊不依赖「被 @」才能回复。根据活跃度动态概率插话，被 @/reply/唤醒前缀时硬触发必回。
- **智能分段流式输出**：AI 用自带分段符（默认 `---` 独占一行）自我分段，逐段发送；无分段时按最大字符兜底切段。
- **全局记忆与自我学习**：基于 Embedding + 余弦相似度的自建向量记忆库（纯 JSON 存储，无外部依赖），跨群可选共享，每次回复自动召回最相关记忆。
- **隐性意图分析**：后台定时分析群内话题，命中「隐性提及/邀请机器人」时提升回复概率。
- **撤回取消**：用户撤回触发消息后，立即停止正在生成的回复，不打断其他对话。
- **主动动作**：允许 AI 用 `[[at:昵称]]` / `[[reply:昵称]]` 主动 @ 或回复群成员。
- **聊天记录注入**：自动读取 AstrBot 持久化的聊天记录，注入当前会话历史与群聊中该发送者的私聊记忆。
- **上下文 LLM 摘要**：更早历史自动交给独立摘要模型做滚动摘要，省 token 且保留关键信息。
- **引用回溯**：用户引用消息时，AI 能读到被引用的原内容（含连锁引用链）。
- **接管模式**：接管 AstrBot 原生 LLM 链路，避免重复回复。

## 安装

1. 在 AstrBot 管理后台「插件市场」/「扩展」中安装本插件（或手动放入 `data/plugins/` 并重启）。
2. 需要 AstrBot `>= 4.26`。

## 快速开始

1. **配置提供商**：打开 AstrBot「模型提供商」页面，配置好聊天模型、识图模型（可选，支持视觉输入）、Embedding 模型。
2. **插件配置**：进入插件配置，在「模型提供商」分组中选择：
   - **聊天主模型**：下拉选择 AstrBot 已配置的聊天提供商。
   - **识图模型**：可选，消息含图片时转文字描述；留空复用聊天模型。
   - **摘要模型**：可选，上下文压缩专用；留空复用聊天模型。
   - **Embedding 提供商 ID**：填写 AstrBot 中 Embedding 提供商的 ID（用于全局记忆检索）。
3. 保存配置并重启插件（或重启 AstrBot）。
4. 运行 `/chatcore` 可查看运行状态。

## 配置说明

所有配置在 AstrBot 管理后台插件配置中可视化完成（`description` 为标题，`hint` 为说明）。

### chat — 聊天主设置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 插件总开关 |
| `takeover` | `true` | 接管 AstrBot 原生 LLM 链路，避免重复回复 |
| `private_force_reply` | `true` | 私聊必回，不参与概率 |
| `group_enabled` | `true` | 群聊启用，关闭后群消息只记录不回复 |
| `markers_enabled` | `true` | 允许 AI 用 `[[at:昵称]]` / `[[reply:昵称]]` 主动 @ 或回复群成员 |

### attention — 注意力机制

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `bubble_base_prob` | `0.02` | 群聊无人搭理时 AI 自发插话的概率（建议 0.01~0.03） |
| `active_max_prob` | `0.3` | 有人与 AI 对话时概率上限 |
| `decay_minutes` | `10.0` | 超过该时长无人维持，概率指数降回基线 |
| `hard_trigger_force` | `true` | 被 @/reply/唤醒前缀 时是否必定回复 |
| `hard_trigger_boost` | `0.1` | 每次被 @/reply 额外提升的概率 |
| `wake_prefix` | `["bot","ai","机器人"]` | 不带 @ 直接喊也能触发的前缀列表 |

### segment — 智能分段

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `delimiter` | `\n---\n` | AI 自我分段符（独占一行的分隔符，如 `---`） |
| `escape_char` | `\` | AI 想输出分段符本身时在其前加该字符 |
| `interval` | `1.0` | 分段发送间隔(秒) |
| `max_segment_chars` | `600` | 单段最大字符数，0 为不限制 |

### context — 智能上下文

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `recent_count` | `10` | 近 N 条完整原文直接喂给 AI |
| `history_count` | `30` | recent 之外的更早消息保留条数 |
| `old_msg_chars` | `40` | 每条历史消息压缩后的最大字符数 |
| `llm_summarize` | `true` | 是否用 LLM 对更早历史做滚动摘要（替代逐条截断） |
| `quote_max_depth` | `15` | 引用回溯的最大嵌套深度 |

### history — 聊天记录注入

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `inject_enabled` | `true` | 自动注入 AstrBot 持久化聊天记录 |
| `max_messages` | `10` | 每次注入的历史消息条数上限 |
| `max_chars` | `1200` | 每次注入的历史文本总字符上限 |

### providers — 模型提供商

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `chat.provider_id` | 空 | 聊天主模型，下拉选择 AstrBot 已配置的提供商 |
| `chat.multimodal` | `false` | 聊天模型是否原生支持识图；开启后图片直接传给聊天模型 |
| `vision.provider_id` | 空 | 识图模型（需支持视觉输入），留空复用聊天模型 |
| `summary.provider_id` | 空 | 上下文压缩摘要专用模型，留空复用聊天模型 |
| `reply_decision.provider_id` | 空 | 首段生成后异步判断自动 reply/@ 的模型，留空关闭 |
| `reply_decision.timeout_seconds` | `8` | 延迟判断最长等待时间，不阻塞首段发送 |
| `embedding.provider_id` | 空 | Embedding 提供商 ID，用于全局记忆向量检索 |

上下文中的当前聊天和压缩历史统一使用用户 `<message uid="..." nickname="...">...</message>`；机器人使用
`<message from="yourself">...</message>`，并有 XML 注释标识自身消息。记忆、画像和外部历史位于 `<reference>` 块中。工具采用按需 schema：系统提示
只注入极短工具目录，需要时输出 `[[tools]]`，下一轮才发送完整 schema。

### memory — 全局记忆与自我学习

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 记忆开关 |
| `shared_across_groups` | `true` | 各群聊的数据/记忆互相共享 |
| `max_recall` | `5` | 每次请求最多召回的记忆片段数 |

### implicit — 隐性意图分析

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 隐性分析开关 |
| `interval_minutes` | `30` | 分析最小间隔(分钟)，避免频繁调用 |
| `prob_boost` | `0.05` | 命中隐性相关话题后的概率提升 |
| `provider_id` | 空 | 隐性分析模型提供商，与聊天模型相互独立；留空则关闭 |
| `prompt` | 内置 | 分析提示词，留空使用默认 |

### recall_cancel — 撤回取消

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 用户撤回触发消息后，停止 AI 回复 |

## 命令

- `/chatcore` — 查看插件运行状态（模型、记忆、隐性分析、主动动作、撤回取消统计、正在进行的对话数）。

## 常见问题

### 群聊里没人 @ 机器人，为什么 AI 也会回？

这是注意力机制的设计：`attention.bubble_base_prob` 控制无人搭理时自发插话的概率（默认 2%），`active_max_prob` 封顶活跃态概率。不想要就都设为 `0`，同时保留 @/唤醒前缀的硬触发。

### AI 忘了一句话里分段怎么办？

`segment.max_segment_chars`（默认 600）会在 AI 忘了分段时按字符兜底切段；也可以调小分段符的复杂度或看 AI 是否遵守了 `delimiter` 写法（建议用一个独占一行的简单标记，如 `---`；若要用换行包围标记，直接输入真实换行符即可，真实换行符不含反斜杠，不要写成 `\n` 字面量）。

### 记忆似乎没生效？

确认「模型提供商」分组的 Embedding 提供商 ID 填对，且 AstrBot 中该提供商是可用的 Embedding 提供商（`/chatcore` 可查看全局记忆是否启用）。

### 撤回后 AI 还在继续回？

确认 `recall_cancel.enabled` 开启（默认开启）。撤回取消依赖 OneBot（aiocqhttp）的撤回上报，请确认平台已开启撤回事件上报。

## 项目结构

```
astrbot_plugin_chatcore/
├── main.py          # 插件入口、消息接管、命令、撤回取消
├── llm.py           # 通过 AstrBot 提供商体系访问模型（LLMProvider / EmbeddingAdapter）
├── context.py       # 上下文记录、压缩历史、缓存友好的消息组装
├── attention.py     # 注意力概率计算
├── segmentation.py  # 智能分段、流式发送
├── memory.py        # 自建向量记忆库（embedding + 余弦相似度，JSON 存储）
├── actions.py       # 主动动作标记解析
└── _conf_schema.json  # 配置项定义
```

更多设计细节见 `arch.md`。

## License

MIT
