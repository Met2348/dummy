# L01 · Memory Overview — 4 类

## 30 秒

> Agent memory = LLM context (working) + 持久 store (episodic/semantic/procedural)。
> 跨 session 记住用户是 agent 与 chatbot 的关键区别。

## 4 类（认知科学借鉴）

| 类 | 期限 | 内容 | 例 |
|----|------|------|---|
| **Working** | 当下对话 | 即时上下文 | "刚才用户问了 X" |
| **Episodic** | 事件 | 时间戳事件 | "2026-06-05 用户聊了 RAG" |
| **Semantic** | 概念 | 抽象事实 | "用户是 ML 工程师，偏 Anthropic" |
| **Procedural** | 技能 | 怎么做 | "调 search 工具时用 query rewrite" |

## 存哪

| 层 | 位置 |
|----|------|
| Working | LLM context window |
| Episodic | Vector DB / 时间序列 DB |
| Semantic | KG / structured store / Vector DB |
| Procedural | code / prompt templates |

## 4 类记忆运作

```
用户问: "我上次问你的那个 RAG 框架是哪个？"
   ↓
Agent 流程:
  1. Working: 看当前对话 → "我上次" 暗示历史
  2. Episodic: search "RAG" in past conversations → 找到 5 天前对话
  3. Semantic: 用户偏好 = 推 LangChain？还是 LlamaIndex？
  4. Procedural: 记忆 retrieve 的 best practice
```

## Memory 工程的 4 大问题

| 问题 | 解 |
|------|---|
| 何时存？ | LLM 自决 (Letta) / 每 turn 强存 / extract 模式 (Mem0) |
| 何时取？ | RAG-style query / 关键词 / 全 read |
| 何时删？ | TTL / LRU / 容量阈 |
| 何时改？ | extract 新事实 conflict 旧 → 更新 |

## Memory vs RAG

| 维度 | RAG | Memory |
|------|-----|--------|
| 内容 | 静态文档 | 动态用户事件 |
| 写入 | 一次性 index | 持续 append/update |
| 时效 | 文档时间戳 | session 时间戳 |
| 隐私 | 全局 | per-user |
| 更新 | 重 index | 增量 |

## 主流系统

| 系统 | 强在 |
|------|------|
| **Letta (前 MemGPT)** | OS 内存类比、自动 paging |
| **Mem0** | LLM extract / update / retrieve |
| **Zep** | 时间序列 + KG 混合 |
| **LangMem** | LangChain memory module |
| **OpenAI Memory** (ChatGPT) | 商业 SaaS |

## 退出条件

- 能默写 4 类
- 能讲存储位置
- 能区分 Memory 与 RAG

## 一句话

> Agent memory = working (LLM ctx) + episodic (事件) + semantic (事实) + procedural (技能) — 跨 session 记得才叫 agent 而非 chatbot。
