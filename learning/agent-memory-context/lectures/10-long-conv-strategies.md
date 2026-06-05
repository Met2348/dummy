# L10 · 长对话策略

## 长对话三大挑战

| 挑战 | 表现 |
|------|------|
| Context window | 1M token 也总有上限 |
| Token 成本 | 每 turn 重传旧消息 |
| 信息衰减 | 早期 fact 后期被淹没 |

## 综合方案 4 层

```
Layer 1: Recent window (最近 N turn 原文)
Layer 2: Rolling summary (旧 N turn 总结)
Layer 3: Semantic memory (核心 facts)
Layer 4: Episodic store (按需 retrieve)
```

## 每 turn 构 prompt 模板

```
SYSTEM: <role + persona>

CORE FACTS (semantic):
  User name: Alice
  User preference: Anthropic Claude
  Project: RAG-legal-agent

CONTEXT SUMMARY (rolling):
  In earlier conversation, user asked about ColBERT,
  we discussed RAG strategies, ...

EPISODIC (retrieved by query):
  [2026-06-01] User mentioned LangChain difficulty
  [2026-06-03] User shared paper link

RECENT (last 5 turns raw):
  User: ...
  Agent: ...

CURRENT TURN:
  User: <new question>
```

## 何时触发 summary

| 条件 | 行 |
|------|---|
| Recent turns > 30 | summarize first half |
| Token > 50% window | aggressive summarize |
| Session end | full session summary → episodic |
| Topic change | per-topic summary |

## 实测数字（Letta blog 2025）

| 设置 | 性能 |
|------|------|
| Pure sliding window | 200 turn 后忘 early facts |
| + summary | 1k turn 仍可 recall |
| + episodic retrieve | 10k turn 仍 work |
| + semantic | 跨 session 仍记 |

## 失败模式

| 模式 | 解 |
|------|---|
| Summary 漏关键 | LLM-judge eval summary 质量 |
| Episodic retrieve 错 | reranker + time filter |
| Semantic conflict | conflict resolver |
| Token 总爆 | budget cap + emergency compress |

## 实现 (`long_conv.py` 预告)

```python
class LongConversation:
    def __init__(self, recent_n=10, summary_threshold=20):
        self.recent = []
        self.summary = ""
        self.recent_n = recent_n
        self.summary_threshold = summary_threshold

    def add_turn(self, role, content, llm=None):
        self.recent.append({"role":role,"content":content})
        if len(self.recent) > self.summary_threshold:
            to_summarize = self.recent[:-self.recent_n]
            new = self._summarize(to_summarize, llm)
            self.summary = self._merge_summary(self.summary, new)
            self.recent = self.recent[-self.recent_n:]

    def build_prompt(self, core_facts: dict, query: str) -> str:
        return f"""SYSTEM
CORE: {core_facts}
SUMMARY: {self.summary}
RECENT: {self.recent}
USER: {query}"""
```

## 退出条件

- 能默写 4 层综合方案
- 能讲 4 触发 summary 条件
- 能写 add_turn 函数

## 一句话

> 长对话 4 层组合 (recent + summary + semantic + episodic) — Letta 在 10k+ turn 仍 work。
