# Topic 5: Agent Memory & Context（记忆 + 上下文）

> Module 7 第 5 专题 · 12 lectures · ~12h
>
> Letta (MemGPT) / Mem0 / 4 memory 类 / 上下文管理 / prompt cache

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Memory 分类 (working / episodic / semantic / procedural) | (intro) |
| L02 | **Letta** (前 MemGPT) — OS-inspired paging | `letta_mock.py` |
| L03 | **Mem0** — extract / update / retrieve | `mem0_mock.py` |
| L04 | Vector DB 选型 (Pinecone/Weaviate/Chroma/Qdrant/Milvus/pgvector) | `vector_store.py` |
| L05 | **Episodic memory** | `episodic_memory.py` |
| L06 | **Semantic memory** + KG | `semantic_memory.py` |
| L07 | Context mgmt (summary / pruning / compression) | `context_mgmt.py` |
| L08 | **Prompt caching** (Anthropic / OpenAI) | `prompt_cache.py` |
| L09 | KV-cache mgmt + PagedAttention | `kv_cache_mock.py` |
| L10 | 长对话策略 | `long_conv.py` |
| L11 | 个性化 memory | `personalization.py` |
| L12 | **Capstone**: 3-layer memory chatbot | `capstone_memory_chat.py` |

## Tags

- `agent-memory` — Module 7 第 5 专题

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"; python learning/agent-memory-context/src/tests/test_memory.py
```

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'learning/agent-memory-context/src'); from capstone_memory_chat import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 4 memory 类对照

| 类 | 期限 | 例 |
|----|------|---|
| Working | 当下对话 | 当前 turn 上下文 |
| Episodic | 事件 | "上周三聊了 RAG" |
| Semantic | 概念 | "用户偏爱 Anthropic 模型" |
| Procedural | 技能 | "如何调 search tool" |

## 一句话

> 4 memory 类 + Letta paging + Mem0 extract-update — 让 agent 记住用户跨 session 的偏好。
