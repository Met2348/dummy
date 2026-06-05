# L12 · Capstone — 3-Layer Memory Chatbot ⭐

## 任务

> 实现一个 chatbot，跑 10 turn 对话，最后能正确 **recall turn 1 中提到的偏好**。

## 3 layer memory

| Layer | 内容 | 实现 |
|-------|------|------|
| Working | 当前 turn ctx | 直接 prompt |
| Episodic | 历次事件 | vector store |
| Semantic | 用户 profile | KG / fact dict |

## 10-turn 测试场景

```
Turn 1:  User: "Hi, I'm Alice. I prefer Anthropic Claude over GPT."
Turn 2:  User: "What's the weather today?"
Turn 3:  User: "Tell me about RAG"
Turn 4:  User: "What's RAG-Fusion?"
Turn 5:  User: "How does ColBERT work?"
Turn 6:  User: "What is GraphRAG?"
Turn 7:  User: "Pros and cons of HippoRAG?"
Turn 8:  User: "Compare BM25 to dense retrieval"
Turn 9:  User: "Which embedding model is best?"
Turn 10: User: "What was my preferred LLM?"  ← 关键 recall test
```

期望 Turn 10 答："Anthropic Claude"。

## 退出条件

- [ ] Turn 1 → semantic memory 抽 "Alice prefers Anthropic Claude"
- [ ] Turn 2-9 全部 episodic store
- [ ] Turn 10 query → recall 正确
- [ ] Profile 持久 (10 turn 后 query 仍命中)

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/agent-memory-context/src'); from capstone_memory_chat import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 预期输出

```markdown
# Memory Chatbot Capstone

## Profile after 10 turns
- name: Alice
- prefers: Anthropic Claude

## Episodic
- 10 episodes stored
- search "weather" → turn 2

## Turn 10 recall
- Q: What was my preferred LLM?
- A: Anthropic Claude
- Source: semantic profile (set in turn 1)

## Verdict: [PASS]
```

## 一句话

> 3-layer memory (working + episodic + semantic) → 10 turn 后仍能 recall turn 1 偏好。
