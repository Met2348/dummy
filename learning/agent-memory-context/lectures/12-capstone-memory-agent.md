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
python learning/agent-memory-context/src/capstone_memory_chat.py
```

> （历史版本用过 CWD 依赖的 `python -c "import sys; sys.path.insert(0,'learning/agent-memory-context/src'); ..."`
> 一行流；已改为直接脚本调用，效果等价但不再依赖"当前目录=repo-root"这个隐藏前提。脚本无 argparse，
> Python 会把脚本自身所在目录插入 `sys.path[0]`，故 `from common import ...` 等裸导入照样能解析。）

## 预期输出

先跑内置 `_self_test`（真断言：`recall_source == "semantic_profile"` 而非只看 verdict 字符串），
再打印完整 markdown 报告。以下是实测逐字输出（`episodic_count` 计的是 user+agent 两侧共 20 条事件，
不是 10）：

```text
[OK] capstone_memory_chat._self_test passed (recall from turn 1 after 9 unrelated turns)
```

```markdown
# 3-Layer Memory Chatbot Capstone

## Profile after 10 turns
- name: Alice
- preference[llm]: Anthropic Claude

## Episodic: 20 events stored

## Turn-by-turn
| # | User | Agent | Source |
|---|------|-------|--------|
| 1 | Hi, my name is Alice. I prefer Anthropic | (mock answer for: Hi, my name is Alice.  | default |
| 2 | What's the weather today? | (mock answer for: What's the weather tod | default |
| 3 | Tell me about RAG. | (mock answer for: Tell me about RAG.) | default |
| 4 | What's RAG-Fusion? | (mock answer for: What's RAG-Fusion?) | default |
| 5 | How does ColBERT work? | (mock answer for: How does ColBERT work? | default |
| 6 | What is GraphRAG? | (mock answer for: What is GraphRAG?) | default |
| 7 | Pros and cons of HippoRAG? | (mock answer for: Pros and cons of Hippo | default |
| 8 | Compare BM25 to dense retrieval. | (mock answer for: Compare BM25 to dense  | default |
| 9 | Which embedding model is best? | (mock answer for: Which embedding model  | default |
| 10 | What was my preferred LLM? | Your preferred LLM is Anthropic Claude. | semantic_profile |

## Verdict: [PASS]
- Final recall: Your preferred LLM is Anthropic Claude.
- Recall source: semantic_profile
```

## 一句话

> 3-layer memory (working + episodic + semantic) → 10 turn 后仍能 recall turn 1 偏好。
