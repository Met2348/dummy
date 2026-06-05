# L07 · Context Management（上下文管理）

## 30 秒

> Context window 有限 (Claude 200k / GPT-4o 128k / Gemini 1.5 1M) 但对话越长越超。
> 4 大策略：**summarize / prune / compress / retrieve**。

## 4 策略

| 策略 | 实现 | 何时 |
|------|------|------|
| **Rolling summary** | 旧 messages → LLM 总结 → 替代 | 长对话 |
| **Sliding window** | 只保留最近 N turns | 简单 |
| **Pruning** | 删低重要度 messages | 中等 |
| **RAG-over-history** | 历史进 vector store，按需 retrieve | 极长对话 |

## Rolling summary

```python
def maintain_context(messages, max_tokens=8000, llm=None):
    if total_tokens(messages) < max_tokens:
        return messages
    # 把旧的 50% 总结成 1 条
    half = len(messages) // 2
    old, recent = messages[:half], messages[half:]
    summary = llm(f"Summarize concisely:\n{old}")
    return [{"role":"system","content":f"[Earlier summary] {summary}"}] + recent
```

## Sliding window 实现

```python
def slide(messages, max_turns=20):
    if len(messages) <= max_turns:
        return messages
    return messages[-max_turns:]
```

简单但丢早期 context。

## 重要度评分 pruning

```python
def score_importance(msg):
    score = 0
    if msg.get("role") == "system": score += 10
    if "name" in msg["content"].lower(): score += 5
    if "?" in msg["content"]: score += 3
    if len(msg["content"]) > 100: score += 1
    return score

def prune(messages, target_tokens=8000):
    scored = sorted(messages, key=score_importance, reverse=True)
    out, tokens = [], 0
    for m in scored:
        if tokens + len(m["content"]) // 4 > target_tokens: break
        out.append(m); tokens += len(m["content"]) // 4
    return sorted(out, key=lambda m: messages.index(m))  # restore order
```

## RAG-over-history

```python
def rag_history(query, history, k=5):
    embeds = [embed(m["content"]) for m in history]
    q_vec = embed(query)
    scored = sorted(zip(embeds, history),
                    key=lambda ev: cos(q_vec, ev[0]),
                    reverse=True)
    return [m for _, m in scored[:k]]
```

只塞 relevant history。

## Compression 极致版

```python
def compress(text, ratio=0.3):
    # 用 small LLM 缩 70%
    return small_llm(f"Compress to {int(ratio*100)}%:\n{text}")
```

[LLMLingua-2 (MS 2024)](https://github.com/microsoft/LLMLingua) 用小模型把 prompt 缩 5-20×。

## 实现 (`context_mgmt.py` 预告)

```python
class ContextManager:
    def __init__(self, strategy="rolling", max_tokens=8000):
        self.strategy = strategy
        self.max_tokens = max_tokens

    def manage(self, messages, llm=None):
        if self.strategy == "rolling": return self._rolling(messages, llm)
        if self.strategy == "slide": return self._slide(messages)
        if self.strategy == "prune": return self._prune(messages)
        if self.strategy == "rag": return self._rag(messages)
        raise ValueError(self.strategy)
```

## 何时选哪个

| 场景 | 推荐 |
|------|------|
| 简单 chatbot | sliding window |
| 长 task agent | rolling summary |
| 5000+ turns | RAG-over-history |
| Token cost 重 | LLMLingua-2 compress |

## 退出条件

- 能列 4 策略
- 能写 rolling summary 函数
- 知道 LLMLingua-2

## 一句话

> Context mgmt 4 法 (summary / slide / prune / RAG) — 不同对话长度选不同，token cost 重时考虑 LLMLingua compress。
