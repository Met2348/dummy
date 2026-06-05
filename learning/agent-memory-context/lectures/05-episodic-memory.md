# L05 · Episodic Memory（事件记忆）

## 30 秒

> Episodic = **按时间排序的对话事件**：
> - "2026-06-05 14:00 用户问了 RAG"
> - "2026-06-04 用户分享了一个论文链接"

```python
@dataclass
class Episode:
    id: str
    user_id: str
    timestamp: float
    actor: str  # "user" / "agent"
    content: str
    summary: str = ""
    embedding: list[float] | None = None
```

## 索引策略

| 维度 | 用 |
|------|---|
| Vector | 语义检索 ("what did we discuss about X") |
| Time | 时序检索 ("yesterday's chat") |
| User | 隔离 (per-user retrieval) |
| Topic | tag-based |
| Sentiment | "frustration" episodes |

## 检索

```python
def episodic_retrieve(query, user_id, time_filter=None, k=5):
    candidates = store.search(
        embed(query),
        k=k*3,
        filter={"user_id": user_id, "timestamp": time_filter},
    )
    return rerank(candidates, query)[:k]
```

## 总结策略（防 storage 爆炸）

| 策略 | 何时 |
|------|------|
| Per-turn raw | 始终存原文 |
| Per-session summary | session 结束 LLM 总结 |
| Daily digest | 每天总结当天 |
| Topic merge | 同 topic 合并 |

## Zep 风格（时间序列 + KG）

[Zep](https://github.com/getzep/zep) 把 episodic memory 做了 enterprise 化：
- 自动 summary
- KG 抽取 (entity + relation)
- TTL + retention policy
- 多租户

## 实现 (`episodic_memory.py` 预告)

```python
class EpisodicMemory:
    def __init__(self):
        self.episodes: list[Episode] = []

    def add(self, user_id, actor, content):
        ep = Episode(
            id=f"ep{len(self.episodes)}",
            user_id=user_id,
            timestamp=time.time(),
            actor=actor,
            content=content,
            embedding=hash_embed(content),
        )
        self.episodes.append(ep)

    def search(self, query, user_id, k=5,
               time_from=None, time_to=None):
        q_vec = hash_embed(query)
        scored = []
        for ep in self.episodes:
            if ep.user_id != user_id: continue
            if time_from and ep.timestamp < time_from: continue
            if time_to and ep.timestamp > time_to: continue
            scored.append((ep, cosine(q_vec, ep.embedding)))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
```

## "what did we discuss" 模式

最常见 episodic 用：
- "what did we discuss yesterday?"
- "what did I tell you about my project?"
- "remind me what I asked last week"

→ time + semantic + user filter 三件套。

## 退出条件

- 能默写 Episode dataclass
- 能列 4 索引维度
- 会写 search with time filter

## 一句话

> Episodic memory = (time + user + content) tuple 序列 + vector index + summary 防爆 — chatbot 的"记忆相册"。
