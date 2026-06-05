# L03 · Chunking 策略

## Chunking = "切对地方 retrieve 才有用"

| 策略 | 边界 | 优势 | 缺点 |
|------|------|------|------|
| **Fixed** | N 字符/token | 简单 | 切句子中间 |
| **Fixed + overlap** | N 字符 + M 重叠 | 跨边界保留 | 冗余 |
| **Sentence** | 句号 | 自然 | 句子长度差异大 |
| **Semantic** | 段间相似度 dip | 主题完整 | 慢、贵 |
| **Proposition** | LLM 拆原子命题 | 最细粒度 | 贵 |
| **Agentic** | LLM 决策每块边界 | 上下文适配 | 最贵 |

## Fixed-size 模板

```python
def fixed_chunk(text, size=500, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks
```

## Semantic chunking（Kamradt 2024）

```
for each consecutive sentence pair:
    sim = cos(embed(s_i), embed(s_{i+1}))
breakpoint = 5th percentile of sim
split at breakpoint
```

→ "话题切换"处切，保持主题连贯。

## Proposition chunking（Chen 2023 "Dense X Retrieval"）

> 用 LLM 把文档拆成原子命题（atomic propositions），每个命题独立 retrieve。

例：
```
原文：「Claude 由 Anthropic 开发。Anthropic 于 2021 创立。」
命题 1：Claude 由 Anthropic 开发。
命题 2：Anthropic 于 2021 创立。
```

## Contextual chunking (Anthropic 2024)

```
每个 chunk 前加一段 LLM 写的"上下文"：
"该块来自 <doc name> 的 <section>，讨论 <topic>。"

→ 让 chunk 自带 doc 上下文，跨 chunk 不再瞎猜。
```

实测 retrieval failure rate 降 49%（Anthropic 2024.09）。

## Chunk size 选择

| 嵌入模型 | 推荐 size |
|---------|-----------|
| text-embedding-3-small | 256-512 |
| text-embedding-3-large | 512-1024 |
| BGE-large | 512 |
| Voyage-3 | 1024 |
| **经验**：太短缺上下文，太长稀释信号 → **平均段落级** | 400-800 |

## 一图选型

```
速度优先   → fixed
质量优先   → contextual
精度优先   → proposition
关系优先   → graphrag chunk = node
```

## 退出条件

- 能列 6 chunking 策略
- 会写 fixed_chunk
- 知道 Contextual chunking 来源（Anthropic 2024）

## 一句话

> Chunk 切不好，RAG 必跑歪 —— fixed 入门，contextual / proposition 进阶。
