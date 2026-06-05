# L04 · Embedding 模型横评

## 2025-2026 主流 embedding 模型

| 模型 | 团队 | dim | MTEB 平均 | 特点 |
|------|------|----:|----------:|------|
| **text-embedding-3-large** | OpenAI | 3072 | 64.6 | 商业默认 |
| text-embedding-3-small | OpenAI | 1536 | 62.3 | 便宜 |
| **voyage-3** | Voyage AI | 1024 | 71.5 | Anthropic 推荐 |
| voyage-3-lite | Voyage | 512 | 68.0 | 小模型 |
| **BGE-large-en-v1.5** | BAAI | 1024 | 64.2 | 开源默认 |
| BGE-M3 | BAAI | 1024 | 65 | 多语言 + 多模态 |
| **E5-mistral-7b-instruct** | Microsoft | 4096 | 66.6 | LLM-based |
| **SFR-Embedding-Mistral** | Salesforce | 4096 | 67.6 | 微调 E5 |
| **gte-large-en-v1.5** | Alibaba | 1024 | 65.4 | 开源 |
| **NV-Embed-v2** | NVIDIA | 4096 | 72.3 | 2024 SOTA |
| **stella-en-1.5B** | 2024 | 8192 | 71.2 | 1.5B 参数 |
| **mxbai-embed-large** | Mixedbread | 1024 | 64.7 | 开源新秀 |

## MTEB benchmark（Massive Text Embedding Benchmark）

- HuggingFace + Cohere 维护
- 56 dataset / 8 task：classification / clustering / reranking / retrieval / STS / summarization
- 排行榜：https://huggingface.co/spaces/mteb/leaderboard

## API vs 本地

| 维度 | API (OpenAI/Voyage) | 本地 (BGE/E5) |
|------|---------------------|---------------|
| 启动 | 即用 | 需 GPU |
| 成本 | $0.13-1.3 / 1M tok | 电费 |
| 隐私 | 上传 | 本地 |
| 延迟 | 100-500ms | 5-50ms (GPU) |
| 维度 | 1.5k-4k | 1k-4k |

## Matryoshka embedding（Aditya 2024）

- OpenAI text-embedding-3 / voyage / Nomic 支持
- 训出来的向量可以**截断**前 N 维仍保有效（套娃式）
- 优势：存储 / 检索可缩放（先粗筛低维，再精排高维）

## 维度选择

| dim | 存储 (1M doc) | 用 |
|-----|------------:|---|
| 256 | ~1 GB | 大规模粗筛 |
| 768 | ~3 GB | 经济型 |
| 1024 | ~4 GB | 默认 |
| 1536 | ~6 GB | OpenAI |
| 3072 | ~12 GB | 高精度 |

## 选型决策树

```
快速 PoC + 不想搭 GPU
  → text-embedding-3-small (OpenAI)

中等规模 + 关心成本
  → BGE-large-en-v1.5 (本地 GPU)

大规模 + 高精度
  → voyage-3 (Anthropic 推荐)

多语言
  → BGE-M3 / voyage-multilingual

最 SOTA
  → NV-Embed-v2 / stella-en-1.5B
```

## 我们 mock 实现

```python
def hash_embed(text: str, dim: int = 64) -> list[float]:
    vec = [0.0] * dim
    for tok in text.lower().split():
        vec[hash(tok) % dim] += 1.0
    norm = (sum(v*v for v in vec)) ** 0.5
    return [v / (norm + 1e-9) for v in vec]
```

bag-of-words 投影 = "无 GPU 教学 embedding"。

## 退出条件

- 能默写 4 商业 + 4 开源模型
- 知道 MTEB 是评测平台
- 知道 Matryoshka 是套娃 truncate

## 一句话

> Embedding 是 RAG 的"语义指南针" —— voyage-3 / BGE-large / text-embedding-3 三选一够 90% 用例。
