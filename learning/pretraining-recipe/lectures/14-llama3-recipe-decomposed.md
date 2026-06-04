# L14 · Llama-3 recipe 拆解

> 14 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · Llama-3 概览

```
8B / 70B / 405B
15T token (高于 Chinchilla 200×)
GQA + 128k vocab + 8k → 128k ctx
RLHF 加强
```

## Slide 2 · 数据

```
50% web (FineWeb-Edu 风格)
30% code
20% other (math, books, dialog)
```

显著比 Llama-2 (90% web) 更精细。

## Slide 3 · 数据过滤

```
- 全语言 fastText 分类
- domain quality classifier
- dedup minhash
- educational classifier
- safety filter
```

## Slide 4 · 模型 architecture

```
8B: 32 layer, hidden 4096, n_head 32, GQA kv_head 8
70B: 80 layer, 8192, n_head 64, kv_head 8
405B: 126 layer, 16384, n_head 128, kv_head 16
RoPE base = 500000 (Llama-2 是 10000)
```

## Slide 5 · 训练 settings

```
optimizer: AdamW (0.9, 0.95) wd 0.1
lr: 1.5e-4 (8B), 1.5e-4 (70B), 8e-5 (405B)
warmup: 8000 step
schedule: cosine to 1.5e-5
batch: 16M token
seq_len: 8192 (主), 131072 后期
```

## Slide 6 · 多阶段

```
phase 1: 0-9.8T  general
phase 2: 9.8-15T 高质 + math + code
phase 3: 15T+   long ctx (8k → 128k YaRN)
```

## Slide 7 · 长 ctx 阶段

```
~ 800B token
1) 8k → 16k (RoPE base ×10)
2) 16k → 32k
3) 32k → 64k
4) 64k → 128k
每阶段 ~ 200B token
```

## Slide 8 · 后训练 (post-training)

```
SFT: 1M+ instruction (人工 + 合成)
DPO 多轮 iterative
RM 训练 + rejection sampling
```

## Slide 9 · scaling laws 应用

```
405B 用 3.8e25 FLOPs
Chinchilla 最优: 13T token
实际: 15T (略 over-train)
8B 用 1.5e24, optimal 75B token, 实 15T → 200×
```

## Slide 10 · 训练资源

```
16k H100 × 30 day (405B)
碳排:: 11000 ton CO2 (Meta 报告)
```

## Slide 11 · benchmark

```
8B:  MMLU 66, HumanEval 60, GSM8K 80
70B: MMLU 79, HumanEval 80, GSM8K 93
405B: MMLU 87, HumanEval 89, GSM8K 96
```

## Slide 12 · 工程关键

```
- 数据质量 > 算法
- 长 ctx 单独阶段
- iterative DPO + RM
- safety 整合到管线
```

## Slide 13 · 复刻成本

```
70B Llama-3 复刻:
  数据收集 + tokenization: 1 month, 几百 GPU
  训练: 15T × 6 N FLOPs ≈ 6.3e24 ≈ 7M H100-hour ≈ $14M
```

不 affordable for 个人。

## Slide 14 · takeaway

```
Llama-3 = scaling + data + 长 ctx + RLHF 综合工程
本 capstone 缩小 1000× 但思路一致
```

## 参考
- Llama-3 tech report (Meta 2024)
