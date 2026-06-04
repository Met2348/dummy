# L01 · 长上下文 2024-2026 全景

> 20 slides | 60 min ⭐⭐⭐⭐⭐

## Slide 1 · 时间线

```
2023.01  GPT-3.5  4k
2023.05  Llama-1   4k
2023.07  Llama-2   4k
2023.09  YaRN     (RoPE scaling)
2024.04  Llama-3  8k → 128k (RoPE scale 8×)
2024.05  Qwen-2.5 32k → 128k
2024.12  Gemini  2M
2025     Claude / GPT 1M+
```

## Slide 2 · 长 context 价值

```
- RAG: 长文档全文 attention
- code: 整 repo
- 对话: 多日历史
- 推理: 长 CoT
```

## Slide 3 · 技术路线

```
1. RoPE scaling (主流): Llama / Qwen / DS-V3
2. Sliding window: Mistral
3. SSM: Mamba / RWKV
4. Hybrid: Jamba
5. Memory tokens: Infini, MemTrap
```

## Slide 4 · 显存挑战

```
KV cache @ 32k for 70B:
  MHA: 84 GB
  GQA: 0.6 GB     ← OK
  MLA: 0.06 GB    ← 极致
```

## Slide 5 · 训练挑战

```
seq_len 128k 训练:
  attention O(L²) 显存
  单卡跑不动 → Ring Attention
```

## Slide 6 · 评测

```
NIAH (Needle in a Haystack):
  在长文档插入随机句, 问 LLM 该句
RULER: 多种 task 综合长 context
LongBench: 中英多任务
```

## Slide 7 · 现状

```
2k → 32k: 基本解决
32k → 128k: 主流，YaRN + 长 ctx 训练
128k → 1M: 部分模型 (Gemini, Claude)
1M+: Ring attention + 多机
```

## Slide 8 · 本系列路线

```
L02-05 RoPE 全套 (PI / NTK / YaRN / 3D)
L06-07 Ring + Striped Attention
L08 Infini-Attention
L09 FA causal 长
L10-12 评测 + 数据 + 陷阱
L13 Capstone Llama-3.2 1B YaRN 32k
```

## Slide 9-20 · 各方法预览（略）

## 参考
- YaRN, Ring Attention, Infini paper
- Llama-3 / Qwen-2.5 tech reports
