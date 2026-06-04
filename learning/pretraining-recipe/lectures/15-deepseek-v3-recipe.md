# L15 · DeepSeek-V3 recipe 拆解

> 14 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · DeepSeek-V3 概览

```
671B (37B active)  MoE
14.8T token
MTP (Multi-Token Prediction) 训练
~$5.6M (相比 Llama-3 节省 5×)
```

## Slide 2 · 架构

```
n_layer: 61
hidden: 7168
attn: MLA (Multi-head Latent Attention)
MoE: aux-loss-free routing, top-8 + shared expert
n_expert: 256 + 1 shared
```

## Slide 3 · MoE 创新

```
aux-loss-free routing:
  - 不再 aux loss 平衡 expert
  - 用 bias 项动态调整
  - 训练更稳, 性能更好
```

## Slide 4 · MTP 训练

```
n_predict_head: 2 (vs 1 default)
loss = CE(t+1) + 0.3 × CE(t+2)
两 head 共享 trunk
推理时 MTP 当 spec decoding draft
```

## Slide 5 · 数据

```
14.8T token mix:
  30% en (FineWeb-Edu-like)
  30% zh
  20% code
  10% math (Math-Pile)
  10% multilingual
```

## Slide 6 · 训练 settings

```
optimizer: AdamW (0.9, 0.95)
lr peak: 2.4e-4 (37B active)
schedule: cosine to 2.4e-5
batch: 16M token (active)
n_step: 14.8T / 16M ≈ 925k
```

## Slide 7 · 通信优化

```
DualPipe (DeepSeek 自创):
  forward/backward 双流水
  通信/计算 重叠 95%+
节省 25% 训练时间
```

## Slide 8 · 多阶段

```
phase 1: 通用 14.8T
phase 2: long ctx 8k → 32k → 128k YaRN
phase 3: SFT
phase 4: RL (GRPO + RM + Rule-based)
```

## Slide 9 · benchmark

```
MMLU: 88.5
GSM8K: 89.3
MATH: 90.2
HumanEval: 82.6
NIAH @ 128k: 100%
```

## Slide 10 · 推理成本

```
37B active → vLLM 4× H100 即可
context: 128k native
比 Llama-3-405B 便宜 10×
```

## Slide 11 · cost saving

```
- FP8 训练 (NV TransformerEngine)
- DualPipe + EP 高效
- MoE 37B active (vs 405B dense)
- MTP 2× draft
合计 5× cost reduction
```

## Slide 12 · 与本 capstone 的差距

```
本 capstone: 270M dense bf16
DeepSeek-V3:
  670B MoE + fp8 + MTP + MLA + DualPipe
  本 capstone 完全无 MoE / fp8 / MTP / Dual
小项目坚持 dense + bf16 即可
```

## Slide 13 · 学习要点

```
- MLA 看 transformer-deep L*
- aux-loss-free 看 moe-architecture L*
- MTP 看本 lecture
- DualPipe 看 scaling-infra L06
- 长 ctx 看 long-context L*
拼起来 = DeepSeek-V3
```

## Slide 14 · takeaway

```
DeepSeek-V3 = MoE + MLA + MTP + DualPipe + FP8 综合工程
开源最强 (2025)
本 capstone 是其千分之一
```

## 参考
- DeepSeek-V3 tech report (2024.12)
