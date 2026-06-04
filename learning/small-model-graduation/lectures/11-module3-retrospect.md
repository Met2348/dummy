# L11 · Module 3 全程回顾

> 16 slides | 50 min ⭐⭐⭐⭐⭐

## Slide 1 · 8 topic 总览

```
T1 data-curation     CommonCrawl → token
T2 transformer-deep  RoPE/MHA/MQA/GQA/MLA/FA/SWA
T3 moe-architecture  GShard/Switch/EC/Mixtral/aux-free
T4 ssm-hybrid        S4/Mamba/RWKV/RetNet/Jamba/Zamba
T5 long-context      PI/NTK/YaRN/Ring/Infini/NIAH
T6 scaling-infra     Chinchilla/FSDP/Megatron/vLLM/SGLang
T7 pretraining-recipe Phi-tiny 270M end-to-end
T8 small-model-graduation (本) 五部曲对照
```

合 118 method, 107 lectures, ~ 122h.

## Slide 2 · 学到的核心问题

```
1. 数据怎么造?
2. transformer 怎么改更高效?
3. 单 dense vs MoE 何时选?
4. SSM 何时优于 transformer?
5. 长 ctx 怎么扩到 100k+?
6. 1B vs 7B vs 70B 训练怎么 scale?
7. SOTA 模型从 0 怎么训出来?
8. 五部曲贡献怎么量化?
```

## Slide 3 · 关键模型时间线

```
2017 transformer (Attention is all)
2018 GPT-1, BERT
2019 GPT-2, T5
2020 GPT-3, ZeRO, Megatron
2021 LLaMA-1, PaLM, FlashAttention
2022 Chinchilla, ChatGPT, Mixtral
2023 GPT-4, LLaMA-2, FA2, Mamba, RWKV
2024 LLaMA-3, GPT-4o, DeepSeek-V3, Phi-3, Qwen-2.5
2025 R1, o1, Claude 3.7, Kimi k1.5
2026 (本课时间)
```

## Slide 4 · 五部曲公式

```
p(y | x ; θ_data, θ_arch, θ_weight, φ_ctx)

θ_data:  从原始 text 学到的 distribution
θ_arch:  transformer 等架构
θ_weight: 训出来的具体参数
φ_ctx:    ctx 处理机制 (RoPE/SSM/sparse)
```

## Slide 5 · 关键 trade-off 汇总

| 选择 | + | - |
|------|---|---|
| dense vs MoE | 推理快 vs 参数 ↑↑ | 推理慢 vs 容量 ↑↑ |
| MHA vs GQA | 容量 vs KV cache |
| trans vs SSM | 强 vs 长 ctx 友好 |
| 短 ctx 强 vs 长 ctx 弱 | 多 train | annealing 阶段 |
| over-train vs Chinchilla | 推理便宜 vs 训便宜 |

## Slide 6 · 推荐路线 (单 GPU 个人)

```
小项目 (< 1B):
  Phi-tiny 风格 dense + GQA + RoPE + SwiGLU
  WSD lr schedule
  bf16 + grad ckpt
  Cosmopedia 类合成数据

不要碰 (< 1B):
  MoE (路由 overhead 占主导)
  TP/PP (单卡)
  fp8 (5090 一般)
```

## Slide 7 · 推荐路线 (中等 7B)

```
Llama-3 风格:
  GQA + RoPE base 500000
  15T token over-train
  长 ctx 分阶段
  WSD 或 cosine
  FSDP + grad ckpt
  H100 8 卡至少
```

## Slide 8 · 推荐路线 (大 70B+)

```
DeepSeek-V3 风格:
  MoE + MLA + MTP
  fp8 训练 + DualPipe
  数据 30% en + 30% zh + ...
  长 ctx YaRN
  TP=8 + PP=8 + EP=N
```

## Slide 9 · 自己最重要的 5 个 takeaway

```
1. 数据质量 > 算法 (5pp 来源)
2. Phi-tiny 风格 (Pre-RMSNorm + RoPE + GQA + SwiGLU)
3. WSD lr (易 resume, 优 cosine)
4. 长 ctx 是独立阶段 (YaRN + 长 doc)
5. capstone 5 ckpt 对照 让贡献量化
```

## Slide 10 · 不重要的 (可后置)

```
- exact ZeRO stage 数学
- TP 具体 op-level 切分
- Megatron-LM 源码
- vLLM 内部调度
- FP8 数值细节
```

知其然即可, 用工具即可.

## Slide 11 · 仍未学到 (留下一程)

```
- SFT 指令微调技术细节 → Module 4
- RLHF / DPO / GRPO → Module 5
- 推理 RL (R1) → Module 6
- 多模态 → Module 7
- agent → Module 7+
```

## Slide 12 · 推荐扩展阅读

```
- The Llama 3 Herd of Models
- DeepSeek-V3 Technical Report
- Mamba: Linear-Time Sequence Modeling
- YaRN: Efficient Context Window Extension
- vLLM paper
- Chinchilla scaling laws
- Phi-3 Technical Report
- Qwen-2.5 Technical Report
```

## Slide 13 · 自查能力

```
[ ] 能解释 RoPE 数学 + 实现一遍
[ ] 能解释 MoE aux-loss-free 路由
[ ] 能解释 YaRN 公式
[ ] 能跑 Phi-tiny 训练 (5090)
[ ] 能解释 FSDP / Megatron 区别
[ ] 能用 vLLM 部署 + 量化
[ ] 能选 model size + token budget
```

## Slide 14 · 自己练手项目

```
1. 从头训 nanoGPT 124M @ Wikitext (~ 4h)
2. 从 ckpt E 做 SFT (Alpaca) (~ 4h)
3. 从 SFT 做 DPO (Anthropic-HH) (~ 6h)
4. 部署到 vLLM + 量化 (1h)
5. 写自己的 tech report
```

## Slide 15 · Module 3 → Module 4 桥梁

```
ckpt E (Phi-tiny 270M Pretrained) 是 Module 4 起点
SFT/RLHF/RL 都从此 ckpt 出发
本 capstone tag: 造改-graduation ⭐⭐⭐⭐⭐
```

## Slide 16 · 总结

```
Module 3 = "造大模型" 完整训练
8 topic, 108 lecture, 118 方法, ~122h
本 capstone 把全部串起来跑通
```

## 参考
- 系列内 Topic 1-7
