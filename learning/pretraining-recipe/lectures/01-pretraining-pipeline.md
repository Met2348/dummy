# L01 · 预训练总流水线

> 14 slides | 40 min ⭐⭐⭐⭐⭐

## Slide 1 · 现代 LLM 预训练全图

```
1. 数据采集 (CommonCrawl / Code / Math)
2. 数据清洗 + 去重 (trafilatura / MinHash / SemDedup)
3. 质量过滤 (FineWeb-Edu / DCLM)
4. tokenization (BPE / tiktoken / SentencePiece)
5. data mixture (40% web + 30% code + ...)
6. shard 化 + memmap
7. 模型架构选 (transformer / MoE / hybrid)
8. 超参选 (μP / WSD)
9. 训练 (FSDP / DeepSpeed)
10. SFT / RLHF 后训练
11. eval + 报告
```

## Slide 2 · 与本系列的关联

```
data-curation     ← L01-L05 (本课 step 1-4)
transformer-deep  ← L07 (本课 step 7 — 选 transformer)
moe-architecture  ← L07 (本课 step 7 — 选 MoE)
ssm-hybrid        ← L07 (本课 step 7 — 选 hybrid)
long-context      ← L08 (本课 — 长 ctx 阶段)
scaling-infra     ← L01-L02 (本课 step 9)
                    ↓
pretraining-recipe ← 拼成一条线
```

## Slide 3 · 资源 vs 模型规模

```
~50M (nanoGPT):  1 GPU 1h
~200M (Phi-tiny): 1 5090 24G  1-2 day
~1B (TinyLlama): 16 A100  ~ 90 day
~7B (Llama-2):   2048 A100  3 weeks
~70B (Llama-3):  1024 H100  数月
```

本系列 capstone 是 270M Phi-tiny @ 5090 1d.

## Slide 4 · 数据规模

```
Chinchilla 1:20:
  270M → 5.4B token
Llama-3 1:1875:
  270M → 500B token (不实际)

折中: 1:50 ~ 1:100 = 13-27B token
```

## Slide 5 · token 预算细分

```
web 70% (FineWeb / DCLM)
code 15% (StarCoder / The Stack)
math 5% (Math-Shepherd / OpenMathInst)
zh 5% (WuDao / CMRC)
书 5% (Books3 / Project Gutenberg)
```

## Slide 6 · 架构选择决策树

```
< 1B + 中文/英文: dense transformer (Phi-3-mini / Qwen-2.5)
1-7B: dense + GQA
7-70B: dense + GQA + maybe MoE
> 70B: MoE 必选 (DeepSeek-V3 / Mixtral 8x22B)
极长 ctx: Mamba/RWKV hybrid (Jamba/Zamba)
```

## Slide 7 · 流程图（详）

```
raw text → clean → dedup → quality filter
        ↓
       tokenizer
        ↓
  [data mix shards] → memmap
        ↓
  [model init μP] ←── [config]
        ↓
   training loop (FSDP)
   ↓ log: loss / grad_norm / MFU
   ↓ ckpt 每 1000 step
        ↓
   eval (HellaSwag / MMLU / lambada / GSM8K)
        ↓
   SFT → RLHF → release
```

## Slide 8 · 工程关键

```
data: shard pickle / numpy memmap, 顺序固定 (resume 友好)
model: μP 让小模型超参直接迁移
training: WSD lr / FSDP / log MFU
eval: tinyMMLU / lm-eval-harness
```

## Slide 9 · 典型 budget 表

| 名 | 参数 | 训 token | GPU-hour | 结果 |
|----|------|---------|----------|------|
| nanoGPT | 50M | 0.3B | 4 (RTX 3090) | OK |
| Phi-1.5 | 1.3B | 30B | 16k (A100) | MMLU 41 |
| Llama-2 7B | 7B | 2T | 184k | MMLU 46 |
| Qwen-2.5 7B | 7B | 18T | ~ 8M | MMLU 75 |
| DeepSeek-V3 | 671B (37B act) | 14.8T | 2.7M | MMLU 88 |

## Slide 10 · 数据 vs 算力 比

```
2020 GPT-3: 数据 << 算力 (over-parameterized)
2022 Chinchilla: 数据 = 算力 (1:20)
2024-: 数据 > 算力 (over-trained, 1:200+)
2026?: 合成数据 + RL 推理 → 数据再扩
```

## Slide 11 · 工业实践 secret sauce

```
- 数据质量 > 算法 (FineWeb-Edu 影响 + 3pp)
- WSD > cosine on small ckpt
- continual pretraining (Phi 6→7)
- 课程学习: 后期增加 code/math
- 长 ctx 单独阶段
```

## Slide 12 · 模型选 Phi 风格 (本 capstone)

```
基础: dense GPT-2 风格
改进 (Phi):
  - WSD lr schedule
  - SwiGLU MLP
  - RMSNorm + 残差 in fp32
  - RoPE
  - GQA
  - 高质量数据
```

## Slide 13 · 270M 是什么规模

```
GPT-2-small:        124M (Wei 2019)
DistilGPT-2:         82M
Phi-1:              1.3B
Phi-1.5:            1.3B
本 capstone:        270M
```

## Slide 14 · 总结

```
预训练 = 数据 + 模型 + 训练 + 评测 拼装
本 topic 跑通 270M 玩具
真训需 1B+ token + 单 5090 1 day
```

## 参考
- Phi-1 (Microsoft 2023)
- TinyLlama paper
- DCLM (Datacomp-LM 2024)
