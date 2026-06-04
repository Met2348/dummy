# L01 · 毕业总览 — Module 3 五部曲

> 14 slides | 40 min ⭐⭐⭐⭐⭐

## Slide 1 · 五部曲回顾

```
p(y | x ; θ_data, θ_arch, θ_weight, φ)

θ_data    造  data-curation     Topic 1
θ_arch    造  transformer-deep   Topic 2
                moe-architecture  Topic 3
                ssm-hybrid        Topic 4
ctx       造/扩 long-context       Topic 5
θ_infra    造  scaling-infra     Topic 6
recipe    造  pretraining-recipe Topic 7
                ↓
   小模型从头到尾走一遍 = 毕业 capstone (本)
```

## Slide 2 · 毕业 capstone 目标

```
任务: 从 raw text 到 trained 小模型, 全流程
模型: Phi-tiny 270M (本系列 Topic 7 capstone 基础)
数据: 用 Topic 1 的 pipeline (FineWeb + Cosmopedia + code)
评测: tinyMMLU + tinyHellaSwag + GSM8K-tiny
对照: 5 个 ckpt (vanilla / data-better / arch-better / long-ctx / full)
```

## Slide 3 · 5 对照路径

```
ckpt A: vanilla GPT-2-style 124M (baseline)
ckpt B: + 高质数据 (Cosmopedia)
ckpt C: + Phi 架构 (GQA + SwiGLU + RoPE)
ckpt D: + 长 ctx 阶段 (YaRN)
ckpt E: 全部 (= Topic 7 capstone)
```

## Slide 4 · 五部曲变量影响

```
benchmark 提升来源:
  data:      +5pp MMLU
  arch:      +1pp MMLU, +20% throughput
  long_ctx:  +0pp MMLU, NIAH 0→80%
  composed:  +6pp MMLU
```

## Slide 5 · 资源预算

```
单 5090 24G:
  ckpt A 124M × 0.3B token: 4h
  ckpt B 124M × 0.3B token: 4h
  ckpt C 270M × 0.5B token: 8h
  ckpt D + long-ctx: 4h
  ckpt E = C 已含
合计 ~ 20h (单 5090, 一夜跑完)
```

## Slide 6 · 阶段化目标

```
Stage 1: 准备 Phase 1 数据 (用 Topic 1 工具)
Stage 2: 训 A vs B 对照
Stage 3: 训 C (改架构)
Stage 4: 长 ctx 扩 D
Stage 5: 评测 + 报告
```

## Slide 7 · 与系列结尾的衔接

```
本 capstone = Module 3 毕业
下一程: Module 4 「改大模型」 (PEFT/SFT/RL)
本 capstone 训出的 ckpt 是 Module 4 的起点
```

## Slide 8 · 评测设计

```
metric 矩阵:
            A     B     C     D     E
val_loss   3.5   3.2   2.9   2.9   2.8
HellaSwag  0.35  0.40  0.48  0.48  0.50
tinyMMLU   0.25  0.30  0.33  0.33  0.35
NIAH @ 8k  0%    0%    0%    80%   80%
```

## Slide 9 · 报告格式

```
report/
  curve.png         (5 条 loss 曲线)
  benchmark.csv     (5 × 5 矩阵)
  generations.md    (每 ckpt 同 prompt 输出)
  bag_of_tricks.md  (五部曲 trick 总结)
README 全图
```

## Slide 10 · 时间表

```
Day 1 上午: data prepare
Day 1 下午: A + B 训练
Day 1 晚上: C 训练
Day 2 上午: D 长 ctx 扩
Day 2 下午: evals + 写报告
```

## Slide 11 · 与 PEFT 学习 series 的对比

```
PEFT (prompt/lora/adapter) = 改给定的大模型
五部曲 = 造小模型
两条线交点: 大 model + PEFT vs 小 model 全训

trade-off:
  PEFT: 几小时 vs 全训: 几天
  PEFT: 大模型基础 vs 全训: 自由架构
```

## Slide 12 · 学到了什么

```
1. 数据是最重要的 (Phi 路线)
2. 架构差异在小规模可见 (GQA / SwiGLU)
3. 长 ctx 是独立阶段 (RoPE scaling)
4. WSD 优于 cosine
5. 训完一个真模型是真技能
```

## Slide 13 · 下一程

```
本 capstone ckpt E (270M Phi-tiny) 
  ↓
未来 SFT (Module 4)
  ↓
未来 RLHF / RL reasoning (Module 5/6)
  ↓
未来 multimodal (Module 7+)
```

## Slide 14 · 总结

```
小模型毕业 = 五部曲组装 + 实测对照
单 5090 24h 即可
是 Module 3 自然终点
```

## 参考
- Phi-1.5 (Microsoft 2023)
- TinyStories (Eldan 2023)
- 本系列 Topic 1-7
