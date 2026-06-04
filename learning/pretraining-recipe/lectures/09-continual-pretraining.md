# L09 · Continual Pretraining

> 12 slides | 35 min ⭐⭐⭐

## Slide 1 · 什么是 continual pretraining (CPT)

```
已 pretrained model + 新数据 → 继续训
不同于 full SFT (CPT 仍是 LM loss, 不指令)
```

## Slide 2 · 用例

```
- 领域适配: 通用模型 → 法律 / 医疗
- 中文增强: Llama 英文 → 加 zh
- code 增强: Llama → CodeLlama
- 长 ctx 扩展: 8k → 128k
```

## Slide 3 · 关键挑战

```
catastrophic forgetting: 学新忘旧
解决: 数据混合 (旧 50% + 新 50%) 或 LoRA
```

## Slide 4 · LR 选择

```
比 pretraining 小 10×:
  pretrain lr: 3e-4
  CPT lr: 3e-5
warmup 仍要
```

## Slide 5 · 数据配比

```
方案 A (复习): 旧 50% + 新 50%
方案 B (回放): 旧 10% + 新 90% (effective for short CPT)
方案 C (LoRA): 100% 新 + 用 LoRA 锁住旧 (Phi-3 用)
```

## Slide 6 · 长 ctx CPT

```
phase 1 (8k → 32k): YaRN + 100M token
phase 2 (32k → 128k): 50M token
phase 3 (annealing): 高质 long doc 10M
```

## Slide 7 · 中文 CPT (Chinese-Llama, BiLLa)

```
phase 1: 加 20k 中文 token 到 vocab
phase 2: 50% zh + 50% en × 100B token
phase 3: SFT instruction 中英
```

## Slide 8 · code CPT (CodeLlama)

```
phase 1: 100% code × 500B token
phase 2: 数据回放 + math
phase 3: long ctx 16k
```

## Slide 9 · CPT loss 曲线

```
开始: spike (新分布)
1-2 epoch 后: 稳定
val_loss 应 < pretrain baseline
```

## Slide 10 · 检测 catastrophic forgetting

```
保留 hold-out test:
  - 旧领域 MMLU
  - 新领域 task
两者都监控
```

## Slide 11 · DeepSeek-V3 例

```
14.8T 通用 + 1T 推理 CPT
最后阶段加 cleaner data
```

## Slide 12 · 总结

```
CPT = 已 pretrained model + 新数据 + 小 lr + 数据回放
长 ctx 扩展是常见 CPT 场景
LoRA CPT 也常用 (Phi-3)
```

## 参考
- CodeLlama (Roziere 2023)
- Chinese-Llama
- Phi-3 (Microsoft 2024)
