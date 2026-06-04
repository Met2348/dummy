# L10 · 蒸馏 (Distillation) for Pretraining

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 蒸馏 idea

```
小 student 学大 teacher 的 logits
比直接 next-token 学习更高效
Hinton 2015 提出, LLM 复兴
```

## Slide 2 · 经典 KD loss

```
loss = alpha * CE(student, hard_label)
       + (1-alpha) * T² * KL(soft_student, soft_teacher)

soft = softmax(logits / T)
```

T (temperature) 通常 4-10.

## Slide 3 · LLM 蒸馏特殊

```
vocab 大 (50k+) → KL over 50k 慢
sequence 长 → 多 token 蒸馏
位置相关 → 不 pool
```

## Slide 4 · DistilBERT (2019) 复刻

```
6 layer student + 12 layer teacher
3 losses: CE + KL(soft) + cosine(hidden)
40% smaller, 97% performance
```

## Slide 5 · MiniLM (2020)

```
attention map 蒸馏:
  KL(student_attn_softmax, teacher_attn_softmax)
+ value relation 蒸馏
```

## Slide 6 · 当代: Phi 蒸馏链

```
Phi-1: 大 model (CodeT5) 生成训练数据 (合成)
Phi-2: GPT-3.5 生成
Phi-3: 不直接 KD, 用合成 + 高质数据
```

## Slide 7 · Distillation from API

```
1. prompt GPT-4
2. response → 训练数据
3. small model SFT on (prompt, response)
合法性: 看 ToS
```

## Slide 8 · 数据合成 vs logits 蒸馏

```
数据合成: 用 LLM 生成 (prompt, response)
logits 蒸馏: 用 LLM logits 直接监督
小项目用数据合成 (无需 teacher 在线)
```

## Slide 9 · DeepSeek-R1 → Qwen 蒸馏

```
R1 生成 800k 高质 reasoning trace
SFT Qwen-7B → Qwen-Distill-7B
benchmark 大幅提升 (MATH +20pp)
```

## Slide 10 · TinyLlama 风格

```
Llama-2 7B 当 teacher
TinyLlama 1.1B student
loss: pure LM loss (无 KD)
"large data" 比 "small distill" 更重要
```

## Slide 11 · 蒸馏 vs 大数据

```
1B token + KD ≈ 5B token + 无 KD
KD 性价比高 (vs 多数据)
```

## Slide 12 · 总结

```
蒸馏: 数据合成 (主流) > logits KD (经典)
本 capstone 不用蒸馏 (直接 LM loss)
真用 → Phi 风格合成
```

## 参考
- Hinton 2015 KD
- DistilBERT 2019
- DeepSeek-R1 蒸馏报告 2025
