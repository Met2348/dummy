# L10 · Speculative Decoding

> 12 slides | 40 min ⭐⭐⭐⭐

## Slide 1 · 推理瓶颈

```
LLM decode: 一次 1 token, autoregressive
带宽 bound (KV cache + weight) > 计算 bound
GPU 算力闲置
```

## Slide 2 · idea

```
小 draft model 一次 predict K token
大 target model 一次 verify K token
verify pass 率高 → 1 step 出多 token
```

## Slide 3 · 算法

```
1. draft: 小模型 generate K token (autoregressive)
2. verify: 大模型一次 forward (并行) 算 logits
3. accept token until divergence (rejection sampling)
4. resample 1 token from divergence point
```

## Slide 4 · 加速比

```
accept rate p (~0.7-0.9)
平均 accept k 个
speedup ≈ 1 + p*K / (1 + K/M)
M = target/draft 比
```

实测 2-3×.

## Slide 5 · draft model 选择

```
small same family: Llama-3-8B → Llama-3-1B
prefix LM: 用 target 自己做 N-gram
EAGLE: target 自带轻量 head
Medusa: target + 多 head 并行预测
```

## Slide 6 · EAGLE-2

```
draft = target 的轻量 head + tree
accept rate 90%+ → 3-4× speedup
DeepSeek-V3 / Llama-3 已集成
```

## Slide 7 · Medusa

```
target + N 个 lm_head (并行预测 N 个未来 token)
top-K 树搜索验证
2-2.5× speedup
```

## Slide 8 · vllm 支持

```bash
vllm serve meta-llama/Llama-3-70B \
  --speculative-model meta-llama/Llama-3-1B \
  --num-speculative-tokens 4
```

## Slide 9 · 限制

```
- batch=1 时收益最大, batch 大时通讯瓶颈
- draft 必须同 tokenizer
- temperature 高时 accept 率降
```

## Slide 10 · Multi-Token Prediction (MTP)

```
DeepSeek-V3 引入: 训练时多 head 并行预测
推理时直接当 draft
不需要外加小模型
```

## Slide 11 · 实测

```
Llama-3-70B + Llama-3-1B draft + temp 0:
  vanilla: 18 tok/s
  spec:    52 tok/s  (~2.9×)
```

## Slide 12 · 总结

```
speculative decoding 是当下 LLM 推理标配
EAGLE-2 / Medusa 是主流
DeepSeek MTP 是新方向
```

## 参考
- Leviathan 2023 Speculative Decoding
- EAGLE-2 (Li 2024)
- Medusa (Cai 2024)
