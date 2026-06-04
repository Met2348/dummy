# L01 · Scaling Laws

> 16 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · 三大 scaling law

```
Kaplan 2020 (OpenAI):      L = f(N, D, C)
Chinchilla 2022 (DeepMind): N : D = 1 : 20 最优
Hoffmann 2022:              重写 Kaplan compute-optimal frontier
```

## Slide 2 · Kaplan 公式

```
L(N) = (N_c / N)^α    α ≈ 0.076
L(D) = (D_c / D)^β    β ≈ 0.095
L(C) = (C_c / C)^γ    γ ≈ 0.050
```

## Slide 3 · Chinchilla 颠覆

Kaplan: 给定 budget, 大模型 + 少数据
Chinchilla: ratio N : D = **1 : 20** (即 token / param ≈ 20)

```
GPT-3 175B + 300B token (1:1.7)  → 严重 undertrained
Chinchilla 70B + 1.4T token (1:20) → 最优
```

## Slide 4 · 70B vs 175B 实测

```
Chinchilla 70B:    MMLU 67.5, MATH 35.7
GPT-3 175B:        MMLU 43.9, MATH 14.6
                     ↑ 70B 小模型反超
```

## Slide 5 · 后 Chinchilla 时代

Llama 系列起：极端 over-train
```
Llama-3 70B + 15T token (1:215)  ← 远超 Chinchilla
Llama-3 8B  + 15T token (1:1875)
Qwen-2.5 7B + 18T (1:2570)
```

为什么？小模型推理便宜。

## Slide 6 · token 越多越好？

理论：edge of overfitting
实际：
```
Llama-3-8B 15T 仍未饱和
Qwen-2.5 18T 仍涨
DeepSeek-V3 14.8T 最优 ratio
```

## Slide 7 · compute-optimal frontier

```
loss vs C (FLOPs) curve:
  小模型 + 多 token 一开始降快
  大模型 + 少 token 一开始降慢但更低
                   ↓
  intersection = optimal model size @ budget
```

## Slide 8 · IsoFLOP 实验

```python
# 固定 compute, 扫描 N
N_list = [70M, 100M, 200M, 500M, 1B, 3B, 7B]
for N in N_list:
    D = budget_flops / (6 * N)
    train_model(N, D); record(loss)
plot(N, loss)
```

最低点 = optimal N。

## Slide 9 · μP scaling

```
Tensor Programs V (Yang 2022)
μ-parametrization: 小模型超参 → 大模型直接复用
省掉大模型 grid search
```

## Slide 10 · scaling law 在 SFT/RLHF

Hoffmann 类似公式：
```
RLHF loss 也满足 L = (D/D_c)^α
但 D 是 preference pair 数
PPO 的 scaling 待研究
```

## Slide 11 · 推理 scaling (新一代)

```
OpenAI o1/R1: test-time compute scaling
"思考越长 → accuracy 越高"
新 axis: thinking_tokens
```

## Slide 12 · scaling 暗面

```
loss 涨/降 ≠ benchmark 涨/降
emergent abilities @ 某 N 阶跃
Hoffmann 警告: 看 task 不看 loss
```

## Slide 13 · 工程实践

```
给定 H100 1k × 1 month budget:
  ↓ compute = 7.5e22 FLOPs
  ↓ Chinchilla: 200B param + 4T token
  实际选: 100B param + 8T token (省推理)
```

## Slide 14 · 各家选择

| 模型 | N | D | ratio |
|------|---|---|-------|
| GPT-3 | 175B | 300B | 1.7 |
| Chinchilla | 70B | 1.4T | 20 |
| LLaMA-2 | 70B | 2T | 28 |
| Llama-3 | 70B | 15T | 215 |
| Qwen-2.5 | 72B | 18T | 250 |
| DeepSeek-V3 | 671B (37B active) | 14.8T | — |

## Slide 15 · 公式实现

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, E=1.69,
                    alpha=0.34, beta=0.28):
    return E + A * N**(-alpha) + B * D**(-beta)

# given budget C = 6*N*D, find optimal N
import scipy.optimize as opt
def neg_loss(logN, C):
    N = 10**logN; D = C/(6*N)
    return -chinchilla_loss(N, D)
```

## Slide 16 · 总结

```
Scaling laws 是预算 → 模型大小工具
Chinchilla ratio 是 baseline (1:20)
现代趋势: over-train 小模型 (推理友好)
```

## 参考
- Kaplan 2020 OpenAI Scaling Laws
- Chinchilla 2022 (Hoffmann et al)
- Yang 2022 μP
