# L05 · mini-Arena 设计

5 ckpt round-robin × 5 题 × 2 ordering = 100 battles。

## 流程

```python
1. for (a, b) in C(5, 2):     # 10 pair
2.     for q in 5 题:
3.         winner_AB = judge(a(q), b(q))
4.         winner_BA = judge(b(q), a(q))
5. fit_BT(battles) → log strengths
6. to_elo(log_s) → 1500 base
```

## 算法核心：MM (Hunter 2004)

```
s_i ← W_i / Σ_j (n_ij / (s_i + s_j))
```

200 iter 收敛。

## 我们的 toy judge

```python
def _length_quality_judge(r_a, r_b):
    """Reward <answer> tags + step-by-step pattern + length cap.
       Penalize obvious leaks like 'step 1: get materials'."""
    score(t) = +0.4 if "answer" or </think>
             + 0.2 if "step" or ":"
             + 0.3 * min(1, len/600)
             - 0.8 if leak pattern
```

## 预期 Elo

```
| rank | ckpt | Elo |
|---|---|---:|
| 1 | r1_tiny  | ~1700 |
| 2 | phi_tiny | ~1600 |
| 3 | dpo      | ~1550 |
| 4 | lora     | ~1400 |
| 5 | vanilla  | ~1300 |
```

vanilla 最低（leaks + 错误推理 23）→ Elo ~1300。
r1_tiny 最高（带 think + answer + 正确推理）。

## 真 Chatbot Arena 对照

| 维度 | 我们 | Chatbot Arena |
|------|------|---------------|
| Judge | toy regex | GPT-4o / 真人 |
| Pairs | 100 | 1M+ |
| 模型 | 5 mock | 100+ 真 |
| 时间 | 1s | 持续更新 |
| Style control | 无 | 有 |

## 一句话

> mini-Arena = 5 ckpt 互打 + BT-Elo，1s 给出排行。
