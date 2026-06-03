# L04 · ORPO — Odds Ratio PO

> 14 slides | 40 min | DPO 家族 — **无 ref model**

---

## Slide 1 · 动机

DPO/IPO 都需要 ref model（SFT ckpt 副本）。
- 显存 ×2
- 推理慢

**问题**：能不能去掉 ref？

---

## Slide 2 · 核心 idea

用 **odds** 替代 log π/π_ref。

odds(y|x) = p(y|x) / (1 - p(y|x))
log odds = log p - log(1 - p)

→ 与 ref 无关。

---

## Slide 3 · ORPO loss

```
L_ORPO = L_SFT(chosen) + λ · L_OR

L_OR = -log sigmoid(log odds_chosen - log odds_rejected)
```

- L_SFT 保住生成能力
- L_OR 拉开 chosen vs rejected

---

## Slide 4 · 推导：odds 替代 log π/π_ref

DPO loss 中 `log π_actor(c|x) / π_ref(c|x)` 衡量 actor 相对 ref 的"偏移"。

ORPO 论文证明：在 SFT 主导下，log odds 也能起类似作用。

→ 数学上不是严格等价，但工程上 SFT loss 把 actor 锚住，odds 可代 ref。

---

## Slide 5 · 算法流程

```python
for batch in dataloader:
    log_p_chosen = -F.cross_entropy(actor(chosen))   # NLL
    log_p_rejected = -F.cross_entropy(actor(rejected))
    L_sft = -log_p_chosen.mean()
    or_c = log_p_chosen - log(1 - exp(log_p_chosen))
    or_r = log_p_rejected - log(1 - exp(log_p_rejected))
    L_or = -log_sigmoid(or_c - or_r).mean()
    L = L_sft + 0.1 * L_or
```

---

## Slide 6 · 显存对比

| 方法 | model 数 | 显存 7B |
|------|--------|---------|
| RLHF (PPO) | 4 | ~50 GB |
| DPO | 2 (actor + ref) | ~28 GB |
| **ORPO** | **1 (actor only)** | **~14 GB** ⭐ |

显存省一半。

---

## Slide 7 · 训练效率

| 方法 | 单 batch 时间 (相对) |
|------|------------------|
| DPO | 1.0 (baseline) |
| ORPO | 0.55 (省 ref forward) |

→ 训练速度近 2×。

---

## Slide 8 · 数值稳定 trick

```python
def log_odds(log_p):
    log_p = log_p.clamp(max=-1e-6)        # 防 log(0)
    return log_p - torch.log1p(-log_p.exp())  # log(1-p) 稳定
```

torch.log1p(-exp(log_p)) 避免溢出。

---

## Slide 9 · 与 DPO/SimPO 对比

| 维度 | DPO | ORPO | SimPO |
|------|-----|------|-------|
| ref model | ✓ | ✗ | ✗ |
| SFT 项 | ✗ | ✓ | ✗ |
| length norm | ✗ | ✗ | ✓ |
| 工程简单度 | 中 | **高** | 高 |
| 小数据效果 | 中 | **强** | 中 |

→ ORPO 在数据少时优势明显。

---

## Slide 10 · 实测结果

ORPO 论文 Phi-2 SFT + ORPO vs DPO：
- MT-Bench: +0.3pp
- AlpacaEval LC: +2.1pp
- 训练时间: -45%

→ 效果略好 + 训练快 + 实现简单。

---

## Slide 11 · 超参建议

```
beta:        0.1
lambda_or:   0.1 (loss balance)
SFT data:    可只用 chosen，或 chosen+rejected (chosen 主导)
```

---

## Slide 12 · 局限

ORPO 仅在 actor 已有"基本"能力时好。base model 不带 SFT 直接 ORPO 效果差。

→ ORPO ≈ SFT + 偏好约束，必须先有 SFT 起点。

---

## Slide 13 · 三轨实现

```
orpo_minimal.py   手写（< 50 行）
orpo_trl.py       trl.ORPOTrainer (生产)
orpo_axolotl      axolotl yaml 配置
```

---

## Slide 14 · 一句话总结

> ORPO = SFT + odds ratio penalty + 无 ref。1 model 训练，效果对标 DPO，显存省一半。

下一讲 L05 — SimPO（更激进的 ref-free）。
