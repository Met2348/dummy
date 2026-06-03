# L05 · SimPO — Simple PO

> 14 slides | 35 min | DPO 家族 — **极简 + length norm + 无 ref**

---

## Slide 1 · 动机

DPO 反复发现：长 response 在 RL 中"占便宜"（reward 累积大）。需要 length normalize。

SimPO = DPO 减掉 ref + 加 length norm + 单超参 γ。

---

## Slide 2 · SimPO 公式

```
r(x, y) = (β / |y|) · log π(y|x) − γ

L_SimPO = -log sigmoid(r_chosen − r_rejected)
        = -log sigmoid(β·(avg_logp_c − avg_logp_r) − γ)
```

- **length-normalized**：避免长 response 占优
- **γ**: target margin，鼓励 chosen−rejected 至少 γ
- **无 ref model**

---

## Slide 3 · 与 DPO 数学对比

```
DPO:    log π_c/π_r_c − log π_r/π_r_r    (有 ref，无 norm)
SimPO:  (1/|c|)·log π_c − (1/|r|)·log π_r − γ/β   (无 ref，length norm)
```

数学上不等价，但 SimPO 在工程中跑得更好。

---

## Slide 4 · 为什么 length norm 重要

DPO 训练后常见：
- chosen 长度 ↗ 30%
- rejected 长度 ↘ 20%
- → "我学会了灌水"

SimPO length norm 消除这个 bias。

---

## Slide 5 · γ 的几何意义

```
margin = β·(avg_logp_c − avg_logp_r)

希望 margin ≥ γ → 即使 actor 给两者概率接近，loss 也非零。
```

γ ≈ 1.0 是 default（force 至少 e 倍 odds 优势）。

---

## Slide 6 · 超参建议

| 超参 | 建议 |
|------|-----|
| β | 2.5 (而非 DPO 的 0.1) |
| γ | 0.5 ~ 1.6 |
| lr | 5e-7 (比 DPO 小) |
| epoch | 1 (overfit 风险) |

注：β 大是因为 length-normalized log_p 数值小。

---

## Slide 7 · 工程优势

```
✓ 1 model 训练 (无 ref) → 显存省一半
✓ implementation < 30 lines
✓ AlpacaEval LC SOTA (2024.05)
```

→ DPO 家族中最受欢迎的之一。

---

## Slide 8 · 实测效果 (论文)

LLaMA-3 8B SFT + SimPO vs 各 baseline:
| 方法 | AlpacaEval 2 LC win |
|------|---------------------|
| SFT | 26.0 |
| DPO | 40.4 |
| ORPO | 38.1 |
| **SimPO** | **44.7** ⭐ |

---

## Slide 9 · 失败模式

| 现象 | 修 |
|------|---|
| 完全不学 | γ 调小至 0.5 |
| 输出短到不自然 | γ 调小 + β 调小 |
| 训崩 | lr 减半 |
| 输出空 | data 质量检查 |

---

## Slide 10 · 三轨实现

```
simpo_minimal.py   手写 length norm + margin loss
simpo_trl.py       trl.CPOTrainer (loss_type="simpo")
simpo_axolotl      yaml 配置
```

---

## Slide 11 · 与 ORPO 选哪个

- 数据多 (≥ 10k pair)：SimPO 略优
- 数据少 (< 5k)：ORPO 略优（SFT 锚住）
- 想最简单：SimPO（< 30 行）

---

## Slide 12 · 与 KTO 互补

KTO 处理"单边"（只有 desired/undesired 标签）；SimPO 处理 pair。
工程上经常先 KTO（大量单边数据），再 SimPO（少量 pair fine-tune）。

---

## Slide 13 · 局限

length norm 假设"每个 token 价值相等"，对长推理任务（GSM8K CoT）可能不利。
→ R1 时代倾向 token-level loss（DAPO Trick 3）。

---

## Slide 14 · 一句话总结

> SimPO = DPO 去 ref + length norm + γ margin = 工程最简 + AlpacaEval SOTA。

下一讲 L06 — CPO。
