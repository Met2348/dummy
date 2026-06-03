# L12 · RainbowPO — 7 个 DPO 变体的统一公式

> 18 slides | 50 min | DPO 家族理论收官

---

## Slide 1 · 动机

到 2024 末，DPO 变体爆炸：
- DPO / IPO / KTO / ORPO / SimPO / CPO / DPOP / sDPO ...

工程师困惑："到底哪个好？"

**RainbowPO 答案**：它们都是同一个 unified loss 在 4 维超参空间的特例。

---

## Slide 2 · 4 维超参

```
1. use_ref     : 是否用 ref model
2. length_norm : 是否 length normalize
3. loss_type   : sigmoid | squared | hinge
4. add_sft     : 是否加 NLL on chosen
```

→ 2 × 2 × 3 × 2 = 24 种组合，常见 7 个被命名。

---

## Slide 3 · Unified loss 公式

```
L_unified = α · L_pref + (1−α) · L_NLL(chosen)

L_pref = f( β·(r_c − r_r) )

r_y = (1/|y|)·log π(y) − (ref ? (1/|y|)·log π_ref(y) : 0)
              ↑ length_norm                     ↑ use_ref

f = -log sigmoid  (DPO/ORPO/SimPO/...)
  | (·−c)²        (IPO)
  | max(0, 1−·)   (hinge)
```

---

## Slide 4 · 7 个变体的超参表

| 名字 | use_ref | length_norm | loss_type | add_sft | β |
|------|---------|-------------|-----------|---------|---|
| DPO | T | F | sigmoid | F | 0.1 |
| IPO | T | F | squared | F | 0.1 |
| KTO | T | F | sigmoid | F | 0.1 |
| **ORPO** | F | F | sigmoid | T | 1.0 |
| **SimPO** | F | T | sigmoid | F | 2.5 |
| CPO | F | F | sigmoid | T | 0.1 |
| DPOP | T | F | sigmoid | F | 0.1 (+ hinge) |

---

## Slide 5 · 三轴分类

```
                  use_ref?
                    ↓
        T (DPO/IPO/KTO/DPOP)        F (ORPO/SimPO/CPO)
                                            ↓
                                      length_norm?
                                            ↓
                              T (SimPO)        F (ORPO/CPO)
                                                  ↓
                                              add_sft?
                                                  ↓
                                        T (ORPO/CPO)  ← 必须
```

→ "无 ref 必须 SFT 锚住或 length norm" 是核心规律。

---

## Slide 6 · 为何这么多变体

每个变体修一个特定问题：

| 变体 | 修 |
|------|---|
| IPO | DPO over-confidence (squared loss) |
| KTO | 无 pair 数据 (单边) |
| ORPO | 省 ref (SFT 锚住) |
| SimPO | length bias (norm) |
| CPO | 同 ORPO |
| DPOP | DPO chosen prob 下降 (hinge) |

→ 每个变体都是工程 patch。

---

## Slide 7 · 实现统一接口

```python
@dataclass
class POConfig:
    use_ref: bool
    length_norm: bool
    loss_type: str
    add_sft: bool
    beta: float
    lambda_sft: float

def unified_po_loss(log_p_*, mask, cfg):
    ...
```

→ 一个函数 + 一个 Config 切换 = 全部变体。

---

## Slide 8 · 横向 benchmark 设计

同基座 (Qwen-0.5B) + 同数据 (Anthropic-HH 1k) + 同 step：
- 看 final reward margin
- 看 chosen_logp 是否下降 (DPOP 反例)
- 看 length 漂移 (SimPO 优势)
- 看训练时长

→ Capstone src 已实现。

---

## Slide 9 · 选型决策树

```
有 pair 数据?
├─ 否 (只有 desired/undesired): KTO
└─ 是:
    ├─ 数据 < 5k: ORPO (SFT 锚住)
    ├─ 数据 ≥ 10k + 担心 length bias: SimPO
    ├─ chosen prob 不能降: DPOP
    ├─ over-confidence 风险: IPO
    └─ default: DPO
```

---

## Slide 10 · 工业实践

- 客服 chatbot: DPO 或 SimPO
- 数学推理 (步级): Step-DPO
- 持续迭代: Iterative DPO + OAIF
- 新方法验证 baseline: SimPO

---

## Slide 11 · 与 RLHF 关系

DPO 家族都是 **offline RL on preference data**。
- 优势: 1 epoch over fixed dataset，工程简单
- 劣势: 无 exploration，无 online sample

→ 高 stakes 任务（safety / 推理）仍需 RLHF/R1。

---

## Slide 12 · 数学：DPO loss 完整推导

从 RLHF 优化目标：
```
max E[r(x,y)] - β·KL(π||π_ref)
```

闭式解：
```
π*(y|x) = π_ref · exp(r/β) / Z
⇒ r(x,y) = β·log(π/π_ref) + β·log Z
```

代入 BT preference：
```
P(c≻r) = sigmoid(r_c - r_r)
       = sigmoid(β·(log π_c/π_r_c - log π_r/π_r_r))
```

→ 这就是 DPO loss。**RL 问题转化为 supervised**。

---

## Slide 13 · 为什么 ORPO 能去 ref

ORPO 用 odds 替代 log ratio：
```
DPO uses:  log π_c/π_r_c - log π_r/π_r_r
ORPO uses: log[π_c/(1-π_c)] - log[π_r/(1-π_r)]
```

ORPO 不是严格 DPO 等价，而是另一种"凸推 preference"的方式。SFT loss 充当 anchor。

---

## Slide 14 · 实测对比 (论文数据)

LLaMA-3 8B base + UltraFeedback：
| Method | AlpacaEval 2 LC | MT-Bench |
|--------|----------------|---------|
| SFT | 26 | 7.8 |
| DPO | 40.4 | 8.0 |
| IPO | 35.5 | 7.9 |
| KTO | 38.5 | 7.95 |
| ORPO | 38.1 | 7.85 |
| **SimPO** | **44.7** | **8.05** |
| CPO | 39.8 | 8.0 |

→ 没有"全场最佳"，但 SimPO 在多数任务靠前。

---

## Slide 15 · 与 RainbowPO 的实验

论文用 unified loss 网格搜 4 维超参，找到新 SOTA：
- use_ref=F, length_norm=T, loss_type=sigmoid, add_sft=T
- 超越 SimPO 1-2pp on AlpacaEval

→ 启示：超参空间未被充分探索。

---

## Slide 16 · 工程意义

之前每个 PO 变体一份代码 → 难维护。
RainbowPO 统一接口：
- 一份 loss 实现
- 一份 trainer
- yaml 切换变体

→ verl / TRL 后续都吸收了这套思想。

---

## Slide 17 · 三轨实现

```
rainbowpo.py     手写统一公式 (本讲核心)
trl.py           trl 各 trainer + RainbowPO 配置 wrap
yaml             axolotl/llamafactory 都已支持
```

---

## Slide 18 · 一句话总结

> 7 个 PO 变体 = 4 维超参的 7 个特例。RainbowPO 统一公式让选型变 "调 yaml" 而非 "改代码"。

🎓 **Topic 3 DPO Family 理论完结。**
下一讲 L13 — 6 方法横向 benchmark Capstone。
