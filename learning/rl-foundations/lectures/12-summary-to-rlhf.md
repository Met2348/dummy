# L12 · 总结 + 引出 RLHF

> 16 slides | 40 min | RL Foundations 系列收尾

---

## Slide 1 · 本系列方法清单

| L | 方法 | 一句话 |
|---|------|--------|
| 1 | REINFORCE | PG 起点，高方差 |
| 2 | A2C | 加 critic 降方差 |
| 3 | TRPO | trust region 二阶 |
| 4 | PPO | clip 一行替代 trust region |
| 5 | GAE | λ 在 TD/MC 间平滑 |
| 6 | PPO tricks | 7 件套工程 |
| 7 | CartPole Lab | 5 算法横向 |
| 8 | PPO for LLM | 4 model 协同 |
| 9 | sentiment toy | 第一次 LLM-RL 实验 |
| 10 | RL Pitfalls | 坑合集 |
| 11 | Capstone | IMDb 完整 |
| 12 | 本讲 | 总结 |

---

## Slide 2 · 横向对照表

| 算法 | critic | KL 约束 | 实现难度 | LLM 适用 |
|------|--------|--------|---------|---------|
| REINFORCE | ✗ | ✗ | 极易 | 仅玩具 |
| A2C | ✓ TD | ✗ | 易 | ✓ |
| TRPO | ✓ | KL ≤ δ | 难（F+CG）| ✗ |
| **PPO** | ✓ | **clip** | 中 | ✓ 行业默认 |

---

## Slide 3 · PPO 在所有 LLM-RL 中的角色

| 算法 | PPO 关系 |
|------|---------|
| InstructGPT PPO | PPO + KL ref penalty |
| GRPO (R1) | PPO clip + 去 critic + group baseline |
| DAPO (2025) | PPO + 4 件套 |
| VAPO (2025) | PPO + Length-Adaptive GAE |
| Dr. GRPO | GRPO + 修 length bias |

→ 所有 RL 用在 LLM 上的算法都是 PPO 的变体或简化。

---

## Slide 4 · 关键公式回顾

```
g_PG = ∇log π · A                            # PG 定理
A_GAE = δ + γλ A_next                        # GAE 反向
L_clip = -min(r·A, clip(r,1±ε)·A)            # PPO clip
KL_approx = E[r - 1 - log r]                 # Schulman approx (cheap)
```

把这 4 行记住，你就有 RL 80% 的"脑内 cheatsheet"。

---

## Slide 5 · 为什么 RL 需要 RM（引出专题 2）

CartPole reward 是环境给的（每步 +1）。LLM 没有这样的环境 reward：
- 怎么定义"好的回答"？人工标注太贵
- 用规则？太死板
- 用预训练 LLM 当 judge？引入新 bias

**InstructGPT 范式**：
1. 收人类**偏好对**（pairwise）
2. 训 RM (Bradley-Terry) 模拟人类偏好
3. 用 RM 当 reward 跑 PPO

→ 这是专题 2 的全部内容。

---

## Slide 6 · RM 的核心 idea

不直接评分（绝对评分难），而是**比较两个 response**：

```
P(y_chosen > y_rejected | x) = sigmoid(r(y_chosen) - r(y_rejected))
```

→ Bradley-Terry 模型。RM 学的是"哪个更好"，不是"打几分"。

工业上 RM 是 LLM + scalar head（输出 1 维），用 BT loss 训。

---

## Slide 7 · 三段式管线

```
        预训练 LLM
            |
            v
          SFT
       （instruction tuning）
            |
            v
          RM 训练
        （pairwise BT loss）
            |
            v
          PPO
       （用 RM 当 reward + KL ref penalty）
            |
            v
       对齐后的 LLM
```

→ InstructGPT (2022)、LLaMA-2、Sparrow 都是这个。

---

## Slide 8 · RLHF 的局限（引出专题 3）

RLHF 复杂：
- 4 个模型协同
- 显存巨大
- 训练慢
- 超参敏感（PPO 痛点 + RM 痛点）

→ DPO (NeurIPS 2023) 出现：**直接从偏好数据训 LLM，无需显式 RM 和 PPO**。

```
L_DPO = -log sigmoid( β log(π_θ(y_w)/π_ref(y_w)) - β log(π_θ(y_l)/π_ref(y_l)) )
```

一条公式，5 行代码，不需要 reward model。

---

## Slide 9 · DPO 之后是 R1（引出专题 5）

DPO 解决了"对齐"，但**推理能力没显著提升**。

R1 (2025.01) 解决方案：
- 不用 RM，用 **rule-based reward**（format + accuracy）
- 算法 GRPO（PPO 的 critic-free 变体）
- 涌现 "aha moment"（wait/recheck/let me reconsider）

→ 专题 5 详讲。

---

## Slide 10 · 系列整体路线（再放一遍）

```
专题 1 ✓ RL 基础（你刚学完）
专题 2   RLHF Classic (InstructGPT 三段式)
专题 3   DPO 家族（去 RM 革命）
专题 4   Process Reward（推理 RL 工具）
专题 5   R1 时代 ⭐⭐⭐（GRPO + DAPO + R1-Zero 复现）
专题 6   2025-2026 SOTA（DAPO/VAPO/PRIME/...）
专题 7   多模态+Agent+毕业 ⭐⭐⭐（五线综合）
```

---

## Slide 11 · 五线综合预告（专题 7 毕业）

PEFT + RL = 大模型对齐"五条线"：

```
                    p(y|x; θ_LM, φ)
                       /  |  \
            Prompt   LoRA  Adapter   ← PEFT 三线
                       \  |  /
                        ↓ ↓ ↓
            (φ 改 model: input / weight / structure)

                       RLHF      R1
                       (改 distribution / trajectory)
```

→ 5 个 ckpt 在同一道 GSM8K 题上对照。

---

## Slide 12 · 一句话总结整个 RL 基础专题

> **PPO 是 LLM-RL 的瑞士军刀。其他算法 (DPO/GRPO/DAPO) 要么是它的简化要么是它的变体。**

掌握 PPO，掌握 RL；
掌握 PPO + KL，掌握 RLHF；
掌握 PPO + clip 几何，掌握 GRPO / DAPO；
掌握 PPO + Fisher 思路，掌握 TRPO / Nash-MD。

---

## Slide 13 · 检查清单

```
[ ] 12 lecture 全过
[ ] 12 notebook 全跑（或至少看完）
[ ] 5 算法手写源码读过（不需要逐行）
[ ] PPO 7 件套清单背得
[ ] capstone 跑过（手写或 trl 都行）
[ ] Spurious Rewards、Over-optimization 警示理解
[ ] tag rl-foundations
```

---

## Slide 14 · 自测题（系列级）

1. 写出 PG 定理 + GAE 公式 + PPO clip。
2. 解释为何 LLM-PPO 需要 4 model。
3. 介绍 PPO 7 件套 + 哪个最重要。
4. Reward Hacking 5 类各举一例 + 防御。
5. 为什么 DPO 之后是 R1（PEFT vs RLHF vs R1 各自改什么）？

---

## Slide 15 · 推荐扩展阅读

- **必读**：InstructGPT (Ouyang 2022)
- **必读**：DPO (Rafailov 2023)
- **必读**：DeepSeekMath GRPO (2024.02)
- **必读**：R1 / R1-Zero (DeepSeek 2025.01)
- **必看**：HuggingFace Deep RL Course
- **可选**：Schulman PhD 论文（PPO/GAE 内功）

---

## Slide 16 · 入口：进入专题 2

```bash
# tag 本专题
git tag rl-foundations

# 下一专题 spec/plan
ls docs/superpowers/specs/2026-06-03-rlhf-classic-design.md
ls docs/superpowers/plans/2026-06-03-rlhf-classic.md

# 开始
cd learning/  # 接下来创建 rlhf-classic/
```

🎓 **专题 1 RL 基础完成！**
