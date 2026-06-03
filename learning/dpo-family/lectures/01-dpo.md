# L01 · DPO — Direct Preference Optimization

> 28 slides | 70 min | DPO Family 第 1 讲（⭐⭐⭐⭐⭐ 必学）

> Rafailov et al. 2023 — NeurIPS 2023 Outstanding Paper

---

## 学习目标

1. 推导 DPO 公式（从 RLHF 的 PPO+KL 出发，闭式解）
2. 理解 DPO 的"隐式 RM"是什么
3. 写出 DPO loss 的 4 行 Pytorch 实现
4. 在 Anthropic-HH 1k 上训练，看 chosen-rejected margin 变化

---

## Slide 1 · RLHF 痛点回顾

专题 2 学到：
- 4 模型协同 → 显存 4×
- PPO 训练超参敏感
- RM 训练独立一阶段
- KL 控制需 adaptive

→ **能否直接从偏好数据训 LLM，跳过 RM 和 PPO？**

---

## Slide 2 · DPO 的承诺

```
                 RLHF                          DPO
        ┌──────────────────┐         ┌──────────────────┐
        │  SFT             │         │  SFT             │
        │  RM 训练         │         │                  │
        │  PPO + KL + 4M   │         │  DPO (1 行 loss) │
        └──────────────────┘         └──────────────────┘
        显存 4×, 1 周                显存 2× (actor+ref), 1 天
```

→ **一行 loss 替代 RM + PPO**。

---

## Slide 3 · 起点：RLHF 优化目标

PPO 的等价目标：

```
max_π   E_x [ E_{y ~ π(.|x)} r(x, y) ] - β · KL(π || π_ref)
```

可以解出 closed-form 最优策略 π*(y|x)：

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x, y) / β)
```

其中 Z(x) 是 partition function (难算)。

---

## Slide 4 · 求逆：从 π* 解出 r

把上式反解：

```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

→ **Reward 可以从 policy 和 ref 算出来！**

→ 隐式 RM：`r(x, y) ∝ β · log(π(y|x) / π_ref(y|x))`

---

## Slide 5 · 代入 Bradley-Terry

RM 学的是 pairwise：

```
L_RM = -E[log sigmoid(r(x, y_w) - r(x, y_l))]
```

把上 slide 的 r 代入：

```
r(x, y_w) - r(x, y_l)
   = β log(π(y_w|x)/π_ref(y_w|x)) + β log Z(x)
   - β log(π(y_l|x)/π_ref(y_l|x)) - β log Z(x)
   = β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x))
```

**Z(x) 消去！** ← DPO 的关键。

---

## Slide 6 · DPO Loss

```
L_DPO(π_θ ; π_ref) = -E_(x,y_w,y_l) [
    log sigmoid(
        β log(π_θ(y_w|x) / π_ref(y_w|x))
      - β log(π_θ(y_l|x) / π_ref(y_l|x))
    )
]
```

一行公式，直接训 actor，**不需要 RM 也不需要 PPO**。

---

## Slide 7 · 几何意义

```
L_DPO = -log sigmoid( β · margin )

margin = log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)
```

- margin > 0 → 模型偏好 chosen，loss 小
- margin < 0 → 模型偏好 rejected，loss 大

→ 训练目标：**让 chosen 概率上升、rejected 下降（相对 ref）**。

---

## Slide 8 · Pytorch 实现（4 行）

```python
log_p_chosen_actor = get_log_probs(model, chosen_ids)
log_p_chosen_ref = get_log_probs(ref, chosen_ids)
log_p_rejected_actor = get_log_probs(model, rejected_ids)
log_p_rejected_ref = get_log_probs(ref, rejected_ids)

log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()
```

→ 4 行就训完了 RLHF 三段管线中的 Stage 2+3。

---

## Slide 9 · β 怎么选

| β | 行为 |
|---|------|
| 0.01 | 接近 SFT，DPO 改动小 |
| **0.1** | **DPO 论文默认** |
| 0.5 | 激进 |
| > 1.0 | DPO 几乎等价 BT-fitting，KL 不再起作用 |

→ 0.1 是 trl 默认，多数 paper 采用。

---

## Slide 10 · 训练超参（typical）

| 超参 | 值 |
|------|---|
| lr | 5e-7 (比 PPO 小 10×) |
| batch | 4 |
| epoch | 3 |
| β | 0.1 |
| max_length | 1024 |
| weight decay | 0 |

→ 5090 24GB + Qwen2.5-0.5B 训 1k pair 约 30 min。

---

## Slide 11 · trl DPOTrainer

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(output_dir="dpo", beta=0.1, learning_rate=5e-7),
    train_dataset=train_ds,
    tokenizer=tokenizer,
)
trainer.train()
```

trl 自动处理 ref forward + masking + loss。

---

## Slide 12 · 监控指标

| 指标 | 期望 |
|------|------|
| margin (log r_w - log r_l) | 从 0 → 增大 |
| reward chosen / rejected | 散开 |
| chosen 概率 | 上升 |
| rejected 概率 | 下降 |
| 训练 loss | 0.69 → 0.1 |

⚠️ **DPO 的"奇怪现象"**：chosen 概率有时反而下降（只是 rejected 下降更多）→ DPOP 修复（L07）。

---

## Slide 13 · DPO vs RLHF 实证

InstructGPT 任务（Stiennon TL;DR）：

| 指标 | RLHF | DPO |
|------|------|-----|
| 偏好胜率 | 64% | 63% |
| 显存 | 4× | 2× |
| 训练时间 | 1 周 | 1 天 |
| 超参敏感性 | 高 | 中 |

→ 性能持平，**资源 / 工程效率显著优**。

---

## Slide 14 · DPO 的局限

1. **off-policy**：用静态偏好数据训，不像 PPO 持续 rollout
2. **chosen 概率下降问题**（L07 详讲）
3. **数据效率**：需 BT 偏好对，比 single-response SFT 数据贵
4. **β 调参敏感**

→ Iterative DPO / Online DPO 解决 1，DPOP 解决 2。

---

## Slide 15 · Capstone preview · 6 方法横向

L13 capstone 同基座同数据跑 6 个 DPO 变体：

| 方法 | 公式 |
|------|------|
| DPO | 上面 |
| IPO | 替换 sigmoid 为 RMSE |
| KTO | 单边偏好（无 pair） |
| ORPO | 无 ref model |
| SimPO | length-normalized |
| CPO | DPO + SFT 加权 |

→ 看哪个 6 维雷达图最优。

---

## Slide 16 · 算法选型决策树

```
有 pair 偏好数据？
    ├── 是 → 想去 ref model 显存？
    │        ├── 是 → ORPO / SimPO
    │        └── 否 → DPO / IPO
    └── 否 (单边)→ KTO
```

工程考虑：
- 训练速度：SimPO > ORPO > DPO
- 性能：DPO ≈ IPO ≈ CPO 略胜 SimPO
- 数据效率：KTO > DPO

---

## Slide 17 · 与 R1 时代的对比

R1 时代用 GRPO：
- on-policy（每步 rollout）
- 用 rule-based reward
- 推理任务

DPO 适合：
- 对齐 / 风格 / 安全
- 静态人类偏好数据
- 通用 chat

→ 两者覆盖不同场景，不冲突。

---

## Slide 18 · DPO 的数学优雅

DPO 的 elegance:
1. 从 RLHF 出发，闭式推导
2. 消去 partition function Z
3. 一行 loss 替代两阶段训练

→ 这是为什么获 NeurIPS Outstanding。

---

## Slide 19 · 推导自测（必练）

5 步代数推导：

1. PPO 等价目标 `max E[r] - β KL(π||π_ref)`
2. 闭式解 `π*(y|x) ∝ π_ref · exp(r/β)`
3. 求逆 `r = β log(π/π_ref) + β log Z`
4. 代入 BT loss
5. Z 消去，得 DPO loss

→ 自己手推一遍。

---

## Slide 20 · 工程小贴士

| 项 | 注意 |
|----|------|
| ref model | 必须 freeze + 与 actor 同 init（通常用 SFT 模型） |
| log_probs | masking 必须正确（仅 response 段计数） |
| lr | 5e-7 起步，过大 → margin 飞涨但不学语义 |
| 数据 | chosen/rejected 长度差异不要太大 |
| β | 从 0.1 起 |

---

## Slide 21 · 经典坑

1. **lr 太大** → margin 飞涨，模型快速过拟合
2. **ref model 错** → 用了 base 而非 SFT
3. **chosen 概率下降** → DPOP 修复（L07）
4. **length bias** → SimPO 修复（L05）

---

## Slide 22 · 后续 12 lecture 预览

| L | 方法 | idea |
|---|------|------|
| 02 | IPO | RMSE 替代 sigmoid |
| 03 | KTO | 单边偏好 |
| 04 | ORPO | 无 ref |
| 05 | SimPO | length norm |
| 06 | CPO | + SFT |
| 07 | DPOP | 修 chosen 概率 |
| 08 | Step-DPO | 步骤级 |
| 09 | Iterative | 多轮迭代 |
| 10 | Online | on-policy 采样 |
| 11 | Nash-LHF | Nash 均衡 |
| 12 | RainbowPO | 7 变体统一 |
| 13 | Capstone | 6 方法横向 |

---

## Slide 23 · 自测题

1. 推导 DPO loss 从 RLHF 出发（5 步）。
2. 隐式 RM `r = β log(π/π_ref)` 的几何意义？
3. 为何 partition function Z 在 BT 中消去？
4. β 取值对训练的影响？
5. DPO 比 RLHF 的工程优势有哪几条？

---

## Slide 24 · 阅读建议

- **必读**：Rafailov 2023 §3-§4（数学推导）
- **必读**：trl DPOTrainer 文档
- **看图**：附录 A 的 chosen/rejected 概率漂移
- 可选：RainbowPO (L12) 看 7 变体统一视角

---

## Slide 25 · 实战入口

```bash
# 训 DPO toy
python learning/dpo-family/src/dpo_minimal.py
python learning/dpo-family/src/dpo_trl.py

# 测试一致性
pytest learning/dpo-family/src/tests/test_dpo_loss_equivalence.py
```

---

## Slide 26 · 一行总结

> DPO = 5 步代数推导得到一行 loss = 替代整个 RM + PPO 训练。

---

## Slide 27 · 下一讲

**L02 IPO** — 用 RMSE 替代 sigmoid，解决 DPO 偏向极端样本的问题。

---

## Slide 28 · 检查清单

- [ ] 手推 DPO 5 步代数
- [ ] 看懂 4 行 Pytorch 实现
- [ ] 跑通 dpo_minimal.py
- [ ] 与 trl DPOTrainer loss 数值一致
- [ ] 看到 margin 上升曲线
