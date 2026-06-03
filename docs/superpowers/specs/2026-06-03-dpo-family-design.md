# DPO Family 学习专题 — 设计文档

> **承接**: rlhf-classic (BT-RM + PPO 三段管线)
> **本专题**: 去 reward model 革命 — 13 个偏好优化方法
> **战略地位**: 2023-2024 最重要的 PEFT/对齐范式革新
> **总体规划**: `C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`

---

## 1. 专题定位

DPO (Rafailov et al. 2023, NeurIPS 2023 best paper) 证明了 **RLHF 不需要显式 reward model** — 通过把 BT loss + PPO + KL 约束闭式求解，得到一个直接基于 preference data 的简单监督 loss。本专题完整覆盖 DPO 衍生家族 13 个方法 + RainbowPO 统一框架。

### 1.1 为什么 DPO 革命如此重要
1. **工程简化**：从 4-model PPO 降为 2-model SFT-like 训练
2. **稳定性**：不再有 reward hacking + KL 爆炸
3. **数据高效**：preference data 直接用，无需 RM 中介
4. **理论优美**：闭式解从最优策略导回

### 1.2 但 DPO 也有问题（衍生方法的动机）
- **chosen 概率反而下降**（DPOP 修复）
- **长度偏置**（SimPO 用 length norm 修复）
- **极端样本敏感**（IPO 修复）
- **需要 reference model 占显存**（ORPO 去掉 ref）
- **pairwise data 难收集**（KTO 用单边偏好）

---

## 2. 方法清单（13 种 + RainbowPO 统一框架）

| # | 方法 | 年份 | 论文 | 核心 idea |
|---|------|------|------|----------|
| 1 | **DPO** | 2023.05 | Rafailov et al., NeurIPS | implicit RM，闭式 loss |
| 2 | **IPO** | 2023.10 | Azar et al., DeepMind | 解决 DPO 偏向极端 |
| 3 | **KTO** | 2024.02 | Ethayarajh et al. | Kahneman-Tversky，单边偏好 |
| 4 | **ORPO** | 2024.03 | Hong et al. | Odds Ratio，**无 reference model** |
| 5 | **SimPO** | 2024.05 | Meng et al., NeurIPS 2024 | 极简，length-norm，无 ref |
| 6 | **CPO** | 2024 | Xu et al. | Contrastive Preference + SFT |
| 7 | **DPOP** | 2024 | Pal et al. | 修复 chosen 概率下降 |
| 8 | **Step-DPO** | 2024 | — | 步骤级 DPO（推理任务）|
| 9 | **Iterative DPO** | 2024 | — | 多轮迭代偏好 |
| 10 | **Online DPO / OAIF** | 2024 | Guo et al. | on-policy 采样 + AI feedback |
| 11 | **sDPO** | 2024 | — | sequential DPO |
| 12 | **Nash-LHF** | 2023 | DeepMind (Nash-MD/INPO/ONPO) | Nash 均衡视角 |
| 13 | **β/Cal/α-DPO** | 2025 | — | DPO 调参变体集 |
| **+** | **RainbowPO** | 2024.10 | Pal et al. | 统一框架，4 维超参覆盖 7 变体 |

---

## 3. Lecture 结构（13 篇 = 11 主线 + 2 综合）

| Lecture | 主方法 | 时长 |
|---------|--------|------|
| **L1** DPO 推导 | DPO 完整 5 步代数推导 | 90 min |
| **L2** IPO | 解决 DPO 极端样本 | 60 min |
| **L3** KTO | 单边偏好，Kahneman-Tversky 损失 | 60 min |
| **L4** ORPO | 消去 reference model 推导 | 90 min |
| **L5** SimPO | length-norm 几何意义 | 90 min |
| **L6** CPO | Contrastive + SFT 组合 | 60 min |
| **L7** DPOP | 修复 chosen 概率下降 + 反例可视化 | 60 min |
| **L8** Step-DPO | 步骤级 DPO + GSM8K 玩具 | 60 min |
| **L9** Iterative DPO + OAIF | 多轮迭代 | 60 min |
| **L10** Online DPO + sDPO | on-policy 采样 | 60 min |
| **L11** Nash-LHF | Nash-MD / INPO / ONPO | 60 min |
| **L12** RainbowPO 统一公式 | 7 变体的 4 维超参覆盖 | 90 min |
| **L13** Capstone：6 方法对照 benchmark | 横向对比 + 雷达图 | 120 min |

**总学时**: 12×70 + 120 = 960 min ≈ 16 hours（**含 DPO 数学推导**）

---

## 4. Lecture 模板（每篇 22-26 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 这个 DPO 变体解决什么问题（vs 原 DPO）
## Slide 3-6: 数学推导（loss 形式 + 与 DPO 的代数差异）
## Slide 7-9: 直觉解释（什么样的数据 / 任务受益）
## Slide 10-14: 代码逐行（minimal + trl 对照）
## Slide 15-17: 论文实验复现
## Slide 18-20: 与 DPO/上一个变体对比
## Slide 21-23: 思考题 + 陷阱
## Slide 24-25: 下节预告
```

---

## 5. 代码两轨策略

| 方法 | minimal | trl | 备注 |
|------|---------|-----|------|
| DPO | ✅ 手写 | ✅ `DPOTrainer` | **强一致** loss < 1e-6 |
| IPO | ✅ 手写 | ✅ `DPOTrainer(loss_type="ipo")` | 强一致 |
| KTO | ✅ 手写 | ✅ `KTOTrainer` | 强一致 |
| ORPO | ✅ 手写 | ✅ `ORPOTrainer` | 强一致 |
| SimPO | ✅ 手写 | ✅ `CPOTrainer(loss_type="simpo")` | 强一致 |
| CPO | ✅ 手写 | ✅ `CPOTrainer` | 强一致 |
| DPOP | ✅ 手写 | — | 自实现 |
| Step-DPO | ✅ 手写 | — | 自实现 |
| Iterative DPO | ✅ 框架 | trl 组合 | 手写迭代 loop |
| Online DPO | ✅ on-policy 采样 | trl 部分支持 | 自实现 |
| Nash-LHF | ✅ Nash-MD 简化 | — | 自实现 |
| RainbowPO | ✅ 统一 config wrapper | 包装 trl | **核心 capstone 代码** |

---

## 6. 一致性测试

```python
def test_dpo_loss_equivalence():     # minimal vs trl loss < 1e-6
def test_ipo_loss():                  # IPO loss 公式正确
def test_kto_loss():                  # KTO 单边数据格式
def test_orpo_no_ref():               # 验证 ORPO 不需要 ref model
def test_simpo_length_norm():         # length norm 数值验证
def test_dpop_chosen_prob_up():       # 反例：DPO 下降，DPOP 上升
def test_rainbowpo_unification():     # 同配置下 RainbowPO ≡ trl Trainer
```

---

## 7. Notebook 结构（13 个）

每个 lecture 一个 ipynb：
1. 数据格式（chosen/rejected pair / KTO 单边）
2. loss 推导可视化（chosen vs rejected log prob）
3. minimal step-by-step
4. trl 对照
5. mini training（200 step）+ loss / reward margin 曲线
6. 与 DPO 对比图
7. 思考题

---

## 8. 环境配置

```
# requirements.txt (Windows native)
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
peft>=0.13
datasets, accelerate
matplotlib, seaborn   # 雷达图
```

**verify_env.py**:
- Part A: trl 各 Trainer init（DPO/KTO/ORPO/CPO）
- Part B: GPU
- Part C: DPOTrainer smoke test (Qwen-0.5B + 5 step)

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `dpo-core` | L1: DPO 推导 + minimal + trl 强一致 | 3 |
| `dpo-variants-1` | L2-L4: IPO + KTO + ORPO | 5 |
| `dpo-variants-2` | L5-L7: SimPO + CPO + DPOP | 5 |
| `dpo-step-online` | L8-L11: Step-DPO + Iterative + Online + Nash | 5 |
| `dpo-rainbow` | L12: RainbowPO 统一 | 2 |
| `dpo-family` | L13: Capstone 6 方法对照 | 3 |

---

## 10. 跨专题衔接

### 与专题 2 RLHF 的关系
DPO 是 RLHF 的"短路版"。L1 必须从 PPO + BT loss 推导 DPO closed-form。**这是 DPO 教学的核心**。

### 与专题 4 Process Reward 的关系
Step-DPO（L8）是 DPO 的步骤级版本，承接专题 4 的 PRM 思想。

### 与专题 5 R1 的关系
- DPO 是 offline preference，R1 是 online RL
- L11 Nash-LHF 与 R1 的 self-play 思想衔接

### 与专题 6 SOTA 的关系
2025 涌现的 β-DPO / Cal-DPO / α-DPO 等会在专题 6 单独深化。

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| DPO 推导难度高 | 高 | 中 | L1 用 5 步代数 + 几何图示 + 配套 notebook 演示 |
| 6 个 trainer config 易混 | 高 | 低 | RainbowPO 统一 config wrapper |
| 基座 GPT-2 偏好不敏感 | 中 | 中 | 用 Qwen2.5-0.5B 替代 |
| DPOP 反例难精心设计 | 中 | 低 | 提供 known-good 反例数据集 |
| length bias 在玩具数据不明显 | 中 | 低 | 用 SimPO 论文原数据复现 |
| 13 lecture 体量大 | 高 | 中 | 部分 lecture (Step-DPO/Online/sDPO) 压到 60 min |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-dpo-2023.md             # Rafailov et al.
├── 02-ipo-2023.md             # Azar et al.
├── 03-kto-2024.md
├── 04-orpo-2024.md
├── 05-simpo-2024.md
├── 06-cpo-2024.md
├── 07-dpop-2024.md
├── 08-step-dpo-2024.md
├── 09-iterative-dpo-oaif-2024.md
├── 10-online-dpo-sdpo-2024.md
├── 11-nash-lhf-2023.md
├── 12-rainbowpo-2024.md       # 统一框架
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-03-dpo-family.md` 的 6 个 Phase 推进。Capstone (L13) 在 Qwen2.5-0.5B + Anthropic-HH 1k 子集上跑 6 个方法（DPO/IPO/KTO/ORPO/SimPO/CPO），输出雷达图对比，单卡 5090 总 ≤ 6h。
