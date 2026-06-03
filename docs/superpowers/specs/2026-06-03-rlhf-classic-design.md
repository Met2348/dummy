# RLHF Classic 学习专题 — 设计文档

> **承接**: rl-foundations (RL 算法基础)
> **本专题**: 完整复刻 InstructGPT 三段管线（SFT → RM → PPO）
> **战略地位**: RL for LLM 系列的"经典代表作"
> **总体规划**: `C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`

---

## 1. 专题定位

RLHF Classic 教用户跑通 **InstructGPT 三段式 (SFT + Reward Model + PPO)**。是历史上第一个让 LLM 学会"按人类偏好回答"的工程范式，至今 OpenAI/Anthropic/Meta 主线仍在沿用（即使内部细节有改进）。

### 1.1 本专题与上下游
- **上游**：专题 1 PPO for LLM 框架已就绪
- **下游**：
  - 专题 3 DPO 会用本专题的 BT-RM 公式推导 implicit reward
  - 专题 4 PRM 会扩展 ORM → PRM
  - 专题 5 R1 会与传统 RLHF 对照

### 1.2 三段式的工程难度梯度
1. **SFT**（最简单）：监督微调，与 PEFT 无差
2. **RM**（中等）：BT loss + pairwise data
3. **PPO**（最难）：4 model 协同、KL 控制、reward hacking 防范

---

## 2. 方法清单（12 种）

| # | 方法 | 年份 | 论文 | 核心 idea |
|---|------|------|------|----------|
| 1 | **InstructGPT** | 2022 | Ouyang et al. | 三段式范式定义 |
| 2 | **SFT** | 2020+ | 多篇 | 监督微调（指令格式）|
| 3 | **BT Reward Model** | 1952/2022 | Bradley-Terry | pairwise → scalar reward |
| 4 | **PPO for LLM (深化)** | 2022+ | OpenAI | token-level + KL ref + 4 model |
| 5 | **Constitutional AI / RLAIF** | 2022 | Bai et al. (Anthropic) | AI 当裁判 |
| 6 | **Sparrow** | 2022 | DeepMind | 规则约束 + 检索 |
| 7 | **LLaMA-2 RLHF** | 2023 | Meta | 迭代式 RLHF (rejection sampling) |
| 8 | **Reward Hacking** | 2022 | Gao et al. | RM over-optimization 问题 |
| 9 | **KL Adaptive Control** | 2022 | OpenAI | KL 系数动态调整 |
| 10 | **Length Bias 治理** | 2023 | — | length normalization tricks |
| 11 | **MaxMin-RLHF** | 2024 | — | 多目标最坏情况优化 |
| 12 | **Multi-objective RLHF** | 2024 | — | helpful + harmless 加权 |

---

## 3. Lecture 结构（12 篇 = 11 主线 + 1 capstone）

| Lecture | 主题 | 主方法 | 时长 |
|---------|------|--------|------|
| **L1** InstructGPT 三段式 | InstructGPT 全景 | 90 min |
| **L2** SFT | 指令微调 + 数据格式 | 60 min |
| **L3** Reward Model | BT loss + pairwise | 90 min |
| **L4** PPO for LLM 深化 | KL penalty / 4 model 协同 | 90 min |
| **L5** RLHF 工程细节 | rollout / KL adaptive / advantage norm | 90 min |
| **L6** RLAIF / Constitutional AI | Anthropic 范式 | 60 min |
| **L7** LLaMA-2 RLHF | rejection sampling + iterative | 60 min |
| **L8** Sparrow | 规则 + 检索 RLHF | 60 min |
| **L9** Reward Hacking | over-optimization 实验 | 90 min |
| **L10** RLHF 陷阱合集 | length bias / sycophancy / 对齐税 | 60 min |
| **L11** 多目标 RLHF | MaxMin / Pareto | 60 min |
| **L12** Capstone：TL;DR 摘要 RLHF | 完整三段管线 | 180 min |

**总学时**: 11×70 + 180 = 950 min ≈ 16 hours

---

## 4. Lecture 模板（PPT-style，每篇 22-28 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 三段式中的位置（每个 lecture 都标注当前在 SFT/RM/PPO 哪段）
## Slide 3-5: 动机 + 核心公式
## Slide 6-10: 工程细节（数据格式、loss、训练循环）
## Slide 11-15: 代码逐行（minimal + trl 对照）
## Slide 16-18: 论文 claim 复现
## Slide 19-20: 与前序方法对比 / 跨专题衔接
## Slide 21-23: 思考题 + 陷阱与警示
## Slide 24-25: 下节预告
```

---

## 5. 代码三轨策略（本专题两轨：minimal + trl）

| 方法 | minimal | trl | 备注 |
|------|---------|-----|------|
| SFT | ✅ 手写 | ✅ `SFTTrainer` | 强一致 |
| BT-RM | ✅ 手写 | ✅ `RewardTrainer` | 强一致 |
| PPO-LLM | ✅ 手写 | ✅ `PPOTrainer` | 弱一致（实现细节多）|
| CAI / RLAIF | ✅ 手写 demo | — | 无标准库 |
| Reward Hacking | ✅ 长度 reward demo | — | 教学性 |
| MaxMin-RLHF | ✅ 手写 | — | 自实现 |
| Multi-obj | ✅ 手写 | — | 自实现 |

**目录约定**:
- `{stage}_minimal.py` — 手写最小实现 (sft / rm / ppo)
- `{stage}_trl.py` — trl 对照
- `cai_demo.py` — Constitutional AI 演示
- `reward_hacking_demo.py` — 长度奖励陷阱演示
- `multiobj_rlhf.py` — 多目标

---

## 6. 一致性测试

```python
def test_sft_loss():            # minimal vs trl SFT loss < 1e-6
def test_rm_bt_loss():          # BT loss 数值一致
def test_rm_accuracy():         # 玩具数据 RM 准确率 > 60%
def test_ppo_pipeline():        # 三段都能跑 + 中间产物保存
def test_kl_under_control():    # 100 step PPO 后 KL < 10
def test_reward_hacking_visible(): # 长度 reward 学到生成超长文本
```

---

## 7. Notebook 结构（12 个）

每个 lecture 一个 ipynb：
1. 三段式所处阶段提示（永远显示当前在哪段）
2. import + 模型加载
3. 数据可视化（preference data 长啥样）
4. minimal step-by-step
5. trl 对照
6. 关键指标可视化（RM accuracy / PPO reward / KL）
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (Windows native)
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
peft>=0.13                # PPO with LoRA 节省显存
datasets, accelerate, scipy
wandb 或 tensorboard
matplotlib, seaborn
```

**verify_env.py**:
- Part A: trl 基础 import + SFTTrainer / RewardTrainer / PPOTrainer 都能 init
- Part B: GPU + sm_120
- Part C: 三段 smoke test（GPT-2 各跑 5 步）

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `rlhf-sft-rm` | L1-L3: 三段式定义 + SFT + RM | 5 |
| `rlhf-ppo-llm` | L4-L5: PPO 深化 + 工程细节 | 4 |
| `rlhf-rlaif` | L6-L8: CAI + Sparrow + LLaMA-2 | 4 |
| `rlhf-pitfalls` | L9-L11: 陷阱 + 多目标 | 4 |
| `rlhf-classic` | L12: Capstone TL;DR 完整管线 | 3 |

---

## 10. 跨专题衔接

### 与专题 3 DPO 的关键桥梁
DPO loss 是从本专题的 PPO + BT-RM 直接推导而来。L3 Reward Model 必须把 BT loss 推导讲透，否则 DPO 推导卡壳。

### 与专题 4 PRM 的衔接
本专题用的是 ORM（outcome RM，整段答案打分），专题 4 扩展到 PRM（每步打分）。

### 与专题 5 R1 的对照
R1 用 RLVR（可验证奖励，规则代替 RM），与本专题的 BT-RM 形成鲜明对比。

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Anthropic-HH 太大 | 高 | 中 | 1k 精选子集脚本 |
| RM 准确率上不去 | 中 | 中 | 期望管理 60-70% |
| PPO 三段失败传染 | 中 | 高 | 每段中间产物保存 + 单元测试 |
| Reward hacking 难复现 | 低 | 低 | L9 用极端长度 reward 必出 |
| 训练时间过长 | 高 | 中 | GPT-2-medium + 1k 数据 + ≤ 5h |
| trl PPOTrainer 在 transformers 5.x 兼容 | 中 | 高 | 独立 venv，pin 版本 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-instructgpt-2022.md
├── 02-bradley-terry-1952.md
├── 03-constitutional-ai-2022.md
├── 04-sparrow-2022.md
├── 05-llama2-2023.md
├── 06-reward-hacking-2022.md      # Gao et al.
├── 07-rlhf-pitfalls-2023.md       # 综合 review
├── 08-maxmin-rlhf-2024.md
├── 09-multiobj-rlhf-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-03-rlhf-classic.md` 的 6 个 Phase 推进，每 Phase 1 commit + 部分 tag。Capstone (L12) 完整跑一遍 GPT-2-medium + Anthropic-HH 1k 子集（总 ≤ 5h on 5090）。
