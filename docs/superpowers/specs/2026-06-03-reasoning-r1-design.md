# Reasoning RL — R1 时代 学习专题 — 设计文档

> **承接**: rlhf-classic + dpo-family + process-reward
> **本专题**: 整个系列的高峰 — GPT-2 微缩复现 R1-Zero（双轨）
> **战略地位**: ⭐⭐⭐⭐⭐ 系列最大卖点
> **环境**: **WSL2 切换点**（verl + Ray + Megatron + vllm 在 Windows 装不上）
> **总体规划**: `C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`

---

## 1. 专题定位

DeepSeek-R1 (2025.01) 重塑了行业 — 它证明：
1. **纯 RL 无 SFT 可以涌现复杂推理**（R1-Zero）
2. **GRPO 算法**（去 critic、组内相对优势）足够支撑大规模推理 RL
3. **rule-based reward**（无需 PRM）+ 长 CoT 是可行路径
4. **aha moment 涌现**（模型自发出现 "wait, let me reconsider"）

本专题完整复刻 R1 的算法 + 工程路径，capstone 是 **GPT-2 微缩复现 R1-Zero 双轨**：
- 教学轨：GPT-2-medium + Countdown-3（必跑，验证 pipeline）
- 挑战轨：Qwen2.5-1.5B + LoRA + GSM8K-tiny（选跑，看 aha moment）

### 1.1 为什么是 R1 时代的"系列高峰"
- 整个 RL for LLM 系列前 4 个专题（PPO/RLHF/DPO/PRM）都在为 R1 时代铺路
- R1-Zero 是历史上第一次"无 SFT 涌现推理"，意义可类比 GPT-3 的 in-context learning
- TinyZero 证明 $30 + Qwen-3B 可复现，门槛已大幅降低

### 1.2 环境切换：Windows → WSL2
- 原因：verl + Ray + Megatron + vllm 在 Windows native 装不上
- 时机：本专题 L00（专门一节 setup lecture）
- 后续：专题 6-7 全部在 WSL2 完成

---

## 2. 方法清单（15 种）

| # | 方法 | 团队/年份 | 论文 | 核心 idea |
|---|------|----------|------|----------|
| 0 | **WSL2 + verl + vllm 环境** | — | — | 环境切换 |
| 1 | **OpenAI o1** | OpenAI 2024.09 | (blog) | 推理时 RL scaling 范式定义 |
| 2 | **GRPO** | DeepSeek 2024.02 | DeepSeekMath | 组内相对优势，去 critic |
| 3 | **DeepSeek-R1-Zero** | DeepSeek 2025.01 | — | **纯 RL 无 SFT 涌现 aha moment** |
| 4 | **DeepSeek-R1** | DeepSeek 2025.01 | — | cold-start + 4 阶段训练 |
| 5 | **Kimi k1.5** | Moonshot 2025.01 | — | long context RL + 多模态 |
| 6 | **RLOO** | Ahmadian 2024 | — | Leave-One-Out baseline |
| 7 | **ReMax** | Li 2023 | — | 去 critic 简化 PPO |
| 8 | **VinePPO** | 2024 | — | Monte Carlo advantage |
| 9 | **REINFORCE++** | OpenRLHF 2025.01 | — | 极简 PPO 替代 |
| 10 | **TinyZero** | UC Berkeley 2025.01 | (blog+repo) | $30 + Qwen-3B 复现 aha |
| 11 | **Open-R1** | HuggingFace 2025.01 | (blog) | 三步走 + 350k MoT 数据 |
| 12 | **Open-Reasoner-Zero** | StepFun 2025.03 | — | 1/10 步数复现 |
| 13 | **Mini-R1** | Schmid 2025 | — | 教学版 |
| 14 | **Spurious Rewards** | 2025.06 | — | ⚠️ Qwen 随机奖励涨 21pt 警示 |

---

## 3. Lecture 结构（15 篇 = 13 主线 + 0 setup + 2 capstone）

| Lecture | 主题 | 时长 |
|---------|------|------|
| **L0** WSL2 + verl + vllm setup | 环境切换 | 60 min |
| **L1** OpenAI o1 范式 | 推理时 RL scaling | 60 min |
| **L2** GRPO 完整推导 | 与 PPO 数学对比 | 90 min |
| **L3** DeepSeek-R1-Zero | 纯 RL 涌现 aha moment | 90 min |
| **L4** DeepSeek-R1 | cold-start + 4 阶段 | 90 min |
| **L5** Kimi k1.5 | long context + 多模态 RL | 60 min |
| **L6** RLOO | Leave-One-Out baseline | 60 min |
| **L7** ReMax | 去 critic | 60 min |
| **L8** VinePPO | Monte Carlo advantage | 60 min |
| **L9** REINFORCE++ | 极简 PPO 替代 | 60 min |
| **L10** TinyZero | $30 复现 + Countdown 任务 | 60 min |
| **L11** Open-R1 | HuggingFace 完整复现路径 | 60 min |
| **L12** Spurious Rewards 警示 | Qwen 随机 reward 涨 21pt | 90 min（⚠️ 必看）|
| **L13** Capstone-A: GPT-2-M 教学轨 | format reward 5%→95% | 180 min |
| **L14** Capstone-B: Qwen-1.5B 挑战轨 | 看 aha moment | 240 min |
| **L15** R1 时代工程 takeaway | 算法 + 工程总结 | 60 min |

**总学时**: 60 + 13×70 + 180 + 240 + 60 = 1450 min ≈ 24 hours（含双轨 capstone）

---

## 4. Lecture 模板（每篇 22-28 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 在 R1 时代谱系中的位置（GRPO/R1-Zero/Kimi/...）
## Slide 3-6: 算法公式（与 PPO/GRPO 对比）
## Slide 7-10: 工程细节（rollout / batch / KL / clip）
## Slide 11-15: 代码逐行（minimal + trl + verl 对照）
## Slide 16-19: 论文实验观察（aha moment / length / accuracy）
## Slide 20-22: 陷阱与警示（Spurious Rewards / length bias）
## Slide 23-25: 与其他 R1 变体对比
## Slide 26-28: 思考题 + 下节预告
```

---

## 5. 代码三轨策略（minimal + trl + verl）

| 方法 | minimal | trl | verl | 备注 |
|------|---------|-----|------|------|
| GRPO | ✅ 手写 | ✅ GRPOTrainer | ✅ recipe | **强一致** |
| R1-Zero 训练 loop | ✅ 完整 | trl + 自定义 | verl 推荐 | 双轨 capstone |
| RLOO | ✅ 手写 | ✅ RLOOTrainer | ✅ | 一致性测试 |
| ReMax | ✅ 手写 | — | ✅ | 自实现 |
| VinePPO | ✅ MC 优势 | — | ✅ | 自实现 |
| REINFORCE++ | ✅ 手写 | — | ✅ | 自实现 |
| TinyZero | — | trl + 数据 | ✅ recipe | 直接跑作者 repo |
| Open-R1 | — | trl 集成 | — | 完整管线 demo |

**目录约定**:
- `{method}_minimal.py` — 手写
- `{method}_trl.py` — trl 对照
- `{method}_verl.py` — verl 配置文件 + 启动脚本
- `r1_zero_track_a/` — GPT-2-M 教学轨完整 src
- `r1_zero_track_b/` — Qwen-1.5B 挑战轨完整 src
- `rewards/` — format_reward.py / accuracy_reward.py / combined_reward.py

---

## 6. 一致性测试

```python
def test_grpo_loss_equiv():      # minimal vs trl GRPO loss < 1e-6
def test_rloo_baseline():        # leave-one-out 数值正确
def test_format_reward():        # regex 正确识别 <think>/<answer>
def test_accuracy_reward():      # GSM8K answer parse 正确
def test_track_a_pipeline():     # 教学轨 200 step format reward 上升
def test_track_b_aha_moment():   # 挑战轨 aha 词频 > 5%（选测）
```

---

## 7. Notebook 结构（15 个）

每个 lecture 一个 ipynb：
1. 该方法在 R1 谱系中位置图
2. 算法公式可视化
3. minimal step-by-step
4. trl / verl 对照
5. mini training（GPT-2-M + Countdown 50 step）
6. 关键指标（format_acc / accuracy / length / aha 词频）
7. 思考题

---

## 8. 环境配置（⚠️ WSL2 切换）

```
# requirements.txt (WSL2 Ubuntu 22.04)
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
verl>=0.4                  # DAPO/GRPO 默认含
vllm>=0.7                  # ⚠️ 必须，rollout 15x 加速
ray>=2.30
peft, bitsandbytes
math-verify, sympy         # verifier
datasets, accelerate
wandb 或 tensorboard
```

**WSL2 setup（L0 lecture 完整覆盖）**:
1. Windows Terminal + WSL2 Ubuntu 22.04
2. CUDA 13.0 + cuDNN
3. Python 3.11 + venv
4. verl 安装 + vllm（known-good Dockerfile 备份）

**verify_env.py 三段式**:
- Part A: torch + transformers + trl + verl import
- Part B: vllm 单卡 inference smoke
- Part C: verl GRPO 5-step smoke + Ray cluster init

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `r1-wsl2-setup` | L0: 环境切换完成 | 2 |
| `r1-grpo-core` | L1-L2: o1 + GRPO 推导 | 3 |
| `r1-deepseek` | L3-L5: R1-Zero + R1 + Kimi | 5 |
| `r1-variants` | L6-L9: RLOO + ReMax + VinePPO + REINFORCE++ | 6 |
| `r1-reproductions` | L10-L11: TinyZero + Open-R1 | 4 |
| `r1-spurious-warning` | L12: Spurious Rewards 警示 | 2 |
| `r1-capstone-track-a` | L13: GPT-2-M 教学轨完成 | 4 |
| `r1-capstone-track-b` | L14: Qwen-1.5B 挑战轨（选测）| 2 |
| `reasoning-r1` | L15 + README | 3 |

---

## 10. Capstone 双轨详细设计

### Capstone-A: 教学轨（必跑）
- **基座**: GPT-2-medium (355M)
- **任务**: Countdown-3（用 3 个数字 + 四则凑目标）
- **数据**: 5k train / 500 val（提供生成器）
- **Reward**: `0.1 × format + 0.9 × accuracy`
  - format: `<think>...</think><answer>...</answer>` regex 严格匹配
  - accuracy: 计算结果是否 == target
- **算法对照**（核心教学价值）:
  1. REINFORCE + mean baseline（200 step，看不稳定）
  2. RLOO k=8（显著稳定化）
  3. GRPO k=8（KL constraint，length 开始涨）
  4. GRPO + DAPO Clip-Higher（aha 早现，但 GPT-2-M 可能不行）
- **预期**: format 5%→95%, accuracy 5%→15%, length 50→150
- **耗时**: 5090 24GB, 4 算法各 1.5h, 总 6h

### Capstone-B: 挑战轨（选跑）
- **基座**: Qwen2.5-1.5B-Base + 4bit LoRA
- **任务**: GSM8K-tiny (500 train / 100 test)
- **Reward**: 同上
- **算法**: 只跑 GRPO + DAPO
- **预期**: accuracy 5%→25%, length 100→250
- **关键观察**: wait/recheck/let me reconsider 词频 5-10%（⭐ **aha moment**）
- **耗时**: 5090 24GB, 单跑 4h

---

## 11. 风险与缓解（系列最大）

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| WSL2 配置失败 | 中 | 高 | L0 lecture 完整覆盖 + Dockerfile 备份 |
| verl + vllm 安装坑 | 高 | 高 | known-good 版本 pin + 故障排查指南 |
| GPT-2-M aha 不出现 | 高 | 中 | 明确"教学轨只追 pipeline" |
| Qwen-1.5B 24GB OOM | 中 | 高 | 4bit + LoRA + max_response=256 + k=4 |
| 4h+ 训练时长 | 高 | 中 | 提供预训 ckpt 给学员下载 |
| **Spurious Rewards 暴击** | 中 | 高 | L12 专 lecture + 强调"reward 涨 ≠ 真涨" |
| GRPO length bias | 中 | 中 | 用 Dr. GRPO 修复（专题 6 详讲）|
| Countdown 数据生成 bug | 低 | 中 | 严格单元测试 |

---

## 12. 论文 / 资料占位

```
papers/
├── 00-o1-blog-2024.md             # OpenAI o1
├── 01-deepseekmath-grpo-2024.md
├── 02-deepseek-r1-2025.md
├── 03-kimi-k1.5-2025.md
├── 04-rloo-2024.md
├── 05-remax-2023.md
├── 06-vineppo-2024.md
├── 07-reinforce-plus-plus-2025.md
├── 08-tinyzero-2025.md            # blog + repo
├── 09-open-r1-2025.md             # HuggingFace blog
├── 10-open-reasoner-zero-2025.md  # StepFun
├── 11-spurious-rewards-2025.md    # ⚠️ 警示
├── 12-r1-engineering-2025.md      # 综合 takeaway
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-03-reasoning-r1.md` 的 9 个 Phase 推进。**双轨 capstone**：教学轨在 L13 必跑（5090 6h），挑战轨在 L14 选跑（5090 4h+）。完成后 tag `reasoning-r1` — 系列高峰。
