# Multimodal + Agent + 五线综合毕业 学习专题 — 设计文档

> **承接**: 全部前 6 专题（PEFT 三 + RL 三）
> **本专题**: 系列毕业 — 多模态/Agent/Long Context/Safety + 五线综合
> **战略地位**: ⭐⭐⭐⭐⭐ 系列最终收官
> **环境**: WSL2（继承专题 5-6）
> **总体规划**: `C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`

---

## 1. 专题定位

本专题是整个 PEFT + RL 学习系列的**毕业作品**，覆盖：
1. **多模态 RL**：Vision-R1 / VLM-R1 / Kimi vision
2. **Agent RL**：WebRL / SWE-Gym / ComputerRL / ToolRL
3. **Long Context RL**：Long2Short / 长文本 RL
4. **Test-Time Compute**：s1 / Don't Overthink / Claude/Gemini thinking
5. **Safety + RL**：Constitutional AI 升级 / Constitutional Classifiers / Safe-RLHF
6. **⭐⭐⭐ 五线综合 lecture**：Prompt + LoRA + Adapter + RLHF + 推理 RL 统一
7. **⭐⭐⭐ 毕业 Capstone**：同一道 GSM8K 题，5 种方法对照

### 1.1 为什么本专题是"毕业作品"
- 用户从 PEFT 入门一路学到 R1，本专题让所有 ckpt 在同一道题上对比
- L13 五线综合 lecture 把 88 方法归约成 4 维统一视角
- 跨模态/Agent 是 PEFT/RL 在 2025+ 的最大应用场景

### 1.2 系列总收官
完成本专题后，用户达成：
- 7 专题 / 88 方法 / 90 lectures / ~101h
- 28 (PEFT) + 60 (RL) = 88 方法横向对比
- 5 种 ckpt 在同一道题上的 inference 对照

---

## 2. 方法清单（12 种主线 + 五线综合 + 毕业作品）

| # | 方法 | 团队/年份 | 论文 | 核心 idea |
|---|------|----------|------|----------|
| 1 | **Vision-R1** ⭐⭐⭐⭐ | 2025.03 | arXiv 2503.06749 | cold-start + R1-style 多模态 |
| 2 | **VLM-R1** ⭐⭐⭐⭐ | OM-AI Lab 2025.02 | GitHub | GRPO 直接训 VLM |
| 3 | **Kimi k1.5 vision** | Moonshot 2025.01 | — | long context + vision 联合 |
| 4 | **WebRL** ⭐⭐⭐⭐ | THU 2024.11 | arXiv 2411.02337 | Web agent RL，自演化课程 |
| 5 | **SWE-Gym** ⭐⭐⭐⭐ | UIUC+CMU 2024.12 | arXiv 2412.21139 | 软件工程 RL benchmark |
| 6 | **ComputerRL / AutoGLM-OS** ⭐⭐⭐⭐ | 智谱 2025 | arXiv 2508.14040 | OS 级 agent，OSWorld 48.9% |
| 7 | **ToolRL** ⭐⭐⭐⭐ | 2025.04 | — | tool-use RL |
| 8 | **Long Context RL** | Kimi/DAPO | — | Long2Short / Token-PG / Overlong |
| 9 | **s1 budget forcing** ⭐⭐⭐⭐ | Stanford 2025.01 | — | "Wait" token 强制延长 |
| 10 | **Don't Overthink** ⭐⭐⭐⭐ | 2025 | — | TTC 不是越长越好 |
| 11 | **Claude/Gemini thinking** | Anthropic/Google 2025 | — | 商业 thinking models |
| 12 | **Constitutional Classifiers** | Anthropic 2025 | — | 2025 jailbreak 防御 |
| **+** | **五线综合 lecture (L13)** | — | — | ⭐⭐⭐ 系列毕业理论高峰 |
| **+** | **毕业 Capstone (L14)** | — | — | ⭐⭐⭐ 同题 5 ckpt 对照 |

---

## 3. Lecture 结构（14 篇 = 11 主线 + 1 多模态 capstone + 1 五线综合 + 1 毕业作品）

| Lecture | 主题 | 时长 |
|---------|------|------|
| **L1** Vision-R1 | cold-start + R1-style 多模态 | 60 min |
| **L2** VLM-R1 | GRPO 直接训 VLM | 60 min |
| **L3** Kimi k1.5 vision | long context + vision 联合 | 60 min |
| **L4** WebRL | Web agent RL | 60 min |
| **L5** SWE-Gym | 软件工程 RL benchmark | 60 min |
| **L6** ComputerRL | OS 级 agent | 60 min |
| **L7** ToolRL | tool-use RL | 60 min |
| **L8** Long Context RL | Long2Short + 长文本 | 60 min |
| **L9** Test-Time Compute | s1 / Don't Overthink | 90 min |
| **L10** Thinking Models 2026 | Claude 4 / Gemini 2.5 thinking | 60 min |
| **L11** Safety + RL | CAI 升级 / Constitutional Classifiers / Safe-RLHF | 60 min |
| **L12** Capstone-1: VLM-R1 玩具复现 | Qwen2-VL-2B + CLEVR counting | 240 min |
| **L13** ⭐ **五线综合 lecture** | Prompt+LoRA+Adapter+RLHF+R1 统一 | 90 min |
| **L14** ⭐ **毕业 Capstone**: 同题 5 ckpt 对照 | 系列最终演示 | 120 min |

**总学时**: 11×65 + 240 + 90 + 120 = 1165 min ≈ 19 hours

---

## 4. Lecture 模板（每篇 22-28 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 本方法的应用领域（vision / agent / long / safety）
## Slide 3-6: 方法核心（数据 + 算法 + reward）
## Slide 7-10: 代码逐行（minimal + 库对照）
## Slide 11-14: 实验结果（论文 + 我们玩具）
## Slide 15-17: 与前序 RL 主线（R1）的关系
## Slide 18-20: 跨模态/Agent 的特殊挑战
## Slide 21-23: 思考题 + 陷阱
## Slide 24-25: 下节预告
```

### L13 五线综合 lecture 特别结构（32 slides）

```
Part I: 五线回顾 (8 slides, 20 min)
  - 每条线的切入点 + 典型方法
  - 三个 PEFT 线（input/weight/structure）+ 两个 RL 线（distribution/trajectory）

Part II: 统一公式 (12 slides, 35 min)
  - 核心命题：p(y|x; θ_LM, φ)，五线只是 φ 的安放位置
  - 三句话一锤定音：
    PEFT 改 model
    RLHF 改 distribution
    R1 改 trajectory
  - 跨主线"等价对"（Prefix ≡ Parallel Adapter / LoRA = Adapter w/o σ）

Part III: 工程选型决策树 (8 slides, 20 min)
  - 4 个真实场景：客服 / 数学竞赛 / SWE agent / 千用户 SaaS

Part IV: 历史观 + 下一程 (4 slides, 15 min)
  - 大模型对齐 5 年史
  - 下一程: MoE / 长上下文 / Continuous Pretraining / World Model RL
```

---

## 5. 代码三轨策略

| 方法 | minimal | 库 1 | 库 2 |
|------|---------|------|------|
| VLM-R1 | ✅ 简化 (Qwen2-VL-2B + GRPO) | trl 适配 | verl |
| Vision-R1 | ✅ cold-start 演示 | trl | verl |
| WebRL | ✅ MiniWoB++ 玩具 | — | — |
| SWE-Gym | — | swe-gym 库 | — |
| ComputerRL | — | 官方 repo | — |
| ToolRL | ✅ 简化 | trl | — |
| s1 budget forcing | ✅ Wait token 实现 | — | — |
| Safe-RLHF | ✅ Lagrangian | trl 适配 | — |
| **unified_view.py** | ✅ 加载 5 ckpt 同题对照 | — | — |

---

## 6. 一致性测试

```python
def test_vlm_r1_reward_up():     # VLM 训完 counting acc 提升
def test_webrl_episode():        # 1 个 episode 跑通
def test_swe_gym_issue():        # 1 个 issue 解出
def test_s1_budget_forcing():    # Wait token 强制延长效果
def test_safe_rlhf_lagrangian(): # 多目标 Lagrange 收敛
def test_five_line_inference():  # 5 ckpt 全部加载 + 同题生成成功
```

---

## 7. Notebook 结构（14 个）

每个 lecture 一个 ipynb：
1. 应用场景介绍
2. 数据示例
3. minimal step-by-step
4. mini training
5. 关键指标可视化
6. 思考题

### L13 五线综合 notebook 特别结构
- 加载 5 个 ckpt（vanilla / LoRA / Adapter / DPO / R1）
- 在同一道 GSM8K 题上做 5 种 inference
- 可视化对比 + 性能表格
- 总结 88 方法的 4 维统一框架

### L14 毕业作品 notebook
- 完整 demo 系列学习成果
- 输出最终的"PEFT + RL 系列学习证书" markdown

---

## 8. 环境配置（WSL2 继承专题 5-6）

```
# requirements.txt (WSL2)
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
verl>=0.4
vllm>=0.7
qwen-vl-utils                # Qwen2-VL
miniwob                      # web agent
swe-gym                      # 软件工程 benchmark
playwright                   # browser agent
math-verify, sympy
matplotlib, seaborn
```

**verify_env.py**:
- Part A: 多模态库 + agent 库 import
- Part B: Qwen2-VL-2B 加载 smoke
- Part C: VLM-R1 GRPO 5-step smoke

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `mm-vision-r1` | L1-L3: Vision-R1 + VLM-R1 + Kimi vision | 5 |
| `mm-agent` | L4-L7: WebRL + SWE-Gym + ComputerRL + ToolRL | 6 |
| `mm-long-ttc` | L8-L10: Long Context + s1 + Thinking | 5 |
| `mm-safety` | L11: Constitutional + Safe-RLHF | 3 |
| `mm-capstone-vlm-r1` | L12: VLM-R1 玩具复现 | 3 |
| `mm-five-line-unified` | L13: 五线综合 lecture | 2 |
| `rl-graduation` | L14: 毕业作品 + README | 4 |

---

## 10. Capstone 详细设计

### Capstone-1: VLM-R1 玩具复现（L12）
- **基座**: Qwen2-VL-2B-Instruct + 4bit LoRA
- **任务**: CLEVR counting（"how many red cubes?"）
- **数据**: 1k train / 200 val
- **Reward**: format + 数值答案 exact match
- **算法**: GRPO
- **预期**: accuracy +20pp，response length 适度增长
- **耗时**: WSL2 5090 24GB, 4h

### Capstone-2: ⭐⭐⭐ 五线综合 — 同题 5 ckpt 对照（L14）
- **题目**: 同一道 GSM8K 题（如 Janet 鸡蛋题）
- **5 种 ckpt**:
  1. **Vanilla**: GPT-2 base 直接推理
  2. **LoRA**: 从 lora-family L01 ckpt（如有）
  3. **Pfeiffer Adapter**: 从 adapter L01 ckpt
  4. **DPO**: 从专题 3 capstone ckpt
  5. **R1-Zero**: 从专题 5 capstone-A ckpt
- **输出**:
  - 5 个 response 对比
  - 推理 trace 可视化（R1 的 thinking vs 其他）
  - 性能表格（accuracy / length / latency）
- **耗时**: 5090 1h
- **意义**: 用户整个学习历程的"成果展"

---

## 11. 跨专题衔接

### 与前 6 专题的依赖
- 专题 1-6 ckpt 全部要保留，本专题 L14 全部加载
- README 必须有"如何运行毕业 Capstone"的指引

### 与未来专题的衔接（README 中预告）
- MoE (Mixtral / DeepSeek-MoE)
- World Model RL
- Continuous Pretraining
- Test-time scaling 深化

---

## 12. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| VLM 显存 OOM | 中 | 高 | Qwen2-VL-2B + 4bit + LoRA |
| WebRL/SWE-Gym 数据准备复杂 | 高 | 中 | 提供 1 个完整 episode 演示 |
| ComputerRL 依赖 OS 环境 | 高 | 中 | 用作者 docker，本专题只演示 |
| 5 ckpt 全加载需要存储 | 中 | 低 | 提供 ckpt 下载脚本 |
| 五线综合 lecture 抽象度高 | 高 | 中 | 大量 ASCII 图 + notebook 数值验证 |
| L14 毕业作品依赖前置 ckpt | 高 | 高 | 提供 fallback ckpt 下载 + 跳过指引 |
| 14 lecture + 双 capstone 过长 | 中 | 低 | L1-L11 控制 60 min，capstone 集中精力 |

---

## 13. 论文 / 资料占位

```
papers/
├── 01-vision-r1-2025.md
├── 02-vlm-r1-2025.md
├── 03-kimi-k1.5-vision-2025.md
├── 04-webrl-2024.md
├── 05-swe-gym-2024.md
├── 06-computer-rl-2025.md
├── 07-tool-rl-2025.md
├── 08-long-context-rl-2025.md
├── 09-s1-budget-forcing-2025.md
├── 10-dont-overthink-2025.md
├── 11-claude-extended-thinking-2025.md
├── 12-gemini-thinking-2025.md
├── 13-constitutional-classifiers-2025.md
├── 14-safe-rlhf-2023.md
└── README.md
```

---

## 14. 实施方案

按 plan 文件 `2026-06-03-multimodal-agent.md` 的 7 个 Phase 推进。**双 capstone**:
- L12 VLM-R1 玩具复现（必跑，4h）
- L14 五线综合毕业作品（必跑，1h，前提是前置 ckpt 都在）

完成后 tag `rl-graduation` — ⭐⭐⭐ 系列收官里程碑。
