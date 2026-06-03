# SSM / Hybrid Implementation Plan

**Goal**: 完整实现 SSM/Mamba/RWKV/Hybrid 学习包（11 lecture + 多个 block 实现 + 6 测试 + 11 notebook + Capstone 130M mini-Mamba）

**Architecture**: 11 lectures = 10 主线 + 1 capstone。三轨代码：手写 minimal + 库（mamba-ssm / rwkv）。WSL2 环境。

**Tech Stack**: torch + mamba-ssm + causal-conv1d + rwkv + einops

**Design 文档**: `docs/superpowers/specs/2026-06-04-ssm-hybrid-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/ssm-hybrid/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + mamba-ssm + causal-conv1d + rwkv + einops

### Task 1.3: environment/verify_env.py
- Part A: torch + mamba-ssm + causal-conv1d import
- Part B: GPU + sm_120
- Part C: Mamba forward 10 step smoke

### Task 1.4: src/common.py
- discretization helpers / scan helpers

### Task 1.5: papers/ 11 个占位 + README index

### Commit: `chore: ssm-hybrid scaffold`

---

## Phase 2: L01-L02 SSM 基础

### Task 2.1: lectures/01-ssm-intro.md (24 slides)
- HiPPO 多项式投影 + 状态空间方程
- 连续 vs 离散

### Task 2.2: lectures/02-s4-s5.md (24 slides)
- S4 卷积形式 + 频域离散化 + S5 简化

### Task 2.3: src/s4_naive.py
- 手写 S4 卷积形式
- HiPPO matrix 初始化

### Task 2.4: src/tests/test_s4_conv_vs_recurrent.py
- 同 input 两种形式输出一致

### Task 2.5: notebooks/01-ssm-intro.ipynb + 02-s4-s5.ipynb

### Commit + Tag: `ssm-foundations`

---

## Phase 3: L03-L05 Mamba 1/2/3

### Task 3.1: lectures/03-mamba.md (32 slides)
- Selective SSM + 硬件 scan kernel
- 与 attention 的本质区别

### Task 3.2: src/mamba_block.py
- 手写 Mamba block (无 selective scan kernel,naive scan)
- 与 mamba_ssm 库对照

### Task 3.3: src/mamba_lib.py
- mamba-ssm 库调用 + benchmark

### Task 3.4: src/tests/test_mamba_correctness.py
- naive vs mamba-ssm 库 < 1e-4

### Task 3.5: lectures/04-mamba2.md (24 slides)
- SSD / 矩阵分解 / 与 attention 等价性

### Task 3.6: src/mamba2_block.py
- Mamba-2 SSD 形式

### Task 3.7: lectures/05-mamba3.md (16 slides)
- 长上下文优化（可选用最新论文 release）

### Task 3.8: notebooks/03-mamba.ipynb + 04-mamba2.ipynb + 05-mamba3.ipynb

### Commit + Tag: `ssm-mamba`

---

## Phase 4: L06-L07 RWKV / RetNet

### Task 4.1: lectures/06-rwkv-7.md (20 slides)
- RWKV-7 linear attention 路线
- 与 Mamba 对比

### Task 4.2: src/rwkv_block.py
- RWKV-7 简化版

### Task 4.3: lectures/07-retnet.md (16 slides)
- RetNet retention mechanism
- 与 RWKV-6 关系

### Task 4.4: notebooks/06-rwkv-7.ipynb + 07-retnet.ipynb

### Commit + Tag: `ssm-rwkv`

---

## Phase 5: L08-L10 Hybrid 设计

### Task 5.1: lectures/08-jamba.md (24 slides)
- AI21 Jamba: Mamba+attn+MoE
- 1:7 attn:mamba 配比

### Task 5.2: src/jamba_block.py
- hybrid layer 配比
- attn 层 + mamba 层 + MoE 层组合

### Task 5.3: lectures/09-zamba.md (16 slides)
- Zamba / Zamba-2 共享 attn + Codestral-Mamba 代码

### Task 5.4: src/zamba_block.py
- 共享 attention 实现

### Task 5.5: lectures/10-hybrid-design.md (20 slides)
- 何时混 / 比例 / 选 layer / 经验法则

### Task 5.6: notebooks/08-jamba.ipynb + 09-zamba.ipynb + 10-hybrid-design.ipynb

### Commit + Tag: `ssm-hybrid-arch`

---

## Phase 6: L11 Capstone

### Task 6.1: lectures/11-capstone-mamba-mini.md (24 slides)
- 130M Mamba 设计 + 训练 recipe

### Task 6.2: src/mini_mamba.py
- 24 层 / d_model 512 / 完整 Mamba

### Task 6.3: src/capstone_train_mini_mamba.py
- 在专题 1 输出 1B-token 上训练
- 与同算力 GPT-mini 对照

### Task 6.4: src/tests/test_long_context_extrapolation.py
- 训 1k 推 4k 的 perplexity 不爆

### Task 6.5: src/tests/test_mini_mamba_train_loss.py
- 500 step loss 单调下降

### Task 6.6: notebooks/11-capstone.ipynb

### Task 6.7: learning/ssm-hybrid/README.md

### Commit + Tag: `ssm-hybrid`

---

## 验证清单

```bash
python learning/ssm-hybrid/environment/verify_env.py
python -m pytest learning/ssm-hybrid/src/tests/ -v
jupyter nbconvert --execute --inplace learning/ssm-hybrid/notebooks/*.ipynb
python learning/ssm-hybrid/src/capstone_train_mini_mamba.py
```

预期：
- env 三段 PASS
- 6 tests PASS
- 11 notebook 跑通
- Capstone Mamba train loss 下降 + 长上下文外推

---

## 总览

- 11 lectures × 平均 75 min = 14h slides
- 多个 block 实现（S4/Mamba/Mamba-2/RWKV/Jamba/Zamba）
- 6 tests
- 11 notebooks
- 1 capstone (130M Mamba)
- 预计 6 commit + 5 tag
- 总时长 12h（lecture + notebook + Capstone）
