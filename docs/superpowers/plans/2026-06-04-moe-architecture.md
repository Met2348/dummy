# MoE Architecture Implementation Plan

**Goal**: 完整实现 MoE 架构学习包（13 lecture + 多个 router 实现 + 7 测试 + 13 notebook + Capstone 4-expert mini-MoE）

**Architecture**: 13 lectures = 12 主线 + 1 capstone。三轨代码：手写 minimal + megablocks 库 + DeepSpeed-MoE 工业级。WSL2 环境（环境切换点）。

**Tech Stack**: torch + flash-attn + megablocks + deepspeed + einops + transformers

**Design 文档**: `docs/superpowers/specs/2026-06-04-moe-architecture-design.md`

---

## Phase 1: 基础设施（WSL2 切换）

### Task 1.1: 目录骨架
- Create `learning/moe-architecture/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: lectures/00-wsl2-megablocks-setup.md (16 slides)
- WSL2 切换（如已在 Module 2 末切，复用）
- megablocks / deepspeed 安装

### Task 1.3: environment/requirements.txt
- torch + flash-attn + megablocks + deepspeed + einops

### Task 1.4: environment/verify_env.py
- Part A: torch + megablocks + deepspeed import
- Part B: GPU + sm_120
- Part C: top-2 MoE forward 10 step smoke

### Task 1.5: src/common.py
- expert_init / load_balance metrics / capacity drop helpers

### Task 1.6: papers/ 12 个占位 + README index

### Commit: `chore: moe-architecture scaffold + WSL2 setup`

---

## Phase 2: L01-L04 路由四方法

### Task 2.1: lectures/01-moe-intro.md (18 slides)
- Shazeer 2017 sparse MoE + gating

### Task 2.2: src/moe_layer_naive.py
- 手写 4 expert + top-2 gating + aux loss
- 200 行清晰教学版

### Task 2.3: lectures/02-gshard.md (20 slides)
- top-2 routing + expert parallel + capacity factor

### Task 2.4: src/gshard_router.py
- top-2 + aux loss + capacity drop

### Task 2.5: lectures/03-switch.md (18 slides)
- top-1 简化 + 1.6T 参数实战

### Task 2.6: src/switch_router.py
- top-1 + load balance

### Task 2.7: lectures/04-expert-choice.md (18 slides)
- 反向路由（experts 选 tokens）

### Task 2.8: src/expert_choice.py
- reversed routing 实现

### Task 2.9: src/tests/test_router_load_balance.py
- 训 100 step,各 expert 利用率 1/k ± 30%

### Task 2.10: notebooks/01-intro.ipynb + 02-gshard.ipynb + 03-switch.ipynb + 04-expert-choice.ipynb

### Commit + Tag: `moe-routing`

---

## Phase 3: L05-L06 Mixtral + DeepSeekMoE

### Task 3.1: lectures/05-mixtral.md (18 slides)
- Mistral 2023 开源 MoE 起点

### Task 3.2: src/mixtral_load.py
- Mixtral-8x7B 加载（4bit）+ 架构统计

### Task 3.3: lectures/06-deepseekmoe.md (22 slides)
- DeepSeek-V2 细粒度 + 共享专家

### Task 3.4: src/deepseekmoe_layer.py
- 细粒度专家 + 共享专家 layer

### Task 3.5: notebooks/05-mixtral.ipynb + 06-deepseekmoe.ipynb

### Commit + Tag: `moe-modern`

---

## Phase 4: L07 Aux-Loss-Free（核心 ⭐）

### Task 4.1: lectures/07-aux-loss-free.md (32 slides)
- 完整数学推导
- 偏置项更新公式 + 与 aux loss 对比

### Task 4.2: src/aux_loss_free.py
- DeepSeek-V3 偏置项实现
- update_rate=1e-3 严格按论文
- bias_init=0

### Task 4.3: src/tests/test_aux_free_stability.py
- 100 step 偏置项收敛 + 负载均衡

### Task 4.4: notebooks/07-aux-loss-free.ipynb
- 偏置项动态可视化 + 与 aux loss 对照

### Commit + Tag: `moe-aux-free`

---

## Phase 5: L08-L10 Phi-MoE / Qwen3-MoE / MoR

### Task 5.1: lectures/08-phi-moe.md (16 slides)
- Phi-3.5-MoE / Phi-4-MoE 小 MoE 路线

### Task 5.2: src/phi_moe_load.py
- Phi-3.5-MoE 加载（4bit）

### Task 5.3: lectures/09-qwen3-moe.md (16 slides)
- Qwen3-A3B / 235B 系列

### Task 5.4: src/qwen3_moe_load.py
- Qwen3-MoE 加载 + 架构统计

### Task 5.5: lectures/10-mor.md (16 slides)
- MoR Mixture of Recursions 新方向

### Task 5.6: notebooks/08-phi-moe.ipynb + 09-qwen3-moe.ipynb + 10-mor.ipynb

### Commit + Tag: `moe-small` + `moe-new`

---

## Phase 6: L11-L12 训练稳定 + 推理优化

### Task 6.1: lectures/11-moe-training.md (26 slides)
- router z-loss / capacity factor / 路由崩塌典型

### Task 6.2: src/router_z_loss.py + crash_demo.py
- z-loss 实现 + 崩塌注入实验

### Task 6.3: lectures/12-moe-inference.md (24 slides)
- expert offload + grouped GEMM + sparse compile

### Task 6.4: src/expert_offload.py + grouped_gemm_demo.py
- 教学版 CPU offload + megablocks grouped GEMM 对照

### Task 6.5: src/tests/test_router_z_loss.py + test_grouped_gemm.py
- z-loss 单调下降 / grouped GEMM 数值正确

### Task 6.6: notebooks/11-training.ipynb + 12-inference.ipynb

### Commit + Tag: `moe-stability`

---

## Phase 7: L13 Capstone

### Task 7.1: lectures/13-capstone-mini-moe.md (28 slides)
- 4-expert mini-MoE 设计
- 把专题 2 GPT-mini MLP 替换成 MoE

### Task 7.2: src/mini_moe.py
- 集成 4 expert × 80M base
- top-2 路由 + Aux-Free 偏置

### Task 7.3: src/capstone_train_mini_moe.py
- 同样 1B token，对比 dense 80M

### Task 7.4: src/tests/test_mini_moe_vs_dense.py
- MoE val ppl < dense × 0.9 (1.5+ ppl 优势)

### Task 7.5: notebooks/13-capstone.ipynb
- 路由热图 + ppl 对照

### Task 7.6: learning/moe-architecture/README.md

### Commit + Tag: `moe-arch`

---

## 验证清单

```bash
python learning/moe-architecture/environment/verify_env.py
python -m pytest learning/moe-architecture/src/tests/ -v
jupyter nbconvert --execute --inplace learning/moe-architecture/notebooks/*.ipynb
python learning/moe-architecture/src/capstone_train_mini_moe.py
```

预期：
- env 三段 PASS
- 7 tests PASS（含 aux-free + z-loss + grouped GEMM + 负载均衡）
- 13 notebook 跑通
- Capstone MoE > dense baseline ≥1.5 ppl

---

## 总览

- 13 lectures × 平均 80 min = 17h slides
- 多个 router + layer 实现
- 7 tests
- 13 notebooks
- 1 capstone
- 预计 8 commit + 7 tag
- 总时长 16h（lecture + notebook + Capstone）
