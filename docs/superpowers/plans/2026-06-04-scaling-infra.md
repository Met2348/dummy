# Scaling Infra Implementation Plan

**Goal**: 完整实现训练 infra 学习包（14 lecture + 多个并行实现 + 6 测试 + 14 notebook + Capstone FSDP 训 350M）

**Architecture**: 14 lectures = 13 主线 + 1 capstone。三轨代码：手写 minimal 玩具 + 库（torch FSDP / DeepSpeed）+ 工业（Megatron-Core）。WSL2 环境 + 推荐 ≥ 2 卡（云租用建议）。

**Tech Stack**: torch + deepspeed + megatron-core + flash-attn + NCCL

**Design 文档**: `docs/superpowers/specs/2026-06-04-scaling-infra-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/scaling-infra/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + deepspeed + megatron-core + flash-attn + nvidia-nccl

### Task 1.3: environment/verify_env.py
- Part A: torch + deepspeed + megatron-core import
- Part B: GPU + NCCL 多 rank 模拟
- Part C: FSDP 2-rank smoke（torchrun --nproc-per-node=2 用 CPU shard）

### Task 1.4: src/common.py
- gpu memory profiler / NCCL collective wrappers

### Task 1.5: papers/ 12 个占位 + README index

### Commit: `chore: scaling-infra scaffold`

---

## Phase 2: L01-L03 Scaling Laws + μP

### Task 2.1: lectures/01-scaling-laws.md (22 slides)
- Kaplan + Chinchilla 推导

### Task 2.2: lectures/02-chinchilla.md (18 slides)
- 数据 / 参数 平衡教训

### Task 2.3: src/scaling_law_fit.py
- 已知点拟合 Chinchilla scaling law

### Task 2.4: lectures/03-mup.md (28 slides)
- μP 超参 transfer 完整推导

### Task 2.5: src/mup_init.py
- μP init + lr transfer 实验
- 64 → 256 hidden 验证

### Task 2.6: src/tests/test_mup_lr_transfer.py
- 64 vs 256 hidden 最优 lr 一致 (±20%)

### Task 2.7: notebooks/01-scaling.ipynb + 02-chinchilla.ipynb + 03-mup.ipynb

### Commit + Tag: `infra-scaling-laws`

---

## Phase 3: L04-L05 精度 + grad accum/ckpt

### Task 3.1: lectures/04-mixed-precision.md (22 slides)
- FP16 / BF16 / FP8 + loss scale

### Task 3.2: src/mixed_precision_demo.py
- FP16 vs BF16 train loss 稳定性对照

### Task 3.3: lectures/05-gradient-accum.md (18 slides)
- 梯度累积 + checkpointing 显存换算力

### Task 3.4: src/grad_accum_demo.py + grad_ckpt_demo.py
- 累积等价大 batch + 重算换显存

### Task 3.5: src/tests/test_grad_accum_equiv.py + test_mixed_precision_loss.py
- 累积等价大 batch / FP vs BF 损失差 < 0.1 ppl

### Task 3.6: notebooks/04-precision.ipynb + 05-accum.ipynb

### Commit + Tag: `infra-precision`

---

## Phase 4: L06 ZeRO 三阶段

### Task 4.1: lectures/06-zero-1-2-3.md (32 slides)
- ZeRO-1: optimizer 分片
- ZeRO-2: + gradient 分片
- ZeRO-3: + parameter 分片
- 通信复杂度推导

### Task 4.2: src/zero_naive.py
- 手写 ZeRO-1 玩具（optimizer state 分片）

### Task 4.3: src/zero_deepspeed.py
- DeepSpeed ZeRO-1/2/3 config 对照

### Task 4.4: src/tests/test_zero_naive_vs_deepspeed.py
- ZeRO-1 手写 vs deepspeed < 1e-4

### Task 4.5: notebooks/06-zero.ipynb
- 显存对照（FP / ZeRO-1 / 2 / 3）

### Commit + Tag: `infra-zero`

---

## Phase 5: L07 FSDP

### Task 5.1: lectures/07-fsdp.md (32 slides)
- PyTorch FSDP 完整流程
- wrap policy + mixed precision + sharding

### Task 5.2: src/fsdp_train.py
- FSDP wrap + 训练完整 example
- transformer wrap policy + cpu offload 可选

### Task 5.3: src/tests/test_fsdp_vs_ddp.py
- FSDP vs DDP 单步 loss < 1e-4

### Task 5.4: notebooks/07-fsdp.ipynb

### Commit + Tag: `infra-fsdp`

---

## Phase 6: L08-L10 TP + PP + 3D

### Task 6.1: lectures/08-tensor-parallel.md (24 slides)
- Megatron TP column-row split

### Task 6.2: src/tp_megatron.py
- column-row 拆分玩具实现

### Task 6.3: lectures/09-pipeline-parallel.md (22 slides)
- GPipe + 1F1B + interleaved

### Task 6.4: src/pp_gpipe.py
- pipeline 微批调度玩具

### Task 6.5: lectures/10-3d-parallel.md (24 slides)
- DP × TP × PP 组合

### Task 6.6: src/tests/test_tp_correctness.py
- column-row split-merge 一致

### Task 6.7: notebooks/08-tp.ipynb + 09-pp.ipynb + 10-3d.ipynb

### Commit + Tag: `infra-parallel`

---

## Phase 7: L11-L14 框架 + Capstone

### Task 7.1: lectures/11-deepspeed.md (22 slides)
- DeepSpeed 完整栈 + ZeRO-Infinity offload

### Task 7.2: src/deepspeed_zero3.py
- ZeRO-3 完整 config

### Task 7.3: lectures/12-megatron-core.md (18 slides)
- Megatron-Core 工业级框架

### Task 7.4: src/megatron_core_minimal.py
- Megatron-Core 启动 + 简单训练 example

### Task 7.5: lectures/13-comm-primitives.md (16 slides)
- NCCL all-reduce / all-gather / 性能分析

### Task 7.6: src/comm_bench.py
- 通信原语 benchmark

### Task 7.7: lectures/14-capstone-fsdp-train.md (28 slides)
- FSDP 训 350M 模型 完整 recipe

### Task 7.8: src/capstone_fsdp_350m.py
- 350M GPT (24 层 / hidden 1024) + FSDP wrap
- 100M-token 子集训练

### Task 7.9: src/tests/test_flops_calculator.py
- 显存 / FLOPs 预测 vs 实测 < 10%

### Task 7.10: notebooks/11-deepspeed.ipynb + 12-megatron.ipynb + 13-comm.ipynb + 14-capstone.ipynb

### Task 7.11: learning/scaling-infra/README.md

### Commit + Tag: `infra-frameworks` + `scaling-infra`

---

## 验证清单

```bash
python learning/scaling-infra/environment/verify_env.py
python -m pytest learning/scaling-infra/src/tests/ -v
jupyter nbconvert --execute --inplace learning/scaling-infra/notebooks/*.ipynb
torchrun --nproc-per-node=2 learning/scaling-infra/src/capstone_fsdp_350m.py
```

预期：
- env 三段 PASS
- 6 tests PASS
- 14 notebook 跑通
- Capstone FSDP 350M 100M-token 训练完成

---

## 总览

- 14 lectures × 平均 85 min = 20h slides
- 多个并行实现（ZeRO/FSDP/TP/PP/3D/Megatron-Core）
- 6 tests
- 14 notebooks
- 1 capstone (FSDP 350M)
- 预计 8 commit + 7 tag
- 总时长 18h（lecture + notebook + Capstone，含多卡云租用 ~$10）
