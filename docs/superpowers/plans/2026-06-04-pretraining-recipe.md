# Pretraining Recipe Implementation Plan

**Goal**: 完整实现预训练 recipe 学习包（16 lecture + 多个 recipe 实现 + 8 测试 + 16 notebook + Capstone 270M Phi-tiny 从 0 训）

**Architecture**: 16 lectures = 14 主线 + 2 capstone。三轨代码：手写 minimal + 库（mosaic streaming / lm-eval-harness） + 工业（megatron / mosaic）。WSL2 + 推荐 1 卡（可选云 4×A100）。

**Tech Stack**: torch + deepspeed + megatron-core + mosaic streaming + lm-eval-harness + lion-pytorch + sophia + mup + wandb

**Design 文档**: `docs/superpowers/specs/2026-06-04-pretraining-recipe-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/pretraining-recipe/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + deepspeed + megatron-core + streaming + lm-eval-harness
- lion-pytorch + sophia + mup + wandb

### Task 1.3: environment/verify_env.py
- Part A: torch + deepspeed + streaming + lm-eval-harness import
- Part B: GPU + sm_120
- Part C: 5 step 训练 smoke

### Task 1.4: src/common.py
- training loop helpers / metric loggers

### Task 1.5: papers/ 15 个占位 + README index

### Commit: `chore: pretraining-recipe scaffold`

---

## Phase 2: L01-L02 lr schedule

### Task 2.1: lectures/01-pretraining-overview.md (18 slides)
- 现代预训练全流程

### Task 2.2: lectures/02-warmup-cosine.md (24 slides)
- linear / cosine / WSD 三段 schedule

### Task 2.3: src/scheduler_wsd.py
- WSD scheduler 实现
- warmup-stable-decay 数学验证

### Task 2.4: src/tests/test_wsd_correctness.py
- 三段 lr 数值正确

### Task 2.5: notebooks/01-overview.ipynb + 02-wsd.ipynb

### Commit + Tag: `pretrain-lr`

---

## Phase 3: L03 optimizer

### Task 3.1: lectures/03-optimizer-choice.md (24 slides)
- AdamW / Lion / Sophia 对比 + 何时用哪个

### Task 3.2: src/optimizer_lion.py
- Lion 实现 + lion-pytorch 对照

### Task 3.3: src/optimizer_sophia.py
- Sophia 实现（Hessian-aware）

### Task 3.4: src/tests/test_lion_vs_adamw.py
- 玩具任务 Lion 收敛验证

### Task 3.5: notebooks/03-optimizer.ipynb

### Commit + Tag: `pretrain-optim`

---

## Phase 4: L04-L06 稳定性 + init + reg

### Task 4.1: lectures/04-grad-clip-loss-spike.md (24 slides)
- 梯度裁剪 + loss spike 处理 rollback / skip / restart

### Task 4.2: src/loss_spike_handler.py
- 自动 rollback 实现
- spike 检测阈值

### Task 4.3: src/tests/test_loss_spike_rollback.py
- 注入 spike,自动 rollback 触发

### Task 4.4: lectures/05-init-strategy.md (18 slides)
- Xavier / He / μP init

### Task 4.5: src/init_strategies.py
- 各 init + μP init 实现

### Task 4.6: lectures/06-dropout-strategy.md (14 slides)
- dropout / weight decay 早期 / 后期策略

### Task 4.7: notebooks/04-spike.ipynb + 05-init.ipynb + 06-dropout.ipynb

### Commit + Tag: `pretrain-stability`

---

## Phase 5: L07 数据加载

### Task 5.1: lectures/07-data-loading.md (22 slides)
- StreamingDataset / WebDataset / Mosaic 流式

### Task 5.2: src/data_streaming.py
- mosaic streaming + 多文件流式迭代

### Task 5.3: src/tests/test_streaming_data.py
- 多文件迭代 + 边界正确

### Task 5.4: notebooks/07-data-loading.ipynb

### Commit + Tag: `pretrain-data`

---

## Phase 6: L08-L10 监控 + eval + ckpt

### Task 6.1: lectures/08-loss-monitor.md (18 slides)
- wandb / tensorboard / 早期警报

### Task 6.2: src/monitor.py
- wandb / tensorboard 双轨 + 自动警报

### Task 6.3: lectures/09-eval-during-train.md (16 slides)
- val loss / downstream tasks 训中评测

### Task 6.4: src/eval_harness.py
- lm-eval-harness 子集 (MMLU 200 / GSM8K 100)

### Task 6.5: lectures/10-checkpoint-strategy.md (22 slides)
- resume / EMA / averaging

### Task 6.6: src/checkpoint_avg.py + ema.py
- model averaging + EMA 实现

### Task 6.7: src/tests/test_resume_continuity.py + test_ema_correctness.py
- resume 后 loss 连续 / EMA 数学正确

### Task 6.8: notebooks/08-monitor.ipynb + 09-eval.ipynb + 10-ckpt.ipynb

### Commit + Tag: `pretrain-monitor`

---

## Phase 7: L11-L14 recipes + annealing + mid

### Task 7.1: lectures/11-phi-recipe.md (28 slides)
- Phi 1/2/3/3.5/4 完整 recipe

### Task 7.2: lectures/12-llama3-recipe.md (24 slides)
- Llama-3 15T + 长上下文 + annealing

### Task 7.3: src/phi_recipe.py + llama3_recipe.py
- 配置文件参考

### Task 7.4: lectures/13-data-annealing.md (18 slides)
- 末段切高质量数据策略

### Task 7.5: src/data_annealing.py
- 末段切换高质量数据 + lr decay 配合

### Task 7.6: lectures/14-mid-training.md (16 slides)
- continued pretraining / 引入新能力

### Task 7.7: notebooks/11-phi.ipynb + 12-llama3.ipynb + 13-annealing.ipynb + 14-mid.ipynb

### Commit + Tag: `pretrain-recipes`

---

## Phase 8: L15-L16 Capstone（系列高峰）

### Task 8.1: lectures/15-capstone-phi-tiny-train.md (40 slides)
- 270M Phi-tiny 完整训练 recipe
- 数据 + 架构 + infra 全集成

### Task 8.2: src/model_phi_tiny.py
- 270M Phi-style (24 层 / hidden 1024 / GQA 16/4 / SwiGLU / RMSNorm / RoPE)

### Task 8.3: src/pretrain_main.py
- 主训练 loop (复用前面所有组件)
- WSD scheduler + Lion + 自动 rollback + EMA

### Task 8.4: src/capstone_train_phi_tiny.py
- 用专题 1 1B-token + Phi 合成数据 5B token
- 完整 20h 训练（或云 4 卡 4h）

### Task 8.5: lectures/16-capstone-evaluation.md (18 slides)
- MMLU / GSM8K / HumanEval 子集评测

### Task 8.6: src/capstone_eval.py
- 评测 270M Phi-tiny on lm-eval-harness 子集

### Task 8.7: src/tests/test_phi_tiny_forward.py + test_capstone_smoke.py
- forward 输出 shape 正确 / 100 step smoke 训练

### Task 8.8: notebooks/15-capstone-train.ipynb + 16-capstone-eval.ipynb

### Task 8.9: learning/pretraining-recipe/README.md

### Commit + Tag: `pretraining-recipe`

---

## 验证清单

```bash
python learning/pretraining-recipe/environment/verify_env.py
python -m pytest learning/pretraining-recipe/src/tests/ -v
jupyter nbconvert --execute --inplace learning/pretraining-recipe/notebooks/*.ipynb
python learning/pretraining-recipe/src/capstone_train_phi_tiny.py  # 真训
python learning/pretraining-recipe/src/capstone_eval.py
```

预期：
- env 三段 PASS
- 8 tests PASS
- 16 notebook 跑通
- Capstone val ppl < 25
- Capstone GSM8K 0-shot ≥ 5%
- Capstone MMLU 5-shot ≥ 30%

---

## 总览

- 16 lectures × 平均 75 min = 20h slides
- 多个 recipe 组件 + 完整训练 pipeline
- 8 tests
- 16 notebooks
- 2 capstone (训 + 评测)
- 预计 8 commit + 7 tag
- 总时长 20h（lecture + notebook + Capstone）
- 注：真训 Capstone 需 20h × 5090 或租云 4×A100 ~$40
