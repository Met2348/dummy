# Long Context Implementation Plan

**Goal**: 完整实现长上下文学习包（13 lecture + 多个 RoPE scaling + Ring attention 实现 + 5 测试 + 13 notebook + Capstone Llama-3.2-1B 8k→32k）

**Architecture**: 13 lectures = 12 主线 + 1 capstone。三轨代码：手写 minimal + 库（flash-attn / ring-flash-attention） + 工业。WSL2 环境。

**Tech Stack**: torch + flash-attn + ring-flash-attention + transformers + peft

**Design 文档**: `docs/superpowers/specs/2026-06-04-long-context-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/long-context/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + flash-attn + ring-flash-attention + transformers + peft

### Task 1.3: environment/verify_env.py
- Part A: flash-attn + ring-flash-attention import
- Part B: GPU + sm_120
- Part C: YaRN scaling smoke + ring naive smoke

### Task 1.4: src/common.py
- RoPE freq helpers / mask helpers

### Task 1.5: papers/ 11 个占位 + README index

### Commit: `chore: long-context scaffold`

---

## Phase 2: L01-L05 RoPE 全套

### Task 2.1: lectures/01-long-context-overview.md (20 slides)
- 2024-2026 长上下文全景

### Task 2.2: lectures/02-rope-scaling-basic.md (18 slides)
- Meta 2023 Position Interpolation

### Task 2.3: src/rope_pi.py
- PI 实现

### Task 2.4: lectures/03-ntk-aware.md (18 slides)
- LocalLlama 社区 NTK-aware

### Task 2.5: src/rope_ntk.py
- NTK-aware scaling

### Task 2.6: lectures/04-yarn.md (32 slides)
- YaRN 完整推导 + NTK by parts + attn temperature

### Task 2.7: src/rope_yarn.py
- YaRN 严格按论文实现

### Task 2.8: lectures/05-rope-3d.md (16 slides)
- 3D RoPE / Q-RoPE 多模态视频

### Task 2.9: src/rope_3d.py
- 3D RoPE 实现

### Task 2.10: src/tests/test_rope_extrapolation.py
- PI / NTK / YaRN 各自在 8k→16k 的 ppl 趋势

### Task 2.11: notebooks/01-overview.ipynb + 02-pi.ipynb + 03-ntk.ipynb + 04-yarn.ipynb + 05-3d-rope.ipynb

### Commit + Tag: `lc-rope`

---

## Phase 3: L06-L07 Ring + Striped Attention

### Task 3.1: lectures/06-ring-attention.md (32 slides)
- Ring Attention 完整算法 + sequence parallel 1M context

### Task 3.2: src/ring_attention_naive.py
- 单卡 naive ring attention 教学版（模拟多卡通信）

### Task 3.3: src/ring_attention_lib.py
- ring-flash-attention 库调用 (多卡可用)

### Task 3.4: src/tests/test_ring_correctness.py
- naive ring vs naive attention < 1e-4

### Task 3.5: lectures/07-striped-attention.md (16 slides)
- Striped Attention 改进

### Task 3.6: notebooks/06-ring.ipynb + 07-striped.ipynb

### Commit + Tag: `lc-ring`

---

## Phase 4: L08-L09 Infini + FA long-context

### Task 4.1: lectures/08-infini-attention.md (22 slides)
- Google 2024 Infini-Attention compressive memory

### Task 4.2: src/infini_attention.py
- 简化版 compressive memory

### Task 4.3: lectures/09-flash-attention-causal.md (16 slides)
- FA 长上下文优化 block-sparse / window

### Task 4.4: notebooks/08-infini.ipynb + 09-fa-causal.ipynb

### Commit + Tag: `lc-infini`

---

## Phase 5: L10-L12 评测 + 数据 + 陷阱

### Task 5.1: lectures/10-needle-haystack.md (18 slides)
- NIAH 设计 + RULER benchmark

### Task 5.2: src/niah_eval.py
- 完整 NIAH 评测脚本

### Task 5.3: src/ruler_eval.py
- RULER 子集（200 题）

### Task 5.4: lectures/11-long-context-data.md (18 slides)
- book/repo packing 策略

### Task 5.5: src/long_data_packing.py
- 长 sample 拼接 + 边界 mask

### Task 5.6: lectures/12-long-context-pitfalls.md (20 slides)
- Lost in middle / 注意力稀释

### Task 5.7: src/tests/test_niah_pass_rate.py + test_long_packing.py
- NIAH 玩具题 PASS / packing 边界 mask 正确

### Task 5.8: notebooks/10-eval.ipynb + 11-data.ipynb + 12-pitfalls.ipynb

### Commit + Tag: `lc-eval`

---

## Phase 6: L13 Capstone

### Task 6.1: lectures/13-capstone-yarn-extension.md (28 slides)
- Llama-3.2-1B 8k → 32k 完整流程

### Task 6.2: src/capstone_yarn_llama32.py
- 加载 Llama-3.2-1B + YaRN scaling
- LoRA 微调 32k 数据（book / pile-narrative / repo）

### Task 6.3: src/tests/test_capstone_niah_32k.py
- NIAH 32k pass rate > 80%

### Task 6.4: notebooks/13-capstone.ipynb
- 32k NIAH 通过率可视化

### Task 6.5: learning/long-context/README.md

### Commit + Tag: `long-context`

---

## 验证清单

```bash
python learning/long-context/environment/verify_env.py
python -m pytest learning/long-context/src/tests/ -v
jupyter nbconvert --execute --inplace learning/long-context/notebooks/*.ipynb
python learning/long-context/src/capstone_yarn_llama32.py
```

预期：
- env 三段 PASS
- 5 tests PASS
- 13 notebook 跑通
- Capstone NIAH 32k pass rate > 80%

---

## 总览

- 13 lectures × 平均 75 min = 16h slides
- 多个 RoPE scaling + ring attention + Infini 实现
- 5 tests
- 13 notebooks
- 1 capstone (Llama-3.2-1B YaRN 32k)
- 预计 6 commit + 5 tag
- 总时长 14h（lecture + notebook + Capstone）
