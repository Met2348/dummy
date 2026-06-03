# Transformer Deep Implementation Plan

**Goal**: 完整实现 Transformer 架构骨架学习包（14 lecture + 多个组件 minimal/lib 三轨 + 7 测试 + 14 notebook + Capstone 80M GPT-mini）

**Architecture**: 14 lectures = 13 主线 + 1 capstone。三轨代码：手写 minimal + 库（flash-attn / triton） + 工业（DeepSeek-V3 / Llama-3 加载）。Windows native，复用 Module 1+2 cu130 环境。

**Tech Stack**: torch + flash-attn + triton + einops + transformers

**Design 文档**: `docs/superpowers/specs/2026-06-04-transformer-deep-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/transformer-deep/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + flash-attn + triton + einops + transformers + tokenizers

### Task 1.3: environment/verify_env.py
- Part A: torch + flash-attn + triton import
- Part B: GPU + sm_120
- Part C: GPT-mini single forward smoke

### Task 1.4: src/common.py
- 复用 Module 1+2 helpers + numerical safe softmax + causal mask

### Task 1.5: papers/ 16 个占位 + README index

### Commit: `chore: transformer-deep scaffold`

---

## Phase 2: L01-L03 PE 全套 + RoPE 推导

### Task 2.1: lectures/01-transformer-recap.md (18 slides)
- Vaswani 2017 完整复习

### Task 2.2: lectures/02-positional-encoding.md (24 slides)
- Sinusoidal / Learned / RoPE / ALiBi / NoPE 横向对比

### Task 2.3: src/pe_sinusoidal.py + pe_alibi.py
- 标准实现 + 可视化

### Task 2.4: lectures/03-rope-deep.md (28 slides)
- RoPE 复数旋转推导 + 相对位置性质 + 推外能力

### Task 2.5: src/rope.py
- 标准 RoPE + position interpolation 接口预留
- 与 flash_attn.rope 一致性测试

### Task 2.6: src/tests/test_rope_consistency.py
- naive vs flash-attn 库 < 1e-4

### Task 2.7: notebooks/01-recap.ipynb + 02-pe.ipynb + 03-rope.ipynb
- RoPE 旋转角可视化（不同 dim 的频率）

### Commit + Tag: `trans-pe`

---

## Phase 3: L04 Attention 变体 (MHA / MQA / GQA / MLA)

### Task 3.1: lectures/04-attention-variants.md (26 slides)
- MHA → MQA → GQA → MLA 演化
- 显存 / 速度 / 性能 trade-off

### Task 3.2: src/mha.py
- 标准 multi-head attention

### Task 3.3: src/mqa.py + gqa.py
- MQA: 1 KV head；GQA: group KV head

### Task 3.4: src/mla.py
- DeepSeek-V2 MLA: low-rank KV 压缩
- 详细注释推导

### Task 3.5: src/tests/test_attention_variants.py
- 4 种 attention 同 input 输出 shape / numerical 验证

### Task 3.6: notebooks/04-attention-variants.ipynb
- 显存对照 + KV cache 大小可视化

### Commit + Tag: `trans-attn-variants`

---

## Phase 4: L05-L06 Norm + Activation

### Task 4.1: lectures/05-normalization.md (18 slides)
- LayerNorm / RMSNorm / Pre vs Post / DeepNorm

### Task 4.2: src/rmsnorm.py + layernorm_compare.py
- 手写 RMSNorm + torch.nn.RMSNorm 对照

### Task 4.3: lectures/06-activation.md (16 slides)
- ReLU / GELU / SwiGLU / GeGLU

### Task 4.4: src/swiglu.py
- SwiGLU / GeGLU + 与 GELU MLP 对照

### Task 4.5: src/tests/test_rmsnorm_swiglu.py
- 数值一致性 + 梯度流验证

### Task 4.6: notebooks/05-norm.ipynb + 06-activation.ipynb

### Commit + Tag: `trans-norm-act`

---

## Phase 5: L07-L08 FlashAttention

### Task 5.1: lectures/07-flash-attention-v1.md (32 slides)
- tiling + online softmax + 内存层级
- 完整算法步骤

### Task 5.2: src/flash_attn_naive.py
- Triton 手写 naive FA1
- block_size + online softmax 实现

### Task 5.3: src/flash_attn_lib.py
- flash-attn 库调用 + benchmark vs naive vs vanilla

### Task 5.4: src/tests/test_flash_attn_correctness.py
- naive Triton vs flash-attn 库 < 1e-4

### Task 5.5: lectures/08-flash-attention-v2-v3.md (24 slides)
- FA2: warp specialization
- FA3: TMA + FP8（H100）

### Task 5.6: src/fa2_fa3_bench.py
- 仅 benchmark 数字 + 论文对照（无实际 FA3 跑）

### Task 5.7: notebooks/07-fa-v1.ipynb + 08-fa-v2-v3.ipynb

### Commit + Tag: `trans-flash`

---

## Phase 6: L09-L11 Paged + SWA + μP

### Task 6.1: lectures/09-paged-attention.md (20 slides)
- vLLM KV cache 分页管理

### Task 6.2: src/paged_attn_demo.py
- 教学版 KV cache 分页演示

### Task 6.3: lectures/10-sliding-window.md (18 slides)
- Mistral / Gemma SWA

### Task 6.4: src/sliding_window.py
- 标准 SWA + flash-attn 接口

### Task 6.5: lectures/11-architecture-search.md (20 slides)
- μP + hyperparam transfer

### Task 6.6: notebooks/09-paged.ipynb + 10-swa.ipynb + 11-arch-search.ipynb

### Commit + Tag: `trans-paged` + `trans-modern`

---

## Phase 7: L12-L14 DeepSeek-V3 + Llama-3 + Capstone

### Task 7.1: lectures/12-deepseek-v3-walkthrough.md (28 slides)
- MLA + DeepSeekMoE + Aux-Free 完整精读

### Task 7.2: lectures/13-llama3-walkthrough.md (22 slides)
- GQA 8/64 + 128k 长上下文

### Task 7.3: src/deepseek_v3_summary.py + llama3_summary.py
- 加载 model + 架构组件统计

### Task 7.4: lectures/14-capstone-build-gpt-mini.md (32 slides)
- 80M GPT-mini 设计 + 训练 recipe

### Task 7.5: src/gpt_mini.py
- 集成 RoPE + RMSNorm + GQA + SwiGLU 的 80M model
- 12 层 / hidden 768 / 12 heads (kv=2)

### Task 7.6: src/capstone_train.py
- 用专题 1 输出的 1B-token + 32k SP tokenizer
- 1 epoch ≤ 6h on 5090

### Task 7.7: src/tests/test_gpt_mini_forward.py + test_kv_cache.py
- forward 输出 shape / 反向梯度 / KV cache 增量解码一致性

### Task 7.8: notebooks/12-deepseek-v3.ipynb + 13-llama3.ipynb + 14-capstone.ipynb

### Task 7.9: learning/transformer-deep/README.md

### Commit + Tag: `transformer-deep`

---

## 验证清单

```powershell
python learning/transformer-deep/environment/verify_env.py
python -m pytest learning/transformer-deep/src/tests/ -v
jupyter nbconvert --execute --inplace learning/transformer-deep/notebooks/*.ipynb
python learning/transformer-deep/src/capstone_train.py --steps 100  # smoke
```

预期：
- env 三段 PASS
- 7 tests PASS
- 14 notebook 跑通
- Capstone GPT-mini 1 epoch val ppl < 30

---

## 总览

- 14 lectures × 平均 80 min = 19h slides
- 多个 src 组件实现（RoPE/MHA/MQA/GQA/MLA/RMSNorm/SwiGLU/FA naive/paged/swa/gpt-mini）
- 7 tests
- 14 notebooks
- 1 capstone (80M GPT-mini)
- 预计 7 commit + 7 tag
- 总时长 16h（lecture + notebook + Capstone）
