# Transformer Deep 学习包

> Module 3「造大模型」**专题 2 / 8** — 现代 Transformer 架构骨架

| 元数据 | 值 |
|--------|----|
| 方法数 | 16 |
| Lecture | 14 |
| 估时 | 16h |
| 环境 | Windows native，复用 Module 1+2 cu130 |
| Design | [`docs/superpowers/specs/2026-06-04-transformer-deep-design.md`](../../docs/superpowers/specs/2026-06-04-transformer-deep-design.md) |
| Plan | [`docs/superpowers/plans/2026-06-04-transformer-deep.md`](../../docs/superpowers/plans/2026-06-04-transformer-deep.md) |

---

## 14 Lecture

| # | Lecture | 关键内容 |
|---|---------|---------|
| L01 | Transformer 完整回顾 | Vaswani → 现代演化 |
| L02 | Positional Encoding 全套 | Sinusoidal / Learned / RoPE / ALiBi / NoPE |
| L03 | **RoPE 深推导** | 复数旋转 + 相对位置 + interleaved |
| L04 | **Attention 变体** | MHA / MQA / GQA / **MLA** |
| L05 | Normalization | LayerNorm / **RMSNorm** / Pre vs Post |
| L06 | Activation | ReLU / GELU / **SwiGLU** |
| L07 | **FlashAttention v1** | tiling + online softmax |
| L08 | FlashAttention v2/v3 | warp specialization + FP8 |
| L09 | PagedAttention | vLLM KV cache 分页 |
| L10 | Sliding Window | Mistral SWA |
| L11 | μP + arch search | 超参 transfer |
| L12 | **DeepSeek-V3 精读** | MLA + DeepSeekMoE + Aux-Free + MTP |
| L13 | Llama-3 精读 | GQA 8/64 + RoPE 500000 + 128k |
| L14 | **Capstone — GPT-mini 80M** | 集成训练 |

---

## 目录结构

```
learning/transformer-deep/
├── README.md
├── environment/
│   ├── requirements.txt
│   └── verify_env.py
├── papers/README.md
├── lectures/  (14 × PPT-style md)
├── src/
│   ├── common.py            # causal_mask / init / safe_softmax
│   ├── pe_sinusoidal.py / pe_alibi.py
│   ├── rope.py              # RoPE interleaved
│   ├── mha.py / mqa.py / gqa.py / mla.py
│   ├── rmsnorm.py
│   ├── swiglu.py
│   ├── flash_attn_naive.py  # Triton-style tiling
│   ├── flash_attn_lib.py    # flash-attn 库调用
│   ├── fa2_fa3_bench.py     # 数字对照
│   ├── paged_attn_demo.py   # vLLM KV pool
│   ├── sliding_window.py    # SWA
│   ├── deepseek_v3_summary.py
│   ├── llama3_summary.py
│   ├── gpt_mini.py          # 80M model
│   ├── capstone_train.py    # 训练 main
│   └── tests/
│       ├── test_rope_consistency.py
│       ├── test_attention_variants.py
│       ├── test_rmsnorm_swiglu.py
│       ├── test_flash_attn_correctness.py
│       ├── test_gpt_mini_forward.py
│       └── test_kv_cache.py
└── notebooks/  (14 × ipynb)
```

---

## 横向对比表

### Attention 变体

| 变体 | KV head | KV cache (70B / 32k) | 实际选用 |
|------|---------|---------------------|---------|
| MHA  | 64 | 5 GB | GPT-3 |
| MQA  | 1 | 0.08 GB | PaLM 早期 |
| **GQA** | 8 | **0.6 GB** | **Llama-3 / Mistral** |
| **MLA** | latent d=512 | **0.06 GB** | **DeepSeek-V3** |

### Norm + Activation

| 模型 | Norm | Act |
|------|------|-----|
| GPT-2/3 | LayerNorm Pre | GELU |
| Llama-2/3 | RMSNorm Pre | SwiGLU |
| DeepSeek-V3 | RMSNorm Pre | SwiGLU |
| Phi-3/4 | RMSNorm Pre | SwiGLU |

→ 现代 LLM 全是 RMSNorm + SwiGLU + Pre-LN。

---

## 验证命令

```powershell
python learning/transformer-deep/environment/verify_env.py
python -m pytest learning/transformer-deep/src/tests/ -v
jupyter nbconvert --execute --inplace learning/transformer-deep/notebooks/*.ipynb
python learning/transformer-deep/src/capstone_train.py --steps 50
```

预期：env 三段 PASS / 7 tests PASS / 14 notebook 跑通 / Capstone loss 持续下降。

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（19/19 PASS）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules transformer-deep
> ```

**18 个 from-scratch 直跑 demo**（均无参直跑，CPU 秒级；纯数值/结构 self-test，真有 assert + 打印真实张量/diff/参数量）：

```powershell
# 位置编码
python learning/transformer-deep/src/pe_sinusoidal.py      # Sinusoidal PE
python learning/transformer-deep/src/pe_alibi.py           # ALiBi 线性偏置
python learning/transformer-deep/src/rope.py               # RoPE（相对位置不变性 sanity）
# Attention 变体（MHA → MQA → GQA → MLA）
python learning/transformer-deep/src/mha.py
python learning/transformer-deep/src/mqa.py
python learning/transformer-deep/src/gqa.py
python learning/transformer-deep/src/mla.py                # 低秩 latent KV 压缩
# Norm + Activation
python learning/transformer-deep/src/rmsnorm.py            # vs torch.nn.RMSNorm 对照
python learning/transformer-deep/src/swiglu.py             # SwiGLU/GeGLU/GELU 三变体
# FlashAttention + 现代 attention
python learning/transformer-deep/src/flash_attn_naive.py   # tiling + online softmax，数值等价 vanilla
python learning/transformer-deep/src/flash_attn_lib.py     # PyTorch SDPA benchmark（GPU）
python learning/transformer-deep/src/fa2_fa3_bench.py      # FA2/FA3 论文报数 + 本机 SDPA TFLOPS（GPU）
python learning/transformer-deep/src/paged_attn_demo.py    # block table + KV pool alloc/share
python learning/transformer-deep/src/sliding_window.py     # Mistral SWA mask
# 架构精读（配置 + 真算 KV-cache GB）
python learning/transformer-deep/src/deepseek_v3_summary.py
python learning/transformer-deep/src/llama3_summary.py
# GPT-mini
python learning/transformer-deep/src/gpt_mini.py           # forward + KV-cache generate
```

**Capstone：GPT-mini 训练**（AdamW + cosine lr + grad clip，mock data）：

```powershell
# 真训（loss 下降）
python learning/transformer-deep/src/capstone_train.py --steps 50
# 快速 smoke（验证训练循环可跑通）
python learning/transformer-deep/src/capstone_train.py --steps 8
```

> **注记**：
> - 全部 demo 为纯数值/结构 self-test，秒级 PASS 是正常的（非 no-op：真有计算 + assert + 打印）。
> - `flash_attn_lib` / `fa2_fa3_bench`：flash-attn 库在 Windows 装不上 → 走 **PyTorch SDPA 真实路径**，flash-attn 分支显式打印 `[SKIP]`（非假成功，主路径在 3080 Ti 真跑出 ms / TFLOPS）。
> - 子模块 `official/repos/tensor2tensor` 是官方参考实现，**不在 runbook 范围**（不跑 / 不收录）。

**测试（V2）**：

```powershell
python -m pytest learning/transformer-deep/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules transformer-deep --tests
```

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `trans-pe` | L01-L03 PE + RoPE |
| `trans-attn-variants` | L04 MHA/MQA/GQA/MLA |
| `trans-norm-act` | L05-L06 RMSNorm + SwiGLU |
| `trans-flash` | L07-L08 FlashAttention |
| `trans-modern` | L09-L11 paged + SWA + μP |
| `transformer-deep` | L12-L14 收口 |

---

## 与其他专题

```
data-curation (专题 1) → tokenizer + corpus
                        ↓
本专题 transformer-deep → GPT-mini ckpt 接口
                        ↓
   专题 3 moe-architecture → MoE 替换 dense MLP
   专题 4 ssm-hybrid → SSM 替换 attn
   专题 7 pretraining-recipe → 真训 270M Phi-tiny
   专题 8 graduation → 80M GPT-mini 作 baseline
```
