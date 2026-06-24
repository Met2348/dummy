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


---
## 🔬 小而真 · 真实模型例子
> 除 toy 外, 本专题附一个**真实小模型** notebook (本地 gpt2/TinyLlama, CPU 离线):
> - [`notebooks/N15-real-gpt2-attention-kv.ipynb`](notebooks/N15-real-gpt2-attention-kv.ipynb) — 真实 gpt2: 注意力矩阵 + 下一 token 分布 + KV cache 加速 (你 toy 学的机制, 真模型版)
> 共享工具见 [`learning/_shared/realmodels.py`](../_shared/realmodels.py)。
