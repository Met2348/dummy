# L01 · Transformer 完整回顾

> 18 slides | 55 min | Transformer Deep 第 1 讲 ⭐⭐⭐⭐

> Vaswani 2017 起点 + 至 2024 的关键演化

---

## 学习目标

1. 回顾 Vaswani 2017 原始 Transformer 全栈
2. 知道现代 LLM 与原始版的差异（PE / Norm / Attn / Activation）
3. 为后续 13 讲铺背景

---

## Slide 1 · Vaswani 2017 架构

```
[Input Embedding + Positional Encoding]
        ↓
[Encoder × 6]   ⟷   [Decoder × 6]
   (Self-Attn)         (Self-Attn)
   (Add+LN)           (Cross-Attn)
   (FFN)              (Add+LN)
   (Add+LN)           (FFN)
                       (Add+LN)
        ↓
[Output linear + softmax]
```

参数量 ~65M (base) / 213M (big)，BLEU 28+。

---

## Slide 2 · 关键组件

```
1. Multi-Head Attention      Q/K/V 投影 + 多头平行
2. Position Encoding (PE)    sinusoidal absolute
3. LayerNorm + Residual      Post-LN
4. FFN (2-layer MLP)         ReLU 或 GELU
5. Softmax + Cross-Entropy   预测下个词
```

每一项都被 2017-2026 改造过 ≥ 1 次。

---

## Slide 3 · Decoder-only (GPT)

```
Embed + PE
   ↓
[Block × N]
   ├─ Self-Attn (causal mask)
   ├─ Add + Norm
   ├─ FFN
   └─ Add + Norm
   ↓
LM head
```

去掉 encoder + cross-attn，纯左到右生成。GPT-1/2/3 / Llama / Qwen 全是 decoder-only。

---

## Slide 4 · 现代 LLM 7 大改动

| 组件 | 2017 | 现代 | 章节 |
|------|------|------|------|
| PE | sinusoidal absolute | **RoPE** / ALiBi | L02-03 |
| Norm | LayerNorm Post | **RMSNorm Pre** | L05 |
| Attn | MHA | **GQA / MLA** | L04 |
| Activation | ReLU | **SwiGLU** | L06 |
| Attn kernel | naive | **FlashAttention** | L07-08 |
| KV cache | flat | **Paged** | L09 |
| 长上下文 | 512 | 32k-1M | L10 / 专题 5 |

---

## Slide 5 · self-attention 公式

```
Attention(Q, K, V) = softmax( Q K^T / √d ) · V
```

- Q ∈ R^{t × d_k}
- K ∈ R^{t × d_k}
- V ∈ R^{t × d_v}

复杂度 O(t² · d)。t 长 → 二次方爆炸（专题 5 主题）。

---

## Slide 6 · multi-head 切分

```
Q W_Q → 切 h 头 → 每头 d_k = d_model / h
```

例：d_model=768, h=12 → 每头 64-d。

不同头学不同关系（句法 / 语义 / 长距等）。

---

## Slide 7 · FFN (2017 版)

```
FFN(x) = ReLU(x W_1 + b_1) · W_2 + b_2
        with d_ff = 4 · d_model
```

参数占 transformer 总数 ~2/3（attention 1/3）。

现代多用 SwiGLU + d_ff ≈ 2.67 d (Llama-2/3)。

---

## Slide 8 · positional encoding (sinusoidal)

```
PE(pos, 2i)   = sin( pos / 10000^{2i/d} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d} )
```

特性：相对位置 = 旋转角差（"前导 RoPE"）。
缺：绝对，无法外推。

---

## Slide 9 · 损失：Cross-Entropy

```
L = -Σ log p(y_t | y_<t, x)
```

per-token，t 为 sequence 长度。LM 是 next-token prediction。

---

## Slide 10 · 训练：teacher forcing

```
input_ids:    [<bos>, w_1, w_2, ..., w_{n-1}]
target_ids:   [w_1,   w_2, w_3, ..., w_n]
```

shift-by-1，所有 token 平行训。

---

## Slide 11 · 推理：autoregressive

```
for t in range(max_len):
    logits = model(prefix_tokens)
    next_token = sample(logits[:, -1])
    prefix_tokens.append(next_token)
```

每步重算 → KV cache 优化（专题 5）。

---

## Slide 12 · KV cache 概念

```
K_cache, V_cache = [], []
for t in range(max_len):
    q_t = ... 
    K_cache.append(k_t)
    V_cache.append(v_t)
    attn(q_t, K_cache, V_cache)
```

K, V 只 append 不变，省去重算。L09 PagedAttention 是其管理优化。

---

## Slide 13 · 模型尺寸表

| 模型 | 参数 | 层 | hidden | head | 年 |
|------|------|----|---|---------|------|
| GPT-2 small | 124M | 12 | 768 | 12 | 2019 |
| GPT-3 | 175B | 96 | 12288 | 96 | 2020 |
| Llama-2 7B | 7B | 32 | 4096 | 32 | 2023 |
| Llama-3 70B | 70B | 80 | 8192 | 64 | 2024 |
| DeepSeek-V3 | 671B (37B act) | 61 | 7168 | 128 | 2024 |

`hidden ≈ √(N) × 100` 的经验。

---

## Slide 14 · 算力消耗

```
FLOPs ≈ 6 × N × D
        N: 参数数
        D: 训练 token 数
```

Llama-3 70B × 15T → 6 × 70e9 × 15e12 ≈ 6.3e24 FLOPs。

5090 单卡 ~ 400 TFLOPS BF16 → 5e17 FLOPs/day → 13.7 万天。多机才能跑。

---

## Slide 15 · 训练 vs 推理

| | 训练 | 推理 |
|---|------|------|
| 算力主体 | matmul 全网络 | attn + matmul |
| 瓶颈 | 参数同步 (DP) | 显存带宽 (KV) |
| 优化 | flash-attn / ZeRO | KV cache / batching |

不同需求 → 专题 6 与本专题 L07-09 分别覆盖。

---

## Slide 16 · transformer 之外的 challengers

- **Mamba / SSM** (专题 4)
- **RWKV** (专题 4)
- **RetNet**
- **Jamba** (Mamba + attn)
- **Hyena / Mega**

2024 - 2025 全是各家试探。截至 2026 Q1 transformer 仍是主流，hybrid 兴起。

---

## Slide 17 · 本专题 14 lecture 路线

```
L01 · 完整回顾 (本讲)
L02 · PE 全套 (sinusoidal / ALiBi / NoPE)
L03 · RoPE 深推导
L04 · Attention 变体 (MHA → MLA)
L05 · Normalization
L06 · Activation (SwiGLU)
L07 · FlashAttention v1
L08 · FlashAttention v2/v3
L09 · PagedAttention
L10 · Sliding Window
L11 · μP + architecture search
L12 · DeepSeek-V3 全栈精读
L13 · Llama-3 精读
L14 · Capstone — 80M GPT-mini
```

---

## Slide 18 · 课后思考

1. 为什么 decoder-only 比 encoder-decoder 更流行？
2. d_ff = 4 · d_model 的依据？
3. Post-LN vs Pre-LN 哪个稳定？为什么 Llama 用 Pre-LN？
4. KV cache 占多少显存？写出公式。

---

## 参考

- Vaswani 2017 (Attention Is All You Need)
- Radford 2019 (GPT-2)
- Llama-3 / DeepSeek-V3 tech reports 2024
