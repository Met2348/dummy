# L07 · FlashAttention v1 — Tiling + Online Softmax

> 32 slides | 90 min | Transformer Deep 第 7 讲 ⭐⭐⭐⭐⭐ 必修

> Dao et al. 2022 / 改变 transformer 训练成本结构

---

## 学习目标

1. 理解 HBM ↔ SRAM 内存层级与 attention 瓶颈
2. 掌握 tiling 分块算法
3. 推导 online softmax 数值稳定形式
4. 写一版 Python / Triton 玩具

---

## Slide 1 · "为什么标准 attention 慢"

标准 `softmax(QK^T) V`：

```
1. QK^T: t × t score matrix → 写入 HBM
2. softmax: 读 t² → 写 t² → 读 t²
3. @V: 读 t² + t·d → 写 t·d
```

HBM-IO 比 FLOP 慢 100×。attention 实际是 **memory-bound** 而非 compute-bound。

---

## Slide 2 · HBM vs SRAM

```
A100:  80 GB HBM (1.5 TB/s)
        20 MB SRAM (19 TB/s)   ← 12× 更快
```

FlashAttention 思路：**把 attention 拆块**，每块在 SRAM 内算完，仅写最终 output 回 HBM。

---

## Slide 3 · Tiling 切块

```
Q ∈ R^{t × d}  → 切 N_q 块, each (B_r × d)
K, V ∈ R^{t × d} → 切 N_k 块, each (B_c × d)
```

outer loop: Q block
  inner loop: K, V block
  在 SRAM 算 partial attention

---

## Slide 4 · 问题：softmax 跨块怎么算

```
softmax_j(Q_i K_j^T)  需要见 **所有 K_j** 才能正确归一化
```

切块后只见局部，怎么"补正"？

→ Online softmax 解决。

---

## Slide 5 · Online softmax (Milakov 2018)

记 `m = max_j x_j`, `l = Σ exp(x_j - m)`：

```
softmax_j(x_j) = exp(x_j - m) / l
```

当新增 `x_{j+1}` 时：

```
m_new = max(m, x_{j+1})
l_new = exp(m - m_new) · l + exp(x_{j+1} - m_new)
```

→ **可流式更新**。FlashAttention 的核心数值魔法。

---

## Slide 6 · FlashAttention 主算法

```
for i = 1..N_q:                              # Q block
    O_i = 0; m_i = -∞; l_i = 0
    for j = 1..N_k:                          # K, V block
        S_ij = Q_i K_j^T / √d                # in SRAM
        m_new = max(m_i, rowmax(S_ij))
        p = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) l_i + rowsum(p)
        O_i = exp(m_i - m_new) O_i + p V_j
        m_i = m_new; l_i = l_new
    O_i = O_i / l_i                          # 写回 HBM
```

---

## Slide 7 · 内存复杂度

```
标准 attention:  O(t²) HBM (score matrix)
FlashAttention:  O(t)  HBM (只 Q, K, V, O)
```

ts 长 → 内存节省巨大。t=8k: 节省 64MB; t=128k: 16GB。

---

## Slide 8 · 速度

FlashAttention v1 paper:
- BERT-base 2-3× 训快
- GPT-2 长上下文 5-7×

后来 FA2 在此基础再 1.5-2×。

---

## Slide 9 · 反向传播

Forward 没存 attention matrix，反向需要重算。
但只重算 attention，不重算 QK 投影 → 计算 +25%，但显存 -90%。

→ "memory for compute" trade-off。

---

## Slide 10 · 块大小 B_r, B_c

```
B_r · d  + B_c · d  + B_r · B_c  ≤  SRAM size
```

A100 SRAM 20MB → 大致 B_r = B_c = 64-128。

各 GPU 不同最优值，FA2 在 H100 优化更精细。

---

## Slide 11 · Causal mask 处理

```
score_{ij} = -inf if j > i  else Q_i K_j / √d
```

FA 实现：跳过完全是 -inf 的 block (j 全 > i)。
```
for j in range(start, ...):
    if j > i + B_r:
        skip  # all -inf
    elif j > i: 
        部分 mask
```

→ 实测对 causal attention 加速 ~ 2×。

---

## Slide 12 · v1 局限

- 单 GPU 单 stream
- 不能完美用尽 H100 TMA / FP8
- mask 灵活性有限

→ FA2, FA3 各自解决。

---

## Slide 13 · FA1 vs vanilla 性能数字

```
seq_len 512:   FA1 / vanilla ≈ 1.0× (短，无显著)
seq_len 2k:    1.5×
seq_len 8k:    3×
seq_len 32k:   vanilla OOM,  FA1 OK
```

长上下文是 FA 的主战场。

---

## Slide 14 · 实务代码 — torch native

```python
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch 2.0+ 自动选 FA backend（若 GPU 支持）。

---

## Slide 15 · 实务代码 — flash-attn lib

```python
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True, dropout_p=0.0)
```

直接 FA2 kernel。Linux + sm_80+ (Ampere+) 必备。

---

## Slide 16 · 在 5090 (sm_120) 上

5090 是 Blackwell sm_120：
- PyTorch 2.5 SDPA 已支持
- flash-attn 2.6+ 部分支持
- FA3 仅 H100/H200 (sm_90)

---

## Slide 17 · Triton naive 玩具

```python
import triton
import triton.language as tl

@triton.jit
def flash_attn_naive_kernel(Q, K, V, O, ...):
    # outer block over Q
    for j in range(0, K_len, BLOCK_J):
        # 加载 K_j, V_j 到 SRAM
        # 计算 S_ij = Q_i @ K_j^T
        # online softmax 更新
        ...
```

实战在 `src/flash_attn_naive.py` (Triton 必装)。

---

## Slide 18 · 数值精度

FA 内部 FP32 累加（accumulator），输入 BF16。
若全 BF16 会有 numerical drift ~0.5pp ppl。

---

## Slide 19 · attention bias 处理

```
out = FA(Q, K, V, attn_bias=B)
```

ALiBi 可表达为 bias；其他 mask 也通过 bias 加。
FA2 起 bias 支持完善。

---

## Slide 20 · Window attention 集成

Mistral SWA 在 FA2 中通过 `window_size` 参数：

```python
flash_attn_func(q, k, v, window_size=(window, 0))
```

L10 详讲 SWA。

---

## Slide 21 · 验证 FA 数值正确

```python
out_vanilla = (softmax(q @ k.T / sqrt(d)) @ v)
out_fa = flash_attn_func(q, k, v)
assert (out_vanilla - out_fa).abs().max() < 1e-3
```

bfloat16 误差 ~ 1e-3 是正常。

---

## Slide 22 · "为什么 FA 是 paper 改 LLM 训练成本"

GPT-3 训练成本：~$4M
若 FA 提前 1 年发布，估计省 ~$1M。

长 context 训练（128k+）几乎全靠 FA + Ring。

---

## Slide 23 · FA 不解决的问题

- 多机通信
- KV cache 管理（vLLM PagedAttention 补）
- 长文 > 100k 单卡（Ring Attention 补，专题 5）

FA 只解决"单层 attention 在单 GPU 的内存效率"。

---

## Slide 24 · 实现完整伪代码

```python
def flash_attn_naive(Q, K, V, B_r=64, B_c=64):
    """all in SRAM-friendly tiles."""
    t, d = Q.shape
    O = zeros_like(Q); l = zeros(t); m = full(t, -inf)
    for i in range(0, t, B_r):
        Qi = Q[i:i+B_r]
        for j in range(0, t, B_c):
            Kj, Vj = K[j:j+B_c], V[j:j+B_c]
            Sij = Qi @ Kj.T / sqrt(d)
            ... # online softmax update O, l, m
    return O / l[:, None]
```

---

## Slide 25 · 测试一致性

```python
out_naive = vanilla_attn(q, k, v)
out_fa    = flash_attn_naive(q, k, v)
diff = (out_naive - out_fa).abs().max()
assert diff < 1e-4
```

fp32 输入下应严格一致。

---

## Slide 26 · Memory math

```
vanilla:  O(t²)
FA:       O(t)
ratio:    t × 节省
```

t=4k: 4000× 节省（含 attention matrix）。

---

## Slide 27 · "compute" 维度

```
FA: 25% 更多 FLOPs (recompute backward)
但 GPU FLOPs 富余，HBM bandwidth 紧
→ 净加速
```

memory-bound → compute-bound 的转换。

---

## Slide 28 · "硬件"维度

FA 是硬件 / kernel 工程典范。同样算法 5090 (sm_120) 与 A100 (sm_80) 性能差异巨大。

→ 每代 GPU 需要新一版 FA。

---

## Slide 29 · 主要变体

```
FlashAttention v1     2022.05    base
v2                    2023.07    warp specialization
v3                    2024.07    TMA + FP8 (H100)
StreamingLLM          2023.10    attention sinks
Ring Attention        2023.10    多 GPU 序列分片
PagedAttention         2023.10    KV cache 分页 (vLLM)
```

→ FA 已演化成"attention 系统工程"家族。

---

## Slide 30 · 课后思考

1. online softmax 数学上为什么正确？
2. SRAM 20MB 限制 B_r 大小最大多少 (d_head=128)？
3. 为什么 backward 重算更省内存？
4. 5090 与 H100 谁能用更新版 FA？

---

## Slide 31 · 代码组织

```
src/flash_attn_naive.py    # Triton naive 教学版
src/flash_attn_lib.py      # flash-attn 库调用
src/fa2_fa3_bench.py       # 数字对比（无真跑 FA3）
```

---

## Slide 32 · 参考

- Dao 2022 (FlashAttention v1)
- Milakov 2018 (Online softmax)
- FlashAttention GitHub repo
- Triton language docs
