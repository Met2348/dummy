# L04 · Attention 变体 — MHA / MQA / GQA / MLA

> 26 slides | 75 min | Transformer Deep 第 4 讲 ⭐⭐⭐⭐⭐ 必修

---

## 学习目标

1. 理解 MHA → MQA → GQA → MLA 的演化动机
2. KV cache 显存如何随之缩减
3. 写出 4 种 attention 的 100 行实现
4. 知道何时选哪种

---

## Slide 1 · 演化总览

```
2017 MHA      h Q head, h K, h V head
2019 MQA      h Q head, 1 K, 1 V head      → KV cache /h
2023 GQA      h Q head, g K, g V (h/g)     → KV cache × g/h
2024 MLA      low-rank K,V compression     → KV cache 极致缩
```

减少 KV head = 减少推理显存 = 增大 batch size。

---

## Slide 2 · MHA (Vaswani 2017)

```
d_head = d_model / h
Q, K, V 三投影：(d_model, h, d_head)
```

每头独立 attention，输出 concat 后再投影。最经典也最贵。

KV cache: `2 · n_layer · h · t · d_head · dtype_size`。

---

## Slide 3 · MQA (Shazeer 2019)

```
h Q head 不变
K, V 各只 1 个 head (共享给所有 Q head)
```

KV cache 减为 `2 · n_layer · 1 · t · d_head · dtype_size` = **/h**。

经验：模型性能 -1-2%，推理快 2-3×。早期 PaLM 用。

---

## Slide 4 · GQA (Ainslie 2023)

```
h Q head; g KV head; 每 h/g Q 共享 1 K, 1 V
```

- g=h → MHA
- g=1 → MQA
- g=8, h=64 → Llama-3 70B

平衡：质量损失小，KV cache 减 8×。事实标准。

---

## Slide 5 · MLA (DeepSeek-V2 2024)

```
K, V 不存全维 d_head，存低秩 compressed c_kv ∈ R^{d_low}
推理时 K = c_kv @ W_K_up, V = c_kv @ W_V_up
```

KV cache: `n_layer · t · d_low` (no h, no 2)。

DeepSeek-V3 128 head 用 d_low=512 → KV cache 极致省。

---

## Slide 6 · KV cache 实测

Llama-3 70B 推 32k context, batch=1:

```
MHA   h=64  : 2·80·64·32k·128·2 = 84 GB
GQA   g=8   :                      10.5 GB   ← 8×
MLA   d=512:                       0.84 GB   ← 100×
```

5090 24GB 上：MHA OOM，GQA OK，MLA 极宽松。

---

## Slide 7 · 选型决策

| 场景 | 推荐 |
|------|------|
| < 7B 小模型 | MHA / GQA(g=h/4) |
| 7-70B 中大 | GQA (g=8) |
| > 100B | MLA |
| 仅 batch=1 推理 | GQA 即可 |
| 高并发服务 | MLA |

---

## Slide 8 · MHA 公式回顾

```
Q = X W_Q ∈ R^{t × h·d_head}
K = X W_K
V = X W_V
切多头: Q, K, V → (b, h, t, d_head)
attn:   softmax(Q K^T / √d_head) V
concat: (b, t, h·d_head) → W_O
```

---

## Slide 9 · MQA 实现

```python
W_Q: (d_model → h · d_head)
W_K: (d_model → 1 · d_head)     # 1 head only
W_V: (d_model → 1 · d_head)
# K, V 在 forward 时 broadcast 给 h 个 Q head
```

---

## Slide 10 · GQA 实现

```python
W_Q: (d → h · d_head)
W_K: (d → g · d_head)
W_V: (d → g · d_head)
# K, V 每个 head 复用 给 (h/g) 个 Q head
K_repeated = K.repeat_interleave(h//g, dim=1)  # (b, h, t, d_head)
```

---

## Slide 11 · MLA 关键 — 压缩

```
W_DKV: (d → d_low)          ← 压缩
W_UK:  (d_low → h·d_head)    ← K 解压
W_UV:  (d_low → h·d_head)    ← V 解压

c = X W_DKV       # 存这个
K = c W_UK        # 用时再算
V = c W_UV
```

KV cache 只存 c (size d_low)。

---

## Slide 12 · MLA 另一个 trick — RoPE 分离

```
K = [K_rope ; K_nope]
                |    |
            正常 RoPE  非 RoPE 部分
K_rope 也压缩存？不能！RoPE 与压缩冲突。
解决：K_rope 单独走 MHA 路线，d_rope 小（如 64）。
```

具体细节见 DeepSeek-V3 论文 Algorithm 1。

---

## Slide 13 · attention score 公式（仅 MHA）

```
score_{i,j} = (q_i · k_j) / √d_head
attn_{i,j}  = softmax_j(score) (causal mask)
out_i       = Σ attn_{i,j} v_j
```

---

## Slide 14 · 训练时差异

| | MHA | GQA | MLA |
|---|-----|-----|-----|
| 参数 | 高 | 中 | 低（W 矩阵小）|
| FLOPs | 标准 | 标准 | 解压多算 |
| 收敛 | 标准 | 略差 | 略差 |

→ 训练时 MHA 仍是最稳。MLA 需 tuning。

---

## Slide 15 · 推理时差异

| | MHA | GQA | MLA |
|---|-----|-----|-----|
| KV cache | 大 | 中 | 极小 |
| 矩阵乘 | 标 | 标 | 略多 |
| batch | 小 | 中 | 大 |

→ 推理偏好 MLA → GQA → MHA。

---

## Slide 16 · 实务对照表

| 模型 | head | g | d_head | KV cache 32k |
|------|------|---|--------|------|
| GPT-3 175B | 96 | =h (MHA) | 128 | 8 GB |
| Llama-2 70B | 64 | =h (MHA) | 128 | 5 GB |
| Llama-3 70B | 64 | 8 (GQA) | 128 | 0.6 GB |
| Mistral 7B | 32 | 8 (GQA) | 128 | 0.25 GB |
| DeepSeek-V3 | 128 | MLA d=512 | 192 | < 0.1 GB |

---

## Slide 17 · `repeat_interleave` GQA 技巧

```python
# K: (b, g, t, d_head),  Q: (b, h, t, d_head)
K = K.repeat_interleave(h // g, dim=1)  # (b, h, t, d_head)
V = V.repeat_interleave(h // g, dim=1)
attn = (Q @ K.transpose(-2, -1)) / sqrt(d_head)
```

或直接用 PyTorch 2.5 `scaled_dot_product_attention` 内置 GQA。

---

## Slide 18 · KV cache 增量解码

```python
class KVCache:
    def __init__(self, ...):
        self.K = []  # list of (b, h_kv, t, d)
        self.V = []
    def append(self, k_new, v_new):
        self.K.append(k_new); self.V.append(v_new)
    def get(self):
        return cat(self.K, dim=-2), cat(self.V, dim=-2)
```

PagedAttention（L09）替换这个简单实现以高效管理碎片。

---

## Slide 19 · "为什么不 LV cache"

L (loss) 不需要 cache—每步 forward 时算完即丢。

只 K, V 需要 cache，因为它们参与下一步 attention（与未来 Q 做 dot）。

---

## Slide 20 · attention quadratic 复杂度

```
全 attn:  O(t² · d_head · h)  per step
KV cache 后:  O(t · d_head · h)  per step (Q 只 1 token)
```

KV cache 把 incremental decoding 从 O(t²) 降到 O(t)。

---

## Slide 21 · attention sink 现象

Mistral 早期发现：训长上下文时，前几个 token 总是有高 attention（即使无关）。
→ Attention Sink。

后续 SoftPick / sinks-aware kernel 是对此的工程修正。

---

## Slide 22 · 数学：MQA 等价 MHA 当 h=g？

```
g=1 (MQA): K, V 1 个 head，h 个 Q 共享
g=h (MHA): K, V h 个独立
```

只看 expressiveness, MHA > GQA > MQA。但实务差距常 < 1pp。

---

## Slide 23 · MLA 训练 trick

DeepSeek-V2 实测：
- W_DKV 初始用 SVD 分解原 MHA W_K, W_V
- 收敛快 30%

不这么做也 OK，只是要多训。

---

## Slide 24 · "fused" attention kernel

- PyTorch `scaled_dot_product_attention` 自动 fuse
- FlashAttention v2 显式 fuse + 处理 mask
- vLLM PagedAttention：fuse + cache 管理

下两讲（L07-09）专门展开。

---

## Slide 25 · 实务代码组织

```
src/mha.py    # baseline
src/mqa.py    # MQA
src/gqa.py    # GQA + groups
src/mla.py    # MLA + low-rank
```

每个 < 80 行 forward。

---

## Slide 26 · 课后思考

1. GQA g=h/4 比 g=h/8 的质量差距？
2. MLA 推理慢于 GQA 吗？理论分析。
3. KV cache 8GB 时是否需要 paged 管理？
4. 模型规模与 KV head 数的经验关系？

---

## 参考

- Shazeer 2019 (MQA)
- Ainslie 2023 (GQA)
- DeepSeek-V2 / V3 2024 (MLA)
- Vaswani 2017 (MHA)
