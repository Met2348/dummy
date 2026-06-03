# L03 · RoPE 深推导 — Rotary Position Embedding

> 28 slides | 80 min | Transformer Deep 第 3 讲 ⭐⭐⭐⭐⭐ 必修

> Su 2021 / 现代 LLM 默认 PE / 长上下文基石

---

## 学习目标

1. 完整推导 RoPE 的复数旋转表达
2. 理解为什么 `<RoPE(q), RoPE(k)>` 仅依赖相对位置
3. 实现 100 行内 PyTorch 版
4. 验证与 flash-attn / transformers 一致

---

## Slide 1 · 问题动机

绝对 PE 加在 embedding 上：
```
attn(q_i, k_j) = q_i · k_j
其中 q_i = (x_i + p_i) W_Q,  k_j = (x_j + p_j) W_K
```

attention 中**位置信息**和**内容信息**纠缠 → 网络分不清，外推弱。

→ 希望 attn(q_i, k_j) **仅依赖** 相对距离 `j - i`。

---

## Slide 2 · 目标方程

设计 f, g 使得：

```
<f(q_i, i), g(k_j, j)> = h(q_i, k_j, j-i)
```

只要满足这个，attention 的 score 只看相对位置，与绝对位置无关。

---

## Slide 3 · 复数解

考虑 d=2，把 (x₀, x₁) 看作复数 z = x₀ + i·x₁。

```
f(z, m) = z · e^{i·m·θ}    （旋转角 m·θ）
g(z, n) = z · e^{i·n·θ}
```

内积：

```
<f, g> = Re( f · ḡ ) = Re( q · e^{i·m·θ} · k̄ · e^{-i·n·θ} )
       = Re( q k̄ · e^{i(m-n)·θ} )
```

→ **仅依赖 m-n**。✓

---

## Slide 4 · 推广到 d 维

把 d 个分量两两分组 → d/2 个复数 pair：

```
group_i: dim (2i, 2i+1) ← 一个复数 z_i = x_{2i} + i x_{2i+1}
```

每 group 用不同频率 θ_i 旋转。

```
θ_i = base^{-2i/d}    （base = 10000 默认）
```

---

## Slide 5 · 矩阵表达

每 group 是 2D 旋转：

```
R(m·θ_i) = [ cos(m·θ_i)  -sin(m·θ_i) ]
           [ sin(m·θ_i)   cos(m·θ_i) ]
```

整 RoPE = block-diagonal 矩阵，d/2 个 2D 旋转块。

---

## Slide 6 · 实现的简化形式

```python
def rope(q, cos, sin):
    # q: (..., t, d)
    # cos, sin: (t, d/2)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    rot1 =  q1 * cos - q2 * sin
    rot2 =  q1 * sin + q2 * cos
    out = torch.empty_like(q)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out
```

直接按 group 切，不需要复数 lib。

---

## Slide 7 · `cos, sin` 的预计算

```python
def build_cos_sin(t, d, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2) / d))  # (d/2,)
    pos = torch.arange(t).float()                            # (t,)
    angles = pos[:, None] * inv_freq[None, :]                # (t, d/2)
    return angles.cos(), angles.sin()
```

每次 forward 前算一次（或 cache）。

---

## Slide 8 · 验证位置相对性

```python
q = ones(d=4); k = ones(d=4)
qr_m = rope(q, ..., m=5)
kr_n = rope(k, ..., n=8)
score = qr_m @ kr_n.T
# score 应只取决于 n - m = 3
```

m=10, n=13 vs m=5, n=8 → score 一致。✓

---

## Slide 9 · "为什么不用复数 dtype"

PyTorch 复数 dtype 慢、kernel 少。用实数 sin/cos 等价表达 + element-wise 即可。flash-attn 也是这样。

---

## Slide 10 · interleaved 还是 split？

两种切分方式：

```
interleaved:  [x0, x1, x2, x3] → pair (x0,x1) (x2,x3)
split:        [x0, x1, x2, x3] → pair (x0,x2) (x1,x3)
```

- 原论文 + flash-attn: **interleaved**
- HuggingFace 早期：split
- → 跨 lib 时务必匹配，否则乱码

---

## Slide 11 · base = 10000 的选择

```
base 大 → 慢频率 → 长距区分能力强但短距弱
base 小 → 快频率 → 反之
```

Llama-1/2: 10000 (4k context)
Llama-3.1: 500000 (128k)
DeepSeek-V3: 1000000 (128k)

base 越大支持越长上下文。

---

## Slide 12 · RoPE 外推问题

```
训 4k base=10000 → 推 8k 直接炸
原因：base 不变 → 远距角度不再可区分
```

解决方案（专题 5 详讲）：
- Position Interpolation: pos / scale
- NTK-aware: 改 base
- YaRN: NTK by parts + attn temp

---

## Slide 13 · "Position Interpolation"

```
m' = m · (L_train / L_new)
```

把推理时位置缩到训练范围内。Meta 2023。

简单粗暴，对 4-32k 有效。

---

## Slide 14 · "NTK-aware scaling"

```
base' = base · (L_new / L_train)^{d / (d - 2)}
```

只改 base 不改 m。理论上保留高频细节。

LocalLlama 社区贡献，2023.06。

---

## Slide 15 · 与 attention 集成

```python
class RoPEAttention(nn.Module):
    def forward(self, x, pos):
        q = self.q(x); k = self.k(x); v = self.v(x)
        # 切多头
        q = q.view(b, t, h, d_head).transpose(1, 2)
        k = k.view(b, t, h, d_head).transpose(1, 2)
        # 应用 RoPE
        cos, sin = build_cos_sin(t, d_head)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        # 然后 attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
```

---

## Slide 16 · KV cache + RoPE

新生成的 token 位置 = `cache_len + new_idx`：

```python
def forward_with_cache(self, x_new, cache):
    pos_new = cache.length + arange(t_new)
    cos, sin = build_cos_sin(at_pos=pos_new)
    q_new = rope(q_new, cos, sin)
    k_new = rope(k_new, cos, sin)
    cache.k = cat([cache.k, k_new], dim=-2)
    cache.v = cat([cache.v, v_new], dim=-2)
```

旧 K 已经旋转过，不再 re-rotate。

---

## Slide 17 · "为什么不旋转 V"

```
attn = softmax(Q_rot @ K_rot^T) @ V
       │                 │       │
       └── 位置在 score 中体现   └── V 保持原内容
```

V 是"原始信息"，旋转会污染内容。Su 2021 设计哲学：**位置只影响相似度，不改变信息**。

---

## Slide 18 · 与 ALiBi 数学对比

```
RoPE: score(i, j) = <q rot_i, k rot_j> = f(<q,k>, j-i)
ALiBi: score(i, j) = <q, k> + m·(j-i)
```

RoPE 是"乘性"，ALiBi 是"加性"。

→ RoPE 表达力强（可学复杂位置模式），ALiBi 更简单更稳定外推。

---

## Slide 19 · 与 flash-attn 集成

```python
from flash_attn.layers.rotary import apply_rotary_emb
q_rot = apply_rotary_emb(q, cos, sin, interleaved=False)
```

flash-attn 的 RoPE 必须用同 base / interleaved 参数。

---

## Slide 20 · 实现细节 — 双精度

cos/sin 在 fp32 算（避免 bfloat16 精度损失）：

```python
inv_freq = inv_freq.float()
angles = (pos.float() @ inv_freq).cos()
return angles.cos().to(q.dtype), ...
```

bfloat16 角度精度只 ~3 位，rotate 后误差累计。

---

## Slide 21 · 实现细节 — half-dim only

只在前 d/2 维 rotate，后 d/2 不动？

某些 lib (early HF Llama) 这样实现：

```
q_rot[:d/2] = rotate(q[:d/2])
q_rot[d/2:] = q[d/2:]    # 不动
```

但现代 RoPE (Llama-2+, flash-attn) 是 **全 d 维都 rotate**。

---

## Slide 22 · 调试 RoPE 的 checklist

```
[ ] interleaved 还是 split？
[ ] base 值？(10000 / 500000 / ...)
[ ] cos/sin 在 fp32 算
[ ] 全 d_head 还是 half rotate？
[ ] KV cache 中 K 是否已 rotate？
[ ] position id 起 0 还是 1？
[ ] 与 lib `apply_rotary_emb` 调用约定一致？
```

每一项都可能导致输出乱码。

---

## Slide 23 · 测试与 lib 一致性

```python
import torch
from flash_attn.layers.rotary import apply_rotary_emb
q = torch.randn(1, 4, 16)
cos, sin = build_cos_sin(4, 16)
my = rope(q, cos, sin)
lib = apply_rotary_emb(q, cos, sin, interleaved=True)
assert (my - lib).abs().max() < 1e-4
```

---

## Slide 24 · "Yarn 之后"的世界

YaRN (2023.09) 后 RoPE 已成事实上的"现代长上下文 PE"标配。

- Llama-3.1 128k: RoPE + YaRN
- Qwen-2.5 128k: RoPE + YaRN
- DeepSeek-V3 128k: RoPE + YaRN-variant

详见专题 5 长上下文。

---

## Slide 25 · RoPE 的"扩展"

```
2D RoPE (图像 patch)
3D RoPE (视频)
M-RoPE (Qwen-VL: 多 modality)
Llama-3 RoPE scaling (3 piece)
```

新方向：multi-modal / multi-scale。专题 5 / 7 多模态部分详讲。

---

## Slide 26 · 实现完整版（伪）

```python
class RoPE(nn.Module):
    def __init__(self, dim, max_seq, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def cos_sin(self, t, device, dtype):
        pos = torch.arange(t, device=device, dtype=torch.float32)
        angles = pos[:, None] * self.inv_freq.to(device).float()[None, :]
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(self, q, k, t=None):
        t = t or q.shape[-2]
        cos, sin = self.cos_sin(t, q.device, q.dtype)
        # apply
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
```

详见 `src/rope.py`。

---

## Slide 27 · 性能

RoPE 计算量：~ O(t · d) per Q/K (vs attention O(t² d))。

flash-attn 把 RoPE 融进 kernel，几乎零成本。

---

## Slide 28 · 课后思考

1. 复数推导中 e^{i·m·θ} 的 m 是位置还是 head id？
2. interleaved vs split 在数学上等价吗？
3. KV cache 必须在 RoPE 之后保存。为什么？
4. base 改大对短距区分能力的影响？

---

## 参考

- Su 2021 (RoPE)
- Liu 2023 (Position Interpolation)
- bloc97 2023 (NTK-aware, social media post)
- Peng et al. 2023 (YaRN)
- flash-attn rotary docs
