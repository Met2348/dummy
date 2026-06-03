# L10 · Sliding Window Attention (SWA)

> 18 slides | 55 min | Transformer Deep 第 10 讲 ⭐⭐⭐

> Mistral 7B / Gemma 引入的 window attention

---

## 学习目标

1. 理解 SWA 的窗口设计
2. 知道为什么 SWA 能"伪长上下文"
3. 用 flash-attn window 接口

---

## Slide 1 · SWA 定义

```
attn(i, j) 仅在 |i - j| ≤ W 时算（否则 0）
```

W = 4k → 每个 token 只看前 4k 个邻居。

---

## Slide 2 · 复杂度

```
全 attn:  O(t²)
SWA:      O(t · W)
```

t=32k, W=4k → 8× 节省。

---

## Slide 3 · 信息传递

```
token at pos i 看 i-W ~ i
↓
层间传递: 第 L 层 token 实际能看到 L · W 个 token
↓
Mistral 32 层 × 4k window → 128k 上下文（理论）
```

→ "effective context = L · W"。

---

## Slide 4 · 哪些模型用 SWA

```
Mistral 7B:       W=4k, ctx 32k (effective 128k)
Mistral 8x7B (Mixtral): W=4k
Gemma 2:          每两层一次 SWA + 全 attn
Phi-3.5:          alternating
```

---

## Slide 5 · 实务图示

```
i=10, W=4:
attn 看 i-4 ~ i = [6, 7, 8, 9, 10]
其余 mask = -inf
```

下三角 + window 双条件 mask。

---

## Slide 6 · 与 causal 组合

```
mask(i, j) = -inf  if  (j > i) or (i - j > W)
           = 0     otherwise
```

bottom-triangular band → "因果窗口"。

---

## Slide 7 · alternate layers

Gemma 2 / Phi-3.5：
```
even layer: full causal
odd layer:  SWA (W=4k)
```

→ "局部窗" + "全局聚合" 交替，效果好。

---

## Slide 8 · flash-attn 接口

```python
from flash_attn import flash_attn_func
out = flash_attn_func(
    q, k, v, causal=True,
    window_size=(W, 0)    # (left, right)
)
```

right=0 → causal + 左 window。

---

## Slide 9 · 与 KV cache 配合

```
SWA + KV cache: 
  仅需保留最近 W 个 KV → KV cache 也 O(W)
```

→ 推理时 KV cache 不需"无限增长"。Mistral 推理省巨多。

---

## Slide 10 · 长上下文真正好处

实测 Mistral 32k:
- 短文 < 4k: 与 GPT-3.5 持平
- 4k-32k: 有效 (跨层传递)
- > 32k: 失效 (训练时无样本)

→ 不能无限延伸，受训练 max len 限制。

---

## Slide 11 · "attention sink" 在 SWA 中

```
最早的 token (pos 0-3) 总有高 attn
↓
SWA 之后 sink token 被滑出 window → 模型输出退化
```

→ StreamingLLM 解决：永久保留前 4-8 个 token 作 "sink"。

---

## Slide 12 · StreamingLLM 修正

```
attention = [前 4 sink tokens + 最近 W tokens]
```

兼顾 sink + 局部，可推 4M tokens。

---

## Slide 13 · 代码实现 — naive mask

```python
def swa_mask(t, W, device):
    pos = torch.arange(t, device=device)
    diff = pos[None, :] - pos[:, None]
    # causal: diff < 0 OR diff > W
    mask = (diff > 0) | (diff < -W)
    return mask  # True = mask out
```

---

## Slide 14 · 速度

```
A100 32k context:
  vanilla full:   OOM
  FA full:        可，但 O(t²)
  FA + SWA W=4k:  4× 更快
```

显著 + 实用。

---

## Slide 15 · 关于"真长"vs "SWA 长"

```
真 32k context (Llama-3 / Qwen 2.5):
   - 全 attn + RoPE scaling
   - 训练成本高
   - 跨远距推理强

SWA 32k (Mistral):
   - 局部窗 + 层间传递
   - 训练成本低
   - 远距推理弱
```

各有取舍。

---

## Slide 16 · SWA + MoE 组合

```
Mixtral 8x7B:  SWA + MoE
```

二者正交，组合后高效 + 高吞吐。后专题 3 详讲 MoE。

---

## Slide 17 · "效果"vs "效率"

Llama-3 8B 训 long ctx 直接 (RoPE + FA) → 性能更好。
Mistral SWA → 效率好。

各家路线不同。

---

## Slide 18 · 课后思考

1. 为什么 layer 间能传递信息？
2. W 越大效果越好？何时饱和？
3. SWA 适合 SFT 数据是 4k 时吗？
4. SWA + Ring Attention 能合用吗？

---

## 参考

- Mistral 7B paper 2023
- Gemma 2 paper 2024
- StreamingLLM (Xiao 2023)
- Mistral SWA blog
