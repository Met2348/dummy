# L02 · Positional Encoding 全套

> 24 slides | 70 min | Transformer Deep 第 2 讲 ⭐⭐⭐⭐

> Sinusoidal / Learned / RoPE / ALiBi / NoPE 横向对比

---

## 学习目标

1. 理解为什么 attention 需要 PE
2. 掌握 5 种主流 PE 的数学与实务差异
3. 知道每种 PE 的"推外"能力
4. 选型决策（绝对 / 相对 / 无 PE）

---

## Slide 1 · 为什么 attention 需要 PE

self-attention 是 **permutation-invariant**：
```
attn([w_1, w_2, w_3]) == attn([w_3, w_1, w_2])
```

→ 无法区分词序，必须额外注入位置信息。

---

## Slide 2 · 5 种 PE 对照

| 方法 | 类型 | 实现位置 | 外推 | 实务 |
|------|------|----------|------|------|
| Sinusoidal | absolute | embed 加 | 差 | 2017 |
| Learned | absolute | embed 加 | 几乎不能 | GPT-2 |
| **RoPE** | relative | Q,K 旋转 | 中（YaRN 后好）| Llama / Qwen |
| ALiBi | relative | attn bias | 优 | BLOOM / Falcon |
| NoPE | none | — | 中 | 长上下文研究 |

主流当前：**RoPE + YaRN scaling**（专题 5 详讲）。

---

## Slide 3 · Sinusoidal PE

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
```

性质：
- 不同维度对应不同频率
- 任意两位置可表示为旋转
- 但是是绝对，外推超过 max_len 后炸

---

## Slide 4 · Learned PE

```python
pos_embed = nn.Embedding(max_seq, d_model)
out = tok_embed(x) + pos_embed(arange(t))
```

GPT-2 / GPT-3 用。简单但：
- 上下文长度固定在训练时
- 无法外推
- 占额外参数（max_seq × d_model）

---

## Slide 5 · RoPE 直觉

把 Q, K 视为复数对，每个 dim 用不同频率旋转：

```
q_rot[2i]   = q[2i] · cos(m·θ_i) - q[2i+1] · sin(m·θ_i)
q_rot[2i+1] = q[2i] · sin(m·θ_i) + q[2i+1] · cos(m·θ_i)
```

m: 位置；θ_i = 10000^{-2i/d}.

关键性质：`<q_rot(m), k_rot(n)> = <q, k> · R(n-m)` → **隐式相对位置**。

L03 详细推导。

---

## Slide 6 · ALiBi (Press 2022)

Attention with Linear Biases：

```
attn(i, j) = Q_i · K_j - m · (j - i)
```

m 是每头一个常数。距离越远，attention 越被压制。

- 完全无 embedding 修改
- 训 512 推 4k 自然外推
- 缺：表达力弱（无法学复杂位置模式）

---

## Slide 7 · ALiBi vs RoPE

| | ALiBi | RoPE |
|---|------|------|
| 实现 | attn bias | Q/K 旋转 |
| 训外推 | 自然 ✓ | 需 YaRN 补 |
| 长上下文 | 优 | 中等 |
| 灵活性 | 低 | 高 |
| 现代采用 | 少 | 主流 |

最近 Llama-3 / Qwen 选 RoPE+YaRN，因为表达力高。BLOOM / Falcon 选 ALiBi。

---

## Slide 8 · NoPE (Kazemnejad 2023)

不加任何 PE。仅靠 causal mask 提供"非对称"。

实验：小模型上 NoPE 与 RoPE 差距小，且外推可控。Long-context 研究中重新被关注。

但实际产品 LLM 还无人采用 NoPE，原因：训练初期收敛慢。

---

## Slide 9 · base 参数 (10000) 的本质

```
θ_i = base^{-2i/d}
```

base 大 → 高频快变 → 短距区分好，长距区分差。

RoPE 默认 10000（Vaswani 2017 沿袭）。
Llama-3.1 改 500_000（为长上下文）。
DeepSeek-V3 调到 1_000_000。

---

## Slide 10 · 改 base 实现外推

```
原训 4k base=10000 → 推 32k 直接炸
↓ 切 base=500000 → 频率慢下来 → 推 32k 可以
```

Position Interpolation / NTK-aware / YaRN 都是改 base 思路的变体（专题 5）。

---

## Slide 11 · Sinusoidal 可视化

```
dim 0  (高频)   ─ 短距区分
dim 32 (中频)  ─ 中距
dim 63 (低频)  ─ 长距区分
```

每个 dim 对应一个频率，全 d 维各自贡献不同尺度。

---

## Slide 12 · ALiBi 头的斜率

```
m_h = 2^(-8h/H)   # 第 h 头的斜率
```

最浅头 m=2⁻⁸ 接近无 bias，最深头 m=2⁻¹⁶ 几乎只看近邻。

→ 不同头自动负责不同尺度。

---

## Slide 13 · 实务对比：训 vs 推

```
任务: 训 512 → 推 4k
              ↓
Sinusoidal:    ppl 飙升
Learned:       完全炸
RoPE 10000:    炸（无 YaRN）
RoPE 500000:   慢慢退化
ALiBi:         几乎不变
NoPE:          中等
```

→ 长上下文专题（5）会深入讲 RoPE scaling。

---

## Slide 14 · 模型 PE 选择速查

| 模型 | PE |
|------|----|
| GPT-2/3 | Learned absolute |
| Llama-1 | RoPE 10000 |
| Llama-2 | RoPE 10000 |
| Llama-3 | RoPE 500000 + RoPE scaling |
| Mistral | RoPE 1000000 |
| BLOOM | ALiBi |
| Falcon | ALiBi |
| Qwen-2.5 | RoPE 1000000 |
| DeepSeek-V3 | RoPE 1000000 |

→ 主流"RoPE + 大 base"。

---

## Slide 15 · PE 是否参与 attention bias？

| 类 | 加在哪 |
|----|---------|
| Sinusoidal/Learned | embedding 之前加 |
| RoPE | Q, K 在 attn 计算前旋转 |
| ALiBi | attn score 上加 bias |

RoPE 不进 V，只改 Q, K。

---

## Slide 16 · RoPE 不进 V 的几何意义

```
attn = softmax(Q_rot K_rot^T / √d) V
                   ↑
                   位置仅影响"相似度"
                   而不影响"信息内容"
```

V 是 "内容"，不该被旋转污染。这是 Su 2021 的设计思想。

---

## Slide 17 · 实务代码 — Sinusoidal

```python
def sinusoidal_pe(t, d, base=10000):
    pos = torch.arange(t).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * -math.log(base) / d)
    pe = torch.zeros(t, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
```

---

## Slide 18 · 实务代码 — ALiBi

```python
def alibi_slopes(num_heads):
    start = 2 ** (-8.0 / num_heads)
    return start ** torch.arange(1, num_heads + 1)

def alibi_bias(t, num_heads, device):
    slopes = alibi_slopes(num_heads).to(device)
    pos = torch.arange(t, device=device)
    bias = -(pos[None, :] - pos[:, None]).abs()
    return slopes[:, None, None] * bias  # (h, t, t)
```

---

## Slide 19 · 实务代码 — Learned PE

```python
class LearnedPE(nn.Module):
    def __init__(self, max_seq, d):
        super().__init__()
        self.pe = nn.Embedding(max_seq, d)
    def forward(self, x, pos):
        return x + self.pe(pos)
```

---

## Slide 20 · RoPE 代码预告

下一讲完整写。这里只列接口：

```python
class RoPE:
    def __init__(self, dim, max_seq, base=10000):
        ...
    def __call__(self, q, k, pos): ...
```

---

## Slide 21 · 选型决策树

```
需求:
  - 训长上下文 ≤ 4k → RoPE 10000 (Llama-2) 或 ALiBi
  - 想 train short / infer long → ALiBi 或 RoPE+YaRN
  - 想 32k+ context → RoPE 500000+ (Llama-3) + YaRN
  - 多模态视频 → 3D RoPE (Qwen2-VL)
```

---

## Slide 22 · 实务陷阱

```
1. RoPE base 选错 → 长上下文炸
2. PE 加在 Q 前还是后 → 与 lib 不一致出错
3. 推理时 pos_id 错位 → 输出乱
4. KV cache + RoPE → 旧 K 已 rot，prefill 时需重 rot
```

---

## Slide 23 · 性能影响

```
计算:
  Sinusoidal: 一次表查
  Learned: 一次 embedding lookup
  RoPE: 旋转 = 2 mul + 1 add 每元素
  ALiBi: 加 bias 矩阵
```

RoPE 是其中最贵的（但仍 << attention 主体）。

---

## Slide 24 · 课后思考

1. 为什么 GPT-2 用 learned PE 不外推？
2. RoPE 比 sinusoidal 强在哪？
3. ALiBi 的 m_h 为什么用指数衰减？
4. 如果 NoPE 比 RoPE 在小模型上无差，为什么 LLM 仍用 RoPE？

---

## 参考

- Vaswani 2017 (Sinusoidal)
- Su 2021 (RoPE)
- Press 2022 (ALiBi)
- Kazemnejad 2023 (NoPE)
- Llama-3 tech report 2024 (RoPE 500000)
