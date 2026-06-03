# L11 · μP + Architecture Search

> 20 slides | 60 min | Transformer Deep 第 11 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 理解 μP (Maximal Update Parametrization) 的核心思想
2. 知道超参可在小模型 transfer 到大模型
3. 大模型 architecture search 的实务

---

## Slide 1 · 标准训练的痛点

```
GPT-3 175B 训练 1 次成本 ~$4M
↓
超参调一次：lr / batch / warmup → 巨贵
```

希望"小模型调好 → 直接 transfer 到大模型"。

---

## Slide 2 · μP 提出

Yang & Hu 2022 (Tensor Programs V)：

```
若把 Linear 层 init/lr 按 (1/width)^x 缩放
→ 最优 lr / init 在 width 间稳定
```

→ 在 100M 上调出的 lr 直接用于 70B。

---

## Slide 3 · μP 三个 scaling rule

```
weight init  ~  N(0, σ² / m)    m = fan-in
forward 输出 = (1/m) Σ
backward grad ~ 不变
lr scale      ~ 1/m
```

→ 各层"动量"保持稳定，与 width 无关。

---

## Slide 4 · μP vs SP (Standard Parametrization)

```
SP:  lr / init 与 width 强相关，需重调
μP:  lr / init 与 width 解耦，可 transfer
```

μP 在 100M → 7B 的 lr 一致；SP 调 5-10×。

---

## Slide 5 · μP 在大模型的应用

| 模型 | μP |
|------|----|
| GPT-3 (2020) | 部分（init 调整）|
| OpenAI o1 | 报道用 μP |
| Llama-2/3 | 自报 SP，部分 μP-like 调整 |
| Cerebras | μP 标配 |

---

## Slide 6 · 调超参的代价节省

```
不用 μP:
  100M 调 → 7B 重调 → 70B 重调  = 3 次大模型实验

μP:
  100M 调 → 7B / 70B 直接用      = 1 次小模型实验
```

→ 100× 成本节省。

---

## Slide 7 · 实务步骤

```
1. 小模型 (~100M) 在小数据 (~10B token) ablation
2. 找 best lr, weight_decay, warmup
3. 直接用同 lr 训大模型
```

如不用 μP 需 sweep 大模型 lr。

---

## Slide 8 · 与 RoPE 配合

RoPE base 与 μP 独立。RoPE base 取决于 context 长度，μP 控制 width-scale。

---

## Slide 9 · μP init 公式

```python
def mup_init(module, d_model):
    for n, p in module.named_parameters():
        if 'weight' in n and p.dim() >= 2:
            fan_in = p.shape[-1]
            std = base_std / math.sqrt(fan_in / d_model)
            nn.init.normal_(p, std=std)
```

---

## Slide 10 · μP lr 调整

```python
for n, p in model.named_parameters():
    if n.endswith('weight') and p.dim() >= 2:
        fan_in = p.shape[-1]
        lr_factor = base_d / fan_in
        param_group_lr = lr_base * lr_factor
```

每层 lr 与 width 反相关。

---

## Slide 11 · architecture search 范畴

不止 hyperparameter — 也包括：
- depth / width 比例
- n_head / d_head
- attn / MLP ratio
- norm placement
- activation

---

## Slide 12 · Llama-3 architecture 决策

```
8B    32层 4096 32head 8 kv
70B   80层 8192 64head 8 kv
405B  126层 16384 128head 8 kv
```

→ depth-width 平衡经验：`depth × width² ≈ const × params`。

---

## Slide 13 · "Chinchilla optimal" 与 architecture

```
D = 20 × N    (token : param)
N 决定 architecture
```

Chinchilla 给出 N，architecture search 决定 depth/width/heads。

---

## Slide 14 · 启发式 architecture 公式

```
d_model = round(64 × √(N/M_ref))
n_head = d_model / 64 (或 128)
d_ff   = 8/3 × d_model (SwiGLU)
n_layer ≈ N / (12 × d_model²)
```

GPT-3 / Llama 都接近这套。

---

## Slide 15 · NAS (Neural Architecture Search)

```
传统 NAS: 上千次小实验
现代 LLM: 经验法则 + 1-2 次 ablation
```

LLM 太大不适合 NAS，多靠 Yang μP + manual design。

---

## Slide 16 · 不同 task 偏好 architecture

```
通用 LLM:    深窄 (Llama)
高吞吐推理: 浅宽 (Phi-3)
多模态:      attention 模块多
长上下文:    多 head + RoPE 配合
```

→ 没有"最优 architecture"。

---

## Slide 17 · μP 的局限

- BN / LayerNorm 是否需相应改 → 复杂
- 工程实现门槛高
- 实际 SP 调一次也能用，团队习惯 SP

→ 学术热，工业部分。

---

## Slide 18 · 工程 checklist

```
[ ] depth × width 比例
[ ] head_dim = 64 或 128
[ ] GQA group 数 (h/8)
[ ] vocab × d_model 不要太大
[ ] init std = 0.02 (SP) 或 μP 公式
[ ] lr = 3e-4 (small) ~ 1e-4 (large)
```

---

## Slide 19 · 与 scaling laws 配合

```
Kaplan (2020):  L ∝ N^{-α}
Chinchilla:     N × 20 ≈ D
μP:             跨 width 超参 transfer
```

三者结合 = 设计大模型的完整框架。

专题 6 详讲 scaling laws。

---

## Slide 20 · 课后思考

1. μP 在 init 阶段就 break SP 假设吗？
2. 如果不用 μP 直接训 70B，how much hyperparam search?
3. depth vs width 比例的实验如何设计？
4. NAS 在 LLM 上为什么少？

---

## 参考

- Yang & Hu 2022 (Tensor Programs V / μP)
- Cerebras-GPT 2023 (μP 应用)
- Kaplan 2020 (scaling laws)
- Llama-3 tech report
