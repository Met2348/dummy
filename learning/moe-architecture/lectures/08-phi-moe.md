# L08 · Phi-MoE — 小 MoE 路线

> 16 slides | 50 min | MoE Architecture 第 8 讲 ⭐⭐⭐

---

## Slide 1 · Phi-3.5-MoE / Phi-4-MoE

Microsoft 小 MoE 路线：

```
Phi-3.5-MoE:    16 × 3.8B = 61B 总,  6.6B 激活, top-2
Phi-4-MoE (估): 16 × 7B = 112B 总,  ~14B 激活, top-2
```

定位：与 Mixtral 同段，质量优于同等 dense。

---

## Slide 2 · Phi-3.5-MoE 性能

| Bench | Phi-3.5-MoE | Mixtral 8x7B |
|-------|--------------|--------------|
| MMLU | 78 | 70 |
| GSM8K | 84 | 60 |
| HumanEval | 70 | 40 |

→ 同等激活下显著优于 Mixtral，凭借 Phi 数据质量。

---

## Slide 3 · 与 Mixtral 架构差异

```
Mixtral:  8 expert × 7B
Phi-MoE:  16 expert × 3.8B
↓
expert 数 ↑, 单 expert 小 → 同等容量
```

中间路线（vs 8 大 / 256 小）。

---

## Slide 4 · 数据是关键

Phi 系列核心思想：**高质量合成数据 > 大量 web**。

```
training:
  ~ 50% 合成 (Phi-3 generates)
  ~ 30% web filtered
  ~ 20% code / math
```

Phi-3.5-MoE 沿用此配方。

---

## Slide 5 · top-2 routing

与 Mixtral 一致。Microsoft 没创新 routing 算法，专注数据。

---

## Slide 6 · 推理 footprint

```
Phi-3.5-MoE 4bit:  ~ 32GB → 5090 24GB OOM
Phi-3.5-MoE FP8:    ~ 62GB → 多卡或 H100
```

不适合单卡 5090。

---

## Slide 7 · Phi-3.5-MoE 加载

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-MoE-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

---

## Slide 8 · 与 DeepSeek-V3 对比

```
Phi-3.5-MoE: 16 routed, top-2, aux loss
DeepSeek-V3: 256 routed + 1 shared, top-8, Aux-Free
```

Phi 简单，DeepSeek 极致。

---

## Slide 9 · 应用场景

```
Phi-MoE: 边缘推理 / 中等 GPU 服务
DeepSeek-V3: 数据中心 / 高并发
Mixtral: 通用研究
```

---

## Slide 10 · 训练

Phi-3.5-MoE 训练数据 ~ 4T token。
比 Mixtral 8x7B ~ 同量级。

---

## Slide 11 · expert utility

Phi 论文报告：各 expert utilization 接近，aux loss 调好。

---

## Slide 12 · Phi-4-MoE 估计

未公开，但根据 trend：
- expert ~ 16
- routing top-2 或 top-3
- 加入更多合成数据

---

## Slide 13 · 小 MoE 的"性价比"

```
8 × 7B + top-2 (Mixtral): 13B 激活
16 × 3.8B + top-2 (Phi):   7B 激活
↓
后者激活更少，但精度更高（数据驱动）
```

---

## Slide 14 · 数据 vs 算法

Phi 团队哲学：
> "Architecture 是 5%, 数据是 95%"

Phi-MoE 是这个哲学的 MoE 版本。

---

## Slide 15 · 加载脚本（src/phi_moe_load.py）

```python
CONFIGS = {
    "Phi-3.5-MoE": dict(n_layer=32, hidden=4096, n_head=32, n_kv=8,
                        n_experts=16, top_k=2, total=61e9, active=6.6e9),
}
```

只展示 architecture 配置，不下载权重。

---

## Slide 16 · 课后思考

1. 16 expert × 3.8B vs 8 × 7B 哪个更利于推理？
2. Phi-MoE 的合成数据能否复刻？
3. Microsoft 会引入 Aux-Free 吗？
4. 小 MoE 是否会取代 dense 7B？

---

## 参考

- Phi-3 / Phi-3.5 reports 2024
- Phi-4 technical report 2024.12
