# L02 · GShard — top-2 + Expert Parallel

> 20 slides | 60 min | MoE Architecture 第 2 讲 ⭐⭐⭐⭐

> Google 2020 / "MoE 上规模"的第一篇工程化论文

---

## Slide 1 · GShard 定位

```
Shazeer 2017: 小规模 LSTM MoE 概念验证
GShard 2020: Transformer + 上千 expert + 多机
```

scale 1000× expert，跨多 TPU 分布。

---

## Slide 2 · top-2 routing

GShard 用 top-2（也是 Shazeer 沿用）：

```
gates = softmax(x @ W_g)         # (n_tok, n_expert)
top2_gates, top2_idx = gates.topk(2)
output = top2_gates[:, 0] · expert[top2_idx[:, 0]] +
         top2_gates[:, 1] · expert[top2_idx[:, 1]]
```

---

## Slide 3 · capacity factor

```
capacity = 1.25 × (n_tokens × top_k / n_experts)
```

每 expert 接 1.25× "理论平均" 数量，超出 token 直接 drop（不进 MoE）。

→ 防 GPU 显存爆。但 drop rate > 5% 模型 ppl 显著上升。

---

## Slide 4 · 负载均衡 aux loss

```
aux = α · (Σ_i f_i × P_i)
where:
    f_i = 实际接到 token 数 / n_tokens     # 实际频率
    P_i = 平均概率 (gates.mean(0))         # router 偏好
α = 0.01
```

f, P 都接近 1/n_experts → loss 最小。

---

## Slide 5 · 与 Shazeer aux 区别

```
Shazeer 2017:   CV(load)² + CV(importance)²
GShard:         f · P (更平滑)
```

GShard 形式更稳定，被 Switch / Mixtral 沿用。

---

## Slide 6 · expert parallel

```
GPU 0:  expert 0, 1, ..., 7
GPU 1:  expert 8, 9, ..., 15
...
```

每 token 按 top_k_idx all-to-all 到对应 GPU。

→ 通信成本高，但容量扩展无上限。

---

## Slide 7 · all-to-all 通信

```
forward:    token → 目标 GPU
backward:   gradient → 源 GPU
```

每 step 2 次 all-to-all（forward + backward）。

是 MoE 训练 bottleneck。

---

## Slide 8 · GShard 论文成绩

```
600B 参数 multilingual translation:
- 100 语言
- 训 2 days on 2048 TPU v3
- BLEU 平均 +13%
```

证明 MoE 在大规模 task 的有效性。

---

## Slide 9 · GShard 路由实现

```python
def gshard_router(x, W_g, n_experts, capacity):
    logits = x @ W_g
    gates = softmax(logits)
    top2_gates, top2_idx = gates.topk(2)
    # capacity 限制
    expert_counts = counter(top2_idx.flatten())
    for e in range(n_experts):
        if expert_counts[e] > capacity:
            # drop overflow
            ...
    return top2_gates, top2_idx, aux_loss
```

---

## Slide 10 · "drop"的实现

```
方案 1: zeroing — overflow token 输出 0
方案 2: skip — overflow token 不进 MoE 整层
```

GShard 用方案 1。Switch 改 方案 2。

---

## Slide 11 · expert utilization

理想：每 expert 1/n_experts 利用率。
实务：初期失衡，aux loss 推动均衡。

```
no aux loss:    expert 0 70%, others < 5%
+ aux loss:     each ~ 1/n_experts ± 10%
```

---

## Slide 12 · 工程坑

```
1. capacity 太小 → drop 多 → loss 高
2. capacity 太大 → 显存爆
3. aux loss 太大 → router 不学
4. aux loss 太小 → 路由崩塌
```

GShard 实测 α = 0.01 是甜点。

---

## Slide 13 · 与 Shazeer 工程差异

| | Shazeer 2017 | GShard 2020 |
|---|--------------|-------------|
| 训练规模 | 单机 | 上千 TPU |
| 多机通信 | 无 | all-to-all |
| Auto-sharding | 无 | 有（XLA）|
| 工业实用 | demo | yes |

---

## Slide 14 · 推理时是否需 capacity

```
训练: 需要（防 OOM）
推理: 不需要（单 token 一次）
```

→ 推理时可放宽 capacity，提高质量。

---

## Slide 15 · GShard → Switch 演进

GShard top-2 → Switch top-1。

Switch 论文证明 top-1 也行，且更简单 / 更省。

---

## Slide 16 · 代码（src/gshard_router.py）

```python
class GShardRouter(nn.Module):
    def __init__(self, d, n_expert, capacity_factor=1.25):
        super().__init__()
        self.W = nn.Linear(d, n_expert, bias=False)
        self.cap_f = capacity_factor
    def forward(self, x):
        logits = self.W(x)
        gates_all = logits.softmax(-1)
        top2_gates, top2_idx = gates_all.topk(2, dim=-1)
        # aux
        f = bincount(top2_idx.flatten()) / x.shape[0]
        p = gates_all.mean(0)
        aux = n_expert * (f * p).sum()
        return top2_gates, top2_idx, aux
```

---

## Slide 17 · expert dropout

GShard 还提出 "expert dropout"：训练中随机 mask 部分 expert。

类似 attention dropout，防 router 过拟合。

---

## Slide 18 · 与 dense scaling 对比

```
dense T5-XXL 11B:  baseline
GShard 600B MoE:   +14% BLEU
```

但 GShard 训练成本 ≈ 5× dense（多通信）。

---

## Slide 19 · 后续影响

GShard 是几乎所有现代 MoE 的工程蓝本：
- Switch, Mixtral, DeepSeek-V3, Phi-MoE 都借鉴 GShard
- 至今 capacity factor 仍是关键超参

---

## Slide 20 · 课后思考

1. capacity factor 在 推理时 应设多少？
2. top-2 比 top-1 提升 1.5pp，多少 额外算力？
3. all-to-all 通信能否替换为更便宜的方案？
4. expert parallel 与 tensor parallel 兼容吗？

---

## 参考

- Lepikhin et al. 2020 (GShard)
- Switch Transformer 2021
- DeepSeek-V3 2024
