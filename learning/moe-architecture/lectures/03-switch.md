# L03 · Switch Transformer — top-1 极简

> 18 slides | 55 min | MoE Architecture 第 3 讲 ⭐⭐⭐⭐

> Google 2021 / 1.6T 参数 MoE 工程证明

---

## Slide 1 · Switch 定位

```
GShard top-2  → Switch top-1
GShard 复杂   → Switch 极简
GShard 100B  → Switch 1.6T (16× 大)
```

→ "更少 expert per token，更简单"。

---

## Slide 2 · top-1 router

```python
def switch_route(x, W_g):
    logits = x @ W_g
    gates_all = softmax(logits)
    top1_gates, top1_idx = gates_all.max(dim=-1, keepdim=True)
    return top1_gates, top1_idx
```

每 token 只走 1 个 expert。

---

## Slide 3 · 算力 50% 省

```
GShard:  每 token × 2 expert FFN
Switch:  每 token × 1 expert FFN
```

FLOPs 减 50%（同 expert size）。

---

## Slide 4 · 容量更紧

```
capacity = 1.0 × (n_tokens / n_experts)
```

无 top_k 乘数，capacity 与 dense token 数一致。

---

## Slide 5 · 1.6T 参数 实战

Switch-XXL 模型：
- 1.6T params
- 2k experts × 800M dense base
- 训 32B token

→ 当时最大开源模型。

---

## Slide 6 · 训练稳定 trick

Switch 论文报告：
- expert dropout 0.1（强 reg）
- aux loss α = 0.01
- capacity factor 1.0

3 个超参一调就稳，比 GShard 易。

---

## Slide 7 · top-1 vs top-2 性能

```
同 7B 总参:
  top-1 (Switch):   baseline
  top-2 (Mixtral):  +0.5-1 pp MMLU
```

top-2 略好，但 top-1 简单稳定。

---

## Slide 8 · Switch 论文公式

aux loss:
```
α · N · Σ_i (f_i × P_i)
```

f_i = 实际频率, P_i = 平均概率。N = expert 数。

---

## Slide 9 · expert dropout

```python
if training:
    expert_mask = bernoulli(0.9)    # drop 10% experts
    gates = gates * expert_mask
```

防止 router 过度依赖少数 expert。

---

## Slide 10 · Switch → Mixtral / DeepSeek 路线

```
Switch top-1   → Mixtral top-2 (回 GShard 风)
                ↓
              DeepSeek top-8 (细粒度多激活)
```

→ "极简单" → "回归 top-2" → "极细粒度"，三阶段。

---

## Slide 11 · 推理时

Switch top-1 推理：
- 每 token 1 个 expert 查找
- 非常快（vs dense）

GPT-4 / Mixtral 部分推理优化沿用 Switch 思想。

---

## Slide 12 · "router 崩塌" 问题

无 aux loss 时：
```
1. router 学早期分配
2. expert 0 总是收到
3. expert 1-N 几乎不更新
4. 模型退化为 dense (1 expert)
```

aux loss 强迫 router 探索。

---

## Slide 13 · Switch 与 ST-MoE

后续 ST-MoE (2022)：
- 加 router z-loss（防 logits 爆炸）
- 更稳定 scaling

后续 (Llama / DeepSeek) 都沿用 z-loss + aux 组合。

---

## Slide 14 · top-1 代码（极简）

```python
class SwitchRouter(nn.Module):
    def __init__(self, d, n_expert):
        self.W = Linear(d, n_expert, bias=False)
    def forward(self, x):
        logits = self.W(x)
        gates_all = logits.softmax(-1)
        gate, idx = gates_all.max(-1, keepdim=True)
        aux = aux_loss(gates_all, idx)
        return gate, idx, aux
```

---

## Slide 15 · 训练实务

```
training:    capacity 1.0 + aux α=0.01 + z-loss α=0.001
inference:   capacity ∞ (no drop)
```

---

## Slide 16 · Switch 失败模式

```
1. 全部 token → 1 expert → 退化 dense
2. expert 不稳 (训练初期 gate 学得太快)
3. aux α 太大 → router 不学
```

aux + z-loss + warmup 联合解决。

---

## Slide 17 · Switch 历史地位

2021 Switch 让"超过 GPT-3"的模型可达，证明 MoE 可上规模。

→ Mixtral / DeepSeek 站在 Switch 肩膀上。

---

## Slide 18 · 课后思考

1. top-1 在小模型 (< 1B) 是否值得？
2. capacity 1.0 是否对长 context 友好？
3. expert dropout 与 attention dropout 是否冗余？
4. Switch 1.6T 推理实际成本？

---

## 参考

- Fedus et al. 2021 (Switch Transformer)
- Zoph et al. 2022 (ST-MoE)
- Mixtral 2024
