# L11 · MoE 训练 — z-loss + Crash 防御

> 26 slides | 75 min | MoE Architecture 第 11 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 知道 MoE 训练的 5 类典型 crash
2. 掌握 router z-loss
3. 知道 capacity factor 实务
4. expert dropout / load balance tricks

---

## Slide 1 · 5 类典型 crash

```
1. router 崩塌 — 全 token 选 1 expert
2. logits 爆炸 — softmax overflow
3. load imbalance — 部分 expert 闲置
4. capacity overflow — 大量 drop
5. expert "死亡" — 永不更新
```

---

## Slide 2 · 解药 - router z-loss

ST-MoE (2022) 提出：
```
z_loss = (log Σ exp(logits))²
        ≈ (logsumexp(logits))²
```

防 logits 整体爆炸。

---

## Slide 3 · z-loss 推导

```
softmax over n_expert:
  P_i = exp(z_i) / Σ exp(z_j)
```

如果 Σ exp(z_j) → ∞，softmax 数值不稳。

z-loss 惩罚 logsumexp，迫使 logits 居中。

---

## Slide 4 · 完整 loss

```
total_loss = ce_loss + α_aux × aux_loss + α_z × z_loss
α_aux = 0.01
α_z = 0.001
```

DeepSeek-V3 用 Aux-Free 后 α_aux=0，但 z_loss 仍保留。

---

## Slide 5 · 实现

```python
def z_loss(logits):
    return torch.logsumexp(logits, dim=-1).pow(2).mean()
```

简单一行。

---

## Slide 6 · capacity factor 实务

```
training:  1.0 - 1.25
inference: ∞ (无 drop)
```

太小 → drop 多 → 信息丢失。
太大 → 显存爆。

---

## Slide 7 · expert dropout

```python
if training:
    # 训练时随机 mask 部分 expert
    mask = bernoulli(0.9)
    gates *= mask
```

类似 dropout，防 router 过度依赖少数 expert。Switch 用 0.1 dropout。

---

## Slide 8 · router warmup

前 1000 step 用 random routing，让所有 expert 都收到 token：

```python
if step < 1000:
    gates = random_uniform_routing(x)
else:
    gates = self.router(x)
```

防止"早期偏好"固化。

---

## Slide 9 · 路由崩塌可视化

```
step 1:    [25%, 25%, 25%, 25%]   ← 均匀
step 100:  [40%, 30%, 20%, 10%]   ← 开始偏
step 500:  [80%, 15%, 4%, 1%]     ← 危险
step 1000: [99%, 1%, 0%, 0%]      ← 崩塌
```

aux loss / Aux-Free 防止此 trajectory。

---

## Slide 10 · 崩塌防御 checklist

```
[ ] aux loss 或 Aux-Free 启用
[ ] z-loss 启用
[ ] expert dropout
[ ] warmup random routing
[ ] capacity factor 1.0+
[ ] monitor expert utilization
```

---

## Slide 11 · 监控指标

```
1. expert_utilization (load_i / total)
2. capacity_overflow_rate
3. aux_loss (or bias range for Aux-Free)
4. z_loss
5. main ce_loss
```

W&B / TensorBoard 实时绘图。

---

## Slide 12 · 注入 crash 实验

```python
# 故意不加 aux → 复现崩塌
model = MoE(d=64, n_expert=4, top_k=2, aux_alpha=0)
# 训 1k step
# expert utilization 应崩塌到 [99, 1, 0, 0]
```

学生体验 crash 后理解 防御。

---

## Slide 13 · DeepSeek-V3 训练栈

```
ce_loss + 0 × aux + 1e-3 × z_loss + Aux-Free bias
```

**最简化**配方，证明 Aux-Free 替代 aux。

---

## Slide 14 · 训练 hyperparam

```
batch (tokens):   4M (1024 × 4k seq)
lr:                3e-4
warmup:            2k step
weight_decay:     0.1
```

MoE 通常用与 dense 同 hyperparams。

---

## Slide 15 · "expert balancing" 工程

```
1. all-to-all 通信占 30% 时间
2. expert imbalance → 部分 GPU 闲置
3. capacity 不均 → bubble
```

均衡是性能优化的关键。

---

## Slide 16 · 实务建议

```
单卡训练: 4-8 expert
多卡训练: 32-128 expert (expert parallel)
跨机训练: 256+ expert (DeepSpeed-MoE)
```

---

## Slide 17 · grad accumulation 配合

```
batch=8, accum=4 → effective batch=32
↓
router 在小 batch 上看不足，更易崩塌
```

→ effective batch 越大越稳。

---

## Slide 18 · z-loss 与 dense 区别

```
dense softmax: rare overflow
MoE softmax (256 expert): 偶有
↓
z-loss 在 MoE 特别有用
```

---

## Slide 19 · 与 attention 不冲突

z-loss 只针对 router logits，不影响 attention softmax。

---

## Slide 20 · 训练时间分解

DeepSeek-V3 训练 step time:
```
attention:   30%
shared FFN:  10%
routed FFN:  20% (sparse)
all-to-all:  30%
其他:        10%
```

all-to-all 是 MoE 训练大头。

---

## Slide 21 · 代码 src/router_z_loss.py

```python
def router_z_loss(logits):
    return torch.logsumexp(logits, dim=-1).pow(2).mean()

# 训练
loss = ce_loss(...) + 1e-3 * router_z_loss(router_logits)
```

---

## Slide 22 · crash_demo.py

```python
# 不加 aux 启崩塌
model = MoE(aux_alpha=0)
# 训 200 step
util = [expert_utilization(model, x) for x in batches]
plot(util)  # 崩塌曲线
```

---

## Slide 23 · 失败案例

GShard 早期发现：
- 所有 token 流向 expert 0
- 1k step 后崩塌

→ 加 aux loss 后才能稳。

---

## Slide 24 · 推理时是否需 z-loss

不需要。z-loss 是训练-time regularizer。

推理时 fwd 即可，但 logits 仍需 fp32 防 overflow。

---

## Slide 25 · 综合训练 recipe

```python
loss = ce_loss
if use_aux_free:
    pass  # bias 自动更新
else:
    loss += 0.01 * aux_loss
loss += 0.001 * router_z_loss
```

5 行覆盖完整 MoE 训练。

---

## Slide 26 · 课后思考

1. z-loss α 取值范围？
2. expert dropout 与 attention dropout 关系？
3. 崩塌检测的自动化指标？
4. Aux-Free 与 z-loss 必须共用吗？

---

## 参考

- Zoph et al. 2022 (ST-MoE)
- Switch Transformer 2021
- DeepSeek-V3 报告 2024
