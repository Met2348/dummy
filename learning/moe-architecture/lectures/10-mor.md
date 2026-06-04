# L10 · MoR — Mixture of Recursions

> 16 slides | 50 min | MoE Architecture 第 10 讲 ⭐⭐⭐

> 2024 新方向 / "MoE 的下一步"探索

---

## Slide 1 · MoR 定位

```
MoE: 选不同 FFN expert
MoR: 选不同"递归深度"
```

每 token 可选 1, 2, 3, ..., n 次重复 same block。

---

## Slide 2 · 动机

```
hard token  → 多次过同 block (更深推理)
easy token  → 1 次即可 (省算力)
```

类比人脑：困难题想久一点。

---

## Slide 3 · 算法

```python
class MoRBlock(nn.Module):
    def forward(self, x):
        depth_logits = self.depth_router(x)         # (n_tok, max_depth)
        depth_gates = softmax(depth_logits)
        top_d_gate, top_d_idx = depth_gates.max(-1)
        out = x
        for d in range(top_d_idx + 1):
            out = self.shared_block(out)
        return out * top_d_gate
```

shared block 复用，深度由 router 决定。

---

## Slide 4 · 与 MoE 关系

```
MoE: 选 expert (路由 N 个 FFN)
MoR: 选深度 (路由同 block 多次)
↓
MoE 在"宽度"维度，MoR 在"深度"维度
```

---

## Slide 5 · 算力 elasticity

```
固定 hardware:
  MoE → 总参容量 ↑
  MoR → 总深度容量 ↑（每 token 自适应）
```

适合 inference budget 不确定场景。

---

## Slide 6 · MoR vs Speculative Decoding

```
Spec decoding: 快小模型预 + 大模型验
MoR:           按需 deep router
```

不同思路达成"自适应算力"。

---

## Slide 7 · 实现细节

```
max_depth = 4
shared block = 1 个 transformer block
router = Linear(d, 4)
```

参数比 normal block 多了 router 和 reuse 控制。

---

## Slide 8 · 训练

```
forward 用 random depth (curriculum)
loss = ce + 0.1 * KL(deep || shallow)
```

防止 router 总是选 max_depth。

---

## Slide 9 · 与 Universal Transformer 关系

```
Universal Transformer (Dehghani 2018):
  动态 halting 每 token
MoR: 类似但用 router 决定 depth
```

MoR 是 UT 的现代 routing 版本。

---

## Slide 10 · 实务复杂

```
1. recursive forward 难 fuse
2. backward 需保存所有 depth 中间值
3. compile 优化困难
```

→ 未在大模型见用。

---

## Slide 11 · MoR 与 R1 推理

```
R1 用 token rollout 探索答案
MoR  用 block depth 探索答案
```

互补但目前 R1 占主流。

---

## Slide 12 · 论文级数字

MoR 论文（2024）：
```
1B 参数 + MoR depth 4:
  GSM8K +5% vs dense
  推理 latency +10% (动态深度)
```

→ 有效，但 +5% 不算革命。

---

## Slide 13 · 是否会主流

```
未来 trends:
- MoE 已主流
- MoR 实验性
- Hybrid (MoE + MoR) 有意思
```

2026 仍处早期。

---

## Slide 14 · MoR + MoE 混合

```
每 layer:
  - MoE 路由 expert
  - MoR 路由 depth
↓
两个独立 router 同时工作
```

理论上 OK，实务未见。

---

## Slide 15 · 类似方向

```
PaR (Pause as a Token)    类似 MoR
Looped Transformers       loop 整个 stack
COLT5 (conditional MLP)   类似 MoE
```

→ "自适应深度"是 active 研究方向。

---

## Slide 16 · 课后思考

1. MoR 在 R1 时代还有价值吗？
2. MoR + R1 token rollout 能不能合？
3. depth 路由的 grad flow 怎么算？
4. 推理时 batch 不同 depth 怎么并？

---

## 参考

- MoR paper 2024 (estimated, 方向已存在)
- Universal Transformer 2018
- PaR (Pause as Token) 2023
