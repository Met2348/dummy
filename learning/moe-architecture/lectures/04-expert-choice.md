# L04 · Expert Choice Routing — 反向路由

> 18 slides | 55 min | MoE Architecture 第 4 讲 ⭐⭐⭐

> Google 2022 / 由 expert 选 token，反向于 token 选 expert

---

## Slide 1 · 反向路由动机

```
token-choice:  token 选 top-k expert     → 有 capacity drop 问题
expert-choice: expert 选 top-k token     → 无 drop，自然均衡
```

→ 反向解决 imbalance。

---

## Slide 2 · expert-choice 算法

```
1. 算 全 (token × expert) logits
2. 对每个 expert，选 top-k token (k = capacity)
3. 每 token 可能被 0 / 1 / 2 / ... expert 选中
4. 多个 expert 选中 → sum
```

每 expert 严格 k 个 token，绝对均衡。

---

## Slide 3 · 实现

```python
def expert_choice_route(x, W, capacity_per_expert):
    logits = x @ W                            # (n_tok, n_expert)
    logits_T = logits.T                        # (n_expert, n_tok)
    # 每 expert 选 top-k token
    expert_gates, token_idx = logits_T.topk(capacity_per_expert, dim=-1)
    return expert_gates, token_idx
```

---

## Slide 4 · 优点

```
1. 无 capacity drop（每 expert 严格 k 个）
2. 不需要 aux loss
3. 训练更稳定
```

---

## Slide 5 · 缺点

```
1. token 可能被 0 个 expert 选 → 输出 0 → 丢信息
2. 推理时 batch=1 不可用（无法对所有 expert top-k）
3. 不适合自回归（每 token 算时不知其他 token）
```

→ 主要用于 encoder / BERT-MoE，不适合 GPT-style。

---

## Slide 6 · 对 LLM 不流行的原因

GPT 自回归推理：
- 每次只生成 1 个 token
- 无法 "expert 选 top-k 的 1 个 token"

→ 现代 LLM (Mixtral / DeepSeek) 都用 token-choice。

---

## Slide 7 · BERT-MoE / 训练时用 expert-choice

```
encoder pretrain:    用 expert-choice (绝对均衡)
decoder finetune:    切回 token-choice (实用)
```

混合策略，部分实务采用。

---

## Slide 8 · 与 GShard 对比

| | GShard | Expert Choice |
|---|--------|---------------|
| 谁选谁 | token → expert | expert → token |
| balance | aux loss 强制 | 算法保证 |
| drop | 有 | 无 |
| 自回归 | yes | no |
| 主流 | yes | no (BERT 限定) |

---

## Slide 9 · 算 token 利用率

```python
util = (token_chosen > 0).float().mean()
# 实测 ~ 80-90%（10-20% token 完全没被任何 expert 选中）
```

→ 10-20% 信息损失。

---

## Slide 10 · "buffer + token 0" 修正

```
对未被选 token: 走 shared expert (共享 MLP)
                ↓
        信息不丢
```

DeepSeekMoE 的 shared expert 部分受此启发。

---

## Slide 11 · 数学上等价的反 routing

```
Expert Choice = Linear assignment problem
                ↓
         Hungarian / auction 算法
```

但实务用 top-k argmax 已足够。

---

## Slide 12 · expert-choice + token-choice 混合

```
forward:
  token-choice top-2
  + 每 expert 再用 expert-choice 补 top-N 不饱和 token
```

Phi-MoE 等用类似混合。

---

## Slide 13 · 代码

```python
class ExpertChoiceRouter(nn.Module):
    def __init__(self, d, n_expert, capacity):
        self.W = Linear(d, n_expert, bias=False)
        self.cap = capacity
    def forward(self, x):
        logits = self.W(x).T                  # (n_expert, n_tok)
        expert_gates, token_idx = logits.topk(self.cap, dim=-1)
        return expert_gates, token_idx
```

---

## Slide 14 · 实务何时用

```
encoder MoE:        expert-choice
decoder MoE:        token-choice
hybrid (Phi):       token-choice + 补 expert-choice
```

---

## Slide 15 · 训练成本

```
expert-choice: 每 step 需排序 all tokens × all experts
              → O(n_tok × n_expert × log) 排序
token-choice:  O(n_tok × n_expert)
```

token-choice 计算更省，是主流原因之一。

---

## Slide 16 · 与 vision Transformer

ViT-MoE 中 expert-choice 较常用，因 vision token 数固定。

V-MoE (Google 2022) 用 expert-choice。

---

## Slide 17 · 总结

```
Expert Choice 是 routing 理论的"反方向"
绝对均衡 + 训练稳定
但不适合 自回归 LLM
→ 仍是有价值的研究方向
```

---

## Slide 18 · 课后思考

1. expert-choice 在 chat 推理时为什么不可用？
2. 10-20% token 丢失对 ppl 影响？
3. 混合 routing 是否值得？
4. ViT-MoE 与 LLM-MoE 算 routing 上的实务区别？

---

## 参考

- Zhou et al. 2022 (Expert Choice Routing)
- Riquelme et al. 2021 (V-MoE)
- Mixtral 2024
