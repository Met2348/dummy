# L08 · 成本工程（跨 ckpt 对照）

## 5 ckpt 的隐性成本

| ckpt | params | latency | $/M-token (in) | total |
|------|--------|---------|----------------|-------|
| vanilla 124M | 124M | 30ms | $0.05 | 最便宜 |
| lora 124M | 124M | 35ms | $0.05 | 接近 vanilla |
| dpo 124M | 124M | 40ms | $0.05 | 安全分类 + 5ms |
| r1_tiny 124M | 124M | 80ms | $0.10 | 长 think，token 多 |
| phi_tiny 270M | 270M | 60ms | $0.08 | 大模型但快 |

## R1 的 hidden cost

R1-style 输出 `<think>...</think><answer>...</answer>`：
- 用户看 `<answer>` 短
- 但 token 计费看全 trace → 5-10× cost

```
vanilla:  "Paris."          → 1 token out
r1_tiny:  "<think>...</think><answer>Paris</answer>"  → 50 token out
```

→ 通用 chat 用 r1_tiny **冤大头**。
→ 难推理才用 r1_tiny。

## 路由决策

```python
def smart_route(query):
    if is_simple(query):    # 知识 Q
        return vanilla      # $0.05
    if needs_reasoning(query):
        return r1_tiny      # $0.10
    if safety_sensitive(query):
        return dpo          # $0.05
    return phi_tiny         # $0.08 (default safe)
```

→ 平均成本 = 加权和。

## 我们 capstone 的成本估算

mini-HELM 跑 5 ckpt × 5 题 mock = 0 cost。
真 HELM 跑 100k token = ~$5。
真 Chatbot Arena 持续 = $$M。

## $/quality 比

```
quality / $ = avg_score / cost
```

| ckpt | quality | $/M-token | $/quality |
|------|---------|----------|-----------|
| vanilla | 0.50 | 0.05 | 0.10 |
| lora | 0.99 | 0.05 | 0.05 |
| dpo | 0.98 | 0.05 | 0.05 |
| r1_tiny | 0.88 | 0.10 | 0.11 |
| phi_tiny | 0.93 | 0.08 | 0.09 |

→ **lora / dpo 性价比最高**（小 cost + 高 quality）。

## 一句话

> 部署 ≠ 选最强 ckpt — 选 quality/$ 比最高的 ckpt。
