# L10 · Dynamic Draft Tree

## 1 · EAGLE-2 的 dynamic tree 算法

每步生成树：
```python
def build_dynamic_tree(h_root, draft_model, max_K=32, max_depth=6):
    leaves = [(h_root, [], 0.0)]   # (state, tokens, logprob)
    final = []
    while leaves and len(final) < max_K:
        leaf = max(leaves, key=lambda l: l[2])    # most promising
        leaves.remove(leaf)
        h, toks, lp = leaf
        if len(toks) >= max_depth:
            final.append(leaf)
            continue
        logits = draft_model(h)
        for tid in topk(logits, 3):
            new_lp = lp + log_softmax(logits)[tid]
            new_h = draft_model.advance(h, tid)
            leaves.append((new_h, toks + [tid], new_lp))
    return final
```

## 2 · 关键：beam search 风格
- 用 priority queue 保留最有希望的路径
- 不浪费 verify 在低 lp 候选

## 3 · target verify
- 把树扁平化为 token list
- 构造 tree mask
- 一次 forward
- 自上而下接受路径：第一个 reject → 截断

## 4 · accept depth 分布
| depth | 比例 |
|-------|-----|
| 1 | 95% (root 几乎总接受) |
| 2 | 75% |
| 3 | 50% |
| 4 | 30% |
| 5 | 15% |
| 6+ | 5% |

每 iter 期望 accept ≈ 4 token.

## 5 · 实现示意
继承 eagle2.py 的 DraftTree 类，加 dynamic 扩展逻辑。
