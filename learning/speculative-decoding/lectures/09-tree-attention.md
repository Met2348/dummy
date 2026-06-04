# L09 · Tree Attention（并行 verify 多 draft）

## 1 · 痛点
draft 一棵树（K 路径），verify 时**必须**一次 forward 算所有 K × depth 的 logits。
若用 batch K → 内存 K× + 重复算共同 prefix。

## 2 · 解：tree attention mask
所有 K 条路径**共享 prefix KV**，只是树的分叉处 attention mask 不同。

```
        root
       /    \
      A      B
     /        \
    Aa         Ba

flat token list: [root, A, Aa, B, Ba]
positions:       [  0 ,  1,  2 , 3 , 4 ]

tree mask:
  Aa attends [0, 1, 2]  (祖先链)
  Ba attends [0, 3, 4]
```

## 3 · 一次 forward 算完
- 输入 [root, A, Aa, B, Ba] (5 token)
- mask 阵让 Aa 不看到 B 分支
- 一次 forward → 得每个 position 的 next-token logits

## 4 · 工程实现
- FlashAttention v2/v3 都支持 `tree_attention_mask`
- vLLM 0.7+ 的 `spec_decode` 集成
- SGLang `--enable-tree-decoding`

## 5 · 性能数字
| K 路径 | naive batch | tree attn |
|--------|-----------|-----------|
| 32 | 32x cost | 1x cost (+ tiny overhead) |

## 6 · 实现：[tree_attention.py](../src/tree_attention.py)
- `build_tree_mask` 构造 NxN bool mask
- `tree_attention_torch` 朴素实现
- 与 batched-naive 对照
