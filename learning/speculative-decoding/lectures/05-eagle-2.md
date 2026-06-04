# L05 · EAGLE-2（Li 2024.06）⭐

## 1 · 痛点（EAGLE-1 单 path）
draft 走单条路径 → 万一 reject 第 2 个 token 浪费后面 3 个。

## 2 · 解：dynamic draft tree
每步 draft：
- 从当前 feature 出 top-3 token 候选
- 每个候选再 expand 出 top-2
- ... 形成树
- runtime 据置信度动态裁剪

```
       root (h_t)
      /  |  \
    A    B   C        depth 1
   / |   |   | \
  Aa Ab Ba  Ca Cb     depth 2 (置信度低剪掉)
   ...
```

## 3 · 关键创新：confidence-based pruning
- 每条路径维护累积 logprob
- 全树只保留 top-K 条路径（K=32）
- 不浪费 verify 在低概率上

## 4 · 收益（Llama-7B / Vicuna-7B）
| 方法 | 加速 |
|------|------|
| EAGLE-1 | 3.0x |
| **EAGLE-2** | **4.0-4.5x** |

## 5 · tree attention verify
- 一次 target forward 算所有 32 路径的 logits
- 用 tree-attention mask（每个 query 只看自己路径祖先）
- 一次 forward 收 ≈ 5-7 accepted token

## 6 · 与 Medusa tree 区别
- Medusa: 静态笛卡尔积 → 树大且无关
- EAGLE-2: 动态 expand by confidence → 树小但精

## 7 · 实现：[eagle2.py](../src/eagle2.py)
- DraftTree class
- top-K 路径选择
- tree mask 构造
