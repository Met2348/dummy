# L04 — FlashAttention 三代演进

## 核心思想

不显式生成 N×N attention 矩阵。outer loop on Q-blocks, inner loop on K/V-blocks, online softmax 累加。

## 算法 (FA-2)

```
for i in 0..N step Br:
    O_i, m_i, l_i = 0, -inf, 0
    Q_i = Q[i:i+Br]
    for j in 0..N step Bc:
        K_j, V_j = K[j:j+Bc], V[j:j+Bc]
        S_ij = Q_i @ K_j^T / sqrt(d)
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
        m_i, l_i = m_new, l_new
    O_i = O_i / l_i
    write O_i
```

## HBM 流量

- Naive: O(N²) (S 和 P 矩阵)
- Flash: O(N×d) (只读 Q/K/V，写 O)
- N=128k, d=128 → 1025× HBM 节省 (与本 capstone 测算一致)

## 三代差异

| | FA-1 (2022) | FA-2 (2023) | FA-3 (2024) |
|--|------------|-------------|-------------|
| 并行化 | head | head + batch + seq | + Hopper async wgmma |
| Backward | 同 forward | 重排 dQ/dK/dV loop | + FP8 backward |
| 峰值 H100 | ~50% | ~70% | **~75%** |

## 工程影响

- 长序列训练成为可能 (32k → 128k → 1M)
- KV cache 仍是 O(N) 容量瓶颈 → PagedAttention / MLA 接力
