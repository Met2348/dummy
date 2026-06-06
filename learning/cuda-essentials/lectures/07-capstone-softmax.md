# L07 — Capstone：Online Softmax

## 问题

Standard softmax 是 3-pass：

```
m = max(x)            # pass 1
e = exp(x - m)        # pass 2
out = e / sum(e)      # pass 3
```

2 次 HBM 读写 → memory bound 主因。

## Online (1-pass) 算法 (Milakov & Gimelshein 2018, FlashAttention 内核)

维护 invariant：
- `m_i = max(x[0..i])`
- `d_i = sum_{j<=i} exp(x[j] - m_i)`

更新规则：

```
m_new = max(m_i, x[i+1])
d_new = d_i * exp(m_i - m_new) + exp(x[i+1] - m_new)
```

`exp(m_i - m_new)` 是"过去 sum 的 rescale 因子"。

## 为什么重要

- **FlashAttention 的核心**：softmax(QK^T) 可与 attention 输出 fuse 同一 kernel
- Online 算法的存在让 attention 从 O(N²) HBM 流量 → O(N) HBM 流量
- 2023 后所有 attention kernel (FlashAttn-2/3, FlashInfer, FlashMLA) 都基于这个范式

## Capstone 退出

```powershell
python learning/cuda-essentials/src/capstone_softmax.py
# expect: [OK] capstone_softmax (online == naive)
```

数值精度：对大值 (1e3+) 仍稳定。
