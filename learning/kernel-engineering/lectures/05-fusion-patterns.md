# L05 — 常见 Fusion 模式

## 三类 fusion

| 类 | 例 | 收益 |
|----|----|-----|
| **Pointwise → pointwise** | gelu(W@x) → 不写中间 h | 省 1 个 N×D HBM roundtrip |
| **Pointwise → matmul** | RMSNorm + W → fused_norm_linear | 省 1 个 N×D HBM roundtrip |
| **Matmul → matmul** | A@B@C → 大 GEMM with epilogue | 通常不 fuse (中间维度太大)，例外 SwiGLU |

## SwiGLU 三 matmul 融合

```
SwiGLU(x) = (Silu(x @ W_gate) * (x @ W_up)) @ W_down
```

- 经典：3 个 matmul + 2 个 elementwise → 5 个 kernel
- 融合：FA3 风格 "matmul + elementwise + matmul" 在同 kernel 内 → 2 个 kernel

## 反例：不要乱 fuse

- 中间 buffer 太大撑爆 SMEM → 反而 register spill → 慢
- Fusion 阻碍并行 (一个长 kernel 占 SM 不放，调度差)
- LayerNorm + Linear fuse 时 mean/var 跨 column 需要全 reduce → 需要 tile-wide 同步

## 工程指标

- nsight compute "Memory Throughput" + "Compute Throughput" 同时高 → 健康
- 一个低一个高 → 还有 fusion 空间
