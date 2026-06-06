# L06 — Roofline 模型实战

## 定义

`achievable_FLOPS = min(peak_FLOPS, BW * arithmetic_intensity)`

- `arithmetic_intensity = ops / bytes_moved`
- ridge point = `peak_FLOPS / BW`
- 一切优化要么提 `ai`，要么提 `BW`，要么提 `peak`

## LLM 常见 op 的 ai (H100 BF16, ridge ≈ 295)

| Op | ai 估算 | 结论 |
|----|--------|------|
| GEMV (1×4096×4096) | ~1 | memory bound (util 极低) |
| 大 GEMM (8k×8k×8k) | ~2700 | compute bound (util 100%) |
| Attention (s=2048, d=128) | ~250 | 边界 (FlashAttn fuse 后 ai 涨) |
| Attention (s=32k) | ~3500 | compute bound (KV cache 巨大) |
| LayerNorm | ~2 | 极端 memory bound |
| GEMM (m=128) | ~250 | 接近 ridge (batch 太小) |

## 工程指导

- 推理小 batch → 几乎所有 op 都 memory bound → **量化** (FP4/INT4) 是 ai 不变的同时把 bytes 减半，等价提升 ai 2×
- 训练大 batch → GEMM compute bound → 上 FP8/FP4 + 大 tile
- LayerNorm / Softmax / Residual → fuse 进相邻 op 节流量
