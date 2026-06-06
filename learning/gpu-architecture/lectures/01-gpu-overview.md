# L01 — GPU 体系结构总览

## 为什么 GPU 适合 LLM

- 数千核心 + 高带宽 HBM = 把"矩阵乘法 + 注意力"打满
- CPU 8-128 核 → GPU H100 132 SM × 128 ALU = 16k+ 并行流
- HBM3 3.35 TB/s vs DDR5 0.05 TB/s → 67×

## 4 代旗舰对照 (2026-06 口径)

| | A100 80G | H100 SXM | H200 SXM | B200 |
|--|---------:|---------:|---------:|-----:|
| BF16 TFLOPS | 312 | 989 | 989 | 2250 |
| FP8 TFLOPS | — | 1979 | 1979 | 4500 |
| FP4 TFLOPS | — | — | — | 9000 |
| HBM | 80 GB | 80 GB | 141 GB | 192 GB |
| HBM BW | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s |
| NVLink | 0.6 TB/s | 0.9 TB/s | 0.9 TB/s | 1.8 TB/s |
| TDP | 400 W | 700 W | 700 W | 1000 W |

## 关键观察

- 算力翻倍速度 (3×/代) > 带宽翻倍速度 (1.5×/代) → ridge point 上移 → 更多 op 变成 memory bound
- Blackwell 引入 FP4 (E2M1) → 4× FP8 算力，但需要 microscaling block (MX-FP4)
- HBM 容量 80 → 192 GB → 单卡能装下 70B BF16 完整权重 + 一定 KV cache

## Roofline 直觉

`achievable_TFLOPS = min(peak_TFLOPS, BW * arithmetic_intensity)`

- ai < ridge → memory bound (优化方向：fuse / 减少 HBM 流量)
- ai > ridge → compute bound (优化方向：用更低精度 / 更大 tile)
- H100 BF16 ridge = 989 / 3.35 ≈ 295 FLOP/byte
- 大 GEMM (m=n=k=4096) ai ≈ 2700 → compute bound ✓
- GEMV (m=1) ai ≈ 1 → memory bound ✗
- LayerNorm ai ≈ 2 → 极端 memory bound (Flash 系列工作主要解决这类)
