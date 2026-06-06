# L04 — SM 占用率

## H100 SM 资源上限

- 2048 threads/SM = 64 warps/SM
- 32 blocks/SM 上限
- 65536 registers/SM (32-bit)
- 228 KB SMEM/SM (可调，与 L1 共片)

## 4 个瓶颈

```
blocks_per_sm = min(
  max_threads / threads_per_block,
  max_warps / warps_per_block,
  max_regs / (regs_per_thread * threads_per_block),
  max_smem / smem_per_block,
  max_blocks
)
occupancy = (blocks * warps_per_block) / max_warps
```

## 经验值

| 情况 | 后果 |
|-----|------|
| regs/thread > 64 | reg bottleneck，occ < 0.5 |
| SMEM/block > 48 KB | block_per_sm 显著掉 |
| threads/block < 64 | 不能填满 warp slot |
| 高 occupancy ≠ 高性能 | 见 Volkov "Better Performance at Lower Occupancy" |

## 关键经验

- 0.25 占用率 + 大量 ILP > 1.0 占用率 + 频繁 stall (常见于 GEMM kernel)
- Triton / CUTLASS 经常**主动降低 occupancy** 换 register file 容量
