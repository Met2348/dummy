# L03 — Warp 级原语

## 核心 API (sm_70+)

| 原语 | 用途 |
|-----|------|
| `__shfl_sync(mask, val, src)` | 任意 lane 间交换 |
| `__shfl_up_sync(mask, val, delta)` | 上邻取 (lane i ← i-delta) |
| `__shfl_down_sync(mask, val, delta)` | 下邻取 (lane i ← i+delta) |
| `__shfl_xor_sync(mask, val, lane_mask)` | 蝶式交换 |
| `__ballot_sync(mask, pred)` | 32-bit mask of who voted true |
| `__any_sync` / `__all_sync` | 全 warp 投票 |

## Tree reduce via shuffle (典范)

```cuda
__device__ float warp_reduce(float v) {
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    return v;       // lane 0 holds result
}
```

5 步完成 32 元素求和，**完全不用 SMEM**。

## 工程价值

- 替代 SMEM-based reduce → 省 SMEM、避免 bank conflict
- 跨 warp 仍需 SMEM 中转 (block-level reduce = warp-reduce + 1 atomic / SMEM-stage + warp-reduce)
- 是 Triton `tl.sum(..., axis=0)` 的底层实现
