# L06 — Reduce 与 Tiled GEMM

## Reduce 三代

1. **Naive**：1 thread 串行加 — O(N)
2. **Brent-Kung 树**：log2(N) 步，每步活跃 thread 减半
3. **Warp shuffle**：log2(32) = 5 步无 SMEM；跨 warp 用 1 个 SMEM stage

## Tiled GEMM (`Volkov 2008` 经典)

```cuda
__shared__ float As[TILE][TILE], Bs[TILE][TILE];
for (int kk = 0; kk < K; kk += TILE) {
    As[ty][tx] = A[row * K + kk + tx];
    Bs[ty][tx] = B[(kk + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < TILE; k++)
        sum += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
```

## HBM 流量分析

- Naive：每个 (i,j) 重读 K 个 A + K 个 B = 2MNK 字节
- Tiled：每个 (ii,jj,kk) tile 读 1 次 → 2 * M * N * K / TILE 字节
- TILE = 32 → 32× HBM 流量减少

## 进阶

- **Cooperative Groups**：grid-level barrier
- **Async copy** (`cp.async`)：H100 重叠 HBM→SMEM 与计算
- **wgmma**：H100 异步 MMA + 双 buffer，最大化 tensor core 利用
