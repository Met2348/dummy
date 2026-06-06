# L01 — CUDA 执行模型

## 三层并行

```
Grid  → 一次 kernel launch
  Block → 同一 SM，共享 SMEM，可同步
    Warp → 32 个 lane lock-step (SIMT)
      Thread → 一个 lane
```

## 启动语法

```cuda
dim3 block(256, 1, 1);
dim3 grid((N + 255) / 256, 1, 1);
my_kernel<<<grid, block>>>(args...);
```

## 全局线程 id 公式 (1D)

```
gid = blockIdx.x * blockDim.x + threadIdx.x
```

边界检查：`if (gid < N) { ... }` 不可省，否则越界。

## Warp 是真正的执行单位

- 32 个 thread 同步执行同一 PC
- 分歧 (if/else) → 一条 path 一条 path 串行执行 (warp divergence)
- 一个 block 256 thread = 8 warp，warp scheduler 在它们之间 hide latency
