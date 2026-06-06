# L02 — Hello World：Vector Add

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) c[gid] = a[gid] + b[gid];
}

// host
cudaMalloc(...); cudaMemcpy(H2D); 
vector_add<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
cudaMemcpy(D2H); cudaFree(...);
```

## 三个习惯

1. ceil-div：`(n + bs - 1) / bs` — 防止漏 tail
2. 边界检查：`if (gid < n)` — block 数向上取整必然有多余 thread
3. cudaCheck：每个 API 调用包 macro，否则错误吞掉

## 性能特征

- vector_add 是 memory bound (ai = 1)，util 100% 时 ≤ 0.5% peak FLOPS
- 这是为什么 nvprof 报 vector_add "low SM efficiency" 不是 bug — 它本来就是 HBM-bound
