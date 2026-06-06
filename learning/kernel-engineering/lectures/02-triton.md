# L02 — Triton 实战

## 最小 GEMM kernel

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K, ...,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :])
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
```

## 关键 idiom

- `tl.constexpr` — 编译期常量，影响生成代码
- `num_stages=k` — software pipeline 深度 → 多缓冲 cp.async
- `tl.dot(...)` — 触发 wgmma / mma
- `tl.load(..., mask=...)` — 边界 mask

## Autotune 注意

- key 决定缓存命中：`['M', 'N', 'K']` 即每个新 shape 触发一次扫描
- 第一次 tune 慢 (几秒)，之后命中 → 0 overhead
- shape 频繁变化 (动态 batch) → autotune 反而成瓶颈，需 lock-in 一个 config
