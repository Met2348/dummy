# L04 — Shared Memory 与 Bank Conflict

## 模型

- 32 个 4-byte bank，循环映射：byte_offset → `(offset / 4) % 32`
- 同 warp 内多 lane 访同 bank **不同字** → N-way conflict，串行 N 次
- 同 lane 访同字 (broadcast) → 不冲突

## 经典反例：stride 32

```cuda
__shared__ float s[32 * 32];
float v = s[threadIdx.x * 32];   // lane i 访 bank (i*32 % 32) = 0 → 31-way conflict
```

## 解法 1：padding

```cuda
__shared__ float s[32 * 33];     // +1 列
float v = s[threadIdx.x * 33];   // 错位掉
```

## 解法 2：swizzle

H100 的 wgmma 期望 SMEM 是 swizzled 排布 (32B / 64B / 128B swizzle pattern)，CUTLASS / Triton 自动生成。

## 解法 3：warp shuffle

能用 shuffle 就不用 SMEM。但跨 warp 必须 SMEM。
