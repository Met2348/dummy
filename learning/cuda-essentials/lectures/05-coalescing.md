# L05 — 全局内存 Coalescing

## 模型

GMEM 以 128B sector 为单位发请求。32 lane × 4 byte = 128 B 完美对齐 → 1 sector → 100% 利用。

## 反例对比

| Pattern | sectors | eff |
|---------|--------:|----:|
| stride 1 (相邻 word) | 1 | 100% |
| stride 2 (跳一个) | 2 | 50% |
| stride 8 | 8 | 12.5% |
| stride 32 | 32 | **3.1%** |

## AoS vs SoA

- AoS (Array of Structs)：`struct {x,y,z}; arr[i]` → lane i 访 12 字节远的 struct → 跨 sector
- SoA (Struct of Arrays)：`x[]`, `y[]`, `z[]` 分开 → 各自连续 → 完美 coalesce
- LLM 张量都是 SoA (一个 [B, S, D] tensor 内存连续)，所以默认 OK

## 调试

`ncu --metrics l1tex__t_sector_pipe_lsu_mem_global_op_ld.sum` 看 sector 数。
