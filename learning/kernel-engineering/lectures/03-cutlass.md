# L03 — CUTLASS / CuTe Layout 代数

## Layout = (shape, stride)

- `Layout((4, 8), (8, 1))` = row-major 4×8
- `Layout((4, 8), (1, 4))` = col-major 4×8
- 任意嵌套：`Layout((4, (2, 4)), (8, (4, 1)))` 表"4 行，每行 8 元素分两组"

## 重要性质

- 任意位置 idx (i, j, k, ...) → linear offset = Σ idx_d × stride_d
- swizzle = stride 用 XOR 变换 → 同 row 不同 col 落不同 bank → 0 conflict

## 经典模式

```cpp
using SmemLayoutA = decltype(composition(
    Swizzle<3, 3, 3>{},                     // 8B × 8B × 8B
    Layout<Shape<_128, _32>>{}));           // BLOCK_M × BLOCK_K
```

`Swizzle<3,3,3>` 意思：高 3 位选 bank-group，低 3 位 XOR mask 重排。

## CUTLASS 3.x 三角架构

```
Collective Mainloop  → loads A/B tiles, MMA
Collective Epilogue  → bias, activation, output stage
TileScheduler        → grid 大小 / stream-K decomposition
```

各自独立可换 → CUTLASS recipes = 三者组合。
