# L02 — GPU 存储层次

## 5 层金字塔 (H100)

| 层 | 容量 | 延迟 | 带宽 | 关键事实 |
|---|------|------|------|---------|
| Registers | 256 KB/SM (65k×32bit) | 1 cycle | ~2 TB/s | 编译期分配，超量 → spill 到 L1 |
| Shared Memory | ≤228 KB/SM | ~30 cycle | 228 TB/s | 程序员手动管，bank conflict 致命 |
| L1 cache | 与 SMEM 共片 | ~30 cycle | 128 TB/s | 自动 |
| L2 cache | 60 MB/GPU | ~200 cycle | 12 TB/s | 跨 SM 共享 |
| HBM3 | 80 GB | ~600 cycle | 3.35 TB/s | DRAM，远 |

## 工程意义

- **Tile 选择**：分块大到能填满 SMEM，但不能超 (228KB-α) 否则占用率掉
- **Bank conflict**：32 个 bank × 4 byte，相邻线程访问相邻 4 字节零冲突
- **L2 持久化**：H100 引入 L2 cache resident control (cudaAccessProperty)
- **HBM stall**：~600 cycle 等待 → 必须用 warp scheduler hide latency

## 典型反例

LayerNorm 写一次 → 读一次 next op，复用 = 0 → 永远卡 HBM 带宽。解决：fuse 进 attention 输出/residual。
