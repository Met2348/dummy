# L07 · All-to-All 通信

## 1 · all-reduce vs all-to-all
| op | semantics | 用途 |
|----|----------|------|
| all-reduce | sum/avg 跨 rank | TP |
| all-gather | concat 跨 rank | activation 收集 |
| reduce-scatter | sum then split | TP 反向 |
| **all-to-all** | rank i send x[i,j] to j | **EP** ⭐ |

## 2 · all-to-all 数据
ranks×ranks 矩阵传输：
- 输入：每 rank 持 [N, ...] 数据
- 输出：每 rank 得 [N, ...] 数据 from all other ranks
- 等价于 N×N 个 P2P send

## 3 · NCCL 实现
- ring all-to-all（高带宽）
- pairwise (低带宽，但简单)
- 跨节点用 RDMA + NVSHMEM

## 4 · 带宽分析
- N rank，每 rank 发 S byte → 总 S·N²
- ring 实现：每 rank 双向 S·(N-1)/N
- NVSwitch (all-to-all native) 比 ring 快 2x

## 5 · 与 KV cache 关系
disaggregated P/D 时：
- prefill GPU 生成 KV
- decode GPU 来取 → P2P send (或 all-to-all if batch)
- 用 NCCL/NVSHMEM

## 6 · 通信库
| 库 | 用途 |
|---|-----|
| **NCCL** | NVIDIA 主流，PyTorch/vLLM 用 |
| **NVSHMEM** | one-sided RDMA |
| **MSCCL** | Microsoft 优化 NCCL |
| **MPI** | 老牌，跨厂商 |

## 7 · 实现：扩展 ep_demo
模拟单进程 all-to-all 时间，对比 ring vs allreduce。
