# L09 · KV Cache 跨节点传输

## 1 · 三种传输方式
| 方式 | 速度 | 适用 |
|------|-----|-----|
| **NCCL P2P** | 700 GB/s (NVLink), 50 GB/s (PCIe) | 同节点 |
| **NVSHMEM** | 同 NCCL，更细粒度 (one-sided) | 高级用法 |
| **RDMA over IB** | 50 GB/s (400 Gb/s IB) | 跨节点 |
| **TCP/IP** | 1 GB/s | 测试 |

## 2 · KV 序列化格式
- 直接 dump tensor 字节
- 量化的 KV (int8/fp8) → 字节数减半
- prefix-cache hit 部分**不传**（已在 decode 节点）

## 3 · 传输策略
- **layer-by-layer**: 一层一层发，与 decode forward overlap
- **block-by-block**: 配合 PagedAttention，按 block 发
- **streaming**: 边 prefill 边发

## 4 · Mooncake 设计
- 单独 KV cache pool 跨节点共享
- 用 NVMe SSD 做 L2 cache（cold KV）
- 用 DRAM 做 L1 cache
- HBM 是 L0

## 5 · 与 prefix caching 协同
- 命中的 prefix block 不传（已 share）
- 只传 user-specific 部分
- 命中率高时 KV 传输代价大幅降

## 6 · 数据流：streaming KV
```
prefill iter i finishes layer i → send layer i KV → decode 收到后 immediate continue
```

## 7 · 实现：[kv_transfer_mock.py](../src/kv_transfer_mock.py)
- 模拟传输时间 = size / bandwidth
- 不同方式对照
- streaming 重叠 vs 批
