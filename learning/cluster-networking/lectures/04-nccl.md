# L04 — NCCL collectives

## 5 个核心 op

| op | 公式 | 用途 |
|----|-----|------|
| `ncclAllReduce` | RS + AG | data parallel 梯度同步 |
| `ncclReduceScatter` | (N-1)/N | ZeRO 优化器分片 |
| `ncclAllGather` | (N-1)/N | EP 输出 / TP 列并行 |
| `ncclBroadcast` | log(N) | 初始权重广播 |
| `ncclAllToAll` | (N-1) hop | MoE 分发 / pipeline schedule |

## 关键关系

```
AllReduce_cost ≈ ReduceScatter_cost + AllGather_cost
```

ZeRO-3 把这两步拆开放在 forward / backward 不同 phase，**总通信量不变**但延迟可隐藏。

## 实战

```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
dist.all_reduce(grad, op=dist.ReduceOp.SUM)
```

环境变量调优：
- `NCCL_ALGO=Ring` / `Tree` / `CollnetDirect`
- `NCCL_PROTO=Simple` / `LL` / `LL128`
- `NCCL_DEBUG=INFO` 看实际选了什么

## 注意

- 跨 PG 的 collective 是非阻塞的，可与 compute overlap
- All-to-all 在 MoE EP 中是瓶颈 → DeepSeek-V3 用 dual-pipe + 选择性 fuse
