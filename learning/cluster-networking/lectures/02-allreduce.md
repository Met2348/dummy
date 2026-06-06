# L02 — All-reduce 算法家族

## Ring (Patarasuk-Yuan 2009)

```
T = 2*(N-1)/N * size / BW
```

- BW-optimal (每 GPU 用满带宽)
- 步数 2(N-1)：随 N 线性 → 大 N 延迟上升
- NCCL 默认 ≤ 8 GPU + NVLink

## Tree (recursive halving)

```
T = 2 * log2(N) * (latency + size/BW)
```

- 步数 O(log N) → 小消息友好
- BW 不优 (每 step 全 size)
- NCCL 用于小消息

## Halving-doubling (Rabenseifner)

```
T = 2 * log2(N) * (latency + (size/N) / BW)
```

- O(log N) 步 + BW-optimal 总流量
- 大 N + 大消息最优
- NCCL 在中等 N 上用

## SHARP (in-network reduction)

- Mellanox/Nvidia IB switch 内嵌 ALU，对 partial sum 在 switch 上做加法
- 步数 = 2 (constant!)，与 N 无关
- 仅对支持 SHARP 的 fabric (Quantum-2 + ConnectX-7 + 兼容驱动)
- 大 N 上加速 10-100×

## NCCL 启发式选择

```
size < 1 KB       → tree
size < 1 MB       → ring
N ≤ 8 + NVLink   → ring
N > 8 + IB + 大size → halving_doubling (or SHARP if available)
```
