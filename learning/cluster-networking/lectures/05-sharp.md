# L05 — SHARP 与 in-network 计算

## SHARP (Scalable Hierarchical Aggregation Reduction Protocol)

- Mellanox/Nvidia 自研，IB switch 内嵌 ALU 单元
- All-reduce 用 2 steps：endpoint → switch → endpoint
- 不再随 N 增加 hop 数 → 大集群"福音"

## 数学

```
T_SHARP = 2 * (link_lat + (size/N) / BW)
```

64 GPU 100 MB 在 IB NDR（`python learning/cluster-networking/src/sharp_inline.py` 实测，`_self_test()` 内置的同一场景）：
- ring: ~4.13 ms
- SHARP: ~0.066 ms → **63× 加速**（512-GPU 同法算出 511×；两者都约等于 `n_gpus-1`，因为 ring 步数
  `2(N-1)` 随 N 线性增长而 SHARP 恒为 2 步，大消息下带宽项主导，时间比收敛到步数比）

## 实战门槛

- Quantum-2 switch (NDR 时代)
- ConnectX-7 NIC
- HPC-X 软件栈 + NCCL_COLLNET_ENABLE=1
- 大模型训练几乎必须 (Meta / 阿里 / 字节 自建 IB 集群都开)

## 2024-2025 进展

- SHARP v3 支持非二次幂 N
- Ultra-Ethernet Consortium 也在做以太网版 in-network reduce
- Microsoft Azure 自研 RDMA over InfiniBand + 自家 in-network 收敛
