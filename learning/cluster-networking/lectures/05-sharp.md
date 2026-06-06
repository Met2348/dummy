# L05 — SHARP 与 in-network 计算

## SHARP (Scalable Hierarchical Aggregation Reduction Protocol)

- Mellanox/Nvidia 自研，IB switch 内嵌 ALU 单元
- All-reduce 用 2 steps：endpoint → switch → endpoint
- 不再随 N 增加 hop 数 → 大集群"福音"

## 数学

```
T_SHARP = 2 * (link_lat + (size/N) / BW)
```

64 GPU 100 MB 在 IB NDR：
- ring: ~30 ms
- SHARP: ~0.5 ms → **60× 加速**

## 实战门槛

- Quantum-2 switch (NDR 时代)
- ConnectX-7 NIC
- HPC-X 软件栈 + NCCL_COLLNET_ENABLE=1
- 大模型训练几乎必须 (Meta / 阿里 / 字节 自建 IB 集群都开)

## 2024-2025 进展

- SHARP v3 支持非二次幂 N
- Ultra-Ethernet Consortium 也在做以太网版 in-network reduce
- Microsoft Azure 自研 RDMA over InfiniBand + 自家 in-network 收敛
