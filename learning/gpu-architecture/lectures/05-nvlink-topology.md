# L05 — NVLink / NVSwitch 拓扑

## 4 代 NVLink

| Gen | 单链 | 每 GPU | 形状 | 出场 |
|----|-----|--------|------|------|
| 2 | 25 GB/s | 300 GB/s | P100 ring | 2017 |
| 3 | 25 GB/s | 600 GB/s | A100 + NVSwitch | 2020 |
| 4 | 50 GB/s | 900 GB/s | H100 + NVSwitch | 2022 |
| 5 | 100 GB/s | 1800 GB/s | B200 + NVSwitch | 2024 |

## NVSwitch 进化

- A100 6 NVSwitch × 8 GPU = full mesh 4.8 TB/s bisection
- H100 4 NVSwitch × 8 GPU = full mesh 7.2 TB/s bisection
- **GB200 NVL72**: 1 个 rack 72 GPU 全互连，130 TB/s bisection — 当作"一个超级 GPU"用

## Ring AllReduce 公式

`time = 2*(N-1)/N * size / per_gpu_BW`

8-GPU H100 1 GB allreduce ≈ 1.94 ms。72-GPU GB200 1 GB allreduce ≈ 1.10 ms (尽管 N 大 9×，BW 也 2×，且 ring 系数趋近 2)。

## PCIe 退化

PCIe Gen5 x16 ≈ 64 GB/s 单向 = 0.128 TB/s 双向 vs NVLink5 1.8 TB/s → **14× 差距**。所以训练集群基本不用 PCIe topo，只在推理小集群里见。
