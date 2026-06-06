# L01 — Fabric 总览

## 三层网络

| 层 | 介质 | 典型 BW (per port) | 典型延迟 | 范围 |
|---|------|-------|----------|------|
| Intra-node (GPU↔GPU) | NVLink 5 | 900 GB/s | 0.4 μs | 1 box |
| Intra-rack | NVLink Switch (NVL72) | 1.8 TB/s | <1 μs | 1 rack |
| Inter-node (Pod) | IB NDR/XDR | 400/800 Gbps | 1-2 μs | 1k+ nodes |
| Inter-DC | Eth 100/400G | — | ms | 跨 AZ |

## 4 类协议

- **NVLink + NVSwitch**：GPU 互连专用，2018+ DGX 标配
- **InfiniBand**：HPC 老兵，NDR (400G) / XDR (800G) 当前主流
- **RoCEv2**：以太网上跑 RDMA，AWS EFA / 阿里 HPN 用
- **Ethernet (CXL/UltraEthernet)**：2025+ UEC consortium 推 800G/1.6T

## 选型

- 单节点训练：NVLink/NVSwitch (无替代品)
- 多节点训练：IB NDR + NCCL + SHARP (Nvidia 全栈)
- 推理集群：RoCEv2/Eth 性价比高 (推理 collectives 少)
- 异构云环境：Eth + RDMA (UltraEthernet 推动)
