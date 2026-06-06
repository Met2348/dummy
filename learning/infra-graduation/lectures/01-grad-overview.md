# L01 — Infra Graduation 总览

## 收官 7 件事

1. **Roofline 心电图** (Topic 1) — 给定 GPU + op，秒回 memory/compute bound
2. **CUDA 写 kernel** (Topic 2) — vector add / reduce / tiled GEMM / online softmax
3. **库选型** (Topic 3) — Triton / CUTLASS / FlashAttn 三家分工
4. **网络协议** (Topic 4) — NVLink / IB / SHARP / NCCL all-reduce
5. **存储管道** (Topic 5) — Lustre + WebDataset + DCP-async
6. **集群编排** (Topic 6) — Slurm + Ray + Young's formula + elastic
7. **总收尾** (本 Topic) — mini-cluster simulator + topology selector + Portfolio v3

## Module 8 在整个学习地图的位置

```
M3 造模型  ← M8 教你它跑在什么硬件上
M4 改模型  ← M8 教你训练集群怎么调
M5 用模型  ← M8 是 vLLM 之下的硬件层
M6 评模型  ← MLPerf 是 infra benchmark
M7 造 agent ← agent 的 inference 也跑 M8 硬件
```

> **M8 是其他所有 module 的物理基础**。学完这层，整个 LLM 全栈通了。
