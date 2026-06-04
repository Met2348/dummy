# L01 · 分布式推理全图

## 1 · 何时分布式
| 模型 | 单卡 fp16 | 单卡 4bit | 何时切片 |
|------|----------|----------|---------|
| Llama-7B | 14 GB | 3.5 GB | 不需要 |
| Llama-70B | 140 GB | 35 GB | 必须 |
| DeepSeek-V3 671B | 1.3 TB | 335 GB | 必须 + EP |

## 2 · 4 类并行
| 名称 | 切什么 | 通信 |
|------|--------|-----|
| **DP** (Data Parallel) | 不切，复制 | grad allreduce |
| **TP** (Tensor Parallel) | 切层内矩阵 | allreduce per layer |
| **PP** (Pipeline Parallel) | 切层（不同 GPU 不同 layer）| send/recv 跨 stage |
| **EP** (Expert Parallel) | 切 MoE expert | all-to-all |

推理 vs 训练区别：
- 训练用 DP/TP/PP 全套
- 推理大多 TP（小）+ PP（中）+ EP（MoE）

## 3 · 推理特有问题
| 问题 | 解 |
|------|---|
| prefill 长 vs decode 快 → bubble | **Disaggregated Prefill/Decode** |
| TP all-reduce 占 30% 时间 | Megatron-LM SP / CP |
| MoE token routing 跨 GPU | EP all-to-all |
| 跨节点带宽 | NVLink + NVSwitch + RDMA |

## 4 · 5090/H100 拓扑
- 单卡 5090：24GB，PCIe 5.0
- 8 × H100: NVLink 4 (900 GB/s), HBM3 80GB × 8 = 640GB
- 16 × H100 跨节点：InfiniBand 400Gb/s

## 5 · 本专题路线
- L02-L03 TP 深入（Megatron 风格）
- L04-L05 PP + 1F1B schedule
- L06-L07 EP for MoE + all-to-all
- L08-L09 **Disaggregated P/D** + KV 传输
- L10 prefix-aware routing
- L11 多节点部署
- L12 Capstone

## 6 · 一句话
> 分布式推理 = **把模型/计算/数据按对的维度切**，**通信带宽别成为瓶颈**。
