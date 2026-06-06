# L05 — TCO (Total Cost of Ownership)

## 三大成本

| 类 | 占比 | 公式 |
|---|-----|-----|
| GPU capex | 70-80% | n_gpus × price |
| Fabric capex | 2-5% | n_nodes × switch_port_cost |
| Storage capex | 1-3% | cap_pb × $50k |
| Power opex (3y) | 10-20% | tdp × pue × $0.10/kWh × 8760 × 3 |
| Floor space + cooling | 5% | data-center fee |

## 真实数字 (2026)

| Asset | Capex |
|-------|------:|
| H100 80GB SXM5 | $30k |
| H200 141GB | $35k |
| B200 192GB | $40k |
| GB200 NVL72 unit (72 GPU) | ~$3M |
| IB NDR switch (64 port) | ~$50k |
| Lustre 1 PB (NVMe + OSS) | ~$50-100k |

## 决策框架

- **Buy or rent**：自训 > 6 个月 → 自建有 ROI；< 6 个月 → 租 Together/CoreWeave
- **H100 vs B200**：B200 单位 token 成本低 2× (FP4 推理)，但单位 token 训练成本高 1.3×
- **NVL72 vs DGX**：NVL72 每 token cost 低 30%，但前期 $3M 跳跃门槛

## 反例

- 买 1000x H100 训 10B 小模型 → 大马拉小车
- 用 PCIe-only 8 GPU 训 70B → 永远训不完 (1 step AR 几秒)
- 不算电费就投资 5MW data center → 3 年电费 = 0.5×capex
