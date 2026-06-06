# L02 — Mini-Cluster Simulator (Capstone-1)

## 设计

```
ClusterBlueprint(n_nodes, gpus_per_node, gpu, fabric, storage)
   ↓
total_flops(model) → pure_compute_s
   + step_comm_s × n_steps → raw_comm_s (informational)
   × overhead_factor (1.25)
   → wall_days
   ↓
total_cost_3y → capex + opex × 3
```

## 18 个场景结果

| Model    | Cluster              | Days | TCO 3y ($M) | Power (kW) |
|----------|----------------------|-----:|------------:|-----------:|
| 8B-1T    | 8x H100 + IB NDR     |  220 |         0.6 |        9.5 |
| 8B-1T    | 64x H100 + IB NDR    |   27 |         2.4 |       75.7 |
| 8B-1T    | 512x H100 + IB NDR   |    3 |        18.1 |      605.7 |
| 70B-5T   | 4096x H100 + IB XDR  |  120 |       136.4 |     4845.3 |
| 70B-5T   | 4096x B200 + IB XDR  |   53 |       169.4 |     6885.8 |
| 405B-10T | 4096x H100 + IB XDR  |  217 |       136.4 |     4845.3 |
| 405B-10T | 4096x B200 + IB XDR  |   95 |       169.4 |     6885.8 |

## 教学结论

- 8B-1T 在 512 H100 上 3 天 → 可在大集群快速复现
- 70B-5T 在 4096 H100 上 120 天 → 接近 Llama-3 实际规模
- B200 vs H100：2-3× speedup，但 power +40%、capex +25%
- TCO 主要被 capex (~80%) 主导，opex 是月级电费
