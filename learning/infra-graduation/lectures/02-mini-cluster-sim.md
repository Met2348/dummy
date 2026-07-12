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

## 18 个场景结果（`python learning/infra-graduation/src/sim/capstone_1.py` 实测 stdout，7 行精选）

| Model    | Cluster              |  Days | TCO 3y ($M) | Power (kW) |
|----------|----------------------|------:|------------:|-----------:|
| 8B-1T    | 8x H100 + IB NDR     | 219.4 |         1.3 |        9.5 |
| 8B-1T    | 64x H100 + IB NDR    |  27.4 |         3.1 |       75.7 |
| 8B-1T    | 512x H100 + IB NDR   |   3.4 |        18.1 |      605.7 |
| 70B-5T   | 4096x H100 + IB XDR  |  18.7 |       137.6 |     4845.6 |
| 70B-5T   | 4096x B200 + IB XDR  |   8.2 |       184.1 |     6922.2 |
| 405B-10T | 4096x H100 + IB XDR  | 217.0 |       137.6 |     4845.6 |
| 405B-10T | 4096x B200 + IB XDR  |  95.4 |       184.1 |     6922.2 |

> 早期版本这张表的数字是撰写时手估的草稿值，跑 runbook 验证时和脚本实测 stdout 逐字比对后发现有出入
> （尤其 70B-5T 那两行差出 6 倍以上）——已用真实 stdout 整表替换。18 场景全量输出见脚本直跑；这里维持
> 原先"7 行精选"的教学取舍不变。

## 教学结论

- 8B-1T 在 512 H100 上 3.4 天 → 可在大集群快速复现
- 70B-5T 在 4096 H100 上仅需 18.7 天，同一 4096x H100 集群跑 405B-10T 要 217.0 天 → 集群规模固定时，
  time-to-train 随 `n_params × n_tokens` 近线性变化（(70×5000)/(405×10000)=0.086 ≈ 18.7/217.0=0.086，
  两者吻合）
- B200 vs H100：3 个场景实测 speedup 均落在 2.27-2.28×（8B-1T/512 节点 2.27×、70B-5T/4096 节点
  2.28×、405B-10T/4096 节点 2.27×），与 Capstone-2（`eval/mlperf_mock.py`）5 个 MLPerf 任务独立算出的
  2.28× 平均加速吻合；同时 power +43%、TCO +32~34%（4096 节点档：power 4845.6→6922.2kW +42.9%，
  TCO 137.6M→184.1M +33.8%）
- TCO 主要被 capex（GPU+fabric+storage）主导，占比 ~90-91%（非早期估计的 ~80%），3 年 opex 仅占
  ~9%——GPU 硬件采购成本远超电费
