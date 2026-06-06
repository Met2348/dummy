# L03 — Topology Selector

## 输入

```python
select(model: ModelSpec, budget_usd: float, max_days: float)
   → ClusterBlueprint | None
```

## 算法

1. 枚举 candidates (3 GPU × 4 n_nodes × 2 fabric × 2 storage = 48)
2. 过滤 budget (`tco_3y_m * 1e6 > budget` 剔除)
3. 过滤 time (`wall_days > max_days` 剔除)
4. 排序 by (TCO ↑, time ↑, blueprint id)
5. 返回最便宜的

## 工程用法

```python
# CTO 视角："7B 模型，500B token，预算 $20M，60 天内"
pick = select(ModelSpec("demo-7B", 7, 500), budget_usd=20e6, max_days=60)
# → 128x H100 + IB XDR
```

## 拓展

- 加 budget per day → 选 spot vs reserved
- 加 power_kw 约束 → 避开机房限电
- 加 fairness → 多任务共享时调度

## 真实工程产品

- Together AI Cluster Planner
- CoreWeave Provisioner
- Lambda Labs CLI
- 这些都是 topology selector + capacity planner 的商业化
