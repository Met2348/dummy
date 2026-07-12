# L06 — Capstone: 7 天训练 ckpt 经济学

## 任务

70B 模型 / 512 GPU / 7 天训练，每小时 ckpt + 24h MTBF，对比 3 种 ckpt 策略。

## 结果

```
Strategy | per-ckpt | blocking (min) | recovery (h) | wasted %
---------|----------|----------------|--------------|---------
full     |    0.70s |           1.96 |         3.50 |    2.10
sharded  |    0.35s |           0.98 |         3.50 |    2.09
async    |   0.009s |           0.00 |         3.50 |    2.08
```

（以上为 `capstone_ckpt_recovery.py` 实测 stdout；async 的 blocking(min) **精确等于 0.00**，不是"很小的非零数"——`async_sharded()` 把 `blocking=False`，`total_overhead()` 里 `blocking_overhead_s = (cost.sec if cost.blocking else 0.0) * n_ckpts` 在 `blocking=False` 时直接短路成 0，与 `cost.sec`（0.009s 的 PCIe 暂存时间）本身大小无关——这正是 async 设计的核心卖点：暂存阶段完全不占用训练关键路径。）

## 教学结论

- 在合理 ckpt_interval (1h) 下，三种策略 wasted% 差异不大 (~2%)
- 大头是**故障恢复重做训练** (3.5h)，不是 ckpt 本身——且三种策略的 recovery(h) 几乎相同（都要重新加载
  ckpt + 补跑半个 interval），async 省的只是**训练期间的 blocking**，不能缩短故障后的恢复时间
- 但若 ckpt_interval 缩到 10 min，full ckpt blocking 时间 6× 增加 → async 优势凸显
- 真正的差距体现在大集群 + 短 interval + 频繁故障的极端场景

## 关键 takeaway

1. ckpt 频率 = sqrt(2 × ckpt_cost × MTBF)
2. async DCP (PyTorch 2.4+) 是 default 选择
3. 多 tier 备份：Lustre (热) + S3 (冷)
4. 测试恢复路径！很多团队只测 ckpt 不测恢复
