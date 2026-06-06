# L06 — Capstone: 24h Cluster 调度仿真

## 配置

- 64 节点 × 8 GPU = 512 GPU
- 8 个混合 priority 训练 job
- 模型 FIFO + backfill 调度 + Young's formula ckpt

## 结果

```
8/8 jobs scheduled
util 0.69 (69%)
MTBF 16.71 h
T_ckpt 347 s
```

## 教学结论

- 真实集群 util 60-80% 是健康范围 (零碎窗口 + maintenance)
- 8 个 job 中只要总 GPU < 512，FIFO+backfill 几乎都能塞下
- 大 cluster MTBF 趋于 hours 级 → ckpt 频率必须从小时降到分钟
- 512 GPU 集群，每 ~6 分钟一次 ckpt (T_opt 347s) 是最优经济点

## 工程 takeaway

1. 监控 `util` + `time_to_first_alloc` + `failure_rate` 三指标
2. 不同 priority 的 QoS 分桶 (production/dev/spot)
3. ckpt 频率随 cluster 大小自动调
4. 用 elastic + spot 节省 cost (Together AI / Lambda Labs)
