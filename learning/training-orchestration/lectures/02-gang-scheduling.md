# L02 — Gang Scheduling

## 为什么

分布式训练所有 rank 必须同时启动 (NCCL init 需要全员到位)，否则：
- 半数 rank 在 timeout 等其他人 → 浪费 GPU
- 部分 rank 启动后再死 → ckpt 不一致

## 实现

- Slurm: `--exclusive` + `--ntasks-per-node` 严格匹配
- K8s: Coscheduler (Kubernetes SIG-Scheduling 2024)
- Volcano: 默认 gang scheduler，云原生选择
- Ray: `PlacementGroup` + STRICT_PACK

## 反例

```
节点池: 8 个节点 × 8 GPU = 64 GPU
job-A 需要 32 GPU (跨 4 节点)
job-B 已占用 6 节点 × 4 GPU = 24 GPU
剩余: 40 GPU 但分布在 6 节点 × 1-4 GPU + 2 节点 × 8 GPU
job-A 等待 8 节点完整空 → 死等
```

解：drain 让小 job 完成，再启 job-A。

## Starvation 监控

- 期望最大 job 大小 > 集群总量 → starvation
- 监控指标：`waiting_jobs_over_quorum` / `time_to_first_alloc`
