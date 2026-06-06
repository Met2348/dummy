# L04 — Checkpoint 策略

## 三代

### 1. Full (legacy DeepSpeed)

```
all_gather → rank 0 全模型 → 单写者写 Lustre
```

70B 在 512 GPU 上：gather 0.35s + write 0.35s ≈ 0.7s blocking

### 2. Sharded (DCP, PyTorch 2.x)

```
每 rank 写自己的 shard (FSDP/ZeRO)
```

但 OSS pool 仍是 BW 上限 → 0.35s blocking

### 3. Async + staging (DCP-async, 2024)

```
GPU → host pinned RAM (PCIe, 独立 per-rank, 几 ms)
host RAM → Lustre (background, 与训练并行)
```

实际 blocking：仅 PCIe 阶段 → <10 ms ⭐

## 故障恢复经济学

512 GPU 一周训练：
- MTBF ≈ 24 小时 → 期望 7 次故障
- 每次恢复 = ckpt 加载 + 半个 ckpt 间隔重做
- 每小时 ckpt → 平均 30 分钟 wasted/failure
- 全程恢复成本 ≈ 3.5 小时

## 工程指导

- ckpt_interval = sqrt(2 × ckpt_cost × MTBF) → 最优频率
- 用 DCP-async (PyTorch 2.4+)
- 同时存 fast tier (Lustre) + slow tier (S3 备份)
