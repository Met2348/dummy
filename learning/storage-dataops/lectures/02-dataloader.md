# L02 — Dataloader pipeline

## 4 阶段

```
[ fetch (Lustre/S3) ] → [ decode (JPEG/JSON) ] → [ augment ] → [ collate ]
       I/O               CPU                       CPU            GPU H2D
```

## PyTorch DataLoader 三要点

- `num_workers=N`：N 个 worker process 并行
- `prefetch_factor=K`：每 worker 提前 K batch
- `pin_memory=True`：H2D 异步 + 锁页内存

## 计算

- 单 sample 时长 = max(stages) / N_workers (并行隐藏)
- 总时长 ≈ N_samples × max_stage / N_workers
- 瓶颈是 decode → 4 worker 4× 加速

## 工程坑

- IterableDataset vs Map: 大数据集必须 IterableDataset (avoid index list 内存爆)
- shuffle buffer：100k-1M 平衡随机性 vs 内存
- 多 epoch 重 shuffle：必须 `worker_init_fn` 设 seed
- 单 worker decode bottleneck → 上 GPU JPEG (nvJPEG / DALI)
