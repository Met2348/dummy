# L05 — Elastic Training (torchrun)

## 核心：rendezvous

```python
# torchrun --rdzv-id=42 --rdzv-backend=c10d --rdzv-endpoint=master:29400 \
#          --nnodes=4:16 --nproc-per-node=8 train.py
```

- `--nnodes=4:16` 表示最小 4 节点，最大 16 节点
- 任意时刻有 ≥4 节点 = quorum，可继续训
- 节点加入/退出触发 generation +1，全员 reload state

## State 必须 elastic-safe

1. 加载最近 ckpt
2. world_size 改变 → 调整 batch size / LR
3. 数据 sampler seeded by (epoch, generation) 保持可重现
4. RNG state 同步 (else 不同 rank generate 不同结果)

## 应用场景

- Spot instance 训练 (AWS / GCP spot 经常 reclaim)
- 大 cluster 部分节点维护 → 训练继续
- 容器编排平台 (K8s + PyTorchJob)

## 限制

- 模型 parallelism (TP/PP) 不能 elastic 改变 shape → 只 elastic DP 维度
- ZeRO-3 + elastic 复杂，需要预留 rank padding
- 实际很多团队选 "fixed world + restart on failure" 而非真 elastic
