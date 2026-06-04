# L04 · DeepSpeed (ZeRO + 3D parallel)

> 14 slides | 45 min ⭐⭐⭐⭐

## Slide 1 · DeepSpeed 三大件

```
ZeRO: data parallel sharding (1/2/3 阶段)
3D parallel: TP + PP + DP
Inference: ZeRO-Infer + KV cache 压缩
```

Microsoft 维护, Megatron-DeepSpeed = Megatron + DS.

## Slide 2 · 配置文件 (ds_config.json)

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "gradient_clipping": 1.0
}
```

## Slide 3 · 启动

```python
import deepspeed

model_engine, _, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters(),
    config="ds_config.json",
)

for batch in loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

## Slide 4 · ZeRO Offload (CPU)

```
显存不够 → 把 optimizer state 推 CPU
带宽: PCIe 32 GB/s (慢)
trade-off: 训得动 vs 慢 30-50%
```

## Slide 5 · ZeRO-Infinity (NVMe)

```
optimizer state → NVMe SSD
能训 1T+ 模型, 但极慢
```

## Slide 6 · Stage 3 限制

```
不能 print(model)
不能 model[i].weight 直接访问
不能 reduce-scatter 跨节点带宽限制
```

## Slide 7 · 3D parallel (Megatron-DeepSpeed)

```json
{
  "tensor_parallel": {"tp_size": 8},
  "pipeline_parallel": {"stages": 8, "method": "1f1b"}
}
```

## Slide 8 · 性能 trick

```
overlap_comm: True       # 通信/计算重叠
allgather_bucket_size: 5e8
reduce_bucket_size: 5e8
```

## Slide 9 · DeepSpeed-Chat

```
针对 RLHF 优化:
- 三段管线 (SFT + RM + PPO) 单机器
- ZeRO + 4-model rollout 优化
```

## Slide 10 · DeepSpeed-MoE

```
Expert Parallel + ZeRO data parallel 联合
适合 200B+ MoE 训练
```

## Slide 11 · 与 FSDP 对比

| 项 | FSDP | DeepSpeed |
|----|------|-----------|
| ZeRO-3 | ✓ | ✓ |
| TP | ✗ | ✓ (Megatron-DS) |
| PP | ✗ | ✓ |
| CPU offload | ✓ | ✓ (更成熟) |
| MoE | 一般 | ✓ (强) |
| ckpt 灵活 | ✓ | ✗ |

## Slide 12 · 启动多卡

```bash
deepspeed --num_gpus 8 train.py \
  --deepspeed_config ds_config.json
```

## Slide 13 · Accelerate + DeepSpeed

```bash
accelerate config  # 选 deepspeed
accelerate launch train.py
```

更简洁, 适合中小项目.

## Slide 14 · 总结

```
DeepSpeed = ZeRO + 3D + offload
推荐: 100B+ 训练 / RLHF / MoE
小项目: FSDP 即可
```

## 参考
- deepspeed.ai
- ZeRO 2020 (Rajbhandari)
