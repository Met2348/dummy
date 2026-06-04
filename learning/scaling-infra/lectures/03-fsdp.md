# L03 · FSDP (Fully Sharded Data Parallel)

> 14 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · FSDP = ZeRO-3 官方 PyTorch 版

```
shard: W, G, O 全分散
forward: all-gather W → 计算 → 释放
backward: all-gather W → 计算 → reduce-scatter G
```

显存 = 1/N (近似)，通信 vs DP 多。

## Slide 2 · 基础用法

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch

torch.distributed.init_process_group("nccl")
model = MyModel().cuda()
model = FSDP(model)
```

## Slide 3 · auto-wrap policy

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

policy = functools.partial(size_based_auto_wrap_policy,
                            min_num_params=1_000_000)
model = FSDP(model, auto_wrap_policy=policy)
```

## Slide 4 · Transformer 专用 policy

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

policy = functools.partial(transformer_auto_wrap_policy,
                            transformer_layer_cls={MyTransformerBlock})
model = FSDP(model, auto_wrap_policy=policy)
```

每个 transformer block 单独 wrap → 通信粒度最优。

## Slide 5 · sharding strategy

```python
from torch.distributed.fsdp import ShardingStrategy

FULL_SHARD       # ZeRO-3 默认
SHARD_GRAD_OP    # ZeRO-2
NO_SHARD         # DDP
HYBRID_SHARD     # 节点内 full, 跨节点复制
```

## Slide 6 · mixed precision

```python
from torch.distributed.fsdp import MixedPrecision

mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)
model = FSDP(model, mixed_precision=mp)
```

## Slide 7 · activation checkpointing

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper \
    import apply_activation_checkpointing

apply_activation_checkpointing(model)
```

通常 + FSDP 联合用，省 activation memory.

## Slide 8 · ckpt 保存

```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                          FullStateDictConfig(offload_to_cpu=True,
                                              rank0_only=True)):
    ckpt = model.state_dict()
    if rank == 0:
        torch.save(ckpt, "model.pt")
```

## Slide 9 · 性能 trick

```
backward_prefetch=BackwardPrefetch.BACKWARD_PRE  # 重叠通信
forward_prefetch=True                            # 重叠 forward
limit_all_gathers=True                            # 控制并发
```

## Slide 10 · FSDP vs DeepSpeed ZeRO

| 项 | FSDP | DeepSpeed |
|----|------|-----------|
| 集成 | PyTorch 内置 | 外部库 |
| 学习曲线 | 中 | 高 (config) |
| ZeRO-3 性能 | 接近 | 略好 |
| CPU offload | 支持 | 更成熟 |
| MoE | 一般 | 强 |

## Slide 11 · 启动

```bash
torchrun --nproc_per_node=4 train.py
```

或 accelerate launch 包装:
```bash
accelerate launch --use_fsdp train.py
```

## Slide 12 · 实战 trap

```
- 不能用 model.cuda() 后再 FSDP (用前 .cuda())
- 不能直接 print(model)
- save ckpt 必须用 state_dict_type 上下文
- backward 后才能 step()
```

## Slide 13 · 5090 24G 实战

```
Llama-3-8B + FSDP(1 GPU):
  W: 16GB, 没法训
↓
Llama-3-8B + FSDP + LoRA:
  W: 16 (frozen) + LoRA: 0.1 GB grad
  ✓ 单卡可训 32k ctx
```

## Slide 14 · 总结

```
FSDP = PyTorch 官方 ZeRO-3 + auto wrap + mixed precision
推荐: 7B-70B 训练, 1-8 卡
```

## 参考
- pytorch.org/docs/stable/fsdp.html
- Zhao 2023 FSDP paper
