# L05 · Megatron-LM Tensor Parallel

> 14 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · TP idea

```
W in [d, 4d]
拆 col: W = [W0, W1, W2, W3]  各 [d, d]
GPU 0..3 各算 X @ Wi
```

## Slide 2 · MLP TP

```
linear1 (d → 4d): col-split
nonlinearity (内部, 无通信)
linear2 (4d → d): row-split
all-reduce 最终 output
```

## Slide 3 · MLP TP 实现 (PyTorch)

```python
class ColumnLinear(nn.Module):
    def __init__(self, d_in, d_out, tp_size):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_in, d_out // tp_size))
    def forward(self, x):
        return x @ self.W

class RowLinear(nn.Module):
    def __init__(self, d_in, d_out, tp_size):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_in // tp_size, d_out))
    def forward(self, x):
        out = x @ self.W
        dist.all_reduce(out)
        return out
```

## Slide 4 · attention TP

```
Q, K, V proj: col-split (按 head)
attention 计算 (内部, head_dim 内)
output proj: row-split → all-reduce
```

每张卡只算自己 head。

## Slide 5 · TP 通信开销

```
每 transformer block 2 次 all-reduce
fwd: 2, bwd: 2 (total 4 / block)
带宽 = 2 × hidden × seq × bf16
节点内 NVLink (900 GB/s) ✓ 可承受
节点间 IB (400 GB/s) ✗ 慢, 不推荐
```

## Slide 6 · 推荐配置

```
TP_size = 节点内 GPU 数 (通常 8)
跨节点用 PP 或 DP
```

## Slide 7 · embedding TP

```
vocab embedding: row-split
GPU 0: vocab [0..1/N]
GPU 1: vocab [1/N..2/N]
...
input id 经 mask + all-reduce 取得 embedding
output head 同理 (但反向)
```

## Slide 8 · 启动 Megatron

```bash
torchrun --nproc_per_node 8 \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 4096 ...
```

## Slide 9 · 与 vanilla 对比性能

```
GPT-3 175B + 1024 GPU:
  DP only: OOM
  TP=8 + DP=128: 24% MFU
  TP=8 + PP=8 + DP=16: 52% MFU
```

## Slide 10 · TP + ZeRO 混合

```
TP=8 切 weight (节点内)
ZeRO=DP_size 切 optimizer state (跨节点)
两者正交, 节省双倍显存
```

## Slide 11 · TP 限制

```
- 必须能整除 (head_count, intermediate_dim) 
- NVLink 必须可用 (否则 TP > 2 太慢)
- ckpt 保存复杂 (合并 shards)
```

## Slide 12 · 5090 单机 TP?

```
24G 24G NVLink? 5090 没 NVLink
PCIe 5.0: 64 GB/s
TP 在 PCIe 上太慢
推荐: 单 5090 + FSDP, 不上 TP
```

## Slide 13 · Megatron-Core (新)

```
Megatron-Core (2023+): 独立库
nvidia/Megatron-LM 已迁过去
更易集成到 HF/TRL
```

## Slide 14 · 总结

```
TP = 节点内 (NVLink) 最强切法
节点间用 PP/DP
单卡用户跳过
```

## 参考
- Megatron-LM (Shoeybi 2019)
- Megatron-Core docs
