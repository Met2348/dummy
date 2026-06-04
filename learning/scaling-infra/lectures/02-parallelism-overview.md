# L02 · 并行训练总览

> 16 slides | 50 min ⭐⭐⭐⭐⭐

## Slide 1 · 4 种并行

```
Data Parallel (DP)       — 复制模型，切 batch
Tensor Parallel (TP)     — 切 weight，row/col
Pipeline Parallel (PP)   — 切 layer，流水
Sequence Parallel (SP)   — 切 seq_len
                            ↓
ZeRO / FSDP             — DP 的 sharded 版
3D parallel             — DP + TP + PP 组合
```

## Slide 2 · DP 简单

```
GPU 0: full model + batch 0
GPU 1: full model + batch 1
all-reduce grad → 更新
```

显存: 4×（每卡放完整模型）
速度: 接近线性

## Slide 3 · DP 瓶颈

```
模型 > 单卡显存 → 必须切
all-reduce 在大集群 (>1k) 慢
```

## Slide 4 · TP (Megatron style)

```
W in [in, out]
切 col: W → [W0, W1]
GPU 0 算 X @ W0, GPU 1 算 X @ W1
forward: all-gather output
backward: all-reduce grad
```

## Slide 5 · TP attention

```
QKV 切 head_dim 维 (column)
output proj 切 row
中间 attention 全在卡内
通信只在 row/col 边界
```

## Slide 6 · PP

```
GPU 0: layer 0-5
GPU 1: layer 6-11
GPU 2: layer 12-17
...
微 batch 流水: F0 F1 F2 | B2 B1 B0
```

## Slide 7 · PP "bubble"

```
朴素 PP: 启动/排空 浪费 50%
1F1B (Megatron-LM): 减少 bubble
Interleaved 1F1B: 进一步减少
```

## Slide 8 · 1F1B

```
F1 F2 F3 F4 | B1 F5 B2 F6 B3 F7 B4 F8 | B5 B6 B7 B8
                ↑ 一旦 F4 完成立即开始 B1
```

## Slide 9 · ZeRO Stage 1/2/3

```
DP 每卡冗余存:
  weights (W)
  gradients (G)
  optimizer state (O)
ZeRO-1: shard O
ZeRO-2: shard O + G
ZeRO-3: shard O + G + W (即 FSDP)
```

## Slide 10 · ZeRO-3 / FSDP

```
forward: all-gather W 段 → 算 → 释放
backward: 同上 + reduce-scatter G
平均显存 / GPU = 1/N
```

## Slide 11 · 3D Parallel

```
70B 在 1024 H100:
  TP=8 (1 node 内 NVLink)
  PP=8 (跨 node)
  DP=16 (跨 cluster)
  total = 8*8*16 = 1024 GPU
```

## Slide 12 · 推荐选择

| 模型大小 | 推荐 |
|---------|------|
| < 7B | DP / FSDP |
| 7B-70B | FSDP (1-8 node) |
| 70B-200B | TP=8 + FSDP |
| 200B+ | TP + PP + DP 3D |

## Slide 13 · Sequence Parallel

```
TP 已切 head, 但 LayerNorm/Dropout 仍全 batch
SP: 把 LN/Dropout 也切 seq 维
节省 activation 显存 1/TP
```

## Slide 14 · Expert Parallel (MoE)

```
专家分布在不同 GPU
all-to-all dispatch tokens → expert GPU
```

DeepSeek-V3 671B 用 EP=64 + DP=16.

## Slide 15 · 实操工具

```
naive: torch.distributed
FSDP: torch.distributed.fsdp (官方)
DeepSpeed: ZeRO + 3D parallel
Megatron-LM: TP/PP/SP 最强
Megatron-DeepSpeed: 两者合体
```

## Slide 16 · 总结

```
DP 简单但卡死
TP 高效但限于 1 个 NVLink 域
PP 跨节点但有 bubble
FSDP 通用 + 易上手
3D 是大规模唯一选择
```

## 参考
- Megatron-LM (Shoeybi 2019)
- ZeRO (Rajbhandari 2020)
- FSDP (PyTorch 2023)
