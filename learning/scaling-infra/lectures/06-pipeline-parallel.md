# L06 · Pipeline Parallel (1F1B + Interleaved)

> 12 slides | 40 min ⭐⭐⭐⭐

## Slide 1 · PP idea

```
GPU 0: layer 0-5
GPU 1: layer 6-11
GPU 2: layer 12-17
GPU 3: layer 18-23
```

切 layer, micro-batch 流水。

## Slide 2 · Naive PP "bubble"

```
F0 F1 F2 F3 | B3 B2 B1 B0
↑ 启动 ramp     ↑ 排空 drain
浪费 (N-1) / N + (N-1) / N ≈ 50% (4 GPU)
```

## Slide 3 · GPipe (Google 2018)

```
将 batch 拆成 M 个 micro-batch:
F0 F1 F2 F3 F4 F5 F6 F7 | B7 ... B0
bubble = (N - 1) / M
M = 32 → bubble 9% (4 GPU)
```

## Slide 4 · 1F1B (Megatron 2021)

```
            F0
        F1     B0
    F2     F1     B1
F3     F2     F1     B2
                          ↓ 一旦 F0/F1/F2/F3 完成立即 backward
```

bubble 与 GPipe 相同, 但 memory 少 (不需要囤 activation)。

## Slide 5 · 1F1B 显存优势

```
GPipe: 要囤 M 个 activation per stage
1F1B: 最多囤 N (#stage)
```

70B + 8 stage + M=32 micro batch:
- GPipe: 32 × 8 = 256 activation
- 1F1B: 8 activation

## Slide 6 · Interleaved 1F1B

```
每 GPU 拥有 2-4 个非连续 layer 段
F0a F0b F1a F1b ... → 进一步减少 bubble
```

## Slide 7 · zero-bubble PP

```
Sun 2024 ZB-H1:
- 把 backward 拆成 input grad + weight grad
- weight grad 可推迟
- bubble → 接近 0
```

## Slide 8 · 实现简化

```python
class PpStage(nn.Module):
    def __init__(self, layers_subset):
        super().__init__()
        self.layers = nn.ModuleList(layers_subset)
    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x

# rank 0 → rank 1 → ... torch.distributed.send/recv
```

## Slide 9 · 与 TP / DP 组合

```
3D: TP=8 (节点内) × PP=8 (跨节点) × DP=16
       8       8       16  = 1024 GPU
```

## Slide 10 · PP 通信粒度

```
PP 通信: 仅在 stage 边界 (每 N/PP layer 一次)
比 TP 少很多
适合跨节点 IB
```

## Slide 11 · PP 限制

```
- bubble 总是存在
- ckpt 跨 stage 合并复杂
- 实现成本高 (1F1B 调度)
```

## Slide 12 · 推荐使用场景

```
70B+ 模型 + 跨节点训练 → 必须 PP
< 70B → FSDP 已够
```

## 参考
- GPipe (Huang 2018)
- 1F1B (Narayanan 2021 Megatron)
- ZB-H1 (Sun 2024)
