# L09 · CUDA Graphs 推理加速

## 1 · 痛点
decode 是 memory-bound + tiny kernel × 上百个 → kernel launch overhead 可观（每 launch ~10 µs，1000 launch = 10 ms）。

## 2 · CUDA Graph 原理
- 一次性 "录制"（capture）一组 kernel + memcpy
- 后续 "回放"（replay）→ 只一次 launch
- 大幅降低 host-side overhead

```python
# capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = model(static_input)

# replay
static_input.copy_(real_input)
graph.replay()
out_real = out.clone()
```

## 3 · 推理痛点
模型 forward 形状每 iter 变（batch / seq）→ 录的 graph 对应不上。

## 4 · 解：bucketing
- 把 batch / seq 分桶（如 1, 2, 4, 8, 16, ...）
- 每个 (B, S) 桶录一个 graph
- 实际请求 round-up 到最近桶 → 走对应 graph

## 5 · vLLM 实现
- `--enforce-eager` 关 graph (debug)
- 默认录 batch ∈ {1, 2, 4, ..., max_batch} 共 ~8 个 graph
- 显存占用 + 2-5%（多份静态 buffer）

## 6 · 收益
| 配置 | 7B decode iter |
|------|---------------|
| eager | 8 ms |
| graph | 3 ms |

batch 越小，graph 收益越大（launch overhead 占比高）。

## 7 · 限制
- 只能 decode 阶段（shape 固定）
- prefill 不能用（长度可变）
- attention 动态 shape 不能录入 graph，要在 graph 外做或用 paged 的 fixed-shape kernel

## 8 · 实现
教学留 stub，不依赖 cuda。
