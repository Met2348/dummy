# L07 · Striped Attention

> 16 slides | 50 min ⭐⭐⭐

## Slide 1 · 改进 Ring

```
Ring: 顺序 K 块 → 部分 GPU 等待
Striped: K 块 stripe 排列 → 更均衡负载
```

## Slide 2 · 数学等价

Striped 在数学上与 Ring 等价，但调度更均衡。

## Slide 3 · 实现

```python
# Ring: K_order = [K_0, K_1, K_2, K_3]
# Striped: K_order = [K_0, K_2, K_1, K_3]  (奇偶分组)
```

## Slide 4 · 性能

```
8 GPU Ring:     某 GPU 闲置 30%
8 GPU Striped:  闲置 < 10%
```

## Slide 5 · causal mask 处理

```
causal: 后 token 不看前 token
↓
某些 round 全 -inf 可 skip
Striped 优化 skip 顺序
```

## Slide 6 · 是否流行

Striped 是 Ring 的工程改进，效果稳定但收益 5-10%。

主流框架（vLLM / Megatron）多用 Ring + 内部优化。

## Slide 7-16 · 详细（略）

## 参考
- Striped Attention paper 2024
- Ring 后续优化
