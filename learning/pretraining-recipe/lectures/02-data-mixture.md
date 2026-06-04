# L02 · 数据配比与课程

> 12 slides | 35 min ⭐⭐⭐⭐⭐

## Slide 1 · "data is king"

```
Llama-3 paper: 数据质量提升比架构更大
DCLM: 配比 + 过滤 → +5pp MMLU
```

## Slide 2 · 经典配比

```
Web 60-70% (FineWeb / RefinedWeb / DCLM)
Code 15-20% (StarCoder / The Stack)
Math 3-5%
zh 5-10%
Books 5%
Wiki 1-2%
```

## Slide 3 · DoReMi (Google 2023)

```
auto-balance 训练数据配比
小代理模型 + 重要性采样
比手工配比涨 1-2pp
```

## Slide 4 · DataComp-LM (DCLM 2024)

```
首个数据配比标准化 benchmark
1B 模型 + 4M GPU-hour budget
最佳: 100% DCLM-Baseline (40% web + ...)
```

## Slide 5 · 课程学习

```
phase 1 (前 80%): general web → 学语言
phase 2 (后 20%): code + math + wiki → 提升推理
```

DeepSeek-V3 / Qwen-2.5 都用。

## Slide 6 · Phi 路线 (textbook quality)

```
不是大量 web, 而是合成 + 教科书
"Textbooks are all you need" (Microsoft 2023)
50% 合成 + 30% 高质量 web + 20% code
```

## Slide 7 · 数据规模 vs 模型

```
1:20 (Chinchilla)
1:200 (Llama-3)
1:2000 (Qwen-2.5)
                  ↑ 继续涨
极限 ≈ 1:10000 (尚未达到)
```

## Slide 8 · token-to-param ratio 实验

```python
ratios = [10, 20, 50, 100, 200, 500, 1000]
for r in ratios:
    train(N=100M, D=N*r)
    eval(MMLU, HellaSwag)
# 实测: 1:200 最优, 之后递减
```

## Slide 9 · 中文配比

```
LLaMA-1: ~0% zh → 中文差
Qwen-2.5: 30% zh → 中文 SOTA
DeepSeek-V3: 30% zh + 30% en + 40% other
```

## Slide 10 · 长 ctx 阶段

```
training 末期: max_seq_len 8k → 32k → 128k
所用 token: 占总 1-3%
但效果显著 (NIAH 提升)
```

## Slide 11 · annealing

```
最后 1% step:
  lr → 1e-5
  高质量数据 only (Books, Wiki, Math)
  → 模型"晶化"
```

WSD 的核心 idea。

## Slide 12 · 总结

```
数据配比是 secret sauce
课程学习 = stage + annealing
WSD 在 stable phase 训, decay phase 注入高质数据
```

## 参考
- DCLM (Datacomp-LM)
- DoReMi (Xie 2023)
- Phi-1.5 (Microsoft)
