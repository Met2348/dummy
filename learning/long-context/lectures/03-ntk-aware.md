# L03 · NTK-aware Scaling

> 18 slides | 55 min ⭐⭐⭐⭐

> LocalLlama 社区 2023 / bloc97

## Slide 1 · NTK-aware 直觉

```
PI: 压缩 position → 高频细节损
NTK-aware: 改 base → 频率分布平滑
```

## Slide 2 · 公式

```
base_new = base × (L_new / L_train)^{d/(d-2)}
```

base 大 → 长波频率慢 → 容纳更长序列。

## Slide 3 · 与 PI 区别

| | PI | NTK |
|---|----|----|
| 改 | position | base |
| 高频 | 损失 | 保留 |
| FT 需求 | 需 1k step | 几乎无需 |

## Slide 4 · 实现

```python
def ntk_aware(t, dim, base=10000, scale=8):
    new_base = base * (scale ** (dim / (dim - 2)))
    inv_freq = 1.0 / (new_base ** (arange(0, dim, 2) / dim))
    ...
```

## Slide 5 · 性能

```
Llama-1 NTK-aware: 2k → 16k 无 FT
ppl 持平训 4k
```

## Slide 6 · 局限

```
对 32k+ 仍需 FT
对极端长 (128k) 还不够
↓
YaRN 解决
```

## Slide 7-18 · 详细（略）

## 参考
- LocalLlama bloc97 reddit post 2023.06
- 各家整合 NTK-aware 算法
