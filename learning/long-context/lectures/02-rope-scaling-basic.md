# L02 · Position Interpolation (PI)

> 18 slides | 55 min ⭐⭐⭐⭐

> Meta 2023.06 / "训 2k 推 16k"

## Slide 1 · PI 直觉

```
RoPE: position m → 旋转 m*θ
训 2k 推 8k → m 跨出训练范围 → 炸
↓
PI: m → m / scale_factor  (压缩 positions)
```

## Slide 2 · 公式

```
原 pos: m ∈ [0, L_train]
new pos: m' = m × (L_train / L_new)
```

例：2k 训 → 16k 推，scale_factor = 8。

## Slide 3 · 实现

```python
def pi_rope(t, dim, base=10000, scale_factor=8.0):
    inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    pos = arange(t).float() / scale_factor   # ← key
    angles = pos[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()
```

## Slide 4 · 优缺

```
优:  简单 1 行修改
缺:  压缩高频信息 → 局部细节损失
    需 fine-tune 才能完全恢复
```

## Slide 5 · 性能

```
Llama-1 7B + PI (no FT): 2k → 16k 可推
ppl 比训 4k 略差
```

## Slide 6 · 实务

```
直接换 inv_freq 加 scale factor
1k step fine-tune 即可补回
```

## Slide 7-18 · 详细数学（略 - 见论文）

## 参考
- Chen et al. 2023 (Position Interpolation)
- Meta 内部 ablation
