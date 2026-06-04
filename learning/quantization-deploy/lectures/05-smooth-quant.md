# L05 · SmoothQuant（Xiao et al., Han 2022）

## 1 · 痛点
W8A8 矩阵乘法理论上快，但 activation 有 outlier → 量化掉精度。

## 2 · idea：把 activation outlier 转移到 weight
`y = X · W = (X · diag(1/s)) · (diag(s) · W) = X_s · W_s`
- `X_s = X / s` (activation 平滑)
- `W_s = s · W` (weight 吸收 outlier)

激活的 outlier "搬到" weight 上，weight 反正要量化 → 净赚。

## 3 · 选 s
对每 channel j：`s_j = max(|X_j|)^α / max(|W_j|)^(1-α)`
α ≈ 0.5 平衡两边。

## 4 · 收益
| 方案 | PPL |
|------|-----|
| fp16 | 5.68 |
| W8A8 naive | 6.50 |
| **SmoothQuant W8A8** | **5.75** |

接近无损 + matmul 用 int8 cuda kernel → 2x 加速。

## 5 · 与 AWQ 区别
| | SmoothQuant | AWQ |
|---|-----------|-----|
| 量 activation? | 是 (int8) | 否 |
| 量 weight? | int8 | int4 |
| matmul kernel | int8 | int4 + fp16 |
| 精度 | 高 | 极高 |
| 速度 | 2x (W8A8) | 1.8x (W4A16) |

两者可组合 → W4A8。

## 6 · 实现：[smooth_quant.py](../src/smooth_quant.py)
- find_scale (α=0.5)
- apply_smoothing
- 误差对照
