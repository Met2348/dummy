# L02 · int8 基础

## 1 · 朴素：min-max
对张量 `x`：
```
s = (x_max - x_min) / 255
zp = round(-x_min / s)
x_int = round(x / s + zp).clip(0, 255)
x_dq = (x_int - zp) * s
```

## 2 · symmetric vs asymmetric
- symmetric: `zp = 0`, `s = max(|x|) / 127`  → 简单，kernel 快
- asymmetric: 用 zp → 精度略高，但 kernel 复杂

LLM 通常用 symmetric for weight, asymmetric for activation。

## 3 · per-tensor / per-channel / per-group
| 粒度 | scale 数 | 精度 | 速度 |
|------|---------|------|------|
| per-tensor | 1 | 差 | 最快 |
| per-channel | `out_features` | 中 | 快 |
| per-group (128) | `out_features × in_features/128` | 高 | 略慢 |

## 4 · 量化误差来源
- **outlier**: 一个 channel 出现 100×大值 → 整列 scale 飞掉 → 其他 channel 量化精度差
- 这就是 LLM.int8() 处理的重点

## 5 · symmetric int8 公式
```python
def quant_sym(x, axis):
    s = x.abs().max(dim=axis) / 127
    q = (x / s).round().clip(-128, 127).int8
    return q, s

def dequant_sym(q, s):
    return q.float * s
```

## 6 · 实现：[int8_basics.py](../src/int8_basics.py)
- `quantize_per_tensor` / `quantize_per_channel` / `quantize_per_group`
- 误差度量（MSE）
