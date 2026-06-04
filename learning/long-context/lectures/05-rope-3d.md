# L05 · 3D RoPE / M-RoPE

> 16 slides | 50 min ⭐⭐⭐

## Slide 1 · 3D RoPE 动机

```
2D RoPE: text 序列 (单一时间维度)
3D RoPE: 图像 patch (x, y) + 时间 → 3D
```

## Slide 2 · Qwen2-VL M-RoPE

```
text token: (t, 0, 0)
image patch: (t, x_patch, y_patch)
video frame: (t, x, y) + frame index
```

3 个独立轴 RoPE，concat 后用。

## Slide 3 · 实现

```python
def m_rope(token, axis_pos):
    dim_per_axis = dim // 3
    rope_t = rope(token[:dim_per_axis], axis_pos[0])
    rope_x = rope(token[dim_per_axis:2*dim_per_axis], axis_pos[1])
    rope_y = rope(token[2*dim_per_axis:], axis_pos[2])
    return concat([rope_t, rope_x, rope_y])
```

## Slide 4 · 与 2D RoPE 对比

```
2D RoPE (Llama): 1 个 axis (sequence)
3D RoPE (Qwen2-VL): 3 个 axis
```

每 axis 独立 RoPE。

## Slide 5 · 应用

```
Qwen2-VL: 主要采用
DeepSeek-VL2: 类似
GPT-4 vision: 未公开
```

## Slide 6 · 视频长 context

```
1 帧 = N image patch (RoPE axes 2,3)
M 帧 = 时间轴 (RoPE axis 1)
↓
1000 帧 × 128 patch = 128k token (类比文本)
```

## Slide 7-16 · 详细（略）

## 参考
- Qwen2-VL paper 2024
- M-RoPE 详解 blog
