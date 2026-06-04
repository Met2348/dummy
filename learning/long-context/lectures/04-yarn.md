# L04 · YaRN — NTK by Parts + Attention Temperature

> 32 slides | 90 min ⭐⭐⭐⭐⭐ 必修

> Peng et al. 2023.09 / 长 ctx 当前标配

## Slide 1 · YaRN 改进

```
PI:  压 position, 损高频
NTK: 改 base, 不够 long
YaRN: NTK by parts + attention temperature
```

→ Llama-3.1 / Qwen-2.5 / DS-V3 都用 YaRN-variant。

## Slide 2 · "by parts" 思路

```
不同 dim 不同 scaling:
- 高频 dim (低 index): 保持原 base (无变化)
- 低频 dim (高 index): 改 base 大
```

→ 保留细节 + 拓展长距。

## Slide 3 · "ramp" 函数

```
ramp(d):
  if d < α_low: 1            (full PI)
  if d > α_high: 0           (no PI)
  else: linear interp
```

每 dim 用 ramp 决定 PI 程度。

## Slide 4 · Attention temperature

YaRN 还引入：
```
attn_scale = √(1 / (0.1 · ln(scale_factor) + 1))
scores = (q @ k^T) / sqrt(d) × attn_scale
```

补偿长 context 下 attention 散开。

## Slide 5 · 完整 YaRN

```
1. NTK-aware base 缩放
2. ramp function 对 dim
3. attention temperature 校准
```

## Slide 6 · 性能

```
Llama-2-7B + YaRN: 4k → 128k 无大损失 ppl
PG-19 ppl @ 128k: 与 4k 基本持平
```

## Slide 7 · 实现 rope_yarn.py

详见 src/rope_yarn.py。

## Slide 8 · 与 Llama-3 scale 配方

```
Llama-3.1 用 "llama3" RoPE scaling type:
  factor=8.0
  low_freq_factor=1.0
  high_freq_factor=4.0
  original_max=8192
```

类 YaRN 变体。

## Slide 9 · DeepSeek-V3 YaRN

```
DS-V3: factor=40, original_max=4096
↓ 推 128k
```

更激进的 scaling 但效果好。

## Slide 10 · attention temp 的作用

```
长 context:
  q · k 范围扩大
  softmax 输出过于 sharp
  attention 集中于少数 token
↓
attn_scale < 1 → 降低 score
→ softmax 平滑
```

## Slide 11-32 · 详细推导 + ramp 函数（略 - 见 YaRN 论文）

## 参考
- Peng et al. 2023 (YaRN)
- Llama-3.1 tech report (llama3 RoPE scaling)
