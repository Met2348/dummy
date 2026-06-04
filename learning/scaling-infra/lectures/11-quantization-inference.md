# L11 · 量化推理 (GPTQ / AWQ / FP8 / int4)

> 14 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · 推理量化目的

```
fp16/bf16: 2 byte/param  → 显存大
int8: 1 byte/param        → 减半
int4: 0.5 byte/param      → 1/4
fp8: 1 byte (H100+)       → 减半 + 不掉精度
```

## Slide 2 · 训练后量化 (PTQ) vs 训练时量化 (QAT)

```
PTQ: 训完了再量化, 部署友好
QAT: 训练时模拟低精度, 更准但成本高
LLM 主流 PTQ.
```

## Slide 3 · GPTQ (Frantar 2023)

```
逐层量化 + 误差补偿 (类 OBS)
group_size: 128
desc_act: True (高频列优先)
开源工具 AutoGPTQ
```

## Slide 4 · GPTQ 公式

```
W_int = round(W / scale + zero)
逐列误差 propagated 到剩余列
最终 W_q s.t. ||X (W - W_q)||_2 最小
```

## Slide 5 · AWQ (Lin 2023)

```
观察: 1% activation 决定 99% 输出
保留这 1% 不量化 (full precision)
其余 4-bit
比 GPTQ 略好 (1pp)
```

## Slide 6 · int4 推理性能

```
Llama-3-8B + int4:
  显存: 4.5 GB (vs 16 GB bf16)
  速度: 1.3-1.5× (decode 带宽减少)
  精度: < 0.5 pp 退化
```

## Slide 7 · FP8

```
H100/B100/5090 硬件原生 fp8
两种格式: e4m3 (动态范围低), e5m2 (动态范围高)
weight: e4m3
activation: e5m2
速度 ≈ bf16 的 1.5-2×, 显存减半
```

## Slide 8 · FP8 (TransformerEngine NVIDIA)

```python
from transformer_engine.pytorch import Linear
fp8_linear = Linear(d_in, d_out, fp8=True)
```

vLLM 已自动支持: `--quantization fp8`.

## Slide 9 · KV cache 量化

```
weight 量化 ≠ KV cache 量化
KV cache 可单独 int4/fp8
```

vllm `--kv-cache-dtype fp8`.

## Slide 10 · SmoothQuant

```
activation 难量化 (outlier 大)
SmoothQuant: 把 activation outlier 转嫁到 weight
后者更易量化
```

## Slide 11 · int8 dynamic vs static

```
dynamic: activation 每次推理时计算 scale
static:  pre-calibrate, 更快
```

## Slide 12 · 5090 量化指南

```
模型 ≤ 7B: bf16 单跑 ✓
7B-32B: fp8 / int4 (AWQ)
> 32B: int4 + offload
```

## Slide 13 · 工具速查

```
weight 量化:
  AutoGPTQ      pip install
  AutoAWQ       pip install
  TransformerEngine  fp8 (H100+)
  bitsandbytes  4/8-bit

推理引擎:
  vLLM         awq / fp8 / gptq
  SGLang       fp8 / awq
  TRT-LLM      fp8 / int8 / int4
```

## Slide 14 · 总结

```
推理量化是部署必备
fp8 (有硬件) > AWQ/GPTQ int4 > int8
KV cache 量化独立配
```

## 参考
- GPTQ (Frantar 2023)
- AWQ (Lin 2023)
- SmoothQuant (Xiao 2023)
- FP8 (NVIDIA Transformer Engine)
