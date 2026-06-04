# L07 · FP8 格式（E4M3 / E5M2）

## 1 · IEEE 754 vs FP8
| dtype | bits | exp | mantissa | 范围 |
|-------|------|-----|---------|------|
| fp32 | 32 | 8 | 23 | ±3e38 |
| fp16 | 16 | 5 | 10 | ±65k |
| bf16 | 16 | 8 | 7 | ±3e38 |
| **E4M3** | 8 | 4 | 3 | ±448 |
| **E5M2** | 8 | 5 | 2 | ±57344 |

## 2 · E4M3 vs E5M2
- E4M3: 精度高，范围小，用于 forward weight/activation
- E5M2: 范围大，精度低，用于 gradient
- NVIDIA Hopper / Blackwell 硬件支持

## 3 · per-tensor 缩放
fp8 范围有限 → 必须 scale：
```
x_fp8 = (x_fp32 / scale).cast(fp8)
y = matmul_fp8(x_fp8, w_fp8) * (x_scale * w_scale)
```

## 4 · 训推一致
- DeepSeek-V3 用 FP8 训练
- 推理直接用同 FP8 权重 → 无量化误差！
- 与传统 PTQ (fp16 训，int4 推) 区别本质

## 5 · 速度
H100/H200 上 FP8 matmul 比 bf16 快 1.5-2x（tensor core）。

## 6 · 软件栈
- PyTorch 2.5+ 原生 fp8 dtype
- TransformerEngine (NVIDIA) FP8 训练库
- vLLM/SGLang 内置 FP8 推理

## 7 · 精度
| 方案 | PPL |
|------|-----|
| fp16 | 5.68 |
| **FP8 (E4M3, per-tensor)** | **5.70** |
| FP8 (E4M3, per-channel) | 5.69 |

接近无损 + 速度 + 显存全省。

## 8 · 5090 (Blackwell) FP8
- 第 5 代 tensor core
- 比 H100 FP8 快 2x
- 5090 24GB 跑 70B FP8 ≈ 35GB ⇒ 仍不够，需 W4

## 9 · 实现：[fp8_demo.py](../src/fp8_demo.py)
- fp8 cast / matmul mock
- 与 fp16 对照
