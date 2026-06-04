# L01 · 量化全图

## 1 · 为什么量化
| 模型 | fp16 显存 | int4 显存 |
|------|----------|----------|
| Llama-7B | 14 GB | **3.5 GB** |
| Llama-70B | 140 GB | **35 GB** |
| Qwen-2.5-72B | 144 GB | 36 GB |

→ 5090 24GB 只能跑 fp16 14B；int4 能跑 70B。

## 2 · 4 个量化维度
| 维度 | 选择 |
|------|-----|
| 对象 | weight / activation / KV cache |
| 精度 | int8 / int4 / fp8 / fp4 |
| 粒度 | per-tensor / per-channel / per-group (128) |
| 时机 | PTQ (post-training) / QAT (训练时) |

## 3 · 三大主流方案 2024-2026
| 方案 | 对象 | 精度 | 何时用 |
|------|-----|------|-------|
| **GPTQ** | weight | int4 | 高精度 PTQ |
| **AWQ** | weight | int4 | 激活感知，PTQ |
| **FP8** | weight + activation | E4M3 | Hopper+ 训推一致 |
| **bitsandbytes int4** | weight | int4 | 端侧极简 |
| **W4A8** | weight int4 + activation int8 | int4/int8 | 极致显存 |

## 4 · 量化的精度损失
| 方案 | PPL ↑ | MMLU ↓ |
|------|-------|--------|
| int8 (LLM.int8()) | +1% | -0.5pp |
| GPTQ 4bit | +3% | -1pp |
| AWQ 4bit | +2% | -0.5pp |
| FP8 | +0.5% | 0 |
| W4A8 | +5% | -2pp |

## 5 · 显存节省 vs 速度
- weight-only 量化：省显存，**decode 加速 1.5-3x**（memory-bound 主导）
- W+A 量化：用 int8/fp8 算 matmul → kernel 也快
- KV 量化：进一步省 50% 显存

## 6 · 本专题路线
- L02-L05: 经典 weight 量化（int8 / GPTQ / AWQ / SmoothQuant）
- L06-L08: FP8 (训推一致 / W4A16 / W4A8)
- L09: KV cache 量化
- L10-L12: 评测 + Capstone
- L13: 量化动物园

## 7 · 一句话
> 量化 = **用精度换显存 + 用 kernel 换速度**。2024-2026 已经做到 int4 几乎无损 → 7B+ 模型上端侧可用。
