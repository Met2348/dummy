# L04 · AWQ（Lin et al. MIT 2024）⭐

## 1 · Key observation
"显著激活通道"（salient channel）只占 1%，但承载主要 information。
保护这 1% channel 的精度 → 整体损失大幅降。

## 2 · 算法
1. **找显著 channel**: 看 activation magnitude `|X_j|`，取 top 1%
2. **per-channel rescale**: weight 对应 channel 乘 `s_j`，activation 除 `s_j`
3. **量化 weight**: 用 round + symmetric int4
4. **deploy**: kernel 自动处理 rescale

## 3 · 关键巧妙
- 不直接保护 weight，而是**重新缩放**让 weight 更"圆"（钝化 outlier）
- 数学上和量 activation 一样，但**不需要 activation 量化** → kernel 简单

## 4 · 数学
原始 matmul `y = W · x`
缩放后 `y = (W · diag(s)) · (diag(1/s) · x) = W_s · x_s`
- `W_s = W · diag(s)` 被 round
- `x_s = x / s` runtime 直接除（fp16 操作）

s 用 grid search 选最优。

## 5 · 精度（Llama-7B）
| 方案 | MMLU | PPL |
|------|------|-----|
| fp16 | 45.3 | 5.68 |
| GPTQ-4bit | 44.5 | 5.85 |
| **AWQ-4bit** | **44.9** | **5.81** |
| W4A8 | 43.0 | 5.95 |

AWQ 略胜 GPTQ。

## 6 · 部署
- autoawq lib 一键
- vLLM, SGLang, TRT-LLM 全部内置 AWQ kernel
- 推理速度比 fp16 快 1.6-2.0x（decode）

## 7 · 与 GPTQ 对比
| | GPTQ | AWQ |
|---|------|-----|
| 数学基础 | 二阶 Hessian | per-channel scale search |
| 校准成本 | 1h | 10min |
| 精度 | 中 | **高** |
| 部署 kernel | 良好 | **极佳** |

## 8 · 实现：[awq_minimal.py](../src/awq_minimal.py)
- per-channel scale search
- 量化 + dequant
- 误差 vs naive 对照
