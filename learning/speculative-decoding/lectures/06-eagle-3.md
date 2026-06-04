# L06 · EAGLE-3（Li 2025）

## 1 · EAGLE-2 → EAGLE-3 改进点
- **Multi-feature input**: draft layer 输入 backbone 多层 feature（不只 last layer）
- **Scaling**: support 32B+ target models
- **Training-time changes**: 引入 teacher-forcing on rolled-out trajectories

## 2 · multi-feature 直觉
last hidden 已经"committed"到 next token；earlier hidden 含更多语义。
draft input concat [h_8, h_16, h_24, h_32] → 信息丰富 → accept rate ↑

## 3 · 收益
| 方法 | 加速 |
|------|------|
| EAGLE-2 | 4.0x |
| **EAGLE-3** | **5.0-6.0x** (Llama-3 70B) |

## 4 · scaling laws
- target 越大，draft 相对越快 → 加速比 ↑
- 70B target + 1B draft layer → 5x
- 7B target + 200M draft → 3x

## 5 · 部署
- HuggingFace 集成（`spec_decode_method="eagle"`）
- vLLM 集成（0.7+）
- sglang 集成

## 6 · 与 Medusa-2 (2024.06) 对比
EAGLE-3 全面胜过 Medusa-2，目前**业界 SOTA**。

## 7 · 一句话
> 投机解码 = "small smart steps" + "verify in parallel"，EAGLE 把这个范式榨干。
