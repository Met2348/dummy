# L03 · Megatron-LM 风格 TP

## 1 · 完整 Transformer 切法
| 算子 | 切法 | 通信 |
|------|------|-----|
| QKV projection | column split | none |
| Attention | per-head 切 | none (heads 独立) |
| Output projection | row split | allreduce |
| MLP up | column split | none |
| MLP gate | column split | none |
| MLP down | row split | allreduce |

每 layer **2 次 allreduce**。

## 2 · Megatron-LM (Shoeybi et al. 2020)
- 标准实现 / NVIDIA 维护
- 推理时被 vLLM/SGLang 复用

## 3 · TP communication 优化
- **Sequence Parallel (SP)**: dropout + LN 也切 → 减 activation 显存 + 减次 allreduce
- **Context Parallel (CP)**: 长 ctx 时再切 sequence 维
- **Async TP**: 通信与计算 overlap

## 4 · MoE + TP
- expert 不分 TP（不切矩阵）→ EP 单独
- shared expert 走 TP
- 例如 Mixtral 8x7B：TP=4 (单 expert) + EP=8 (8 expert)

## 5 · TP=2 vs TP=4 vs TP=8
| TP | 7B 单 token 延迟 (decode) |
|----|-------------------------|
| 1 | 8 ms |
| 2 | 6 ms |
| 4 | 5 ms |
| 8 | 4.5 ms (allreduce overhead dominant) |

→ 边际收益递减；7B 用 TP=2 即可

## 6 · 实现继承 tp_demo.py
"""
