# L08 · FP8 训练（回顾 Module 3 scaling-infra）

## 1 · 痛点
fp16/bf16 训练已是主流，但 fp16 / bf16 matmul 在 H100/Blackwell 上**没用到全部 TFLOPs**。

## 2 · FP8 训练流程
1. forward: fp8 matmul (E4M3)
2. backward: fp8 gradient (E5M2)
3. accumulate: fp32 (master weight)
4. 更新: 在 fp32 上 step optimizer

## 3 · 关键工程
- **gradient scaling**: fp8 范围窄 → 必须缩放避免下溢
- **dynamic loss scaling**: 实时调整
- **mixed precision master weight**: fp32 主权重，每步从 fp32 量到 fp8 forward

## 4 · DeepSeek-V3 的 FP8 训练
- 全程 FP8 weight + activation
- 训出 → 直接用 FP8 部署
- 训练时间相对 bf16 节省 30%
- 推理无任何量化损失

## 5 · NVIDIA TransformerEngine
官方 FP8 训练库：
- 自动 quant/dequant
- 自动 scaling 调整
- 用 1 行代码切换 fp8/bf16

```python
import transformer_engine.pytorch as te
with te.fp8_autocast(enabled=True):
    out = model(x)
```

## 6 · vs 别的训练精度
| 精度 | speedup vs fp32 | 训推一致? |
|------|----------------|----------|
| bf16 | 2x | bf16→bf16 |
| fp16 | 2x | fp16→fp16 |
| **fp8** | **3-4x** | **fp8→fp8** ⭐ |

## 7 · 影响
2025-2026 中小开源模型大量用 FP8 训练（DeepSeek-V3 引爆）。推理时直接拿 FP8 ckpt 不必再量化。

## 8 · 实现
本课为概念课，回顾 Module 3 scaling-infra 的 FP8 training 章节。
