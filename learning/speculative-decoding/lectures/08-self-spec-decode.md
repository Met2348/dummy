# L08 · Self-Speculative Decoding

## 1 · 思路
用**同一个模型**作 draft，只是 forward 时跳过某些 layer。

跳过策略：
- **Skip layer**: 跳过 16/32 层 → draft 比 target 快 2x
- **Skip head**: 减 attention head
- **Skip MoE expert**: 只激活 top-1 expert

## 2 · 优势
- 完全无需训 draft / Medusa head / EAGLE layer
- draft 与 target 共享所有权重 → 无额外显存

## 3 · 缺点
- accept rate 显著低于 EAGLE（layer skip 引入 bias）
- 加速 1.5-2x，不如 EAGLE 的 3-5x

## 4 · 何时用
- 内存极度紧张
- 不想训练
- 接受较低 accept rate

## 5 · 与 chained skipping
- 不是固定 skip → 自适应 skip
- 用某些 layer 的"重要性"动态决定

## 6 · LayerSkip (Meta 2024.04)
- 训练 layer dropout → 模型本身可跳层
- 推理时 self-spec 加速 1.86x

## 7 · 实现 stub
教学：mock_skip_forward 略
