# L03 · Medusa（Cai et al., Princeton 2024）

## 1 · 痛点
独立 draft model 不够好 — Llama-7B 要找另一个能力相当的小 draft 不容易。

## 2 · 核心 idea
**让 target model 自己当 draft**：在最后一层加几个**额外的 LM head**，每个 head 预测第 i 个未来 token。

```
shared backbone
       ↓
    [last hidden h]
    /   |    |    \
  head0 head1 head2 head3   ← Medusa heads
   ↓     ↓     ↓     ↓
   y_1   y_2   y_3   y_4
```

- head_i 训练目标：`logits_i = head_i(h)` → next-i token

## 3 · 训练
- 冻结 backbone
- 训练 `M = 4-5` 个 head（每个一个 MLP）
- 数据：原 SFT 数据
- LoRA-rank ≈ 0.1 → 训快、占小

## 4 · 推理：tree-based verification
每步：
1. forward backbone 得 h
2. 各 head 同时输出 top-k 候选
3. 形成"candidate tree"（笛卡尔积裁剪）
4. tree attention 一次 verify 所有路径

## 5 · 收益
| 模型 | 加速 |
|------|------|
| Llama-7B | 2.2-2.8x |
| Vicuna-13B | 2.3x |
| Llama-3 8B | 2x |

## 6 · 与 classic 对比
| 维度 | classic | Medusa |
|------|---------|--------|
| draft 来源 | 独立小模型 | 同模 + heads |
| 训练成本 | 训 + 部署两模型 | 只训 heads |
| accept rate | 0.5-0.8 | 0.6-0.85 |
| 显存 | 大 + 小 | 大 + 小 head |

## 7 · 缺点
- head 数固定，不能动态调整
- tree 太大 → verify 也变慢
- 长尾 token 学不好

## 8 · 实现：[medusa_heads.py](../src/medusa_heads.py)
- 4 个 head 简化模型
- top-k 候选 + tree verify
