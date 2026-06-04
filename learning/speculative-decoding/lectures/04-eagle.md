# L04 · EAGLE（Li et al., THU 2024）

## 1 · 进一步：feature-level draft
Medusa 让 heads 预测 token id。EAGLE 让 heads 预测**下一层的 feature (hidden state)**：

```
Medusa:  h → head_i → logits_i → token_i
EAGLE:   h_t, token_t → draft_layer → h_{t+1} → LM_head → token_{t+1}
```

draft_layer 是一个**单 transformer 层**，输入 (last hidden + current token embed)。

## 2 · 为什么更好
- feature 比 token 信息丰富（dense vector vs sparse id）
- 一个 draft_layer 可以**链式自回归** k 步
- 不像 Medusa 各 head 独立 → EAGLE 后续 token 利用前一 token 的 prediction

## 3 · 训练
- 冻结 backbone
- 训练 1 个 transformer layer（draft layer）
- LM head 复用 target 的
- 数据：用 target 自己生成的 trajectory

## 4 · 收益（Llama-7B 基准）
| 方法 | 加速 |
|------|------|
| Medusa | 2.5x |
| **EAGLE** | **3.0x** |

## 5 · accept rate 数字
- accept k=3 时 ≈ 4.5 token / iter
- accept k=4 时 ≈ 5.5 token / iter

## 6 · 缺陷
- 单 path draft → 浪费机会
- 树宽固定

→ EAGLE-2 解：dynamic tree

## 7 · 实现：[eagle_minimal.py](../src/eagle_minimal.py)
- 单层 draft model
- feature 自回归 k=4 步
- LM head 复用
