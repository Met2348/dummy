# L01 · 评测毕业 = 系列大考

## 25-topic 全图

```
Module 1 PEFT (3)        ┐
Module 3 造大模型 (8)     ├─→ Module 6 评测/安全 (本 capstone)
Module 4 改大模型 (7)     │       ↓
Module 5 用大模型 (7)     ┘  25-topic Portfolio ⭐
```

合计 25 prior + Topic 7 (eval-graduation) + Module 6 前 6 = **32 专题完结**。

## 评测毕业的 3 大 Capstone

| Capstone | 内容 | 代码 |
|---|---|---|
| **1: mini-HELM** | 5 ckpt × 4 维 score matrix | `mini_helm.py` |
| **2A: mini-Arena** | 5 ckpt round-robin + BT-Elo | `mini_arena.py` |
| **2B: 红队矩阵** | 5 ckpt × 3 攻击 ASR | `mini_red_team.py` |
| **2C: 防御加固** | 同矩阵 + classifier 兜底 | `mini_defense.py` |
| **3: Portfolio** | 一份完整 README | `portfolio.py` |

## 5 个 ckpt 元数据

| ckpt | 出处 | 推理 | 安全 |
|------|------|------|------|
| `vanilla` (124M) | Module 3 baseline | none | weak |
| `lora` (124M) | Module 1 PEFT | brief | medium |
| `dpo` (124M) | Module 4 RLHF | yes | strong |
| `r1_tiny` (124M) | Module 4 R1 | strong | medium |
| `phi_tiny` (270M) | Module 3 pretrain | clean | strong |

## 本 Topic 14 lecture

L01-L02：定位 + ckpt 全集
L03-L08：mini-HELM / mini-Arena / red-team / defense / cost
L09-L11：portfolio 设计 + blog README + 决策树
L12: Capstone-1 mini-HELM
L13: Capstone-2 mini-Arena + 红队 + 防御
L14: Capstone-3 Portfolio README ⭐⭐⭐

## 一句话

> Module 6 收官 = 25 专题学完后的"出门作品集"。
