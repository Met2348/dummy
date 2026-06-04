# L08 · ARC-AGI — Chollet 真智力测试

## 背景

François Chollet 2019 发起 **Abstraction and Reasoning Corpus**：
- 反对"靠记忆/规模 scale 出来的智力"
- 测真 generalization：每题都是新颖 task

## 数据

- **400 训练 + 400 公开测试 + 100 私有测试 (V1)**
- 每题：3-5 个输入→输出 grid 示例 + 1 个测试输入
- 模型需推断 transformation 规则，输出 grid

```
Example 1:  Input   Output
            ░░▓     ▓░░
            ░▓▓     ▓▓░
            ▓▓▓     ▓▓▓

Example 2:  ...

Test:       Input  → ?
            ░▓░    
            ▓▓░    
            ▓▓▓
```

## 评测

- **2 attempts**（每题最多 2 个 guess）
- **exact match** of output grid
- 没 partial credit

## 分数史

| 时间 | 模型 | ARC-AGI |
|------|------|---------|
| 2019 | brute-force solver | ~30% |
| 2023 | GPT-4 | ~14% |
| 2024 | Claude 3.5 Sonnet | ~21% |
| 2024.06 | MindsAI ensemble | ~55% |
| **2024.12** | **o3 (low compute)** | **75.7%** |
| **2024.12** | **o3 (high compute)** | **87.5%** |
| 人类 | ~85% (avg adult) |

→ **o3 是第一个超人类的 LLM**，但代价是 $1000+/题 算力。

## ARC-AGI-2 (2025.03)

V1 被 o3 攻破后，Chollet 发布 V2：
- 同样规则，新题目
- 更难推断
- 2025 主流模型 < 5%

## 为什么难

ARC 要求：
1. **few-shot pattern recognition**（3-5 demo）
2. **空间/几何推理**
3. **组合规则**（旋转 + 颜色映射 + 计数）
4. **抗 prior**：每题独一无二

不能靠"在 corpus 里见过类似题"。

## 实操

我们 src 没专门 runner（grid eval 工程量大）。
推荐方法：
- 真要做用 `arckit` 库
- 或读 François 的 toy notebook

## 一句话

> ARC-AGI 是 Chollet 给 LLM 立的"AGI 标尺"，o3 第一次跨过，但 V2 重启。
