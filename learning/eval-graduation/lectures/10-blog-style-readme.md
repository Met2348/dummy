# L10 · blog 风格 README

## 反例：枯燥 README

```
This repo contains 25 topics on LLM engineering.
Topic 1: prompt tuning
Topic 2: lora
...
```

→ 简历 / 招聘官扫一眼就 close。

## 正例：blog 风格

```markdown
# 32 专题 LLM 学习马拉松

> 6 月底跑完，从 GPT-2 fine-tune 到 R1-Zero 复现 + 5 ckpt 红队评测

## 你能学到什么

1. **造模型**：从 0 预训 GPT-2 / Phi-tiny ...
2. **改模型**：LoRA / Adapter / DPO ...
...

## 5 ckpt 对比

[包含 mini-HELM 4 维表]

## 选型决策树

[包含决策树代码]

## 你最该读的 5 个 topic（如果只看 1 周）

1. reasoning-r1 — R1-Zero aha moment 双轨
2. dpo-family — 7 个 PO 变体 + Rainbow
3. quantization-deploy — AWQ/GPTQ/SmoothQuant
4. constitutional-classifiers (in safety-defense)
5. eval-graduation — 这个，看作品集
```

## 7 大 blog 元素

| 元素 | 干嘛 |
|------|------|
| **Hook** | 1 句话点题 |
| **结构 ToC** | 易扫 |
| **示例 / 实操** | 代码 + 输出 |
| **图表 / ASCII** | 视觉冲击 |
| **决策树 / FAQ** | 实用价值 |
| **Tag / Badge** | 信任信号 |
| **CTA** | clone / star / contact |

## SEO 友好

- 标题含关键词：`LLM`, `PEFT`, `RL`, `R1-Zero`, `Constitutional Classifiers`
- 描述前 160 char 关键
- 加 `topic` tags：`llm`, `peft`, `rlhf`, `dpo`, `r1-zero`, `red-team`

## 一句话

> README 不是文档，是宣言书 — blog 风格 + 视觉 + CTA。
