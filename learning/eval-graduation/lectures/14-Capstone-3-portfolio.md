# L14 · Capstone-3 ⭐⭐⭐⭐⭐⭐ — Portfolio 出门作品集

## 这是 Module 6 收官 capstone

不止 1 个 markdown 表，是**整个学习系列的"毕业作品"**。

## 跑

```python
from portfolio import write_portfolio
path = write_portfolio("portfolio.md")
print(f"Portfolio written to {path}")
```

## Portfolio 长这样

```markdown
# 32-topic LLM Learning Portfolio

> 2026-06-05 · Module 6 收官，全系列完结

## 25 专题时间线
### Module 1 PEFT
  1. `prompt-tuning`
  2. `lora`
  3. `adapter-tuning`
### Module 3 造大模型
  4. `data-curation`
  ...
### Module 6 评测/安全
  ...
  32. `eval-graduation`

## 5-ckpt 元数据表
[5 行 ckpt 表]

## Capstone-1: mini-HELM
[5×4 score 表]

## mini-HELM 雷达 (r1_tiny)
[ASCII radar]

## Capstone-2A: mini-Arena
[Elo 排行表]

## Capstone-2B: 红队 ASR
[5×3 红队矩阵]

## Capstone-2C: 防御加 ASR 降低
[防御对照表]

## 选型决策树
[5 段决策树]

## 我能做什么
[5 大画像 + 简历用]
```

## 整个 portfolio 大约 200 行 markdown，5 分钟可读完。

## 用法

1. **简历附件**：直接当"个人 LLM 工程师 ID 卡"
2. **GitHub README**：repo 主页 markdown
3. **面试讲解**：每节 1 分钟，10 分钟总览
4. **Linkedin post**：摘要 + 链接

## 关键文献

- HELM (Stanford 2022)
- Chatbot Arena Elo (LMSYS 2024)
- HarmBench (CMU 2024)
- Constitutional Classifiers (Anthropic 2025)
- ...

## "我能做什么"画像

```
你已具备：

1. 造模型 — 从 0 训 GPT-2 / Phi-tiny
2. 改模型 — LoRA / Adapter / DPO / R1-Zero
3. 用模型 — vLLM / SGLang / 量化 / 分布式
4. 评模型 — 25 bench × 多 judge × Arena
5. 守模型 — 红队 + 4 层防御 + Constitutional Classifiers
6. 评 4 主轴 — knowledge/reasoning/safety/efficiency

= 2026 年的 LLM 全栈工程师
```

## 退出条件

- portfolio.md 文件生成（约 200 行）
- 包含 32 个 topic enumerated
- 含 Capstone 1/2A/2B/2C 全部表
- 含决策树 + "我能做什么"
- 文件可 push 到 GitHub 当 README

## git tag 收官

```bash
git tag 评-graduation
git tag module6-complete
```

最后 2 个 tag 完结整个 Module 6（也是整个学习系列）。

## 一句话

> Capstone-3 ⭐⭐⭐⭐⭐⭐ = 把 32 专题写成 1 份 portfolio.md — 你的"LLM 工程师 ID 卡"。
