# L09 · Portfolio 设计

## 目标

```
25 专题学习 → 1 份 markdown Portfolio → 出门作品集
```

## 9 大组件

1. **标题 + 日期** — 收官时间戳
2. **25-topic 时间线** — 4 大 module 列表
3. **5 ckpt 元数据表** — 出处 + 能力画像
4. **Capstone-1: mini-HELM** — 4 维 score
5. **Capstone-1.5: ASCII radar** — 视觉
6. **Capstone-2A: mini-Arena** — Elo 排行
7. **Capstone-2B: 红队 ASR** — 5 ckpt × 3 attack
8. **Capstone-2C: 防御加固** — ASR 降低对照
9. **选型决策树** — 场景 → ckpt
10. **"我能做什么"画像** — 简历用

## blog 风格原则

- 一句话 hook（开头）
- 视觉表格 > 文字段落
- ASCII 图（雷达 / 表）
- 每节小结
- 每节附实操命令

## 跑 portfolio

```python
from portfolio import gen_portfolio, write_portfolio

# 直接 print
print(gen_portfolio())

# 写文件
write_portfolio("portfolio.md")
```

## 接 Linkedin / 简历

```
"完成 32 专题 LLM 学习系列（GitHub: ...）：
- 造 (Module 3): 8 专题, 从 0 训 GPT-2 / Phi-tiny
- 改 (Module 4): 7 专题, RL / PPO / DPO / R1-Zero
- 用 (Module 5): 7 专题, vLLM / SGLang / 量化 / 分布式
- 评 (Module 6): 7 专题, MMLU / HumanEval / Arena / 红队 / 安全
- PEFT (Module 1): 3 专题, LoRA / Prompt / Adapter
共 100+ 主要论文复现 + 教学 mock + 双 capstone graduation"
```

## 一句话

> Portfolio = 你的"LLM 全栈工程师 ID 卡"，1 份 markdown 全说清。
