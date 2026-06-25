# L01 · Tool → Analyst → Scientist：三级自主性阶梯

## 1. 为什么需要一把尺子

2025–26 年，"AI 做科研"的系统井喷，名字一个比一个大：AI Scientist、co-scientist、
AI-Researcher、NovelSeek……如果你按**名字/自称**理解它们，会得出"科研已经被自动化了"的错觉。
综述 [A Survey on LLM-based Agents for Science (2505.13259)](https://arxiv.org/abs/2505.13259)
给了一把更冷静的尺子——按**自主程度**分三级：

| 级别 | 一句话 | 谁定问题 | 典型动作 |
|------|--------|---------|---------|
| **Tool** | AI 是工具，人主导每一步 | 人 | 检索、翻译、写一段代码、画一张图 |
| **Analyst** | AI 自动跑通"实验→分析"，但**问题/假设由人给** | 人 | 给定 idea，自动设计实验、跑、出结论 |
| **Scientist** | AI **自己定问题**并闭合"创意→实验→分析"整条链 | AI | 自生成 idea、自跑、自写、自评 |

> 关键分水岭是**两个**，不是一个：① 闭合了多少环；② **问题是谁定的**。
> 一个能自动跑完整实验流水线、但 idea 是人喂的系统，再强也只是 **Analyst**。

## 2. 我们把这把尺子变成代码

`classifier.py` 的 `evidenced_level()` 把上表写成可执行规则：

```python
closes_loop = has_ideation and (experiment 且 analysis 都自动)
if closes_loop and not human_sets_problem:   return "scientist"
if has_core_exec or len(自动化的环) >= 3:     return "analyst"
return "tool"
```

它**只读证据字段**（`automates / human_sets_problem / independent_verification`），
不读系统自称的 `claimed_level`。于是"自称"与"证据"被强行拆成两栏——这正是下一讲要榨取的张力。

## 3. 试一个边界

`Darwin Gödel Machine` 自称能"自己改进自己"（听着像 Scientist）。但证据上：它在**给定的
benchmark** 上进化 agent 代码 —— 问题是人定的。按尺子，它是 **Analyst**。
跑 `python src/run.py --system "Darwin Godel Machine"` 看判定理由。

> 这不是贬低它——Analyst 已经很强。是提醒你：**"自我改进"≠"自己定研究问题"**。
> 把这两件事混为一谈，是这个领域最常见的话术滑坡。

## 4. 动手

1. `python src/run.py --show map`，对着三行（scientist/analyst/tool）想：
   为什么 Scientist 那行 `exp/ana/wri` 列都满，`rev`（评审）却也有——它评的是**谁的**论文？（埋给 L03）
2. 找一个你最近读到的系统，口头给它的三个证据字段打分，预测它落在哪级，再加进 `systems.py` 验证。
