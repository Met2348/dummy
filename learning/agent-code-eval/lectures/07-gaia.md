# L07 · GAIA — Meta General AI Assistant

**Mialon et al. 2023** · arXiv 2311.12983

## 数据

- **466 真实世界 task** （Lvl 1: 165, Lvl 2: 250, Lvl 3: 50）
- 任务跨：网页搜索 / 文件处理 / 数学 / 逻辑 / 多模态
- **答案唯一**：1 个字符串 / 数字 / 列表
- **需要工具**：search engine / Python / browser / OCR

## 例题（Lvl 1）

```
Q: "Find the population of Tokyo in 2023, then compute 10% of it."
Required tools: web search + calculator
A: "1395000" (1.395M × 10%)
```

## 例题（Lvl 2）

```
Q: "Look at the attached PDF. Find the SECOND-most cited paper in
its references, then tell me the first author's last name."
Required tools: PDF parse + citation lookup
```

## 例题（Lvl 3）

```
Q: "On the Wikipedia page for 'Theory of Everything', under the
'Modern attempts' section, count the number of citation markers
that point to papers published after 2010."
Required tools: web browse + parse + filter
```

## 评测

- exact match（答案唯一）
- 多次尝试不允许

## 分数

| Agent | GAIA score |
|-------|-----------|
| GPT-4 + plugins | 14.6% |
| AutoGPT | 4% |
| **AutoGen + GPT-4o** | **39%** |
| **OpenDeepResearch** | **55%** |
| Open-source | 持续追赶 |
| 人类 | 92% |

## 为什么有用

GAIA 测的就是 **"真实工作流"**：
- 不是单 task，而是**多工具组合**
- 不是 demo prompt，而是**真实办公任务**
- 没有 partial credit，要求**全对**

## ToolUse 关键技术

1. **Plan-Act loop**：先规划工具序列
2. **Reflexion**：失败后回看
3. **Memory**：跨步骤记得已查的事实
4. **多模态**：PDF / 图片必备

## 实操

我们没专门 runner（任务复杂、需要真实 API）。
推荐：用 `smolagents` 或 OpenAI Assistants API 跑官方 GAIA 子集。

## 一句话

> GAIA = 让 LLM 当个能干活的"实习生"。
