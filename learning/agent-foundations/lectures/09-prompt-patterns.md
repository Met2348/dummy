# L09 · System Prompt Patterns

## Agent system prompt 5 大组件

```
1. ROLE       「你是 X」
2. CONSTRAINTS「不能 Y, 必须 Z」
3. TOOLS      「可用 tool 列表 + schema」
4. EXAMPLES   「示范 thought-action-obs 一轮」
5. OUTPUT     「输出格式 schema」
```

## 模板

```
You are an expert {role}. Help user achieve {goal}.

Constraints:
- Never call >5 tools per question
- Use ONLY provided tools
- If uncertain, ask clarifying question

Tools:
{tool_schemas}

Format:
Thought: ...
Action: tool_name(args)
Observation: ...
... (repeat)
Final Answer: <answer>

Example:
Q: What's 2025 + 3?
Thought: simple arith, use calculator
Action: calculator(2025+3)
Observation: 2028
Final Answer: 2028
```

## ROLE 设计

| Role | 效果 |
|------|------|
| "expert researcher" | 写更长 thought |
| "code assistant" | 偏写代码 |
| "concise QA" | 短输出 |
| "Socratic tutor" | 反问 |

**注意**：role 不是魔法，过度依赖会 sycophancy。

## CONSTRAINTS 反例

| 写法 | 问题 |
|------|------|
| "Don't be wrong" | 没意义 |
| "Be safe" | 模糊 |
| "Don't say bad words" | 容易 jailbreak |

## CONSTRAINTS 正例

| 写法 | 优势 |
|------|------|
| "Max 5 tool calls" | 量化 |
| "If unclear, ask user" | actionable |
| "Refuse if violates X policy" | 引到 policy |

## 输出格式：JSON vs Markdown vs XML

| 格式 | 优势 | 用 |
|------|------|---|
| JSON | 易解析 | 程序消费 |
| Markdown | 易读 | 用户看 |
| XML | 嵌套强 | Anthropic best practice |

Anthropic 官方推荐 XML tag 包裹推理：
```
<thinking>...</thinking>
<answer>...</answer>
```

## Few-shot examples 注意

- **1-3 个就够**（更多反而 overfit）
- **覆盖边界 case**（错误 / 拒绝 / 多步）
- **示范完整循环**（thought-action-obs-final）

## "Prompt as code" 工程化

| 实践 | 工具 |
|------|------|
| Version control | git |
| A/B test | LangSmith / promptfoo |
| Eval | RAGAS / G-Eval |
| Templating | Jinja / f-string + 占位 |

## 退出条件

- 能列 5 组件
- 能写一个 ReAct system prompt
- 知道 Anthropic XML tag 实践

## 一句话

> Prompt 是 agent 的 OS —— ROLE + CONSTRAINTS + TOOLS + EXAMPLES + OUTPUT 5 组件，工程化对待。
