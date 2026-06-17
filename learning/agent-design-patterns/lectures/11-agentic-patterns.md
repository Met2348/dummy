# L11 · Agentic Patterns 全景 + 12-Factor Agents

## 三套互补的"模式语言"

业界对 agent 设计有几套常被引用的归纳,角度不同但互补。

### A) Anthropic《Building Effective Agents》— 本专题主线
5 workflow + 1 agent(L04-L09)。强调**从简单往复杂爬**。

### B) Andrew Ng 4 大 agentic design patterns
更偏"能力维度":

| 模式 | 含义 | 对应本仓库 |
|------|------|-----------|
| **Reflection** | 自我批评再改 | evaluator-optimizer(L08)/ `agent-foundations` Reflexion |
| **Tool Use** | 调外部工具 | augmented LLM(L03)/ `tool-use-mcp` 专题 |
| **Planning** | 先规划再执行 | orchestrator(L07)/ `agent-foundations` Plan-Execute |
| **Multi-Agent** | 多角色协作 | `multi-agent-orchestration` 专题 |

### C) CoALA(认知架构)
把 agent 拆成:memory(工作/情景/语义)+ action space(内部推理 vs 外部工具)+ decision loop。是更学术的统一框架,呼应 `agent-memory-context` 专题。

## 12-Factor Agents(工程纪律)

借鉴 12-Factor App,Dexter Horthy 提出让 agent **可靠、可维护**的工程原则,精选几条最该记的:

| Factor | 说的事 | 为什么重要 |
|--------|--------|-----------|
| **Own your prompts** | prompt 是源码,纳入版本管理 | 别让框架把 prompt 藏起来 |
| **Own your context window** | 自己掌控塞什么进 context | 即 L10 的 context engineering |
| **Tools are structured outputs** | 工具调用本质是结构化输出 | 用 schema 约束,别靠解析自由文本 |
| **Stateless reducer** | agent = (state, event) → new state | 可重放、可测试、可恢复 |
| **Small, focused agents** | 小而专,别造全能体 | 呼应 L02"start simple" |
| **Explicit control flow** | 关键路径用代码,不全交给模型 | workflow > agent 的工程化表达 |

> 12-Factor 的精神和本专题一致:**能用确定性代码控制的地方,就别交给模型的自由发挥**。

## 怎么用这三套

- 设计时先用 **Anthropic 阶梯**决定 workflow/agent;
- 用 **Ng 4 模式**检查能力是否齐(要不要 reflection?要不要 planning?);
- 用 **12-Factor** 做工程化落地的 checklist(L13 会汇成清单)。

## 退出条件
- [ ] 能把 Ng 4 模式映射到本仓库对应专题
- [ ] 记住 12-Factor 里至少 3 条并解释
- [ ] 理解三套模式语言不是竞争而是互补
