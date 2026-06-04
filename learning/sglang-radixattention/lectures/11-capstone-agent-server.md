# L11 · Capstone — Agent 推理服务

## 1 · 目标
- 用 radix tree + grammar FSM + frontend DSL 跑一个**真实** agent 服务
- 量化 radix 命中率 / jump-forward 节省 / 调度开销

## 2 · 任务
ReAct agent + 4 个 mock tool（search / calc / weather / python）

```python
@function
def agent(s, query):
    s += SYSTEM_PROMPT + f"\nQ: {query}\n"
    for step in range(5):
        s += f"Thought {step}: "
        s += Gen(f"t{step}", stop=["\n"])
        s += f"\nAction {step}: "
        s += Gen(f"a{step}", stop=["\n"])
        if "Final" in s.vars[f"t{step}"]:
            break
        obs = TOOL_RUNNER.run(s.vars[f"a{step}"])
        s += f"\nObservation: {obs}\n"
```

## 3 · 跑 32 并发 query
- system_prompt 共享 → radix tree 命中率 ≥ 70%
- 每个 query 同一 SYSTEM_PROMPT prefix（2000 token）
- 真实 agent：tool 输出后续 prompt 也常有重叠

## 4 · 指标 + 退出条件
- radix hit_rate ≥ 70% ✓
- jump-forward 在 `Action:` `Thought:` 上触发 ≥ 10 次 / agent
- 总 forward 数 < 朴素 chain 实现的 30%

## 5 · 与 vLLM baseline 对比
- baseline = 同 query 走 OpenAI API 5 次（multi-turn 无 radix）
- 期望 SGLang 模式吞吐 3-5x

## 6 · 实现：[agent_server.py](../src/agent_server.py)
- 集成 radix_tree + frontend_lang + agent_patterns
- 32 并发模拟
- 输出 metrics JSON

## 7 · 一句话
> agent 服务的本质 = **prompt 拼起来 → 让 radix tree 自然共享 → 让 grammar FSM 自然结构化**。
