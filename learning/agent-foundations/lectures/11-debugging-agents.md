# L11 · Agent 调试

## Agent 调试与普通代码不同

| 维度 | 普通代码 | Agent |
|------|---------|-------|
| 决定 | 你写 | LLM 决 |
| 复现 | 100% | LLM 采样可能不同 |
| Bug 位置 | stack trace | prompt / tool / state |
| 修法 | 改代码 | 改 prompt + few-shot + tool desc |

## 5 大调试维度

| 维度 | 工具 |
|------|------|
| **Trace** | 全 thought-action-obs 记录 |
| **Replay** | 同 seed + tool mock 重放 |
| **Token usage** | per-call breakdown |
| **Tool errors** | 抽 error trace |
| **HITL** | 中途人工介入 |

## Trace 格式

```json
{
  "trace_id": "abc",
  "steps": [
    {"step": 1, "thought": "...", "action": "search('X')", "obs": "..."},
    {"step": 2, "thought": "...", "action": "calc(3*4)", "obs": "12"}
  ],
  "final": "12",
  "tokens": {"in": 800, "out": 120},
  "duration_ms": 4321
}
```

## 商业 trace 工具

| 工具 | 团队 |
|------|------|
| **LangSmith** | LangChain |
| **W&B Weave** | Weights & Biases |
| **Helicone** | YC |
| **Phoenix** | Arize |
| **Langfuse** | OSS |

## Replay 模式

```python
# 录制
recorder.start()
react_loop(question, llm, tools)
recorder.save("trace.json")

# 重放（tool mock 用录制结果）
mock_tools = recorder.load_tools("trace.json")
react_loop(question, llm, mock_tools)
```

→ 同 LLM seed 下，调 prompt 即可对照效果。

## 5 个最常见 bug

| Bug | 症状 | 修 |
|-----|------|----|
| Tool format | LLM 写错 action name | 加 few-shot 示范 |
| Infinite loop | step > max | 加重复检测 |
| Tool error 静默 | 失败但 LLM 继续 | observation 写明 error |
| Sycophancy | 同意用户错的 | system prompt 加 critical |
| Hallucination | 无 tool 时编 | strict reflection step |

## HITL 在调试中的角色

```python
def run_with_hitl(graph, state):
    while not state.done:
        state = graph.step(state)
        if state.needs_review:
            user_input = input(f"Approve? {state.last_action}")
            state.user_decision = user_input
```

→ 关键步骤前停下来等人确认。

## 我们手写版（`tracing.py` 预告）

```python
@dataclass
class Trace:
    steps: list
    tokens_in: int = 0
    tokens_out: int = 0
    def add(self, step): self.steps.append(step)
    def to_md(self): ...
    def save(self, path): json.dump(...)
```

## 退出条件

- 能列 5 大调试维度
- 能写一个 Trace 类
- 知道 LangSmith / W&B Weave 至少一个

## 一句话

> Agent debug = trace + replay + HITL —— LangSmith / W&B Weave / Phoenix 5 家选一。
