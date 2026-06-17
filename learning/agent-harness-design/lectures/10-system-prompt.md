# L10 · System Prompt 工程

## harness 拥有 system prompt

12-Factor Agents 第一条:**own your prompts**。system prompt 是 harness 对模型的"标准指令",决定 agent 的人格、能力边界、行为约束。它是源码,该版本管理,不该被框架藏起来。

## 一个 system prompt 装什么

```
[角色]      你是谁、目标是什么
[工具清单]   有哪些工具、各自描述(模型据此选用)
[环境上下文] cwd / OS / 用户 / 时间 等运行时事实
[行为约束]   何时调工具、何时收尾、安全红线
```

## src 走读

[system_prompt.py](../src/harness/system_prompt.py):

```python
def build_system_prompt(role, tools, env=None):
    lines = [role, "", "## Tools"]
    for t in tools:
        ro = " (read-only)" if t.read_only else ""
        lines.append(f"- {t.name}: {t.description}{ro}")   # 工具描述注入
    if env:
        lines += ["", "## Environment"] + [f"- {k}: {v}" for k,v in env.items()]
    lines += ["", "## Behavior",
              "- Call a tool when you need info/action; otherwise answer directly.",
              "- When complete, reply with a final answer and no tool call."]
    return "\n".join(lines)
```

[mini_harness.py](../src/mini_harness.py) 在每次 `run` 开头把它作为第一条 system 消息塞进 context。

## 设计要点

1. **工具说明动态生成**:工具增删,system prompt 自动跟着变(别手写一份会漂移的清单)。
2. **环境注入**:cwd/OS/时间等"模型不知道的运行时事实"要喂进去——本仓库的 environment 块就是例子。
3. **行为约束要明确"何时收尾"**:否则模型可能无谓地继续调工具或过早停。
4. **read-only 标注顺带给模型看**:它能感知哪些是"安全"操作。

## 和 context 的关系

system prompt 是 context 里**最该被保护的部分**——L06 的 compaction 永远不动 system 消息。它被压掉 = agent 失忆 + 丢工具。

## 与"用大模型"的呼应

prompt 本身的写法(few-shot、CoT 触发、格式约束)在 prompt-tuning / 各 serving 专题讲过;本节关注的是 **harness 如何系统化地构造和注入** system prompt,而不是 prompt 的措辞技巧。

## 退出条件
- [ ] 列出 system prompt 的四大块
- [ ] 理解工具清单为何要动态生成
- [ ] 记住 system prompt 在 compaction 里要保护
