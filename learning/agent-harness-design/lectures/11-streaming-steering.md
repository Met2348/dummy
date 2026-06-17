# L11 · 流式输出与中途打断(Steering)

> 本节是概念 + 在 mini-harness 里的挂载点说明(流式涉及 I/O,本仓库 loop 是同步确定性版,这里讲清"真 harness 怎么做、该插在哪")。

## 两个相关能力

| 能力 | 说的事 |
|------|--------|
| **Streaming** | 模型边生成边吐 token,不等整段完成 |
| **Steering** | 用户在 agent 跑的过程中插话纠偏 |

二者都关乎**交互性**:让用户不必干等一个黑箱跑完。

## 为什么重要

- agent 一回合可能很久(多次 tool 调用)。流式让用户**实时看到进展**(在调哪个工具、想什么)。
- agent 跑偏时,用户能**当场喊停/纠偏**,而不是等它烧完预算才发现。Claude Code 里你打断输入新消息,就是 steering。

## 在 agentic loop 里的挂载点

回看 [loop.py](../src/harness/loop.py) 的循环,steering 插在**每回合边界**:

```
for turn in range(max_turns):
    # ← steering 检查点:有没有新的用户消息?有就插进 context
    resp = model.respond(context.messages)      # ← streaming:逐 token 回调
    ...
    for tc in resp.tool_calls:                   # ← 也可在工具间检查打断
        ...
```

实现要点:
1. **回合边界注入**:把用户中途消息作为新 message 加进 context,下一回合模型就看到了。
2. **协作式中断**:在安全点(回合/工具之间)检查中断标志,而不是硬 kill(避免半成品状态)。
3. **流式是回调**:`model.respond` 在真实现里接收一个 `on_token` 回调,边收边显示。

## 设计要点

| 点 | 说明 |
|----|------|
| 中断要"协作式" | 在安全点停,别 kill 到一半 |
| steering 消息进 context | 当成新 user turn,模型自然纳入 |
| 流式 ≠ 改变控制流 | 它只改"怎么显示",loop 结构不变 |
| 工具执行中也可中断 | 长工具要支持取消 |

## 与本仓库 mini-harness 的差距(诚实说明)

本仓库 loop 是**同步、确定性**的(便于测试),没有真流式/异步打断。但**插入点已经是对的**:回合边界。把 `model.respond` 换成异步流式客户端、在循环顶部加一个"读取待处理用户消息"的步骤,就得到可 steering 的 harness。

## 退出条件
- [ ] 区分 streaming(怎么显示)与 steering(中途纠偏)
- [ ] 指出二者在 agentic loop 的挂载点
- [ ] 理解"协作式中断"为何优于硬 kill
