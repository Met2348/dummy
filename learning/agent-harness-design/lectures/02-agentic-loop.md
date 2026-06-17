# L02 · Agentic Loop — 引擎的心脏

## 控制流

harness 的本质就是这个循环:

```
while 没结束 and 还有回合:
    (必要时压缩 context)
    resp = model(messages)
    if resp 是 end_turn:  return 文本        # 模型说完事了
    for 每个 tool_call:
        权限检查 → dispatch → 把结果写回 context
    (loop guard:无进度就停)
```

这就是 Anthropic / OpenAI 工具使用协议的运行时实现。

## src 走读

[loop.py](../src/harness/loop.py) 的 `run_loop` 把所有组件串起来:

```python
for turn in range(max_turns):
    if context.over_budget(): context.compact()          # L05-06
    resp = model.respond(context.messages)               # model.py
    tracker.add_model(tin, resp.out_tokens())            # L13
    if resp.stop_reason == "end_turn":
        return resp.text                                  # 完成
    for tc in resp.tool_calls:
        d = permissions.check(tool, tc.args)              # L09
        if d.action == "deny": <写入拒绝结果>; continue
        result = registry.dispatch(tc.name, tc.args)     # L03
        context.add("tool", result, name=tc.name)        # 结果回灌
    if guard.record(signature): return None              # L12 loop guard
return None                                                # 撞 max_turns
```

## 三个必须答对的设计点

1. **停机**:模型 `end_turn`、`max_turns` 上限、loop guard 无进度——三道闸缺一不可。`_self_test` 验证了三种退出路径(正常完成 / 权限阻断后完成 / guard 触发返回 None)。
2. **回合 = 一次 model 调用**:tool_use 不算结束,结果回灌后**再调一次模型**才推进。这就是为什么 agent 调用数不可预测(L9 的成本现实)。
3. **每步都要可观测**:轨迹涌现,必须 trace,否则无法复盘(L13)。

## 协议数据结构

[model.py](../src/harness/model.py) 定义了边界:

```python
@dataclass
class ToolCall:   name; args; id
@dataclass
class ModelResponse:
    text; tool_calls
    @property
    def stop_reason(self): return "tool_use" if self.tool_calls else "end_turn"
```

`MockModel` 用注入的 `brain(messages)->ModelResponse` 模拟真模型——loop 逻辑因此可在零网络、零权重下被完整测试。

## 退出条件
- [ ] 能默写 agentic loop 的伪代码
- [ ] 说清"一回合 = 一次 model 调用"及其成本含义
- [ ] 记住三道停机闸
