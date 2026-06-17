# L03 · Tool 执行层

## 职责

模型只会"说"要调哪个工具(tool_call),**真正执行、收结果、把结果格式化回灌**是 harness 的事。这层要做对四件:

1. **注册**:工具名 → 实现 + 描述 + 元数据
2. **dispatch**:按名查找、传参、调用
3. **结构化结果**:统一 `{"ok", "value"|"error"}` 信封
4. **错误 surface**:异常变成结果,不抛出、不吞掉

## src 走读

[tools.py](../src/harness/tools.py):

```python
class ToolRegistry:
    def dispatch(self, name, args=None) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return {"ok": True, "value": tool(**(args or {}))}
        except Exception as e:                      # surface,不 swallow
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
```

`_self_test` 覆盖:正常返回、未知工具、缺参数(TypeError 被 surface)、批量 dispatch。

## 为什么结果必须是结构化信封

回灌进 context 的是 `{"ok": False, "error": "..."}` 而不是裸异常或空串。这样:
- **模型看得见错误**,能在下一回合反应(重试 / 换路 / 报告)。
- **trace 看得见错误**,能复盘。
- 这正是 design-patterns 专题 [failure-modes](../../agent-design-patterns/lectures/12-failure-modes.md) 里"silent failure"反模式的根治。

## 并行 tool calls

一回合模型可能同时要调多个工具。`dispatch_many` 顺序执行(逻辑并行),返回 `[(call, result)]`。真 harness 会真并发(I/O bound 的工具并行收益大),但回灌顺序要稳定,便于 trace 对齐。

## 设计要点

| 点 | 说明 |
|----|------|
| 信封统一 | 所有工具同一返回形状,loop 无需特判 |
| 未知工具不崩 | 返回 error 信封,模型可改用别的 |
| 参数错误 surface | 让模型看到"你少传了 b",它能补 |
| read_only 标记 | 供权限层判断(L09) |

## 退出条件
- [ ] 说清"模型说、harness 做"的分工
- [ ] 理解结构化信封为何是错误处理的根基
- [ ] 知道并行 dispatch 的收益与回灌顺序约束
