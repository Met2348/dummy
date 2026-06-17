# L09 · 权限 / 审批系统

## 为什么这是 harness 最关键的安全层

agent 能调工具 = 能影响真实世界(写文件、删数据、转账、发邮件)。权限系统是**意图与副作用之间的闸门**。没有它,一个跑偏的 agent 能造成不可逆破坏。

## src 走读

[permissions.py](../src/harness/permissions.py) 的三种模式:

```python
class PermissionManager:
    def _decide(self, name, read_only, args) -> Decision:
        if name in self.deny:  return Decision("deny", "deny-list")
        if name in self.allow: return Decision("allow", "allow-list")
        if self.mode == "auto":     return Decision("allow", "auto")
        if self.mode == "readonly":
            return Decision("allow","read-only") if read_only \
                else Decision("deny","writes blocked in readonly mode")
        if self.mode == "ask":
            return Decision("allow","approved") if self.ask_handler(name,args) \
                else Decision("deny","user declined")
```

| 模式 | 行为 | 适用 |
|------|------|------|
| `auto` | 除 deny-list 全放行 | 信任环境 / 沙箱内 |
| `readonly` | 只放行 read-only 工具 | 探查、不可改 |
| `ask` | 交给 ask_handler 决定 | 人审高风险操作 |

allow-list / deny-list 是横切快捷:无论哪种模式,deny-list 一票否决、allow-list 一票放行。

## loop 里的拦截点

[loop.py](../src/harness/loop.py) 在 dispatch **之前**检查:

```python
d = permissions.check(tool, tc.args)
if d.action == "deny":
    context.add("tool", {"ok": False, "error": f"permission denied: {d.reason}"}, ...)
    continue                          # 拒绝也 surface 成结果,模型能看到并反应
```

关键:**拒绝不是静默丢弃,而是 surface 成工具错误**。capstone 的 readonly 跑就展示了——agent 看到 "permission denied",优雅地报告"算出来了但没能保存",而不是假装成功。

## 设计要点

1. **默认最小权限**:生产里别 `auto`,从 `readonly` / `ask` 起步。
2. **危险操作永远过闸**:写/删/转账/外发,即使 auto 也该进 deny 或 ask。
3. **拒绝要可见**:surface 给模型 + trace,不静默(回顾 design-patterns silent-failure)。
4. **sandbox 是补充不是替代**:权限管"准不准",sandbox 管"就算调了能波及多大"。

## 与设计层的呼应

design-patterns 专题 [design-checklist](../../agent-design-patterns/lectures/13-design-checklist.md) 的"危险操作有护栏 / human-in-the-loop"——就落在这一层实现。

## 退出条件
- [ ] 说清三种权限模式 + allow/deny-list 的优先级
- [ ] 理解拒绝为何要 surface 而非静默
- [ ] 记住"默认最小权限 + 危险操作必过闸"
