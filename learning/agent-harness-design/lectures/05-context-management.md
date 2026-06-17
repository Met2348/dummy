# L05 · Context 管理 — 窗口即预算

## 核心心智

长跑 agent 里,context window 是**最稀缺的资源**。每多塞一条消息,就花一份 token、慢一点、且稀释信号。harness 必须主动管理"窗口里装什么"。

## src 走读

[context.py](../src/harness/context.py) 的 `ContextWindow`:

```python
@dataclass
class ContextWindow:
    budget: int
    messages: list
    def used(self):       return sum(est_tokens(m["content"]) for m in self.messages)
    def over_budget(self): return self.used() > self.budget
    def headroom(self):   return self.budget - self.used()
```

loop 每回合开头检查预算,超了就 `compact()`(详见 L06)。

## 进窗口前的三道闸

| 闸 | 干什么 | 在哪 |
|----|--------|------|
| **裁剪** | 大工具输出留头尾、丢中段 | 工具结果回灌前 |
| **预算核算** | 知道当前用了多少、还剩多少 | `used()` / `headroom()` |
| **compaction** | 逼近上限时压缩旧轮 | `compact()`(L06) |

## 工具输出裁剪(回顾 design-patterns)

design-patterns 专题 [context_engineering.py](../../agent-design-patterns/src/context_engineering.py) 的 `trim_tool_output` 演示了头尾保留法:

```
"<前半>…[trimmed ~N tokens]…<后半>"
```

读大文件、跑长命令时,这一步是省窗口的关键。harness 应在工具结果**回灌前**自动套用。

## 消息分层:不是所有消息一样贵/一样重要

| 层 | 例子 | 处置 |
|----|------|------|
| system | 角色、工具说明 | 永远保留(L06 compact 不动 system) |
| 近因 | 最近几轮 | 逐字保留 |
| 远因 | 久远轮次 | 压成摘要 |
| 工具大输出 | 文件/命令 | 裁剪后存,或 offload 到 memory |

## 与 long-context 专题的分工

- `long-context`(造模型):把窗口**做大**(RoPE/YaRN/Ring Attention)。
- 本节(harness):在**给定窗口**里精打细算。

窗口再大也有上限,且越满越贵越慢——所以两者都要。

## 退出条件
- [ ] 把"窗口即预算"当默认心智
- [ ] 说清进窗口前的三道闸
- [ ] 理解消息分层与各自处置策略
