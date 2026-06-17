# L13 · 可观测性 — Trace 与 Cost

## 为什么 agent 必须可观测

agent 的轨迹是**涌现的**——同样的输入,可能走不同的路。出了问题,没有 trace 就只能干瞪眼。可观测性是 agent 工程从"玩具"到"生产"的分水岭。

## src 走读

[tracing.py](../src/harness/tracing.py) 两件套:

```python
@dataclass
class Span:                       # 一条轨迹事件
    kind: str   # system|model|tool|perm|ctx|mem|subagent|loop|done
    label: str
    detail: str = ""

class Trace:
    def add(self, kind, label, detail=""): ...
    def count(self, kind): ...    # 统计某类事件数
    def render(self): ...          # 人读输出

@dataclass
class CostTracker:
    model_calls; tool_calls; tokens_in; tokens_out
    def add_model(self, tin, tout): ...
    def usd(self): ...             # 估算花费
```

loop 每一步都 `trace.add(...)` + 更新 tracker。capstone 跑出来的就是一条完整 trace:

```
[  system] prompt :: 3 tools, 2 env vars
[   model] turn-0 :: tool_use
[    perm] read_config :: allow: allow-list
[    tool] read_config :: {'ok': True, 'value': {...}}
[   model] turn-1 :: tool_use
...
[    done] final :: Done. Report saved...
cost: {'model_calls': 4, 'tool_calls': 3, 'usd': 0.00265}
```

## 该记录什么

| 维度 | 记什么 | 用途 |
|------|--------|------|
| **结构** | 每步 kind/label/detail | 复盘"它走了哪条路" |
| **成本** | model/tool 调用数、token、$ | 预算、优化 |
| **决策** | 权限 allow/deny、guard 触发 | 安全审计 |
| **错误** | 工具失败、重试次数 | 排障 |

## 设计要点

1. **trace 是一等公民**:不是 debug 时才加,是每步默认就记。
2. **cost 实时累计**:capstone 里 ask 跑 0.00265 美元、3 次工具——这种可量化是 agent 经济性的基础。
3. **可重放**:有了完整 trace(+ 确定性 mock),可以重跑复现——本仓库的所有 `_self_test` 就是"可重放"的极致。
4. **span 分类要稳定**:固定的 kind 集合让 `count()`/过滤/可视化都好做。

## 与评测的衔接

L15 评测 harness 时,trace 和 cost 就是评测信号的来源(走了几步、花了多少、有没有触发护栏)。可观测性是评测的前提。

## 退出条件
- [ ] 说清为什么涌现轨迹必须可观测
- [ ] 列出 trace 该记录的四个维度
- [ ] 理解"确定性 + 完整 trace = 可重放"
