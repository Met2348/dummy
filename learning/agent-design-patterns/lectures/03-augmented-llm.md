# L03 · 原子构件:Augmented LLM

## 所有模式的最小积木

在谈 5 大 workflow 之前,先认识它们共同的"原子":**augmented LLM**——一个被增强了三种能力的 LLM 调用。

```
            ┌──────────────────────────┐
   query →  │        LLM               │ → answer
            │   ↑        ↑       ↑      │
            │ retrieval tools  memory   │
            └──────────────────────────┘
```

| 增强 | 给了 LLM 什么 | 例子 |
|------|--------------|------|
| **Retrieval** | 读外部知识 | RAG(见 `rag-essential` 专题) |
| **Tools** | 影响外部世界 | 查 API、跑代码、写文件 |
| **Memory** | 跨轮记住状态 | session state、长期记忆 |

> 每个 workflow / agent 节点,都是一个 augmented LLM。模式只是把这些积木**连起来的方式**。

## 关键设计点:增强不是越多越好

新手常犯:给 LLM 塞 30 个工具、灌满检索结果。结果是 **context rot**(信号被噪声稀释)和 **tool sprawl**(选错工具)。L12 专门讲这俩反模式。

设计 augmented LLM 时问三句:
1. 这个工具/检索,**这一步真的需要吗**?
2. 工具的**描述**清楚到模型不会选错吗?
3. 返回结果会不会**撑爆 context**?(需要裁剪吗——L10)

## 工具的接口约定(本专题的 mock)

[common.py](../src/common.py) 里的 `Tool` 是最小抽象:

```python
@dataclass
class Tool:
    name: str          # 模型看到的名字
    description: str    # 模型据此决定何时调用 ← 最重要
    fn: Callable        # 实际执行
```

`description` 是工具设计的灵魂——模型靠它判断"这个场景该不该用我"。Topic 9 的 [04-tool-design](../../agent-harness-design/lectures/04-tool-design.md) 会深入工具设计原则。

## 为什么把"LLM 调用"做成确定性 mock

本专题 [common.py](../src/common.py) 的 `MockLLM` 用注入的 `responder` 返回确定性文本,并用 `CostTracker` 记调用数/token。

> 道理:**模式的教学价值在"怎么连积木",不在 LLM 本身**。把模型固定成可预测的 mock,你才能清楚看到不同模式的控制流和成本差异(capstone 的对照表正是靠这个)。

## 退出条件
- [ ] 能画出 augmented LLM 的三种增强
- [ ] 理解"每个节点 = 一个 augmented LLM"
- [ ] 知道增强过度会导致 context rot / tool sprawl
