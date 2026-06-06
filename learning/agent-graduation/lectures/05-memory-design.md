# L05 · DRA Memory 设计

## 3 memory layer

| Layer | 内容 | 寿命 |
|-------|------|------|
| Working | 当前 plan + notes | 1 run |
| Run-level | 跨 sub_q notes | 1 run |
| Cross-session | user preference, past topics | 永久 |

## Working memory

```python
@dataclass
class DRAWorkingMemory:
    query: str
    plan: list[str]
    notes: dict[str, list[dict]]  # sub_q → docs
    draft: str = ""
    citations: list[dict] = field(default_factory=list)
```

每 run 重建。

## Run-level memory

```
检索 doc cache，避免重 fetch:
fetched = {"url_A": markdown_A, ...}

Sub-q 之间共享，避免重复 search。
```

## Cross-session memory

```
User 偏好：
  - "Always cite at least 3 sources"
  - "Prefer Anthropic / vendor-neutral"
  - "Avoid Wikipedia primary"

存到 Mem0/Letta-style store, 下次 run 自动加载。
```

## Memory + cost

```
每 run cache fetch 结果 → 节省 80% web 调用
跨 run cache → 节省 30% LLM 调用
prompt cache → 节省 90% input token
```

## 退出条件

- 能列 3 layer
- 能写 DRAWorkingMemory
- 知道 cache 节省比例

## 一句话

> DRA memory = working / run / cross-session 3 layer + cache — 80% tool / 30% LLM 节省。
