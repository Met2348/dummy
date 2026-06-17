# L08 · Memory 子系统 — 跨会话的持久记忆

## context 之外的记忆

context window 是**短期工作记忆**,会话一结束就没了。要让 agent 跨会话记住东西(用户偏好、项目约定、上次的结果),需要**持久记忆**——落到文件/DB。

```
工作记忆 (context window)   ← 短期,易失,贵
持久记忆 (file / DB)        ← 长期,廉价,需主动读写
```

## src 走读

[memory.py](../src/harness/memory.py) 是最小 file-based KV:

```python
class Memory:
    def __init__(self, path): self.data = self._load()   # 从 JSON 文件读
    def set(self, key, value): self.data[key]=value; self._save()
    def get(self, key, default=None): return self.data.get(key, default)
    def _load(self):
        try: return json.load(open(self.path))
        except Exception: return {}                        # 损坏不崩,降级为空
```

`_self_test` 验证:set/get、**跨实例持久**(新 `Memory(path)` 读得到旧写入)、损坏文件降级为空不抛异常。

## 设计要点

1. **损坏要降级,不要崩**:记忆文件坏了,agent 应当空记忆继续跑,而不是整个挂掉(`_load` 的 try/except)。
2. **主动读写**:记忆不会自动进 context——harness 要在合适时机 `get` 进提示、`set` 存结论。
3. **结构化 key**:`last_budget`、`user.timezone`,别一锅塞。

## 记忆的种类(呼应 agent-memory-context 专题)

| 类型 | 内容 | 本仓库对应 |
|------|------|-----------|
| 工作记忆 | 当前对话 | `ContextWindow` |
| 情景记忆 | 过去发生了什么 | Memory + 时间戳 |
| 语义记忆 | 事实/偏好 | Memory KV |
| 程序记忆 | 怎么做某事 | system prompt / 技能 |

`agent-memory-context` 专题(Letta/Mem0)讲记忆的**算法**;本节讲 harness 怎么**挂载**一个记忆后端。

## 真 harness 的 file-based memory

Claude Code 的 `CLAUDE.md` / memory 目录、`.remember/` handoff——都是这个思路的工业版:用**人类可读的文件**当持久记忆,既给 agent 用,也给人看、可版本管理(12-Factor: own your context)。

## 退出条件
- [ ] 区分工作记忆与持久记忆
- [ ] 说清"损坏降级、主动读写"两条
- [ ] 把四种记忆类型映射到 harness 组件
