# L02 · Letta（前 MemGPT, Packer 2023）

## 30 秒核心

> MemGPT/Letta 把 LLM context 当作 **OS 内存**：
> - **Main Context**：always in prompt (system + recent dialogue + key facts)
> - **External Memory**：archive，需要时 LLM 主动 swap 进 main

CoLM 2024，开源 [letta-ai/letta](https://github.com/letta-ai/letta) 2024 改名 Letta。

## OS 类比

```
RAM (main context, in prompt)
  ↕  (swap in/out via tool calls)
DISK (external archive, vector store)
```

LLM 通过 tool call 主动管 memory：
- `archival_memory_insert(text)`
- `archival_memory_search(query)`
- `core_memory_replace(label, new_value)`

## Main context 结构

```
+-------------------------+
| System instructions     |
+-------------------------+
| Core memory:            |
|   <human>: ...          |   ← about user
|   <persona>: ...        |   ← agent's role
+-------------------------+
| Recent messages (FIFO)  |
+-------------------------+
| Total <= context_window |
+-------------------------+
```

## 何时 swap

LLM 自己决定：
- 看到新事实 → archival_memory_insert
- 用户问历史 → archival_memory_search
- 核心事实变 → core_memory_replace

## Letta 强项

| 强 | 解释 |
|----|------|
| LLM 自主管 memory | 不需写规则 |
| OS 类比直观 | 工程师易懂 |
| 跨 session 持久 | 数据库 backend |
| Open-source | 自托管 |

## Letta 弱项

| 弱 | 解释 |
|----|------|
| LLM 须支持 tool calls | 老模型不行 |
| Search 准确性 | 取决于 archive 实现 |
| Token 成本 | 每 turn 都 build main context |

## 实现 (`letta_mock.py` 预告)

```python
@dataclass
class LettaMemory:
    system: str = ""
    core: dict = field(default_factory=dict)  # {human:..., persona:...}
    recent: list = field(default_factory=list)
    archive: list[tuple[str, str]] = field(default_factory=list)  # (id, text)
    max_recent: int = 20

    def add_message(self, msg: str): ...
    def archival_insert(self, text: str): ...
    def archival_search(self, query: str, k=3): ...
    def core_replace(self, label: str, new_value: str): ...

    def build_main_context(self) -> str:
        return f"SYSTEM: {self.system}\nCORE: {self.core}\nRECENT: {self.recent}"
```

## 数字

[Letta 2025] 长对话 (100k+ turns) 通过 swap 让 8k context 模拟 unlimited 看起来。

## 与 OpenAI Memory 区别

| 维度 | Letta | OpenAI Memory |
|------|-------|---------------|
| Open-source | ✓ | ✗ |
| Self-host | ✓ | ✗ |
| LLM-managed | ✓ | ✓ (mostly) |
| Multi-agent | ✓ | ✗ |

## 退出条件

- 能默写 main vs external context
- 能列 4 archival tool calls
- 知道 Letta 之前叫 MemGPT

## 一句话

> Letta = OS 类比的 LLM memory — main context 是 RAM, archive 是 disk, LLM 自己 tool-call 管 swap。
