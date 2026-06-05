# L06 · Magentic-One（Microsoft 2024.11）

## 30 秒核心

> Magentic-One = **Orchestrator + 4 worker** 通用多 agent 系统，无 SOP，靠 LLM 决调度。

Microsoft 2024.11 发布，对标 GPT-4o-as-agent。

## 5 个角色

| Agent | 职责 |
|-------|------|
| **Orchestrator** ⭐ | 中央决策，写 plan + 决下一个 worker |
| WebSurfer | 浏览器 (Chromium) |
| FileSurfer | 文件系统读写 |
| Coder | 写 + 改代码 |
| ComputerTerminal | shell 执行 |

## Orchestrator 内核

Orchestrator 维护两个 ledger：

```
Task Ledger: (持久)
  - Facts established
  - Educated guesses
  - Open questions
  - Plan

Progress Ledger: (每轮更新)
  - Is task done?
  - Are we in a loop?
  - Are we making progress?
  - Next speaker?
  - Next instruction
```

每轮 Orchestrator 问自己：
- 完成了吗？→ END
- 卡住了吗？→ 重写 plan
- 否则 → 下一个 worker + 指令

## 基准成绩

| Benchmark | Magentic-One | GPT-4o single |
|-----------|-------------:|--------------:|
| GAIA | 38.0% | 6.7% |
| WebArena | 32.8% | 12.0% |
| AssistantBench | 27.7% | — |

→ Orchestrator + worker 比单 GPT-4o 大幅强。

## 实现要点（`magentic_one_mock.py` 预告）

```python
class Orchestrator:
    def __init__(self, workers):
        self.workers = workers
        self.task_ledger = {}
        self.progress_ledger = {}

    def run(self, task, max_rounds=20):
        self.task_ledger = self._init_ledger(task)
        for round in range(max_rounds):
            self.progress_ledger = self._update_progress()
            if self.progress_ledger["done"]:
                return self.task_ledger["final"]
            if self.progress_ledger["stuck"]:
                self.task_ledger = self._replan()
            next_w = self.progress_ledger["next_worker"]
            instr = self.progress_ledger["next_instruction"]
            result = self.workers[next_w].execute(instr)
            self._integrate(result)
        return None
```

## 与 MetaGPT 区别

| 维度 | MetaGPT | Magentic-One |
|------|---------|--------------|
| 流程 | Hard-coded SOP | LLM 决 |
| Role | 软件公司角色 | 通用 4 worker |
| 任务范围 | 软件开发 | 通用 web/code/file |
| 灵活性 | 低 | 高 |

## 强弱

| 强 | 弱 |
|----|----|
| 通用 | Orchestrator LLM 必须强 |
| GAIA SOTA | $$ 贵 |
| 工程模板好 | 4 worker 重 |

## 退出条件

- 能列 5 agent + 2 ledger
- 知道 GAIA 38% 数字
- 与 MetaGPT 区分 SOP vs 动态规划

## 一句话

> Magentic-One = Orchestrator + 4 worker + 2 ledger — LLM 决调度而非 hard SOP，GAIA 38%。
