# L03 · CrewAI（2024）

## 30 秒核心

> CrewAI = **Role + Goal + Backstory** 三件套定义 agent，Process 决执行流程 (sequential / hierarchical)。

2024 上半年最快爆红的 multi-agent 框架，GitHub 30k+ star。

## 核心抽象

| 抽象 | 含义 |
|------|------|
| `Agent` | role + goal + backstory + tools |
| `Task` | description + agent + expected_output |
| `Crew` | agents + tasks + process |
| `Process` | sequential (Process.sequential) / hierarchical |

## 完整例

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Researcher",
    goal="Uncover cutting-edge developments in {topic}",
    backstory="You're a seasoned researcher...",
    tools=[search_tool],
)

writer = Agent(
    role="Content Writer",
    goal="Craft engaging articles...",
    backstory="...",
)

task1 = Task(
    description="Research about {topic}",
    agent=researcher,
    expected_output="A bullet list of insights",
)

task2 = Task(
    description="Write a blog post",
    agent=writer,
    expected_output="A 500-word article",
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
)

result = crew.kickoff(inputs={"topic": "LLM 2025"})
```

## Process 类型

| Process | 含义 |
|---------|------|
| sequential | task 顺序执行，前 task output → 后 task context |
| hierarchical | manager_agent 决调度 |

## CrewAI 强项

| 强 | 解释 |
|----|------|
| 入门极快 | 30 行代码出 crew |
| Role 抽象漂亮 | role/goal/backstory 直观 |
| 工具集成 | LangChain tools 直接用 |
| 社区大 | 30k+ star |

## CrewAI 弱项

| 弱 | 解释 |
|----|------|
| 状态管理弱 | 跨 task 复杂 state 难 |
| Conditional 流程 | 不如 LangGraph 灵活 |
| Process 模式少 | 只 2 种 |

## 实现 (`crewai_mock.py` 预告)

```python
@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    tools: list = field(default_factory=list)

class Crew:
    def __init__(self, agents, tasks, process="sequential"):
        ...
    def kickoff(self, inputs):
        context = {}
        for task in self.tasks:
            agent = task.agent
            output = agent.execute(task.description, context, inputs)
            context[task.id] = output
        return context[self.tasks[-1].id]
```

## 退出条件

- 能默写 role/goal/backstory + task + crew + process
- 知道 sequential vs hierarchical
- 能写 30 行 crew

## 一句话

> CrewAI = role-based agent + 顺序 task — 入门 30 行，30k+ star 不是没原因。
