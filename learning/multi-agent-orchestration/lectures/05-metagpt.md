# L05 · MetaGPT（DeepWisdom 2023）

## 30 秒核心

> MetaGPT = "**软件公司"**模拟：PM + Architect + Engineer + QA，按 **SOP** (Standard Operating Procedure) 协作。

2023.08 论文，2024 ICLR Oral，代码生成 multi-agent 早期里程碑。

## 角色定义

| Role | 职责 | 产出 |
|------|------|------|
| Product Manager | 写 PRD | PRD.md |
| Architect | 系统设计 | system_design.md |
| Project Manager | 任务拆分 | task_list.md |
| Engineer | 写代码 | code/*.py |
| QA Engineer | 测试 | test_report.md |

## SOP（Standard Operating Procedure）

```
User: "Build a 2048 game"
  ↓
PM → PRD.md
  ↓
Architect → system_design.md (UML / API)
  ↓
PM → task_list.md
  ↓
Engineer → game.py (按 task_list)
  ↓
QA → test_2048.py + bug report
  ↓
Engineer 修 → repeat 直到 QA pass
```

## Why SOP works

| 论点 | 论据 |
|------|------|
| 拆 role 让 prompt 短 | 每 agent 只看自己 role |
| Document handoff | 每 step 出文档 → 可审 |
| 失败定位准 | PRD 错 vs 代码错 vs test 错 → 明确 |
| 与人类公司类比 | 易理解 |

## HumanEval 数字（论文）

| Method | pass@1 |
|--------|-------:|
| GPT-4 single | 67% |
| AutoGPT | 74% |
| MetaGPT | **85.9%** |

## 弱项

| 弱 | 解释 |
|----|------|
| Token 重 | 多 role × 多 doc |
| Hard-coded SOP | 不灵活，定制难 |
| 任务窄 | 软件开发友好，其他差 |
| 部署成本 | 每个真任务 $$$ |

## 实现 (`metagpt_mock.py` 预告)

```python
@dataclass
class Role:
    name: str
    profile: str  # "Product Manager"
    actions: list[str]  # ["WritePRD"]

class Engineer(Role):
    def write_code(self, task_list, design):
        # mock: 按 design 生成代码
        return generated_code

class MetaGPTPipeline:
    def kickoff(self, requirement):
        prd = self.pm.write_prd(requirement)
        design = self.architect.design(prd)
        tasks = self.pm.split_tasks(design)
        code = self.engineer.write_code(tasks, design)
        report = self.qa.test(code, tasks)
        return {"prd":prd,"design":design,"code":code,"report":report}
```

## 退出条件

- 能默写 5 role + 5 doc
- 能讲 SOP 是什么
- 知道 HumanEval 85.9% 数字

## 一句话

> MetaGPT = 模拟软件公司 SOP — 5 role 接力，HumanEval 85.9%，但 token 贵。
