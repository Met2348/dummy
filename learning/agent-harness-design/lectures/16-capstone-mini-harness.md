# L16 · Capstone — Mini-Harness 跑通多步任务

## 目标

把前 15 节的组件**组装成一个能跑的 harness**,在一个真实多步任务上跑通,并展示:完整 trace、tool 调用序列、context、cost、**权限拦截**、memory。

## 任务

> "读取团队配置 → 计算月度席位预算 → 保存报告。"

工具:`read_config`(read-only)、`multiply`(read-only)、`write_report`(可写)。

## src 走读

[mini_harness.py](../src/mini_harness.py) 组装,[capstone/run_task.py](../src/capstone/run_task.py) 跑两遍做对照:

```python
h = Harness(MockModel(budget_brain), reg,
            role="You are a budgeting assistant.",
            env={"cwd": "/work", ...},
            permissions=PermissionManager(mode=mode, allow=["read_config","multiply"]))
result = h.run("Read the team config, compute the monthly seat budget, and save a report.")
```

## 运行

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-harness-design/src/capstone/run_task.py
```

## 两遍对照(核心教学)

### Run 1 — mode=ask(写操作批准)
```
[  system] prompt :: 3 tools, 2 env vars
[   model] turn-0 :: tool_use
[    perm] read_config :: allow: allow-list
[    tool] read_config :: {'ok': True, 'value': {'team':'infra','seats':6,...}}
[   model] turn-1 :: tool_use
[    perm] multiply :: allow: allow-list
[    tool] multiply :: {'ok': True, 'value': 150}
[   model] turn-2 :: tool_use
[    perm] write_report :: allow: approved
[    tool] write_report :: {'ok': True, 'value': 'report.md (41 chars)'}
[   model] turn-3 :: end_turn
[    done] final :: Done. Report saved...
cost: {'model_calls': 4, 'tool_calls': 3, 'usd': 0.00265}
```

### Run 2 — mode=readonly(写操作拒绝)
```
[    perm] write_report :: deny: writes blocked in readonly mode
[    done] final :: Budget computed = $150, but the report could not be saved
          (permission denied). Surfacing this instead of pretending success.
```

## 读这个对照

> **同一个 model、同一组工具、同一个任务**——只改了 `permission mode`,副作用就从"发生"变成"被拦截"。被拒的写操作 **surface 成工具错误**,agent 看见后优雅地报告失败,而不是假装成功。

这一张对照浓缩了整个专题:harness 的每个组件(权限、错误 surface、trace、cost、loop)都在这一次运行里各司其职。

## 毕业要义

> agent 的能力 = model × harness。你已经亲手造了那个 harness:agentic loop、工具执行、context 管理、权限、记忆、可观测——把 MockModel 换成真 LLM 客户端,形状一模一样。

## 退出条件
- [ ] 跑通 capstone,看到两遍对照
- [ ] 能解释 readonly 那遍为什么仍优雅收尾
- [ ] 能把每条 trace 行对应到某一节讲的组件
