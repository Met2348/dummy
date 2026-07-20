# 团队运营计划骨架模板

> 配套 L1-L5、`src/team_ops_audit.py`。五块骨架,每块给出具体写作要点。写完用 `team_ops_audit.py` 的 `audit()` 自检有没有漏项/敷衍项。

## 1. 多项目并行时间分配 (time_allocation)
配L1。每周固定时间块分给哪个项目,而不是"有空就做"——先钉死主线课题的深度块,再分配救火时间;同一时间块只处理一个项目,减少切换的注意力残留。

_填写:_
```


```

## 2. 合作者/实习生招募标准 (recruiting_criteria)
配L2。写清楚必须具备的能力(必须项)和可以现学的能力(可现学项),不是"越强越好"式的空洞高要求;附一个有边界(2-4小时)、贴近真实工作的试做任务设计。

_填写:_
```


```

## 3. 远程/异步协作规范 (async_protocol)
配L3。按消息紧急程度分层的响应时限约定(阻塞型/非阻塞型/真正紧急各多久响应),以及哪些日常事项该用书面文档替代同步会议。

_填写:_
```


```

## 4. 新人onboarding文档 (onboarding_docs)
配L4。新人第一周该看什么(最小必要阅读集)+ 该跑通什么(一条可验证的golden path,写明预期输出),外加术语表和"谁负责什么"地图。

_填写:_
```


```

## 5. 跨专业沟通桥接 (cross_discipline_bridge)
配L5。和非ML背景的合作者(工程师/PM/设计师)有没有一张双向的术语对照表,以及谁对哪类决定拥有最终拍板权。

_填写:_
```


```

---

## 自检

```python
from team_ops_audit import blank_ops_plan, audit, render

plan = blank_ops_plan()
plan["time_allocation"] = "..."
plan["recruiting_criteria"] = "..."
plan["async_protocol"] = "..."
plan["onboarding_docs"] = "..."
plan["cross_discipline_bridge"] = "..."

print(render(plan))
chk = audit(plan)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])
```

(五块都填完之后用上面的代码跑一遍自检——`ready` 为真才说明每一节都有实质内容,而不是随手写一句话敷衍过去;`audit()` 会把内容过短的项单独标出来。)
