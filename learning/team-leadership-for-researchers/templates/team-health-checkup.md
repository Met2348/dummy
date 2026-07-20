# 团队健康度诊断表

> 配套 L5、`src/team_health_check.py`。四个信号对应团队功能失调最常见的四种早期迹象, 每个信号给出具体自问。写完用 `team_health_check.py` 的 `diagnose()` 自检有没有偏低的风险信号。

## 诊断对象: __________ (团队/项目组名称, 诊断日期: __________)

## 1. 心理安全感 (psychological_safety)

成员敢在组会上说"我不知道"或承认失败吗? 还是大家只报喜不报忧?

_填写(打分1-5 + 依据):_
```


```

## 2. 工作量分配均衡 (workload_balance)

是否有人长期超负荷而有人明显清闲? 参照 L1 的任务难度阶梯, 检查负荷是否和能力/阶段匹配。

_填写(打分1-5 + 依据):_
```


```

## 3. 冲突是否被公开处理 (conflict_visibility)

上次意见分歧(尤其是authorship/功劳分配这类敏感话题, 参照 L4)是摊开谈了还是不了了之? 注意: "团队从没吵过架"不是健康信号, 可能是分歧被系统性回避。

_填写(打分1-5 + 依据):_
```


```

## 4. 成长路径可见度 (growth_visibility)

成员知道自己半年后会成长成什么样子吗? 有没有具体、可核对的成长里程碑, 而不是一句笼统的"你会越来越好"?

_填写(打分1-5 + 依据):_
```


```

---

## 自检

```python
from team_health_check import blank_checkup, diagnose, render

c = blank_checkup()
c["psychological_safety"] = {"score": 0, "note": "..."}
c["workload_balance"] = {"score": 0, "note": "..."}
c["conflict_visibility"] = {"score": 0, "note": "..."}
c["growth_visibility"] = {"score": 0, "note": "..."}

print(render(c))
diag = diagnose(c)
print("✅ 暂无明显风险信号" if diag["healthy"] else diag["risks"])
```

(填完四个信号之后, 用上面的代码跑一遍诊断——任何一项打分≤2都会被标记为风险信号; 对每个风险信号, 回到对应讲义(L1任务分配/L4冲突处理)找一个具体的干预动作, 而不是只停留在"发现了问题"这一步。)
