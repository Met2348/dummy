# 经费与资源计划骨架模板

> 配套 L1-L5、`src/funding_plan_audit.py`。五块骨架, 每块给出具体写作要点。写完用 `funding_plan_audit.py` 的 `audit()` 自检有没有漏项, 再用 `reviewer_focus()` 看评审最可能追问哪几处。

## 1. 预算论证 (budget_justification)
配L1。每一项花费都要对应一个具体产出(哪个实验/哪个里程碑), 而不是笼统的"研究经费"; 分清direct costs和indirect cost/F&A, 后者由机构和资助方事先谈定, 不是申请人自己能改的变量。

_填写:_
```


```

## 2. 算力资源规划 (compute_plan)
配L2。写清楚需要多少GPU-hours、什么时候用、走哪种算力来源(组内集群/外部大型资源/云计算), 以及申请不到时的具体备选方案——不是一句"需要更多算力"。

_填写:_
```


```

## 3. 数据管理规划 (data_management)
配L3。数据存哪(存储与备份策略)/谁能访问(权限与隐私处理)/论文发表后是否公开(共享与留存策略), 对照FAIR原则逐条检查, 不要写"按需共享"这类模糊表述。

_填写:_
```


```

## 4. 多机构合作协议 (collaboration_mou)
配L4。谁出资源/谁担责任/怎么分署名, 在项目启动阶段就书面写清楚, 包括合作方中途退出时的处理条款, 而不是等出结果之后再吵。

_填写:_
```


```

## 5. 第三方/供应商合规审查 (vendor_compliance)
配L5。用到的API/数据集的使用许可是否覆盖你打算发表或商用的用途, 是否需要签署数据处理协议(DPA), 在立项阶段核查而非投稿/上线前才想起来。

_填写:_
```


```

---

## 自检

```python
from funding_plan_audit import blank_funding_plan, audit, render, reviewer_focus

plan = blank_funding_plan()
plan["budget_justification"] = "..."
plan["compute_plan"] = "..."
plan["data_management"] = "..."
plan["collaboration_mou"] = "..."
plan["vendor_compliance"] = "..."

print(render(plan))
chk = audit(plan)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])

# 如果有薄弱环节, 看评审最可能追问什么
for q in reviewer_focus(plan):
    print("?", q)
```

(五块都填完之后用上面的代码跑一遍自检——`ready` 为真才说明每一节都有实质内容; 如果`reviewer_focus()`输出了追问问题, 说明对应那一节还需要补充, 补完再跑一遍直到没有追问为止。)
