# 开放科学实践自检清单模板

> 配套 L1-L5、`src/open_science_audit.py`。五块骨架, 每块给出具体写作要点。写完用 `open_science_audit.py` 的 `audit()` 自检有没有漏项。

## 1. 跨学科术语对照表 (interdisciplinary_glossary)
配L1。如果这个项目涉及和其他学科的合作者共事, 写出至少几组"你的领域术语 ⇄ 对方领域等价说法 ⇄ 备注(是否完全对等)"; 如果暂时不涉及跨学科合作, 也要写明这一判断本身, 而不是简单留空。

_填写:_
```


```

## 2. 公众沟通材料 (public_communication)
配L2。有没有一份面向非专业读者、不夸大结论的摘要(倒金字塔结构、术语首次出现就解释); 有没有找非本领域的朋友试读过, 确认对方复述出来的结论没有悄悄加上原文不支持的断言。

_填写:_
```


```

## 3. 预注册计划 (preregistration)
配L3。假设和分析计划是否在正式收集数据/跑实验之前就已经写下来并打上公开时间戳(比如上传OSF); 明确区分哪些是提前计划好的验证性主张、哪些是数据到手之后才发现的探索性发现。

_填写:_
```


```

## 4. 代码/数据发布规范 (artifact_release_plan)
配L3(及L4竞赛场景下的代码复现验证)。投稿/参赛时代码是否已经整理好放在(匿名)仓库; accept/夺冠后打算什么时候、以什么形式转为正式公开(依赖版本锁定、README、复现命令), 而不是"以后再整理"的空头支票。

_填写:_
```


```

## 5. 学术社交媒体边界 (social_media_boundary)
配L5。个人社交媒体简介是否已经声明"观点仅代表个人, 不代表所在机构"; 更重要的是, 你是否已经想清楚哪些话题该用专业身份发言、哪些该主动淡化机构关联, 而不是仅仅依赖那一句免责声明。

_填写:_
```


```

---

## 自检

```python
from open_science_audit import blank_release_plan, audit, render

plan = blank_release_plan()
plan["interdisciplinary_glossary"] = "..."
plan["public_communication"] = "..."
plan["preregistration"] = "..."
plan["artifact_release_plan"] = "..."
plan["social_media_boundary"] = "..."

print(render(plan))
chk = audit(plan)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])
```

(五块都填完之后用上面的代码跑一遍自检——`ready` 为真才说明每一节都有实质内容, 而不是随手写一句话敷衍过去。)
