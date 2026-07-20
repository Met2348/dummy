# 求职材料包五块骨架填空worksheet

> 配套 L2-L5、`src/application_kit_audit.py`。五块骨架对应求职材料包必须具备的说服力要素,每块给出具体写作要点。写完用 `application_kit_audit.py` 的 `audit()` 自检有没有漏项/空泛套话,再用 `negotiation_focus()` 把薄弱环节转成谈判/面试前该问自己的问题。

## 1. CV/履历定制 (cv_tailoring)

针对具体岗位调整了侧重, 而不是海投同一份CV。写清楚: 这次投递的岗位最看重什么, 你把哪些项目/经历排到了前面, 弱化了哪些和这个岗位关联较弱的内容(呼应 L5 CV版本化管理的"按受众派生"原则)。

_填写:_
```



```

## 2. 推荐人网络与请求策略 (recommenders)

提前沟通、给推荐人素材包, 而不是临时才开口(呼应 L2)。写清楚: 你选了哪几位推荐人、为什么是他们而不是头衔更大但了解你有限的人、你给了对方什么具体素材(CV+岗位描述+希望强调的具体事例)、请求的时间点是不是留出了4-8周。

_填写:_
```



```

## 3. Job talk/面试陈述大纲 (talk_outline)

有清晰的一条研究主线, 而不是罗列做过的项目(呼应 L3)。写清楚: 如果只能讲一条主线是哪一条、挑了哪1-2个代表作、未来5年计划具体到了什么程度(而非"继续深入这个方向"式空话)。

_填写:_
```



```

## 4. 谈判准备 (negotiation_prep)

明确底线/优先级(薪资/起始经费/团队规模), 而不是被动接受首次offer(呼应 L4)。写清楚: 你的锚点数字是多少、底线是多少、当前真实的BATNA(备选方案)是什么——如果BATNA这一栏写不出具体内容, 这本身就是需要优先补强的信号。

_填写:_
```



```

## 5. 跨材料叙事一致性 (narrative_consistency)

CV/cover letter/job talk讲的是同一个故事。写清楚: 用一句话概括你的研究主线, 检查这句话是否在CV、cover letter、job talk里的表述基本一致, 不会让人觉得是两个人写的材料。

_填写:_
```



```

---

## 自检

```python
from application_kit_audit import blank_package, audit, render, negotiation_focus

package = blank_package()
package["cv_tailoring"] = "..."
package["recommenders"] = "..."
package["talk_outline"] = "..."
package["negotiation_prep"] = "..."
package["narrative_consistency"] = "..."

print(render(package))
chk = audit(package)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])
for q in negotiation_focus(package):
    print("?", q)
```

(填完五块之后, 用上面的代码跑一遍自检——`ready` 为真才说明骨架完整; 如果某一节被标记为"空泛套话", 换成具体可验证的事实, 而不是"综合能力强"这类可以套在任何人身上的描述。)
