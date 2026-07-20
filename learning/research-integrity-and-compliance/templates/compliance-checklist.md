# 科研合规自检清单模板

> 配套 L1-L5、`src/compliance_checklist.py`。五块骨架, 每块给出具体写作要点。写完用 `compliance_checklist.py` 的 `audit()` 自检有没有漏项。

## 1. 署名协议 (authorship_agreement)
配L1。谁一作/谁通讯/贡献比例是否在项目早期就书面确认过(邮件/会议纪要/署名协议模板均可), 而不是"大家心照不宣"; 如果真的发生署名纠纷, 你打算先找谁沟通(直接沟通→机构调解→正式申诉的哪一级)。

_填写:_
```


```

## 2. IRB/伦理审查状态 (irb_status)
配L2。判断项目是否涉及人类被试或敏感数据(含众包标注/人类评测阶段), 如果涉及, 写下该走哪种审查类别(豁免/快速/全审)及理由; 如果确实不涉及, 也要写明判断依据, 不要简单留空。

_填写:_
```


```

## 3. 知识产权披露 (ip_disclosure)
配L3。这项工作是否可能有商业价值/可专利性, 是否已经(或打算)按机构要求提交发明披露表(IDF), IDF提交时间是否安排在公开发表之前。

_填写:_
```


```

## 4. 出口管制/跨境合规 (export_control)
配L4。国际合作中涉及的对象/技术/数据, 是否核实过受限清单、是否核实过基础研究豁免是否成立、跨境数据传输是否有具体合规安排(而非"应该没问题")。

_填写:_
```


```

## 5. 负责任的风险披露计划 (disclosure_plan)
配L5。如果研究涉及安全漏洞/危险能力的发现, 打算先报告给谁、给多久的响应窗口、公开发表时打算保留哪些技术细节不直接公开。

_填写:_
```


```

---

## 自检

```python
from compliance_checklist import blank_compliance, audit, render

project = blank_compliance()
project["authorship_agreement"] = "..."
project["irb_status"] = "..."
project["ip_disclosure"] = "..."
project["export_control"] = "..."
project["disclosure_plan"] = "..."

print(render(project))
chk = audit(project)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])
```

(五块都填完之后用上面的代码跑一遍自检——`ready` 为真才说明每一节都有实质内容, 而不是随手写一句话敷衍过去。)
