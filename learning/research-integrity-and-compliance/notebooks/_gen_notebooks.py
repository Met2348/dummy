"""生成 research-integrity-and-compliance notebooks (N1 合规自检). 跑后 nbconvert --execute."""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)

PATHS = """import sys
from pathlib import Path
SRC = Path.cwd().parent / "src"
sys.path.insert(0, str(SRC))"""

# ───────────────────────── N1 audit-compliance ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 自检一个项目的科研合规骨架 (Audit Your Research Compliance Plan)

> 配套 L1-L5 · **真实科研动作**: 写一版科研合规五块骨架(哪怕是雏形), 用 `compliance_checklist.py`
> 自动检查漏项, 覆盖你现在(或未来可能)真实要处理的一个研究项目, 不需要虚构材料。"""),

    code(PATHS + """
import compliance_checklist as cc
print("五块骨架:", [name for _, name, _ in cc.SECTIONS])"""),

    md("""## 1. 写一版完整的科研合规计划(五块都填)"""),

    code("""good = cc.blank_compliance()
good["authorship_agreement"] = "项目启动会上书面确认: A一作(实现+实验), B通讯(idea+资源), 已发邮件存档。"
good["irb_status"] = "本项目不涉及人类被试或敏感个人数据, 已在项目文档标注'不适用'并说明理由。"
good["ip_disclosure"] = "方法涉及的核心算法已按学校要求提交disclosure表, 等待技术转移办公室反馈。"
good["export_control"] = "国际合作方来自受限清单外地区, 已核实无出口管制限制, 存档确认邮件。"
good["disclosure_plan"] = "若发现模型存在可被滥用的能力, 先内部上报安全团队, 90天coordinated disclosure后再公开细节。"

print(cc.render(good))"""),

    md("## 2. 完整性自检"),

    code("""chk = cc.audit(good)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])
print("risk_flags:", chk["risk_flags"])"""),

    md("""## 3. 反面: 完全没考虑合规的项目(常见新手版本)"""),

    code("""bad = cc.blank_compliance()
bad["authorship_agreement"] = "大家心照不宣。"
for i in cc.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 缺IRB审查状态+缺知识产权披露+缺出口管制核实+缺负责任披露计划, 外加一句'大家心照不宣'的"
      " 署名'协议', 这正是评审/合规审查最容易戳穿的敷衍申请——这些漏洞往往不是能力问题,"
      " 而是没人提醒就压根没想到要处理。")"""),

    md("""## 4. 反思(本专题L1-L5收官)

你刚把"科研诚信与合规"这句容易被简化成"别造假就行"的空话, 变成了一份可核查、可复盘的
五块骨架。带走:

- 五块骨架对应五个完全不同性质的问题(署名协议/IRB审查/知识产权披露/出口管制/负责任披露),
  缺一块就是评审/合规审查最容易追问倒你的地方。
- `render()`把骨架可视化,`audit()`查漏, 两者配合让你在真正提交/公开之前就完成一次
  "模拟被追问", 而不是把这一步留到真正出事的时候。
- 这份骨架和`research-direction-proposal`(9.0)的开题报告自检、`research-funding-and-resources`
  (9.14)的经费/资源自检是同一套"骨架+audit+render"方法论在"诚信与合规"这一具体维度上的
  应用, 缺一块骨架的代价往往不是"别人觉得你没写完", 而是"真出事时, 不懂流程本身就是
  二次伤害"。

> **本专题(research-integrity-and-compliance)收官**: L1不端调查与authorship仲裁 → L2 IRB/伦理
> 审查 → L3知识产权与成果转化 → L4国际合作合规 → L5安全与负责任披露, 完整走完了科研诚信与
> 合规这条常年隐身、直到出事才被想起的深水线。"""),
]
nbformat.write(n1, HERE / "N1-audit-compliance.ipynb")
print("written N1")
