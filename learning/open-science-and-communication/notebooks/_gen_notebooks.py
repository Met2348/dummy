"""生成 open-science-and-communication notebooks (N1 开放科学实践自检). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 audit-open-science ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 自检一个项目的开放科学实践骨架 (Audit Your Open Science Plan)

> 配套 L1-L5 · **真实科研动作**: 写一版开放科学实践五块骨架(哪怕是雏形), 用 `open_science_audit.py`
> 自动检查漏项, 覆盖你现在(或未来可能)真实要处理的一个研究项目/竞赛/公开发言场景, 不需要虚构材料。"""),

    code(PATHS + """
import open_science_audit as osa
print("五块骨架:", [name for _, name, _ in osa.SECTIONS])"""),

    md("""## 1. 写一版完整的开放科学实践计划(五块都填)"""),

    code("""good = osa.blank_release_plan()
good["interdisciplinary_glossary"] = "和认知科学合作者共建了一份'注意力'一词在两个学科的定义对照。"
good["public_communication"] = "写了一段200字的非专业摘要, 请非本领域朋友试读确认能看懂且不夸大结论。"
good["preregistration"] = "在OSF上预注册了核心假设和统计检验方法, 时间戳早于正式实验开始。"
good["artifact_release_plan"] = "投稿时代码已整理好放在匿名仓库, accept后24小时内公开正式仓库。"
good["social_media_boundary"] = "个人twitter简介已声明'观点仅代表个人, 不代表所在机构'。"

print(osa.render(good))"""),

    md("## 2. 完整性自检"),

    code("""chk = osa.audit(good)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])"""),

    md("""## 3. 反面: 完全没考虑开放科学的项目(常见新手版本)"""),

    code("""bad = osa.blank_release_plan()
bad["interdisciplinary_glossary"] = "反正对方应该听得懂我说的话。"
for i in osa.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 缺公众沟通材料+缺预注册计划+缺代码/数据发布规范+缺学术社交媒体边界, 外加一句"
      " '反正对方应该听得懂我说的话'的敷衍'术语对照', 这正是HARKing式事后包装、"
      " '代码以后会整理'式空头支票、以及社交媒体身份混淆最容易发生的地方——这些漏洞往往"
      " 不是能力问题, 而是没人提醒就压根没想到要处理。")"""),

    md("""## 4. 反思(本专题L1-L5收官)

你刚把"开放科学与科学传播"这句容易被简化成"把代码放上GitHub"的空话, 变成了一份可核查、可复盘的
五块骨架。带走:

- 五块骨架对应五个完全不同性质的问题(跨学科术语对照/公众沟通/预注册/代码数据发布/社交媒体边界),
  缺一块就是评审/公众/共同体/机构最容易追问倒你的地方。
- `render()`把骨架可视化, `audit()`查漏, 两者配合让你在真正合作/发表/参赛/发言之前就完成一次
  "模拟被追问", 而不是把这一步留到真正引发误解或争议的时候。
- 这份骨架和`research-integrity-and-compliance`(9.15)的合规自检、`research-funding-and-resources`
  (9.14)的经费/资源自检是同一套"骨架+audit+render"方法论在"开放与传播"这一具体维度上的
  应用, 缺一块骨架的代价往往不是"别人觉得你没写完", 而是"素不相识的外部读者/公众/记者/竞赛
  共同体, 从一开始就没有办法核验你说的话"。

> **本专题(open-science-and-communication)收官**: L1跨学科合作方法论 → L2科学传播与公众沟通 →
> L3开放科学实践(预注册/Registered Report/代码发布) → L4竞赛与Challenge的组织与参与策略 →
> L5学术社交媒体的边界与风险管理, 完整走完了"公开"这件事在合作者/公众/共同体/竞赛/社交媒体
> 五种不同场域下, 分别需要被认真对待的具体方式。"""),
]
nbformat.write(n1, HERE / "N1-audit-open-science.ipynb")
print("written N1")
