"""生成 research-funding-and-resources notebooks (N1 经费/资源计划自检). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 audit-funding-plan ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 自检一份经费与资源计划 (Audit Your Funding & Resource Plan)

> 配套 L1-L5 · **真实科研动作**: 写一版经费/资源计划五块骨架(哪怕是雏形), 用 `funding_plan_audit.py`
> 自动检查漏项, 再用 `reviewer_focus()` 把薄弱环节转成评审最可能追问的问题清单, 覆盖你现在(或未来
> 可能)真实要申请的一份经费/资源, 不需要虚构材料。"""),

    code(PATHS + """
import funding_plan_audit as fpa
print("五块骨架:", [name for _, name, _ in fpa.SECTIONS])"""),

    md("""## 1. 写一版完整的经费/资源计划(五块都填)"""),

    code("""good = fpa.blank_funding_plan()
good["budget_justification"] = "80% 预算用于GPU-hours(对应3个大规模实验, 呼应L1预算论证), 20%用于会议差旅。"
good["compute_plan"] = "需要8×A100持续2个月(呼应L2算力规划); 若砍半, 优先保留可解释性实验, 砍训练规模。"
good["data_management"] = "训练数据存内部集群(呼应L3 DMP), 论文发表后仅公开代码和评测脚本, 不公开原始训练数据。"
good["collaboration_mou"] = "对方实验室出算力, 我方出算法设计(呼应L4), 通讯作者由数据主要贡献方担任, 已书面确认。"
good["vendor_compliance"] = "核实过所用API的商用条款(呼应L5), 确认允许用于论文发表和非商业研究。"

print(fpa.render(good))"""),

    md("## 2. 完整性自检"),

    code("""chk = fpa.audit(good)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])"""),

    md("""## 3. 反面: 敷衍的经费/资源计划(常见新手版本)"""),

    code("""bad = fpa.blank_funding_plan()
bad["budget_justification"] = "申请50万元用于开展本研究的相关支出。"
bad["compute_plan"] = "需要更多算力。"
for i in fpa.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 缺数据管理规划+缺多机构合作协议+缺第三方合规审查, 外加两句笼统的空话,"
      " 这正是评审小组见得最多、也最容易被戳穿的敷衍申请。")"""),

    md("""## 4. reviewer_focus(): 把薄弱环节转成评审追问预判

这是本工具区别于单纯"查漏补缺"的关键一步: `audit()`只告诉你"缺了什么",
`reviewer_focus()`进一步告诉你"评审会怎么追问这个缺口",让你在提交前就
准备好回答, 而不是被问倒在现场。"""),

    code("""print("如果拿这份敷衍计划去答辩/评审, 最可能被追问:")
for q in fpa.reviewer_focus(bad):
    print("  ?", q)
print("\\n→ 每一条追问都直接对应回一讲: 预算论证→L1, 算力规划→L2,"
      " 数据管理→L3, 多机构协议→L4, 供应商合规→L5, 缺哪块就回哪一讲补。")"""),

    md("""## 5. 反思(本专题L1-L5收官)

你刚把"经费与资源规划"这句容易被简化成"申请更多经费/算力"的空话, 变成了一份
可核查、可复盘的五块骨架。带走:

- 五块骨架对应五个完全不同粒度的问题(预算论证/算力规划/数据管理/多机构协议/
  供应商合规), 缺一块就是评审/合规审查最容易追问倒你的地方。
- `audit()`查漏, `reviewer_focus()`把漏项转成具体追问, 两者配合让你在提交前
  就完成一次"模拟被追问", 而不是把这一步留到真正的评审现场。
- 这份骨架和`research-direction-proposal`(9.0)的开题报告自检、`research-team-operations`
  (9.13)的团队运营自检是同一套"骨架+audit+追问预判"方法论在"经费与资源"这一
  具体维度上的应用, 缺一块骨架的代价往往不是"评审觉得你没写完", 而是"评审觉得
  你没有认真规划过这件事"。

> **本专题(research-funding-and-resources)收官**: L1经费申请全流程 → L2算力资源规划 →
> L3数据管理规划 → L4多机构合作管理 → L5供应商/API合规评估, 完整走完了经费与资源
> 规划的五块日常操作基本功。"""),
]
nbformat.write(n1, HERE / "N1-audit-funding-plan.ipynb")
print("written N1")
