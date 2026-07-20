"""生成 research-visibility-negotiation notebooks (N1 求职材料包自检). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 audit-application-kit ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 自检一份求职材料包 (Audit Your Application Kit)

> 配套 L2-L5 · **真实科研动作**: 写一版求职材料包五块骨架(哪怕是雏形), 用 `application_kit_audit.py`
> 自动检查漏项/空泛套话, 再把检查结果转成谈判/面试前该主动准备的问题清单。"""),

    code(PATHS + """
import application_kit_audit as aa
print("五块骨架:", [name for _, name, _ in aa.SECTIONS])"""),

    md("""## 1. 写一版完整的材料包(五块都填)"""),

    code("""good = aa.blank_package()
good["cv_tailoring"] = "针对XX Lab的招聘方向, 把interp相关项目排在前面, 弱化早期工程复现经历。"
good["recommenders"] = "提前2周联系导师+合作者, 附上目标岗位描述和自己想强调的3个点。"
good["talk_outline"] = "主线: 从机制可解释性方法论到大规模验证, 一条技术演进链。"
good["negotiation_prep"] = "底线: 起薪不低于X, 优先要独立预算而非团队规模。"
good["narrative_consistency"] = "三份材料统一用'从机制理解到工程验证'这条主线, 已交叉核对。"

print(aa.render(good))"""),

    md("## 2. 完整性自检"),

    code("""chk = aa.audit(good)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])"""),

    md("""## 3. 反面: 敷衍的材料包(常见新手版本)"""),

    code("""bad = aa.blank_package()
bad["cv_tailoring"] = "综合能力强, 适合任何岗位。"
for i in aa.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 缺推荐人策略+缺job talk主线+缺谈判准备+缺叙事一致性, 外加CV这一节还用了"
      " '综合能力强'这种空泛套话——这是评审/面试官见得最多、也最容易被戳穿的组合。")"""),

    md("""## 4. L4用: 把薄弱环节转成谈判/面试前的准备清单"""),

    code("""print("如果拿这份敷衍材料包去谈判/面试, 最需要先补的问题是:")
for q in aa.negotiation_focus(bad):
    print("  ?", q)
print("\\n→ 这正是L4「谈判技巧」的准备核心: 与其等被问倒, 不如自己先用同一套检查跑一遍,"
      " 主动想清楚锚点/底线/BATNA, 而不是临场被动应答。")"""),

    md("""## 5. 反思(本专题L1-L5收官)

你刚完整走了一遍: 经营可见度(L1) → 维护推荐人关系(L2) → 设计job talk(L3) → 准备谈判(L4) →
CV版本化管理(L5)这条主线, 并用工具给自己(或未来)的求职材料包做了一次完整自检。带走:

- 材料包不是简历关键词堆砌, 五块骨架(CV定制/推荐人策略/job talk主线/谈判准备/叙事一致性)缺一不可。
- 空泛套话(如"综合能力强")比缺项更隐蔽, 但同样会被评审一眼看穿——具体、可验证的事实才有说服力。
- 谈判准备不是临场发挥, 是提前想清楚锚点/底线/BATNA; BATNA弱恰恰提示要回头补强L1的可见度经营
  和L2的推荐人关系维护, 而不是指望单靠谈判话术翻盘。

> **本专题(research-visibility-negotiation)收官**: L1可见度经营 → L2推荐信策略 → L3设计job talk →
> L4谈判技巧 → L5 CV版本化管理。加上`career-pathways`(9.10)的求职流程全景, "科研生涯周期"体系
> 至此完成前两块拼图, 接下来会转向团队带教与向下管理。"""),
]
nbformat.write(n1, HERE / "N1-audit-application-kit.ipynb")
print("written N1")
