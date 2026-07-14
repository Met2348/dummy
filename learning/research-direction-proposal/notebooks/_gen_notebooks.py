"""生成 research-direction-proposal notebooks (N1 方向打分对比 / N2 开题报告自检). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 score-a-direction ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 给候选研究方向打分对比 (Score Candidate Directions)

> 配套 L1 · **真实科研动作**: 列出你脑子里真实存在的2个候选研究方向(哪怕只是模糊念头),
> 用四维打分框架(兴趣/实验室积累/资助趋势/职业规划)真的打一次分, 对比出结论。

不需要虚构材料 —— 换成你自己现在真的在纠结的两个方向去跑一遍。"""),

    code(PATHS + """
import direction_scorer as ds
print("四个打分维度:", [ds.DIMENSIONS[k][0] for k in ds.DIMENSIONS])"""),

    md("""## 1. 建两个候选, 逐维度打分(必须写依据, 空口打分等于没打分)"""),

    code("""a = ds.blank_candidate("方向A: 长上下文推理效率")
a["scores"]["interest"] = {"score": 4, "note": "读过的相关论文都主动做了笔记, 是真兴趣"}
a["scores"]["lab_fit"] = {"score": 5, "note": "long-context专题的代码和数据直接能复用"}
a["scores"]["funding_fit"] = {"score": 3, "note": "关注度稳定但不算上升期, 竞争者不少"}
a["scores"]["career_fit"] = {"score": 4, "note": "工业界效率方向岗位需求持续存在"}

b = ds.blank_candidate("方向B: 多模态对齐评测")
b["scores"]["interest"] = {"score": 2, "note": "是最近才因为热度关注, 还没深入读过几篇"}
b["scores"]["lab_fit"] = {"score": 1, "note": "实验室没有多模态数据/算力积累, 要从零搭"}
b["scores"]["funding_fit"] = {"score": 5, "note": "明显上升期, 顶会track专门扩了"}
b["scores"]["career_fit"] = {"score": 3, "note": "方向新, 但和自己规划的NLP路线略有偏离"}

print(ds.render([a, b]))"""),

    md("""## 2. 打分完整性自检(每维度必须有分数+依据)"""),

    code("""for c in [a, b]:
    chk = ds.audit(c)
    status = "✅ 完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])
    print(f"{c['name']}: {status}")"""),

    md("""## 3. 反面: 只打分不写依据"""),

    code("""bad = ds.blank_candidate("方向C: 敷衍打分示例")
bad["scores"]["interest"] = {"score": 5, "note": ""}
for i in ds.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 只填数字不写理由, 等于没打分 —— 半年后你自己都不记得当初为什么打这个分。")"""),

    md("""## 4. 总分最高 ≠ 直接选它

方向A总分更高, 但方向B的资助趋势明显更强。这正是L2「项目级可行性评估」要接着回答的问题:
方向A的lab_fit优势能否弥补funding_fit的中庸, 需要结合更长尺度的可行性判断, 不能只看总分表。"""),

    code("""ranked = ds.compare([a, b])
print("排序结果(仅供参考, 不是最终决定):")
for i, c in enumerate(ranked, 1):
    print(f"  {i}. {c['name']} (总分 {ds.total(c)}, 最弱项: {c['_weakest']})")
print("\\n下一步: 把总分靠前的1-2个候选, 拿去 N2 之前先走一遍 templates/feasibility-worksheet.md (L2)。")"""),

    md("""## 5. 反思

你刚把"自己想个方向"这句空话, 变成了一份可比较、可留档的候选清单。带走:
- 兴趣/实验室积累/资助趋势/职业规划四维缺一不可, 只看兴趣或只追热点都会踩坑。
- 打分必须带依据, 不然半年后自己都解释不清当初的判断。
- 总分只是参考, 最终决定还要过L2的项目级可行性评估这一关。

下一步: 去 L2 讲义, 对总分靠前的候选跑一遍 `templates/feasibility-worksheet.md`。"""),
]
nbformat.write(n1, HERE / "N1-score-a-direction.ipynb")
print("written N1")

# ───────────────────────── N2 audit-your-proposal ─────────────────────────
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 自检一份开题报告草稿 (Audit Your Proposal Draft)

> 配套 L3-L4 · **真实科研动作**: 写一版开题报告七块骨架(哪怕是雏形), 用 `proposal_audit.py`
> 自动检查漏项/薄弱项, 再把检查结果转成开题答辩最可能被追问的问题清单。"""),

    code(PATHS + """
import proposal_audit as pa
print("七块骨架:", [name for _, name, _ in pa.SECTIONS])"""),

    md("""## 1. 写一版完整的proposal(七块都填)"""),

    code("""good = pa.blank_proposal()
good["background"] = "长上下文推理在超过32k token后普遍出现注意力稀释, 但该现象缺乏系统量化。"
good["gap"] = "现有工作只报告了端到端准确率下降, 未定位是注意力层还是位置编码的问题。"
good["question"] = "假设: 在64k token设定下, 注意力稀释导致的准确率下降比位置编码外推误差高≥2倍。"
good["method"] = "构造分层探针实验, 分别固定注意力/位置编码为理想值, 对比准确率恢复幅度。"
good["timeline"] = "第1-2月复现baseline; 第3-4月完成探针实验; 第5月出第一版结果; 第6月开始写作。"
good["risks"] = "若探针实验无法分离两个因素, 备选方案是改用因果干预(activation patching)定位。"
good["contribution"] = "首次量化区分长上下文两种退化机制的相对贡献, 指导后续工程优化优先级。"

print(pa.render(good))"""),

    md("## 2. 完整性自检"),

    code("""chk = pa.audit(good)
print("✅ 骨架完整" if chk["ready"] else chk["issues"])"""),

    md("""## 3. 反面: 敷衍的proposal(常见新手版本)"""),

    code("""bad = pa.blank_proposal()
bad["background"] = "长上下文很重要。"
bad["question"] = "研究长上下文推理问题。"
bad["timeline"] = "预计一年完成。"
for i in pa.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 缺文献缺口定位 + 研究问题不可证伪 + 时间线模糊 + 无风险预案,"
      " 这是评审小组见得最多、也最容易被戳穿的四类问题。")"""),

    md("""## 4. L4用: 把薄弱环节转成开题答辩追问预判"""),

    code("""print("如果拿这份敷衍proposal去答辩, 评审最可能追问:")
for q in pa.defense_focus(bad):
    print("  ?", q)
print("\\n→ 这正是L4「开题答辩」的准备核心: 与其等被问, 不如自己先用同一套检查跑一遍,"
      " 主动亮出风险预案, 比回避风险更能建立评审信任。")"""),

    md("""## 5. 反思(本专题L1-L4收官)

你刚完整走了一遍: 打分选方向(L1/N1) → 项目可行性(L2) → 写proposal并自检(L3/N2) → 预判答辩追问(L4)。带走:

- proposal不是idea卡的放大版, 是给评审看的正式文档, 七块骨架缺一不可。
- 可证伪的研究问题 + 具体里程碑 + 诚实的风险预案, 是最容易被忽视但最影响评审信任的三块。
- 开题答辩被打回不是失败(呼应 L0「回路A」), 是流程设计的校验点在正常工作——退回L1重新收窄,
  比跑完两年实验才发现方向有问题, 代价小得多。

> **本专题(research-direction-proposal)收官**: L0地图 → L1选方向 → L2评可行性 → L3写proposal → L4扛答辩。
> 加上已有9个科研技能专题的16个环节, 20环节全流程地图至此完整落地。"""),
]
nbformat.write(n2, HERE / "N2-audit-your-proposal.ipynb")
print("written N2")
