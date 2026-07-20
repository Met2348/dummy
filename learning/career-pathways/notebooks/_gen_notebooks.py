"""生成 career-pathways notebooks (N1 职业路径打分对比). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 score-a-career-path ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 给候选职业路径打分对比 (Score Candidate Career Paths)

> 配套 L1-L5 · **真实科研动作**: 列出你脑子里真实存在的2个候选职业路径(哪怕只是模糊念头,
> 比如"工业界research scientist" vs "学术界tenure-track"), 用四维打分框架
> (技能匹配度/入行门槛与准备度/长期稳定性与成长空间/当前市场窗口期)真的打一次分, 对比出结论。

不需要虚构材料 —— 换成你自己现在真的在纠结的两条路径去跑一遍。"""),

    code(PATHS + """
import career_path_scorer as cs
print("四个打分维度:", [cs.DIMENSIONS[k][0] for k in cs.DIMENSIONS])"""),

    md("""## 1. 建两个候选, 逐维度打分(必须写依据, 空口打分等于没打分)"""),

    code("""a = cs.blank_path("路径A: 工业界research scientist")
a["scores"]["skill_fit"] = {"score": 4, "note": "已有的复现/工程能力直接对口(呼应L2 research portfolio里能讲清楚的深度项目)"}
a["scores"]["entry_barrier"] = {"score": 3, "note": "还缺一篇一作顶会论文撑门面, 但portfolio已有雏形"}
a["scores"]["stability_growth"] = {"score": 4, "note": "头部lab research scientist晋升路径清楚"}
a["scores"]["market_timing"] = {"score": 4, "note": "前沿lab仍在扩招research岗(L3提到的市场时机信号)"}

b = cs.blank_path("路径B: 走tenure-track学术界")
b["scores"]["skill_fit"] = {"score": 3, "note": "教学/带组经验几乎为零, L5讲的招生/带教技能还没练过"}
b["scores"]["entry_barrier"] = {"score": 2, "note": "需要博后(L4)+多篇一作+成体系的research statement(L1)"}
b["scores"]["stability_growth"] = {"score": 3, "note": "tenure后稳定但晋升周期长, 前几年生存压力大(L5)"}
b["scores"]["market_timing"] = {"score": 2, "note": "faculty岗位僧多粥少, 竞争空前激烈"}

print(cs.render([a, b]))"""),

    md("""## 2. 打分完整性自检(每维度必须有分数+依据)"""),

    code("""for p in [a, b]:
    chk = cs.audit(p)
    status = "✅ 完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])
    print(f"{p['name']}: {status}")"""),

    md("""## 3. 反面: 只打分不写依据"""),

    code("""bad = cs.blank_path("路径C: 敷衍打分示例")
bad["scores"]["market_timing"] = {"score": 5, "note": ""}
for i in cs.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 只填数字不写理由, 等于没打分 —— 半年后你自己都不记得当初为什么打这个分,"
      " 更没法在L3讲的『定期重新评估』时对比出变化。")"""),

    md("""## 4. 总分最高 ≠ 直接选它

路径A总分更高, 但这不意味着路径B(学术界)就该直接出局。这正是L3「学界⇄业界转换策略」和
L4「博士后阶段选择与规划」要接着回答的问题: 市场窗口期(market_timing)和长期稳定性
(stability_growth)这两个维度本来就会随时间变化, 总分只是当下这一次打分的参考,
不是一次性定案的终局决定。"""),

    code("""ranked = cs.compare([a, b])
print("排序结果(仅供参考, 不是最终决定):")
for i, p in enumerate(ranked, 1):
    print(f"  {i}. {p['name']} (总分 {cs.total(p)}, 最弱项: {p['_weakest']})")
print("\\n下一步: 如果最弱项是market_timing或stability_growth, 去看L3的信号识别框架;"
      " 如果在纠结学术界内部的博后阶段, 去看L4的选组四维度。")"""),

    md("""## 5. 反思(呼应L1-L5)

你刚把"到底走学术界还是工业界"这句焦虑的自我拉扯, 变成了一份可比较、可留档的候选清单。带走:

- 技能匹配度/入行门槛/长期稳定性/市场窗口期四维缺一不可, 只看当下的舒适度或只追一时热度都会踩坑。
- 打分必须带依据, 不然半年后自己都解释不清当初的判断——这也是L3讲的"主动转换"和"被动转换"的核心区别:
  前者有据可查, 后者是事到临头才慌忙决定。
- 总分只是参考, 市场窗口期和长期稳定性会随时间变化(L3), 这份打分值得每隔一段时间重新跑一次,
  而不是当作一次性定案。
- 如果路径涉及"要不要先读博后", 下一步去L4用同一套工具对比"直接工作"和"博后1-2年再定"两条子路径。

下一步: 把打分结果和依据存好, 过几个月(比如收到新offer、方向有新进展时)重新跑一遍这个notebook,
对比两次打分有没有变化、变化的原因是什么。"""),
]
nbformat.write(n1, HERE / "N1-score-a-career-path.ipynb")
print("written N1")
