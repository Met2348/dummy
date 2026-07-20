"""生成 team-leadership-for-researchers notebooks (N1 反馈质量自检 / N2 团队健康度诊断). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 give-feedback ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 给一次真实的向下反馈打质量分 (Give Feedback, Then Audit It)

> 配套 L2 · **真实科研动作**: 回想你最近一次给 junior/intern 的反馈(或者你自己收到过的一次),
> 用 `feedback_quality.py` 的四维度框架真的打一次分, 再把敷衍版本改写成具体可执行的版本。

不需要虚构材料 —— 换成你自己真实经历过的一次反馈去跑一遍。"""),

    code(PATHS + """
import feedback_quality as fq
print("四个打分维度:", [fq.DIMENSIONS[k][0] for k in fq.DIMENSIONS])"""),

    md("""## 1. 反面案例: 一句"继续努力"式的反馈

先记录一次常见的敷衍反馈, 用四维度打分——大概率会在 specificity 和 actionability 上现出原形。"""),

    code("""vague = fq.blank_feedback("师弟B")
vague["scores"]["specificity"] = {"score": 1, "note": ""}
vague["scores"]["actionability"] = {"score": 1, "note": ""}
vague["scores"]["balance"] = {"score": 3, "note": "顺口夸了一句'整体不错'"}
vague["scores"]["timing"] = {"score": 2, "note": "拖到两周后的组会才提, 对方已经不太记得当时的上下文"}

print(fq.render(vague))
chk = fq.audit(vague)
print("\\n" + ("✅ 合格" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))"""),

    md("""## 2. 改写: 把同一件事说具体

同一个问题, 换成点名具体行为 + 给出可执行下一步的版本。对照 L2 第2节"从叹气到可执行"的走查示例。"""),

    code("""good = fq.blank_feedback("师弟B")
good["scores"]["specificity"] = {"score": 4, "note": "点名了具体是哪次实验汇报里漏报了方差, 而不是笼统说'汇报要仔细'"}
good["scores"]["actionability"] = {"score": 4, "note": "明确说了下次汇报模板必须包含均值+方差两列"}
good["scores"]["balance"] = {"score": 4, "note": "先肯定了这次消融组设计得很完整, 再指出报告环节的具体缺口"}
good["scores"]["timing"] = {"score": 4, "note": "组会当场就指出来了, 没有拖到下次"}

print(fq.render(good))
chk = fq.audit(good)
print("\\n" + ("✅ 合格" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))"""),

    md("""## 3. 完整性自检: 只打分不写依据也会被拦下"""),

    code("""careless = fq.blank_feedback("反面示例: 敷衍打分")
careless["scores"]["specificity"] = {"score": 5, "note": ""}
for i in fq.audit(careless)["issues"]:
    print("⚠", i)
print("\\n→ 只填数字不写依据, 跟没打分一样——你自己下次都记不清当时具体指的是哪件事。")"""),

    md("""## 4. 对比与反思

两版反馈打分差距最大的维度往往是 specificity 和 actionability——这两项恰好是最容易被"我已经说得挺清楚了"这种自我感觉良好蒙蔽的地方。带走:

- "仔细点""继续努力"这类话没有具体锚点, 对方无法定位到底是哪个动作被评价了。
- balance 不是凑够"一句表扬配一句批评"的公式, 认可和改进建议都要具体。
- 时效性: 当场/当天给出的反馈, 信息密度最高, 拖得越久对方复盘成本越高。

下一步: 打开 `N2-diagnose-team.ipynb`, 把镜头从"一次反馈"拉宽到"整个团队的健康状况"。"""),
]
nbformat.write(n1, HERE / "N1-give-feedback.ipynb")
print("written N1")

# ───────────────────────── N2 diagnose-team ─────────────────────────
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 给一个真实团队做健康度诊断 (Diagnose Your Team)

> 配套 L5 · **真实科研动作**: 用 `team_health_check.py` 给你现在(或曾经)所在的真实团队做
> 一次四信号诊断, 对每个偏低的信号写下一个具体的干预动作, 而不是只停留在"发现了问题"这一步。"""),

    code(PATHS + """
import team_health_check as thc
print("四个健康信号:", [name for _, name, _ in thc.SIGNALS])"""),

    md("""## 1. 给一个真实(或假想)团队做完整诊断

3人小组的例子(呼应 L5 第4节的走查示例): 工作量分配没问题, 但心理安全感和冲突可见度都偏低。"""),

    code("""team = thc.blank_checkup()
team["psychological_safety"] = {"score": 2, "note": "组会上很少有人主动说自己实验失败了"}
team["workload_balance"] = {"score": 4, "note": "目前3个人任务量分布均衡"}
team["conflict_visibility"] = {"score": 2, "note": "上次authorship意见不合, 私下嘀咕但没摊开谈"}
team["growth_visibility"] = {"score": 3, "note": "有semi-annual review但反馈比较笼统"}

print(thc.render(team))"""),

    md("""## 2. 一个容易被忽视的陷阱: 不填分数 ≠ 健康

`diagnose()` 只统计分数在 1-2 分之间的信号为风险——如果一个信号根本没填分数(留在初始的0分),
它不会被标记为风险, 但这**不代表这个团队在这一项上是健康的, 只代表你根本没有去诊断它**。"""),

    code("""unassessed = thc.blank_checkup()  # 什么都没填, 全部保持初始状态
diag = thc.diagnose(unassessed)
print("诊断结果:", "✅ 暂无明显风险信号" if diag["healthy"] else diag["risks"])
print("\\n→ 这份'诊断'之所以显示健康, 只是因为没有人真的花时间去看这四个信号——"
      "'没有数据'和'数据显示健康'是两回事, 定期主动诊断比等风险信号自己冒出来更重要。")"""),

    md("""## 3. 从诊断到行动: 把偏低信号转成具体动作

对照 L5 第5节的干预对照表, 把第1节诊断出的两个风险信号各转成一个具体动作。"""),

    code("""diag = thc.diagnose(team)
print("需要行动的风险信号:")
for r in diag["risks"]:
    print(" ", r)

actions = {
    "心理安全感": "下次组会由带教者先主动分享一次自己判断失误的具体案例, 做出示范",
    "冲突是否被公开处理": "主动约authorship分歧的两位当事人单独谈一次, 对照git提交记录复盘贡献(呼应L4)",
}
print("\\n对应的具体干预动作:")
for name, action in actions.items():
    print(f"  {name}: {action}")"""),

    md("""## 4. 反思(本专题L1-L5收官)

你刚完整走了一遍: 给新手分配任务+review(L1) → 打一次反馈质量分(L2/N1) → 主持一场组会(L3)
→ 处理一次冲突(L4) → 诊断一个团队的健康度(L5/N2)。带走:

- 反馈的四个维度里, specificity 和 actionability 最容易被自我感觉良好蒙蔽。
- 团队健康诊断要定期主动做, "没有人抱怨"和"没有人去看"是完全不同的两件事。
- 心理安全感是团队效能最强的预测因子(Project Aristotle), 且和"降低标准"是两回事。

> **本专题(team-leadership-for-researchers)收官**: L1任务阶梯与code review → L2向下反馈
> → L3组会与brainstorm主持 → L4冲突公开化处理 → L5团队健康诊断。从"带一个新手"到
> "诊断一整个团队", 这条"科研生涯周期"线接下来会转向团队运营的更多实务问题, 但都建立在
> 本专题这一层基本功之上。"""),
]
nbformat.write(n2, HERE / "N2-diagnose-team.ipynb")
print("written N2")
