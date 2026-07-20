"""生成 academic-community-engagement notebooks (N1 邀约打分对比). 跑后 nbconvert --execute."""
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

# ───────────────────────── N1 score-an-engagement ─────────────────────────
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 给学术共同体参与邀约打分对比 (Score an Academic Engagement Opportunity)

> 配套 L1-L5 · **真实科研动作**: 列出你自己实际收到过(或可以合理设想)的2个学术共同体参与邀约
> (比如"顶会workshop program committee邀请" vs "不知名期刊单篇审稿邀请", 或者"组织一场workshop的念头" vs
> "一次PC邀约"), 用四维打分框架(时间成本/可见度增益/人脉价值/回馈价值)真的打一次分, 对比出结论。

不需要虚构材料 —— 换成你自己现在真的收到过、或者纠结要不要接的邀约去跑一遍。"""),

    code(PATHS + """
import engagement_scorer as es
print("四个打分维度:", [es.DIMENSIONS[k][0] for k in es.DIMENSIONS])"""),

    md("""## 1. 建两个候选邀约, 逐维度打分(必须写依据, 空口打分等于没打分)"""),

    code("""a = es.blank_engagement("邀约A: 顶会workshop program committee")
a["scores"]["time_cost"] = {"score": 3, "note": "预计需审6-8篇, 约15小时(呼应L1第1节PC时间投入估算)"}
a["scores"]["visibility_gain"] = {"score": 4, "note": "PC名单会公开挂在workshop官网, 是该子领域的一次曝光"}
a["scores"]["network_value"] = {"score": 4, "note": "PC群里都是这个细分领域的活跃研究者(呼应L4长期合作的起点)"}
a["scores"]["reciprocity"] = {"score": 5, "note": "自己至今没审过一次稿, 欠了共同体不少"}

b = es.blank_engagement("邀约B: 不知名期刊审稿(单篇)")
b["scores"]["time_cost"] = {"score": 4, "note": "单篇预计3小时"}
b["scores"]["visibility_gain"] = {"score": 1, "note": "期刊领域内认可度低"}
b["scores"]["network_value"] = {"score": 1, "note": "匿名审稿, 认识不到任何人"}
b["scores"]["reciprocity"] = {"score": 5, "note": "同样是在回馈共同体"}

print(es.render([a, b]))"""),

    md("""## 2. 打分完整性自检(每维度必须有分数+依据)"""),

    code("""for e in [a, b]:
    chk = es.audit(e)
    status = "✅ 完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])
    print(f"{e['name']}: {status}")"""),

    md("""## 3. 反面: 只打分不写依据"""),

    code("""bad = es.blank_engagement("邀约C: 敷衍打分示例")
bad["scores"]["visibility_gain"] = {"score": 5, "note": ""}
for i in es.audit(bad)["issues"]:
    print("⚠", i)
print("\\n→ 只填数字不写理由, 等于没打分 —— 半年后你自己都不记得当初为什么打这个分,"
      " 更没法在下次收到类似邀约时对比出'这次和上次比到底有什么不同'。")"""),

    md("""## 4. 总分最高 ≠ 直接接受

邀约A总分更高, 但这不意味着它就该无条件接受。L1第2节提到: 四维之间不是互相独立、地位对等的——
如果`time_cost`打了很低的分(意味着会严重冲突手头的关键截止日期), 即便另外三维都很高、总分看起来很漂亮,
也应该认真考虑推掉或协商延后。反过来, `reciprocity`打了满分但其他三维都偏低的邀约, 也值得接受——
即便对个人回报不大, 至少不亏欠共同体, 这是打分该纳入的诚实动机, 不是唯功利论。"""),

    code("""ranked = es.compare([a, b])
print("排序结果(仅供参考, 不是最终决定):")
for i, e in enumerate(ranked, 1):
    print(f"  {i}. {e['name']} (总分 {es.total(e)}, 最弱项: {e['_weakest']})")
print("\\n下一步: 如果最弱项是time_cost, 先去看手头的deadline日历再决定;"
      " 如果在纠结要不要自己发起一场活动而不是加入别人的, 去看L2组织workshop的流程。")"""),

    md("""## 5. 反思(呼应L1-L5)

你刚把"这份邀约到底接不接"这句凭直觉的一时纠结, 变成了一份可比较、可留档的候选清单。带走:

- 时间成本/可见度增益/人脉价值/回馈价值四维缺一不可, "来者不拒"和"全部推掉"(L1开篇的两种典型错误反应)
  都是没有评估这一步的情绪化直觉。
- 打分必须带依据, 不然半年后自己都解释不清当初的判断——这也是为什么`audit()`强制拦截"打了分却不写理由"
  的敷衍打分。
- 总分只是参考, 不是终局决定: 一个维度的极端低分(尤其是`time_cost`)可以否决总分很高的邀约,
  这和L1第2节"四维不是互相独立、地位对等"的提醒是同一件事。
- 如果这份邀约是"要不要自己发起一场活动"而不是"要不要加入别人的名单", 下一步去L2用提案六要点
  (动机/为什么是现在/差异化/组织者分工/意向嘉宾/审稿规划)走一遍mini proposal骨架。
- 如果你发现自己至今几乎没有被邀请审过稿, 去看L5"第一次被看见"的几条正当入口
  (导师授权的secondary审稿/主动填写公开招募表单/OpenReview高质量公开评论), 这不是玄学人脉,
  是一套可以主动触发的正反馈循环。

下一步: 把打分结果和依据存好, 下次再收到类似邀约时重新跑一遍这个notebook, 对比这次和上次的判断
有没有变化、变化的原因是什么(比如手头deadline变了, 还是这份邀约本身的可见度/人脉价值和上次不一样)。"""),
]
nbformat.write(n1, HERE / "N1-score-an-engagement.ipynb")
print("written N1")
