"""
feedback_quality.py — 向下反馈质量自检: 给junior研究者的反馈是否具体、可执行, 而不是
"继续努力"式的空话。

四个维度 (对应 L2):
  specificity   具体行为而非泛泛评价
  actionability 给出可执行的下一步
  balance       认可与改进建议的平衡
  timing        问题发生后多久给出反馈

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "specificity": ("具体行为而非泛泛评价", "是否点名了具体的代码/实验/会议发言, 而不是'做得不错'"),
    "actionability": ("给出可执行的下一步", "对方听完知道下一步具体做什么"),
    "balance": ("认可与改进建议的平衡", "只有批评或只有表扬都会失真"),
    "timing": ("时效性", "问题发生后多久给出反馈, 拖太久对方记不清上下文"),
}


def blank_feedback(mentee: str) -> dict:
    return {"mentee": mentee, "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS}}


def audit(feedback: dict) -> dict:
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = feedback["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或越界: 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了{score}分却没写依据")
    return {"issues": issues, "ready": not issues}


def render(feedback: dict) -> str:
    lines = [f"=== 给 {feedback['mentee']} 的反馈质量自检 ==="]
    for key, (name, _) in DIMENSIONS.items():
        d = feedback["scores"][key]
        lines.append(f"{name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    fb = blank_feedback("师弟A")
    fb["scores"]["specificity"] = {"score": 2, "note": "只说了'实验部分再仔细点', 没点名具体哪个脚本"}
    fb["scores"]["actionability"] = {"score": 2, "note": "没说清'仔细点'具体指做什么动作"}
    fb["scores"]["balance"] = {"score": 3, "note": "全是改进意见, 没提认可的部分"}
    fb["scores"]["timing"] = {"score": 4, "note": "当天代码review时就给了"}
    print(render(fb))
    chk = audit(fb)
    print("\n" + ("✅ 反馈质量合格" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
    print("\n→ specificity和actionability都偏低: '仔细点'不是反馈, 是叹气。"
          " 改成'第47行的seed没有固定, 三次跑出的结果对不上, 建议这周把所有实验脚本的"
          "seed显式写进config' 才是可执行的反馈。")
