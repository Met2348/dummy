"""
career_path_scorer.py — 职业路径候选打分工具: 把"academic还是industry还是postdoc-first"
这种开放性选择, 变成可比较的候选清单。

四个维度 (对应 L1-L5):
  skill_fit        技能匹配度 —— 现有技能栈离这条路径的入门要求差多远
  entry_barrier    入行门槛与准备度 —— 需要补多少东西(证书/作品集/人脉)才够格投递
  stability_growth 长期稳定性与成长空间 —— 5-10年后这条路径的天花板和地板
  market_timing    当前市场窗口期 —— 这条赛道现在是扩张期还是收缩期

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "skill_fit": ("技能匹配度", "现有技能栈离这条路径的入门要求差多远?"),
    "entry_barrier": ("入行门槛与准备度", "需要补多少东西(证书/作品集/人脉)才够格投递?"),
    "stability_growth": ("长期稳定性与成长空间", "5-10年后这条路径的天花板和地板分别在哪?"),
    "market_timing": ("当前市场窗口期", "这条赛道现在是扩张期还是收缩期?"),
}


def blank_path(name: str) -> dict:
    return {"name": name, "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS}}


def total(path: dict) -> int:
    return sum(d["score"] for d in path["scores"].values())


def audit(path: dict) -> dict:
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = path["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或分数越界(应为1-5): 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了 {score} 分却没写依据")
    return {"issues": issues, "ready": not issues}


def compare(paths: list[dict]) -> list[dict]:
    ranked = sorted(paths, key=total, reverse=True)
    for p in ranked:
        weakest_key = min(p["scores"], key=lambda k: p["scores"][k]["score"])
        p["_weakest"] = DIMENSIONS[weakest_key][0]
    return ranked


def render(paths: list[dict]) -> str:
    ranked = compare(paths)
    lines = ["=== 职业路径候选对比 ==="]
    for i, p in enumerate(ranked, 1):
        lines.append(f"\n{i}. {p['name']}  (总分 {total(p)}/20, 最弱项: {p['_weakest']})")
        for key, (name, _) in DIMENSIONS.items():
            d = p["scores"][key]
            lines.append(f"   {name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = blank_path("路径A: 工业界research scientist")
    a["scores"]["skill_fit"] = {"score": 4, "note": "已有的复现/工程能力直接对口"}
    a["scores"]["entry_barrier"] = {"score": 3, "note": "还缺一篇一作顶会论文撑门面"}
    a["scores"]["stability_growth"] = {"score": 4, "note": "头部lab research scientist晋升路径清楚"}
    a["scores"]["market_timing"] = {"score": 4, "note": "前沿lab仍在扩招research岗"}

    b = blank_path("路径B: 走tenure-track学术界")
    b["scores"]["skill_fit"] = {"score": 3, "note": "教学/带组经验几乎为零"}
    b["scores"]["entry_barrier"] = {"score": 2, "note": "需要博后+多篇一作+成体系的研究agenda"}
    b["scores"]["stability_growth"] = {"score": 3, "note": "tenure后稳定但晋升周期长"}
    b["scores"]["market_timing"] = {"score": 2, "note": "faculty岗位僧多粥少, 竞争空前激烈"}

    print(render([a, b]))
    print("\n=== 打分完整性自检 ===")
    for p in [a, b]:
        chk = audit(p)
        status = "✅ 完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])
        print(f"{p['name']}: {status}")
