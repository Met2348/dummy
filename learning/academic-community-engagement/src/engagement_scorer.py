"""
engagement_scorer.py — 学术共同体参与邀约打分: 审稿邀请/PC职位/workshop组织邀约,
不是来者不拒, 也不是全部推掉, 而是按四个维度系统比较该不该接。

四个维度 (对应 L1-L5):
  time_cost       时间成本 —— 这项service大概占用多少小时
  visibility_gain 可见度/影响力增益 —— 对你在这个领域的存在感有多大帮助
  network_value   人脉/合作价值 —— 会不会认识潜在合作者/推荐人
  reciprocity     对共同体的回馈价值 —— 你是不是也在"占便宜不还"

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "time_cost": ("时间成本", "这项service大概占用多少小时? 值不值当前阶段投入?"),
    "visibility_gain": ("可见度/影响力增益", "接了之后, 领域内对你的存在感会有实质提升吗?"),
    "network_value": ("人脉/合作价值", "会不会因此认识潜在合作者/推荐人?"),
    "reciprocity": ("对共同体的回馈价值", "你是不是也一直在'索取审稿意见却从不审别人'?"),
}


def blank_engagement(name: str) -> dict:
    return {"name": name, "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS}}


def total(engagement: dict) -> int:
    return sum(d["score"] for d in engagement["scores"].values())


def audit(engagement: dict) -> dict:
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = engagement["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或越界: 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了{score}分却没写依据")
    return {"issues": issues, "ready": not issues}


def compare(engagements: list[dict]) -> list[dict]:
    ranked = sorted(engagements, key=total, reverse=True)
    for e in ranked:
        weakest_key = min(e["scores"], key=lambda k: e["scores"][k]["score"])
        e["_weakest"] = DIMENSIONS[weakest_key][0]
    return ranked


def render(engagements: list[dict]) -> str:
    ranked = compare(engagements)
    lines = ["=== 学术共同体参与邀约对比 ==="]
    for i, e in enumerate(ranked, 1):
        lines.append(f"\n{i}. {e['name']}  (总分 {total(e)}/20, 最弱项: {e['_weakest']})")
        for key, (name, _) in DIMENSIONS.items():
            d = e["scores"][key]
            lines.append(f"   {name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = blank_engagement("邀约A: 顶会workshop program committee")
    a["scores"]["time_cost"] = {"score": 3, "note": "预计需审6-8篇, 约15小时"}
    a["scores"]["visibility_gain"] = {"score": 4, "note": "PC名单会公开挂在workshop官网"}
    a["scores"]["network_value"] = {"score": 4, "note": "PC群里都是这个细分领域的活跃研究者"}
    a["scores"]["reciprocity"] = {"score": 5, "note": "自己至今没审过一次稿, 欠了共同体不少"}

    b = blank_engagement("邀约B: 不知名期刊审稿(单篇)")
    b["scores"]["time_cost"] = {"score": 4, "note": "单篇预计3小时"}
    b["scores"]["visibility_gain"] = {"score": 1, "note": "期刊领域内认可度低"}
    b["scores"]["network_value"] = {"score": 1, "note": "匿名审稿, 认识不到任何人"}
    b["scores"]["reciprocity"] = {"score": 5, "note": "同样是在回馈共同体"}

    print(render([a, b]))
