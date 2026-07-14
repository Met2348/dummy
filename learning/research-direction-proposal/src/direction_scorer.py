"""
direction_scorer.py — 研究方向候选打分工具: 把"自己想个方向"从一句空话变成可比较的候选清单.

为什么需要它: 面对"你自己想个方向"这种开放性任务, 新手最常见的失败模式是只凭一时兴趣拍板,
或者反过来纯粹追热点却忽略自己实验室积累用不上。四个维度缺一不可地放在一起比较, 才能避免
"选了个自己做不动/没人关心/和现有积累完全脱节"的方向。

四个维度 (对应 L1):
  interest    兴趣真实度 —— 是三分钟热度还是能撑得下去的真兴趣
  lab_fit     实验室积累契合度 —— 能不能复用现有代码/数据/算力/师门经验
  funding_fit 资助与关注度趋势 —— 这个方向未来几年是否还有经费和读者
  career_fit  职业规划契合度 —— 做完这个方向是否指向你想去的下一步

纯 stdlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMENSIONS = {
    "interest": ("兴趣真实度", "半年后每天还愿意打开这个话题的论文吗, 还是现在新鲜劲一过就想换?"),
    "lab_fit": ("实验室积累契合度", "现有代码/数据/算力/师门经验能复用多少? 从零搭基础设施的代价有多大?"),
    "funding_fit": ("资助与关注度趋势", "这个方向未来3-5年经费/审稿人关注度是在涨还是在退潮?"),
    "career_fit": ("职业规划契合度", "做完这个方向, 是否指向你想去的下一份工作/下一个研究方向?"),
}


def blank_candidate(name: str) -> dict:
    return {
        "name": name,
        "scores": {k: {"score": 0, "note": ""} for k in DIMENSIONS},
    }


def total(candidate: dict) -> int:
    return sum(d["score"] for d in candidate["scores"].values())


def audit(candidate: dict) -> dict:
    """检查打分是否敷衍: 每个维度必须有分数(1-5)和依据笔记, 不能只填数字不写理由."""
    issues = []
    for key, (name, _) in DIMENSIONS.items():
        d = candidate["scores"].get(key, {})
        score = d.get("score", 0)
        note = d.get("note", "")
        if not (1 <= score <= 5):
            issues.append(f"「{name}」缺分数或分数越界(应为1-5): 当前 {score}")
        if score and not note.strip():
            issues.append(f"「{name}」打了 {score} 分却没写依据 —— 空口打分等于没打分")
    return {"issues": issues, "ready": not issues}


def compare(candidates: list[dict]) -> list[dict]:
    """按总分降序排列候选方向, 附带各维度短板提示."""
    ranked = sorted(candidates, key=total, reverse=True)
    for c in ranked:
        weakest_key = min(c["scores"], key=lambda k: c["scores"][k]["score"])
        c["_weakest"] = DIMENSIONS[weakest_key][0]
    return ranked


def render(candidates: list[dict]) -> str:
    ranked = compare(candidates)
    lines = ["=== 候选研究方向对比 ==="]
    for i, c in enumerate(ranked, 1):
        lines.append(f"\n{i}. {c['name']}  (总分 {total(c)}/20, 最弱项: {c['_weakest']})")
        for key, (name, _) in DIMENSIONS.items():
            d = c["scores"][key]
            lines.append(f"   {name}: {d['score']}分 —— {d['note'] or '(未填依据)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = blank_candidate("方向A: 长上下文推理效率")
    a["scores"]["interest"] = {"score": 4, "note": "读过的相关论文都主动做了笔记, 是真兴趣"}
    a["scores"]["lab_fit"] = {"score": 5, "note": "long-context专题的代码和数据直接能复用"}
    a["scores"]["funding_fit"] = {"score": 3, "note": "关注度稳定但不算上升期, 竞争者不少"}
    a["scores"]["career_fit"] = {"score": 4, "note": "工业界效率方向岗位需求持续存在"}

    b = blank_candidate("方向B: 多模态对齐评测")
    b["scores"]["interest"] = {"score": 2, "note": "是最近才因为热度关注, 还没深入读过几篇"}
    b["scores"]["lab_fit"] = {"score": 1, "note": "实验室没有多模态数据/算力积累, 要从零搭"}
    b["scores"]["funding_fit"] = {"score": 5, "note": "明显上升期, 顶会track专门扩了"}
    b["scores"]["career_fit"] = {"score": 3, "note": "方向新, 但和自己规划的NLP路线略有偏离"}

    print(render([a, b]))
    print("\n=== 打分完整性自检 ===")
    for c in [a, b]:
        chk = audit(c)
        status = "✅ 完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])
        print(f"{c['name']}: {status}")
    print("\n→ 总分最高不代表直接选它: 方向A总分更高, 但方向B的资助趋势明显更强 —— "
          "这正是L2「项目级可行性评估」要接着回答的问题: 方向A的lab_fit优势能否弥补"
          "funding_fit的中庸, 需要结合更长尺度的可行性判断, 不能只看这一张总分表。")
