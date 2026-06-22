"""
talk_planner.py — 把一篇论文/研究装配成一个有时间预算的 talk 分镜 (storyboard), 并自检结构.

为什么需要它: 新手做 talk 的头号错误是「把论文从头念一遍」—— 但 talk 是**听觉、单向、限时**
的媒介, 和论文 (视觉、可回看、不限时) 完全不同。一个好 talk 是为这个媒介**重新设计**的:
一个 takeaway、少字、视觉化、严格卡时间。这个工具帮你:
  1. 按 talk 时长**分配时间预算**到各部分 (motivation/approach/result/takeaway)。
  2. 估算**幻灯片数**是否合理 (经验: 平均 ~1 张/分钟, 别堆太密)。
  3. **自检结构**: 有没有唯一 takeaway? 时间超不超? motivation 是否过长 (新手通病)?

纯 stdlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 一个会议 talk 的标准结构 + 各部分**建议时间占比** (经验值)
TALK_STRUCTURE = [
    ("motivation", "动机/问题", 0.20, "为什么这个问题重要 + gap. 别超时! 新手最爱在这拖太久"),
    ("approach", "方法/思路", 0.30, "核心 idea (不是所有细节). 一张 Figure1 讲清"),
    ("result", "关键结果", 0.30, "主结果 + 那张交互效应图. 带误差棒 (诚实)"),
    ("takeaway", "结论/意义", 0.20, "唯一的 takeaway + future work. 让人记住一句话"),
]

SLIDES_PER_MIN = 1.0     # 经验: 平均约 1 张/分钟 (复杂图可多停, 过渡页少停)
MOTIVATION_CAP = 0.30    # motivation 超过 30% 就是"铺垫太长"的红线


def plan(total_min: float, takeaway: str, slides: dict | None = None) -> dict:
    """生成 talk 分镜: 各部分时间预算 + 建议幻灯片数. slides 可给每部分实际打算用几张."""
    parts = []
    for key, name, frac, hint in TALK_STRUCTURE:
        minutes = round(total_min * frac, 1)
        suggested_slides = max(1, round(minutes * SLIDES_PER_MIN))
        actual = (slides or {}).get(key)
        parts.append({
            "key": key, "name": name, "minutes": minutes,
            "suggested_slides": suggested_slides,
            "actual_slides": actual, "hint": hint,
        })
    return {"total_min": total_min, "takeaway": takeaway, "parts": parts}


def audit(plan_obj: dict) -> dict:
    """自检 talk 结构: takeaway 是否唯一且具体 / 时间是否超 / motivation 是否过长 / slide 是否过密."""
    issues = []
    # 1. takeaway
    tk = plan_obj.get("takeaway", "").strip()
    if not tk:
        issues.append("没有写明唯一 takeaway —— 一个 talk 必须让听众记住一句话")
    elif len(tk) > 60:
        issues.append("takeaway 太长 (>60 字), 浓缩成一句话 —— 它要能被复述")

    # 2. motivation 占比
    mot = next(p for p in plan_obj["parts"] if p["key"] == "motivation")
    if mot["minutes"] / plan_obj["total_min"] > MOTIVATION_CAP:
        issues.append(f"motivation 占比偏高 —— 新手通病, 听众更想早点听到你做了什么")

    # 3. slide 密度 (若给了 actual)
    actual_total = sum(p["actual_slides"] for p in plan_obj["parts"] if p["actual_slides"])
    if actual_total:
        density = actual_total / plan_obj["total_min"]
        if density > 1.8:
            issues.append(f"幻灯片过密 ({actual_total}张/{plan_obj['total_min']}分≈{density:.1f}张/分), "
                          f"每张停留太短, 听众跟不上 —— 砍 slide 或加时间")
    return {"issues": issues, "ok": not issues}


def render(plan_obj: dict, audit_obj: dict | None = None) -> str:
    lines = [f"Talk 分镜 ({plan_obj['total_min']} 分钟)",
             f"唯一 takeaway: 「{plan_obj['takeaway']}」\n"]
    lines.append(f"{'部分':<12}{'时间':>6}{'建议slide':>10}   提示")
    lines.append("-" * 64)
    for p in plan_obj["parts"]:
        slide_str = f"{p['suggested_slides']}"
        if p["actual_slides"]:
            slide_str += f"(实{p['actual_slides']})"
        lines.append(f"{p['name']:<12}{p['minutes']:>5}m{slide_str:>10}   {p['hint'][:24]}")
    if audit_obj is not None:
        lines.append("\n结构自检:")
        if audit_obj["ok"]:
            lines.append("  ✅ 结构合理")
        else:
            for i in audit_obj["issues"]:
                lines.append(f"  ⚠ {i}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = plan(12, "Robust-DPO 的优势随偏好噪声增大而扩大",
             slides={"motivation": 3, "approach": 4, "result": 5, "takeaway": 2})
    print(render(p, audit(p)))
    print("\n--- 反面: motivation 太长 + 无 takeaway ---")
    bad = plan(10, "")
    bad["parts"][0]["minutes"] = 4.0  # 强行把 motivation 拉到 40%
    print(render(bad, audit(bad)))
