"""
meeting_prep.py — 导师 meeting 准备工具: 把"和导师聊聊"变成有议程、有明确 ask 的高效会.

为什么需要它: 导师时间极有限 (一周可能只有 30 分钟给你)。新手最大的浪费是: 没准备就去,
流水账汇报进展, 没有明确问题, 聊完什么决定都没做。高效 meeting 的关键是**结构化 + 有明确
的 ask** (你要导师帮你决定/解锁什么)。这个工具把一次 meeting 组织成四块, 并检查你有没有
带着明确的问题去 —— 没有 ask 的 meeting 基本是浪费双方时间。

四块结构:
  progress   自上次以来做了什么 (简洁, 1-2 句/项, 不流水账)
  blockers   卡在哪 (具体! 不是"有点难", 而是"X 试了 A/B 都不行")
  decisions  需要导师帮你决定的 (带选项 + 你的倾向, 而非开放式"咋办")
  next       下一步计划 (让导师确认方向对不对)

纯 stdlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("progress", "进展", "自上次以来做了什么. 简洁, 1-2 句/项. 别流水账"),
    ("blockers", "卡点", "具体卡在哪. 不是'有点难', 而是'X 试了 A/B 都失败, 因为...'"),
    ("decisions", "需决策", "要导师帮定的事. 带选项 + 你的倾向, 而非开放式'咋办'"),
    ("next", "下一步", "你的计划, 让导师确认方向"),
]


def build_agenda(progress=None, blockers=None, decisions=None, next_steps=None) -> dict:
    return {
        "progress": list(progress or []),
        "blockers": list(blockers or []),
        "decisions": list(decisions or []),
        "next": list(next_steps or []),
    }


def audit(agenda: dict) -> dict:
    """检查 meeting 准备是否高效: 有没有明确 ask / 卡点是否具体 / 决策是否带选项."""
    issues = []
    if not agenda.get("decisions") and not agenda.get("blockers"):
        issues.append("没有任何 ask (决策/卡点) —— 纯汇报型 meeting 浪费导师时间; "
                      "想清楚'我要他帮我什么'")
    for d in agenda.get("decisions", []):
        # 一个好的决策项应包含选项 (启发式: 含 'vs' / '还是' / 'A/B' / '选项')
        if not any(t in d for t in ["vs", "还是", "或", "选项", "A/B", "/"]):
            issues.append(f"决策项缺选项: 「{d[:30]}...」—— 带上你的备选 + 倾向, 别开放式提问")
    if not agenda.get("progress"):
        issues.append("没列进展 —— 哪怕'卡住了'也是进展, 让导师知道你在动")
    return {"issues": issues, "ready": not issues}


def render(agenda: dict) -> str:
    lines = ["=== 导师 Meeting 议程 ===\n"]
    for key, name, hint in SECTIONS:
        lines.append(f"## {name}  ({hint[:30]}...)")
        items = agenda.get(key, [])
        if items:
            for it in items:
                lines.append(f"  - {it}")
        else:
            lines.append("  (无)")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    good = build_agenda(
        progress=["跑完 Robust-DPO 噪声消融, 40% 噪声下 +13 点 (p<0.001)",
                  "复现了 baseline DPO 并公平调参"],
        blockers=["在 70B 上跑不动 (显存不够), 试了梯度检查点和 LoRA 都还 OOM"],
        decisions=["投 EMNLP (6月ddl, 时间紧) 还是 NeurIPS (5月ddl, 更对口)? 我倾向 EMNLP, 因为..."],
        next_steps=["补不同噪声类型的消融", "写 method 节初稿"],
    )
    print(render(good))
    print("准备就绪?", audit(good)["ready"])
    print("\n--- 反面: 纯汇报, 无 ask ---")
    bad = build_agenda(progress=["看了几篇论文"])
    for i in audit(bad)["issues"]:
        print("⚠", i)
