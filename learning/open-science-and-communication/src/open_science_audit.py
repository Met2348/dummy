"""
open_science_audit.py — 开放科学实践自检: 检查一个项目是否具备可复现/可核验的开放
科学实践, 而不是发表后"代码以后会整理"的空头支票。

五块骨架 (对应 L1-L5):
  interdisciplinary_glossary 跨学科术语对照表 (和其他学科合作者对齐关键概念)
  public_communication        公众沟通材料 (面向非专业读者的准确摘要)
  preregistration             预注册计划 (提前公开假设和分析计划, 防止事后诸葛亮)
  artifact_release_plan       代码/数据发布规范 (发表时同步公开, 而非"以后")
  social_media_boundary       学术社交媒体边界 (公开发言哪些代表个人、哪些代表机构)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("interdisciplinary_glossary", "跨学科术语对照表", "和其他学科合作者是否对齐了关键概念的定义"),
    ("public_communication", "公众沟通材料", "有没有一份面向非专业读者、准确不夸大的摘要"),
    ("preregistration", "预注册计划", "假设和分析计划是否在跑实验前就已公开记录"),
    ("artifact_release_plan", "代码/数据发布规范", "发表时是否同步公开代码/数据, 而不是'以后整理'"),
    ("social_media_boundary", "学术社交媒体边界", "公开发言哪些代表个人观点、哪些代表机构, 是否写清楚过"),
]


def blank_release_plan() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(plan: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = plan.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
    return {"issues": issues, "ready": not issues}


def render(plan: dict) -> str:
    lines = ["=== 开放科学实践自检 ==="]
    for key, name, hint in SECTIONS:
        content = plan.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


if __name__ == "__main__":
    good = blank_release_plan()
    good["interdisciplinary_glossary"] = "和认知科学合作者共建了一份'注意力'一词在两个学科的定义对照。"
    good["public_communication"] = "写了一段200字的非专业摘要, 请非本领域朋友试读确认能看懂且不夸大结论。"
    good["preregistration"] = "在OSF上预注册了核心假设和统计检验方法, 时间戳早于正式实验开始。"
    good["artifact_release_plan"] = "投稿时代码已整理好放在匿名仓库, accept后24小时内公开正式仓库。"
    good["social_media_boundary"] = "个人twitter简介已声明'观点仅代表个人, 不代表所在机构'。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
