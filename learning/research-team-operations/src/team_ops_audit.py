"""
team_ops_audit.py — 团队运营计划自检: 检查一份多人协作计划是否具备可执行的运营骨架,
而不是"大家自己看着办"式的隐性协调。

五块骨架 (对应 L1-L5):
  time_allocation         多项目并行时间分配计划
  recruiting_criteria     合作者/实习生招募标准
  async_protocol          远程/异步协作规范
  onboarding_docs         新人onboarding文档计划
  cross_discipline_bridge 跨专业背景沟通桥接计划

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("time_allocation", "多项目并行时间分配", "每周固定时间块分给哪个项目, 而不是'有空就做'"),
    ("recruiting_criteria", "合作者/实习生招募标准", "写清楚必须具备的能力和可以现学的能力, 不是'越强越好'"),
    ("async_protocol", "远程/异步协作规范", "响应时限/会议时区/文档同步方式写清楚"),
    ("onboarding_docs", "新人onboarding文档", "新人第一周能不看你就跑起来baseline"),
    ("cross_discipline_bridge", "跨专业沟通桥接", "和非ML背景的合作者(工程师/PM)有没有共享术语表"),
]


def blank_ops_plan() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(plan: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = plan.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
        elif len(content) < 15:
            issues.append(f"「{name}」内容过短, 疑似敷衍")
    return {"issues": issues, "ready": not issues}


def render(plan: dict) -> str:
    lines = ["=== 团队运营计划骨架 ==="]
    for key, name, hint in SECTIONS:
        content = plan.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


if __name__ == "__main__":
    good = blank_ops_plan()
    good["time_allocation"] = "周一三五上午专属项目A, 周二四上午项目B, 下午统一留给突发协作。"
    good["recruiting_criteria"] = "必须: 会读PyTorch代码。可现学: 具体某个子领域背景。"
    good["async_protocol"] = "24小时内响应non-urgent消息, 每周一份async周报替代部分会议。"
    good["onboarding_docs"] = "首周: 环境搭建doc + 跑通一个baseline脚本 + 结对review一次PR。"
    good["cross_discipline_bridge"] = "和工程团队共享一份术语对照表(如'ablation'对应他们的'A/B测试')。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
