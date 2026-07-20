"""
funding_plan_audit.py — 经费与资源计划自检: 检查一份经费/资源申请是否具备评审会追问的
几块内容, 而不是只有一句"需要更多算力"。

五块骨架 (对应 L1-L5):
  budget_justification 预算论证 (每一项花费为什么必要)
  compute_plan          算力资源规划 (需要多少/什么时候用/备选方案)
  data_management       数据管理规划 (DMP, 存储/共享/留存策略)
  collaboration_mou     多机构合作协议 (谁出资源/谁担责任/怎么分署名)
  vendor_compliance     第三方/供应商合规审查 (数据处理协议/使用许可)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("budget_justification", "预算论证", "每一项花费都对应具体产出, 而不是笼统的'研究经费'"),
    ("compute_plan", "算力资源规划", "写清楚需要多少GPU-hours/什么时候用/申请不到时的备选方案"),
    ("data_management", "数据管理规划", "数据存哪/谁能访问/论文发表后是否公开, 全部写清楚"),
    ("collaboration_mou", "多机构合作协议", "谁出资源/谁担责任/怎么分署名, 提前写清楚而不是出结果后再吵"),
    ("vendor_compliance", "第三方/供应商合规审查", "用到的API/数据集的使用许可是否核实过"),
]


def blank_funding_plan() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(plan: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = plan.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
    return {"issues": issues, "ready": not issues,
            "weak_sections": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(plan: dict) -> str:
    lines = ["=== 经费与资源计划骨架 ==="]
    for key, name, hint in SECTIONS:
        content = plan.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


def reviewer_focus(plan: dict) -> list[str]:
    chk = audit(plan)
    prompts = {
        "预算论证": "这一项花费如果砍掉一半, 研究计划还能执行吗?",
        "算力资源规划": "申请的算力被砍到一半, B计划是什么?",
        "数据管理规划": "论文发表后数据/代码是否公开? 隐私数据怎么处理?",
        "多机构合作协议": "如果合作方中途退出, 责任怎么划分?",
        "第三方/供应商合规审查": "这些数据/API的使用许可覆盖你打算发表的用途吗?",
    }
    return [prompts.get(w, f"「{w}」请准备被追问") for w in chk["weak_sections"]]


if __name__ == "__main__":
    good = blank_funding_plan()
    good["budget_justification"] = "80% 预算用于GPU-hours(对应3个大规模实验), 20%用于会议差旅。"
    good["compute_plan"] = "需要8×A100持续2个月; 若砍半, 优先保留可解释性实验, 砍训练规模。"
    good["data_management"] = "训练数据存内部集群, 论文发表后仅公开代码和评测脚本, 不公开原始训练数据。"
    good["collaboration_mou"] = "对方实验室出算力, 我方出算法设计, 通讯作者由数据主要贡献方担任, 已书面确认。"
    good["vendor_compliance"] = "核实过所用API的商用条款, 确认允许用于论文发表和非商业研究。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ " + "; ".join(chk["issues"])))
