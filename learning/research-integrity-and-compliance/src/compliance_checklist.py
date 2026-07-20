"""
compliance_checklist.py — 科研合规自检: 检查一个项目在诚信/伦理/合规上是否有明显漏洞,
这些漏洞往往不是能力问题, 而是"没人提醒就压根没想到要处理"的盲区。

五块骨架 (对应 L1-L5):
  authorship_agreement 署名协议 (谁一作/贡献声明是否书面化)
  irb_status            IRB/伦理审查状态 (涉及人类被试/敏感数据是否走过审查)
  ip_disclosure         知识产权披露 (涉及专利/商业敏感内容是否提前披露)
  export_control        出口管制/跨境合规 (国际合作涉及的技术/数据跨境限制)
  disclosure_plan       负责任的风险披露计划 (发现安全漏洞/危险能力后的披露流程)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("authorship_agreement", "署名协议", "谁一作/谁通讯/贡献比例是否书面确认过, 而非默认约定俗成"),
    ("irb_status", "IRB/伦理审查状态", "涉及人类被试/敏感数据的项目是否已过伦理审查"),
    ("ip_disclosure", "知识产权披露", "涉及潜在专利/商业敏感方法, 是否已按机构要求提前披露"),
    ("export_control", "出口管制/跨境合规", "国际合作中涉及的技术/数据是否核实过出口管制限制"),
    ("disclosure_plan", "负责任的风险披露计划", "如果研究涉及安全漏洞/危险能力, 披露流程和时间线是否提前规划"),
]


def blank_compliance() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(project: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = project.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
    return {"issues": issues, "ready": not issues,
            "risk_flags": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(project: dict) -> str:
    lines = ["=== 科研合规自检 ==="]
    for key, name, hint in SECTIONS:
        content = project.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


if __name__ == "__main__":
    good = blank_compliance()
    good["authorship_agreement"] = "项目启动会上书面确认: A一作(实现+实验), B通讯(idea+资源), 已发邮件存档。"
    good["irb_status"] = "本项目不涉及人类被试或敏感个人数据, 已在项目文档标注'不适用'并说明理由。"
    good["ip_disclosure"] = "方法涉及的核心算法已按学校要求提交disclosure表, 等待技术转移办公室反馈。"
    good["export_control"] = "国际合作方来自受限清单外地区, 已核实无出口管制限制, 存档确认邮件。"
    good["disclosure_plan"] = "若发现模型存在可被滥用的能力, 先内部上报安全团队, 90天coordinated disclosure后再公开细节。"
    print(render(good))
    chk = audit(good)
    print("\n" + ("✅ 骨架完整" if chk["ready"] else "⚠ 风险: " + "; ".join(chk["issues"])))

    print("\n--- 反面: 完全没考虑合规的项目 ---")
    bad = blank_compliance()
    bad["authorship_agreement"] = "大家心照不宣。"
    for i in audit(bad)["issues"]:
        print("⚠", i)
