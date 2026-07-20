"""
application_kit_audit.py — 求职材料包自检工具: 检查一套申请材料是否具备说服力, 而不是
简历关键词堆砌 + 千篇一律的cover letter。

五块骨架 (对应 L2-L5):
  cv_tailoring          CV/履历定制
  recommenders          推荐人网络与请求策略
  talk_outline          Job talk/面试陈述大纲
  negotiation_prep      谈判准备
  narrative_consistency 跨材料叙事一致性 (CV/cover letter/job talk讲的是不是同一个故事)

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("cv_tailoring", "CV/履历定制", "针对具体岗位调整了侧重, 而不是海投同一份CV"),
    ("recommenders", "推荐人网络与请求策略", "提前沟通、给推荐人素材包, 而不是临时才开口"),
    ("talk_outline", "Job talk/面试陈述大纲", "有清晰的一条研究主线, 而不是罗列做过的项目"),
    ("negotiation_prep", "谈判准备", "明确底线/优先级(薪资/起始经费/团队规模), 而不是被动接受首次offer"),
    ("narrative_consistency", "跨材料叙事一致性", "CV/cover letter/job talk讲的是同一个故事"),
]

_VAGUE_HINTS = ["综合能力强", "学习能力强", "沟通能力好"]


def blank_package() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(package: dict) -> dict:
    issues = []
    for key, name, hint in SECTIONS:
        content = package.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
            continue
        if any(h in content for h in _VAGUE_HINTS):
            issues.append(f"「{name}」用了空泛套话(如'综合能力强') —— 换成具体可验证的事实")
    return {"issues": issues, "ready": not issues,
            "weak_sections": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(package: dict) -> str:
    lines = ["=== 求职材料包骨架 ==="]
    for key, name, hint in SECTIONS:
        content = package.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


def negotiation_focus(package: dict) -> list[str]:
    chk = audit(package)
    prompts = {
        "CV/履历定制": "针对目标岗位重新排列CV的重点顺序了吗?",
        "推荐人网络与请求策略": "给推荐人发过你的CV+这份岗位的具体侧重了吗?",
        "Job talk/面试陈述大纲": "如果只能讲一条研究主线, 是哪一条?",
        "谈判准备": "薪资/起始经费/团队规模, 你的底线分别是多少?",
        "跨材料叙事一致性": "CV和job talk里的故事是否会让人觉得是两个人?",
    }
    return [prompts.get(w, f"「{w}」这一节请补完") for w in chk["weak_sections"]]


if __name__ == "__main__":
    good = blank_package()
    good["cv_tailoring"] = "针对XX Lab的招聘方向, 把interp相关项目排在前面, 弱化早期工程复现经历。"
    good["recommenders"] = "提前2周联系导师+合作者, 附上目标岗位描述和自己想强调的3个点。"
    good["talk_outline"] = "主线: 从机制可解释性方法论到大规模验证, 一条技术演进链。"
    good["negotiation_prep"] = "底线: 起薪不低于X, 优先要独立预算而非团队规模。"
    good["narrative_consistency"] = "三份材料统一用'从机制理解到工程验证'这条主线, 已交叉核对。"
    print(render(good))
    chk = audit(good)
    print("\n完整性自检:", "✅ 骨架完整" if chk["ready"] else chk["issues"])

    print("\n--- 反面: 敷衍的材料包 ---")
    bad = blank_package()
    bad["cv_tailoring"] = "综合能力强, 适合任何岗位。"
    for i in audit(bad)["issues"]:
        print("⚠", i)
    print("\n--- 谈判/面试前准备清单 ---")
    for q in negotiation_focus(bad):
        print("?", q)
