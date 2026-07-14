"""
proposal_audit.py — 开题报告自检工具: 检查一份proposal草稿是否具备正式立项该有的骨架.

为什么需要它: idea卡(critical-reading-gap专题的产出物)是给自己看的简略记录, 开题报告是给
评审小组看的正式文档。新手最常见的失败是直接把idea卡展开成大白话, 漏了评审小组必然会问的
几块内容 —— 尤其是"风险预案"这一节, 没有它会显得对自己的方向毫无自我怀疑, 反而降低可信度。

七块骨架 (对应 L3):
  background   研究背景与动机
  gap          文献综述与研究缺口定位 (呼应 critical-reading-gap 的 gap 分类学)
  question     具体研究问题/假设 (要求可证伪, 呼应 critical-reading-gap L4 的可证伪标准)
  method       初步方法设想
  timeline     时间线与里程碑规划
  risks        风险与应对预案
  contribution 预期贡献

也用于L4开题答辩准备: audit的结果直接对应评审小组最可能追问的薄弱环节。

纯 stdlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECTIONS = [
    ("background", "研究背景与动机", "为什么这个问题值得做, 讲清楚背景而不是直接跳进技术细节"),
    ("gap", "文献综述与研究缺口", "指出具体哪篇/哪类工作没做到什么, 而不是泛泛的文献罗列"),
    ("question", "具体研究问题/假设", "必须可证伪 —— 存在'跑出什么结果就承认假设不成立'的可能"),
    ("method", "初步方法设想", "不要求完整方案, 但要有具体到能被追问细节的雏形"),
    ("timeline", "时间线与里程碑", "按阶段给出可核对的时间点, 不是一句'预计一年完成'"),
    ("risks", "风险与应对预案", "至少列出方法可能失败的场景, 以及每个风险对应的B计划"),
    ("contribution", "预期贡献", "假装做完了, 用一句话说出结论会让人'哦!'还是'so what?'"),
]

_FALSIFIABLE_HINTS = ["比", "优于", "达到", "≥", ">=", "高于", "低于", "在...下"]
_VAGUE_TIMELINE_HINTS = ["尽快", "预计一年", "尽早", "适时"]


def blank_proposal() -> dict:
    return {key: "" for key, _, _ in SECTIONS}


def audit(proposal: dict) -> dict:
    """检查proposal是否具备正式立项该有的骨架, 返回缺项和薄弱项列表."""
    issues = []
    for key, name, hint in SECTIONS:
        content = proposal.get(key, "").strip()
        if not content:
            issues.append(f"缺「{name}」这一节 —— {hint}")
            continue
        if key == "question" and not any(h in content for h in _FALSIFIABLE_HINTS):
            issues.append(f"「{name}」看起来不够可证伪 —— 没有出现比较级/阈值类表述, "
                          f"检查是否只是一句模糊的研究方向陈述而非具体假设")
        if key == "timeline" and any(h in content for h in _VAGUE_TIMELINE_HINTS):
            issues.append(f"「{name}」用词过于模糊(如'尽快'/'预计一年') —— "
                          f"改成带具体阶段和核对点的里程碑")
        if key == "risks" and len(content) < 20:
            issues.append(f"「{name}」内容过短, 疑似敷衍 —— 风险预案至少要点出"
                          f"一个具体失败场景和对应B计划")
    return {"issues": issues, "ready": not issues,
            "weak_sections": [i.split("「")[1].split("」")[0] for i in issues if "「" in i]}


def render(proposal: dict) -> str:
    lines = ["=== 开题报告骨架 ==="]
    for key, name, hint in SECTIONS:
        content = proposal.get(key, "")
        lines.append(f"\n## {name}")
        lines.append(content if content else f"  (空 —— {hint})")
    return "\n".join(lines)


def defense_focus(proposal: dict) -> list[str]:
    """L4用: 把audit发现的薄弱环节, 转成评审小组最可能追问的问题清单."""
    chk = audit(proposal)
    prompts = {
        "background": "为什么现在做这个问题, 而不是三年前/三年后?",
        "文献综述与研究缺口": "这个缺口真的没人做过吗? 你查得够全面吗?",
        "具体研究问题/假设": "如果实验结果和你预期相反, 你打算怎么解读?",
        "初步方法设想": "如果这个方法不work, 你的备选方案是什么?",
        "时间线与里程碑": "如果第一年拿不到预期结果, 后面时间还够吗?",
        "风险与应对预案": "这个方向最可能失败在哪一步? 你想过吗?",
        "预期贡献": "假设做完了, 谁会真的在意这个结论?",
    }
    return [prompts.get(w, f"「{w}」这一节请准备被追问") for w in chk["weak_sections"]]


if __name__ == "__main__":
    good = blank_proposal()
    good["background"] = "长上下文推理在超过32k token后普遍出现注意力稀释, 但该现象缺乏系统量化。"
    good["gap"] = "现有工作(如XXX 2025)只报告了端到端准确率下降, 未定位是注意力层还是位置编码的问题。"
    good["question"] = "假设: 在64k token设定下, 注意力稀释导致的准确率下降比位置编码外推误差高≥2倍。"
    good["method"] = "构造分层探针实验, 分别固定注意力/位置编码为理想值, 对比准确率恢复幅度。"
    good["timeline"] = "第1-2月复现baseline; 第3-4月完成探针实验; 第5月出第一版结果; 第6月开始写作。"
    good["risks"] = "若探针实验无法分离两个因素, 备选方案是改用因果干预(activation patching)定位。"
    good["contribution"] = "首次量化区分长上下文两种退化机制的相对贡献, 指导后续工程优化优先级。"

    print(render(good))
    chk = audit(good)
    print("\n完整性自检:", "✅ 骨架完整" if chk["ready"] else chk["issues"])

    print("\n--- 反面: 敷衍的proposal ---")
    bad = blank_proposal()
    bad["background"] = "长上下文很重要。"
    bad["question"] = "研究长上下文推理问题。"
    bad["timeline"] = "预计一年完成。"
    for i in audit(bad)["issues"]:
        print("⚠", i)

    print("\n--- L4用: 把薄弱环节转成答辩追问预判 ---")
    for q in defense_focus(bad):
        print("?", q)
