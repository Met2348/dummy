"""
paper_assembler.py — 把 9.1-9.6 的研究产物装配成论文骨架, 并检查叙事链完整性.

为什么需要它: 你做完 9.1-9.6, 手里有一堆零散产物 —— mini-survey(9.2)、gap卡(9.3)、
假设卡(9.4)、实验结果(9.4/9.5)、出版级图(9.6)。论文不是把它们堆起来, 而是用一条**叙事**
把它们串成闭合证据链 (这是 how_to_write_a_paper 的核心: 叙事先行)。这个工具做两件事:
  1. 把你的产物**映射**到标准论文骨架的各节 (assemble)。
  2. **审计叙事链**: 每个 claim 有没有对应的 evidence? 哪节还空着? (narrative_audit)

它不替你写论文 (写作细则全在 how_to_write_a_paper 技能包)。它帮你**在动笔前**确认:
材料齐不齐、证据链通不通。纯 stdlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 标准顶会论文骨架 (对应 how_to_write_a_paper/references/sections.md)
SECTIONS = [
    ("title", "标题", "一句话点出贡献, 不卖关子"),
    ("abstract", "摘要", "问题→方法→关键结果→意义, 4-6 句"),
    ("intro", "引言", "动机 + gap + 我们做了什么 + 贡献列表"),
    ("related", "相关工作", "领域地图 (9.2), 指出我们补的洞"),
    ("method", "方法", "可证伪假设的方法实现 + Figure 1 示意图 (9.6)"),
    ("experiments", "实验", "主结果 + 公平 baseline (9.4-L3) + 带误差棒图 (9.6)"),
    ("ablation", "消融", "消融矩阵 + 交互效应 (9.4-L4)"),
    ("discussion", "讨论", "为什么 work + 适用边界"),
    ("limitations", "局限", "诚实写出, 含不可复现点 (9.5)"),
]

# 产物 → 喂给哪些节
ARTIFACT_TO_SECTION = {
    "narrative": ["title", "abstract", "intro"],
    "mini_survey": ["related", "intro"],
    "gap": ["intro", "related"],
    "hypothesis": ["method", "intro"],
    "results": ["experiments", "abstract"],
    "ablation": ["ablation"],
    "figures": ["method", "experiments"],
    "limitations": ["limitations"],
}


def assemble_skeleton(artifacts: dict) -> str:
    """把产物字典装配成一个带占位的论文骨架 markdown."""
    lines = ["# 论文骨架 (auto-assembled)\n",
             "> 由 9.1-9.6 产物装配. 写作细则见 `how_to_write_a_paper` 技能包 (叙事先行).\n"]
    # 反向索引: 每节有哪些产物可用
    sec_arts: dict[str, list[str]] = {k: [] for k, _, _ in SECTIONS}
    for art, secs in ARTIFACT_TO_SECTION.items():
        if artifacts.get(art):
            for s in secs:
                if s in sec_arts:
                    sec_arts[s].append(art)
    for key, name, hint in SECTIONS:
        lines.append(f"## {name}")
        lines.append(f"<!-- {hint} -->")
        avail = sec_arts.get(key, [])
        if avail:
            lines.append(f"可用素材: {', '.join(avail)}")
        else:
            lines.append("⚠ 暂无对应素材 —— 需补 (见 narrative_audit)")
        lines.append("")
    return "\n".join(lines)


def narrative_audit(artifacts: dict, claims: list[dict] | None = None) -> dict:
    """审计叙事链: ① 哪些节缺素材; ② 每个 claim 是否有 evidence 支撑.

    claims: [{"claim": "...", "evidence": "results"|None}] —— 论文的核心主张及其证据来源。
    """
    # 1. 节覆盖
    covered = {k: False for k, _, _ in SECTIONS}
    for art, secs in ARTIFACT_TO_SECTION.items():
        if artifacts.get(art):
            for s in secs:
                covered[s] = True
    missing = [name for (k, name, _) in SECTIONS if not covered[k]]

    # 2. claim→evidence 链
    claim_report = []
    if claims:
        for c in claims:
            ev = c.get("evidence")
            ok = bool(ev) and bool(artifacts.get(ev))
            claim_report.append({"claim": c["claim"], "evidence": ev, "supported": ok})

    n_sec = sum(covered.values())
    unsupported = [c for c in claim_report if not c["supported"]]
    return {
        "sections_covered": n_sec, "sections_total": len(SECTIONS),
        "missing_sections": missing,
        "claims": claim_report,
        "unsupported_claims": unsupported,
        "ready": (not missing) and (not unsupported),
    }


def render_audit(report: dict) -> str:
    lines = [f"叙事链审计: {report['sections_covered']}/{report['sections_total']} 节有素材"]
    if report["missing_sections"]:
        lines.append(f"  ⚠ 缺素材的节: {', '.join(report['missing_sections'])}")
    if report["claims"]:
        lines.append("  claim → evidence:")
        for c in report["claims"]:
            mark = "✅" if c["supported"] else "❌"
            lines.append(f"    {mark} 「{c['claim']}」← {c['evidence'] or '(无证据!)'}")
    verdict = "✅ 材料齐、证据链闭合, 可动笔" if report["ready"] \
        else "⚠ 还有缺口, 补齐再写 (无证据的 claim 是审稿人攻击点)"
    lines.append(f"  裁决: {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    arts = {
        "narrative": "Robust-DPO 在噪声偏好下更鲁棒",
        "mini_survey": "偏好优化领域地图",
        "gap": "DPO 假设偏好无噪声",
        "hypothesis": "40%噪声下 Robust-DPO win-rate ≥ DPO+5",
        "results": "noise=0.4: +13点, p<0.001",
        "ablation": "method×noise 全因子, 交互显著",
        "figures": "交互效应图 + Figure 1",
        # 故意漏 limitations 看审计能不能抓出
    }
    claims = [
        {"claim": "Robust-DPO 在高噪声下显著更优", "evidence": "results"},
        {"claim": "提升来自鲁棒机制而非工程 trick", "evidence": "ablation"},
        {"claim": "在所有规模上都成立", "evidence": None},  # 没证据!
    ]
    print(assemble_skeleton(arts)[:400], "...\n")
    print(render_audit(narrative_audit(arts, claims)))
