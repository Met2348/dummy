"""把分级结果铺成"自主性阶梯 × 生命周期"二维地图，并渲染成纯文本表。

地图行 = 三级阶梯（按证据），列 = 七环生命周期；格子里是"该级别有几个系统自动化了该环"。
另提供逐系统对照表：自称 → 证据级别 + hype/可信度旗标。
"""
from __future__ import annotations

from .classifier import LEVELS, classify
from .systems import STAGES, STAGE_CODE, SYSTEM_CATALOG


def classify_all(catalog=None):
    cat = SYSTEM_CATALOG if catalog is None else catalog
    return [classify(s) for s in cat]


def ladder_lifecycle_grid(classifications):
    """level -> stage -> 自动化该环的系统数。"""
    grid = {lvl: {st: 0 for st in STAGES} for lvl in LEVELS}
    for c in classifications:
        for st in c.coverage:
            grid[c.evidenced_level][st] += 1
    return grid


def render_map(classifications) -> str:
    grid = ladder_lifecycle_grid(classifications)
    by_level = {lvl: [c.name for c in classifications if c.evidenced_level == lvl]
                for lvl in LEVELS}
    header = "  level     | " + " ".join(f"{STAGE_CODE[s]:>3}" for s in STAGES) + " | systems"
    lines = ["自主性阶梯 × 生命周期（格子=该级别自动化该环的系统数）", header,
             "  " + "-" * (len(header) - 2)]
    for lvl in reversed(LEVELS):   # scientist 在上
        cells = " ".join(f"{grid[lvl][s]:>3}" for s in STAGES)
        names = ", ".join(by_level[lvl]) or "—"
        lines.append(f"  {lvl:9} | {cells} | {len(by_level[lvl])}: {names}")
    return "\n".join(lines)


def render_table(classifications) -> str:
    lines = ["逐系统：自称 → 证据级别（[!]=hype gap，(self)=仅自评/可信度低）",
             f"  {'system':24}{'claimed':>10}{'evidenced':>11}  flags"]
    for c in sorted(classifications, key=lambda c: (-c.hype_gap, c.name)):
        flags = []
        if c.hype_gap > 0:
            flags.append(f"[!]hype+{c.hype_gap}")
        if c.self_verified_only:
            flags.append("(self)self-verified")
        lines.append(f"  {c.name:24}{c.claimed_level:>10}{c.evidenced_level:>11}  "
                     + " ".join(flags))
    return "\n".join(lines)
