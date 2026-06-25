"""Reviewer 角色：一个**不负责生成、只负责挑错**的独立 agent（Reflexion 思想）。

研究 agent 的内核里，"自我批判"必须由一个职责分离的角色来做——这样它才会去查
"你引的文献到底检索到没有""有没有 baseline""新颖度是不是吹了"，而不是替作者圆场。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Critique:
    ungrounded_ids: tuple   # 引了但检索集里没有的 id（= 幻觉引用）
    flags: tuple            # 其它问题（缺 baseline / 新颖度过吹 等）
    suggestions: tuple      # 给 revise 的具体修法

    @property
    def ok(self) -> bool:
        return not self.ungrounded_ids and not self.flags


def review(draft, retrieved) -> Critique:
    retrieved_ids = {p.arxiv_id for p in retrieved}
    grounded = [c for c in draft.cited_ids if c in retrieved_ids]
    ungrounded = tuple(c for c in draft.cited_ids if c not in retrieved_ids)

    flags, sugg = [], []
    if ungrounded:
        flags.append(f"幻觉引用 {len(ungrounded)} 处：{','.join(ungrounded)} 不在检索集内")
        sugg.append("删除所有检索集外的引用（不许凭空引）")
    if not draft.has_baseline:
        flags.append("缺 baseline：没有可比对照，结论无法判强弱")
        sugg.append("补一个明确 baseline")
    # 新颖度 vs 可行性：自评很高却只有薄弱接地 → 大概率是 ideation-execution gap 的前兆
    if draft.claimed_novelty >= 0.7 and len(grounded) < 2:
        flags.append(f"新颖度自评 {draft.claimed_novelty:.2f} 过高，但仅 {len(grounded)} 篇接地支撑")
        sugg.append("把新颖度下调到与证据相称，并标注可行性风险")

    return Critique(ungrounded_ids=ungrounded, flags=tuple(flags), suggestions=tuple(sugg))
