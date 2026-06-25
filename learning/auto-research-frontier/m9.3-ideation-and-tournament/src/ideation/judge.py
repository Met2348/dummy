"""LLM-as-judge 的缩小版：只看**文案**给 novelty 打分，并对"自家点子"有自偏好。

关键诚实点：评委**看不到 feasibility**（要执行才知道），只能从文本表面特征猜新颖度。
这正是 ideation 阶段的根本局限——也是它系统性看走眼的原因。
复用 llm-judge-arena 的"成对偏好 → 排序"思路（见 tournament.py 的 Elo）。
"""
from __future__ import annotations

# 评委眼里的"高级感"热词（命中越多越显新颖）——注意：和真实价值无关
BUZZWORDS = ("新颖", "exotic", "重参数化", "突破", "cutting-edge", "全新",
             "自适应", "大胆", "激进", "novel", "breakthrough")

SELF_BIAS = 0.3   # 对 source=="self" 的点子，凭空加的偏好分（= 自我吹捧）


def novelty(idea) -> float:
    """从文案数热词，归一化到 0..1。纯表面，不碰 config/feasibility。"""
    hits = sum(1 for w in BUZZWORDS if w in idea.text)
    return min(hits / 3.0, 1.0)


def judge_score(idea, use_self_bias: bool = True) -> float:
    """评委给的总分 = 文本新颖度 + （若是自家点子）自偏好加成。**不含可行性**。"""
    s = novelty(idea)
    if use_self_bias and idea.source == "self":
        s += SELF_BIAS
    return s


def prefer(a, b, use_self_bias: bool = True) -> str:
    """成对评判：判 judge_score 高者胜；平手按 id 字典序（确定性）。返回胜者 id。"""
    sa, sb = judge_score(a, use_self_bias), judge_score(b, use_self_bias)
    if sa > sb or (sa == sb and a.id < b.id):
        return a.id
    return b.id
