"""阶段 5：mock 自动评审 —— 故意做成**可被 game**的，用来教 grading-its-own-homework。

真 AI Scientist 用 LLM 给自己的论文打分，"自己给自己改作业"是公认的循环性陷阱。
这里我们把这个缺陷**显式做出来并可复现**：评审的 novelty/overall 主要看"报告里宣称的
效果有多大"，于是一篇**夸大（甚至伪造）大效果**的报告能骗到高分——哪怕实验根本不可靠。

`demonstrate_gaming()` 把这点跑给你看：诚实的小真效果 vs 注水的大假效果，后者分更高。
教训：**别信自动评审的分，要看实验本身。**
"""
from __future__ import annotations

from typing import Dict


def _clip(x: float) -> float:
    return max(1.0, min(10.0, x))


def review(report_text: str, comparison: Dict) -> Dict:
    """给"论文"打分（1–10）。故意有缺陷：novelty 主要奖励"宣称的大效果"。"""
    delta = abs(comparison.get("delta", 0.0))

    # clarity：奖励结构完整（有图/表/假设）—— 这部分还算合理
    clarity = 4.0
    for marker in ("## 假设", "| 指标 |", "![", "## 结论"):
        if marker in report_text:
            clarity += 1.5

    # soundness：奖励"报了误差棒/多种子"—— 也还算合理
    soundness = 4.0
    if "±" in report_text:
        soundness += 2.0
    if "seeds" in report_text or "种子" in report_text:
        soundness += 1.5

    # novelty：★缺陷★ 只看"宣称的效果有多大"，完全不管它是否可靠/真实
    novelty = _clip(3.0 + 40.0 * delta)

    overall = _clip(0.5 * novelty + 0.3 * soundness + 0.2 * clarity)
    return {
        "novelty": round(_clip(novelty), 1),
        "soundness": round(_clip(soundness), 1),
        "clarity": round(_clip(clarity), 1),
        "overall": round(overall, 1),
        "recommendation": "accept" if overall >= 6 else "reject",
        "_caveat": "novelty 只看宣称效果大小 → 可被夸大/伪造刷分；勿信此分，看实验本身",
    }


def demonstrate_gaming() -> Dict:
    """跑给你看：注水的大假效果 比 诚实的小真效果 评分更高。"""
    honest_report = (
        "## 假设\n小真效果\n## 方法\nseeds=5\n| 指标 | 值 |\n结果 0.88 ± 0.01\n"
        "![fig](f.png)\n## 结论\n小幅提升"
    )
    honest_cmp = {"delta": 0.02, "combined_std": 0.01}        # 诚实：小但真
    rigged_report = "## 假设\n惊天效果\n结果 0.99\n## 结论\nSOTA"   # 没误差棒/没图/没表
    rigged_cmp = {"delta": 0.40, "combined_std": 0.0}          # 注水：巨大且"零方差"

    honest = review(honest_report, honest_cmp)
    rigged = review(rigged_report, rigged_cmp)
    return {
        "honest_small_real": honest,
        "rigged_big_fake": rigged,
        "gamed": rigged["overall"] > honest["overall"],
    }
