"""Best-of-N (BoN) + Reranking with RM/PRM.

Pipeline:
    1. 生成 N candidates (sample N times)
    2. 每个 candidate 用 RM (or PRM-aggregated) 打分
    3. 取分数最高的作为最终输出
"""
from __future__ import annotations

import torch


def best_of_n(
    candidates: list[str],
    scores: list[float],
) -> tuple[str, int, float]:
    """返回最高分 candidate, 索引, 分数."""
    idx = int(torch.tensor(scores).argmax().item())
    return candidates[idx], idx, scores[idx]


def majority_vote(answers: list[str]) -> tuple[str, dict]:
    """提取最常见答案."""
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts


def weighted_bon(answers: list[str], scores: list[float]) -> tuple[str, dict]:
    """每个 distinct answer 累加分数，取最高."""
    weight: dict[str, float] = {}
    for a, s in zip(answers, scores):
        weight[a] = weight.get(a, 0) + s
    best = max(weight, key=weight.get)
    return best, weight


def bon_compare_strategies(answers: list[str], rm_scores: list[float],
                            ground_truth: str) -> dict:
    """对比 greedy / majority / BoN / weighted-BoN."""
    greedy = answers[0]
    maj, maj_counts = majority_vote(answers)
    bo_n, _, _ = best_of_n(answers, rm_scores)
    w_bo_n, _ = weighted_bon(answers, rm_scores)
    return {
        "greedy": (greedy, greedy == ground_truth),
        "majority": (maj, maj == ground_truth),
        "bon": (bo_n, bo_n == ground_truth),
        "weighted_bon": (w_bo_n, w_bo_n == ground_truth),
    }


if __name__ == "__main__":
    print("BoN strategies smoke test\n" + "=" * 50)
    # 模拟 8 个 candidate
    answers = ["7", "8", "7", "5", "7", "7", "8", "6"]
    rm_scores = [0.9, 0.4, 0.85, 0.2, 0.92, 0.88, 0.5, 0.3]
    gt = "7"
    result = bon_compare_strategies(answers, rm_scores, gt)
    for k, (ans, ok) in result.items():
        mark = "✓" if ok else "✗"
        print(f"  {k:14s} → {ans}  {mark}")
