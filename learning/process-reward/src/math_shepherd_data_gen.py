"""Math-Shepherd 风格自动生成 PRM 数据 (Wang 2024).

idea: 不需要人工标 step 级 label，用 MC rollout 自动生成.
    1. 从某 step prefix 出发 rollout N 次
    2. 若 >70% 到正确答案 → label = good
    3. 若 <30% → bad
    4. 中间 → neutral
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepLabel:
    step_text: str
    label: str           # "good" | "neutral" | "bad"
    success_rate: float  # rollout 成功率


def label_from_rollouts(success_count: int, total: int,
                        upper: float = 0.7, lower: float = 0.3) -> str:
    rate = success_count / max(total, 1)
    if rate >= upper:
        return "good"
    if rate <= lower:
        return "bad"
    return "neutral"


def auto_label_steps(question: str, steps: list[str], rollout_fn,
                     n_per_step: int = 8) -> list[StepLabel]:
    """每个 step prefix 做 n_per_step 次 rollout，按成功率打 label.

    rollout_fn(question, step_prefix) -> bool (是否到正确答案).
    """
    labels = []
    for i, step in enumerate(steps):
        prefix = "\n".join(steps[: i + 1])
        success = sum(rollout_fn(question, prefix) for _ in range(n_per_step))
        rate = success / n_per_step
        label = label_from_rollouts(success, n_per_step)
        labels.append(StepLabel(step, label, rate))
    return labels


def to_prm_training_jsonl(question: str, steps_labels: list[StepLabel]) -> dict:
    """转 PRM 训练 jsonl 格式."""
    return {
        "question": question,
        "steps": [s.step_text for s in steps_labels],
        "labels": [s.label for s in steps_labels],
        "success_rates": [s.success_rate for s in steps_labels],
    }


if __name__ == "__main__":
    print("Math-Shepherd PRM 数据自动生成 demo\n" + "=" * 50)

    # mock：假装我们有 rollout_fn，能从 prefix 接着算
    import random
    random.seed(42)

    def mock_rollout(_q, prefix: str) -> bool:
        # 简化：prefix 越长越容易对（更多 hints）
        return random.random() < 0.3 + 0.15 * prefix.count("\n")

    q = "Janet has 16 eggs. She eats 3 and sells 6. How many left?"
    steps = [
        "Step 1: Start with 16 eggs.",
        "Step 2: She eats 3, so 16 - 3 = 13.",
        "Step 3: She sells 6, so 13 - 6 = 7.",
        "Answer: 7",
    ]
    labels = auto_label_steps(q, steps, mock_rollout, n_per_step=10)
    for lbl in labels:
        marker = {"good": "OK", "neutral": "?", "bad": "BAD"}[lbl.label]
        print(f"  {marker} [{lbl.label:7s}] rate={lbl.success_rate:.1%} | {lbl.step_text[:50]}")

    out = to_prm_training_jsonl(q, labels)
    print(f"\nPRM 训练样本（JSON 格式）:\n  keys: {list(out.keys())}")
    print("  -> 可拼接成 1k-10k 条，训出 PRM。")
