"""s1 Budget Forcing for Test-Time Scaling.

idea: 推理时若模型早 stop（产 </think>），强行注入 "Wait" 让它继续想.
配 budget 控制：min_think_tokens / max_think_tokens.
"""
from __future__ import annotations

import re


def budget_force_decode_mock(
    initial_response: str,
    min_think_tokens: int,
    max_think_tokens: int,
    judge_continue,
):
    """模拟 budget forcing.

    若 think 长度 < min: 注入 "Wait" 让它继续
    若 think 长度 > max: 强制 close </think>
    """
    think_match = re.search(r"<think>(.*?)</think>", initial_response, re.DOTALL)
    if not think_match:
        return initial_response, "no_think"
    think = think_match.group(1)
    think_len = len(think.split())

    if think_len < min_think_tokens:
        # 注入 Wait, 强制继续
        rest = judge_continue(think + " Wait, let me reconsider...")
        return initial_response.replace(think, think + " Wait, " + rest), "extended"

    if think_len > max_think_tokens:
        truncated = " ".join(think.split()[:max_think_tokens])
        return initial_response.replace(think, truncated), "truncated"

    return initial_response, "within_budget"


def analyze_budget_distribution(responses: list[str]) -> dict:
    """分析 think 段长度分布."""
    lens = []
    for r in responses:
        m = re.search(r"<think>(.*?)</think>", r, re.DOTALL)
        if m:
            lens.append(len(m.group(1).split()))
    if not lens:
        return {"count": 0}
    import statistics
    return {
        "count": len(lens),
        "mean": statistics.mean(lens),
        "median": statistics.median(lens),
        "min": min(lens),
        "max": max(lens),
    }


if __name__ == "__main__":
    print("s1 Budget Forcing demo\n" + "=" * 50)

    def mock_continue(prefix: str) -> str:
        return "double-checking: yes 7 is correct"

    short = "<think>16-9=7</think><answer>7</answer>"
    long_text = "<think>" + " ".join(["analysis"] * 100) + "</think><answer>x</answer>"

    out, status = budget_force_decode_mock(short, 30, 100, mock_continue)
    print(f"[short input]: status={status}")
    print(f"  -> {out[:80]}...")

    out, status = budget_force_decode_mock(long_text, 30, 50, mock_continue)
    print(f"\n[long input]: status={status}")
    print(f"  -> think len after truncate is about 50")

    stats = analyze_budget_distribution([short, long_text])
    print(f"\nDistribution: {stats}")
