"""R1-Zero accuracy reward — Countdown / GSM8K verification."""
from __future__ import annotations

import re


def gsm8k_extract_answer(text: str) -> str | None:
    """GSM8K 答案在 #### 后面。

    例: "...so the answer is 42. #### 42"
    """
    m = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def gsm8k_reward(predicted_answer: str | None, ground_truth: str) -> float:
    """提取数字答案后比对。"""
    if predicted_answer is None:
        return 0.0
    try:
        return 1.0 if float(predicted_answer) == float(ground_truth) else 0.0
    except (ValueError, TypeError):
        return 0.0


def countdown_reward(predicted: str | None, numbers: list[int], target: int) -> float:
    """Countdown：用给定数字 + 四则运算等于 target.

    predicted 形如 "(3+5)*4" 或 "32"。
    """
    if predicted is None:
        return 0.0
    expr = predicted.strip()
    # 提取所有数字
    used = re.findall(r"\d+", expr)
    if sorted(used) != sorted(str(n) for n in numbers):
        return 0.0
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return 1.0 if result == target else 0.0
    except Exception:
        return 0.0


def _self_test():
    print("gsm8k:", gsm8k_extract_answer("the answer is 42. #### 42"))     # 42
    print("gsm8k bad:", gsm8k_extract_answer("answer: 42"))                # None
    print("gsm8k reward:", gsm8k_reward("42", "42"))                       # 1.0
    print("countdown:", countdown_reward("3*4+5", [3, 4, 5], 17))          # 1.0
    print("countdown bad:", countdown_reward("3+4+5", [3, 4, 5], 17))      # 0.0


if __name__ == "__main__":
    _self_test()
