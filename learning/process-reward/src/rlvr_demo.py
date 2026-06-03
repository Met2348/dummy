"""RLVR — Reinforcement Learning with Verifiable Rewards.

idea: 用规则 / 程序 / 数学校验当 reward，避免 RM hacking.
适用：数学 / 代码 / 形式化推理.
"""
from __future__ import annotations

import re


def gsm8k_reward(response: str, ground_truth: str) -> float:
    """提取 #### 后的答案，与 ground truth 对照."""
    m = re.search(r"####\s*([+-]?\d+\.?\d*)", response)
    if not m:
        return 0.0
    pred = m.group(1).strip()
    return 1.0 if pred == ground_truth.strip() else 0.0


def code_reward(code: str, test_cases: list[tuple[str, str]]) -> float:
    """运行 code 用 test cases 验证."""
    import io
    import contextlib

    passed = 0
    for input_str, expected in test_cases:
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec_globals = {"input": lambda: input_str}
                exec(code, exec_globals)
            if buf.getvalue().strip() == expected.strip():
                passed += 1
        except Exception:
            continue
    return passed / max(len(test_cases), 1)


def equation_reward(response: str, target: float, tol: float = 1e-3) -> float:
    """Countdown: 提取最后一个表达式 → eval → 与 target 对比."""
    expressions = re.findall(r"[\d+\-*/().\s]+", response)
    for expr in reversed(expressions):
        try:
            val = eval(expr, {"__builtins__": {}}, {})
            if abs(val - target) < tol:
                return 1.0
        except Exception:
            continue
    return 0.0


def format_reward(response: str,
                  pattern: str = r"<think>.+?</think>\s*<answer>.+?</answer>") -> float:
    """检查格式合规."""
    return 1.0 if re.search(pattern, response, re.DOTALL) else 0.0


def combined_rlvr_reward(response: str, gt_answer: str,
                          alpha: float = 0.1) -> dict:
    """混合 reward = alpha · format + (1-alpha) · accuracy."""
    f = format_reward(response)
    a = gsm8k_reward(response, gt_answer)
    return {
        "format": f,
        "accuracy": a,
        "total": alpha * f + (1 - alpha) * a,
    }


if __name__ == "__main__":
    print("RLVR demo\n" + "=" * 50)
    # 1. GSM8K
    print("\n[GSM8K]")
    for resp, gt in [
        ("Janet has 16 - 3 - 6 = 7 eggs. #### 7", "7"),
        ("I think it's 8. #### 8", "7"),
        ("No structured answer", "7"),
    ]:
        r = gsm8k_reward(resp, gt)
        print(f"  reward={r:.1f} | {resp[:50]}")

    # 2. Countdown
    print("\n[Countdown target=24]")
    for resp in ["Use 6*4 = 24", "2+3 = 5", "(8-2)*4 = 24", "no math"]:
        r = equation_reward(resp, target=24)
        print(f"  reward={r:.1f} | {resp}")

    # 3. Code
    print("\n[Code]")
    code = "print(int(input()) * 2)"
    cases = [("3", "6"), ("5", "10"), ("0", "0")]
    r = code_reward(code, cases)
    print(f"  code reward={r:.1%}")

    # 4. Format
    print("\n[Format reward]")
    for resp in [
        "<think>x</think><answer>7</answer>",
        "no tags here",
        "<think>x</think>missing answer",
    ]:
        r = format_reward(resp)
        print(f"  reward={r:.1f} | {resp[:40]}")

    # 5. Combined
    print("\n[Combined]")
    full = "<think>16-3-6=7</think><answer>#### 7</answer>"
    print(f"  {combined_rlvr_reward(full, '7')}")
