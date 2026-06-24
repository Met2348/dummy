"""Thinking budget forcing - early-stop / Wait injection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class BudgetResult:
    tokens: List[str]
    forced_close: bool
    thinking_tokens: int
    answer_tokens: int


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANS_OPEN = "<answer>"
ANS_CLOSE = "</answer>"


def generate_with_budget(stream: Iterable[str], budget_tokens: int) -> BudgetResult:
    """Consume `stream` token-by-token; force-close <think> when budget exhausted."""
    out: List[str] = []
    in_think = False
    forced = False
    think_count = 0
    for tok in stream:
        if tok == THINK_OPEN:
            in_think = True
        out.append(tok)
        if in_think and tok != THINK_OPEN:
            think_count += 1
        if tok == THINK_CLOSE:
            in_think = False
        if in_think and think_count >= budget_tokens:
            # force-close
            out.append(THINK_CLOSE)
            in_think = False
            forced = True
            break
    return BudgetResult(
        tokens=out,
        forced_close=forced,
        thinking_tokens=think_count,
        answer_tokens=len(out) - think_count,
    )


def inject_wait_tokens(stream: List[str], every_n: int = 50) -> List[str]:
    """Inject 'Wait' tokens to elongate thinking, s1-style."""
    out: List[str] = []
    in_think = False
    count = 0
    for tok in stream:
        out.append(tok)
        if tok == THINK_OPEN:
            in_think = True
        elif tok == THINK_CLOSE:
            in_think = False
        if in_think:
            count += 1
            if count % every_n == 0:
                out.append("Wait")
    return out


def demo() -> None:
    print("=== Thinking budget forcing（s1 风格：预算耗尽强制收尾 + Wait 注入）===")
    stream = ["<think>"] + [f"t{i}" for i in range(20)] + ["</think>", "<answer>", "42", "</answer>"]
    r = generate_with_budget(stream, budget_tokens=8)
    print(f"budget=8: thinking_tokens={r.thinking_tokens} forced_close={r.forced_close}")
    print(f"  强制收尾后末段: {r.tokens[-4:]}")
    short = ["<think>"] + [f"t{i}" for i in range(5)] + ["</think>"]
    print(f"inject_wait(every_n=2): {inject_wait_tokens(short, every_n=2)}")


if __name__ == "__main__":
    demo()
