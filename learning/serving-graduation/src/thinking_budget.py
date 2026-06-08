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
