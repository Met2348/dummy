"""Verifier demo — sympy-style equivalence checking.

Real `math-verify` library uses sympy to handle algebraic equivalences
(x+y == y+x, 1/2 == 0.5, sin(0)==0). Here we ship a teaching version
that handles the common cases without external deps.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional


def normalize_math(s: str) -> str:
    """Strip whitespace, latex command markers, dollar signs."""
    s = s.strip()
    s = s.replace("$", "")
    s = s.replace("\\,", "").replace("\\;", "")
    # Unwrap \boxed{...} (preserve braces of nested \frac{a}{b})
    m = re.search(r"\\boxed\{(.+)\}\s*$", s)
    if m:
        s = m.group(1)
    s = re.sub(r"\s+", "", s)
    return s


def parse_to_float(s: str) -> Optional[float]:
    s = normalize_math(s)
    if not s:
        return None
    # Fraction "a/b"
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    # Percentage "50%"
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            return None
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    # plain number
    try:
        return float(s)
    except ValueError:
        return None


def equiv(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """Equivalence check across algebraic forms."""
    fp = parse_to_float(pred)
    fg = parse_to_float(gold)
    if fp is not None and fg is not None:
        return abs(fp - fg) <= tol
    return normalize_math(pred) == normalize_math(gold)


def _self_test() -> int:
    assert equiv("0.5", "1/2")
    assert equiv("0.5", "\\frac{1}{2}")
    assert equiv("50%", "0.5")
    assert equiv("\\boxed{18}", "18")
    assert equiv("\\boxed{ 18 }", "18")
    assert not equiv("17", "18")
    assert equiv("Paris", "Paris")  # non-numeric fallback
    assert not equiv("Paris", "Lyon")
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"math_verify_demo.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
