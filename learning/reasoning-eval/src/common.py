"""Shared utilities for reasoning-eval Topic 2."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional


ModelFn = Callable[[str, int], str]


@dataclass
class ReasoningResult:
    qid: str
    pred: str
    gold: str
    correct: bool
    meta: Dict = field(default_factory=dict)


# === Answer extraction ===

_GSM8K_ANS_RE = re.compile(r"####\s*(-?[\d,\.]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)[:\s]+(-?[\d,\.\$/]+)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"(-?\d+(?:[\.,]\d+)?)")


def extract_gsm8k(text: str) -> Optional[str]:
    """Extract '#### N' style or last number."""
    m = _GSM8K_ANS_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).replace("$", "").replace(",", "").strip()
    nums = _LAST_NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def extract_boxed(text: str) -> Optional[str]:
    m = _BOXED_RE.search(text)
    return m.group(1).strip() if m else None


# === Numeric equivalence ===

def safe_to_fraction(s: str) -> Optional[Fraction]:
    """Try parse '1/2', '0.5', '50%' into Fraction."""
    s = s.strip().replace(" ", "")
    try:
        if "/" in s and "\\" not in s:
            num, den = s.split("/")
            return Fraction(int(num), int(den))
        if s.endswith("%"):
            return Fraction(float(s[:-1])) / 100
        return Fraction(s) if "." not in s else Fraction(float(s)).limit_denominator(10**6)
    except (ValueError, ZeroDivisionError):
        return None


def numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    fa = safe_to_fraction(a)
    fb = safe_to_fraction(b)
    if fa is not None and fb is not None:
        return abs(float(fa - fb)) <= tol
    return a.strip().lower() == b.strip().lower()


# === Mock models ===

def make_mock_model(answers: Dict[str, str], default: str = "0") -> ModelFn:
    def _fn(prompt: str, max_new_tokens: int = 256) -> str:
        for key, ans in answers.items():
            if f"[qid={key}]" in prompt:
                return ans
        return default
    return _fn


def make_dummy_model(constant_answer: str = "0") -> ModelFn:
    def _fn(prompt: str, max_new_tokens: int = 256) -> str:
        return f"The answer is {constant_answer}. #### {constant_answer}"
    return _fn


# === Pass@k for sampled answers ===

def pass_at_k(per_sample: List[List[bool]]) -> Dict[int, float]:
    """Given per-task lists of k boolean correctness, compute pass@1/4/k."""
    if not per_sample:
        return {}
    out: Dict[int, float] = {}
    ks = [1, 4]
    max_k = max(len(s) for s in per_sample)
    if max_k not in ks:
        ks.append(max_k)
    for k in ks:
        if k > max_k:
            continue
        scores = []
        for sols in per_sample:
            top_k = sols[:k]
            scores.append(1.0 if any(top_k) else 0.0)
        out[k] = sum(scores) / len(scores)
    return out


def accuracy(rs: List[ReasoningResult]) -> float:
    return (sum(1 for r in rs if r.correct) / len(rs)) if rs else 0.0


# === Self test ===

def _self_test() -> int:
    # gsm8k extraction
    assert extract_gsm8k("So Janet has 16-3-4=9 eggs and earns 9*2=$18. #### 18") == "18"
    assert extract_gsm8k("Final answer: 42") == "42"
    assert extract_gsm8k("nothing here") is None
    # boxed
    assert extract_boxed("So \\boxed{42}.") == "42"
    # numeric_equal
    assert numeric_equal("0.5", "1/2")
    assert numeric_equal("50%", "0.5")
    assert numeric_equal("18", "18.0")
    assert not numeric_equal("17", "18")
    # pass_at_k
    pak = pass_at_k([[True, False], [False, False], [False, True]])
    assert pak[1] == 1/3  # only first has True at index 0
    assert pak[2] == 2/3  # 1 + 0 + 1 / 3
    # mock model
    m = make_mock_model({"q1": "#### 7"})
    assert "#### 7" in m("[qid=q1] x+5=12")
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"common.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
