"""Common utilities for eval-foundations Topic 1.

Mock model backend + prompt formatters + simple accuracy aggregator.
Everything runs on CPU with zero external deps — designed for teaching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class EvalSample:
    """One eval sample. `gold` is the ground-truth answer string."""
    qid: str
    prompt: str
    gold: str
    meta: Dict = field(default_factory=dict)


@dataclass
class EvalResult:
    qid: str
    pred: str
    gold: str
    correct: bool
    meta: Dict = field(default_factory=dict)


# Mock backend protocol: callable taking (prompt, max_new_tokens) -> generated text
ModelFn = Callable[[str, int], str]


def make_mock_model(answers: Dict[str, str], default: str = "A") -> ModelFn:
    """Build a mock model that returns canned answer per qid prefix in prompt."""
    def _fn(prompt: str, max_new_tokens: int = 32) -> str:
        for key, ans in answers.items():
            if f"[qid={key}]" in prompt:
                return ans
        return default
    return _fn


def make_random_model(seed: int = 0, choices: str = "ABCD") -> ModelFn:
    """Deterministic 'random' baseline — useful for chance-level comparison."""
    import hashlib
    def _fn(prompt: str, max_new_tokens: int = 32) -> str:
        h = hashlib.md5(prompt.encode()).digest()[0]
        return choices[h % len(choices)]
    return _fn


# === Answer extraction ===

_LETTER_RE = re.compile(r"\b([A-D])\b")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_FINAL_RE = re.compile(r"final answer[:\s]+([^\n.]+)", re.IGNORECASE)


def extract_letter(text: str) -> Optional[str]:
    """Extract first A/B/C/D from generation."""
    m = _LETTER_RE.search(text.strip())
    return m.group(1) if m else None


def extract_boxed(text: str) -> Optional[str]:
    """Extract \\boxed{...} content."""
    m = _BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract 'Final answer: X' style."""
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def normalize_answer(s: str) -> str:
    """Lowercase + strip punctuation for fuzzy match."""
    s = s.lower().strip()
    s = re.sub(r"[\s\.\,\!\?\"\'\(\)]+", "", s)
    return s


# === Prompt formatting ===

def format_multiple_choice(question: str, options: List[str], k_shot: List[Tuple[str, List[str], str]] = None) -> str:
    """Format an n-shot MCQ prompt. options is list like ['Paris', 'London', ...]."""
    lines: List[str] = []
    for q, opts, ans in (k_shot or []):
        lines.append(_one_shot(q, opts, ans))
        lines.append("")
    lines.append(_one_shot(question, options, ans=None))
    return "\n".join(lines)


def _one_shot(q: str, opts: List[str], ans: Optional[str]) -> str:
    body = [f"Question: {q}"]
    for i, opt in enumerate(opts):
        body.append(f"{chr(ord('A')+i)}. {opt}")
    body.append("Answer:")
    if ans is not None:
        body.append(f" {ans}")
    return "\n".join(body)


# === Aggregation ===

def accuracy(results: List[EvalResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def group_accuracy(results: List[EvalResult], by: str) -> Dict[str, float]:
    """Group accuracy by a meta field (e.g., 'subject' for MMLU)."""
    buckets: Dict[str, List[EvalResult]] = {}
    for r in results:
        key = r.meta.get(by, "_")
        buckets.setdefault(key, []).append(r)
    return {k: accuracy(v) for k, v in buckets.items()}


# === Self test ===

def _self_test() -> int:
    failed = 0
    # extract_letter
    assert extract_letter("Answer: A.") == "A"
    assert extract_letter("I think B is correct") == "B"
    assert extract_letter("hello") is None
    # extract_boxed
    assert extract_boxed("So \\boxed{42}.") == "42"
    # normalize_answer
    assert normalize_answer("Hello, World!") == "helloworld"
    # mock model
    m = make_mock_model({"q1": "A", "q2": "B"})
    assert m("[qid=q1] Question...") == "A"
    assert m("[qid=q2] Question...") == "B"
    assert m("[qid=qX] Question...") == "A"  # default
    # accuracy
    rs = [EvalResult("1", "A", "A", True), EvalResult("2", "B", "A", False)]
    assert accuracy(rs) == 0.5
    return failed


if __name__ == "__main__":
    f = _self_test()
    print(f"common.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
