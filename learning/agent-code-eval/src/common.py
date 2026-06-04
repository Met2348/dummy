"""Shared utilities for agent-code-eval Topic 3."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


ModelFn = Callable[[str, int], str]


@dataclass
class CodeResult:
    qid: str
    pred_code: str
    passed: bool
    error: Optional[str] = None
    meta: Dict = field(default_factory=dict)


# === Code extraction ===

_CODE_RE = re.compile(r"```(?:python)?\n(.+?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract first python code block; fall back to full text."""
    m = _CODE_RE.search(text)
    if m:
        return m.group(1).rstrip()
    return text.strip()


# === Sandboxed exec ===

FORBIDDEN_PATTERNS = (
    "import os", "import sys", "import subprocess", "import shutil",
    "__import__", "open(", "exec(", "eval(", "compile(",
    "globals(", "locals(", "vars(", "dir(",
    "input(", "exit(", "quit(",
)


def is_safe_code(code: str) -> bool:
    """Surface-level safety check. Real sandbox needs AST + subprocess."""
    return not any(pat in code for pat in FORBIDDEN_PATTERNS)


def safe_exec(code: str, test_assertions: str, timeout: float = 2.0) -> Optional[str]:
    """Exec code + tests in a restricted namespace.

    Returns:
        None on success
        Error string on failure
    """
    if not is_safe_code(code):
        return "BLOCKED: forbidden pattern"
    ns: Dict = {
        "__builtins__": {
            "range": range, "len": len, "sum": sum, "abs": abs,
            "min": min, "max": max, "sorted": sorted, "reversed": reversed,
            "list": list, "tuple": tuple, "dict": dict, "set": set,
            "str": str, "int": int, "float": float, "bool": bool,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "all": all, "any": any, "round": round, "pow": pow,
            "ord": ord, "chr": chr, "print": lambda *a, **k: None,
            "AssertionError": AssertionError, "ValueError": ValueError,
            "TypeError": TypeError, "Exception": Exception,
            "isinstance": isinstance, "type": type, "callable": callable,
            "None": None, "True": True, "False": False,
        }
    }
    try:
        exec(code, ns)
        exec(test_assertions, ns)
        return None
    except AssertionError as e:
        return f"AssertionError: {e}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# === Mock models ===

def make_mock_model(answers: Dict[str, str], default: str = "") -> ModelFn:
    def _fn(prompt: str, max_new_tokens: int = 512) -> str:
        for key, ans in answers.items():
            if f"[qid={key}]" in prompt:
                return ans
        return default
    return _fn


# === Aggregation ===

def pass_rate(results: List[CodeResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def _self_test() -> int:
    # extract_code
    assert extract_code("```python\ndef f(): return 1\n```") == "def f(): return 1"
    assert extract_code("def g(): return 2") == "def g(): return 2"
    # safe_exec ok
    err = safe_exec("def add(a,b): return a+b", "assert add(2,3) == 5")
    assert err is None, err
    # safe_exec failure
    err = safe_exec("def add(a,b): return a-b", "assert add(2,3) == 5")
    assert err is not None and "AssertionError" in err
    # forbidden
    err = safe_exec("import os\nprint(os)", "pass")
    assert err is not None and "BLOCKED" in err
    # mock
    m = make_mock_model({"q1": "code1", "q2": "code2"})
    assert m("[qid=q1] ...") == "code1"
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"common.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
