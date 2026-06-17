"""Tiny shared utilities for the harness."""
from __future__ import annotations

import itertools
from typing import Any

_counter = itertools.count(1)


def new_id(prefix: str = "id") -> str:
    """Deterministic-per-process monotonic id (no Math.random needed)."""
    return f"{prefix}_{next(_counter)}"


def est_tokens(text: Any) -> int:
    """~4 chars/token back-of-envelope estimate."""
    return max(1, len(str(text)) // 4)


def preview(x: Any, n: int = 56) -> str:
    s = str(x).replace("\n", " / ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _self_test() -> None:
    a, b = new_id("tool"), new_id("tool")
    assert a != b and a.startswith("tool_")
    assert est_tokens("abcdefgh") == 2 and est_tokens("") == 1
    assert preview("x" * 100).endswith("…") and preview("short") == "short"
    print("[OK] harness.util._self_test passed")


if __name__ == "__main__":
    _self_test()
