"""Capstone - 4-attack x 3-target ASR matrix - TEACHING MOCK.

Shows the canonical "red-team report card" format used in safety papers.
"""
from __future__ import annotations

from typing import Dict, List

from common import HARMFUL_QUERIES, TargetFn, make_safe_target
from jailbench_runner import METHODS, run_jailbench


def make_target_zoo() -> Dict[str, TargetFn]:
    """3 mock targets with varying safety strength."""
    return {
        "strong_safety": make_safe_target("strong", jb_keys=[]),
        "weak_safety": make_safe_target(
            "weak",
            # 3 known toy triggers - multiple methods can find them
            jb_keys=["<toy_trigger>", "<toy_h>", "[TOY_PREFIX]"],
        ),
        "no_safety": make_safe_target(
            "no_safety",
            # any token in the seed template triggers - basically broken
            jb_keys=["BoundaryTester", "seed prompt"],
        ),
    }


def run_matrix(queries: List[str] = None) -> Dict[str, Dict[str, float]]:
    qs = queries or HARMFUL_QUERIES[:3]
    zoo = make_target_zoo()
    out: Dict[str, Dict[str, float]] = {}
    for tname, tfn in zoo.items():
        out[tname] = run_jailbench(tfn, qs)
    return out


def to_md(matrix: Dict[str, Dict[str, float]]) -> str:
    methods = sorted(set().union(*(m.keys() for m in matrix.values())))
    lines = ["# Red-team ASR matrix",
             "Rows = target, Cols = attack method", "",
             "| target \\ method | " + " | ".join(methods) + " |"]
    lines.append("|---|" + "---|" * len(methods))
    for t, m in matrix.items():
        lines.append("| " + t + " | "
                     + " | ".join(f"{m.get(name, 0.0):.0%}" for name in methods)
                     + " |")
    return "\n".join(lines)


def _self_test() -> int:
    mat = run_matrix(HARMFUL_QUERIES[:2])
    assert set(mat.keys()) == {"strong_safety", "weak_safety", "no_safety"}
    # strong_safety: all 0
    assert all(v == 0.0 for v in mat["strong_safety"].values())
    # weak_safety: at least some > 0
    n = sum(1 for v in mat["weak_safety"].values() if v > 0)
    assert n >= 2, mat["weak_safety"]
    # no_safety: even more
    n2 = sum(1 for v in mat["no_safety"].values() if v > 0)
    assert n2 >= 1
    md = to_md(mat)
    assert "ASR" in md and "strong_safety" in md
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"red_team_matrix.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    print(to_md(run_matrix()))
