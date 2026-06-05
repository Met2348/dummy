"""JailbreakBench / HarmBench standardized eval — TEACHING MOCK.

Real bench: 100-500 standardized harmful queries × N attack methods,
report ASR (Attack Success Rate) per method × target.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from common import (
    AttackResult,
    HARMFUL_QUERIES,
    TargetFn,
    attack_success_rate,
    make_safe_target,
)
import gcg_minimal
import pair_minimal
import autodan_minimal
import crescendo_demo


METHODS: Dict[str, Callable[[TargetFn, List[str]], List[AttackResult]]] = {
    "gcg": lambda t, qs: gcg_minimal.run_gcg_bench(t, qs, max_iters=16),
    "pair": lambda t, qs: pair_minimal.run_pair_bench(t, qs, max_attempts=8),
    "autodan": lambda t, qs: autodan_minimal.run_autodan_bench(t, qs, n_gen=4),
    "crescendo": lambda t, qs: crescendo_demo.run_crescendo_bench(t, qs),
}


def run_jailbench(target: TargetFn, queries: List[str] = None) -> Dict[str, float]:
    qs = queries or HARMFUL_QUERIES[:3]
    out: Dict[str, float] = {}
    for name, fn in METHODS.items():
        out[name] = attack_success_rate(fn(target, qs))
    return out


def to_md(target_name: str, asr: Dict[str, float]) -> str:
    lines = [f"# JailbreakBench results — `{target_name}`",
             "| method | ASR |", "|---|---:|"]
    for k, v in asr.items():
        lines.append(f"| {k} | {v:.0%} |")
    return "\n".join(lines)


def _self_test() -> int:
    # Safe target — all methods ~0% ASR
    safe = make_safe_target("safe", jb_keys=[])
    asr_safe = run_jailbench(safe, HARMFUL_QUERIES[:2])
    assert all(v == 0.0 for v in asr_safe.values()), asr_safe
    # Vulnerable target — at least some methods succeed
    vuln = make_safe_target("vuln", jb_keys=["{!}"])
    asr_vuln = run_jailbench(vuln, HARMFUL_QUERIES[:2])
    n_succeed = sum(1 for v in asr_vuln.values() if v > 0)
    assert n_succeed >= 2, asr_vuln
    md = to_md("vuln", asr_vuln)
    assert "ASR" in md
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"jailbench_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
