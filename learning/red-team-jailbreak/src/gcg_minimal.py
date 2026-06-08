"""GCG (Zou 2023) - gradient-based suffix search - TEACHING MOCK.

Real GCG optimizes an adversarial suffix token-by-token using gradients.
This file keeps only the search-loop shape and uses a harmless toy target.

This MOCK simulates the search loop without real gradients. The mock
target has a small set of "vulnerable" suffix tokens; the optimizer
tries random combinations and converges to a suffix that flips refusal.

NOTE: This is for teaching the algorithm structure only.
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Tuple

from common import (
    AttackResult,
    HARMFUL_QUERIES,
    TargetFn,
    attack_success_rate,
    classify_response,
    make_safe_target,
)


# Mock "candidate tokens" for adversarial suffix search.
# In real GCG these are token IDs from vocab.
CANDIDATE_TOKENS = [
    "<toy_a>", "<toy_b>", "<toy_c>", "<toy_d>",
    "<toy_trigger>", "<toy_e>", "<toy_f>", "<toy_g>",
    "<toy_h>", "<toy_i>", "<toy_j>", "<toy_k>",
    "<toy_l>", "<toy_m>", "<toy_n>", "<toy_o>",
]


def candidate_suffix(seed: int, length: int = 3) -> str:
    """Construct a suffix by seeded sampling from candidate tokens."""
    rng = random.Random(seed)
    return " " + " ".join(rng.choices(CANDIDATE_TOKENS, k=length))


def gcg_attack(target: TargetFn, query: str,
                max_iters: int = 32, suffix_len: int = 3) -> AttackResult:
    """Search a suffix that triggers compliance.

    Returns the best-found AttackResult. If no suffix works, returns
    the last refusal.
    """
    best: AttackResult = AttackResult(
        query=query, attack_prompt=query, response="", label="refused", method="gcg",
    )
    for i in range(max_iters):
        suffix = candidate_suffix(seed=i, length=suffix_len)
        full = query + suffix
        resp = target(full)
        label = classify_response(resp)
        result = AttackResult(
            query=query, attack_prompt=full, response=resp,
            label=label, method="gcg",
        )
        if label == "complied":
            return result
        best = result
    return best


def run_gcg_bench(target: TargetFn, queries: List[str],
                   max_iters: int = 32) -> List[AttackResult]:
    return [gcg_attack(target, q, max_iters=max_iters) for q in queries]


def _self_test() -> int:
    # Safe target with no vulnerability -> ASR = 0
    safe = make_safe_target("safe", jb_keys=[])
    rs = run_gcg_bench(safe, HARMFUL_QUERIES[:3], max_iters=16)
    assert attack_success_rate(rs) == 0.0
    # Vulnerable target -> GCG should find the trigger
    # Pick a token that IS in CANDIDATE_TOKENS so suffix search can find it.
    vuln = make_safe_target("vuln", jb_keys=["<toy_trigger>"])
    rs2 = run_gcg_bench(vuln, HARMFUL_QUERIES[:3], max_iters=64)
    # Should succeed on at least 1
    assert attack_success_rate(rs2) > 0.0, rs2
    # candidate_suffix deterministic
    assert candidate_suffix(0) == candidate_suffix(0)
    assert candidate_suffix(0) != candidate_suffix(1)
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"gcg_minimal.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
