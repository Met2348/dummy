"""Classic Speculative Decoding (Leviathan 2023 / Chen 2023).

Correctness theorem: the marginal distribution of accepted tokens equals
the target distribution p, irrespective of how poor the draft q is.  We
exercise this in tests via large-sample empirical comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import math
import random

from common import softmax, sample_from, SpecMetrics


def rejection_sample(p: List[float], q: List[float], drafted: int, rng: random.Random) -> int:
    """Single-token accept/reject for a drafted token x = `drafted`.

    Returns the resampled token id (possibly = drafted if accepted).
    """
    r = rng.random()
    ratio = p[drafted] / max(q[drafted], 1e-12)
    if r < ratio:
        return drafted
    # rejected: sample from norm(max(0, p-q))
    residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    s = sum(residual)
    if s <= 0:
        return drafted   # fallback (numerically rare)
    residual = [r / s for r in residual]
    return sample_from(residual, rng)


def speculative_decode_step(
    p_target: List[List[float]],     # k+1 target probs at positions 0..k
    q_draft: List[List[float]],      # k draft probs at positions 0..k-1
    drafted_ids: List[int],
    rng: random.Random,
) -> List[int]:
    """One speculative iteration: returns accepted token ids (length 1..k+1)."""
    accepted: List[int] = []
    for i, draft in enumerate(drafted_ids):
        # rejection sample at position i
        r = rng.random()
        ratio = p_target[i][draft] / max(q_draft[i][draft], 1e-12)
        if r < ratio:
            accepted.append(draft)
        else:
            # reject: sample residual at position i and STOP further accept
            residual = [max(0.0, pj - qj) for pj, qj in zip(p_target[i], q_draft[i])]
            s = sum(residual)
            if s <= 0:
                accepted.append(draft)
            else:
                accepted.append(sample_from([r / s for r in residual], rng))
            return accepted
    # All drafted tokens were accepted, so draw one bonus token from p_target[k].
    accepted.append(sample_from(p_target[len(drafted_ids)], rng))
    return accepted


def run_classic_spec(
    draft_fn: Callable[[List[int]], List[float]],
    target_fn: Callable[[List[int]], List[float]],
    prompt: List[int],
    n_tokens: int,
    k: int,
    seed: int = 0,
) -> tuple[List[int], SpecMetrics]:
    rng = random.Random(seed)
    out = list(prompt)
    m = SpecMetrics()
    while len(out) - len(prompt) < n_tokens:
        # 1. draft k tokens
        drafted_ids: List[int] = []
        q_per_step: List[List[float]] = []
        local = list(out)
        for _ in range(k):
            q = draft_fn(local)
            tid = sample_from(q, rng)
            drafted_ids.append(tid)
            q_per_step.append(q)
            local.append(tid)

        # 2. target verify positions 0..k (k+1 distributions)
        p_per_step: List[List[float]] = []
        local2 = list(out)
        for i in range(k + 1):
            p = target_fn(local2)
            p_per_step.append(p)
            if i < k:
                local2.append(drafted_ids[i])

        # 3. accept loop
        accepted = speculative_decode_step(p_per_step, q_per_step, drafted_ids, rng)
        out.extend(accepted)
        m.n_iters += 1
        m.n_drafted += k
        m.n_accepted += len(accepted) - 1   # subtract bonus
        m.n_tokens_out += len(accepted)
    return out[len(prompt) : len(prompt) + n_tokens], m
