"""Capstone — compare classic / medusa / eagle1 / eagle2 across 5 tasks.

Uses synthetic target distributions parametrised by 'task difficulty':
- code/math/json => low-entropy target => high accept rate
- story/qa => high-entropy => lower accept rate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List
import json
import math
import random

from common import softmax, sample_from, SpecMetrics
from classic_spec_decode import run_classic_spec
from medusa_heads import MedusaHeads, medusa_step
from eagle_minimal import EagleDraft, eagle_step
from eagle2 import build_dynamic_tree
from spec_eval import sim_speedup


VOCAB = 64


def make_target(entropy: float, seed: int = 0) -> Callable[[List[int]], List[float]]:
    """Higher entropy means flatter (harder) distribution."""
    base = [random.Random(seed).gauss(0, 1) for _ in range(VOCAB)]
    def fn(prefix: List[int]) -> List[float]:
        shift = sum(prefix) % VOCAB if prefix else 0
        logits = [(b * (1.0 - entropy) + 0.5 * math.sin(i + shift) * entropy) for i, b in enumerate(base)]
        return softmax(logits)
    return fn


TASKS = {
    "code":  dict(entropy=0.30),
    "math":  dict(entropy=0.20),
    "json":  dict(entropy=0.20),
    "qa":    dict(entropy=0.55),
    "story": dict(entropy=0.80),
}


def eval_method(method: str, task: str, n_tokens: int = 50, seed: int = 0) -> Dict:
    target_fn = make_target(TASKS[task]["entropy"], seed=seed)
    rng = random.Random(seed)
    m = SpecMetrics()
    prompt = [0, 1, 2]
    out: List[int] = []

    if method == "classic":
        draft_fn = make_target(TASKS[task]["entropy"] + 0.25, seed=seed + 7)
        out, m = run_classic_spec(draft_fn, target_fn, prompt, n_tokens=n_tokens, k=4, seed=seed)
    elif method == "medusa":
        heads = MedusaHeads(n_heads=4, noise=0.45)
        local = list(prompt)
        while len(out) < n_tokens:
            accepted, _ = medusa_step(target_fn, heads, local, rng)
            out.extend(accepted)
            local.extend(accepted)
            m.n_iters += 1
            m.n_drafted += heads.n_heads
            m.n_accepted += len(accepted) - 1
            m.n_tokens_out += len(accepted)
    elif method == "eagle1":
        draft = EagleDraft(k=4, noise=0.30)
        local = list(prompt)
        while len(out) < n_tokens:
            accepted, _ = eagle_step(target_fn, draft, local, rng)
            out.extend(accepted)
            local.extend(accepted)
            m.n_iters += 1
            m.n_drafted += draft.k
            m.n_accepted += len(accepted) - 1
            m.n_tokens_out += len(accepted)
    elif method == "eagle2":
        # tree of K=8 leaves: take longest accepted path
        draft = EagleDraft(k=4, noise=0.22)
        local = list(prompt)
        while len(out) < n_tokens:
            tree = build_dynamic_tree(target_fn, local, K=4, max_depth=5, branch=2, noise=0.20, rng=rng)
            # naive verify: pick longest tree path with all-accept ratio
            best = max(tree, key=lambda t: len(t.tokens))
            # verify each token
            accepted: List[int] = []
            tmp = list(local)
            for tid in best.tokens:
                p = target_fn(tmp)
                q_lp = best.logprob / max(len(best.tokens), 1)    # avg lp
                q = math.exp(q_lp)
                if rng.random() < p[tid] / max(q, 1e-12):
                    accepted.append(tid)
                    tmp.append(tid)
                else:
                    residual = [max(0.0, pj - q) for pj in p]
                    s = sum(residual)
                    tok = sample_from([r / s for r in residual] if s > 0 else p, rng)
                    accepted.append(tok)
                    break
            # bonus
            bonus = sample_from(target_fn(tmp), rng)
            accepted.append(bonus)
            out.extend(accepted)
            local.extend(accepted)
            m.n_iters += 1
            m.n_drafted += len(best.tokens)
            m.n_accepted += len(accepted) - 1
            m.n_tokens_out += len(accepted)
    else:
        raise ValueError(method)

    return {
        "method": method,
        "task": task,
        "n_iters": m.n_iters,
        "n_drafted": m.n_drafted,
        "n_accepted": m.n_accepted,
        "n_tokens_out": m.n_tokens_out,
        "accept_rate": round(m.accept_rate, 3),
        "MAU": round(m.mau, 2),
        "sim_speedup": round(sim_speedup(m), 2),
    }


def run_all(n_tokens: int = 50, seed: int = 0) -> List[Dict]:
    out = []
    for method in ("classic", "medusa", "eagle1", "eagle2"):
        for task in TASKS.keys():
            out.append(eval_method(method, task, n_tokens=n_tokens, seed=seed))
    return out


def to_md(rows: List[Dict]) -> str:
    head = "| method | task | accept | MAU | sim_speedup |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['method']} | {r['task']} | {r['accept_rate']} | {r['MAU']} | {r['sim_speedup']}x |"
        for r in rows
    )
    return head + "\n" + body


if __name__ == "__main__":
    rows = run_all(n_tokens=40)
    print(to_md(rows))
    print("\nJSON:")
    print(json.dumps(rows, indent=2))
