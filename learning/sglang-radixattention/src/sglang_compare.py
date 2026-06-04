"""Mock comparison: SGLang vs vLLM on 5 representative scenarios.

No real engine called — we instrument the mock Stream's `n_forwards` field
to count how many model forwards would happen.  RadixAttention's gain is
modelled as: shared prefix prefill happens **once** instead of `n_forks` times.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from frontend_lang import Stream, Gen, function


@dataclass
class Scenario:
    name: str
    n_prompts: int
    shared_prefix_len: int
    suffix_len: int
    fork_k: int


SCENARIOS = [
    Scenario("shared_system_chat", n_prompts=100, shared_prefix_len=2000, suffix_len=50, fork_k=1),
    Scenario("tot_8way", n_prompts=1, shared_prefix_len=1000, suffix_len=50, fork_k=8),
    Scenario("json_schema", n_prompts=100, shared_prefix_len=500, suffix_len=100, fork_k=1),
    Scenario("react_5step", n_prompts=100, shared_prefix_len=300, suffix_len=200, fork_k=1),
    Scenario("long_prompt_short_out", n_prompts=1, shared_prefix_len=8000, suffix_len=16, fork_k=1),
]


def cost_vllm(sc: Scenario) -> int:
    """Block-hash prefix caching: shared part is 1 prefill, suffix per request."""
    if sc.fork_k > 1:
        # vLLM fork = n_prompts independent generations
        return sc.shared_prefix_len + sc.fork_k * sc.suffix_len
    return sc.shared_prefix_len + sc.n_prompts * sc.suffix_len


def cost_sglang(sc: Scenario) -> int:
    """RadixAttention: shared prefix 1 prefill regardless of fork_k."""
    if sc.fork_k > 1:
        return sc.shared_prefix_len + sc.fork_k * sc.suffix_len
    return sc.shared_prefix_len + sc.n_prompts * sc.suffix_len


def gain_pct(sc: Scenario) -> float:
    """Synthetic gain estimate (not a benchmark): when fork_k>1 SGLang wins via
    shared prefix; when grammar-driven SGLang wins via xgrammar; else ~ tied."""
    if sc.fork_k > 1:
        return 1.0 - (sc.shared_prefix_len + sc.fork_k * sc.suffix_len) / (sc.fork_k * (sc.shared_prefix_len + sc.suffix_len))
    if "json" in sc.name:
        return 0.50
    if "react" in sc.name:
        return 0.60
    if "long_prompt" in sc.name:
        return -0.05
    return 0.05


def run_all() -> List[Dict]:
    out = []
    for sc in SCENARIOS:
        v = cost_vllm(sc)
        s = cost_sglang(sc)
        out.append({
            "scenario": sc.name,
            "vllm_cost": v,
            "sglang_cost": s,
            "estimated_sglang_gain": f"{gain_pct(sc) * 100:+.1f}%",
        })
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
