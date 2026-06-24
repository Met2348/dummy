"""Mock comparison: SGLang vs vLLM on 5 representative scenarios.

No real engine called.  We model only the **prefill-token cost**: how many
prompt tokens each engine must prefill.  RadixAttention's modelled gain is that
a shared prefix is prefilled **once** instead of `fork_k` times when a request
forks (tree-of-thought, parallel sampling, ...).

The displayed `sglang_prefill_gain` is derived **directly** from the two cost
columns (`1 − sglang/vllm`), so it can never contradict them.  Advantages this
cost model does NOT capture (xgrammar jump-forward on structured output, KV
reuse across agent steps, kernel differences) are listed as a separate
qualitative note, not as a fabricated percentage.  A real tok/s benchmark needs
a machine with `sglang` installed (Linux + CUDA).
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
    """vLLM prefill tokens.

    fork_k>1: forks are **independent** generations — each re-prefills the
    shared prefix (no cross-fork sharing) → ``fork_k·(prefix+suffix)``.
    fork_k==1: block-hash prefix caching dedups the identical prefix across the
    n_prompts requests, so the shared part is prefilled once.
    """
    if sc.fork_k > 1:
        return sc.fork_k * (sc.shared_prefix_len + sc.suffix_len)
    return sc.shared_prefix_len + sc.n_prompts * sc.suffix_len


def cost_sglang(sc: Scenario) -> int:
    """SGLang/RadixAttention prefill tokens: shared prefix prefilled **once**
    regardless of fork_k (the radix tree keeps it resident)."""
    if sc.fork_k > 1:
        return sc.shared_prefix_len + sc.fork_k * sc.suffix_len
    return sc.shared_prefix_len + sc.n_prompts * sc.suffix_len


def prefill_gain_pct(sc: Scenario) -> float:
    """SGLang 相对 vLLM 的 prefill-token 节省，**纯由上面两列 cost 推出**。

    只反映 RadixAttention 的共享前缀收益；fork_k==1 且前缀相同时两引擎都能
    缓存共享前缀 → 0%（差异来自下面 other_advantage 列的未建模机制）。
    """
    v, s = cost_vllm(sc), cost_sglang(sc)
    return 1.0 - s / v


def other_advantage(sc: Scenario) -> str:
    """本 prefill-token 模型**未建模**的其它 SGLang 优势（定性说明，非数字）。"""
    if sc.fork_k > 1:
        return "(已由 prefill_gain 量化：共享前缀只 prefill 一次)"
    if "json" in sc.name:
        return "xgrammar + jump-forward 跳过确定性 token 的解码（未建模）"
    if "react" in sc.name:
        return "多步 agent 跨 step 复用 KV + 约束解码（未建模）"
    if "long_prompt" in sc.name:
        return "超长前缀两引擎都缓存，差异主要在内核实现（未建模）"
    return "前缀相同，两引擎前缀缓存基本打平"


def run_all() -> List[Dict]:
    out = []
    for sc in SCENARIOS:
        v = cost_vllm(sc)
        s = cost_sglang(sc)
        out.append({
            "scenario": sc.name,
            "vllm_prefill_tokens": v,
            "sglang_prefill_tokens": s,
            "sglang_prefill_gain": f"{prefill_gain_pct(sc) * 100:+.1f}%",
            "other_advantage_not_modeled": other_advantage(sc),
        })
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
