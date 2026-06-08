"""Paper-shaped examples for SGLang and RadixAttention.

This file does not implement a real LLM runtime. It connects the paper's
systems ideas to small measurable quantities:

- naive prefill recomputes every prompt token;
- RadixAttention stores prompt prefixes in a radix tree and reuses matched KV;
- compressed FSM / jump-forward decoding can skip forced literal spans.

Run:
    .\\.venv\\Scripts\\python.exe learning\\sglang-radixattention\\src\\sglang_original_minimal.py
"""
from __future__ import annotations

from dataclasses import dataclass

from grammar_fsm import compile_literal
from jump_forward import jump_forward
from radix_tree import RadixTree


SYSTEM = list(range(100, 124))
FEW_SHOT = list(range(200, 220))
TOOL_TEMPLATE = list(range(300, 310))


@dataclass(frozen=True)
class RadixReuseReport:
    naive_prefill_tokens: int
    radix_prefill_tokens: int
    cached_prompt_tokens: int
    hit_rate: float
    total_nodes: int

    @property
    def saved_prefill_tokens(self) -> int:
        return self.naive_prefill_tokens - self.radix_prefill_tokens


def lm_program_prompts() -> list[list[int]]:
    """Return prompts with multi-level sharing similar to LM programs."""
    prompts: list[list[int]] = []

    # Agent calls: shared system prompt and tool template, different questions.
    for q in range(8):
        prompts.append(SYSTEM + TOOL_TEMPLATE + [400 + q, 401 + q])

    # Few-shot benchmark calls: shared examples, different final questions.
    for q in range(6):
        prompts.append(SYSTEM + FEW_SHOT + [500 + q])

    # Forked branch-solve-merge style calls: same prefix, branch dimension differs.
    branch_prefix = SYSTEM + [600, 601, 602]
    for dim in range(4):
        prompts.append(branch_prefix + [700 + dim])

    return prompts


def measure_radix_reuse(prompts: list[list[int]]) -> RadixReuseReport:
    """Compare naive prefill cost with radix-tree prefix reuse."""
    tree = RadixTree(cap=1_000_000)
    naive = sum(len(p) for p in prompts)
    cached = 0

    for prompt in prompts:
        _leaf, matched = tree.insert(prompt)
        cached += matched

    radix_prefill = naive - cached
    return RadixReuseReport(
        naive_prefill_tokens=naive,
        radix_prefill_tokens=radix_prefill,
        cached_prompt_tokens=cached,
        hit_rate=cached / max(naive, 1),
        total_nodes=tree.total_nodes(),
    )


def forced_literal_jump() -> dict[str, object]:
    """Show how a compressed FSM can skip a deterministic literal span."""
    literal = '{"summary":"'
    fsm = compile_literal(literal)
    forced, new_state = jump_forward(fsm, state=0)
    return {
        "literal": literal,
        "normal_decode_steps": len(literal),
        "jump_forward_steps": 1 if forced else 0,
        "forced": forced,
        "new_state": new_state,
    }


def summarize() -> dict[str, object]:
    report = measure_radix_reuse(lm_program_prompts())
    jump = forced_literal_jump()
    return {
        "naive_prefill_tokens": report.naive_prefill_tokens,
        "radix_prefill_tokens": report.radix_prefill_tokens,
        "saved_prefill_tokens": report.saved_prefill_tokens,
        "hit_rate": round(report.hit_rate, 3),
        "total_radix_nodes": report.total_nodes,
        "jump_forward": jump,
    }


def _self_test() -> None:
    result = summarize()
    assert result["saved_prefill_tokens"] > 0
    assert result["hit_rate"] > 0.50
    jump = result["jump_forward"]
    assert jump["forced"] == jump["literal"]
    assert jump["normal_decode_steps"] > jump["jump_forward_steps"]

    for key, value in result.items():
        print(f"{key}: {value}")
    print("sglang_original_minimal self-test passed")


if __name__ == "__main__":
    _self_test()
