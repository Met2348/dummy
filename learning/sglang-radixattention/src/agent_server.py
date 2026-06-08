"""Capstone: 32-concurrent ReAct agent server with radix-cache stats.

Pieces composed:
- RadixTree                (radix_tree.py)
- frontend_lang.Stream     (frontend_lang.py)
- React loop               (agent_patterns.py)
- ConstrainedSampler stub  (constrained_sampler.py)

We don't drive a real LLM; instead a mock generator yields short canned tokens
so the workshop runs offline. Production replacement: swap the generator with
your sglang.RuntimeEndpoint client.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List
import json
import time

from radix_tree import RadixTree
from frontend_lang import Stream, Gen, function


SYSTEM_PROMPT = (
    "You are a helpful agent. Use tools when needed.\n"
    "Available tools: search, calc, weather, python.\n"
    "Always end with 'Final Answer: <answer>'.\n"
) * 25     # blow up to ~2000 token equivalent


def tokenize(text: str) -> List[int]:
    """Hash chars in 4-char groups as fake tokens (avoids real tokenizer dep)."""
    out: List[int] = []
    for i in range(0, len(text), 4):
        chunk = text[i : i + 4]
        out.append(hash(chunk) & 0xFFFFF)
    return out


def mock_tool(action: str) -> str:
    a = action.lower()
    if "search" in a:
        return "found 3 results."
    if "calc" in a:
        return "= 42"
    if "weather" in a:
        return "rainy, 18C"
    if "python" in a:
        return "stdout: ok"
    return "no-op"


def mock_generator(prompt: str, name: str, max_tokens: int = 16) -> str:
    """Cycle through canned thoughts/actions so the agent terminates."""
    n_thought = sum(1 for k in [] if False)  # placeholder
    if name.startswith("t"):
        idx = int(name[1:])
        if idx >= 2:
            return "Final Answer: 42"
        return "I should look this up."
    if name.startswith("a"):
        idx = int(name[1:])
        return ["search query", "calc 6*7"][idx % 2]
    return "<no-gen>"


@dataclass
class AgentMetrics:
    n_agents: int = 0
    n_forwards: int = 0
    radix_hits: int = 0
    radix_misses: int = 0
    elapsed: float = 0.0
    tool_calls: Counter = field(default_factory=Counter)


@function
def react_agent(s: Stream, query: str, max_steps: int = 3) -> None:
    s += SYSTEM_PROMPT + f"\nQ: {query}\n"
    for step in range(max_steps):
        s += f"Thought {step}: "
        s += Gen(f"t{step}", max_tokens=10)
        if "Final Answer" in s.vars[f"t{step}"]:
            break
        s += f"\nAction {step}: "
        s += Gen(f"a{step}", max_tokens=8)
        obs = mock_tool(s.vars[f"a{step}"])
        s += f"\nObservation: {obs}\n"


def run_server(n_agents: int = 32, max_steps: int = 3) -> AgentMetrics:
    tree = RadixTree(cap=1_000_000)
    metrics = AgentMetrics(n_agents=n_agents)
    t0 = time.perf_counter()
    queries = [f"What is {i} * 7?" for i in range(n_agents)]
    for q in queries:
        s = react_agent(q, max_steps, generator=mock_generator)
        # post-hoc: insert final prompt into radix to measure share
        tokens = tokenize(s.prompt)
        prev_hits = tree.hits
        leaf, matched = tree.insert(tokens)
        # tool calls counted from the recorded vars
        for k, v in s.vars.items():
            if k.startswith("a"):
                if "search" in v: metrics.tool_calls["search"] += 1
                elif "calc" in v: metrics.tool_calls["calc"] += 1
                elif "weather" in v: metrics.tool_calls["weather"] += 1
                elif "python" in v: metrics.tool_calls["python"] += 1
        metrics.n_forwards += s.n_forwards
        metrics.radix_hits += matched
        metrics.radix_misses += len(tokens) - matched
    metrics.elapsed = time.perf_counter() - t0
    return metrics


def metrics_json(m: AgentMetrics) -> Dict:
    total = m.radix_hits + m.radix_misses
    hit_rate = m.radix_hits / max(total, 1)
    return {
        "n_agents": m.n_agents,
        "n_forwards": m.n_forwards,
        "radix_hit_rate": round(hit_rate, 3),
        "radix_hits": m.radix_hits,
        "radix_total": total,
        "elapsed_s": round(m.elapsed, 3),
        "tool_calls": dict(m.tool_calls),
        "forwards_per_agent": round(m.n_forwards / m.n_agents, 2),
    }


if __name__ == "__main__":
    m = run_server(n_agents=32, max_steps=3)
    print(json.dumps(metrics_json(m), indent=2))
