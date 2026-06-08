"""Cost-controlled agent evaluation helpers inspired by AI Agents That Matter."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRun:
    """One evaluated agent configuration.

    accuracy is a task success rate in [0, 1].
    variable_cost_usd is paid each time the agent runs on one task.
    fixed_cost_usd is one-time search or optimization cost.
    """

    name: str
    accuracy: float
    variable_cost_usd: float
    fixed_cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def amortized_cost(self, n_runs: int = 1) -> float:
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")
        return self.variable_cost_usd + self.fixed_cost_usd / n_runs


def token_cost_usd(
    tokens_in: int,
    tokens_out: int,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    """Compute dollar cost from token counts and per-million-token prices."""
    return round(
        tokens_in / 1_000_000 * price_in_per_mtok
        + tokens_out / 1_000_000 * price_out_per_mtok,
        6,
    )


def dominates(a: AgentRun, b: AgentRun, n_runs: int = 1) -> bool:
    """Return true if a is at least as accurate and cheaper than b."""
    a_cost = a.amortized_cost(n_runs)
    b_cost = b.amortized_cost(n_runs)
    no_worse = a.accuracy >= b.accuracy and a_cost <= b_cost
    strictly_better = a.accuracy > b.accuracy or a_cost < b_cost
    return no_worse and strictly_better


def pareto_frontier(runs: list[AgentRun], n_runs: int = 1) -> list[AgentRun]:
    """Keep configurations not dominated on accuracy and cost."""
    frontier = [
        candidate
        for candidate in runs
        if not any(dominates(other, candidate, n_runs) for other in runs)
    ]
    return sorted(frontier, key=lambda r: (r.amortized_cost(n_runs), -r.accuracy, r.name))


def cost_controlled_score(
    run: AgentRun,
    n_runs: int = 1,
    cost_weight: float = 1.0,
    latency_weight: float = 0.0,
) -> float:
    """A tiny utility function for downstream agent selection."""
    return round(
        run.accuracy
        - cost_weight * run.amortized_cost(n_runs)
        - latency_weight * run.latency_s,
        6,
    )


def best_by_utility(
    runs: list[AgentRun],
    n_runs: int = 1,
    cost_weight: float = 1.0,
    latency_weight: float = 0.0,
) -> AgentRun:
    if not runs:
        raise ValueError("runs must not be empty")
    return max(
        runs,
        key=lambda r: cost_controlled_score(
            r,
            n_runs=n_runs,
            cost_weight=cost_weight,
            latency_weight=latency_weight,
        ),
    )


HOLDOUT_BY_GENERALITY = {
    "distribution-specific": "hold out in-distribution samples",
    "task-specific": "hold out out-of-distribution samples",
    "domain-general": "hold out tasks",
    "fully-general": "hold out domains",
}


def required_holdout(generality: str) -> str:
    try:
        return HOLDOUT_BY_GENERALITY[generality]
    except KeyError as exc:
        allowed = ", ".join(sorted(HOLDOUT_BY_GENERALITY))
        raise ValueError(f"unknown generality; expected one of {allowed}") from exc


def benchmark_shortcut_risk(
    generality: str,
    actual_holdout: str,
    cost_reported: bool,
    code_and_data_released: bool,
    standardized_eval: bool,
) -> int:
    """Return a simple 0-5 risk score for benchmark overinterpretation."""
    risk = 0
    expected = required_holdout(generality)
    if actual_holdout != expected:
        risk += 2
    if not cost_reported:
        risk += 1
    if not code_and_data_released:
        risk += 1
    if not standardized_eval:
        risk += 1
    return risk


def novelqa_cost_scenarios(
    novel_tokens: int,
    question_tokens: int,
    answer_tokens: int,
    n_questions: int,
    retrieved_tokens_per_question: int,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> dict[str, float]:
    """Compare batched benchmark cost with downstream sequential QA cost."""
    benchmark_batched = token_cost_usd(
        tokens_in=novel_tokens + question_tokens * n_questions,
        tokens_out=answer_tokens * n_questions,
        price_in_per_mtok=price_in_per_mtok,
        price_out_per_mtok=price_out_per_mtok,
    )
    long_context_sequential = token_cost_usd(
        tokens_in=(novel_tokens + question_tokens) * n_questions,
        tokens_out=answer_tokens * n_questions,
        price_in_per_mtok=price_in_per_mtok,
        price_out_per_mtok=price_out_per_mtok,
    )
    rag_sequential = token_cost_usd(
        tokens_in=(retrieved_tokens_per_question + question_tokens) * n_questions,
        tokens_out=answer_tokens * n_questions,
        price_in_per_mtok=price_in_per_mtok,
        price_out_per_mtok=price_out_per_mtok,
    )
    return {
        "benchmark_batched": benchmark_batched,
        "long_context_sequential": long_context_sequential,
        "rag_sequential": rag_sequential,
    }


def humaneval_style_baselines() -> list[str]:
    return [
        "zero-shot model",
        "retry same model after visible test failure",
        "warming with increasing temperature",
        "escalation from cheap model to stronger model",
    ]


def _self_test() -> None:
    runs = [
        AgentRun("complex-agent", accuracy=0.91, variable_cost_usd=0.90),
        AgentRun("warming", accuracy=0.91, variable_cost_usd=0.30),
        AgentRun("escalation", accuracy=0.93, variable_cost_usd=0.25),
        AgentRun("cheap-baseline", accuracy=0.82, variable_cost_usd=0.03),
        AgentRun("optimized-agent", accuracy=0.92, variable_cost_usd=0.18, fixed_cost_usd=50.0),
    ]

    short_horizon = {r.name for r in pareto_frontier(runs, n_runs=10)}
    long_horizon = {r.name for r in pareto_frontier(runs, n_runs=10_000)}
    assert "complex-agent" not in short_horizon
    assert "escalation" in short_horizon
    assert "optimized-agent" in long_horizon

    winner = best_by_utility(runs, n_runs=10_000, cost_weight=0.5)
    assert winner.name in {"escalation", "optimized-agent"}

    assert required_holdout("domain-general") == "hold out tasks"
    risk = benchmark_shortcut_risk(
        "domain-general",
        actual_holdout="hold out in-distribution samples",
        cost_reported=False,
        code_and_data_released=True,
        standardized_eval=False,
    )
    assert risk == 4

    costs = novelqa_cost_scenarios(
        novel_tokens=200_000,
        question_tokens=80,
        answer_tokens=120,
        n_questions=20,
        retrieved_tokens_per_question=2_000,
        price_in_per_mtok=1.0,
        price_out_per_mtok=3.0,
    )
    assert costs["long_context_sequential"] > costs["benchmark_batched"] * 10
    assert costs["long_context_sequential"] > costs["rag_sequential"] * 10

    assert len(humaneval_style_baselines()) == 4
    print("[OK] eval.agent_eval_matter._self_test passed")


if __name__ == "__main__":
    _self_test()
