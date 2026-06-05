"""Multi-agent cost analyzer."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CostReport:
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    n_llm_calls: int = 0
    by_agent: dict[str, dict] = field(default_factory=dict)

    def add_call(self, agent: str, tin: int, tout: int) -> None:
        self.total_tokens_in += tin
        self.total_tokens_out += tout
        self.n_llm_calls += 1
        ag = self.by_agent.setdefault(agent, {"tin": 0, "tout": 0, "n": 0})
        ag["tin"] += tin
        ag["tout"] += tout
        ag["n"] += 1

    def usd(self, in_price: float = 0.003, out_price: float = 0.015) -> float:
        return round(
            self.total_tokens_in / 1000 * in_price
            + self.total_tokens_out / 1000 * out_price,
            6,
        )


def simulate_single_agent(n_tokens: int = 500) -> CostReport:
    r = CostReport()
    r.add_call("single", tin=n_tokens, tout=n_tokens // 4)
    return r


def simulate_multi_agent(n_agents: int = 3, n_rounds: int = 3,
                         base_tokens: int = 500) -> CostReport:
    r = CostReport()
    for round_idx in range(n_rounds):
        ctx_growth = 1 + 0.5 * round_idx
        for ag_idx in range(n_agents):
            tin = int(base_tokens * ctx_growth * (1 + 0.3 * ag_idx))
            tout = int(base_tokens / 4 * ctx_growth)
            r.add_call(f"agent_{ag_idx}", tin=tin, tout=tout)
    return r


def compare_single_vs_multi() -> dict:
    single = simulate_single_agent()
    multi = simulate_multi_agent()
    ratio_tokens = (multi.total_tokens_in + multi.total_tokens_out) / max(
        1, single.total_tokens_in + single.total_tokens_out
    )
    ratio_cost = multi.usd() / max(1e-9, single.usd())
    return {
        "single": single,
        "multi": multi,
        "ratio_tokens": round(ratio_tokens, 1),
        "ratio_cost": round(ratio_cost, 1),
    }


def _self_test() -> None:
    single = simulate_single_agent()
    assert single.n_llm_calls == 1
    assert single.usd() > 0

    multi = simulate_multi_agent(n_agents=3, n_rounds=3)
    assert multi.n_llm_calls == 9
    assert len(multi.by_agent) == 3

    cmp = compare_single_vs_multi()
    assert cmp["ratio_tokens"] > 10, cmp["ratio_tokens"]
    assert cmp["ratio_cost"] > 10, cmp["ratio_cost"]
    print(f"[OK] cost_analyzer._self_test passed (multi/single token ratio={cmp['ratio_tokens']}x)")


if __name__ == "__main__":
    _self_test()
