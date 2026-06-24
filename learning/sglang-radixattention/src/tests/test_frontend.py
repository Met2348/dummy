"""Tests for frontend DSL + agent patterns + scenario cost model."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frontend_lang import Stream, Gen, Select, function
from agent_patterns import react_loop, tree_of_thought, self_consistency
from sglang_compare import run_all, SCENARIOS, cost_vllm, cost_sglang


def test_stream_append_string():
    s = Stream()
    s += "hello "
    s += "world"
    assert s.prompt == "hello world"


def test_stream_gen_records_var():
    s = Stream()
    s += "prefix"
    s += Gen("answer", max_tokens=5)
    assert "answer" in s.vars
    assert s.n_forwards == 1


def test_stream_fork_independent_branches():
    s = Stream()
    s += "shared"
    forks = s.fork(3)
    forks[0] += " A"
    forks[1] += " B"
    forks[2] += " C"
    assert forks[0].prompt == "shared A"
    assert forks[1].prompt == "shared B"
    assert s.prompt == "shared"   # parent unchanged


def test_function_decorator_returns_stream():
    @function
    def f(s, x):
        s += f"input={x}"
        s += Gen("o")
    out = f("hi")
    assert isinstance(out, Stream)
    assert "input=hi" in out.prompt


def test_react_loop_produces_thoughts_and_actions():
    out = react_loop("hi", max_steps=2)
    # at least t0, a0, t1, a1
    assert "t0" in out.vars and "a0" in out.vars
    assert "Observation:" in out.prompt


def test_tot_consults_all_branches():
    text = tree_of_thought("Q?", k=4)
    assert "Q?" in text
    assert "Answer:" in text


def test_self_consistency_returns_string():
    res = self_consistency("Q?", k=3)
    assert isinstance(res, str)
    assert len(res) > 0


def test_compare_all_scenarios_runs():
    rows = run_all()
    assert len(rows) == len(SCENARIOS)
    names = {r["scenario"] for r in rows}
    assert "tot_8way" in names


def test_compare_costs_nonzero():
    for sc in SCENARIOS:
        assert cost_vllm(sc) > 0
        assert cost_sglang(sc) > 0


def test_fork_scenario_radix_strictly_cheaper():
    # fork_k>1 (tot_8way): RadixAttention prefills the shared prefix once while
    # vLLM forks re-prefill it each time -> SGLang must be strictly cheaper, and
    # the displayed gain (derived from the columns) must be positive & consistent.
    from sglang_compare import prefill_gain_pct
    tot = next(sc for sc in SCENARIOS if sc.fork_k > 1)
    v, s = cost_vllm(tot), cost_sglang(tot)
    assert v > s, "fork case must show vLLM > SGLang (else the demo is a no-op)"
    assert abs(prefill_gain_pct(tot) - (1.0 - s / v)) < 1e-9
