"""Capstone smoke + radix hit-rate target."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent_server import run_server, metrics_json


def test_agent_server_completes():
    m = run_server(n_agents=8, max_steps=3)
    assert m.n_agents == 8
    assert m.n_forwards > 0


def test_radix_hit_rate_high_with_shared_system_prompt():
    """32 agents sharing the same long SYSTEM_PROMPT → hit_rate ≥ 0.70."""
    m = run_server(n_agents=32, max_steps=3)
    j = metrics_json(m)
    assert j["radix_hit_rate"] >= 0.70, f"hit_rate {j['radix_hit_rate']} too low"


def test_forwards_per_agent_modest():
    m = run_server(n_agents=8, max_steps=3)
    j = metrics_json(m)
    # Each agent does 1 thought + maybe 1 action per step; cap loosely
    assert j["forwards_per_agent"] <= 10, f"forwards/agent={j['forwards_per_agent']} unexpectedly high"


def test_tool_calls_recorded():
    m = run_server(n_agents=4, max_steps=3)
    j = metrics_json(m)
    assert sum(j["tool_calls"].values()) > 0
