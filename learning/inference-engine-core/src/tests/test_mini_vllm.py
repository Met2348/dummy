"""Tests for the mini-vLLM capstone — smoke + throughput sanity."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_vllm import run_case, run_all, CASES


def test_case_1_short_short_completes():
    r = run_case(1)
    assert r["tokens_out"] == CASES[1]["n"] * CASES[1]["m"]
    assert r["throughput"] > 0


def test_case_2_long_prompt_completes():
    r = run_case(2)
    assert r["tokens_out"] == CASES[2]["n"] * CASES[2]["m"]


def test_case_3_long_output_completes():
    r = run_case(3)
    assert r["tokens_out"] == CASES[3]["n"] * CASES[3]["m"]


def test_case_4_big_batch_no_oom():
    r = run_case(4)
    assert r["tokens_out"] == CASES[4]["n"] * CASES[4]["m"]


def test_case_5_streaming_one_request():
    r = run_case(5)
    assert r["tokens_out"] == CASES[5]["m"]


def test_run_all_returns_5():
    rs = run_all()
    assert len(rs) == 5
    names = {r["name"] for r in rs}
    assert names == {c["name"] for c in CASES.values()}
