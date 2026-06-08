"""Graduation tests - agent + budget + router + R1 deploy + 5-line E2E."""
import sys, pathlib, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent_inference_demo import run_multi_turn
from thinking_budget import generate_with_budget, inject_wait_tokens, THINK_OPEN, THINK_CLOSE
from multi_model_router import route, estimate_total_cost, TIERS
from vlm_serve import VlmRequest, vlm_generate, encode_image_mock
from embedding_serve import embed_text, embed_batch, cosine
from r1_tiny_deploy.serve import MockR1Model, run_demo, QUESTIONS
from graduation_e2e.ckpts import load_all, CKPTS, QUESTION
from graduation_e2e.compare import run_compare, to_md
from graduation_e2e.report import write_report
from serving_scorecard import (
    GraduationSLO,
    effective_goodput,
    rank_candidates,
    score_report,
)


# ---- Agent inference -----

def test_agent_cached_saves_tokens():
    s = run_multi_turn(turns=5, base_history=200, per_turn_new=50)
    assert s.cached_prefill_tokens < s.naive_prefill_tokens
    # cached saves at least 60%
    assert s.cached_prefill_tokens < 0.4 * s.naive_prefill_tokens


# ---- Thinking budget -----

def test_budget_force_closes_long_thinking():
    stream = [THINK_OPEN] + ["w" for _ in range(100)] + [THINK_CLOSE, "answer"]
    r = generate_with_budget(iter(stream), budget_tokens=10)
    assert r.forced_close
    assert r.thinking_tokens >= 10


def test_budget_allows_short_thinking():
    stream = [THINK_OPEN] + ["w" for _ in range(5)] + [THINK_CLOSE, "answer"]
    r = generate_with_budget(iter(stream), budget_tokens=50)
    assert not r.forced_close


def test_wait_injection_lengthens_thinking():
    stream = [THINK_OPEN] + [f"t{i}" for i in range(200)] + [THINK_CLOSE]
    out = inject_wait_tokens(stream, every_n=20)
    assert out.count("Wait") >= 5


# ---- Multi-model router -----

def test_router_picks_small_for_simple():
    t = route("hi")
    assert t is TIERS[0]


def test_router_picks_thinking_for_complex():
    q = "Please derive step-by-step a proof of optimisation of complex algorithm prove" * 20
    t = route(q)
    assert t.name == "thinking-o3"


def test_router_cost_breakdown_includes_all_tiers():
    queries = ["hi", "explain backprop in detail step-by-step", "ok"] * 5
    cost = estimate_total_cost(queries)
    assert set(cost.keys()) == {t.name for t in TIERS}
    assert sum(cost.values()) > 0


# ---- VLM mock -----

def test_vlm_response_mentions_images():
    r = vlm_generate(VlmRequest(text="describe", images=["img1", "img2"]))
    assert "2 images" in r


def test_vlm_no_images_returns_zero():
    r = vlm_generate(VlmRequest(text="describe", images=None))
    assert "0 images" in r


# ---- Embedding -----

def test_embed_text_returns_vector_of_dim():
    v = embed_text("hello", dim=32)
    assert len(v) == 32


def test_embed_batch_matches_individual():
    texts = ["a", "b", "c"]
    batch = embed_batch(texts)
    for i, t in enumerate(texts):
        assert batch[i] == embed_text(t)


def test_cosine_self_one():
    v = embed_text("x")
    assert abs(cosine(v, v) - 1.0) < 1e-6


# ---- R1-tiny deploy -----

def test_r1_tiny_demo_runs_5_questions():
    out = run_demo()
    assert len(out) == 5
    for r in out:
        assert r["thinking_present"]
        assert r["answer"]


def test_r1_tiny_stream_yields_think_and_answer_tags():
    model = MockR1Model()
    text = "".join(model.stream(QUESTIONS[0]))
    assert "<think>" in text and "</think>" in text
    assert "<answer>" in text and "</answer>" in text


# ---- 5-line graduation -----

def test_load_5_ckpts():
    ck = load_all()
    assert len(ck) == 5
    assert set(ck.keys()) == {"vanilla", "lora", "dpo", "r1_zero", "phi_tiny"}


def test_compare_yields_5_results():
    rep = run_compare()
    assert rep["question"] == QUESTION
    assert len(rep["results"]) == 5


def test_compare_md_contains_all_ckpts():
    md = to_md(run_compare())
    for k in CKPTS.keys():
        assert k in md
    assert "Ablation contributions" in md


def test_compare_vanilla_wrong_others_correct():
    rep = run_compare()
    res = {r["ckpt"]: r for r in rep["results"]}
    assert res["vanilla"]["correct"] is False
    assert all(res[k]["correct"] for k in ("lora", "dpo", "r1_zero", "phi_tiny"))


def test_compare_r1_zero_has_thinking_trace():
    rep = run_compare()
    res = next(r for r in rep["results"] if r["ckpt"] == "r1_zero")
    assert "<think>" in res["response"]


def test_write_report_creates_md_and_json():
    with tempfile.TemporaryDirectory() as d:
        paths = write_report(d)
        assert "report.md" in paths
        assert "report.json" in paths
        for name in paths:
            assert os.path.exists(os.path.join(d, name))


# ---- Graduation scorecard -----

def test_scorecard_marks_wrong_or_slow_outputs_as_not_good():
    scores = score_report(run_compare(), GraduationSLO(max_ttft_ms=70, max_tpot_ms=8))
    by_ckpt = {s.ckpt: s for s in scores}

    assert by_ckpt["vanilla"].passes is False
    assert by_ckpt["r1_zero"].passes is False
    assert by_ckpt["lora"].passes is True
    assert by_ckpt["dpo"].passes is True
    assert by_ckpt["phi_tiny"].passes is True


def test_scorecard_goodput_is_request_rate_times_attainment():
    scores = score_report(run_compare(), GraduationSLO(max_ttft_ms=70, max_tpot_ms=8))
    summary = effective_goodput(scores, request_rate_rps=2.0)

    assert summary["attainment"] == 0.6
    assert summary["goodput_rps"] == 1.2


def test_rank_candidates_prefers_cheapest_passing_model():
    scores = score_report(run_compare(), GraduationSLO(max_ttft_ms=70, max_tpot_ms=8))
    ranked = rank_candidates(scores)

    assert ranked[0].ckpt == "lora"
    assert ranked[0].passes is True
