"""Tests for OpenAI compat + SSE + metrics + cost + trt build config."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from openai_api_server import (
    build_models_response, build_completion_response,
    build_stream_chunk, build_error,
    validate_chat_request, mock_generate, mock_stream,
)
from streaming_sse import sse_encode, sse_done, chunks_to_sse, parse_sse_line
from metrics_prometheus import Counter, Histogram, REQS, TTFT, render_all
from cost_calc import (
    Workload, Deployment, dollars_per_million_tokens, cost_for_workload,
    cache_hit_savings, cost_aware_route,
)
from trtllm_build import TrtLlmBuildConfig
from clipper_original_minimal import (
    AdaptiveBatcher,
    ModelContainer,
    PredictionCache,
    best_effort_ensemble,
    exp3_probabilities,
    exp3_update,
)


# ---- OpenAI compat -----

def test_models_response_shape():
    r = build_models_response()
    assert r["object"] == "list"
    assert len(r["data"]) >= 1
    assert "id" in r["data"][0]


def test_completion_response_includes_usage():
    req = {"model": "mock", "messages": [{"role": "user", "content": "hi"}]}
    r = build_completion_response(req, "Hello", prompt_tokens=1, out_tokens=1)
    assert r["model"] == "mock"
    assert r["usage"]["total_tokens"] == 2


def test_stream_chunk_with_finish():
    c = build_stream_chunk("m", {"content": "x"})
    assert c["choices"][0]["delta"] == {"content": "x"}
    done = build_stream_chunk("m", {}, finish="stop")
    assert done["choices"][0]["finish_reason"] == "stop"


def test_error_shape():
    e = build_error("oops")
    assert e["error"]["type"] == "invalid_request_error"


# ---- Validation -----

def test_validate_chat_request_missing_model():
    err = validate_chat_request({"messages": [{"role": "user", "content": "hi"}]})
    assert err and "model" in err


def test_validate_chat_request_missing_messages():
    err = validate_chat_request({"model": "m"})
    assert err and "messages" in err


def test_validate_chat_request_invalid_role():
    err = validate_chat_request({"model": "m", "messages": [{"role": "evil", "content": "x"}]})
    assert err and "role" in err


def test_validate_chat_request_ok():
    err = validate_chat_request({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert err is None


def test_mock_generate_returns_text_and_counts():
    text, p, o = mock_generate({"model": "m", "messages": [{"role": "user", "content": "hi there"}]})
    assert isinstance(text, str)
    assert p >= 1
    assert o >= 1


# ---- SSE -----

def test_sse_encode_format():
    out = sse_encode({"a": 1})
    assert out.startswith("data: ")
    assert out.endswith("\n\n")


def test_sse_done_format():
    assert sse_done() == "data: [DONE]\n\n"


def test_parse_sse_done_signal():
    assert parse_sse_line("data: [DONE]") == {"_done": True}


def test_parse_sse_normal_line():
    p = parse_sse_line('data: {"x": 1}')
    assert p == {"x": 1}


def test_chunks_to_sse_terminates_with_done():
    items = list(chunks_to_sse([{"a": 1}, {"a": 2}]))
    assert items[-1] == "data: [DONE]\n\n"
    assert len(items) == 3


# ---- Metrics -----

def test_counter_inc_and_render():
    c = Counter("test_c", "help", labels=["m"])
    c.inc(("model_a",), 1)
    c.inc(("model_a",), 2)
    rendered = c.render()
    assert "model_a" in rendered
    assert "test_c{m=\"model_a\"} 3" in rendered


def test_histogram_observe_and_percentile():
    h = Histogram("test_h", "help")
    for v in [0.01, 0.02, 0.5, 0.8, 1.2]:
        h.observe(v)
    assert h.count_ == 5
    p50 = h.percentile(0.5)
    assert p50 >= 0.025


def test_render_all_includes_both_metrics():
    REQS.inc(("m",), 1)
    TTFT.observe(0.1)
    out = render_all()
    assert "llm_requests_total" in out
    assert "llm_ttft_seconds" in out


# ---- Cost -----

def test_dollars_per_million_tokens():
    d = Deployment("h100", gpu_cost_per_hour=4.0, tok_per_s_per_gpu=5000.0)
    cost = dollars_per_million_tokens(d)
    assert 0.2 <= cost <= 0.3


def test_cost_for_workload_scales_linearly():
    d = Deployment("h100", gpu_cost_per_hour=4.0, tok_per_s_per_gpu=5000.0)
    w1 = Workload(qps=10, avg_in_tokens=100, avg_out_tokens=100)
    w2 = Workload(qps=20, avg_in_tokens=100, avg_out_tokens=100)
    assert abs(cost_for_workload(d, w2) / cost_for_workload(d, w1) - 2.0) < 0.01


def test_cache_hit_savings_positive():
    d = Deployment("h100", gpu_cost_per_hour=4.0, tok_per_s_per_gpu=5000.0)
    w = Workload(qps=10, avg_in_tokens=1000, avg_out_tokens=100)
    s = cache_hit_savings(d, w, hit_rate=0.8, hit_discount=0.9)
    assert s > 0


def test_cost_aware_route_picks_small_for_simple():
    small = Deployment("small", 1.0, 2000)
    large = Deployment("large", 8.0, 1500)
    assert cost_aware_route(0.1, small, large) is small
    assert cost_aware_route(0.9, small, large) is large


# ---- TRT-LLM build -----

def test_trtllm_build_cli_contains_required_flags():
    cli = TrtLlmBuildConfig(checkpoint_dir="./ckpt", output_dir="./engine").to_cli()
    assert "trtllm-build" in cli
    assert "--checkpoint_dir ./ckpt" in cli
    assert "--max_batch_size 32" in cli
    assert "--use_paged_context_fmha enable" in cli


# ---- Clipper paper mechanisms -----

def test_prediction_cache_lru_hit_and_evict():
    cache = PredictionCache(capacity=2)
    cache.put("m1", "x1", "y1")
    cache.put("m1", "x2", "y2")

    assert cache.fetch("m1", "x1") == "y1"

    cache.put("m1", "x3", "y3")
    assert cache.fetch("m1", "x2") is None
    assert cache.fetch("m1", "x1") == "y1"


def test_adaptive_batcher_increases_then_backs_off():
    batcher = AdaptiveBatcher(slo_ms=20.0, max_batch_size=4)
    batcher.observe(batch_size=4, latency_ms=12.0)
    assert batcher.max_batch_size == 5

    batcher.observe(batch_size=5, latency_ms=25.0)
    assert batcher.max_batch_size == 4


def test_batcher_respects_queue_and_current_cap():
    batcher = AdaptiveBatcher(slo_ms=20.0, max_batch_size=8)
    assert batcher.choose_batch_size(queue_len=3) == 3
    assert batcher.choose_batch_size(queue_len=20) == 8


def test_best_effort_ensemble_drops_straggler_at_deadline():
    models = [
        ModelContainer("fast_a", fixed_ms=2.0, per_item_ms=1.0, accuracy=0.80),
        ModelContainer("fast_b", fixed_ms=3.0, per_item_ms=1.0, accuracy=0.82),
        ModelContainer("slow", fixed_ms=50.0, per_item_ms=5.0, accuracy=0.95),
    ]

    result = best_effort_ensemble(models, query="x", deadline_ms=10.0)

    assert result["used_models"] == ["fast_a", "fast_b"]
    assert result["missing_models"] == ["slow"]
    assert result["confidence"] == 2 / 3


def test_exp3_update_increases_probability_after_reward():
    weights = {"a": 1.0, "b": 1.0}
    before = exp3_probabilities(weights)
    after_weights = exp3_update(weights, chosen="a", reward=1.0, probability=before["a"])
    after = exp3_probabilities(after_weights)

    assert after["a"] > before["a"]
