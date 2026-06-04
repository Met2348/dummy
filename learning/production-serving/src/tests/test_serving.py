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
