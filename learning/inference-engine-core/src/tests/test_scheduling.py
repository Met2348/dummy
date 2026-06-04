"""Tests for continuous batching, chunked prefill, prefix cache."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import Request
from continuous_batching import Engine, make_request
from chunked_prefill import ChunkedPrefillEngine, PrefillState
from prefix_cache import PrefixCache


def test_continuous_batching_finishes_all():
    eng = Engine(max_running=4, kv_budget=200, eos_id=-1)
    for i in range(10):
        eng.add_request(make_request(i, prompt_len=8, max_new=5))
    m = eng.run()
    assert len(eng.finished) == 10
    assert m.n_tokens_out == 50


def test_continuous_batching_kv_budget_starvation_avoided():
    # Tight KV budget should still complete by serialising
    eng = Engine(max_running=8, kv_budget=30, eos_id=-1)
    for i in range(5):
        eng.add_request(make_request(i, prompt_len=4, max_new=3))
    m = eng.run()
    assert len(eng.finished) == 5
    assert m.n_tokens_out == 15


def test_chunked_prefill_processes_long_prompt():
    eng = ChunkedPrefillEngine(max_tokens_per_iter=32)
    long_req = Request(rid=0, prompt_ids=list(range(200)), max_new_tokens=3)
    eng.add_request(long_req)
    log = eng.run_until_idle()
    prefill_total = sum(c[1] for step in log for c in step["prefill_chunks"] if c[0] == 0)
    assert prefill_total == 200, f"expected 200 prefill tokens, got {prefill_total}"
    assert long_req in eng.finished


def test_chunked_prefill_mixes_decode_with_prefill():
    eng = ChunkedPrefillEngine(max_tokens_per_iter=32)
    eng.decoding.append(Request(rid=99, prompt_ids=[0]*10, max_new_tokens=50))  # warm decoder
    eng.add_request(Request(rid=0, prompt_ids=list(range(64)), max_new_tokens=1))
    # at least one iter should have BOTH decode and prefill chunk
    log = eng.run_until_idle()
    mixed = [s for s in log if s["decode"] > 0 and s["prefill_chunks"]]
    assert mixed, "expected at least one iter with mixed decode+prefill"


def test_prefix_cache_hits_rise_with_shared_system_prompt():
    cache = PrefixCache(block_size=4, cap=32)
    sys_prompt = list(range(20))
    for q in range(20):
        cache.mount(sys_prompt + [100 + q])
    assert cache.hit_rate > 0.7, f"hit_rate {cache.hit_rate:.3f} too low"


def test_prefix_cache_lru_eviction():
    cache = PrefixCache(block_size=4, cap=4)
    # 6 unique prompts of one block each
    for i in range(6):
        cache.mount(list(range(i * 100, i * 100 + 4)))
    assert len(cache.table) == 4
