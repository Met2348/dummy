"""Tests for scheduling policies + sampler + naive attention."""
import sys, pathlib, math
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import Request
from scheduling_policies import schedule_order
from sampling import SamplerConfig, sample_one, top_k_top_p_mask, apply_repetition_penalty
from attention_naive import naive_attention


def _req(rid, plen, mlen, prio=0):
    r = Request(rid=rid, prompt_ids=list(range(plen)), max_new_tokens=mlen)
    r.priority = prio
    return r


def test_fcfs_keeps_insertion_order():
    reqs = [_req(0, 10, 5), _req(1, 5, 5), _req(2, 1, 100)]
    assert schedule_order(reqs, "fcfs") == [0, 1, 2]


def test_sjf_picks_shortest_first():
    reqs = [_req(0, 100, 100), _req(1, 5, 5), _req(2, 20, 20)]
    assert schedule_order(reqs, "sjf") == [1, 2, 0]


def test_priority_picks_highest_first():
    reqs = [_req(0, 10, 5, prio=1), _req(1, 5, 5, prio=10), _req(2, 1, 100, prio=5)]
    assert schedule_order(reqs, "priority") == [1, 2, 0]


def test_sampler_temperature_zero_is_argmax():
    torch.manual_seed(0)
    logits = torch.randn(50)
    cfg = SamplerConfig(temperature=0)
    tok = sample_one(logits, torch.tensor([], dtype=torch.long), cfg)
    assert tok == int(logits.argmax().item())


def test_top_k_mask_keeps_only_k():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    mask = top_k_top_p_mask(logits, top_k=2, top_p=1.0, min_p=0.0)
    assert mask.tolist() == [False, True, False, False, True]


def test_top_p_mask_keeps_nucleus():
    # all-equal logits + top_p=0.6 keeps ~ ceil(0.6*5)=3 tokens
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    mask = top_k_top_p_mask(logits, top_k=0, top_p=0.6, min_p=0.0)
    assert mask.sum().item() >= 3


def test_repetition_penalty_reduces_repeats():
    logits = torch.tensor([1.0, 1.0, 1.0])
    out = apply_repetition_penalty(logits, torch.tensor([0]), penalty=2.0)
    assert out[0].item() < logits[0].item()
    assert out[1].item() == logits[1].item()


def test_naive_attention_softmax_normalised():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = naive_attention(q, k, v, causal=True)
    assert out.shape == (1, 2, 4, 8)
    assert not torch.isnan(out).any()


def test_causal_attention_blocks_future():
    """Compare row 0 of causal vs non-causal — should differ when seq>1."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out_c = naive_attention(q, k, v, causal=True)
    out_n = naive_attention(q, k, v, causal=False)
    # row 0 should differ because non-causal sees future tokens
    assert not torch.allclose(out_c[0, 0, 0], out_n[0, 0, 0], atol=1e-3)
