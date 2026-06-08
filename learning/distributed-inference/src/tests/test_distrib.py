"""Tests for TP/PP/EP mocks + disagg + routing."""
import sys, pathlib, random
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tp_demo import ColumnSplitLinear, RowSplitLinear, TpMlp
from pp_demo import gpipe_bubble, interleaved_bubble, schedule_naive, render_grid
from ep_demo import MoEEpDemo, all_to_all_time_ms
from disaggregated_mock import WorkloadConfig, HardwareConfig, colocate, disagg, kv_transfer_ms
from kv_transfer_mock import transfer_time_ms, kv_payload_bytes, streaming_overlap, batched_no_overlap, BANDWIDTHS_GBPS
from routing_policies import Router, evaluate, prompt_hash
from capstone_disagg import run_all, to_md


def test_column_split_matches_single():
    torch.manual_seed(0)
    X = torch.randn(8, 64)
    W = torch.randn(64, 128)
    ref = X @ W
    out = ColumnSplitLinear(W, n_shards=4).forward(X)
    assert torch.allclose(ref, out, atol=1e-5)


def test_row_split_matches_single():
    torch.manual_seed(0)
    X = torch.randn(8, 64)
    W = torch.randn(64, 32)
    ref = X @ W
    X_shards = list(torch.chunk(X, 4, dim=-1))
    out = RowSplitLinear(W, n_shards=4).forward(X_shards)
    assert torch.allclose(ref, out, atol=1e-5)


def test_tp_mlp_matches_single():
    torch.manual_seed(0)
    X = torch.randn(4, 32)
    W_up = torch.randn(32, 128)
    W_down = torch.randn(128, 32)
    mlp = TpMlp(W_up, W_down, n_shards=4)
    out_tp = mlp.forward_tp(X)
    out_single = mlp.forward_single(X)
    assert torch.allclose(out_tp, out_single, atol=1e-4)


def test_gpipe_bubble_decreases_with_micro():
    b_few = gpipe_bubble(n_stages=4, n_micro=4)
    b_many = gpipe_bubble(n_stages=4, n_micro=16)
    assert b_many < b_few


def test_interleaved_better_than_naive():
    naive = gpipe_bubble(n_stages=4, n_micro=8)
    inter = interleaved_bubble(n_stages=4, n_micro=8, n_chunks=2)
    assert inter < naive


def test_schedule_grid_shape():
    grid = schedule_naive(4, 4)
    assert len(grid) == 4
    assert len(grid[0]) == 4 + 4 - 1
    # ensure render works
    assert "F1" in render_grid(grid)


def test_moe_assign_expert_to_gpu():
    moe = MoEEpDemo(n_experts=8, n_gpus=4, top_k=2)
    assert moe.assign_expert_to_gpu(0) == 0
    assert moe.assign_expert_to_gpu(2) == 1
    assert moe.assign_expert_to_gpu(7) == 3


def test_moe_load_imbalance_under_uniform():
    moe = MoEEpDemo(n_experts=8, n_gpus=4, top_k=2)
    rng = random.Random(0)
    routes = moe.route_tokens(list(range(1000)), rng)
    loads = moe.load_per_gpu(routes)
    imb = moe.load_imbalance(loads)
    assert imb < 0.2, f"imbalance {imb:.3f} too high"


def test_all_to_all_time_positive():
    t = all_to_all_time_ms(n_ranks=8, bytes_per_rank=1_000_000, bw_gbps=900)
    assert t > 0


def test_disagg_near_better_than_colocate_for_long_prompt():
    w = WorkloadConfig(n_reqs=32, prompt_len=4096, out_len=64)
    hw = HardwareConfig()
    colo = colocate(w, hw)
    da = disagg(w, hw, cross_node=False)
    assert da["tok_per_s"] > colo["tok_per_s"] * 1.2


def test_kv_transfer_remote_slower_than_near():
    near = transfer_time_ms(1_000_000, "nvlink_4")
    far = transfer_time_ms(1_000_000, "ib_400g")
    assert far > near * 5


def test_streaming_overlap_smaller_than_batch():
    s = streaming_overlap(prefill_ms=100, transfer_ms=80, decode_ms=200)
    b = batched_no_overlap(prefill_ms=100, transfer_ms=80, decode_ms=200)
    assert s < b


def test_kv_payload_bytes_grows_with_seq():
    a = kv_payload_bytes(1024, 8, 128)
    b = kv_payload_bytes(2048, 8, 128)
    assert b == 2 * a


def test_router_prefix_hash_consistent():
    r1 = Router(4)
    r2 = Router(4)
    # Same prompt maps to the same replica.
    assert r1.route([1, 2, 3, 4, 5, 6, 7, 8], "prefix_hash") == r2.route([1, 2, 3, 4, 5, 6, 7, 8], "prefix_hash")


def test_router_hit_rate_higher_for_prefix_aware():
    prompts = [[i, 1, 2, 3, 4, 5] for i in range(2)] * 20
    rr = evaluate("round_robin", prompts, n_replicas=4)
    pa = evaluate("load_aware_prefix", prompts, n_replicas=4)
    assert pa["hit_rate"] >= rr["hit_rate"] - 0.05


def test_capstone_3_configs():
    rows = run_all()
    assert len(rows) == 3
    configs = {r["config"] for r in rows}
    assert configs == {"colocate", "disagg-near", "disagg-remote"}
    md = to_md(rows)
    assert "TTFT" in md
