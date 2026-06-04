"""Tests across all spec-decode methods + tree attention + capstone."""
import sys, pathlib, math, random
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import softmax, sample_from, SpecMetrics
from classic_spec_decode import run_classic_spec, rejection_sample
from medusa_heads import MedusaHeads, medusa_step
from eagle_minimal import EagleDraft, eagle_step
from eagle2 import build_dynamic_tree
from lookahead import NgramPool, lookahead_step
from tree_attention import build_tree_mask, tree_attention_torch
from spec_eval import sim_speedup
from capstone_eagle3 import run_all, eval_method, make_target


def test_softmax_normalised():
    p = softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-9


def test_rejection_sample_returns_valid_token():
    rng = random.Random(0)
    p = [0.5, 0.3, 0.2]
    q = [0.4, 0.4, 0.2]
    out = rejection_sample(p, q, drafted=1, rng=rng)
    assert 0 <= out < 3


def test_classic_spec_outputs_correct_length():
    target = make_target(0.3, seed=0)
    draft = make_target(0.5, seed=7)
    out, m = run_classic_spec(draft, target, prompt=[0, 1], n_tokens=20, k=4, seed=0)
    assert len(out) >= 20
    assert m.n_iters > 0


def test_medusa_step_produces_tokens():
    target = make_target(0.3)
    heads = MedusaHeads(n_heads=3, noise=0.5)
    rng = random.Random(0)
    accepted, _ = medusa_step(target, heads, prefix=[0, 1, 2], rng=rng)
    assert len(accepted) >= 1
    assert len(accepted) <= heads.n_heads + 1


def test_eagle1_step_produces_tokens():
    target = make_target(0.3)
    draft = EagleDraft(k=4, noise=0.3)
    rng = random.Random(0)
    accepted, _ = eagle_step(target, draft, prefix=[0, 1, 2], rng=rng)
    assert 1 <= len(accepted) <= 5


def test_eagle2_build_tree_returns_K_leaves():
    target = make_target(0.3)
    rng = random.Random(0)
    leaves = build_dynamic_tree(target, prefix=[0, 1], K=4, max_depth=4, branch=2, rng=rng)
    assert 1 <= len(leaves) <= 4
    assert all(len(l.tokens) <= 4 for l in leaves)


def test_lookahead_pool_lookup():
    pool = NgramPool(n=2)
    pool.add_sequence([1, 2, 3, 4, 5])
    assert pool.lookup([1, 2]) == 3
    assert pool.lookup([2, 3]) == 4
    assert pool.lookup([99, 99]) is None


def test_tree_mask_builds_correctly():
    # Tree: 0 (root) → 1, 2 ; 1 → 3 ; 2 → 4
    parent = [-1, 0, 0, 1, 2]
    mask = build_tree_mask(parent)
    # node 3 sees ancestors 3, 1, 0
    assert mask[3].tolist() == [True, True, False, True, False]
    # node 4 sees ancestors 4, 2, 0
    assert mask[4].tolist() == [True, False, True, False, True]


def test_tree_attention_runs():
    torch.manual_seed(0)
    parent = [-1, 0, 0, 1, 2]
    q = torch.randn(5, 8)
    k = torch.randn(5, 8)
    v = torch.randn(5, 8)
    out = tree_attention_torch(q, k, v, parent)
    assert out.shape == (5, 8)
    assert not torch.isnan(out).any()


def test_sim_speedup_above_one_for_high_accept():
    m = SpecMetrics(n_iters=10, n_drafted=40, n_accepted=32, n_tokens_out=50)
    assert sim_speedup(m) > 1.0


def test_capstone_eval_method_returns_full_row():
    row = eval_method("eagle1", "code", n_tokens=20, seed=0)
    for key in ("method", "task", "accept_rate", "MAU", "sim_speedup", "n_iters"):
        assert key in row


def test_capstone_all_methods_produce_output():
    """4 methods × 5 tasks → 20 rows, every method gets positive token throughput."""
    rows = run_all(n_tokens=30, seed=0)
    assert len(rows) == 20
    for r in rows:
        assert r["n_tokens_out"] >= 30
        assert r["MAU"] > 0
        assert r["sim_speedup"] > 0
