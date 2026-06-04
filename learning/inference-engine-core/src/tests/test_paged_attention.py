"""Numerical and structural tests for naive + paged KV caches."""
import math
import sys
import pathlib
from contextlib import contextmanager
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from naive_kv import NaiveKvPool, demo_fragmentation
from paged_kv import PagedKvPool, BlockTable, utilization
from paged_attention_triton import paged_attention_torch


@contextmanager
def raises(exc_type, match=None):
    try:
        yield
    except exc_type as e:
        if match and match not in str(e):
            raise AssertionError(f"expected {match!r} in error, got {e!r}")
        return
    raise AssertionError(f"expected {exc_type.__name__}, no exception raised")


def test_naive_fragmentation_high():
    info = demo_fragmentation(B=8, max_len=2048, avg_len=512)
    assert info["wasted_pct"] > 0.5, f"expected >50% waste, got {info['wasted_pct']:.3f}"


def test_paged_pool_no_oom_under_capacity():
    pool = PagedKvPool(n_blocks=8, block_size=16, n_kv_heads=4, head_dim=32, n_layers=1)
    table = BlockTable(pool)
    # 7 blocks worth of tokens
    for i in range(7 * 16):
        k = torch.randn(4, 32, dtype=torch.float16)
        v = torch.randn(4, 32, dtype=torch.float16)
        table.append_token(0, k, v)
    assert table.n_tokens == 7 * 16
    assert pool.n_free() == 1


def test_paged_pool_oom_at_capacity():
    pool = PagedKvPool(n_blocks=2, block_size=4, n_kv_heads=2, head_dim=8, n_layers=1)
    table = BlockTable(pool)
    for i in range(8):
        table.append_token(0, torch.randn(2, 8, dtype=torch.float16), torch.randn(2, 8, dtype=torch.float16))
    with raises(RuntimeError, match="OOM"):
        table.append_token(0, torch.randn(2, 8, dtype=torch.float16), torch.randn(2, 8, dtype=torch.float16))


def test_paged_pool_utilization_high():
    """Variable-length writes still keep utilization >85% (paged beats naive)."""
    pool = PagedKvPool(n_blocks=32, block_size=8, n_kv_heads=2, head_dim=16, n_layers=1)
    tables = [BlockTable(pool) for _ in range(4)]
    lengths = [40, 17, 25, 8]
    for t, length in zip(tables, lengths):
        for i in range(length):
            t.append_token(0, torch.randn(2, 16, dtype=torch.float16), torch.randn(2, 16, dtype=torch.float16))
    util = utilization(tables)
    assert util > 0.85, f"paged util {util:.3f} unexpectedly low"


def test_paged_pool_fork_shares_blocks():
    pool = PagedKvPool(n_blocks=8, block_size=4, n_kv_heads=2, head_dim=8, n_layers=1)
    parent = BlockTable(pool)
    for i in range(10):
        parent.append_token(0, torch.randn(2, 8, dtype=torch.float16), torch.randn(2, 8, dtype=torch.float16))
    free_before = pool.n_free()
    child = parent.fork()
    assert pool.n_free() == free_before, "fork must not consume blocks"
    assert all(b1 == b2 for b1, b2 in zip(parent.block_ids, child.block_ids))


def test_paged_attention_matches_dense():
    """paged_attention_torch == dense attention when blocks are filled in order."""
    torch.manual_seed(42)
    block_size, n_blocks, n_kv_heads, d_h = 4, 8, 2, 16
    n_heads = 4
    n_tokens = 13
    k_pool = torch.zeros(n_blocks, block_size, n_kv_heads, d_h)
    v_pool = torch.zeros(n_blocks, block_size, n_kv_heads, d_h)

    K_dense = torch.randn(n_tokens, n_kv_heads, d_h)
    V_dense = torch.randn(n_tokens, n_kv_heads, d_h)
    block_table = []
    for i, tok in enumerate(range(n_tokens)):
        bid = i // block_size
        if bid >= len(block_table):
            block_table.append(bid + 2)   # arbitrary physical id
        phys = block_table[bid]
        slot = i % block_size
        k_pool[phys, slot] = K_dense[i]
        v_pool[phys, slot] = V_dense[i]

    q = torch.randn(n_heads, d_h)
    rep = n_heads // n_kv_heads
    K_rep = K_dense.repeat_interleave(rep, dim=1)
    V_rep = V_dense.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(d_h)
    scores = torch.einsum("hd,thd->th", q, K_rep) * scale
    attn = torch.softmax(scores, dim=0)
    ref = torch.einsum("th,thd->hd", attn, V_rep)

    out = paged_attention_torch(q, block_table, n_tokens, k_pool, v_pool)
    assert torch.allclose(out, ref, atol=1e-5)
