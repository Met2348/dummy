"""Radix tree correctness tests."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from radix_tree import RadixTree


def test_first_insert_full_miss():
    tree = RadixTree()
    leaf, matched = tree.insert([1, 2, 3, 4])
    assert matched == 0
    assert tree.n_tokens == 4


def test_second_insert_full_prefix_hit():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4])
    leaf, matched = tree.insert([1, 2, 3, 4, 5, 6])
    assert matched == 4
    assert tree.n_tokens == 6


def test_third_insert_partial_prefix_splits():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    leaf, matched = tree.insert([1, 2, 3, 8, 9])
    assert matched == 3
    # after split we expect the original "[1,2,3,4,5]" node to live as
    # [1,2,3]+[4,5] subtree; total tokens 5 + 2 (8,9) = 7
    assert tree.n_tokens == 7


def test_disjoint_prompts_share_nothing():
    tree = RadixTree()
    tree.insert([1, 2, 3])
    leaf, matched = tree.insert([10, 20, 30])
    assert matched == 0


def test_acquire_release_refcount():
    tree = RadixTree()
    leaf, _ = tree.insert([1, 2, 3, 4])
    tree.acquire(leaf)
    cur = leaf
    rc_path = []
    while cur is not tree.root:
        rc_path.append(cur.refcount)
        cur = cur.parent
    assert all(r == 1 for r in rc_path)
    tree.release(leaf)
    cur = leaf
    while cur is not tree.root:
        assert cur.refcount == 0
        cur = cur.parent


def test_evict_removes_zero_ref_leaf():
    """Evict frees ≥3 tokens; the older insert is the LRU victim."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4])           # inserted first → smaller last_access
    tree.insert([10, 11, 12])           # newer
    freed = tree.evict(want_tokens=3)
    assert freed >= 3
    # Older branch evicted; newer 3-token branch survives.
    assert tree.n_tokens == 3


def test_hit_rate_rises_with_shared_prefix():
    tree = RadixTree()
    sys_prompt = list(range(20))
    for q in range(10):
        tree.insert(sys_prompt + [100 + q])
    # First call had 20-token miss; subsequent 9 calls each hit 20 tokens.
    assert tree.hit_rate > 0.7
