"""跑全部 19 个 problems 模块的 _self_test() + catalog/tracker 一致性检查。"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "problems"))

MODULES = [
    "p01_arrays_two_pointers",
    "p02_sliding_window",
    "p03_hashing",
    "p04_strings",
    "p05_linked_list",
    "p06_stack_queue",
    "p07_binary_search",
    "p08_binary_tree",
    "p09_bst",
    "p10_backtracking",
    "p11_graph_bfs_dfs",
    "p12_dp_basics",
    "p13_dp_advanced",
    "p14_greedy",
    "p15_heap_topk",
    "p16_monotonic_stack",
    "p17_union_find",
    "p18_prefix_sum",
    "p19_bit_math",
    # Phase 2：进阶补充（_ii）
    "p01_arrays_two_pointers_ii",
    "p02_sliding_window_ii",
    "p03_hashing_ii",
    "p04_strings_ii",
    "p05_linked_list_ii",
    "p06_stack_queue_ii",
    "p07_binary_search_ii",
    "p08_binary_tree_ii",
    "p09_bst_ii",
    "p10_backtracking_ii",
    "p11_graph_bfs_dfs_ii",
    "p12_dp_basics_ii",
    "p13_dp_advanced_ii",
    "p14_greedy_ii",
    "p15_heap_topk_ii",
    "p16_monotonic_stack_ii",
    "p17_union_find_ii",
    "p18_prefix_sum_ii",
    "p19_bit_math_ii",
    # Phase 2：全新分类
    "p20_trie",
    "p21_design",
    "p22_advanced_graph",
    "p23_matrix_simulation",
    "p24_advanced_strings",
    "p25_concurrency",
]


def main() -> int:
    passed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    print(f"=== {passed}/{len(MODULES)} modules passed ===")

    import catalog
    catalog._self_test()

    import tracker
    tracker._self_test()

    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
