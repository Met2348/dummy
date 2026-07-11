"""跑全部 77 个 problems 模块（Phase1核心19 + Phase2进阶19+新6 + Phase3竞赛级25+新8）
的 _self_test() + catalog/tracker 一致性检查。"""
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
    # Phase 3：竞赛级补充（_iii）
    "p01_arrays_two_pointers_iii",
    "p02_sliding_window_iii",
    "p03_hashing_iii",
    "p04_strings_iii",
    "p05_linked_list_iii",
    "p06_stack_queue_iii",
    "p07_binary_search_iii",
    "p08_binary_tree_iii",
    "p09_bst_iii",
    "p10_backtracking_iii",
    "p11_graph_bfs_dfs_iii",
    "p12_dp_basics_iii",
    "p13_dp_advanced_iii",
    "p14_greedy_iii",
    "p15_heap_topk_iii",
    "p16_monotonic_stack_iii",
    "p17_union_find_iii",
    "p18_prefix_sum_iii",
    "p19_bit_math_iii",
    "p20_trie_iii",
    "p21_design_iii",
    "p22_advanced_graph_iii",
    "p23_matrix_simulation_iii",
    "p24_advanced_strings_iii",
    "p25_concurrency_iii",
    # Phase 3：全新竞赛级分类
    "p26_number_theory",
    "p27_segment_tree_bit",
    "p28_string_matching_advanced",
    "p29_computational_geometry",
    "p30_advanced_graph_ii",
    "p31_bitmask_dp",
    "p32_game_theory_combinatorics",
    "p33_advanced_data_structures",
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
