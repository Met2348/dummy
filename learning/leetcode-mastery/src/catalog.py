"""100 题元数据登记表 —— 单一数据源。

README 的统计表、tracker 的种子数据都从这里生成，避免多处手动同步出偏差。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    id: str            # 如 "lc1"，对应 problems/*.py 里的测试用例 id
    leetcode_no: int
    name: str
    category: str       # 对应 lectures/NN-*.md 的分类名
    module: str         # 对应 problems/pNN_*.py 的模块名（不含 .py）
    difficulty: str     # "E" | "M" | "H"


PROBLEMS: list[Problem] = [
    # 01 数组与双指针
    Problem("lc1", 1, "两数之和", "数组与双指针", "p01_arrays_two_pointers", "E"),
    Problem("lc26", 26, "删除有序数组中的重复项", "数组与双指针", "p01_arrays_two_pointers", "E"),
    Problem("lc27", 27, "移除元素", "数组与双指针", "p01_arrays_two_pointers", "E"),
    Problem("lc15", 15, "三数之和", "数组与双指针", "p01_arrays_two_pointers", "M"),
    Problem("lc11", 11, "盛最多水的容器", "数组与双指针", "p01_arrays_two_pointers", "M"),
    Problem("lc42", 42, "接雨水", "数组与双指针", "p01_arrays_two_pointers", "H"),
    # 02 滑动窗口
    Problem("lc3", 3, "无重复字符的最长子串", "滑动窗口", "p02_sliding_window", "M"),
    Problem("lc209", 209, "长度最小的子数组", "滑动窗口", "p02_sliding_window", "M"),
    Problem("lc438", 438, "找到字符串中所有字母异位词", "滑动窗口", "p02_sliding_window", "M"),
    Problem("lc567", 567, "字符串的排列", "滑动窗口", "p02_sliding_window", "M"),
    Problem("lc76", 76, "最小覆盖子串", "滑动窗口", "p02_sliding_window", "H"),
    # 03 哈希表
    Problem("lc217", 217, "存在重复元素", "哈希表", "p03_hashing", "E"),
    Problem("lc242", 242, "有效的字母异位词", "哈希表", "p03_hashing", "E"),
    Problem("lc49", 49, "字母异位词分组", "哈希表", "p03_hashing", "M"),
    Problem("lc128", 128, "最长连续序列", "哈希表", "p03_hashing", "M"),
    Problem("lc380", 380, "O(1) 时间插入、删除和获取随机元素", "哈希表", "p03_hashing", "M"),
    # 04 字符串
    Problem("lc344", 344, "反转字符串", "字符串", "p04_strings", "E"),
    Problem("lc14", 14, "最长公共前缀", "字符串", "p04_strings", "E"),
    Problem("lc5", 5, "最长回文子串", "字符串", "p04_strings", "M"),
    Problem("lc151", 151, "反转字符串中的单词", "字符串", "p04_strings", "M"),
    Problem("lc6", 6, "Z 字形变换", "字符串", "p04_strings", "M"),
    Problem("lc8", 8, "字符串转换整数 atoi", "字符串", "p04_strings", "M"),
    # 05 链表
    Problem("lc206", 206, "反转链表", "链表", "p05_linked_list", "E"),
    Problem("lc21", 21, "合并两个有序链表", "链表", "p05_linked_list", "E"),
    Problem("lc141", 141, "环形链表", "链表", "p05_linked_list", "E"),
    Problem("lc19", 19, "删除链表的倒数第 N 个结点", "链表", "p05_linked_list", "M"),
    Problem("lc2", 2, "两数相加", "链表", "p05_linked_list", "M"),
    Problem("lc23", 23, "合并 K 个升序链表", "链表", "p05_linked_list", "H"),
    # 06 栈与队列
    Problem("lc20", 20, "有效的括号", "栈与队列", "p06_stack_queue", "E"),
    Problem("lc232", 232, "用栈实现队列", "栈与队列", "p06_stack_queue", "E"),
    Problem("lc155", 155, "最小栈", "栈与队列", "p06_stack_queue", "M"),
    Problem("lc227", 227, "基本计算器 II", "栈与队列", "p06_stack_queue", "M"),
    Problem("lc224", 224, "基本计算器", "栈与队列", "p06_stack_queue", "H"),
    # 07 二分查找
    Problem("lc704", 704, "二分查找", "二分查找", "p07_binary_search", "E"),
    Problem("lc35", 35, "搜索插入位置", "二分查找", "p07_binary_search", "E"),
    Problem("lc34", 34, "在排序数组中查找元素的第一个和最后一个位置", "二分查找", "p07_binary_search", "M"),
    Problem("lc33", 33, "搜索旋转排序数组", "二分查找", "p07_binary_search", "M"),
    Problem("lc4", 4, "寻找两个正序数组的中位数", "二分查找", "p07_binary_search", "H"),
    # 08 二叉树遍历与构造
    Problem("lc94", 94, "二叉树的中序遍历", "二叉树遍历与构造", "p08_binary_tree", "E"),
    Problem("lc104", 104, "二叉树的最大深度", "二叉树遍历与构造", "p08_binary_tree", "E"),
    Problem("lc226", 226, "翻转二叉树", "二叉树遍历与构造", "p08_binary_tree", "E"),
    Problem("lc102", 102, "二叉树的层序遍历", "二叉树遍历与构造", "p08_binary_tree", "M"),
    Problem("lc105", 105, "从前序与中序遍历序列构造二叉树", "二叉树遍历与构造", "p08_binary_tree", "M"),
    Problem("lc236", 236, "二叉树的最近公共祖先", "二叉树遍历与构造", "p08_binary_tree", "M"),
    Problem("lc124", 124, "二叉树中的最大路径和", "二叉树遍历与构造", "p08_binary_tree", "H"),
    # 09 二叉搜索树
    Problem("lc108", 108, "将有序数组转换为二叉搜索树", "二叉搜索树", "p09_bst", "E"),
    Problem("lc98", 98, "验证二叉搜索树", "二叉搜索树", "p09_bst", "M"),
    Problem("lc230", 230, "二叉搜索树中第 K 小的元素", "二叉搜索树", "p09_bst", "M"),
    Problem("lc235", 235, "二叉搜索树的最近公共祖先", "二叉搜索树", "p09_bst", "M"),
    # 10 回溯
    Problem("lc78", 78, "子集", "回溯", "p10_backtracking", "M"),
    Problem("lc46", 46, "全排列", "回溯", "p10_backtracking", "M"),
    Problem("lc39", 39, "组合总和", "回溯", "p10_backtracking", "M"),
    Problem("lc22", 22, "括号生成", "回溯", "p10_backtracking", "M"),
    Problem("lc79", 79, "单词搜索", "回溯", "p10_backtracking", "M"),
    Problem("lc51", 51, "N 皇后", "回溯", "p10_backtracking", "H"),
    # 11 图 BFS/DFS
    Problem("lc200", 200, "岛屿数量", "图 BFS/DFS", "p11_graph_bfs_dfs", "M"),
    Problem("lc133", 133, "克隆图", "图 BFS/DFS", "p11_graph_bfs_dfs", "M"),
    Problem("lc207", 207, "课程表", "图 BFS/DFS", "p11_graph_bfs_dfs", "M"),
    Problem("lc210", 210, "课程表 II", "图 BFS/DFS", "p11_graph_bfs_dfs", "M"),
    Problem("lc994", 994, "腐烂的橘子", "图 BFS/DFS", "p11_graph_bfs_dfs", "M"),
    Problem("lc127", 127, "单词接龙", "图 BFS/DFS", "p11_graph_bfs_dfs", "H"),
    # 12 DP 基础/序列
    Problem("lc70", 70, "爬楼梯", "DP 基础/序列", "p12_dp_basics", "E"),
    Problem("lc198", 198, "打家劫舍", "DP 基础/序列", "p12_dp_basics", "M"),
    Problem("lc53", 53, "最大子数组和", "DP 基础/序列", "p12_dp_basics", "M"),
    Problem("lc152", 152, "乘积最大子数组", "DP 基础/序列", "p12_dp_basics", "M"),
    Problem("lc300", 300, "最长递增子序列", "DP 基础/序列", "p12_dp_basics", "M"),
    Problem("lc322", 322, "零钱兑换", "DP 基础/序列", "p12_dp_basics", "M"),
    Problem("lc139", 139, "单词拆分", "DP 基础/序列", "p12_dp_basics", "M"),
    # 13 DP 进阶
    Problem("lc121", 121, "买卖股票的最佳时机", "DP 进阶", "p13_dp_advanced", "E"),
    Problem("lc122", 122, "买卖股票的最佳时机 II", "DP 进阶", "p13_dp_advanced", "M"),
    Problem("lc416", 416, "分割等和子集", "DP 进阶", "p13_dp_advanced", "M"),
    Problem("lc1143", 1143, "最长公共子序列", "DP 进阶", "p13_dp_advanced", "M"),
    Problem("lc72", 72, "编辑距离", "DP 进阶", "p13_dp_advanced", "H"),
    Problem("lc312", 312, "戳气球", "DP 进阶", "p13_dp_advanced", "H"),
    # 14 贪心
    Problem("lc55", 55, "跳跃游戏", "贪心", "p14_greedy", "M"),
    Problem("lc45", 45, "跳跃游戏 II", "贪心", "p14_greedy", "M"),
    Problem("lc134", 134, "加油站", "贪心", "p14_greedy", "M"),
    Problem("lc435", 435, "无重叠区间", "贪心", "p14_greedy", "M"),
    Problem("lc763", 763, "划分字母区间", "贪心", "p14_greedy", "M"),
    # 15 堆/Top-K
    Problem("lc215", 215, "数组中的第K个最大元素", "堆/Top-K", "p15_heap_topk", "M"),
    Problem("lc347", 347, "前 K 个高频元素", "堆/Top-K", "p15_heap_topk", "M"),
    Problem("lc692", 692, "前K个高频单词", "堆/Top-K", "p15_heap_topk", "M"),
    Problem("lc253", 253, "会议室 II", "堆/Top-K", "p15_heap_topk", "M"),
    Problem("lc295", 295, "数据流的中位数", "堆/Top-K", "p15_heap_topk", "H"),
    # 16 单调栈/单调队列
    Problem("lc496", 496, "下一个更大元素 I", "单调栈/单调队列", "p16_monotonic_stack", "E"),
    Problem("lc739", 739, "每日温度", "单调栈/单调队列", "p16_monotonic_stack", "M"),
    Problem("lc84", 84, "柱状图中最大的矩形", "单调栈/单调队列", "p16_monotonic_stack", "H"),
    Problem("lc239", 239, "滑动窗口最大值", "单调栈/单调队列", "p16_monotonic_stack", "H"),
    # 17 并查集
    Problem("lc547", 547, "省份数量", "并查集", "p17_union_find", "M"),
    Problem("lc684", 684, "冗余连接", "并查集", "p17_union_find", "M"),
    Problem("lc721", 721, "账户合并", "并查集", "p17_union_find", "M"),
    # 18 前缀和与差分
    Problem("lc303", 303, "区域和检索-数组不可变", "前缀和与差分", "p18_prefix_sum", "E"),
    Problem("lc238", 238, "除自身以外数组的乘积", "前缀和与差分", "p18_prefix_sum", "M"),
    Problem("lc560", 560, "和为 K 的子数组", "前缀和与差分", "p18_prefix_sum", "M"),
    Problem("lc1109", 1109, "航班预订统计", "前缀和与差分", "p18_prefix_sum", "M"),
    # 19 位运算与数学
    Problem("lc136", 136, "只出现一次的数字", "位运算与数学", "p19_bit_math", "E"),
    Problem("lc191", 191, "位1的个数", "位运算与数学", "p19_bit_math", "E"),
    Problem("lc169", 169, "多数元素", "位运算与数学", "p19_bit_math", "E"),
    Problem("lc50", 50, "Pow(x, n)", "位运算与数学", "p19_bit_math", "M"),
    Problem("lc7", 7, "整数反转", "位运算与数学", "p19_bit_math", "M"),
]


def categories() -> list[str]:
    seen: list[str] = []
    for p in PROBLEMS:
        if p.category not in seen:
            seen.append(p.category)
    return seen


def stats() -> dict[str, int]:
    d = {"E": 0, "M": 0, "H": 0}
    for p in PROBLEMS:
        d[p.difficulty] += 1
    return d


def _self_test() -> None:
    assert len(PROBLEMS) == 100, f"应为 100 题，实际 {len(PROBLEMS)}"
    assert len(categories()) == 19
    ids = [p.id for p in PROBLEMS]
    assert len(ids) == len(set(ids)), "存在重复 id"
    nos = [p.leetcode_no for p in PROBLEMS]
    assert len(nos) == len(set(nos)), "存在重复 LeetCode 编号"
    for p in PROBLEMS:
        assert p.difficulty in ("E", "M", "H")
        assert p.module.startswith("p") and "_" in p.module
    s = stats()
    assert s["E"] + s["M"] + s["H"] == 100
    print(f"[PASS] catalog: 100 题元数据完整 (E={s['E']} M={s['M']} H={s['H']}, 19 类)")


if __name__ == "__main__":
    _self_test()
