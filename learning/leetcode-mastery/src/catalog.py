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
    tier: str = "core"  # "core"(Phase1 100题) | "advanced"(Phase2 198题) | "frontier"(Phase3 竞赛级)


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

    # ══════════════════ Phase 2：进阶补充（advanced tier，198 题）══════════════════
    # 01 数组与双指针 · 进阶
    Problem("lc18", 18, "四数之和", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    Problem("lc16", 16, "最接近的三数之和", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    Problem("lc611", 611, "有效三角形的个数", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    Problem("lc31", 31, "下一个排列", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    Problem("lc75", 75, "颜色分类", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    Problem("lc41", 41, "缺失的第一个正数", "数组与双指针", "p01_arrays_two_pointers_ii", "H", "advanced"),
    Problem("lc287", 287, "寻找重复数", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    Problem("lc88", 88, "合并两个有序数组", "数组与双指针", "p01_arrays_two_pointers_ii", "E", "advanced"),
    Problem("lc80", 80, "删除有序数组中的重复项II", "数组与双指针", "p01_arrays_two_pointers_ii", "M", "advanced"),
    # 02 滑动窗口 · 进阶
    Problem("lc1004", 1004, "最大连续1的个数III", "滑动窗口", "p02_sliding_window_ii", "M", "advanced"),
    Problem("lc1493", 1493, "删掉一个元素以后全为1的最长子数组", "滑动窗口", "p02_sliding_window_ii", "M", "advanced"),
    Problem("lc424", 424, "替换后的最长重复字符", "滑动窗口", "p02_sliding_window_ii", "M", "advanced"),
    Problem("lc30", 30, "串联所有单词的子串", "滑动窗口", "p02_sliding_window_ii", "H", "advanced"),
    Problem("lc632", 632, "最小区间", "滑动窗口", "p02_sliding_window_ii", "H", "advanced"),
    Problem("lc992", 992, "K个不同整数的子数组", "滑动窗口", "p02_sliding_window_ii", "H", "advanced"),
    Problem("lc643", 643, "子数组最大平均数I", "滑动窗口", "p02_sliding_window_ii", "E", "advanced"),
    # 03 哈希表 · 进阶
    Problem("lc205", 205, "同构字符串", "哈希表", "p03_hashing_ii", "E", "advanced"),
    Problem("lc290", 290, "单词规律", "哈希表", "p03_hashing_ii", "E", "advanced"),
    Problem("lc350", 350, "两个数组的交集II", "哈希表", "p03_hashing_ii", "E", "advanced"),
    Problem("lc149", 149, "直线上最多的点数", "哈希表", "p03_hashing_ii", "H", "advanced"),
    Problem("lc202", 202, "快乐数", "哈希表", "p03_hashing_ii", "E", "advanced"),
    Problem("lc387", 387, "字符串中的第一个唯一字符", "哈希表", "p03_hashing_ii", "E", "advanced"),
    Problem("lc454", 454, "四数相加II", "哈希表", "p03_hashing_ii", "M", "advanced"),
    # 04 字符串 · 进阶
    Problem("lc43", 43, "字符串相乘", "字符串", "p04_strings_ii", "M", "advanced"),
    Problem("lc68", 68, "文本左右对齐", "字符串", "p04_strings_ii", "H", "advanced"),
    Problem("lc71", 71, "简化路径", "字符串", "p04_strings_ii", "M", "advanced"),
    Problem("lc165", 165, "比较版本号", "字符串", "p04_strings_ii", "M", "advanced"),
    Problem("lc316", 316, "去除重复字母", "字符串", "p04_strings_ii", "M", "advanced"),
    Problem("lc12", 12, "整数转罗马数字", "字符串", "p04_strings_ii", "M", "advanced"),
    Problem("lc13", 13, "罗马数字转整数", "字符串", "p04_strings_ii", "E", "advanced"),
    Problem("lc838", 838, "推多米诺", "字符串", "p04_strings_ii", "M", "advanced"),
    # 05 链表 · 进阶
    Problem("lc25", 25, "K个一组翻转链表", "链表", "p05_linked_list_ii", "H", "advanced"),
    Problem("lc92", 92, "反转链表II", "链表", "p05_linked_list_ii", "M", "advanced"),
    Problem("lc138", 138, "随机链表的复制", "链表", "p05_linked_list_ii", "M", "advanced"),
    Problem("lc142", 142, "环形链表II", "链表", "p05_linked_list_ii", "M", "advanced"),
    Problem("lc143", 143, "重排链表", "链表", "p05_linked_list_ii", "M", "advanced"),
    Problem("lc148", 148, "排序链表", "链表", "p05_linked_list_ii", "M", "advanced"),
    Problem("lc160", 160, "相交链表", "链表", "p05_linked_list_ii", "E", "advanced"),
    Problem("lc24", 24, "两两交换链表中的节点", "链表", "p05_linked_list_ii", "M", "advanced"),
    # 06 栈与队列 · 进阶
    Problem("lc150", 150, "逆波兰表达式求值", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc394", 394, "字符串解码", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc946", 946, "验证栈序列", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc1249", 1249, "移除无效的括号", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc636", 636, "函数的独占时间", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc1472", 1472, "设计浏览器历史记录", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc1381", 1381, "设计一个支持增量操作的栈", "栈与队列", "p06_stack_queue_ii", "M", "advanced"),
    Problem("lc682", 682, "棒球比赛", "栈与队列", "p06_stack_queue_ii", "E", "advanced"),
    # 07 二分查找 · 进阶
    Problem("lc74", 74, "搜索二维矩阵", "二分查找", "p07_binary_search_ii", "M", "advanced"),
    Problem("lc240", 240, "搜索二维矩阵II", "二分查找", "p07_binary_search_ii", "M", "advanced"),
    Problem("lc153", 153, "寻找旋转排序数组中的最小值", "二分查找", "p07_binary_search_ii", "M", "advanced"),
    Problem("lc162", 162, "寻找峰值", "二分查找", "p07_binary_search_ii", "M", "advanced"),
    Problem("lc875", 875, "爱吃香蕉的珂珂", "二分查找", "p07_binary_search_ii", "M", "advanced"),
    Problem("lc1011", 1011, "在D天内送达包裹的能力", "二分查找", "p07_binary_search_ii", "M", "advanced"),
    Problem("lc410", 410, "分割数组的最大值", "二分查找", "p07_binary_search_ii", "H", "advanced"),
    Problem("lc154", 154, "寻找旋转排序数组中的最小值II", "二分查找", "p07_binary_search_ii", "H", "advanced"),
    # 08 二叉树遍历与构造 · 进阶
    Problem("lc144", 144, "二叉树的前序遍历", "二叉树遍历与构造", "p08_binary_tree_ii", "E", "advanced"),
    Problem("lc145", 145, "二叉树的后序遍历", "二叉树遍历与构造", "p08_binary_tree_ii", "E", "advanced"),
    Problem("lc543", 543, "二叉树的直径", "二叉树遍历与构造", "p08_binary_tree_ii", "E", "advanced"),
    Problem("lc113", 113, "路径总和II", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc437", 437, "路径总和III", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc662", 662, "二叉树最大宽度", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc199", 199, "二叉树的右视图", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc297", 297, "二叉树的序列化与反序列化", "二叉树遍历与构造", "p08_binary_tree_ii", "H", "advanced"),
    Problem("lc106", 106, "从中序与后序遍历序列构造二叉树", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc958", 958, "二叉树的完全性检验", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc1110", 1110, "删点成林", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    Problem("lc1325", 1325, "删除给定值的叶子节点", "二叉树遍历与构造", "p08_binary_tree_ii", "M", "advanced"),
    # 09 二叉搜索树 · 进阶
    Problem("lc700", 700, "二叉搜索树中的搜索", "二叉搜索树", "p09_bst_ii", "E", "advanced"),
    Problem("lc701", 701, "二叉搜索树中的插入操作", "二叉搜索树", "p09_bst_ii", "M", "advanced"),
    Problem("lc450", 450, "删除二叉搜索树中的节点", "二叉搜索树", "p09_bst_ii", "M", "advanced"),
    Problem("lc96", 96, "不同的二叉搜索树", "二叉搜索树", "p09_bst_ii", "M", "advanced"),
    Problem("lc95", 95, "不同的二叉搜索树II", "二叉搜索树", "p09_bst_ii", "M", "advanced"),
    Problem("lc538", 538, "把二叉搜索树转换为累加树", "二叉搜索树", "p09_bst_ii", "M", "advanced"),
    Problem("lc99", 99, "恢复二叉搜索树", "二叉搜索树", "p09_bst_ii", "H", "advanced"),
    # 10 回溯 · 进阶
    Problem("lc90", 90, "子集II", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc47", 47, "全排列II", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc40", 40, "组合总和II", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc216", 216, "组合总和III", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc17", 17, "电话号码的字母组合", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc93", 93, "复原IP地址", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc131", 131, "分割回文串", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc37", 37, "解数独", "回溯", "p10_backtracking_ii", "H", "advanced"),
    Problem("lc494", 494, "目标和", "回溯", "p10_backtracking_ii", "M", "advanced"),
    Problem("lc52", 52, "N皇后II", "回溯", "p10_backtracking_ii", "H", "advanced"),
    Problem("lc301", 301, "删除无效的括号", "回溯", "p10_backtracking_ii", "H", "advanced"),
    Problem("lc1980", 1980, "找出不同的二进制字符串", "回溯", "p10_backtracking_ii", "M", "advanced"),
    # 11 图 BFS/DFS · 进阶
    Problem("lc130", 130, "被围绕的区域", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc417", 417, "太平洋大西洋水流问题", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc695", 695, "岛屿的最大面积", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc1091", 1091, "二进制矩阵中的最短路径", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc329", 329, "矩阵中的最长递增路径", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "H", "advanced"),
    Problem("lc542", 542, "01矩阵", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc909", 909, "蛇梯棋", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc934", 934, "最短的桥", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc797", 797, "所有可能的路径", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc1971", 1971, "寻找图中是否存在路径", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "E", "advanced"),
    Problem("lc1466", 1466, "重新规划路线", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "M", "advanced"),
    Problem("lc1791", 1791, "找出星型图的中心节点", "图 BFS/DFS", "p11_graph_bfs_dfs_ii", "E", "advanced"),
    # 12 DP 基础/序列 · 进阶
    Problem("lc213", 213, "打家劫舍II", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc337", 337, "打家劫舍III", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc91", 91, "解码方法", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc279", 279, "完全平方数", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc343", 343, "整数拆分", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc62", 62, "不同路径", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc63", 63, "不同路径II", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc918", 918, "环形子数组的最大和", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc673", 673, "最长递增子序列的个数", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc1043", 1043, "分隔数组以得到最大和", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc264", 264, "丑数II", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc1027", 1027, "最长等差数列", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    Problem("lc926", 926, "将字符串翻转到单调递增", "DP 基础/序列", "p12_dp_basics_ii", "M", "advanced"),
    # 13 DP 进阶（背包/区间/编辑距离） · 进阶
    Problem("lc123", 123, "买卖股票的最佳时机III", "DP 进阶", "p13_dp_advanced_ii", "H", "advanced"),
    Problem("lc188", 188, "买卖股票的最佳时机IV", "DP 进阶", "p13_dp_advanced_ii", "H", "advanced"),
    Problem("lc309", 309, "最佳买卖股票时机含冷冻期", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc714", 714, "买卖股票的最佳时机含手续费", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc518", 518, "零钱兑换II", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc115", 115, "不同的子序列", "DP 进阶", "p13_dp_advanced_ii", "H", "advanced"),
    Problem("lc97", 97, "交错字符串", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc1035", 1035, "不相交的线", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc516", 516, "最长回文子序列", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc1039", 1039, "多边形三角剖分的最低得分", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc646", 646, "最长数对链", "DP 进阶", "p13_dp_advanced_ii", "M", "advanced"),
    Problem("lc10", 10, "正则表达式匹配", "DP 进阶", "p13_dp_advanced_ii", "H", "advanced"),
    Problem("lc44", 44, "通配符匹配", "DP 进阶", "p13_dp_advanced_ii", "H", "advanced"),
    # 14 贪心 · 进阶
    Problem("lc56", 56, "合并区间", "贪心", "p14_greedy_ii", "M", "advanced"),
    Problem("lc57", 57, "插入区间", "贪心", "p14_greedy_ii", "M", "advanced"),
    Problem("lc452", 452, "用最少数量的箭引爆气球", "贪心", "p14_greedy_ii", "M", "advanced"),
    Problem("lc861", 861, "翻转矩阵后的得分", "贪心", "p14_greedy_ii", "M", "advanced"),
    Problem("lc1046", 1046, "最后一块石头的重量", "贪心", "p14_greedy_ii", "E", "advanced"),
    Problem("lc135", 135, "分发糖果", "贪心", "p14_greedy_ii", "H", "advanced"),
    Problem("lc402", 402, "移掉K位数字", "贪心", "p14_greedy_ii", "M", "advanced"),
    Problem("lc968", 968, "监控二叉树", "贪心", "p14_greedy_ii", "H", "advanced"),
    # 15 堆 / Top-K · 进阶
    Problem("lc373", 373, "查找和最小的K对数字", "堆/Top-K", "p15_heap_topk_ii", "M", "advanced"),
    Problem("lc378", 378, "有序矩阵中第K小的元素", "堆/Top-K", "p15_heap_topk_ii", "M", "advanced"),
    Problem("lc502", 502, "IPO", "堆/Top-K", "p15_heap_topk_ii", "H", "advanced"),
    Problem("lc767", 767, "重构字符串", "堆/Top-K", "p15_heap_topk_ii", "M", "advanced"),
    Problem("lc621", 621, "任务调度器", "堆/Top-K", "p15_heap_topk_ii", "M", "advanced"),
    Problem("lc1834", 1834, "单线程CPU", "堆/Top-K", "p15_heap_topk_ii", "M", "advanced"),
    Problem("lc1642", 1642, "可以到达的最远建筑", "堆/Top-K", "p15_heap_topk_ii", "M", "advanced"),
    Problem("lc703", 703, "数据流中的第K大元素", "堆/Top-K", "p15_heap_topk_ii", "E", "advanced"),
    # 16 单调栈/单调队列 · 进阶
    Problem("lc503", 503, "下一个更大元素II", "单调栈/单调队列", "p16_monotonic_stack_ii", "M", "advanced"),
    Problem("lc901", 901, "股票价格跨度", "单调栈/单调队列", "p16_monotonic_stack_ii", "M", "advanced"),
    Problem("lc456", 456, "132模式", "单调栈/单调队列", "p16_monotonic_stack_ii", "M", "advanced"),
    Problem("lc85", 85, "最大矩形", "单调栈/单调队列", "p16_monotonic_stack_ii", "H", "advanced"),
    Problem("lc1856", 1856, "子数组最小乘积的最大值", "单调栈/单调队列", "p16_monotonic_stack_ii", "M", "advanced"),
    Problem("lc1475", 1475, "商品折扣后的最终价格", "单调栈/单调队列", "p16_monotonic_stack_ii", "E", "advanced"),
    Problem("lc1673", 1673, "找出最具竞争力的子序列", "单调栈/单调队列", "p16_monotonic_stack_ii", "M", "advanced"),
    # 17 并查集 · 进阶
    Problem("lc685", 685, "冗余连接II", "并查集", "p17_union_find_ii", "H", "advanced"),
    Problem("lc990", 990, "等式方程的可满足性", "并查集", "p17_union_find_ii", "M", "advanced"),
    Problem("lc1061", 1061, "按字典序排列最小的等效字符串", "并查集", "p17_union_find_ii", "M", "advanced"),
    Problem("lc947", 947, "移除最多的同行或同列石头", "并查集", "p17_union_find_ii", "M", "advanced"),
    Problem("lc839", 839, "相似字符串组", "并查集", "p17_union_find_ii", "H", "advanced"),
    Problem("lc1101", 1101, "彼此熟识的最早时间", "并查集", "p17_union_find_ii", "M", "advanced"),
    # 18 前缀和与差分 · 进阶
    Problem("lc304", 304, "二维区域和检索-矩阵不可变", "前缀和与差分", "p18_prefix_sum_ii", "M", "advanced"),
    Problem("lc528", 528, "按权重随机选择", "前缀和与差分", "p18_prefix_sum_ii", "M", "advanced"),
    Problem("lc974", 974, "和可被K整除的子数组", "前缀和与差分", "p18_prefix_sum_ii", "M", "advanced"),
    Problem("lc523", 523, "连续的子数组和", "前缀和与差分", "p18_prefix_sum_ii", "M", "advanced"),
    Problem("lc1094", 1094, "拼车", "前缀和与差分", "p18_prefix_sum_ii", "M", "advanced"),
    Problem("lc1893", 1893, "检查是否区域内所有整数都被覆盖", "前缀和与差分", "p18_prefix_sum_ii", "E", "advanced"),
    Problem("lc1732", 1732, "找到最高海拔", "前缀和与差分", "p18_prefix_sum_ii", "E", "advanced"),
    # 19 位运算与数学 · 进阶
    Problem("lc137", 137, "只出现一次的数字II", "位运算与数学", "p19_bit_math_ii", "M", "advanced"),
    Problem("lc260", 260, "只出现一次的数字III", "位运算与数学", "p19_bit_math_ii", "M", "advanced"),
    Problem("lc190", 190, "颠倒二进制位", "位运算与数学", "p19_bit_math_ii", "E", "advanced"),
    Problem("lc371", 371, "两整数之和", "位运算与数学", "p19_bit_math_ii", "M", "advanced"),
    Problem("lc172", 172, "阶乘后的零", "位运算与数学", "p19_bit_math_ii", "M", "advanced"),
    Problem("lc29", 29, "两数相除", "位运算与数学", "p19_bit_math_ii", "M", "advanced"),
    Problem("lc231", 231, "2的幂", "位运算与数学", "p19_bit_math_ii", "E", "advanced"),
    # 20 Trie 前缀树（新分类）
    Problem("lc208", 208, "实现 Trie (前缀树)", "Trie 前缀树", "p20_trie", "M", "advanced"),
    Problem("lc211", 211, "添加与搜索单词-数据结构设计", "Trie 前缀树", "p20_trie", "M", "advanced"),
    Problem("lc212", 212, "单词搜索II", "Trie 前缀树", "p20_trie", "H", "advanced"),
    Problem("lc677", 677, "键值映射", "Trie 前缀树", "p20_trie", "M", "advanced"),
    Problem("lc1268", 1268, "搜索推荐系统", "Trie 前缀树", "p20_trie", "M", "advanced"),
    # 21 设计题（新分类）
    Problem("lc146", 146, "LRU 缓存", "设计题", "p21_design", "M", "advanced"),
    Problem("lc460", 460, "LFU 缓存", "设计题", "p21_design", "H", "advanced"),
    Problem("lc355", 355, "设计推特", "设计题", "p21_design", "M", "advanced"),
    Problem("lc1146", 1146, "快照数组", "设计题", "p21_design", "M", "advanced"),
    Problem("lc622", 622, "设计循环队列", "设计题", "p21_design", "M", "advanced"),
    Problem("lc705", 705, "设计哈希集合", "设计题", "p21_design", "E", "advanced"),
    # 22 高级图论（新分类）
    Problem("lc743", 743, "网络延迟时间", "高级图论", "p22_advanced_graph", "M", "advanced"),
    Problem("lc787", 787, "K站中转内最便宜的航班", "高级图论", "p22_advanced_graph", "M", "advanced"),
    Problem("lc1584", 1584, "连接所有点的最小费用", "高级图论", "p22_advanced_graph", "M", "advanced"),
    Problem("lc785", 785, "判断二分图", "高级图论", "p22_advanced_graph", "M", "advanced"),
    Problem("lc1319", 1319, "连通网络的操作次数", "高级图论", "p22_advanced_graph", "M", "advanced"),
    Problem("lc802", 802, "找到最终的安全状态", "高级图论", "p22_advanced_graph", "M", "advanced"),
    Problem("lc1976", 1976, "到达目的地的方案数", "高级图论", "p22_advanced_graph", "M", "advanced"),
    # 23 矩阵与模拟（新分类）
    Problem("lc48", 48, "旋转图像", "矩阵与模拟", "p23_matrix_simulation", "M", "advanced"),
    Problem("lc54", 54, "螺旋矩阵", "矩阵与模拟", "p23_matrix_simulation", "M", "advanced"),
    Problem("lc73", 73, "矩阵置零", "矩阵与模拟", "p23_matrix_simulation", "M", "advanced"),
    Problem("lc289", 289, "生命游戏", "矩阵与模拟", "p23_matrix_simulation", "M", "advanced"),
    Problem("lc59", 59, "螺旋矩阵II", "矩阵与模拟", "p23_matrix_simulation", "M", "advanced"),
    Problem("lc36", 36, "有效的数独", "矩阵与模拟", "p23_matrix_simulation", "M", "advanced"),
    # 24 高级字符串算法（新分类）
    Problem("lc28", 28, "找出字符串中第一个匹配项的下标(KMP)", "高级字符串算法", "p24_advanced_strings", "M", "advanced"),
    Problem("lc459", 459, "重复的子字符串模式", "高级字符串算法", "p24_advanced_strings", "E", "advanced"),
    Problem("lc647", 647, "回文子串", "高级字符串算法", "p24_advanced_strings", "M", "advanced"),
    Problem("lc1044", 1044, "最长重复子串", "高级字符串算法", "p24_advanced_strings", "H", "advanced"),
    # 25 并发（新分类，可选层）
    Problem("lc1114", 1114, "按序打印", "并发", "p25_concurrency", "E", "advanced"),
    Problem("lc1115", 1115, "交替打印FooBar", "并发", "p25_concurrency", "M", "advanced"),
    Problem("lc1195", 1195, "交替打印字符串", "并发", "p25_concurrency", "M", "advanced"),

    # ══════════════════ Phase 3：竞赛级/Frontier Lab 补充（frontier tier）══════════════════
    # 01 数组与双指针 · 竞赛级
    Problem("lc665f", 665, "非递减数列", "数组与双指针", "p01_arrays_two_pointers_iii", "M", "frontier"),
    Problem("lc581f", 581, "最短无序连续子数组", "数组与双指针", "p01_arrays_two_pointers_iii", "M", "frontier"),
    Problem("lc448f", 448, "找到所有数组中消失的数字", "数组与双指针", "p01_arrays_two_pointers_iii", "E", "frontier"),
    Problem("lc769f", 769, "最多能完成排序的块", "数组与双指针", "p01_arrays_two_pointers_iii", "M", "frontier"),
    Problem("lc768f", 768, "最多能完成排序的块II", "数组与双指针", "p01_arrays_two_pointers_iii", "H", "frontier"),
    Problem("lc826f", 826, "安排工作以达到最大收益", "数组与双指针", "p01_arrays_two_pointers_iii", "M", "frontier"),
    Problem("lc986f", 986, "区间列表的交集", "数组与双指针", "p01_arrays_two_pointers_iii", "M", "frontier"),
    Problem("lc925f", 925, "长按键入", "数组与双指针", "p01_arrays_two_pointers_iii", "E", "frontier"),
    # 02 滑动窗口 · 竞赛级
    Problem("lc1052f", 1052, "爱生气的书店老板", "滑动窗口", "p02_sliding_window_iii", "M", "frontier"),
    Problem("lc1423f", 1423, "可获得的最大点数", "滑动窗口", "p02_sliding_window_iii", "M", "frontier"),
    Problem("lc1652f", 1652, "拆炸弹", "滑动窗口", "p02_sliding_window_iii", "E", "frontier"),
    Problem("lc1839f", 1839, "所有元音按顺序排布的最长子字符串", "滑动窗口", "p02_sliding_window_iii", "M", "frontier"),
    Problem("lc795f", 795, "区间子数组个数", "滑动窗口", "p02_sliding_window_iii", "M", "frontier"),
    Problem("lc187f", 187, "重复的DNA序列", "滑动窗口", "p02_sliding_window_iii", "M", "frontier"),
    # 03 哈希表 · 竞赛级
    Problem("lc447f", 447, "回旋镖的数量", "哈希表", "p03_hashing_iii", "E", "frontier"),
    Problem("lc1002f", 1002, "查找共用字符", "哈希表", "p03_hashing_iii", "E", "frontier"),
    Problem("lc1207f", 1207, "独一无二的出现次数", "哈希表", "p03_hashing_iii", "E", "frontier"),
    Problem("lc953f", 953, "验证外星语词典", "哈希表", "p03_hashing_iii", "E", "frontier"),
    Problem("lc274f", 274, "H指数", "哈希表", "p03_hashing_iii", "M", "frontier"),
    Problem("lc525f", 525, "连续数组", "哈希表", "p03_hashing_iii", "M", "frontier"),
    Problem("lc220f", 220, "存在重复元素III", "哈希表", "p03_hashing_iii", "H", "frontier"),
    # 04 字符串 · 竞赛级
    Problem("lc583f", 583, "两个字符串的删除操作", "字符串", "p04_strings_iii", "M", "frontier"),
    Problem("lc686f", 686, "重复叠加字符串匹配", "字符串", "p04_strings_iii", "M", "frontier"),
    Problem("lc819f", 819, "最常见的单词", "字符串", "p04_strings_iii", "E", "frontier"),
    Problem("lc481f", 481, "神奇字符串", "字符串", "p04_strings_iii", "M", "frontier"),
    Problem("lc809f", 809, "情感丰富的文字", "字符串", "p04_strings_iii", "M", "frontier"),
    Problem("lc848f", 848, "字母移位", "字符串", "p04_strings_iii", "M", "frontier"),
    Problem("lc1417f", 1417, "重新格式化字符串", "字符串", "p04_strings_iii", "E", "frontier"),
    # 05 链表 · 竞赛级
    Problem("lc61f", 61, "旋转链表", "链表", "p05_linked_list_iii", "M", "frontier"),
    Problem("lc86f", 86, "分隔链表", "链表", "p05_linked_list_iii", "M", "frontier"),
    Problem("lc82f", 82, "删除排序链表中的重复元素II", "链表", "p05_linked_list_iii", "M", "frontier"),
    Problem("lc83f", 83, "删除排序链表中的重复元素", "链表", "p05_linked_list_iii", "E", "frontier"),
    Problem("lc445f", 445, "两数相加II", "链表", "p05_linked_list_iii", "M", "frontier"),
    Problem("lc234f", 234, "回文链表", "链表", "p05_linked_list_iii", "E", "frontier"),
    Problem("lc1721f", 1721, "交换链表中的节点", "链表", "p05_linked_list_iii", "M", "frontier"),
    # 06 栈与队列 · 竞赛级
    Problem("lc1544f", 1544, "整理字符串", "栈与队列", "p06_stack_queue_iii", "E", "frontier"),
    Problem("lc856f", 856, "括号的分数", "栈与队列", "p06_stack_queue_iii", "M", "frontier"),
    Problem("lc1963f", 1963, "使字符串平衡的最小交换次数", "栈与队列", "p06_stack_queue_iii", "M", "frontier"),
    Problem("lc921f", 921, "使括号有效的最少添加", "栈与队列", "p06_stack_queue_iii", "M", "frontier"),
    Problem("lc1190f", 1190, "反转每对括号间的子串", "栈与队列", "p06_stack_queue_iii", "M", "frontier"),
    Problem("lc1614f", 1614, "括号的最大嵌套深度", "栈与队列", "p06_stack_queue_iii", "E", "frontier"),
    Problem("lc32f", 32, "最长有效括号", "栈与队列", "p06_stack_queue_iii", "H", "frontier"),
    # 07 二分查找 · 竞赛级
    Problem("lc81f", 81, "搜索旋转排序数组II", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc540f", 540, "有序数组中的单一元素", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc275f", 275, "H指数II", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc1898f", 1898, "可移除字符的最大数目", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc1300f", 1300, "转变数组后最接近目标值的数组和", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc1482f", 1482, "制作m束花所需的最少天数", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc1552f", 1552, "两球之间的磁力", "二分查找", "p07_binary_search_iii", "M", "frontier"),
    Problem("lc719f", 719, "找出第K小的数对距离", "二分查找", "p07_binary_search_iii", "H", "frontier"),
    # 08 二叉树遍历与构造 · 竞赛级
    Problem("lc100f", 100, "相同的树", "二叉树遍历与构造", "p08_binary_tree_iii", "E", "frontier"),
    Problem("lc129f", 129, "求根节点到叶节点数字之和", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc116f", 116, "填充每个节点的下一个右侧节点指针", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc117f", 117, "填充每个节点的下一个右侧节点指针II", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc863f", 863, "二叉树中所有距离为K的结点", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc1123f", 1123, "最深叶节点的最近公共祖先", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc998f", 998, "最大二叉树II", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc1372f", 1372, "二叉树中的最长交错路径", "二叉树遍历与构造", "p08_binary_tree_iii", "M", "frontier"),
    Problem("lc987f", 987, "二叉树的垂序遍历", "二叉树遍历与构造", "p08_binary_tree_iii", "H", "frontier"),
    # 09 二叉搜索树 · 竞赛级
    Problem("lc653f", 653, "两数之和IV-输入BST", "二叉搜索树", "p09_bst_iii", "E", "frontier"),
    Problem("lc501f", 501, "二叉搜索树中的众数", "二叉搜索树", "p09_bst_iii", "E", "frontier"),
    Problem("lc530f", 530, "二叉搜索树的最小绝对差", "二叉搜索树", "p09_bst_iii", "E", "frontier"),
    Problem("lc669f", 669, "修剪二叉搜索树", "二叉搜索树", "p09_bst_iii", "M", "frontier"),
    Problem("lc1008f", 1008, "前序遍历构造二叉搜索树", "二叉搜索树", "p09_bst_iii", "M", "frontier"),
    Problem("lc1305f", 1305, "两棵二叉搜索树中的所有元素", "二叉搜索树", "p09_bst_iii", "M", "frontier"),
    Problem("lc1382f", 1382, "将二叉搜索树变平衡", "二叉搜索树", "p09_bst_iii", "M", "frontier"),
    # 10 回溯 · 竞赛级
    Problem("lc77f", 77, "组合", "回溯", "p10_backtracking_iii", "M", "frontier"),
    Problem("lc1079f", 1079, "活字印刷", "回溯", "p10_backtracking_iii", "M", "frontier"),
    Problem("lc967f", 967, "连续差相同的数字", "回溯", "p10_backtracking_iii", "M", "frontier"),
    Problem("lc401f", 401, "二进制手表", "回溯", "p10_backtracking_iii", "E", "frontier"),
    Problem("lc1219f", 1219, "黄金矿工", "回溯", "p10_backtracking_iii", "M", "frontier"),
    Problem("lc87f", 87, "扰乱字符串", "回溯", "p10_backtracking_iii", "H", "frontier"),
    Problem("lc473f", 473, "火柴拼正方形", "回溯", "p10_backtracking_iii", "M", "frontier"),
    Problem("lc1863f", 1863, "找出所有子集的异或总和再求和", "回溯", "p10_backtracking_iii", "E", "frontier"),
    # 11 图 BFS/DFS · 竞赛级
    Problem("lc1129f", 1129, "颜色交替的最短路径", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "M", "frontier"),
    Problem("lc924f", 924, "尽量减少恶意软件的传播", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "M", "frontier"),
    Problem("lc1042f", 1042, "不邻接植花", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "M", "frontier"),
    Problem("lc1631f", 1631, "最小体力消耗路径", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "M", "frontier"),
    Problem("lc1926f", 1926, "迷宫中离入口最近的出口", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "M", "frontier"),
    Problem("lc1293f", 1293, "网格中的最短路径-有障碍物消除", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "H", "frontier"),
    Problem("lc864f", 864, "获取所有钥匙的最短路径", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "H", "frontier"),
    Problem("lc1263f", 1263, "推箱子", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "H", "frontier"),
    Problem("lc1298f", 1298, "从盒子里获得的最大糖果数", "图 BFS/DFS", "p11_graph_bfs_dfs_iii", "H", "frontier"),
    # 12 DP 基础/序列 · 竞赛级
    Problem("lc1289f", 1289, "下降路径最小和II", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc132f", 132, "分割回文串II", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc1553f", 1553, "吃掉N个橘子的最少天数", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc1420f", 1420, "生成数组", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc940f", 940, "不同的子序列II", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc887f", 887, "鸡蛋掉落", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc1751f", 1751, "最多可以参加的会议数目II", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc1937f", 1937, "扣分后的最大得分", "DP 基础/序列", "p12_dp_basics_iii", "M", "frontier"),
    Problem("lc1531f", 1531, "压缩字符串II", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc1187f", 1187, "使数组严格递增", "DP 基础/序列", "p12_dp_basics_iii", "H", "frontier"),
    Problem("lc1140f", 1140, "石子游戏II", "DP 基础/序列", "p12_dp_basics_iii", "M", "frontier"),
    # 13 DP 进阶 · 竞赛级
    Problem("lc174f", 174, "地下城游戏", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc1691f", 1691, "堆叠长方体的最大高度", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc1770f", 1770, "执行乘法运算的最大分数", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc730f", 730, "统计不同回文子序列", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc546f", 546, "移除盒子", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc902f", 902, "最大为N的数字组合", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc1074f", 1074, "元素和为目标值的子矩阵数量", "DP 进阶", "p13_dp_advanced_iii", "M", "frontier"),
    Problem("lc828f", 828, "统计子串中的唯一字符", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc1000f", 1000, "合并石头的最低成本", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    Problem("lc1547f", 1547, "切棍子的最小成本", "DP 进阶", "p13_dp_advanced_iii", "H", "frontier"),
    # 14 贪心 · 竞赛级
    Problem("lc881f", 881, "救生艇", "贪心", "p14_greedy_iii", "M", "frontier"),
    Problem("lc948f", 948, "令牌放置", "贪心", "p14_greedy_iii", "M", "frontier"),
    Problem("lc1005f", 1005, "K次取反后最大化的数组和", "贪心", "p14_greedy_iii", "E", "frontier"),
    Problem("lc870f", 870, "优势洗牌", "贪心", "p14_greedy_iii", "M", "frontier"),
    Problem("lc1029f", 1029, "两地调度", "贪心", "p14_greedy_iii", "M", "frontier"),
    Problem("lc1996f", 1996, "游戏中弱角色的数量", "贪心", "p14_greedy_iii", "M", "frontier"),
    Problem("lc630f", 630, "课程表III", "贪心", "p14_greedy_iii", "H", "frontier"),
    # 15 堆/Top-K · 竞赛级
    Problem("lc407f", 407, "接雨水II", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    Problem("lc218f", 218, "天际线问题", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    Problem("lc1439f", 1439, "有序矩阵中的第k个最小数组和", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    Problem("lc1675f", 1675, "数组的最小偏移量", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    Problem("lc358f", 358, "K距离间隔重排字符串", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    Problem("lc857f", 857, "雇佣K名工人的最低成本", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    Problem("lc480f", 480, "滑动窗口中位数", "堆/Top-K", "p15_heap_topk_iii", "H", "frontier"),
    # 16 单调栈/单调队列 · 竞赛级
    Problem("lc321f", 321, "拼接最大数", "单调栈/单调队列", "p16_monotonic_stack_iii", "H", "frontier"),
    Problem("lc962f", 962, "最大宽度坡", "单调栈/单调队列", "p16_monotonic_stack_iii", "M", "frontier"),
    Problem("lc1124f", 1124, "表现良好的最长时间段", "单调栈/单调队列", "p16_monotonic_stack_iii", "M", "frontier"),
    Problem("lc1499f", 1499, "满足不等式的最大值", "单调栈/单调队列", "p16_monotonic_stack_iii", "H", "frontier"),
    Problem("lc907f", 907, "子数组最小值之和", "单调栈/单调队列", "p16_monotonic_stack_iii", "M", "frontier"),
    # 17 并查集 · 竞赛级
    Problem("lc765f", 765, "情侣牵手", "并查集", "p17_union_find_iii", "H", "frontier"),
    Problem("lc1202f", 1202, "交换字符串中的元素", "并查集", "p17_union_find_iii", "M", "frontier"),
    Problem("lc1258f", 1258, "近义词句子", "并查集", "p17_union_find_iii", "M", "frontier"),
    Problem("lc1489f", 1489, "找到最小生成树里的关键边和伪关键边", "并查集", "p17_union_find_iii", "H", "frontier"),
    Problem("lc1697f", 1697, "检查是否存在有效路径-带边长限制", "并查集", "p17_union_find_iii", "H", "frontier"),
    # 18 前缀和与差分 · 竞赛级
    Problem("lc1521f", 1521, "找出最接近目标值的函数值", "前缀和与差分", "p18_prefix_sum_iii", "H", "frontier"),
    Problem("lc1685f", 1685, "有序数组中差绝对值之和", "前缀和与差分", "p18_prefix_sum_iii", "M", "frontier"),
    Problem("lc1310f", 1310, "子数组异或查询", "前缀和与差分", "p18_prefix_sum_iii", "M", "frontier"),
    Problem("lc1546f", 1546, "和为目标值的最大数目不重叠非空子数组数目", "前缀和与差分", "p18_prefix_sum_iii", "M", "frontier"),
    Problem("lc1477f", 1477, "找两个和为目标值且不重叠的子数组", "前缀和与差分", "p18_prefix_sum_iii", "M", "frontier"),
    # 19 位运算与数学 · 竞赛级
    Problem("lc201f", 201, "数字范围按位与", "位运算与数学", "p19_bit_math_iii", "M", "frontier"),
    Problem("lc393f", 393, "UTF-8编码验证", "位运算与数学", "p19_bit_math_iii", "M", "frontier"),
    Problem("lc1178f", 1178, "猜字谜", "位运算与数学", "p19_bit_math_iii", "H", "frontier"),
    Problem("lc1611f", 1611, "使数字变为0的最少操作次数", "位运算与数学", "p19_bit_math_iii", "H", "frontier"),
    Problem("lc782f", 782, "变为棋盘", "位运算与数学", "p19_bit_math_iii", "H", "frontier"),
    # 20 Trie 前缀树 · 竞赛级
    Problem("lc140f", 140, "单词拆分II", "Trie 前缀树", "p20_trie_iii", "H", "frontier"),
    Problem("lc472f", 472, "连接词", "Trie 前缀树", "p20_trie_iii", "H", "frontier"),
    Problem("lc425f", 425, "单词方块", "Trie 前缀树", "p20_trie_iii", "H", "frontier"),
    Problem("lc588f", 588, "设计内存文件系统", "Trie 前缀树", "p20_trie_iii", "H", "frontier"),
    Problem("lc648f", 648, "单词替换", "Trie 前缀树", "p20_trie_iii", "M", "frontier"),
    # 21 设计题 · 竞赛级
    Problem("lc432f", 432, "全O(1)的数据结构", "设计题", "p21_design_iii", "H", "frontier"),
    Problem("lc895f", 895, "最大频率栈", "设计题", "p21_design_iii", "H", "frontier"),
    Problem("lc1206f", 1206, "设计跳表", "设计题", "p21_design_iii", "H", "frontier"),
    Problem("lc729f", 729, "我的日程安排表I", "设计题", "p21_design_iii", "M", "frontier"),
    # 22 高级图论 · 竞赛级
    Problem("lc1928f", 1928, "规定时间内到达终点的最小花费", "高级图论", "p22_advanced_graph_iii", "H", "frontier"),
    Problem("lc1786f", 1786, "从第一个节点出发到最后一个节点的受限路径数", "高级图论", "p22_advanced_graph_iii", "M", "frontier"),
    Problem("lc882f", 882, "细分图中的可到达节点", "高级图论", "p22_advanced_graph_iii", "H", "frontier"),
    Problem("lc1210f", 1210, "穿过迷宫的最少移动次数", "高级图论", "p22_advanced_graph_iii", "H", "frontier"),
    Problem("lc1368f", 1368, "使网格图至少有一条有效路径的最小代价", "高级图论", "p22_advanced_graph_iii", "M", "frontier"),
    Problem("lc2045f", 2045, "到达目的地的第二短时间", "高级图论", "p22_advanced_graph_iii", "M", "frontier"),
    # 23 矩阵与模拟 · 竞赛级
    Problem("lc885f", 885, "螺旋矩阵III", "矩阵与模拟", "p23_matrix_simulation_iii", "M", "frontier"),
    Problem("lc1329f", 1329, "将矩阵按对角线排序", "矩阵与模拟", "p23_matrix_simulation_iii", "M", "frontier"),
    Problem("lc1727f", 1727, "重新排列后的最大子矩阵", "矩阵与模拟", "p23_matrix_simulation_iii", "M", "frontier"),
    Problem("lc311f", 311, "稀疏矩阵的乘法", "矩阵与模拟", "p23_matrix_simulation_iii", "M", "frontier"),
    Problem("lc1275f", 1275, "找出井字棋的获胜者", "矩阵与模拟", "p23_matrix_simulation_iii", "E", "frontier"),
    # 24 高级字符串算法 · 竞赛级
    Problem("lc214f", 214, "最短回文串", "高级字符串算法", "p24_advanced_strings_iii", "H", "frontier"),
    Problem("lc1392f", 1392, "最长快乐前缀", "高级字符串算法", "p24_advanced_strings_iii", "H", "frontier"),
    Problem("lc1316f", 1316, "不同的循环子字符串数量", "高级字符串算法", "p24_advanced_strings_iii", "H", "frontier"),
    Problem("lc2223f", 2223, "字符串的Score和", "高级字符串算法", "p24_advanced_strings_iii", "H", "frontier"),
    # 25 并发 · 竞赛级（可选层）
    Problem("lc1116f", 1116, "打印零与奇偶数", "并发", "p25_concurrency_iii", "M", "frontier"),
    Problem("lc1226f", 1226, "哲学家进餐", "并发", "p25_concurrency_iii", "M", "frontier"),
    Problem("lc1117f", 1117, "H2O生成", "并发", "p25_concurrency_iii", "M", "frontier"),
    # 26 数论与数学进阶（新分类）
    Problem("lc1979g", 1979, "找出数组的最大公约数", "数论与数学进阶", "p26_number_theory", "E", "frontier"),
    Problem("lc780g", 780, "到达终点", "数论与数学进阶", "p26_number_theory", "H", "frontier"),
    Problem("lc204g", 204, "计数质数", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("lc372g", 372, "超级次方", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("lc1922g", 1922, "统计好数字", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("lc1015g", 1015, "可被K整除的最小整数", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("lc633g", 633, "平方数之和", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("lc829g", 829, "连续整数求和", "数论与数学进阶", "p26_number_theory", "H", "frontier"),
    Problem("custom-extgcd", -1, "扩展欧几里得算法 ext_gcd", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("custom-modinv", -2, "模逆元 mod_inverse", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("custom-crt", -3, "中国剩余定理 crt", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("custom-powmod", -4, "快速幂取模 pow_mod", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("custom-fibmat", -5, "矩阵快速幂求斐波那契 fib_matrix_pow", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    Problem("custom-comb", -6, "组合数取模 n_choose_r_mod", "数论与数学进阶", "p26_number_theory", "M", "frontier"),
    # 27 线段树与树状数组进阶（新分类）
    Problem("lc307g", 307, "区域和检索-数组可修改", "线段树与树状数组进阶", "p27_segment_tree_bit", "M", "frontier"),
    Problem("lc315g", 315, "计算右侧小于当前元素的个数", "线段树与树状数组进阶", "p27_segment_tree_bit", "H", "frontier"),
    Problem("lc327g", 327, "区间和的个数", "线段树与树状数组进阶", "p27_segment_tree_bit", "H", "frontier"),
    Problem("lc493g", 493, "翻转对", "线段树与树状数组进阶", "p27_segment_tree_bit", "H", "frontier"),
    Problem("lc1649g", 1649, "通过指令创建有序数组", "线段树与树状数组进阶", "p27_segment_tree_bit", "H", "frontier"),
    Problem("lc850g", 850, "矩形面积II", "线段树与树状数组进阶", "p27_segment_tree_bit", "H", "frontier"),
    Problem("lc715g", 715, "Range模块", "线段树与树状数组进阶", "p27_segment_tree_bit", "H", "frontier"),
    Problem("custom-fenwick", -7, "树状数组 FenwickTree", "线段树与树状数组进阶", "p27_segment_tree_bit", "M", "frontier"),
    Problem("custom-segtree", -8, "线段树 SegmentTree(区间加+区间和,懒标记)", "线段树与树状数组进阶", "p27_segment_tree_bit", "M", "frontier"),
    Problem("custom-segtreemax", -9, "线段树 SegmentTreeMax(单点更新+区间最值)", "线段树与树状数组进阶", "p27_segment_tree_bit", "M", "frontier"),
    # 28 字符串匹配进阶（新分类）
    Problem("lc796g", 796, "旋转字符串", "字符串匹配进阶", "p28_string_matching_advanced", "E", "frontier"),
    Problem("lc1668g", 1668, "最大重复子字符串", "字符串匹配进阶", "p28_string_matching_advanced", "E", "frontier"),
    Problem("lc1092g", 1092, "最短公共超序列", "字符串匹配进阶", "p28_string_matching_advanced", "H", "frontier"),
    Problem("custom-zfunc", -10, "Z函数 z_function", "字符串匹配进阶", "p28_string_matching_advanced", "M", "frontier"),
    Problem("custom-zpattern", -11, "Z函数模式匹配 find_pattern_occurrences_z", "字符串匹配进阶", "p28_string_matching_advanced", "M", "frontier"),
    Problem("custom-ahocorasick", -12, "Aho-Corasick自动机", "字符串匹配进阶", "p28_string_matching_advanced", "H", "frontier"),
    Problem("custom-suffixarray", -13, "后缀数组 build_suffix_array", "字符串匹配进阶", "p28_string_matching_advanced", "M", "frontier"),
    Problem("custom-distinctsub", -14, "不同子串计数(对应LC1698思路)", "字符串匹配进阶", "p28_string_matching_advanced", "H", "frontier"),
    Problem("custom-lcsubstr", -15, "最长公共子串 longest_common_substring", "字符串匹配进阶", "p28_string_matching_advanced", "M", "frontier"),
    # 29 计算几何（新分类）
    Problem("lc812g", 812, "最大三角形面积", "计算几何", "p29_computational_geometry", "E", "frontier"),
    Problem("lc939g", 939, "最小面积矩形", "计算几何", "p29_computational_geometry", "M", "frontier"),
    Problem("lc963g", 963, "最小面积矩形II", "计算几何", "p29_computational_geometry", "M", "frontier"),
    Problem("lc1401g", 1401, "圆和矩形是否重叠", "计算几何", "p29_computational_geometry", "M", "frontier"),
    Problem("lc587g", 587, "安装栅栏", "计算几何", "p29_computational_geometry", "H", "frontier"),
    Problem("custom-cross", -16, "二维叉积 cross", "计算几何", "p29_computational_geometry", "E", "frontier"),
    Problem("custom-segintersect", -17, "线段相交 segments_intersect", "计算几何", "p29_computational_geometry", "M", "frontier"),
    Problem("custom-convexhull", -18, "凸包-Andrew单调链 convex_hull", "计算几何", "p29_computational_geometry", "H", "frontier"),
    Problem("custom-polygonarea", -19, "多边形面积-鞋带公式 polygon_area", "计算几何", "p29_computational_geometry", "M", "frontier"),
    # 30 高级图论II（新分类）
    Problem("lc1192g", 1192, "找出关键连接", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("lc1568g", 1568, "使陆地分离的最少天数", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("lc1483g", 1483, "树节点的第K个祖先", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("lc1615g", 1615, "最大网络秩", "高级图论II", "p30_advanced_graph_ii", "M", "frontier"),
    Problem("lc1345g", 1345, "跳跃游戏IV", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("lc1627g", 1627, "图连通性和阈值", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("lc2492g", 2492, "两个城市间路径的最小分数", "高级图论II", "p30_advanced_graph_ii", "M", "frontier"),
    Problem("custom-tarjanscc", -20, "Tarjan求强连通分量(SCC)", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("custom-tarjanbridge", -21, "Tarjan求桥(bridges)", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("custom-articulation", -22, "求割点(articulation points)", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("custom-lcabinary", -23, "LCA倍增法(Binary Lifting)", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    Problem("custom-maxflow", -24, "最大流(Edmonds-Karp)", "高级图论II", "p30_advanced_graph_ii", "H", "frontier"),
    # 31 位运算与状态压缩DP（新分类）
    Problem("lc464g", 464, "我能赢吗", "位运算与状态压缩DP", "p31_bitmask_dp", "M", "frontier"),
    Problem("lc698g", 698, "划分为k个相等的子集", "位运算与状态压缩DP", "p31_bitmask_dp", "M", "frontier"),
    Problem("lc847g", 847, "访问所有节点的最短路径", "位运算与状态压缩DP", "p31_bitmask_dp", "H", "frontier"),
    Problem("lc1125g", 1125, "最小的必要团队", "位运算与状态压缩DP", "p31_bitmask_dp", "H", "frontier"),
    Problem("lc1349g", 1349, "参加考试的最大学生数", "位运算与状态压缩DP", "p31_bitmask_dp", "H", "frontier"),
    Problem("lc1494g", 1494, "并行课程II", "位运算与状态压缩DP", "p31_bitmask_dp", "H", "frontier"),
    Problem("lc1655g", 1655, "分配重复整数", "位运算与状态压缩DP", "p31_bitmask_dp", "H", "frontier"),
    Problem("lc526g", 526, "优美的排列", "位运算与状态压缩DP", "p31_bitmask_dp", "M", "frontier"),
    Problem("lc1723g", 1723, "完成所有工作的最短时间", "位运算与状态压缩DP", "p31_bitmask_dp", "H", "frontier"),
    Problem("lc1986g", 1986, "完成任务的最少工作时间段", "位运算与状态压缩DP", "p31_bitmask_dp", "M", "frontier"),
    # 32 博弈论与组合数学（新分类）
    Problem("lc292g", 292, "Nim游戏", "博弈论与组合数学", "p32_game_theory_combinatorics", "E", "frontier"),
    Problem("lc294g", 294, "翻转游戏II", "博弈论与组合数学", "p32_game_theory_combinatorics", "M", "frontier"),
    Problem("lc375g", 375, "猜数字大小II", "博弈论与组合数学", "p32_game_theory_combinatorics", "M", "frontier"),
    Problem("lc877g", 877, "石子游戏", "博弈论与组合数学", "p32_game_theory_combinatorics", "M", "frontier"),
    Problem("lc913g", 913, "猫和老鼠", "博弈论与组合数学", "p32_game_theory_combinatorics", "H", "frontier"),
    Problem("lc486g", 486, "预测赢家", "博弈论与组合数学", "p32_game_theory_combinatorics", "M", "frontier"),
    Problem("lc1927g", 1927, "Sum Game", "博弈论与组合数学", "p32_game_theory_combinatorics", "M", "frontier"),
    Problem("lc1406g", 1406, "石子游戏III", "博弈论与组合数学", "p32_game_theory_combinatorics", "H", "frontier"),
    Problem("lc1690g", 1690, "石子游戏VII", "博弈论与组合数学", "p32_game_theory_combinatorics", "M", "frontier"),
    # 33 高级数据结构（新分类，981真实题 + Skiplist对应LC1206确定性变体custom + 6个自实现）
    Problem("lc981g", 981, "基于时间的键值存储", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
    Problem("custom-skiplist", -25, "跳表确定性变体 SimpleSkipList(对应LC1206)", "高级数据结构", "p33_advanced_data_structures", "H", "frontier"),
    Problem("custom-sparsetable", -26, "稀疏表 SparseTable(静态区间最值,O(1)查询)", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
    Problem("custom-undounionfind", -27, "可撤销并查集 UndoableUnionFind", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
    Problem("custom-monoqueue", -28, "手写单调队列 MonotonicQueue", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
    Problem("custom-sqrtdecomp", -29, "分块 SqrtDecomposition(区间加+区间和)", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
    Problem("custom-xortrie", -30, "01-字典树 XorTrie(最大异或对)", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
    Problem("custom-bloomfilter", -31, "布隆过滤器 SimpleBloomFilter(确定性哈希)", "高级数据结构", "p33_advanced_data_structures", "M", "frontier"),
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
    assert len(PROBLEMS) == 544, (
        f"应为 544 题(100 core + 198 advanced + 246 frontier[215真题+31自实现])，"
        f"实际 {len(PROBLEMS)}"
    )
    assert len(categories()) == 33, f"应为 33 类，实际 {len(categories())}"
    ids = [p.id for p in PROBLEMS]
    assert len(ids) == len(set(ids)), "存在重复 id"
    # leetcode_no 唯一性：真实题号互不相同；自实现算法用负数占位，同样各自唯一
    real_nos = [p.leetcode_no for p in PROBLEMS if p.leetcode_no > 0]
    assert len(real_nos) == len(set(real_nos)), "存在重复 LeetCode 编号"
    custom_nos = [p.leetcode_no for p in PROBLEMS if p.leetcode_no < 0]
    assert len(custom_nos) == len(set(custom_nos)), "自实现算法占位编号重复"
    for p in PROBLEMS:
        assert p.difficulty in ("E", "M", "H")
        assert p.tier in ("core", "advanced", "frontier")
        assert p.module.startswith("p") and "_" in p.module
    s = stats()
    assert s["E"] + s["M"] + s["H"] == 544
    core = [p for p in PROBLEMS if p.tier == "core"]
    advanced = [p for p in PROBLEMS if p.tier == "advanced"]
    frontier = [p for p in PROBLEMS if p.tier == "frontier"]
    assert len(core) == 100
    assert len(advanced) == 198
    assert len(frontier) == 246
    print(
        f"[PASS] catalog: 544 题元数据完整 (E={s['E']} M={s['M']} H={s['H']}, "
        f"33 类, core={len(core)} advanced={len(advanced)} frontier={len(frontier)})"
    )


if __name__ == "__main__":
    _self_test()
