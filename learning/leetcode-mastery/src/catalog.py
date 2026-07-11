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
    tier: str = "core"  # "core"（Phase 1 起步 100 题）| "advanced"（Phase 2 进阶 198 题）


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
    assert len(PROBLEMS) == 298, f"应为 298 题(100 core + 198 advanced)，实际 {len(PROBLEMS)}"
    assert len(categories()) == 25, f"应为 25 类，实际 {len(categories())}"
    ids = [p.id for p in PROBLEMS]
    assert len(ids) == len(set(ids)), "存在重复 id"
    nos = [p.leetcode_no for p in PROBLEMS]
    assert len(nos) == len(set(nos)), "存在重复 LeetCode 编号"
    for p in PROBLEMS:
        assert p.difficulty in ("E", "M", "H")
        assert p.tier in ("core", "advanced")
        assert p.module.startswith("p") and "_" in p.module
    s = stats()
    assert s["E"] + s["M"] + s["H"] == 298
    core = [p for p in PROBLEMS if p.tier == "core"]
    advanced = [p for p in PROBLEMS if p.tier == "advanced"]
    assert len(core) == 100
    assert len(advanced) == 198
    print(
        f"[PASS] catalog: 298 题元数据完整 (E={s['E']} M={s['M']} H={s['H']}, "
        f"25 类, core={len(core)} advanced={len(advanced)})"
    )


if __name__ == "__main__":
    _self_test()
