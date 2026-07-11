"""15 个高频 LeetCode pattern，每个一道范例解 + 复杂度注释。

用法：认出题目属于哪个 pattern（骨架），再套模板。研究岗 coding 轮的"地板"覆盖面。
每个函数 docstring 标注：代表题、思路、时间/空间复杂度。
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque


# ── 1. 双指针 ────────────────────────────────────────────────────────────
def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int]:
    """代表题：Two Sum II（有序）。左右指针向中间夹逼。O(n) / O(1)。"""
    i, j = 0, len(nums) - 1
    while i < j:
        s = nums[i] + nums[j]
        if s == target:
            return (i, j)
        if s < target:
            i += 1
        else:
            j -= 1
    return (-1, -1)


# ── 2. 滑动窗口 ──────────────────────────────────────────────────────────
def longest_unique_substring(s: str) -> int:
    """代表题：无重复字符最长子串。右扩左缩，维护窗口内字符集。O(n) / O(k)。"""
    seen = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return best


# ── 3. 哈希 / 频次 ───────────────────────────────────────────────────────
def group_anagrams(words: list[str]) -> list[list[str]]:
    """代表题：字母异位词分组。用排序后字符串当 key。O(n·k log k)。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())


# ── 4. 二分查找 ──────────────────────────────────────────────────────────
def search_insert(nums: list[int], target: int) -> int:
    """代表题：搜索插入位置（lower_bound）。O(log n) / O(1)。"""
    lo, hi = 0, len(nums)                # 左闭右开
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ── 5. BFS（网格/图最短路） ──────────────────────────────────────────────
def shortest_grid_path(grid: list[list[int]]) -> int:
    """代表题：二值网格最短路（0 可走）。层序 BFS。O(R·C)。返回步数或 -1。"""
    if not grid or grid[0][0] == 1:
        return -1
    R, C = len(grid), len(grid[0])
    q = deque([(0, 0, 1)])
    seen = {(0, 0)}
    while q:
        r, c, d = q.popleft()
        if (r, c) == (R - 1, C - 1):
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1


# ── 6. DFS / 回溯 ────────────────────────────────────────────────────────
def subsets(nums: list[int]) -> list[list[int]]:
    """代表题：子集。回溯选/不选。O(n·2^n)。"""
    res: list[list[int]] = []

    def bt(start, path):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i + 1, path)
            path.pop()

    bt(0, [])
    return res


# ── 7. DP（1D） ──────────────────────────────────────────────────────────
def house_robber(nums: list[int]) -> int:
    """代表题：打家劫舍。dp[i]=max(dp[i-1], dp[i-2]+nums[i])。O(n) / O(1)。"""
    prev = prev2 = 0
    for x in nums:
        prev, prev2 = max(prev, prev2 + x), prev
    return prev


# ── 8. DP（2D / 网格） ───────────────────────────────────────────────────
def min_path_sum(grid: list[list[int]]) -> int:
    """代表题：最小路径和。dp[i][j]=grid+min(上,左)。O(R·C)。"""
    R, C = len(grid), len(grid[0])
    dp = [[0] * C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            best = 0
            if i and j:
                best = min(dp[i - 1][j], dp[i][j - 1])
            elif i:
                best = dp[i - 1][j]
            elif j:
                best = dp[i][j - 1]
            dp[i][j] = grid[i][j] + best
    return dp[-1][-1]


# ── 9. 堆 / Top-K ────────────────────────────────────────────────────────
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """代表题：前 K 高频元素。大小为 K 的最小堆。O(n log k)。"""
    freq = Counter(nums)
    return [x for x, _ in heapq.nlargest(k, freq.items(), key=lambda kv: kv[1])]


# ── 10. 区间 ─────────────────────────────────────────────────────────────
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """代表题：合并区间。按起点排序后线性合并。O(n log n)。"""
    intervals.sort()
    out: list[list[int]] = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


# ── 11. 链表（快慢指针） ─────────────────────────────────────────────────
def has_cycle(adj: dict[int, int], start: int) -> bool:
    """代表题：环形链表（Floyd 判圈）。adj[node]=next（或缺失=None）。O(n) / O(1)。"""
    slow = fast = start
    while fast in adj and adj[fast] in adj:
        slow = adj[slow]
        fast = adj[adj[fast]]
        if slow == fast:
            return True
    return False


# ── 12. 单调栈 / 括号 ────────────────────────────────────────────────────
def daily_temperatures(temps: list[int]) -> list[int]:
    """代表题：每日温度（下一个更大元素）。单调递减栈。O(n)。"""
    res = [0] * len(temps)
    stack: list[int] = []                # 存下标，温度递减
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res


# ── 13. 树（递归遍历） ───────────────────────────────────────────────────
def max_depth(tree: dict, root) -> int:
    """代表题：二叉树最大深度。tree[node]=(left,right)。O(n)。"""
    if root is None:
        return 0
    left, right = tree.get(root, (None, None))
    return 1 + max(max_depth(tree, left), max_depth(tree, right))


# ── 14. 并查集 ───────────────────────────────────────────────────────────
class UnionFind:
    """代表题：连通分量数 / 冗余连接。路径压缩 + 按秩合并。近 O(α(n))。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1


# ── 15. 贪心 ─────────────────────────────────────────────────────────────
def can_jump(nums: list[int]) -> bool:
    """代表题：跳跃游戏。维护能到达的最远下标。O(n) / O(1)。"""
    reach = 0
    for i, x in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + x)
    return True


PATTERNS = [
    ("双指针", "Two Sum II / 三数之和 / 盛水容器"),
    ("滑动窗口", "无重复最长子串 / 最小覆盖子串"),
    ("哈希频次", "字母异位词 / 两数之和 / 最长连续序列"),
    ("二分查找", "搜索插入 / 旋转数组 / 峰值 / 二分答案"),
    ("BFS", "网格最短路 / 腐烂橘子 / 单词接龙"),
    ("DFS回溯", "子集 / 全排列 / 组合总和 / N皇后"),
    ("DP-1D", "打家劫舍 / 爬楼梯 / 最长递增子序列"),
    ("DP-2D", "最小路径和 / 编辑距离 / 最长公共子序列"),
    ("堆TopK", "前K高频 / 合并K链表 / 数据流中位数"),
    ("区间", "合并区间 / 插入区间 / 会议室"),
    ("链表快慢", "环形链表 / 中点 / 相交 / 重排"),
    ("单调栈", "每日温度 / 柱状图最大矩形 / 括号匹配"),
    ("树递归", "最大深度 / 路径和 / 最近公共祖先"),
    ("并查集", "连通分量 / 冗余连接 / 账户合并"),
    ("贪心", "跳跃游戏 / 加油站 / 分发糖果"),
]


def _self_test() -> None:
    assert two_sum_sorted([2, 7, 11, 15], 9) == (0, 1)
    assert longest_unique_substring("abcabcbb") == 3
    assert len(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) == 3
    assert search_insert([1, 3, 5, 6], 5) == 2 and search_insert([1, 3, 5, 6], 2) == 1
    assert shortest_grid_path([[0, 0], [1, 0]]) == 3
    assert len(subsets([1, 2, 3])) == 8
    assert house_robber([2, 7, 9, 3, 1]) == 12
    assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert has_cycle({0: 1, 1: 2, 2: 0}, 0) is True
    assert has_cycle({0: 1, 1: 2}, 0) is False
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert max_depth({1: (2, 3), 2: (4, None)}, 1) == 3
    uf = UnionFind(5)
    uf.union(0, 1); uf.union(1, 2); uf.union(3, 4)
    assert uf.count == 2
    assert can_jump([2, 3, 1, 1, 4]) is True and can_jump([3, 2, 1, 0, 4]) is False
    assert len(PATTERNS) == 15
    print(f"[PASS] patterns: 15 pattern 范例解全部正确")


if __name__ == "__main__":
    _self_test()
