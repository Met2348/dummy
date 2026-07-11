"""16 单调栈/单调队列 · 竞赛级补充(Part III)：单调栈与归并、前缀和、单调队列
优化结合的 Frontier Lab 面试难题，不重复 Part I/II 已经讲过的骨架，只挑
竞赛级变体。"""
from __future__ import annotations

from collections import deque


# ── LC321 拼接最大数(Hard) ────────────────────────────────────────────────
def _max_subsequence(nums: list[int], length: int) -> list[int]:
    """用单调栈从 nums 中选出长度为 length、保持相对顺序的"数值最大"子序列。"""
    if length == 0:
        return []
    stack: list[int] = []
    drop = len(nums) - length
    for x in nums:
        while stack and drop > 0 and stack[-1] < x:
            stack.pop()
            drop -= 1
        stack.append(x)
    return stack[:length]


def _greater(a: list[int], i: int, b: list[int], j: int) -> bool:
    """比较 a[i:] 和 b[j:] 的字典序，a[i:] 严格更大则返回 True。"""
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            return a[i] > b[j]
        i += 1
        j += 1
    return (len(a) - i) > (len(b) - j)


def _merge(a: list[int], b: list[int]) -> list[int]:
    """把两个数字子序列按"每步都取字典序更大的那个序列的当前首位"归并成一个
    最大数（不是普通的两路归并排序，而是"贪心比较剩余后缀"）。"""
    result: list[int] = []
    i = j = 0
    while i < len(a) or j < len(b):
        if _greater(a, i, b, j):
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result


def max_number(nums1: list[int], nums2: list[int], k: int) -> list[int]:
    """
    【题意】给定两个数组 nums1、nums2（分别代表两个数字的各位数字），以及
    整数 k，从两个数组中一共选出 k 个数字（可以只选其中一个数组的、也可以
    两边都选），必须保持各自数组内部的相对顺序，拼出的整体数值最大。返回这
    k 个数字组成的数组。
    【思路】把问题拆解成两步："单调栈选子序列" + "两个有序子序列的最大归并"。
    第一步：枚举从 nums1 里选 i 个、从 nums2 里选 k-i 个（i 的范围要同时满足
    `0 <= i <= len(nums1)` 且 `0 <= k-i <= len(nums2)`），对每种切分，用
    "单调栈保留最大子序列"的技巧（和 LC402 移掉K位数字同源：维护一个递减栈，
    当前数字比栈顶大且还有"删除名额"就弹栈）分别求出 nums1 的最大长度-i
    子序列和 nums2 的最大长度-(k-i) 子序列——这一步保证了"各自数组内部选出的
    部分"已经是这个长度下能选到的最大值。第二步：把这两个子序列**归并**成一个
    长度为 k 的结果——但这不是普通的两路归并排序，而是每一步都要比较"两个
    子序列从当前位置开始的剩余后缀，谁的字典序更大"，取字典序更大的那个序列的
    当前首位放到结果里（如果只比较当前首位数字大小，会在两个子序列当前首位
    数字相等时错误地"随便选一个"，必须往后看足够远才能确定哪边继续选下去
    整体更大）。对所有 i 的切分方式得到的候选结果取字典序最大的一个，就是
    全局最优解。
    【复杂度】设 m = len(nums1)，n = len(nums2)：单次"单调栈选子序列"是
    O(m) 或 O(n)；单次"归并两个长度为 i、k-i 的子序列"最坏是 O((i+(k-i))^2)
    = O(k^2)（因为比较剩余后缀最坏要扫到底）；枚举 i 有 O(min(k,m)+1) 种，
    整体最坏 O(k^2 * min(k, m))。
    【易错点】① 归并阶段最容易犯的错误是只比较 `a[i]` 和 `b[j]` 的大小，
    而不是比较两条剩余序列 `a[i:]` 和 `b[j:]` 的字典序——当 a[i] == b[j]
    时，选哪一边完全取决于"往后谁能给出更大的下一个数字"，必须继续比较后续
    元素直到分出大小（或某一边耗尽）；② 枚举切分 i 的范围必须同时裁剪两端，
    遗漏 `i <= min(k, m)` 或 `k - i <= n` 会在数组越界或者请求负长度子序列时
    出错。
    """
    m, n = len(nums1), len(nums2)
    best: list[int] | None = None
    for i in range(max(0, k - n), min(k, m) + 1):
        candidate = _merge(_max_subsequence(nums1, i), _max_subsequence(nums2, k - i))
        if best is None or candidate > best:
            best = candidate
    return best if best is not None else []


# ── LC962 最大宽度坡(Medium) ──────────────────────────────────────────────
def max_width_ramp(nums: list[int]) -> int:
    """
    【题意】数组 nums 中的"坡"是满足 i < j 且 nums[i] <= nums[j] 的下标对
    (i, j)，坡的宽度定义为 j - i。求数组中最大的坡宽度，不存在则返回 0。
    【思路】单调栈只从**左边**收集"有潜力当坡起点"的候选：从左到右扫描，只有
    当当前值比栈顶（对应的数值）更小时才入栈——这保证栈中数值从底到顶严格
    递减，因此栈里任意一个下标都可能是"比它右边某个更大或相等的值"构成坡的
    起点（如果一个数比它左边所有已扫描的数都大，它永远不可能作为坡的起点，
    因为拿它左边任意一个更早、更小的下标当起点，坡宽度都会更大，所以只有
    "创下新低"的下标才值得留在候选栈里）。然后从右到左扫描整个数组（作为坡的
    终点 j），只要栈顶下标对应的数值 <= nums[j]，就说明这一对 (栈顶, j) 构成
    一个合法的坡，弹出栈顶并更新最大宽度——这里要弹出而不是保留，因为对于
    "终点"逐渐左移的后续扫描，栈顶这个（更靠右的）候选起点不可能再产生比
    "当前这次用它"更宽的坡了（j 只会越来越小）。
    【复杂度】时间 O(n)（构建单调栈一次遍历 + 反向扫描消耗栈一次遍历，
    每个下标各入栈出栈至多一次），空间 O(n)。
    【易错点】① 反向扫描时弹栈条件是 `nums[stack[-1]] <= nums[j]`（非严格），
    题目定义坡允许 `nums[i] <= nums[j]` 相等的情况，写成严格小于会漏掉合法
    的坡；② 构建候选栈时如果写成"只要比栈顶大就不管"，会让栈丢失单调递减性
    从而漏掉一些本该被视为潜在起点的下标——必须是"严格更小才入栈"。
    """
    stack: list[int] = []
    for i, x in enumerate(nums):
        if not stack or nums[stack[-1]] > x:
            stack.append(i)
    best = 0
    for j in range(len(nums) - 1, -1, -1):
        while stack and nums[stack[-1]] <= nums[j]:
            best = max(best, j - stack.pop())
    return best


# ── LC1124 表现良好的最长时间段(Medium) ───────────────────────────────────
def longest_wpi(hours: list[int]) -> int:
    """
    【题意】hours 表示某员工每天的工作小时数。一天如果工作时长严格超过 8
    小时称为"劳累日"，否则称为"摸鱼日"。一段区间是"表现良好的时间段"当且仅当
    这段区间里"劳累日"的数量严格多于"摸鱼日"的数量。求最长的表现良好时间段
    长度。
    【思路】把"劳累日"记为 +1、"摸鱼日"记为 -1，问题转化为经典的"前缀和 +
    哈希表"母题：定义前缀和 s[i] = 前 i 天的 +1/-1 之和，区间 (j, i] 是表现
    良好的当且仅当 `s[i] - s[j] > 0`，即 `s[i] > s[j]`。扫描到第 i 天时分两种
    情况：① 如果当前前缀和 s[i] > 0，说明从第 0 天到第 i 天整体就是表现良好
    的，直接用 `i+1` 更新答案；② 如果 s[i] <= 0，需要找一个**最早**出现过的
    j 使得 s[j] = s[i] - 1（这样 `s[i] - s[j] = 1 > 0`，且因为只需要"严格
    大于"，找 s[i]-1 而不是任意更小的值就足够，同时因为前缀和每步只变化
    ±1，`s[i]-1` 是恰好比 s[i] 差 1 的最近可能值，能保证区间最长）——用哈希
    表记录每个前缀和**第一次出现**的下标（第一次出现才能让区间最长），查表
    即可 O(1) 得到 j。
    【复杂度】时间 O(n)（一次线性扫描，配合哈希表 O(1) 查询/插入），空间
    O(n)（哈希表最多存 n 个不同的前缀和取值）。
    【易错点】① 哈希表必须只记录"某个前缀和第一次出现的下标"，如果后面
    同样的前缀和又出现一次并覆盖了旧下标，会让查到的 j 变大，区间变短，
    答案偏小；② 容易在 s[i] <= 0 时忘记也要检查 `s[i] > 0` 单独成立的整体
    区间这一支——这一支不需要查哈希表（相当于 j = -1，此时"下标 -1"这个
    起点从未真正被记录进哈希表里）。
    """
    pos: dict[int, int] = {}
    s = 0
    best = 0
    for i, h in enumerate(hours):
        s += 1 if h > 8 else -1
        if s > 0:
            best = max(best, i + 1)
        elif s - 1 in pos:
            best = max(best, i - pos[s - 1])
        if s not in pos:
            pos[s] = i
    return best


# ── LC1499 满足不等式的最大值(Hard) ───────────────────────────────────────
def find_max_value_of_equation(points: list[list[int]], k: int) -> int:
    """
    【题意】给定按 x 坐标严格递增排序的点集 points[i] = [xi, yi]，以及整数
    k，求 `yi + yj + |xi - xj|`（其中 i < j 且 `xj - xi <= k`）的最大值
    （因为 x 递增，`|xi-xj| = xj - xi`）。
    【思路】把要最大化的式子拆成两个独立部分：`yi + yj + (xj - xi) =
    (yi - xi) + (xj + yj)`。当扫描到点 j 时，`xj + yj` 是已知的固定值，
    问题变成"在窗口 `xj - xi <= k` 的范围内，找最大的 `yi - xi`"——这正是
    "滑动窗口最大值"的单调队列母题（LC239 的变体，只是窗口条件从下标距离
    换成了 x 坐标距离）。维护一个单调递减的双端队列，队列里存 `(yi - xi, xi)`
    对：从队头弹出所有"x 坐标已经超出窗口"（`xj - 队头.xi > k`）的过期候选；
    此时队头就是窗口内 `yi - xi` 的最大值，用它加上当前的 `xj + yj` 更新
    答案；再把当前点的 `(yj - xj, xj)` 按照"从队尾弹出所有比它小的候选"
    的方式插入队尾，保持队列从队头到队尾单调递减——这样每个候选只有在被
    严格更大的新候选"淘汰"或者"滑出窗口"时才会离开队列，均摊下来每个点只
    入队出队一次。
    【复杂度】时间 O(n)（每个下标最多入队出队各一次），空间 O(n)。
    【易错点】① 必须先弹出过期的队头、更新答案之后，再把当前点插入队列——
    如果先插入再弹出，可能会把"刚插入的自己"当成窗口内的候选和自己配对
    （i、j 必须是两个不同的点）；② 队列里比较、维护单调性时要用 `yi - xi`
    这个组合值，而不是分别维护 xi、yi 两个单调队列——拆开维护无法保证"整体
    组合值"的单调性。
    """
    dq: deque[tuple[int, int]] = deque()  # (y - x, x)，从队头到队尾单调递减
    best = float("-inf")
    for x, y in points:
        while dq and x - dq[0][1] > k:
            dq.popleft()
        if dq:
            best = max(best, dq[0][0] + x + y)
        while dq and dq[-1][0] <= y - x:
            dq.pop()
        dq.append((y - x, x))
    return int(best)


# ── LC907 子数组最小值之和(Medium) ────────────────────────────────────────
def sum_subarray_mins(arr: list[int]) -> int:
    """
    【题意】给定整数数组 arr，对每一个连续子数组求最小值，再把所有子数组的
    最小值累加起来，结果对 1e9+7 取模。
    【思路】"贡献法 + 单调栈"：与其枚举每个子数组再求最小值（O(n^2) 个子
    数组），不如反过来枚举**每个元素**，计算它作为"某些子数组的最小值"总共
    贡献了多少次。对 arr[i]，它能充当最小值的子数组，左边界最远能扩展到
    "左边第一个严格更小的元素"之后一格，右边界最远能扩展到"右边第一个小于
    等于它的元素"之前一格——用两次单调栈分别预处理 left[i]（左边第一个更小
    元素的下标，不存在记 -1）和 right[i]（右边第一个"小于等于"它的元素的
    下标，不存在记 n）。这样以 arr[i] 为最小值的子数组恰好有
    `(i - left[i]) * (right[i] - i)` 个，把 `arr[i] * (i-left[i]) *
    (right[i]-i)` 累加到答案里，对所有 i 求和即为最终结果。
    【复杂度】时间 O(n)（两次单调栈预处理各一次线性扫描），空间 O(n)。
    【易错点】计算 left 和 right 时如果两边都用**严格小于**（或者都用
    **小于等于**）判断相同元素的归属边界，会在数组存在重复元素时把同一个
    子数组的最小值重复计数到两个不同的下标上——正确做法是左右两侧的比较
    符号必须"一边严格、一边不严格"（本实现左边界找"严格更小"、右边界找
    "小于等于"），这样每个子数组的最小值恰好被其中**最靠右**的那个最小值
    下标计数一次，不重不漏。
    """
    mod = 10**9 + 7
    n = len(arr)
    left = [0] * n
    stack: list[int] = []
    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        left[i] = i - (stack[-1] if stack else -1)
        stack.append(i)
    right = [0] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        right[i] = (stack[-1] if stack else n) - i
        stack.append(i)
    total = 0
    for i in range(n):
        total += arr[i] * left[i] * right[i]
    return total % mod


def _self_test() -> None:
    assert max_number([3, 4, 6, 5], [9, 1, 2, 5, 8, 3], 5) == [9, 8, 6, 5, 3]
    assert max_number([6, 7], [6, 0, 4], 5) == [6, 7, 6, 0, 4]
    assert max_number([3, 9], [8, 9], 3) == [9, 8, 9]

    assert max_width_ramp([6, 0, 8, 2, 1, 5]) == 4
    assert max_width_ramp([9, 8, 1, 0, 1, 9, 4, 0, 4, 1]) == 7

    assert longest_wpi([9, 9, 6, 0, 6, 6, 9]) == 3
    assert longest_wpi([6, 6, 6]) == 0

    assert find_max_value_of_equation([[1, 3], [2, 0], [5, 10], [6, -10]], 1) == 4
    assert find_max_value_of_equation([[0, 0], [3, 0], [9, 2]], 3) == 3

    assert sum_subarray_mins([3, 1, 2, 4]) == 17
    assert sum_subarray_mins([11, 81, 94, 43, 3]) == 444

    print(
        "[PASS] p16_monotonic_stack_iii: 5 题(拼接最大数/最大宽度坡/"
        "表现良好的最长时间段/满足不等式的最大值/子数组最小值之和)全部通过"
    )


if __name__ == "__main__":
    _self_test()
