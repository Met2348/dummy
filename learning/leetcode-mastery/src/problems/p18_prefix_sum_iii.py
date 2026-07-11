"""前缀和与差分 · Phase 3 竞赛级补充（Part III）：把"前缀和"的运算从加法/异或继续
扩展到按位与（且要维护的不是一个值而是一个动态的候选集合），补两道更贴近"前缀和 +
滑动窗口/贪心"结合的非重叠子数组计数难题，外加一道轻量的前缀异或练习题，共 5 道题。"""
from __future__ import annotations


def closest_to_target(arr: list[int], target: int) -> int:
    """
    【题意】给整数数组 arr 和整数 target，定义 `func(arr, l, r)` 为 `arr[l..r]`
    （闭区间）所有元素的按位与（AND）。枚举所有合法的 `(l, r)`，求 `func(arr, l, r)`
    和 target 的绝对差的最小值。

    【思路】这是前缀和思想的一个变体——普通前缀和维护"到当前位置为止的一个固定
    累积值"，但按位与不满足"可以直接用两个前缀值相减还原任意子区间"这个性质（与
    运算没有逆运算），所以不能照搬 `prefix[r+1] - prefix[l]` 的经典写法。关键突破口
    是**按位与具有单调不增性**：区间越长，AND 的结果只可能不变或变小（多与一个数
    只会消掉更多的 1，不会增加）。这意味着，固定右端点 r，所有以 r 结尾的子区间
    `arr[l..r]`（l 从 0 到 r）的 AND 值，**最多只有 O(log(max_value)) 种不同的
    取值**——因为每次把左端点继续往左扩展、AND 值要么不变、要么至少熄灭一个二进制
    位，而一个数最多有 32 个二进制位，值最多变化 32 次就会稳定或归零。

    于是把"维护一个前缀和数值"升级成"维护一个集合：所有以当前位置结尾的子数组的
    AND 值都有哪些"。扫描到位置 i、元素为 `x` 时，新的集合 = "上一个位置的集合里
    每个值都和 x 做一次 AND" ∪ "{x} 本身"（对应"子数组只包含 x 自己"这一种情况）。
    因为集合大小有 O(log(max_value)) 的上界，这一步更新的开销很小；每次更新完集合，
    立刻扫一遍集合里所有值，更新"离 target 最近的距离"这个全局最优解。

    【复杂度】时间 O(n log(max_value))（n 个位置，每个位置维护的候选集合大小是
    O(log(max_value))，且更新、扫描该集合都是这个量级）；空间 O(log(max_value))
    （当前候选集合的大小）。

    【易错点】
    - 容易忽略"子数组可以只包含 arr[i] 自己"这一种情况，只做"和上一个集合里的值
      AND"会漏掉这一支，必须显式地把 `{x}` 并入新集合。
    - 不能像普通前缀和那样只维护一个累积值——按位与没有逆运算，"整体减去前面一段"
      这种技巧在这里不成立，必须维护一整个候选值集合才能覆盖所有可能的左端点。
    - 全局最优解的更新要放在**每次集合更新之后**都做一次，而不是等扫描完整个数组
      再统一算——因为不同结尾位置的候选集合互不包含，最优答案可能出现在任意一个
      结尾位置。
    """
    best = abs(arr[0] - target)
    current = {arr[0]}
    for x in arr[1:]:
        current = {v & x for v in current} | {x}
        best = min(best, min(abs(v - target) for v in current))
    return best


def get_sum_absolute_differences(nums: list[int]) -> list[int]:
    """
    【题意】给一个已经**非递减排序**的整数数组 nums，构造并返回一个等长数组
    result，`result[i]` 等于 `nums[i]` 和数组中所有其他元素的绝对差之和。

    【思路】暴力做法是对每个 i 都遍历一遍数组求绝对差之和，O(n²)。这里的突破口是
    利用"数组已排序"这个前提，把每个位置的绝对值拆开：对位置 i，它左边的所有元素
    `nums[0..i-1]` 都 ≤ `nums[i]`，绝对差就是 `nums[i] - nums[j]`（不需要加绝对值
    符号）；它右边的所有元素 `nums[i+1..n-1]` 都 ≥ `nums[i]`，绝对差就是
    `nums[j] - nums[i]`。这样，`result[i]` 就能拆成两部分：
    "左边贡献" = `nums[i] * i - prefix_sum(0..i-1)`（i 个比 nums[i] 小或相等的数，
    每个都贡献一次 `nums[i]` 减去它自己）；
    "右边贡献" = `suffix_sum(i+1..n-1) - nums[i] * (n-i-1)`（右边 n-i-1 个数各自
    减去 `nums[i]`）。
    两部分都能用一个边扫边累积的前缀和变量 `prefix` 算出来（`prefix` 恰好是
    `nums[0..i-1]` 的和，`total - prefix - nums[i]` 恰好是 `nums[i+1..n-1]` 的和），
    不需要额外构造一份完整的前缀和数组，一次线性扫描即可算出所有 `result[i]`。

    【复杂度】时间 O(n)（一趟扫描，维护累积前缀和 + 总和）；空间 O(1)（除输出数组
    外，只需要两个累积变量）。

    【易错点】
    - 这个"拆掉绝对值"的推导**依赖数组已经排序**这个前提——如果拿到一个未排序的
      数组直接套这套公式，会得到错误结果，必须先确认（或者先排序）再使用。
    - 左边贡献的计数是 `i`（下标本身，因为 0-indexed 时下标 i 之前恰好有 i 个元素），
      右边贡献的计数是 `n - i - 1`，两个计数很容易在实现时写反或者差一。
    - `prefix` 变量必须在算完 `result[i]` **之后**才累加上 `nums[i]`（先用后加），
      如果提前把 `nums[i]` 计入 `prefix`，会把当前元素自己错误地算进"左边"的贡献里。
    """
    n = len(nums)
    total = sum(nums)
    prefix = 0
    result = []
    for i, x in enumerate(nums):
        right_sum = total - prefix - x
        left_contrib = x * i - prefix
        right_contrib = right_sum - x * (n - i - 1)
        result.append(left_contrib + right_contrib)
        prefix += x
    return result


def xor_queries(arr: list[int], queries: list[list[int]]) -> list[int]:
    """
    【题意】给正整数数组 arr 和查询数组 queries，`queries[i] = [left, right]`。对每
    条查询，求 `arr[left] xor arr[left+1] xor ... xor arr[right]`（闭区间），返回
    所有查询的结果数组。

    【思路】和"和为 K 的子数组""区域和检索"是同一个"前缀信息表，O(1) 回答区间查询"
    母题在异或运算上的应用：异或满足 `a ^ a = 0` 和结合律，所以前缀异或数组
    `prefix[i] = arr[0] ^ arr[1] ^ ... ^ arr[i-1]`（`prefix[0] = 0` 作哨兵）满足
    `prefix[r+1] ^ prefix[l] = arr[l] ^ ... ^ arr[r]`——因为 `prefix[l]` 恰好是
    `arr[0..l-1]` 的异或，`prefix[r+1] ^ prefix[l]` 会让 `arr[0..l-1]` 这一段在
    异或中出现两次而相互抵消，只剩下 `arr[l..r]` 这一段。构造一次前缀异或数组
    O(n)，之后每条查询都是 O(1) 的两个前缀值异或。

    【复杂度】时间 O(n + q)（n 为数组长度，q 为查询数，构造前缀数组 + 每条查询
    O(1)）；空间 O(n)（前缀异或数组）。

    【易错点】
    - 和前缀和版本一样，前缀数组要开 `n+1` 长、`prefix[0] = 0` 作哨兵，查询
      `[left, right]` 对应的是 `prefix[right+1] ^ prefix[left]`，容易在下标上
      漏掉这个"多 1"的偏移，写成 `prefix[right] ^ prefix[left]` 少异或了
      `arr[right]` 这一项。
    - 容易把异或和加减法搞混，误以为可以用 `prefix[r+1] - prefix[l]` 之类的减法
      运算——异或没有"减法"的概念，正确的"逆操作"就是再异或一次同样的值
      （`a ^ b ^ b = a`），必须用 `^` 而不是 `-`。
    """
    n = len(arr)
    prefix = [0] * (n + 1)
    for i, x in enumerate(arr):
        prefix[i + 1] = prefix[i] ^ x
    return [prefix[r + 1] ^ prefix[left] for left, r in queries]


def max_non_overlapping(nums: list[int], target: int) -> int:
    """
    【题意】给整数数组 nums（可能含负数）和整数 target，求最多能找到多少个"和恰好
    等于 target"的非空、互不重叠的连续子数组。

    【思路】"子数组和等于定值"是"和为 K 的子数组"那套"前缀和 + 哈希表"母题的经典
    信号，但这里多了一个"非重叠、要最多个数"的约束，需要在母题基础上叠加一个贪心
    策略：**只要在当前扫描位置发现了一个合法子数组（前缀和恰好比 target 早出现过
    对应的差值），就立刻把它计入答案、并且清空所有"当前正在累积的前缀和历史记录"，
    从下一个位置重新开始扫描**。这个"发现一个就立刻截断、清零重来"的贪心之所以
    正确，是因为：一旦确定了某个位置 i 是某个合法子数组的结尾，为了给后面留出尽量
    多的、互不重叠的合法子数组的空间，让这个子数组"尽量早结束"（不再继续往右扩展
    寻找更长但结尾更晚的合法子数组）总是不会更差——更早地把这一段"消耗掉"，给
    右边剩余的部分留出的空间只多不少。

    实现上维护一个"当前这一段（从上次重置点开始）出现过的前缀和集合" `seen`（初始
    含 0，对应"从当前起点算起、前缀和为 0"这个基准点）和当前段的累积和 `cur`；每
    加入一个新元素就检查 `cur - target` 是否在 `seen` 里——命中就说明存在一个合法
    子数组恰好在这里结束，计数 +1，并把 `seen` 重置成 `{0}`、`cur` 重置成 0，从
    下一个元素重新开始下一段的扫描。

    【复杂度】时间 O(n)（虽然外层和内层是两重 while 循环，但每个元素只会被扫描一次
    就被计入某一段或触发重置，均摊仍是线性）；空间 O(n)（最坏情况下 `seen` 集合要
    存下一段内所有不同的前缀和）。

    【易错点】
    - "发现即重置"这一步必须真正清空 `seen` 和 `cur`（重新只含 `{0}`、`cur=0`），
      不能只是把计数 +1 而继续沿用之前累积的前缀和历史——如果不重置，后续找到的
      "合法子数组"可能会和刚刚计入答案的这一段发生重叠，破坏"非重叠"这个约束。
    - `seen` 必须预置 `{0}`（对应"这一段从头开始就恰好等于 target"的情况），这是
      "和为 K 的子数组"母题里同样的易错点，在这道题的"分段"写法里，每次重置后都
      要重新预置这个 `{0}`，而不是只在最开始预置一次。
    - 贪心"发现即截断"的正确性依赖于"只要求最多个数，不要求最长或字典序最小"这个
      具体的优化目标——如果题目要求变成别的（比如"总覆盖长度最长"），同一套贪心
      不再直接适用，需要重新论证。
    """
    seen = {0}
    cur = 0
    count = 0
    for x in nums:
        cur += x
        if cur - target in seen:
            count += 1
            seen = {0}
            cur = 0
        else:
            seen.add(cur)
    return count


def min_sum_of_lengths(arr: list[int], target: int) -> int:
    """
    【题意】给正整数数组 arr 和整数 target，找两个"和恰好等于 target"的非空、互不
    重叠的连续子数组，使得这两个子数组的长度之和最小；如果找不出这样的两个子数组，
    返回 -1。

    【思路】因为 `arr` 保证全为正数，"和恰好等于 target 的子数组"可以用滑动窗口
    在 O(n) 内逐个找到（正数数组的前缀和严格递增，窗口左边界只需要单调向右收缩，
    不需要回退）。但本题真正的难点在"非重叠、且要两个子数组长度之和最小"这个组合
    优化目标上——这里借助一个辅助数组 `best_ending_by`，`best_ending_by[i]` 表示
    "**完全落在 `arr[0..i]` 范围内**的、和为 target 的子数组里，最短的那一个的
    长度"（如果不存在这样的子数组，记为无穷大）。这是前缀和思想的一个推广：不再
    只维护"到当前位置为止的一个数值"，而是维护"到当前位置为止、已知的最优子结构"，
    以便后续位置能够 O(1) 地查询"在我左边、且和我不重叠的最优选择是什么"。

    滑动窗口每找到一个新的合法窗口 `arr[left..right]`（和恰好为 target，长度
    `right-left+1`），就去查 `best_ending_by[left-1]`（"这个窗口开始之前、最优的
    不重叠候选"）——如果这个值不是无穷大，说明存在一个和当前窗口不重叠的更早的
    合法子数组，把两者长度相加就是一个候选答案，用它更新全局最优 `ans`。同时更新
    `best_ending_by[right] = min(best_ending_by[right-1], 当前窗口长度)`，把"目前
    为止见过的最短合法子数组长度"结转给后面的位置查询。

    【复杂度】时间 O(n)（滑动窗口左右指针各自最多移动 n 次，均摊线性）；空间 O(n)
    （`best_ending_by` 辅助数组）。

    【易错点】
    - `best_ending_by[i]` 的定义是"落在 `arr[0..i]` 范围内的最短合法子数组长度"，
      不是"以 i 结尾的合法子数组长度"——这个区别很关键：即使 `arr[i]` 本身不是任何
      合法子数组的结尾，`best_ending_by[i]` 也要从 `best_ending_by[i-1]` 继承过来
      （"最优候选"不会因为当前位置没有新发现而丢失），忘记继承会导致后面的查询
      漏掉更早位置本该找到的最优候选。
    - 查询"和当前窗口不重叠的最优候选"要用 `best_ending_by[left - 1]`（当前窗口
      左边界的前一个位置），不能用 `best_ending_by[right]`（那样会把当前窗口自己
      算进"更早的候选"里，破坏"不重叠"的约束）；`left == 0` 时（窗口从头开始）
      没有更早的位置可用，视为无穷大。
    - 这个解法依赖 arr 全为正数这一前提，才能使用单调的滑动窗口；如果数组可能含
      负数或 0，需要换成前缀和 + 哈希表的写法（类似 `max_non_overlapping`），不能
      直接套用滑动窗口。
    """
    n = len(arr)
    inf = float("inf")
    best_ending_by = [inf] * n
    ans = inf
    left = 0
    cur = 0
    for right, x in enumerate(arr):
        cur += x
        while cur > target:
            cur -= arr[left]
            left += 1
        prev_best = best_ending_by[right - 1] if right > 0 else inf
        if cur == target:
            length = right - left + 1
            before = best_ending_by[left - 1] if left > 0 else inf
            if before != inf:
                ans = min(ans, before + length)
            best_ending_by[right] = min(prev_best, length)
        else:
            best_ending_by[right] = prev_best
    return ans if ans != inf else -1


def _self_test() -> None:
    assert closest_to_target([9, 12, 3, 7, 15], 5) == 2
    assert closest_to_target([1000000, 1000000, 1000000], 1) == 999999

    assert get_sum_absolute_differences([2, 3, 5]) == [4, 3, 5]
    assert get_sum_absolute_differences([1, 4, 6, 8, 10]) == [24, 15, 13, 15, 21]

    assert xor_queries([1, 3, 4, 8], [[0, 1], [1, 2], [0, 3], [3, 3]]) == [2, 7, 14, 8]
    assert xor_queries([4, 8, 2, 10], [[2, 3], [1, 3], [0, 0], [0, 3]]) == [8, 0, 4, 4]

    assert max_non_overlapping([1, 1, 1, 1, 1], 2) == 2
    assert max_non_overlapping([-1, 3, 5, 1, 4, 2, -9], 6) == 2
    assert max_non_overlapping([-2, 6, 6, 3, 5, 4, 1, 2, 8], 10) == 3

    assert min_sum_of_lengths([3, 2, 2, 4, 3], 3) == 2
    assert min_sum_of_lengths([7, 3, 4, 7], 7) == 2
    assert min_sum_of_lengths([4, 3, 2, 6, 2, 3, 4], 6) == -1
    assert min_sum_of_lengths([3, 1, 1, 1, 5, 1, 2, 1], 3) == 3

    print(
        "[PASS] p18_prefix_sum_iii: 5 题"
        "（找出最接近目标值的函数值/有序数组中差绝对值之和/子数组异或查询/"
        "和为目标值的最大数目不重叠非空子数组数目/"
        "找两个和为目标值且不重叠的子数组）全部通过"
    )


if __name__ == "__main__":
    _self_test()
