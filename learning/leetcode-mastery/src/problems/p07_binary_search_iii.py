"""二分查找 · 竞赛级补充（Part III）：不重讲框架，专攻"二分答案"这一子类型里
最容易在面试/竞赛中卡住的变体，以 Hard 难度为主的 8 道题。

Part II 已经建立了"二分答案"的骨架——候选答案有一个取值范围 + 一个单调的
check(候选) 函数。这批题把"check 函数怎么写"这件事逼到了更难的角落：check
本身可能需要另一层双指针/滑动窗口才能算清楚（LC719 数对距离），或者"答案"
根本不是显然的数值区间、需要先想清楚单调性到底藏在哪个变量上（LC1898/LC1300/
LC1552/LC1482）。此外也补充两道"非二分答案"但同样考验区间不变量维护能力的
经典变体（LC81 含重复元素的旋转数组、LC540 用奇偶下标性质做二分）。
"""
from __future__ import annotations


# ── LC81 搜索旋转排序数组 II（Medium） ────────────────────────────────────
def search_ii(nums: list[int], target: int) -> bool:
    """
    【题意】和 LC33（Part I）相同的旋转数组查找问题，但这次数组**允许存在重复
        元素**（原本非降序排列后被旋转过），判断 target 是否存在（只要求返回
        布尔值，不要求下标，因为重复元素下标本身就不唯一）。
    【思路】LC33 靠比较 `nums[lo]` 和 `nums[mid]` 判断"哪一半有序"，这个判断
        在没有重复时是可靠的。但一旦 `nums[lo] == nums[mid] == nums[hi]`，
        这三个值相等时完全无法判断哪一半有序（比如 `[3,1,2,3,3]` 和
        `[3,3,3,1,2]` 在 lo/mid/hi 都取到 3 时形状可能完全不同）。这时候唯一
        安全的做法是**同时收缩 `lo` 和 `hi` 各一格**，放弃这一步的判断——因为
        被丢弃的这两个位置的值和留下来的值相等，不会丢失 target 可能存在的
        信息。除了这一条新增的特判分支，其余逻辑和 LC33 完全一样。
    【复杂度】时间：平均 O(log n)，最坏 O(n)（数组几乎全是重复元素时，
        `lo += 1; hi -= 1` 这个分支会被连续触发很多次，退化成线性扫描）；
        空间 O(1)。
    【易错点】特判分支的判断条件必须是**三者都相等**（`nums[lo] == nums[mid]
        == nums[hi]`），只判断其中两者相等是不够的——比如 `nums[lo] ==
        nums[mid]` 但 `nums[hi]` 不同，仍然可以靠和 LC33 一样的逻辑判断出
        哪一半有序，不需要放弃这一步；这个特判必须放在"判断哪一半有序"的
        逻辑**之前**检查，否则会先被 `nums[lo] <= nums[mid]` 这类条件提前
        分流到错误的分支。
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return True
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1
            hi -= 1
        elif nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False


# ── LC540 有序数组中的单一元素（Medium） ──────────────────────────────────
def single_non_duplicate(nums: list[int]) -> int:
    """
    【题意】一个升序数组，除了一个元素只出现一次之外，其余每个元素都恰好出现
        两次，要求 O(log n) 时间、O(1) 空间找出这个只出现一次的元素。
    【思路】关键观察：**只要还没扫到"单独的那个元素"，每一对重复元素的第一个
        必然出现在偶数下标上**（下标从 0 开始）——因为在它之前，前面所有元素
        都是"每两个占用两个下标"的完整对，不会打乱这个奇偶节律。一旦"单独的
        那个元素"出现在某个偶数下标之前，这个节律就会被打破：从它开始，原本
        该在偶数下标的"每对的第一个"全部往后错开了一位，变成出现在奇数下标上。
        于是二分的判断依据是：取偶数下标 `mid`（如果 mid 是奇数就减一变成偶数），
        比较 `nums[mid]` 和 `nums[mid+1]` 是否相等——相等说明这一对还没被打乱，
        单独元素在 mid 右边（`lo = mid + 2`）；不相等说明节律已经被打乱，单独
        元素在 mid 或其左边（`hi = mid`）。
    【复杂度】时间 O(log n)，空间 O(1)。
    【易错点】必须先把 `mid` 修正成偶数（`if mid % 2 == 1: mid -= 1`）再去
        比较 `nums[mid]` 和 `nums[mid+1]`——如果 mid 恰好落在奇数下标上直接
        比较，比较的其实是"某一对的第二个"和"下一对的第一个"，这两者之间的
        相等关系和"节律是否被打乱"无关，会得出错误的收缩方向；收缩语句必须是
        `lo = mid + 2`（跳过完整的一对），写成 `mid + 1` 会导致 `mid` 下次
        又落在奇数下标上，需要多做一次没必要的修正。
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid % 2 == 1:
            mid -= 1
        if nums[mid] == nums[mid + 1]:
            lo = mid + 2
        else:
            hi = mid
    return nums[lo]


# ── LC275 H 指数 II（Medium） ─────────────────────────────────────────────
def h_index_ii(citations: list[int]) -> int:
    """
    【题意】给一个**已经按升序排好**的引用次数数组 citations，求 h 指数——即
        存在 h 篇论文每篇至少被引用 h 次，且其余论文引用次数都不超过 h（取满足
        条件的最大 h）。要求 O(log n)。
    【思路】数组已经升序排列，第 `i` 个下标（0-indexed）右边（含自身）一共有
        `n - i` 篇论文；如果 `citations[i] >= n - i`，说明"从下标 i 到末尾"
        这 `n - i` 篇论文每篇引用数都至少是 `n - i`（因为升序，`citations[i]`
        是这一段里最小的，只要它自己都 `>= n - i`，右边更大的当然也满足），
        也就是说 `h = n - i` 是一个合法的候选值。要找**最大**的合法 h，等价于
        找**最小**的满足 `citations[i] >= n - i` 的下标 `i`（i 越小，
        `n - i` 越大）。这是标准的 lower_bound 二分。
    【复杂度】时间 O(log n)，空间 O(1)。
    【易错点】二分的谓词是 `citations[mid] >= n - mid`，不是直接拿
        `citations[mid]` 和某个固定值比较——`n - mid` 这个"门槛"本身也随
        下标变化，容易漏看这一点直接套用普通二分模板；最终答案是
        `n - lo`（下标转换成"还剩多少篇"），不是 `lo` 本身。
    """
    n = len(citations)
    lo, hi = 0, n - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if citations[mid] >= n - mid:
            hi = mid - 1
        else:
            lo = mid + 1
    return n - lo


# ── LC1898 可移除字符的最大数目（Medium，二分答案） ───────────────────────
def maximum_removals(s: str, p: str, removable: list[int]) -> int:
    """
    【题意】字符串 p 是 s 的子序列。给一个下标数组 removable（s 中一些下标，
        互不相同）。选一个 k（`0 <= k <= len(removable)`），把 `removable`
        的**前 k 个**下标对应的字符从 s 中删除，要求删除之后 p 仍然是剩下
        字符组成的字符串的子序列。求满足条件的最大 k。
    【思路】这是"二分答案"骨架里"check 函数本身就是一次子序列匹配"的变体：
        候选答案是 k，`check(k)`（"删掉前 k 个下标后 p 还是不是子序列"）具有
        单调性——**删得越少，越不容易破坏子序列关系**，也就是说如果
        `check(k)` 为真，那么对任意 `k' < k`，`check(k')` 也一定为真（删的
        更少，约束更宽松）。这个单调性（越大越难满足、越小越容易满足）保证了
        可以对 k 在 `[0, len(removable)]` 上二分，找**最大的使 check 为真的
        k**。`check(k)` 的实现是标准的双指针子序列匹配：先把前 k 个下标标记
        为"已删除"，再用双指针扫一遍 s（跳过被删除的下标），看能不能按顺序
        匹配完整个 p。
    【复杂度】时间 O((n + m) log m)（n = len(s)，m = len(removable)，每次
        check 是 O(n) 的双指针扫描，二分需要 O(log m) 轮），空间 O(k)
        （标记被删除下标的集合）。
    【易错点】这题二分找的是"最大的可行 k"（谓词从 True 逐渐变成 False，
        找 True/False 分界线的右端），收缩方向和"找最小可行答案"（LC875 那种）
        正好相反：`check(mid)` 为真时应该 `lo = mid`（尝试更大的 k，而不是
        `hi = mid`），配合上取整的 `mid = lo + (hi - lo + 1) // 2`（向上取整）
        才能避免死循环——如果沿用 LC875 那种下取整 mid 和 `hi = mid` 的写法，
        会在 `lo` 和 `hi` 只差 1 时反复卡在同一个 mid 上出不来。
    """

    def is_subsequence_after_removal(k: int) -> bool:
        removed = set(removable[:k])
        j = 0
        for i, ch in enumerate(s):
            if i in removed:
                continue
            if j < len(p) and ch == p[j]:
                j += 1
        return j == len(p)

    lo, hi = 0, len(removable)
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # 向上取整，配合 lo = mid 避免死循环
        if is_subsequence_after_removal(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


# ── LC1300 转变数组后最接近目标值的数组和（Medium，二分答案） ─────────────
def closest_to_target(arr: list[int], target: int) -> int:
    """
    【题意】给数组 arr 和目标值 target，选一个整数 value：把 arr 中所有大于
        value 的元素都变成 value，使变换后数组的和尽量接近 target（绝对差最
        小）。如果有多个 value 效果相同，返回其中最小的那个。
    【思路】定义 `mutated_sum(value) = sum(min(x, value) for x in arr)`——
        这是关于 value **单调不减**的函数（value 越大，每个元素被"封顶"得
        越少，总和只会更大或不变）。这个单调性意味着可以二分 value，找到
        **最小的、使 `mutated_sum(value) >= target` 成立的 value**（lower_
        bound）。但这还没完——真正最优的 value 可能是这个 lower_bound，也
        可能是它的前一个值（`lower_bound - 1`，此时 `mutated_sum` 略小于
        target）：因为"最接近"可能是从下方逼近也可能从上方逼近，二分只能
        帮你锁定"恰好跨过 target 的那个临界点"，跨过前后两个候选值都要拿出来
        比较绝对差，才能确定真正最优的那一个。
    【复杂度】时间 O(n log(max(arr)))（每次二分迭代都要 O(n) 遍历算
        mutated_sum，一共 O(log(max)) 次迭代），空间 O(1)。
    【易错点】二分只能找到"跨过 target 的临界点"，不能直接把这个临界点当成
        答案返回——必须额外比较 `lower_bound` 和 `lower_bound - 1` 两个候选
        谁的绝对差更小；"多个候选效果相同（绝对差相等）时返回最小值"这条
        规则容易被忽略，比较时如果 `mutated_sum(lo - 1)` 和 `mutated_sum(lo)`
        到 target 的距离相等，必须优先返回较小的 `lo - 1`（用 `<=` 而不是
        `<` 做比较）。
    """

    def mutated_sum(value: int) -> int:
        return sum(min(x, value) for x in arr)

    lo, hi = 0, max(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mutated_sum(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(mutated_sum(lo - 1) - target) <= abs(mutated_sum(lo) - target):
        return lo - 1
    return lo


# ── LC1482 制作 m 束花所需的最少天数（Medium，二分答案） ──────────────────
def min_days(bloom_day: list[int], m: int, k: int) -> int:
    """
    【题意】花园里有 n 朵花，第 i 朵花会在第 `bloomDay[i]` 天开放。做一束花
        需要用 k 朵**相邻**且已经开放的花；一朵花只能用于一束花。求能凑出 m
        束花的最少等待天数；如果无论等多久都凑不出，返回 -1。
    【思路】"二分答案"骨架：二分等待的天数 `day`。`bouquets_possible(day)`
        （"等到第 day 天，最多能凑出几束花"）是关于 day **单调不减**的
        函数——等待的天数越多，开放的花越多（或不变），能凑的花束数也只会
        更多或不变。这个单调性支持在 `[min(bloomDay), max(bloomDay)]` 上
        二分，找**最小的、使 `bouquets_possible(day) >= m` 的 day**。
        `bouquets_possible(day)` 用贪心统计：从左到右扫一遍花园，维护"当前
        连续已开放的花的数量" `consecutive`，每凑够 k 朵连续开放的花就算一束、
        清零计数器重新开始数下一段；一旦遇到还没开放的花，`consecutive` 也要
        清零（不连续的花不能拼成一束）。
    【复杂度】时间 O(n log(max(bloomDay) - min(bloomDay)))，空间 O(1)。
    【易错点】必须在二分之前先判断 `m * k > len(bloomDay)`（总共需要的花数
        超过花园里的花总数），这种情况无论等多久都不可能凑够，要直接返回
        -1——否则二分会在一个根本不存在解的区间里瞎找，得到一个错误的"最小
        天数"；统计 `bouquets_possible` 时，"够 k 朵就清零重新计数"和"遇到
        未开放的花也要清零"这两处清零逻辑都不能漏，只处理其中一种会多算或
        漏算花束数。
    """
    if m * k > len(bloom_day):
        return -1

    def bouquets_possible(day: int) -> int:
        count = 0
        consecutive = 0
        for bloom in bloom_day:
            if bloom <= day:
                consecutive += 1
                if consecutive == k:
                    count += 1
                    consecutive = 0
            else:
                consecutive = 0
        return count

    lo, hi = min(bloom_day), max(bloom_day)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if bouquets_possible(mid) >= m:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ── LC1552 两球之间的磁力（Medium，二分答案） ─────────────────────────────
def max_distance(position: list[int], m: int) -> int:
    """
    【题意】n 个篮子分布在数轴上（位置由 position 给出），要把 m 个球放进 m
        个不同的篮子里，使"任意两球之间磁力的最小值"尽可能**大**（磁力定义
        为两球位置差的绝对值）。求这个最大化后的最小磁力。
    【思路】"最大化最小值"是二分答案的另一种典型信号（对应 LC875/LC1011
        那种"最小化最大值"）。先把 position 排序，二分候选的"最小间距"
        force：`can_place(force)`（"用贪心的方式，从左到右依次放球，只要
        当前篮子和上一个放球位置的距离 `>= force` 就放一个球，最终能放下的
        球数是否 `>= m`"）是关于 force **单调不增**的函数——force 越大，
        每两个球之间要求的间距越宽，能放下的球自然越少。这个单调性支持二分
        force，找**最大的、使 `can_place(force)` 仍为真的 force**。
    【复杂度】时间 O(n log n + n log(max(position) - min(position)))
        （排序 + 二分，每次 check 是 O(n) 贪心扫描），空间 O(1)（不计排序）。
    【易错点】这题和 LC1898 一样是"找最大可行值"，收缩方向要写成
        `mid = lo + (hi - lo + 1) // 2` 配合 `can_place(mid)` 为真时
        `lo = mid`——如果照抄 LC1482/LC875 那种"找最小可行值"的下取整 mid
        写法，会在区间只剩两个候选时死循环；`can_place` 贪心时第一个球必须
        放在排序后的第一个篮子里（`last = position[0]`），从第二个篮子开始
        才判断"距离是否达标"，不能凭空跳过第一个篮子。
    """
    position = sorted(position)

    def can_place(force: int) -> bool:
        count = 1
        last = position[0]
        for pos in position[1:]:
            if pos - last >= force:
                count += 1
                last = pos
        return count >= m

    lo, hi = 1, position[-1] - position[0]
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if can_place(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


# ── LC719 找出第 K 小的数对距离（Hard，二分答案） ─────────────────────────
def kth_smallest_pair_distance(nums: list[int], k: int) -> int:
    """
    【题意】数组 nums 中任意两个下标 `i < j`，定义"数对距离" =
        `abs(nums[i] - nums[j])`。所有这样的数对距离构成一个多重集合，求其中
        第 k 小的距离值。
    【思路】做法见本文件对应 lecture 的深挖部分——核心是把"直接枚举/排序所有
        数对距离"（`O(n^2 log n)`，数对总数是 `O(n^2)` 级别，n 较大时会
        超时）转化成"二分答案 + check 用双指针统计"：先把 nums 排序，二分
        候选的距离上限 `d`，`pairs_with_distance_at_most(d)`（"有多少个数对
        的距离 `<= d`"）是关于 d **单调不减**的函数（d 越大，满足条件的数对
        只会更多或不变）。找**最小的、使这个计数 `>= k` 的 d**，就是第 k 小
        的距离。计数函数本身也不能暴力枚举所有数对（否则退化回 O(n^2)），
        要用滑动窗口/双指针：排序后固定右端点 `right`，用一个只会单调右移的
        左指针 `left` 维护"最小的、使 `nums[right] - nums[left] <= d`
        成立的左边界"，则以 `right` 结尾、距离 `<= d` 的数对个数就是
        `right - left`（`left` 到 `right-1` 之间的每一个下标和 right 配对
        都满足条件）。左指针全程只增不减，配合右指针的移动，整个统计过程是
        均摊 O(n) 的。
    【复杂度】时间 O(n log n + n log(max(nums) - min(nums)))（排序 + 二分，
        每轮 check 用双指针是均摊 O(n)），空间 O(log n)（排序的递归栈，不计
        输入本身）。
    【易错点】计数函数里的双指针 `left` 必须在**外层循环之间保持状态、不重置
        为 0**——如果每次固定新的 `right` 都把 `left` 重新设回 0 再重新扫，
        计数函数本身就退化成 O(n^2)，二分的意义就被抵消了；数组必须先排序，
        否则"距离 `<= d`"这个条件和下标的单调关系无法用双指针维护；二分的
        上界是 `nums[-1] - nums[0]`（排序后整个数组的最大距离），不是
        `max(nums)` 本身。
    """
    nums = sorted(nums)
    n = len(nums)

    def pairs_with_distance_at_most(d: int) -> int:
        count = 0
        left = 0
        for right in range(n):
            while nums[right] - nums[left] > d:
                left += 1
            count += right - left
        return count

    lo, hi = 0, nums[-1] - nums[0]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if pairs_with_distance_at_most(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _self_test() -> None:
    assert search_ii([2, 5, 6, 0, 0, 1, 2], 0) is True
    assert search_ii([2, 5, 6, 0, 0, 1, 2], 3) is False
    assert search_ii([1, 0, 1, 1, 1], 0) is True

    assert single_non_duplicate([1, 1, 2, 3, 3, 4, 4, 8, 8]) == 2
    assert single_non_duplicate([3, 3, 7, 7, 10, 11, 11]) == 10

    assert h_index_ii([0, 1, 3, 5, 6]) == 3
    assert h_index_ii([1, 2, 100]) == 2

    assert maximum_removals("abcacb", "ab", [3, 1, 0]) == 2
    assert maximum_removals("abcbddddd", "abcd", [3, 2, 1, 4, 5, 6]) == 1
    assert maximum_removals("abcab", "abc", [0, 1, 2, 3, 4]) == 0

    assert closest_to_target([4, 9, 3], 10) == 3
    assert closest_to_target([2, 3, 5], 10) == 5

    assert min_days([1, 10, 3, 10, 2], 3, 1) == 3
    assert min_days([1, 10, 3, 10, 2], 3, 2) == -1
    assert min_days([7, 7, 7, 7, 12, 7, 7], 2, 3) == 12

    assert max_distance([1, 2, 3, 4, 7], 3) == 3
    assert max_distance([5, 4, 3, 2, 1, 1000000000], 2) == 999999999

    assert kth_smallest_pair_distance([1, 3, 1], 1) == 0
    assert kth_smallest_pair_distance([1, 1, 1], 2) == 0
    assert kth_smallest_pair_distance([1, 6, 1], 3) == 5

    print(
        "[PASS] p07_binary_search_iii: 8 题"
        "（搜索旋转排序数组II/有序数组中的单一元素/H指数II/可移除字符的最大数目/"
        "转变数组后最接近目标值的数组和/制作m束花所需的最少天数/两球之间的磁力/"
        "找出第K小的数对距离）全部通过"
    )


if __name__ == "__main__":
    _self_test()
