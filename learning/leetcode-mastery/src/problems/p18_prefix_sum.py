"""分类 18：前缀和与差分 —— 区域和检索 / 除自身以外数组的乘积 / 和为K的子数组 / 航班预订统计。

前缀和的核心思想：把"多次查询区间和，每次都要重新把区间内元素加一遍"的重复计算，
预处理成"花一次 O(n) 算出前缀和数组，之后每次查询只需要两个前缀和相减，O(1)"。
差分数组是前缀和的逆运算：如果需要"多次对一个区间内所有元素批量加上某个值，最后
一次性还原出结果数组"，直接在原数组上暴力加是 O(n) 每次，而差分数组只需要在区间的
两个端点各改一个数（O(1) 每次），所有操作做完后再对差分数组求一次前缀和就能还原出
最终结果。
"""
from __future__ import annotations

from collections import defaultdict


class NumArray:
    """
    【题意】给定整数数组 nums（构造后数组不再改变），实现 `sum_range(left, right)`，
    返回下标 `[left, right]`（闭区间，含两端）内所有元素之和，会被多次调用。

    【思路】如果每次 `sum_range` 都老老实实把 `nums[left:right+1]` 加一遍，单次 O(n)，
    调用 m 次就是 O(n*m)——而区间会反复覆盖同一批元素，属于重复计算。既然数组构造后
    不再变化，可以在构造时一次性预处理出前缀和数组 `prefix`，其中 `prefix[i]` 表示
    `nums[0..i-1]` 的和（`prefix[0] = 0` 作为哨兵，避免 `left=0` 时要特判"左边界前面
    没有元素"）。这样 `sum_range(left, right) = prefix[right+1] - prefix[left]`——
    减去的是"不属于这个区间、但被 prefix[right+1] 多算进去的" `nums[0..left-1]` 那部分。
    预处理一次 O(n)，之后每次查询都是 O(1) 的减法。

    【复杂度】构造 O(n) 时间 O(n) 空间；每次 `sum_range` 查询 O(1)。

    【易错点】
    - 前缀和数组要开 `n+1` 长度、`prefix[0]=0` 作为哨兵，否则 `left=0` 时的区间和
      需要单独特判，容易漏掉这个边界或者写出 off-by-one。
    - 查询公式是 `prefix[right+1] - prefix[left]`（不是 `prefix[right] - prefix[left]`）,
      因为 `prefix[i]` 定义的是"前 i 个元素"之和，取到下标 right 的元素需要
      `prefix[right+1]`。
    """

    def __init__(self, nums: list[int]):
        self.prefix = [0] * (len(nums) + 1)
        for i, x in enumerate(nums):
            self.prefix[i + 1] = self.prefix[i] + x

    def sum_range(self, left: int, right: int) -> int:
        return self.prefix[right + 1] - self.prefix[left]


def product_except_self(nums: list[int]) -> list[int]:
    """
    【题意】给数组 nums，返回一个新数组 answer，其中 `answer[i]` 等于 nums 中除
    `nums[i]` 以外所有元素的乘积。不能使用除法（否则遇到 0 会出问题，且题目明确禁止），
    要求 O(n) 时间，尽量 O(1) 额外空间（不计输出数组）。

    【思路】"除自身以外的乘积"可以拆成两部分："i 左边所有元素的乘积"乘以"i 右边所有
    元素的乘积"。这正是前缀和思想的乘法版本：先从左到右扫一遍，`answer[i]` 先只存
    "左前缀积"（`nums[0..i-1]` 的乘积，`answer[0]=1` 作为没有左边元素时的哨兵）；
    再从右到左扫一遍，用一个变量 `right_product` 累积"右后缀积"，把它乘进
    `answer[i]` 里（`answer[i] *= right_product`，然后再更新
    `right_product *= nums[i]` 供更左边的位置使用）。两趟扫描都是 O(n)，且第二趟
    复用了 answer 数组本身来存左前缀积，不需要额外开一个数组存右后缀积，做到了
    O(1) 额外空间（不算输出数组）。

    【复杂度】时间 O(n)（两趟线性扫描）；空间 O(1) 额外空间（不计输出数组 answer）。

    【易错点】
    - 想用"先算总乘积，再对每个位置除以 nums[i]"——这是被题目明确禁止的除法思路，
      而且数组里出现 0 时会直接除零崩溃，必须用左右前缀积的思路规避。
    - 数组中若有一个 0：则除了那个 0 所在位置，其余位置的 answer 都应该是 0
      （因为它们的乘积里都包含了这个 0）；若有两个及以上 0，则所有位置都是 0。
      左右前缀积的写法天然处理了这些情况，不需要额外特判，但容易在脑内验证时想岔。
    - 第二趟扫描的顺序：必须先用 `right_product` 乘 `answer[i]`，再更新
      `right_product`（否则会把 `nums[i]` 自己也算进右后缀积里，多乘了一次自己）。
    """
    n = len(nums)
    answer = [1] * n
    for i in range(1, n):
        answer[i] = answer[i - 1] * nums[i - 1]
    right_product = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= right_product
        right_product *= nums[i]
    return answer


def subarray_sum(nums: list[int], k: int) -> int:
    """
    【题意】给整数数组 nums 和整数 k，求"和恰好等于 k"的连续子数组的个数（nums 可能
    含负数）。

    【思路】暴力解是枚举所有子数组的左右端点，对每个区间都重新算一遍和，O(n^2)或
    O(n^3)。用前缀和可以把"任意区间和"变成两个前缀和的差：设 `prefix[i]` 是
    `nums[0..i-1]` 的和，则子数组 `nums[j..i-1]` 的和等于 `prefix[i] - prefix[j]`。
    要求这个差等于 k，也就是 `prefix[j] = prefix[i] - k`。于是问题转化成：扫描到
    位置 i 时，"前面出现过多少次前缀和等于 `prefix[i] - k`"，这正好可以用一个哈希表
    边扫边记录"每个前缀和数值出现过多少次"来 O(1) 查到——这是前缀和技巧里最常被
    考的变体：把"区间和等于定值"转化成"两个前缀和之差"，再用哈希表把 O(n) 的
    "从当前位置往回找"变成 O(1) 查表。从左到右扫一遍，边维护当前前缀和 cur，边先
    查表加上 `count[cur - k]`（不含当前这个前缀和自身,避免用空子数组"自己减自己"
    凑出 k=0 的情况被错误计入两次），再把 cur 自己计入哈希表供后面的位置使用。

    【复杂度】时间 O(n)（一趟扫描 + 哈希表 O(1) 查询更新）；空间 O(n)（哈希表最多记
    n+1 个不同前缀和）。

    【易错点】
    - 哈希表必须预置 `{0: 1}`（表示"前缀和为 0"在还没扫描任何元素时就已经出现一次）
      ——否则会漏掉"从数组开头到当前位置"这一段恰好等于 k 的子数组（这类子数组
      需要 `prefix[j]=0` 且 j=0，如果不预置 0，这次计数永远查不到）。
    - 必须先查表统计答案、再把当前前缀和计入表中——顺序反了会把"空子数组"错误地
      当成合法答案多算一次（比如 k=0 时，若先把 cur 计入表再查 `count[cur-0]`，
      会把自己也算进去）。
    - k 可以是负数，nums 也可能含负数，哈希表方案对这些情况天然成立，不需要额外
      处理（不像滑动窗口那样要求元素非负）。
    """
    count: dict[int, int] = defaultdict(int)
    count[0] = 1  # 前缀和为 0 出现过一次（对应"从头开始"的子数组）
    cur = 0
    total = 0
    for x in nums:
        cur += x
        total += count[cur - k]
        count[cur] += 1
    return total


def corp_flight_bookings(bookings: list[list[int]], n: int) -> list[int]:
    """
    【题意】有 n 个航班，编号 1 到 n。bookings 中每条 `[first, last, seats]` 表示
    "在航班 first 到 last（闭区间，含两端）上各预订了 seats 个座位"。求每个航班最终
    被预订的总座位数，返回长度为 n 的数组。

    【思路】如果对每条预订记录都直接在结果数组的 `[first-1, last-1]` 区间里逐个加
    seats，最坏情况下一条记录要改 O(n) 个位置，m 条记录总共 O(n*m)——区间批量增加
    如果每次都逐格改，就是重复劳动。差分数组是前缀和的逆运算，专门用来解决"多次
    区间批量加，最后一次性还原"：构造一个和结果数组等长的差分数组 diff，对每条
    `[first, last, seats]`，只需要在 `diff[first-1] += seats`（这个位置开始，后面
    的前缀和都会带上这 seats）、以及 `diff[last] -= seats`（如果 last 没有超出数组
    范围，在这里减掉，抵消 seats 对 last 之后位置的影响）——每条记录都是 O(1) 的两次
    修改，与区间长度无关。所有记录处理完后，对 diff 数组做一次前缀和
    （`result[i] = result[i-1] + diff[i]`），前缀和运算本身就是差分的逆运算，
    这样"标记开始、标记结束"的信息就会在前缀和累加的过程中，自然地只在
    `[first-1, last-1]` 这段区间内生效。

    【复杂度】时间 O(n + m)（m 条预订记录各 O(1) 修改差分数组，最后 O(n) 求一次前缀和
    还原结果，比逐格暴力加的 O(n*m) 快得多）；空间 O(n)（差分数组，也可以复用输出
    数组本身）。

    【易错点】
    - 航班编号从 1 开始，题目给的 `first/last` 要转换成 0-indexed 数组下标时，
      `diff[first-1] += seats` 里的 -1 容易漏掉或多减。
    - `diff[last] -= seats` 用的是"原始（1-indexed）的 last"，不是 `last - 1`——因为
      差分数组里，"减法标记"要打在"最后一个受影响位置的下一个位置"上，
      1-indexed 的 last 换算成 0-indexed 后正好是 `last - 1 + 1 = last`。如果 last
      恰好等于 n（数组末尾），这次减法操作会落在 `diff[n]`，需要保证 diff 数组开了
      `n+1` 长度，避免越界（或者提前判断 `last < n` 才做这次减法）。
    - 求前缀和还原结果的这一步不能漏——直接返回差分数组本身不是最终答案,必须再
      对差分数组做一次前缀和累加。
    """
    diff = [0] * (n + 1)  # 多开一位避免 last==n 时的越界
    for first, last, seats in bookings:
        diff[first - 1] += seats
        diff[last] -= seats
    result = [0] * n
    result[0] = diff[0]
    for i in range(1, n):
        result[i] = result[i - 1] + diff[i]
    return result


def _self_test() -> None:
    na = NumArray([-2, 0, 3, -5, 2, -1])
    assert na.sum_range(0, 2) == 1
    assert na.sum_range(2, 5) == -1
    assert na.sum_range(0, 5) == -3

    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]

    assert subarray_sum([1, 1, 1], 2) == 2
    assert subarray_sum([1, 2, 3], 3) == 2

    assert corp_flight_bookings([[1, 2, 10], [2, 3, 20], [2, 5, 25]], 5) == [10, 55, 45, 25, 25]

    print("[PASS] p18_prefix_sum: 区域和检索 / 除自身以外的乘积 / 和为K的子数组 / 航班预订统计 全部正确")


if __name__ == "__main__":
    _self_test()
