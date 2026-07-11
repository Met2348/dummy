"""二分查找 · 进阶补充（Part II）：不重讲框架，重点引入"二分答案"这一子类型，扩大变体覆盖面的 8 道题。

统一约定沿用 Part I：lower_bound 风格的二分一律用左闭右开 [lo, hi) 写法。本文件新增的
"二分答案"题目，套的也是同一套骨架——只是二分的对象从"数组下标"换成了"答案的取值范围"。
"""
from __future__ import annotations


def search_matrix(matrix: list[list[int]], target: int) -> bool:
    """
    【题意】给定一个 m×n 矩阵，每行从左到右升序，且每行第一个数都大于上一行最后一个数
    （整个矩阵拉平后完全升序）。判断 target 是否存在于矩阵中。
    【思路】"每行首元素大于上一行尾元素"这个条件，意味着把矩阵按行拼接起来就是一个
    普通的一维升序数组，唯一的区别只是"下标 i" 需要通过 `i // cols` 和 `i % cols`
    换算成"第几行第几列"。二分查找的骨架完全不变，只是把 `nums[mid]` 换成了
    `matrix[mid // cols][mid % cols]`。
    【复杂度】时间 O(log(m·n))，空间 O(1)。
    【易错点】1) 这道题的前提是"整个矩阵拉平后严格升序"，如果每行升序但行与行之间
    的值域有重叠（比如下一行的首元素反而更小），就不能用这种一维二分，要换成 LC240
    的做法；2) 换算行列下标时容易写反 `//` 和 `%`，行号是 `mid // cols`，列号是
    `mid % cols`；3) 矩阵为空或某一行为空要提前特判，否则 `cols = len(matrix[0])`
    会直接抛异常。
    """
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target:
            return True
        elif val < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


def search_matrix_ii(matrix: list[list[int]], target: int) -> bool:
    """
    【题意】给定一个矩阵，每一行从左到右升序、每一列从上到下升序，但不保证"下一行首
    元素大于上一行尾元素"（行与行之间的值域可能重叠）。判断 target 是否存在。
    【思路】因为行列值域可能重叠，LC74 那种"拉平成一维数组二分"的前提不成立了，这题
    换一个角度：从矩阵的**右上角**开始看，这个位置有一个特殊性质——它是"当前所在行的
    最大值"，同时也是"当前所在列的最小值"。所以每次比较它和 target：如果它比 target
    大，说明整列（从这个位置往下）都比它大、更不可能等于 target，可以把这一整列排除，
    向左移动一格；如果它比 target 小，说明整行（从这个位置往左）都比它小，可以把这一
    整行排除，向下移动一格。每一步都能排除一整行或一整列，最多走 m+n 步。
    【复杂度】时间 O(m+n)（不是 O(log)，因为这题的搜索空间不再具有"拉平后单调"的
    性质，只能利用行列有序做线性排除，而不能做真正的二分），空间 O(1)。
    【易错点】1) 起点必须是右上角（或者对称地选左下角），选错起点（比如左上角或右下
    角）不具备"行最大、列最小"这种能同时排除一行或一列的性质，算法会失效；2) 循环
    终止条件是 `r < rows and c >= 0`，两个边界都要检查，只检查一个会在其中一个方向
    越界；3) 这题经常被误当成 LC74 一样直接用一维二分做，一定要先确认"行首是否大于
    上一行行尾"这个前提是否成立，不成立就不能套用 LC74 的解法。
    """
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    r, c = 0, cols - 1
    while r < rows and c >= 0:
        val = matrix[r][c]
        if val == target:
            return True
        elif val > target:
            c -= 1
        else:
            r += 1
    return False


def find_min(nums: list[int]) -> int:
    """
    【题意】一个原本升序的数组被旋转过（无重复元素），找出其中的最小值，要求 O(log n)。
    【思路】和 Part I 的 LC33（旋转数组找 target）是同一个"哪一半有序"的判断框架，
    但这题更简单：不需要判断 target 落不落在有序半的值域里，只需要判断"最小值在
    左半还是右半"。比较 `nums[mid]` 和 `nums[hi]`：如果 `nums[mid] > nums[hi]`，
    说明从 mid 到 hi 之间发生了"掐断"（旋转点在 mid 右边，包含 mid），最小值一定
    在 mid 右边，`lo = mid + 1`；如果 `nums[mid] <= nums[hi]`，说明 mid 到 hi
    这一段本身已经是升序的（没有被掐断），最小值只可能是 mid 自己或者在 mid 左边，
    所以保留 mid（`hi = mid`，不是 `mid - 1`）。
    【复杂度】时间 O(log n)，空间 O(1)。
    【易错点】1) 一定要拿 `nums[mid]` 和 `nums[hi]` 比较，而不是和 `nums[lo]`
    比较——用 `nums[lo]` 比较时，当区间已经是"没被掐断的一小段"时无法区分"最小值
    是不是 lo 自己"这种边界情况，容易死循环；2) `hi = mid`（左闭右开思路下的等号
    写法）而不是 `hi = mid - 1`，因为 `nums[mid] <= nums[hi]` 时 mid 本身仍是
    候选答案，不能排除掉；3) 这题假设无重复元素，如果数组允许重复（LC154），
    `nums[mid] == nums[hi]` 时无法判断该往哪收缩，需要额外处理，见下方 LC154。
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid
    return nums[lo]


def find_peak_element(nums: list[int]) -> int:
    """
    【题意】给定数组 nums（可以想象成 `nums[-1] = nums[n] = -inf`），峰值元素是
    严格大于左右相邻元素的元素。找出任意一个峰值的下标，要求 O(log n)；数组可能
    存在多个峰值，返回其中任意一个即可。
    【思路】表面上数组整体无序，似乎二分无从下手，但关键观察是：**只要沿着数值上升
    的方向走，一定能走到一个峰值**（因为两端视为 -inf，一路上升不可能一直冲出数组
    边界而没有遇到峰值）。所以每次比较 `nums[mid]` 和 `nums[mid+1]`：如果
    `nums[mid] < nums[mid+1]`，说明从 mid 往右走数值在上升，峰值一定存在于
    `[mid+1, hi]`（mid 自己不可能是峰值，因为右边比它大），`lo = mid + 1`；否则
    `nums[mid] >= nums[mid+1]`，说明峰值在 `[lo, mid]`（mid 自己有可能就是峰值，
    保留它），`hi = mid`。
    【复杂度】时间 O(log n)，空间 O(1)。
    【易错点】1) 这题不要求找"全局最大值"，只要求任意一个"局部峰值"，所以不能想着
    先排序或者全局扫描再二分，直接按"往上坡方向走"的单调性二分即可；2) 比较的是
    `nums[mid]` 和 `nums[mid+1]`（相邻的下一个），而不是 `nums[mid]` 和
    `nums[hi]`，这题的单调性依据和 LC153 完全不同，不要混用比较对象；3) 边界
    `mid+1` 不会越界，因为循环条件 `lo < hi` 保证了 mid 严格小于 hi，`mid+1`
    最大也只能等于 hi，不会超出数组范围。
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def min_eating_speed(piles: list[int], h: int) -> int:
    """
    【题意】珂珂要在 h 小时内吃完 n 堆香蕉（每小时选一堆、以速度 speed 根/小时进食，
    一堆吃不完这一小时也不会去吃下一堆，下一小时继续吃这一堆剩下的），求能在 h 小时
    内吃完所有香蕉的最小速度 speed。
    【思路】这是"二分答案"这一子类型的入门题：不在数组下标上二分，而是在"speed 的
    取值范围"上二分。为什么能对 speed 二分——因为 check(speed)（"用这个速度能否在
    h 小时内吃完"）具有单调性：speed 越大，吃完所有香蕉花的总时间越短（或不变），
    这是一个关于 speed 单调不增的函数，因此"能否在 h 小时内吃完"这个布尔谓词，
    随 speed 增大只会从 False 翻转到 True 一次，不会来回跳变——这正是二分能work
    的前提（谓词单调）。二分的区间是 `[1, max(piles)]`（速度至少是 1，最大不需要
    超过最大的那一堆，因为再快也一小时吃完一堆）。check(speed) 用贪心/模拟计算：
    每堆需要 `ceil(pile / speed)` 小时，累加就是总耗时。
    【复杂度】时间 O(n · log(max(piles)))（每次二分迭代都要 O(n) 遍历算总耗时，
    一共 O(log(max)) 次迭代），空间 O(1)。
    【易错点】1) `ceil(pile / speed)` 在整数运算里要写成 `(pile + speed - 1) //
    speed`，不能直接用 `pile // speed`（会少算没吃完的那部分时间）；2) 二分的
    收缩方向是"check 通过（能吃完）就尝试更小的 speed（hi = mid），check 不通过
    就必须增大 speed（lo = mid + 1）"——本质是在找"最小的、能让 check 为 True 的
    speed"，也就是 lower_bound 风格的二分，写反方向会算出能吃完但不是最小的速度；
    3) 下界不能从 0 开始（速度为 0 永远吃不完，且做除法会报错），必须从 1 开始。
    """

    def hours_needed(speed: int) -> int:
        return sum((pile + speed - 1) // speed for pile in piles)

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if hours_needed(mid) <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo


def ship_within_days(weights: list[int], days: int) -> int:
    """
    【题意】传送带上的包裹必须按 weights 数组给定的顺序装船（不能打乱顺序），船有
    固定的载重上限 capacity，每天卸货前尽量多装货（不超过 capacity 就继续装下一个），
    求能在 days 天内运完所有包裹的最小 capacity。
    【思路】和爱吃香蕉的珂珂是同一个"二分答案"骨架：不二分下标，二分 capacity 本身。
    单调性依据是：capacity 越大，能装的天数就越少（或不变）——"用给定 capacity
    能否在 days 天内运完"这个谓词，随 capacity 增大只会从 False 翻转到 True 一次。
    二分区间的下界是 `max(weights)`（capacity 再小也必须能装下最重的那一件，
    否则永远没法把它装船），上界是 `sum(weights)`（一天全部装完，天数一定够）。
    check(capacity) 用贪心模拟：按顺序累加当前这一天已装的重量，一旦加上下一件会
    超过 capacity，就新开一天重新累加。
    【复杂度】时间 O(n · log(sum(weights)))，空间 O(1)。
    【易错点】1) 包裹必须按给定顺序装船，不能重新排序或挑着装，贪心模拟时要按数组
    原顺序遍历，不能想着"轻的先装、重的后装"这类调整顺序的贪心；2) 二分下界必须是
    `max(weights)` 而不是 1——如果最重的一件超过了当前二分到的 capacity，这个
    capacity 根本不可行，从 1 开始二分虽然结果依然正确但会浪费效率，更重要的是
    check 函数如果没处理这种"单件超重"的情况会导致死循环（永远也装不完这一件）；
    3) 收缩方向同样是"check 通过就尝试更小的 capacity（hi = mid），不通过就增大
    （lo = mid + 1）"。
    """

    def days_needed(capacity: int) -> int:
        days_used = 1
        cur_load = 0
        for w in weights:
            if cur_load + w > capacity:
                days_used += 1
                cur_load = 0
            cur_load += w
        return days_used

    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if days_needed(mid) <= days:
            hi = mid
        else:
            lo = mid + 1
    return lo


def split_array(nums: list[int], k: int) -> int:
    """
    【题意】把非负整数数组 nums 按顺序分割成 k 个非空的连续子数组，使得"这 k 个子
    数组各自的和"里最大的那个尽可能小，返回这个最小化之后的最大子数组和。
    【思路】仍然是"二分答案"骨架，二分的对象是"最大子数组和的上限" limit：
    limit 越大，需要分割的段数就越少（或不变，因为限制越宽松，每段能装的元素越多）——
    "在限制 limit 之下最少需要分成几段"这个函数随 limit 增大单调不增，所以"能否
    用不超过 k 段做到限制为 limit"这个谓词也具有二分能利用的单调性。二分区间下界是
    `max(nums)`（每段至少要能装下最大的那个数，否则那个数所在的段永远超限），上界
    是 `sum(nums)`（合成一段，段数为 1，一定 <= k）。check(limit) 用贪心统计：
    像 LC1011 一样按顺序累加，超过 limit 就新开一段，统计出总共需要几段。
    【复杂度】时间 O(n · log(sum(nums)))，空间 O(1)。
    【易错点】1) 和 LC1011 几乎是同一道题换了个问法——"求最小运载能力"和"求最大
    子数组和的最小值"本质上是同一个二分答案模型，做完这题应该能看出两者是同构的；
    2) 收缩方向：`segments_needed(mid) <= k` 说明当前 limit 足够宽松（甚至可能
    过于宽松），要尝试更小的 limit（`hi = mid`），反之说明 limit 太紧、分出的段数
    超过了 k，要放宽 limit（`lo = mid + 1`）；3) 下界必须是 `max(nums)` 而不是
    0 或 1，否则某个单独的大数无法被任何一段装下，check 函数会得到不合理的段数。
    """

    def segments_needed(limit: int) -> int:
        segments = 1
        cur_sum = 0
        for x in nums:
            if cur_sum + x > limit:
                segments += 1
                cur_sum = 0
            cur_sum += x
        return segments

    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if segments_needed(mid) <= k:
            hi = mid
        else:
            lo = mid + 1
    return lo


def find_min_ii(nums: list[int]) -> int:
    """
    【题意】和 LC153 相同的旋转数组找最小值问题，但这次数组里**允许存在重复元素**，
    求最小值，仍要求尽量做到 O(log n)（最坏情况会退化）。
    【思路】LC153 的判断依据是比较 `nums[mid]` 和 `nums[hi]`，靠这两者严格
    大于/小于/等于来判断"哪一半有序、最小值在哪"。但一旦允许重复，会出现
    `nums[mid] == nums[hi]` 的情况——这时候完全无法判断最小值在左半还是右半
    （比如 [1,1,1,0,1] 和 [1,0,1,1,1] 在 mid、hi 都取到相同值 1 时，形状却
    完全不同，最小值位置也不同）。这题在 LC153 的基础上只需要多加一个分支：
    `nums[mid] == nums[hi]` 时，因为不能确定方向，但 `nums[hi]` 这个副本至少
    还有 `nums[mid]` 这个跟它相等的元素可以代替它的"代表性"，所以可以安全地
    把 `hi` 收缩一格（`hi -= 1`）而不会丢失最小值——最小值不可能只存在于被丢弃的
    这一个位置，因为它有一个值相等的"替身" `nums[mid]` 仍然留在区间里。
    【复杂度】时间：平均 O(log n)，最坏 O(n)（当数组几乎全是重复元素时，
    `hi -= 1` 这个分支会被连续触发很多次，退化成线性扫描）；空间 O(1)。
    【易错点】1) 这题相比 LC153（以及 Part I 的 LC33 无重复版）多出的唯一一步
    特殊处理就是 `nums[mid] == nums[hi]` 分支，其余两个分支（`>` 和 `<`）和
    LC153 完全一样，不要误以为要重新设计整个二分逻辑；2) `nums[mid] == nums[hi]`
    时只能安全地收缩 `hi`（`hi -= 1`），不能直接当成"有序，hi = mid"来处理，
    否则在退化数据（比如全部相同）下会漏掉真正的最小值所在位置；3) 正是因为
    这一步只能"一格一格"收缩，最坏复杂度才会从 O(log n) 退化到 O(n)，这是
    "含重复元素"这个条件带来的本质代价，不是实现写得不够好。
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1
        elif nums[mid] < nums[hi]:
            hi = mid
        else:
            hi -= 1
    return nums[lo]


def _self_test() -> None:
    matrix74 = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    assert search_matrix(matrix74, 3) is True
    assert search_matrix(matrix74, 13) is False

    matrix240 = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]
    assert search_matrix_ii(matrix240, 5) is True
    assert search_matrix_ii(matrix240, 20) is False

    assert find_min([3, 4, 5, 1, 2]) == 1
    assert find_min([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_min([11, 13, 15, 17]) == 11

    assert find_peak_element([1, 2, 3, 1]) == 2
    assert find_peak_element([1, 2, 1, 3, 5, 6, 4]) in (1, 5)

    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23

    assert ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
    assert ship_within_days([3, 2, 2, 4, 1, 4], 3) == 6

    assert split_array([7, 2, 5, 10, 8], 2) == 18
    assert split_array([1, 2, 3, 4, 5], 2) == 9

    assert find_min_ii([1, 3, 5]) == 1
    assert find_min_ii([2, 2, 2, 0, 1]) == 0

    print(
        "[PASS] p07_binary_search_ii: 8 题"
        "（搜索二维矩阵/搜索二维矩阵II/寻找旋转排序数组中的最小值/寻找峰值/"
        "爱吃香蕉的珂珂/在D天内送达包裹的能力/分割数组的最大值/"
        "寻找旋转排序数组中的最小值II）全部通过"
    )


if __name__ == "__main__":
    _self_test()
