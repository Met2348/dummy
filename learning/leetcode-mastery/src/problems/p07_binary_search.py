"""分类 07·二分查找 —— 5 道题，从「精确查找」到「区间边界」再到「跨数组切割」。

统一约定：本文件里凡是写成 lower_bound 风格的二分，都用**左闭右开** `[lo, hi)` 写法——
循环不变量是「答案一定在 [lo, hi) 里」，循环结束时 `lo == hi` 就是答案，不需要再判断
"循环退出后 lo 和 hi 谁对谁错"，这是新手最容易纠结的地方。LC704 因为要求"找不到返回 -1"、
且是最经典写法，保留传统的**左闭右闭** `[lo, hi]` 写法作为对照，方便体会两种写法的区别。
"""
from __future__ import annotations


# ── LC704 二分查找（Easy） ────────────────────────────────────────────────
def search(nums: list[int], target: int) -> int:
    """
    【题意】给一个升序数组 nums 和目标值 target，返回 target 的下标；不存在返回 -1。
    【思路】数组有序 → 每次看中点，能排除掉一半区间，是二分查找最原始的形态。这里用
        左闭右闭 [lo, hi]：循环不变量是"答案如果存在，一定在闭区间 [lo, hi] 里"。
        nums[mid] == target 直接返回；nums[mid] < target 说明 mid 及左边都不可能是答案，
        lo = mid + 1；反之 hi = mid - 1。循环条件必须是 lo <= hi（注意等号），因为
        [lo, hi] 是闭区间，lo == hi 时区间里还有一个元素没检查完。
    【复杂度】时间 O(log n)：每轮区间减半。空间 O(1)。
    【易错点】闭区间写法里 while 条件写成 lo < hi（漏了等号）会漏检查最后一个元素；
        mid 越界：Python 不会整数溢出，但其他语言 (lo+hi) 可能溢出，习惯写
        lo + (hi - lo) // 2 更安全。
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# ── LC35 搜索插入位置（Easy） ─────────────────────────────────────────────
def search_insert(nums: list[int], target: int) -> int:
    """
    【题意】升序数组里找 target 的下标；不存在的话，返回"把 target 按顺序插入后它应该
        在的下标"（即数组里第一个 >= target 的位置）。
    【思路】这就是标准的 lower_bound：找"第一个满足 nums[i] >= target 的 i"。用左闭
        右开 [lo, hi)：不变量是"答案一定在 [lo, hi) 里"。nums[mid] < target 说明 mid
        本身及左边都不满足条件，答案在右半开区间 [mid+1, hi)，lo = mid + 1；否则
        （nums[mid] >= target）mid 有可能就是答案，但也可能左边还有更靠前的满足条件
        的位置，所以把 mid 留在候选区间里，hi = mid（不是 mid - 1）。循环结束
        lo == hi 就是答案，因为空区间 [lo, lo) 意味着"再没有更早的候选了"。
    【复杂度】时间 O(log n)，空间 O(1)。
    【易错点】左闭右开写法里如果把 hi 初始化成 len(nums) - 1（当成闭区间），当 target
        比所有元素都大时会漏掉"插入到末尾"这个答案；循环条件必须是 lo < hi（不能带等
        号），否则死循环或多算一轮。
    """
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ── LC34 在排序数组中查找元素的第一个和最后一个位置（Medium） ─────────────
def _lower_bound(nums: list[int], target: int) -> int:
    """第一个 >= target 的下标（不存在则返回 len(nums)）。"""
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _upper_bound(nums: list[int], target: int) -> int:
    """第一个 > target 的下标（不存在则返回 len(nums)）。"""
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def search_range(nums: list[int], target: int) -> list[int]:
    """
    【题意】升序数组（可能有重复元素）里找 target 出现的第一个和最后一个下标，
        组成 [first, last]；不存在则返回 [-1, -1]。
    【思路】"第一个位置"就是 lower_bound(target)；"最后一个位置"就是
        upper_bound(target) - 1（第一个比 target 大的位置往前挪一格）。这题的关键
        insight 是：一次二分只能回答"某个单调谓词第一次成立在哪"，而"第一个 == target"
        和"最后一个 == target"是两个不同的谓词，所以老老实实二分两次，而不是想办法
        一次二分搞定——强行合并成一次反而更容易写错。
    【复杂度】时间 O(log n)（两次二分），空间 O(1)。
    【易错点】lower_bound 求出来之后一定要检查越界和值是否真的等于 target（比如
        target 比所有元素都大，lower_bound 会返回 len(nums)，直接 nums[left] 会
        IndexError）；空数组也要单独能跑通，不要漏判。
    """
    left = _lower_bound(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    right = _upper_bound(nums, target) - 1
    return [left, right]


# ── LC33 搜索旋转排序数组（Medium） ───────────────────────────────────────
def search_rotated(nums: list[int], target: int) -> int:
    """
    【题意】一个原本升序的数组，在某个未知位置整体旋转过（比如 [0,1,2,4,5,6,7]
        旋转成 [4,5,6,7,0,1,2]），数组内无重复元素。在其中找 target 的下标，
        不存在返回 -1，要求 O(log n)。
    【思路】旋转数组整体不是有序的，但有一个关键性质：**任取一个 mid，
        [lo, mid] 和 [mid, hi] 这两段里必然有一段是完全有序的**（另一段可能还是
        被旋转过的）。所以每一步先判断"哪一半是有序的"（比较 nums[lo] 和 nums[mid]：
        nums[lo] <= nums[mid] 说明左半 [lo, mid] 有序，否则右半 [mid, hi] 有序），
        再看 target 是否落在那段**有序**区间的值域里——落在里面就往那一半收缩，
        不在里面就说明 target 只可能在另一半，往另一半收缩。这样每次都能排除一半，
        复杂度依然是 O(log n)，只是收缩方向的判断从"比大小"变成"先定位有序段"。
    【复杂度】时间 O(log n)，空间 O(1)。
    【易错点】判断有序段的边界写错等号：nums[lo] <= nums[mid]（注意是 <=，因为
        lo == mid 时左半只有一个元素，也算"有序"）；判断 target 是否落在有序段值域
        时边界也要用 <=/<  与"有序段是否包含 mid 本身"对齐，否则会把 target 恰好
        等于边界值的情况判错方向。
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                 # 左半 [lo, mid] 有序
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                      # 右半 [mid, hi] 有序
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


# ── LC4 寻找两个正序数组的中位数（Hard） ──────────────────────────────────
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    【题意】给两个各自升序的数组 nums1、nums2，返回把它们合并后（不真的合并）的
        中位数，要求时间复杂度 O(log(m+n))。
    【思路】题目要求的是 log 级别复杂度——真的 merge 两个数组是 O(m+n)，说明这题
        本质不是"排序/合并"问题，而是"**在哪里切一刀，把两个数组分成左右两半，
        使得左半所有数 <= 右半所有数、且左半元素总数等于总长度的一半**"的问题。
        一旦找到这个切割点，中位数就直接由切割点附近的 4 个数算出来，不需要合并。
        为什么能对着数组"切一刀"用二分找：给定"左半应该有几个元素"（由总长度决定，
        是固定值 half），nums1 里切多少个（记 i），nums2 里就必须切 half - i 个（记
        j）——i 和 j 是绑定的，只需要对 i 二分。i 增大 1，nums1 左边界的值
        （nums1[i-1]）单调不减，这保证了二分的单调性：如果 nums1 的左边界比 nums2
        的右边界还大（说明 nums1 切多了），就把 i 往左移；反之往右移。
        为了保证复杂度是 O(log(min(m,n)))，二分永远在**较短的数组**上做（较长的
        数组的切割点 j = half - i 自动确定，不需要额外二分）。
    【复杂度】时间 O(log(min(m, n)))；空间 O(1)。
    【易错点】i 或 j 取到数组边界（i==0 或 i==m）时，对应的"左边界值/右边界值"要
        分别当成 -inf / +inf 处理，忘记这一步会在边界数组越界或比较错误；总长度
        为奇数/偶数时中位数的取法不同（奇数取"左半最大值"，偶数取"左右半分界处
        两个数的平均值"），容易漏掉其中一支；一定要先判断哪个数组更短再二分，
        否则会在较长数组上二分导致复杂度不达标（本题虽然功能上仍然正确，但会
        偏离题目对复杂度的要求）。
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    total = m + n
    half = (total + 1) // 2                        # 左半应该有多少个元素

    lo, hi = 0, m
    while lo <= hi:
        i = lo + (hi - lo) // 2                     # nums1 切出的元素个数
        j = half - i                                # nums2 切出的元素个数

        left1 = nums1[i - 1] if i > 0 else float("-inf")
        right1 = nums1[i] if i < m else float("inf")
        left2 = nums2[j - 1] if j > 0 else float("-inf")
        right2 = nums2[j] if j < n else float("inf")

        if left1 <= right2 and left2 <= right1:
            if total % 2 == 1:
                return float(max(left1, left2))
            return (max(left1, left2) + min(right1, right2)) / 2
        elif left1 > right2:
            hi = i - 1                              # nums1 切多了，往左收缩
        else:
            lo = i + 1                               # nums1 切少了，往右扩大
    raise ValueError("输入不是两个有序数组")


def _self_test() -> None:
    assert search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert search([-1, 0, 3, 5, 9, 12], 2) == -1

    assert search_insert([1, 3, 5, 6], 5) == 2
    assert search_insert([1, 3, 5, 6], 2) == 1
    assert search_insert([1, 3, 5, 6], 7) == 4
    assert search_insert([1, 3, 5, 6], 0) == 0

    assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    assert search_range([], 0) == [-1, -1]

    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search_rotated([1], 0) == -1

    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5

    print("[PASS] p07_binary_search: 5 题（LC704/35/34/33/4）全部通过")


if __name__ == "__main__":
    _self_test()
