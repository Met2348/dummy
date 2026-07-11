"""数组与双指针：一次遍历哈希查找 / 两端夹逼 / 原地覆盖式修改的 6 道代表题。"""
from __future__ import annotations


def two_sum(nums: list[int], target: int) -> list[int]:
    """
    【题意】给定整数数组 nums 和目标值 target，返回和为 target 的两个数的下标（保证恰好一个解，不能重复用同一个元素）。
    【思路】暴力解是两层循环枚举所有 (i, j)，O(n^2)。注意到对每个 nums[i]，我们只是想知道
    "前面是否出现过 target - nums[i] 这个值，出现在哪个下标"——这正是哈希表的强项：一次遍历同时
    做"记录见过的值"和"查询互补值"，把内层循环的线性查找变成 O(1) 查字典，边遍历边建表即可。
    【复杂度】时间 O(n)，空间 O(n)。
    【易错点】1) 不能用同一个元素两次，所以要"先查后存"，保证查到的下标不会是自己；2) 值可能重复
    （如 [3,3]），必须用"值→下标"的字典而不是用值本身去重；3) 只需一次遍历，边遍历边查边建表，不要
    先建完整的值→下标表再单独查一遍（结果一样但多做了一次遍历）。
    """
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []


def remove_duplicates(nums: list[int]) -> int:
    """
    【题意】给定非递减排序数组 nums，原地删除重复项使每个元素只出现一次，返回新长度 k；
    nums 的前 k 个位置必须恰好是去重后的结果（k 之后的内容不作要求）。
    【思路】数组已经有序，意味着重复元素一定相邻，完全不需要哈希表去重。用两个指针：slow 指向
    "下一个该写入的位置"，fast 向右扫描寻找"和上一个保留值不同的新值"。因为是原地覆盖写入，
    slow 永远 <= fast，覆盖 nums[slow] 时不会破坏还没被 fast 读到的值。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 空数组要特判，否则访问 nums[0] 会越界；2) 比较对象是"上一个保留下来的值"
    （nums[slow-1]）而不是"上一个遍历到的原始值"，否则连续三个及以上相同值时容易漏判；
    3) 这题不改变 Python list 的长度，只要求前 k 项正确、返回值是 k，容易误以为要 del 多余元素。
    """
    if not nums:
        return 0
    slow = 1
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow - 1]:
            nums[slow] = nums[fast]
            slow += 1
    return slow


def remove_element(nums: list[int], val: int) -> int:
    """
    【题意】给定数组 nums 和值 val，原地移除所有等于 val 的元素，返回剩余元素个数 k；
    nums 前 k 项包含且只包含剩下的元素，顺序不作要求。
    【思路】既然顺序不重要，就不必像"保留相对顺序"那样把后面元素一个个往前搬——这是双指针的另一
    变体：left 从头扫描，right 指向"当前有效区间的最后一个位置"；遇到 nums[left] 等于 val 时，
    直接把区间最后一个还没检查过的元素换到 left 这个坑位（O(1) 交换），而不是整体搬移 O(n)。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 交换后 left 不能前进，因为刚换过来的新值还没被检查过，必须留在原地重新判断；
    2) 循环条件要用 while left <= right（right 会不断收缩），用固定长度的 for 循环容易多判断已经
    移出有效区间的元素；3) 这题结果顺序不定，测试时要用 sorted() 或 set() 比较，不能直接比较 list。
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        if nums[left] == val:
            nums[left] = nums[right]
            right -= 1
        else:
            left += 1
    return left


def three_sum(nums: list[int]) -> list[list[int]]:
    """
    【题意】给定整数数组，找出所有和为 0 且不重复的三元组 [a,b,c]（下标不同，值可以重复但
    三元组本身不能重复出现在结果里）。
    【思路】暴力解是三层循环枚举 O(n^3)，还要额外处理"怎么判断两个三元组算重复"。关键 insight：
    先排序，排序之后"不重复"就变成了"相邻相同的候选直接跳过"，非常好判断；同时排序还带来另一个
    好处——固定第一个数 nums[i] 后，剩下"找两数之和等于 -nums[i]"退化成了有序数组上的双指针夹逼
    （left/right 从两端向中间收缩），把内层的 O(n^2) 降到 O(n)。
    【复杂度】时间 O(n^2)（排序 O(n log n) 可忽略 + 外层 O(n) * 内层双指针 O(n)），空间 O(log n)
    ~O(n)（排序本身的开销，不计输出）。
    【易错点】1) 三处"跳过重复"都要处理：外层 i 跳过和上一个 i 相同的值、内层 left/right 移动后
    都要跳过和自己刚才相同的值，漏掉任何一处都会出现重复三元组；2) 排序后若 nums[i] > 0 可以直接
    break（后面都更大，三数之和不可能为 0），这是常见剪枝但不是正确性必须；3) 双指针夹逼要用
    while left < right 严格小于，避免两指针撞在一起或越界死循环。
    """
    nums = sorted(nums)
    n = len(nums)
    res: list[list[int]] = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] > 0:
            break
        left, right = i + 1, n - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s < 0:
                left += 1
            elif s > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
    return res


def max_area(height: list[int]) -> int:
    """
    【题意】给定每条竖线的高度数组，任选两条线与 x 轴构成容器，返回能盛的最大水量
    （面积 = 两线中较矮的高度 * 两线下标之差）。
    【思路】暴力解枚举所有 (i,j) 计算面积，O(n^2)。关键 insight：从两端 i=0, j=n-1 开始
    （此时宽度最大），每次只移动"较矮的那一端"——因为面积由较矮的一侧决定，若移动较高的一侧，
    宽度变小而高度上限依然不会超过原来的矮板，面积只会更差；只有移动矮的一侧才有"遇到更高的板
    从而抵消宽度损失"的可能。这样每一步收缩都不会丢失最优解，把 O(n^2) 降到 O(n)。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 面积公式是 min(height[i], height[j]) * (j - i)，容易忘记取 min 而错用较高一侧；
    2) 移动指针时用"严格小于"来判断移动哪一侧，相等时移动哪侧都不影响正确性，但要保证只移动一侧；
    3) 循环结束条件是 left < right，别写成 <=（那样宽度为 0，没有意义）。
    """
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        h = min(height[left], height[right])
        best = max(best, h * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best


def trap(height: list[int]) -> int:
    """
    【题意】给定每根柱子的高度数组（宽度为 1），计算下雨后能接住的雨水总量。
    【思路】暴力解：对每个位置 i，能接的水 = min(左边最高, 右边最高) - height[i]，需要对每个 i
    都各扫一遍左右两边求最高，O(n^2)；用两个数组预存每个位置的"左边最高/右边最高"可以降到 O(n)
    但要额外 O(n) 空间。再进一步的 insight：从两端向中间收缩时，只要 left_max < right_max，就能
    确定"位置 left 这一格的积水只取决于 left_max"（因为右边一定存在比 left_max 更高的柱子挡住），
    根本不需要知道右边具体最高是多少，从而不需要预存数组，把空间降到 O(1)。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 容易一上来就想预处理左右最高数组（也对，但不是最优空间，值得再想一步）；
    2) 边界柱子（最左/最右）不可能积水，双指针写法自然处理了这一点，不需要额外特判；
    3) 更新 left_max/right_max 要在"结算当前格子积水"之前完成，先取 max 再结算，逻辑要自洽。
    """
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    total = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            total += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            total += right_max - height[right]
    return total


def _self_test() -> None:
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]

    nums1 = [1, 1, 2]
    k1 = remove_duplicates(nums1)
    assert k1 == 2 and nums1[:k1] == [1, 2]
    nums2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    k2 = remove_duplicates(nums2)
    assert k2 == 5 and nums2[:k2] == [0, 1, 2, 3, 4]

    nums3 = [3, 2, 2, 3]
    k3 = remove_element(nums3, 3)
    assert k3 == 2 and sorted(nums3[:k3]) == [2, 2]
    nums4 = [0, 1, 2, 2, 3, 0, 4, 2]
    k4 = remove_element(nums4, 2)
    assert k4 == 5 and sorted(nums4[:k4]) == [0, 0, 1, 3, 4]

    assert sorted(three_sum([-1, 0, 1, 2, -1, -4])) == sorted([[-1, -1, 2], [-1, 0, 1]])
    assert three_sum([0, 1, 1]) == []
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]

    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert max_area([1, 1]) == 1

    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap([4, 2, 0, 3, 2, 5]) == 9

    print("[PASS] p01_arrays_two_pointers: 6 题（两数之和/删除重复项/移除元素/三数之和/盛水容器/接雨水）全部通过")


if __name__ == "__main__":
    _self_test()
