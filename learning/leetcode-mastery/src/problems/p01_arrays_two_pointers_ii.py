"""数组与双指针 · 进阶补充（Part II）：不重讲框架，扩大双指针变体覆盖面的 9 道题。"""
from __future__ import annotations


def four_sum(nums: list[int], target: int) -> list[list[int]]:
    """
    【题意】给定整数数组 nums 和目标值 target，返回所有和为 target 且不重复的四元组 [a,b,c,d]
    （下标互不相同）。
    【思路】是三数之和的直接延伸：排序后再多套一层外层循环固定第一个数，内层退化成"三数之和找
    -nums[i]"，而三数之和内部又是"固定第二个数 + 双指针夹逼"——四层嵌套里只有最内层是真正的
    双指针，外面两层都是"固定一个数，把问题降一维"。
    【复杂度】时间 O(n^3)（两层固定 + 双指针夹逼），空间 O(log n)~O(n)（排序开销，不计输出）。
    【易错点】1) 两层外层循环（i 和 j）都要各自做"跳过重复"处理，且 j 的跳重复要满足
    `j > i + 1`（不能跳到比 i 还靠前的位置比较）；2) 内层双指针命中后同样要对 left/right
    各自跳过重复值，四处去重逻辑漏一处都会出现重复四元组；3) 数据范围可能包含很大的数导致
    int 相加溢出（Python 不存在这个问题，但换成 Java/C++ 实现时要注意用 long 存中间和）。
    """
    nums = sorted(nums)
    n = len(nums)
    res: list[list[int]] = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            left, right = j + 1, n - 1
            while left < right:
                s = nums[i] + nums[j] + nums[left] + nums[right]
                if s < target:
                    left += 1
                elif s > target:
                    right -= 1
                else:
                    res.append([nums[i], nums[j], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
    return res


def three_sum_closest(nums: list[int], target: int) -> int:
    """
    【题意】给定整数数组 nums 和目标值 target，找出和最接近 target 的三个数，返回这三个数的和
    （假设恰好只有一个最优解）。
    【思路】和三数之和同一套排序 + 双指针骨架，但这题不是"找到等于 0 就收集"，而是"每算出一个和
    就和当前最优比较差距，差距更小就更新"；双指针的移动方向依据也从"是否等于 0"换成"是否等于
    target"：和小于 target 就右移 left 让和变大，和大于 target 就左移 right 让和变小，恰好相等
    时可以直接提前返回（差距为 0，不可能更优）。
    【复杂度】时间 O(n^2)，空间 O(log n)~O(n)（排序开销）。
    【易错点】1) 最优解的初始值不能随便设成 0 或者一个哨兵大数，要先算一个真实存在的三元组和
    （比如前三个数的和）当初始值，否则可能被一个根本不存在的"更优解"污染；2) 比较"哪个更接近"
    要用 abs(s - target)，容易漏掉 abs 直接比较差值本身；3) 这题不需要跳过重复值——因为只要
    返回一个和，不要求返回具体三元组、更不要求去重，跳重复反而可能跳过更优解。
    """
    nums = sorted(nums)
    n = len(nums)
    best = nums[0] + nums[1] + nums[2]
    for i in range(n - 2):
        left, right = i + 1, n - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if abs(s - target) < abs(best - target):
                best = s
            if s < target:
                left += 1
            elif s > target:
                right -= 1
            else:
                return s
    return best


def triangle_number(nums: list[int]) -> int:
    """
    【题意】给定非负整数数组 nums，统计其中能组成三角形的三元组个数（任意两边之和大于第三边）。
    【思路】排序后，只要固定最大边 nums[k]，就只需要判断"较小两边之和是否大于 nums[k]"——这时候
    双指针的收获点变成"批量计数"而不是"找到一组就都收集"：固定 k 后 left/right 在 [0, k-1] 内
    夹逼，若 nums[left]+nums[right] > nums[k]，说明 [left, right-1] 里任意一个数配合 right 都能
    成立（因为排序后更靠左的数只会更小，仍然满足或更容易满足"和大于 nums[k]"这个不等式的方向
    需要重新想清楚：这里固定的是 right，让 left 在 [left, right) 区间内都成立），一次性
    count += right - left，然后收缩 right；否则说明 nums[left] 太小，只能右移 left 换更大的数。
    【复杂度】时间 O(n^2)（排序 O(n log n) 可忽略 + 外层固定最大边 O(n) * 内层双指针 O(n)），
    空间 O(log n)~O(n)。
    【易错点】1) 外层要从最大的 k 开始固定，内层双指针在 [0, k-1] 里收缩，若固定成最小边反而
    没有这种单调批量计数的性质；2) 命中条件是 nums[left] + nums[right] > nums[k] 时一次性
    count += right - left（而不是 +1 再逐个移动），这是本题双指针相比暴力 O(n^3) 提速的关键；
    3) 数组中可能有 0，三边中有一条为 0 无法组成三角形，排序后 0 会被自然过滤（0 加任何数都不
    可能大于一个正数最大边，除非最大边也是 0，这时同样不构成三角形），不需要额外特判。
    """
    nums = sorted(nums)
    n = len(nums)
    count = 0
    for k in range(n - 1, 1, -1):
        left, right = 0, k - 1
        while left < right:
            if nums[left] + nums[right] > nums[k]:
                count += right - left
                right -= 1
            else:
                left += 1
    return count


def next_permutation(nums: list[int]) -> None:
    """
    【题意】原地将 nums 修改为它的下一个字典序排列；如果已经是最大排列，则修改为最小排列
    （即完全升序）。不返回值，直接原地修改。
    【思路】想让排列变大且变化幅度最小，直觉是"尽量保留尽可能长的后缀不变，只在后缀的前一个
    位置做一次小改动"。从右往左找第一个满足 nums[i] < nums[i+1] 的位置 i（这个位置之后的后缀
    已经是降序、即当前后缀的最大排列），说明"从 i 开始变大"是最小的改动位置；在后缀里找到比
    nums[i] 大的最小值和它交换（双指针视角：一左一右两个位置的定位 + 交换），换完之后后缀依然
    降序，需要整体反转后缀使其变成升序（后缀的最小排列），这样"i 位置变大 + 后缀取最小"合起来
    就是全局意义上最小的"下一个更大排列"。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 如果从右往左找不到满足 nums[i] < nums[i+1] 的位置（整个数组降序，已是最大
    排列），要特判直接整体反转变成最小排列，容易忘记这个分支；2) 找"比 nums[i] 大的最小值"时
    因为后缀本身降序，从右往左找到第一个大于 nums[i] 的值就是最小的那个，不需要额外排序；
    3) 交换之后必须反转后缀而不是保持原样，很多人只做了交换就以为结束了，但后缀此时仍是降序
    （当前后缀能构成的最大值），不反转的话结果不是"下一个"排列而是"很靠后的一个"排列。
    """
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = reversed(nums[i + 1:])


def sort_colors(nums: list[int]) -> None:
    """
    【题意】给定只含 0、1、2 的数组，原地排序使相同颜色相邻，且按 0、1、2 顺序排列（荷兰国旗
    问题），不能使用额外的计数排序数组、要求一趟遍历原地完成。
    【思路】用三个指针把数组分成四段：[0, left) 确定是 0，[left, cur) 确定是 1，[cur, right]
    还未处理，(right, n) 确定是 2。cur 负责逐个读取未处理的元素：遇到 0 就和 left 位置交换（把
    0 甩到已确定区的末尾）并将 left、cur 都前进；遇到 2 就和 right 位置交换（把 2 甩到已确定区
    开头）但 cur 不能前进（因为换过来的新值还没检查过）；遇到 1 就直接前进。是"删除有序数组
    重复项"那类双指针分工的三路加强版。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 和 right 交换后 cur 绝对不能前进，换过来的值可能还是 2 或者需要继续处理的值，
    这是本题最容易犯的错（和处理 0 的分支搞混，误以为两个分支都要前进）；2) 循环条件要用
    `cur <= right` 而不是 `cur < right`（否则会漏掉最后一个未处理的位置）；3) 每次交换后不需要
    重新检查交换过去的旧值是否越界，因为交换只发生在已确定区和未处理区的边界上，天然安全。
    """
    left, cur, right = 0, 0, len(nums) - 1
    while cur <= right:
        if nums[cur] == 0:
            nums[left], nums[cur] = nums[cur], nums[left]
            left += 1
            cur += 1
        elif nums[cur] == 2:
            nums[cur], nums[right] = nums[right], nums[cur]
            right -= 1
        else:
            cur += 1


def first_missing_positive(nums: list[int]) -> int:
    """
    【题意】给定未排序整数数组，找出其中没有出现的最小正整数，要求时间 O(n)、空间 O(1)。
    【思路】答案必然落在 [1, n+1] 区间内（n 个数最多能填满 1..n，若都填满答案就是 n+1）。既然
    答案范围有限，可以把数组本身当作一张"值→位置"的哈希表：把每个满足 1<=x<=n 的值 x 尽量换到
    下标 x-1 的位置上（原地哈希，本质是双指针"读/写分离"思想的变体——用交换代替额外的哈希表
    存储）。换完之后从头扫描，第一个"nums[i] != i+1"的位置，i+1 就是缺失的最小正数。
    【复杂度】时间 O(n)（虽然内层是 while 循环，但每次交换都会让至少一个数归位，归位后不会再被
    换出去，所以总交换次数不超过 n），空间 O(1)（原地复用输入数组，不额外开数组或哈希表）。
    【易错点】1) while 循环条件必须同时检查值域 `1 <= nums[i] <= n` 和"目标位置的值还不是自己"
    （`nums[nums[i]-1] != nums[i]`），漏掉后者会在有重复值时死循环（比如 [1,1] 会不断交换同一对
    值）；2) 交换的目标下标是 `nums[i] - 1` 而不是 `nums[i]`，一次下标偏移写错整个算法就错；
    3) 最后扫描的返回值是 `i + 1` 而不是 `i`，容易在这里丢一个 1 的偏移。
    """
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            target = nums[i] - 1
            nums[i], nums[target] = nums[target], nums[i]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def find_duplicate(nums: list[int]) -> int:
    """
    【题意】给定长度为 n+1、值域在 [1, n] 的数组（因此必然恰好有一个值重复，可能重复多次），
    要求不修改数组、只用 O(1) 额外空间，找出这个重复的值。
    【思路】把 nums[i] 看成"从节点 i 指向节点 nums[i]"的一条有向边，因为值域是 [1, n] 而数组
    长度是 n+1，必然存在两个不同下标指向同一个值，也就是链表视角下"有一个入度为 2 的节点"——
    这意味着从任意起点出发不断跳 `i -> nums[i]` 必然会进入一个环，且环的入口正是那个重复值。
    这就把问题转化成了经典的 Floyd 判圈算法（快慢指针找链表环入口），是双指针思想在"数组模拟
    链表"场景下的延伸，而不是在真正的数组下标区间上做双指针。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 一定不能修改数组来做标记（比如把访问过的位置取负号），题目明确要求不能修改；
    2) 判圈分两阶段：先用快慢指针（slow 走一步、fast 走两步）找到相遇点，再让一个指针回到起点、
    两个指针都改成每次走一步，第二次相遇的位置才是环入口（重复值），漏掉第二阶段直接返回第一次
    相遇点的值是常见错误；3) 起点 slow=fast=nums[0] 的初始化容易和"下标 0"混淆，这里 nums[0]
    是第一步要跳到的节点，不是下标本身。
    """
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow2 = nums[0]
    while slow2 != slow:
        slow2 = nums[slow2]
        slow = nums[slow]
    return slow


def merge_sorted_array(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """
    【题意】nums1、nums2 都是非递减排序数组，nums1 的长度是 m+n（前 m 个是有效元素，后 n 个是
    占位的 0），把 nums2 的 n 个元素合并进 nums1，使 nums1 长度为 m+n 时整体有序。原地修改，
    不返回值。
    【思路】如果像归并排序那样从前往后合并，写入位置会覆盖 nums1 里还没读到的有效元素（因为
    nums1 前 m 位既是"读"的数据源又是"写"的目标）。关键 insight 是反过来：从两个数组的**末尾**
    开始比较，把较大的值写到 nums1 的**末尾**——nums1 后面 n 个位置本来就是占位的 0，从后往前写
    永远不会覆盖还没被读取的前部有效数据，这是"双指针方向"服务于"原地写不覆盖"这一约束的典型
    例子。
    【复杂度】时间 O(m+n)，空间 O(1)。
    【易错点】1) 三个指针 i（nums1 有效区末尾）、j（nums2 末尾）、k（nums1 总末尾）都要从
    右往左走，写反成从左往右会覆盖数据；2) 循环结束条件只需要 `j >= 0`（nums2 还有剩余），
    因为 nums1 剩余的元素已经在正确位置上，不需要再搬；3) i 可能提前耗尽（nums2 的元素都比
    nums1 剩余的大），此时循环里 `if i >= 0 and nums1[i] > nums2[j]` 的短路判断要先检查
    `i >= 0`，否则会越界访问 nums1[-1] 之类的错误下标（Python 里 nums1[-1] 不会报错但语义错误，
    在别的语言里会直接越界）。
    """
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1


def remove_duplicates_ii(nums: list[int]) -> int:
    """
    【题意】给定非递减排序数组 nums，原地删除多余的重复项，使每个元素最多出现两次，返回新长度
    k；nums 前 k 项必须是结果。
    【思路】是"删除有序数组中的重复项"（每个元素最多出现一次）的直接推广：那题判断"是否写入"
    看的是"当前值是否等于上一个保留值"（nums[slow-1]），这题把比较对象从"前一个保留值"改成
    "前两个保留值"（nums[slow-2]）——因为允许保留两份，只有当前值和"倒数第二个保留位置"的值
    相同时，才说明这是第三次出现，才需要跳过；slow/fast 的读写分工完全不变。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) slow 的初始值要设为 2（而不是 1），因为前两个元素永远可以直接保留，不需要
    判断；2) 数组长度小于等于 2 时要提前返回 n，直接走后面的循环逻辑会因为 slow-2 变成负数
    下标而访问到错误的比较对象；3) 比较对象是 nums[slow-2] 而不是 nums[fast-2]——slow 才是
    "已保留区间"的游标，fast-2 指向的是原始输入里的旧位置，语义完全不同，写混是这题最典型的
    bug。
    """
    n = len(nums)
    if n <= 2:
        return n
    slow = 2
    for fast in range(2, n):
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    return slow


def _self_test() -> None:
    assert sorted(four_sum([1, 0, -1, 0, -2, 2], 0)) == sorted(
        [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    )
    assert four_sum([2, 2, 2, 2, 2], 8) == [[2, 2, 2, 2]]

    assert three_sum_closest([-1, 2, 1, -4], 1) == 2
    assert three_sum_closest([0, 0, 0], 1) == 0

    assert triangle_number([2, 2, 3, 4]) == 3
    assert triangle_number([4, 2, 3, 4]) == 4

    nums_p1 = [1, 2, 3]
    next_permutation(nums_p1)
    assert nums_p1 == [1, 3, 2]
    nums_p2 = [3, 2, 1]
    next_permutation(nums_p2)
    assert nums_p2 == [1, 2, 3]
    nums_p3 = [1, 1, 5]
    next_permutation(nums_p3)
    assert nums_p3 == [1, 5, 1]

    nums_c1 = [2, 0, 2, 1, 1, 0]
    sort_colors(nums_c1)
    assert nums_c1 == [0, 0, 1, 1, 2, 2]
    nums_c2 = [2, 0, 1]
    sort_colors(nums_c2)
    assert nums_c2 == [0, 1, 2]

    assert first_missing_positive([1, 2, 0]) == 3
    assert first_missing_positive([3, 4, -1, 1]) == 2
    assert first_missing_positive([7, 8, 9, 11, 12]) == 1

    assert find_duplicate([1, 3, 4, 2, 2]) == 2
    assert find_duplicate([3, 1, 3, 4, 2]) == 3

    nums1_a = [1, 2, 3, 0, 0, 0]
    merge_sorted_array(nums1_a, 3, [2, 5, 6], 3)
    assert nums1_a == [1, 2, 2, 3, 5, 6]
    nums1_b = [1]
    merge_sorted_array(nums1_b, 1, [], 0)
    assert nums1_b == [1]

    nums_d1 = [1, 1, 1, 2, 2, 3]
    k1 = remove_duplicates_ii(nums_d1)
    assert k1 == 5 and nums_d1[:5] == [1, 1, 2, 2, 3]
    nums_d2 = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    k2 = remove_duplicates_ii(nums_d2)
    assert k2 == 7 and nums_d2[:7] == [0, 0, 1, 1, 2, 3, 3]

    print(
        "[PASS] p01_arrays_two_pointers_ii: 9 题"
        "（四数之和/最接近的三数之和/有效三角形个数/下一个排列/颜色分类/"
        "缺失的第一个正数/寻找重复数/合并有序数组/删除有序数组重复项II）全部通过"
    )


if __name__ == "__main__":
    _self_test()
