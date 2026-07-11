"""前缀和与差分 · 进阶补充（Part II）：不重讲框架，把前缀和扩展到二维矩阵、取模场景，
把差分数组扩展到更多"区间批量标记"场景，共 7 道题。"""
from __future__ import annotations

import bisect
import random
from collections import defaultdict


class NumMatrix:
    """
    【题意】给定二维整数矩阵 matrix（构造后不再变化），实现 `sum_region(row1, col1,
    row2, col2)`，返回左上角 `(row1, col1)`、右下角 `(row2, col2)` 这个矩形区域
    （闭区间，含边界）内所有元素之和，会被多次调用。

    【思路】这是一维前缀和（`NumArray`，见 p18_prefix_sum.py）向二维的直接推广：
    一维是"前 i 个数的和"，二维是"左上角到 (i-1, j-1) 这个矩形的和"。构造时预计算
    二维前缀和数组 `prefix`（比原矩阵多一行一列，`prefix[0][*]` 和 `prefix[*][0]`
    全为 0 作哨兵），递推式是
    `prefix[i+1][j+1] = prefix[i][j+1] + prefix[i+1][j] - prefix[i][j] + matrix[i][j]`
    ——"上边那一块的和" + "左边那一块的和" - "左上角被重复算了两次的那一块" +
    "当前格子自己"，这正是容斥原理（加两次单独区域、减一次重叠区域）。查询时同样用
    容斥原理反过来推：整个大矩形（从原点到 (row2, col2)）的和，减去"多算的上边一条"
    （从原点到 (row1-1, col2)）、减去"多算的左边一条"（从原点到 (row2, col1-1)），
    再加回"被减了两次的左上角那一块"（从原点到 (row1-1, col1-1)），O(1) 完成。

    【复杂度】构造 O(rows × cols) 时间、O(rows × cols) 空间；每次 `sum_region`
    查询 O(1)。

    【易错点】
    - 前缀和数组要开 `(rows+1) × (cols+1)`，第 0 行、第 0 列全部置 0 作哨兵，否则
      `row1=0` 或 `col1=0` 时需要单独特判边界，容易漏掉或写出 off-by-one。
    - 查询公式的四项容斥关系（加两次减一次），符号写反或者漏掉某一项都会导致结果
      错误，最容易漏的是"加回被减了两次的左上角"这一项。
    - 下标转换：`prefix` 里的下标永远比原矩阵下标"多 1"（因为多了一行一列哨兵），
      查询时用 `row2+1`、`col1`（不是 `col1+1` 时的哨兵位置该用哪个）容易搞混，
      记忆技巧是"要保留的边界用 +1，要排除的边界用原值"。
    """

    def __init__(self, matrix: list[list[int]]):
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
        self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(rows):
            for j in range(cols):
                self.prefix[i + 1][j + 1] = (
                    self.prefix[i][j + 1]
                    + self.prefix[i + 1][j]
                    - self.prefix[i][j]
                    + matrix[i][j]
                )

    def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
        p = self.prefix
        return (
            p[row2 + 1][col2 + 1]
            - p[row1][col2 + 1]
            - p[row2 + 1][col1]
            + p[row1][col1]
        )


class Solution528:
    """
    【题意】给定一个正整数权重数组 w，实现 `pick_index()`：按权重正比的概率随机返回
    一个下标（`w[i]` 越大，下标 i 被选中的概率越高），会被多次调用。

    【思路】"按权重随机选择下标"可以转化成"把 [0, total) 这个区间按权重切成 n 段，
    随机落一个点，看落在哪一段"——这正是前缀和的经典应用："累积和数组" `prefix` 天然
    把权重变成了一系列递增的分段边界（`prefix[i]` 是前 i+1 个权重的累积和），某个随机
    点落在第几段，等价于"prefix 数组里第一个严格大于这个随机点的下标"，可以用二分查找
    （`bisect_right`）在 O(log n) 内定位，而不必线性扫描累积和数组。真正生成随机数的
    部分（`random.random()`）本身不可测试（每次调用结果不同），所以把"给定一个具体的
    随机数、查表定位下标"这部分拆成一个独立的确定性方法 `_pick_from_target`，
    `pick_index` 只是拿真实随机数调用它——这样测试只需要固定几个"模拟随机数"输入，
    验证前缀和 + 二分查找这部分核心逻辑是否正确，不需要对随机性本身做统计断言。

    【复杂度】构造 O(n)（n 为权重个数）；每次 `pick_index`/`_pick_from_target`
    O(log n)（二分查找）。

    【易错点】
    - 二分查找要用 `bisect_right`（找第一个"严格大于"目标值的下标），如果误用
      `bisect_left`，当随机数目标恰好等于某个前缀和边界值时会偏移到错误的段。
    - `target_int = int(target * total)` 这一步的取整方式必须和 `bisect_right`
      配合一致：`target_int` 落在 `[0, total)` 范围内（`target` 属于 `[0, 1)`），
      对应关系是"累积和数组里第一个 > target_int 的位置"，这个对应关系如果换成
      "严格大于等于"会导致边界值被分到相邻的错误一段。
    - 测试不应该依赖 `random.random()` 的具体返回值做断言（真随机不可复现），必须
      像这里一样拆出一个接受确定性输入的内部方法单独测试。
    """

    def __init__(self, w: list[int]):
        self.prefix: list[int] = []
        total = 0
        for x in w:
            total += x
            self.prefix.append(total)
        self.total = total

    def _pick_from_target(self, target: float) -> int:
        target_int = int(target * self.total)
        return bisect.bisect_right(self.prefix, target_int)

    def pick_index(self) -> int:
        return self._pick_from_target(random.random())


def subarrays_div_by_k(nums: list[int], k: int) -> int:
    """
    【题意】给整数数组 nums 和整数 k，求"和能被 k 整除"的连续子数组个数（nums 可能
    含负数）。

    【思路】这是 Part I "和为 K 的子数组"的取模变体，母题结构完全一致——只是判断条件
    从"两个前缀和之差恰好等于 k"换成了"两个前缀和之差能被 k 整除"。关键的同余定理：
    设 `prefix[i]`、`prefix[j]`（i<j）是两个前缀和，子数组 `nums[i..j-1]` 的和
    是 `prefix[j] - prefix[i]`；这个差能被 k 整除，当且仅当
    `prefix[j] % k == prefix[i] % k`（两个数模 k 同余，它们的差必然是 k 的倍数）。
    于是把"求子数组和被 k 整除的个数"转化成"统计前缀和模 k 之后，相同余数出现了多少
    次、每一对相同余数的组合都贡献一个合法子数组"：用哈希表边扫边记录"每种余数目前
    出现过几次"，扫到新位置时先查表加上"当前余数已出现的次数"（这些次数各自对应一个
    以当前位置结尾、和能被 k 整除的子数组），再把当前余数计入表中。Python 的 `%`
    运算对负数结果本身就是非负的（比如 `-3 % 5 == 2`），不需要额外处理成"数学意义上
    的非负余数"，这点和很多其他语言（如 C++/Java 的 `%` 对负数取模可能得到负值）不同。

    【复杂度】时间 O(n)（一趟扫描 + 哈希表 O(1) 查询更新）；空间 O(min(n, k))
    （哈希表最多记 k 个不同余数）。

    【易错点】
    - 哈希表必须预置 `{0: 1}`——对应"从数组开头到当前位置"这一段前缀和本身就能被 k
      整除的情况（这类子数组要求"当前前缀和取模后为 0"，如果不预置，这次计数会漏掉）。
    - 必须先查表统计答案、再把当前余数计入表中——顺序反了会把"空子数组"错误地计入。
    - Python 的负数取模已经是非负结果，不需要像有的教程写的那样再手动 `+= k` 后
      `% k` 两次处理，多此一举（虽然无害，但容易让人误以为 Python 的 `%` 行为和
      C++ 一样需要修正）。
    """
    count: dict[int, int] = defaultdict(int)
    count[0] = 1
    cur = 0
    total = 0
    for x in nums:
        cur = (cur + x) % k
        total += count[cur]
        count[cur] += 1
    return total


def check_subarray_sum(nums: list[int], k: int) -> bool:
    """
    【题意】给整数数组 nums 和整数 k，判断是否存在一个长度至少为 2 的连续子数组，
    其元素和是 k 的倍数（k 的倍数包括 0 本身，比如子数组和恰好为 0 也算）。

    【思路】和 `subarrays_div_by_k` 是同一个"前缀和取模 + 哈希表"母题，但这里只需要
    "存在性"而不是"计数"，所以哈希表记录的是"每种余数第一次出现的下标"而不是"出现
    次数"——一旦同一个余数再次出现，说明这两个下标之间的子数组和能被 k 整除，此时只
    需要检查这两个下标的距离是否 ≥2（题目要求子数组长度至少为 2）。用"第一次出现的
    下标"而不是"最近一次出现的下标"是关键：要让两个下标之间的距离尽量大，才更容易
    满足"长度至少为 2"这个约束，所以同一个余数只需要记一次、以后遇到相同余数时不要
    覆盖更新它的下标。

    【复杂度】时间 O(n)；空间 O(min(n, k))。

    【易错点】
    - 哈希表要预置 `{0: -1}`（表示"前缀和为 0"在下标 -1，也就是"还没开始扫描"时就
      已经出现），这样如果前 m 个数本身的和就能被 k 整除（m≥2），用
      `m - 1 - (-1) = m >= 2` 能正确判断成立。
    - 同一个余数第二次及以后出现时，不能更新它在哈希表里记录的下标——只应该记录
      "第一次"出现的位置，否则会错误地缩小两次出现之间的距离，可能把本该成立的
      情况误判为"长度不足 2"。
    - k 的取值：题目允许 k 为任意正整数，`cur % k` 在 k=0 的情况下会报除零错误，
      但本题约束保证 k≥1，不需要额外处理 k=0 的分支。
    """
    seen: dict[int, int] = {0: -1}
    cur = 0
    for i, x in enumerate(nums):
        cur = (cur + x) % k
        if cur in seen:
            if i - seen[cur] >= 2:
                return True
        else:
            seen[cur] = i
    return False


def car_pooling(trips: list[list[int]], capacity: int) -> bool:
    """
    【题意】trips 中每条 `[numPassengers, from, to]` 表示"从 from 站上车、到 to 站
    下车，car 上此时段多了 numPassengers 名乘客"（左闭右开区间 `[from, to)`）。判断
    整趟行程中车上乘客数是否始终不超过 capacity。

    【思路】和"航班预订统计"（Part I）是同一个"差分数组处理区间批量加"的母题，区别
    只是这里的"区间"是站点区间而不是航班编号区间、且是左闭右开（上车站点开始计入，
    下车站点不再计入）。构造一个差分数组 diff：对每条 `[num, start, end]`，
    `diff[start] += num`（从这一站开始，车上多了这些人）、`diff[end] -= num`
    （到这一站，这些人已经下车，不再计入）。所有记录处理完后，对 diff 求一次前缀和
    还原出"每一站车上实际乘客数"，逐站检查是否超过 capacity，一旦超过立即返回 False；
    全程都不超过则返回 True。

    【复杂度】时间 O(n + m)（m 条行程记录各 O(1) 修改差分数组，n 为站点范围，最后
    O(n) 求前缀和逐站检查）；空间 O(n)（差分数组）。

    【易错点】
    - 站点区间是左闭右开 `[from, to)`——下车站点 to 本身不再计入这些乘客，`diff[to]`
      要减掉（而不是 `diff[to+1]`），如果按闭区间的习惯多减一位会导致 to 这一站的
      人数被错误地提前减少。
    - 差分数组的大小要覆盖所有 trips 里出现过的最大 `to` 值（否则会越界），最简单的
      做法是先扫一遍所有 trips 取最大的 to 决定数组长度。
    - 一旦某一站超过 capacity 应该立刻返回 False（不需要扫完全程），但也不能只检查
      "上车那一刻"就断言超载——必须对差分数组做完整的前缀和累积后逐站检查，因为某一
      站超载可能是多趟行程的乘客同时在车上叠加导致的，不是任何单一一条 trip 记录能
      直接看出来的。
    """
    max_stop = max((t[2] for t in trips), default=0)
    diff = [0] * (max_stop + 1)
    for num, start, end in trips:
        diff[start] += num
        diff[end] -= num
    cur = 0
    for d in diff:
        cur += d
        if cur > capacity:
            return False
    return True


def is_covered(ranges: list[list[int]], left: int, right: int) -> bool:
    """
    【题意】给一批区间 ranges（`[starti, endi]`，闭区间），判断 `[left, right]`
    这个区间内的每一个整数，是否都至少被 ranges 中的某一个区间覆盖到。

    【思路】仍然是"差分数组处理区间批量标记"的母题：与其对每个区间都逐个整数标记
    "被覆盖了一次"（区间可能很长，逐点标记浪费），不如用差分数组只在区间的两个端点
    各改一次——对每个 `[start, end]`（闭区间），`diff[start] += 1`（从这一点开始，
    覆盖计数多 1）、`diff[end + 1] -= 1`（从这一点之后，不再享受这次覆盖，抵消掉）。
    所有区间标记完后，对 diff 求一次前缀和，`cur` 在某个位置的值就是"这个位置被多少
    个区间覆盖"；只需要检查 `[left, right]` 里每个位置的 cur 是否都 ≥1，只要有一个
    位置 cur<=0（一次都没被覆盖），就说明没有完全覆盖，返回 False。

    【复杂度】时间 O(V + m)（V 为坐标值域大小，m 为区间个数）；空间 O(V)。

    【易错点】
    - "减 1"的标记位置是 `end + 1`（因为区间是闭区间，end 这一点本身仍然要被计入
      覆盖），如果写成 `diff[end] -= 1` 会让 end 这一点被提前排除出覆盖范围。
    - 差分数组大小要预留出 `end + 1` 可能达到的最大下标，否则会越界（本题坐标范围
      通常较小，直接按题目给定的值域上界开数组即可）。
    - 只需要检查 `[left, right]` 范围内的点是否都被覆盖，不需要检查这个区间之外的
      点——容易多此一举地去验证整个坐标轴的覆盖情况。
    """
    diff = [0] * 52
    for start, end in ranges:
        diff[start] += 1
        diff[end + 1] -= 1
    cur = 0
    for x in range(1, 51):
        cur += diff[x]
        if left <= x <= right and cur <= 0:
            return False
    return True


def largest_altitude(gain: list[int]) -> int:
    """
    【题意】一条路径起点海拔为 0，`gain[i]` 表示从第 i 个点到第 i+1 个点的海拔变化量
    （可正可负）。求这条路径上出现过的最高海拔（起点的 0 也算作候选之一）。

    【思路】"每一步的海拔"正是"从起点累加变化量"这件事，本质就是一维前缀和——只是
    这里不需要把整个前缀和数组都存下来，因为只关心"历史最大值"，边扫边维护一个当前
    累积海拔 `cur` 和一个历史最大值 `best`（初始都是 0，对应起点）即可，不需要额外
    O(n) 空间存下完整前缀和数组。这是前缀和思想里最轻量的应用形式：**当只需要"某个
    维护中的累积量曾经达到过的极值"、而不需要"任意历史时刻的具体前缀和"时，滚动一个
    变量代替整个前缀和数组就够了**。

    【复杂度】时间 O(n)（一趟扫描）；空间 O(1)。

    【易错点】
    - `best` 的初始值必须是 0（起点海拔），而不是 `gain[0]` 或负无穷——如果所有
      `gain` 都是负数，最高点其实就是起点本身，漏掉这个初始候选会算错。
    - 不需要真的构造完整的前缀和数组再取 `max`——虽然那样做也能得到正确答案，但
      白白多用了 O(n) 空间，这题是练习"什么时候可以用滚动变量代替完整前缀和数组"
      的一个好例子。
    """
    cur = 0
    best = 0
    for g in gain:
        cur += g
        best = max(best, cur)
    return best


def _self_test() -> None:
    nm = NumMatrix(
        [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5],
        ]
    )
    assert nm.sum_region(2, 1, 4, 3) == 8
    assert nm.sum_region(1, 1, 2, 2) == 11
    assert nm.sum_region(1, 2, 2, 4) == 12

    sol528 = Solution528([1, 3])
    assert sol528._pick_from_target(0.1) == 0
    assert sol528._pick_from_target(0.5) == 1

    assert subarrays_div_by_k([4, 5, 0, -2, -3, 1], 5) == 7
    assert subarrays_div_by_k([5], 9) == 0

    assert check_subarray_sum([23, 2, 4, 6, 7], 6) is True
    assert check_subarray_sum([23, 2, 6, 4, 7], 6) is True
    assert check_subarray_sum([23, 2, 6, 4, 7], 13) is False

    assert car_pooling([[2, 1, 5], [3, 3, 7]], 4) is False
    assert car_pooling([[2, 1, 5], [3, 3, 7]], 5) is True

    assert is_covered([[1, 2], [3, 4], [5, 6]], 2, 5) is True
    assert is_covered([[1, 10], [10, 20]], 21, 21) is False

    assert largest_altitude([-5, 1, 5, 0, -7]) == 1
    assert largest_altitude([-4, -3, -2, -1, 4, 3, 2]) == 0

    print(
        "[PASS] p18_prefix_sum_ii: 7 题"
        "（二维区域和检索-矩阵不可变/按权重随机选择/和可被K整除的子数组/"
        "连续的子数组和/拼车/检查是否区域内所有整数都被覆盖/找到最高海拔）全部通过"
    )


if __name__ == "__main__":
    _self_test()
