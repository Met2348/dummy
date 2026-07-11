"""分类 27：线段树与树状数组进阶 —— Phase 3 竞赛级新分类。

前缀和（18 类）只能处理"数组构造后不再变化"的静态场景：一旦出现"边查询区间信息、
边修改数组"的需求，前缀和就失效了——每次修改都要重新花 O(n) 重建整个前缀和数组，
和暴力没有本质区别。树状数组（Fenwick Tree / Binary Indexed Tree）和线段树
（Segment Tree）正是解决"支持修改的区间查询"问题的核心武器：两者都能把"单点更新"
和"区间查询"都做到 O(log n)，树状数组用一段简短的位运算实现，常数更小、代码更短，
但只能维护"满足结合律且有逆运算"的信息（如求和、异或）；线段树用显式的二叉树结构
实现，代码稍长，但更通用——除了区间和，还能维护区间最值、配合"懒标记"做区间批量
更新，也能通过扩展节点信息支持"区间覆盖"这类更复杂的场景（进而支持扫描线等高级
技巧）。这一章把"自己实现" Fenwick 树 / 线段树的能力，和 6 道 LeetCode 上依赖这两种
结构的经典难题串起来。
"""
from __future__ import annotations

import bisect

MOD = 10**9 + 7


class FenwickTree:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：树状数组（Fenwick Tree /
    Binary Indexed Tree）。要求支持 `update(i, delta)`——把下标 i（1-indexed）的值
    累加 delta；`query(i)`——返回前缀 `[1, i]` 的和；`range_query(l, r)`——返回区间
    `[l, r]`（1-indexed，闭区间）的和。目标是让单点更新和前缀/区间查询都比"暴力
    O(n) 重新扫一遍数组"更快。

    【思路】树状数组的核心技巧是用一个长度为 n+1 的数组 `tree`，让 `tree[i]` 恰好
    维护"以 i 结尾、长度为 `i & (-i)`（i 的二进制表示中最低一位 1 所代表的值）的
    一段区间和"。这个精巧的分段方式带来两个性质：1）更新下标 i 时，只需要不断跳到
    "下一个会覆盖到 i 的区间"（`i += i & (-i)`），最多跳 O(log n) 次就能把所有受
    影响的 `tree[j]` 都更新到；2）查询前缀和 `[1, i]` 时，只需要不断跳到"当前区间
    的前一段"（`i -= i & (-i)`），把沿途经过的 `tree[j]` 加起来，也是 O(log n) 步
    就能拼出完整的前缀和——因为任何一个前缀区间 `[1, i]` 都可以被分解成
    O(log n) 个"树状数组式"分段区间的并集（这是二进制拆分的必然结果：i 的二进制
    表示里有多少个 1，前缀 `[1, i]` 就能被分解成多少段）。区间和 `range_query(l, r)`
    直接复用"前缀和相减"（`query(r) - query(l - 1)`），这一步和前缀和（18 类）的
    思路完全一致，树状数组只是把"预处理前缀和数组"换成了"能快速更新的前缀和结构"。
    【复杂度】`update`/`query` 均为 O(log n)；`range_query` 是两次 `query` 相减，
    仍是 O(log n)；空间 O(n)。
    【易错点】
    - 树状数组约定俗成用 **1-indexed**（下标从 1 开始），因为 `i & (-i)` 这个技巧
      依赖"下标为 0 时二进制全 0，`0 & (-0) == 0`，会导致死循环（永远跳不动）"这个
      边界问题——如果不小心把下标 0 传进 `update`/`query`，会陷入死循环，调用方
      必须自己把 0-indexed 的数组下标 +1 转换成 1-indexed 再传进来。
    - `update(i, delta)` 传入的是"变化量" delta（可正可负），不是"新值"——如果要把
      某个位置的值从 old 改成 new，调用方要自己算出 `delta = new - old` 再调用，
      这一点和线段树的 `update_range`（也是传变化量）保持一致，但很容易被误用成
      "直接把 new 传进去"。
    - `range_query(l, r)` 在 `l > r`（空区间）时要返回 0，否则 `query(r) - query(l-1)`
      在这种非法输入下会算出没有意义的负数或异常。
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, l: int, r: int) -> int:
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)


class NumArray:
    """
    【题意】LC307·区域和检索-数组可修改（Medium）。给定整数数组 nums，需要支持两种
    操作交替调用任意多次：`update(index, val)` 把 `nums[index]` 改成 val；
    `sum_range(left, right)` 返回下标 `[left, right]`（闭区间）的和。

    【思路】这是 LC303（18 类，`NumArray` 不可变版本）的进阶版——一旦允许 update，
    "预处理一次前缀和数组"的方案就失效了：每次 update 都要重新花 O(n) 重建整个前缀
    和数组，退化成暴力。用 `FenwickTree` 替代前缀和数组：构造时把 nums 的每个值都
    当作一次 `update` 插入树状数组（下标 +1 转成 1-indexed）；之后 `update(index,
    val)` 只需要算出这个位置的变化量 `val - 当前值`，再调用一次树状数组的 update，
    O(log n)；`sum_range(left, right)` 直接调用 `range_query`，也是 O(log n)。
    另外用一个普通数组 `self.nums` 缓存"当前每个位置的值"，是为了让 update 时能
    O(1) 算出变化量，不需要反查树状数组。

    【复杂度】构造 O(n log n)（n 次插入，每次 O(log n)）；update/sum_range 均为
    O(log n)；空间 O(n)。

    【易错点】
    - 树状数组是 1-indexed，而 nums 的下标是 0-indexed——`update`/`sum_range` 里
      传给 `FenwickTree` 的下标都要 +1，`sum_range(left, right)` 对应
      `range_query(left + 1, right + 1)`，容易漏掉某一处 +1 导致结果整体错位。
    - `update` 必须先算出 `delta = val - self.nums[index]`、再更新 `self.nums
      [index] = val`，顺序反了会导致算出的 delta 永远是 0（用新值减新值）。
    """

    def __init__(self, nums: list[int]) -> None:
        self.nums = list(nums)
        self.fenwick = FenwickTree(len(nums))
        for i, x in enumerate(nums):
            self.fenwick.update(i + 1, x)

    def update(self, index: int, val: int) -> None:
        delta = val - self.nums[index]
        self.nums[index] = val
        self.fenwick.update(index + 1, delta)

    def sum_range(self, left: int, right: int) -> int:
        return self.fenwick.range_query(left + 1, right + 1)


def count_smaller(nums: list[int]) -> list[int]:
    """
    【题意】LC315·计算右侧小于当前元素的个数（Hard）。给定整数数组 nums，返回数组
    counts，其中 `counts[i]` 是 `nums[i]` 右边（下标严格大于 i）比 `nums[i]` 小的
    元素个数。

    【思路】暴力做法是对每个 i 都扫一遍右边所有元素数比自己小的个数，O(n^2)。关键
    转化：**把"值域"当作树状数组的下标，从右往左扫描数组，每插入一个元素之前，先
    查询"比它小的元素已经插入了多少个"**——因为是从右往左扫，"已经插入的元素"恰好
    就是"当前元素右边的所有元素"，这一步查询的结果就是这个位置的答案。具体做法：
    1) 先对 nums 做坐标压缩（因为原始值域可能很大或含负数，树状数组的下标必须是
    连续的正整数）：把所有不同的值排序去重，每个值映射到它在有序序列里的排名
    （1-indexed）；2) 建一棵大小为"不同值个数"的树状数组；3) 从右往左扫描 nums，
    对每个 `nums[i]`，先查询 `fenwick.query(rank - 1)`（排名严格小于当前排名的
    元素，已经插入了多少个）作为 `counts[i]`，再把当前元素的排名插入树状数组
    （`fenwick.update(rank, 1)`）。这样每个位置的查询，看到的都是"树状数组里已经
    插入的、且排名更小"的计数——而由于是从右往左扫，"已经插入"意味着"下标在当前
    位置右边"，两个条件（右边 + 更小）被这一次查询同时满足。

    【复杂度】时间 O(n log n)（坐标压缩排序 O(n log n) + n 次树状数组操作各
    O(log n)）；空间 O(n)（坐标压缩表 + 树状数组）。

    【易错点】
    - 必须严格按"先查询、再插入"的顺序处理每个位置——如果先插入当前元素再查询，
      会把自己也算进"比自己小的个数"里（虽然自己不可能比自己小，但如果数组里有
      重复值，插入顺序错误会导致相同值的元素被错误地计入彼此的答案）。
    - 查询用的是 `rank - 1`（排名严格小于当前排名），不是 `rank`——树状数组的
      `query(rank)` 会连当前排名本身的计数也一起算进去，多算了"和自己相等"的
      元素。
    - 坐标压缩要用 `sorted(set(nums))` 去重后再排序，直接对 `sorted(nums)` 做
      排名会让相同的值获得不同的排名，破坏"值相同 → 排名相同"这个前提。
    """
    sorted_unique = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}
    fenwick = FenwickTree(len(sorted_unique))
    result = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = rank[nums[i]]
        result[i] = fenwick.query(r - 1)
        fenwick.update(r, 1)
    return result


def count_range_sum(nums: list[int], lower: int, upper: int) -> int:
    """
    【题意】LC327·区间和的个数（Hard）。给定整数数组 nums 和两个整数 lower、upper，
    统计"区间和落在 `[lower, upper]`"的（下标）区间个数——区间和 `S(i, j)` 定义为
    `nums[i..j]`（闭区间）所有元素之和。

    【思路】和"和为 K 的子数组"（18 类）同源：设 `prefix[k]` 为 `nums[0..k-1]` 的
    前缀和（`prefix[0] = 0`），任意区间 `nums[i..j]` 的和就是 `prefix[j+1] -
    prefix[i]`。要求这个差落在 `[lower, upper]`，等价于：
    `prefix[j+1] - upper <= prefix[i] <= prefix[j+1] - lower`。于是从左到右扫描
    `prefix` 数组（下标记作 k = 0..n），扫到 k 时，统计"前面（下标 < k）已经出现
    过的 prefix 值里，有多少个落在 `[prefix[k]-upper, prefix[k]-lower]` 这个区间
    内"，累加到答案里，再把 `prefix[k]` 自己计入"已出现过的 prefix 值"集合。这是
    "和为 K 的子数组"从"哈希表精确计数"到"树状数组区间计数"的推广——因为这里要问
    的不是"等于某个定值"而是"落在某个区间"，用哈希表只能精确匹配，必须换成能做
    "区间计数"的树状数组。做法：对整个 prefix 数组做坐标压缩（排序去重、映射排名），
    对每个 k，用 `bisect` 在压缩后的有序值域上定位 `prefix[k]-upper` 和
    `prefix[k]-lower` 对应的排名边界，再用 `fenwick.range_query` 查询"已插入的
    prefix 值里，排名落在这个边界内的个数"。

    【复杂度】时间 O(n log n)（prefix 数组 n+1 个值，坐标压缩排序 O(n log n)，
    每个 k 一次 O(log n) 的树状数组区间查询 + 一次插入）；空间 O(n)。

    【易错点】
    - 查询边界 `[prefix[k]-upper, prefix[k]-lower]` 不是 prefix 数组里的原始值，
      必须用 `bisect_left`/`bisect_right` 在"压缩后的有序值域"上找到对应的排名
      范围，不能直接拿这两个边界值去查 `rank` 字典（因为它们大概率不在 prefix
      数组本身出现过的值集合里）。
    - `bisect_right` 用于上界（`<= high` 的计数是 `bisect_right(sorted_unique,
      high)`）、`bisect_left` 用于下界（`< low` 的计数是 `bisect_left
      (sorted_unique, low)`，用它作为要减去的"前缀计数"）——两者不能对调，否则
      会把边界值本身算重或漏算。
    - 必须先查询"已经插入的、落在区间内的个数"，再把当前 `prefix[k]` 插入树状
      数组——顺序反了会把"空区间 `[k, k]`"（i=j=k 对应的 0 长度区间，没有实际
      意义）错误地计入统计。
    """
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    sorted_unique = sorted(set(prefix))
    fenwick = FenwickTree(len(sorted_unique))

    total = 0
    for p in prefix:
        low_bound = p - upper
        high_bound = p - lower
        hi_pos = bisect.bisect_right(sorted_unique, high_bound)
        lo_pos = bisect.bisect_left(sorted_unique, low_bound)
        total += fenwick.query(hi_pos) - fenwick.query(lo_pos)
        rank = bisect.bisect_left(sorted_unique, p) + 1
        fenwick.update(rank, 1)
    return total


def reverse_pairs(nums: list[int]) -> int:
    """
    【题意】LC493·翻转对（Hard）。给定整数数组 nums，"翻转对"定义为满足
    `0 <= i < j < len(nums)` 且 `nums[i] > 2 * nums[j]` 的下标对 `(i, j)`，返回
    翻转对的总数。

    【思路】和 LC315"计算右侧小于当前元素的个数"是同一个"从右往左扫 + 树状数组
    计数已插入元素"的框架，区别只在于比较条件从"严格小于"换成了"大于两倍关系"。
    从右往左扫描到 `nums[i]` 时，要统计的是"已经插入的（也就是下标在 i 右边的）
    元素里，有多少个满足 `nums[j] < nums[i] / 2`"——为了避免浮点数比较的精度
    问题，把 `nums[j] < nums[i] / 2` 改写成等价的整数不等式
    `nums[j] <= floor((nums[i] - 1) / 2)`（因为 nums[j] 是整数，"严格小于
    nums[i]/2" 和 "小于等于 (nums[i]-1) 除以 2 向下取整" 对任意整数 nums[i]
    都完全等价，包括 nums[i] 为负数的情况——Python 的 `//` 本身就是向下取整除法，
    对负数也成立）。同样对原始 nums 数组做坐标压缩，从右往左扫时，先用 `bisect`
    在压缩值域上定位 `floor((nums[i]-1)/2)` 对应的排名上界，查询树状数组里"已
    插入、排名不超过这个上界"的个数累加进答案，再把 `nums[i]` 自己的排名插入
    树状数组。

    【复杂度】时间 O(n log n)（坐标压缩 O(n log n) + n 次树状数组操作各
    O(log n)）；空间 O(n)。

    【易错点】
    - 直接写成 `nums[j] < nums[i] / 2` 用浮点数比较，在 nums[i] 很大时可能因为
      浮点精度丢失而算错——必须转换成整数形式 `2 * nums[j] < nums[i]`（等价于
      `nums[j] <= (nums[i]-1)//2`），全程用整数运算。
    - `(nums[i] - 1) // 2` 在 nums[i] 为负数时，Python 的 `//` 是向下取整（不是
      向零取整），这恰好是我们需要的语义，但如果心算成 C/Java 那种"向零截断"的
      除法语义，会在负数样例上算错边界。
    - 和 LC315 一样，必须严格"先查询、再插入"当前元素，顺序反了会把不满足
      `i < j` 的情况错误地计入。
    """
    sorted_unique = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}
    fenwick = FenwickTree(len(sorted_unique))

    count = 0
    for i in range(len(nums) - 1, -1, -1):
        threshold = (nums[i] - 1) // 2
        pos = bisect.bisect_right(sorted_unique, threshold)
        count += fenwick.query(pos)
        fenwick.update(rank[nums[i]], 1)
    return count


def create_sorted_array(instructions: list[int]) -> int:
    """
    【题意】LC1649·通过指令创建有序数组（Hard）。从空数组 nums 开始，依次把
    instructions 里的每个数插入 nums（插入后 nums 始终保持有序），每次插入的代价
    是 `min(当前 nums 里严格小于该数的个数, 当前 nums 里严格大于该数的个数)`，
    求所有插入代价之和，对 `1e9+7` 取模。

    【题意补充】这是 LC315/LC493 那套"树状数组统计已插入元素"框架的直接扩展——
    区别在于这里每次插入都要同时统计"比我小的个数"和"比我大的个数"两个方向，
    取较小值累加。

    【思路】用一个大小等于 `max(instructions)` 的树状数组，按顺序（不是从右往左，
    这里天然就是"边插入边查询"，插入顺序就是 instructions 的原始顺序）扫描
    instructions：对每个 `x = instructions[i]`，插入之前，`less =
    fenwick.query(x - 1)` 就是"已经插入的、严格小于 x 的个数"；`greater = i -
    fenwick.query(x)`——`i` 是当前已经插入的元素总数（因为是插入第 i+1 个元素前，
    已经插入了 i 个），`fenwick.query(x)` 是"已插入的、小于等于 x 的个数"，两者
    相减就是"已插入的、严格大于 x 的个数"。取 `min(less, greater)` 累加进答案
    （取模），再把 x 插入树状数组。因为 `instructions[i]` 的值域在题目约束下不大
    （1 到 1e5），这里直接用值本身作为树状数组下标，不需要额外坐标压缩。

    【复杂度】时间 O(n log V)（n 为 instructions 长度，V 为值域上限，每个元素一次
    O(log V) 的插入 + 两次 O(log V) 查询）；空间 O(V)。

    【易错点】
    - `greater` 的计算依赖"当前已插入元素总数 i"，这个数必须是"插入 x 之前"的
      总数（即循环变量 i 本身，从 0 开始），如果先插入再计算 greater，会把 x
      自己也算进"已插入总数"里，导致 `greater` 多减了一次。
    - 取模只需要作用在"累加的总代价"上，每一步的 `min(less, greater)` 本身不会
      超出正常整数范围，不需要每一步都取模，但求和过程中如果不定期取模，Python
      虽然不会溢出（大整数），也没有实际错误，只是不符合"取模运算"的题目要求；
      最终返回值必须对 1e9+7 取模。
    - 树状数组下标要开到 `max(instructions)`（不是 `len(instructions)`），如果
      直接用数组长度当树状数组大小，遇到数值超过数组长度的输入会下标越界。
    """
    max_val = max(instructions)
    fenwick = FenwickTree(max_val)
    total_cost = 0
    for i, x in enumerate(instructions):
        less = fenwick.query(x - 1)
        greater = i - fenwick.query(x)
        total_cost = (total_cost + min(less, greater)) % MOD
        fenwick.update(x, 1)
    return total_cost


def rectangle_area(rectangles: list[list[int]]) -> int:
    """
    【题意】LC850·矩形面积 II（Hard）。给定若干个轴对齐矩形（`[x1, y1, x2, y2]`
    表示左下角和右上角坐标），求它们的并集覆盖的总面积（重叠部分只算一次），对
    `1e9+7` 取模。

    【思路】标准"扫描线 + 线段树"技巧：想象一条水平线从下往上扫过整个平面，扫描线
    在每个 y 位置上，被多少个矩形覆盖，决定了这一瞬间"活跃的 x 区间总长度"；把
    y 方向的变化拆成一系列"事件"——每个矩形在 `y1` 处触发一次"这段 x 区间的覆盖
    次数 +1"，在 `y2` 处触发一次"-1"。按 y 从小到大处理这些事件，每处理完一个
    事件、进入下一个不同的 y 之前，用"当前活跃的 x 覆盖总长度 * y 方向的跨度"累加
    到总面积里。核心数据结构是一棵在 x 方向上做过坐标压缩的线段树，每个节点维护
    两个量：`count`（这个节点对应的 x 区间被多少个矩形"完整覆盖"，即区间加/减的
    懒标记，但这里不需要下推，因为查询永远只问根节点的总覆盖长度）、`covered`
    （这个节点对应的 x 区间里，实际被覆盖的长度）。`covered` 的更新规则是本题的
    关键：如果 `count > 0`（这个节点整个区间都被至少一个矩形完整覆盖），
    `covered` 直接等于这个节点代表的 x 区间长度；否则（`count == 0`），如果是
    叶子节点，`covered = 0`（没有更细粒度的信息了）；如果不是叶子节点，
    `covered` 等于左右子节点 `covered` 之和（虽然这个节点本身没有被"完整"覆盖，
    但子节点内部可能有一部分被覆盖）——这正是线段树处理"区间覆盖计数"问题的经典
    写法，`count` 只在完全被区间更新命中的节点上才 +1/-1，不需要懒标记下推到叶子。

    【复杂度】时间 O(n log n)（n 个矩形产生 2n 个事件，排序 O(n log n)；每个事件
    触发一次 O(log n) 的线段树区间更新）；空间 O(n)（坐标压缩后的 x 轴 + 线段树）。

    【易错点】
    - 线段树的叶子节点对应的不是"一个 x 坐标点"，而是"两个相邻压缩坐标之间的一段
      区间"（`[xs[i], xs[i+1]]`），所以线段树的叶子数是"不同 x 坐标个数 - 1"，
      矩形的 `[x1, x2]` 要先分别映射到坐标压缩后的下标，再转换成"叶子区间下标"
      `[idx(x1), idx(x2) - 1]`（右端点要 -1，因为矩形覆盖的是"格子"而不是"坐标
      点"），这一步 off-by-one 是本题最容易出错的地方。
    - `covered` 在 `count == 0` 时，必须区分"是否是叶子节点"——叶子节点直接置 0，
      非叶子节点要用子节点的 `covered` 之和，如果不加这个区分，会把"内部有一部分
      仍被更小的子矩形覆盖"的情况错误地清零。
    - 事件按 y 排序后，累加面积时用的是"当前 y 和上一个处理过的 y 之间的差"乘以
      "处理这批事件之前（！）线段树根节点的 covered 值"——必须先结算面积、再应用
      当前这批事件的加减，顺序反了会把"还没生效"的覆盖状态提前计入面积。
    - 最终结果要对 1e9+7 取模，矩形坐标范围可以到 1e9，总面积可能远超一般整数
      范围（虽然 Python 不会溢出，但题目要求取模）。
    """
    xs = sorted(set(x for rect in rectangles for x in (rect[0], rect[2])))
    x_index = {x: i for i, x in enumerate(xs)}
    m = len(xs) - 1
    if m <= 0:
        return 0

    count = [0] * (4 * m)
    covered = [0] * (4 * m)

    def update(node: int, node_l: int, node_r: int, l: int, r: int, val: int) -> None:
        if r < node_l or node_r < l:
            return
        if l <= node_l and node_r <= r:
            count[node] += val
        else:
            mid = (node_l + node_r) // 2
            update(node * 2, node_l, mid, l, r, val)
            update(node * 2 + 1, mid + 1, node_r, l, r, val)
        if count[node] > 0:
            covered[node] = xs[node_r + 1] - xs[node_l]
        elif node_l == node_r:
            covered[node] = 0
        else:
            covered[node] = covered[node * 2] + covered[node * 2 + 1]

    events = []
    for x1, y1, x2, y2 in rectangles:
        l, r = x_index[x1], x_index[x2] - 1
        events.append((y1, l, r, 1))
        events.append((y2, l, r, -1))
    events.sort()

    area = 0
    prev_y = events[0][0]
    for y, l, r, val in events:
        area += covered[1] * (y - prev_y)
        update(1, 0, m - 1, l, r, val)
        prev_y = y
    return area % MOD


class RangeModule:
    """
    【题意】LC715·Range 模块（Hard）。设计一个数据结构追踪数轴上的若干"半开区间"
    `[left, right)`：`add_range(left, right)` 把这个区间标记为"被追踪"（和已有
    区间有重叠也要正确合并）；`query_range(left, right)` 判断 `[left, right)`
    内的每一个实数是否都正被追踪；`remove_range(left, right)` 取消追踪这个区间
    （可能把已有区间切成两段）。

    【思路】题目坐标范围可以到 1e9，如果用数组式线段树按坐标建树，数组要开到
    1e9 级别，内存无法接受——真正在竞赛/工程中会用"动态开点线段树"（只在真正被
    访问到的区间才创建节点）解决这个问题，但那需要额外一层"节点动态分配"的实现
    复杂度。这里用一个等价、更轻量的经典替代方案：**维护一个按起点排序、彼此不
    重叠的半开区间列表**，本质上是把"整条数轴的覆盖状态"压缩成"只记录被覆盖的
    连续段"，效果类似于线段树在"区间总是被批量成段覆盖/清除"这种场景下的最终
    效果（活跃区间数量远小于坐标范围）。三个操作都基于同一个二分查找辅助函数
    `_find_first_overlap_or_after(left)`——在按起点有序的区间列表上，二分找到
    "第一个终点 > left 的区间"（即第一个可能和 `[left, ...)` 有交集或排在它之后
    的区间）：`add_range` 从这个位置开始，把所有起点 `<= right` 的区间都合并进
    新区间；`remove_range` 从这个位置开始，把所有起点 `< right` 的区间都拆分
    （区间在 left/right 之外的残留部分重新作为独立区间保留）；`query_range` 直接
    检查这个位置的区间是否完整覆盖 `[left, right)`。

    【复杂度】设当前维护的区间数为 K：`add_range`/`remove_range` 均为 O(K)（最坏
    情况下一次操作要合并/拆分 O(K) 个相邻区间，二分查找本身是 O(log K)）；
    `query_range` 为 O(log K)（只需要二分定位，不需要遍历）。

    【易错点】
    - 区间是**半开**的 `[left, right)`——`query_range(16, 17)` 在恰好把 `[14, 16)`
      移除后仍然返回 True，因为 16 这个点属于剩下的 `[16, 20)`，而不属于被移除的
      `[14, 16)`；如果把区间理解成闭区间，这类边界判断会出错。
    - `remove_range` 拆分区间时，只有当原区间的左端 `s < left` 时才需要保留左边
      残留段 `[s, left)`，只有当原区间右端 `e > right` 时才需要保留右边残留段
      `[right, e)`——两个条件是独立的 if，不是 if/else，一个区间可能同时保留
      左右两段残留（比如 remove 的区间完全被原区间夹在中间）。
    - `add_range` 合并时，新区间的左右端点要取"新区间本身"和"所有被合并的旧区间"
      两者的 min/max，不能想当然地认为新区间一定比旧区间更宽。
    """

    def __init__(self) -> None:
        self.intervals: list[list[int]] = []

    def _find_first_overlap_or_after(self, left: int) -> int:
        lo, hi = 0, len(self.intervals)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.intervals[mid][1] < left:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def add_range(self, left: int, right: int) -> None:
        i = self._find_first_overlap_or_after(left)
        j = i
        new_left, new_right = left, right
        while j < len(self.intervals) and self.intervals[j][0] <= right:
            new_left = min(new_left, self.intervals[j][0])
            new_right = max(new_right, self.intervals[j][1])
            j += 1
        self.intervals[i:j] = [[new_left, new_right]]

    def query_range(self, left: int, right: int) -> bool:
        i = self._find_first_overlap_or_after(left)
        if i >= len(self.intervals):
            return False
        s, e = self.intervals[i]
        return s <= left and right <= e

    def remove_range(self, left: int, right: int) -> None:
        i = self._find_first_overlap_or_after(left)
        j = i
        pieces = []
        while j < len(self.intervals) and self.intervals[j][0] < right:
            s, e = self.intervals[j]
            if s < left:
                pieces.append([s, left])
            if e > right:
                pieces.append([right, e])
            j += 1
        self.intervals[i:j] = pieces


class SegmentTree:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：线段树，支持区间和查询 +
    区间批量更新（懒标记）。要求 `update_range(l, r, val)`——把下标 `[l, r]`
    （0-indexed，闭区间）内每个元素都加上 val；`query_range(l, r)`——返回下标
    `[l, r]` 的元素和。

    【思路】线段树把数组递归地二分成一棵二叉树：每个节点代表一段连续下标区间，
    叶子节点代表单个下标，非叶子节点的值等于左右子节点值的合并（这里是求和）。
    区间更新如果每次都递归到叶子节点逐个修改，退化成 O(n)；用**懒标记**优化：
    当一次区间更新恰好完整覆盖某个节点代表的区间时，不需要往下更新它的子节点，
    只需要在当前节点记一个"懒标记"（表示"这个节点下面的所有元素都还欠着一次
    +val 的更新，还没真正下推"），并直接更新当前节点缓存的区间和（`+= val *
    区间长度`）；只有当后续操作需要真正进入这个节点的子节点时（区间查询或区间
    更新跨越了这个节点的边界），才通过 `_push_down` 把懒标记"下推"一层——子节点
    各自累加这个懒标记、更新各自缓存的区间和，然后清空当前节点的懒标记。这样
    "更新"和"查询"都只需要访问 O(log n) 个节点，且每个节点的下推是均摊的，不会
    重复做无用功。

    【复杂度】构造 O(n)；`update_range`/`query_range` 均为 O(log n)；空间
    O(n)（数组式存储，通常开 4n 大小保证足够）。

    【易错点】
    - 每次递归进入一个节点的子节点之前（无论是 update 还是 query），都必须先
      `_push_down` 检查并下推懒标记——如果只在 update 里下推、query 里忘记下推，
      会读到"还没真正生效"的过期区间和。
    - 懒标记下推时，子节点累加的是"懒标记值 * 子节点对应的区间长度"，不是简单
      的 "+= 懒标记值"——一个常见错误是忘记乘以区间长度，把"区间加"误写成"单点
      加"的逻辑。
    - 数组式线段树的节点数组要开到 `4 * n`（而不是 `2 * n`），因为满二叉树在
      n 不是 2 的幂时，最后一层可能需要额外的空间，`4n` 是保证不越界的经验值。
    """

    def __init__(self, arr: list[int]) -> None:
        self.n = len(arr)
        self.sum = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: list[int], node: int, l: int, r: int) -> None:
        if l == r:
            self.sum[node] = arr[l]
            return
        mid = (l + r) // 2
        self._build(arr, node * 2, l, mid)
        self._build(arr, node * 2 + 1, mid + 1, r)
        self.sum[node] = self.sum[node * 2] + self.sum[node * 2 + 1]

    def _push_down(self, node: int, l: int, r: int) -> None:
        if self.lazy[node] == 0:
            return
        mid = (l + r) // 2
        left_len = mid - l + 1
        right_len = r - mid
        self.lazy[node * 2] += self.lazy[node]
        self.sum[node * 2] += self.lazy[node] * left_len
        self.lazy[node * 2 + 1] += self.lazy[node]
        self.sum[node * 2 + 1] += self.lazy[node] * right_len
        self.lazy[node] = 0

    def update_range(self, l: int, r: int, val: int) -> None:
        self._update(1, 0, self.n - 1, l, r, val)

    def _update(self, node: int, node_l: int, node_r: int, l: int, r: int, val: int) -> None:
        if r < node_l or node_r < l:
            return
        if l <= node_l and node_r <= r:
            self.sum[node] += val * (node_r - node_l + 1)
            self.lazy[node] += val
            return
        self._push_down(node, node_l, node_r)
        mid = (node_l + node_r) // 2
        self._update(node * 2, node_l, mid, l, r, val)
        self._update(node * 2 + 1, mid + 1, node_r, l, r, val)
        self.sum[node] = self.sum[node * 2] + self.sum[node * 2 + 1]

    def query_range(self, l: int, r: int) -> int:
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node: int, node_l: int, node_r: int, l: int, r: int) -> int:
        if r < node_l or node_r < l:
            return 0
        if l <= node_l and node_r <= r:
            return self.sum[node]
        self._push_down(node, node_l, node_r)
        mid = (node_l + node_r) // 2
        return self._query(node * 2, node_l, mid, l, r) + self._query(node * 2 + 1, mid + 1, node_r, l, r)


class SegmentTreeMax:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：线段树的区间最值版本，
    支持 `update(i, val)`——把下标 i（0-indexed）的值改成 val（单点更新）；
    `query_range(l, r)`——返回下标 `[l, r]`（闭区间）内的最大值。

    【思路】和 `SegmentTree`（求和版本）共享同一套"二叉树分治"骨架，区别只在于
    "合并两个子节点信息"的方式从"相加"换成了"取最大值"（`max(左子树最大值, 右
    子树最大值)`），以及因为这里只需要**单点**更新（不是区间批量更新），不需要
    懒标记——单点更新只会影响"从根到这个叶子"这一条路径上的 O(log n) 个节点，
    每个节点更新完直接用子节点的最新值重新算一次 max 即可，不存在"整个子树集体
    延迟更新"的需求。这个对比本身就是一个重要的设计判断力练习：**懒标记只在
    "区间批量更新"场景下才需要，单点更新的线段树永远不需要懒标记**。

    【复杂度】构造 O(n)；`update`/`query_range` 均为 O(log n)；空间 O(n)。

    【易错点】
    - 区间查询如果完全落在当前节点区间之外，要返回"最大值的单位元"负无穷
      （`float("-inf")`），不能返回 0——如果数组里全是负数，返回 0 会把"不存在
      任何有效元素"和"存在一个值为 0 的元素"混淆，导致 `max` 合并出错误的结果。
    - 因为没有懒标记，千万不要在这个类里照搬 `SegmentTree` 的"区间批量更新"
      逻辑——如果直接把区间更新的代码复制过来却不做懒标记下推，会导致"区间"更新
      只真正生效在部分节点上，查询结果不一致。
    - `update(i, val)` 是把值**替换**成 val（不是像 `FenwickTree.update` 那样传
      变化量 delta），调用方如果沿用树状数组那套"传 delta"的直觉调用这个方法，
      会把语义搞反。
    """

    def __init__(self, arr: list[int]) -> None:
        self.n = len(arr)
        self.tree = [float("-inf")] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: list[int], node: int, l: int, r: int) -> None:
        if l == r:
            self.tree[node] = arr[l]
            return
        mid = (l + r) // 2
        self._build(arr, node * 2, l, mid)
        self._build(arr, node * 2 + 1, mid + 1, r)
        self.tree[node] = max(self.tree[node * 2], self.tree[node * 2 + 1])

    def update(self, i: int, val: int) -> None:
        self._update(1, 0, self.n - 1, i, val)

    def _update(self, node: int, l: int, r: int, i: int, val: int) -> None:
        if l == r:
            self.tree[node] = val
            return
        mid = (l + r) // 2
        if i <= mid:
            self._update(node * 2, l, mid, i, val)
        else:
            self._update(node * 2 + 1, mid + 1, r, i, val)
        self.tree[node] = max(self.tree[node * 2], self.tree[node * 2 + 1])

    def query_range(self, l: int, r: int) -> int:
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node: int, node_l: int, node_r: int, l: int, r: int) -> float:
        if r < node_l or node_r < l:
            return float("-inf")
        if l <= node_l and node_r <= r:
            return self.tree[node]
        mid = (node_l + node_r) // 2
        return max(
            self._query(node * 2, node_l, mid, l, r),
            self._query(node * 2 + 1, mid + 1, node_r, l, r),
        )


def _self_test() -> None:
    # ---- FenwickTree：与暴力 O(n) 重新求和的数组交叉验证多组操作序列 ----
    n = 10
    brute = [0] * n
    fenwick = FenwickTree(n)
    ops = [(1, 5), (3, 2), (7, -4), (10, 9), (5, 1), (3, 3), (9, -6), (1, 8), (6, 4), (10, -2)]
    queries = [(1, 10), (2, 8), (1, 1), (10, 10), (3, 7), (5, 9)]
    for i, delta in ops:
        brute[i - 1] += delta
        fenwick.update(i, delta)
        for l, r in queries:
            assert fenwick.range_query(l, r) == sum(brute[l - 1:r]), (i, delta, l, r)

    # ---- LC307 区域和检索-数组可修改 ----
    num_array = NumArray([1, 3, 5])
    assert num_array.sum_range(0, 2) == 9
    num_array.update(1, 2)
    assert num_array.sum_range(0, 2) == 8

    # ---- LC315 计算右侧小于当前元素的个数 ----
    assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert count_smaller([-1, -1]) == [0, 0]
    assert count_smaller([2, 0, 1]) == [2, 0, 0]

    # ---- LC327 区间和的个数 ----
    assert count_range_sum([-2, 5, -1], -2, 2) == 3
    assert count_range_sum([0], 0, 0) == 1

    # ---- LC493 翻转对 ----
    assert reverse_pairs([1, 3, 2, 3, 1]) == 2
    assert reverse_pairs([2, 4, 3, 5, 1]) == 3

    # ---- LC1649 通过指令创建有序数组（三个官方示例） ----
    assert create_sorted_array([1, 5, 6, 2]) == 1
    assert create_sorted_array([1, 2, 3, 6, 5, 4]) == 3
    assert create_sorted_array([1, 3, 3, 3, 2, 4, 2, 1, 2]) == 4

    # ---- LC850 矩形面积 II ----
    assert rectangle_area([[0, 0, 2, 2], [1, 0, 2, 3], [1, 0, 3, 1]]) == 6
    assert rectangle_area([[0, 0, 1000000000, 1000000000]]) == (10**18) % MOD

    # ---- LC715 Range 模块 ----
    rm = RangeModule()
    rm.add_range(10, 20)
    rm.remove_range(14, 16)
    assert rm.query_range(10, 14) is True
    assert rm.query_range(13, 15) is False
    assert rm.query_range(16, 17) is True

    # ---- 自实现 SegmentTree（区间加 + 区间和）：与暴力数组模拟交叉验证 ----
    base = [3, -1, 4, 1, 5, 9, 2, 6]
    brute2 = list(base)
    seg = SegmentTree(base)
    range_ops = [(1, 4, 10), (0, 7, -3), (2, 2, 100), (5, 6, 7), (0, 0, -50), (3, 7, 2)]
    check_ranges = [(0, 7), (1, 3), (2, 2), (0, 0), (4, 6), (5, 5)]
    for l, r, val in range_ops:
        for k in range(l, r + 1):
            brute2[k] += val
        seg.update_range(l, r, val)
        for ql, qr in check_ranges:
            assert seg.query_range(ql, qr) == sum(brute2[ql:qr + 1]), (l, r, val, ql, qr)

    # ---- 自实现 SegmentTreeMax（单点更新 + 区间最大值）：与暴力数组模拟交叉验证 ----
    base_max = [5, 2, 8, 1, 9, 3, 7, 4]
    brute3 = list(base_max)
    seg_max = SegmentTreeMax(base_max)
    point_ops = [(0, 20), (4, -1), (7, 15), (2, 0), (5, 30), (1, 100)]
    check_ranges_max = [(0, 7), (0, 3), (4, 7), (2, 5), (1, 1), (6, 6)]
    for i, val in point_ops:
        brute3[i] = val
        seg_max.update(i, val)
        for ql, qr in check_ranges_max:
            assert seg_max.query_range(ql, qr) == max(brute3[ql:qr + 1]), (i, val, ql, qr)

    print(
        "[PASS] p27_segment_tree_bit: FenwickTree/SegmentTree/SegmentTreeMax + "
        "307/315/327/493/1649/850/715 全部正确"
    )


if __name__ == "__main__":
    _self_test()
