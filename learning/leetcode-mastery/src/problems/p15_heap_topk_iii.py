"""15 堆/Top-K · 竞赛级补充(Part III)：Frontier Lab 面试公认的"堆经典难题"——
三维接雨水的边界收缩堆、扫描线天际线、多路归并求第k小、堆+贪心置换、
延迟队列重排字符串，不重复 Part I/II 的骨架，只挑竞赛级题目。"""
from __future__ import annotations

import heapq
from collections import Counter, deque


# ── LC407 接雨水II(Hard) ─────────────────────────────────────────────────
def trap_rain_water_2d(height_map: list[list[int]]) -> int:
    """
    【题意】给定一个 m x n 的二维高度图 heightMap，每个格子是一个单位宽度的
    柱子，求下雨之后这块地形总共能接住多少单位的水（边界格子的水会直接流走，
    无法存水）。
    【思路】二维接雨水不能像一维接雨水(LC42)那样用两个指针从数组两端向中间
    收缩——一维只有左右两侧边界，能明确"当前该处理哪一侧的哪个位置"；二维的
    "边界"是整整一圈，任何一个内部格子的储水量取决于**环绕它一整圈的最低点**，
    而不是简单地取"左边界最大值"和"右边界最大值"两个方向。正确做法是"从最矮
    的边界格子开始，像剥洋葱一样一圈一圈往里收缩"（一种"木桶效应"：一圈围栏
    能存多少水取决于最矮的那根栏杆）：把最外一圈的所有格子连同高度放进一个
    最小堆，每次弹出堆里**当前已知边界中最矮**的那个格子——因为水面高度不可能
    超过"环绕这片区域的最低缺口"，这个最矮格子就是当前能确定的、真正制约储水
    上限的短板。处理它的 4 个邻居：如果邻居没被访问过，它的储水量就是
    `max(0, 当前最矮高度 - 邻居高度)`；无论邻居本身高度是多少，都要把邻居以
    `max(当前最矮高度, 邻居高度)` 作为新的有效高度重新推入堆——这代表"从这个
    邻居往更深处看，能确定的水位下限"单调不减地向内扩散，这也是必须用堆而不能
    用普通队列（BFS）的原因：展开顺序必须严格按"当前已知边界里最矮的先处理"，
    否则会有格子被更高的水位提前错误地"定型"，把它内侧本该更小的储水量覆盖成
    偏大的估计值。
    【复杂度】时间 O(mn log(mn))（每个格子入堆出堆各一次），空间 O(mn)
    （visited 数组 + 堆）。
    【易错点】① 更新邻居的有效高度必须取 `max(当前弹出高度, 邻居原始高度)`
    而不是直接用邻居原始高度——如果邻居比当前弹出的高度还高，它应该以自己的
    真实高度参与后续比较，而不是被错误地"拉低"；② 容易把这题误当成一维接雨水
    的简单推广，尝试对每一行/每一列各自跑一次一维接雨水——这是错误的，二维
    环绕结构无法分解成行、列独立的一维子问题，必须用"边界最小堆向内扩散"。
    """
    if not height_map or not height_map[0]:
        return 0
    m, n = len(height_map), len(height_map[0])
    if m < 3 or n < 3:
        return 0
    visited = [[False] * n for _ in range(m)]
    heap: list[tuple[int, int, int]] = []
    for i in range(m):
        for j in range(n):
            if i in (0, m - 1) or j in (0, n - 1):
                heapq.heappush(heap, (height_map[i][j], i, j))
                visited[i][j] = True
    water = 0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and not visited[ni][nj]:
                visited[ni][nj] = True
                water += max(0, h - height_map[ni][nj])
                heapq.heappush(heap, (max(h, height_map[ni][nj]), ni, nj))
    return water


# ── LC218 天际线问题(Hard) ────────────────────────────────────────────────
def get_skyline(buildings: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定若干矩形建筑 buildings[i] = [left, right, height]，求这些
    建筑从远处看形成的轮廓线，用一系列"关键点" [x, y] 表示（每个关键点是轮廓
    某条水平线段的左端点，最后一个点的 y 恒为 0，表示轮廓终止；输出中不能有
    连续两个高度相同的水平线段）。
    【思路】扫描线 + 最大堆：把每栋建筑拆成两个"事件"——在 left 处是"建筑
    开始"事件（带着它的高度和结束位置），在 right 处是"建筑结束"事件。把所有
    事件按 x 坐标排序后从左到右扫描，同一时刻"当前轮廓的高度"就是"所有还没
    结束的建筑里最高的那个"——这正是一个需要支持"插入新高度"和"查询当前最大值"
    的动态最值问题，用一个最大堆（存 (-高度, 结束位置) 使其可以按最小堆实现）
    再配合**懒删除**（堆顶如果对应的结束位置 <= 当前扫描到的 x，说明这栋楼已
    经结束了，直接弹出丢弃，不需要真的从堆中间删除）来维护。每处理完一个 x
    坐标的所有事件后，如果"懒删除后的堆顶高度"和上一个记录的轮廓高度不同，
    就说明轮廓在这里发生了台阶变化，记录一个新的关键点。为了让"地面"这条
    基线恒定存在，堆里预置一个 (0, +inf) 的哨兵，保证堆永远非空、且高度降到
    0 时能正确闭合轮廓。
    【复杂度】时间 O(n log n)（n 为建筑数量，排序 + 每栋楼的两个事件各触发
    O(log n) 的堆操作），空间 O(n)。
    【易错点】① 懒删除必须用 `while` 循环清理堆顶所有已经过期的记录，而不是
    只检查一次——同一时刻可能有多栋楼同时结束；② 同一个 x 坐标既有"开始"
    事件又有"结束"事件时，必须先处理完这个 x 坐标上的**所有**事件、让堆状态
    稳定下来之后，再和"上一个记录的高度"比较一次决定是否要产生新关键点，
    如果对每个事件都单独比较一次，会在同一 x 坐标下错误地插入多个关键点。
    """
    events = sorted(
        {(l, -h, r) for l, r, h in buildings} | {(r, 0, 0) for l, r, h in buildings}
    )
    result: list[list[int]] = [[0, 0]]
    live: list[tuple[int, float]] = [(0, float("inf"))]  # (-高度, 结束位置)
    for x, neg_h, r in events:
        while live[0][1] <= x:
            heapq.heappop(live)
        if neg_h:
            heapq.heappush(live, (neg_h, r))
        if result[-1][1] != -live[0][0]:
            result.append([x, -live[0][0]])
    return result[1:]


# ── LC1439 有序矩阵中的第k个最小数组和(Hard) ──────────────────────────────
def kth_smallest_array_sum(mat: list[list[int]], k: int) -> int:
    """
    【题意】给定一个每行都非递减排序的 m x n 矩阵 mat，每行恰好选一个元素
    组成一个长度为 m 的数组，求所有这样的组合中，数组和第 k 小的那个值。
    【思路】这是 LC373(查找和最小的K对数字) 的多路推广："逐行合并、每次只
    保留前 k 小"：维护一个候选和列表 sums（初始为 [0]），每处理一行，就把
    "当前 sums 里的每个候选和" 与 "这一行的元素" 做一次多路归并——这一步本身
    又是"合并 len(sums) 条有序序列（每条序列固定用 sums[i]，只让行内下标 j
    递增）"的堆技巧：初始把每条序列的第一个候选 (sums[i]+row[0], i, 0) 放进
    堆，每弹出一个候选就加入结果，同时把同一条序列的下一个候选（行内 j+1）
    补进堆，直到凑够 k 个或者堆耗尽。合并完这一行后，sums 更新为这至多 k 个
    最小和，继续和下一行合并。因为矩阵每行本身已经有序，每条"序列"（固定用
    某个旧候选和 + 这一行递增的元素）天然是非递减的，多路归并取前 k 小是
    正确的；而且每一步都只保留 min(k, 剩余组合数) 个候选，避免了组合数随行数
    指数爆炸。
    【复杂度】时间 O(m * k log k)（m 行，每行归并最多产出 k 个新候选，每次
    堆操作 O(log k)），空间 O(k)。
    【易错点】① 每处理完一行都必须把 sums 裁剪到最多 k 个——如果不裁剪，
    sums 的长度会随行数指数增长（第 i 行处理后长度是 min(k, n^i)），退化成
    枚举全部组合；② 归并时容易把"沿用哪个旧候选 sums[i]"和"行内前进到哪个
    下标 j"两个维度弄混，堆里的三元组必须同时记录两者才能正确定位下一个
    候选。
    """
    sums = [0]
    for row in mat:
        heap: list[tuple[int, int, int]] = [
            (sums[i] + row[0], i, 0) for i in range(len(sums))
        ]
        heapq.heapify(heap)
        merged: list[int] = []
        while heap and len(merged) < k:
            total, i, j = heapq.heappop(heap)
            merged.append(total)
            if j + 1 < len(row):
                heapq.heappush(heap, (sums[i] + row[j + 1], i, j + 1))
        sums = merged
    return sums[-1]


# ── LC1675 数组的最小偏移量(Hard) ─────────────────────────────────────────
def minimum_deviation(nums: list[int]) -> int:
    """
    【题意】数组 nums 全为正整数，每个元素可以任意次执行：偶数除以 2，或者
    奇数乘以 2。数组的"偏移量"定义为数组最大值和最小值之差，求经过若干次操作
    后能达到的最小偏移量。
    【思路】关键观察：奇数只能变大（乘 2 变成偶数），而偶数可以不断变小
    （除 2 直到变成奇数）——所以每个数在整个操作空间里能到达的"上界"，就是
    "把它先翻倍一次（如果原本是奇数）"之后得到的那个偶数，此后它只会单调
    递减（不断除以 2）。基于这个观察，预处理时把所有奇数先翻倍一次（这一步
    等价于"抢占"这个数唯一一次有意义的变大机会），然后问题转化为："这些数
    只能不断变小（除以 2 直到变成奇数为止），求过程中出现过的最小偏移量"。
    用一个最大堆维护当前数组，同时用一个变量追踪当前最小值：每次贪心地把
    **当前最大值**除以 2（缩小最大值是唯一可能减小偏移量的操作，因为最小值
    只会不变或者被这次除法产生的新值刷新），更新堆和最小值，重新计算偏移量并
    维护历史最优；只要堆顶（当前最大值）还是偶数就继续这个过程，一旦堆顶变成
    奇数，它已经无法再变小，继续缩小其它数不会再改善这个瓶颈，算法终止。
    【复杂度】时间 O(n log n + n log(max_val))（初始建堆 O(n)，之后每个数
    最多被除 log(max_val) 次，每次堆操作 O(log n)），空间 O(n)。
    【易错点】① 容易忘记预处理"奇数先翻倍一次"这一步，直接对原数组建堆——
    这样会漏掉"奇数变大后可能让最大值不再是瓶颈"的情形；② 循环终止条件是
    "堆顶（当前最大值）为偶数"，如果写成某个固定次数或者忽略这个条件，会在
    堆顶已经是奇数、继续除只会让它变成非整数时出错。
    """
    heap: list[int] = []
    min_val = float("inf")
    for x in nums:
        if x % 2 == 1:
            x *= 2
        heap.append(-x)
        min_val = min(min_val, x)
    heapq.heapify(heap)
    best = -heap[0] - min_val
    while heap[0] % 2 == 0:
        cur = -heapq.heappop(heap)
        cur //= 2
        min_val = min(min_val, cur)
        heapq.heappush(heap, -cur)
        best = min(best, -heap[0] - min_val)
    return best


# ── LC358 K距离间隔重排字符串(Hard) ───────────────────────────────────────
def rearrange_string_k_distance_apart(s: str, k: int) -> str:
    """
    【题意】给定字符串 s 和整数 k，重新排列 s 使得任意两个相同字符之间的
    距离至少为 k；如果不可能，返回空字符串 ""。
    【思路】"最大堆 + 冷却队列"：统计每个字符的频次放进最大堆（按频次）。
    维护一个时间指针 time（也就是结果字符串当前的长度）：每个时间步，先看
    冷却队列队首是否已经"冷却期满"（它的 ready_time == time），如果是就把它
    放回堆里重新参与"当前最高频"的竞争（要用 while 而不是 if，因为同一时刻
    可能有多个字符同时冷却期满）；然后从堆顶取出当前频次最高的字符放到结果
    末尾（贪心：优先消耗最高频字符，是为了让它尽量分散，避免它在字符串末尾
    "剩余量太集中"而无法满足间距）；如果这个字符还有剩余次数，就不能立刻把它
    放回堆，而是记录到冷却队列，标记它要等到 `time + k` 之后才能重新可用。
    如果某个时间步冷却队列腾出后堆仍然为空，说明剩下的字符全部还在冷却中、
    没有任何字符可以填入这个位置，判定无解，返回 ""。
    【复杂度】时间 O(n log c)（n = len(s)，c 为字符集大小，每个字符入堆出堆
    各常数次），空间 O(c + k)（堆 + 冷却队列）。
    【易错点】① 冷却队列的"是否可以放回堆"判断必须用 `while`，只用 `if`
    会漏掉同一时刻有多个字符同时解冻的情况；② 判断无解的时机是"清空了所有
    已解冻的字符之后，堆仍然为空但还有未排完的字符"，如果在检查冷却队列之前
    就判断堆是否为空，会把"马上就要解冻、只是还没来得及放回堆"的字符误判为
    无解。
    """
    if k <= 1:
        return s
    freq = Counter(s)
    heap = [(-cnt, ch) for ch, cnt in freq.items()]
    heapq.heapify(heap)
    result: list[str] = []
    wait_queue: deque[tuple[int, str, int]] = deque()  # (剩余频次(负数), 字符, 解冻时间)
    time = 0
    while heap or wait_queue:
        while wait_queue and wait_queue[0][2] == time:
            neg_cnt, ch, _ = wait_queue.popleft()
            heapq.heappush(heap, (neg_cnt, ch))
        if not heap:
            return ""
        neg_cnt, ch = heapq.heappop(heap)
        result.append(ch)
        neg_cnt += 1
        if neg_cnt < 0:
            wait_queue.append((neg_cnt, ch, time + k))
        time += 1
    return "".join(result)


# ── LC857 雇佣K名工人的最低成本(Hard) ─────────────────────────────────────
def mincost_to_hire_workers(quality: list[int], wage: list[int], k: int) -> float:
    """
    【题意】有 n 名工人，quality[i] 是第 i 名工人的工作能力，wage[i] 是他的
    最低期望薪资。需要雇佣恰好 k 名工人组成一个薪资组，规则：① 组内每人薪资
    不低于自己的最低期望；② 组内薪资必须与能力成正比（能力是别人两倍则薪资
    也必须是两倍）。求组成这样一个 k 人组的最小总薪资。
    【思路】"排序 + 堆的置换贪心"：对每个工人，`ratio = wage[i] / quality[i]`
    就是"如果这名工人是组内薪资基准，其他人应该按这个比例支付"的下限比例
    ——一旦某名工人在组内，实际支付给他的薪资至少是 `ratio_组内最大值 *
    quality[i]`，所以要让总成本最小，组内实际采用的比例应该恰好等于"组内
    ratio 最大的那个人的 ratio"（比这个更低就违反了那个人的最低期望）。把
    所有工人按 ratio **升序**排序后从左到右扫描，这样处理到第 i 个工人时，
    它的 ratio 是"目前扫描过的所有工人里最大的"——只要选中的 k 人组是这些
    已扫描工人的子集，当前工人的 ratio 就一定是组内的支付比例。用一个最大堆
    维护"当前已扫描、候选进组"的工人 quality：每扫描一个新工人就把它的
    quality 计入堆和累加和；一旦堆的大小超过 k，就贪心地弹出**quality 最大**
    的那个（用最少的能力凑够薪资总量最划算——保留 quality 之和最小的 k
    个人，配合当前 ratio 计算出的候选总成本才会最小）；堆大小恰好为 k 时，
    候选总成本 = quality 之和 * 当前工人的 ratio，对所有候选取最小值。
    【复杂度】时间 O(n log n)（排序 + 每个工人最多入堆出堆各一次），空间
    O(k)。
    【易错点】① 候选总成本必须用"当前正在扫描的工人的 ratio"而不是堆里
    quality 最大那个工人的 ratio——因为排序保证了当前工人的 ratio 才是
    已扫描集合里的最大值；② 堆里应保留 **quality 较小** 的 k 个人（用最大堆
    弹出最大的 quality），而不是随意选择，否则同一 ratio 下总成本不是最小。
    """
    workers = sorted(zip(wage, quality), key=lambda w: w[0] / w[1])
    heap: list[int] = []  # 最大堆(取负数)，存当前候选组的 quality
    quality_sum = 0
    best = float("inf")
    for w, q in workers:
        heapq.heappush(heap, -q)
        quality_sum += q
        if len(heap) > k:
            quality_sum += heapq.heappop(heap)  # 堆里存负数，加负数=减去最大 quality
        if len(heap) == k:
            best = min(best, quality_sum * w / q)
    return best


# ── LC480 滑动窗口中位数(Hard) ─────────────────────────────────────────────
def median_sliding_window(nums: list[int], k: int) -> list[float]:
    """
    【题意】给定数组 nums 和窗口大小 k，窗口从最左滑到最右，每次滑动一格，
    求每个窗口内 k 个数的中位数（k 为偶数时取中间两个数的平均值）。
    【思路】"双堆 + 懒删除"（LC295 数据流的中位数的滑动窗口版）：维护一个
    最大堆 small（存负数，代表数值较小的一半）和一个最小堆 large（代表数值
    较大的一半），并始终保持 `len(small) == len(large)` 或
    `len(small) == len(large) + 1`——这样中位数要么是 small 堆顶，要么是
    small 堆顶和 large 堆顶的平均值。难点在于"滑出窗口的旧元素"不能直接从
    堆中间删除（堆不支持随机删除），所以用**懒删除**：用一个计数字典记录
    "这个值还欠多少次删除"，只在某个堆的堆顶恰好是待删除值时才真正弹出，
    平时任由无效值留在堆的深处，只要不影响堆顶查询就不必立刻清理。因为
    size 计数（small_size/large_size）是"逻辑上有效"的个数（不含懒删除已
    记账的），每次插入或删除新元素后都要重新平衡两个堆的逻辑大小，并在重新
    平衡前后调用 prune 清理堆顶的失效值，确保堆顶始终反映真实的当前分界。
    【复杂度】时间 O(n log k)（每个元素入堆出堆 O(log k)，n 次滑动），空间
    O(k)。
    【易错点】① 判断某个滑出的值属于 small 还是 large 时，必须用"当前
    (懒删除之后的) small 堆顶"作比较，而不是它当初插入时的分界——分界会
    随着后续插入/删除动态移动，但只要比较规则前后一致，逻辑计数依然正确；
    ② 每次真正需要读取 small[0]/large[0]（计算中位数或做平衡判断）之前，
    必须先调用 prune 清理堆顶的懒删除标记，否则会读到一个已经"名义上不存在"
    的过期值。
    """
    small: list[int] = []  # 最大堆(取负数)，数值较小的一半
    large: list[int] = []  # 最小堆，数值较大的一半
    delayed: dict[int, int] = {}
    small_size = 0
    large_size = 0

    def prune(heap: list[int], is_small: bool) -> None:
        while heap:
            val = -heap[0] if is_small else heap[0]
            if delayed.get(val, 0) > 0:
                delayed[val] -= 1
                if delayed[val] == 0:
                    del delayed[val]
                heapq.heappop(heap)
            else:
                break

    def rebalance() -> None:
        nonlocal small_size, large_size
        prune(small, True)
        prune(large, False)
        if small_size > large_size + 1:
            val = -heapq.heappop(small)
            heapq.heappush(large, val)
            small_size -= 1
            large_size += 1
            prune(small, True)
        elif large_size > small_size:
            val = heapq.heappop(large)
            heapq.heappush(small, -val)
            large_size -= 1
            small_size += 1
            prune(large, False)

    result: list[float] = []
    for i, x in enumerate(nums):
        if small and x <= -small[0]:
            heapq.heappush(small, -x)
            small_size += 1
        else:
            heapq.heappush(large, x)
            large_size += 1
        rebalance()

        if i >= k - 1:
            if k % 2 == 1:
                result.append(float(-small[0]))
            else:
                result.append((-small[0] + large[0]) / 2.0)

            out = nums[i - k + 1]
            delayed[out] = delayed.get(out, 0) + 1
            if out <= -small[0]:
                small_size -= 1
            else:
                large_size -= 1
            rebalance()
    return result


def _self_test() -> None:
    assert trap_rain_water_2d(
        [[1, 4, 3, 1, 3, 2], [3, 2, 1, 3, 2, 4], [2, 3, 3, 2, 3, 1]]
    ) == 4
    assert (
        trap_rain_water_2d(
            [
                [3, 3, 3, 3, 3],
                [3, 2, 2, 2, 3],
                [3, 2, 1, 2, 3],
                [3, 2, 2, 2, 3],
                [3, 3, 3, 3, 3],
            ]
        )
        == 10
    )

    assert get_skyline([[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]) == [
        [2, 10],
        [3, 15],
        [7, 12],
        [12, 0],
        [15, 10],
        [20, 8],
        [24, 0],
    ]
    assert get_skyline([[0, 2, 3], [2, 5, 3]]) == [[0, 3], [5, 0]]

    assert kth_smallest_array_sum([[1, 3, 11], [2, 4, 6]], 5) == 7
    assert kth_smallest_array_sum([[1, 10, 10], [1, 4, 5], [2, 3, 6]], 7) == 9

    assert minimum_deviation([1, 2, 3, 4]) == 1
    assert minimum_deviation([4, 1, 5, 20, 3]) == 3
    assert minimum_deviation([2, 10, 8]) == 3

    r1 = rearrange_string_k_distance_apart("aabbcc", 3)
    assert r1 == "abcabc"
    assert rearrange_string_k_distance_apart("aaabc", 3) == ""
    r3 = rearrange_string_k_distance_apart("aaadbbcc", 2)
    assert Counter(r3) == Counter("aaadbbcc")
    last_seen: dict[str, int] = {}
    for i, ch in enumerate(r3):
        assert ch not in last_seen or i - last_seen[ch] >= 2
        last_seen[ch] = i

    assert abs(mincost_to_hire_workers([10, 20, 5], [70, 50, 30], 2) - 105.0) < 1e-5
    assert (
        abs(mincost_to_hire_workers([3, 1, 10, 10, 1], [4, 8, 2, 2, 7], 3) - 30.66667)
        < 1e-4
    )

    assert median_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [
        1.0,
        -1.0,
        -1.0,
        3.0,
        5.0,
        6.0,
    ]
    assert median_sliding_window([1, 2, 3, 4, 2, 3, 1, 4, 2], 3) == [
        2.0,
        3.0,
        3.0,
        3.0,
        2.0,
        3.0,
        2.0,
    ]

    print(
        "[PASS] p15_heap_topk_iii: 7 题(接雨水II/天际线问题/矩阵第k个最小数组和/"
        "数组的最小偏移量/K距离间隔重排字符串/雇佣K名工人的最低成本/"
        "滑动窗口中位数)全部通过"
    )


if __name__ == "__main__":
    _self_test()
