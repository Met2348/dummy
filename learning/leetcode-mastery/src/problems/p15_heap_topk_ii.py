"""15 堆/Top-K · 进阶补充(Part II)：练"堆的更多应用场景"——多路归并、贪心决策
辅助、流式处理，不重复 Part I 的"维护大小为 k 的堆"骨架，只扩变体。"""
from __future__ import annotations

import heapq
from collections import Counter, deque


# ── LC373 查找和最小的K对数字(Medium) ────────────────────────────────────
def k_smallest_pairs(
    nums1: list[int], nums2: list[int], k: int
) -> list[list[int]]:
    """
    【题意】给定两个非递减排序的整数数组 nums1、nums2 和整数 k，定义一对数
    (u, v)（u 来自 nums1，v 来自 nums2），求所有可能的数对中，和最小的 k 个
    （按 [u, v] 的形式返回，结果数量不超过 min(k, len(nums1)*len(nums2))）。
    【思路】直接枚举 nums1 x nums2 的所有组合再排序是 O(mn log(mn))，而且很多
    组合的和明显偏大，根本不需要算——这是典型的"多路归并找前 k 小"场景。把
    nums2 的每个下标 j 都看成一条"有序序列"：`nums1[0]+nums2[j] <= nums1[1]+nums2[j]
    <= ...`（因为 nums1 本身非递减），于是整个问题变成"合并 len(nums2) 条有序
    序列，找前 k 小"。初始只把每条序列的第一个元素——也就是 (nums1[0]+nums2[j], 0, j)
    ——放进最小堆（而不是真的枚举所有 m*n 对），每弹出一个 (和, i, j)，如果
    i+1 还在 nums1 范围内，就把这条序列的下一个候选 (nums1[i+1]+nums2[j], i+1, j)
    补进堆里。堆里始终只保留"每条序列当前最靠前、还没被消费的候选"，弹出 k 次
    即得到全局前 k 小。
    【复杂度】时间 O((m+k) log m)（m = len(nums2)，初始建堆 O(m log m)，之后
    最多再弹/推 k 次，每次 O(log m)），空间 O(m)。
    【易错点】容易先把 m*n 个和全部算出来再排序——当 m、n 很大而 k 很小时，这样
    做的浪费和堆方法的差距会非常明显；另外补充候选时容易写成"j+1"而不是
    "i+1"（这里固定的是 j，递增的是 i，方向和常见的"两个数组"写法容易搞反）。
    """
    if not nums1 or not nums2:
        return []
    heap: list[tuple[int, int, int]] = []
    for j in range(len(nums2)):
        heapq.heappush(heap, (nums1[0] + nums2[j], 0, j))
    result: list[list[int]] = []
    while heap and len(result) < k:
        _total, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1):
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
    return result


# ── LC378 有序矩阵中第K小的元素(Medium) ──────────────────────────────────
def kth_smallest_matrix(matrix: list[list[int]], k: int) -> int:
    """
    【题意】给定一个 n x n 矩阵，矩阵每行、每列都按升序排列，求矩阵中第 k 小的
    元素。
    【思路】和 LC373 是同一个"多路归并"骨架：把矩阵的每一行都看成一条有序序列，
    初始把每行的第一个元素 (matrix[i][0], i, 0) 放进最小堆，每弹出一个
    (值, 行号, 列号)，如果该行还有下一列，就把同一行的下一个元素补进堆。
    弹够 k 次，最后一次弹出的值就是第 k 小。这样只需要"归并"最多 k 个元素，
    不需要把整个矩阵摊平排序。
    【复杂度】时间 O(k log n)（n 为矩阵边长，堆大小恒为 n），空间 O(n)。
    【易错点】容易把"行"和"列"的推进方向弄反——应该是同一行内列号 +1（因为
    行内是升序的），而不是跨行推进；另外堆的初始大小是 n（矩阵边长）而不是 k，
    k 只决定弹出的次数。
    """
    n = len(matrix)
    heap: list[tuple[int, int, int]] = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)
    value = matrix[0][0]
    for _ in range(k):
        value, row, col = heapq.heappop(heap)
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    return value


# ── LC502 IPO(Hard) ──────────────────────────────────────────────────────
def find_maximized_capital(
    k: int, w: int, profits: list[int], capital: list[int]
) -> int:
    """
    【题意】初始资本为 w，最多可以做 k 个项目（做完一个项目获得对应利润，利润
    累加进资本，可用于解锁资本要求更高的项目），每个项目 i 需要资本 >= capital[i]
    才能启动，profits[i] 是启动后能获得的利润。求做完最多 k 个项目后能达到的
    最大资本。
    【思路】这是"贪心 + 堆"的组合：贪心的部分是"在当前资本能负担的所有项目里，
    永远选利润最大的那个去做"——因为做项目的顺序不影响"哪些项目最终能做"（只要
    资本单调不减），既然反正要做 k 个，每一步选当前可行集合里利润最大的，得到的
    总利润一定不会比其它选择顺序更差。实现上按 capital 排序后用一个指针把
    "资本要求 <= 当前 w 的项目"逐个放进一个最大堆（按利润排序），每一轮从堆顶
    取出利润最大的项目做掉、更新 w，再看看 w 增长后又能解锁哪些新项目。重复
    k 轮或堆空为止。
    【复杂度】时间 O(n log n + k log n)（排序 + 最多 n 次入堆 + k 次弹堆），
    空间 O(n)。
    【易错点】容易在每一轮都重新扫描一遍所有项目找"当前能做的里利润最大的"，
    退化成 O(k*n)；正确做法是维护一个指针，已经放进堆里的项目不需要重复扫描，
    只需要把"资本要求 <= w 但还没入堆"的项目继续往堆里推。
    """
    projects = sorted(zip(capital, profits))
    heap: list[int] = []  # 最大堆(取负数)，存当前可负担项目的利润
    i, n = 0, len(projects)
    for _ in range(k):
        while i < n and projects[i][0] <= w:
            heapq.heappush(heap, -projects[i][1])
            i += 1
        if not heap:
            break
        w += -heapq.heappop(heap)
    return w


# ── LC767 重构字符串(Medium) ─────────────────────────────────────────────
def reorganize_string(s: str) -> str:
    """
    【题意】给定字符串 s，重新排列其中的字符，使任意两个相邻字符都不相同；如果
    不可能做到，返回空字符串。
    【思路】贪心 + 最大堆：统计每个字符的出现频率，每一步都从当前剩余字符里取
    出频率最高的那个字符放到结果末尾——只要不是"和上一个刚放的字符相同"，这样
    做能最大程度地把高频字符分散开。用一个最大堆（取负频率）维护候选，每次弹出
    堆顶字符使用一次（频率 -1），但不能立刻把它塞回堆——如果立刻塞回去，下一步
    很可能又弹到它自己（如果它的频率仍然最高），造成相邻重复；所以用一个"延迟
    一步"的技巧：把上一步用过的字符缓存到 prev，等这一步用完新字符之后，再把
    prev（如果还有剩余次数）放回堆里参与下一轮竞争。是否有解的判断：如果某个
    字符的频率超过 `(len(s)+1)//2`，说明它无论怎么摆放都必然会有两个相邻，
    直接判定无解。
    【复杂度】时间 O(n log c)（n = len(s)，c 为字符集大小，每个字符入堆出堆
    各常数次），空间 O(c)。
    【易错点】① 忘记先判断"最高频率是否超过 (n+1)//2"这个可行性条件，直接跑
    贪心可能死循环或者得到错误的相邻重复结果；② "延迟一步"这个技巧容易漏掉——
    如果每次弹出字符后立刻把剩余次数塞回堆，最高频字符会连续被选中导致相邻
    重复，必须等下一轮再放回。
    """
    freq = Counter(s)
    max_freq = max(freq.values()) if freq else 0
    if max_freq > (len(s) + 1) // 2:
        return ""
    heap = [(-cnt, ch) for ch, cnt in freq.items()]
    heapq.heapify(heap)
    result: list[str] = []
    prev: tuple[int, str] | None = None
    while heap:
        cnt, ch = heapq.heappop(heap)
        result.append(ch)
        cnt += 1  # 存的是负数，用掉一次频率就 +1(向 0 靠近)
        if prev is not None and prev[0] < 0:
            heapq.heappush(heap, prev)
        prev = (cnt, ch)
    return "".join(result)


# ── LC621 任务调度器(Medium) ─────────────────────────────────────────────
def least_interval(tasks: list[str], n: int) -> int:
    """
    【题意】给定任务列表 tasks（每个元素是任务类型），以及冷却时间 n：同一种
    任务两次执行之间必须间隔至少 n 个单位时间（可以用"待命"填充间隔，或者
    执行其它任务）。求完成所有任务需要的最短总时间。
    【思路】这题常见的解法是数学公式，但这里用堆模拟来呼应"堆"这个主题：统计
    每种任务的频次，放进一个最大堆（按频次）。每一个时间单位，从堆顶取出当前
    频次最高的任务执行一次（频次 -1），如果这个任务还有剩余次数，就不能立刻把
    它放回堆里竞争——它需要冷却 n 个时间单位之后才能再次被调度。用一个队列
    （存 (剩余频次, 可以重新入堆的时间点)）来"暂存"刚执行过、还没冷却完的任务；
    每走一个时间单位，检查队首任务是否已经冷却完毕（可入堆时间 == 当前时间），
    冷却完就把它放回堆参与下一轮竞争。只要堆和队列有一个非空，就继续推进时间；
    如果某一时刻堆为空但队列非空，说明所有能执行的任务都在冷却中，这一个单位
    时间只能空转（待命），时间正常 +1，但不消耗任何任务。
    【复杂度】时间 O(T)（T 为最终总时间，T <= n_tasks * (n+1) 量级），空间
    O(种类数)。
    【易错点】① 容易忘记"某一时间单位堆为空、只有队列在冷却"这种情况下，
    时间依然要推进（这就是"待命"耗费的时间片），而不是跳过这个时间单位；
    ② 队列里存的应该是"可以重新入堆的绝对时间点"（当前时间 + n），而不是
    "还需要等待的相对时长"，否则每一步都要额外维护倒计时，容易出错。
    """
    freq = Counter(tasks)
    heap = [-cnt for cnt in freq.values()]
    heapq.heapify(heap)
    time = 0
    queue: deque[list[int]] = deque()  # [剩余频次(负数), 可重新入堆的时间点]
    while heap or queue:
        time += 1
        if heap:
            cnt = heapq.heappop(heap) + 1
            if cnt != 0:
                queue.append([cnt, time + n])
        if queue and queue[0][1] == time:
            heapq.heappush(heap, queue.popleft()[0])
    return time


# ── LC1834 单线程CPU(Medium) ─────────────────────────────────────────────
def get_order(tasks: list[list[int]]) -> list[int]:
    """
    【题意】给定任务列表，每个任务是 [enqueueTime, processingTime]，单线程 CPU
    每次只能处理一个任务：如果 CPU 空闲且有多个任务已经入队，优先处理耗时最短的
    （耗时相同则原始下标小的优先）；如果当前没有任何任务入队，CPU 会一直等到
    下一个任务入队为止。返回任务被处理的顺序（按原始下标）。
    【思路】这是"排序 + 堆"的组合：先把任务按 enqueueTime 排序，方便按时间顺序
    知道"此刻已经有哪些任务入队了"。用一个时间指针模拟：如果堆里没有可处理的
    任务（当前时间点还没有任务入队），直接把时间跳到下一个未入队任务的
    enqueueTime（不需要一个时间单位一个时间单位地空转，直接跳跃更高效）；否则，
    把所有 enqueueTime <= 当前时间的任务都放进一个最小堆（按 (processingTime, 原下标)
    排序），从堆顶取出耗时最短（同耗时取下标最小）的任务处理，把时间推进
    processingTime。重复直到所有任务处理完。
    【复杂度】时间 O(n log n)（排序 + 每个任务入堆出堆各一次），空间 O(n)。
    【易错点】① 堆为空时不能傻等（一个时间单位一个时间单位地推进），必须直接
    把时间跳到"下一个未入队任务的 enqueueTime"，否则任务量大、时间跨度长时会
    严重超时；② 堆里比较的键必须是 (processingTime, 原下标) 的二元组而不是
    只比较 processingTime——耗时相同时题目要求下标小的优先，Python 的元组
    比较刚好能同时满足这两个排序需求。
    """
    n = len(tasks)
    order = sorted(range(n), key=lambda i: tasks[i][0])
    heap: list[tuple[int, int]] = []
    result: list[int] = []
    time = 0
    i = 0
    while len(result) < n:
        while i < n and tasks[order[i]][0] <= time:
            idx = order[i]
            heapq.heappush(heap, (tasks[idx][1], idx))
            i += 1
        if not heap:
            time = tasks[order[i]][0]
            continue
        proc, idx = heapq.heappop(heap)
        time += proc
        result.append(idx)
    return result


# ── LC1642 可以到达的最远建筑(Medium) ────────────────────────────────────
def furthest_building(heights: list[int], bricks: int, ladders: int) -> int:
    """
    【题意】沿着一排建筑往前走，从第 i 栋走到第 i+1 栋：如果后一栋更矮或一样高，
    免费通过；如果更高，需要用砖头（花费 = 高度差块砖）或者用一架梯子（不管
    高度差多少，消耗一整架梯子）翻过去。给定砖头总数和梯子总数，求最远能走到
    第几栋（下标）。
    【思路】"堆贪心置换"的经典应用：梯子应该留给"高度差最大"的那几次爬升，因为
    梯子不计成本、一架顶所有差值，用在差值越大的地方越"划算"；而砖头按差值
    数量消耗，适合处理差值较小的爬升。具体做法：从左到右遍历每一次需要爬升的
    地方（高度差 > 0），先假设"来者不拒，全部记到一个最小堆里当作用了梯子"；
    一旦梯子的"账"记多了（堆的大小超过了梯子总数），就说明必须有一次爬升要
    从"梯子"降级为"用砖头支付"——贪心地选择堆里**最小的**那次爬升差值降级
    （用 heappop 弹出最小值），用砖头支付这个差值，把这块"最小的差值"腾出来的
    梯子名额，留给后面可能出现的更大差值。如果某一步砖头不够支付了（bricks < 0），
    说明走不到这一步之后了，返回当前下标。如果一路都能走完，返回最后一个下标。
    【复杂度】时间 O(n log ladders)（堆大小最多维持在 ladders+1），空间
    O(ladders)。
    【易错点】① 容易把"堆里存的是差值"和"用梯子还是用砖头"搞混——正确理解是
    "堆里的都是暂时用梯子记账的爬升，只有被挤出堆的那个才真正改用砖头支付"；
    ② 判断条件是堆大小 `> ladders`（严格大于）才触发置换，等于时说明梯子刚好
    够用，不需要提前把某次爬升降级为砖头支付。
    """
    heap: list[int] = []
    for i in range(len(heights) - 1):
        diff = heights[i + 1] - heights[i]
        if diff <= 0:
            continue
        heapq.heappush(heap, diff)
        if len(heap) > ladders:
            bricks -= heapq.heappop(heap)
            if bricks < 0:
                return i
    return len(heights) - 1


# ── LC703 数据流中的第K大元素(Easy) ──────────────────────────────────────
class KthLargest:
    """
    【题意】设计一个类，初始化时给定 k 和一个初始数字流 nums，之后每次调用
    add(val) 往流里加入一个新数字，并返回加入之后整个流里第 k 大的数字。
    【思路】和 LC215（数组中的第K个最大元素）是完全一样的骨架，只是从"一次性
    给定数组"变成了"数据流式地不断加入"：始终维护一个**大小恰好为 k 的最小堆**，
    堆顶就是当前所有数字里第 k 大的那个。初始化时把 nums 全部入堆后裁剪到大小
    k；每次 add 把新值入堆，若堆大小超过 k 就弹出堆顶（最小的那个），再返回
    此刻的堆顶。
    【复杂度】初始化：时间 O(n log k)（n = len(nums)）；add：时间 O(log k)，
    空间 O(k)。
    【易错点】容易在初始化阶段就地维护一个大小为 len(nums) 的堆而不裁剪到 k
    ——这样虽然结果不错，但失去了"堆大小恒为 k"这个技巧本该带来的空间优势，
    并且如果后续 add 很多次，堆会越长越大。
    """

    def __init__(self, k: int, nums: list[int]) -> None:
        self.k = k
        self.heap = list(nums)
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


def _self_test() -> None:
    assert k_smallest_pairs([1, 7, 11], [2, 4, 6], 3) == [[1, 2], [1, 4], [1, 6]]
    assert k_smallest_pairs([1, 1, 2], [1, 2, 3], 2) == [[1, 1], [1, 1]]

    assert kth_smallest_matrix([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8) == 13

    assert find_maximized_capital(2, 0, [1, 2, 3], [0, 1, 1]) == 4

    aab_result = reorganize_string("aab")
    assert all(aab_result[i] != aab_result[i + 1] for i in range(len(aab_result) - 1))
    assert sorted(aab_result) == sorted("aab")
    assert reorganize_string("aaab") == ""

    assert least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
    assert least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6

    assert get_order([[1, 2], [2, 4], [3, 2], [4, 1]]) == [0, 2, 3, 1]
    assert get_order([[7, 10], [7, 12], [7, 5], [7, 4], [7, 2]]) == [4, 3, 2, 0, 1]

    assert furthest_building([4, 2, 7, 6, 9, 14, 12], 5, 1) == 4
    assert furthest_building([4, 12, 2, 7, 3, 18, 20, 3, 19], 10, 2) == 7

    kl = KthLargest(3, [4, 5, 8, 2])
    assert kl.add(3) == 4
    assert kl.add(5) == 5
    assert kl.add(10) == 5
    assert kl.add(9) == 8
    assert kl.add(4) == 8

    print(
        "[PASS] p15_heap_topk_ii: 8 题(K对最小和/矩阵第K小/IPO/重构字符串/"
        "任务调度器/单线程CPU/最远建筑/数据流第K大)全部通过"
    )


if __name__ == "__main__":
    _self_test()
