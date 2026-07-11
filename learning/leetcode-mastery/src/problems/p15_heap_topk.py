"""15 堆/Top-K：五道题的规范实现。Python 只有最小堆(heapq)，最大堆用取负数模拟。"""
from __future__ import annotations

import heapq
from collections import Counter


# ── LC215 数组中的第K个最大元素(Medium) ──────────────────────────────────
def find_kth_largest(nums: list[int], k: int) -> int:
    """
    【题意】给定未排序数组 nums 和整数 k，返回数组中第 k 大的元素（注意是排序后
    第 k 大，不是第 k 个不同的值）。
    【思路】不需要对整个数组排序（O(n log n)），只需要维护一个**大小恰好为 k 的
    最小堆**：从左到右把元素依次入堆，一旦堆的大小超过 k 就弹出堆顶（也就是当前堆里
    最小的那个）。扫完整个数组后，堆里剩下的 k 个元素正是"全局最大的 k 个数"，而堆顶
    （最小堆的堆顶是堆内最小值）就是这 k 个数里最小的一个——也正是全局第 k 大。
    【复杂度】时间 O(n log k)（每个元素最多一次入堆一次出堆，堆大小恒为 k），
    空间 O(k)。
    【易错点】容易搞反"维护大小为 k 的最小堆"和"维护大小为 k 的最大堆"——如果用最大堆
    维护全部 n 个元素再弹 k-1 次，虽然结果也对，但堆大小是 O(n) 而不是 O(k)，失去了
    这个技巧本该有的空间/时间优势。
    """
    heap: list[int] = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]


# ── LC347 前 K 个高频元素(Medium) ────────────────────────────────────────
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    【题意】给定整数数组 nums 和整数 k，返回出现频率前 k 高的元素（顺序不限）。
    【思路】先用 Counter 一次遍历统计每个数出现的次数，把问题转化成"从 (数值, 频次)
    这些 pair 里，按频次找 Top-K"——这就是标准的"维护大小为 k 的最小堆"模式（和
    LC215 是同一个骨架，只是排序的键从"数值本身"换成了"频次"）：依次把 (频次, 数值)
    入堆，堆大小超过 k 就弹出频次最小的那个，最后堆里剩下的 k 个 pair 就是频次前 k 高
    的元素。
    【复杂度】时间 O(n log k)（n 为数组长度，去重后不同数值数一般远小于 n），
    空间 O(n + k)。
    【易错点】容易直接对所有 (数值, 频次) 按频次整体排序再切前 k 个，这是 O(m log m)
    （m 为不同数值个数）——当 k 远小于 m 时，维护大小为 k 的堆比整体排序更省。
    """
    freq = Counter(nums)
    heap: list[tuple[int, int]] = []
    for val, cnt in freq.items():
        heapq.heappush(heap, (cnt, val))
        if len(heap) > k:
            heapq.heappop(heap)
    return [val for _cnt, val in heap]


# ── LC692 前K个高频单词(Medium) ──────────────────────────────────────────
def top_k_frequent_words(words: list[str], k: int) -> list[str]:
    """
    【题意】给定字符串数组 words 和整数 k，返回出现频率前 k 高的单词；如果两个单词
    频率相同，按字典序**升序**排在前面（返回结果本身也要求按"频率降序、同频按字典序
    升序"整体有序，不是顺序任意）。
    【思路】和 LC347 的区别只在"排序键"从单一的频次变成了"频次 + 字典序"的复合键。
    技巧是把频次取负、单词保持正序，组成排序键 (-cnt, word)：对这个键做**最小堆**，
    弹出的第一名自然是 -cnt 最小（也就是 cnt 最大，频率最高）；频次相同时 -cnt 相等，
    比较退化到 word 本身，取正序就自然是字典序升序排前面。直接对全部 (-cnt, word)
    建堆，然后连续弹出 k 次，弹出的顺序就正好是题目要求的最终排序，不需要额外再排一次。
    【复杂度】时间 O(m log m)（m 为不同单词数，建堆 O(m) + 弹 k 次 O(k log m)，
    最坏 O(m log m)），空间 O(m)。
    【易错点】只把频次取负、却忘记单词要保持"正序"（如果也取反或者用其他方式比较，
    就会把"频率相同时字典序升序"这个次要条件搞反）——(频率取负, 字符串正序) 这个
    组合键要同时处理对，才能一次 heappop 就拿到最终顺序。
    """
    freq = Counter(words)
    heap = [(-cnt, word) for word, cnt in freq.items()]
    heapq.heapify(heap)
    result: list[str] = []
    for _ in range(k):
        _neg_cnt, word = heapq.heappop(heap)
        result.append(word)
    return result


# ── LC253 会议室 II(Medium) ──────────────────────────────────────────────
def min_meeting_rooms(intervals: list[list[int]]) -> int:
    """
    【题意】给定若干会议的 [start, end) 时间区间，求同一时刻最多有多少场会议在
    同时进行，即最少需要准备多少间会议室。
    【思路】按会议**开始时间**排序后依次处理，用一个最小堆维护"当前正在进行的所有
    会议里，最早结束的那个时刻"。每来一个新会议，先看堆顶（最早结束时间）是否
    <= 当前会议的开始时间——如果是，说明那间会议室已经空出来了，可以直接复用
    （弹出堆顶，把它的结束时间换成新会议的结束时间）；如果不是，说明现有会议室都
    还占用着，必须新开一间（直接把新会议的结束时间入堆，不弹出）。扫描结束后堆的
    大小就是同时并存的最大会议室数。（等价写法：把所有起点、终点分别排序后用双指针
    做扫描线，效果相同，这里选堆实现是因为它更直接地对应"当前占用的房间集合"这个概念。）
    【复杂度】时间 O(n log n)（排序 + 每个会议一次堆操作），空间 O(n)。
    【易错点】判断"能否复用房间"时要用 `<=`（结束时刻和开始时刻相等，视为可以立刻
    复用，因为区间是 [start, end) 半开的），写成严格 `<` 会多开出不必要的房间。
    """
    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda iv: iv[0])
    heap: list[int] = []  # 当前占用中的会议室的结束时间
    for start, end in ordered:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    return len(heap)


# ── LC295 数据流的中位数(Hard) ───────────────────────────────────────────
class MedianFinder:
    """
    【题意】设计一个数据结构，支持源源不断地 add_num(num) 添加数字，并能随时
    find_median() 查询目前为止所有数字的中位数（偶数个数时取中间两个的平均值）。
    【思路】如果每次都排序，add 一次 O(n log n) 太慢。核心 insight：把数据流从中间
    劈成两半——用一个最大堆 small 存"较小的一半"，一个最小堆 large 存"较大的一半"，
    并始终维持"两堆大小相差不超过 1"这个不变量。这样一来：
      - 若两堆一样大，中位数 = (small 堆顶 + large 堆顶) / 2；
      - 若 small 比 large 多一个，中位数就是 small 的堆顶。
    因为中位数只可能出现在"较小一半的最大值"和"较大一半的最小值"这两个位置附近，
    只要这两个数随时能在 O(1) 取到（堆顶），中位数查询就是 O(1)。add_num 时先把新数
    丢进 small，再把 small 的堆顶"倒"进 large 保证 small 里的数都 <= large 里的数
    （因为 small 存的是较小一半，它的最大值也不该超过 large 的最小值），最后如果
    large 的元素数超过了 small，就从 large 倒一个回 small，把"大小差不超过 1"这个
    不变量重新扳平。Python 的 heapq 只提供最小堆，用"存入取负数"的技巧把它当最大堆用：
    small 里实际存的是原数的相反数，堆顶 -small[0] 才是"较小一半"里的最大值。
    【复杂度】add_num: 时间 O(log n)（两次堆操作）；find_median: 时间 O(1)；
    空间 O(n)（两个堆合起来存了全部数字）。
    【易错点】① 只用一个堆无法同时快速拿到"较小一半的最大值"和"较大一半的最小值"
    ——这正是需要两个堆而不是一个的原因；② 维持不变量时容易漏掉"新数不能直接进
    正确的那一堆，必须统一从 small 入堆再倒一次"这一步，否则 small 存的值范围可能
    超过 large，导致中位数计算错误；③ 取 small 堆顶时忘记再取一次负号（堆里存的是
    负数）。
    """

    def __init__(self) -> None:
        self.small: list[int] = []  # 最大堆(取负数模拟)，存较小的一半
        self.large: list[int] = []  # 最小堆，存较大的一半

    def add_num(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2


def _self_test() -> None:
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4

    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert top_k_frequent([1], 1) == [1]

    assert top_k_frequent_words(
        ["i", "love", "leetcode", "i", "love", "coding"], 2
    ) == ["i", "love"]
    assert top_k_frequent_words(
        ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], 4
    ) == ["the", "is", "sunny", "day"]

    assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_meeting_rooms([[7, 10], [2, 4]]) == 1

    m = MedianFinder()
    m.add_num(1)
    m.add_num(2)
    assert m.find_median() == 1.5
    m.add_num(3)
    assert m.find_median() == 2

    print("[PASS] p15_heap_topk: 5 题(第K大/前K高频/前K高频单词/会议室II/数据流中位数)全部通过")


if __name__ == "__main__":
    _self_test()
