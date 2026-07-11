"""滑动窗口 · 进阶补充（Part II）：不重讲框架，扩大"收缩条件"多样性覆盖面的 7 道题。"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict


def longest_ones(nums: list[int], k: int) -> int:
    """
    【题意】给定只含 0/1 的数组 nums 和整数 k，最多可以把 k 个 0 翻转成 1，求翻转后最长的连续
    全 1 子数组长度（等价说法：窗口内最多容忍 k 个 0）。
    【思路】收缩条件是"窗口内 0 的个数超过 k"——维护一个 zeros 计数器，right 每纳入一个 0 就
    加一；一旦 zeros > k（违反约束，必须收缩，这是和"无重复字符最长子串"同一方向的收缩条件：
    违反了才缩），就收缩 left 直到 zeros 重新 <= k。全程只需要维护一个计数器，不需要真的做
    翻转操作。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 收缩循环要用 `while zeros > k` 而不是 `if`，虽然这题每次 right 最多让 zeros
    多 1、收缩一次通常就够，但写成 `while` 更能体现"收缩到重新合法为止"这个通用心法，不容易
    在改写成别的约束时出错；2) 只有移出窗口的元素是 0 时才需要把 zeros 减一，容易忘记这个
    判断，无脑对每次移出都减一；3) 答案是"窗口长度"而非"翻转次数"，容易和 LC1493 的"允许删除
    一个"混淆返回值是否要再减一（这题不需要减，因为翻转的 0 仍然算作窗口内的元素）。
    """
    left = 0
    zeros = 0
    best = 0
    for right, x in enumerate(nums):
        if x == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


def longest_subarray(nums: list[int]) -> int:
    """
    【题意】给定只含 0/1 的数组，必须恰好删掉一个元素（不能不删），求删除后最长的连续全 1
    子数组长度。
    【思路】"删掉一个元素后全 1"等价于"原数组里存在一个窗口，其中最多包含 1 个 0"——把这个 0
    看作将被删除的那个位置。于是复用 longest_ones 的骨架，把阈值从 k 换成 1；但因为题目要求
    "必须删除"一个元素（哪怕数组全是 1，也要强制删掉一个），最终答案要在"最多 1 个 0 的最长
    窗口长度"基础上减 1（相当于强制从窗口里拿掉一个位置，无论这个位置原来是不是 0）。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 最容易忘记的是最后的 `- 1`——这是本题和 longest_ones 唯一的区别，很多人直接
    抄 longest_ones 的模板却忘了这一步，导致对全 1 数组算出的长度多 1；2) 全部是 1 的数组
    （比如 [1,1,1]）依然要执行"减 1"，因为题目强制要求删除一个元素，不能因为数组已经全 1 就
    豁免这次删除；3) 数组只有一个元素时，删除后剩下空数组，长度为 0，代入公式（窗口长度 1
    减 1）天然得到 0，不需要额外特判。
    """
    left = 0
    zeros = 0
    best = 0
    for right, x in enumerate(nums):
        if x == 0:
            zeros += 1
        while zeros > 1:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best - 1


def character_replacement(s: str, k: int) -> int:
    """
    【题意】给定字符串 s（大写字母）和整数 k，最多可以替换 k 个字符为任意字符，求替换后最长的
    "全部字符相同"的连续子串长度。
    【思路】收缩条件写成"窗口长度 - 窗口内出现次数最多的字符次数 > k"——这个差值就是"把窗口
    变成单一字符还需要替换多少个字符"，超过 k 说明当前窗口不可行，必须收缩。这里有一个不太
    直觉但很关键的实现技巧：max_freq（窗口内历史最高的单字符频次）在窗口收缩时**不需要主动
    下调**，因为答案只关心"出现过的最大可行窗口"，一个基于过时 max_freq 算出的、暂时不合法的
    窗口不会让 best 变大，等右指针继续前进、真正出现更大的合法窗口时 best 自然会被正确刷新。
    【复杂度】时间 O(n)，空间 O(字符集大小)。
    【易错点】1) 全程不下调 max_freq 是本题最反直觉的地方，很多人会强行去维护"当前窗口精确的
    最大频次"，代码更复杂而且容易出错，其实完全没有必要，因为 best 只增不减、错误地偏大的
    max_freq 至多让窗口暂时不收缩、不会污染最终答案；2) 收缩条件是 `(right-left+1) - max_freq
    > k` 而不是 `>=`，等于 k 时窗口仍然合法（恰好用满 k 次替换机会）；3) 计数字典要用
    `dict.get(ch, 0)` 或 `Counter` 而不是假设字符一定已经在字典里，否则首次出现的字符会
    KeyError。
    """
    count: dict[str, int] = {}
    left = 0
    max_freq = 0
    best = 0
    for right, ch in enumerate(s):
        count[ch] = count.get(ch, 0) + 1
        max_freq = max(max_freq, count[ch])
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


def find_substring(s: str, words: list[str]) -> list[int]:
    """
    【题意】给定字符串 s 和长度相同的单词列表 words（单词可以重复），找出 s 中恰好是 words
    所有单词按任意顺序拼接而成的子串的起始下标（不允许多字符、不允许少字符）。
    【思路】和"找字母异位词"是同一个"定长窗口 + 词频比较"模型，只是这里窗口内比较的基本单位
    从"单个字符"换成了"整个单词"：窗口总长度固定为 `len(words) * len(word)`，对每个候选起始
    位置，把窗口按单词长度切成若干段，统计每段单词出现的次数，和 words 的词频表逐一比较是否
    完全一致（多退少补都不行）。
    【复杂度】时间 O((n - total_len) * num_words)（n = len(s)，对每个起始位置都要切出
    num_words 个单词做比较，未使用逐字符滑动优化），空间 O(num_words)（词频表）。
    【易错点】1) 单词长度必须一致才能这样切分，题目保证了这一点，但如果误用变长单词就完全不
    适用这个模型；2) 词频比较要用"计数是否超过需求"而不是"集合是否包含"，因为 words 里允许
    重复单词（比如 ["word","good","best","word"] 里 word 出现两次）；3) 起始位置的合法范围是
    `range(len(s) - total_len + 1)`，容易多算或少算一个边界导致漏判最后一个可能的起点。
    """
    if not s or not words:
        return []
    word_len = len(words[0])
    num_words = len(words)
    total_len = word_len * num_words
    if len(s) < total_len:
        return []
    need = Counter(words)
    res: list[int] = []
    for i in range(len(s) - total_len + 1):
        seen: dict[str, int] = defaultdict(int)
        ok = True
        for j in range(num_words):
            start = i + j * word_len
            w = s[start:start + word_len]
            if w not in need:
                ok = False
                break
            seen[w] += 1
            if seen[w] > need[w]:
                ok = False
                break
        if ok:
            res.append(i)
    return res


def smallest_range(nums: list[list[int]]) -> list[int]:
    """
    【题意】给定 k 个各自非递减排序的整数列表，找一个最小的区间 [a, b]，使得每个列表里至少
    有一个数落在这个区间内。
    【思路】把"窗口"从"一个数组上的连续下标区间"泛化成"同时在 k 个有序数组上各选一个下标形成
    的组合"：用一个最小堆维护"当前每个列表各自选中的那个数"里最小的一个，同时维护这些数里的
    最大值 cur_max。堆顶（当前最小值）到 cur_max 的区间就是一个候选答案；每次弹出堆顶后，把
    该列表的下一个数（一定更大，因为列表有序）压入堆，更新 cur_max，重复直到某个列表被耗尽
    （耗尽后不可能再凑齐"每个列表都有代表"，必须停止）。
    【复杂度】时间 O(N log k)（N 是所有列表长度之和，每个数最多入堆出堆一次，堆大小恒为 k），
    空间 O(k)。
    【易错点】1) 终止条件是"任意一个列表被取到末尾"，而不是所有列表都被取完——因为一旦某个
    列表没有更多候选可换，就再也不可能同时覆盖所有 k 个列表；2) 每次只应该把"当前堆顶所属列表"
    的下一个数压入堆，而不是同时推进所有列表，这样才能保证堆里始终恰好是"每个列表当前选中的
    那个数"；3) 更新答案要用"严格小于"（区间更短才更新），且区间长度的比较要用 `cur_max - val`
    而不是先入为主地比较 val 本身。
    """
    heap: list[tuple[int, int, int]] = []
    cur_max = float("-inf")
    for i, lst in enumerate(nums):
        heapq.heappush(heap, (lst[0], i, 0))
        cur_max = max(cur_max, lst[0])
    best_start, best_end = float("-inf"), float("inf")
    while True:
        val, i, j = heapq.heappop(heap)
        if cur_max - val < best_end - best_start:
            best_start, best_end = val, cur_max
        if j + 1 == len(nums[i]):
            break
        next_val = nums[i][j + 1]
        cur_max = max(cur_max, next_val)
        heapq.heappush(heap, (next_val, i, j + 1))
    return [best_start, best_end]


def _at_most_k_distinct(nums: list[int], k: int) -> int:
    """辅助函数：统计"至多包含 k 种不同数字"的子数组个数（k <= 0 时天然返回 0）。"""
    if k <= 0:
        return 0
    count: dict[int, int] = {}
    left = 0
    total = 0
    for right, x in enumerate(nums):
        count[x] = count.get(x, 0) + 1
        while len(count) > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                del count[nums[left]]
            left += 1
        total += right - left + 1
    return total


def subarrays_with_k_distinct(nums: list[int], k: int) -> int:
    """
    【题意】给定整数数组 nums 和整数 k，统计恰好包含 k 种不同整数的连续子数组个数。
    【思路】"恰好 k 种"的收缩条件不像"至多 k 种"那样能直接用一次收缩循环表达——恰好等于某个
    值，不满足滑动窗口"越收缩越合法"的单调性（一个子数组去掉一个字符，distinct 数可能不变
    也可能减少，无法简单用一个方向的收缩来维护"恰好"）。这里的核心转化技巧是组合数学里常见的
    "恰好 = 至多 - 至多少一个"：`恰好K个 = 至多K个 - 至多(K-1)个`，因为"至多 K 种"这个条件
    对收缩窗口是单调的（可以直接复用一个更简单的辅助函数 `_at_most_k_distinct`），把一个不好
    直接滑窗的问题，转化成两次"性质良好"的滑窗问题相减。
    【复杂度】时间 O(n)（两次线性扫描），空间 O(k)。
    【易错点】1) 千万不要尝试直接维护"恰好 k 种"的收缩循环，这个方向没有单调性，写出来的代码
    大概率是错的；2) `_at_most_k_distinct(nums, k - 1)` 在 k=1 时退化成 `_at_most_k_distinct
    (nums, 0)`，必须能正确返回 0（辅助函数已经对 `k <= 0` 做了特判），否则 k=1 这个边界会
    出错；3) 这个"至多 - 至多"的减法技巧不是本题独有，只要看到"恰好满足某个计数条件"的子数组
    /子串统计题，都可以先想一下能不能转化成这种形式。
    """
    return _at_most_k_distinct(nums, k) - _at_most_k_distinct(nums, k - 1)


def find_max_average(nums: list[int], k: int) -> float:
    """
    【题意】给定整数数组 nums 和整数 k，找出长度为 k 的连续子数组的最大平均值。
    【思路】窗口大小固定为 k，属于滑动窗口里最简单的"定长窗口平移"变体：先求出前 k 个数的和
    作为初始窗口，之后每平移一步只需要"减去移出窗口的那个数、加上新移入的那个数"（O(1) 更新），
    不需要每次都重新求和整个窗口。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 全程只需要维护"窗口和"，最后再统一除以 k 求平均，不要在每一步都做除法（多做
    无谓的浮点运算，还可能因为提前转成小数而在比较大小时引入不必要的精度问题）；2) 平移窗口
    时新旧元素的下标关系是 `nums[i] - nums[i - k]`，容易把 `i - k` 写成 `i - k - 1` 或
    `i - k + 1` 之类的偏移错误；3) k 大于数组长度是不合法输入，本实现假设调用方保证 k 合法，
    不做防御性检查（和 Phase 1 其它函数的假设一致）。
    """
    window = sum(nums[:k])
    best = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i - k]
        best = max(best, window)
    return best / k


def _self_test() -> None:
    assert longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2) == 6
    assert longest_ones(
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3
    ) == 10

    assert longest_subarray([1, 1, 0, 1]) == 3
    assert longest_subarray([0, 1, 1, 1, 0, 1, 1, 0, 1]) == 5
    assert longest_subarray([1, 1, 1]) == 2

    assert character_replacement("ABAB", 2) == 4
    assert character_replacement("AABABBA", 1) == 4

    assert sorted(find_substring("barfoothefoobarman", ["foo", "bar"])) == [0, 9]
    assert find_substring(
        "wordgoodgoodgoodbestword", ["word", "good", "best", "word"]
    ) == []

    assert smallest_range(
        [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
    ) == [20, 24]

    assert subarrays_with_k_distinct([1, 2, 1, 2, 3], 2) == 7
    assert subarrays_with_k_distinct([1, 2, 1, 3, 4], 3) == 3

    assert abs(find_max_average([1, 12, -5, -6, 50, 3], 4) - 12.75) < 1e-6

    print(
        "[PASS] p02_sliding_window_ii: 7 题"
        "（最大连续1的个数III/删除元素后全1最长子数组/替换后最长重复字符/"
        "串联所有单词的子串/最小区间/K个不同整数的子数组/子数组最大平均数I）全部通过"
    )


if __name__ == "__main__":
    _self_test()
