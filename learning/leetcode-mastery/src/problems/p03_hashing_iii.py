"""哈希表 · Phase 3 竞赛级补充：7 道题，前 4 道巩固"计数/去重/双射"基础用法，后 3 道换成更有
挑战性的真哈希设计题（前缀和配合哈希、桶哈希、原地哈希-进阶），匹配竞赛级定位。不重讲基础框架。
"""
from __future__ import annotations

from collections import Counter


def number_of_boomerangs(points: list[list[int]]) -> int:
    """
    【题意】LC447.回旋镖的数量。平面上 n 个互不相同的点，"回旋镖"是一个有序三元组
    (i, j, k)，满足 i 到 j 的距离等于 i 到 k 的距离（j、k 顺序不同算不同的回旋镖）。求回旋镖
    总数。
    【思路】枚举每个点 i 当"顶点"，用哈希表统计"其余每个点到 i 的距离"出现的次数（用距离的
    平方代替开根号后的真实距离，避免浮点误差且更快）：如果有 c 个点到 i 的距离相同，这 c 个点
    里任选 2 个（有序）都能和 i 组成一个回旋镖，贡献 `c * (c - 1)` 种（排列数 P(c,2)）。对每个
    顶点 i 分别统计、累加所有距离分组的贡献，就是总回旋镖数。
    【复杂度】时间 O(n^2)（外层枚举顶点 i，内层对其余 n-1 个点计算距离并计数），空间 O(n)
    （每个顶点的距离计数表，最坏情况有 n-1 个不同距离）。
    【易错点】1) 距离要用平方值（`dx*dx + dy*dy`）而不是开根号后的浮点数，浮点距离在哈希表里
    做 key 存在精度误差风险（同一距离因为开根号精度不同被误判为不同 key）；2) 每个点到自己的
    距离是 0，也会被计入这个点自己的计数表里，但因为只有它自己一个点距离为 0（distinct points
    前提下），贡献 `1 * 0 = 0`，不影响结果，不需要专门排除自身；3) 结果是"有序"三元组，也就是
    `c*(c-1)` 而不是组合数 `c*(c-1)/2`，J 和 K 互换算两个不同的回旋镖。
    """
    total = 0
    for x1, y1 in points:
        dist_count: dict[int, int] = {}
        for x2, y2 in points:
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist_count[d] = dist_count.get(d, 0) + 1
        for cnt in dist_count.values():
            total += cnt * (cnt - 1)
    return total


def common_chars(words: list[str]) -> list[str]:
    """
    【题意】LC1002.查找共用字符。给定字符串数组 words，返回在**每一个**字符串里都出现过的
    字符列表（重复字符按最少出现次数重复输出），顺序任意。
    【思路】"在所有字符串里都出现" 的次数不能超过任何一个字符串里该字符的出现次数——也就是
    "取所有字符串中该字符计数的最小值"。用第一个字符串的字符计数表 Counter 作为初始候选，
    然后依次和后面每个字符串的计数表做"逐 key 取最小值"（`Counter` 的 `&` 运算正好定义为
    "取两个计数表每个 key 的最小值"，语义完全匹配这里的需求），全部处理完之后剩下的计数表就是
    答案，用 `.elements()` 按计数展开成列表。
    【复杂度】时间 O(sum of len(word))（每个字符串遍历一次建计数表），空间 O(1)（字符集大小
    固定为 26，计数表大小有上界，不随输入增长）。
    【易错点】1) 不能用 set 求交集再判断"是否出现"，那样会丢失"同一个字符出现几次"这个信息
    （比如两个字符串都有两个 'l'，用 set 求交集只能得到一次 'l'）；2) `Counter & Counter`
    是"取每个 key 计数的最小值"，不是"取并集"或者"相加"，如果自己手写这个逻辑，遍历某一个
    Counter 的 key 判断 `min(c1[k], c2[k])` 时要注意如果某个 key 只在其中一个 Counter 里出现
    （另一个是默认值 0），最小值自然是 0，不需要额外的存在性判断（`Counter` 对不存在的 key
    默认返回 0，这一点和普通 dict 不同）；3) `words` 至少有一个元素，用 `words[0]` 初始化前
    不需要额外判空，但如果 `words` 为空列表访问 `words[0]` 会抛异常，实际题目保证长度 >= 1。
    """
    common = Counter(words[0])
    for w in words[1:]:
        common &= Counter(w)
    return list(common.elements())


def unique_occurrences(arr: list[int]) -> bool:
    """
    【题意】LC1207.独一无二的出现次数。给定整数数组 arr，判断每个不同值的出现次数是否互不
    相同（是则返回 True）。
    【思路】用哈希表（`Counter`）统计每个值出现的次数，得到一组"次数"的多重集合；题目要求
    这些次数本身互不相同，也就是"把这组次数再丢进一个 set 里，大小应该和原来的次数个数完全
    相等"（如果有两个值出现次数相同，去重后 set 会变小）。
    【复杂度】时间 O(n)（一次遍历建计数表 + 一次对计数值集合去重），空间 O(n)（最坏情况 arr
    里的值互不相同，计数表大小为 n）。
    【易错点】1) 判断对象是"每个值出现的次数"这个二级统计量，而不是"值本身"——很容易一上来
    就写成判断 arr 里数值是否重复，那是另一个问题（LC217）；2) 用 `set(counts)` 去重后要比较
    的是"长度是否相等"，而不是直接判断 set 和原 counts 是否内容相同（counts 本身是一个多重
    值列表，两者类型不同不能直接比较相等）；3) 负数、0 都是合法的值，`Counter` 对任意可哈希
    的值都适用，不需要对值域做特殊处理。
    """
    counts = Counter(arr).values()
    return len(counts) == len(set(counts))


def is_alien_sorted(words: list[str], order: str) -> bool:
    """
    【题意】LC953.验证外星语词典。order 给出外星语字母表的字母顺序（26 个小写字母的一个
    排列），判断 words 是否按这个字母表顺序字典序排列。
    【思路】把 order 映射成"字符 → 排名"的哈希表（`{c: i for i, c in enumerate(order)}`），
    这样比较两个"外星语单词"的大小，就等价于把每个单词转换成"排名序列"后按普通列表比较规则
    （Python 原生支持逐元素比较列表大小，含"较短列表是较长列表前缀时更小"这条规则，正好和
    字典序定义一致）比较相邻两个转换后的列表。只要 words 里任意相邻一对不满足"前者 <= 后者"，
    整体就不是排序好的。
    【复杂度】时间 O(总字符数)（每个单词转换一次排名序列，序列之间的比较也是线性的），空间
    O(总字符数)（存储所有单词转换后的排名序列）。
    【易错点】1) 必须转换成"排名"之后再比较，不能直接比较字符本身——字符的 ASCII 顺序和 order
    定义的外星语顺序通常不一致，直接 `word1 < word2` 用的是标准字母表顺序，是错误的比较依据；
    2) 一个单词是另一个单词的前缀时（比如 "app" 和 "apple"），字典序规定"更短的前缀在前"，
    这一点 Python 的列表比较（`[0,15,15] < [0,15,15,11,4]` 为 True）天然满足，不需要额外
    处理，但如果自己手写比较函数容易漏掉这个边界；3) `rank` 表要覆盖 order 里全部 26 个字母，
    如果 order 长度不足 26（题目保证是完整排列，但防御性编程时要留意）会导致某些字符查表时
    KeyError。
    """
    rank = {c: i for i, c in enumerate(order)}

    def to_key(word: str) -> list[int]:
        return [rank[c] for c in word]

    return all(to_key(words[i]) <= to_key(words[i + 1]) for i in range(len(words) - 1))


def h_index(citations: list[int]) -> int:
    """
    【题意】LC274.H指数。给定研究者每篇论文的被引用次数数组 citations，求其 h 指数——最大的
    h 使得至少有 h 篇论文的引用次数都 >= h，且其余论文的引用次数都 <= h。
    【思路】排序后从高到低找临界点是常见思路，但可以用"桶计数"（哈希表思想的数组化版本）做到
    O(n)：因为 h 指数的取值不可能超过论文总数 n，把每篇论文的引用次数"压缩"到 [0, n] 区间
    （引用次数超过 n 的论文，只要知道它 >= n 就够了，不需要区分具体多大，所以统一放进
    `min(citations[i], n)` 这个桶），用一个长度 n+1 的桶数组统计每个"引用次数"（0..n）对应
    多少篇论文。然后从 h=n 往 h=0 倒着扫，累加"引用次数 >= 当前 h"的论文数 `total`（每次把
    桶 buckets[h] 计入 total，因为我们是从大到小扫描，累加起来的 total 天然就是"引用次数 >= h
    的论文数"），一旦 `total >= h` 成立，这个 h 就是答案（从大到小找到的第一个满足条件的 h
    必然是最大的）。
    【复杂度】时间 O(n)（一次分桶 + 一次倒序累加，都是线性扫描，避免了排序的 O(n log n)），
    空间 O(n)（桶数组长度 n+1）。
    【易错点】1) 桶下标要用 `min(c, n)` 做截断，而不是直接用 `c` 当下标——引用次数可能远大于
    论文篇数 n，直接当下标会数组越界，而且"具体大多少"对 h 指数的计算毫无意义（只要知道
    >= n 即可）；2) 倒序累加时 `total` 是"迄今为止" `>= h` 的论文数的滚动和，不能在每个 h
    处重新从头计算，否则退化成 O(n^2)；3) 找到第一个满足 `total >= h` 的 h 就要立刻返回，
    因为是从大到小扫描，第一个满足条件的必然是最大的合法 h，继续往下扫只会找到更小的答案。
    """
    n = len(citations)
    buckets = [0] * (n + 1)
    for c in citations:
        buckets[min(c, n)] += 1
    total = 0
    for h in range(n, -1, -1):
        total += buckets[h]
        if total >= h:
            return h
    return 0


def find_max_length(nums: list[int]) -> int:
    """
    【题意】LC525.连续数组。给定只含 0、1 的数组 nums，求 0 和 1 数量相等的最长连续子数组
    的长度。
    【思路】把 0 看成 -1、1 看成 +1，问题就转化成"求前缀和第一次和某个更早的前缀和相等时，
    两个位置之间的最大距离"——因为一段区间的和为 0，等价于这段区间左右两个端点（不含左端点，
    即前缀和的下标）对应的前缀和相等。用哈希表 first_index 记录"每个前缀和第一次出现的下标"
    （初始化 `{0: -1}`，代表"还没开始扫描时前缀和是 0，位置在 -1"，这样处理从下标 0 开始就
    满足条件的情况）：扫描时累加当前前缀和，如果这个前缀和值之前出现过，说明中间这一段的和
    为 0（0/1 数量相等），用 `当前下标 - 第一次出现的下标` 更新最长长度；如果没出现过，记录
    这是它第一次出现的位置（只记录第一次，因为要让长度最大，配对的位置应该尽量靠左）。
    【复杂度】时间 O(n)，空间 O(n)（哈希表最多存 n+1 个不同的前缀和值）。
    【易错点】1) 初始化 `first_index` 必须包含 `{0: -1}` 这个哨兵，否则会漏掉"从数组开头
    就满足条件"的情况（比如整个数组本身 0/1 数量相等）；2) 同一个前缀和值只应该记录**第一次**
    出现的下标——如果每次出现都更新下标，后续用"当前下标 - 记录的下标"算出来的会是更短的区间
    而不是最长的；3) 0 要映射成 -1 而不是保持 0，如果直接对 0/1 求和，任何全 0 或全 1 的前缀
    都不会产生"和相等"这种可判定的信号，必须用 +1/-1 这种"对称"的映射才能让"数量相等"对应到
    "前缀和相等"这个可哈希判断的条件。
    """
    first_index: dict[int, int] = {0: -1}
    running = 0
    best = 0
    for i, x in enumerate(nums):
        running += 1 if x == 1 else -1
        if running in first_index:
            best = max(best, i - first_index[running])
        else:
            first_index[running] = i
    return best


def contains_nearby_almost_duplicate(nums: list[int], index_diff: int, value_diff: int) -> bool:
    """
    【题意】LC220.存在重复元素 III。给定数组 nums 和两个整数 indexDiff、valueDiff，判断是否
    存在一对下标 i != j，同时满足 `abs(i - j) <= indexDiff` 且 `abs(nums[i] - nums[j]) <=
    valueDiff`。
    【思路】维护一个大小不超过 indexDiff 的滑动窗口，用"桶哈希"技巧在窗口内做 O(1) 近似查找：
    把值域按宽度 `w = valueDiff + 1` 分桶（`bucket_id = x // w`），使得**同一个桶里任意两个
    数的差必然 <= valueDiff**（宽度设计的关键）。对每个新元素 x：如果它所在的桶已经有值，
    两者差必然满足条件，直接判真；否则还要检查相邻的左右两个桶（因为满足条件的另一个数可能
    恰好落在相邻桶里，只是没被分进同一个桶），只有相邻桶都不满足才把 x 放进桶里。每处理一个
    新下标就把滑出窗口（下标差超过 indexDiff）的旧值从桶里移除，保证桶里任何时刻只有窗口内
    的元素。
    【复杂度】时间 O(n)（每个元素的桶操作是 O(1)），空间 O(min(n, indexDiff))（桶哈希表大小
    不超过窗口内的元素个数）。
    【易错点】1) 桶宽度必须是 `valueDiff + 1` 而不是 `valueDiff`——如果宽度设成 valueDiff，
    同一个桶内最大差可能达到 `valueDiff`（宽度为 w 的桶内两数最大差是 w-1），宽度差 1 会导致
    漏判边界情况；2) 这个技巧在很多语言（如 C++/Java）里因为整数除法对负数是"向零截断"而非
    "向下取整"，需要对负数特殊处理，否则同样大小的负数会被错误分到不同桶——**Python 的 `//`
    运算符对负数是向下取整（floor division），天然规避了这个经典陷阱**，但如果照抄其他语言的
    实现要格外小心这一点；3) `value_diff` 为负数时条件不可能满足（绝对值不可能小于负数），
    要在最开始特判直接返回 False，否则桶宽度会变成 <= 0 导致除法逻辑出错。
    """
    if value_diff < 0:
        return False
    buckets: dict[int, int] = {}
    width = value_diff + 1
    for i, x in enumerate(nums):
        bucket_id = x // width
        if bucket_id in buckets:
            return True
        if bucket_id - 1 in buckets and abs(x - buckets[bucket_id - 1]) <= value_diff:
            return True
        if bucket_id + 1 in buckets and abs(x - buckets[bucket_id + 1]) <= value_diff:
            return True
        buckets[bucket_id] = x
        if i >= index_diff:
            old_bucket = nums[i - index_diff] // width
            if buckets.get(old_bucket) == nums[i - index_diff]:
                del buckets[old_bucket]
    return False


def _self_test() -> None:
    assert number_of_boomerangs([[0, 0], [1, 0], [2, 0]]) == 2
    assert number_of_boomerangs([[1, 1], [2, 2], [3, 3]]) == 2
    assert number_of_boomerangs([[1, 1]]) == 0

    assert sorted(common_chars(["bella", "label", "roller"])) == sorted(["e", "l", "l"])
    assert sorted(common_chars(["cool", "lock", "cook"])) == sorted(["c", "o"])

    assert unique_occurrences([1, 2, 2, 1, 1, 3]) is True
    assert unique_occurrences([1, 2]) is False
    assert unique_occurrences([-3, 0, 1, -3, 1, 1, 1, -3, 10, 0]) is True

    assert is_alien_sorted(["hello", "leetcode"], "hlabcdefgijkmnopqrstuvwxyz") is True
    assert (
        is_alien_sorted(["word", "world", "row"], "worldabcefghijkmnpqstuvxyz") is False
    )
    assert is_alien_sorted(["apple", "app"], "abcdefghijklmnopqrstuvwxyz") is False

    assert h_index([3, 0, 6, 1, 5]) == 3
    assert h_index([1, 3, 1]) == 1

    assert find_max_length([0, 1]) == 2
    assert find_max_length([0, 1, 0]) == 2
    assert find_max_length([0, 1, 0, 1, 0, 1, 1]) == 6

    assert contains_nearby_almost_duplicate([1, 2, 3, 1], 3, 0) is True
    assert contains_nearby_almost_duplicate([1, 5, 9, 1, 5, 9], 2, 3) is False
    assert contains_nearby_almost_duplicate([1, 0, 1, 1], 1, 2) is True

    print(
        "[PASS] p03_hashing_iii: 7 题"
        "（回旋镖的数量/查找共用字符/独一无二的出现次数/验证外星语词典/"
        "H指数/连续数组/存在重复元素III）全部通过"
    )


if __name__ == "__main__":
    _self_test()
