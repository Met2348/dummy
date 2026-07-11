"""滑动窗口 · Phase 3 竞赛级补充：7 道强调"多技巧组合"和边界严谨性的窗口变体，不重讲基础框架。"""
from __future__ import annotations


def max_satisfied(customers: list[int], grumpy: list[int], minutes: int) -> int:
    """
    【题意】LC1052.爱生气的书店老板。customers[i] 是第 i 分钟进店的顾客数，grumpy[i] 为 1
    表示老板这一分钟很生气（这一分钟的顾客都不满意）。老板有一次"secret technique"，可以让
    自己连续 minutes 分钟内保持不生气。求全天顾客满意总数的最大值。
    【思路】把答案拆成两部分：一部分是"天生就会满意"的顾客（grumpy[i]==0 的那些分钟，不受
    technique 影响，直接累加为 base）；另一部分是"technique 能追加挽回多少个生气分钟的顾客"，
    这是一个长度固定为 minutes 的滑动窗口最大子数组和问题——窗口内只统计 grumpy[i]==1 的那些
    customers[i]（因为窗口盖住 grumpy==0 的分钟不会带来额外收益，本来就已经算在 base 里）。
    用定长窗口平移（进一个出一个）求出这部分的最大值 extra，最终答案是 base + extra。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) base 只统计 grumpy[i]==0 的分钟，千万不要把 technique 窗口覆盖的 grumpy==0
    分钟也重复计入 window_gain，否则会重复计算；2) 窗口滑动时"移出"的判断要看被移出那一分钟
    原本是否生气（grumpy[i-minutes]==1），如果那一分钟本来就不生气，减去它的 customers 会
    错误地把已经计入 base 的部分也减掉；3) technique 是"可选"的，如果全天都不生气，extra 会
    自然算出 0（不影响最终答案是 base），不需要特判"要不要使用 technique"。
    """
    base = sum(c for c, g in zip(customers, grumpy) if g == 0)
    window_gain = 0
    extra = 0
    for i, (c, g) in enumerate(zip(customers, grumpy)):
        if g == 1:
            window_gain += c
        if i >= minutes and grumpy[i - minutes] == 1:
            window_gain -= customers[i - minutes]
        extra = max(extra, window_gain)
    return base + extra


def max_score(card_points: list[int], k: int) -> int:
    """
    【题意】LC1423.可获得的最大点数。一排卡牌 cardPoints，每次只能从最左或最右取一张，必须
    恰好取 k 张，求取到的点数总和最大值。
    【思路】正面枚举"每一步从左还是从右取"是指数级的，但反过来想：**取走的 k 张牌永远是"最左
    的若干张 + 最右的若干张"这两段拼起来**，等价于"剩下 n-k 张牌构成一段连续区间"。于是问题
    转化成："在长度为 n 的数组里，找一个长度为 n-k 的连续窗口，使得窗口外（也就是被取走）的
    总和最大"——这就是总和减去一个长度固定的滑动窗口的最小和。实现上更直接的写法是：先假设
    "全部 k 张都从左边取"（初始窗口 = 前 k 个数的和），然后每次把窗口最左边的一个换成从右边
    再多拿一个（左边少拿一个、右边多拿一个），一共有 k+1 种"从左拿 k-i 张、从右拿 i 张"的
    组合（i=0..k），取其中的最大值。
    【复杂度】时间 O(k)（枚举 k+1 种切分方式），空间 O(1)。
    【易错点】1) 这题的窗口不是"从左往右平移一次"，而是"左边界持续右缩、右边界持续左扩"，两个
    方向同时变化，容易和普通定长滑窗搞混，写的时候要清楚 `card_points[n-i]` 和
    `card_points[k-i]` 分别对应哪一步的增减；2) k 等于数组长度 n 的边界情况（必须拿走所有牌）
    要能正确处理——按上面的算法会自然算出总和，不需要特判；3) 初始窗口是"前 k 个数的和"而不是
    "0"，如果初始化成 0 再累加 k 次会把第一次的比较基准算错。
    """
    n = len(card_points)
    window = sum(card_points[:k])
    best = window
    for i in range(1, k + 1):
        window += card_points[n - i] - card_points[k - i]
        best = max(best, window)
    return best


def decrypt(code: list[int], k: int) -> list[int]:
    """
    【题意】LC1652.拆炸弹。code 是循环数组，k>0 时把第 i 位替换成"接下来 k 个数之和"，k<0 时
    替换成"前面 |k| 个数之和"，k==0 时全部替换成 0。
    【思路】这是"固定大小窗口在环形数组上平移"的定长滑动窗口：k>0 时，位置 i 对应的窗口是
    `code[i+1 .. i+k]`（下标取模 n）；从 i 移动到 i+1 时，窗口整体后移一位——移出
    `code[i+1]`，移入 `code[i+k+1]`（都取模 n）。k<0 的情况对称：位置 i 对应窗口是
    `code[i-|k| .. i-1]`，从 i 到 i+1 时，移出 `code[i-|k|]`，移入 `code[i]`。先计算出 i=0
    处的初始窗口和，再用"移出一个、移入一个"的方式 O(1) 递推后续位置，而不是每个位置都重新
    累加 k 个数（那样是 O(n*k)）。
    【复杂度】时间 O(n)（初始窗口 O(k)，之后每步 O(1)，k<=n 所以整体是 O(n)），空间 O(n)
    （返回结果数组）。
    【易错点】1) 环形数组的下标必须处处取模 `% n`，包括"移出/移入"用到的下标计算，漏掉任何
    一处取模都会在 i 接近数组边界时越界或算错；2) k>0 和 k<0 是两套不同方向的递推公式（移出/
    移入的下标表达式不同），不能用同一套代码直接套用两种情况，容易把方向搞反；3) k==0 必须在
    最开始就特判直接返回全 0 数组，因为"接下来 0 个数之和"这件事没有对应的窗口可以滑动。
    """
    n = len(code)
    res = [0] * n
    if k == 0:
        return res
    ak = abs(k)
    if k > 0:
        window = sum(code[s % n] for s in range(1, ak + 1))
        res[0] = window
        for i in range(1, n):
            window += code[(i + ak) % n] - code[i % n]
            res[i] = window
    else:
        window = sum(code[(-s) % n] for s in range(1, ak + 1))
        res[0] = window
        for i in range(1, n):
            window += code[(i - 1) % n] - code[(i - 1 - ak) % n]
            res[i] = window
    return res


def longest_beautiful_substring(word: str) -> int:
    """
    【题意】LC1839.所有元音按顺序排布的最长子字符串。word 只含元音字母。若一个子串同时满足
    "5 个元音 a/e/i/o/u 都至少出现一次"且"字母严格按 a<=e<=i<=o<=u 的顺序非降排列"，则称为
    beautiful；求最长 beautiful 子串的长度，不存在则返回 0。
    【思路】直接用"扩张 + 收缩"的窗口很难判断"顺序是否被破坏"这个条件（一旦出现降序就必须
    整体重新开始，而不是像普通窗口那样只收缩左边界）。更干净的做法是先把 word 压缩成"连续
    相同字符"的分组列表（游程编码），比如 "aaeiiou" 变成 [(a,2),(e,1),(i,2),(o,1),(u,1)]。
    这样"一段非降的 beautiful 子串"就对应"连续 5 个分组，字符恰好是 a,e,i,o,u"——因为分组
    之间字符不同，只要相邻分组不违反字母序（分组本身就是逐段扫描时天然形成的），检查任意连续
    5 个分组的字符序列是否恰好是 "aeiou"，是的话这 5 段的字符总数就是一个候选答案。
    【复杂度】时间 O(n)（游程编码一次遍历 + 之后检查窗口是分组数量级，仍是线性），空间 O(n)
    （最坏情况每个字符自成一组）。
    【易错点】1) 游程编码分组本身不保证"整体非降"——如果字符出现降序（比如 "ea"，e 后面接
    比它小的 a），这两个字符会被分成两个独立的组，但它们之间不满足非降关系，直接检查"5 个
    连续分组字符是否为 aeiou"这个条件天然就排除了降序的情况（因为如果中间夹杂了降序，5 个
    连续分组的字符序列不可能恰好是 a,e,i,o,u 这个严格递增的顺序）；2) 判断连续 5 组的字符
    序列要精确匹配 "aeiou" 这五个字符按顺序排列，而不是"包含这 5 个字符"；3) 循环范围是
    `range(len(groups) - 4)`，分组数不足 5 个时应该自然得到空区间（不会报错），不需要额外
    特判。
    """
    order = "aeiou"
    groups: list[tuple[str, int]] = []
    for ch in word:
        if groups and groups[-1][0] == ch:
            c, cnt = groups[-1]
            groups[-1] = (c, cnt + 1)
        else:
            groups.append((ch, 1))
    best = 0
    for i in range(len(groups) - 4):
        five = groups[i : i + 5]
        if "".join(c for c, _ in five) == order:
            best = max(best, sum(cnt for _, cnt in five))
    return best


def num_subarray_bounded_max(nums: list[int], left: int, right: int) -> int:
    """
    【题意】LC795.区间子数组个数。给定数组 nums 和整数 left、right，统计有多少个非空连续
    子数组，其最大值落在闭区间 [left, right] 内。
    【思路】直接维护"窗口最大值在 [left, right] 内"这个条件不具备滑动窗口需要的单调性（和
    LC992 K 个不同整数的子数组同源）：正确的转化是 **"最大值在 [left, right] 内的子数组数"
    ="最大值 <= right 的子数组数" - "最大值 <= left-1 的子数组数"**——这是一次容斥，把一个
    "范围限定"的计数问题拆成两个"上界限定"的计数问题相减。而"最大值 <= x 的子数组个数"是一个
    经典的线性扫描：维护一个"连续可行段长度" run，只要当前元素 <= x，run 就自增（这个元素能
    和它前面所有可行的元素组成新的合法子数组，增加的合法子数组个数恰好等于 run），一旦遇到
    元素 > x，run 直接清零（任何跨过这个元素的子数组都不可能满足"最大值 <= x"）。
    【复杂度】时间 O(n)（两次线性扫描），空间 O(1)。
    【易错点】1) 辅助函数 `count_at_most(bound)` 每次累加的是当前的 run（而不是 1），因为
    以当前位置为右端点、且满足条件的子数组一共有 run 个（左端点可以是 run 个不同的合法起点）；
    2) 计算"下界"用的是 `left - 1` 而不是 `left`，因为要排除的是"最大值严格小于 left"的子
    数组，写成 `count_at_most(left)` 会多减掉"最大值恰好等于 left"的那部分，答案偏小；
    3) run 清零的条件是"当前元素 > bound"，不是">= bound"，等于 bound 的元素仍然是合法的。
    """

    def count_at_most(bound: int) -> int:
        total = 0
        run = 0
        for x in nums:
            if x <= bound:
                run += 1
            else:
                run = 0
            total += run
        return total

    return count_at_most(right) - count_at_most(left - 1)


def find_repeated_dna_sequences(s: str) -> list[str]:
    """
    【题意】LC187.重复的DNA序列。s 只含 'A'/'C'/'G'/'T'，找出所有在 s 中出现超过一次的
    长度为 10 的子串，结果不重复（每种重复子串只返回一次）。
    【思路】这是长度固定为 10 的窗口在字符串上平移：用一个哈希 set 记录"已经见过的 10 位
    子串"，每平移一步就取出当前窗口对应的子串，若它已经在 seen 集合里，说明这是第二次及以后
    出现，加入结果集合（用 set 去重，保证同一个重复子串不会被加入多次）；否则把它加进 seen。
    数据规模不大时直接对每个窗口做字符串切片即可；如果 s 很长，可以把 'A''C''G''T' 编码成
    2 bit（4 种字符共需 2 位），10 个字符编码进一个 20 位整数，用整数代替字符串做哈希 key，
    进一步减小常数开销（本题给出的实现为了清晰起见使用字符串切片版本）。
    【复杂度】时间 O(n)（n 为 s 的长度，每步切片是 O(10) 常数），空间 O(n)（最坏情况 seen
    集合存储接近 n 个不同的子串）。
    【易错点】1) 循环范围是 `range(len(s) - 9)`（窗口起点最多到 len(s)-10），写成
    `len(s) - 10` 会漏掉最后一个合法窗口；2) 结果集合必须用 set 而不是 list 去重，否则一个
    出现 3 次以上的子串会被重复加入结果多次；3) "已出现过"的判断依据是 seen 集合而不是结果
    集合本身——用结果集合去判断"是否已经加过"在逻辑上也能工作，但用两个独立的集合（seen 记录
    "见过"，repeated 记录"已判定为重复"）意图更清晰、不容易在后续维护时改错。
    """
    seen: set[str] = set()
    repeated: set[str] = set()
    for i in range(len(s) - 9):
        sub = s[i : i + 10]
        if sub in seen:
            repeated.add(sub)
        else:
            seen.add(sub)
    return list(repeated)


def _self_test() -> None:
    assert max_satisfied([1, 0, 1, 2, 1, 1, 7, 5], [0, 1, 0, 1, 0, 1, 0, 1], 3) == 16

    assert max_score([1, 2, 3, 4, 5, 6, 1], 3) == 12
    assert max_score([2, 2, 2], 2) == 4
    assert max_score([9, 7, 7, 9, 7, 7, 9], 7) == 55

    assert decrypt([5, 7, 1, 4], 3) == [12, 10, 16, 13]
    assert decrypt([1, 2, 3, 4], 0) == [0, 0, 0, 0]
    assert decrypt([2, 4, 9, 3], -2) == [12, 5, 6, 13]

    assert longest_beautiful_substring("aeiaaioaaaaeiiiiouuuooaauuaeiu") == 13
    assert longest_beautiful_substring("aeeeiiiioooauuuaeiou") == 5

    assert num_subarray_bounded_max([2, 1, 4, 3], 2, 3) == 3
    assert num_subarray_bounded_max([2, 9, 2, 5, 6], 2, 8) == 7

    assert sorted(find_repeated_dna_sequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")) == sorted(
        ["AAAAACCCCC", "CCCCCAAAAA"]
    )
    assert find_repeated_dna_sequences("AAAAAAAAAAAAA") == ["AAAAAAAAAA"]

    print(
        "[PASS] p02_sliding_window_iii: 6 题"
        "（爱生气的书店老板/可获得的最大点数/拆炸弹/所有元音按顺序排布的最长子字符串/"
        "区间子数组个数/重复的DNA序列）全部通过（滑动窗口中位数改归入 15-heap-topk 类，避免和该类重复）"
    )


if __name__ == "__main__":
    _self_test()
