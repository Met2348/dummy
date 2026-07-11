"""分类 24：高级字符串算法 —— KMP / 中心扩展 / Rabin-Karp 滚动哈希，真正需要专门算法的字符串题。"""
from __future__ import annotations


def _build_lps(pattern: str) -> list[int]:
    """KMP 预处理：lps[i] 表示 pattern[0..i] 这一段里，"最长的、既是前缀又是后缀"的
    长度（这个前缀/后缀本身不能等于整个 pattern[0..i]，必须是真前缀/真后缀）。这是
    KMP 算法的核心数据结构，str_str_kmp 和 repeated_substring_pattern 共用。"""
    n = len(pattern)
    lps = [0] * n
    length = 0
    i = 1
    while i < n:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps


def str_str_kmp(haystack: str, needle: str) -> int:
    """
    【题意】在字符串 haystack 中找出 needle 第一次出现的起始下标；如果 needle 不是
    haystack 的子串，返回 -1。
    【思路】朴素做法是对 haystack 的每个起点都暴力比较一遍 needle，一旦某个字符
    不匹配就把起点往后挪一位、从头重新比较——这样做的浪费在于：匹配失败之前，我们
    其实已经知道"已经匹配上的这一段"长什么样，却因为起点只挪了一位，白白扔掉了这段
    已知信息。KMP 的核心 insight 是：当匹配到 needle 的第 j 个字符失败时，我们知道
    haystack 里刚刚被匹配的这一段等于 `needle[0..j-1]`；如果 `needle[0..j-1]` 自己
    存在一个"长度为 L 的前缀，同时也是它的后缀"，那么这个前缀必然已经在 haystack
    里出现过（就是刚匹配上的那段的开头部分），下一次比较可以直接从 `needle[L]` 继续
    比较，而不必把 needle 的指针整个退回到 0——这正是 lps 数组（最长相同前后缀长度）
    存的信息：失配时，把 needle 的指针 j 跳到 `lps[j-1]`，跳过一大截已经证明不可能
    匹配的位置，haystack 的指针 i 则完全不需要回退。
    【复杂度】时间 O(m+n)（构造 lps 数组 O(m)，主循环里 haystack 指针只前进不回退，
    均摊 O(n)；m=len(needle), n=len(haystack)）；空间 O(m)（lps 数组）。
    【易错点】1) 失配时容易直接把 j 重置为 0（退化成朴素算法），而不是跳到
    `lps[j-1]`，白白丢失了 KMP 相对暴力法的加速效果；2) `lps[j-1]` 里的下标是
    "长度"，不是"指向 needle 里某个具体字符的下标"，容易和 j 本身混淆；3) needle
    为空串时按惯例应直接返回 0（在任意字符串的开头，空串总能"找到"）。
    """
    if not needle:
        return 0
    lps = _build_lps(needle)
    i = j = 0
    while i < len(haystack):
        if haystack[i] == needle[j]:
            i += 1
            j += 1
            if j == len(needle):
                return i - j
        elif j != 0:
            j = lps[j - 1]
        else:
            i += 1
    return -1


def repeated_substring_pattern(s: str) -> bool:
    """
    【题意】给定字符串 s，判断它是否能由某个更短的子串重复多次拼接而成（比如
    "abab" 是 "ab" 重复两次，"abcabcabcabc" 是 "abc" 重复四次）。
    【思路】最直接的写法是利用一个巧妙的字符串性质：如果 s 确实是某个子串重复
    k>=2 次拼接而成，那么把 s 和自己拼接成 s+s，去掉首尾各一个字符之后，s 一定还能
    在这个 (s+s)[1:-1] 里被找到（因为循环移位之后 s 依然完整出现在中间某个位置）；
    反过来如果 s 不是循环重复的，这种巧合不会发生。这个一行写法（`s in (s+s)[1:-1]`）
    虽然简洁，但更能体现"进阶理解"的是用 KMP 的 lps 数组：算出整个字符串 s 自身的
    lps 数组，取最后一个值 `border = lps[-1]`（即 s 的"最长相同前后缀"长度）；如果
    这个 border > 0，且 `len(s)` 恰好能被 `len(s) - border` 整除，说明 s 存在一个
    长度为 `len(s) - border` 的"最小循环节"，可以重复 `len(s) // (len(s) - border)`
    次拼出整个 s——这是因为"最长相同前后缀"这个概念和"字符串的最小周期"在数学上
    是等价的：s 有一个长度为 p 的周期，当且仅当 s 存在一个长度为 `len(s) - p` 的
    相同前后缀。本文件用 lps 数组的写法作为主实现，直接复用和 str_str_kmp 相同的
    KMP 预处理逻辑。
    【复杂度】时间 O(n)（构造 lps 数组）；空间 O(n)。
    【易错点】1) `border == 0`（不存在任何相同前后缀）时不能直接用
    `len(s) % (len(s) - border) == 0` 下结论——这时 `len(s) - border == len(s)`，
    取模恒为 0，会被误判为"存在循环节"，必须先判断 `border > 0` 排除这种情况；
    2) 容易把"最小循环节长度"记成 border 本身，而不是 `len(s) - border`——border
    是"相同前后缀的长度"，循环节长度是"整个串长度减去这个重叠部分"。
    """
    lps = _build_lps(s)
    n = len(s)
    border = lps[-1] if n > 0 else 0
    return border > 0 and n % (n - border) == 0


def count_substrings(s: str) -> int:
    """
    【题意】给定字符串 s，统计它一共有多少个回文子串（同一个子串如果在不同位置出现
    多次，按出现次数分别计数，不是只数不同的回文子串种类）。
    【思路】每一个回文串都有一个"中心"——奇数长度回文串的中心是一个字符，偶数长度
    回文串的中心是相邻两个字符之间的空隙。枚举所有可能的中心（n 个字符中心 + n-1
    个空隙中心，统一写成 n 个字符中心 + n 个"字符与自己右边字符之间"的中心，多出的
    一个越界情况在扩展时会被边界检查自然挡掉），从每个中心往两边同时扩展，只要
    左右两个字符相等就说明又找到了一个以这个中心为对称轴的回文串，计数器加一，
    继续往外扩展，直到左右字符不相等或者越界为止。中心扩展法的复杂度是 O(n^2)，
    对面试场景已经足够；如果需要把复杂度进一步压到 O(n)，需要用 Manacher 算法（利用
    "回文串关于中心对称"这个性质，复用已经算出的对称位置的信息避免重复扩展），但
    Manacher 的实现复杂度和收益比，在真实面试里通常不要求手写，了解存在即可。
    【复杂度】时间 O(n^2)（n 个中心，每个中心最坏扩展 O(n) 次）；空间 O(1)（不算
    输出的计数器）。
    【易错点】1) 只枚举"字符中心"（奇数长度回文）会漏掉所有偶数长度的回文串，
    必须对每个位置额外再做一次"以该字符和右边字符之间的空隙为中心"的扩展；2) 扩展
    循环的边界条件 `l >= 0 and r < n` 必须写全，少写一个会在字符串两端出现越界
    访问。
    """
    n = len(s)
    count = 0

    def expand(l: int, r: int) -> None:
        nonlocal count
        while l >= 0 and r < n and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1

    for center in range(n):
        expand(center, center)
        expand(center, center + 1)
    return count


def longest_dup_substring(s: str) -> str:
    """
    【题意】给定字符串 s，找出其中最长的、至少出现两次（可以重叠）的子串；如果不
    存在这样的重复子串，返回空字符串。
    【思路】这题的两个维度分别用两种技巧解决：1) "最长"这个要求具有单调性——如果
    长度为 L 的重复子串存在，不代表长度 L-1 的重复子串一定不存在（截断后依然重复），
    所以严格来说不是"长度越大越难满足"的单调性，但可以证明"是否存在长度为 L 的重复
    子串"这个判定，天然支持二分查找长度（如果长度 L 存在重复，L 更小的窗口大概率也
    能找到重复，二分查找答案是本题标准做法）；2) 对固定长度 L，怎么快速判断"是否
    存在两个起点不同、内容相同的长度 L 子串"——如果每次都直接比较子串内容是 O(L)，
    结合外层滑动窗口整体是 O(nL)，用 Rabin-Karp 滚动哈希可以把"判断某个长度 L 的
    子串是否出现过"降到均摊 O(1)：把每个长度 L 的窗口看成一个 base 进制数取模的
    哈希值，窗口每滑动一位，新哈希值可以用旧哈希值通过"减去滑出的字符、乘 base、
    加上滑入的字符"的公式 O(1) 递推出来，不需要重新计算整个窗口。用一个 hash ->
    [起点列表] 的字典记录见过的哈希值；由于不同子串可能哈希碰撞（尤其是模数不够大
    或不是质数时），查到相同哈希后必须再真正比较一次子串内容确认，才能返回。
    【复杂度】时间 O(n log n)（二分 O(log n) 轮，每轮滚动哈希扫描 O(n)）；空间
    O(n)（每轮的哈希表）。
    【易错点】1) 取模用的 mod 必须是一个足够大的质数（这里用 2^61-1，已知的
    Mersenne 质数），如果 mod 选得太小或者不是质数，会有大量哈希碰撞，需要退化成
    到处做真实字符串比较，失去滚动哈希本该有的效率优势；2) 哈希相同不代表子串
    真的相同（碰撞的可能性始终存在），发现哈希匹配后必须用真正的字符串切片比较
    做二次确认，否则会返回错误答案；3) 滚动更新哈希的公式里，"减去滑出字符"这一项
    要乘以 `base^(L-1)`（滑出的字符在窗口最高位，对应的权重是 `base` 的 L-1 次方），
    容易写成 `base^L` 导致递推公式整体错位。
    """
    n = len(s)
    nums = [ord(ch) - ord("a") for ch in s]
    base = 31
    mod = (1 << 61) - 1  # 2^61-1 是已知的 Mersenne 质数，冲突概率足够低

    def search(length: int) -> int:
        if length == 0:
            return 0
        h = 0
        for i in range(length):
            h = (h * base + nums[i]) % mod
        seen: dict[int, list[int]] = {h: [0]}
        power = pow(base, length - 1, mod)
        for start in range(1, n - length + 1):
            h = ((h - nums[start - 1] * power) * base + nums[start + length - 1]) % mod
            if h in seen:
                candidate = s[start : start + length]
                if any(
                    s[prev : prev + length] == candidate for prev in seen[h]
                ):
                    return start
                seen[h].append(start)
            else:
                seen[h] = [start]
        return -1

    lo, hi = 1, n - 1
    best_start, best_len = -1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        start = search(mid)
        if start != -1:
            best_start, best_len = start, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return s[best_start : best_start + best_len] if best_start != -1 else ""


def _self_test() -> None:
    assert str_str_kmp("sadbutsad", "sad") == 0
    assert str_str_kmp("leetcode", "leeto") == -1

    assert repeated_substring_pattern("abab") is True
    assert repeated_substring_pattern("aba") is False
    assert repeated_substring_pattern("abcabcabcabc") is True

    assert count_substrings("abc") == 3
    assert count_substrings("aaa") == 6

    assert longest_dup_substring("banana") == "ana"
    assert longest_dup_substring("abcd") == ""

    print(
        "[PASS] p24_advanced_strings: 4/4 题通过 "
        "(KMP找子串下标/重复的子字符串模式/回文子串/最长重复子串)"
    )


if __name__ == "__main__":
    _self_test()
