"""分类 24（Phase 3 竞赛级补充）：高级字符串算法进阶 —— 把 KMP 家族的几种变体
（lps 数组用在"回文性"上、lps 数组用在"周期性/自身对称性"上）和 Z 函数放在一起
对比着学，再补一道 Rabin-Karp 风格的双指针去重题。"""
from __future__ import annotations


def _build_lps(pattern: str) -> list[int]:
    """KMP 预处理：lps[i] 表示 pattern[0..i] 这一段里，"最长的、既是真前缀又是真
    后缀"的长度。shortest_palindrome 和 longest_happy_prefix 共用这个工具函数，但
    应用的对象和挖掘出的含义完全不同（详见各自的【思路】）。"""
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


def shortest_palindrome(s: str) -> str:
    """
    【题意】给定字符串 s，只能在字符串**前面**添加字符，把它转换成回文串。求这样
    能得到的最短回文串。
    【思路】设最终答案是在 s 前面拼接了一段字符 `prefix` 之后得到的（`prefix + s`
    是回文串），要让拼接的部分最短，等价于要让 s 自身"从下标 0 开始的最长回文前缀"
    尽量长——如果这个最长回文前缀的长度是 L，那么剩下的那一段 `s[L:]` 反转之后
    正好就是需要拼在前面的 `prefix`（回文串左右对称，s 前面缺的那部分，正好是
    s 后面多出来那部分的镜像）。问题转化成：怎么求"s 从下标 0 开始的最长回文
    前缀长度"？做法是构造一个新串 `s + '#' + reverse(s)`（用一个 s 中不会出现的
    分隔符 `#` 隔开两段，防止后面求 lps 时"越过中点"把整个新串错误地匹配到自己
    身上），对这个新串求 KMP 的 lps 数组，取最后一个值 `lps[-1]`——这个值就是
    "s 的前缀"和"reverse(s) 的后缀"能重合的最大长度。由于 `reverse(s)` 的后缀
    等于 s 的前缀反转后的结果，"这一段能重合"意味着"s 这一段前缀正读、反读相同"，
    也就是回文，所以 `lps[-1]` 恰好就是 s 从头开始的最长回文前缀长度 L。求出 L
    之后，把 `s[L:]` 反转，拼接到 s 前面即可。
    【复杂度】时间 O(n)（构造新串 O(n)，求 lps 数组 O(n)）；空间 O(n)（新串本身
    和 lps 数组都是 O(n) 大小）。
    【易错点】1) 必须在 s 和 reverse(s) 之间插入一个分隔符（且这个分隔符不能是
    s 里会出现的字符），否则如果 s 中有一部分和它的反转发生交叉重叠，lps 数组
    可能会算出一个"跨过中点"的错误长度，误把不属于同一段的字符匹配到一起；2) 最终
    答案是把 `s[L:]` **反转后**拼到前面，而不是把 `s[L:]` 原样拼接——原样拼接
    得到的字符串不是回文串；3) 容易把"最长相同前后缀"和"最长回文前缀"这两个
    概念直接划等号，误以为对 s 自身求一次 lps 就能拿到答案——这个转换只有在通过
    `s + '#' + reverse(s)` 这个特定构造之后才成立，对 s 自身求 lps 得到的是完全
    不同的信息（s 自身的周期性，见 longest_happy_prefix 的做法）。
    """
    if not s:
        return s
    rev = s[::-1]
    combined = s + "#" + rev
    lps = _build_lps(combined)
    overlap = lps[-1]
    return rev[: len(s) - overlap] + s


def longest_happy_prefix(s: str) -> str:
    """
    【题意】定义字符串的"快乐前缀"是既是它的非空真前缀、又是它的非空真后缀的
    子串（不能等于原字符串自身）。给定字符串 s，返回它最长的快乐前缀；不存在则
    返回空字符串。
    【思路】这是 KMP 的 lps 数组最直接、最不需要拐弯的应用——lps 数组的定义本身
    就是"pattern[0..i] 这一段里，最长的、既是真前缀又是真后缀的长度"，对整个字符
    串 s 自身求 lps 数组，最后一个值 `lps[-1]`（对应 `pattern[0..n-1]`，也就是整个
    s）刚好就是"整个 s 里最长的、既是真前缀又是真后缀的长度"，这正是"最长快乐
    前缀"的定义。和 shortest_palindrome 不同，这里不需要在 s 和任何其他串之间拼接
    分隔符——shortest_palindrome 是在 s 和 reverse(s) **两个不同的字符串**之间求
    最长公共前后缀，需要分隔符防止越界比较；这里是同一个字符串对自己求 lps，"真
    前缀/真后缀不能等于整个串"这个约束已经由 lps 的定义本身保证，不会出现越界
    问题。求出 `L = lps[-1]` 之后，直接返回 `s[:L]`。
    【复杂度】时间 O(n)（构造 lps 数组）；空间 O(n)（lps 数组）。
    【易错点】1) 容易和 shortest_palindrome 的做法混淆，误以为这里也需要构造
    `s + '#' + reverse(s)` 这样的拼接串——完全不需要，直接对 s 自身求 lps 即可，
    多此一举地拼接反而会引入无意义的复杂度，而且拼接之后 `lps[-1]` 的含义会
    变成完全不同的东西（s 和 reverse(s) 的公共前后缀，而不是 s 自身的相同前后缀）；
    2) `lps[-1]` 得到的是一个长度，返回值应该是切片 `s[:L]`，不能直接返回这个
    长度数字；3) 当 `lps[-1] == 0`（s 不存在任何相同前后缀）时，`s[:0]` 天然就是
    空字符串，不需要额外写一个"不存在就返回空串"的特判分支。
    """
    if not s:
        return ""
    lps = _build_lps(s)
    return s[: lps[-1]]


def distinct_echo_substrings(text: str) -> int:
    """
    【题意】给定字符串 text，返回其中能写成"某个字符串与自身拼接"（形如 `a + a`，
    a 是非空字符串）的、不同的非空子串的数目——同一段内容在不同位置重复出现只算
    一种。
    【思路】能写成 `a+a` 形式的子串，长度必然是偶数；设这个子串的一半长度是 L，
    它就等价于"从某个起点开始，长度为 L 的一段，和紧跟在它后面、同样长度为 L 的
    一段完全相同"。枚举 L（从 1 到 n//2），对每个固定的 L 用一对相隔 L 的下标
    `i` 和 `j = i + L` 一起向右滑动，维护一个计数器 `matched` 表示"当前已经连续
    验证了多少对 `text[i] == text[i+L]`"：每次匹配就 `matched += 1`，不匹配就清零；
    一旦 `matched == L`，说明找到了一个完整的、长度为 2L 的回声子串（这个子串的
    起点是 `i - L + 1`，因为发现匹配时下标 i 已经往前走了 L-1 步），把它加入一个
    哈希集合去重，然后把 `matched` 减一（而不是直接清零）——这是为了让窗口继续
    往右滑动一格时，依然能利用"除了刚滑出窗口的那个字符外，其余部分仍然匹配"这个
    信息，检测出相互重叠的下一个回声子串，如果直接清零会漏掉这种重叠的情况。数据
    范围较小（`n <= 2000`）时直接用字符串切片存入集合即可通过；如果要把复杂度进一
    步压到规避 O(L) 的切片开销，可以像 longest_dup_substring 那样引入 Rabin-Karp
    滚动哈希做去重判断（哈希相同后仍需切片二次确认），思路是同一套工具。
    【复杂度】时间 O(n^2)（枚举 L 是 O(n)，每个 L 下双指针扫描是 O(n)，加上字符串
    切片和哈希集合操作，在 n<=2000 的范围内可以接受）；空间 O(n^2)（最坏情况下
    哈希集合里存储的所有子串长度总和）。
    【易错点】1) 一旦 `matched == L` 记录了一个回声子串之后，`matched` 应该减一
    而不是直接清零——否则会漏掉紧接着重叠的下一个回声子串（比如 "aaaa" 这种高度
    重复的输入，多个回声子串首尾相接）；2) 回声子串在原串中的实际起点是
    `i - L + 1`，不是当前的 `i`——因为发现 `matched == L` 时，`i` 已经是"最后一次
    成功匹配"的下标，起点要往回退 `L - 1` 位；3) 只需要枚举到 `L <= n // 2`，因为
    `a+a` 的总长度是 `2L`，必须不超过整个字符串长度 n，枚举更大的 L 既没有意义也
    会导致下标越界。
    """
    n = len(text)
    seen: set[str] = set()
    for length in range(1, n // 2 + 1):
        matched = 0
        for i in range(n - length):
            j = i + length
            if text[i] == text[j]:
                matched += 1
            else:
                matched = 0
            if matched == length:
                start = i - length + 1
                seen.add(text[start : start + 2 * length])
                matched -= 1
    return len(seen)


def _z_function(s: str) -> list[int]:
    """Z 函数（Z-array）：z[i] 表示 s 从下标 i 开始的后缀，与 s 本身（从下标 0
    开始的前缀）的最长公共前缀长度；按惯例 z[0] 不参与这个定义（自己和自己比较
    没有意义）。维护一个"已知的、最靠右的匹配窗口" [l, r]（表示 s[l..r-1] 已经
    确认和 s 的某个前缀相同）：处理下标 i 时，如果 i 落在窗口内，可以先用
    z[i-l] 作为一个下界估计（因为这一段之前已经间接验证过和某个前缀相同），
    但仍然需要在这个基础上继续暴力扩展验证，不能直接假设 z[i] 就等于 z[i-l]；
    如果 i 落在窗口外，只能从 0 开始暴力扩展。这个"复用已知窗口、避免把验证过的
    信息扔掉重新扫一遍"的思路，和 KMP 的 lps 数组是同源的 insight，只是 Z 函数
    直接给出"每个后缀和整个串的最长公共前缀"，语义上比 lps 数组更直接契合
    sum_of_scores 这道题。"""
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z


def sum_of_scores(s: str) -> int:
    """
    【题意】给定字符串 s（长度为 n），把 s 的长度为 i 的后缀记作 `s_i`
    （`s_i = s[n-i:]`，特别地 `s_n = s` 本身）。定义 `s_i` 的 score 是 `s_i` 和
    `s` 的最长公共前缀长度。求所有 `i` 从 1 到 n 的 score 之和。
    【思路】这是 Z 函数教科书式的直接应用：Z 函数定义 `z[k]` 表示 s 从下标 k
    开始的后缀，与整个 s 的最长公共前缀长度。本题里 `s_i`（长度为 i 的后缀）
    恰好就是"从下标 `n-i` 开始的后缀"，所以 `s_i` 的 score 正好等于
    `z[n-i]`；而 `s_n = s` 自己和自己的最长公共前缀显然是整个长度 n（这一项
    对应 `i = n`，即下标 0 处，但 Z 函数按惯例不在下标 0 处计算——需要单独把
    这一项加上，而不是套用 `z[0]`）。求出整个 Z 数组之后，把 `z[1]` 到
    `z[n-1]` 全部加起来（这部分恰好覆盖了 `i` 从 1 到 `n-1` 的所有 score，
    只是加总的顺序和 `i` 的对应关系被打乱了，但求和是可交换的，不影响结果），
    再加上 `n`（`i=n` 这一项），就是最终答案。
    【复杂度】时间 O(n)（Z 函数是均摊线性的经典结论——窗口右边界 `r` 全程只会
    单调右移，不会回退，因此内层 while 循环总共执行的次数不会超过 O(n)）；
    空间 O(n)（Z 数组）。
    【易错点】1) `z[0]` 按惯例不参与"和整个串的最长公共前缀"这个定义，必须在
    主循环里从下标 1 开始；本题里"整个字符串自己"这一项（对应 `i=n`）要单独
    按 `n` 计入总和，不能不小心地把 `z[0]`（未定义或恒为 0）当成这一项使用，
    否则会漏掉这个最大的一项；2) 窗口 `[l, r]` 内"可以复用 `z[i-l]`"这一步只
    给出一个下界（`min(r-i, z[i-l])`），后面仍然必须继续暴力扩展验证，不能
    直接假设 `z[i] = z[i-l]` 就完事，因为 `z[i-l]` 可能会超出窗口 `[l,r]`
    剩余的长度；3) 窗口 `[l, r]` 只应该在"这一轮实际扩展出的匹配右边界超过
    当前 r"时才更新，如果每轮都无条件更新窗口，会破坏"窗口是已知最靠右匹配
    区间"这个不变量，导致后续下标错误地复用了不该复用的信息。
    """
    n = len(s)
    z = _z_function(s)
    return n + sum(z[1:])


def _self_test() -> None:
    assert shortest_palindrome("aacecaaa") == "aaacecaaa"
    assert shortest_palindrome("abcd") == "dcbabcd"
    assert shortest_palindrome("") == ""

    assert longest_happy_prefix("level") == "l"
    assert longest_happy_prefix("ababab") == "abab"
    assert longest_happy_prefix("leetcodeleet") == "leet"
    assert longest_happy_prefix("a") == ""

    assert distinct_echo_substrings("abcabcabc") == 3
    assert distinct_echo_substrings("leetcodeleetcode") == 2

    assert sum_of_scores("babab") == 9
    assert sum_of_scores("azbazbzaz") == 14

    print(
        "[PASS] p24_advanced_strings_iii: 4/4 题通过 "
        "(最短回文串/最长快乐前缀/不同的循环子字符串数量/字符串的Score和)"
    )


if __name__ == "__main__":
    _self_test()
