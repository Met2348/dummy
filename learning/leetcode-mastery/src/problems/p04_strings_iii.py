"""LeetCode 分类 04·字符串 Phase 3 竞赛级补充（Part III）：LCS 变体（两个字符串的删除
操作）+ 取模避免整数溢出的自我拼接匹配（重复叠加字符串匹配）+ 正则清洗与词频统计
（最常见的单词）+ 自描述序列的增量构造（神奇字符串）+ 行程编码分组比较（情感丰富的
文字）+ 后缀和取模的批量偏移（字母移位）+ 双数组交替合并（重新格式化字符串），共 7
道范例，比 Part II 更强调"边界条件的完整性"和"避免朴素解法带来的复杂度或数值风险"。"""
from __future__ import annotations

import re
from collections import Counter


def min_distance_delete(word1: str, word2: str) -> int:
    """
    【题意】给定两个小写字母字符串 word1、word2，每一步可以删除其中一个字符串里的
    任意一个字符，求使 word1 和 word2 变得完全相同所需要的最少删除步数。
    【思路】"删到相同"等价于"保留一个公共子序列，其余字符全部删掉"，而要让删除次数
    最少，保留下来的这个公共子序列必须尽量长——这正是最长公共子序列（LCS，LC1143）
    的定义。所以先用标准的二维 DP 求出 word1、word2 的 LCS 长度：`dp[i][j]` 表示
    `word1[:i]` 和 `word2[:j]` 的 LCS 长度，字符相同就在对角线基础上 +1，不同就取
    "丢弃 word1 当前字符" 和 "丢弃 word2 当前字符" 两种选择中的较大者。求出 LCS 长度
    后，word1 需要删掉 `len(word1) - lcs` 个字符，word2 需要删掉 `len(word2) - lcs`
    个字符，两者相加就是总删除步数：`len(word1) + len(word2) - 2*lcs`。
    【复杂度】时间 O(m·n)，空间 O(m·n)（可以用滚动数组把空间降到 O(min(m,n))，这里为了
    突出"和 LCS 是同一个 DP 表"这一点，保留完整二维表）。
    【易错点】
    1) 不能直接套用"编辑距离"（LC72）的 DP 转移——编辑距离允许替换操作（一步把一个
    字符改成另一个），而本题只允许删除，如果错误地引入替换转移，会把答案算小；
    2) 最终答案不是 LCS 长度本身，而是 `m + n - 2*lcs`——这一步转换很容易漏掉，只
    求出 LCS 长度就直接返回，会漏掉"两边各自还要删掉多少"这一步换算；
    3) `dp` 数组下标 i、j 分别对应 `word1[:i]`、`word2[:j]`（前 i、j 个字符构成的
    前缀），比较字符时要用 `word1[i-1]`、`word2[j-1]`（下标错一位是 LCS 类 DP 最
    常见的越界/偏移 bug）。
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return m + n - 2 * lcs


def repeated_string_match(a: str, b: str) -> int:
    """
    【题意】给定两个字符串 a、b，求最少把 a 重复拼接多少次，才能使 b 成为拼接结果的
    子串；如果无论重复多少次都不可能，返回 -1（重复 0 次视为空字符串）。
    【思路】重复次数不是漫无目的地一次次尝试再检查——可以先确定一个"下界"：拼接结果
    的长度至少要达到 `len(b)`，才有可能包含 b，这个下界就是 `ceil(len(b)/len(a))`
    次（记为 `count`）。但只拼到刚好覆盖长度还不够：b 有可能"骑跨"在这个拼接结果的
    末尾，需要再多拼一次 a 才能让骑跨的部分完整出现。所以只需要按顺序尝试 `count` 次
    和 `count + 1` 次这两种拼接结果，用朴素的子串检查（Python 的 `in`）分别判断 b
    是否是子串，只要有一种命中就返回对应的重复次数，两种都不命中则说明无论怎么拼都
    不可能，返回 -1（可以证明这两次尝试已经足够，不需要尝试第三次）。
    【复杂度】时间 O((m+n)·n)，m=len(a)、n=len(b)（拼接后字符串长度是 O(m+n)，`in`
    做一次子串查找最坏是 O((m+n)·n)，这里不追求引入 KMP 的严格线性解法，两次朴素
    子串检查已经足够通过本题给定的数据规模）；空间 O(m+n)（存储拼接结果）。
    【易错点】
    1) 下界必须用 `ceil(len(b)/len(a))` 而不是 `len(b)//len(a)`（向下取整会少拼
    一次，导致拼接结果长度不够，必然无法包含和它一样长甚至更长的 b）；
    2) 只判断"拼到覆盖长度那一次"是不够的，必须再多试一次（`count+1`），因为 b 完全
    可能横跨在这次拼接结果的最后一小段和补上的下一份 a 的开头之间；
    3) 两次都不命中时要返回 -1，不能误以为"再多拼几次总会撞上"——如果 a、b 压根不
    共享字符集（比如 a 只含 'a'，b 里有 'w'），拼多少次都不可能包含 b。
    """
    m, n = len(a), len(b)
    count = -(-n // m)  # 等价于 math.ceil(n / m)，避免额外 import
    candidate = a * count
    if b in candidate:
        return count
    candidate += a
    if b in candidate:
        return count + 1
    return -1


def most_common_word(paragraph: str, banned: list[str]) -> str:
    """
    【题意】给定一段文本 paragraph（可能夹杂大小写、逗号、句号等标点）和一个禁用词
    列表 banned（全小写、不含标点），返回文本中出现频率最高、且不在禁用列表里的单词
    （题目保证答案唯一）。返回结果需要是小写。
    【思路】核心是"先把脏文本清洗成干净的单词列表，再做词频统计"这两步分开处理，不要
    试图一边扫描字符一边同时做大小写归一化、标点剔除、分词三件事。用正则
    `re.findall(r"[a-z]+", paragraph.lower())` 一步到位：先把整段文本转成小写（大小
    写不敏感），再按"连续的小写字母"切出所有单词——标点符号、数字、空格都不匹配这个
    正则，天然被当作单词之间的分隔符丢弃掉，不需要手写"是不是字母"的逐字符判断。清洗
    完之后，用 `Counter` 对"不在 banned 集合里的单词"计数，取出现次数最多的那个。
    【复杂度】时间 O(L)，L 为 paragraph 的字符总数（正则扫描一遍、计数一遍都是线性）；
    空间 O(W)，W 为不同单词的个数。
    【易错点】
    1) `banned` 必须转成 `set` 再做成员判断，如果直接对 list 用 `in`，禁用词很多时
    会退化成 O(n·k) 的重复线性查找；
    2) 正则必须写成 `[a-z]+`（小写），并且要先对整个 paragraph 调用 `.lower()`，
    不能指望正则自己做大小写不敏感匹配——如果直接对原始大小写混合文本跑 `[a-z]+`，
    像 "BALL" 这种全大写单词会被完全匹配不到（大写字母不在 `[a-z]` 范围内）；
    3) 不能用 `paragraph.split()` 之类的简单分词，标点符号会残留在单词末尾（比如
    "ball," 会被当成和 "ball" 不同的两个词），导致同一个单词被错误地拆分统计。
    """
    banned_set = set(banned)
    words = re.findall(r"[a-z]+", paragraph.lower())
    counts = Counter(w for w in words if w not in banned_set)
    return counts.most_common(1)[0][0]


def magical_string(n: int) -> int:
    """
    【题意】神奇字符串 S 只由字符 '1'、'2' 组成，且满足一个自描述性质：把 S 中连续
    相同字符的"游程长度"依次记录下来，得到的序列恰好就是 S 自身（S 以 "1221121221
    221121122……" 开头）。给定整数 n，返回 S 的前 n 个字符中 '1' 的个数。
    【思路】这是"用已经生成的部分，指导生成接下来的部分"的自描述序列构造问题——不能
    先假设整个序列已知再去分析它，而要模拟它"边生成边读取自己"的过程。已知 S 以
    `[1, 2, 2]` 开头。维护一个指针 `i`（从下标 2 开始，指向"当前正在被读取、用来决定
    接下来生成几个字符"的那个位置），以及序列本身 `s`（同时既是"正在被构造的结果"，
    也是"正在被读取的游程长度表"）：每一轮取 `s[-1]`（上一个生成的字符）算出下一个
    要生成的字符 `cur`（`1` 和 `2` 交替，用 `3 - pre` 一步算出对方），再看 `s[i]`
    这个"游程长度"决定 `cur` 要连续写几次，写完后 `i` 前移一位，读取下一个游程长度。
    重复直到写够 n 个字符为止，最后数一下前 n 个字符里 '1' 的个数。
    【复杂度】时间 O(n)，空间 O(n)（存储整个序列；可以用双指针只保留必要状态做到更省
    空间，这里为了让"序列同时是自己的游程长度表"这一自描述性质更直观，直接保留完整
    列表）。
    【易错点】
    1) 序列必须以 `[1, 2, 2]` 这三个字符作为固定起点，如果只从 `[1]` 开始重新推导，
    读取指针 `i` 和写入位置会对不上（这三个字符是打破"鸡生蛋"循环依赖的初始条件，
    需要记住而不是临时推导）；
    2) 字符交替用 `3 - pre` 而不是写 `if/else` 分支——因为字符只可能是 1 或 2，
    `3 - 1 = 2`、`3 - 2 = 1` 天然实现"取另一个值"，比写两条分支更不容易在扩展到
    "1/2 以外的字符集"时出错（虽然本题字符集固定，这个技巧仍值得记住）；
    3) 读取指针 `i` 每次只前移一位（读完这一个游程长度就换下一个），不能因为一次写入
    了多个字符就误以为 `i` 也要跟着多移几位——`i` 索引的是"游程长度"这个序列本身的
    下标，和"实际写了多少个字符"是两回事。
    """
    if n <= 0:
        return 0
    s = [1, 2, 2]
    i = 2
    while len(s) < n:
        pre = s[-1]
        cur = 3 - pre
        s += [cur] * s[i]
        i += 1
    return s[:n].count(1)


def expressive_words(s: str, words: list[str]) -> int:
    """
    【题意】人们打字时会拉长某些字母表达情绪（比如 "hello" 拉长成
    "heeellooo"）。规则是：每次"拉伸"操作，只能把一组连续相同的字母，从当前长度扩展
    到 3 个或以上（不能把只有 1 个的字母组扩展成 2 个，扩展后必须至少有 3 个）。给定
    拉长后的字符串 s 和候选单词列表 words，统计有多少个候选单词可以经过零次或多次
    这样的拉伸操作，变成 s。
    【思路】先把"逐字符比较"转化成"逐组比较"——用行程编码（RLE）把 s 和每个候选词都
    拆成 `[(字符, 连续出现次数), ...]` 的分组列表（比如 "heeellooo" 拆成
    `[('h',1),('e',3),('l',2),('o',3)]`）。两个字符串能通过拉伸互相转换，当且仅当
    它们的分组数量相同、每组字符相同，且每组的次数满足："word 的这一组次数
    `c2` 不能超过 s 对应这一组的次数 `c1`（拉伸只能增多，不能减少）；如果 `c2` 严格
    小于 `c1`，那么 `c1` 必须达到 3 或以上（因为拉伸操作本身要求扩展到至少 3 个，
    如果 s 里这一组连 3 个都不到，说明它根本没有被拉伸过，此时 `c2` 必须和 `c1`
    严格相等）"。
    【复杂度】时间 O(Σ|words[i]| + |s|)，对每个候选词分组和比较都是线性；空间
    O(|s| + max|words[i]|)（分组结果的存储）。
    【易错点】
    1) 分组数量不同（比如字母种类或出现顺序不一致）必须直接判定不匹配，不能只比较
    "字母集合"——分组的顺序本身就编码了字母在字符串里的先后位置信息；
    2) "次数相等"这种情况必须始终允许通过，不能因为 `c1 < 3` 就一律拒绝——只有
    "次数不相等且 `c1 < 3`" 才应该被拒绝，`c1 < 3` 但 `c2 == c1`（两边压根没拉伸）
    是完全合法的；
    3) 判断顺序上，如果 `c2 > c1`（候选词这一组反而比 s 更长），必须直接判不匹配，
    因为拉伸操作只能让 s 里的组变得更长，候选词的组次数不可能超过 s 对应组的次数。
    """

    def rle(t: str) -> list[tuple[str, int]]:
        groups: list[tuple[str, int]] = []
        i, n = 0, len(t)
        while i < n:
            j = i
            while j < n and t[j] == t[i]:
                j += 1
            groups.append((t[i], j - i))
            i = j
        return groups

    s_groups = rle(s)
    matched = 0
    for w in words:
        w_groups = rle(w)
        if len(w_groups) != len(s_groups):
            continue
        ok = True
        for (c1, n1), (c2, n2) in zip(s_groups, w_groups):
            if c1 != c2 or n2 > n1:
                ok = False
                break
            if n2 < n1 and n1 < 3:
                ok = False
                break
        if ok:
            matched += 1
    return matched


def shifting_letters(s: str, shifts: list[int]) -> str:
    """
    【题意】给定小写字母字符串 s 和长度相同的整数数组 shifts。`shifts[i] = x`
    表示"把 s 的前 i+1 个字符，每个都循环移位 x 次"（'z' 移位一次变回 'a'）。所有
    `shifts` 里的操作要依次全部施加到 s 上，返回最终结果。
    【思路】如果真的按 `shifts` 的顺序，对"前 i+1 个字符"逐次整体移位，会有大量重复
    劳动（下标 0 的字符几乎每次操作都会被移位一次）。换个角度看："下标 i 的字符总共
    被移位了多少次"，其实就是所有满足 `j >= i` 的 `shifts[j]` 加起来的总和——这正是
    一个**后缀和**。所以只需要从右往左扫一遍 s，同时维护一个累加的后缀和 `total`
    （每往左走一位，把当前的 `shifts[i]` 累加进去），当前字符最终的移位量就是这个
    `total`，直接对 26 取模、循环移位即可得到最终字符，不需要真的对每个前缀做一次
    整体移位。
    【复杂度】时间 O(n)，一次从右往左的线性扫描；空间 O(n)（结果字符列表，如果算上
    输入字符串本身转 list 的开销）。
    【易错点】
    1) 累加后缀和必须从右往左扫描（`total` 在每一步累加 `shifts[i]` 而不是提前算好
    整个数组的总和再逐步减去），如果从左往右扫描且每步直接加总和会导致每个位置都
    被加上"包括它自己在内的全部后续位移"这个理解出现偏差，容易搞反累加方向；
    2) `total` 必须每一步都对 26 取模（而不是等到最后才取模）——题目里 `shifts[i]`
    最大可以到 1e9，量级很大的多个数字连续相加，虽然 Python 整数不会溢出，但及时
    取模是一个"迁移到会溢出的语言（如 32 位整型的 C++/Java）也同样正确"的好习惯；
    3) 移位公式是 `(原字符的字母序号 + total) % 26`，不能忘记先减掉 `ord('a')`
    把字符换算成 0-25 的字母序号，再加移位量、取模，最后加回 `ord('a')` 换算回
    字符——直接对 ASCII 码取模会得到完全无意义的字符。
    """
    n = len(s)
    result = list(s)
    total = 0
    for i in range(n - 1, -1, -1):
        total = (total + shifts[i]) % 26
        idx = (ord(result[i]) - ord("a") + total) % 26
        result[i] = chr(ord("a") + idx)
    return "".join(result)


def reformat(s: str) -> str:
    """
    【题意】给定一个由小写字母和数字混合组成的字符串 s，重新排列它，使得任意相邻两个
    字符的"类型"（字母/数字）都不同——即字母和数字必须交替出现。如果无法做到，返回
    空字符串；能做到的话，返回任意一个满足条件的合法排列。
    【思路】先把字母和数字分别收集成两个列表。能交替排列的充要条件是：两个列表的长度
    差不超过 1——如果某一类字符数量比另一类多 2 个或以上，无论怎么摆放，多出来的那
    一类里必然有两个同类字符挨在一起（抽屉原理）。满足条件后，构造时让"数量较多（或
    相等）的那一类"打头：从较长的列表和较短的列表里各取一个字符，依次交替拼接，较短
    列表耗尽后，如果较长列表还剩最后一个字符（只有长度差恰好为 1 时才会发生），把它
    补在结果末尾。
    【复杂度】时间 O(n)，空间 O(n)（拆分出的两个列表 + 结果列表）。
    【易错点】
    1) 判断"能否重排"的条件是两个列表长度之差的绝对值 `> 1` 就无解，不能只检查
    "较多的一类是否严格多 2 个"这一个方向，两类互换后的情况也要覆盖（用 `abs` 一次
    性覆盖两个方向）；
    2) 交替拼接必须从"数量较多（或相等）的那一类"开始，如果两类数量不相等却从较少的
    那一类开始拼接，交替到最后较少的那一类会先耗尽，而较多的那一类还剩字符没地方放，
    会漏掉合法的排列结尾；
    3) 这道题允许多个合法答案，测试时不应该假设"结果必须和某一个特定字符串逐字符
    相等"（除非能证明这道题在给定输入下答案唯一），本课的自测里选用的样例经过验证后
    确实和"先数字类/字母类哪边更长就从哪边开始交替"这个构造方式产出的结果一致，但
    这只是这几个样例的巧合，不是这道题本身的性质。
    """
    digits = [c for c in s if c.isdigit()]
    letters = [c for c in s if c.isalpha()]
    if abs(len(digits) - len(letters)) > 1:
        return ""

    long_group, short_group = (
        (digits, letters) if len(digits) >= len(letters) else (letters, digits)
    )
    result: list[str] = []
    for i in range(len(short_group)):
        result.append(long_group[i])
        result.append(short_group[i])
    if len(long_group) > len(short_group):
        result.append(long_group[-1])
    return "".join(result)


def _self_test() -> None:
    assert min_distance_delete("sea", "eat") == 2
    assert min_distance_delete("leetcode", "etco") == 4

    assert repeated_string_match("abcd", "cdabcdab") == 3
    assert repeated_string_match("a", "aa") == 2
    assert repeated_string_match("a", "a") == 1
    assert repeated_string_match("abc", "wxyz") == -1

    assert (
        most_common_word(
            "Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"]
        )
        == "ball"
    )

    assert magical_string(6) == 3
    assert magical_string(1) == 1
    assert magical_string(4) == 2

    assert expressive_words("heeellooo", ["hello", "hi", "helo"]) == 1
    assert expressive_words("zzzzzyyyyy", ["zzyy", "zy", "zyy"]) == 3

    assert shifting_letters("abc", [3, 5, 9]) == "rpl"

    assert reformat("a0b1c2") == "0a1b2c"
    assert reformat("leetcode") == ""
    assert reformat("1229857369") == ""
    assert reformat("covid2019") == "c2o0v1i9d"
    assert reformat("ab123") == "1a2b3"

    print(
        "[PASS] p04_strings_iii: 7/7 题通过 "
        "(两个字符串的删除操作/重复叠加字符串匹配/最常见的单词/神奇字符串/"
        "情感丰富的文字/字母移位/重新格式化字符串)"
    )


if __name__ == "__main__":
    _self_test()
