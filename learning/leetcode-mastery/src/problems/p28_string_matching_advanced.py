"""分类 28（Phase 3 竞赛级新分类）：字符串匹配进阶 —— Z 函数、Aho-Corasick 自动机、
后缀数组、最长公共子串。24 类的 KMP/中心扩展/Rabin-Karp 解决的是"单模式串在单文本
里匹配"；这一类把武器库扩展到"多模式串同时匹配"（Aho-Corasick）和"所有后缀相关的
结构化问题"（后缀数组、最长公共子串），并且用 Z 函数展示"同一类前后缀匹配问题"的
另一种表述方式。"""
from __future__ import annotations

from collections import deque


def rotate_string(s: str, goal: str) -> bool:
    """
    【题意】LeetCode 796·Easy。给定字符串 s 和 goal，判断 s 经过若干次"循环左移"
    （把最左边的字符搬到最右边）之后，能否变成 goal。
    【思路】s 的所有循环移位结果，都必然是 `s + s`（s 自我拼接一次）的某个长度为
    len(s) 的连续子串——因为循环移位相当于"把 s 的某个后缀接到前缀前面"，而
    `s+s` 恰好把 s 的所有"后缀+前缀"组合都平铺在了一条更长的字符串里。因此只需
    要判断 goal 是否是 `s+s` 的子串，再额外检查长度是否相等（否则 s="a"、
    goal="aa" 这种长度不同但 goal 恰好是 s+s 子串的错误情况会被误判为 True）。
    这道题本身可以用内置的 `in` 运算符一行写完，但更能体现"进阶"的地方是把它接到
    KMP：用 `str_str_kmp`（或本文件里的 Z 函数版本）在 `s+s` 里查找 goal，
    同样可以做到线性时间，且避免依赖 Python 字符串 `in` 背后隐藏的实现细节。
    【复杂度】时间 O(n)（构造 s+s 是 O(n)，在其中查找长度为 n 的子串在 KMP/Z 函数
    下也是线性）；空间 O(n)（s+s 这个新字符串）。
    【易错点】1) 忘记先比较 `len(s) == len(goal)`，导致 s 是 goal 的"真前缀"这种
    长度不等的情况被误判；2) 空字符串边界：s 和 goal 都为空时应视为可以匹配
    （题目约束保证长度 >= 1，但独立实现时要留意）。
    """
    if len(s) != len(goal):
        return False
    if not s:
        return True
    return goal in (s + s)


def max_repeating_substring(sequence: str, word: str) -> int:
    """
    【题意】LeetCode 1668·Easy。给定字符串 sequence 和 word，求最大的 k，使得
    word 重复 k 次拼接后的结果（word*k）是 sequence 的子串；若 word 本身都不是
    sequence 的子串，返回 0。
    【思路】最直接的写法是从 k=1 开始不断尝试 `word * k in sequence`，一旦某个
    k 失败就停止，返回 k-1——因为如果 word 重复 k 次都无法作为子串出现，重复
    更多次也不可能出现（一旦断开就不会重新连上），这个判定天然具有单调性，不需要
    额外的复杂算法，暴力递增在 sequence 长度不大时就是最简洁清晰的做法。更适合
    大规模输入的写法是先用 KMP 在 sequence 里定位 word 的所有出现位置，再检查
    这些位置是否首尾相接（前一个出现的结束位置正好是后一个出现的开始位置），
    统计最长的连续相接段——但由于本题约束里字符串都很短，直接暴力尝试的写法
    更清晰，也是官方题解认可的标准解法。
    【复杂度】时间 O(k_max * n)（每次 `in` 判断是 O(n)，最坏尝试 O(n/len(word))
    次）；空间 O(k_max * len(word))（构造 word*k 这个临时字符串）。
    【易错点】1) 循环终止条件容易写反——应该是"当 word*(k+1) 不再是子串时停止，
    返回 k"，而不是"当 word*k 不是子串时返回 k"（会少算一次）；2) word 本身不是
    sequence 子串时要返回 0，不能因为循环从 k=1 开始漏判这个基础情况。
    """
    k = 0
    candidate = word
    while candidate in sequence:
        k += 1
        candidate += word
    return k


def shortest_common_supersequence(str1: str, str2: str) -> str:
    """
    【题意】LeetCode 1092·Hard。给定两个字符串 str1 和 str2，返回同时以 str1 和
    str2 为子序列的最短字符串（如果有多个满足条件的最短串，返回任意一个）。
    【思路】这是"字符串算法 + DP"结合的典型：最短公共父序列的长度公式是
    `len(str1) + len(str2) - LCS(str1, str2)`——直觉是，str1 和 str2 共享的
    最长公共子序列（LCS）部分只需要在结果里出现一次，其余不属于 LCS 的字符各自
    保留。做法分两步：1) 先用标准的二维 DP 算出 LCS 长度表 `dp[i][j]`（str1 前
    i 个字符和 str2 前 j 个字符的 LCS 长度）；2) 从 `dp[m][n]` 开始倒着回溯——
    如果 `str1[i-1] == str2[j-1]`，说明这个字符属于 LCS，只需要写一次，i、j
    都往前退一步；否则，看 `dp[i-1][j]` 和 `dp[i][j-1]` 哪个更大，说明"跳过"
    str1 当前字符或 str2 当前字符不会破坏 LCS 长度，就把这个没被跳过的字符
    单独写进结果（因为它不属于 LCS，必须原样保留），对应指针退一步。回溯得到的
    字符序列是反着的，最后需要整体反转。
    【复杂度】时间 O(m*n)（DP 表構造 + 回溯路径最长 O(m+n)）；空间 O(m*n)（DP
    表，可以用滚动数组优化到 O(min(m,n))，但回溯重建结果需要完整的表）。
    【易错点】1) 回溯时如果两个字符不相等，必须比较 `dp[i-1][j]` 和
    `dp[i][j-1]` 的大小来决定退哪一个指针，而不能随便退一个——退错方向会导致
    重建出的字符串丢字符或长度不对；2) 回溯过程是从后往前构造字符，容易忘记
    最后要把结果整体反转；3) 当 i 或 j 已经退到 0 时，剩下的另一个字符串的
    剩余部分要原样整体拼接进去，不能遗漏。
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j = m, n
    merged: list[str] = []
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            merged.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            merged.append(str1[i - 1])
            i -= 1
        else:
            merged.append(str2[j - 1])
            j -= 1
    while i > 0:
        merged.append(str1[i - 1])
        i -= 1
    while j > 0:
        merged.append(str2[j - 1])
        j -= 1
    merged.reverse()
    return "".join(merged)


def z_function(s: str) -> list[int]:
    """
    【custom】Z 函数：计算每个位置 i（0 <= i < n）上，`s[i:]` 与整个字符串 s
    本身的最长公共前缀长度，记作 z[i]（约定 z[0] = 0，因为"和自己完全重合"这个
    平凡情况通常不计入，遵循大多数教材的惯例）。
    【思路】暴力做法是对每个 i 都从头逐字符比较 `s[i:]` 和 `s`，是 O(n^2)。
    Z 函数用一个"目前已知匹配得最靠右的窗口" `[l, r]`（这个窗口是某个之前算出的
    `z[k]` 对应的、和 s 前缀相同的那一段区间，r 是这类窗口里右端点最靠右的一个）
    来复用信息：算 z[i] 时，如果 i 落在 `[l, r]` 内部，那么 `s[i:]` 和
    `s[i-l:]` 在 `[i, r]` 这一段内容必然相同（因为 `[l,r]` 本身就是和 s 前缀
    相同的一段），所以可以直接借用 `z[i-l]` 作为初始猜测（但要和 `r-i+1`
    取较小值，因为超出窗口的部分没有保证），再从这个初始值继续暴力扩展；如果 i
    在窗口外，只能老老实实从 0 开始暴力扩展。每次暴力扩展成功都会把窗口右端点
    往右推，均摊下来总扩展次数是 O(n)。
    【复杂度】时间 O(n)（虽然内层有暴力扩展，但依赖窗口 [l,r] 只会整体右移，
    均摊线性）；空间 O(n)（z 数组）。
    【易错点】1) 窗口内借用 `z[i-l]` 时忘记和 `r-i+1` 取 min，会把"窗口内保证
    相同"的这段长度错误地当成了扩展的下限，可能导致后续暴力扩展比较时越界或
    读到不该读的字符；2) 更新窗口 `[l, r]` 的条件应该是"本次扩展得到的右端点
    `i+z[i]-1` 比当前 r 更大"才更新，不是每次都无条件更新；3) `z[0]` 按惯例
    设为 0（不是 len(s)），如果不做这个特殊处理，后续如果拿 z 数组去做"最长
    公共前后缀"之类的推理容易被 z[0] 干扰。
    """
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


def find_pattern_occurrences_z(text: str, pattern: str) -> list[int]:
    """
    【custom】用 Z 函数实现模式匹配：返回 pattern 在 text 中所有出现的起始下标
    （和 KMP 解决的是同一个问题，用来对照体会两种算法"表述不同、效果等价"）。
    【思路】构造拼接串 `pattern + sep + text`（sep 是一个保证不会出现在 pattern
    或 text 中的分隔符，这里用 `"\x00"`，避免 pattern 恰好和拼接串的某个跨越
    分隔符的片段发生假匹配），对这个拼接串跑一遍 Z 函数。拼接串中"以 text 里
    某个位置开头的后缀"如果和 pattern 本身的公共前缀长度恰好等于
    `len(pattern)`，就说明 pattern 在这个位置完整出现了一次——这正是
    z[i] == len(pattern) 的位置。这一实现从 KMP 的"失配指针跳转"切换到了
    "计算每个位置与整体前缀的最长公共前缀"这个完全不同的视角，但解决的是同一类
    问题，呼应 24 类 KMP 和这一类 Z 函数"表述不同、目的相同"的关系。
    【复杂度】时间 O(len(pattern) + len(text))（Z 函数本身线性）；空间
    O(len(pattern) + len(text))（拼接串和 z 数组）。
    【易错点】1) 忘记加分隔符直接拼接 `pattern + text`，会导致 pattern 末尾
    和 text 开头恰好续接的情况产生错误的"假匹配"；2) 结果下标要减去
    `len(pattern) + 1`（pattern 长度 + 分隔符 1 个字符）换算回 text 里的真实
    起始下标，容易漏减或多减分隔符这一位；3) pattern 为空串时按惯例每个位置
    都算"出现"，需要单独处理，否则空 pattern 会导致构造出的拼接串行为异常。
    """
    if not pattern:
        return list(range(len(text) + 1))
    combined = pattern + "\x00" + text
    z = z_function(combined)
    offset = len(pattern) + 1
    plen = len(pattern)
    return [i - offset for i in range(offset, len(combined)) if z[i] == plen]


class AhoCorasick:
    """
    【custom】Aho-Corasick 自动机：一次预处理多个模式串（patterns），支持对任意
    text 一次扫描就找出所有模式串的所有出现位置，而不必对每个模式串单独跑一遍
    KMP（如果模式串有 k 个、每个长度均值 m，单独跑 KMP 是 O(k*(m+n))，
    Aho-Corasick 建自动机是 O(sum(len(patterns)))，查询是 O(n + 出现次数)，
    和模式串数量 k 基本无关）。
    【思路】核心结构是"Trie 树 + 失败指针（fail pointer）"，可以类比成"KMP 的
    next 数组从一条链搬到了一棵树上"：先把所有 patterns 插入一棵 Trie，Trie 上
    每个节点代表"从根到这里的路径"对应的某个（或多个）模式串的前缀。然后用 BFS
    从根开始逐层给每个节点算一个 fail 指针：`fail[node]` 指向 Trie 里"另一个
    节点，其代表的字符串是 node 代表字符串的最长真后缀，并且这个真后缀本身也是
    某个模式串的前缀"——这和 KMP 里 `lps[i]` 的含义（最长相同前后缀）完全类比，
    只是 KMP 的"链"变成了 Trie 的"树"。在 text 上扫描时，从根节点出发，每读入
    一个字符尝试沿 Trie 边走；如果当前节点没有这个字符的边（失配），就沿着 fail
    指针往回跳（而不是retreat 到根重新开始），直到找到一个有这条边的节点或者
    跳到根为止——这正是"失配时沿失败指针跳转，不回退到根重新开始"的核心加速点。
    每到达一个节点，都要顺着它的 fail 指针链（以及自身）检查是否有某个模式串在
    此刻结尾，累计所有匹配。
    【复杂度】时间：构造 Trie + fail 指针 O(sum(len(patterns)))；search(text)
    是 O(len(text) + 总匹配次数)（虽然沿 fail 链检查"输出"看起来可能是每步
    O(depth)，但在标准实现里用"输出链合并"或者简单的每节点输出集合预计算可以
    摊销掉，这里为了实现简洁，直接在 BFS 阶段把每个节点的 fail 链上所有可能的
    模式串输出预先合并到该节点，查询时只需 O(1) 取用）。空间 O(sum(len(
    patterns)) * 字符集大小)（Trie 节点数与总模式串长度成正比）。
    【易错点】1) fail 指针必须用 BFS（按 Trie 深度从浅到深）计算，不能用 DFS，
    因为计算深度为 d 的节点的 fail 指针依赖深度小于 d 的节点已经算好；2) 从
    根节点出发的直接子节点，其 fail 指针应该指向根节点本身（根没有比它更短的
    真后缀可退），这是 BFS 的初始条件，漏掉会导致后续所有节点的 fail 指针连锁
    出错；3) 每个节点的"输出集合"要包含它自己代表的模式串（如果这个节点恰好是
    某个模式串的终点）以及它 fail 指针指向的节点的输出集合（因为 fail 指向的
    字符串是当前字符串的后缀，如果那也是某个模式串，同样应该在这个位置报告
    出来），只统计自身终点会漏掉"作为其他模式串后缀出现"的匹配。
    """

    def __init__(self, patterns: list[str]) -> None:
        self.patterns = list(patterns)
        # 每个节点用 dict 表示: {"children": {char: node_id}, "fail": int, "output": set[str]}
        self._children: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._output: list[set[str]] = [set()]

        for pattern in self.patterns:
            node = 0
            for ch in pattern:
                if ch not in self._children[node]:
                    self._children.append({})
                    self._fail.append(0)
                    self._output.append(set())
                    self._children[node][ch] = len(self._children) - 1
                node = self._children[node][ch]
            self._output[node].add(pattern)

        # BFS 计算 fail 指针
        queue: deque[int] = deque()
        for ch, child in self._children[0].items():
            self._fail[child] = 0
            queue.append(child)
        while queue:
            node = queue.popleft()
            for ch, child in self._children[node].items():
                queue.append(child)
                f = self._fail[node]
                while f != 0 and ch not in self._children[f]:
                    f = self._fail[f]
                # 标准写法：fail[child] = children[fail[node]].get(ch, root)
                self._fail[child] = self._children[f].get(ch, 0)
                self._output[child] |= self._output[self._fail[child]]

    def search(self, text: str) -> dict[str, list[int]]:
        """返回每个模式串在 text 中出现的所有起始下标（升序）。"""
        result: dict[str, list[int]] = {p: [] for p in self.patterns}
        node = 0
        for i, ch in enumerate(text):
            while node != 0 and ch not in self._children[node]:
                node = self._fail[node]
            node = self._children[node].get(ch, 0)
            for pattern in self._output[node]:
                result[pattern].append(i - len(pattern) + 1)
        for pattern in result:
            result[pattern].sort()
        return result


def build_suffix_array(s: str) -> list[int]:
    """
    【custom】后缀数组简化构造：返回把 s 的所有后缀按字典序排序后，各后缀起始
    下标组成的数组（不追求 O(n log n) 的倍增法，直接用"切片 + 排序"的
    O(n^2 log n) 简单实现，教学重点是"后缀数组是什么、能干什么"，而不是最优
    构造算法）。
    【思路】s 一共有 n 个后缀：`s[0:]`, `s[1:]`, ..., `s[n-1:]`。后缀数组就是
    "如果把这 n 个后缀按字符串字典序从小到大排序，排序后每个位置对应的原始起始
    下标"。直接用 Python 内置 `sorted`，以 `(下标, s[下标:])` 为 key 按字符串
    比较排序即可——每次比较两个后缀是 O(n) 字符比较，排序本身 O(n log n) 次
    比较，总体 O(n^2 log n)。后缀数组是很多"全体后缀相关问题"的基础设施：比如
    排序后相邻两个后缀的最长公共前缀（LCP）之和可以用来求"不同子串个数"
    （LeetCode 1698，Number of Distinct Substrings in a String 就是这个应用，
    该题为付费题，这里用后缀数组配合下面的 `count_distinct_substrings` 演示
    同样的思路），最长重复子串问题（LC1044，24 类已用滚动哈希+二分实现）也可以
    用排序后相邻后缀的 LCP 最大值来解。
    【复杂度】时间 O(n^2 log n)（n 次切片各 O(n) 空间，排序时比较函数最坏 O(n)，
    n log n 次比较）；空间 O(n^2)（保存所有切片这一简化实现的代价，倍增法可以
    做到 O(n)）。
    【易错点】1) 排序 key 只用 `s[i:]`（切片本身）而忘记原始下标 i，排序后就
    丢失了"这个后缀在原字符串里的起始位置"这个后缀数组本该保留的核心信息；
    2) 空字符串 s 时后缀数组应该是空列表（没有任何后缀，包括不存在"空后缀"
    这一说），容易误把 n=0 的情况和"只有一个后缀"搞混。
    """
    n = len(s)
    indices = list(range(n))
    indices.sort(key=lambda i: s[i:])
    return indices


def count_distinct_substrings(s: str) -> int:
    """
    【custom】统计字符串 s 一共有多少个"不同"的子串（同一内容的子串只算一次），
    这正是 LeetCode 1698（Number of Distinct Substrings in a String）的题意，
    该题是付费题，这里用后缀数组 + LCP（最长公共前缀）实现同等效果的自实现版本。
    【思路】s 的子串总数（不去重）是 `n*(n+1)/2`（每个后缀贡献 `n-i` 个不同
    起点相同、长度不同的子串，等价于每个后缀的所有前缀）。如果把所有后缀按字典
    序排好（后缀数组），相邻两个排好序的后缀之间的"最长公共前缀"长度
    `lcp[k]`，恰好等于"排在后面这个后缀的、和排在前面这个后缀重复计数的子串
    个数"——因为一个后缀的所有前缀本质就是"以这个后缀起始下标为左端点的所有
    子串"，而字典序相邻的两个后缀之间共享的前缀部分，就是会被重复计数的那些
    子串。所以 `distinct = n*(n+1)/2 - sum(lcp[k] for k in 1..n-1)`：先假设
    所有子串都不同（总数上限），再减去"由于和字典序相邻的后缀共享前缀而被
    重复计数"的部分。
    【复杂度】时间 O(n^2)（后缀数组构造是本实现的瓶颈，O(n^2 log n)；计算相邻
    LCP 是额外 O(n^2)）；空间 O(n^2)（保存后缀切片）。
    【易错点】1) 只统计"排序后紧邻的一对"后缀的 LCP，不能跨着比较不相邻的两个
    后缀——字典序排序保证了"和当前后缀共享前缀最长的另一个后缀，必然是排序后
    紧邻的那一个"，这是这个公式成立的关键前提；2) 求和是从下标 1 开始（第 0
    个后缀没有"前一个"可比较），漏掉边界会导致结果偏大；3) 空字符串应返回 0。
    """
    n = len(s)
    if n == 0:
        return 0
    sa = build_suffix_array(s)
    total = n * (n + 1) // 2
    overlap = 0
    for k in range(1, n):
        a, b = s[sa[k - 1] :], s[sa[k] :]
        m = min(len(a), len(b))
        common = 0
        while common < m and a[common] == b[common]:
            common += 1
        overlap += common
    return total - overlap


def longest_common_substring(s1: str, s2: str) -> str:
    """
    【custom】最长公共子串（注意不是最长公共子序列 LCS——子串要求连续，子序列
    允许跳跃删除字符，13 类已经练过后者/编辑距离一脉，这里专门练"连续"版本）：
    返回 s1 和 s2 的最长公共连续子串；如果有多个长度相同的最长解，返回任意一个。
    【思路】标准区间 DP：`dp[i][j]` 表示"以 s1[i-1] 结尾、以 s2[j-1] 结尾"的
    最长公共子串长度（注意这个定义要求这个公共子串必须真的"用 s1[i-1] 和
    s2[j-1] 收尾"，不是"s1 前 i 个字符和 s2 前 j 个字符之间任意位置的最长公共
    子串"——这个"必须以当前位置收尾"的约束是子串 DP 和子序列 DP 最本质的区别）。
    转移：如果 `s1[i-1] == s2[j-1]`，说明可以把 `dp[i-1][j-1]` 这段公共子串
    再往后延伸一个字符，`dp[i][j] = dp[i-1][j-1] + 1`；如果两个字符不相等，
    "以这两个位置收尾的公共子串"根本不存在（子串要求连续，一旦断开这里的公共
    子串就必须结束），`dp[i][j] = 0`（这里是和 LCS DP 最大的不同：LCS 在两个
    字符不等时会取 `max(dp[i-1][j], dp[i][j-1])` 继续尝试跳过某个字符，子串
    DP 不允许跳跃，直接归零）。全局最长公共子串的长度就是整张 dp 表里的最大值，
    记录下取得最大值时的 `(i, j)`，往回切片 `s1[i-length:i]` 就是答案。
    【复杂度】时间 O(len(s1) * len(s2))（二维 DP 表的每个格子 O(1) 转移）；
    空间 O(len(s1) * len(s2))（可以用滚动数组优化到 O(min(len(s1),len(s2)))，
    这里为了清晰保留完整表）。
    【易错点】1) 字符不相等时错误地沿用 LCS 的转移公式
    `max(dp[i-1][j], dp[i][j-1])`，会把"允许跳跃拼接"的子序列逻辑混进"必须
    连续"的子串问题里，得到错误的过大结果；2) 找到最大值后切片时容易切错端点
    ——`dp[i][j]` 存的是"以 s1 下标 i-1 结尾"的长度，切片应该是
    `s1[i-length:i]`（左闭右开），不是 `s1[i-length-1:i-1]` 之类的偏移。
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best_len, best_end = 0, 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_end = i
    return s1[best_end - best_len : best_end]


def _brute_force_z(s: str) -> list[int]:
    """暴力逐位比较，用于交叉验证 z_function。"""
    n = len(s)
    z = [0] * n
    for i in range(1, n):
        k = 0
        while i + k < n and s[k] == s[i + k]:
            k += 1
        z[i] = k
    return z


def _brute_force_aho_corasick(patterns: list[str], text: str) -> dict[str, list[int]]:
    """对每个 pattern 单独用 str.find 循环查找所有出现位置，用于交叉验证。"""
    result: dict[str, list[int]] = {}
    for pattern in patterns:
        occurrences = []
        if pattern:
            start = text.find(pattern)
            while start != -1:
                occurrences.append(start)
                start = text.find(pattern, start + 1)
        result[pattern] = occurrences
    return result


def _brute_force_suffix_array(s: str) -> list[int]:
    """直接对所有后缀切片排序，用于交叉验证 build_suffix_array。"""
    return sorted(range(len(s)), key=lambda i: s[i:])


def _brute_force_longest_common_substring(s1: str, s2: str) -> int:
    """暴力枚举 s1 的所有子串，检查是否也是 s2 的子串，取最长长度，用于交叉验证。
    仅用于短字符串（<=15），避免 O(n^3) 级别的枚举太慢。"""
    best = 0
    n = len(s1)
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = s1[i:j]
            if len(sub) > best and sub in s2:
                best = len(sub)
    return best


def _self_test() -> None:
    # ---- 796 Rotate String ----
    assert rotate_string("abcde", "cdeab") is True
    assert rotate_string("abcde", "abced") is False

    # ---- 1668 Maximum Repeating Substring ----
    assert max_repeating_substring("ababc", "ab") == 2
    assert max_repeating_substring("ababc", "ba") == 1
    assert max_repeating_substring("ababc", "ac") == 0

    # ---- 1092 Shortest Common Supersequence ----
    result = shortest_common_supersequence("abac", "cab")
    assert len(result) == 5  # len(str1)+len(str2)-LCS = 4+3-2 = 5
    # 交叉验证：结果必须同时把 str1 和 str2 作为子序列包含
    def _is_subsequence(sub: str, full: str) -> bool:
        it = iter(full)
        return all(ch in it for ch in sub)

    assert _is_subsequence("abac", result)
    assert _is_subsequence("cab", result)

    # ---- Z 函数：暴力交叉验证多个测试字符串 ----
    for test_s in ["aaabaab", "abcabcabc", "aaaaaa", "abababab", "z", ""]:
        assert z_function(test_s) == _brute_force_z(test_s), f"z_function 与暴力结果不一致: {test_s!r}"

    # ---- 用 Z 函数做模式匹配，和 str.find 全部出现位置交叉验证 ----
    def _all_occurrences_naive(text: str, pattern: str) -> list[int]:
        occurrences = []
        start = text.find(pattern)
        while start != -1:
            occurrences.append(start)
            start = text.find(pattern, start + 1)
        return occurrences

    for text, pattern in [
        ("ababcababab", "aba"),
        ("aaaaaa", "aa"),
        ("abcdef", "xyz"),
        ("mississippi", "issi"),
    ]:
        assert find_pattern_occurrences_z(text, pattern) == _all_occurrences_naive(text, pattern)

    # ---- Aho-Corasick：暴力逐个 str.find 交叉验证 ----
    patterns = ["he", "she", "his", "hers"]
    text = "ahishers"
    ac = AhoCorasick(patterns)
    assert ac.search(text) == _brute_force_aho_corasick(patterns, text)

    patterns2 = ["a", "ab", "bab", "bc", "bca", "c", "caa"]
    text2 = "abccab"
    ac2 = AhoCorasick(patterns2)
    assert ac2.search(text2) == _brute_force_aho_corasick(patterns2, text2)

    # ---- 后缀数组：暴力排序交叉验证 ----
    for test_s in ["banana", "abcabcabc", "aaaa", "z", "mississippi"]:
        assert build_suffix_array(test_s) == _brute_force_suffix_array(test_s)

    # ---- 不同子串个数（1698 风格）：暴力枚举所有子串放入 set 交叉验证 ----
    def _brute_force_distinct_substrings(s: str) -> int:
        subs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n + 1):
                subs.add(s[i:j])
        return len(subs)

    for test_s in ["aabbaba", "aaaa", "abcabc", "z", ""]:
        assert count_distinct_substrings(test_s) == _brute_force_distinct_substrings(test_s)

    # ---- 最长公共子串：DP 结果与暴力枚举交叉验证（长度<=15）----
    for s1, s2 in [
        ("abcdxyz", "xyzabcd"),
        ("zxabcdezy", "yzabcdezx"),
        ("abc", "def"),
        ("aaaa", "aa"),
    ]:
        result = longest_common_substring(s1, s2)
        assert result in s1 and result in s2
        assert len(result) == _brute_force_longest_common_substring(s1, s2)

    print(
        "[PASS] p28_string_matching_advanced: 9/9 项通过 "
        "(796 Rotate String / 1668 Maximum Repeating Substring / "
        "1092 Shortest Common Supersequence / Z函数 / Z函数模式匹配 / "
        "Aho-Corasick / 后缀数组 / 不同子串个数 / 最长公共子串)"
    )


if __name__ == "__main__":
    _self_test()
