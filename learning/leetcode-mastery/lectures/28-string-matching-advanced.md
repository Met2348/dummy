# L28 · 字符串匹配进阶 —— Z 函数 / Aho-Corasick / 后缀数组 / 最长公共子串

> 对应代码：`src/problems/p28_string_matching_advanced.py`。

## 这一章在整个字符串算法体系里的位置

这批是字符串算法的"重型武器库"——24 类的 KMP 解决的是"一个模式串在一段文本里
匹配"，这一类把同一个问题从三个方向继续深挖：

- **Z 函数**和 **KMP** 解决的其实是同一类问题（前后缀匹配 / 模式串定位），只是
  表述方式不同：KMP 靠 `lps` 数组记录"模式串自身的最长相同前后缀"、失配时靠这个
  数组跳转；Z 函数直接算"每个后缀和整个字符串本身的最长公共前缀"，把模式匹配
  问题转化成"在 `pattern + 分隔符 + text` 上找哪些位置的 Z 值等于 pattern 长度"。
  两者复杂度都是 O(n)，但思考角度完全不同，能同时掌握有助于理解"字符串匹配"这个
  问题本质上有多种等价的刻画方式。
- **Aho-Corasick** 把 KMP 从"一个模式串"扩展到"多个模式串同时匹配"——如果要在一段
  文本里同时找 100 个敏感词，对每个词单独跑一遍 KMP 是 O(100·n)，Aho-Corasick
  只需要一次线性扫描就能找到所有词的所有出现位置。
- **后缀数组**（以及配套的 LCP 数组）是处理"所有后缀相关问题"的通用工具——重复
  子串、不同子串计数、最长公共子串，本质上都是"把所有后缀排好序之后，答案就藏在
  相邻后缀的比较结果里"这同一个思路的不同应用。

## 深挖：Aho-Corasick 自动机 —— Trie 上的"KMP"

**先想清楚要解决什么问题**：给定一组模式串 `patterns = [p1, p2, ..., pk]` 和一段
文本 `text`，要找出每个 `pi` 在 `text` 里的所有出现位置。如果对每个 `pi` 单独
调用一次 KMP，复杂度是 `O(sum(len(pi)) + k·n)`——当 k 很大（比如敏感词库有几万个
词）时，`k·n` 这一项会成为瓶颈。Aho-Corasick 的目标是把它降到
`O(sum(len(pi)) + n)`，让复杂度和模式串的**数量**基本无关。

**核心结构 = Trie + 失败指针（fail pointer）**。第一步是把所有模式串插入一棵
Trie 树——Trie 上每个节点天然代表"从根到这里的路径"这一个字符串（它可能是某个
模式串的前缀，也可能恰好是某个模式串本身）。

```python
for pattern in self.patterns:
    node = 0
    for ch in pattern:
        if ch not in self._children[node]:
            self._children.append({})
            ...
            self._children[node][ch] = len(self._children) - 1
        node = self._children[node][ch]
    self._output[node].add(pattern)
```

第二步，也是整个算法的精髓：给 Trie 上每个节点算一个 **fail 指针**，`fail[node]`
指向"另一个节点，其代表的字符串是 `node` 代表字符串的**最长真后缀**，同时这个
真后缀本身也必须是某个模式串的前缀（也就是必须真实存在于 Trie 里）"。这个定义
和 KMP 里 `lps[i]`（"`pattern[0..i]` 的最长相同真前后缀长度"）是**同一个概念**，
只是 KMP 的字符串只有一条链，"退到哪里"用一个整数下标就能表示；Aho-Corasick 把
多个模式串同时塞进一棵树，"退到哪里"就变成了"退到树上的哪个节点"，所以 `lps`
数组升级成了 `fail` 指针。

`fail` 指针必须用 **BFS**（按 Trie 深度从浅到深）计算，因为算深度为 `d` 的节点
的 `fail` 指针，依赖它的父节点（深度 `d-1`）的 `fail` 指针已经算好：

```python
queue: deque[int] = deque()
for ch, child in self._children[0].items():
    self._fail[child] = 0          # 根的直接子节点，fail 指针指回根
    queue.append(child)
while queue:
    node = queue.popleft()
    for ch, child in self._children[node].items():
        queue.append(child)
        f = self._fail[node]
        while f != 0 and ch not in self._children[f]:
            f = self._fail[f]
        self._fail[child] = self._children[f].get(ch, 0)
        self._output[child] |= self._output[self._fail[child]]
```

**匹配阶段是这个结构真正发挥作用的地方**：从根节点出发，逐字符扫描 `text`，
每读入一个字符尝试沿 Trie 边往下走；如果当前节点没有这条边（失配），**不是**
退回根节点重新开始（那样就退化成对每个起点都暴力试一遍），而是**沿着 fail 指针
往回跳**，直到找到一个有这条边的节点，或者跳到根为止——这正是"KMP 失配时跳
`lps[j-1]`、不回退到 0 重新开始"这个加速思想在树上的推广。每到达一个节点，都要
检查它（以及它 fail 链上所有祖先，已经在建树阶段合并进 `output` 集合里）是否有
某个模式串在此刻结尾。

**和 24 类 KMP 的关系一句话总结**：KMP 是"一条链上的失败指针"，Aho-Corasick 是
"一棵树上的失败指针"；本质是同一个"失配时别从头再来，跳到已知的最长有效后缀"的
思想，只是数据结构从字符串升级成了 Trie。

## 其余项点睛

- **796 Rotate String**：判断 `s` 循环移位后能否变成 `goal`。关键观察是 `s` 的
  所有循环移位结果都必然是 `s+s` 的子串，所以只需要判断 `goal in (s+s)` 并额外
  检查长度相等。可以用 KMP 或 Z 函数在 `s+s` 里查找 `goal` 做到严格线性时间。
- **1668 Maximum Repeating Substring**：求最大的 `k` 使 `word*k` 是 `sequence`
  的子串。这个判定天然单调（`word` 重复 k 次都接不上，重复更多次也不可能突然
  接上），直接从 `k=1` 递增尝试到失败为止即可，是"看似需要复杂算法、实际暴力
  最简洁"的题目。
- **1092 Shortest Common Supersequence**：求同时以两个字符串为子序列的最短串，
  本质是"字符串 + DP"的结合——先用标准 LCS 的 DP 求最长公共子序列，再从 DP 表
  倒着回溯：相等的字符只写一次（属于 LCS），不相等时看 `dp[i-1][j]` 和
  `dp[i][j-1]` 哪个更大决定保留哪个字符，最后把回溯结果反转。
- **Z 函数**：`z[i]` 表示 `s[i:]` 与 `s` 本身的最长公共前缀长度。用一个"目前
  已知匹配到最靠右"的窗口 `[l, r]` 复用之前算出的结果，均摊 O(n)；在窗口内可以
  直接借用 `z[i-l]` 作为初始猜测（但要和 `r-i+1` 取 min），再继续暴力扩展。
- **后缀数组 + 不同子串计数**：把字符串所有后缀按字典序排序，排序后相邻两个
  后缀的最长公共前缀（LCP）之和，恰好是"被重复计数的子串数量"——用总子串数
  `n(n+1)/2` 减去这个和，就是不同子串个数（这正是付费题 1698 的解法思路，这里
  用自实现的简化后缀数组演示同样的推理，不依赖付费题本身）。
- **最长公共子串**（不是子序列）：`dp[i][j]` 表示"必须以 `s1[i-1]` 和
  `s2[j-1]` 结尾"的最长公共子串长度，字符不相等时直接归零（不像 LCS 那样可以
  跳跃取 max）——"必须以当前位置收尾、不允许跳跃"是子串 DP 和子序列 DP 最本质
  的区别。
