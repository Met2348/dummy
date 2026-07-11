"""分类 20：Trie 前缀树 —— 大量字符串的公共前缀查询，复杂度只和字符串长度有关。"""
from __future__ import annotations

import bisect


class Trie:
    """
    【题意】实现一棵前缀树，支持 insert(word) 插入单词、search(word) 判断某个完整单词
    是否被插入过、starts_with(prefix) 判断是否存在任意一个已插入单词以 prefix 为前缀。
    【思路】用嵌套 dict 模拟 Trie 节点：self.children 本身就是根节点，一个节点是一个
    dict，key 是字符，value 是"走这个字符之后到达的下一个节点"（也是一个 dict）。插入
    时沿着单词逐字符往下走，节点不存在就用 setdefault 创建；走完整个单词后，在终点节点
    里塞一个哨兵键 "#"，表示"从根到这里的这条路径，恰好是一个完整插入过的单词"（而不只
    是别的单词的前缀）。search 和 starts_with 的唯一区别就是：走到路径尽头后，search 还
    要额外检查终点节点是否有 "#" 标记，starts_with 只要求路径存在即可——这正是"是完整
    单词"和"只是某个单词的前缀"的本质区别。
    【复杂度】insert/search/starts_with 均为 O(L)，L 是单词或前缀的长度，和 Trie 里
    已经存了多少个单词无关；空间 O(所有单词的字符总数)（公共前缀的节点会被共享，不会
    重复存储）。
    【易错点】1) search("app") 在插入过 "apple" 但没插入过 "app" 时必须返回 False——
    路径是存在的（apple 的前几个字符经过 app），但终点节点没有 "#" 标记，容易只判断
    "路径是否存在"而忘记检查这个终点标记；2) 用嵌套 dict 时，要用 setdefault 而不是
    直接 node[ch] = {}，否则会把已经存在的、可能还挂着后续字符的子树整个覆盖掉。
    """

    def __init__(self) -> None:
        self.children: dict[str, dict] = {}

    def insert(self, word: str) -> None:
        node = self.children
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = True

    def _find_node(self, s: str) -> dict | None:
        node = self.children
        for ch in s:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and "#" in node

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None


class WordDictionary:
    """
    【题意】实现一个支持通配符的单词字典：add_word(word) 添加单词；search(word) 判断
    是否存在一个已添加的单词与 word 匹配，word 里的 '.' 可以匹配任意一个字符。
    【思路】底层结构和 Trie 完全一样（嵌套 dict + "#" 终点标记），区别只在 search 的
    匹配逻辑：普通字符必须精确匹配、只能走这一条边；但遇到 '.' 时，当前节点的每一个
    子节点都是候选的下一跳，必须逐一尝试——这就是"和普通字符串匹配唯一的不同"，需要用
    DFS/回溯遍历所有候选分支，只要有一条分支能匹配到单词结尾（且终点有 "#" 标记）就
    算成功，不能只探一条路径就下结论。
    【复杂度】最坏情况（word 全是 '.'）时间 O(26^L)（每一位都要尝试 26 个子节点），
    但实际 Trie 里的分支数远小于 26，通常远快于这个上界；空间 O(所有单词字符总数)。
    【易错点】1) 遇到 '.' 时忘记排除掉哨兵键 "#" 本身（"#" 也是 node 的一个 key，但它
    不代表"匹配了一个字符"，如果把它当成候选分支往下递归会出错）；2) 只要任意一个分支
    匹配成功就应该立刻返回 True，不能等所有分支都尝试完才判断，否则会做很多不必要的
    搜索（虽然结果依然正确，但效率明显变差）。
    """

    def __init__(self) -> None:
        self.children: dict[str, dict] = {}

    def add_word(self, word: str) -> None:
        node = self.children
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = True

    def search(self, word: str) -> bool:
        def dfs(node: dict, i: int) -> bool:
            if i == len(word):
                return "#" in node
            ch = word[i]
            if ch == ".":
                for key, child in node.items():
                    if key != "#" and dfs(child, i + 1):
                        return True
                return False
            if ch not in node:
                return False
            return dfs(node[ch], i + 1)

        return dfs(self.children, 0)


def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    """
    【题意】给定字母网格 board 和单词列表 words，找出所有能在网格里通过"水平/竖直相邻、
    同一个格子不能重复使用"的路径拼出来的单词，返回这些单词（不要求顺序、不重复）。
    【思路】如果对每个单词单独在网格上做一次 79 题那样的"单词搜索"DFS，复杂度是
    O(单词数 * 网格搜索)，大量单词之间共享的前缀（比如 "oath" 和 "oat" 前三个字母
    一样）会被重复地在网格上搜索好几遍。更好的做法是把全部 words 一次性建成一棵 Trie
    （在单词末尾节点存整个单词字符串，而不是普通的 True 标记，方便直接取用），然后
    只对网格做**一次**遍历：从每个格子出发做 DFS，但每一步只往 Trie 当前节点还存在的
    分支走——Trie 天然把"从当前路径出发，还有没有单词可能以它为前缀"这个判断变成
    O(1) 的字典查找，不匹配任何单词前缀的分支会被立刻剪掉，不会浪费时间探索。额外的
    优化：一旦某个 Trie 节点被找到对应单词就从节点里删掉这个单词标记（防止重复加入
    结果），如果一个 Trie 节点被走空了（不再指向任何单词），就把它从父节点里剪掉——
    这样搜索会越搜索分支越少，越搜越快。
    【复杂度】建 Trie O(所有单词字符总数)；网格 DFS 最坏 O(R*C*4^L)（L 是单词最大
    长度），但 Trie 剪枝在实践中让它远快于"逐词单独搜索"的朴素做法。空间 O(Trie 大小)。
    【易错点】1) 网格 DFS 时必须临时"淹没"当前格子（比如改成 "#"）防止同一条路径
    重复用同一个格子，递归返回前要恢复原字符，否则会污染后续从别的起点出发的搜索；
    2) 找到一个单词后如果不清除 Trie 里的终点标记，同一个单词可能因为存在多条路径
    被重复加入结果列表；3) 别忘了"提前剪枝"这个优化不是必须的（不加也能算对），但
    如果单词量很大、网格很大，不剪枝会明显变慢，属于这道 Hard 题真正的难点所在。
    """
    root: dict = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = w

    if not board or not board[0]:
        return []
    rows, cols = len(board), len(board[0])
    found: list[str] = []

    def dfs(r: int, c: int, parent: dict) -> None:
        letter = board[r][c]
        curr = parent[letter]
        word = curr.pop("$", None)
        if word is not None:
            found.append(word)

        board[r][c] = "#"
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] in curr:
                dfs(nr, nc, curr)
        board[r][c] = letter

        if not curr:
            parent.pop(letter)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] in root:
                dfs(r, c, root)
    return found


class MapSum:
    """
    【题意】实现一个 key-value 结构：insert(key, val) 插入或覆盖（同一个 key 重复
    insert 会用新值覆盖旧值，不是累加）；sum(prefix) 返回所有以 prefix 为前缀的
    key 对应 val 的总和。
    【思路】如果每次 sum 都遍历一遍所有 key 判断前缀再求和，时间和 key 的总数成正比。
    更好的做法是把"前缀和"直接维护在 Trie 的节点上：每个 Trie 节点额外存一个 "_sum"
    字段，表示"以从根到这个节点的路径为前缀的所有 key，它们 val 的总和"。insert 时，
    先算出这次 insert 相对于该 key 旧值的增量 delta（新值 - 旧值，第一次插入旧值视为
    0），然后沿着 key 的每一个字符往下走，把路径上每一个节点的 "_sum" 都加上 delta——
    因为这个 key 本身就是这些前缀节点"会被计入"的一员，它的值变化多少，这些前缀节点
    的总和就该变化多少。这样 insert 是 O(key 长度)，sum 只需要沿 prefix 走到对应
    节点直接读 "_sum"，也是 O(prefix 长度)，不需要遍历任何其他 key。
    【复杂度】insert: O(L)；sum: O(L)（L 为对应字符串长度）；空间 O(所有 key 字符
    总数)。
    【易错点】1) 重复 insert 同一个 key 时，如果直接把新 val 累加到路径节点上（而不是
    先算出"增量 delta"），会把旧值也重复计入，变成"累加"而不是题目要求的"覆盖"；
    2) 必须额外用一个 key_val 字典记住每个 key 当前的值，否则算不出正确的 delta。
    """

    def __init__(self) -> None:
        self.children: dict[str, dict] = {}
        self.key_val: dict[str, int] = {}

    def insert(self, key: str, val: int) -> None:
        delta = val - self.key_val.get(key, 0)
        self.key_val[key] = val
        node = self.children
        for ch in key:
            node = node.setdefault(ch, {"_sum": 0})
            node["_sum"] = node.get("_sum", 0) + delta

    def sum(self, prefix: str) -> int:
        node = self.children
        for ch in prefix:
            if ch not in node:
                return 0
            node = node[ch]
        return node.get("_sum", 0)


def suggested_products(products: list[str], search_word: str) -> list[list[str]]:
    """
    【题意】给定商品名称列表 products 和用户正在输入的 search_word，模拟"搜索框自动
    联想"：对 search_word 每输入一个字符（形成一个更长的前缀），从 products 里找出
    最多 3 个以当前前缀开头、字典序最小的商品，返回每一步对应的候选列表。
    【思路】虽然这类"前缀查询"很容易联想到 Trie，但这道题其实用排序 + 二分更直接：
    先把 products 整体按字典序排序，那么"所有以某个前缀 p 开头的商品"在排序后的数组里
    一定是**连续的一段**，并且这一段里最靠前的 3 个，天然就是字典序最小的 3 个（因为
    整个数组已经有序）。用 bisect_left 找到"第一个字典序 >= p 的商品"的位置（也就是
    这一段连续区间的起点），然后往后最多取 3 个、且要再检查一下它们是否真的以 p 为
    前缀（避免起点之后的商品虽然字典序更大但根本不共享这个前缀）。逐个字符累积前缀，
    重复这个"二分定位 + 取前 3 个"的过程即可。
    【复杂度】排序 O(n log n)；之后每个前缀一次二分查找 O(log n)，一共 O(m log n)
    （m 为 search_word 长度）；空间 O(n)（排序后的数组）。
    【易错点】1) 忘记先对 products 排序就直接二分，bisect 的前提是数组本身有序，
    不排序会得到错误的位置；2) 二分定位到的起点之后的元素不一定真的以当前前缀开头
    （它们只是字典序不小于前缀），必须额外用 startswith 校验，不能假设二分结果一定
    合法；3) 取候选时要注意数组末尾越界，`min(i + 3, len(products))` 要写对，否则
    在候选不足 3 个时会抛出下标越界或漏取。
    """
    products = sorted(products)
    result: list[list[str]] = []
    prefix = ""
    for ch in search_word:
        prefix += ch
        i = bisect.bisect_left(products, prefix)
        candidates = [
            products[j] for j in range(i, min(i + 3, len(products)))
            if products[j].startswith(prefix)
        ]
        result.append(candidates)
    return result


def _self_test() -> None:
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.starts_with("app") is True
    trie.insert("app")
    assert trie.search("app") is True

    wd = WordDictionary()
    wd.add_word("bad")
    wd.add_word("dad")
    wd.add_word("mad")
    assert wd.search("pad") is False
    assert wd.search("bad") is True
    assert wd.search(".ad") is True
    assert wd.search("b..") is True

    board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    assert sorted(find_words([row[:] for row in board], ["oath", "pea", "eat", "rain"])) == [
        "eat",
        "oath",
    ]

    ms = MapSum()
    ms.insert("apple", 3)
    assert ms.sum("ap") == 3
    ms.insert("app", 2)
    assert ms.sum("ap") == 5

    assert suggested_products(
        ["mobile", "mouse", "moneypot", "monitor", "mousepad"], "mouse"
    ) == [
        ["mobile", "moneypot", "monitor"],
        ["mobile", "moneypot", "monitor"],
        ["mouse", "mousepad"],
        ["mouse", "mousepad"],
        ["mouse", "mousepad"],
    ]

    print(
        "[PASS] p20_trie: 5/5 题通过 "
        "(实现Trie/添加与搜索单词/单词搜索II/键值映射/搜索推荐系统)"
    )


if __name__ == "__main__":
    _self_test()
