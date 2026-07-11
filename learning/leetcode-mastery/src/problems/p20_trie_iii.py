"""分类 20：Trie 前缀树 —— Phase 3 竞赛级补充：Trie 与回溯/记忆化/DP 的深度结合。"""
from __future__ import annotations


def word_break_ii(s: str, word_dict: list[str]) -> list[str]:
    """
    【题意】给定字符串 s 和词典 word_dict，在 s 中插入空格，使得每一段都是词典里的单词，
    返回所有可能的分割方案（每种方案用空格拼成一个句子），词典中的单词可以被重复使用；
    不存在任何合法分割时返回空列表。
    【思路】这是 139 题"单词拆分"（只问 True/False）的加强版——不仅要判断能不能拆，还要
    把所有拆法都找出来。朴素回溯的问题在于：不同的分割路径经常会在某个相同的 `start`
    位置"会师"（比如从 0 拆到 4、或从 0 拆到 2 再到 4，之后从位置 4 出发能拼出哪些后续
    句子是完全相同的子问题），如果每次都重新展开这个子问题，会产生大量重复计算，退化成
    指数级；必须在回溯外面加一层"以 start 位置为 key"的记忆化，保证每个 start 位置对应
    的"后续所有句子"只真正计算一次，之后所有路径复用同一份结果——这正是"回溯 + 记忆化"
    模式的核心动机：回溯负责穷举分支，记忆化负责消除分支之间重叠的子问题。这里额外用
    Trie 代替"截取子串再查 set"：从 start 出发沿着 s 的字符往下走 Trie 的边，每走到一个
    带 "#" 标记的节点就说明 s[start:end+1] 是词典里的一个合法单词，比每一步都做字符串
    切片再哈希查找更直接地复用了 Trie 的路径信息。
    【复杂度】时间 O(n^2)（n 个 start 位置，每个位置最多沿 Trie 往前走到字符串末尾，
    且每个位置只展开一次，是记忆化保证的）加上还原句子本身的开销（最坏情况下方案数可能
    指数级，这是问题本身决定的，不是算法的锅）；空间 O(n^2)（记忆化表 + 存储的句子）。
    【易错点】1) 记忆化的 key 必须是 `start` 位置本身，不能把"之前拼接的前缀句子"也
    编码进 key，否则不同路径抵达同一个 start 时会被当成不同的子问题，记忆化形同虚设；
    2) 空字符串结尾（`start == len(s)`）要返回 `[""]` 而不是 `[]`，因为 `[]` 表示
    "无法拆分"，`[""]` 才表示"成功拆完、且没有更多后缀需要拼接"，两者语义完全不同，
    混淆会导致有效结果被误判为拆分失败；3) 拼接单词和后续句子时要注意空格：只有当
    `rest` 非空时才需要在中间插入空格，否则最后一个单词后面会多一个空格。
    """
    n = len(s)
    root: dict = {}
    for w in word_dict:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["#"] = True

    memo: dict[int, list[str]] = {}

    def backtrack(start: int) -> list[str]:
        if start == n:
            return [""]
        if start in memo:
            return memo[start]
        sentences: list[str] = []
        node = root
        for end in range(start, n):
            ch = s[end]
            if ch not in node:
                break
            node = node[ch]
            if "#" in node:
                word = s[start : end + 1]
                for rest in backtrack(end + 1):
                    sentences.append(word if not rest else word + " " + rest)
        memo[start] = sentences
        return sentences

    return backtrack(0)


def concatenated_words(words: list[str]) -> list[str]:
    """
    【题意】给定一个不含重复项的单词列表 words，找出其中所有的"连接词"——一个连接词
    是指完全由列表里**至少两个**较短单词首尾拼接而成的字符串（拼接用到的单词允许重复
    使用同一个短单词）。返回所有连接词组成的列表。
    【思路】先把全部单词插入一棵 Trie（哨兵键 "#" 标记单词结尾）。对每个单词 word 单独
    判断"能否由 Trie 里的其他单词拼接而成"：用一个长度为 `len(word)+1` 的 dp 数组，
    `dp[i]` 表示"word 的前缀 word[:i] 能否被 Trie 里的单词完整拼接覆盖"，`dp[0] = True`
    (空前缀天然成立)。从每个 `dp[i]` 为 True 的位置 i 出发，沿 Trie 往后走 word 的
    字符，每遇到一个 "#" 标记（即 word[i:j] 是词典中的一个单词）就把 `dp[j]` 置 True——
    唯一的例外是 `i == 0 and j == len(word)`，也就是"整个单词本身就是词典里的一个词"这
    种情况必须排除，因为这只是把 word 自己当成唯一的一段，不满足"至少两段"的要求（Trie
    里因为插入了全部单词，天然包含 word 自身，如果不排除这种情况，任何单词都会被误判
    成"连接词"）。
    【复杂度】时间 O(sum(L_i^2))（对每个单词 i，双重循环最坏是长度的平方，Trie 沿途
    字符不匹配会提前 break）；空间 O(sum(L_i))（Trie 大小）。
    【易错点】1) 忘记排除 `i==0 and j==len(word)` 这个"整词自我匹配"的特例，会把每一个
    单词都错误地判定为连接词；2) 直接对每个单词现场再插入/删除 Trie 会让实现变复杂且
    容易出错——更简单稳妥的做法是先把全部单词一次性插入 Trie（含单词自身），只在 dp
    转移那一步排除"自我匹配"这一种情况即可，不需要为每个单词维护"排除自己"的独立 Trie；
    3) 空字符串或超短单词要注意 dp 数组越界，虽然本题约束保证单词长度 >= 1，但下标
    `j + 1` 不能超过 `len(word)`。
    """
    root: dict = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["#"] = True

    def can_form(word: str) -> bool:
        n = len(word)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n):
            if not dp[i]:
                continue
            node = root
            for j in range(i, n):
                ch = word[j]
                if ch not in node:
                    break
                node = node[ch]
                if "#" in node and not (i == 0 and j + 1 == n):
                    dp[j + 1] = True
        return dp[n]

    return [w for w in words if w and can_form(w)]


def word_squares(words: list[str]) -> list[list[str]]:
    """
    【题意】给定一批长度相同的单词（互不重复），找出所有能组成"单词方块"的单词序列：
    一个单词方块由 k 个长度为 k 的单词组成，要求第 i 行读出来的单词和第 i 列读出来的
    单词完全相同（对所有 i 成立）。返回所有满足条件的方块（每个方块内单词顺序重要）。
    【思路】枚举方块第一行的每一个候选单词作为起点，然后逐行往下确定：如果前 row 行
    已经确定，那么第 row 行必须以"已确定的前 row 列，在第 row 行位置的字符"拼成的
    前缀开头（这是"行=列"这个约束反过来推导出的必要条件）。例如已经填了
    ["wall","area"]，要填第三行时，这一行必须以 "l"+"r" = "le"（第 0、1 行第 2 列
    的字符）为前缀。如果每次都线性扫描全部单词判断是否以这个前缀开头，复杂度会随着
    候选单词数线性增长；用 Trie 把"所有以某前缀开头的单词"预先关联到对应节点上（每个
    Trie 节点额外挂一个 `_w` 列表，记录"路径到这个节点的字符串，是哪些完整单词的
    前缀"），把"前缀查询"变成一次 O(前缀长度) 的路径遍历 + 直接读列表，不需要对候选
    单词做任何线性扫描。
    【复杂度】时间 O(候选单词数 * k * 平均前缀命中数)（k 为单词长度，具体上界依赖数据，
    但相比"每行都线性扫描全部单词判断前缀"的朴素回溯已经快很多）；空间 O(所有单词
    字符总数 * k)（每个 Trie 节点的 `_w` 列表最坏可能记录大量单词）。
    【易错点】1) 计算第 row 行需要匹配的前缀时，必须取"已确定的前 row 个单词，各自
    第 row 列的字符"拼起来，而不是随便取某一个已确定单词的某个字符，位置对应关系
    (行号 row 对应列号 row) 很容易搞混；2) 每次尝试一个候选单词后，如果最终没能拼出
    完整方块，必须把这个候选单词从当前方块里弹出（标准回溯"试探-撤销"），否则会污染
    后续同一层的其他候选；3) 结果去重/顺序：方块内部单词的顺序是有意义的（决定了哪行
    对应哪个单词），但方块之间彼此顺序不重要，比较结果时应该用"结果集合"而不是要求
    列表顺序完全一致。
    """
    if not words:
        return []
    root: dict = {}
    for w in words:
        node = root
        node.setdefault("_w", []).append(w)
        for ch in w:
            node = node.setdefault(ch, {})
            node.setdefault("_w", []).append(w)

    def words_with_prefix(prefix: str) -> list[str]:
        node = root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        return node.get("_w", [])

    k = len(words[0])
    squares: list[list[str]] = []
    square: list[str] = []

    def backtrack(row: int) -> None:
        if row == k:
            squares.append(square[:])
            return
        prefix = "".join(square[i][row] for i in range(row))
        for candidate in words_with_prefix(prefix):
            square.append(candidate)
            backtrack(row + 1)
            square.pop()

    for w in words:
        square.append(w)
        backtrack(1)
        square.pop()
    return squares


class FileSystem:
    """
    【题意】设计一个内存文件系统：ls(path) 返回该路径下内容——如果 path 是文件，返回
    只含该文件名的列表；如果是目录，返回目录下所有文件/子目录名（按字典序排列）。
    mkdir(path) 创建目录，路径上缺失的中间目录一并创建。addContentToFile(filePath,
    content) 若文件不存在则创建并写入内容，若已存在则把内容追加到末尾（同时隐式创建
    缺失的中间目录）。readContentFromFile(filePath) 返回文件当前的完整内容。
    【思路】路径本身就是"由 / 分隔的一串名字"，天然是一棵 Trie：每一级目录名对应
    Trie 的一条边，走到某个节点就代表"当前处于这一层目录"。用嵌套 dict 表示每个节点，
    每个节点维护两个独立的子映射——"_dirs"（子目录名 -> 子节点）和"_files"（文件名
    -> 文件内容字符串），之所以分开而不是像 20 章其他 Trie 那样共用一个 children，是
    因为文件和目录在同一层可能重名程度的语义不同（此题保证同层不重名，但文件是"叶子
    终点"，目录还能继续往下走，分开存储让 ls 的"是文件还是目录"判断更直接，不需要额外
    哨兵标记）。mkdir 和 addContentToFile 共享同一段"沿路径逐级 setdefault 创建缺失
    目录"的逻辑；ls 需要先判断路径最后一段是文件名还是目录名——如果是文件，直接返回
    只含这个文件名的单元素列表；如果是目录，返回该目录下 "_dirs" 和 "_files" 两个
    映射的 key 合并后按字典序排序的列表。
    【复杂度】设路径深度为 d：mkdir/addContentToFile/readContentFromFile 均为
    O(d)；ls 是 O(d + m log m)（m 为目录下的直接子项数，排序的开销）；空间 O(所有
    路径名字符总数 + 文件内容总长度)。
    【易错点】1) ls 在路径最后一段既可能是文件也可能是目录时，必须先检查"父节点的
    _files 里是否有这个名字"，命中就直接返回单文件列表，不能继续往 _dirs 里找同名
    目录（本题保证文件名和目录名在同一层不会重复，但判断顺序错了会导致逻辑分支错乱）；
    2) addContentToFile 对已存在的文件是"追加"而不是"覆盖"，必须用
    `.get(name, "") + content` 而不是直接赋值覆盖旧内容；3) 根路径 "/" 分割后是空
    列表（`"/".split("/")` 产生两个空字符串，需要过滤掉），必须正确处理"路径就是
    根目录"这个边界情况，否则 mkdir("/a") 之类的操作会把空字符串当成一层目录名。
    """

    def __init__(self) -> None:
        self.root: dict = {"_dirs": {}, "_files": {}}

    def _split(self, path: str) -> list[str]:
        return [p for p in path.split("/") if p]

    def ls(self, path: str) -> list[str]:
        parts = self._split(path)
        node = self.root
        for i, part in enumerate(parts):
            if part in node["_files"]:
                return [part]
            node = node["_dirs"][part]
        return sorted(list(node["_dirs"].keys()) + list(node["_files"].keys()))

    def mkdir(self, path: str) -> None:
        node = self.root
        for part in self._split(path):
            node = node["_dirs"].setdefault(part, {"_dirs": {}, "_files": {}})

    def addContentToFile(self, filePath: str, content: str) -> None:
        parts = self._split(filePath)
        node = self.root
        for part in parts[:-1]:
            node = node["_dirs"].setdefault(part, {"_dirs": {}, "_files": {}})
        filename = parts[-1]
        node["_files"][filename] = node["_files"].get(filename, "") + content

    def readContentFromFile(self, filePath: str) -> str:
        parts = self._split(filePath)
        node = self.root
        for part in parts[:-1]:
            node = node["_dirs"][part]
        return node["_files"][parts[-1]]


def replace_words(dictionary: list[str], sentence: str) -> str:
    """
    【题意】词典 dictionary 里的每个单词都是一个"词根"，句子 sentence 中的某个单词如果
    以某个词根为前缀，就说这个单词是该词根的"衍生词"，应该被替换成词根本身；如果一个
    衍生词能匹配多个词根，选**最短**的那个词根替换。返回替换后的句子。
    【思路】把所有词根插入一棵 Trie。对句子里的每个单词，从头沿 Trie 往下走：一旦走到
    某个带 "#" 标记的节点，就说明"从单词开头到当前走过的这一段"是一个合法词根——因为
    是**从根开始按顺序走**，第一次遇到 "#" 标记时对应的前缀，天然就是所有匹配词根里
    **最短**的那个（更长的词根需要多走几步字符才能到达），不需要额外比较长度，只要
    找到第一个 "#" 就可以立刻停下来返回这个前缀。如果一直走到单词结尾都没遇到 "#"
    （或者中途 Trie 没有对应字符的边），说明这个单词没有任何词根匹配，原样保留。
    【复杂度】建 Trie O(词典字符总数)；替换每个单词最坏 O(单词长度)，整体 O(句子字符
    总数)；空间 O(词典字符总数)。
    【易错点】1) 必须在"第一次"遇到 "#" 标记时就停止并返回，不能贪心地继续往下走
    试图找更长的词根——题目要求的是最短词根，走得越深匹配到的词根反而越长，这是本题
    最容易理解反的地方；2) 如果单词本身就是某个词根的真前缀但从未形成过完整词根（比如
    单词是 "ca"，词根里只有 "cat"），必须原样保留这个单词，不能因为路径存在就当成
    匹配成功——路径存在只代表"是某个词根的前缀"，不代表"自己就是一个词根"；3) 句子要
    按空格分词、替换后再用空格拼接回去，注意首尾不能引入多余空格。
    """
    root: dict = {}
    for w in dictionary:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["#"] = True

    def find_root(word: str) -> str:
        node = root
        for i, ch in enumerate(word):
            if ch not in node:
                return word
            node = node[ch]
            if "#" in node:
                return word[: i + 1]
        return word

    return " ".join(find_root(w) for w in sentence.split())


def _self_test() -> None:
    assert sorted(word_break_ii("catsanddog", ["cat", "cats", "and", "sand", "dog"])) == sorted(
        ["cats and dog", "cat sand dog"]
    )
    assert sorted(
        word_break_ii(
            "pineapplepenapple", ["apple", "pen", "applepen", "pine", "pineapple"]
        )
    ) == sorted(
        [
            "pine apple pen apple",
            "pineapple pen apple",
            "pine applepen apple",
        ]
    )
    assert word_break_ii("catsandog", ["cats", "dog", "sand", "and", "cat"]) == []

    assert sorted(
        concatenated_words(
            [
                "cat",
                "cats",
                "catsdogcats",
                "dog",
                "dogcatsdog",
                "hippopotamuses",
                "rat",
                "ratcatdogcat",
            ]
        )
    ) == sorted(["catsdogcats", "dogcatsdog", "ratcatdogcat"])
    assert concatenated_words(["cat", "dog", "catdog"]) == ["catdog"]

    squares = word_squares(["area", "lead", "wall", "lady", "ball"])
    assert {tuple(sq) for sq in squares} == {
        ("wall", "area", "lead", "lady"),
        ("ball", "area", "lead", "lady"),
    }

    fs = FileSystem()
    assert fs.ls("/") == []
    fs.mkdir("/a/b/c")
    fs.addContentToFile("/a/b/c/d", "hello")
    assert fs.ls("/") == ["a"]
    assert fs.readContentFromFile("/a/b/c/d") == "hello"
    fs.addContentToFile("/a/b/c/d", " world")
    assert fs.readContentFromFile("/a/b/c/d") == "hello world"
    assert fs.ls("/a/b/c/d") == ["d"]
    assert fs.ls("/a/b") == ["c"]

    assert (
        replace_words(["cat", "bat", "rat"], "the cattle was rattled by the battery")
        == "the cat was rat by the bat"
    )

    print(
        "[PASS] p20_trie_iii: 5/5 题通过 "
        "(单词拆分II/连接词/单词方块/设计内存文件系统/单词替换)"
    )


if __name__ == "__main__":
    _self_test()
