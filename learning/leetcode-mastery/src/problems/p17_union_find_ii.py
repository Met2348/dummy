"""并查集 · 进阶补充（Part II）：不重讲框架，扩大"有向图找环/字符等价类/坐标当节点/
按时间顺序动态连通"这几类并查集变体覆盖面的 6 道题。"""
from __future__ import annotations


class UnionFind:
    """通用并查集：路径压缩 + 按秩合并。本文件独立定义一份，和 p17_union_find.py 里的
    实现同构，供下面 4 道题（冗余连接II、等式方程可满足性、移除石头、彼此熟识的最早时间）
    共用。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True


def find_redundant_directed_connection(edges: list[list[int]]) -> list[int]:
    """
    【题意】n 个节点、n 条边，本该构成一棵"每个节点最多一个父节点"的有根树（树根没有
    父节点），现在多加了一条边。找出删掉之后能让剩余的边重新构成一棵合法有根树的那条边；
    如果有多个候选，返回在输入中最后出现的那一条。

    【思路】这是 Part I 684 题"冗余连接"的有向图版本，但比无向图版本多一层分支——
    无向图里"多余的边"只有一种成因（成环），直接按输入顺序找到第一条"两端已连通"的边
    即可；有向图里"多余的边"有两种可能成因，必须先分情况：
    1) 某个节点的入度为 2（有两条边都指向它）——这说明这两条边里必然有一条是多余的，
       但具体是哪一条，光看入度看不出来，需要"分别尝试去掉其中一条、看剩下的边能不能
       构成合法树"来判断。记这两条候选边为 candidate1（较早出现的那条）、candidate2
       （较晚出现的那条）。
    2) 不存在入度为 2 的节点，但边集里有环——这种情况退化成和无向图版本相同的处理：
       按并查集正常union，第一次遇到"两端已连通"就是那条多余边。
    实现分两遍扫描：第一遍只记录"每个节点的父节点"（`parent_in_tree`），一旦发现某个
    节点已经有父节点、又来了一条边指向它，说明入度为 2，记下 candidate1/candidate2、
    并记住这条边在输入里的位置 `skip_index`（先跳过它，第二遍验证时优先"假设多余的是
    candidate2"）。第二遍对所有边（若存在 candidate2 就跳过 skip_index 那条）做正常的
    并查集 union：如果扫描中提前发现"两端已连通"（出现环），说明就算去掉了 candidate2
    也还有环，真正的多余边必须是 candidate1（因为图中不可能同时有两条独立的冗余边）；
    如果扫完都没有发现环，说明去掉 candidate2 之后其余的边already能构成合法树，那么
    candidate2 就是答案。如果第一遍根本没有发现入度为 2 的节点（candidate1 为 None），
    第二遍找到的"两端已连通"的边就直接是答案（等价于 Part I 684 题的逻辑）。

    【复杂度】时间 O(n·α(n))（两遍线性扫描 + 并查集操作均摊 O(1)）；空间 O(n)。

    【易错点】
    - 最容易漏掉的分支是"入度为 2 但两条候选边都不参与成环"的情况——这种时候真正的
      多余边就是 candidate2（较晚出现的那条），必须在第二遍扫描"完全没有发现环"时也
      能正确返回 candidate2，不能默认成"没找到环就返回空列表"。
    - 判断"要跳过哪条边"应该用边在输入中的位置索引（`skip_index`），而不是拿边的
      值去做列表比较——如果输入里恰好出现内容相同的边（虽然本题限制下不常见），按值
      比较可能跳过错误的一条。
    - 第二遍扫描一旦发现"两端已连通"，要立刻返回 candidate1（而不是继续扫描或者返回
      当前这条边本身）——这是本题区别于无向图版本最核心的一步：出现环就说明
      candidate2 不是真正的冗余边，必须是 candidate1。
    """
    n = len(edges)
    parent_in_tree = [0] * (n + 1)  # parent_in_tree[v] = u 表示存在边 u -> v
    candidate1: list[int] | None = None
    candidate2: list[int] | None = None
    skip_index = -1

    for i, (u, v) in enumerate(edges):
        if parent_in_tree[v] != 0:
            candidate1 = [parent_in_tree[v], v]
            candidate2 = [u, v]
            skip_index = i
        else:
            parent_in_tree[v] = u

    uf = UnionFind(n + 1)  # 节点编号 1..n，下标 0 空置
    for i, (u, v) in enumerate(edges):
        if i == skip_index:
            continue
        if uf.find(u) == uf.find(v):
            return candidate1 if candidate1 is not None else [u, v]
        uf.union(u, v)
    return candidate2 if candidate2 is not None else []


def equations_possible(equations: list[str]) -> bool:
    """
    【题意】给一批形如 "a==b" 或 "a!=b" 的方程（变量都是单个小写字母），判断是否存在
    一种变量赋值方式，能让所有方程同时成立。

    【思路】"==" 是等价关系，天然具有传递性（a==b 且 b==c 则 a==c），这正好是并查集
    要维护的东西；而 "!=" 只是一个"最终检查条件"，不具备传递性（a!=b 且 b!=c 不能推出
    a 和 c 的关系），不能直接参与 union。所以分两遍处理：第一遍只扫描所有 "==" 方程，
    把左右两个字母 union 到同一个集合（26 个字母，并查集大小固定为 26）；第二遍扫描
    所有 "!=" 方程，检查左右两个字母是否被判定为"已经连通"——如果连通了，说明既要求
    它们相等（来自某条或某几条 == 的传递链）又要求它们不相等，产生矛盾，直接返回
    False。两遍都扫完没有矛盾，返回 True。

    【复杂度】时间 O(n)（n 为方程条数，两遍线性扫描 + 并查集操作均摊 O(1)）；
    空间 O(1)（并查集大小固定 26，不随输入增长）。

    【易错点】
    - 必须先处理完全部 "==" 再处理 "!="，不能在同一遍扫描里混着处理——如果 "!=" 出现
      在某条后续会让两者连通的 "==" 之前，交替处理会因为处理顺序而误判（比如
      ["a!=b", "b==a"] 如果边扫边判，会在检查 "a!=b" 时因为此时 a、b 还没被 union
      而错误地放过，正确做法必须是 union 全部完成之后再统一检查）。
    - 字母到并查集下标的映射是 `ord(ch) - ord('a')`，容易在两处（== 处理、!= 处理）
      写得不一致导致下标错位。
    - 形如 "a==a" 或 "a!=a" 这种自等式："==a==a" union 自己和自己是无害的空操作；
      但 "a!=a" 一定矛盾（一个字母不可能不等于自己），这个情况会被
      `find(a) == find(a)` 恒为真自然捕获到，不需要额外特判。
    """
    uf = UnionFind(26)
    for eq in equations:
        if eq[1] == "=":
            a, b = ord(eq[0]) - ord("a"), ord(eq[3]) - ord("a")
            uf.union(a, b)
    for eq in equations:
        if eq[1] == "!":
            a, b = ord(eq[0]) - ord("a"), ord(eq[3]) - ord("a")
            if uf.find(a) == uf.find(b):
                return False
    return True


def smallest_equivalent_string(s1: str, s2: str, base_str: str) -> str:
    """
    【题意】s1、s2 等长，`s1[i]` 和 `s2[i]` 被视为等价字母，等价关系具有传递性。基于
    这批等价关系，把 base_str 里每个字母都替换成"它所在等价类里字典序最小的字母"，
    返回替换后的字符串。

    【思路】26 个字母上的并查集，但 union 的方式和常规写法不同：常规并查集（比如
    p17_union_find.py 里按秩合并的版本）只保证"同一集合内的元素被正确分组"，根节点
    具体是谁并不重要；这道题恰恰要求"代表元必须是字典序最小的那个字母"，如果直接套用
    按秩合并的通用 UnionFind 类，秩大的树会成为根，根不一定是字典序最小的字母，结果就
    可能算错。所以这里手写一份不带秩、只带"代表元恒为较小值"规则的 union：合并两个字母
    的集合时，比较两个根节点的字母大小，让字典序更小的根节点成为新的根（另一个根挂到它
    下面），这样保证了任意时刻"根节点 == 该集合字典序最小的字母"这个不变量。扫描 s1、s2
    对应位置逐对 union 完成后，base_str 里每个字符只需要 find 到它的根，根对应的字母
    就是答案里这个位置该填的字母。

    【复杂度】时间 O(L + m)（L 是 s1/s2 长度，m 是 base_str 长度，并查集操作均摊
    O(1)）；空间 O(1)（并查集大小固定 26）。

    【易错点】
    - 不能直接复用"按秩合并"的通用并查集类——按秩合并只保证效率，不保证根节点是
      字典序最小的元素，必须手写"比较字母大小决定谁当根"的 union 规则。
    - 合并时判断根节点大小要在两个字母各自先 find 到根之后再比较（`find(a)` 和
      `find(b)`），不能直接比较 a、b 本身的字母大小——a、b 可能已经不是各自集合的根。
    - 如果两个字母 find 出来的根已经相同，要直接跳过（无操作），不要误再执行一次挂载，
      否则可能把根挂到自己身上制造环。
    """
    parent = list(range(26))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for c1, c2 in zip(s1, s2):
        union(ord(c1) - ord("a"), ord(c2) - ord("a"))

    return "".join(chr(find(ord(c) - ord("a")) + ord("a")) for c in base_str)


def remove_stones(stones: list[list[int]]) -> int:
    """
    【题意】二维平面上有若干块石头，`stones[i] = [xi, yi]`。每次可以移除一块石头，
    前提是"存在另一块石头和它同行或同列"。求最多能移除多少块石头。

    【思路】"同行或同列就能建立联系"暗示了并查集的用武之地，但这里的并查集节点不是
    石头本身，而是"行号"和"列号"——把每块石头 `[x, y]` 看成"把行号 x 和列号 y 连到
    一起"这一条边（列号需要加一个偏移量，避免和行号的编号范围冲突，两者才能共用同一个
    并查集下标空间）。同一个连通分量里的所有石头，都可以通过"沿着同行/同列的关系链"
    逐一移除到只剩一块——因为只要连通分量里还有 ≥2 块石头，就一定能找到"某块石头和
    另一块同行或同列"这个前提，反复移除，最终每个连通分量恰好能留下一块石头（不能
    再移除，因为剩下这一块行、列都不再和其他石头共享）。所以答案就是
    "石头总数 - 连通分量个数"。连通分量个数直接用同一块石头的行号（或列号，二者
    find 出的根相同）去重统计即可，不需要额外遍历整个并查集数组。

    【复杂度】时间 O(n·α(n))（n 为石头数，每块石头一次 union）；空间 O(n)（并查集
    数组大小取决于坐标值域上限，而不是石头数量本身）。

    【易错点】
    - 并查集节点建在"坐标值"上，而不是"石头的下标"上——如果建在石头下标上，需要
      两两比较石头是否同行同列，退化成 O(n^2)，丢失了"行/列共享"这个结构。
    - 列号必须加偏移量（比如 +10000）再作为并查集下标，否则行号 3 和列号 3 会被
      误认为是同一个节点，把本不相关的石头错误地连到一起。
    - 统计连通分量个数时，要对"每块石头"取它行号（或列号）的根去重，而不是对"并查集
      数组里所有下标"去重——后者会把从未被任何石头用到的行/列也算作独立分量，多算。
    """
    OFFSET = 10000
    uf = UnionFind(2 * OFFSET + 1)
    for x, y in stones:
        uf.union(x, y + OFFSET)
    roots = {uf.find(x) for x, _ in stones}
    return len(stones) - len(roots)


def num_similar_groups(strs: list[str]) -> int:
    """
    【题意】给一组长度相同、且互为彼此字母重排（anagram）的字符串。定义"X 和 Y 相似"
    为"X 恰好交换两个位置的字符就能变成 Y，或者 X 本身就等于 Y"。相似关系具有传递性，
    把 strs 分成若干"相似组"，求组数。

    【思路】判断两个字符串是否相似，直接逐位比较有多少个位置字符不同：因为题目保证了
    所有字符串互为字母重排，只要"不同的位置数恰好是 0 或 2"，就一定能通过交换这两个
    不同的位置让两个串相等（如果不同位置数是 0，本身相同；如果恰好是 2，交换这两个
    位置就能把 X 变成 Y；如果超过 2，无论怎么交换都无法只用一次交换消除 2 个以上的
    差异）。相似关系一旦确定，就是并查集的连通问题：两两比较所有字符串对，只要相似
    就 union 到同一个集合，最终并查集里的连通分量数就是相似组的组数——因为并查集
    自动帮我们处理了"传递性"（A 和 B 相似、B 和 C 相似，即使 A 和 C 本身差异超过 2
    个位置、直接比较不相似，也会因为都 union 到 B 所在的集合而被正确分进同一组）。

    【复杂度】时间 O(n^2 · L)（n 为字符串个数，L 为字符串长度，两两比较是 O(n^2)，
    每次比较 O(L)；并查集操作均摊 O(1)）；空间 O(n)。

    【易错点】
    - 判断相似不能简单地统计"字符集合是否相同"（比如用 Counter 比较）——那只能说明
      两个字符串是彼此的字母重排，不能说明"只需要一次交换"就能互相转化，必须逐位比较
      不同位置的数量。
    - 比较不同位置数时一旦超过 2 就应该提前返回 False（不需要数出精确的差异数），
      是一个小优化，但更重要的是不能把判断写成"不同位置数为偶数就相似"这种错误的
      放宽条件——4 个不同位置也是偶数，但不可能通过一次交换消除。
    - 这个"逐位比较差异数"的判断依赖题目"所有字符串都是彼此字母重排"这个前提；如果
      放到字符集合都不一定相同的字符串上，逐位比较的结论就不成立了。
    """
    def is_similar(a: str, b: str) -> bool:
        diff = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diff += 1
                if diff > 2:
                    return False
        return True

    n = len(strs)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_similar(strs[i], strs[j]):
                uf.union(i, j)
    return uf.count


def earliest_acq(logs: list[list[int]], n: int) -> int:
    """
    【题意】n 个人，logs 中每条 `[timestamp, x, y]` 表示 x 和 y 在 timestamp 这一刻
    成为朋友（朋友关系具有传递性：朋友的朋友也算认识）。求"所有人彼此都认识"这件事
    最早发生在哪个时间戳；如果按所有 log 处理完也无法让所有人连通，返回 -1。

    【思路】"随着时间推移、关系不断累积、要问何时达到某个连通性条件"是并查集"动态陆续
    加边"这类场景的典型应用（呼应 Part I 里"并查集 vs BFS/DFS"那一节的结论：边是随时间
    逐条到达的，不适合每来一条边就重新 BFS 一次全图）。先把 logs 按时间戳升序排序
    （题目不保证输入本身有序），然后按顺序逐条处理：对每条 `[timestamp, x, y]`，
    把 x 和 y union 到一起；每次 union 之后检查并查集的连通分量数 `uf.count` 是否已经
    降到 1——一旦降到 1，说明"刚好在这个时间戳"所有人第一次变得彼此连通，直接返回这个
    timestamp（因为是按时间顺序处理的，第一次达到 count==1 的时刻就是最早的时刻）。
    如果处理完所有 log 连通分量数仍然大于 1，说明这批关系不足以让所有人连通，返回 -1。

    【复杂度】时间 O(m log m + m·α(n))（m 为 log 条数，排序占主导；并查集操作均摊
    O(1)）；空间 O(n + m)（并查集数组 + 排序产生的副本）。

    【易错点】
    - 必须先按时间戳排序——如果直接按输入顺序处理，遇到 logs 本身不是按时间排好序的
      测试数据就会得到错误答案（题目通常不保证输入有序，是很容易被忽略的前提）。
    - 检查"是否已经全部连通"要用 `uf.count == 1`，而不是数"union 成功的次数是否等于
      n-1"——虽然两者在树形结构下等价，但如果同一对人重复出现在多条 log 里（union
      会返回 False、不消耗 count），用"成功次数计数"容易被这种重复边干扰，直接看
      `uf.count` 更直接可靠。
    - 每次 union 之后都要检查是否达到 1（而不是等所有 log 处理完再检查一次），否则
      会错过"最早"这个要求，只能拿到"是否最终连通"的结果而不是最早时间戳。
    """
    uf = UnionFind(n)
    for timestamp, x, y in sorted(logs):
        uf.union(x, y)
        if uf.count == 1:
            return timestamp
    return -1


def _self_test() -> None:
    assert find_redundant_directed_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert find_redundant_directed_connection(
        [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5]]
    ) == [4, 1]

    assert equations_possible(["a==b", "b!=a"]) is False
    assert equations_possible(["b==a", "a==b"]) is True

    assert smallest_equivalent_string("parker", "morris", "parser") == "makkek"
    assert smallest_equivalent_string("hello", "world", "hold") == "hdld"

    assert remove_stones([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]]) == 5
    assert remove_stones([[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]) == 3

    assert num_similar_groups(["tars", "rats", "arts", "star"]) == 2

    assert earliest_acq(
        [
            [20190101, 0, 1],
            [20190104, 3, 4],
            [20190107, 2, 3],
            [20190211, 1, 5],
            [20190224, 2, 4],
            [20190301, 0, 3],
            [20190312, 1, 2],
            [20190322, 4, 5],
        ],
        6,
    ) == 20190301

    print(
        "[PASS] p17_union_find_ii: 6 题"
        "（冗余连接II/等式方程的可满足性/最小的等效字符串/移除最多的石头/"
        "相似字符串组/彼此熟识的最早时间）全部通过"
    )


if __name__ == "__main__":
    _self_test()
