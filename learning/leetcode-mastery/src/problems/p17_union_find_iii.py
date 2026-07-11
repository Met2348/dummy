"""并查集 · Phase 3 竞赛级补充（Part III）：把并查集用到"配对关系建图求最少交换"
（情侣牵手）、"下标分组重排"与"字符串同义词组合展开"两道并查集+其他技巧结合的中等题，
以及两道 Frontier Lab 面试里出现频率很高的并查集难题——最小生成树的关键边/伪关键边
判定（1489）、离线按限制排序处理的连通性查询（1697），共 5 道题。"""
from __future__ import annotations

from itertools import product


class UnionFind:
    """通用并查集：路径压缩 + 按秩合并。和 p17_union_find.py / p17_union_find_ii.py
    里的实现同构，本文件独立定义一份供下面 5 道题共用。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
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


def min_swaps_couples(row: list[int]) -> int:
    """
    【题意】2n 个座位排成一排，`row[i]` 是坐在第 i 个座位的人的编号。情侣编号成对：
    (0,1)、(2,3)……每次交换可以选任意两个人互换座位，求让每对情侣都相邻而坐所需的
    最少交换次数。

    【思路】把"第 i 对情侣"看成并查集里的一个节点（编号 `person // 2`），扫描座位
    数组，每两个相邻座位 `(row[2i], row[2i+1])` 实际坐着的两个人，各自属于某一对
    情侣（`row[2i] // 2`、`row[2i+1] // 2`），把这两个"情侣编号"union 到一起——这
    建立了一张"哪些情侣被迫挤在了同一组座位段里、需要互相腾挪"的关系图。如果这两个
    座位上坐的恰好是同一对情侣，union 是无效操作（两者已经是同一个节点），不消耗
    连通分量；如果坐的是不同情侣的两个人，说明这一组座位段和另一组座位段"纠缠"在
    一起，必须通过若干次交换才能解开。

    这里的关键洞察是"一个连通分量需要的最少交换次数 = 分量大小 - 1"：如果 k 对情侣
    因为座位安排彼此纠缠成一个连通分量（形成一个"置换环"），恰好需要 k-1 次交换才能
    把这个环拆解成 k 对各自独立相邻的情侣（这是排列环分解的标准结论：把一个长度为 k
    的置换环还原成恒等置换，需要 k-1 次对换）。对所有连通分量求和，等价于
    "情侣总对数 n - 连通分量个数"，因为 `sum(component_size - 1) = n - components`。

    【复杂度】时间 O(n·α(n))（n 为情侣对数，一次线性扫描 + 并查集操作均摊 O(1)）；
    空间 O(n)。

    【易错点】
    - 并查集节点建在"情侣对编号"（`person // 2`）上，而不是"座位编号"或"人的编号"
      上——如果直接对人的编号建并查集，无法直接用"分量数"得到答案，需要额外一层
      映射，容易把问题复杂化。
    - 答案是 `n - uf.count`（分量数），不是"union 成功的次数"——虽然在这道题里
      "座位对数"恰好等于"情侣对数" n，union 成功次数和 `n - uf.count` 数值上相等，
      但从"分量数"出发理解更直接地对应"每个连通分量需要 分量大小-1 次交换"这个
      核心结论，不容易在稍加变形的题目上出错。
    - 容易忘记"两个座位坐的恰好是同一对情侣"这种已经满足条件的情况也要 union（虽然
      union 会返回 False、不消耗分量），如果跳过这一对不处理，不影响最终分量计数
      的正确性，但会让代码逻辑不完整、不好复用到"顺便统计每对是否已就位"这类扩展
      需求上。
    """
    n = len(row) // 2
    uf = UnionFind(n)
    for i in range(0, len(row), 2):
        uf.union(row[i] // 2, row[i + 1] // 2)
    return n - uf.count


def smallest_string_with_swaps(s: str, pairs: list[list[int]]) -> str:
    """
    【题意】给字符串 s 和一批下标对 pairs，`pairs[i] = [a, b]` 表示"下标 a 和下标 b
    上的字符可以任意次交换"。求能变成的字典序最小的字符串。

    【思路】"可以任意次交换"意味着 pairs 描述的下标关系具有传递性——如果 (0,1) 和
    (1,2) 都可以交换，那么下标 0、1、2 上的字符实际上可以**任意重排**（通过若干次
    两两交换实现任意排列），这正是并查集要处理的"连通分量内任意重排"这件事：先用
    并查集把所有相互可达的下标分到同一组，再对每一组内部单独处理——取出这组下标
    对应的字符，把字符排序、把下标排序，字典序最小的字符依次填回字典序最小的下标，
    这样填法能保证这一组内部达到的字典序最小（一个连通分量内部可以任意排列，排序
    后一一对应就是这个子问题的最优解），各组互不影响（不同组之间没有交换关系），
    所以对每组分别贪心即可得到全局最优。

    【复杂度】时间 O(n log n)（并查集操作均摊 O(1)；主要开销是对每个分组内的下标和
    字符分别排序，最坏情况下退化成对整个字符串排序）；空间 O(n)。

    【易错点】
    - 分组之后必须同时对"组内下标"和"组内字符"分别排序，再按位置一一对应填回——
      只排序字符、不排序下标（或者反过来）会导致字符填错位置。
    - 按并查集根节点分组时，用 `defaultdict(list)` 收集"根 -> 下标列表"，不能直接
      按"下标在 pairs 里第一次出现的位置"分组，否则会把本该属于同一分量、但没有
      直接出现在同一条 pair 里的下标错误地拆散。
    - 结果要用 `list(s)` 之类的可变结构去填、最后 `"".join(...)`，直接对字符串
      本身按下标赋值在 Python 里是不允许的（字符串不可变），容易忘记这一层转换。
    """
    n = len(s)
    uf = UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    result = list(s)
    for indices in groups.values():
        indices.sort()
        chars = sorted(result[i] for i in indices)
        for idx, ch in zip(indices, chars):
            result[idx] = ch
    return "".join(result)


def generate_sentences(synonyms: list[list[str]], text: str) -> list[str]:
    """
    【题意】给一批同义词对 synonyms（同义关系具有传递性），以及一句句子 text（单词
    以空格分隔）。把 text 里每个出现在某个同义词组里的单词，替换成该组里任意一个
    同义词，返回所有可能生成的句子，按字典序排序。

    【思路】同义关系的传递性（"happy 等价 joy、joy 等价 cheerful"意味着"happy、joy、
    cheerful"三者互相等价）正是并查集要维护的"等价类"，和 Part II 的"等式方程的
    可满足性"、"按字典序排列最小的等效字符串"是同一类"用并查集求传递闭包分组"的
    应用，区别在于这里的并查集节点是字符串（单词）而不是整数或单个字符，需要先用
    一个 `word -> id` 的字典把单词映射成整数下标才能喂给并查集。求出每个单词所在
    的等价类之后，对每个类内的单词排序，建一张"单词 -> 该单词所在组的排序后单词
    列表"的映射表；对 text 里每个单词，如果它出现在某个同义词组里，候选替换项就是
    这个组的单词列表，否则候选只有它自己。最终答案是"每个位置从各自候选列表里选一
    个"的笛卡尔积——`itertools.product` 直接枚举了这个"对每个单词位置做一次独立选择"
    的过程，本质上等价于教科书写法里"逐词回溯、每层递归尝试当前词的所有候选、拼完
    整句再收集"的 DFS，只是用标准库的笛卡尔积表达更简洁。

    【复杂度】时间 O(∏|group_i|)（最坏情况下是所有候选组合数之积，即指数级——这是
    本题问题规模本身就很小（synonyms 最多 10 条、text 最多 10 个单词）才能接受暴力
    枚举的原因）；空间同样是 O(结果数量 × 句子长度)（保存所有生成的句子）。

    【易错点】
    - 单词到并查集下标的映射必须用一个统一的字典（比如先收集 synonyms 里出现过的
      所有不同单词、分配连续整数 id），不能假设单词已经是连续整数，这是本题和纯
      整数/字符并查集（如 26 个字母）最大的实现差异。
    - text 里的单词如果压根没出现在任何 synonyms 里，候选列表就是"只有它自己"，
      不能因为它不在并查集映射表里就报错或跳过。
    - 最终返回结果必须整体按字典序排序（`sorted(...)`），`itertools.product` 本身
      不保证输出顺序就是字典序（它按各候选列表的原始顺序做嵌套遍历），必须显式
      排序，即使每个组内部的候选列表已经各自有序，笛卡尔积展开后整体顺序仍然
      需要重新排序才能保证正确。
    """
    word_id: dict[str, int] = {}

    def get_id(word: str) -> int:
        if word not in word_id:
            word_id[word] = len(word_id)
        return word_id[word]

    for a, b in synonyms:
        get_id(a)
        get_id(b)  # 先扫一遍，把所有出现过的单词分配好连续整数 id

    uf = UnionFind(len(word_id))
    for a, b in synonyms:
        uf.union(word_id[a], word_id[b])

    groups: dict[int, list[str]] = {}
    for word, wid in word_id.items():
        groups.setdefault(uf.find(wid), []).append(word)
    for group in groups.values():
        group.sort()

    def candidates(word: str) -> list[str]:
        if word in word_id:
            return groups[uf.find(word_id[word])]
        return [word]

    options = [candidates(w) for w in text.split()]
    sentences = [" ".join(combo) for combo in product(*options)]
    return sorted(sentences)


def find_critical_and_pseudo_edges(n: int, edges: list[list[int]]) -> list[list[int]]:
    """
    【题意】n 个节点、边集 edges（`edges[i] = [u, v, weight]`，边有原始下标）构成一张
    带权连通无向图。定义：删除某条边后，如果图的最小生成树（MST）权值必然增大（或者
    图变得不连通），这条边就是"关键边"；如果存在某个 MST 包含这条边、也存在另一个
    MST 不包含它（即这条边"可有可无"，但强制选上也不影响能否达到最小权值），这条边
    就是"伪关键边"。返回 `[关键边下标列表, 伪关键边下标列表]`。

    【思路】这是并查集在"最小生成树"场景里最经典的应用——Kruskal 算法本身就是"边按
    权重排序、贪心 union、跳过成环边"的并查集流程，而"关键边/伪关键边"的判定，本质
    是对同一套 Kruskal 流程做"删除某条边重新跑一遍"和"强制先选某条边再跑一遍"这两种
    变体，分别对应两条判定规则：

    1. **关键边的判定——"删除这条边后 MST 权值必然变大"**：如果一条边是关键边，说明
       它是所有可能的 MST 都必须包含的边（唯一能连接某个"割"的最小权边，没有替代
       选项）。验证方法就是字面意思：假设不能用这条边（在 Kruskal 流程里直接跳过
       它），重新贪心构造一遍 MST，如果新的 MST 权值比原始最小权值更大（或者图变得
       不连通，视为权值无穷大），说明少了这条边就无法达到原来的最优解，它是关键边。

    2. **伪关键边的判定——"强制先选这条边，MST 权值不变"**：如果一条边不是关键边
       （删掉它 MST 权值不变，说明存在不含它的最优方案），还需要进一步判断它是不是
       "至少有一个 MST 会用到它"（伪关键边）还是"任何最优 MST 都不会用到它"（普通
       边，两个列表都不进）。验证方法是反过来强制：在 Kruskal 流程开始前，先无条件
       把这条边选上（对应 union 这条边的两个端点、累加它的权重），再对其余边正常
       跑贪心，如果这样跑出来的总权值恰好等于原始的最小权值，说明"选上这条边"完全
       不影响最优性，存在包含它的 MST，因此它是伪关键边。

    实现上先算出一次不加任何限制的基准 MST 权值 `base`，再对每条边分别调用同一个
    `mst_weight(skip, force)` 辅助函数做上述两种变体测试——这个辅助函数是本题的
    核心复用点：`skip` 参数实现判定 1（关键边），`force` 参数实现判定 2（伪关键边），
    两者共享同一套"排序边、按并查集贪心加边、数已选边数是否达到 n-1"的 Kruskal 主
    循环，只是在跳过/强制这一个细节上分叉。

    【复杂度】时间 O(m² · α(n))（m 为边数，对每条边都要重新跑一次 O(m·α(n)) 的
    Kruskal，m ≤ 200 时完全可接受）；空间 O(n + m)。

    【易错点】
    - 关键边判定必须用"权值是否变大"而不是"图是否仍连通"来判断——图不连通只是
      "权值变大"的一种极端情况（可以视为权值变成无穷大），如果只检查连通性会漏掉
      "图仍然连通、但用了更贵的替代边导致总权值变大"这种更常见的关键边情形。
    - 伪关键边判定必须是"先无条件强制选上这条边，再对其余边正常贪心"，而不是
      "如果这条边不参与成环就认为它可能进入某个 MST"——只有真正跑一遍强制选它的
      完整 Kruskal 流程、比较总权值是否等于基准最优值，才能准确判断它是否真的能
      出现在某个最优 MST 里。
    - 判断关键边和伪关键边是"先后互斥"的关系（先判是否关键边，不是关键边才继续判
      是否伪关键边），如果颠倒判断顺序或者两个条件都独立判断而不是 elif，可能把
      一条边同时归进两个列表，或者因为"强制选上关键边后权值当然不变"这一天然
      性质，把关键边也错误地放进伪关键边列表。
    """
    m = len(edges)
    order = sorted(range(m), key=lambda i: edges[i][2])

    def mst_weight(skip: int = -1, force: int = -1) -> float:
        uf = UnionFind(n)
        weight = 0
        edges_used = 0
        if force != -1:
            u, v, w = edges[force]
            uf.union(u, v)
            weight += w
            edges_used += 1
        for i in order:
            if i == skip or i == force:
                continue
            u, v, w = edges[i]
            if uf.union(u, v):
                weight += w
                edges_used += 1
        return weight if edges_used == n - 1 else float("inf")

    base = mst_weight()
    critical: list[int] = []
    pseudo: list[int] = []
    for i in range(m):
        if mst_weight(skip=i) > base:
            critical.append(i)
        elif mst_weight(force=i) == base:
            pseudo.append(i)
    return [critical, pseudo]


def distance_limited_paths_exist(
    n: int, edge_list: list[list[int]], queries: list[list[int]]
) -> list[bool]:
    """
    【题意】n 个节点的无向图，`edge_list[i] = [u, v, dis]` 表示 u、v 之间有一条距离
    为 dis 的边（同一对节点间可能有多条边）。给一批查询 `queries[j] = [p, q, limit]`，
    对每条查询判断：是否存在一条从 p 到 q 的路径，使得路径上**每一条边**的距离都
    严格小于 limit。返回每条查询的布尔答案。

    【思路】如果对每条查询都独立地"只用距离 < limit 的边建一张子图、再 BFS/DFS 判断
    p、q 是否连通"，每条查询最坏要花 O(边数) 重建 + 遍历一次图，查询多的话总复杂度
    是 O(查询数 × 边数)，代价很高。这里的关键突破口是一条**单调性**：如果 p、q 在
    "只用距离 < L1 的边"这张子图里连通，那么在"只用距离 < L2 的边"（L2 > L1）这张
    包含了前者所有边、还可能更多的子图里，p、q 必然仍然连通——连通性只会随着可用
    边变多而增加，不会减少。这条单调性意味着，如果把所有查询按 limit 从小到大排序、
    依次处理，那么"这一批查询已经用到的边"可以在处理下一条（limit 更大的）查询时
    继续复用、只需要把新解锁的边（权重落在 [上一次的 limit, 这一次的 limit) 之间的
    边）追加进并查集，而不需要每次都从空并查集重新建图——这是并查集在"离线处理、
    按某个单调量排序后增量维护"这一模式下的经典应用（呼应 Part I"并查集 vs BFS/DFS"
    小节的结论：边是逐步解锁的，用并查集增量维护比每次重新遍历全图更高效）。

    具体实现：把 edge_list 按距离升序排序，把 queries 的下标按 limit 升序排序（必须
    额外保留原始下标，因为答案要按查询的原始顺序返回）；用一个指针 `ptr` 追踪"已经
    加入并查集的边"，对每条（按 limit 排好序处理的）查询，先把所有距离严格小于当前
    limit、且还没加入过的边都 union 进并查集（指针只会前进、不会回退，这是"排序后
    离线处理"能达到线性总扫描量的关键），再直接用 `find(p) == find(q)` 回答这条
    查询。

    【复杂度】时间 O((m + q) log(m + q) + (m + q)·α(n))（m 为边数，q 为查询数，
    排序边和排序查询各占主导，并查集操作总量摊还是线性的）；空间 O(m + q + n)。

    【易错点】
    - 查询必须按 limit 排序、但答案必须按查询的**原始下标**填回结果数组——如果直接
      按排序后的顺序输出会导致答案和查询错位，必须像本实现一样用
      `sorted(range(len(queries)), key=...)` 保留原始下标。
    - 追加边的指针 `ptr` 一旦前进就不会回退（因为查询已经按 limit 升序处理，之前
      解锁的边对后续 limit 更大的查询仍然有效），如果每条查询都重置指针从头开始
      扫描，就退化成了 O(查询数 × 边数) 的暴力做法，丢失了排序带来的复用优势。
    - 边的加入条件是"距离严格小于 limit"（`< limit`，不是 `<= limit`）——因为题目
      要求路径上每条边的距离**严格小于** limit，如果误用小于等于会把恰好等于
      limit 的边也算作可用边，导致某些查询被错误地判定为连通。
    """
    edges_sorted = sorted(edge_list, key=lambda e: e[2])
    query_order = sorted(range(len(queries)), key=lambda i: queries[i][2])

    uf = UnionFind(n)
    answer = [False] * len(queries)
    ptr = 0
    for qi in query_order:
        p, q, limit = queries[qi]
        while ptr < len(edges_sorted) and edges_sorted[ptr][2] < limit:
            u, v, _ = edges_sorted[ptr]
            uf.union(u, v)
            ptr += 1
        answer[qi] = uf.find(p) == uf.find(q)
    return answer


def _self_test() -> None:
    assert min_swaps_couples([0, 2, 1, 3]) == 1
    assert min_swaps_couples([3, 2, 0, 1]) == 0

    assert smallest_string_with_swaps("dcab", [[0, 3], [1, 2]]) == "bacd"
    assert smallest_string_with_swaps("dcab", [[0, 3], [1, 2], [0, 2]]) == "abcd"
    assert smallest_string_with_swaps("cba", [[0, 1], [1, 2]]) == "abc"

    assert generate_sentences(
        [["happy", "joy"], ["sad", "sorrow"], ["joy", "cheerful"]],
        "I am happy today but was sad yesterday",
    ) == [
        "I am cheerful today but was sad yesterday",
        "I am cheerful today but was sorrow yesterday",
        "I am happy today but was sad yesterday",
        "I am happy today but was sorrow yesterday",
        "I am joy today but was sad yesterday",
        "I am joy today but was sorrow yesterday",
    ]
    assert generate_sentences(
        [["happy", "joy"], ["cheerful", "glad"]],
        "I am happy today but was sad yesterday",
    ) == [
        "I am happy today but was sad yesterday",
        "I am joy today but was sad yesterday",
    ]

    assert find_critical_and_pseudo_edges(
        5,
        [[0, 1, 1], [1, 2, 1], [2, 3, 2], [0, 3, 2], [0, 4, 3], [3, 4, 3], [1, 4, 6]],
    ) == [[0, 1], [2, 3, 4, 5]]
    assert find_critical_and_pseudo_edges(
        4, [[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 3, 1]]
    ) == [[], [0, 1, 2, 3]]

    assert distance_limited_paths_exist(
        3, [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]], [[0, 1, 2], [0, 2, 5]]
    ) == [False, True]
    assert distance_limited_paths_exist(
        5, [[0, 1, 10], [1, 2, 5], [2, 3, 9], [3, 4, 13]], [[0, 4, 14], [1, 4, 13]]
    ) == [True, False]

    print(
        "[PASS] p17_union_find_iii: 5 题"
        "（情侣牵手/交换字符串中的元素/近义词句子/"
        "找到最小生成树里的关键边和伪关键边/检查是否存在有效路径-带边长限制）全部通过"
    )


if __name__ == "__main__":
    _self_test()
