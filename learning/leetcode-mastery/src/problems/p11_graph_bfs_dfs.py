"""图 BFS/DFS 专题：岛屿数量 / 克隆图 / 课程表 / 课程表 II / 腐烂的橘子 / 单词接龙。

选择依据：BFS 天然按"层"扩展，适合求最短路/最少步数；DFS 沿着一条路径走到底再回退，
适合求连通性、穷举所有路径。图题的第一步永远是想清楚"怎么表示这张图"——可能是显式的
邻接表/邻接矩阵/网格，也可能是隐式的（比如单词接龙里，每个单词是一个节点，"只差一个
字母"就是一条边，这张图从来没有被真正构造出来，而是在 BFS 过程中现算现用）。
"""
from __future__ import annotations

from collections import defaultdict, deque


# ── LC200 岛屿数量 ───────────────────────────────────────────────────────
def num_islands(grid: list[list[str]]) -> int:
    """
    【题意】给一个由 '1'（陆地）和 '0'（水）组成的二维网格，岛屿由水平/竖直相邻的陆地
    连接而成（网格四周视为被水包围），返回岛屿（连通块）的数量。
    【思路】这是"求连通分量个数"的模板题：按行列扫描每个格子，一旦遇到还没访问过的
    陆地，就说明发现了一座新岛屿（计数 +1），随即用 DFS 把这座岛屿所有相连的陆地全部
    标记为已访问——这里偷懒直接把 grid[r][c] 原地改成 '0'（俗称"淹没"），比额外开一个
    visited 集合更省内存；后续扫描再碰到这些格子时，会被当成水直接跳过，不会被重复
    计数。
    【复杂度】时间 O(R*C)（每个格子最多被访问常数次）；空间最坏 O(R*C)（整个网格都是
    陆地连成一片时的递归栈深度）。
    【易错点】1) 忘记"淹没"已访问过的陆地（或者用了 visited 集合却忘记及时标记），
    导致同一座岛屿的不同格子被反复当成新岛屿起点，重复计数；2) 直接在传入的 grid 上
    原地修改会破坏调用方的原始数据，如果后续还要用原始网格，记得先传入拷贝。
    """
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])

    def sink(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"
        sink(r + 1, c)
        sink(r - 1, c)
        sink(r, c + 1)
        sink(r, c - 1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                sink(r, c)
    return count


# ── LC133 克隆图 ─────────────────────────────────────────────────────────
class GraphNode:
    """无向图节点：val 是编号，neighbors 是相邻节点的引用列表。"""

    def __init__(self, val: int = 0, neighbors: list["GraphNode"] | None = None):
        self.val = val
        self.neighbors: list[GraphNode] = neighbors if neighbors is not None else []


def build_graph(adj_list: list[list[int]]) -> "GraphNode | None":
    """测试辅助函数：把 LeetCode 风格的 adjList（1-indexed）构造成真正的节点/引用图，
    返回编号为 1 的节点作为入口。"""
    if not adj_list:
        return None
    nodes = {i + 1: GraphNode(i + 1) for i in range(len(adj_list))}
    for i, neighbors in enumerate(adj_list, start=1):
        nodes[i].neighbors = [nodes[j] for j in neighbors]
    return nodes[1]


def graph_to_adj(node: "GraphNode | None") -> list[list[int]]:
    """测试辅助函数：从 node 出发遍历整张图，按 val 从小到大转换回 adjList 形式
    （每个节点的邻居 val 列表也排序），方便比较两张图结构是否相同。"""
    if node is None:
        return []
    seen: dict[int, GraphNode] = {}
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.val in seen:
            continue
        seen[cur.val] = cur
        for nb in cur.neighbors:
            if nb.val not in seen:
                stack.append(nb)
    return [sorted(nb.val for nb in seen[v].neighbors) for v in sorted(seen)]


def clone_graph(node: "GraphNode | None") -> "GraphNode | None":
    """
    【题意】给定无向连通图中某一个节点的引用，返回整张图的深拷贝（克隆出全部节点和
    全部边，克隆图和原图不能共享任何节点对象）。
    【思路】图和树最大的区别是"可能存在环"，如果直接朴素地"每访问一个节点就递归它的
    邻居"，遇到环会无限递归下去。解法是用一个哈希表记录"原节点 val -> 已经克隆出的
    节点"：DFS 到一个节点时先查这张表，如果已经克隆过，直接返回克隆节点的引用而不是
    继续往下递归——这一步"先查表再决定要不要递归"正是"有环图遍历"和"树遍历"最核心的
    区别，也天然防止了死循环和重复克隆。
    【复杂度】时间 O(V+E)（每个节点、每条边各处理一次）；空间 O(V)（哈希表 + 递归栈）。
    【易错点】1) 忘记先查哈希表就直接对邻居递归，遇到环会导致无限递归、栈溢出；
    2) 新建克隆节点时，如果 neighbors 列表直接引用了原图节点（而不是递归得到的克隆
    节点），会导致"克隆图"和"原图"共享同一批节点对象，根本没有做到深拷贝。
    """
    if node is None:
        return None
    cloned: dict[int, GraphNode] = {}

    def dfs(cur: GraphNode) -> GraphNode:
        if cur.val in cloned:
            return cloned[cur.val]
        copy = GraphNode(cur.val)
        cloned[cur.val] = copy
        copy.neighbors = [dfs(nb) for nb in cur.neighbors]
        return copy

    return dfs(node)


# ── LC207 课程表 ─────────────────────────────────────────────────────────
def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """
    【题意】一共有 num_courses 门课（编号 0..num_courses-1），prerequisites[i]=[a,b]
    表示"想学课程 a 必须先学完课程 b"；判断是否存在一种可以学完所有课程的顺序。
    【思路】把"课程"看成有向图的节点，"先修关系"看成有向边 b->a（先学 b 才能学 a）。
    "能否学完所有课程"等价于"这张有向图里有没有环"——如果 a 依赖 b、b 依赖 c、c 又
    依赖 a，这三门课谁都没法作为"第一门"开始学。用 Kahn 算法（BFS 拓扑排序）判环：
    先算出每个节点的入度（有多少门课是它的直接前提），把入度为 0 的课程（没有任何
    未满足的先修要求，可以立刻学）全部入队；每次从队列取出一门课当作"学完"，把它
    指向的所有后继课程入度减一，如果某门课的入度减到 0，说明它的前提也都学完了，
    随即入队。最后比较"被处理过的课程总数"是否等于 num_courses——如果图里有环，
    环上的课程入度永远减不到 0，永远不会入队，被处理数就会小于总数。
    【复杂度】时间 O(V+E)（V=num_courses，E=len(prerequisites)）；空间 O(V+E)。
    【易错点】1) 建图方向搞反（建成 a->b 而不是 b->a），会得到完全相反甚至无意义的
    拓扑关系；2) 误以为"只统计初始入度为 0 的课程数"就能判断有没有环，其实必须统计
    "整个 BFS 过程里处理过的课程总数"才能正确识别环的存在。
    """
    graph: dict[int, list[int]] = defaultdict(list)
    indegree = [0] * num_courses
    for a, b in prerequisites:  # b -> a：先学 b 才能学 a
        graph[b].append(a)
        indegree[a] += 1

    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    visited = 0
    while q:
        cur = q.popleft()
        visited += 1
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return visited == num_courses


# ── LC210 课程表 II ──────────────────────────────────────────────────────
def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    【题意】在 can_finish 判断"能否学完所有课程"的基础上，如果可以学完，返回一种具体
    的合法学习顺序（拓扑序）；如果不可能（存在环），返回空列表。
    【思路】和 can_finish 是同一套 Kahn 算法，唯一区别是不只统计处理过的课程数量，而是
    把每次出队的课程按顺序记录下来——BFS 出队的顺序天然就是一个合法拓扑序，因为一门课
    出队时，它所有的前提课程一定已经出队过（不然它的入度不会被减到 0），这就保证了
    "前提永远排在被依赖的课程之前"。
    【复杂度】时间 O(V+E)；空间 O(V+E)。
    【易错点】1) 误以为拓扑序是唯一的——当图里同时存在多个入度为 0 的课程时，拓扑序
    可以有多种合法结果，校验答案时不能死板地比较某一个具体序列，而要检查"每条依赖边
    在结果里的相对先后顺序是否满足"；2) 忘记处理"因为存在环导致 BFS 提前结束"的
    情况，返回了一个不完整的顺序而不是规定的空列表。
    """
    graph: dict[int, list[int]] = defaultdict(list)
    indegree = [0] * num_courses
    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1

    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    order: list[int] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return order if len(order) == num_courses else []


# ── LC994 腐烂的橘子 ─────────────────────────────────────────────────────
def oranges_rotting(grid: list[list[int]]) -> int:
    """
    【题意】网格里 0=空格、1=新鲜橘子、2=腐烂橘子；每过一分钟，每个腐烂橘子会让它
    上下左右相邻的新鲜橘子也变腐烂；返回让所有新鲜橘子都腐烂所需的最少分钟数，如果
    有橘子永远无法被感染到，返回 -1。
    【思路】这是"多源 BFS"的经典场景：不是从一个起点向外扩散，而是把所有初始腐烂的
    橘子同时作为第 0 层一起入队，再统一向外扩散——因为它们是"同时"开始感染各自邻居
    的，BFS 按层扩展的过程天然对应"每过一分钟"这个时间概念，每扩展一层就是过去了
    一分钟。
    【复杂度】时间 O(R*C)；空间 O(R*C)（队列最坏情况下要存下所有格子）。
    【易错点】1) 只从某一个腐烂橘子开始做单源 BFS，而不是把所有初始腐烂橘子同时
    入队，会得到偏大甚至错误的分钟数；2) 忘记特判"一开始就没有新鲜橘子"这种情况
    （这里用 fresh==0 直接返回 0），否则容易在 BFS 循环边界上多算或少算一分钟。
    """
    rows, cols = len(grid), len(grid[0])
    q: deque[tuple[int, int]] = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                q.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    minutes = 0
    while q and fresh > 0:
        minutes += 1
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    q.append((nr, nc))
    return minutes if fresh == 0 else -1


# ── LC127 单词接龙 ───────────────────────────────────────────────────────
def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    """
    【题意】给定起始单词 begin_word、目标单词 end_word 和单词表 word_list，每一步只能
    改变单词里的一个字母，且变换出的中间单词必须存在于 word_list 中；求从 begin_word
    到 end_word 最短变换序列的单词个数（含首尾），如果无法到达返回 0。
    【思路】这题最反直觉的地方是"图在哪里"——题目从没显式给出节点和边，但可以把
    word_list 里每个单词看成图上的一个节点，"只改一个字母就能互相到达"看成一条边，
    问题就变成了这张隐式图上的"无权最短路"。无权图的最短路正是 BFS 的拿手戏：BFS
    逐层扩展，第一次到达 end_word 时所走过的层数，一定就是最短步数——因为 BFS 保证
    同一层内所有节点距起点的步数相同，不可能存在一条更短、却还没被发现的路径。具体
    展开"边"的方式：对当前单词的每一位，尝试替换成 a-z 里的其他 25 个字母，如果替换
    后的新单词在 word_list 对应的集合里且没访问过，就是一条合法的边，可以入队继续
    往外扩展。
    【复杂度】时间 O(M^2*N)（M=单词长度，N=单词表大小；每个出队单词要在 M 个位置各
    尝试 25 种替换，每次替换构造新字符串是 O(M)）；空间 O(M*N)（word set + visited）。
    【易错点】1) 忘记把 word_list 转成 set，导致"新单词是否合法"这一步退化成对列表
    的线性查找，整体复杂度大幅上升；2) 忘记把 begin_word 本身加入 visited（即使它
    可能不在 word_list 里），导致后续搜索绕回 begin_word 造成重复甚至无限扩展；
    3) 忘记特判 end_word 不在 word_list 里的情况——按题意 end_word 也必须是一个
    "合法的中间/终点单词"，如果它本身都不在词表里，就不可能是任何变换序列的终点，
    应直接返回 0。
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    q: deque[tuple[str, int]] = deque([(begin_word, 1)])
    visited = {begin_word}
    while q:
        word, steps = q.popleft()
        if word == end_word:
            return steps
        for i in range(len(word)):
            for ch in alphabet:
                if ch == word[i]:
                    continue
                nxt = word[:i] + ch + word[i + 1 :]
                if nxt in word_set and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, steps + 1))
    return 0


def _self_test() -> None:
    grid1 = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
    ]
    assert num_islands([row[:] for row in grid1]) == 1
    grid2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    assert num_islands([row[:] for row in grid2]) == 3

    adj_list = [[2, 4], [1, 3], [2, 4], [1, 3]]
    original = build_graph(adj_list)
    clone = clone_graph(original)
    assert clone is not original
    assert graph_to_adj(clone) == graph_to_adj(original)
    assert graph_to_adj(original) == [sorted(nbrs) for nbrs in adj_list]

    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False

    prereqs = [[1, 0], [2, 0], [3, 1], [3, 2]]
    order = find_order(4, prereqs)
    assert len(order) == 4
    for a, b in prereqs:
        assert order.index(b) < order.index(a)
    assert find_order(2, [[1, 0], [0, 1]]) == []

    assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
    assert oranges_rotting([[0, 2]]) == 0

    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0

    print(
        "[PASS] p11_graph_bfs_dfs: 6 道图 BFS/DFS 题"
        "（岛屿数量/克隆图/课程表/课程表II/腐烂的橘子/单词接龙）全部通过"
    )


if __name__ == "__main__":
    _self_test()
