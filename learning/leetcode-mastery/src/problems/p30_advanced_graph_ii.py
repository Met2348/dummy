"""分类 30（Phase 3 竞赛级新分类）：高级图论 II —— Tarjan 找桥/割点/强连通分量、
LCA 倍增法、网络流。22 类的"高级图论"解决的是"带权最短路/最小生成树/二分图"；这一类
更硬核——研究图本身的结构性质（割点、桥、强连通分量），把"树上路径查询"从 O(n) 暴力
跳父亲加速到 O(log n)（倍增法），以及回答"一条边最多能通过多少流量"这类网络流问题。
"""
from __future__ import annotations

from collections import defaultdict, deque


# ══════════════════════ 自实现算法 1：Tarjan 求强连通分量(SCC) ══════════════════════
def tarjan_scc(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """
    【题意】自实现算法，无对应单一 LeetCode 题号。给定 n 个节点的有向图 edges=[(u,v),...]，
    求所有的强连通分量(SCC)——一个 SCC 是一个最大的节点集合，集合内任意两点 a、b 都
    存在 a 到 b 和 b 到 a 的路径。返回所有 SCC（每个 SCC 是一个节点列表）。
    【思路】Tarjan 算法用一次 DFS 同时求出所有 SCC：给每个节点记录两个值——`disc`
    （DFS 访问它的时间戳）和 `low`（它自己以及它所有还"在栈上"的后代，通过任意条
    返祖边能到达的最早时间戳）。同时维护一个显式栈，节点在被访问时立刻入栈。DFS
    回溯到某个节点 u 时，如果发现 `low[u] == disc[u]`，说明 u 是它所在 SCC 里
    最先被访问、且没有任何后代能"跳出"这个 SCC 追溯到比 u 更早的祖先——这意味着
    从栈顶一直弹出节点直到弹出 u 本身，这些被弹出的节点恰好构成一个完整的 SCC。
    "还在栈上"这个条件至关重要：如果一条边指向的节点已经访问过、但已经不在栈上
    （说明它所在的 SCC 已经被处理完并弹出了），那么这条边只是"跨 SCC 的边"，不能
    用来更新 `low`，否则会把两个本该分开的 SCC 错误地合并成一个。
    【复杂度】时间 O(V+E)（一次 DFS）；空间 O(V+E)（邻接表 + 栈 + disc/low 数组）。
    【易错点】1) 更新 `low[u]` 时必须区分"树边"（后代还没访问过，递归下去后用
    `low[后代]` 更新自己）和"返祖边/跨边"（后代已访问，此时只能用 `disc[后代]`
    更新，而且必须检查后代是否还在栈上，不在栈上的跨 SCC 边不能用来更新）；
    2) 判断 SCC 边界的条件是 `low[u] == disc[u]`，写成 `low[u] == 0` 之类的
    近似判断会在时间戳不从 0 开始或有多个连通块时出错；3) 图不一定弱连通，必须
    对每一个还没访问过的节点都单独触发一次 DFS。
    """
    graph: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    indices = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    node_stack: list[int] = []
    result: list[list[int]] = []
    counter = 0

    for start in range(n):
        if indices[start] != -1:
            continue
        indices[start] = low[start] = counter
        counter += 1
        node_stack.append(start)
        on_stack[start] = True
        work = [(start, iter(graph[start]))]

        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if indices[nxt] == -1:
                    indices[nxt] = low[nxt] = counter
                    counter += 1
                    node_stack.append(nxt)
                    on_stack[nxt] = True
                    work.append((nxt, iter(graph[nxt])))
                    advanced = True
                    break
                elif on_stack[nxt]:
                    low[node] = min(low[node], indices[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == indices[node]:
                comp: list[int] = []
                while True:
                    w = node_stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                result.append(comp)
    return result


# ══════════════════════ 自实现算法 2：Tarjan 求桥(bridges) ══════════════════════
def find_bridges(n: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    【题意】自实现算法，无对应单一 LeetCode 题号。给定 n 个节点的无向图 edges，
    求所有的"桥"——一条边如果被删除后会让图的连通分量数增加，这条边就是桥。
    【思路】桥的判定思路和求割点、SCC 是同一个 low-link 家族：DFS 时记录每个节点
    的 `disc`（发现时间）和 `low`（自己 + 子树内所有节点，通过一条返祖边能到达的
    最早发现时间）。对于树边 (parent -> child)，如果 `low[child] > disc[parent]`，
    说明 child 的整棵子树里，没有任何一条边能"绕回"parent 或更早的祖先——也就是说，
    唯一连接这棵子树和外部的路径就是 parent-child 这条边本身，删掉它子树就断开了，
    这条边就是桥。这里的关键细节是**无向图的返祖边不能立刻用来更新 low**：因为
    (u,v) 这条边在邻接表里对 u、v 各存了一份，DFS 从 parent 走到 child 时，child
    的邻接表里会包含 parent 自己——必须靠"边的编号"而不是"节点编号"来判断某条边
    是不是刚刚走过来的那条父边，否则父边会被误当成一条返祖边，导致 low 值被错误地
    拉低，从而漏判所有的桥。
    【复杂度】时间 O(V+E)；空间 O(V+E)。
    【易错点】1) 用节点编号判断"是否是父边"在重边(parent 和 child 之间有多条边)
    场景下会出错——必须用边的下标去重，本实现给每条边分配一个 idx，邻接表存
    (邻居, 边idx)，跳过的是"边idx == 父边idx"而不是"邻居 == 父节点"；2) 桥的判定
    条件是 `low[child] > disc[parent]`（严格大于），如果写成 `>=` 会把"child 能
    通过返祖边刚好回到 parent 自己"（不算桥，因为环还在）误判为桥；3) 图不一定
    连通，要对每个未访问的节点单独触发 DFS。
    """
    graph: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        graph[u].append((v, idx))
        graph[v].append((u, idx))

    disc = [-1] * n
    low = [0] * n
    timer = 0
    bridges: list[tuple[int, int]] = []

    for start in range(n):
        if disc[start] != -1:
            continue
        disc[start] = low[start] = timer
        timer += 1
        stack = [(start, -1, iter(graph[start]))]

        while stack:
            node, parent_edge, it = stack[-1]
            advanced = False
            for nxt, eidx in it:
                if eidx == parent_edge:
                    continue
                if disc[nxt] == -1:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append((nxt, eidx, iter(graph[nxt])))
                    advanced = True
                    break
                else:
                    low[node] = min(low[node], disc[nxt])
            if advanced:
                continue
            stack.pop()
            if stack:
                parent_node = stack[-1][0]
                low[parent_node] = min(low[parent_node], low[node])
                if low[node] > disc[parent_node]:
                    a, b = parent_node, node
                    bridges.append((a, b) if a < b else (b, a))
    return bridges


# ══════════════════════ 自实现算法 3：求割点(articulation points) ══════════════════════
def find_articulation_points(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """
    【题意】自实现算法，无对应单一 LeetCode 题号。给定 n 个节点的无向图 edges，求所有
    的"割点"——一个节点如果被删除后（连带它相关的边一起删除）会让图的连通分量数增加，
    这个节点就是割点。
    【思路】和求桥用的是同一套 disc/low DFS，但判定条件和"根节点"的特判都不同。对于
    非根节点 u，它在 DFS 树里的某个孩子 v 如果满足 `low[v] >= disc[u]`（注意这里是
    `>=`，桥是 `>`），说明 v 的子树完全没有办法绕过 u 连到 u 的祖先——删掉 u，v 所在
    的子树就会和图的其余部分断开，所以 u 是割点。等号成立的情况(`low[v] == disc[u]`)
    对应"v 的子树里最多绕回 u 自己"，虽然 u-v 这条边不是桥（因为经过 u 还能连通），
    但 u 本身仍然是必经点。DFS 树的**根节点**是特例：它没有"父亲"可以依赖，只有当它
    在 DFS 树里有 **两个或以上的孩子**时才是割点——因为如果只有一个孩子，根节点其实
    可以被绕过（子树内部本来就连通，根只是恰好第一个被访问），只有当根同时连接着两棵
    互不相通的子树时，删掉根才会真正断开这两部分。
    【复杂度】时间 O(V+E)；空间 O(V+E)。
    【易错点】1) 根节点不能直接套用"低于等于父亲发现时间"这条规则，必须单独统计
    DFS 树里根节点的直接孩子数，>= 2 才算割点；2) 割点的判定是 `low[child] >=
    disc[u]`，比桥的判定条件宽松（用 >= 而不是 >），一个节点可能是多条边的公共
    割点，但每条边不一定都是桥；3) 和求桥一样，为避免把"父边"误当成返祖边，实现
    里跳过父边时假设图是简单图(无重边)，重边场景需要额外用边编号去重。
    """
    graph: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    disc = [-1] * n
    low = [0] * n
    timer = 0
    is_cut = [False] * n

    for start in range(n):
        if disc[start] != -1:
            continue
        root_children = 0
        disc[start] = low[start] = timer
        timer += 1
        stack = [(start, -1, iter(graph[start]))]

        while stack:
            node, parent, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt == parent:
                    continue
                if disc[nxt] == -1:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    if node == start:
                        root_children += 1
                    stack.append((nxt, node, iter(graph[nxt])))
                    advanced = True
                    break
                else:
                    low[node] = min(low[node], disc[nxt])
            if advanced:
                continue
            stack.pop()
            if stack:
                parent_node = stack[-1][0]
                low[parent_node] = min(low[parent_node], low[node])
                if parent_node != start and low[node] >= disc[parent_node]:
                    is_cut[parent_node] = True
        if root_children > 1:
            is_cut[start] = True

    return [u for u in range(n) if is_cut[u]]


def critical_connections(n: int, connections: list[list[int]]) -> list[list[int]]:
    """
    【题意】LeetCode 1192·Hard。n 台服务器由 connections 中的无向连接组成一张网络，
    一条连接如果被删除后会让某两台服务器无法互相到达，就是一条"关键连接"。返回所有
    关键连接。
    【思路】"删除后让某两点无法互达的边"正是桥的定义——这道题就是 `find_bridges`
    在真实场景下的直接应用。一条边不是桥，当且仅当它在图的某个环上（环上任意一条边
    被删掉，环上的点仍然能通过环的另一侧互相到达）；只有那些"不在任何环上"的边，
    删除后才会真正断开连通性。
    【复杂度】时间 O(V+E)；空间 O(V+E)，与 `find_bridges` 相同。
    【易错点】与 `find_bridges` 一致：必须用边的编号（而不是节点编号）来判断是否是
    刚走过来的父边，否则重边或者"子节点邻接表里包含父节点自身"这件事会让某些真正
    的桥被漏判。
    """
    edges = [(u, v) for u, v in connections]
    bridges = find_bridges(n, edges)
    return [[a, b] for a, b in bridges]


# ══════════════════════ 自实现算法 4：LCA 倍增法(Binary Lifting) ══════════════════════
class LCABinaryLifting:
    """
    【题意】自实现算法，无对应单一 LeetCode 题号。给定一棵有根树（n 个节点，
    `parent_of_each_node[i]` 是节点 i 的父节点，根节点的父节点可以是任意占位值，
    根节点自身用 `root` 指定），预处理后能以 O(log n) 回答任意两个节点的最近公共
    祖先(LCA)查询。
    【思路】暴力解法是"两个节点各自往上跳到根节点，记录路径，找最深的公共节点"，
    单次查询 O(树高)，树退化成链时是 O(n)。倍增法用"空间换时间"的经典套路——
    预处理出每个节点的第 1、2、4、8... 个祖先（`up[k][v]` 表示 v 的第 2^k 个祖先），
    因为任意距离 d 都能唯一地分解成若干个 2 的幂次之和（d 的二进制表示），"跳 d 步"
    可以拆成最多 O(log n) 次"跳 2^k 步"，而 `up[k][v] = up[k-1][up[k-1][v]]`——
    跳 2^k 步等于先跳 2^(k-1) 步、再跳 2^(k-1) 步，这个递推关系让预处理只需要
    O(n log n)。查询 LCA(u, v) 分两阶段：第一阶段把更深的节点用倍增跳到和另一个
    节点同一深度；第二阶段两个节点一起往上跳、但只跳到"跳完之后两者仍不相同"的
    最大幅度（从大到小尝试每个 2^k），跳到最后两者的父节点就是 LCA——之所以要"跳到
    不相同为止"而不是直接跳到 depth=0，是因为如果贪心地一步跳到共同祖先，可能会
    跳过 LCA 本身，跳到 LCA 的祖先去。
    【复杂度】预处理时间/空间 O(n log n)；单次查询 O(log n)。
    【易错点】1) 倍增表的层数 `log` 必须满足 `2^log >= n`，层数不够会导致深度差
    很大的两个节点跳不到同一深度；2) 第二阶段"从大到小尝试 2^k"时，判断条件是
    `up[k][u] != up[k][v]`（两者跳了之后仍不同才跳），如果写成"相同就跳"会直接
    跳过 LCA；3) 查询结束后 LCA 是 `up[0][u]`（也就是最后 u、v 相同深度但不同节点时，
    它们共同的直接父亲），不是 u 或 v 本身（除非其中一个恰好是另一个的祖先，这种
    情况在深度对齐阶段就已经让 u == v，被前面的特判直接返回）。
    """

    def __init__(self, n: int, parent_of_each_node: list[int], root: int) -> None:
        self.n = n
        self.root = root
        self.log = max(1, (n - 1).bit_length()) if n > 1 else 1
        self.up: list[list[int]] = [[root] * n for _ in range(self.log + 1)]
        self.depth = [0] * n

        children: list[list[int]] = [[] for _ in range(n)]
        for node in range(n):
            if node == root:
                continue
            children[parent_of_each_node[node]].append(node)

        visited = [False] * n
        queue: deque[int] = deque([root])
        visited[root] = True
        while queue:
            u = queue.popleft()
            for v in children[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.up[0][v] = u
                    queue.append(v)

        for k in range(1, self.log + 1):
            for v in range(n):
                self.up[k][v] = self.up[k - 1][self.up[k - 1][v]]

    def query(self, u: int, v: int) -> int:
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for k in range(self.log + 1):
            if (diff >> k) & 1:
                u = self.up[k][u]
        if u == v:
            return u
        for k in range(self.log, -1, -1):
            if self.up[k][u] != self.up[k][v]:
                u = self.up[k][u]
                v = self.up[k][v]
        return self.up[0][u]


class TreeAncestor:
    """
    【题意】LeetCode 1483·Hard。给定 n 个节点的树（`parent[i]` 是节点 i 的父节点，
    根节点 0 的 `parent[0] == -1`），实现 `getKthAncestor(node, k)`：返回 node 的
    第 k 个祖先；如果不存在（跳出根节点范围）返回 -1。题目会有大量查询。
    【思路】这是倍增法思想最直接的应用——`up[k][v]` 表示 v 的第 2^k 个祖先（不存在
    则记为 -1，且 -1 的任意次跳跃结果都定义成 -1，方便递推时不用特判"半路跳出树外"
    的情况）。查询第 k 个祖先时，把 k 按二进制分解，依次尝试从低位到高位跳（每一位
    是 1 就跳对应的 2^i 步），中途一旦跳到 -1 就可以提前结束——因为 -1 的任何后续
    跳跃都还是 -1。因为有"大量查询"，每次查询摊销 O(log n)，相比每次都朴素地
    沿 parent 指针跳 k 次（最坏 O(n) 一次）要快得多，这正是"预处理 O(n log n) +
    多次查询"场景下倍增法的价值所在。
    【复杂度】预处理时间/空间 O(n log n)；单次查询 O(log n)。
    【易错点】1) `up[k][v]` 递推依赖 `up[k-1][v]` 是否已经是 -1，如果不对 -1 做
    特殊处理直接当数组下标用会越界；2) k 可能超过树的最大深度，查询过程中一旦
    中间结果变成 -1 就应该直接返回 -1，不能继续跳；3) 层数 `log` 必须覆盖 n 的
    对数级别，层数不够会导致 k 较大时无法一次跳够。
    """

    def __init__(self, n: int, parent: list[int]) -> None:
        self.log = max(1, (n - 1).bit_length()) if n > 1 else 1
        self.up: list[list[int]] = [[-1] * n for _ in range(self.log + 1)]
        self.up[0] = list(parent)
        for k in range(1, self.log + 1):
            for v in range(n):
                mid = self.up[k - 1][v]
                self.up[k][v] = -1 if mid == -1 else self.up[k - 1][mid]

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.log + 1):
            if node == -1:
                break
            if (k >> i) & 1:
                node = self.up[i][node]
        return node


# ══════════════════════ 自实现算法 5：最大流(Edmonds-Karp) ══════════════════════
def max_flow(
    n: int, edges_with_capacity: list[tuple[int, int, int]], source: int, sink: int
) -> int:
    """
    【题意】自实现算法，无对应单一 LeetCode 题号。给定 n 个节点的有向网络，
    edges_with_capacity=[(u,v,capacity),...] 表示 u 到 v 有一条容量为 capacity 的
    有向边，求从 source 到 sink 能同时传输的最大流量。
    【思路】Edmonds-Karp 是 Ford-Fulkerson 方法的一个具体实现：只要残量网络里还存在
    一条从 source 到 sink 的路径（"增广路"），就沿着这条路径把流量增加"路径上的
    瓶颈容量"（路径里剩余容量最小的那条边）；反复找增广路直到找不到为止，这时候
    的总流量就是最大流。用 BFS（而不是任意 DFS）寻找增广路是 Edmonds-Karp 相对于
    朴素 Ford-Fulkerson 的关键改进——BFS 保证每次找到的是**边数最少**的增广路，
    这个限制保证了算法在有理数容量下一定会在多项式次增广内终止（O(VE) 次），
    而不是像任意路径的 Ford-Fulkerson 那样在某些精心构造的图上退化成指数级。
    每次增广后，正向边的剩余容量减少路径流量，同时**反向边的剩余容量增加同样的
    流量**——反向边的意义是"允许后续增广路撤销掉一部分之前分配得不够好的流量"，
    如果没有反向边，某些图会因为"先分配的流量方向选早了"而错过真正的最大流。
    【复杂度】时间 O(V*E^2)（Edmonds-Karp 的标准复杂度，最多 O(VE) 次增广，每次
    BFS 找增广路是 O(E)）；空间 O(V+E)（残量图的邻接表）。
    【易错点】1) 忘记给每条边建一条初始容量为 0 的反向边，会导致某些图算出的
    最大流偏小（因为没法"退流"重新分配）；2) 增广时要用**上一轮 BFS 记录的
    parent 指针**回溯路径、取路径上所有边的剩余容量的最小值作为这次增广的流量，
    而不是想当然地假设瓶颈是路径的第一条边或最后一条边；3) 多条边连接同一对
    (u, v) 节点时容量应该累加而不是覆盖，否则会漏算部分容量。
    """
    graph: dict[int, dict[int, int]] = defaultdict(dict)
    for u, v, cap in edges_with_capacity:
        graph[u][v] = graph[u].get(v, 0) + cap
        graph[v].setdefault(u, 0)

    def bfs_find_path() -> dict[int, int] | None:
        parent: dict[int, int] = {source: source}
        queue: deque[int] = deque([source])
        while queue:
            u = queue.popleft()
            if u == sink:
                return parent
            for v, cap in graph[u].items():
                if cap > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        return parent if sink in parent else None

    total = 0
    while True:
        parent = bfs_find_path()
        if parent is None or sink not in parent:
            break
        # 回溯路径求瓶颈容量
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, graph[u][v])
            v = u
        # 沿路径更新正向边(减少)和反向边(增加)的剩余容量
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = u
        total += path_flow
    return total


def min_days_to_disconnect_island(grid: list[list[int]]) -> int:
    """
    【题意】LeetCode 1568·Hard。给定 0-1 网格，1 表示陆地、0 表示水，"岛屿"是
    4 方向连通的陆地极大连通块。网格"连通"当且仅当恰好只有一个岛屿。每天可以把
    任意一个陆地格子变成水。求让网格变得"不连通"（0 个或 >=2 个岛屿）所需的最少
    天数。
    【思路】关键观察：答案只可能是 0、1 或 2。0 天：网格本来就不连通（已经是 0 个
    或 >=2 个岛屿）。2 天是理论上界——对任意一个连通的陆地形状，总能找到 2 个
    格子，删掉后就能把它切成两半（这是一个不需要严格证明的经验上界，本题的数据
    规模 m,n<=30 也支持用暴力搜索验证是否 1 天就够）。所以算法是暴力试探：先数
    一遍岛屿数，不等于 1 直接返回 0；否则逐一尝试删除每一个陆地格子（删除后立刻
    恢复），只要有一次删除后岛屿数不等于 1，就返回 1；如果试遍所有陆地格子都不行，
    说明这个形状"足够健壮"，返回上界 2。
    【复杂度】时间 O((mn)^2)（最坏情况下要对每个陆地格子都重新做一次 O(mn) 的
    flood fill 计数）；空间 O(mn)。
    【易错点】1) "岛屿数"包含"0 个岛屿"（整个网格全是水）也算不连通，如果只检查
    "是否 >= 2 个岛屿"会漏掉这种情况；2) 尝试删除某个格子后必须把它还原，否则会
    污染后续的尝试；3) 计数岛屿时一旦发现数量超过 1 就可以提前返回，不需要精确数
    出所有岛屿数量，只关心"是否恰好等于 1"。
    """
    grid = [row[:] for row in grid]
    m, n = len(grid), len(grid[0])

    def count_islands() -> int:
        seen = [[False] * n for _ in range(m)]
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and not seen[i][j]:
                    count += 1
                    stack = [(i, j)]
                    seen[i][j] = True
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < m
                                and 0 <= ny < n
                                and grid[nx][ny] == 1
                                and not seen[nx][ny]
                            ):
                                seen[nx][ny] = True
                                stack.append((nx, ny))
                    if count > 1:
                        return count
        return count

    if count_islands() != 1:
        return 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                grid[i][j] = 0
                disconnected = count_islands() != 1
                grid[i][j] = 1
                if disconnected:
                    return 1
    return 2


def maximal_network_rank(n: int, roads: list[list[int]]) -> int:
    """
    【题意】LeetCode 1615·Medium。n 个城市由 roads 中的无向道路连接，两个不同城市
    的"网络级别"是与它们直接相连的道路总数（如果两城市间有直连道路，这条道路只算
    一次）。求整个基础设施的最大网络级别（所有城市对里网络级别的最大值）。
    【思路】直接枚举所有城市对 (i, j)：网络级别等于 `degree[i] + degree[j]`，如果
    i、j 之间恰好有一条直连道路，这条道路被两边的度数各计了一次，需要减去 1 去重。
    是否直连可以用一个哈希集合预处理判断，O(1) 查询。城市不要求连通（题目允许整个
    基础设施不连通），所以枚举时不能加"必须存在路径"这类多余的限制。
    【复杂度】时间 O(n^2 + m)（n^2 枚举城市对，m 是道路数用于预处理度数和直连集合）；
    空间 O(n + m)。
    【易错点】1) 直连的道路只能减 1 次，不能因为存了 (i,j) 和 (j,i) 两份就减两次；
    2) 网络级别的定义是"两个不同城市"，不能把同一个城市和自己配对；3) 城市之间
    不要求整体连通，某些城市可能是孤立点（度数为 0），仍然要参与枚举。
    """
    degree = [0] * n
    connected: set[tuple[int, int]] = set()
    for u, v in roads:
        degree[u] += 1
        degree[v] += 1
        connected.add((min(u, v), max(u, v)))

    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            rank = degree[i] + degree[j]
            if (i, j) in connected:
                rank -= 1
            best = max(best, rank)
    return best


def jump_game_iv(arr: list[int]) -> int:
    """
    【题意】LeetCode 1345·Hard。给定数组 arr，从下标 0 出发，每一步可以跳到
    i+1、i-1，或任意满足 arr[i]==arr[j] 的下标 j。求到达最后一个下标所需的最少
    步数。
    【思路】把每个下标看成图上的一个节点，三种跳法就是三类边，问题变成"无权图上
    从 0 到 n-1 的最短路"，天然用 BFS。关键优化：如果直接把"所有值相同的下标互相
    连边"建成显式邻接表，最坏情况下（比如全部元素相同）边数是 O(n^2)。优化方式是
    "用完即弃"——第一次从某个值出发扩展"所有同值下标"之后，把这个值对应的下标列表
    清空；因为同值的下标之间两两可达（一步跳到任意同值下标），一旦其中任意一个
    被访问过，这一整组同值下标要么已经在 BFS 的当前层或更早层被加入过队列，要么
    会在处理这一层时被这次操作一次性全部加入，没有必要在后续任何时刻重新展开这一
    组，这样每个值分组只会被完整展开一次，总的跳转边数降到 O(n)。
    【复杂度】时间 O(n)（每个下标最多入队一次，每个值分组最多被展开一次）；
    空间 O(n)（值到下标列表的哈希表 + BFS 队列 + 访问标记）。
    【易错点】1) 不做"用完即弃"优化，纯粹按边展开会在存在大量重复值时退化到
    O(n^2)甚至 TLE；2) 清空值分组时机要放在"已经把这组下标全部加入 BFS 候选"
    之后，不能在展开之前就清空；3) 数组长度为 1 时直接返回 0（已经在终点），
    如果不特判会进入循环但逻辑上也能得出 0，特判只是让边界更清晰。
    """
    n = len(arr)
    if n == 1:
        return 0

    value_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, v in enumerate(arr):
        value_to_indices[v].append(i)

    visited = [False] * n
    visited[0] = True
    queue: deque[int] = deque([0])
    steps = 0
    while queue:
        for _ in range(len(queue)):
            i = queue.popleft()
            if i == n - 1:
                return steps
            neighbors = [i - 1, i + 1] + value_to_indices[arr[i]]
            value_to_indices[arr[i]] = []
            for j in neighbors:
                if 0 <= j < n and not visited[j]:
                    visited[j] = True
                    queue.append(j)
        steps += 1
    return -1  # 题目保证一定可达，理论上不会执行到这里


def graph_connectivity_with_threshold(
    n: int, threshold: int, queries: list[list[int]]
) -> list[bool]:
    """
    【题意】LeetCode 1627·Hard。城市编号 1..n，如果两个城市 x、y 存在一个严格大于
    threshold 的公共因数，就在它们之间连一条边。对每个查询 [a,b]，判断 a、b 是否
    连通（直接或间接）。
    【思路】如果暴力枚举每一对城市判断是否有公共因数会是 O(n^2)，无法接受。换个
    角度：与其枚举"城市对"，不如枚举"因数" z（从 threshold+1 到 n），把所有是 z
    的倍数的城市两两连通——因为它们都以 z 为公共因数，天然属于同一个连通分量。
    用并查集实现"连通"：对每个因数 z，把 z 和它的倍数 2z、3z、...都合并到同一个
    集合。这样枚举因数 + 枚举倍数的总操作次数是调和级数 O(n log n)，而不是 O(n^2)。
    合并完所有因数关系后，两个城市连通当且仅当它们在并查集里的根相同。
    【复杂度】时间 O(n log n * α(n) + q)（n log n 是因数+倍数的调和级数枚举，q 是
    查询数，并查集操作均摊 O(α(n))）；空间 O(n)。
    【易错点】1) 因数 z 要从 `threshold + 1` 开始枚举，不是从 1 开始——严格大于
    threshold 才算合法公共因数；2) threshold 为 0 时，1 是每个数的因数且
    `1 > 0` 成立，此时所有城市都会通过因数 1 连通到一起，容易误以为 threshold=0
    是特殊情况需要单独处理，其实按通用逻辑跑就是正确的；3) 城市编号从 1 开始，
    并查集数组要开到 n+1 大小，用下标 0 会浪费但不影响正确性，混淆下标偏移会
    导致越界或查询结果整体错位。
    """
    parent = list(range(n + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for z in range(threshold + 1, n + 1):
        for multiple in range(2 * z, n + 1, z):
            union(z, multiple)

    return [find(a) == find(b) for a, b in queries]


def minimum_score_path(n: int, roads: list[list[int]]) -> int:
    """
    【题意】LeetCode 2492·Medium。n 个城市，roads[i]=[a,b,distance] 表示 a、b
    之间有一条距离为 distance 的双向道路。一条"路径"的得分定义为路径上所有道路里
    距离的最小值，路径允许重复经过同一条道路、重复经过同一个城市。求从城市 1 到
    城市 n 的路径里，得分最小可以是多少。
    【思路】因为允许重复走同一条边、反复经过同一个城市，"路径"这个概念被极大地
    放宽了——只要城市 1 和城市 n 处在同一个连通分量里，这个连通分量内**任意一条
    边**都可以被"绕路"利用上（先从 1 走到那条边的一个端点，来回走一次这条边，
    再走回主路线继续走到 n）。这意味着答案就是"包含城市 1 和城市 n 的那个连通
    分量内，所有边的最小距离"，问题退化成一次简单的连通块内边权求最小值：从
    城市 1 出发做一次 DFS/BFS，遍历这个连通分量能碰到的所有边，取距离最小值。
    【复杂度】时间 O(n + m)（一次 DFS/BFS 遍历连通分量）；空间 O(n + m)。
    【易错点】1) 容易想复杂去找"最短路径"或"限定不重复访问的路径"，但题目明确
    允许重复经过城市和道路，本质和"路径形状"无关，只和"连通分量内的边集合"有关；
    2) 遍历时要在访问每一条边时都更新最小值（哪怪边的另一端已经访问过），而不是
    只在"发现新节点"时才看边权，否则会漏掉"连通分量内部、但两端都已访问过"的边；
    3) 城市编号从 1 开始，数组/字典大小和起点终点的下标要对应留意，不要按 0-index
    习惯直接用城市 n 当数组越界的下标。
    """
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, dist in roads:
        graph[u].append((v, dist))
        graph[v].append((u, dist))

    visited = [False] * (n + 1)
    stack = [1]
    visited[1] = True
    min_score = float("inf")
    while stack:
        u = stack.pop()
        for v, dist in graph[u]:
            min_score = min(min_score, dist)
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return min_score


def _self_test() -> None:
    # ---- 自实现算法交叉验证 ----
    # Tarjan SCC：手工构造两个已知 SCC 结构的图
    scc1 = tarjan_scc(5, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)])
    assert {frozenset(c) for c in scc1} == {
        frozenset({0, 1, 2}),
        frozenset({3}),
        frozenset({4}),
    }
    scc2 = tarjan_scc(4, [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)])
    assert {frozenset(c) for c in scc2} == {frozenset({0, 1}), frozenset({2, 3})}

    # 找桥：三角环 + 链，三角环内部无桥，链上全是桥
    bridges1 = find_bridges(5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    assert {frozenset(b) for b in bridges1} == {frozenset((2, 3)), frozenset((3, 4))}
    # 纯链：每条边都是桥
    bridges2 = find_bridges(4, [(0, 1), (1, 2), (2, 3)])
    assert {frozenset(b) for b in bridges2} == {
        frozenset((0, 1)),
        frozenset((1, 2)),
        frozenset((2, 3)),
    }
    # 两个环由一座桥连接
    bridges3 = find_bridges(
        6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    )
    assert {frozenset(b) for b in bridges3} == {frozenset((2, 3))}

    # 割点：同一张"三角环+链"图，割点应为 2 和 3
    cuts1 = find_articulation_points(5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    assert cuts1 == [2, 3]
    # 纯链：中间两个节点是割点，两端叶子不是
    cuts2 = find_articulation_points(4, [(0, 1), (1, 2), (2, 3)])
    assert cuts2 == [1, 2]

    # LC1192 关键连接：桥的直接应用
    assert {frozenset(c) for c in critical_connections(4, [[0, 1], [1, 2], [2, 0], [1, 3]])} == {
        frozenset((1, 3))
    }
    assert critical_connections(2, [[0, 1]]) == [[0, 1]]

    # LCA 倍增法：对两棵不同形状的树做穷举式交叉验证（暴力法：往上爬到根记录祖先集合）
    def brute_lca(parent_arr: list[int], root: int, u: int, v: int) -> int:
        ancestors_u = set()
        x = u
        while True:
            ancestors_u.add(x)
            if x == root:
                break
            x = parent_arr[x]
        y = v
        while y not in ancestors_u:
            y = parent_arr[y]
        return y

    # 树 A：LC1483 官方样例的树形结构 parent=[-1,0,0,1,1,2,2]
    parent_a = [-1, 0, 0, 1, 1, 2, 2]
    lca_a = LCABinaryLifting(7, parent_a, root=0)
    for uu in range(7):
        for vv in range(7):
            assert lca_a.query(uu, vv) == brute_lca(parent_a, 0, uu, vv)

    # 树 B：退化成一条链，验证树很"高"时也正确
    parent_b = [-1, 0, 1, 2, 3, 4]
    lca_b = LCABinaryLifting(6, parent_b, root=0)
    for uu in range(6):
        for vv in range(6):
            assert lca_b.query(uu, vv) == brute_lca(parent_b, 0, uu, vv)

    # LC1483 TreeAncestor：官方样例
    tree_ancestor = TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2])
    assert tree_ancestor.get_kth_ancestor(3, 1) == 1
    assert tree_ancestor.get_kth_ancestor(5, 2) == 0
    assert tree_ancestor.get_kth_ancestor(6, 3) == -1

    # 最大流(Edmonds-Karp)：3 个手工验证的网络
    # 网络 1：CLRS 经典例题 s=0 v1=1 v2=2 v3=3 v4=4 t=5，最大流已知为 23
    flow1 = max_flow(
        6,
        [
            (0, 1, 16),
            (0, 2, 13),
            (2, 1, 4),
            (1, 3, 12),
            (3, 2, 9),
            (2, 4, 14),
            (4, 3, 7),
            (3, 5, 20),
            (4, 5, 4),
        ],
        source=0,
        sink=5,
    )
    assert flow1 == 23
    # 网络 2："菱形"图，源点总容量 3+2=5 恰好是瓶颈，最大流应为 5
    flow2 = max_flow(
        4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3), (1, 2, 1)], source=0, sink=3
    )
    assert flow2 == 5
    # 网络 3：简单链，最大流是链上最小容量
    flow3 = max_flow(4, [(0, 1, 5), (1, 2, 3), (2, 3, 7)], source=0, sink=3)
    assert flow3 == 3

    # ---- 真实 LeetCode 题 ----
    assert min_days_to_disconnect_island(
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    ) == 2
    assert min_days_to_disconnect_island([[1, 1]]) == 2
    assert min_days_to_disconnect_island([[1, 0, 1, 0]]) == 0
    assert min_days_to_disconnect_island([[1, 1, 1]]) == 1

    assert maximal_network_rank(4, [[0, 1], [0, 3], [1, 2], [1, 3]]) == 4
    assert (
        maximal_network_rank(5, [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4]]) == 5
    )
    assert (
        maximal_network_rank(8, [[0, 1], [1, 2], [2, 3], [2, 4], [5, 6], [5, 7]]) == 5
    )

    assert (
        jump_game_iv([100, -23, -23, 404, 100, 23, 23, 23, 3, 404]) == 3
    )
    assert jump_game_iv([7]) == 0
    assert jump_game_iv([7, 6, 9, 6, 9, 6, 9, 7]) == 1

    assert graph_connectivity_with_threshold(6, 2, [[1, 4], [2, 5], [3, 6]]) == [
        False,
        False,
        True,
    ]
    assert graph_connectivity_with_threshold(
        6, 0, [[4, 5], [3, 4], [3, 2], [2, 6], [1, 3]]
    ) == [True, True, True, True, True]
    assert graph_connectivity_with_threshold(
        5, 1, [[4, 5], [4, 5], [3, 2], [2, 3], [3, 4]]
    ) == [False, False, False, False, False]

    assert minimum_score_path(4, [[1, 2, 9], [2, 3, 6], [2, 4, 5], [1, 4, 7]]) == 5
    assert minimum_score_path(4, [[1, 2, 2], [1, 3, 4], [3, 4, 7]]) == 2

    print(
        "[PASS] p30_advanced_graph_ii: 12/12 通过 "
        "(Tarjan-SCC/find_bridges/find_articulation_points/LC1192/"
        "LCA倍增/LC1483/max_flow/LC1568/LC1615/LC1345/LC1627/LC2492)"
        "（LC1489关键边/伪关键边改归入 17-union-find 类，避免和该类重复）"
    )


if __name__ == "__main__":
    _self_test()
