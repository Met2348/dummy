"""分类 22：高级图论 —— 带权最短路(Dijkstra/Bellman-Ford)、最小生成树、二分图判定的延伸。

这批题是 Phase 1 图论（BFS/DFS/并查集）的自然延伸：当边有权重时要用 Dijkstra/
Bellman-Ford 求最短路；当要用最少的边连通所有点时要用最小生成树；当要判断"能否
二染色"时要用染色法；当要判断"从某点出发是否必然走向死路"时，反图 + 拓扑排序
（或三色 DFS）是标准解法。
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque


def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    """
    【题意】n 个节点(编号 1..n)组成有向带权图，times[i]=[u,v,w] 表示 u 到 v 有一条
    耗时 w 的单向边。从节点 k 出发发送信号，求"所有节点都收到信号"所需的最短时间；
    如果有节点永远收不到，返回 -1。
    【思路】"从一个源点出发，到所有其他点的最短距离"正是单源最短路径问题，边权非负
    时用 Dijkstra：维护一个"当前已确定最短距离"的集合，每次从集合外贪心选一个距离
    最小的节点确定下来（用最小堆维护候选，堆顶就是当前候选里距离最小的）。和 BFS 的
    核心区别是——BFS 隐含假设"每条边权重都是 1"，按层扩展就能保证第一次到达即最短；
    但边权不同时，"先出队的节点不代表真正距离更小"这个假设不成立了，必须显式比较
    距离、用堆选出全局候选里最小的那个，才能保证每个节点第一次被"确定"下来时，这个
    距离就是全局最短（不会再被更短的路径更新）。所有节点都被确定之后，答案就是这些
    最短距离里的最大值（因为要等最慢到达的那个节点，"所有节点都收到"的时刻才成立）；
    如果确定的节点数小于 n，说明有节点从 k 不可达，返回 -1。
    【复杂度】时间 O((V+E) log V)（每条边最多松弛一次，堆操作 O(log V)）；空间
    O(V+E)。
    【易错点】1) 堆里可能因为同一个节点被多次以不同距离入堆（还没被访问、但后来又
    找到更短路径），出堆时必须检查"这个距离是否还是当前已知最短距离"，如果堆顶距离
    比记录的更大，说明是过期数据，直接跳过，否则会用旧的、更差的距离误判；2) 最终
    答案是全部最短距离的**最大值**，不是"从 k 出发走过的边数最多的那条路径"，也不是
    简单地把所有距离相加；3) 忘记判断"是否所有节点都被访问到"，只看堆是否清空
    不能说明可达性，图不连通时堆会提前耗尽。
    """
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist: dict[int, int] = {k: 0}
    heap: list[tuple[int, int]] = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    if len(dist) < n:
        return -1
    return max(dist.values())


def find_cheapest_price(
    n: int, flights: list[list[int]], src: int, dst: int, k: int
) -> int:
    """
    【题意】n 个城市间有若干带价格的单向航班 flights=[u,v,price]，求从 src 到 dst、
    **最多经停 k 站**（也就是最多飞 k+1 段航班）的最便宜价格；不存在这样的路线返回 -1。
    【思路】这里不能直接套 Dijkstra——Dijkstra 的贪心正确性依赖"一旦一个节点的最短
    距离被确定，后面不会再被更新"，但这道题引入了"最多经停 k 站"这个额外的次数限制，
    可能"最便宜的路线"反而经停更多站、被这个限制排除掉，导致 Dijkstra 的贪心假设失效。
    正确做法是 Bellman-Ford 的"分层松弛"思想：只允许做 k+1 轮"用一条边松弛"的操作
    （对应最多经停 k 站 = 最多飞 k+1 段航班），第 i 轮结束时，dist 数组存的是"最多用
    i 条边能到达每个城市的最便宜价格"。关键细节是每一轮松弛必须基于**上一轮结束时**
    的 dist 快照，而不是本轮已经更新过的 dist——否则同一轮里会连续用两条边松弛，
    变相突破了"最多 k+1 条边"的限制。
    【复杂度】时间 O(k * E)（E 为航班数，k+1 轮，每轮扫一遍所有航班）；空间 O(n)。
    【易错点】1) 直接在原地更新 dist 而不拷贝快照，会让"经停次数"的限制被悄悄突破，
    对本例这种数据可能得到偏低（错误地更优）的价格；2) 松弛前要检查起点 dist[u]
    是否已经可达（不是 inf），否则会把"不可达 + 航班价格"这种无意义的加法当成合法
    候选值；3) k 代表"经停站数"，实际允许的边数是 k+1，容易在循环次数上差一。
    """
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        new_dist = dist.copy()
        for u, v, w in flights:
            if dist[u] != INF and dist[u] + w < new_dist[v]:
                new_dist[v] = dist[u] + w
        dist = new_dist
    return dist[dst] if dist[dst] != INF else -1


def min_cost_connect_points(points: list[list[int]]) -> int:
    """
    【题意】给定平面上若干个点，任意两点 i、j 之间连接的代价是它们的曼哈顿距离
    |xi-xj|+|yi-yj|；求把所有点连通所需的最小总代价。
    【思路】"用最少代价的边把所有点连通"正是最小生成树(MST)问题。这里用 Prim 算法：
    维护一个"已经在生成树里"的点集合，以及每个还没入树的点到"当前树"的最短连接距离
    `min_edge`；每一步贪心选出 `min_edge` 最小的那个未入树点，把它加入树（这条边一定
    在某棵最小生成树里，因为它是连接"树内"和"树外"这一刀切割上最短的边——这是 MST
    的割性质），再用这个新加入的点去尝试更新其余未入树点的 `min_edge`。之所以选
    Prim 而不是 Kruskal：这里的图是稠密图（任意两点都有一条边，边数是 O(n^2)），
    Prim 每一步只需要考虑"树到每个未入树点的最短距离"，天然利用了这种稠密结构，不需要
    像 Kruskal 那样先把 O(n^2) 条边全部排序。
    【复杂度】时间 O(n^2 log n)（用堆实现 Prim，n 个点、每次更新 min_edge 最多入堆一次，
    共 O(n^2) 次距离计算）；空间 O(n^2)（最坏情况堆里的候选边数）。
    【易错点】1) 堆里同一个点可能因为 min_edge 被多次更新而重复入堆，出堆时必须检查
    该点是否已经在树里（`in_mst`），否则同一个点会被计入总代价两次；2) 距离要用曼哈顿
    距离（绝对值之和），不是欧几里得距离；3) 初始化时只有起点（下标 0，选哪个点作为
    起点都不影响最终答案）的 `min_edge` 是 0，其余点是无穷大，不能全部初始化成 0。
    """
    n = len(points)
    in_mst = [False] * n
    total = 0
    visited_count = 0
    heap: list[tuple[int, int]] = [(0, 0)]  # (与当前树的最短距离, 点下标)
    while heap and visited_count < n:
        cost, u = heapq.heappop(heap)
        if in_mst[u]:
            continue
        in_mst[u] = True
        visited_count += 1
        total += cost
        for v in range(n):
            if not in_mst[v]:
                d = abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                heapq.heappush(heap, (d, v))
    return total


def is_bipartite(graph: list[list[int]]) -> bool:
    """
    【题意】给定无向图的邻接表 graph（graph[u] 是 u 的所有邻居），判断这张图是否是
    二分图——即能否把所有节点分成两组，使得每一条边的两个端点都分别属于不同的组。
    【思路】"能否二染色"用染色法：任选一个未染色的节点染成颜色 A，BFS/DFS 扩展到它的
    每个邻居，邻居必须染成相反的颜色 B；如果扩展到某个邻居时发现它已经染过色、且颜色
    和"应该染的颜色"冲突（也就是和当前节点同色），说明存在一条边连接了同色的两个点，
    图不是二分图。图不一定连通，必须对每一个还没被访问过的节点都单独触发一次染色
    过程（分别处理每个连通分量），漏掉任何一个连通分量都可能让"看似成功染色"的假象
    掩盖住另一个分量里的矛盾。
    【复杂度】时间 O(V+E)（每个节点、每条边最多访问一次）；空间 O(V)。
    【易错点】1) 只从节点 0 开始染色，一旦图不连通，其余连通分量永远不会被处理，
    可能漏掉这些分量内部的矛盾（错误地判定为 True）；2) 判断"颜色冲突"要用
    `color[v] == color[u]`，而不是简单判断 `color[v] != 0`——已经染色但染的正是
    期望的相反色，是完全合法的情况，不能当成矛盾；3) 用 0 表示"未染色"时，两种颜色
    不能用 0 和别的数表示（容易和"未染色"混淆），这里选 1 和 -1 分别表示两种颜色，
    互为相反数，判断"另一染色"直接取负号即可。
    """
    n = len(graph)
    color = [0] * n
    for start in range(n):
        if color[start] != 0:
            continue
        color[start] = 1
        queue: deque[int] = deque([start])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if color[v] == 0:
                    color[v] = -color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def make_connected(n: int, connections: list[list[int]]) -> int:
    """
    【题意】n 台电脑(编号 0..n-1)，connections 是若干条已有的网线连接；每次操作可以
    拔掉一条网线接到别处；求最少需要多少次操作才能让所有电脑连通，如果网线数量根本
    不够（连一棵生成树都连不出来）返回 -1。
    【思路】n 台电脑连通至少需要 n-1 条边（一棵生成树），如果 `len(connections) <
    n-1`，边的数量连基本需求都不满足，直接返回 -1。否则用并查集扫一遍所有已有连接：
    真正让两个不同连通分量合并的连接才是"有用的"，那些连接的两点本来就已经连通的边
    就是"多余的网线"——这些多余网线数量一定 >= 需要拔掉重接的次数（因为题目保证边数
    足够）。扫完之后，`当前连通分量数 - 1` 就是"还需要多少次拔线重接才能把所有分量
    连成一个"——每次操作本质是"消耗一条多余的网线，把两个分量合并成一个"，重复
    分量数-1 次就能把所有分量并成一个。
    【复杂度】时间 O(E * α(n))（E 为连接数，并查集操作均摊近似 O(1)）；空间 O(n)。
    【易错点】1) 忘记先检查"边数是否 >= n-1"这个必要条件，如果边本身就不够，后面
    并查集怎么合并都连不成一棵生成树，必须在合并之前就直接返回 -1；2) 答案是
    "剩余连通分量数 - 1"，不是"多余网线的数量"——虽然题目保证多余网线数量一定
    >= 需要的操作次数，但正确的操作次数就是让分量数减到 1 所需的最少合并次数；
    3) 并查集要在遍历全部 connections 后才统计最终分量数，不能中途提前下结论。
    """
    if len(connections) < n - 1:
        return -1

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = n
    for a, b in connections:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            components -= 1
    return components - 1


def eventual_safe_nodes(graph: list[list[int]]) -> list[int]:
    """
    【题意】给定有向图 graph，"安全节点"定义为：从这个节点出发，不管走哪条路径，
    最终都会走到一个"终端节点"（出度为 0 的节点），不会陷入任何环。返回所有安全节点
    （按编号升序）。
    【思路】直接判断"从每个点出发是否可能进入环"，如果对每个点都单独做一次 DFS
    会有大量重复计算。更高效的做法是反向思考：出度为 0 的节点（终端节点）天然安全；
    一个节点如果它指向的所有邻居都安全，那它自己也安全（不管走哪条出边，最终都会
    落到某个安全节点上，再传递到终端）。这是一个"从终端向外传播安全性"的过程，
    正好对应反图上的拓扑排序：建一张反图（原图里 u->v 的边，反图里变成 v->u），
    把原图里出度为 0 的节点作为拓扑排序的起点（在反图语境下，它们是"入度"为 0 的
    起点，这里的"入度"复用原图的出度数组来递减模拟），用 Kahn 算法一层层剥离——
    每次确定一个安全节点，就去反图里找"谁指向它"，把这些节点的（原图）出度减一，
    一旦某个节点的出度减到 0，说明它指向的所有节点都已经被确定为安全，它自己也
    立刻变安全，加入队列。
    【复杂度】时间 O(V+E)；空间 O(V+E)。
    【易错点】1) 容易把"出度"和"反图里的连接"搞混——递减的是**原图**里这个节点的
    出度（表示"它指向的、还没被确定为安全的邻居数量"），但触发递减的连接来自**反图**
    （要找"谁指向了刚被确定安全的这个节点"）；2) 初始入队的是原图里出度为 0 的节点，
    不是随便挑的节点；3) 也可以用三色 DFS（白=未访问/灰=正在这条路径上/黑=已确认
    安全）等价实现，本质上是同一件事——检测并剥离"当前节点是否必然导向环"，反图
    拓扑排序是其中更直观地对应"安全性传播"的写法。
    """
    n = len(graph)
    reverse_graph: list[list[int]] = [[] for _ in range(n)]
    outdegree = [0] * n
    for u in range(n):
        outdegree[u] = len(graph[u])
        for v in graph[u]:
            reverse_graph[v].append(u)

    queue: deque[int] = deque(u for u in range(n) if outdegree[u] == 0)
    safe = [False] * n
    while queue:
        node = queue.popleft()
        safe[node] = True
        for pred in reverse_graph[node]:
            outdegree[pred] -= 1
            if outdegree[pred] == 0:
                queue.append(pred)
    return [u for u in range(n) if safe[u]]


def count_paths(n: int, roads: list[list[int]]) -> int:
    """
    【题意】n 个城市间有若干无向带权道路 roads=[u,v,time]，求从城市 0 到城市 n-1
    "所有耗时最短"的路径一共有多少条（对 10**9+7 取模）。
    【思路】这是 Dijkstra 求最短路的自然延伸："最短距离"之外，还要统计"有多少条不同
    路径能达到这个最短距离"。在标准 Dijkstra 松弛的同时，额外维护 `ways[v]`（从起点
    到 v、按最短距离走的方案数）：当发现一条更短的路径能到达 v（`nd < dist[v]`），
    说明之前累积的所有方案都不再是最短的了，直接用 `ways[u]`（到达 u 的方案数）覆盖
    重置 `ways[v]`；当发现一条**同样短**的路径能到达 v（`nd == dist[v]`），说明这是
    另一条同样最短的路线，两条路线的方案数应该相加，`ways[v] += ways[u]`——这正是
    "最短路计数"的核心：方案数的传递严格跟着最短距离的更新逻辑走，"更短"是覆盖，
    "一样短"是累加。
    【复杂度】时间 O((V+E) log V)（标准 Dijkstra 的复杂度，计数只是常数级的额外
    维护）；空间 O(V+E)。
    【易错点】1) 分不清"覆盖"和"累加"的触发条件——只有严格更短时才覆盖重置方案数，
    严格相等时才累加，写反了会导致方案数被错误清零或者重复叠加；2) 忘记对结果取模
    `10**9+7`，方案数可能是指数级增长的大数；3) 起点自身的初始方案数是 1（"什么都
    不做，已经在起点"算一种方案），漏了这个初始化会导致最终答案恒为 0。
    """
    MOD = 10**9 + 7
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, w in roads:
        graph[u].append((v, w))
        graph[v].append((u, w))

    INF = float("inf")
    dist = [INF] * n
    ways = [0] * n
    dist[0] = 0
    ways[0] = 1
    heap: list[tuple[int, int]] = [(0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                ways[v] = ways[u]
                heapq.heappush(heap, (nd, v))
            elif nd == dist[v]:
                ways[v] = (ways[v] + ways[u]) % MOD
    return ways[n - 1] % MOD


def _self_test() -> None:
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2

    assert (
        find_cheapest_price(
            4, [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]], 0, 3, 1
        )
        == 700
    )

    assert min_cost_connect_points([[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]]) == 20

    assert is_bipartite([[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]) is False
    assert is_bipartite([[1, 3], [0, 2], [1, 3], [0, 2]]) is True

    assert make_connected(4, [[0, 1], [0, 2], [1, 2]]) == 1
    assert make_connected(6, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]) == 2
    assert make_connected(6, [[0, 1], [0, 2], [0, 3], [1, 2]]) == -1

    assert eventual_safe_nodes([[1, 2], [2, 3], [5], [0], [5], [], []]) == [2, 4, 5, 6]

    assert (
        count_paths(
            7,
            [
                [0, 6, 7],
                [0, 1, 2],
                [1, 2, 3],
                [1, 3, 3],
                [6, 3, 3],
                [3, 5, 1],
                [6, 5, 1],
                [2, 5, 1],
                [0, 4, 5],
                [4, 6, 2],
            ],
        )
        == 4
    )

    print(
        "[PASS] p22_advanced_graph: 7/7 题通过 "
        "(网络延迟时间/K站中转最便宜航班/连接所有点的最小费用/判断二分图/"
        "连通网络的操作次数/找到最终的安全状态/到达目的地的方案数)"
    )


if __name__ == "__main__":
    _self_test()
