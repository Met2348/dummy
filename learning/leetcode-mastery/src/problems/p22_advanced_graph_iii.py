"""分类 22：高级图论 —— Phase 3 竞赛级补充：Dijkstra 与 DP/BFS 状态扩展的组合变体。

这批题都不是"裸" Dijkstra：1928/1368 是"最短路径"和"额外约束（时间预算/改向代价）"
交织在一起的分层 DP；1786 是"先用 Dijkstra 求出一份距离数组，再在这份距离上做计数
DP"；882 是"边本身被拆成一串新节点"的 Dijkstra 变体；1210/2045 则说明"最短路"思想
不是 Dijkstra 专属——当状态本身比"一个节点编号"更复杂（蛇的位置+朝向、到达同一点的
"次短"而非"最短"）时，BFS 配合精心设计的状态定义同样能求出最短/次短路径。
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque


def min_cost_to_reach_destination_in_time(
    max_time: int, edges: list[list[int]], passing_fees: list[int]
) -> int:
    """
    【题意】n 个城市间有若干双向道路 edges[i]=[u,v,time]，每经过一个城市（包括起点和
    终点）都要付一次该城市的通行费 passing_fees[i]。从城市 0 出发，要求总耗时不超过
    max_time 分钟到达城市 n-1，求满足这个时间限制的最小总通行费；无法在限定时间内
    到达则返回 -1。
    【思路】这是"限时最短路"和"787 题限次数最短路"同一类套路的再变形：单纯 Dijkstra
    的贪心假设——"一个节点一旦被确定最短距离/最小花费，就不会再被更优路径替代"——在这
    里不成立，因为**耗费时间更多的路径可能通行费更低**，"花费最小"和"耗时最小"是两个
    可能互相冲突的目标，必须同时追踪。标准做法是按"总耗时"分层做 DP：`dp[t][j]`
    表示"用不超过 t 分钟的总耗时到达城市 j"所需的最小通行费，`dp[0][0] = fee[0]`
    （起点自身也要付费）。从 t=1 递推到 max_time：先把 `dp[t]` 初始化成 `dp[t-1]`
    的拷贝（"这一分钟什么都不做，之前已经达到的花费依然有效"），再用每一条边
    `(u, v, w)`（w <= t 时才可能通过这条边到达）尝试松弛：如果 `dp[t-w][u]` 可行，
    那么"先用 t-w 分钟到达 u，再花 w 分钟走到 v"这条路径在总耗时 t 内可行，花费是
    `dp[t-w][u] + fee[v]`。因为 `t - w < t`，`dp[t-w]` 早在处理到 t 这一层之前就已
    经算完，不会有"用还没算出来的未来结果推导当前结果"的循环依赖问题。最终答案是
    `dp[max_time][n-1]`——注意是固定读取"城市 n-1"这一列，不能取整层里的最小值，
    因为那样会把"根本没往前走、停在起点"的花费也算进去，虽然起点本身的花费更小，但
    压根没有到达终点，不是一个合法答案。
    【复杂度】时间 O(max_time * E)（E 为道路数，每一层耗时都要扫一遍全部边）；空间
    O(max_time * n)（n 为城市数，需要保留每一层的 dp 数组，也可以用滚动数组优化到
    O(n)，但保留全部层更直观地体现"每一层都是一个独立的、不超过某个时间预算的子问题"
    这个教学点）。
    【易错点】1) 每一层必须先把上一层的结果原样"继承"下来（`dp[t] = dp[t-1][:]`），
    再叠加新的松弛，否则会把"总耗时恰好等于 t"和"总耗时不超过 t"这两个不同的语义
    搞混，导致原本已经可行的、耗时更短的方案在后面的层反而"丢失"；2) 起点城市 0 自身
    的通行费必须在 `dp[0][0]` 里就计入，很容易漏掉"起点也要付费"这个细节；3) 图是
    双向的，松弛时两个方向都要做（`u->v` 和 `v->u`），只处理一个方向会漏掉合法路径；
    4) 取最终答案时如果写成 `min(dp[max_time])`（取整层的最小值）而不是
    `dp[max_time][n-1]`，会把"起点本身花费更低、但根本没有前往终点"这种无效状态
    也纳入比较——因为 `dp[t][0]` 恒等于 `fee[0]`（停在起点显然一直"可行"），这个
    值几乎总是全局最小，如果误取整层最小值，算出来的答案会恒等于起点的通行费，
    看似"通过"了简单用例，实则完全没有验证是否真的到达了终点。
    """
    n = len(passing_fees)
    INF = float("inf")
    dp = [[INF] * n for _ in range(max_time + 1)]
    dp[0][0] = passing_fees[0]
    for t in range(1, max_time + 1):
        dp[t] = dp[t - 1][:]
        for u, v, w in edges:
            if w > t:
                continue
            if dp[t - w][u] != INF:
                cand = dp[t - w][u] + passing_fees[v]
                if cand < dp[t][v]:
                    dp[t][v] = cand
            if dp[t - w][v] != INF:
                cand = dp[t - w][v] + passing_fees[u]
                if cand < dp[t][u]:
                    dp[t][u] = cand
    best = dp[max_time][n - 1]
    return -1 if best == INF else best


def number_of_restricted_paths(n: int, edges: list[list[int]]) -> int:
    """
    【题意】n 个节点(编号 1..n)组成带权无向连通图，定义 distanceToLastNode(x) 为节点
    x 到节点 n 的最短距离。一条"受限路径"是从节点 1 到节点 n 的一条路径
    [z0=1, z1, ..., zk=n]，要求路径上 distanceToLastNode 严格递减
    （distanceToLastNode(zi) > distanceToLastNode(zi+1)）。求受限路径总数，对
    10^9+7 取模。
    【思路】分两步：第一步用 Dijkstra 求出**每个节点到节点 n 的最短距离**（把 n 当成
    Dijkstra 的源点，因为求的是"到 n"而不是"从 1 出发"的距离，反过来跑一次单源最短
    路即可，图本身无向所以这个距离是对称的）；第二步在这份距离数组之上做记忆化 DFS
    计数：从节点 1 出发，能走到的下一个节点 v 必须满足 `dist[v] < dist[u]`（这正是
    "受限路径"的定义），把这样的边保留下来，会天然形成一个 DAG（因为每一步 dist
    严格变小，不可能出现环），在这个 DAG 上做"从 1 到 n 的路径计数"就是标准的
    记忆化 DFS：`ways(u) = sum(ways(v) for v in neighbors(u) if dist[v] < dist[u])`，
    边界 `ways(n) = 1`。distanceToLastNode 严格递减这个约束，本质上是把"计数"这个
    本来可能有环、路径数无穷的问题，通过"只往 dist 更小的方向走"限制成了一个在 DAG
    上可以安全递归 + 记忆化的有限计数问题。
    【复杂度】Dijkstra 部分 O((V+E) log V)；DFS 计数部分 O(V+E)（每个节点的 `ways`
    只计算一次，靠记忆化保证）；空间 O(V+E)。
    【易错点】1) 第一步 Dijkstra 的源点是 **n**，不是 1——很容易看到"从节点 1 出发"
    就习惯性地把 1 当成源点，但 distanceToLastNode 的定义明确是"到节点 n 的距离"；
    2) DFS 计数时比较条件必须是**严格小于**（`dist[v] < dist[u]`），如果写成
    `<=`，会把"距离相同、事实上不构成合法受限路径"的边也当成合法转移，且相等的情况下
    还可能在两个方向都满足"<="，引入环导致递归不终止；3) 别忘了对 10^9+7 取模，
    路径数在稠密图上可能是指数级增长的大数。
    """
    MOD = 10**9 + 7
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    INF = float("inf")
    dist = [INF] * (n + 1)
    dist[n] = 0
    heap: list[tuple[int, int]] = [(0, n)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    memo: dict[int, int] = {}

    def dfs(u: int) -> int:
        if u == n:
            return 1
        if u in memo:
            return memo[u]
        total = 0
        for v, _w in graph[u]:
            if dist[v] < dist[u]:
                total = (total + dfs(v)) % MOD
        memo[u] = total
        return total

    return dfs(1)


def reachable_nodes_subdivided_graph(
    edges: list[list[int]], max_moves: int, n: int
) -> int:
    """
    【题意】n 个节点的无向图，每条边 edges[i]=[u,v,cnt] 被"细分"成 cnt 个新增中间
    节点（也就是把原来长度为 1 的一条边，拉长成一条含 cnt+1 段、cnt 个中间点的链）。
    从节点 0 出发，最多走 max_moves 步（每一步等价于原图/细分链上沿一条边移动到相邻
    节点），求能到达的节点总数（包括原图节点和新细分出来的中间节点）。
    【思路】直接在"细分后"的巨大图上跑 Dijkstra 太浪费——中间节点数量可能远超原图
    节点数。关键 insight 是拆成两部分算：(1) 把每条原图边 `(u,v,cnt)` 看成一条长度为
    `cnt+1` 的加权边（细分后走完整条边恰好要 cnt+1 步），在**原图节点**上跑一次
    Dijkstra，得到从节点 0 到每个原图编号节点的最短距离 `dist[]`；一个原图节点
    `i` 可达当且仅当 `dist[i] <= max_moves`。(2) 对每条原图边 `(u,v,cnt)`，这条边上
    有 cnt 个中间节点，从 u 这一侧最多能往里走 `max(0, max_moves - dist[u])` 个中间
    节点，从 v 这一侧最多能往里走 `max(0, max_moves - dist[v])` 个——这两段"够得着"
    的中间节点可能有重叠（如果两段之和已经 >= cnt，说明整条边上的中间节点都能被至少
    一侧覆盖到），所以这条边上实际可达的中间节点数是 `min(cnt, 两段之和)`，取 min
    正是为了避免在两段都够得着整条边时被重复计数。全部原图可达节点数，加上每条边上
    可达的中间节点数之和，就是最终答案。
    【复杂度】时间 O((V+E) log V)（Dijkstra 本身，跑在原图节点规模上，不需要真的
    展开中间节点）；空间 O(V+E)。
    【易错点】1) 千万不要真的把每条边展开成 cnt 个新节点再跑 Dijkstra——cnt 可能很大，
    这正是本题要求"只在原图节点规模上计算"的原因，边权直接用 `cnt+1` 表示"走完整条
    细分边需要的步数"；2) 计算某条边上可达的中间节点数时，两侧"够得着"的步数都要先
    和 0 取 max（如果 `dist[u] > max_moves`，说明从 u 这一侧根本走不到这条边上的任何
    中间节点，不能出现负数）；3) 最终这条边上可达节点数是 `min(cnt, a+b)` 而不是
    `a+b` 本身——`a+b` 可能超过这条边实际拥有的中间节点总数 cnt，多出来的部分是两侧
    重叠不该重复计入的。
    """
    graph: dict[int, dict[int, int]] = defaultdict(dict)
    for u, v, cnt in edges:
        graph[u][v] = cnt
        graph[v][u] = cnt

    INF = float("inf")
    dist = [INF] * n
    dist[0] = 0
    visited = [False] * n
    heap: list[tuple[int, int]] = [(0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, cnt in graph[u].items():
            nd = d + cnt + 1
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    reachable_nodes = sum(1 for d in dist if d <= max_moves)
    total_sub = 0
    for u, v, cnt in edges:
        a = max(0, max_moves - dist[u]) if dist[u] != INF else 0
        b = max(0, max_moves - dist[v]) if dist[v] != INF else 0
        total_sub += min(cnt, a + b)
    return reachable_nodes + total_sub


def minimum_moves_snake(grid: list[list[int]]) -> int:
    """
    【题意】n*n 网格里有一条占据两个相邻格子的"蛇"，初始水平躺在 (0,0)-(0,1)，目标是
    移动到 (n-1,n-2)-(n-1,n-1)。规则：水平状态下可以整体右移（需要右边再一格为空）
    或整体下移（需要正下方两格都为空）；水平状态下如果正下方两格都为空还可以顺时针
    旋转成竖直状态；竖直状态同理可以下移/右移/逆时针转回水平（对称规则）。求到达
    目标状态的最少移动次数，无法到达返回 -1。
    【思路】"最少移动次数"是最短路径的信号，但这里"节点"不再是网格上的一个坐标，而是
    "蛇的完整姿态"——用 `(r, c, orientation)` 三元组表示状态：`orientation=0`（水平）
    时蛇占据 `(r,c)` 和 `(r,c+1)`；`orientation=1`（竖直）时占据 `(r,c)` 和
    `(r+1,c)`。把每种姿态当成图上的一个"节点"，每一种合法动作（右移/下移/旋转）当成
    一条边权为 1 的边，问题就变成了在这张"状态图"上，从初始姿态到目标姿态的 BFS 最短
    路——因为所有边权都是 1，BFS 逐层扩展第一次到达目标状态时，层数就是最少移动次数，
    完全不需要 Dijkstra。核心技巧是把"物理规则"精确翻译成"状态转移"：水平蛇的下移和
    顺时针旋转共享同一个前提条件（正下方两格都为空），只是转移到的新状态不同（前者
    还是水平、后者变竖直），这提醒我们规则里"能不能动"和"动完变成什么姿态"是两件
    需要分别处理但共享前提判断的事情。
    【复杂度】时间 O(n^2)（状态总数是 O(n^2 * 2)——每个 (r,c) 位置乘以两种朝向，
    每个状态只入队一次，均摊 O(1) 转移）；空间 O(n^2)（visited 集合和 BFS 队列）。
    【易错点】1) 水平蛇的"下移"和"顺时针旋转"，虽然目标姿态不同，但都要求
    `(r+1,c)` 和 `(r+1,c+1)` 同时为空——这是最容易在写代码时漏掉的对称关系，容易
    只检查了移动方向上的空格而漏了旋转所需的另一侧；2) 状态用三元组 `(r,c,o)` 去重，
    如果只用 `(r,c)` 去重会把"同一个位置、不同朝向"的两个完全不同的状态错误地合并成
    一个，导致漏掉某些合法路径；3) 目标状态是"竖直方向蛇占据 (n-1,n-2)-(n-1,n-1)"
    吗？不是——目标终点是水平朝向，占据 (n-1,n-2) 和 (n-1,n-1) 这两个格子，也就是
    `orientation=0`，容易搞混目标状态的朝向。
    """
    n = len(grid)

    def is_open(r: int, c: int) -> bool:
        return 0 <= r < n and 0 <= c < n and grid[r][c] == 0

    start = (0, 0, 0)
    target = (n - 1, n - 2, 0)
    if start == target:
        return 0
    visited = {start}
    queue: deque[tuple[int, int, int]] = deque([start])
    moves = 0
    while queue:
        moves += 1
        for _ in range(len(queue)):
            r, c, o = queue.popleft()
            candidates: list[tuple[int, int, int]] = []
            if o == 0:
                if is_open(r, c + 2):
                    candidates.append((r, c + 1, 0))
                if is_open(r + 1, c) and is_open(r + 1, c + 1):
                    candidates.append((r + 1, c, 0))
                    candidates.append((r, c, 1))
            else:
                if is_open(r + 2, c):
                    candidates.append((r + 1, c, 1))
                if is_open(r, c + 1) and is_open(r + 1, c + 1):
                    candidates.append((r, c + 1, 1))
                    candidates.append((r, c, 0))
            for nxt in candidates:
                if nxt == target:
                    return moves
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
    return -1


def minimum_cost_grid_path(grid: list[list[int]]) -> int:
    """
    【题意】m*n 网格，grid[i][j] 是一个方向标记（1=右 2=左 3=下 4=上），表示"站在这个
    格子上，默认下一步会被推向哪个相邻格子"。从 (0,0) 出发，沿着标记方向走能到
    (m-1,n-1) 就是一条"有效路径"，代价为 0；你可以花费 1 的代价修改任意格子的标记方向
    （改成你需要的方向），求让 (0,0) 到 (m-1,n-1) 存在至少一条有效路径所需的最小总
    代价。
    【思路】把每个格子看成图上一个节点，从格子 (r,c) 出发，往它标记的方向走一步，边权
    是 0（顺着标记走不用改）；往其余三个方向走，边权是 1（需要把这个格子的标记改成
    对应方向）。这样问题变成了标准的"单源最短路径"，但边权只有 0 和 1 两种取值——
    这种特殊情况下可以用比 Dijkstra 更轻量的 **0-1 BFS**：用双端队列代替 Dijkstra 的
    二叉堆，遇到 0 权边就把新状态加入队列**前端**（相当于"优先处理，因为它不增加总
    代价，应该和当前节点享有同等的优先级"），遇到 1 权边就加入队列**后端**（代价更高，
    排到后面处理）。因为队列里任意时刻的代价值最多相差 1（要么是当前层、要么是当前层
    +1），这个"双端插入"的技巧保证了出队顺序始终是按代价非递减排列的，效果等价于
    Dijkstra，但因为不需要堆的 O(log n) 调整，整体是纯线性的 O(节点数+边数)。
    【复杂度】时间 O(m*n)（0-1 BFS 对每个节点、每条边只需要常数次处理，不需要堆的
    对数开销）；空间 O(m*n)。
    【易错点】1) 0-1 BFS 的正确性依赖"用双端队列 + 0 权推前/1 权推后"这个具体规则，
    如果写成普通 BFS（所有新状态都推到队尾）就退化成了对边权不敏感的层序遍历，会
    算出错误的最小代价；2) 判断"顺着标记走"和"逆着标记走"时，方向编号 1/2/3/4 分别
    对应右/左/下/上，容易在 `(dr, dc)` 的映射上写反；3) 到达 (m-1, n-1) 之后应该
    直接读 `cost[m-1][n-1]`，不需要也不应该等到整个网格所有格子都处理完才停止——虽然
    这里为了教学清晰没有加"提前退出"优化，但要清楚这只是常数级的效率取舍，不影响
    正确性。
    """
    m, n = len(grid), len(grid[0])
    dirs = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    INF = float("inf")
    cost = [[INF] * n for _ in range(m)]
    cost[0][0] = 0
    dq: deque[tuple[int, int]] = deque([(0, 0)])
    while dq:
        r, c = dq.popleft()
        for d, (dr, dc) in dirs.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                w = 0 if grid[r][c] == d else 1
                if cost[r][c] + w < cost[nr][nc]:
                    cost[nr][nc] = cost[r][c] + w
                    if w == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))
    return cost[m - 1][n - 1]


def second_minimum_time_to_reach(
    n: int, edges: list[list[int]], time: int, change: int
) -> int:
    """
    【题意】n 个节点组成无向连通图，每条边通过耗时都是 time 分钟；每个节点上有一个
    红绿灯，每隔 change 分钟在红/绿之间切换一次（所有节点同步切换，初始都是绿灯），
    只能在绿灯时离开一个节点（红灯时只能停留等待，不能停在半路）。求从节点 1 到节点 n
    的"严格次短"时间（次短定义为"严格大于最短时间的最小值"，即使最短路径有多条也
    只算一次，需要真正走出一条更长的路径或走出"来回一趟"的绕路）。
    【思路】因为每条边耗时相同，"到达某节点最少经过多少条边"和"到达它的最短时间"是
    等价的，可以先用 BFS 算出"层数"意义下的最短和次短路径长度：对每个节点维护两个
    距离 `dist1`（最短）和 `dist2`（次短，要求严格大于 dist1），BFS 逐层扩展，一个
    节点第一次被访问时填入 `dist1`；后续再次被访问到时，只要这次的层数和 `dist1`
    不同（说明是走了一条不同长度的路径到达，而不是同一最短路径的另一种走法），就
    可以把这个层数记为 `dist2`。这样即使图中不存在"天然更长"的次短路径（比如整张图
    是一条链，唯一路径就是最短路径本身），BFS 也能通过"来回走一条边再折返"（层数
    = 最短层数 + 2）自然地找到这个"次短"（因为无向图里从 u 走到邻居再走回 u 恰好
    多花 2 层，且天然严格大于最短层数）。拿到 `dist2[n]`（次短需要经过的边数）之后，
    再模拟"走这么多条边、每条边耗时 time 分钟，每次到达新节点后如果恰逢红灯就要
    等到下一次变绿"这个过程，换算出真实时间：当前累计时间对 `2*change` 取模，如果
    落在 `[change, 2*change)` 区间说明当前是红灯，需要先等到这一轮变绿（补足到下一个
    `2*change` 的整数倍），再加上一段 `time`。
    【复杂度】时间 O(V+E)（BFS 部分）+ O(次短边数)（模拟红绿灯部分，边数上界是
    O(V)）；空间 O(V+E)。
    【易错点】1) "次短"是严格大于最短的**最小值**，不是"最短路径数量为 1 时就必须
    绕路 2 层"这么简单的特判——正确做法是让 BFS 自然地记录"第二种不同层数抵达该
    节点"的情况，不需要单独判断"是否存在天然次短路径"这两种情况，统一处理更不容易
    出错；2) 红绿灯的判断必须用"当前已用时间 mod (2*change)"，不能只看"已经过了多少
    条边"，因为每条边耗时相同但红绿灯周期和边耗时未必对齐；3) 处于红灯时要等到下一次
    变绿即可出发，不能等待"一整个周期"，等待时长是 `2*change - (t mod 2*change)`，
    多算一整个周期或少算都会导致时间计算偏差。
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    dist1 = [-1] * (n + 1)
    dist2 = [-1] * (n + 1)
    dist1[1] = 0
    queue: deque[tuple[int, int]] = deque([(1, 0)])
    while queue:
        node, d = queue.popleft()
        nd = d + 1
        for nb in graph[node]:
            if dist1[nb] == -1:
                dist1[nb] = nd
                queue.append((nb, nd))
            elif dist2[nb] == -1 and nd != dist1[nb]:
                dist2[nb] = nd
                if nb == n:
                    break
                queue.append((nb, nd))

    moves = dist2[n]
    result = 0
    for _ in range(moves):
        remainder = result % (2 * change)
        if remainder >= change:
            result += 2 * change - remainder
        result += time
    return result


def _self_test() -> None:
    assert (
        min_cost_to_reach_destination_in_time(
            30,
            [[0, 1, 10], [1, 2, 10], [2, 5, 10], [0, 3, 1], [3, 4, 10], [4, 5, 15]],
            [5, 1, 2, 20, 20, 3],
        )
        == 11
    )
    assert (
        min_cost_to_reach_destination_in_time(
            29,
            [[0, 1, 10], [1, 2, 10], [2, 5, 10], [0, 3, 1], [3, 4, 10], [4, 5, 15]],
            [5, 1, 2, 20, 20, 3],
        )
        == 48
    )

    assert (
        number_of_restricted_paths(
            5,
            [[1, 2, 3], [1, 3, 3], [2, 3, 1], [1, 4, 2], [5, 2, 2], [3, 5, 1], [5, 4, 10]],
        )
        == 3
    )
    assert (
        number_of_restricted_paths(
            7,
            [
                [1, 3, 1],
                [4, 1, 2],
                [7, 3, 4],
                [2, 5, 3],
                [5, 6, 1],
                [6, 7, 2],
                [7, 5, 3],
                [2, 6, 4],
            ],
        )
        == 1
    )

    assert (
        reachable_nodes_subdivided_graph([[0, 1, 10], [0, 2, 1], [1, 2, 2]], 6, 3) == 13
    )
    assert (
        reachable_nodes_subdivided_graph(
            [[0, 1, 4], [1, 2, 6], [0, 2, 8], [1, 3, 1]], 10, 4
        )
        == 23
    )

    grid1 = [
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
    ]
    assert minimum_moves_snake(grid1) == 11
    grid2 = [
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
    ]
    assert minimum_moves_snake(grid2) == 9

    assert (
        minimum_cost_grid_path(
            [[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1], [2, 2, 2, 2]]
        )
        == 3
    )
    assert minimum_cost_grid_path([[1, 1, 3], [3, 2, 2], [1, 1, 4]]) == 0

    assert (
        second_minimum_time_to_reach(
            5, [[1, 2], [1, 3], [1, 4], [3, 4], [4, 5]], 3, 5
        )
        == 13
    )
    assert second_minimum_time_to_reach(2, [[1, 2]], 3, 2) == 11

    print(
        "[PASS] p22_advanced_graph_iii: 6/6 题通过 "
        "(规定时间内到达终点的最小花费/从第一个节点出发到最后一个节点的受限路径数/"
        "细分图中的可到达节点/穿过迷宫的最少移动次数/"
        "使网格图至少有一条有效路径的最小代价/到达目的地的第二短时间)"
    )


if __name__ == "__main__":
    _self_test()
