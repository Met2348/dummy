"""图 BFS/DFS 专题 Part III（竞赛级补充）：颜色交替的最短路径 / 尽量减少恶意软件的
传播 / 不邻接植花 / 最小体力消耗路径 / 迷宫中离入口最近的出口 / 网格中的最短路径-
有障碍物消除 / 获取所有钥匙的最短路径 / 推箱子 / 从盒子里获得的最大糖果数。

不重复讲 Part I/II 已经建立的"BFS 求最短路、DFS 求连通性"选择依据，本文件聚焦这批
题共享的一条核心线索——**扩展状态空间**：普通 BFS 的状态只是"在哪个格子/节点"，
但这批题里"到过同一个格子"不代表"处于同一种情况"，必须把额外信息（走到这里时
带的边颜色、剩余的障碍消除次数、已经收集到的钥匙集合、箱子和玩家各自的位置）一起
编码进 visited 的状态里，BFS 才不会漏掉"看似重复、其实处境不同"的合法路径。这也是
真实 Frontier Lab 面试里"看起来是道简单 BFS 题，追问一句就要加状态"的经典陷阱。
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque


# ── LC1129 颜色交替的最短路径 ────────────────────────────────────────────
def shortest_alternating_paths(
    n: int, red_edges: list[list[int]], blue_edges: list[list[int]]
) -> list[int]:
    """
    【题意】给定 n 个节点（编号 0~n-1）的有向图，边分两种颜色（红/蓝），分别由
    `red_edges`、`blue_edges` 给出；求从节点 0 出发，到每个节点 x 的"颜色交替
    路径"（相邻两条边必须一红一蓝，交替进行）的最短长度，如果不存在这样的路径，
    对应位置填 -1。
    【思路】本类"扩展状态空间"最直接的一个例子：如果只用"当前在哪个节点"作为
    BFS 的状态，会漏掉"同一个节点，通过不同颜色的边到达"这两种截然不同的处境——
    比如节点 3 如果是"刚经过一条红边到达"，下一步只能走蓝边；如果是"刚经过一条
    蓝边到达"，下一步只能走红边，这是两种完全不同的后续可能性，不能合并成一个
    状态去重。于是把 BFS 的状态从 `node` 扩展成 `(node, last_color)`，`visited`
    也变成对 `(node, color)` 二元组去重。初始时节点 0 同时以"红/蓝两种虚拟起始
    颜色"入队（因为从 0 出发的第一步，可以选红边也可以选蓝边，两者都合法），这样
    第一步既能走红边也能走蓝边。用 `dist[node][color]` 记录"以 color 结尾到达
    node 的最短步数"，最终每个节点的答案是它两种颜色状态里的较小值（如果两种
    颜色都不可达才是 -1）。
    【复杂度】时间 O(V+E)（每个 `(node, color)` 状态最多入队一次，V 个节点、每种
    颜色各一份，E 条边各检查一次）；空间 O(V+E)。
    【易错点】1) 只用 `node` 做 visited 去重（照抄普通 BFS 的写法），会把"红边
    到达"和"蓝边到达"这两种处境合并成一个状态，一旦其中一种先被访问，另一种
    颜色本该存在的更优路径就被错误地剪掉；2) 初始状态忘记让节点 0 同时以两种
    颜色入队，只以一种颜色开始，会让另一种颜色的第一步永远走不出去；3) 求某个
    节点的最终答案时忘记在两个颜色状态里取较小值，直接固定用某一种颜色的
    `dist`，会把"这个节点其实可以通过另一种颜色的路径更快到达"的情况漏掉。
    """
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v in red_edges:
        graph[u].append((v, 0))
    for u, v in blue_edges:
        graph[u].append((v, 1))

    dist = [[-1, -1] for _ in range(n)]
    dist[0] = [0, 0]
    visited = {(0, 0), (0, 1)}
    q = deque([(0, 0), (0, 1)])
    while q:
        node, color = q.popleft()
        for nxt, edge_color in graph[node]:
            if edge_color != color and (nxt, edge_color) not in visited:
                visited.add((nxt, edge_color))
                dist[nxt][edge_color] = dist[node][color] + 1
                q.append((nxt, edge_color))

    ans: list[int] = []
    for i in range(n):
        candidates = [d for d in dist[i] if d != -1]
        ans.append(min(candidates) if candidates else -1)
    return ans


# ── LC924 尽量减少恶意软件的传播 ─────────────────────────────────────────
def min_malware_spread(graph: list[list[int]], initial: list[int]) -> int:
    """
    【题意】给定 n 个节点的无向图（用 n×n 邻接矩阵 graph 表示），`initial` 是一批
    初始被恶意软件感染的节点；感染规则是"两个直接相连的节点，只要有一个被感染，
    另一个最终也会被感染"，感染会持续扩散直到不能再传播为止。从 `initial` 里
    恰好移除一个节点（移除后它就不再是"初始感染源"，但如果它仍和其他感染源连通，
    还是会被传染回来），求移除哪一个节点能让最终被感染的节点总数最少；如果有多个
    节点效果相同，返回下标最小的那个。
    【思路】感染会传满"整个连通分量"，所以真正决定答案的不是节点本身，而是它所在
    的**连通分量**：用并查集把整张图按连通性分组，同一个分量内的所有节点要么全部
    被感染，要么（如果分量里不含任何 `initial` 节点）全部不被感染。对每个连通
    分量，如果里面恰好只有一个 `initial` 节点（"独占"这个分量），移除它就能让
    整个分量幸免于难，收益等于这个分量的大小；如果一个分量里有两个或以上的
    `initial` 节点，移除其中任意一个都没用——分量还是会被剩下的感染源点燃，收益
    为 0。于是只需要在"独占某个分量"的 `initial` 节点里，选分量最大的那个；如果
    没有任何节点独占一个分量（所有 `initial` 节点都和别的 `initial` 节点共享
    分量），移除谁都不影响最终感染数，按题目要求返回下标最小的 `initial` 节点。
    【复杂度】时间 O(n²·α(n))（邻接矩阵本身就是 O(n²)，并查集操作均摊接近
    O(1)）；空间 O(n)（并查集的 parent/size 数组）。
    【易错点】1) 误以为"移除度数最大的 initial 节点"或者"移除被感染节点数最多的
    那个"就是答案——这两种直觉都不对,真正要比较的是"这个节点独占的连通分量有
    多大",如果它的分量里还有别的 initial 节点,移除它完全没有收益;2) 遍历
    `initial` 时如果不按下标升序处理、且用 `>=` 而不是严格 `>` 比较分量大小，
    可能在同等大小的分量之间选出下标更大的节点，违反"多个候选选下标最小"这一
    要求；3) 忘记处理"没有任何节点独占分量"这一兜底情况，直接返回未初始化的
    变量或者最后一次循环残留的值，得到不确定的错误答案。
    """
    n = len(graph)
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] == 1:
                union(i, j)

    root_count = Counter(find(node) for node in initial)
    best_node = min(initial)
    best_size = -1
    for node in sorted(initial):
        root = find(node)
        if root_count[root] == 1 and size[root] > best_size:
            best_size = size[root]
            best_node = node
    return best_node


# ── LC1042 不邻接植花 ────────────────────────────────────────────────────
def garden_no_adj(n: int, paths: list[list[int]]) -> list[int]:
    """
    【题意】有 n 个花园（1-indexed，编号 1~n），`paths` 描述若干双向小路直接连接
    两个花园；每个花园里种一种花（编号 1~4），要求任意一条小路连接的两个花园
    种的花不同；题目保证每个花园直接相连的花园数不超过 3，且保证一定存在合法
    方案，返回任意一种合法的种花方案。
    【思路】这是一道"伪装成图论"的贪心/构造题，之所以放进图 BFS/DFS 专题，是
    因为第一步仍然是标准的"编号+边列表建邻接表"（和课程表、寻找图中是否存在
    路径是同一套建图方式）。建完图后不需要 BFS/DFS 遍历，只需要按花园编号顺序
    逐个决定颜色：对花园 `i`，看它已经确定颜色的邻居们用掉了哪些颜色（`used`
    集合），从 1~4 里选一个不在 `used` 里的颜色。**为什么这一定能选到**——题目
    保证每个花园最多有 3 条边，也就是最多 3 个邻居，而颜色一共有 4 种，最坏情况
    3 个邻居占满 3 种颜色，也一定还剩至少 1 种颜色可用，不需要回溯或试错。
    【复杂度】时间 O(V+E)（建图 O(E)，之后每个节点只需要看一遍邻居 O(度数)，
    总和是 O(E)）；空间 O(V+E)。
    【易错点】1) 题目节点编号是 1-indexed，如果建图或结果数组的下标处理不一致
    （比如建图用 1-indexed 但结果数组按 0-indexed 输出），会导致颜色赋值和最终
    返回的花园编号错位；2) 误以为需要回溯/试错才能保证合法（这是很多新手看到
    "图染色"就联想到的第一反应），实际上"度数上限 3 < 颜色数 4"这个约束保证了
    贪心一定成功，不需要回溯，写成回溯反而是不必要的过度设计；3) 处理邻居颜色
    时如果邻居还没被处理过（颜色初始值是 0，表示"还没分配"），不能把 0 也当成
    "已占用的颜色"排除掉——本实现里颜色范围是 1~4，`used` 集合天然不会包含 0，
    不会误伤，但如果换成从 0 开始编号颜色就需要格外小心这个边界。
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for x, y in paths:
        graph[x - 1].append(y - 1)
        graph[y - 1].append(x - 1)

    ans = [0] * n
    for garden in range(n):
        used = {ans[neighbor] for neighbor in graph[garden]}
        for flower in range(1, 5):
            if flower not in used:
                ans[garden] = flower
                break
    return ans


# ── LC1631 最小体力消耗路径 ──────────────────────────────────────────────
def min_effort_path(heights: list[list[int]]) -> int:
    """
    【题意】给定 rows×cols 的高度矩阵 heights，从左上角 (0,0) 走到右下角
    (rows-1,cols-1)，每步只能上下左右移动；一条路径的"体力消耗"定义为路径上
    相邻两格高度差绝对值的**最大值**（不是总和），求体力消耗最小的路径对应的
    体力值。
    【思路】这不是"边权求和的最短路"，而是"边权取最大值的最短路"——目标函数从
    "sum" 变成了 "max"，普通 BFS/Dijkstra 求的是路径权重和最小，这里要换成
    **改造版 Dijkstra**：用小根堆存 `(当前路径上已经出现过的最大高度差, r, c)`，
    每次弹出堆顶（当前已知"体力消耗"最小的格子），如果这个格子就是终点直接
    返回；否则向四个方向扩展，新状态的"体力消耗"是
    `max(当前路径的体力消耗, 新旧格子高度差)`——注意这里是取 max 而不是像普通
    Dijkstra 那样把边权累加,只有当这个新的体力消耗比"已知到达该格子的最优体力
    消耗"更优时才更新并入堆（标准的 Dijkstra 松弛操作，只是"松弛"用的是 max
    而不是 +）。
    【复杂度】时间 O(R·C·log(R·C))（每个格子最多以更优的体力值入堆几次，堆
    操作 O(log(R·C))）；空间 O(R·C)（effort 矩阵 + 堆）。
    【易错点】1) 照抄普通 Dijkstra 的松弛公式 `new_cost = cur_cost + weight`
    （边权求和），这里必须换成 `new_cost = max(cur_cost, weight)`——本题的目标
    函数是路径上的最大边权，不是边权总和，用错公式会把"体力消耗"算成"体力消耗
    总和"，得到偏大很多的错误答案；2) 忘记堆里弹出的记录可能是"过期"的（同一个
    格子曾经以更差的体力值入过堆），如果不加 `cur_effort > effort[r][c]: continue`
    这一判断,会用过期数据继续扩展,浪费计算但不影响最终正确性(因为 Dijkstra
    保证第一次弹出某格子时就是最优解，后续过期记录的扩展不会产生更优结果，只是
    冗余)；3) 误以为这是无权图，直接套用 BFS 逐层扩展——BFS 假设每条边代价
    相同，但这里每条边的"代价"是高度差，大小不一，必须用堆维护"当前最优"而不是
    普通队列。
    """
    rows, cols = len(heights), len(heights[0])
    effort = [[float("inf")] * cols for _ in range(rows)]
    effort[0][0] = 0
    heap: list[tuple[int, int, int]] = [(0, 0, 0)]
    while heap:
        cur_effort, r, c = heapq.heappop(heap)
        if (r, c) == (rows - 1, cols - 1):
            return cur_effort
        if cur_effort > effort[r][c]:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_effort = max(cur_effort, abs(heights[nr][nc] - heights[r][c]))
                if new_effort < effort[nr][nc]:
                    effort[nr][nc] = new_effort
                    heapq.heappush(heap, (new_effort, nr, nc))
    return 0


# ── LC1926 迷宫中离入口最近的出口 ────────────────────────────────────────
def nearest_exit(maze: list[list[str]], entrance: list[int]) -> int:
    """
    【题意】给定 m×n 的迷宫 maze（'.' 是空地，'+' 是墙），起点是 `entrance`；
    "出口"定义为迷宫**边界上**的某个空地格子，且不能是入口本身；每步只能上下
    左右移动到相邻空地，求从入口到最近出口的最少步数，不存在则返回 -1。
    【思路】标准的无权图最短路 BFS，本题的陷阱全在"出口"这个概念的边界条件：
    1) 出口必须是边界格子（`r==0 or r==rows-1 or c==0 or c==cols-1`），内部的
    空地不算出口，哪怕它是"死胡同"；2) 入口本身即使恰好在边界上，也**不算**
    出口——如果直接把入口的四个邻居当普通格子处理，会漏掉"入口本身在边界"这
    个特判。写法上从入口出发做标准 BFS，每次扩展到一个新的、未访问过的空地时，
    先检查它是否满足"在边界上"，满足就立刻返回当前步数+1；不满足就正常入队
    继续扩展。因为已经把"入口自身"排除在外（判断的是"新扩展到的格子"，不包括
    起点），不需要额外写"排除入口"的特判代码，起点本身从来不会被当成候选出口
    去检查。
    【复杂度】时间 O(R·C)（每个格子最多入队一次）；空间 O(R·C)（visited 集合 +
    队列）。
    【易错点】1) 把"入口在边界上是否要特判成出口"这件事搞反——如果把起点也
    放进"检查是否为出口"的逻辑里，会在还没迈出任何一步时就错误地返回 0；本
    实现通过"只检查新扩展到的邻居格子，不检查起点自身"天然规避了这个问题，
    但如果换一种写法（比如先把起点入队再统一检查）就必须显式排除起点；2) 忘记
    "出口"要求是边界格子这个约束，把任何"能到达且是空地"的格子都当成出口，
    第一层扩展到的第一个空地就会被误判为答案；3) 判断"是否在边界"时写错四个
    方向的条件（比如漏掉 `c==0` 这一支只判断了行不判断列），会漏掉从左右两侧
    边界离开的合法出口。
    """
    rows, cols = len(maze), len(maze[0])
    er, ec = entrance
    visited = {(er, ec)}
    q = deque([(er, ec, 0)])
    while q:
        r, c, steps = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and maze[nr][nc] == "."
                and (nr, nc) not in visited
            ):
                if nr == 0 or nr == rows - 1 or nc == 0 or nc == cols - 1:
                    return steps + 1
                visited.add((nr, nc))
                q.append((nr, nc, steps + 1))
    return -1


# ── LC1293 网格中的最短路径-有障碍物消除 ─────────────────────────────────
def shortest_path_with_obstacle_elimination(grid: list[list[int]], k: int) -> int:
    """
    【题意】给定 m×n 的 0/1 网格 grid（0 空地、1 障碍），最多可以消除 k 个障碍
    （消除后那一格就能正常走），求从 (0,0) 走到 (m-1,n-1) 的最少步数，不可能到达
    时返回 -1。
    【思路】本类"扩展状态空间"的又一个典型：如果只用 `(r,c)` 做 visited，会漏掉
    "同一个格子，用不同数量的剩余消除次数到达"这两种不同处境——比如以剩 3 次
    消除机会到达 (2,2)，和以剩 0 次消除机会到达 (2,2)，后续能走的路完全不同,
    不能因为"格子相同"就认为是同一个状态而合并。把 BFS 状态扩展成
    `(r, c, remain)`（`remain` 是到达这里时还剩多少次消除机会），`visited` 也
    对这个三元组去重；每走一步,如果目标格子是障碍就消耗一次 remain（`remain -
    grid[nr][nc]`，因为 grid 的值恰好是 0/1，天然可以直接相减代替"if 是障碍则
    -1"的分支判断），只要消耗后 `remain>=0` 就是合法状态。有一个重要的提前
    返回优化：如果 `k >= rows+cols-2`（终点到起点的曼哈顿距离，也是不消除任何
    障碍时最少要走的步数），消除次数已经充裕到可以把沿途所有障碍都清空，直接
    走曼哈顿距离最短路径即可，答案就是 `rows+cols-2`，不需要真的跑 BFS。
    【复杂度】时间 O(R·C·K)（状态数是 `(r,c,remain)` 的组合，最坏 O(R·C·K)个
    不同状态，每个状态检查 4 个方向）；空间同量级（visited 集合 + 队列）。
    【易错点】1) 只用 `(r,c)` 做 visited（照抄普通 01 矩阵/腐烂的橘子的写法），
    会把"剩余消除次数不同"的两种处境错误地合并成一个状态，一旦某个格子先被
    "消除次数较少"的路径访问过，"消除次数更多、其实还能走更远"的路径就被
    误判为"已访问"而提前剪掉，可能得到偏大甚至错误的 -1；2) 漏掉 `k >=
    rows+cols-2` 这一优化不算错误（不加也能算出正确答案），但在 k 很大、网格
    也较大时,状态空间 `(r,c,remain)` 会显著膨胀,不加这条优化容易在真实数据上
    超时；3) 用 `remain - grid[nr][nc]` 而不是先判断"是不是障碍"再消耗，逻辑上
    等价但如果误写成 `remain - 1`（不管目标格子是不是障碍都消耗一次），会把
    "走进空地"也错误地当成消耗了一次消除机会。
    """
    rows, cols = len(grid), len(grid[0])
    if k >= rows + cols - 2:
        return rows + cols - 2

    visited = {(0, 0, k)}
    q = deque([(0, 0, k, 0)])
    while q:
        r, c, remain, steps = q.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return steps
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nremain = remain - grid[nr][nc]
                if nremain >= 0 and (nr, nc, nremain) not in visited:
                    visited.add((nr, nc, nremain))
                    q.append((nr, nc, nremain, steps + 1))
    return -1


# ── LC864 获取所有钥匙的最短路径（重点难题：状态压缩 BFS）────────────────
def shortest_path_all_keys(grid: list[str]) -> int:
    """
    【题意】给定 m×n 的网格 grid：`'.'` 空地、`'#'` 墙、`'@'` 起点、小写字母是
    钥匙、大写字母是对应的锁（`'a'`/`'A'` 是一对，最多到 `'f'`/`'F'`，钥匙总数
    1~6 把）；每步上下左右移动，走到钥匙格子自动拾取，没有对应钥匙不能走进锁
    格子；求拿到所有钥匙所需的最少步数，不可能拿全返回 -1。
    【思路】本类"扩展状态空间"最经典的载体——**状态压缩（bitmask）BFS**。如果
    只用 `(r,c)` 做 visited，会漏掉"同一个格子，携带的钥匙集合不同"这一关键
    差异：比如起点附近有把锁 `'A'`，第一次经过时没有钥匙 `'a'`，只能绕路；等
    绕到别处捡到 `'a'` 之后再回到锁 `'A'` 门口，这时候是可以推门而入的——如果
    visited 只记录坐标,第一次经过就会把这个格子标记为"访问过",导致携带了钥匙
    之后重新经过同一格子被错误地剪掉,从而找不到真正的最短路。解法是把 BFS 的
    状态从 `(r,c)` 扩展成 `(r, c, keys)`，`keys` 是一个整数位掩码，第 i 位为 1
    表示"已经拿到第 i 把钥匙"（`'a'` 对应第 0 位，`'b'` 对应第 1 位，以此类推）。
    每走到一个新格子：如果是墙直接跳过；如果是锁且当前 `keys` 里没有对应位，
    说明这道锁打不开，跳过；如果是钥匙，把 `keys` 对应位置 1（`keys | (1 <<
    idx)`）得到新的钥匙集合；用新的 `(nr, nc, nkeys)` 三元组判断是否访问过。
    当某个状态的 `keys` 等于"全部钥匙都拿到"的目标掩码（用一开始扫描网格时
    统计出的钥匙总数 `all_keys` 判断）时，当前步数就是答案。
    【复杂度】时间 O(R·C·2^K)（K 是钥匙数、最多 6，状态数是坐标数乘以 2^K 种
    钥匙组合，每个状态检查 4 个方向）；空间同量级（visited 集合 + 队列）。
    【易错点】1) 只用 `(r,c)` 做 visited（这是本题最容易踩的坑，因为大多数网格
    BFS 题——岛屿、腐烂的橘子、01矩阵——坐标本身就足以代表状态，很容易惯性地
    照搬），会把"带着不同钥匙集合经过同一格子"的情况错误合并，某些必须绕远路
    再折返的最优解会被漏掉；2) 判断"是否是锁"和"是否有对应钥匙"时，字母和位
    掩码的映射算错（比如 `'A'` 应该查 `keys` 的第 0 位而不是第 1 位，`ord(ch) -
    ord('A')` 才是正确的位下标），会导致明明有钥匙却被判定为锁着，或者反过来
    没钥匙也能通过；3) 判断"是否集齐所有钥匙"时,用"当前 keys 的二进制 1 的个数
    是否等于钥匙总数"来判断也可行,但更直接、更不容易出错的写法是提前用网格里
    实际出现的钥匙种类算出目标掩码 `all_keys`,直接比较 `keys == all_keys`,而
    不是去数 popcount。
    """
    rows, cols = len(grid), len(grid[0])
    start = (0, 0)
    all_keys = 0
    for r in range(rows):
        for c in range(cols):
            ch = grid[r][c]
            if ch == "@":
                start = (r, c)
            elif ch.islower():
                all_keys |= 1 << (ord(ch) - ord("a"))

    visited = {(start[0], start[1], 0)}
    q = deque([(start[0], start[1], 0, 0)])
    while q:
        r, c, keys, steps = q.popleft()
        if keys == all_keys:
            return steps
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            ch = grid[nr][nc]
            if ch == "#":
                continue
            if ch.isupper() and not (keys >> (ord(ch) - ord("A")) & 1):
                continue
            nkeys = keys | (1 << (ord(ch) - ord("a"))) if ch.islower() else keys
            state = (nr, nc, nkeys)
            if state not in visited:
                visited.add(state)
                q.append((nr, nc, nkeys, steps + 1))
    return -1


# ── LC1263 推箱子（重点难题：BFS 状态空间扩展）──────────────────────────
def min_push_box(grid: list[list[str]]) -> int:
    """
    【题意】仓库推箱子游戏：网格里 `'#'` 是墙、`'.'` 是空地、`'S'` 是玩家起始
    位置、`'B'` 是箱子起始位置、`'T'` 是箱子要到达的目标；玩家每步可以走到相邻
    的非墙格子，如果玩家走向箱子所在的方向，箱子会被"推"到箱子的下一格（前提
    是箱子的下一格也不是墙且没超出边界），玩家不能穿过箱子；求把箱子推到目标
    所需的最少推动次数（不是玩家的移动步数），无法完成返回 -1。
    【思路】本题的状态如果只记录"箱子在哪"，会严重低估问题的复杂度——箱子能不能
    被继续推、往哪个方向推，取决于玩家能不能先走到箱子"背后"那一格,而玩家能不能
    走到那一格,又依赖箱子当前挡没挡路。所以必须把状态**同时**扩展成
    "箱子位置 + 玩家位置" 四元组 `(box_r, box_c, player_r, player_c)`——这是比
    864 的"坐标+钥匙集合"更进一步的状态扩展：这里额外信息本身就是"另一个实体的
    完整坐标"，而不是一个压缩的位掩码。外层 BFS 按"推动次数"分层：对当前状态,
    尝试把箱子往四个方向推——箱子从 `(br,bc)` 被推向 `(nbr,nbc)`，前提是
    `(nbr,nbc)` 本身可走（不是墙、不越界），且玩家必须能够先走到"箱子背后"那一
    格 `(ppr,ppc)=(br-dr, bc-dc)`（推动方向的反方向,即推之前玩家要站的位置）
    ——这一步"玩家能否从当前 `(pr,pc)` 走到 `(ppr,ppc)`,且中途不能穿过箱子
    `(br,bc)`"本身又是一次独立的 BFS/DFS（在 `can_reach` 里实现），是"BFS 里
    嵌套一次子 BFS"的写法。如果可以推动，新状态是 `(nbr, nbc, br, bc)`——推动
    之后箱子到了新位置,玩家则站到了箱子被推动前的位置（也就是原来箱子的位置）。
    【复杂度】时间 O((R·C)²)（外层状态数是 O(R·C)——箱子位置一定，玩家实际上
    只需要区分"箱子的哪一侧"最多 4 种,但保守估计是 R·C 个玩家位置,所以状态总数
    O((R·C)²)；每个状态还要跑一次 O(R·C) 的可达性子 BFS，总体 O((R·C)³) 更保守，
    但箱子位置和玩家位置的组合在实践中远小于这个上界）；空间同量级（visited
    集合 + 队列 + 子 BFS 的临时 visited）。
    【易错点】1) 状态只记录箱子位置、不记录玩家位置——这是本题最容易漏掉的
    扩展，会误以为"箱子在同一个位置就是同一个状态"，但玩家站在箱子的不同侧面，
    能推动箱子的方向完全不同，合并状态会把"暂时推不动但换个方向站位就能推动"
    的情况错误地剪掉；2) 检查"玩家能否走到箱子背后"时，如果子 BFS 忘记把箱子
    当前位置当成"临时的墙"（玩家不能穿过箱子本体），会算出玩家可以直接穿过箱子
    走到对面，得到过于乐观、错误的可达性判断；3) 推动后的新玩家位置弄错——
    推箱子一次之后,玩家应该站在箱子被推动前的旧位置（因为玩家紧跟在箱子后面
    推它),而不是新算出来的"箱子背后"位置 `(ppr,ppc)`,这两个位置在写代码时
    容易搞混。
    """
    rows, cols = len(grid), len(grid[0])
    player = box = target = (0, 0)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                player = (r, c)
            elif grid[r][c] == "B":
                box = (r, c)
            elif grid[r][c] == "T":
                target = (r, c)

    def is_free(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "#"

    def can_reach(
        player_pos: tuple[int, int], box_pos: tuple[int, int], dest: tuple[int, int]
    ) -> bool:
        if dest == player_pos:
            return True
        seen = {player_pos}
        stack = deque([player_pos])
        while stack:
            r, c = stack.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if is_free(nr, nc) and (nr, nc) != box_pos and (nr, nc) not in seen:
                    if (nr, nc) == dest:
                        return True
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return False

    start = (box[0], box[1], player[0], player[1])
    visited = {start}
    q = deque([(start, 0)])
    while q:
        (br, bc, pr, pc), pushes = q.popleft()
        if (br, bc) == target:
            return pushes
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr, nbc = br + dr, bc + dc
            ppr, ppc = br - dr, bc - dc
            if is_free(nbr, nbc) and is_free(ppr, ppc):
                if can_reach((pr, pc), (br, bc), (ppr, ppc)):
                    new_state = (nbr, nbc, br, bc)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append((new_state, pushes + 1))
    return -1


# ── LC1298 从盒子里获得的最大糖果数 ──────────────────────────────────────
def max_candies(
    status: list[int],
    candies: list[int],
    keys: list[list[int]],
    contained_boxes: list[list[int]],
    initial_boxes: list[int],
) -> int:
    """
    【题意】有若干个盒子，`status[i]` 表示盒子 i 初始是开着（1）还是锁着（0），
    `candies[i]` 是里面的糖果数，`keys[i]` 是打开盒子 i 之后能获得的、可以打开
    其他盒子的钥匙列表，`contained_boxes[i]` 是盒子 i 里面还装着的其他盒子；
    给定一开始就拥有的盒子列表 `initial_boxes`，只有"手上有这个盒子"且"这个
    盒子是开着的（或者手上有对应钥匙）"才能打开它、拿到里面的糖果和钥匙/盒子，
    求最多能拿到多少糖果。
    【思路】这是一个"每次打开一个盒子会解锁更多可能性"的多源 BFS——把每个
    "已经拥有但还没打开"的盒子看成一个待处理节点，一旦满足"开着的或者有钥匙"
    这个条件就打开它、把糖果计入总数，同时把打开它获得的新钥匙、新盒子加入
    队列继续处理。关键的状态维护是**两个集合的双向解锁关系**：`have_boxes`
    记录"手上实际持有哪些盒子"（可能还没打开），`have_keys` 记录"手上有哪些
    盒子的钥匙"；一个盒子能被打开需要"拥有它 且 (它本来就开着 或 手上有它的
    钥匙)"，而这两个条件可能以任意顺序被满足——可能先拿到盒子、后来才捡到它的
    钥匙，也可能先捡到钥匙、后来才拿到盒子本体，所以每次"新拿到一个盒子"或者
    "新拿到一把钥匙"都要重新尝试用 `try_open` 检查它是否现在已经可以被打开了，
    而不能假设"先获得盒子就一定能立刻打开"。用 `opened` 集合防止同一个盒子被
    打开、计数两次。
    【复杂度】时间 O(n)（每个盒子最多被打开一次，每次打开最多贡献 O(度数) 的
    钥匙/子盒子扩展，均摊下来整体是线性的）；空间 O(n)（have_boxes/have_keys/
    opened 三个集合 + 队列）。
    【易错点】1) 假设"先获得盒子的顺序"和"先获得钥匙的顺序"总是能对齐（比如
    只在"新增盒子"这个事件里检查能否打开,却忘了"新增钥匙"也可能让一个早就
    持有但当时打不开的旧盒子突然可以打开了）,这样会漏掉后来才解锁的盒子里的
    糖果,算出偏小的答案——这正是这道题作为"每次打开解锁更多可能性"的 BFS
    容易踩的坑;2) 忘记用 `opened` 集合去重,如果一个盒子的引用通过不同路径
    被重复加入队列（比如两个不同盒子都在 `containedBoxes` 里装了同一个盒子的
    情形，虽然题目保证"每个盒子最多被包含一次"，但防御性地去重是好习惯），
    会把它的糖果重复计数;3) 误以为"手上有盒子"和"这个盒子已经打开"是同一件
    事,直接把 `have_boxes` 当成"可以拿糖果"的条件,会在盒子还锁着、也没有对应
    钥匙时就提前把糖果计入总数。
    """
    have_boxes = set(initial_boxes)
    have_keys: set[int] = set()
    opened: set[int] = set()
    total = 0
    q: deque[int] = deque()

    def try_open(box: int) -> None:
        nonlocal total
        if box in opened or box not in have_boxes:
            return
        if status[box] == 0 and box not in have_keys:
            return
        opened.add(box)
        total += candies[box]
        q.append(box)

    for b in initial_boxes:
        try_open(b)

    while q:
        box = q.popleft()
        for k in keys[box]:
            have_keys.add(k)
            try_open(k)
        for cb in contained_boxes[box]:
            have_boxes.add(cb)
            try_open(cb)

    return total


def _self_test() -> None:
    assert shortest_alternating_paths(3, [[0, 1], [1, 2]], []) == [0, 1, -1]
    assert shortest_alternating_paths(3, [[0, 1]], [[2, 1]]) == [0, 1, -1]
    assert shortest_alternating_paths(3, [[1, 0]], [[2, 1]]) == [0, -1, -1]
    assert shortest_alternating_paths(3, [[0, 1]], [[1, 2]]) == [0, 1, 2]

    assert min_malware_spread([[1, 1, 0], [1, 1, 0], [0, 0, 1]], [0, 1]) == 0
    assert min_malware_spread([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 2]) == 0
    assert min_malware_spread([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [1, 2]) == 1

    ans_garden = garden_no_adj(3, [[1, 2], [2, 3], [3, 1]])
    assert len(ans_garden) == 3
    for x, y in [[1, 2], [2, 3], [3, 1]]:
        assert ans_garden[x - 1] != ans_garden[y - 1]
        assert 1 <= ans_garden[x - 1] <= 4 and 1 <= ans_garden[y - 1] <= 4

    assert min_effort_path([[1, 2, 2], [3, 8, 2], [5, 3, 5]]) == 2
    assert min_effort_path([[1, 2, 3], [3, 8, 4], [5, 3, 5]]) == 1
    assert (
        min_effort_path(
            [[1, 2, 1, 1, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 1, 1, 2, 1]]
        )
        == 0
    )

    assert (
        nearest_exit(
            [["+", "+", ".", "+"], [".", ".", ".", "+"], ["+", "+", "+", "."]], [1, 2]
        )
        == 1
    )
    assert (
        nearest_exit([["+", "+", "+"], [".", ".", "."], ["+", "+", "+"]], [1, 0]) == 2
    )
    assert nearest_exit([[".", "+"]], [0, 0]) == -1

    assert (
        shortest_path_with_obstacle_elimination(
            [[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]], 1
        )
        == 6
    )
    assert (
        shortest_path_with_obstacle_elimination([[0, 1, 1], [1, 1, 1], [1, 0, 0]], 1) == -1
    )

    assert shortest_path_all_keys(["@.a.#", "###.#", "b.A.B"]) == 8
    assert shortest_path_all_keys(["@..aA", "..B#.", "....b"]) == 6

    assert (
        min_push_box(
            [
                ["#", "#", "#", "#", "#", "#"],
                ["#", "T", "#", "#", "#", "#"],
                ["#", ".", ".", "B", ".", "#"],
                ["#", ".", "#", "#", ".", "#"],
                ["#", ".", ".", ".", "S", "#"],
                ["#", "#", "#", "#", "#", "#"],
            ]
        )
        == 3
    )
    assert (
        min_push_box(
            [
                ["#", "#", "#", "#", "#", "#"],
                ["#", "T", ".", ".", "#", "#"],
                ["#", ".", "#", "B", ".", "#"],
                ["#", ".", ".", ".", ".", "#"],
                ["#", ".", ".", ".", "S", "#"],
                ["#", "#", "#", "#", "#", "#"],
            ]
        )
        == 5
    )

    assert (
        max_candies(
            [1, 0, 1, 0],
            [7, 5, 4, 100],
            [[], [], [1], []],
            [[1, 2], [3], [], []],
            [0],
        )
        == 16
    )
    assert (
        max_candies(
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [[1, 2, 3, 4, 5], [], [], [], [], []],
            [[1, 2, 3, 4, 5], [], [], [], [], []],
            [0],
        )
        == 6
    )

    print(
        "[PASS] p11_graph_bfs_dfs_iii: 9 道图 BFS/DFS 竞赛级补充题"
        "（颜色交替的最短路径/尽量减少恶意软件的传播/不邻接植花/最小体力消耗路径/"
        "迷宫中离入口最近的出口/网格中的最短路径-有障碍物消除/获取所有钥匙的最短路径/"
        "推箱子/从盒子里获得的最大糖果数）全部通过"
    )


if __name__ == "__main__":
    _self_test()
