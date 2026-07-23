# 14 · 图论基础(Graphs Basics)

> 总览见 [00-roadmap.md](00-roadmap.md)。图是[08类树](08-trees.md)的推广(树是无环连通图的特例)——DFS/BFS 的代码框架和树遍历高度相似,新增的复杂度来自"图可能有环""图可能不连通""边可能有方向"这几个树没有的性质。本类打基础,[15类](15-graphs-advanced.md)在此之上讲最短路径/最小生成树等进阶算法。

---

## 1. 图的表示方法:邻接表与邻接矩阵

**签名/是什么:**
```
邻接表: dict/list, 每个节点存一个"它连接到哪些节点"的列表 —— 空间O(V+E)
邻接矩阵: 二维数组, matrix[i][j]表示i到j是否有边(或边权) —— 空间O(V^2)
```

**一句话:** 邻接表和邻接矩阵是图的两种基础存储方式,选择哪种取决于图的**稠密程度**和**具体需要的操作**——邻接表在稀疏图(边数远小于V²)上更省空间,邻接矩阵在需要频繁"判断任意两点是否直接相连"时更快(O(1) 而不是遍历邻接表)。

**底层机制/为什么这样设计:** 邻接表的空间复杂度是 O(V+E)——每条边只需要在对应节点的列表里记一次(无向图记两次,因为两个方向都要能查到);邻接矩阵的空间复杂度是 O(V²),不管图里实际有多少条边,矩阵的大小只取决于节点数。这意味着对于稀疏图(比如社交网络,大部分人之间没有直接联系),邻接表能节省大量空间;对于稠密图(边数接近 V² 的量级)或者需要频繁做"这两个点是否相连"的 O(1) 查询,邻接矩阵反而更合适。这个选择本质上是[01类知识点1](01-complexity-and-python-builtins.md#1-时间复杂度与空间复杂度分析方法论)"根据具体场景特征选择合适的数据结构"这个方法论在图这个具体结构上的应用。

**AI 研究/工程场景:** [torch-deep-dive](../torch-deep-dive/00-roadmap.md) 系列讲过的计算图(autograd graph),本质上就是一个有向无环图——大规模神经网络的计算图节点数可能达到数万甚至更多,但每个节点的直接依赖(边)数量通常很有限(稀疏),这类计算图的内部表示天然倾向于类似邻接表的稀疏存储方式,而不是为每一对节点都分配一个矩阵位置。

**可运行例子:**
```python
from collections import defaultdict

def build_adjacency_list(n, edges, directed=False):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    return adj

def build_adjacency_matrix(n, edges, directed=False):
    matrix = [[0] * n for _ in range(n)]
    for u, v in edges:
        matrix[u][v] = 1
        if not directed:
            matrix[v][u] = 1
    return matrix

edges = [(0, 1), (1, 2), (0, 2)]
adj_list = build_adjacency_list(3, edges)
adj_matrix = build_adjacency_matrix(3, edges)

assert sorted(adj_list[0]) == [1, 2]
assert adj_matrix[0][1] == 1 and adj_matrix[1][0] == 1   # 无向图矩阵对称
assert adj_matrix[0][2] == 1
assert adj_matrix[1][1] == 0   # 自己到自己默认没有边

# 有向图:两种表示不再对称
directed_edges = [(0, 1), (1, 2)]
d_adj = build_adjacency_list(3, directed_edges, directed=True)
d_matrix = build_adjacency_matrix(3, directed_edges, directed=True)
assert d_adj[0] == [1] and d_adj[1] == [2] and d_adj[2] == []
assert d_matrix[0][1] == 1 and d_matrix[1][0] == 0   # 有向图矩阵不对称

# 空间对比:验证邻接表在稀疏图上确实比邻接矩阵省空间(用节点/边的记录数量做量级比较,不是精确字节数)
import sys
n_large = 1000
sparse_edges = [(i, i + 1) for i in range(n_large - 1)]   # 极稀疏:只有n-1条边(一条链)
sparse_adj_list = build_adjacency_list(n_large, sparse_edges)
sparse_adj_matrix = build_adjacency_matrix(n_large, sparse_edges)

list_entries = sum(len(v) for v in sparse_adj_list.values())   # 邻接表实际存储的条目数
matrix_entries = n_large * n_large                                # 邻接矩阵固定是n^2个位置

assert list_entries < matrix_entries / 100   # 稀疏图下邻接表存储的条目数远少于邻接矩阵

print(f"OK: 邻接表与邻接矩阵在有向/无向图上表示正确(含矩阵对称性验证); "
      f"n={n_large}的稀疏图(链式结构)下, 邻接表存储条目数={list_entries}, "
      f"邻接矩阵位置数={matrix_entries}(邻接表远小于矩阵, 验证稀疏图下的空间优势)")
```
本机实测:有向图和无向图的两种表示方式均正确构建,无向图的邻接矩阵确认对称;在 1000 个节点的稀疏图(链式结构,仅 999 条边)下,邻接表实际存储的条目数远小于邻接矩阵固定的 100 万个位置,直观验证了稀疏图场景下邻接表的空间优势。

**面试怎么问 + 追问链:** "什么情况下应该选邻接矩阵而不是邻接表?" → 追问"如果需要频繁执行'两点之间是否有边'这个查询,邻接表能不能优化到接近O(1)?"(可以——把邻接表的每个列表换成集合(`set`),`v in adj[u]` 的查询复杂度从 O(度数) 降到 O(1) 均摊,这样即使用邻接表存储,也能获得接近邻接矩阵的查询效率,同时保留邻接表在稀疏图下的空间优势;这个追问检验的是能否看出"邻接表 vs 邻接矩阵"不是非黑即白的选择,可以通过调整邻接表内部使用的容器类型来兼顾两方面的优势)。

**常见坑:**
1. 无向图只添加了一个方向的边(比如只写 `adj[u].append(v)` 忘记 `adj[v].append(u)`)——这会让图退化成"看起来无向,实际上部分边是单向"的错误表示,DFS/BFS 遍历可能因此漏掉本该可达的节点。
2. 邻接矩阵在图节点数很大但边很稀疏时依然被使用——即使功能上没有错误,O(V²) 的空间开销在 V 很大时(比如几十万节点)可能直接导致内存溢出,这类场景应该优先考虑邻接表。

---

## 2. DFS 遍历与应用:连通分量与环检测

**签名/是什么:**
```
连通分量计数: 对每个未访问节点触发一次新的DFS, DFS触发次数就是连通分量数
无向图环检测: DFS时记录"从哪个节点过来的"(parent), 
             遇到已访问节点且不是parent,说明存在环
```

**一句话:** DFS 在图上的两个经典应用——统计连通分量(图被分成几个"互不相连的部分")和检测是否存在环——都建立在"visited 集合"这个基础机制上,区别在于连通分量数的是"DFS 被重新触发了几次",环检测靠的是"遇到了一个已访问、但不是直接来源的节点"。

**底层机制/为什么这样设计:** 连通分量计数的原理很直接:一次完整的 DFS(从某个起点出发,能访问到的全部节点)恰好覆盖一个连通分量;遍历所有节点,每当遇到一个还没被任何一次 DFS 访问过的节点,说明发现了一个新的连通分量,触发一次新的 DFS 并计数——这个"触发次数"正是最终答案。无向图环检测需要额外传递 `parent`(当前节点是从哪个节点走过来的)这个信息:无向图的每条边在邻接表里都会存两次(`u→v` 和 `v→u`),如果不排除"直接来源"这个方向,DFS 必然会立刻发现"刚才来的那条边通向的节点已经被访问过",把这个正常现象误判成环——`parent` 参数正是用来避免这种"假环"的误判。

**AI 研究/工程场景:** [huggingface-deep-dive 06 类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练里,判断"一组计算节点之间的通信拓扑是否连通"(所有节点是否都能通过某种路径互相通信)本质上就是连通分量问题;计算图(autograd graph)理论上不应该存在环(否则梯度计算会陷入循环依赖),环检测的思路是这类图结构完整性校验的基础。

**可运行例子:**
```python
from collections import defaultdict

def count_connected_components(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    count = 0
    def dfs(node):
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                dfs(nb)
    for i in range(n):
        if i not in visited:
            count += 1
            dfs(i)
    return count

assert count_connected_components(5, [(0, 1), (1, 2), (3, 4)]) == 2   # {0,1,2}和{3,4}两个分量
assert count_connected_components(4, []) == 4                            # 没有任何边,每个节点自成一个分量
assert count_connected_components(1, []) == 1                            # 单节点

def has_cycle_undirected(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                if dfs(nb, node):
                    return True
            elif nb != parent:   # 遇到已访问节点,但不是"来源",说明存在环
                return True
        return False
    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True
    return False

assert has_cycle_undirected(3, [(0, 1), (1, 2), (2, 0)]) is True    # 三角形,有环
assert has_cycle_undirected(3, [(0, 1), (1, 2)]) is False             # 一条链,无环
assert has_cycle_undirected(2, [(0, 1)]) is False                        # 单条边,无环(不能误判)
assert has_cycle_undirected(4, []) is False                              # 没有任何边

print("OK: 连通分量计数与无向图环检测在边界情况(无边/单节点/单条边)下全部正确")
```
本机实测:连通分量计数在无边、单节点这几类边界情况下均正确(每个孤立节点各自算一个分量);环检测在三角形(有环)、链(无环)、单条边(不能被误判为环)这几类情况下均正确。

**面试怎么问 + 追问链:** "无向图环检测为什么需要传递 `parent` 参数,有向图检测环需要吗?" → 追问"有向图的环检测,应该怎么实现?"(有向图不能用"是否是parent"这个判断(因为有向边天然只有一个方向,不存在"来源边"造成的假环问题),需要用类似[知识点4](14-graphs-basics.md#4-拓扑排序kahn算法与-dfs-两种实现)拓扑排序DFS版本里的"三色标记法"(未访问/正在访问中/已完成),如果DFS过程中遇到一个"正在访问中"(还在当前递归路径上)的节点,才说明真的存在环;这个追问检验的是能否理解"有向图"和"无向图"的环检测,虽然都叫"环检测",内部机制却完全不同,不能把无向图的parent技巧直接套用到有向图上)。

**常见坑:**
1. 无向图环检测忘记传递或使用 `parent` 参数——会把每一条边都误判成环(因为无向边天然会在邻接表里形成"去而复返"的假象)。
2. 把无向图的环检测逻辑直接套用到有向图上——这是上面追问链已经指出的错误,有向图和无向图的环检测需要不同的机制,不能混用。

---

## 3. BFS 遍历与应用:无权图最短路径

**签名/是什么:**
```
BFS天然按"距起点的距离"分层展开(呼应08类知识点2层序遍历),
第一次访问到某个节点时记录的距离,就是从起点到它的最短距离(仅对无权图成立)
```

**一句话:** 在边权都相同(或者说"无权")的图上,BFS 第一次到达某个节点时所走过的层数,就是从起点到该节点的最短路径长度——这是[08类知识点2](08-trees.md#2-层序遍历与-bfs)"层序遍历天然按层展开"这个性质在图上的直接推广,是无权图最短路径问题最简单高效的解法。

**底层机制/为什么这样设计:** BFS 保证了"距起点为 k 的所有节点,一定在'距起点为 k-1 的所有节点'全部被处理完之后才会被访问到"——这个严格的分层顺序保证了每个节点第一次被访问到的那一刻,记录的距离必然是最短的(如果存在更短的路径,那条路径对应的层数会更早,该节点应该在更早的层就被访问到,和"第一次访问"矛盾)。**这个结论仅在无权图(或者所有边权相同)上成立**——如果边有不同的权重,"经过的边数最少"不再等价于"路径总权重最小",需要改用[15类](15-graphs-advanced.md)的 Dijkstra 等专门处理带权图的算法。

**具体走一遍**(用下面"可运行例子"里真实测试的图:边 `[(0,1),(1,2),(2,3),(0,4),(4,3)]`,节点5没有任何边、孤立存在):
```
连接关系摊开看: 0-1, 1-2, 2-3, 0-4, 4-3
从0到3有两条路径: 0→1→2→3(3条边) 和 0→4→3(2条边) —— BFS应该找到更短的那条(2条边)
```
真实运行 `bfs_shortest_path` 的过程(按节点被"第一次发现"的顺序记录):

| 处理的节点 | 距起点距离 | 发现的新邻居 | 说明 |
|---|---|---|---|
| 0 | 0 | 1(距离1)、4(距离1) | 起点的两个直接邻居 |
| 1 | 1 | 2(距离2) | 1 的另一个邻居 0 已访问,跳过 |
| 4 | 1 | 3(距离2) | 4 的另一个邻居 0 已访问,跳过 |
| 2 | 2 | (无) | 2 的邻居 3 **已经**被访问过(上一步刚被 4 标记),不会被错误地覆盖成距离 3 |
| 3 | 2 | (无) | 邻居 2、4 都已访问 |

节点 3 同时是"距离1的节点4"和"距离2的节点2"的邻居——BFS 严格按层处理,第 1 层(节点1、4)必须**全部**处理完,才会轮到处理它们发现的下一层节点。轮到处理节点 4(第1层)时,3 还没被访问过,于是 3 被标记为距离 2;等到后来节点 2(本身已经是第2层)也尝试把 3 标记一遍时,3 已经有距离记录了,这次尝试被直接跳过——即使 3 确实也是 2 的邻居,BFS 保证了它最终拿到的一定是**最先**被发现的那个、也就是最短的距离。这正是"经过4到3,距离2,比经过1,2到3更短"这条 assert 注释背后的真实执行过程。节点 5 从头到尾没有被任何边连到,不会出现在 `dist` 字典里。

**AI 研究/工程场景:** [huggingface-deep-dive 11 类](../huggingface-deep-dive/11-hub-and-sharing.md)讲过模型版本之间的依赖关系(比如某个 fine-tune 版本基于哪个基座模型训练而来),如果要计算"两个模型版本之间,经过了几次微调迭代"(不考虑每次微调的具体"距离"权重,只关心迭代次数),BFS 最短路径正是这类"边数最少"问题的标准解法。

**可运行例子:**
```python
from collections import defaultdict, deque

def bfs_shortest_path(n, edges, start):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist

result = bfs_shortest_path(6, [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)], 0)
assert result[0] == 0
assert result[1] == 1 and result[4] == 1         # 直接相连,距离1
assert result[3] == 2                                # 经过4到3,距离2(比经过1,2到3更短)
assert 5 not in result                                # 节点5不连通,不应该出现在结果里

assert bfs_shortest_path(1, [], 0) == {0: 0}          # 单节点图
assert bfs_shortest_path(3, [], 0) == {0: 0}          # 没有边,只有起点自己可达

# 交叉验证:小规模下用暴力枚举所有路径,找最短路径长度,和BFS结果对照
def brute_shortest_path(n, edges, start):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    dist = {start: 0}
    frontier = {start}
    step = 0
    while frontier:
        step += 1
        next_frontier = set()
        for node in frontier:
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = step
                    next_frontier.add(nb)
        frontier = next_frontier
    return dist

import random
random.seed(70)
for _ in range(15):
    n = random.randint(2, 10)
    edges = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(random.randint(0, 15))]
    edges = [(u, v) for u, v in edges if u != v]
    assert bfs_shortest_path(n, edges, 0) == brute_shortest_path(n, edges, 0)

print("OK: BFS最短路径在边界情况(单节点/无边)下全部正确, "
      "15组随机测试与暴力分层扩展方法结果完全一致")
```
本机实测:BFS 最短路径在验证案例中正确识别出"经过4到3"(距离2)比"经过1,2到3"(距离3)更短;单节点、无边这几类边界情况均正确;不连通的节点正确地没有出现在结果里;15 组随机测试中,BFS 和暴力分层扩展方法结果完全一致。

**面试怎么问 + 追问链:** "为什么 BFS 能保证第一次访问到某节点时的距离就是最短距离,DFS 不行吗?" → 追问"如果一定要用 DFS 求无权图最短路径,可以吗?会有什么代价?"(可以,但需要遍历所有可能路径并取最小值(不能像BFS一样"第一次访问就是最短"),复杂度会退化,不再具有BFS这种"一次遍历即得最短距离"的效率;这个追问检验的是能否理解BFS"按层扩展"这个特性和"最短距离"之间的因果关系,不是DFS完全做不到,而是DFS的遍历顺序不具备BFS这种天然的最优性保证)。

**常见坑:**
1. 用 BFS 求带权图的最短路径(边权不同)——本知识点已经强调这个方法只对无权图/边权相同的图成立,带权图需要用[15类](15-graphs-advanced.md)的 Dijkstra 等算法。
2. 忘记检查目标节点是否连通就直接访问结果字典对应的距离——如果目标节点和起点不连通,`dist` 字典里不会有这个键,直接用 `dist[target]` 访问会抛出 `KeyError`,应该先用 `in` 检查或者用 `.get()` 提供默认值。

---

## 4. 拓扑排序:Kahn 算法与 DFS 两种实现

**签名/是什么:**
```
Kahn算法: 维护每个节点的入度, 每次取出入度为0的节点加入结果, 
         并将它指向的节点入度减1, 重复直到队列为空
DFS版本: 后序遍历(所有依赖都处理完才把自己加入结果), 最后把结果反转
```

**一句话:** 拓扑排序给有向无环图(DAG)里的节点排出一个顺序,使得每条边 `u→v` 里 `u` 都排在 `v` 前面——Kahn 算法(基于入度,类似 BFS)和 DFS 版本(基于后序遍历再反转)是两种殊途同归的实现方式,都能同时检测出"图里是否存在环"(拓扑排序只对无环图有意义)。

**底层机制/为什么这样设计:** Kahn 算法的直觉:入度为 0 的节点没有任何前置依赖,可以立即排在最前面;把它"移除"后(它指向的所有节点入度各减1),可能会产生新的入度为0的节点,重复这个过程——如果最终处理的节点数少于图的总节点数,说明剩下的节点都困在某个环里(永远无法让入度降到0),据此可以判断图中存在环。DFS 版本的直觉更微妙:对每个节点做 DFS,只有当它的**所有**后继节点都已经被完全处理完(递归返回)之后,才把当前节点加入结果——这保证了结果列表里,每个节点都排在它所有后继节点的**后面**;最后把整个列表反转,就得到了"前置节点在前"的正确拓扑顺序。DFS 版本检测环需要额外的三色标记(未访问/正在访问中/已完成):如果 DFS 过程中遇到一个"正在访问中"(还在当前递归路径上)的节点,说明存在环。

**AI 研究/工程场景:** [torch-deep-dive](../torch-deep-dive/00-roadmap.md) 系列讲过反向传播的执行顺序——计算图必须按照"一个节点的所有输入都已经计算完成"这个顺序反向遍历才能正确计算梯度,这正是拓扑排序(在反向图上)的直接应用;更广泛地说,任何"任务A必须在任务B之前完成"这类依赖关系调度问题(比如构建系统决定编译顺序、[huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练任务依赖),拓扑排序都是标准解法。

**可运行例子:**
```python
from collections import defaultdict, deque

def topo_sort_kahn(n, edges):
    adj = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1
    q = deque([i for i in range(n) if indegree[i] == 0])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nb in adj[node]:
            indegree[nb] -= 1
            if indegree[nb] == 0:
                q.append(nb)
    return order if len(order) == n else None   # 长度不足n说明存在环

def topo_sort_dfs(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    visited = [0] * n   # 0=未访问, 1=正在访问, 2=已完成
    order = []
    has_cycle = [False]
    def dfs(node):
        visited[node] = 1
        for nb in adj[node]:
            if visited[nb] == 1:
                has_cycle[0] = True
                return
            if visited[nb] == 0:
                dfs(nb)
        visited[node] = 2
        order.append(node)
    for i in range(n):
        if visited[i] == 0:
            dfs(i)
    return None if has_cycle[0] else order[::-1]

def is_valid_topo_order(order, n, edges):
    """独立校验函数:检查一个排序是否真的满足所有边的先后约束"""
    if order is None:
        return False
    position = {node: i for i, node in enumerate(order)}
    return all(position[u] < position[v] for u, v in edges)

kahn_result = topo_sort_kahn(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
dfs_result = topo_sort_dfs(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
assert is_valid_topo_order(kahn_result, 4, [(0, 1), (0, 2), (1, 3), (2, 3)])
assert is_valid_topo_order(dfs_result, 4, [(0, 1), (0, 2), (1, 3), (2, 3)])
# 两种算法给出的具体顺序可能不同(都合法),但都必须通过独立校验

assert topo_sort_kahn(3, [(0, 1), (1, 2), (2, 0)]) is None    # 有环,无法拓扑排序
assert topo_sort_dfs(3, [(0, 1), (1, 2), (2, 0)]) is None

assert topo_sort_kahn(3, []) is not None   # 没有任何边,任意顺序都合法
assert len(topo_sort_kahn(3, [])) == 3

print(f"OK: Kahn算法与DFS版本的拓扑排序, 独立校验函数确认结果都满足'前置节点在前'的约束"
      f"(kahn={kahn_result}, dfs={dfs_result}, 顺序不同但都合法); "
      f"两种算法在有环图上都正确返回None; 无边图下正确返回包含全部节点的合法顺序")
```
本机实测:Kahn 算法和 DFS 版本在同一个 DAG 上给出的具体顺序不同(`[0,1,2,3]` 和 `[0,2,1,3]`),但用独立的校验函数(不复用求解逻辑本身)验证,两者都满足"每条边的起点排在终点前面"这个拓扑排序的核心约束;两种算法在含环的图上都正确返回 `None`;无边图下正确返回一个包含全部节点的合法顺序。

**面试怎么问 + 追问链:** "Kahn 算法和 DFS 版本的拓扑排序,复杂度分别是多少?" → 追问"如果题目要求'返回字典序最小的拓扑排序',哪种实现更容易改造?"(Kahn 算法更容易——只需要把维护候选节点的普通队列换成最小堆(呼应[07类](07-heaps-and-priority-queues.md)),每次取入度为0的节点里编号最小的那个;DFS 版本要做到"字典序最小"没有这么直接的改造方式,因为DFS的访问顺序和最终反转后的结果之间的关系不够直观;这个追问检验的是能否理解两种等价算法在面对具体变体需求时,可扩展性可能不同,不能认为两种写法在所有场景下都是完全可以互换的)。

**常见坑:**
1. Kahn 算法忘记检查最终 `order` 的长度是否等于 `n`——如果图中存在环,某些节点的入度永远无法降到0,不会被加入结果,如果不检查长度就直接返回,会静默地返回一个不完整的顺序而不报告"图里其实有环"这个重要信息。
2. DFS 版本忘记在最后反转结果列表——后序遍历得到的顺序本身是"后继节点在前、前置节点在后",必须反转才能得到正确的拓扑顺序,这是这个实现方式里最容易遗漏的一步。

---

## 5. 并查集基础实现

**签名/是什么:**
```
find(x): 查找x所在集合的代表元素(根节点)
union(x, y): 合并x和y所在的两个集合
```

**一句话:** 并查集(Union-Find / Disjoint Set)维护一组不相交的集合,支持"查询两个元素是否属于同一集合"和"合并两个集合"这两个操作——每个集合内部用一棵"树"表示,`parent[x]` 指向 x 的父节点,一直往上找到 `parent[root]==root` 的节点就是这个集合的代表元素。

**底层机制/为什么这样设计:** 并查集的基础实现,`find` 操作沿着 `parent` 指针链一直往上走直到找到根节点(自己指向自己的节点);`union` 操作把其中一棵树的根节点,挂到另一棵树的根节点下面,完成两个集合的合并。这个基础版本的效率问题在于:如果树的形状退化(比如一长串链式结构,类似[03类知识点3](03-linked-lists.md#3-快慢指针与环检测floyd-判圈算法)提到的链表退化问题),`find` 操作的复杂度会退化到 O(n)——[知识点6](14-graphs-basics.md#6-并查集优化路径压缩--按秩合并)会讲的两个优化技巧,正是为了避免这种退化。

**AI 研究/工程场景:** [15类知识点4](15-graphs-advanced.md#4-最小生成树kruskal算法)最小生成树 Kruskal 算法直接依赖并查集判断"加入这条边会不会形成环"(如果两个端点已经在同一集合,加入这条边就会成环);更一般地,任何"动态判断/合并一组元素的连通性"问题(比如判断一批网络节点是否都能互相通信)都是并查集的经典应用场景。

**可运行例子:**
```python
class UnionFindBasic:
    def __init__(self, n):
        self.parent = list(range(n))   # 初始时,每个元素自成一个集合,自己是自己的代表

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

uf = UnionFindBasic(6)
assert uf.find(0) == 0   # 初始时,每个元素的代表就是自己

uf.union(0, 1)
uf.union(1, 2)
assert uf.find(0) == uf.find(2)   # 0和2现在应该在同一个集合
assert uf.find(0) != uf.find(3)     # 0和3依然是不同集合

uf.union(3, 4)
assert uf.find(3) == uf.find(4)
assert uf.find(0) != uf.find(3)     # 两个集合{0,1,2}和{3,4}依然独立

uf.union(2, 4)   # 合并两个集合
assert uf.find(0) == uf.find(4)   # 现在应该全部在同一个集合了

assert uf.find(5) == 5   # 从未被union过的元素,依然自成一个集合

# 交叉验证:并查集判断的连通性,和用DFS/BFS判断连通分量的结果必须一致
from collections import defaultdict

def connected_via_dfs(n, unions, x, y):
    adj = defaultdict(list)
    for a, b in unions:
        adj[a].append(b)
        adj[b].append(a)
    visited = {x}
    stack = [x]
    while stack:
        node = stack.pop()
        if node == y:
            return True
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return x == y

import random
random.seed(71)
for _ in range(10):
    n = 8
    uf_test = UnionFindBasic(n)
    unions = []
    for _ in range(random.randint(0, 6)):
        a, b = random.randint(0, n-1), random.randint(0, n-1)
        uf_test.union(a, b)
        unions.append((a, b))
    for x in range(n):
        for y in range(n):
            assert (uf_test.find(x) == uf_test.find(y)) == connected_via_dfs(n, unions, x, y)

print("OK: 并查集基础实现在多次合并操作后, 连通性判断与DFS独立验证结果完全一致")
```
本机实测:基础并查集在多次合并操作后,正确维护了集合的连通关系;10 组随机测试中(每组包含随机数量的合并操作),并查集对任意两个元素"是否连通"的判断,和用 DFS 独立重新计算连通性的结果完全一致。

**面试怎么问 + 追问链:** "并查集的 `find` 操作最坏情况复杂度是多少?" → 追问"什么样的合并顺序,会让基础版本的并查集退化成最坏情况?"(如果每次都把新元素合并成一条长链(比如依次 `union(0,1)`, `union(1,2)`, `union(2,3)`...),`parent` 指针会形成一条长链,`find` 操作需要沿着整条链走到底,退化成 O(n);这个追问检验的是能否具体构造出触发最坏情况的输入模式,而不是抽象地知道"最坏情况是O(n)"这个结论)。

**常见坑:**
1. `union` 操作不做"是否已经在同一集合"的判断就直接合并——虽然功能上通常不会出错(重复合并同一集合本身是无害的),但这类判断的缺失容易掩盖代码逻辑上的疏漏,养成先判断再操作的习惯更稳妥。
2. 误以为 `find(x)` 返回的值(代表元素)是固定不变的——随着更多 `union` 操作的进行,同一个元素所在集合的代表元素(根节点)可能会发生变化,不能把某次 `find` 调用的结果缓存下来长期使用而不重新查询。

---

## 6. 并查集优化:路径压缩与按秩合并

**签名/是什么:**
```
路径压缩(path compression): find时,把沿途经过的所有节点直接指向根节点,压平树的高度
按秩合并(union by rank): union时,把"矮"的树挂到"高"的树下面,而不是随意挂靠
```

**一句话:** 路径压缩和按秩合并是并查集的两个经典优化,单独使用任何一个都能显著改善最坏情况,两者结合后,并查集的操作复杂度能做到均摊几乎是 O(1)(严格地说是反阿克曼函数 O(α(n)),增长极其缓慢,实践中可以视为常数)。

**底层机制/为什么这样设计:** 路径压缩发生在 `find` 操作的过程中——一旦找到根节点,顺便把刚才沿途经过的所有节点的 `parent` 直接指向根节点(而不是维持原来一层层的链式指向),这样下次再查询这些节点时,一步就能到达根节点,不需要重复走原来的长路径。按秩合并发生在 `union` 操作时——用一个 `rank`(近似于树的高度)记录每棵树的"深度",合并时总是把较矮的树挂到较高的树下面,这样能避免树的高度不必要地增长(如果随意合并,矮树挂在深树上会让整棵树变得更深)。这两个优化分别从"操作时顺便优化结构"和"合并时主动控制结构增长"两个角度,防止[知识点5](14-graphs-basics.md#5-并查集基础实现)提到的树退化问题——[知识点1](01-complexity-and-python-builtins.md#1-时间复杂度与空间复杂度分析方法论)以来反复出现的"用一点额外的操作开销,换取长期使用时的效率"这个权衡,在这里又一次出现。

**AI 研究/工程场景:** [15类知识点4](15-graphs-advanced.md#4-最小生成树kruskal算法)Kruskal 算法在处理大规模图(边数可能达到百万级)时,并查集的操作次数和边数同量级——如果不做这两个优化,`find` 操作的真实性能可能会成为整个算法的瓶颈,这不是理论上的担忧,本知识点会用真实计时验证这个差距。

**可运行例子:**
```python
import time

class UnionFindBasic:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

class UnionFindOptimized:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # 路径压缩:递归返回时顺便压平路径
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:   # 按秩合并:矮树挂到高树下面
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

# 功能正确性验证:优化版本的连通性判断,必须和基础版本完全一致
basic = UnionFindBasic(20)
opt = UnionFindOptimized(20)
import random
random.seed(72)
operations = [(random.randint(0, 19), random.randint(0, 19)) for _ in range(15)]
for a, b in operations:
    basic.union(a, b)
    opt.union(a, b)
for x in range(20):
    for y in range(20):
        assert (basic.find(x) == basic.find(y)) == (opt.find(x) == opt.find(y))

# 真实计时验证:构造一个"链式合并"的最坏场景(呼应知识点5的追问链),对比优化前后的真实性能差异
n = 3000
basic_worst = UnionFindBasic(n)
t0 = time.perf_counter()
for i in range(n - 1):
    basic_worst.union(i, i + 1)   # 链式合并,基础版本会退化成一条长链
for _ in range(3000):
    basic_worst.find(0)             # 每次find都要走完整条链
basic_time = time.perf_counter() - t0

opt_worst = UnionFindOptimized(n)
t0 = time.perf_counter()
for i in range(n - 1):
    opt_worst.union(i, i + 1)
for _ in range(3000):
    opt_worst.find(0)
opt_time = time.perf_counter() - t0

assert basic_time > opt_time * 20   # 优化版本应该有数量级级别的速度优势

print(f"OK: 优化版本(路径压缩+按秩合并)的连通性判断与基础版本完全一致; "
      f"链式合并最坏场景下(n={n}), 基础版本={basic_time:.4f}s, 优化版本={opt_time:.4f}s"
      f"(优化版本快{basic_time/opt_time:.0f}倍)")
```
本机实测:优化版本和基础版本在多次随机合并操作后,连通性判断结果完全一致;在专门构造的"链式合并"最坏场景下(3000 个节点依次两两合并),基础版本耗时约 0.22s,优化版本(路径压缩+按秩合并)仅耗时约 0.001s,快了约 220 倍——这个巨大的差距直接验证了这两个优化在真实最坏场景下的价值,不是理论上的边际改进。

**面试怎么问 + 追问链:** "路径压缩和按秩合并,分别从哪个角度优化并查集?两者可以只用一个吗?" → 追问"如果只用路径压缩、不用按秩合并(随意合并树),复杂度会退化吗?"(不会显著退化——单独使用路径压缩,均摊复杂度已经能达到 O(log n),已经是很大的改进(相比基础版本可能的 O(n));两个优化结合能进一步逼近 O(α(n))这个理论最优,但即使只用一个,相比完全不优化的版本已经有质的飞跃;这个追问检验的是能否理解这两个优化各自独立的价值,而不是把它们当作"必须同时使用才有效"的捆绑技巧)。

**常见坑:**
1. 路径压缩的递归写法(`self.parent[x] = self.find(self.parent[x])`)在树很深、且没有做任何优化的情况下,第一次调用可能触发较深的递归——如果预期图的规模很大(节点数远超默认递归深度限制),需要考虑用迭代写法实现路径压缩,避免[01类知识点6](01-complexity-and-python-builtins.md#6-递归的时间空间复杂度分析)提到的递归深度限制问题。
2. 按秩合并时,合并方向判断写反(应该把矮树挂到高树下,却写成了相反的方向)——虽然功能上依然正确(连通性判断不会出错),但会失去按秩合并本该带来的性能优化效果,退化成接近"随意合并"的效率。

---

## 7. 二分图判定

**签名/是什么:**
```
二分图: 能把所有节点分成两组，使得每条边的两个端点分别属于不同组
BFS/DFS染色法: 从任意节点开始，交替染成两种颜色，
             如果发现相邻节点颜色相同，说明不是二分图
```

**一句话:** 判断一个图是否是二分图,标准做法是尝试用两种颜色给所有节点染色,约束"每条边连接的两个节点颜色必须不同"——如果染色过程中出现矛盾(相邻节点被迫染成同一种颜色),说明这个图不是二分图。

**底层机制/为什么这样设计:** 从任意一个未染色的节点开始,染成颜色 A,它的所有邻居必须染成颜色 B(否则违反"相邻节点颜色不同"的约束),这些邻居的邻居又必须染回颜色 A,如此交替——这本质上是 BFS/DFS 遍历的过程中额外附加一个"颜色约束"。如果在扩展过程中,某个节点已经被染过色,且这次要求的颜色和它已有的颜色冲突,说明无法用两种颜色合法地给整个图染色,图不是二分图。**二分图有一个和"环"密切相关的数学性质**:一个图是二分图,当且仅当它不包含任何"长度为奇数的环"(奇环)——直觉上,沿着一个环交替染色,如果环的长度是奇数,染色会在环闭合的地方产生冲突(A-B-A-B-...最后一个节点和起点相邻,但颜色算下来会相同)。

**AI 研究/工程场景:** 二分图判定和匹配问题([15类知识点8](15-graphs-advanced.md#8-二分图最大匹配匈牙利算法)会展开)在推荐系统里有直接应用——比如"用户"和"商品"这两类节点,用户和购买过的商品之间连边,这类"两组节点、边只在组间存在"的结构天然是二分图,匹配算法(比如为用户匹配最合适的商品)建立在这个结构性质之上。

**可运行例子:**
```python
from collections import defaultdict, deque

def is_bipartite(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    color = {}
    for start in range(n):
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if nb not in color:
                    color[nb] = 1 - color[node]   # 染成相反的颜色
                    q.append(nb)
                elif color[nb] == color[node]:      # 相邻节点颜色冲突
                    return False
    return True

assert is_bipartite(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) is True    # 4个节点的环(偶环),是二分图
assert is_bipartite(3, [(0, 1), (1, 2), (2, 0)]) is False              # 三角形(奇环),不是二分图
assert is_bipartite(2, [(0, 1)]) is True                                  # 单条边,显然是二分图
assert is_bipartite(3, []) is True                                        # 没有任何边,平凡二分图(任意染色都合法)
assert is_bipartite(5, [(0, 1), (2, 3)]) is True                          # 不连通图,分别判断每个分量

# 验证"是否是二分图"和"是否包含奇环"这个数学性质的等价关系
def has_odd_cycle_via_bipartite_check(n, edges):
    return not is_bipartite(n, edges)

def brute_find_any_odd_cycle_length(n, edges):
    """暴力找图中是否存在奇数长度的环(小规模验证用,不追求效率)"""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    found_odd = [False]
    def dfs(node, start, depth, visited_path):
        if depth > 1 and node == start:
            if depth % 2 == 1:
                found_odd[0] = True
            return
        if depth > n:
            return
        for nb in adj[node]:
            if nb == start and depth >= 2:
                if (depth + 1) % 2 == 1:
                    found_odd[0] = True
            elif nb not in visited_path:
                dfs(nb, start, depth + 1, visited_path | {nb})
    for s in range(n):
        dfs(s, s, 0, {s})
    return found_odd[0]

import random
random.seed(73)
for _ in range(15):
    n = random.randint(2, 6)
    edges = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(random.randint(0, 8))]
    edges = [(u, v) for u, v in edges if u != v]
    assert has_odd_cycle_via_bipartite_check(n, edges) == brute_find_any_odd_cycle_length(n, edges)

print("OK: 二分图判定在偶环/奇环/单边/无边/不连通图等边界情况下全部正确, "
      "15组随机测试验证了'非二分图'与'存在奇环'这个数学性质的等价关系")
```
本机实测:二分图判定在偶数长度环(是二分图)、奇数长度环(不是二分图)、单条边、无边、不连通图这几类情况下均正确;15 组随机测试中,验证了"图不是二分图"和"图中存在奇数长度的环"这两个判断在具体案例上的等价性,不只是抽象的数学结论。

**面试怎么问 + 追问链:** "为什么二分图不能包含奇数长度的环?" → 追问"如果一个图不连通,包含多个连通分量,判断二分图需要注意什么?"(必须对每个连通分量分别判断,不能只从一个起点开始染色就断定整个图——不同连通分量之间没有边连接,某个分量是二分图不代表另一个分量也是,本知识点的实现用外层循环遍历所有节点、跳过已染色的节点,正是为了处理这种不连通的情况;这个追问检验的是能否注意到图"不保证连通"这个常被忽略的前提,类似[知识点2](14-graphs-basics.md#2-dfs-遍历与应用连通分量与环检测)连通分量计数需要对每个未访问节点触发新一轮遍历的道理)。

**常见坑:**
1. 只从节点0开始染色,不检查图是否连通——如果图不连通,某些节点可能永远不会被访问到,程序可能因为遗漏这些节点的检查而错误地返回"是二分图"(即使这些未被检查的节点所在的分量实际上有奇环)。
2. 判断颜色冲突的逻辑写错(比如把 `elif color[nb] == color[node]` 误写成检查其他条件)——这类判断条件错误可能导致把真正的二分图误判为非二分图,或者相反,必须谨慎核对"冲突"的准确含义(相邻节点颜色相同才算冲突)。

---

## 8. 图论基础常见坑

**签名/是什么:**
```
有向图/无向图DFS处理方式混淆 -> 环检测等逻辑出错
忘记标记已访问节点 -> 无限递归/死循环
```

**一句话:** 图论算法最常见的两类坑——把有向图和无向图的遍历/环检测逻辑搞混(这两者机制不同,不能直接套用),以及忘记正确维护 `visited` 集合导致重复访问甚至死循环——本类[知识点2](14-graphs-basics.md#2-dfs-遍历与应用连通分量与环检测)已经展示过前者的一个具体表现,这里从"忘记标记访问"这个更直接的错误角度补充。

**底层机制/为什么这样设计:** 图和树最大的结构性区别是"图可能有环"——树的 DFS/BFS 天然不会重复访问同一个节点(因为树没有环,不存在"绕回来"的路径),但图上如果不显式维护 `visited` 集合并在每次访问前检查,遍历很可能陷入"A访问B,B又访问回A,A再次访问B..."这样的无限循环。这个问题在树的遍历代码([08类](08-trees.md))里不会出现,是很多人从树的直觉迁移到图时容易踩的坑——写图的遍历代码时,"检查并标记 visited"不是可选的优化,是保证算法能够终止的必要条件。

**AI 研究/工程场景:** 这类"数据结构从'无环保证'(树)变成'可能有环'(图)之后,原有的算法直觉失效"的现象,在实际工程里同样会出现——比如把一个假设"依赖关系是树状"的系统(简单的层级配置)扩展成允许"依赖关系是图状"(更灵活但可能出现循环依赖)的系统时,原有的处理逻辑必须重新审视是否还成立,这是一个具有普遍意义的工程教训,不只是算法题的注意事项。

**可运行例子:**
```python
from collections import defaultdict

# 坑1: 忘记维护visited集合,在有环图上DFS会真实导致无限递归
# 注:最初设想用一个"最大深度计数器"当安全阀提前拦截,但现场验证发现这个安全阀根本没有机会生效——
# 因为Python自己的默认递归深度限制(1000)比想设置的计数器阈值更早触发,
# 所以这里改为直接捕获Python原生抛出的RecursionError,这样更真实地反映了实际发生的情况
def buggy_dfs_no_visited(adj, start):
    """故意不维护visited"""
    def dfs(node):
        for nb in adj[node]:
            dfs(nb)
    dfs(start)

# 构造一个有环的图(哪怕只有一个自环或者简单的双向边,不做visited检查就会立刻递归爆炸)
cyclic_adj = defaultdict(list)
cyclic_adj[0] = [1]
cyclic_adj[1] = [0]   # 0和1互相指向,形成一个简单环

raised = False
try:
    buggy_dfs_no_visited(cyclic_adj, 0)
except RecursionError:
    raised = True
assert raised   # 真实复现:不维护visited导致真正的无限递归,直接撞上Python的递归深度上限

def correct_dfs_with_visited(adj, start):
    visited = set()
    result = []
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        result.append(node)
        for nb in adj[node]:
            dfs(nb)
    dfs(start)
    return result

assert correct_dfs_with_visited(cyclic_adj, 0) == [0, 1]   # 正确处理环,不会无限递归

# 坑2: 把无向图的环检测逻辑(依赖parent参数)直接套用到有向图上,产生错误判断
def cycle_check_undirected_style(n, directed_edges):
    """错误地用无向图的parent技巧检测有向图的环"""
    adj = defaultdict(list)
    for u, v in directed_edges:
        adj[u].append(v)   # 只加一个方向,这是有向图的正确表示
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                if dfs(nb, node):
                    return True
            elif nb != parent:   # 错误:这个判断逻辑是为无向图设计的,不适用于有向图
                return True
        return False
    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True
    return False

# 构造一个无环的有向图(比如 0->1, 0->2, 1->2 这种"菱形但没有真正环"的结构),
# 但错误的检测逻辑会因为1和2都指向了"已访问"的节点而产生误判的风险
no_cycle_directed = [(0, 1), (0, 2), (1, 2)]
wrong_check_result = cycle_check_undirected_style(3, no_cycle_directed)
# 用正确的有向图环检测方法(三色标记法,呼应知识点4)做对照
def cycle_check_directed_correct(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    state = [0] * n   # 0=未访问, 1=访问中, 2=已完成
    def dfs(node):
        state[node] = 1
        for nb in adj[node]:
            if state[nb] == 1:
                return True
            if state[nb] == 0 and dfs(nb):
                return True
        state[node] = 2
        return False
    for i in range(n):
        if state[i] == 0:
            if dfs(i):
                return True
    return False

correct_check_result = cycle_check_directed_correct(3, no_cycle_directed)
assert correct_check_result is False   # 真实情况:这个有向图没有环
# 错误方法在这个具体案例上是否会给出不同答案,取决于具体的遍历顺序,现场对比记录真实结果

print(f"OK: 现场复现DFS不维护visited导致的真实无限递归(直接撞上Python自身的RecursionError, "
      f"而不是靠自定义安全阀提前拦截——最初设想的'计数器安全阀'反而没有机会生效, "
      f"因为Python原生的递归深度限制先一步触发了); "
      f"有向图环检测的正确方法(三色标记法)确认这个菱形结构无环={not correct_check_result}, "
      f"错误套用无向图技巧的检测结果={not wrong_check_result}"
      f"(两种方法在这个具体案例上{'恰好一致' if wrong_check_result==correct_check_result else '出现分歧,证明了错误方法确实不可靠'})")
```
本机实测(含一次真实的方法调整):不维护 `visited` 集合的 DFS,在一个简单的双向边(0↔1)构成的环上,真实触发了无限递归——最初设计了一个"最大深度计数器"当安全阀,想在真正撞上 Python 自身递归限制之前优雅地拦截,但现场验证发现这个安全阀的阈值(2000)设得比 Python 默认递归深度限制(1000)更高,根本没有机会生效,Python 自己的 `RecursionError` 先一步触发了——于是改为直接捕获这个原生异常,这样反而更真实地反映了"不维护 visited 会有什么后果"这件事本身(直接撞上解释器的硬性限制,不是一个可以被自定义逻辑温和拦截的可控错误);把无向图环检测的 `parent` 判断逻辑直接套用到有向图上,和正确的三色标记法在这个具体测试案例上的结果被现场对比记录——即使两种方法在某个具体输入上碰巧给出相同结果,也不能证明错误方法本身是可靠的(呼应[11类知识点6](11-greedy-algorithms.md#6-贪心常见坑)"不能仅凭个别案例通过就断言方法正确"这条纪律)。

**面试怎么问 + 追问链:** "写图的 DFS/BFS 遍历代码,`visited` 集合应该在什么时候标记——是访问到节点时立刻标记,还是等处理完这个节点之后才标记?" → 追问"这个标记时机的选择,会不会影响算法的正确性?"(应该在访问到节点、加入待处理队列/开始递归的那一刻立刻标记,而不是等处理完才标记——如果标记时机延后,同一个节点可能在它自己还没被标记之前,通过另一条路径被重复加入队列/触发重复的递归调用,这在 BFS 场景下会导致节点被多次加入队列(虽然通常不会导致无限循环,但会有冗余的重复处理),在某些 DFS 场景下则可能真的引发错误;这个追问检验的是对"标记时机"这个容易被忽视的细节是否有精确的理解,不是只知道"要标记 visited"这个粗略的原则)。

**常见坑:**
1. 图的遍历代码完全没有 `visited` 集合,或者有但检查/更新的时机不对——本知识点已经具体复现了完全不维护时的真实死循环后果。
2. 不假思索地把"链表/树"场景下的直觉(不需要考虑重复访问)迁移到图的场景——图和树/链表最本质的区别就是"可能存在环",任何在无环结构上成立的简化假设,搬到图上都需要重新审视是否还适用。

---

*本篇 8 个知识点全部在仓库根目录 `.venv` 真实测试验证(含真实计时的并查集优化效果、真实死循环的现场复现、以及与暴力解法的交叉验证)。*
