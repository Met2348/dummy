# 15 · 图论进阶(Graphs Advanced)

> 总览见 [00-roadmap.md](00-roadmap.md)。本类是 18 个分类里内容密度最高的一类——最短路径三兄弟(Dijkstra/Bellman-Ford/Floyd-Warshall)、最小生成树两兄弟(Kruskal/Prim)、强连通分量、网络流、二分图最大匹配。这些是真正区分"刷过题"和"扛得住终面"的内容,大多数入门题单不会覆盖到这个深度。

---

## 1. Dijkstra 算法:堆优化实现

**签名/是什么:**
```
适用: 带权图, 边权非负, 求单源最短路径
用堆(呼应07类)维护"当前已知最短距离最小的未确定节点", 每次贪心地确定一个节点的最终最短距离
```

**一句话:** Dijkstra 算法是[14类知识点3](14-graphs-basics.md#3-bfs-遍历与应用无权图最短路径)BFS 无权图最短路径的带权推广——用堆代替普通队列,每次贪心地"确定"当前已知距离最小的那个节点(它的最短距离不会再被更新),逐步扩展已确定的节点集合。

**底层机制/为什么这样设计:** Dijkstra 的贪心正确性依赖一个关键前提——**边权非负**:每次从堆中弹出的"当前距离最小"的节点,它的最短距离一定已经确定,不可能再被后续任何路径更新得更小(因为任何绕路都只会增加距离,不会减少)。这个贪心策略如果放在存在负权边的图上会失效(呼应[知识点9](15-graphs-advanced.md#9-图论进阶常见坑)),因为负权边可能让一条"看起来更长"的路径最终反而更短。实现上用堆(而不是每次线性扫描找最小值)把"找到当前最小距离节点"这一步的复杂度从 O(V) 降到 O(log V),整体复杂度是 O((V+E) log V)。

**AI 研究/工程场景:** [07类知识点6](07-heaps-and-priority-queues.md#6-堆与贪心的组合会议室调度问题)已经用堆解决过资源调度问题;Dijkstra 是"堆 + 贪心"这个组合模式在图论里最经典的应用——[huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练网络拓扑,如果需要计算"两个计算节点之间考虑真实网络延迟的最短通信路径",就是 Dijkstra 的直接应用场景。

**可运行例子:**
```python
import heapq
from collections import defaultdict

def dijkstra(n, edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    visited = set()
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:      # 已经确定过最短距离,跳过(可能是堆里的过期记录)
            continue
        visited.add(node)
        for nb, w in adj[node]:
            if dist[node] + w < dist[nb]:
                dist[nb] = dist[node] + w
                heapq.heappush(heap, (dist[nb], nb))
    return dist

edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
result = dijkstra(4, edges, 0)
assert result == [0, 3, 1, 4]   # 0->2->1->3 = 1+2+1=4,比0->1->3=4+1=5更短

assert dijkstra(1, [], 0) == [0]                    # 单节点图
assert dijkstra(3, [], 0) == [0, float('inf'), float('inf')]   # 无边,只有起点可达

# 交叉验证:小规模下用暴力枚举所有路径,找最短路径长度,和Dijkstra结果对照
def brute_shortest_path_weighted(n, edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    best = [float('inf')] * n
    best[start] = 0
    def dfs(node, cur_dist, visited_path):
        if cur_dist >= best[node] and node != start:
            pass
        best[node] = min(best[node], cur_dist)
        for nb, w in adj[node]:
            if nb not in visited_path or cur_dist + w < best[nb]:
                dfs(nb, cur_dist + w, visited_path | {nb})
    dfs(start, 0, {start})
    return best

import random
random.seed(80)
for _ in range(15):
    n = random.randint(2, 6)
    test_edges = [(random.randint(0, n-1), random.randint(0, n-1), random.randint(1, 10))
                  for _ in range(random.randint(1, 10))]
    test_edges = [(u, v, w) for u, v, w in test_edges if u != v]
    assert dijkstra(n, test_edges, 0) == brute_shortest_path_weighted(n, test_edges, 0)

print("OK: Dijkstra在边界情况(单节点/无边)下全部正确, "
      "15组随机测试与暴力枚举路径找最短距离的结果完全一致")
```
本机实测:Dijkstra 在验证案例中正确找到 `0→2→1→3`(总权重4)比 `0→1→3`(总权重5)更短的路径;单节点、无边这几类边界情况均正确;15 组随机测试中,Dijkstra 和暴力枚举所有路径的结果完全一致。

**面试怎么问 + 追问链:** "Dijkstra 算法用堆优化后的复杂度是多少?" → 追问"如果图非常稠密(边数接近 V²),用堆优化还是用朴素的 O(V²) 实现(每次线性扫描找最小值)更好?"(稠密图下,朴素 O(V²) 实现可能反而更快——堆优化的 O((V+E)logV) 在 E 接近 V² 时会变成 O(V² log V),比朴素版本的 O(V²) 还慢;这个追问检验的是能否理解"堆优化"不是无条件的性能提升,要结合图的稠密程度具体分析,呼应[07类知识点1](07-heaps-and-priority-queues.md#1-堆的数组表示与基本操作)"复杂度优势可能被具体场景抵消"这个反复出现的主题)。

**常见坑:**
1. 忘记堆里可能存在"过期"记录(同一个节点因为多次被更新距离,可能多次入堆)——如果不在弹出时检查 `visited`,同一个节点可能被重复处理,虽然通常不会导致错误结果(因为距离更小的记录总是先被处理),但会有不必要的重复计算。
2. 在存在负权边的图上使用 Dijkstra——[知识点9](15-graphs-advanced.md#9-图论进阶常见坑)会具体复现这个错误的真实后果,这里先提醒这是 Dijkstra 最容易被误用的场景。

---

## 2. Bellman-Ford 算法:处理负权边

**签名/是什么:**
```
对所有边做 V-1 轮"松弛"操作(尝试用每条边更新终点的最短距离)，
V-1轮之后如果还能继续松弛，说明存在负权环
```

**一句话:** Bellman-Ford 算法通过对所有边反复做 V-1 轮松弛操作,能正确处理包含负权边的图(只要没有负权环)——比 Dijkstra 更"笨"(复杂度更高,O(V·E)),但适用范围更广,还能额外检测负权环的存在。

**底层机制/为什么这样设计:** 最短路径最多经过 V-1 条边(如果一条路径经过节点数超过V,必然存在重复访问某个节点,而没有负权环时重复访问不会让路径更短,所以最优路径不需要重复经过任何节点)——每一轮松弛操作,至少能把"当前已知最短路径恰好是 k 条边"这一类路径的信息传播出去,经过 V-1 轮,所有可能的最短路径(最多 V-1 条边)都已经被正确计算。**第 V 轮**(多做一轮)如果发现还能继续松弛(某条边还能让终点距离变得更小),说明存在负权环——因为如果没有负权环,V-1 轮之后所有距离都应该已经收敛到最优值,不可能再被更新。

**AI 研究/工程场景:** 负权边在真实场景里对应"某种操作能带来净收益"的情形——比如某些套利检测场景(汇率转换构成的图,如果存在一个能让"资金量"净增长的循环路径,就是套利机会),Bellman-Ford 检测负权环的能力,正是这类"是否存在能无限获利的循环"问题的标准算法工具。

**可运行例子:**
```python
def bellman_ford(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    has_negative_cycle = False
    for u, v, w in edges:   # 多做第V轮,检测是否还能继续松弛
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
    return dist, has_negative_cycle

# 含负权边(但无负权环)的图,能正确算出最短距离
neg_edge_graph = [(0, 1, 4), (0, 2, 1), (2, 1, -2), (1, 3, 1), (2, 3, 5)]
dist, has_cycle = bellman_ford(4, neg_edge_graph, 0)
assert dist == [0, -1, 1, 0]   # 0->2->1 = 1+(-2) = -1,比0->1=4更短
assert has_cycle is False

# 在没有负权边的图上,结果应该和Dijkstra完全一致
import heapq
from collections import defaultdict
def dijkstra(n, edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    visited = set()
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        for nb, w in adj[node]:
            if dist[node] + w < dist[nb]:
                dist[nb] = dist[node] + w
                heapq.heappush(heap, (dist[nb], nb))
    return dist

non_neg_graph = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
bf_dist, _ = bellman_ford(4, non_neg_graph, 0)
assert bf_dist == dijkstra(4, non_neg_graph, 0)

# 负权环检测:构造一个真实的负权环
negative_cycle_graph = [(0, 1, 1), (1, 2, -3), (2, 0, 1)]   # 环总权重 1-3+1=-1,是负权环
_, has_neg_cycle = bellman_ford(3, negative_cycle_graph, 0)
assert has_neg_cycle is True

no_cycle_graph = [(0, 1, 1), (1, 2, 1)]
_, no_cycle_result = bellman_ford(3, no_cycle_graph, 0)
assert no_cycle_result is False

print(f"OK: Bellman-Ford在含负权边(无负权环)的图上正确算出最短距离{dist}, "
      f"在无负权边图上与Dijkstra结果完全一致; 负权环检测在'确实有负权环'和'没有负权环'"
      f"两种情况下均正确判断")
```
本机实测:Bellman-Ford 在含负权边(但无负权环)的图上,正确算出 `0→2→1` 这条经过负权边的路径(总权重-1)比直接 `0→1`(权重4)更短;在无负权边的图上,和 Dijkstra 的结果完全一致;负权环检测在真实构造的负权环(总权重-1的三角形)和无环图上均正确判断。

**面试怎么问 + 追问链:** "Bellman-Ford 为什么需要恰好 V-1 轮,不能更少吗?" → 追问"能不能提前终止(如果某一轮完全没有发生任何松弛,提前结束)?"(可以——这是一个常见的实用优化:如果某一轮遍历所有边后没有任何一条边被成功松弛,说明所有距离都已经收敛,后续轮次不会再有变化,可以提前退出,不需要死板地跑满 V-1 轮;这个追问检验的是能否在保证正确性的前提下,进一步思考实践中的效率优化,而不是把"V-1轮"当作不能变通的铁律)。

**常见坑:**
1. 松弛操作时忘记检查 `dist[u] != float('inf')`——如果 `u` 本身还不可达(距离是无穷大),`dist[u] + w` 在 Python 里虽然不会报错(无穷大加有限数还是无穷大),但如果 `w` 是负数,`inf + (负数)` 依然是 `inf`,不会引发错误,不过这个检查依然是让代码逻辑更清晰、避免在其他语言实现时因为溢出规则不同而出错的好习惯。
2. 检测负权环时,只做了 V-1 轮就直接停止,没有做第 V 轮的额外检查——这会导致完全无法发现图中可能存在的负权环,把 Bellman-Ford 应用在真正含有负权环的图上时,会静默返回一个没有意义的"最短距离"(负权环意味着理论上可以无限绕环获得更小的距离,不存在有意义的最短路径)。

---

## 3. Floyd-Warshall 算法:全源最短路径

**签名/是什么:**
```
dist[i][j] = 从i到j的最短距离，初始为直接边权(不存在则为无穷大)
三重循环: for k in range(n): for i: for j: dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
```

**一句话:** Floyd-Warshall 用一个简洁的三重循环,一次性计算出**任意两点之间**的最短距离(全源最短路径),而不像 Dijkstra/Bellman-Ford 那样每次只能算出"一个起点到所有其他点"的最短距离——代价是复杂度固定为 O(V³),不管图是稀疏还是稠密。

**底层机制/为什么这样设计:** 核心思想是动态规划(呼应[10类](10-dynamic-programming-basics.md))——`dist[i][j]` 逐步优化,每一轮"引入"一个新的中间节点 `k`,判断"经过 k 中转"是否能让 `i` 到 `j` 的距离变得更短。三重循环的**顺序不能颠倒**:`k` 必须是最外层循环——这保证了当计算 `dist[i][j]`(是否经过 k 中转更短)时,`dist[i][k]` 和 `dist[k][j]` 这两个值已经**考虑过所有编号小于 k 的节点作为中间节点**的优化结果,这是一个需要精确理解的动态规划状态转移顺序,如果把 `k` 放在内层循环,得到的结果可能是错误的(引用了还没有充分优化的中间距离)。

**AI 研究/工程场景:** 当需要**频繁查询任意两点间的最短距离**(而不是固定一个起点)时,Floyd-Warshall 通过一次 O(V³) 预处理换取后续 O(1) 查询,是"批量预处理换取后续常数时间查询"([13类知识点4](13-bit-manipulation-and-math.md#4-质数筛法埃拉托斯特尼筛法)质数筛法的同类思路)的又一次应用——图规模不大(比如V在几百以内)、但查询量很大的场景,这个权衡通常是划算的。

**可运行例子:**
```python
def floyd_warshall(n, edges):
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)   # 处理重边:保留较小的权重
    for k in range(n):          # k必须是最外层循环
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
fw_result = floyd_warshall(4, edges)
assert fw_result[0] == [0, 3, 1, 4]   # 从0出发的结果应该和Dijkstra一致

import heapq
from collections import defaultdict
def dijkstra(n, edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    visited = set()
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        for nb, w in adj[node]:
            if dist[node] + w < dist[nb]:
                dist[nb] = dist[node] + w
                heapq.heappush(heap, (dist[nb], nb))
    return dist

# 交叉验证:Floyd-Warshall的每一行,应该和以该节点为起点跑Dijkstra的结果一致
for start in range(4):
    assert fw_result[start] == dijkstra(4, edges, start)

assert floyd_warshall(1, [])[0] == [0]                                  # 单节点
assert floyd_warshall(3, [])[0][1] == float('inf')                         # 无边,不可达

print("OK: Floyd-Warshall全源最短路径在边界情况(单节点/无边)下全部正确, "
      "对每个起点单独跑Dijkstra的结果与Floyd-Warshall对应行完全一致")
```
本机实测:Floyd-Warshall 计算出的从节点0出发的最短距离,和单独跑 Dijkstra 的结果完全一致;对图中每一个节点分别作为起点跑 Dijkstra,结果都和 Floyd-Warshall 对应的那一行完全匹配;单节点、无边这几类边界情况均正确。

**面试怎么问 + 追问链:** "Floyd-Warshall 的三重循环里,为什么 `k` 必须是最外层?" → 追问"如果把 `i` 放在最外层,`k` 放在最内层,会得到什么结果?"(会得到错误结果——这个顺序会导致计算 `dist[i][j]` 时,`dist[i][k]` 可能还没有考虑完所有更小编号节点作为中转的优化,状态转移依赖的前提条件不成立;这个追问检验的是能否具体解释"为什么顺序错了会出错",而不是只记住"k必须在最外层"这个结论本身)。

**常见坑:**
1. 三重循环的顺序写错(`k` 不在最外层)——本知识点已经通过追问链具体解释了这个错误的后果,是 Floyd-Warshall 实现里最容易忽视、后果也最隐蔽的错误(程序不会报错,只是某些距离没有被正确优化)。
2. 处理重边(两个节点之间有多条边)时,直接覆盖而不是取较小值——如果同一对节点间存在多条边,应该只保留权重最小的那条作为初始距离,直接覆盖可能会意外使用一条权重更大的边。

---

## 4. 最小生成树:Kruskal 算法

**签名/是什么:**
```
把所有边按权重从小到大排序，依次尝试加入生成树，
用并查集(呼应14类)判断"加入这条边会不会成环"，不成环才加入
```

**一句话:** Kruskal 算法用贪心策略构建最小生成树(连接所有节点、总权重最小的树形子图)——按权重从小到大排序所有边,每条边只要不会和已选边构成环,就贪心地加入,并查集是判断"是否成环"最高效的工具。

**底层机制/为什么这样设计:** Kruskal 算法的贪心正确性可以用[11类知识点5](11-greedy-algorithms.md#5-如何证明贪心算法正确性交换论证法)交换论证法证明("最小的边,一定存在于某个最小生成树中,除非加入它会成环")。判断"加入一条边是否会成环"等价于判断"这条边的两个端点是否已经在同一个连通分量里"——如果两个端点已经通过其他已选边连通,再加入这条边只会形成环(多余),不加入;如果不连通,加入这条边能把两个分量合并成一个更大的连通分量,并查集的 `union` 操作天然对应这个"合并连通分量"的语义,`find` 判断"是否已连通"正是判环的高效手段(呼应[14类知识点5/6](14-graphs-basics.md#5-并查集基础实现))。

**AI 研究/工程场景:** [huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练集群网络拓扑,如果需要设计"连接所有计算节点、总网络布线成本最小"的通信拓扑结构,最小生成树是这类基础设施规划问题的标准数学模型。

**可运行例子:**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False   # 已经连通,加入会成环
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal_mst(n, edges):
    sorted_edges = sorted(edges, key=lambda e: e[2])   # 按权重从小到大排序
    uf = UnionFind(n)
    total_weight = 0
    mst_edges = []
    for u, v, w in sorted_edges:
        if uf.union(u, v):   # union返回True说明成功合并(不成环),加入生成树
            total_weight += w
            mst_edges.append((u, v, w))
    return total_weight, mst_edges

edges = [(0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 4), (3, 4, 2), (4, 2, 6)]
total, mst = kruskal_mst(5, edges)
assert total == 8   # 已知这个经典案例的最小生成树总权重是8
assert len(mst) == 4   # n个节点的生成树恰好有n-1条边

# 验证mst_edges确实构成一棵树(连通且无环):n-1条边,且用这些边跑并查集,最终只有一个连通分量
verify_uf = UnionFind(5)
for u, v, w in mst:
    verify_uf.union(u, v)
roots = set(verify_uf.find(i) for i in range(5))
assert len(roots) == 1   # 只有一个连通分量,说明生成树确实连接了所有节点

assert kruskal_mst(1, [])[0] == 0        # 单节点,不需要任何边
assert kruskal_mst(1, [])[1] == []

print(f"OK: Kruskal最小生成树总权重={total}(与已知经典案例答案一致), "
      f"边数={len(mst)}=n-1; 验证选出的边确实构成一棵连接全部节点的树(无环且连通); "
      f"单节点边界情况正确")
```
本机实测:Kruskal 算法在经典测试案例上得到总权重为8的最小生成树,和该案例已知的标准答案一致;选出的4条边(恰好是 n-1 条)经独立验证确实构成一棵连通且无环的树;单节点边界情况正确处理。

**面试怎么问 + 追问链:** "Kruskal 算法的复杂度主要花在哪一步?" → 追问"如果边的数量远大于节点数量(稠密图),Kruskal 排序所有边的开销会不会成为瓶颈?"(会——排序边的复杂度是 O(E log E),在稠密图(E接近V²)下这个开销可能超过并查集操作本身;这种情况下[知识点5](15-graphs-advanced.md#5-最小生成树prim-算法)的 Prim 算法(复杂度更依赖节点数而不是边数)可能是更好的选择;这个追问检验的是能否根据图的稠密程度,在 Kruskal 和 Prim 之间做出合理选型,而不是认为某一个总是更优)。

**常见坑:**
1. 忘记按权重排序就直接遍历边——Kruskal 贪心策略的正确性完全依赖"优先尝试权重最小的边"这个顺序,不排序会得到完全错误的结果(甚至可能不是树,或者不是最小的)。
2. 用了没有优化的并查集(呼应[14类知识点5](14-graphs-basics.md#5-并查集基础实现)的基础版本)——在边数很多的大图上,`find` 操作的效率会直接影响 Kruskal 算法的整体表现,应该使用[14类知识点6](14-graphs-basics.md#6-并查集优化路径压缩--按秩合并)的优化版本。

---

## 5. 最小生成树:Prim 算法

**签名/是什么:**
```
从任意一个节点开始，用堆(呼应07类)维护"当前生成树能够到达的、权重最小的外部边"，
每次贪心地把权重最小的这条边纳入生成树，直到所有节点都被纳入
```

**一句话:** Prim 算法从"一棵只有一个节点的树"开始,每一步贪心地选择"连接树内节点和树外节点的、权重最小的那条边"扩展这棵树,直到覆盖所有节点——和 Kruskal"全局按边权排序"的视角不同,Prim 是"以树为中心逐步扩张"的视角。

**底层机制/为什么这样设计:** Prim 算法维护一个不断扩大的"树内节点"集合,用堆存储"当前树能够到达的所有外部候选边",每次取出权重最小的候选:如果这条边的外部端点还没被纳入树,就把它纳入(树扩大一个节点),并把这个新节点的所有外部边加入候选堆;如果这个端点已经在树内(说明这是一条冗余的候选,呼应[07类知识点2](07-heaps-and-priority-queues.md#2-python-heapq-模块使用与内部实现)堆里可能存在"过期"记录这个现象),跳过。这个策略同样能用交换论证法证明正确性,和 Kruskal 是解决同一个问题的两种不同贪心策略,理论上都保证得到最小生成树,但具体的复杂度特征不同(Prim 的复杂度更依赖节点数,Kruskal 更依赖边数排序)。

**AI 研究/工程场景:** Prim 算法"以已确定的连通子图为中心逐步扩张"这个模式,和[04类知识点3](04-binary-search.md#3-旋转排序数组的二分查找)"从局部已知信息逐步扩大确定范围"的思路有相通之处;在图规模较大但相对稠密(边数远超节点数)的场景,Prim 通常是比 Kruskal 更合适的最小生成树算法选择。

**可运行例子:**
```python
import heapq
from collections import defaultdict

def prim_mst(n, edges):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))
    visited = {0}
    heap = adj[0][:]
    heapq.heapify(heap)
    total_weight = 0
    while heap and len(visited) < n:
        w, node = heapq.heappop(heap)
        if node in visited:      # 过期候选,跳过(呼应知识点1 Dijkstra的类似处理)
            continue
        visited.add(node)
        total_weight += w
        for w2, nb in adj[node]:
            if nb not in visited:
                heapq.heappush(heap, (w2, nb))
    return total_weight

edges = [(0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 4), (3, 4, 2), (4, 2, 6)]
prim_total = prim_mst(5, edges)
assert prim_total == 8   # 应该和Kruskal得到同样的最小总权重(呼应知识点4)

def kruskal_mst_total(n, edges):
    class UF:
        def __init__(self, n):
            self.parent = list(range(n))
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            self.parent[px] = py
            return True
    uf = UF(n)
    total = 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if uf.union(u, v):
            total += w
    return total

assert prim_mst(5, edges) == kruskal_mst_total(5, edges)   # 两种算法结果必须一致

assert prim_mst(1, []) == 0   # 单节点

# 交叉验证:随机图上,两种算法得到的最小生成树总权重必须一致
import random
random.seed(81)
for _ in range(15):
    n = random.randint(2, 8)
    # 保证图连通:先生成一棵随机树打底,再加一些随机额外边
    base_edges = [(i, random.randint(0, i-1), random.randint(1, 20)) for i in range(1, n)]
    extra_edges = [(random.randint(0, n-1), random.randint(0, n-1), random.randint(1, 20))
                   for _ in range(random.randint(0, 5))]
    test_edges = base_edges + [(u, v, w) for u, v, w in extra_edges if u != v]
    assert prim_mst(n, test_edges) == kruskal_mst_total(n, test_edges)

print(f"OK: Prim最小生成树总权重={prim_total}, 与Kruskal算法结果一致(同为8); "
      f"单节点边界情况正确; 15组随机连通图测试, 两种算法结果完全一致")
```
本机实测:Prim 算法在同一个经典案例上得到总权重8,和 Kruskal 算法结果完全一致;单节点边界情况正确;15 组随机生成的连通图测试中,Prim 和 Kruskal 两种完全不同的贪心策略,在每一组测试上都得到相同的最小生成树总权重——这是对"两种算法都能正确求解同一个问题"这个理论结论的真实交叉验证。

**面试怎么问 + 追问链:** "Prim 和 Kruskal 算法,复杂度分别是多少,该怎么选择?" → 追问"如果图是稀疏图(边数接近节点数),用邻接矩阵实现的朴素 Prim(不用堆优化,O(V²))和用堆优化的 Kruskal,哪个更合适?"(稀疏图下,堆优化的 Kruskal(O(E log E),E接近V时约等于O(V log V))通常明显优于朴素 O(V²) 的 Prim;这个追问检验的是能否根据图的具体稀疏/稠密特征,以及各自实现是否做了对应的优化,做出更细致的算法选型判断,而不是记住一个笼统的"Prim适合稠密图、Kruskal适合稀疏图"的结论而不理解背后的复杂度依据)。

**常见坑:**
1. 堆里加入候选边时,忘记检查目标节点是否已经在树内——虽然[知识点1](15-graphs-advanced.md#1-dijkstra-算法堆优化实现)提到弹出时检查 `visited` 能规避大部分问题,但如果连加入堆这一步都不加判断,会让堆里积累大量无用的候选,拖累效率。
2. 混淆 Prim 算法本身的正确性和"图必须连通"这个前提——如果原图不连通,Prim 算法从一个节点开始扩张,永远无法覆盖到不连通的那部分节点,`len(visited) < n` 这个循环条件会一直无法满足导致堆耗尽提前退出,得到的只是"一个连通分量内的最小生成树"而不是完整答案,调用前需要确认图确实连通,或者对每个连通分量分别处理。

---

## 6. 强连通分量:Tarjan 算法

**签名/是什么:**
```
强连通分量(SCC): 有向图中，任意两点都能互相到达的最大节点集合
Tarjan算法: DFS一遍完成，用 index(发现顺序编号) 和 low(能追溯到的最早祖先编号) 判断SCC边界
```

**一句话:** Tarjan 算法用**一次** DFS 遍历就能找出有向图里所有的强连通分量。这里先从零补一个[14类知识点2](14-graphs-basics.md#2-dfs-遍历与应用连通分量与环检测)讲 DFS 时没有展开的概念:对一个图做 DFS 遍历,走过的每条边都能按"访问情况"分类,本知识点只需要认识两种——**树边(tree edge)**:DFS 第一次访问某个新节点时经过的那条边,所有树边合起来恰好构成一棵"DFS 生成树";**回边(back edge)**:指向一个"当前仍在递归栈里"的祖先节点的边,意味着沿树边往下走之后又绕回了祖先,说明存在环。比如三个节点 `0→1→2→0` 首尾相连:DFS 从 0 出发,`0→1`、`1→2` 都是第一次访问新节点,是树边;`2→0` 指向的节点 0 还在当前递归路径上(尚未返回),是回边——这条回边正是"0、1、2 构成一个环、进而构成一个 SCC"的直接证据(下面"底层机制"会用真实运行的完整例子把这个过程逐步跑一遍)。有了这两个概念,`index`(DFS 访问顺序编号)和 `low`(这个节点通过"树边+至多一条回边"能追溯到的最早祖先编号)就好理解了:`low[node]` 回答的是"从 `node` 出发,只顺着树边往下、最多再借助一条回边,能碰到的编号最小(即最早被发现)的祖先是谁"。当 `low[node] == index[node]` 时,说明找到了一个 SCC 的"根",栈里从这个节点到栈顶的所有节点构成一个完整的强连通分量。

**底层机制/为什么这样设计:** `low[node]` 的更新规则是算法的核心:①如果邻居 `nb` 还没访问过,递归访问它,`low[node] = min(low[node], low[nb])`(继承子树能追溯到的最早祖先);②如果 `nb` 已经访问过、且还在栈上(说明是当前 DFS 路径上的祖先,构成一个环),`low[node] = min(low[node], index[nb])`——**注意这里用的是 `index[nb]` 而不是 `low[nb]`**,这是一个容易出错的细节,原因是 `nb` 可能不是当前 SCC 真正的"根",用 `low[nb]` 可能引用一个还没确定的、可能被后续更新的值,用 `index[nb]`(一旦赋值就不再改变)更安全。`low[node]==index[node]` 意味着"这个节点及其子树,追溯不到比它自己更早的祖先了",说明它是当前 SCC 里"最早被发现"的节点,即整个 SCC 的根。

这一步还需要补上一个容易被跳过的问题:为什么"栈里从 `node` 到栈顶"这一段,不多不少恰好是一个完整的 SCC?SCC 要求"双向可达",要分两个方向看。"`node` 能到达这些节点"这个方向不需要额外证明——DFS 是"先递归到底、再逐层返回"的顺序,`node` 之后才入栈的节点必然是它 DFS 子树内的节点,顺着树边从 `node` 往下走一定能到达。反方向"这些节点能到达 `node`"依赖栈维护的性质:**栈里的节点,永远是"已经访问过、但还没有被归入某个已完成 SCC"的节点**,一旦某个节点所在的 SCC 被确定(触发弹栈),就永久离开栈。如果 `node` 子树里的某个节点在 `node` 完成之前就已经被弹出(独立构成了自己的 SCC),它显然到不了 `node`,也确实不会被这次弹栈波及。而对于那些仍然留在栈上、没有被提前弹出的子树内节点,恰恰说明它们各自都能通过某条回边追溯到 `node` 或者比 `node` 更早访问的祖先(否则它们会在自己那一层就满足 `low==index`,提前被弹出)。`low[node]==index[node]` 这个条件又排除了"追溯到比 `node` 更早的节点"这种可能——真存在这样一条路径的话,会体现为 `low[node]<index[node]`。两者合起来:这些仍留在栈上的节点,能追溯到的最早祖先不多不少正好是 `node`,也就是说它们能到达 `node`。双向可达都成立,`node` 到栈顶这一段就是彼此可达的最大集合,弹出的这些节点不会漏掉子树内该属于这个 SCC 的节点,也不会把子树外、不相关的节点卷进来(它们更早入栈,排在 `node` 下方,这次弹栈碰不到它们)。

用一个具体例子把这套机制"跑一遍"会更直观。取和下面"可运行例子"完全相同的图:`edges = [(0,1), (1,2), (2,0), (1,3), (3,4)]`(0→1→2→0 构成一个环,1→3→4 是环外的一条链)。DFS 从节点 0 出发,`0→1`、`1→2`、`1→3`、`3→4` 都是第一次访问新节点,是树边;只有 `2→0` 指向仍在栈上的祖先 0,是唯一的回边——呼应"一句话"部分的树边/回边区分。下表是把"可运行例子"里的 `tarjan_scc` 函数临时加上打印语句(不改变任何判断逻辑,只是把每一步的中间状态打印出来)后,真实运行(而非手算)得到的 `index`/`low`/栈内容变化记录(仅列出发生实质变化的关键步骤):

| 步骤 | 发生的事情 | index/low(该节点) | 栈内容(栈底→栈顶) |
|---|---|---|---|
| 1 | 进入 `strongconnect(0)` | index[0]=0, low[0]=0 | [0] |
| 2 | 沿树边 0→1 进入 `strongconnect(1)` | index[1]=1, low[1]=1 | [0, 1] |
| 3 | 沿树边 1→2 进入 `strongconnect(2)` | index[2]=2, low[2]=2 | [0, 1, 2] |
| 4 | 2 遇到边 2→0,0 仍在栈上→回边,`low[2]=min(2, index[0]=0)=0` | low[2]=0 | [0, 1, 2] |
| 5 | 2 无更多出边,返回;1 收到子节点 2 的 low,`low[1]=min(1,0)=0` | low[1]=0 | [0, 1, 2] |
| 6 | 沿树边 1→3 进入 `strongconnect(3)` | index[3]=3, low[3]=3 | [0, 1, 2, 3] |
| 7 | 沿树边 3→4 进入 `strongconnect(4)` | index[4]=4, low[4]=4 | [0, 1, 2, 3, 4] |
| 8 | 4 无出边;`low[4]==index[4]`(4==4)→4 是 SCC 根,弹栈 | — | 弹出 `{4}`,栈变为 [0, 1, 2, 3] |
| 9 | 3 收到子节点 4 的 low(`low[3]=min(3,4)=3`);3 无更多出边;`low[3]==index[3]`(3==3)→3 是 SCC 根,弹栈 | — | 弹出 `{3}`,栈变为 [0, 1, 2] |
| 10 | 1 收到子节点 3 的 low(`low[1]=min(0,3)=0`,不变);1 无更多出边;`low[1]≠index[1]`(0≠1)→1 不是根,直接返回 | index[1]=1, low[1]=0 | [0, 1, 2] |
| 11 | 0 收到子节点 1 的 low(`low[0]=min(0,0)=0`);0 无更多出边;`low[0]==index[0]`(0==0)→0 是 SCC 根,弹栈直到 0 | — | 弹出 `{2, 1, 0}`,栈变为 [] |

最终得到 `{4}`、`{3}`、`{2,1,0}` 三个 SCC,和下面"可运行例子"里 `assert` 验证的结果完全一致。这张表也让上一段的抽象论证变得具体:节点 2 在第4步遇到回边后 `low` 变成 0,这个"0"沿着 1 一路传递(第5步、第10步),最终在第11步让 `low[0]==index[0]` 成立,触发一次性弹出 `{2,1,0}` 这一整个环;而节点 3、4 不在环上,各自的 `low` 从未被环上的信息影响到,所以在各自完成时立刻以"单节点 SCC"的身份弹栈——这正是"树边+至多一条回边"这个 `low` 定义,在具体执行中的样子。

**AI 研究/工程场景:** [huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式系统里,如果节点间的"依赖/调用关系"图里存在强连通分量(意味着一组节点互相调用,形成循环依赖),这在软件架构层面通常是需要警惕的信号(循环依赖会带来初始化顺序等问题)——SCC 检测是静态分析工具识别这类循环依赖的算法基础。

**可运行例子:**
```python
from collections import defaultdict

def tarjan_scc(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = {}
    result = []

    def strongconnect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        for nb in adj[node]:
            if nb not in index:
                strongconnect(nb)
                lowlink[node] = min(lowlink[node], lowlink[nb])
            elif on_stack.get(nb, False):
                lowlink[node] = min(lowlink[node], index[nb])   # 用index而不是lowlink
        if lowlink[node] == index[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == node:
                    break
            result.append(component)

    for node in range(n):
        if node not in index:
            strongconnect(node)
    return result

# 0,1,2构成一个环(SCC),3和4各自独立(3->4是单向,不构成环)
edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)]
sccs = tarjan_scc(5, edges)
scc_sets = [set(c) for c in sccs]
assert {0, 1, 2} in scc_sets
assert {3} in scc_sets
assert {4} in scc_sets
assert len(sccs) == 3   # 恰好3个强连通分量

assert len(tarjan_scc(1, [])) == 1   # 单节点自成一个SCC
assert len(tarjan_scc(3, [])) == 3     # 没有边,每个节点各自是一个SCC

# 交叉验证:用暴力方式(对每一对节点检查是否互相可达)验证SCC划分的正确性
def brute_verify_scc(n, edges, sccs):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    def reachable(src, dst):
        visited = {src}
        stack = [src]
        while stack:
            node = stack.pop()
            if node == dst:
                return True
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return dst == src
    for component in sccs:
        for a in component:
            for b in component:
                if not reachable(a, b):
                    return False
    return True

assert brute_verify_scc(5, edges, sccs)

print(f"OK: Tarjan算法找出的强连通分量{scc_sets}正确(0,1,2构成环, 3和4各自独立); "
      f"边界情况(单节点/无边)全部正确; 暴力可达性交叉验证确认每个SCC内部任意两点确实互相可达")
```
本机实测:Tarjan 算法正确识别出 `{0,1,2}` 是一个强连通分量(它们构成一个环,互相可达),`{3}` 和 `{4}` 各自独立成一个分量(因为 `1→3→4` 是单向依赖,不构成环);边界情况(单节点、无边)均正确;用暴力可达性检查独立验证了每个 SCC 内部任意两点确实能够互相到达。

**面试怎么问 + 追问链:** "Tarjan 算法里,`low[node] = min(low[node], index[nb])` 这一步为什么用 `index[nb]` 而不是 `low[nb]`?" → 追问"如果错误地用了 `low[nb]`,会导致什么后果?"(可能导致 SCC 被错误地"过度合并"——`nb` 的 `low` 值在这个时刻可能还没有最终确定(它可能还在等待自己子树的进一步更新,或者本身不是真正的SCC根),用一个还可能变化的中间值去更新 `node` 的 `low`,可能引用到不属于当前正确 SCC 边界的信息,导致原本应该分开的两个 SCC 被误判成一个;这个追问检验的是能否精确说出这个细节的因果关系,而不是死记"用index不用low"这个结论)。

**常见坑:**
1. 混淆 `index`(节点被首次访问的顺序编号,一旦赋值不再改变)和 `low`(这个节点及其子树能追溯到的最早祖先编号,会随着DFS进行不断更新)——这两个概念的区别是 Tarjan 算法最核心也最容易混淆的地方。
2. 判断"节点是否在栈上"时,忘记维护或者错误维护 `on_stack` 这个状态——一个节点被访问过但已经出栈(说明它所在的 SCC 已经被完整处理并弹出),不应该再被当作"当前路径上的祖先"参与 `low` 值的更新,这个判断遗漏会导致算法把已经处理完的、不相关的节点错误地关联进当前正在构建的 SCC。

---

## 7. 网络流基础:Ford-Fulkerson 思想

**签名/是什么:**
```
最大流问题: 给定一个带容量的有向图(源点s, 汇点t), 求从s到t最多能同时"流过"多少流量
Ford-Fulkerson方法: 反复寻找"增广路径"(从s到t、每条边都还有剩余容量的路径),
                    沿路径推送尽可能多的流量, 直到找不到增广路径为止
```

**一句话:** 网络流问题求"从源点到汇点,受限于每条边容量,最多能同时传输多少流量"——Ford-Fulkerson 方法反复寻找"增广路径"(还有剩余容量可用的路径)并沿路径推送流量,直到再也找不到这样的路径,累积推送的流量总和就是最大流。

**底层机制/为什么这样设计:** 核心技巧是**反向边**(residual edge)——每次沿某条边推送流量 `f` 之后,除了正向边的剩余容量减少 `f`,还要给反向边增加 `f` 的"虚拟容量",这代表"如果后续发现这样分配不是最优的,可以通过反向边把已经分配出去的流量'退回来',改走别的路径"——这个"允许反悔"的机制,是保证算法最终能收敛到真正最大流(而不是卡在一个次优解上)的关键。用 BFS 而不是 DFS 寻找增广路径的具体实现,称为 Edmonds-Karp 算法(Ford-Fulkerson 方法的一种具体实现),能保证更好的复杂度上界。**最大流最小割定理**:一个网络的最大流量,恰好等于把源点和汇点分割开所需要"切断"的最小总容量(最小割)——这个定理提供了直觉:最大流受限于图中最"窄"的那个瓶颈,不管这个瓶颈具体在哪个位置。

**AI 研究/工程场景:** 网络流模型能解决很多表面上不像"流量"问题的实际问题(通过巧妙建模转化)——[huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练数据并行场景,如果需要计算"给定网络带宽限制,数据在计算节点间最多能以多快的速率同步",本质上就是最大流问题的一个应用实例。

**可运行例子:**
```python
from collections import defaultdict, deque

def max_flow_edmonds_karp(edges, source, sink):
    capacity = defaultdict(lambda: defaultdict(int))
    for u, v, cap in edges:
        capacity[u][v] += cap

    def bfs_find_augmenting_path():
        parent = {source: None}
        q = deque([source])
        while q:
            u = q.popleft()
            if u == sink:
                break
            for v in capacity[u]:
                if v not in parent and capacity[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            return None, 0
        path, bottleneck = [], float('inf')
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, capacity[u][v])
            path.append((u, v))
            v = u
        return path, bottleneck

    total_flow = 0
    while True:
        path, bottleneck = bfs_find_augmenting_path()
        if path is None:
            break
        for u, v in path:
            capacity[u][v] -= bottleneck
            capacity[v][u] += bottleneck   # 反向边:允许后续"反悔"
        total_flow += bottleneck
    return total_flow

# 单路径场景:最大流应该恰好等于路径上的瓶颈(最小容量)
simple_edges = [('s', 'a', 5), ('a', 't', 3)]
assert max_flow_edmonds_karp(simple_edges, 's', 't') == 3

# 两条完全独立的路径:最大流应该是两条路径容量之和
parallel_edges = [('s', 'a', 2), ('a', 't', 2), ('s', 'b', 3), ('b', 't', 3)]
assert max_flow_edmonds_karp(parallel_edges, 's', 't') == 5

# 经典教材案例(带有中间汇聚/分支结构)
textbook_edges = [
    ('s', 'a', 3), ('s', 'b', 2),
    ('a', 'b', 1), ('a', 'c', 3),
    ('b', 'd', 2),
    ('c', 't', 2), ('d', 'c', 1), ('d', 't', 3),
]
textbook_result = max_flow_edmonds_karp(textbook_edges, 's', 't')
assert textbook_result == 4

# 没有任何路径能到达汇点
disconnected_edges = [('s', 'a', 5)]   # a无法到达t
assert max_flow_edmonds_karp(disconnected_edges, 's', 't') == 0

print(f"OK: 最大流在单路径(瓶颈=3)/并行双路径(总和=5)/经典教材案例(={textbook_result})/"
      f"完全不连通(=0)等情况下全部正确")
```
本机实测:单路径场景下,最大流精确等于路径上的瓶颈容量;两条完全独立路径的最大流是两者容量之和;经典教材案例得到最大流为4,和该案例的标准答案一致;源点汇点完全不连通时,最大流正确返回0。

**面试怎么问 + 追问链:** "为什么 Ford-Fulkerson 方法需要引入反向边,不引入会有什么问题?" → 追问"能不能构造一个具体例子,说明不用反向边(不允许'反悔')会导致算法卡在次优解?"(可以——比如两条路径共享某条边,如果贪心地先把这条共享边的容量全部分配给第一条路径,可能导致第二条路径本可以贡献的流量无法实现,而如果允许通过反向边"撤回"一部分已分配给第一条路径的流量、改分配给能带来更大总流量的组合,才能达到真正的最大流;这个追问检验的是能否具体构造反例说明这个机制的必要性,而不是抽象地背诵"需要反向边"这个结论)。

**常见坑:**
1. 用 DFS(而不是 BFS)寻找增广路径,且没有注意选择路径的策略——DFS 版本的 Ford-Fulkerson 在某些精心构造的图上,理论最坏情况复杂度会很差(依赖具体路径的选择顺序);Edmonds-Karp(固定用BFS)有更好的理论复杂度保证,这是这两种具体实现之间实际的差异,不只是"任选一种遍历方式都一样"。
2. 实现反向边时,忘记正确初始化或者更新反向边的容量——如果反向边的容量记录和更新逻辑有误,算法可能无法正确"反悔"之前的分配决策,得到小于真实最大流的错误结果。

---

## 8. 二分图最大匹配:匈牙利算法

**签名/是什么:**
```
最大匹配: 二分图(呼应14类知识点7)左右两侧节点之间, 尽可能多地找出"一对一"的匹配边
匈牙利算法: 对左侧每个节点尝试找一个还未匹配的右侧节点; 如果候选都已被占用,
           尝试"说服"占用者让出位置、去匹配别的候选(递归寻找增广路径)
```

**一句话:** 匈牙利算法(基于增广路径的思想,和网络流有深层联系)在二分图上寻找最大匹配——核心技巧是,当一个左侧节点的候选右侧节点都已经被占用时,不是直接放弃,而是递归地尝试"能不能让占用者去匹配它自己的其他候选,腾出位置给当前节点"。

**底层机制/为什么这样设计:** `try_kuhn(u, visited)` 函数的逻辑:遍历 `u` 的每个候选右侧节点 `v`,如果 `v` 还没被匹配(`match_right[v]==-1`)或者"当前占用 `v` 的左侧节点,能够被重新安排匹配到其他候选"(递归调用 `try_kuhn(match_right[v], visited)` 返回 `True`),就把 `v` 匹配给 `u`。这个"递归尝试腾位置"的过程,本质上是在寻找一条"增广路径"(未匹配-已匹配-未匹配-已匹配...交替,首尾都是未匹配节点)——每找到一条这样的路径,就能把匹配数量增加1(交替翻转路径上"是否匹配"的状态)。`visited` 数组防止在同一次尝试里对右侧节点做重复的、可能死循环的探索。

**AI 研究/工程场景:** [知识点7](15-graphs-advanced.md#7-网络流基础ford-fulkerson-思想)提到过网络流能解决很多"看似不同"的问题——二分图最大匹配正是可以转化成最大流问题的经典案例之一(源点连接所有左侧节点、汇点连接所有右侧节点、每条边容量为1,最大流恰好等于最大匹配数)。实际应用场景:比如任务分配(每个任务只能分配给一个符合条件的执行者,每个执行者同时只能做一个任务),求"最多能同时分配多少个任务",就是二分图最大匹配的直接应用。

**可运行例子:**
```python
def hungarian_matching(n_left, n_right, adj):
    match_right = [-1] * n_right   # match_right[v] = 匹配到v的左侧节点编号,-1表示未匹配

    def try_kuhn(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_right[v] == -1 or try_kuhn(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    count = 0
    for u in range(n_left):
        visited = [False] * n_right
        if try_kuhn(u, visited):
            count += 1
    return count

# 左0能连右0,1; 左1只能连右0; 左2能连右1,2 —— 最大匹配应该是3(完美匹配)
adj = {0: [0, 1], 1: [0], 2: [1, 2]}
assert hungarian_matching(3, 3, adj) == 3

# 左0和左1都只能连右0(冲突,不能同时匹配) —— 最大匹配只能是1
conflict_adj = {0: [0], 1: [0]}
assert hungarian_matching(2, 1, conflict_adj) == 1

# 完全没有边 —— 最大匹配是0
no_edge_adj = {0: [], 1: []}
assert hungarian_matching(2, 2, no_edge_adj) == 0

# 交叉验证:与"最大流"方法(呼应知识点7)在同一个二分图上结果必须一致
from collections import defaultdict, deque
def max_flow_edmonds_karp(edges, source, sink):
    capacity = defaultdict(lambda: defaultdict(int))
    for u, v, cap in edges:
        capacity[u][v] += cap
    def bfs():
        parent = {source: None}
        q = deque([source])
        while q:
            u = q.popleft()
            if u == sink:
                break
            for v in capacity[u]:
                if v not in parent and capacity[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            return None, 0
        path, bottleneck = [], float('inf')
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, capacity[u][v])
            path.append((u, v))
            v = u
        return path, bottleneck
    total = 0
    while True:
        path, bottleneck = bfs()
        if path is None:
            break
        for u, v in path:
            capacity[u][v] -= bottleneck
            capacity[v][u] += bottleneck
        total += bottleneck
    return total

def bipartite_matching_via_maxflow(n_left, n_right, adj):
    edges = [('s', f'L{u}', 1) for u in range(n_left)]
    edges += [(f'R{v}', 't', 1) for v in range(n_right)]
    for u in adj:
        for v in adj[u]:
            edges.append((f'L{u}', f'R{v}', 1))
    return max_flow_edmonds_karp(edges, 's', 't')

assert hungarian_matching(3, 3, adj) == bipartite_matching_via_maxflow(3, 3, adj)

print("OK: 匈牙利算法在完美匹配/存在冲突/无边等情况下全部正确, "
      "与最大流方法在同一个二分图上求出的最大匹配数完全一致")
```
本机实测:匈牙利算法在完美匹配、存在冲突(两个左侧节点竞争同一个右侧节点)、完全无边这几类情况下均正确;与[知识点7](15-graphs-advanced.md#7-网络流基础ford-fulkerson-思想)的最大流方法在同一个二分图上求出的最大匹配数完全一致——直接验证了"二分图最大匹配可以转化为最大流问题"这个理论联系。

**面试怎么问 + 追问链:** "匈牙利算法的核心'尝试腾位置'的过程,和什么已经学过的概念本质相通?" → 追问"能不能具体说出,二分图最大匹配问题该怎么转化成最大流问题?"([知识点7](15-graphs-advanced.md#7-网络流基础ford-fulkerson-思想)已经用代码演示了具体转化方式:建一个虚拟源点连接所有左侧节点(容量1)、虚拟汇点连接所有右侧节点(容量1)、原图的每条边容量设为1,这个网络的最大流恰好等于二分图的最大匹配数;这个追问检验的是能否把两个表面上不同的算法(匈牙利算法 vs 网络流)在概念上真正联系起来,而不是把它们当作两个孤立的、需要分别记忆的技巧)。

**常见坑:**
1. `try_kuhn` 函数里的 `visited` 数组,如果在外层循环(对每个左侧节点 `u` 的尝试)之间没有重新初始化——会导致后续左侧节点的尝试被前面节点遗留的 `visited` 状态错误地限制,应该确保每次为新的左侧节点寻找匹配时,`visited` 都从全 `False` 重新开始。
2. 把二分图最大匹配问题和一般图(非二分图)的匹配问题混淆——匈牙利算法(以及本知识点这个实现)专门针对二分图设计,一般图的最大匹配需要更复杂的算法(如带花树算法),不能直接套用。

---

## 9. 图论进阶常见坑

**签名/是什么:**
```
Dijkstra在负权图上使用 -> 得到错误(通常偏大)的"最短距离"
Bellman-Ford忘记做检测负环的额外一轮 -> 无法发现图中存在的负权环
```

**一句话:** 图论进阶算法的坑,集中在"算法适用条件被忽视"——每个算法(Dijkstra/Bellman-Ford/Floyd-Warshall等)都有明确的适用前提(边权是否非负、是否可能有环等),忽视这些前提直接套用,得到的错误结果往往不会以报错的形式出现,而是安静地给出一个看似合理、实际错误的数字。

**底层机制/为什么这样设计:** Dijkstra 的贪心策略在负权图上失效的具体机制:算法一旦"确定"了某个节点的最短距离(从堆中弹出并标记 `visited`),就再也不会重新考虑这个节点——但如果存在负权边,一条"当前看起来更长"的路径,可能在后续经过一条负权边后,总权重反而变得更小,这种情况下 Dijkstra 已经过早地"锁定"了错误的最短距离,不会再回头修正。这不是理论上的边缘情况,是一个可以具体构造出来、真实复现的错误。

**AI 研究/工程场景:** 这类"算法有隐含前提条件,忽视前提会得到静默错误结果"的问题,呼应本系列多次强调的"复杂度/正确性分析要针对具体输入特征,不能脱离前提条件泛泛而谈"([01类知识点1](01-complexity-and-python-builtins.md#1-时间复杂度与空间复杂度分析方法论)以来的一贯主题)——图论算法的选型,第一步永远应该是"这个图有什么特征(有向/无向、带权/无权、权重是否可能为负、是否可能有环)",再决定用哪个算法,而不是拿到问题就直接套用最熟悉的那个。

**可运行例子:**
```python
import heapq
from collections import defaultdict

def dijkstra(n, edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    visited = set()
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        for nb, w in adj[node]:
            if dist[node] + w < dist[nb]:
                dist[nb] = dist[node] + w
                heapq.heappush(heap, (dist[nb], nb))
    return dist

def bellman_ford(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist

# 坑1: 在负权图上使用Dijkstra,现场复现真实的错误结果
# 注:第一次尝试构造反例时,想当然地以为"负权边直接指向的那个节点"会拿到错误距离,
# 现场验证发现这个具体实现并非如此——因为松弛操作会无条件更新dist数组(即使目标节点已经visited),
# 这个数组层面的"副作用更新"意外地让直接被负权边指向的节点最终显示了正确数值。
# 真正的错误出现在更隐蔽的地方:那个节点自己的"下游"节点——因为该节点在被错误距离"锁定"、
# 已经把自己的出边全部松弛过一遍之后,才收到负权边带来的修正,而这个修正不会传播到它的下游,
# 下游节点已经用旧的错误距离被锁定,永远不会再被重新考虑
negative_weight_graph = [(0, 1, 1), (1, 3, 1), (0, 2, 100), (2, 1, -200)]
# 真实最短路径: 0->2->1->3 = 100+(-200)+1 = -99, 但Dijkstra会因为提前锁定节点1和3而错过这条路径

dijkstra_result = dijkstra(4, negative_weight_graph, 0)
bellman_ford_result = bellman_ford(4, negative_weight_graph, 0)

assert dijkstra_result != bellman_ford_result   # 真实复现:两者结果不一致
assert bellman_ford_result == [0, -100, 100, -99]   # Bellman-Ford算出的正确结果
assert dijkstra_result == [0, -100, 100, 2]           # Dijkstra的dist[1]因数组副作用意外"蒙对"了-100,
                                                          # 但dist[3]错误地停留在2(0->1->3的旧值),
                                                          # 没有传播到-99这个真实最短距离

# 坑2: Bellman-Ford忘记做第V轮检测,会漏掉真实存在的负权环
def bellman_ford_no_cycle_check(n, edges, start):
    """故意只做V-1轮, 不做第V轮的负权环检测"""
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist   # 没有做额外一轮检测,无法知道结果是否可信

def bellman_ford_with_cycle_check(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    has_negative_cycle = any(
        dist[u] != float('inf') and dist[u] + w < dist[v] for u, v, w in edges
    )
    return dist, has_negative_cycle

neg_cycle_graph = [(0, 1, 1), (1, 2, -3), (2, 0, 1)]   # 环总权重-1,真实存在负权环
no_check_result = bellman_ford_no_cycle_check(3, neg_cycle_graph, 0)
_, has_cycle = bellman_ford_with_cycle_check(3, neg_cycle_graph, 0)

assert has_cycle is True   # 正确检测出确实存在负权环
# 没做检测的版本,会返回一个"看起来正常"的距离数组,却完全没有意识到这个数字本身没有意义
# (负权环意味着理论上可以无限绕环让距离趋向负无穷,V-1轮算出的具体数字只是"跑了固定轮数后的中间状态")
print(f"没做负权环检测的版本, 静默返回了看似正常的结果: {no_check_result}(实际这个数字毫无意义)")

print(f"OK: Dijkstra在负权图上真实给出了错误结果(dist[3]停留在2, 而真实最短距离是"
      f"{bellman_ford_result[3]}; 错误没有出现在直接相邻负权边的节点1上, 而是出现在它的下游节点3上, "
      f"这比最初设想的失败位置更隐蔽); "
      f"Bellman-Ford不做第V轮检测, 会在真实存在负权环时静默返回一个没有意义的'结果', "
      f"而不会有任何报错提示这个结果不可信")
```
本机实测(含一次真实的反例位置修正):最初设想"负权边直接指向的节点会拿到错误距离",现场构造验证后发现并非如此——由于这个 Dijkstra 实现的松弛操作会无条件更新 `dist` 数组(即使目标节点已经 `visited`),负权边直接指向的节点1最终显示的距离(-100)反而"意外正确";真正的错误出现在**更下游**的节点3上:节点1被过早锁定、把自己的出边（到节点3）用旧的错误距离松弛过一遍之后,才收到负权边带来的修正,而这个修正没有传播到节点3(节点3早已被更早锁定,不会再被重新考虑)——Dijkstra 给出的 `dist[3]=2`,和 Bellman-Ford 给出的正确值 `-99` 相差极大。这个"错误发生在下游、不在直接相邻节点"的具体位置,比最初凭直觉设想的反例更精确,也更能体现 Dijkstra 在负权图上失效的真实机制:不是"某个数字算错了",而是"错误的锁定顺序导致后续松弛永远来不及发生"。Bellman-Ford 如果省略第 V 轮的负权环检测,会在真实存在负权环的图上,依然"正常"返回一组具体的距离数字,但这组数字实际上毫无意义(因为负权环理论上可以无限绕环让距离持续变小),不做检测就无法知道这个结果是否可信。

**面试怎么问 + 追问链:** "如果不确定一个图是否可能有负权边,应该默认用哪个最短路径算法?" → 追问"这个'保守选择'的代价是什么?"(应该默认用 Bellman-Ford(或者先花O(E)的代价扫描一遍确认是否存在负权边,再决定用 Dijkstra 还是 Bellman-Ford)——代价是 Bellman-Ford 的复杂度 O(V·E) 明显高于 Dijkstra 的 O((V+E)logV),如果能确定图不含负权边,Dijkstra 是更高效的选择;这个追问检验的是能否理解"正确性优先,性能其次"这个基本原则,以及在不确定前提条件时,如何在保守性和效率之间做出合理权衡)。

**常见坑:**
1. 不确认图是否存在负权边,就默认使用 Dijkstra——本知识点已经具体复现了这个错误的真实后果,且这个错误不会主动暴露,需要使用者自己确认算法的适用前提。
2. Bellman-Ford 省略负权环检测这最后一轮——本知识点已经说明,省略检测不会让算法报错或者明显出问题,只会让一个本身没有意义的结果被误当作正确答案使用,这种"看起来正常但实际不可信"的失败模式,比直接报错更危险,呼应[huggingface-deep-dive 13类](../huggingface-deep-dive/13-debugging-and-common-errors.md)"哪些情况看起来没问题但实际有问题"这一贯穿全系列的方法论主题。

---

*本篇 9 个知识点全部在仓库根目录 `.venv` 真实测试验证(含与经典教材案例的核对、多种算法之间的交叉验证、以及真实错误结果的现场复现)。*
