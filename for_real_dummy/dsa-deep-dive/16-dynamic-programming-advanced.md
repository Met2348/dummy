# 16 · 动态规划进阶(Dynamic Programming Advanced)

> 总览见 [00-roadmap.md](00-roadmap.md)。[10类](10-dynamic-programming-basics.md)打好了 DP 方法论的地基,本类是真正的"终面区分度"所在——区间DP、状压DP、数位DP、树形DP进阶、期望DP、博弈DP,每一种都是独立的思维模式,大多数标准新手向题单不会覆盖到这个深度。

---

## 1. 区间 DP:石子合并类问题

**签名/是什么:**
```
dp[i][j] = 合并区间[i,j]内所有石子堆的最小代价
枚举分割点k: dp[i][j] = min(dp[i][k] + dp[k+1][j]) + sum(stones[i..j])
```

**一句话:** 区间 DP 的状态是"一段连续区间",转移方程枚举"在区间内的哪个位置分割成两个子区间",石子合并问题(把一排石子堆两两合并成一堆,每次合并代价是两堆重量之和,求合并成一堆的最小总代价)是这类问题的标准范例。

**底层机制/为什么这样设计:** `dp[i][j]` 代表"把区间 `[i,j]` 内所有石子合并成一堆的最小代价"——枚举分割点 `k`,意味着"先把 `[i,k]` 合并成一堆、`[k+1,j]` 合并成另一堆,最后这两堆再合并",`dp[i][k]+dp[k+1][j]` 是前两步的代价,再加上最后一次合并的代价(整个区间 `[i,j]` 的重量总和,用前缀和 O(1) 算出)。这类区间 DP 的遍历顺序必须**按区间长度从小到大**——计算 `dp[i][j]` 依赖所有比它短的子区间已经算出,这和[15类知识点3](15-graphs-advanced.md#3-floyd-warshall-算法全源最短路径)Floyd-Warshall"k必须是最外层循环"背后的道理相通:必须保证依赖的子问题在被引用时已经计算完毕。

**AI 研究/工程场景:** [huggingface-deep-dive 04类](../huggingface-deep-dive/04-datasets-mechanics.md)讲过的批量数据拼接场景,如果需要把多个变长序列两两合并成固定长度的批次、且合并本身有成本(比如需要重新计算某些统计量),求"最优合并顺序使总成本最小",在抽象结构上和石子合并是同一类问题。

**可运行例子:**
```python
def min_cost_merge_stones(stones):
    n = len(stones)
    if n <= 1:
        return 0
    prefix = [0] * (n + 1)
    for i, s in enumerate(stones):
        prefix[i + 1] = prefix[i] + s
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):        # 区间长度从小到大
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = min(dp[i][k] + dp[k + 1][j] for k in range(i, j)) + (prefix[j + 1] - prefix[i])
    return dp[0][n - 1]

assert min_cost_merge_stones([1, 2, 3, 4]) == 19
assert min_cost_merge_stones([4]) == 0        # 单堆,不需要合并
assert min_cost_merge_stones([1, 1]) == 2       # 两堆,一次合并,代价就是1+1=2
assert min_cost_merge_stones([]) == 0            # 空输入

# 交叉验证:用暴力递归(枚举所有可能的合并分割方式)对照验证
from functools import lru_cache

def brute_merge_stones(stones):
    @lru_cache(maxsize=None)
    def solve(t):
        if len(t) <= 1:
            return 0
        best = float('inf')
        for k in range(1, len(t)):
            cost = solve(t[:k]) + solve(t[k:]) + sum(t)
            best = min(best, cost)
        return best
    return solve(tuple(stones))

import random
random.seed(90)
for _ in range(15):
    test_stones = [random.randint(1, 10) for _ in range(random.randint(1, 7))]
    assert min_cost_merge_stones(test_stones) == brute_merge_stones(test_stones)

print("OK: 区间DP(石子合并)在边界情况(空/单堆/两堆)下全部正确, "
      "15组随机测试与暴力递归枚举所有分割方式的结果完全一致")
```
本机实测:区间 DP 在边界情况(空输入、单堆、两堆)下均正确;15 组随机测试中,区间 DP 和暴力递归枚举所有可能分割方式的结果完全一致。

**面试怎么问 + 追问链:** "区间 DP 为什么必须按区间长度从小到大的顺序计算?" → 追问"如果改成按 `i` 从小到大、`j` 从小到大的顺序(而不是按长度)遍历,会出问题吗?"(可能会——比如计算 `dp[0][5]` 时可能需要用到 `dp[1][4]`(长度4,比 `[0,5]` 短),如果外层按 `i` 从小到大、内层按 `j` 从小到大遍历,处理 `i=0` 这一整行时,`dp[1][4]` 这类"i更大但区间更短"的值可能还没被计算过;这个追问检验的是能否具体验证遍历顺序是否真的保证了所有依赖都已就绪,而不是想当然地认为"顺序遍历"就足够)。

**常见坑:**
1. 遍历顺序不是按区间长度而是按端点顺序——这是区间 DP 最容易写错的地方,上面的追问链已经具体分析了这个错误的后果。
2. 忘记用前缀和 O(1) 计算区间总和,而是每次都重新遍历区间求和——这会让区间 DP 的整体复杂度从 O(n³) 退化到 O(n⁴),在 n 较大时可能造成明显的性能问题。

---

## 2. 状态压缩 DP:旅行商问题(TSP)

**签名/是什么:**
```
dp[mask][u]: 已经访问过mask这个集合(用二进制位表示)的所有城市,且当前停留在u,的最小代价
mask的第i位是1，表示第i个城市已经被访问过
```

**一句话:** 状态压缩 DP 用一个整数的二进制位表示"一个集合的访问状态"(呼应[09类知识点2](09-backtracking.md#2-子集问题位运算解法与回溯解法对比)子集的位运算编码)——旅行商问题(访问所有城市恰好一次并回到起点,求最小总路程)的状态需要同时记录"已访问哪些城市"和"当前在哪个城市",前者用位掩码压缩表示,把原本指数级的"访问顺序"问题转化成多项式级的动态规划。

**底层机制/为什么这样设计:** 朴素地枚举所有访问顺序是 O(n!),状态压缩 DP 把"访问顺序"这个信息压缩成"访问了哪些城市的集合"(不关心具体顺序,只关心集合内容),状态数是 `2^n * n`(mask的可能取值 × 当前所在城市),每个状态的转移是 O(n)(尝试访问下一个未访问的城市),总复杂度是 O(2^n * n²)——虽然依然是指数级(TSP本身是NP-hard问题,不存在多项式算法),但相比 O(n!) 已经是巨大的改进(比如 n=15 时,15! 约13亿,而 2^15*15²约7300万,快了将近20倍,n越大这个差距越悬殊)。

**AI 研究/工程场景:** [huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练任务调度,如果需要为一批计算任务规划"访问顺序"(比如数据在多个节点间流转,每次切换节点有固定开销,求总开销最小的访问顺序),且任务数量有限(通常不超过20个左右,是状压DP能处理的规模),就是TSP的直接应用场景。

**可运行例子:**
```python
def tsp(dist):
    n = len(dist)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0   # 初始状态:只访问过城市0(mask=1,即二进制的...0001),当前停留在城市0
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask][u] == float('inf'):
                continue
            for v in range(n):
                if mask & (1 << v):   # v已经访问过,跳过
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
    full_mask = (1 << n) - 1
    return min(dp[full_mask][u] + dist[u][0] for u in range(n) if dp[full_mask][u] != float('inf'))

dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
assert tsp(dist_matrix) == 80

# 交叉验证:暴力枚举所有排列(适用于小规模)对照验证
from itertools import permutations

def tsp_brute(dist):
    n = len(dist)
    best = float('inf')
    for perm in permutations(range(1, n)):
        path = [0] + list(perm) + [0]
        cost = sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))
        best = min(best, cost)
    return best

assert tsp(dist_matrix) == tsp_brute(dist_matrix)

assert tsp([[0]]) == 0   # 单城市,不需要移动

import random
random.seed(93)
for _ in range(8):
    n = random.randint(2, 6)
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = random.randint(1, 20)
            d[i][j] = d[j][i] = w
    assert tsp(d) == tsp_brute(d)

print("OK: 状压DP求解TSP在单城市边界情况下正确, 与暴力枚举全排列的结果(80)完全一致, "
      "8组随机小规模测试进一步交叉验证")
```
本机实测:状压 DP 在经典4城市案例上得到最优总路程80,和暴力枚举全部排列的结果完全一致;单城市边界情况正确;8 组随机小规模测试进一步交叉验证了状压 DP 和暴力枚举的结果始终一致。

**面试怎么问 + 追问链:** "状压 DP 能处理的城市数量上限大概是多少,为什么?" → 追问"如果城市数量达到30个,状压DP还可行吗,应该怎么办?"(状压DP的状态数是 `2^n`,n=20时约100万,n=30时超过10亿,内存和时间都难以承受——这种规模下应该转向近似算法(比如遗传算法、模拟退火这类启发式算法,不保证最优解但能在合理时间内给出足够好的解);这个追问检验的是能否理解状压DP本身依然是指数级算法,只是把原本更糟糕的O(n!)改进到O(2^n \* n²),存在一个实际可行的规模上限,不是"万能"的解决方案)。

**常见坑:**
1. 初始状态设置错误(比如没有正确设置 `dp[1][0]=0` 这个起点)——状压DP的状态转移完全依赖正确的初始条件,起点设置错误会导致所有后续状态都是错误的。
2. 最终答案忘记加上"从最后一个城市回到起点"这一段距离——TSP要求的是一个完整的环路(访问所有城市后必须回到出发点),不能只计算"访问完所有城市"的开销,忘记这一步是这道题最容易漏掉的细节。

---

## 3. 数位 DP:统计满足条件的数字个数

**签名/是什么:**
```
逐位构造数字, 用 (当前处理到第几位, 上一位数字是什么, 是否仍然贴着上界, 是否已经开始有效数字) 
这几个维度描述状态, 记忆化搜索避免重复计算
```

**一句话:** 数位 DP 解决"统计 `[0,n]` 范围内,满足某种和数字组成相关的条件的数字有多少个"这类问题——核心技巧是**逐位构造**数字,用"是否贴着上界"(tight)这个状态维度,避免真的去逐个检查 `[0,n]` 范围内的每一个数字。

**底层机制/为什么这样设计:** 直接遍历 `[0,n]` 检查每个数字是否满足条件是 O(n),当 n 很大(比如10^18)时完全不可行——数位DP把问题转化成"逐位决定每一位填什么数字",用记忆化搜索避免对相同的"状态"重复计算。`tight`(是否贴着上界)是这类DP最核心的状态维度:如果当前构造的数字前缀,已经比 `n` 对应前缀小(不贴着上界),那么后续每一位可以自由选择0~9(不再受 `n` 的限制);如果前缀恰好等于 `n` 的对应前缀(贴着上界),后续这一位最多只能选到 `n` 在这一位的数字(否则会超过n)。`started`(是否已经开始有效数字,用于处理前导零)是另一个常见的状态维度——前导零在多数场景下不应该被当作"数字的一部分"参与条件判断。

**AI 研究/工程场景:** 数位DP本身在AI/ML领域不是高频直接场景,但它代表的"通过状态设计,把一个看似必须逐一枚举的问题转化为可高效计算的动态规划"这个思路,是本系列反复出现的核心方法论——理解这个具体但技巧性很强的应用,能加深对"如何设计DP状态"这个更一般性问题的理解。

**可运行例子:**
```python
from functools import lru_cache

def count_valid_numbers(n_str, is_valid_digit_pair):
    """统计[0, n]范围内, 相邻数字都满足is_valid_digit_pair的数字个数"""
    @lru_cache(maxsize=None)
    def dp(pos, prev_digit, tight, started):
        if pos == len(n_str):
            return 1
        limit = int(n_str[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            new_started = started or d > 0
            if started and not is_valid_digit_pair(prev_digit, d):
                continue   # 只有已经开始构造有效数字时,才检查相邻位约束(跳过前导零阶段)
            total += dp(pos + 1, d if new_started else -1, new_tight, new_started)
        return total
    return dp(0, -1, True, False)

def no_consecutive_equal(prev, cur):
    return prev != cur

result = count_valid_numbers("20", no_consecutive_equal)
assert result == 20   # [0,20]范围内,没有任何一个数字(最多两位)会有相邻位相同

# 用一个真正会触发约束的场景验证(比如数字11因为相邻位相同,应该被排除)
result_50 = count_valid_numbers("50", no_consecutive_equal)

def brute_count(limit, is_valid_digit_pair):
    count = 0
    for n in range(limit + 1):
        s = str(n)
        valid = all(is_valid_digit_pair(int(s[i - 1]), int(s[i])) for i in range(1, len(s)))
        if valid:
            count += 1
    return count

assert result_50 == brute_count(50, no_consecutive_equal)   # 应该排除11,22,33,44这四个数字

assert count_valid_numbers("0", no_consecutive_equal) == 1   # 只有0本身

# 交叉验证:更大规模下,数位DP与暴力逐个枚举检查的结果完全一致
for limit_str in ["9", "20", "99", "123"]:
    assert count_valid_numbers(limit_str, no_consecutive_equal) == brute_count(int(limit_str), no_consecutive_equal)

print(f"OK: 数位DP在边界情况(n=0)下正确; [0,50]范围内结果={result_50}, "
      f"与暴力逐个检查结果一致(正确排除了11/22/33/44这几个相邻位相同的数字); "
      f"多组不同规模的上界与暴力枚举结果完全一致")
```
本机实测:数位DP在 `n=0` 这个边界情况下正确(只有0本身满足条件);在 `[0,50]` 范围内,数位DP和暴力逐个检查的结果完全一致,正确排除了11、22、33、44这几个相邻位相同的数字;多组不同规模的上界测试进一步确认了数位DP和暴力枚举方法的一致性。

**面试怎么问 + 追问链:** "数位DP的记忆化搜索里,为什么 `tight` 为 `True` 的状态通常不能被缓存复用?" → 追问"如果不区分`tight`状态,直接用`(pos, prev_digit)`做缓存键,会有什么问题?"(`tight=True` 的状态,后续每一位的可选范围依赖上界 `n` 在对应位的具体数字,这个限制是"一次性"的、和具体的 `n` 绑定的,不能像 `tight=False` 的状态那样被当作"通用"结果缓存复用;如果缓存键不包含 `tight`,会把"贴着上界、选择受限"的计算结果和"不贴上界、自由选择"的计算结果混淆,产生错误结果;这个追问检验的是能否理解为什么 `tight` 必须是缓存键的一部分,这是数位DP实现里最容易被忽视但影响正确性的细节)。

**常见坑:**
1. 缓存键遗漏 `tight` 维度(如上面追问链所述)——这会导致记忆化搜索错误地复用了不该复用的中间结果。
2. 前导零处理不当(`started` 状态维度设计或使用有误)——比如把前导零错误地当作"数字的一部分"参与相邻位约束检查,会导致类似 "007" 被误判为存在"00"这样的相邻相同数字,而实际上 "007" 代表的数字就是7,不应该有这个约束。

---

## 4. 树形 DP 进阶:树上背包问题

**签名/是什么:**
```
每个节点自身"占用"一定资源(比如1个名额), 子树内每个节点各有一个价值,
在总资源(容量)限制下, 选择树中的一个连通子集, 使总价值最大 
—— 树形结构上的0/1背包(呼应10类)
```

**一句话:** 树上背包问题结合了[08类](08-trees.md)树形DP(自底向上传递状态)和[10类知识点5](10-dynamic-programming-basics.md#5-01-背包)0/1背包(容量限制下的价值最大化)——每个节点的DP状态本身是一个"背包数组"(不同容量对应的最优价值),父节点合并多个子节点的背包数组时,需要做一次"分组背包"式的合并。

**底层机制/为什么这样设计:** `dp[node][c]` 代表"在以 `node` 为根的子树中,选择一个包含 `node` 自身、总共占用 `c` 个名额的连通子集,能获得的最大价值"——因为要求"连通",必须包含根节点本身,这是树上背包和普通背包的关键区别(普通背包的物品互相独立,树上背包的"物品"必须构成一棵连通的子树)。合并子节点的背包数组时,用类似分组背包的思路:依次把每个子节点的背包"并入"当前的背包状态,对每个可能的总容量,枚举"分配给这个新子节点多少名额、分配给之前已合并部分多少名额"的所有组合,取最优——这个"逐个子节点合并、每次合并做一次容量分配"的过程,复杂度分析比普通背包更精细(总复杂度可以证明是 O(n\*capacity²) 量级,不是简单的子节点数乘以容量,这是"树上背包复杂度证明"这个经典问题的结论,面试很少要求推导，但理解合并过程本身很重要)。

**AI 研究/工程场景:** [huggingface-deep-dive 06类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练资源分配,如果计算资源本身有层级依赖关系(比如必须先分配父级资源才能进一步细分子级资源,构成树形结构),在总资源限制下选择价值最大的资源子集,就是树上背包问题的直接应用场景。

**可运行例子:**
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []

def tree_knapsack_max_value(root, capacity, values):
    def dfs(node):
        dp = [0] * (capacity + 1)
        for c in range(1, capacity + 1):
            dp[c] = values[node.val]   # 至少要有1个名额才能选中node自己
        for child in node.children:
            child_dp = dfs(child)
            new_dp = dp[:]
            for c in range(capacity, 0, -1):
                for k in range(0, c):   # k是分配给这个子节点子树的名额数
                    if dp[c - k] + child_dp[k] > new_dp[c]:
                        new_dp[c] = dp[c - k] + child_dp[k]
            dp = new_dp
        return dp

    return max(dfs(root))

root = TreeNode(0)
child1 = TreeNode(1)
child2 = TreeNode(2)
root.children = [child1, child2]
values = {0: 5, 1: 3, 2: 4}

# 容量2: 选root(5) + 价值更高的child2(4) = 9,好于选root+child1(5+3=8)
assert tree_knapsack_max_value(root, 2, values) == 9
assert tree_knapsack_max_value(root, 1, values) == 5    # 容量1,只能选root自己
assert tree_knapsack_max_value(root, 3, values) == 12    # 容量足够,全部选上 5+3+4=12

single_node = TreeNode(0)
assert tree_knapsack_max_value(single_node, 5, {0: 7}) == 7   # 单节点,不管容量多大,最多选这一个

# 交叉验证:暴力枚举所有"连通且包含根节点"的子集,找最大价值,对照验证
def brute_tree_knapsack(root, capacity, values):
    all_nodes = []
    def collect(node):
        all_nodes.append(node)
        for c in node.children:
            collect(c)
    collect(root)
    n = len(all_nodes)
    best = 0
    for mask in range(1 << n):
        selected = [all_nodes[i] for i in range(n) if mask & (1 << i)]
        if not selected or root not in selected or len(selected) > capacity:
            continue
        selected_vals = set(id(x) for x in selected)
        # 检查连通性:每个非根节点的父节点也必须被选中(简化实现,遍历检查)
        def is_connected(node, selected_ids):
            if id(node) not in selected_ids:
                return True
            for c in node.children:
                if id(c) in selected_ids and not is_connected(c, selected_ids):
                    return False
                if id(c) in selected_ids:
                    continue
            return True
        # 简化:只验证选中的节点集合,每个选中的非根节点,其父节点也在集合内
        def build_parent_map(node, parent, pmap):
            pmap[id(node)] = parent
            for c in node.children:
                build_parent_map(c, node, pmap)
        pmap = {}
        build_parent_map(root, None, pmap)
        valid = all(pmap[id(n)] is None or id(pmap[id(n)]) in selected_vals for n in selected)
        if valid:
            total = sum(values[n.val] for n in selected)
            best = max(best, total)
    return best

assert tree_knapsack_max_value(root, 2, values) == brute_tree_knapsack(root, 2, values)
assert tree_knapsack_max_value(root, 3, values) == brute_tree_knapsack(root, 3, values)

print("OK: 树上背包在边界情况(容量1/容量足够/单节点)下全部正确, "
      "与暴力枚举所有连通子集的结果完全一致")
```
本机实测:树上背包在容量恰好只够选根节点、容量足够选择全部节点、单节点这几类边界情况下均正确;和暴力枚举所有"连通且包含根节点"的子集、逐一计算价值取最大值的方法,结果完全一致。

**面试怎么问 + 追问链:** "树上背包为什么要求选出的子集必须包含根节点、且连通?" → 追问"如果不要求连通(可以选子树中任意的节点子集),这个问题会退化成什么?"(会退化成普通的0/1背包问题——每个节点独立,选或不选互不影响,不再需要考虑树形结构和父子依赖关系;这个追问检验的是能否理解"连通性约束"是树上背包区别于普通背包的核心特征,这个约束正是"树形"这个前缀真正体现价值的地方)。

**常见坑:**
1. 合并子节点背包数组时,内层循环容量的遍历方向写错(应该从大到小,呼应[10类知识点5](10-dynamic-programming-basics.md#5-01-背包)0/1背包的类似坑)——如果方向不对,可能导致同一个子节点的贡献被重复计算。
2. 忘记"选中的子集必须包含根节点"这个约束——如果初始化 `dp[node][0]=0`(允许不选根节点自己)而不是要求至少占用1个名额选中根节点,会让子树内的其他节点在"根节点未被选中"的情况下也能贡献价值,违反了树上背包的连通性要求。

---

## 5. 概率 DP / 期望 DP

**签名/是什么:**
```
E[state]: 从某个状态出发, 到达目标状态的期望步数/期望代价
状态转移基于概率的加权平均: E[s] = 1 + sum(P(s->s') * E[s'] for s' in 后继状态)
```

**一句话:** 期望DP求解"从某个起始状态出发,经过一个带随机性的过程,到达目标状态的期望步数/期望代价"——转移方程不再是简单的取最值,而是按每种后继状态发生的概率加权求和,这类问题的正确性**可以用真实随机模拟(蒙特卡洛方法)交叉验证**,这是期望DP区别于其他DP变体的一个独特验证手段。

**底层机制/为什么这样设计:** 以"掷骰子直到累计点数达到目标值,求期望投掷次数"为例:`E[s]` 代表"当前累计点数是 `s` 时,还需要投掷的期望次数"——从状态 `s` 投一次骰子,会等概率地转移到 `s+1, s+2, ..., s+6` 这6种状态之一,所以 `E[s] = 1 + (E[s+1]+E[s+2]+...+E[s+6])/6`("1"是这一次投掷本身消耗的步数,后面是转移到各个后继状态后,各自还需要的期望步数按概率加权平均)。这类问题通常需要**从目标状态往回递推**(`E[已达标] = 0`),因为期望值的计算依赖"后续还需要多少步",这个信息只有从终点往回看才能确定。

**AI 研究/工程场景:** [huggingface-deep-dive 09类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过的训练过程涉及大量随机性(数据打乱顺序、dropout等),如果要理论上估计"某个随机训练过程平均需要多少步才能达到某个收敛标准",期望DP提供的分析框架(尽管实际训练场景通常复杂到难以精确建模)在概念上是相通的;更直接的应用是強化学习理论中的价值函数计算,本质上就是期望DP在马尔可夫决策过程上的应用。

**可运行例子:**
```python
import random

def expected_steps_dp(target, sides=6):
    """期望DP: 掷一个sides面骰子, 求累计点数达到target的期望步数"""
    E = [0.0] * (target + sides + 1)   # 多开一些空间避免越界(达到或超过target时期望为0)
    for s in range(target - 1, -1, -1):
        E[s] = 1 + sum(E[s + i] for i in range(1, sides + 1)) / sides
    return E[0]

dp_result = expected_steps_dp(10)

def monte_carlo_verify(target, sides=6, trials=200000, seed=91):
    """真实随机模拟大量试验, 取平均步数, 作为DP结果的独立交叉验证"""
    random.seed(seed)
    total_steps = 0
    for _ in range(trials):
        s, steps = 0, 0
        while s < target:
            s += random.randint(1, sides)
            steps += 1
        total_steps += steps
    return total_steps / trials

mc_result = monte_carlo_verify(10)

# 期望DP的结果和真实随机模拟的平均值应该非常接近(允许蒙特卡洛方法本身的统计误差)
assert abs(dp_result - mc_result) < 0.05

assert expected_steps_dp(1) == 1.0   # 目标为1,期望恰好1步(骰子任何结果都>=1)
assert abs(expected_steps_dp(0) - 0.0) < 1e-9   # 目标为0,已经达标,期望0步

print(f"OK: 期望DP计算目标为10的期望步数={dp_result:.4f}, "
      f"20万次真实蒙特卡洛模拟得到的平均步数={mc_result:.4f}, "
      f"两者相差仅{abs(dp_result-mc_result):.4f}(在统计误差范围内, "
      f"验证了期望DP公式的正确性); 边界情况(target=0/1)全部正确")
```
本机实测:期望DP计算"累计点数达到10"所需的期望步数为3.3237;用20万次真实蒙特卡洛模拟(不是理论推导,是真的写代码掷骰子20万次统计平均值)得到的结果是3.3263,两者相差仅0.0026,在统计误差的合理范围内——这个交叉验证方式(真实随机模拟 vs 理论DP公式)是期望类DP问题特有的、其他DP变体通常做不到的独立验证手段。

**面试怎么问 + 追问链:** "期望DP和普通DP最大的实现差异是什么?" → 追问"除了理论推导,你怎么确信一个期望DP的转移方程写对了?"(蒙特卡洛模拟是一个实用的独立验证手段——写一个直接模拟随机过程的暴力程序,跑足够多次取平均,如果和DP理论值接近,能提供很强的正确性信心;这个追问检验的是能否在纯数学推导之外,提供一种更"经验主义"的验证思路,这也是本系列贯穿全程"真实验证优于理论推导"这个纪律,在期望类问题上一个特别自然、特别有说服力的应用场景)。

**常见坑:**
1. 状态转移方向搞反(应该从目标状态往回递推,却尝试从起点正向递推)——期望DP的"未来期望"依赖信息通常需要从终点反向传播,正向递推在很多期望DP问题里根本无法列出正确的转移方程(不知道"后续还需要多少步"这个关键信息)。
2. 数组越界或者边界状态(比如骰子点数可能超过target的情况)处理不当——本知识点的实现特意多开了 `sides` 长度的数组空间,避免 `s接近target时`,`s+i` 可能超出数组范围导致索引错误。

---

## 6. 单调队列优化 DP

**签名/是什么:**
```
dp[i] = nums[i] + max(dp[i-k], dp[i-k+1], ..., dp[i-1])
朴素做法: 内层再嵌套一次循环找最大值, O(n*k)
优化: 用单调队列(呼应05类)维护滑动窗口内的最大值, O(n)
```

**一句话:** 当DP转移方程里出现"从前面固定窗口大小的一段状态里取最值"这种模式,朴素实现会引入一层额外的循环(让复杂度多乘一个 k),用[05类知识点3](05-stacks-and-queues.md#3-单调队列滑动窗口最大值)的单调队列技巧维护这个滑动窗口最大值,能把这层额外的复杂度优化掉。

**底层机制/为什么这样设计:** 转移方程 `dp[i] = nums[i] + max(dp[i-k..i-1])` 本质上是"每计算一个新的 `dp[i]`,都要在一个滑动窗口(大小为k)里找最大值"——这正是[05类知识点3](05-stacks-and-queues.md#3-单调队列滑动窗口最大值)已经解决过的问题,只是这里窗口里存的是"已经算出的DP值"而不是原始数组的值。把单调队列直接套进DP的递推过程:每算出一个新的 `dp[i]`,按照单调队列的规则把它加入队列(维护单调性、移除过期的队首),下一次计算 `dp[i+1]` 时,队首就是当前窗口内的最大值,不需要重新扫描整个窗口。

**AI 研究/工程场景:** [huggingface-deep-dive 12类](../huggingface-deep-dive/12-inference-optimization.md)讲过的滑动窗口注意力机制(限制每个 token 只关注最近k个位置),如果需要在生成过程中动态维护"最近k步的某个最优指标"(比如某种早停判断依据),单调队列优化DP的思路能让这类维护操作保持在均摊O(1),而不是每步都重新扫描窗口。

**可运行例子:**
```python
from collections import deque

def max_sum_with_jump_limit(nums, k):
    """dp[i] = nums[i] + max(dp[i-k..i-1]), 用单调队列优化"""
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])   # 存下标, dp[dq]从队首到队尾单调递减
    for i in range(1, n):
        while dq and dq[0] < i - k:   # 队首超出窗口范围
            dq.popleft()
        dp[i] = nums[i] + dp[dq[0]]
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
    return dp[-1]

assert max_sum_with_jump_limit([1, -1, -2, 4, -7, 3], 2) == 7

def brute_jump(nums, k):
    n = len(nums)
    dp = [float('-inf')] * n
    dp[0] = nums[0]
    for i in range(1, n):
        for j in range(max(0, i - k), i):
            dp[i] = max(dp[i], nums[i] + dp[j])
    return dp[-1]

assert max_sum_with_jump_limit([1, -1, -2, 4, -7, 3], 2) == brute_jump([1, -1, -2, 4, -7, 3], 2)

assert max_sum_with_jump_limit([5], 3) == 5   # 单元素

# 交叉验证:多组随机测试,单调队列优化版本与朴素O(n*k)版本结果必须一致
import random
random.seed(94)
for _ in range(20):
    nums = [random.randint(-20, 20) for _ in range(random.randint(1, 15))]
    k = random.randint(1, len(nums))
    assert max_sum_with_jump_limit(nums, k) == brute_jump(nums, k)

# 真实计时对比:验证单调队列优化在较大规模下确实比朴素O(n*k)更快
import time
big_nums = [random.randint(-100, 100) for _ in range(50000)]
big_k = 1000

t0 = time.perf_counter(); max_sum_with_jump_limit(big_nums, big_k); opt_time = time.perf_counter() - t0
t0 = time.perf_counter(); brute_jump(big_nums, big_k); brute_time = time.perf_counter() - t0

assert opt_time < brute_time / 5   # 优化版本应该明显更快

print(f"OK: 单调队列优化DP在单元素边界情况下正确, 20组随机测试与朴素O(n*k)解法结果一致; "
      f"n=50000,k=1000规模下, 优化版本={opt_time:.4f}s, 朴素版本={brute_time:.4f}s"
      f"(优化版本快{brute_time/opt_time:.0f}倍)")
```
本机实测:单调队列优化DP在单元素边界情况下正确;20 组随机测试中和朴素 O(n·k) 解法结果完全一致;在 n=50000、k=1000 的规模下,单调队列优化版本明显快于朴素版本,真实验证了这个优化在大规模输入下的效率价值。

**面试怎么问 + 追问链:** "什么样的DP转移方程,能用单调队列优化?" → 追问"如果转移方程是 `dp[i] = nums[i] + min(dp[i-k..i-1])`(求最小值而不是最大值),单调队列的维护方向需要怎么改?"(需要维护单调**递增**的队列(而不是递减)——求最大值时保留"更大的候选、淘汰更小的",求最小值时方向刚好相反,这个"维护方向要不要反过来"的判断,呼应[05类知识点8](05-stacks-and-queues.md#8-单调栈--单调队列常见坑)已经强调过的"维护方向搞反"这类常见坑,在DP的场景下再次需要留意)。

**常见坑:**
1. 单调队列维护的时机和DP递推的顺序搞混(应该先用队首更新 `dp[i]`,再把 `dp[i]` 加入队列,顺序不能颠倒)——如果先把 `dp[i]` 加入队列再用队首更新自己,可能会用到 `dp[i]` 自己刚被加入的值,这不符合转移方程"只依赖之前状态"的要求。
2. 忘记先做"移除过期队首"这一步就直接使用队首值——如果队首下标已经超出了当前窗口 `[i-k, i-1]` 的范围,不移除就直接使用会引用一个不该被考虑的、过时的状态。

---

## 7. 编辑距离进阶变体:通配符匹配

**签名/是什么:**
```
dp[i][j]: s的前i个字符, 能否被p的前j个字符(含?和*通配符)匹配
'?' 匹配任意单个字符; '*' 匹配任意长度(含0)的任意子串
```

**一句话:** 通配符匹配是[10类知识点4](10-dynamic-programming-basics.md#4-二维-dp路径类问题与编辑距离)编辑距离系列问题的一个变体——不再是"计算变换代价",而是"判断能否匹配",转移方程的核心变化在于 `*` 这个特殊字符对应的转移逻辑:它既可以匹配空(不消耗s的字符),也可以匹配任意长度的子串(消耗s的一个字符,但*本身继续保留在后续匹配中可用)。

**底层机制/为什么这样设计:** `dp[i][j]` 表示 `s` 的前 `i` 个字符能否被 `p` 的前 `j` 个字符匹配——如果 `p[j-1]` 是普通字符或 `?`,转移和标准编辑距离的"字符匹配"分支类似(`dp[i][j] = dp[i-1][j-1]`,前提是字符相同或者是`?`)。如果 `p[j-1]` 是 `*`,它有两种"用法":①让 `*` 匹配空字符串(不消耗s,`dp[i][j] = dp[i][j-1]`,看s的前i个字符能否被p去掉这个`*`后的前j-1个字符匹配);②让 `*` 多匹配一个s的字符(`dp[i][j] = dp[i-1][j]`,看s去掉一个字符后的前i-1个,能否被同样这个`*`结尾的p的前j个字符匹配,`*`本身可以继续吃掉更多字符)——这两种情况取"或"(只要有一种能成立,当前状态就是可匹配的)。

**AI 研究/工程场景:** [huggingface-deep-dive 11类](../huggingface-deep-dive/11-hub-and-sharing.md)讲过 Hub 上模型文件的路径匹配场景,类似 `*.safetensors` 这种通配符模式匹配文件名,底层判断逻辑和这里的DP思路是一致的(虽然实际工程中会直接用 `fnmatch` 这类标准库,不需要手写,但理解其内部匹配逻辑有助于排查一些边界情况下的匹配行为差异)。

**可运行例子:**
```python
def is_match_wildcard(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True   # 空字符串匹配空模式
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]   # 空字符串只能被"全是*"的模式匹配
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]   # *匹配一个字符 或 *匹配空
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]

assert is_match_wildcard('aa', 'a') is False
assert is_match_wildcard('aa', '*') is True
assert is_match_wildcard('cb', '?a') is False
assert is_match_wildcard('adceb', '*a*b') is True
assert is_match_wildcard('acdcb', 'a*c?b') is False

assert is_match_wildcard('', '') is True             # 空字符串匹配空模式
assert is_match_wildcard('', '*') is True             # 空字符串匹配纯*模式
assert is_match_wildcard('a', '') is False             # 非空字符串不能匹配空模式

# 交叉验证:与Python标准库fnmatch(功能上等价的通配符匹配)对照
import fnmatch
import random
random.seed(92)
mismatches = 0
for _ in range(30):
    test_s = ''.join(random.choice('ab') for _ in range(random.randint(0, 6)))
    test_p = ''.join(random.choice(['a', 'b', '?', '*']) for _ in range(random.randint(0, 5)))
    mine = is_match_wildcard(test_s, test_p)
    reference = fnmatch.fnmatchcase(test_s, test_p)
    if mine != reference:
        mismatches += 1

assert mismatches == 0

print("OK: 通配符匹配在标准案例与边界情况(空字符串/空模式)下全部正确, "
      "30组随机测试与Python标准库fnmatch.fnmatchcase结果完全一致(0处不一致)")
```
本机实测:通配符匹配在标准测试案例和边界情况(空字符串、空模式)下均正确;30 组随机测试中,自己实现的DP解法和 Python 标准库 `fnmatch.fnmatchcase` 结果完全一致,没有任何一处不一致——这是一个有独立、权威参照标准(标准库)可以对照验证的DP问题,验证的说服力比只有暴力递归对照更强。

**面试怎么问 + 追问链:** "通配符匹配里 `*` 对应两种转移(`dp[i-1][j]` 和 `dp[i][j-1]`),这两种分别对应什么直觉?" → 追问"如果 `p` 里连续出现多个 `*`(比如 `'**'`),会不会影响算法的正确性?"(不会——连续的`*`在效果上等价于一个`*`(因为`*`本身已经能匹配任意长度包括0),算法不需要特殊处理连续`*`的情况,状态转移方程天然能正确处理(第一个`*`可以匹配0个字符,把匹配任务完全交给第二个`*`,效果和只有一个`*`完全一样);这个追问检验的是能否验证算法在"看似边界但其实转移方程本身已经隐式覆盖"的情况下依然正确,而不是遇到边界场景就默认需要额外特判)。

**常见坑:**
1. 初始化 `dp[0][j]`(空字符串s,匹配模式p的前j个字符)时,忘记处理"p的前j个字符全部是`*`"这种情况——如果初始化不对,即使s是空字符串,只要p以若干个`*`开头也应该能匹配,遗漏这个初始化会导致误判。
2. 混淆通配符匹配(`*`匹配任意长度子串)和正则表达式匹配(`*`表示前一个字符出现0次或多次,语义完全不同)——这是两个表面相似但语义不同的问题,不能把其中一个的转移方程直接套用到另一个上。

---

## 8. 博弈类 DP:判断先手必胜

**签名/是什么:**
```
dp[i][j]: 面对区间[i,j]的局面, 当前轮到的玩家, 相对于对手最多能多拿多少分
石子游戏: 每次只能从区间两端取一堆, 求先手是否必胜(能拿到更多分数)
```

**一句话:** 博弈类DP求解"两个都采取最优策略的玩家轮流决策,先手是否能赢"这类问题——状态定义的关键技巧是:不直接记录"当前玩家能拿多少分",而是记录"当前玩家相对于对手,净赢多少分"(可能是负数,代表当前玩家净输多少),这个"相对分数"的设计让状态转移能自然地统一处理双方轮流决策的对称性。

**底层机制/为什么这样设计:** 石子游戏(排成一排的石子堆,每次只能从两端取一堆,双方都想让自己拿到的石子总数最多)是这类问题的经典代表。`dp[i][j]` 定义为"面对区间 `[i,j]`,当前决策者能比对手多拿多少分(可能是负数)"——如果当前玩家选择拿走最左边的堆(`piles[i]`),那么对手接下来面对的是区间 `[i+1,j]`,对手在这个子区间里"相对当前玩家的净赢分"是 `dp[i+1][j]`,所以当前玩家选这一步之后的相对净赢分是 `piles[i] - dp[i+1][j]`(拿到的分数,减去对手在剩余区间能净赢的分数,因为对手的"净赢"对我方而言就是净输)。取"选左边"和"选右边"两种决策中相对净赢分更大的那个,就是 `dp[i][j]` 的值——这个"用减法表达对手视角"的技巧,是博弈类DP最核心也最不直观的设计。

**AI 研究/工程场景:** 博弈论DP在强化学习的对抗性训练场景(比如自博弈训练,两个策略互相对抗)中有理论上的联系——虽然实际的强化学习通常不会用穷举式的DP求解(状态空间太大),但"评估一个局面对当前决策者有多大优势"这个核心概念(对应博弈DP里的"相对净赢分"),和强化学习里的价值函数(value function)在概念上是相通的。

**可运行例子:**
```python
def can_win_stone_game(piles):
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]   # 只剩一堆,直接拿走,净赢分就是这一堆的数量
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
    return dp[0][n - 1] > 0   # 先手净赢分大于0,说明先手能赢(或至少不输,取决于题目对"胜利"的具体定义)

assert can_win_stone_game([5, 3, 4, 5]) is True
assert can_win_stone_game([3, 7, 2, 3]) is True
assert can_win_stone_game([1]) is True     # 单堆,先手直接拿走,必赢

# 现场发现一处需要修正的过度断言:最初以为"两堆相同"时先手依然算"赢"(净赢分0,松散地当"赢"处理),
# 但can_win_stone_game用的是严格>0判断"净赢",实测发现[5,5]的净赢分确实是0,不满足严格>0,
# 函数如实返回False——这不是bug,是"净赢分"这个指标在两堆相同这种对称局面下的真实、正确表现
assert can_win_stone_game([5, 5]) is False   # 净赢分恰好是0(5-5),不满足严格'净赢'的条件

# 交叉验证:用极小极大搜索(minimax,不依赖DP的具体状态设计,独立验证结果)对照
from functools import lru_cache

def minimax_margin(piles):
    """返回净赢分本身(不只是True/False), 方便观察具体数值"""
    n = len(piles)
    @lru_cache(maxsize=None)
    def solve(i, j):
        if i > j:
            return 0
        return max(piles[i] - solve(i + 1, j), piles[j] - solve(i, j - 1))
    return solve(0, n - 1)

import random
random.seed(95)
for _ in range(20):
    test_piles = [random.randint(1, 20) for _ in range(random.randint(1, 10) * 2)]  # 偶数堆
    assert can_win_stone_game(test_piles) == (minimax_margin(test_piles) > 0)

# 重新验证一个更精确的数学性质:堆数为偶数时,先手的净赢分恒 >= 0("不会输",但不保证严格必胜,
# 严格必胜额外要求"总石子数为奇数"这类排除平局的条件——[5,5]就是一个总数为偶数、真实打平的反例)
min_margin_seen = float('inf')
tie_examples = []
for _ in range(500):
    n = random.choice([2, 4, 6, 8])
    even_piles = [random.randint(1, 30) for _ in range(n)]
    margin = minimax_margin(even_piles)
    min_margin_seen = min(min_margin_seen, margin)
    if margin == 0:
        tie_examples.append(even_piles)
assert min_margin_seen >= 0   # 真正能严格证明的性质:偶数堆先手净赢分恒>=0,即"不会输"
assert len(tie_examples) > 0   # 真实存在平局案例(不是理论上的空集合),[5,5]不是孤例

print(f"OK: 博弈DP与独立的极小极大搜索结果完全一致(20组随机测试); "
      f"500组随机偶数堆测试中最小净赢分={min_margin_seen}(验证'先手不会输'这个精确表述), "
      f"其中真实观察到{len(tie_examples)}组平局案例(如{tie_examples[0]}), "
      f"证明'偶数堆先手必胜'这个更强的说法并不总是严格成立,'不会输'才是仅凭堆数奇偶性能保证的结论")
```
本机实测(含一次对定理表述过度概括的真实修正):最初想当然地把"偶数堆先手必胜"当作可以直接使用的结论,在验证 `[5,5]` 这个具体案例时才发现——两堆相同时,先手净赢分恰好是0,`can_win_stone_game` 用严格 `>0` 判断"净赢",如实返回 `False`,不是bug。进一步用500组随机偶数堆测试排查后确认:仅凭"堆数为偶数"这个条件,能严格证明的结论是"先手净赢分恒 `>=0`"(即先手不会输,但可能打平),**不是**"先手总能净赢"——真实统计中确实存在平局案例(不止 `[5,5]` 这一个,是一类真实存在的情况)。"先手总能严格获胜"这个更强的版本,需要额外的前提(比如原始 LeetCode 877 题目给定的"总石子数一定是奇数"这个约束,天然排除了平局的可能性),脱离这个前提,"必胜"就应该改口成更精确的"不会输"。这是一次典型的"先给出以为正确的结论,现场验证后发现表述过头,及时收窄到真正站得住脚的版本"的真实过程。

**面试怎么问 + 追问链:** "博弈DP里 `dp[i][j] = max(piles[i]-dp[i+1][j], piles[j]-dp[i][j-1])` 这个减法是什么意思?" → 追问"如果博弈规则更复杂(比如每次可以从两端各取任意数量,不限于1堆),这个'相对净赢分'的建模思路还适用吗?"(思路依然适用,但状态定义需要扩展(比如需要额外维护"这次最多能取几堆"这个约束),核心的"用减法表达对手视角"这个建模技巧是可以推广的通用方法,不是只对这一道具体题目有效;这个追问检验的是能否把这个具体技巧提炼成一个可迁移的通用建模思路,而不是记住一道题的具体解法)。

**常见坑:**
1. 状态定义直接记录"当前玩家能拿多少分"(而不是"相对对手净赢多少分")——这种更直觉的状态定义,会让转移方程难以正确表达"对手也在做最优决策、且对手的收益是我方的损失"这个博弈的核心特征,通常会导致转移方程写不对或者写得远比"净赢分"的定义法复杂。
2. 忽略博弈问题的"零和"前提(呼应上面的"减法"设计)——如果问题本身不是零和博弈(比如双方的目标不是完全对立的),这套"相对净赢分"的建模方式就不再适用,需要重新分析具体的博弈规则。
3. 凭记忆里"背过的结论"(比如"偶数堆先手必胜")直接下断言,不针对当前题目的具体约束重新核实——本知识点的可运行例子已经现场演示了这类结论有多容易被不加区分地过度概括:"堆数为偶数"单独成立时只能推出"先手不会输",要得到更强的"严格必胜",还需要额外确认题目是否排除了平局的可能(比如总石子数为奇数这类约束),两个不同强度的结论不能混着用。

---

## 9. DP 进阶常见坑

**签名/是什么:**
```
状压DP位运算写错 -> 状态转移逻辑错误(比如mask&(1<<i)和mask|(1<<i)用混)
数位DP前导零处理不当 -> 相邻位约束在前导零阶段被错误应用
```

**一句话:** DP进阶的坑,集中体现在"状态设计本身的复杂性带来更多犯错的机会"——状压DP的位运算细节、数位DP的多维状态(尤其是前导零处理)、树上背包的合并顺序,这些进阶技巧的正确性门槛比基础DP更高,任何一个状态维度的细节疏漏都可能导致结果错误。

**底层机制/为什么这样设计:** 这类错误的共同特点是:进阶DP的状态设计通常包含多个维度(比如数位DP的 `pos, prev, tight, started` 四个维度),每个维度都有自己精确的语义和边界处理规则——基础DP([10类](10-dynamic-programming-basics.md))通常只有一两个维度,犯错空间相对有限;进阶DP的维度越多,需要同时保持正确的细节就越多,任何一个维度的语义理解偏差,都可能在特定的、不那么显然的输入上暴露问题(呼应本系列反复强调的"随手测试几个例子未必能发现所有问题",在进阶DP场景下尤其如此,因为暴露问题所需的输入往往比基础DP更"刁钻"、更需要专门构造)。

**AI 研究/工程场景:** 这类"状态维度越多、犯错空间越大"的现象,和[huggingface-deep-dive 09类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过的多组超参数联合调优场景有相通之处——参数维度越多,组合空间越大,任何一个维度设置不当都可能让整体结果偏离预期,系统性的交叉验证(而不是零散抽查)在这类高维问题上格外重要。

**可运行例子:**
```python
# 坑1: 状压DP位运算写错,现场复现真实后果
def tsp_buggy_bitwise(dist):
    """故意把 mask & (1<<v) 误写成 mask | (1<<v) 做'是否已访问'的判断"""
    n = len(dist)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask][u] == float('inf'):
                continue
            for v in range(n):
                if mask | (1 << v):   # 错误:这个条件永远为真(mask|(1<<v)必然非零,除非mask和1<<v都是0)
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
    full_mask = (1 << n) - 1
    valid_finals = [dp[full_mask][u] + dist[u][0] for u in range(n) if dp[full_mask][u] != float('inf')]
    return min(valid_finals) if valid_finals else "NO_VALID_TOUR_FOUND"

dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
buggy_result = tsp_buggy_bitwise(dist_matrix)
# 真实复现:错误的条件判断(mask|(1<<v)恒真,除非两者都是0)导致内层循环永远continue,
# 状态转移完全无法进行,最终找不到任何完整路径
assert buggy_result == "NO_VALID_TOUR_FOUND"

def tsp_correct(dist):
    n = len(dist)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask][u] == float('inf'):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
    full_mask = (1 << n) - 1
    return min(dp[full_mask][u] + dist[u][0] for u in range(n) if dp[full_mask][u] != float('inf'))

assert tsp_correct(dist_matrix) == 80   # 正确版本能找到真实最优路径

# 坑2: 数位DP前导零处理不当,现场复现真实后果
from functools import lru_cache

def count_buggy_leading_zero(n_str):
    """故意把前导零也当作'数字的一部分'参与相邻位相同的判断"""
    @lru_cache(maxsize=None)
    def dp(pos, prev_digit, tight):
        if pos == len(n_str):
            return 1
        limit = int(n_str[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            if prev_digit != -1 and d == prev_digit:   # 错误:没有区分是否是前导零阶段
                continue
            total += dp(pos + 1, d, new_tight)
        return total
    return dp(0, -1, True)

def count_correct_with_leading_zero_handling(n_str):
    @lru_cache(maxsize=None)
    def dp(pos, prev_digit, tight, started):
        if pos == len(n_str):
            return 1
        limit = int(n_str[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            new_started = started or d > 0
            if started and d == prev_digit:
                continue
            total += dp(pos + 1, d if new_started else -1, new_tight, new_started)
        return total
    return dp(0, -1, True, False)

buggy_count = count_buggy_leading_zero("20")
correct_count = count_correct_with_leading_zero_handling("20")
# 两个版本在这个具体案例上是否一致,取决于n_str的具体取值——现场对比记录真实结果
# (对于"20"这种较小的数字,前导零的实际影响范围有限,用更能暴露问题的例子进一步验证)
buggy_count_100 = count_buggy_leading_zero("100")
correct_count_100 = count_correct_with_leading_zero_handling("100")
assert buggy_count_100 != correct_count_100   # 在更大范围内,前导零处理不当确实会导致真实的计数偏差

print(f"OK: 现场复现状压DP位运算写错导致的真实后果(完全找不到任何有效路径, "
      f"而不是找到一个'看似合理但错误'的次优解——这是一个比预想更彻底的失败模式); "
      f"现场复现数位DP前导零处理不当导致的计数偏差(n=100时, 错误版本={buggy_count_100}, "
      f"正确版本={correct_count_100})")
```
本机实测:状压DP位运算写错(`mask|(1<<v)` 误代替 `mask&(1<<v)`)的真实后果,是这个错误条件恒为真,导致内层循环完全无法进行任何状态转移,最终连一条完整路径都找不到——这比"找到一个错误的次优解"更彻底,是一种更容易在测试阶段就被发现的失败模式(完全没有输出,而不是有输出但数值不对);数位DP如果不正确处理前导零(没有区分"是否已经开始构造有效数字"),在 `n=100` 这样的规模下,真实产生了和正确处理版本不同的计数结果。

**面试怎么问 + 追问链:** "写状压DP代码时,你会怎么系统性地检查位运算逻辑有没有写错?" → 追问"除了对照小规模的暴力解法,有没有更针对性的检查方法?"(可以针对位运算本身写独立的小测试——比如单独验证"判断某一位是否被设置"、"设置某一位"、"清除某一位"这几个基础位运算操作在几个具体数值上的行为是否符合预期,把"位运算基础操作是否正确"和"DP状态转移逻辑是否正确"这两层关注点分开验证,而不是只在整个算法跑完之后才对照最终结果,这样出问题时更容易定位到底是哪一层出的错;这个追问检验的是能否把复杂问题拆解成独立可验证的小单元,这是调试进阶DP这类多状态维度问题时特别重要的能力)。

**常见坑:**
1. 状压DP的位运算符号用混(`&` 判断、`|` 设置、`^` 翻转,这几个运算符号如果搞混,会产生完全不同的语义)——本知识点已经现场复现了一次这类错误导致的彻底失败(而不是次优解)。
2. 数位DP的前导零处理遗漏——本知识点已经具体验证了这个疏漏在足够大的规模下会产生真实的、可测量的计数偏差,不是理论上的边缘情况。

---

*本篇 9 个知识点全部在仓库根目录 `.venv` 真实测试验证(含与暴力解法/标准库/蒙特卡洛模拟的交叉验证、真实计时的优化效果、以及真实bug的现场复现)。*
