"""DP 基础/序列 · 竞赛级补充（Part III）：Frontier Lab SWE 面试难度的 11 道竞赛级/最难层
DP 题（网格决策优化 / 区间DP求最少分割 / 记忆化+数学剪枝 / 三维计数DP+前缀和优化 /
去重计数DP / 二分优化经典Hard DP / 区间DP+二分 / 二维DP+前后缀最值优化 / 字符串状态压缩DP /
DP+二分查找 / 博弈DP）。
"""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache


# ── LC1289 下降路径最小和 II(Hard) ───────────────────────────────────────
def min_falling_path_sum_ii(grid: list[list[int]]) -> int:
    """
    【题意】给定 n×n 矩阵 grid，从每一行恰好选一个元素组成一条"下降路径"，要求相邻两行选中的
    列**不能相同**（这是和普通"下降路径最小和"唯一的区别：普通版只禁止同列，这版额外禁止
    "同列"，即列必须严格改变），求这条路径的最小元素和。
    【思路】dp[j] 定义为"走到当前行、选中第 j 列时的最小路径和"。朴素做法是对每一行的每个
    j，枚举上一行所有列 j'≠j 取 min(dp_prev[j'])，这是 O(n^3)；关键优化：**只需要记录上一行
    dp 数组里最小值 min1 和它的列下标 min1_idx，以及严格意义上的次小值 min2**（注意 min2 是
    "全局次小"，不是"排除 min1_idx 之外的最小值"里可能出现的重复列同值歧义——只要 min1 只在
    一个位置取得，次小值就一定来自不同列）。这样对当前行的列 j：如果 j == min1_idx（继承上一
    行最小值会撞到同一列，不允许），必须转而使用 min2；否则可以放心使用 min1（因为 min1 所在
    列必然不是 j）。把 O(n) 的"排除当前列找最小值"降到 O(1)，整体从 O(n^3) 降到 O(n^2)。
    【复杂度】时间 O(n^2)（每行 O(n) 求 min1/min2，再 O(n) 更新新一行），空间 O(n)（只保留
    滚动的一行，不需要完整二维数组）。
    【易错点】1) 求 min1/min2 时如果矩阵中存在多个相等的最小值（比如两列都是全局最小），
    "次小值"仍然应该按数值比较得到（可能和 min1 数值相等），而不是想当然地认为 min2 一定
    严格大于 min1——这是这道题最隐蔽的边界，很多人会把"次小"误理解为"第二种不同的值"；
    2) 只维护"最小值+其列下标"是不够的，必须同时维护次小值，否则当当前列恰好是 min1_idx
    时无法降级取值，会退化回 O(n) 扫描；3) 初始化第一行的 dp 就是 grid[0] 本身（没有"上一行"
    的限制），如果误加了不存在的约束会让第一行结果出错。
    """
    n = len(grid)
    prev = grid[0][:]
    for i in range(1, n):
        min1 = min2 = float("inf")
        min1_idx = -1
        for j, v in enumerate(prev):
            if v < min1:
                min2 = min1
                min1 = v
                min1_idx = j
            elif v < min2:
                min2 = v
        cur = [0] * n
        for j in range(n):
            cur[j] = grid[i][j] + (min2 if j == min1_idx else min1)
        prev = cur
    return min(prev)


# ── LC132 分割回文串 II(Hard) ────────────────────────────────────────────
def min_cut(s: str) -> int:
    """
    【题意】把字符串 s 分割成若干子串，使每个子串都是回文串，求最少需要多少次切割（切割次数
    = 分段数 - 1）。这是 LC131「分割回文串」（回溯枚举所有方案）的进阶版：131 要列出所有
    合法分割，132 只要最少切割次数，天然是一道区间 DP + 一维 DP 的组合题。
    【思路】分两步：① 用区间 DP 预处理 is_pal[i][j]="s[i:j+1] 是否是回文串"，转移和"最长回文
    子序列"一节的判断方式一致：s[i]==s[j] 且（区间长度<=2 或内部 is_pal[i+1][j-1] 为真）；
    ② dp[i] 定义为"s[0:i+1]（前 i+1 个字符）最少需要多少次切割"，枚举切割点 j（0<=j<=i）,
    如果 is_pal[j][i] 为真（s[j:i+1] 本身就是一个合法回文段），dp[i] = min(dp[i], 0 if j==0
    else dp[j-1]+1)——如果 j==0，说明 s[0:i+1] 整体就是回文串，不需要切割（0 次）；否则在
    s[j-1] 和 s[j] 之间切一刀，代价是 dp[j-1]+1。
    【复杂度】时间 O(n^2)（预处理回文表 O(n^2)，主 DP 双层循环也是 O(n^2)），空间 O(n^2)
    （回文表；dp 数组本身只需 O(n)）。
    【易错点】1) 回文表 is_pal 的遍历顺序必须保证 is_pal[i+1][j-1] 先算出来（i 从大到小、j
    从小到大，和"最长回文子序列"一节完全同源），顺序反了会读到默认值 False，把真正的回文串
    误判为非回文；2) 主 DP 里"j==0 时 dp[i]=0"这个特判很容易漏掉，如果统一写成 dp[j-1]+1，
    j=0 时 dp[-1] 会取到数组最后一个元素（Python 负索引不报错但语义完全错误），必须显式
    处理 j==0 的情况；3) dp 数组初始化成"最坏情况"（比如每个字符单独切割，dp[i]=i）是可选的
    优化剪枝，但即使不加这个初始上界，只要保证每个 i 都能通过某个 is_pal[j][i]（至少
    j==i 单字符自身回文）转移到，正确性不受影响，只是没有这个初始值会导致 min() 在没有任何
    合法转移时出错——好在单字符总是回文，这个退化情况不会发生。
    """
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True
    dp = [0] * n
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        dp[i] = min(dp[j - 1] + 1 for j in range(1, i + 1) if is_pal[j][i])
    return dp[n - 1]


# ── LC1553 吃掉 N 个橘子的最少天数(Hard) ─────────────────────────────────
def min_days(n: int) -> int:
    """
    【题意】有 n 个橘子，每天可以选择：吃 1 个；如果 n 能被 2 整除，吃掉 n/2 个（剩下 n/2 个）；
    如果 n 能被 3 整除，吃掉 2*(n/3) 个（剩下 n/3 个）。每天只能选一种操作，求吃完所有橘子的
    最少天数。n 最大可达 2×10^9，朴素地对每个 n 做线性 DP（dp[n] 依赖 dp[n-1]）在这个规模下
    根本不可能跑完，这是本题真正的难点。
    【思路】关键观察：无论 n 多大，最优策略一定会尽量"跳到"n/2 或 n/3 这种能被大步压缩的
    状态，而不会真的一个一个吃——如果 n 不能被 2 或 3 整除，就先吃掉 n%2（或 n%3）个凑整，
    再整除。这样 f(n) 的递归只会用到 O(log n) 个不同的子问题（每次至少减半或减到三分之一），
    用字典做记忆化搜索即可，而不需要为每个 0~n 之间的整数都建立数组。转移方程：
    f(n) = min( n%2 + 1 + f(n//2), n%3 + 1 + f(n//3) )，其中 "n%2 + 1" 表示先吃掉 n%2 个
    单个橘子（凑出能被 2 整除的数，each 花 1 天），再用 1 天执行"吃一半"操作；n%3 同理。
    base case：f(0)=0（没有橘子，0 天），f(1)=1（只能吃 1 个，1 天）。
    【复杂度】时间 O(log^2 n)（递归树深度 O(log n)，每层最多产生常数个新的不同子问题，用
    lru_cache/字典记忆化后不会重复计算），空间 O(log n)（记忆化表和递归栈的规模）。
    【易错点】1) 如果直接写 dp[n] 数组一路推到 n（线性 DP），在 n 高达 2×10^9 时会直接
    内存爆炸或严重超时，必须转换成"记忆化搜索 + 只探索 O(log n) 个关键状态"的思路，这是
    本题从"看起来像基础 DP"变成"Hard"的核心原因；2) 转移方程里容易漏掉"先吃单个橘子凑整"
    这一步的 +1 天代价，直接写成 f(n//2) 而不accounting 处理 n%2 那部分橘子；3) 记忆化容器
    必须用支持任意大整数键的字典（而不是数组），因为中间产生的 n//2、n//3 依然可能是很大的
    数，不能像常规 dp 数组那样预先分配连续内存。
    """
    from functools import lru_cache as _lru_cache

    @_lru_cache(maxsize=None)
    def f(x: int) -> int:
        if x <= 1:
            return x
        return min(x % 2 + 1 + f(x // 2), x % 3 + 1 + f(x // 3))

    result = f(n)
    f.cache_clear()
    return result


# ── LC1420 生成数组(Hard) ────────────────────────────────────────────────
def num_of_arrays(n: int, m: int, k: int) -> int:
    """
    【题意】构造长度为 n、每个元素取值范围 [1, m] 的数组 arr，定义"搜索代价"为：从左到右扫描
    数组，维护当前已知最大值（初始 -1），每当遇到严格更大的元素就把最大值更新一次并计数加 1，
    最终这个计数就是 search_cost。求有多少种不同的数组 arr，使得 search_cost 恰好等于 k
    （结果对 1e9+7 取模）。这是一道"数位 DP 风格"的计数题——不是真正对十进制数位做 DP，而是
    和数位 DP 共享同一种"逐位构造、按状态计数"的设计思想。
    【思路】三维状态 dp[i][j][c]="构造了 i 个元素、当前最大值恰好是 j、search_cost 恰好是 c"
    的方案数。转移分两种：① 第 i+1 个元素取值在 [1, j] 之间（不超过当前最大值 j，最大值和
    cost 都不变），有 j 种选法，贡献 j*dp[i][j][c]；② 第 i+1 个元素取值恰好等于某个新的最大值
    j（要求 j 严格大于之前的最大值 prev_j），cost 加 1，贡献 sum(dp[i][prev_j][c-1] for
    prev_j < j)。base case：i=1 时，唯一一个元素本身就是这个数组的最大值，所以对所有
    j in [1,m]，dp[1][j][1]=1（cost 恰好是 1，因为第一个元素总会触发一次"发现新最大值"）。
    最终答案是对所有 j 求和 dp[n][j][k]。
    【复杂度】时间 O(n·m^2·k)（这里对"prev_j < j 求和"直接暴力累加，实践中可以用前缀和把
    这一步优化到 O(1)，从而整体降到 O(n·m·k)；本实现为清晰起见保留暴力累加版本），
    空间 O(m·k)（只保留当前长度这一层的二维状态，滚动掉 i 这一维）。
    【易错点】1) "不超过 j 的方案数"这一项的系数是 j（可以取 1..j 中任意一个值都不会破坏
    "当前最大值仍是 j"这个条件），漏写这个系数、或错误地写成 m 而不是 j，会把状态转移的
    基数搞错；2) "产生新最大值"这一项要求 prev_j **严格小于** j（不能等于，等于的话最大值
    没有变化、cost 也不该增加，那属于第①种转移），如果把严格小于误写成小于等于会重复计数；
    3) k=0 时表示"从未更新过最大值"，但根据 base case，长度至少为 1 的数组第一个元素必然
    触发一次更新，所以 k=0 时答案恒为 0（唯一合法情况是 n=0，但题目保证 n>=1），需要在最外层
    对 k==0 提前返回 0，否则 dp[1][j][0] 全部是默认值 0，虽然结果依然正确，但显式提前返回
    可以避免不必要的三维数组分配。
    """
    MOD = 10**9 + 7
    if k == 0:
        return 0
    dp = [[0] * (k + 1) for _ in range(m + 1)]
    for j in range(1, m + 1):
        dp[j][1] = 1
    for _ in range(2, n + 1):
        new_dp = [[0] * (k + 1) for _ in range(m + 1)]
        for j in range(1, m + 1):
            for c in range(1, k + 1):
                val = j * dp[j][c] % MOD
                for prev_j in range(1, j):
                    val = (val + dp[prev_j][c - 1]) % MOD
                new_dp[j][c] = val
        dp = new_dp
    return sum(dp[j][k] for j in range(1, m + 1)) % MOD


# ── LC940 不同的子序列 II(Hard) ──────────────────────────────────────────
def distinct_subseq_ii(s: str) -> int:
    """
    【题意】给定字符串 s，求 s 的不同非空子序列（不要求连续，按字符值比较、不看来源下标）个数，
    结果对 1e9+7 取模。和 LC115「不同的子序列」（在两个字符串之间做匹配计数）不同，这道题是
    单一字符串内部"数不同子序列种类数"，天然需要处理去重。
    【思路】维护长度 26 的数组 ends，ends[c] 表示"以字符 c 结尾的不同子序列一共有多少种"。
    扫描到新字符 ch 时，新增的"以 ch 结尾的子序列"由两部分构成：单独的 "ch" 本身（+1），以及
    "把 ch 接在任意一个已有子序列后面"（当前所有子序列的总数 total，注意这里的 total 包含
    了以任意字符结尾的子序列，接上 ch 之后变成新的以 ch 结尾的子序列）。这两部分之和就是
    ends[ch] 的**新值**（不是累加，而是**覆盖**）——覆盖而非累加正是这里天然去重的关键：
    如果之前已经存在若干个以 ch 结尾的子序列，这次重新计算 ends[ch] 会用"当前的 total"
    重新生成一整套"以 ch 结尾"的子序列，这批新子序列已经完全覆盖了旧的那批（旧的每一种在
    新一轮的 total 里都被计入过一次贡献），所以直接覆盖不会丢方案，也不会重复计数。
    最终答案是所有 26 个字母的 ends 之和。
    【复杂度】时间 O(n)（对每个字符只需 O(26) 求和，实践中可以用一个滚动 total 变量避免
    每次重新 sum(ends)，这里为清晰起见保留 sum 写法），空间 O(26)=O(1)。
    【易错点】1) 最容易犯的错误是把 ends[ch] 写成"累加"（ends[ch] += total+1）而不是覆盖，
    这会导致同一个以 ch 结尾的子序列被重复计数（比如 "aa"：第二个 a 出现时，如果用累加，
    "a" 和 "aa" 这两个方案会被数出两份）；2) total 必须是"覆盖前"的全部 26 个字母之和（
    也就是"这次要接上 ch 的候选对象"是包括 ch 自己旧值在内的全体），不能只统计"除 ch
    以外的 25 个字母"，因为"a"接一个新的"a"变成"aa"是合法的、需要被计入的新子序列；
    3) 结果要对 1e9+7 取模，且中间过程也要及时取模，避免 total 无限增长导致大数运算变慢
    （虽然 Python 大整数不会溢出，但保持取模是标准实践，方便和其他语言实现对齐）。
    """
    MOD = 10**9 + 7
    ends = [0] * 26
    for ch in s:
        idx = ord(ch) - ord("a")
        total = sum(ends) % MOD
        ends[idx] = (total + 1) % MOD
    return sum(ends) % MOD


# ── LC887 鸡蛋掉落(Hard) ─────────────────────────────────────────────────
def super_egg_drop(k: int, n: int) -> int:
    """
    【题意】给定 k 个相同的鸡蛋和 n 层楼，存在一个未知的临界楼层 f（0<=f<=n），从 f 以上的
    楼层扔鸡蛋会碎，从 f 及以下扔不会碎。每次可以选一个还没碎的鸡蛋从任意楼层扔下，碎了就
    少一个可用鸡蛋，没碎可以重复使用。求在**最坏情况下**确定 f 所需的最少扔鸡蛋次数。
    【思路】朴素状态定义 dp[k][n]="k 个鸡蛋、n 层楼最坏情况下最少试验次数"，转移是枚举
    第一次扔的楼层 x：dp[k][n]=1+min_x( max(dp[k-1][x-1]（碎了，往下找）, dp[k][n-x]
    （没碎，往上找）) )，这是 O(k·n^2)（对每个 (k,n) 都要枚举 O(n) 个 x），n 到 10^4 时
    直接超时。**换一个状态定义可以把复杂度降下来**：dp[eggs][moves]="用 eggs 个鸡蛋、
    最多允许 moves 次操作，最坏情况下能确定的最大楼层数"（可以理解为"给定操作预算，能
    覆盖多少层"，是原问题的"对偶"视角）。转移：dp[eggs][moves] = dp[eggs-1][moves-1]
    （这一次扔的鸡蛋碎了，剩 eggs-1 个鸡蛋、moves-1 次操作能确定的楼层数，作为"下方能
    探测的层数"）+ dp[eggs][moves-1]（没碎，剩 eggs 个鸡蛋、moves-1 次操作能探测的楼层数，
    作为"上方能探测的层数"）+ 1（当前这一层本身）。这个转移是 O(1) 的，两层循环枚举
    eggs 和 moves 即可，总复杂度降到 O(k·m)，其中 m 是最终答案（moves 从 1 递增直到
    dp[k][m]>=n 为止，m 的上界是 O(log n)）。
    【复杂度】时间 O(k·log n)（moves 最多到能覆盖 n 层所需的天数，指数增长所以是
    O(log n) 量级），空间 O(k)（一维滚动数组，从鸡蛋数高到低更新，避免同一轮内重复用到
    "本轮已经更新过"的值）。
    【易错点】1) 这道题最大的陷阱是朴素二维 DP 在 n=10^4 时的 O(k·n^2) 会严重超时，必须
    换成"给定操作次数反过来问最大楼层覆盖数"这个对偶状态，很多人会在"要不要用二分优化"
    和"要不要换状态定义"之间纠结，实际上换状态定义（dp[eggs][moves]）比对朴素 dp 加二分
    优化（O(k·n·logn)）更彻底、更优；2) 一维滚动数组更新顺序必须是 eggs **从大到小**（
    倒序），因为 dp[eggs][moves-1] 这一项需要读到"上一轮 moves-1"时的值，如果正序更新,
    eggs-1 这个位置会被提前更新成"本轮"的值，等于错误地让同一轮内的两个不同鸡蛋数共享了
    未来才该出现的信息；3) 终止条件是"dp[k][当前 moves] 首次 >= n"，而不是"恰好等于 n"，
    因为能确定的楼层数是一个覆盖范围，只要覆盖范围达到或超过 n 就说明这个天数足够。
    """
    dp = [0] * (k + 1)
    m = 0
    while dp[k] < n:
        m += 1
        for eggs in range(k, 0, -1):
            dp[eggs] = dp[eggs] + dp[eggs - 1] + 1
    return m


# ── LC1751 最多可以参加的会议数目 II(Hard) ───────────────────────────────
def max_value(events: list[list[int]], k: int) -> int:
    """
    【题意】给定若干场会议 events[i]=[start,end,value]（同一时刻只能参加一场，且结束日和
    另一场的开始日相同也算冲突），最多可以参加 k 场，求能获得的最大价值和。这是「会议室 II」
    「无重叠区间」等区间调度问题和"最多选 k 个物品的背包"结合的进阶版：既要考虑时间不重叠，
    还要考虑"最多选 k 个"这个数量上限，因此不能用纯贪心，需要区间 DP（记忆化搜索）+ 二分。
    【思路】先把 events 按开始时间排序（排序后才能对"下一个不冲突的会议"做二分查找）。定义
    dfs(i, remain)="从下标 i 开始考虑后面所有会议，还能参加 remain 场，最大价值"。对每个
    会议 i 有两种选择：① 跳过它，dfs(i+1, remain)；② 参加它，价值 events[i][2] 加上
    dfs(next_i, remain-1)，其中 next_i 是"结束时间严格大于 events[i][1] 的第一个会议下标",
    用 bisect_right 在按开始时间排序的数组上二分查出来（这也是"区间DP+二分"里"二分"的
    含义：不是对答案二分，而是对"下一个可选区间"做二分定位，避免线性扫描造成 O(n^2)）。两种
    选择取 max，用记忆化（i, remain）避免重复计算相同的子问题。base case：i 越界或 remain=0
    时返回 0。
    【复杂度】时间 O(n·k·log n)（n·k 个状态，每个状态的转移里有一次二分查找），空间
    O(n·k)（记忆化表）。
    【易错点】1) 二分查找的对象必须是"所有会议的开始时间数组"（排序后），且要找的是"第一个
    开始时间严格大于当前会议结束时间"的位置——用 bisect_right(starts, events[i][1])，如果
    误用 bisect_left 会把"结束日等于下一场开始日"这种不允许同时参加的情况错误地当作可以
    衔接的合法情况；2) 排序时必须按"开始时间"排序而不是"结束时间"，这样二分时的"下一个
    候选下标"落在一段连续区间内才有意义（结束时间排序会破坏这个性质，二分出来的下标不再
    对应"时间上紧随其后"的会议）；3) 记忆化的两个维度 (i, remain) 缺一不可——只记 i 会
    导致"还剩多少个名额"这个信息丢失，不同的 remain 对应完全不同的子问题，必须联合两个
    维度做缓存键。
    """
    events_sorted = sorted(events)
    n = len(events_sorted)
    starts = [e[0] for e in events_sorted]
    memo: dict[tuple[int, int], int] = {}

    def dfs(i: int, remain: int) -> int:
        if i == n or remain == 0:
            return 0
        key = (i, remain)
        if key in memo:
            return memo[key]
        best = dfs(i + 1, remain)
        nxt = bisect_right(starts, events_sorted[i][1])
        best = max(best, events_sorted[i][2] + dfs(nxt, remain - 1))
        memo[key] = best
        return best

    return dfs(0, k)


# ── LC1937 扣分后的最大得分(Medium) ──────────────────────────────────────
def max_points(points: list[list[int]]) -> int:
    """
    【题意】给定 m×n 矩阵 points，每一行必须选一个格子，得分是该格子的值；如果相邻两行选的
    列分别是 c1、c2，要扣除 abs(c1-c2) 分，求最终能获得的最大总分。
    【思路】朴素 dp[r][c]=max over c'( dp[r-1][c'] - abs(c-c') ) + points[r][c]，对每个 c
    枚举所有 c' 是 O(n^2)；关键优化是把 abs(c-c') 拆成两种情况分别处理：当 c'<=c 时
    dp[r-1][c']-abs(c-c') = (dp[r-1][c']-c') + c，只依赖"c' 及其左侧的最大 (dp[r-1][c']-c')"，
    可以用一次**从左到右**扫描维护前缀最大值 left[c] = max(left[c-1]-1, dp[r-1][c])（这里
    "-1" 精妙地表达了"c 每往右移一格，之前那些 c' 的距离惩罚都要多算 1"，等价于滚动地把
    "-c'" 的贡献转换成"每步扣 1"）；当 c'>=c 时同理，用**从右到左**扫描维护 right[c]。
    最终 dp[r][c] = points[r][c] + max(left[c], right[c])。这把 O(n) 的"扫描找最优 c'"降到
    O(1) 的前后缀最值查表，整体从 O(m·n^2) 降到 O(m·n)。
    【复杂度】时间 O(m·n)，空间 O(n)（滚动数组，只保留上一行的 dp）。
    【易错点】1) left[c] 的递推 max(left[c-1]-1, dp[c]) 里的 "-1" 极易被误理解为"减去
    abs(c-c')"，但其实这个 -1 只是"每向右移动一格，所有已经计入 left 的候选者的有效值统一
    衰减 1"这一操作的滚动实现，不能拆开单独理解某个 c' 的即时惩罚；2) 必须同时维护 left 和
    right 两个方向的前后缀最值并取 max，只算一个方向会漏掉"最优的 c' 其实在 c 右侧"这种
    情况；3) 第一行没有"上一行"的扣分约束，dp 初始化直接等于 points[0]，如果误加了不存在
    的惩罚项会让第一行结果偏小。
    """
    m, n = len(points), len(points[0])
    dp = points[0][:]
    for r in range(1, m):
        left = [0] * n
        left[0] = dp[0]
        for c in range(1, n):
            left[c] = max(left[c - 1] - 1, dp[c])
        right = [0] * n
        right[-1] = dp[-1]
        for c in range(n - 2, -1, -1):
            right[c] = max(right[c + 1] - 1, dp[c])
        dp = [points[r][c] + max(left[c], right[c]) for c in range(n)]
    return max(dp)


# ── LC1531 压缩字符串 II(Hard) ───────────────────────────────────────────
def get_length_of_optimal_compression(s: str, k: int) -> int:
    """
    【题意】字符串的"游程编码"（run-length encoding）把连续相同字符压缩成"字符+计数"（计数为
    1 时不写数字，比如 "aaabcccd" 压缩成 "a3bc3d"）。最多可以删除 s 中的 k 个字符，求删除后
    能得到的最短压缩编码长度。
    【思路】记忆化搜索，状态是 (i, last_char, last_cnt, k_remain)：i 表示已经处理到第 i 个
    字符，last_char/last_cnt 表示"当前正在累积的这一段连续字符"是什么、已经累积了多长（这
    一段还没有被压缩编码写进最终结果里，是"悬而未决"的一段），k_remain 是还能删除多少个
    字符。对每个字符 s[i] 有两种决策：① 删除它（k_remain 减 1，last_char/last_cnt 不变，
    因为删除的字符不参与任何一段的计数）；② 保留它——如果 s[i]==last_char，累积长度变成
    last_cnt+1，这一步对压缩长度的**增量**是 calc_len(last_cnt+1)-calc_len(last_cnt)（比如
    从 9 个变成 10 个，编码从 "x9"(2字符) 变成 "x10"(3字符)，增量是 1，而不是重新计算整个
    编码长度）；如果 s[i]!=last_char，说明上一段已经定型，开启新的一段，增量是
    calc_len(1)=1（新段初始只有 1 个字符）。取两种决策中的最小值。
    【复杂度】时间 O(n^2·k)（状态数是 i×last_cnt×k_remain，其中 last_cnt 最多到 n，
    last_char 由 s[i] 直接决定不算独立维度；实践中很多状态不可达，实际运行远小于上界），
    空间同状态数。
    【易错点】1) "保留 s[i] 且和 last_char 相同"这一分支必须用**增量**而不是重新计算
    整个编码总长度，很多人会错误地在每一步都调用 calc_len(整个字符串到目前为止的编码)，
    这样虽然逻辑上也能算对，但会把原本简洁的状态转移搞复杂，且容易在"增量"和"全量"两种
    写法之间混用出 bug；2) calc_len 函数的分段边界要精确：0 个字符长度 0，1 个字符长度 1
    （不写数字），2~9 个字符长度 2（1 位数字），10~99 个字符长度 3（2 位数字），100 个
    字符长度 4——这些边界如果记错任何一个，整个 DP 的增量计算都会连锁出错；3) 记忆化的
    k_remain 维度必须允许在"删除到 0"之后依然能继续递归（只是不能再删除），如果把 k=0
    误判为"必须立即终止递归"会漏掉"k 用完之后继续正常保留剩余字符"的合法路径。
    """

    def calc_len(cnt: int) -> int:
        if cnt == 0:
            return 0
        if cnt == 1:
            return 1
        if cnt < 10:
            return 2
        if cnt < 100:
            return 3
        return 4

    n = len(s)

    @lru_cache(maxsize=None)
    def dp(i: int, last: str, last_cnt: int, k_remain: int) -> int:
        if k_remain < 0:
            return float("inf")
        if i == n:
            return 0
        best = float("inf")
        if k_remain > 0:
            best = dp(i + 1, last, last_cnt, k_remain - 1)
        if s[i] == last:
            best = min(
                best,
                calc_len(last_cnt + 1) - calc_len(last_cnt) + dp(i + 1, last, last_cnt + 1, k_remain),
            )
        else:
            best = min(best, calc_len(1) + dp(i + 1, s[i], 1, k_remain))
        return best

    result = dp(0, "", 0, k)
    dp.cache_clear()
    return result


# ── LC1187 使数组严格递增(Hard) ──────────────────────────────────────────
def make_array_increasing(arr1: list[int], arr2: list[int]) -> int:
    """
    【题意】给定两个数组 arr1、arr2，每次操作可以选 arr1 中一个下标 i 和 arr2 中一个下标 j，
    执行 arr1[i]=arr2[j]，求让 arr1 严格递增所需的最少操作次数；如果无法做到，返回 -1。
    【思路】先对 arr2 排序去重（重复值和顺序都不影响可选替换值的"集合"）。用字典 dp 表示
    "处理到当前位置为止，以某个具体数值 last 结尾、且严格递增"所需的最少操作数，key 是
    last 的值，value 是操作数（同一个 last 值取所有可能路径中的最小操作数）。对 arr1 的
    每个元素 num，从上一步的 dp 字典枚举 (last, ops)，有两种转移：① 不替换 num，前提是
    num>last（保持递增），转移到 (num, ops)；② 用 arr2 中"严格大于 last 的最小值"替换 num
    （用 bisect_right 二分查找这个最小合法替换值——因为要严格递增，替换值必须大于 last，
    而"最小的合法替换值"是贪心最优的选择：它给后面留下最大的、后续依然能找到更多合法替换值
    的空间），转移到 (该替换值, ops+1)。如果某一步新字典为空，说明无法继续，直接返回 -1。
    最终答案是最后一步字典里所有 value 的最小值。
    【复杂度】时间 O(n·log(n)·log(m))（n 个元素，每个元素最多枚举 dp 字典里 O(n) 个不同的
    last，每次转移里有一次二分查找；实践中不同 last 值的数量通常远小于最坏情况），
    空间 O(n)（dp 字典的规模）。
    【易错点】1) "替换成严格大于 last 的**最小**值"是这里的贪心核心——如果替换成一个更大
    的合法值，虽然当下同样满足递增，但会让后续更难找到"比它还大"的下一个替换值，属于典型的
    "贪心选择性质"，误用非最小的合法替换值虽然不会让程序崩溃，但会让某些本该可行的方案被
    误判为不可行；2) dp 字典的 key 需要去重合并（用 min 取较优操作数），不能对同一个 last
    值保留多条独立路径，否则字典会不必要地膨胀，且容易在后续转移中重复扩展本该被剪掉的
    劣质分支；3) 初始 sentinel 必须小于 arr1 里所有可能出现的值（本题约束 arr1[i]>=1，用
    -1 作为"最初的 last"是安全的），如果 sentinel 设置不当（比如用 0 而数组里也可能出现
    0 这种边界），会让第一个元素的"不替换"分支被错误地拒绝或错误地接受。
    """
    arr2_sorted = sorted(set(arr2))
    INF = float("inf")
    dp: dict[int, int] = {-1: 0}
    for num in arr1:
        new_dp: dict[int, int] = {}
        for last, ops in dp.items():
            if num > last:
                new_dp[num] = min(new_dp.get(num, INF), ops)
            idx = bisect_right(arr2_sorted, last)
            if idx < len(arr2_sorted):
                candidate = arr2_sorted[idx]
                new_dp[candidate] = min(new_dp.get(candidate, INF), ops + 1)
        if not new_dp:
            return -1
        dp = new_dp
    return min(dp.values())


# ── LC1140 石子游戏 II(Medium) ───────────────────────────────────────────
def stone_game_ii(piles: list[int]) -> int:
    """
    【题意】Alice 和 Bob 轮流取石子堆（Alice 先手），每回合可以取走从当前剩余堆里最前面的
    X 堆（1<=X<=2M），取完后 M 更新为 max(M, X)（M 初始为 1）。两人都采取最优策略，求 Alice
    最终能获得的最大石子数。这是一道零和博弈 DP，核心难度在于状态里要携带"当前允许的取堆
    上限"这个动态变化的参数。
    【思路】dfs(i, M)="从下标 i 开始的剩余堆，当前玩家在给定上限 M 下，能获得的最大石子数"
    （注意这里"当前玩家"是相对的——不管是 Alice 还是 Bob，轮到谁走都调用同一个 dfs，因为
    双方都按最优策略行事，函数天然对称）。枚举这一步取 X 堆（1<=X<=2M），取完之后对手面对
    (i+X, max(M,X)) 这个新状态、同样会按最优策略行动，能拿到 dfs(i+X, max(M,X))，那么
    "当前玩家"能拿到的就是"剩余堆的总和 - 对手接下来能拿到的"，取 X 使这个值最大。
    base case：当 i+2M>=n 时（剩余堆数不超过 2M），可以一次性全部拿走（X 取到剩余堆数以内
    的最大值总能覆盖全部剩余堆），直接返回剩余堆的总和。用 (i, M) 做记忆化避免重复计算。
    【复杂度】时间 O(n^2)（状态数 O(n·n)，因为 M 的取值范围最坏是 O(n)；每个状态转移枚举
    X 是 O(n)，但由于 base case 的剪枝，M 增长很快，实践中远小于理论上界），空间 O(n^2)
    （记忆化表；后缀和数组是 O(n)）。
    【易错点】1) "当前玩家能拿到的石子数 = 后缀和 - 对手接下来最优能拿到的"，这个"用总量
    减对方最优解"的博弈 DP 套路很容易被想成"直接枚举 X 后对己方的贡献是多少"，正确的写法
    必须通过"总量减对手"间接得到，因为己方这一步拿到多少堆，直接对应的是"接下来对手在剩下
    堆里能拿多少"这个子问题，而不是直接可以枚举出来的量；2) M 的更新是 max(M, X) 而不是
    直接替换成 X——即使这次取的堆数比之前的上限小，未来允许的上限也不会缩小，这是"lasting
    upper bound"的语义，写成直接赋值 X 会让上限错误缩水；3) base case 的判断条件是
    i+2*M>=n（能拿完剩余所有堆），如果误写成 i+M>=n 会提前触发终止，少考虑了"未来还能
    用 2M 这个上限"的可能性，导致漏掉更优的分段策略。
    """
    n = len(piles)
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + piles[i]

    @lru_cache(maxsize=None)
    def dfs(i: int, m: int) -> int:
        if i >= n:
            return 0
        if i + 2 * m >= n:
            return suffix_sum[i]
        best = 0
        for x in range(1, 2 * m + 1):
            best = max(best, suffix_sum[i] - dfs(i + x, max(m, x)))
        return best

    result = dfs(0, 1)
    dfs.cache_clear()
    return result


def _self_test() -> None:
    assert min_falling_path_sum_ii([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 13
    assert min_falling_path_sum_ii([[7]]) == 7

    assert min_cut("aab") == 1
    assert min_cut("a") == 0
    assert min_cut("ab") == 1

    assert min_days(10) == 4
    assert min_days(6) == 3
    assert min_days(1) == 1

    assert num_of_arrays(2, 3, 1) == 6
    assert num_of_arrays(5, 2, 3) == 0
    assert num_of_arrays(9, 1, 1) == 1

    assert distinct_subseq_ii("abc") == 7
    assert distinct_subseq_ii("aba") == 6
    assert distinct_subseq_ii("aaa") == 3

    assert super_egg_drop(1, 2) == 2
    assert super_egg_drop(2, 6) == 3
    assert super_egg_drop(3, 14) == 4

    assert max_value([[1, 2, 4], [3, 4, 3], [2, 3, 1]], 2) == 7
    assert max_value([[1, 2, 4], [3, 4, 3], [2, 3, 10]], 2) == 10
    assert max_value([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], 3) == 9

    assert max_points([[1, 2, 3], [1, 5, 1], [3, 1, 1]]) == 9
    assert max_points([[1, 5], [2, 3], [4, 2]]) == 11

    assert get_length_of_optimal_compression("aaabcccd", 2) == 4
    assert get_length_of_optimal_compression("aabbaa", 2) == 2
    assert get_length_of_optimal_compression("aaaaaaaaaaa", 0) == 3

    assert make_array_increasing([1, 5, 3, 6, 7], [1, 3, 2, 4]) == 1
    assert make_array_increasing([1, 5, 3, 6, 7], [4, 3, 1]) == 2
    assert make_array_increasing([1, 5, 3, 6, 7], [1, 6, 3, 3]) == -1

    assert stone_game_ii([2, 7, 9, 4, 4]) == 10
    assert stone_game_ii([1]) == 1

    print(
        "[PASS] p12_dp_basics_iii: 11 题"
        "（下降路径最小和II/分割回文串II/吃掉N个橘子的最少天数/生成数组/"
        "不同的子序列II/鸡蛋掉落/最多可以参加的会议数目II/扣分后的最大得分/"
        "压缩字符串II/使数组严格递增/石子游戏II）全部通过"
    )


if __name__ == "__main__":
    _self_test()
