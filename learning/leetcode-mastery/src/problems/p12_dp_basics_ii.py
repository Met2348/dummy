"""DP 基础/序列 · 进阶补充（Part II）：不重讲框架，扩大 dp[i] 定义变体覆盖面的 13 道题
（环形数组拆解 / 树形 DP / 计数型 DP / 完全背包 / 多维度 dp）。
"""
from __future__ import annotations

import math
from collections import deque


# ── LC213 打家劫舍 II(Medium) ────────────────────────────────────────────
def _rob_linear(nums: list[int]) -> int:
    """线性版打家劫舍（LC198 的核心递推），供 rob_ii 复用：dp[i]=max(dp[i-1], dp[i-2]+nums[i])，
    用两个滚动变量 prev（dp[i-1]）、prev2（dp[i-2]）代替整个数组。"""
    prev, prev2 = 0, 0
    for x in nums:
        prev, prev2 = max(prev, prev2 + x), prev
    return prev


def rob_ii(nums: list[int]) -> int:
    """
    【题意】和 LC198 一样是"相邻不能都偷"，但这次房子首尾相连成一个环，第一间和最后一间
    也算相邻，求能偷到的最大金额。
    【思路】环形约束的本质是"第一间和最后一间不能同时偷"，而不是"两个都不能偷"。把它拆成
    两个互斥又覆盖所有合法方案的线性子问题：① 强制不偷最后一间（此时第一间可以自由决策），
    对 nums[:-1] 跑线性打家劫舍；② 强制不偷第一间，对 nums[1:] 跑线性打家劫舍。这两种情况
    合起来覆盖了"首尾至少有一个不偷"的所有合法方案（不可能两个都偷，但"两个都不偷"的方案
    在①②里都被计入过，不影响取 max 的正确性），取较大值即为答案。只有一间房子时环形和线性
    没有区别，需要单独特判（否则 nums[:-1] 或 nums[1:] 会变成空数组，两个都返回 0）。
    【复杂度】时间 O(n)，空间 O(1)（_rob_linear 内部只用滚动变量，切片产生的两个子数组各是
    O(n) 空间，若追求严格 O(1) 可以传下标区间而非真的切片，这里为清晰起见保留切片写法）。
    【易错点】1) 忘记单独特判 len(nums)==1，会错误地对空数组调用 _rob_linear（虽然
    _rob_linear 对空数组天然返回 0，不会报错，但语义上是"偷了 0 间房"而不是"唯一一间房
    自己"，二者恰好在这道题里数值不同：nums=[5] 时正确答案是 5，若不特判会算成 0）；
    2) 误以为要拆成"偷第一间"和"偷最后一间"两种情况分别递归，其实应该是"不偷最后一间"
    和"不偷第一间"（即排除法而不是选择法），少数人会在这里想反导致漏解。
    """
    if len(nums) == 1:
        return nums[0]
    return max(_rob_linear(nums[:-1]), _rob_linear(nums[1:]))


# ── LC337 打家劫舍 III(Medium) ───────────────────────────────────────────
class TreeNode:
    """二叉树节点：本文件内部共享的测试辅助结构，独立定义，不跨文件 import。"""

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values: list) -> TreeNode | None:
    """按 LeetCode 层序数组（None 表示该位置没有节点，其子树不再展开）构造二叉树，
    仅供本文件 _self_test 使用，独立实现、不依赖其他题目文件里的同名工具。"""
    if not values or values[0] is None:
        return None
    it = iter(values)
    root = TreeNode(next(it))
    queue = deque([root])
    while queue:
        node = queue.popleft()
        try:
            left_val = next(it)
        except StopIteration:
            break
        if left_val is not None:
            node.left = TreeNode(left_val)
            queue.append(node.left)
        try:
            right_val = next(it)
        except StopIteration:
            break
        if right_val is not None:
            node.right = TreeNode(right_val)
            queue.append(node.right)
    return root


def rob_iii(root: TreeNode | None) -> int:
    """
    【题意】把打家劫舍搬到二叉树上：房子是树的节点，"相邻不能同时偷"变成"直接相连的父子
    节点不能同时偷"（隔代、兄弟之间没有限制），求整棵树能偷到的最大金额。
    【思路】树形 DP：对每个节点递归返回一个二元组 (skip, take)，skip 表示"不偷当前节点时，
    以它为根的子树能拿到的最大金额"，take 表示"偷当前节点时的最大金额"——**必须返回两个值
    而不是一个"这棵子树的最优解"**，因为父节点做决策时，如果父节点想偷，子节点必须处在
    "不偷"的状态（不能只知道子树整体最优是多少，还要单独知道"子节点被强制不偷"这一种情况
    下的最优值）。转移：take = node.val + left.skip + right.skip（偷了当前节点，两个孩子
    都不能偷）；skip = max(left.skip, left.take) + max(right.skip, right.take)（不偷当前
    节点，每个孩子各自独立选偷或不偷中更优的一种）。空节点的基准情况是 (0, 0)。根节点的
    答案是 max(skip, take)。
    【复杂度】时间 O(n)（每个节点访问一次），空间 O(h)（递归栈深度等于树高 h）。
    【易错点】1) 只让递归返回一个"这棵子树的最大金额"（不区分偷/不偷根节点），父节点就
    没法判断"我偷了以后，孩子那边到底损失了多少"，会把本不该共存的"父子都偷"的方案错误地
    合并进来；2) take 的转移必须用两个孩子的 skip 而不是 max(skip,take)——因为偷了当前
    节点，孩子就是被强制不偷，不能自由选择；3) 空节点要返回 (0, 0) 而不是 None 或抛异常，
    否则递归到叶子节点的孩子时会出错。
    """

    def dfs(node: TreeNode | None) -> tuple[int, int]:
        if node is None:
            return 0, 0
        left_skip, left_take = dfs(node.left)
        right_skip, right_take = dfs(node.right)
        take = node.val + left_skip + right_skip
        skip = max(left_skip, left_take) + max(right_skip, right_take)
        return skip, take

    skip, take = dfs(root)
    return max(skip, take)


# ── LC91 解码方法(Medium) ────────────────────────────────────────────────
def num_decodings(s: str) -> int:
    """
    【题意】数字字符串 s 按 'A'->1, 'B'->2, ..., 'Z'->26 的规则可以解码成字母串，一个数字
    可能单独解码（1~9）也可能和前一位组成两位数一起解码（10~26），求 s 一共有多少种不同的
    解码方式。
    【思路】dp[i] 定义为"s 的前 i 个字符（s[:i]）有多少种解码方式"。对 s[i-1]（第 i 个字符,
    即最后一个字符）只有两种"消耗"方式：① 它自己单独解码，要求 s[i-1] != '0'（'0' 没有对应
    字母），贡献 dp[i-1] 种方案；② 它和前一位 s[i-2] 组成两位数一起解码，要求这两位组成的数
    落在 10~26 之间，贡献 dp[i-2] 种方案。两种方式互不冲突（i-1 位置要么单独、要么被并进
    两位数，二选一），dp[i] 是两者之和。初始条件 dp[0]=1（空串只有一种"什么都不解码"的方案，
    是递推链的地基），dp[1] 要单独判断 s[0] 是否为 '0'。
    【复杂度】时间 O(n)，空间 O(1)（可以用两个滚动变量代替数组，这里为清晰起见保留数组）。
    【易错点】1) 忘记特判前导 '0'——'0' 既不能单独解码，也不能作为两位数的十位以外的位置
    单独存在（但可以是两位数的个位，比如 "10"、"20" 合法），"06" 这个样例专门用来检验这一点：
    dp[1] 因为 s[0]='0' 必须是 0（无法单独解码），后面即使两位数 "06"=6 也不在 10~26 范围
    内，最终答案是 0；2) 两位数的判断范围是 [10, 26] 而不是 [1, 26] 或 [0, 26]，10 以下的
    两位数（比如 "05"）不构成合法的两位编码；3) dp[0]=1 这个"空串地基"如果漏掉或写成 0，
    dp[2] 依赖 dp[0] 的两位数解码分支会全部算错。
    """
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1 if s[0] != "0" else 0
    for i in range(2, n + 1):
        if s[i - 1] != "0":
            dp[i] += dp[i - 1]
        two_digit = int(s[i - 2 : i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]
    return dp[n]


# ── LC279 完全平方数(Medium) ─────────────────────────────────────────────
def num_squares(n: int) -> int:
    """
    【题意】给定正整数 n，求最少需要多少个完全平方数（1, 4, 9, 16, ...）相加等于 n。
    【思路】和零钱兑换（LC322）结构完全一样的完全背包：把"完全平方数"当作面值不限次数的
    硬币，dp[i] 定义为"凑出 i 所需的最少完全平方数个数"。先枚举出所有不超过 n 的完全平方数
    作为"硬币面值"，dp[i]=min(dp[i], dp[i-sq]+1)，对每个 i 遍历所有可用的 sq。因为每个完全
    平方数可以重复使用（比如 12=4+4+4，4 用了三次），这是完全背包，外层按金额从小到大遍历，
    内层遍历"硬币"面值即可，不需要像 01 背包那样倒序。初始条件 dp[0]=0（凑出 0 需要 0 个）。
    【复杂度】时间 O(n·sqrt(n))（i 从 1 到 n，每个 i 最多遍历 sqrt(n) 个完全平方数），
    空间 O(n)。
    【易错点】1) 生成候选完全平方数时容易漏掉上界，必须包含到恰好 <=n 的最大平方数（用
    math.isqrt(n) 而不是 int(sqrt(n))，后者在浮点误差下可能少算一个，比如某些 n 恰好是
    完全平方数时会被向下取整漏掉）；2) 把这题误当成 01 背包写成倒序遍历，会导致同一个平方数
    不能被重复使用，某些答案（如 12=4+4+4）会被漏掉、算出偏大的错误结果；3) dp 数组初始值
    如果用 0 而不是一个"不可能"的哨兵（比如 float('inf')），min 比较会被 0 污染，所以除
    dp[0] 外其余位置要初始化成一个明确大于任何合法答案的值。
    """
    squares = [i * i for i in range(1, math.isqrt(n) + 1)]
    dp = [0] + [float("inf")] * n
    for i in range(1, n + 1):
        for sq in squares:
            if sq > i:
                break
            dp[i] = min(dp[i], dp[i - sq] + 1)
    return int(dp[n])


# ── LC343 整数拆分(Medium) ───────────────────────────────────────────────
def integer_break(n: int) -> int:
    """
    【题意】给定正整数 n（n>=2），把它拆分成至少两个正整数之和，使这些正整数的乘积最大，
    返回这个最大乘积。
    【思路】dp[i] 定义为"把 i 拆分成至少两个正整数之和，能得到的最大乘积"。枚举第一刀切在
    位置 j（1<=j<i），剩下的 i-j 有两种处理方式：不再继续拆，直接乘 (i-j)（也就是"只拆成
    两段"）；或者继续拆，用 dp[i-j] 代替 (i-j)（"拆成三段及以上"）。dp[i]=max 在所有 j 上
    取 max(j*(i-j), j*dp[i-j])。之所以两种都要比较，是因为"继续拆"不一定总是更优——比如
    i-j=2 或 3 时，不拆比拆开更优（2、3 拆开后每份至少是 1，乘积反而变小），dp 数组会自动
    通过比较 (i-j) 和 dp[i-j] 的大小来做出正确选择，不需要手工判断"什么时候该停止拆分"。
    【复杂度】时间 O(n^2)（两层循环枚举 i 和切分点 j），空间 O(n)。
    【易错点】1) 容易漏掉"不再继续拆"这个分支，只写 j*dp[i-j]，会在 (i-j) 较小（如
    i-j=1,2,3）时算出偏小的错误答案，因为对小数字继续硬拆反而更差；2) dp[1] 未被显式赋值,
    默认是 0，容易担心它会污染结果——但因为"拆出一份大小为 1"从来不会是最优选择（1 乘任何
    数都不会变大），dp[1]=0 参与比较时会被 j*(i-j) 分支自然压过，不影响最终答案；3) n=2 这个
    最小输入只能拆成 1+1，如果代码里对"至少两个正整数"这个约束理解错误（比如误判 n 本身
    也是候选值之一），会把 dp[2] 错误地算成 2 而不是 1。
    """
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
    return dp[n]


# ── LC62 不同路径(Medium) ────────────────────────────────────────────────
def unique_paths(m: int, n: int) -> int:
    """
    【题意】一个 m 行 n 列的网格，机器人从左上角出发，每次只能向右或向下移动一格，求走到
    右下角一共有多少条不同的路径。
    【思路】dp[i][j] 定义为"走到第 i 行第 j 列格子的路径数"。因为只能从上方或左方过来，
    dp[i][j]=dp[i-1][j]+dp[i][j-1]，边界（第一行、第一列）只有一种走法（一路直走），
    dp[0][j]=1、dp[i][0]=1。用一维滚动数组压缩空间：初始化整行为 1（对应第一行），之后
    对每一行从左到右原地更新 dp[j]+=dp[j-1]（此时 dp[j] 还保留着"上一行"的值，加上
    dp[j-1]（本行已经算好的左边格子）就等于新一行的 dp[j]）。
    【复杂度】时间 O(m·n)，空间 O(n)（一维滚动数组；也可以用组合数公式 C(m+n-2, m-1) 做到
    O(min(m,n)) 甚至 O(1)，这里保留 dp 写法便于和后面的"不同路径 II"对比）。
    【易错点】1) 一维滚动数组更新顺序必须从左到右（j 递增），因为 dp[j-1] 需要用"这一行
    刚更新过的新值"，而 dp[j] 本身在被更新前仍是"上一行"的旧值——这和 01 背包"必须倒序"的
    要求恰好相反，容易和背包类问题的遍历顺序搞混；2) 初始化整行为 1 对应的是"第一行"，如果
    误初始化成全 0 会导致第一行永远算出 0。
    """
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]


# ── LC63 不同路径 II(Medium) ─────────────────────────────────────────────
def unique_paths_ii(obstacle_grid: list[list[int]]) -> int:
    """
    【题意】和 LC62 一样的网格走法（只能右/下），但网格里有些格子是障碍物（值为 1），机器人
    不能经过，求到达右下角的不同路径数；如果起点或终点本身是障碍物，返回 0。
    【思路】在 LC62 的一维滚动 dp 基础上加一条规则：如果当前格子是障碍物，dp[j] 直接置 0
    （无论之前累积了多少条路径，撞上障碍物这一格就全部作废，"路径数为 0"会自动通过加法
    传播——后面的格子如果依赖这个 0，路径数也不会凭空变多）；否则才执行 dp[j]+=dp[j-1]
    （j>0 时）。初始条件不再是固定的"整行为 1"，而是要看起点是否是障碍物：dp[0] 一开始设为
    1 或 0，取决于 obstacle_grid[0][0]。
    【复杂度】时间 O(m·n)，空间 O(n)。
    【易错点】1) 障碍物格子必须先把 dp[j] 清零，再判断是否要加 dp[j-1]——顺序反了（先加
    再清零）会让障碍物那一格短暂地"合法"存在过一次；2) 起点是障碍物时要让 dp[0] 从一开始
    就是 0，不能像 LC62 一样无脑设成 1；3) 每一行开头（j=0）那一列不能执行 dp[0]+=dp[-1]
    这种越界操作，必须用 j>0 单独判断"是否允许从左边格子转移过来"。
    """
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [0] * n
    dp[0] = 1 if obstacle_grid[0][0] == 0 else 0
    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]
    return dp[-1]


# ── LC918 环形子数组的最大和(Medium) ─────────────────────────────────────
def max_subarray_sum_circular(nums: list[int]) -> int:
    """
    【题意】给定环形数组 nums（最后一个元素和第一个元素相邻），求环形意义下连续子数组
    （不能重复取同一个元素，也就是子数组长度不能超过 n）的最大和。
    【思路】环形子数组只有两种形态：① 不跨越"接缝"（数组末尾到开头的那个边界），退化成
    普通的最大子数组和（LC53 的 Kadane 算法）；② 跨越接缝，这种情况下"选中的部分"其实是
    "两端各选一段"，等价于"总和减去中间没被选中的那一段"，而"没被选中的那一段"要让整体和
    最大，就要让它自己的和最小——也就是 total - 最小子数组和。两种形态取较大值即为答案。
    最小子数组和用同样的 Kadane 思路反过来做（取 min 而不是 max）。**边界陷阱**：如果数组
    全是负数，最小子数组和会恰好等于整个数组的总和（因为全负时"最小的连续子数组"就是数组
    本身），这时候"total - best_min" 会算出 0，代表"选一个空的环形部分"，但题目要求子数组
    至少有一个元素，这种情况必须退化成只用形态①的结果（即普通 Kadane 得到的 best_max，
    此时一定是数组中的最大单个负数）。
    【复杂度】时间 O(n)（一次遍历同时维护 Kadane 最大值和最小值），空间 O(1)。
    【易错点】1) 判断"是否需要排除环形情况"要用 best_min == total（而不是 best_max<0
    之类的启发式），这是全负数组的精确特征：只有当"最小子数组"吞掉了整个数组时，"跨接缝"
    的候选才会退化成空集；2) 维护 Kadane 时最大值和最小值必须用同一种初始化方式（从
    nums[0] 开始而不是从 0 开始），从 0 开始初始化在全负情况下会把 best_min 算错（把 0 这个
    从未真实出现过的"空前缀"当成了候选值）；3) 容易忘记同时维护 total（数组总和），它是
    "total - best_min" 这条公式的必需输入。
    """
    total = 0
    cur_max = best_max = nums[0]
    cur_min = best_min = nums[0]
    for x in nums:
        total += x
    for x in nums[1:]:
        cur_max = max(x, cur_max + x)
        best_max = max(best_max, cur_max)
        cur_min = min(x, cur_min + x)
        best_min = min(best_min, cur_min)
    if best_min == total:
        return best_max
    return max(best_max, total - best_min)


# ── LC673 最长递增子序列的个数(Medium) ───────────────────────────────────
def find_number_of_lis(nums: list[int]) -> int:
    """
    【题意】给定整数数组 nums，求其中最长严格递增子序列（LIS）的**个数**（不同下标组合算
    不同方案，即使子序列的值相同）。
    【思路】在 LC300 朴素 O(n^2) 解法（dp[i]="以 nums[i] 结尾的 LIS 长度"）的基础上，额外
    维护 cnt[i]="以 nums[i] 结尾、且长度恰好为 dp[i] 的 LIS 有多少种"。枚举 j<i 且
    nums[j]<nums[i] 时分两种情况更新：① 如果 dp[j]+1 严格大于当前 dp[i]，说明找到了一条
    更长的路径，dp[i] 和 cnt[i] 都要**重置**成 dp[j]+1 和 cnt[j]（之前累积的方案数作废，
    因为它们对应的是更短的长度）；② 如果 dp[j]+1 恰好等于当前 dp[i]，说明这是另一条同样
    长度的路径，cnt[i] 要**累加** cnt[j]（而不是重置）。最终答案是"长度等于全局最大 dp 值"
    的所有位置的 cnt 之和（因为最长递增子序列可能以不同下标结尾，每个都要计入）。
    【复杂度】时间 O(n^2)，空间 O(n)（两个长度为 n 的数组）。
    【易错点】1) 长度相等时忘记"累加"而是直接覆盖 cnt[i]=cnt[j]，会漏掉其他同样能延伸出
    相同长度的路径；2) 长度更长时忘记同步重置 cnt[i]（只更新了 dp[i] 但 cnt[i] 还留着旧
    长度对应的计数），会把"新的最长长度"和"旧的方案数"错误地拼在一起；3) 最终求和时只统计
    "长度等于最大长度"的位置，容易漏掉这一步过滤，直接把所有 cnt 加起来（那样统计的是所有
    长度的方案总数，而不是最长的那些）。
    """
    n = len(nums)
    if n == 0:
        return 0
    length = [1] * n
    count = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                if length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
                    count[i] = count[j]
                elif length[j] + 1 == length[i]:
                    count[i] += count[j]
    max_len = max(length)
    return sum(c for l, c in zip(length, count) if l == max_len)


# ── LC1043 分隔数组以得到最大和(Medium) ──────────────────────────────────
def max_sum_after_partitioning(arr: list[int], k: int) -> int:
    """
    【题意】把数组 arr 分成若干个连续子数组，每个子数组长度不超过 k，把每个子数组中的所有
    元素都替换成该子数组的最大值，求替换后整个数组元素之和的最大可能值。
    【思路】dp[i] 定义为"arr 前 i 个元素（arr[:i]）经过最优分段替换后的最大和"。枚举"最后一
    段"的长度 length（从 1 到 min(k, i)），这一段是 arr[i-length:i]，替换成这段的最大值
    cur_max 之后贡献 cur_max*length，加上前面 i-length 个元素的最优解 dp[i-length]。边
    枚举 length 边动态更新 cur_max（每次把新纳入的元素 arr[i-length] 和已有的 cur_max 取
    最大值），避免对每个 length 重新扫一遍这一段求最大值。初始条件 dp[0]=0（空数组和为 0）。
    【复杂度】时间 O(n·k)（n 个位置，每个位置最多枚举 k 种分段长度），空间 O(n)。
    【易错点】1) 内层枚举 length 时必须"从 1 递增"并同步更新 cur_max，如果反过来想"先确定
    这一段再算最大值"会退化成每次都重新扫描一遍这一段，复杂度变成 O(n·k^2)；2) length 的
    上界是 min(k, i) 而不是固定的 k——当 i<k 时（比如数组开头位置），可用的元素不够 k 个,
    必须用 i 兜底，否则会往下标 -1 甚至更小的地方越界取值；3) cur_max 的更新用的是
    arr[i-length]（新纳入这一段最左侧的元素），而不是 arr[i-1]（这一段最右侧的元素）——
    因为 length 递增时，每次新加入的是"往左扩展"的那个元素。
    """
    n = len(arr)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        cur_max = 0
        for length in range(1, min(k, i) + 1):
            cur_max = max(cur_max, arr[i - length])
            dp[i] = max(dp[i], dp[i - length] + cur_max * length)
    return dp[n]


# ── LC264 丑数 II(Medium) ────────────────────────────────────────────────
def nth_ugly_number(n: int) -> int:
    """
    【题意】丑数是只含质因子 2、3、5 的正整数（1 也算丑数），求第 n 个丑数。
    【思路】三指针法：维护一个已经算出来的丑数序列 ugly（按从小到大的顺序），以及三个指针
    i2, i3, i5，分别表示"下一个候选丑数是 ugly[i2]*2、ugly[i3]*3、ugly[i5]*5"——也就是说
    每个指针都在追踪"已生成序列里的某个丑数乘以对应质因子之后，第一个还没被用过的候选值"。
    每一步取三个候选中的最小值作为新的丑数（这保证了生成顺序单调递增，不会漏掉更小的候选）；
    如果某个候选恰好等于这个最小值，说明它已经被"消费"了，对应指针要前进一位（用 if 而不是
    elif，因为可能同时有两个甚至三个候选相等，例如 6=2*3 既是 ugly[?]*2 也是 ugly[?]*3，
    这时两个指针都要前进，否则会把同一个丑数在序列里重复生成两次）。
    【复杂度】时间 O(n)，空间 O(n)。
    【易错点】1) 三个"指针前进"的 if 判断必须相互独立（都用 if 各自判断，不能写成
    if/elif/else 只允许一个分支生效），否则遇到 6、12 这类同时是多个质因子候选的丑数时会
    生成重复值，导致后续序列整体错位；2) 初始丑数序列要从 [1] 开始（1 本身是第一个丑数），
    如果漏掉这个基准会导致整条序列的起点错误；3) n=1 时应直接返回 1，循环体
    range(1, n) 在 n=1 时天然不会执行，返回 ugly[0]=1，不需要额外特判，但容易担心这里的
    边界而误加不必要的特判代码。
    """
    ugly = [1] * n
    i2 = i3 = i5 = 0
    for i in range(1, n):
        next2, next3, next5 = ugly[i2] * 2, ugly[i3] * 3, ugly[i5] * 5
        nxt = min(next2, next3, next5)
        ugly[i] = nxt
        if nxt == next2:
            i2 += 1
        if nxt == next3:
            i3 += 1
        if nxt == next5:
            i5 += 1
    return ugly[n - 1]


# ── LC1027 最长等差数列(Medium) ──────────────────────────────────────────
def longest_arith_seq_length(nums: list[int]) -> int:
    """
    【题意】给定整数数组 nums，求其中最长等差子序列（下标递增、相邻两项差值恒定，不要求
    下标连续）的长度。
    【思路】比 LIS 多一个维度：不能只按"结尾下标"分状态，还要按"公差"分状态，否则无法判断
    "延续的是哪一个公差"。dp[i][diff] 定义为"以下标 i 结尾、公差为 diff 的等差数列最长
    长度"，用字典（而不是二维数组，因为 diff 的取值范围可能很大且稀疏）实现每个下标一个
    dict。枚举 j<i，diff=nums[i]-nums[j]，如果 dp[j] 里已经存在同样的 diff（意味着存在一条
    以 nums[j] 结尾、公差为 diff 的等差数列），dp[i][diff]=dp[j][diff]+1；如果 dp[j] 里没有
    这个 diff（.get(diff, 1) 兜底为 1），说明最短也能构成"nums[j], nums[i]"这样长度为 2 的
    等差数列。全局答案是所有 dp[i][diff] 中的最大值。
    【复杂度】时间 O(n^2)（外层 i、内层 j 各 O(n)，dict 查找摊销 O(1)），空间 O(n^2)
    （最坏情况下每个下标都要为 O(n) 个不同的 diff 建立条目）。
    【易错点】1) dp[j].get(diff, 1) 的默认值必须是 1 而不是 0——即使 j 之前完全没有以 diff
    为公差的记录，"nums[j] 和 nums[i]" 这两个数本身也已经构成一条长度为 2 的等差数列（把
    dp[j][diff] 隐式理解成"长度为 1 的数列"，加上 nums[i] 变成长度 2），如果默认值写成 0，
    算出来的长度会全部少 1；2) 用二维列表按 diff 直接索引（而非字典）容易因为 diff 可能是
    负数或跨度很大导致数组越界或浪费大量空间，字典是这里更稳妥的选择；3) 更新 dp[i][diff]
    时要和"已经存在的 dp[i][diff]"比较取较大值（不同的 j 可能推出相同的 diff 但长度不同），
    直接覆盖会丢掉之前更优的结果。
    """
    n = len(nums)
    if n == 0:
        return 0
    dp: list[dict[int, int]] = [dict() for _ in range(n)]
    best = 1
    for i in range(n):
        for j in range(i):
            diff = nums[i] - nums[j]
            length = dp[j].get(diff, 1) + 1
            if length > dp[i].get(diff, 0):
                dp[i][diff] = length
                best = max(best, length)
    return best


# ── LC926 将字符串翻转到单调递增(Medium) ─────────────────────────────────
def min_flips_mono_incr(s: str) -> int:
    """
    【题意】给定只含 '0'/'1' 的字符串 s，每次操作可以把某一位从 '0' 翻转成 '1' 或反过来，
    求最少翻转多少位，能让整个字符串变成"单调递增"（形如 0...01...1，前面全 0、后面全 1，
    允许全 0 或全 1）。
    【思路】维护两个滚动状态：zero 表示"扫描到当前位为止，把这个前缀变成全 '0' 所需的最少
    翻转次数"；one 表示"扫描到当前位为止，把这个前缀变成合法的单调递增串（不要求全 1，可以
    是若干 0 后面跟若干 1）所需的最少翻转次数"。遇到字符 '0' 时：zero 不需要翻转就能延续
    "全 0"状态，直接沿用旧的 zero；one 的新状态要么延续旧的"单调递增"状态但把这个 0 翻转成
    1（因为 one 要求整体单调递增，如果前面已经进入了"1 区"，这个新的 0 必须翻转），要么从
    "全 0"状态转过来同样翻转这个 0，取 min(zero, one)+1。遇到字符 '1' 时：zero 要把这个 1
    翻转成 0 才能保持"全 0"，zero+1；one 不需要翻转就能延续（无论之前是"全 0"还是"已经在
    1 区"，接一个天然的 1 都合法），直接取 min(zero, one)（沿用较优的旧状态，不产生新翻转）。
    最终答案是 min(zero, one)（整个字符串可以是全 0，也可以是"若干 0 后接若干 1"）。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 容易把 one 理解成"必须全是 1"，但实际上 one 表示的是"整个前缀已经是合法
    的单调递增串"（可能仍全是 0），这也是为什么最终答案要在 zero 和 one 之间取 min 而不是
    直接返回 one——如果 one 的定义已经涵盖了全 0 的情况，两者理论上不会让 one 比 zero 更差,
    但代码层面仍需要显式比较，漏掉这个 min 在某些边界字符串上会返回错误的偏大值；2) 遇到
    '0' 时更新 one 必须基于"更新前"的 zero 和 one（同一步内不能用已经被更新过的 zero 去算
    one，否则相当于用了未来的信息），需要在同一次循环体内先读旧值再统一赋新值，或者像这里
    这样利用 Python 的元组同时赋值天然保证"同一时刻的旧值"参与运算；3) 初始值 zero=one=0
    对应"空前缀"，如果误设成 float('inf') 之类的哨兵，第一个字符的状态转移就会被污染。
    """
    zero = one = 0
    for ch in s:
        if ch == "0":
            zero, one = zero, min(zero, one) + 1
        else:
            zero, one = zero + 1, min(zero, one)
    return min(zero, one)


def _self_test() -> None:
    assert rob_ii([2, 3, 2]) == 3
    assert rob_ii([1, 2, 3, 1]) == 4
    assert rob_ii([0]) == 0

    assert rob_iii(build_tree([3, 2, 3, None, 3, None, 1])) == 7
    assert rob_iii(build_tree([3, 4, 5, 1, 3, None, 1])) == 9

    assert num_decodings("12") == 2
    assert num_decodings("226") == 3
    assert num_decodings("06") == 0

    assert num_squares(12) == 3
    assert num_squares(13) == 2

    assert integer_break(2) == 1
    assert integer_break(10) == 36

    assert unique_paths(3, 7) == 28
    assert unique_paths(3, 2) == 3

    assert unique_paths_ii([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2
    assert unique_paths_ii([[0, 1], [0, 0]]) == 1

    assert max_subarray_sum_circular([1, -2, 3, -2]) == 3
    assert max_subarray_sum_circular([5, -3, 5]) == 10
    assert max_subarray_sum_circular([-3, -2, -3]) == -2

    assert find_number_of_lis([1, 3, 5, 4, 7]) == 2
    assert find_number_of_lis([2, 2, 2, 2, 2]) == 5

    assert max_sum_after_partitioning([1, 15, 7, 9, 2, 5, 10], 3) == 84

    assert nth_ugly_number(10) == 12

    assert longest_arith_seq_length([3, 6, 9, 12]) == 4
    assert longest_arith_seq_length([9, 4, 7, 2, 10]) == 3
    assert longest_arith_seq_length([20, 1, 15, 3, 10, 5, 8]) == 4

    assert min_flips_mono_incr("00110") == 1
    assert min_flips_mono_incr("010110") == 2
    assert min_flips_mono_incr("00011000") == 2

    print(
        "[PASS] p12_dp_basics_ii: 13 题"
        "（打家劫舍II/III/解码方法/完全平方数/整数拆分/不同路径/不同路径II/"
        "环形子数组最大和/最长递增子序列个数/分隔数组最大和/丑数II/"
        "最长等差数列/单调递增翻转）全部通过"
    )


if __name__ == "__main__":
    _self_test()
