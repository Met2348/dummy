"""13 DP 进阶：01 背包、双序列比较(LCS/编辑距离)、区间 DP(戳气球) 六道题的规范实现。"""
from __future__ import annotations


# ── LC121 买卖股票的最佳时机(Easy) ───────────────────────────────────────
def max_profit(prices: list[int]) -> int:
    """
    【题意】给定每天的股价数组 prices，只允许买一次、之后卖一次（必须先买后卖），
    求能获得的最大利润；如果无法获利返回 0。
    【思路】把"选一对 (buy_day, sell_day), buy_day < sell_day, 使 price[sell]-price[buy]
    最大"翻译成"枚举卖出日，对每一天只关心它之前出现过的最低价格"——不需要对每一对
    下标暴力枚举 O(n^2)，只需要一次从左到右扫描，边走边维护"目前为止见过的最低价"，
    每天都用 当天价格 - 目前最低价 去更新答案。这是"用一个滚动变量替代重复子问题"
    的最简单例子，也是后面所有背包/双序列 DP 的雏形：把"过去的最优信息"压成一个变量。
    【复杂度】时间 O(n)：一次遍历；空间 O(1)：只用两个变量。
    【易错点】① 容易写成暴力枚举所有 (i, j) 导致 O(n^2)；② 容易错误地直接用
    max(prices) - min(prices)，但如果全局最低点出现在全局最高点之后，这是非法的
    （必须先买后卖）——本算法天然规避了这个问题，因为 min_price 只会更新成"当前
    位置之前"的最低价。
    """
    min_price = float("inf")
    best = 0
    for p in prices:
        if p < min_price:
            min_price = p
        elif p - min_price > best:
            best = p - min_price
    return best


# ── LC122 买卖股票的最佳时机 II(Medium) ──────────────────────────────────
def max_profit_multi(prices: list[int]) -> int:
    """
    【题意】同样给定每天股价，但这次允许多次买卖（同一天可以先卖再买，不能同时持有
    多笔），求能获得的最大总利润。
    【思路】这题是状态机 DP 的贪心简化版：完整写法是定义 dp[i][0] = 第 i 天"不持有
    股票"的最大利润、dp[i][1] = 第 i 天"持有股票"的最大利润，转移
    dp[i][0] = max(dp[i-1][0], dp[i-1][1] + price[i])，
    dp[i][1] = max(dp[i-1][1], dp[i-1][0] - price[i])。
    但因为交易次数不受限，可以证明"总利润 = 所有相邻上升区间差价之和"——只要明天比
    今天贵，就当作"今天买、明天卖"，把每一段正的差价都吃到；这等价于状态机 DP 展开后
    抵消掉了中间态，不需要真的维护两个数组。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】容易去找"波峰波谷对"这种更复杂的实现，其实只需要比较相邻两天的差值，
    正的就累加；也容易忘记这题和 LC121 的本质区别只在"能不能多次交易"。
    """
    total = 0
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            total += diff
    return total


# ── LC416 分割等和子集(Medium) ───────────────────────────────────────────
def can_partition(nums: list[int]) -> bool:
    """
    【题意】给定一个正整数数组，判断能否把它分成两个子集，使两个子集的元素和相等。
    【思路】两个子集和相等 <=> 存在一个子集的和恰好等于 sum(nums)/2（另一个子集自动
    是剩下的部分）。于是问题转化成标准 01 背包："能不能从 nums 里选出若干个数（每个
    数只能用一次），凑出目标和 target = sum/2"。dp[j] 表示"是否能凑出和为 j"，
    dp[0] = True（什么都不选，和为 0）；对每个数 num，从大到小更新 dp[j] |= dp[j-num]。
    【复杂度】时间 O(n·target)，空间 O(target)（一维滚动数组）。
    【易错点】① 总和为奇数时直接返回 False（不可能平分成两个整数和相等的子集），别漏判；
    ② 01 背包的内层循环必须**从 target 倒序遍历到 num**——如果正序遍历，当处理到 dp[j]
    时可能已经用了"这一轮刚更新过的 dp[j-num]"，相当于同一个数被选了两次，退化成"完全
    背包"（可重复选）；倒序保证更新 dp[j] 时用的是上一轮（还没加入当前 num）的状态。
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            if dp[j - num]:
                dp[j] = True
    return dp[target]


# ── LC1143 最长公共子序列(Medium) ────────────────────────────────────────
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    【题意】给定两个字符串，求它们最长公共子序列（可以不连续，但必须保持相对顺序）
    的长度；不存在公共子序列返回 0。
    【思路】双序列 DP 的标准模板：定义 dp[i][j] = text1 前 i 个字符与 text2 前 j 个
    字符的最长公共子序列长度。两个下标 i, j 同时向前推进：如果 text1[i-1] == text2[j-1]
    （两个字符相等，可以一起纳入子序列），dp[i][j] = dp[i-1][j-1] + 1；否则这两个字符
    至少有一个对最终子序列没有直接贡献，取"丢弃 text1 的第 i 个字符"和"丢弃 text2 的
    第 j 个字符"两种情况的较大者：dp[i][j] = max(dp[i-1][j], dp[i][j-1])。
    【复杂度】时间 O(m·n)，空间 O(m·n)（可用滚动数组压到 O(n)，此处为教学清晰保留二维）。
    【易错点】dp 数组习惯性开成 (m+1)×(n+1) 并把下标 0 留给"空串"这一基准情况，容易
    在取字符时忘记 -1（dp[i][j] 对应的是 text1[i-1] 而不是 text1[i]）。
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ── LC72 编辑距离(Hard) ──────────────────────────────────────────────────
def min_distance(word1: str, word2: str) -> int:
    """
    【题意】给定两个字符串 word1、word2，每次操作可以对 word1 插入一个字符、删除一个
    字符、或替换一个字符，求把 word1 变成 word2 所需的最少操作数。
    【思路】双序列 DP 的通用模板（和 LCS 结构一样，只是转移含义不同）：dp[i][j] 表示
    "把 word1 前 i 个字符变成 word2 前 j 个字符"所需的最少操作数。三种转移分支各对应
    一个下标的移动：
      - 插入一个字符（在 word1 末尾插入 word2[j-1]，让它俩对齐）→ 消耗 word2 一个字符，
        i 不动、j 退一步：dp[i][j-1] + 1；
      - 删除一个字符（删掉 word1[i-1]）→ 消耗 word1 一个字符，i 退一步、j 不动：
        dp[i-1][j] + 1；
      - 替换一个字符（把 word1[i-1] 换成 word2[j-1]）→ 两边各退一步：dp[i-1][j-1] + 1。
    如果 word1[i-1] == word2[j-1]，这两个字符天然相等，不需要操作，直接
    dp[i][j] = dp[i-1][j-1]（成本 0）；否则三种操作取最小值。
    【复杂度】时间 O(m·n)，空间 O(m·n)。
    【易错点】① 初始化边界 dp[i][0] = i（word2 是空串，word1 前 i 个字符全部删除）、
    dp[0][j] = j（word1 是空串，需要插入 j 个字符）容易漏写或写反；② 三个操作对应的
    下标移动方向记混——插入对应"消耗对方"、删除对应"消耗自己"，替换是"两边都消耗"。
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


# ── LC312 戳气球(Hard) ───────────────────────────────────────────────────
def max_coins(nums: list[int]) -> int:
    """
    【题意】给定气球数组 nums（每个数是气球上的数字），每次戳破一个气球 i 可以得到
    nums[left]·nums[i]·nums[right] 个硬币（left、right 是当前"还没被戳破"的相邻气球，
    如果不存在则视为 1），求把所有气球戳破能得到的最大硬币数。
    【思路】区间 DP 最反直觉的一步：如果按照"先戳哪个气球"去想，戳完一个气球后左右
    相邻关系会变化，子问题之间互相纠缠、无法独立求解。正确的做法是**反过来想"最后
    戳破哪个气球"**：先在首尾各补一个值为 1 的虚拟气球（数组变成
    [1] + nums + [1]），定义 dp[i][j] = 戳破**开区间** (i, j) 内所有气球（不包括 i、j
    本身）能得到的最大硬币数。枚举这个开区间里**最后一个**被戳破的气球 k（i < k < j）——
    因为 k 是最后戳的，戳它的时候 (i, k) 和 (k, j) 两段还都是完整的，戳 k 能拿到的硬币
    数就固定是 nums[i]·nums[k]·nums[j]，与 (i,k)、(k,j) 两段"如何戳"完全解耦，
    于是 dp[i][j] = max_k( dp[i][k] + dp[k][j] + nums[i]·nums[k]·nums[j] )。这就是
    "枚举最后一步"把互相纠缠的子问题拆成互不干扰的两半的经典手法。
    【复杂度】时间 O(n^3)（区间长度 × 左端点 × 枚举 k），空间 O(n^2)。
    【易错点】① 忘记首尾补 1 气球，导致边界气球的系数算错；② dp[i][j] 定义的是
    "开区间 (i,j) 内部"而不含 i、j 本身，i、j 只是"当前还健在的边界"，新手很容易
    把它误当成"闭区间"来枚举 k，导致重复计算或漏算边界项。
    """
    balloons = [1] + nums + [1]
    n = len(balloons)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left + 1, right):
                score = balloons[left] * balloons[k] * balloons[right]
                total = dp[left][k] + dp[k][right] + score
                if total > dp[left][right]:
                    dp[left][right] = total
    return dp[0][n - 1]


def _self_test() -> None:
    assert max_profit([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit([7, 6, 4, 3, 1]) == 0

    assert max_profit_multi([7, 1, 5, 3, 6, 4]) == 7
    assert max_profit_multi([1, 2, 3, 4, 5]) == 4
    assert max_profit_multi([7, 6, 4, 3, 1]) == 0

    assert can_partition([1, 5, 11, 5]) is True
    assert can_partition([1, 2, 3, 5]) is False

    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("abc", "def") == 0

    assert min_distance("horse", "ros") == 3
    assert min_distance("intention", "execution") == 5

    assert max_coins([3, 1, 5, 8]) == 167
    assert max_coins([1, 5]) == 10

    print("[PASS] p13_dp_advanced: 6 题(股票x2/01背包/LCS/编辑距离/戳气球)全部通过")


if __name__ == "__main__":
    _self_test()
