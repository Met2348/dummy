"""回溯专题 Part III（竞赛级补充）：组合 / 活字印刷 / 连续差相同的数字 /
二进制手表 / 黄金矿工 / 扰乱字符串 / 火柴拼正方形 / 找出所有子集的异或总和再求和。
（优美的排列改归入 31-bitmask-dp 类，避免和该类重复）

不重复讲 Part I/II 已经建立的"做选择 -> 递归 -> 撤销选择"模板和"排序+同层跳过重复"
去重技巧，本文件聚焦 Frontier Lab 级别面试更看重的两条线索：1) 回溯天然是指数级的，
一旦子问题会被不同路径反复问到相同的问题（比如"s1 的某个子串和 s2 的某个子串是否
互为扰乱字符串"），必须叠加"记忆化"把重复子问题降到多项式级，扰乱字符串是这一技巧
最经典的载体；2) 回溯剪枝可以来自完全不同的地方——不仅是"用过的下标/字典序"，还
可以是"多桶均分的对称性剪枝"（火柴拼正方形）、"网格里原地淹没再复原"（黄金矿工）。
"""
from __future__ import annotations

from collections import Counter


# ── LC77 组合 ────────────────────────────────────────────────────────────
def combine(n: int, k: int) -> list[list[int]]:
    """
    【题意】给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合（顺序
    不作要求，但一个组合内部不能有重复数字，两个组合之间也不能内容相同）。
    【思路】回溯基本功中的基本功——"从 start 开始往后扫，每选一个数下一层从
    i+1 开始"，和 Part I 的子集几乎是同一套骨架，区别只是这里的终止条件是
    "len(path)==k 才收集"而不是"每个节点都收集"。核心剪枝：如果从当前下标 i 到 n
    剩下的数字个数（`n - i + 1`）已经不够凑满 `k - len(path)` 个,那么这一层 for
    循环后面的下标也肯定不够,直接 break,不用一个个试。这个剪枝在 n 较大、k 较大
    时能显著减少无意义的递归调用。
    【复杂度】时间 O(C(n,k)·k)（每个组合要花 O(k) 拷贝 path）；空间 O(k) 递归深度。
    【易错点】1) 忘记这个"剩余数字不够"的剪枝，程序依然能给出正确答案,但在 n 很大
    时会浪费大量时间尝试注定失败的分支；2) 剪枝条件里的 `n - i + 1` 容易差一，
    需要想清楚"从 i 到 n（含两端）一共有多少个数"再对比还差多少个；3) 忘记
    `path[:]` 拷贝直接 `append(path)`，导致后续 pop 把已收集的组合改空。
    """
    res: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, n + 1):
            if n - i + 1 < k - len(path):
                break
            path.append(i)
            backtrack(i + 1)
            path.pop()

    backtrack(1)
    return res


# ── LC1079 活字印刷 ──────────────────────────────────────────────────────
def num_tile_possibilities(tiles: str) -> int:
    """
    【题意】给定一串大写字母瓷砖 tiles（可能含重复字母），每个瓷砖最多用一次，
    统计能拼出的所有"非空字母序列"（顺序不同算不同序列）的数目。
    【思路】和"全排列 II"是同一个"候选集合含重复元素"的场景，但这里连长度都不
    固定——任意长度（1 到 len(tiles)）的序列都要计数。用 `Counter` 统计每种字母
    剩余可用的数量，而不是给每个下标单独维护 `used` 数组：这样"两个下标值相同的
    瓷砖互相替换"这件事天然被去重（因为遍历的候选是"哪些字母还有剩余"，不是"哪个
    下标"），不需要像全排列 II 那样额外写"同层跳过重复"的判断。每进入一层递归就
    计数 +1（因为这一层对应的路径本身就是一个合法的非空序列），然后对每种还有剩余
    的字母，减少它的计数、递归、再恢复计数。
    【复杂度】时间 O(sum_{k=1}^{n} P(n,k))（n 是瓷砖数，最坏各字母互不相同时的
    有序排列总数，n<=7 所以可接受）；空间 O(n) 递归深度 + O(26) 计数表。
    【易错点】1) 误用"下标去重"的思路（照抄全排列 II 的 `used`+排序去重），会把
    "两个位置字母相同"和"两个位置字母不同"混为一谈，需要额外写同层跳过逻辑，比
    直接用 Counter 麻烦且容易出错；2) 忘记序列长度是任意的（不需要凑满
    len(tiles)），如果只在递归到底（用完所有瓷砖）时才计数，会漏掉所有更短的
    序列；3) 递归本身既是"计数点"又是"继续展开的起点"，容易漏加当前层这一次的
    +1（只在叶子计数，漏掉中间节点）。
    """
    counts = Counter(tiles)

    def backtrack() -> int:
        total = 0
        for ch in list(counts.keys()):
            if counts[ch] > 0:
                total += 1
                counts[ch] -= 1
                total += backtrack()
                counts[ch] += 1
        return total

    return backtrack()


# ── LC967 连续差相同的数字 ───────────────────────────────────────────────
def numbers_same_consec_diff(n: int, k: int) -> list[int]:
    """
    【题意】给定 n 和 k，返回所有长度为 n、且相邻两位数字之差的绝对值恰好等于 k
    的整数（不能有前导零，n>=2 时首位不能是 0；顺序不作要求）。
    【思路】逐位构造：第一位必须是 1-9（避免前导零），之后每一位只有两种候选——
    在上一位数字基础上 `+k` 或 `-k`，只要结果落在 0-9 之间就是合法候选（k=0 时
    这两种候选会重合，用集合去重避免同一个数字被生成两次）。凑够 n 位后把 path
    拼成整数收集。这是本类里"候选个数从固定的一批（1-9、0-9）收窄到最多 2 个"
    的回溯，剪枝几乎是免费的——非法候选（超出 0-9 范围）直接被 for 循环的边界
    条件排除，不需要额外判断。
    【复杂度】时间 O(9·2ⁿ⁻¹)（首位 9 种选择，之后每一位最多 2 种分支，是一棵
    分支因子很小的树，n<=9 时完全可接受）；空间 O(n) 递归深度。
    【易错点】1) 第一位也用"上一位 ±k"的通用逻辑，会漏掉"第一位没有上一位"这个
    特殊情况，需要单独处理第一位可以是 1-9 中任意数字；2) k=0 时 `+k` 和 `-k`
    算出同一个候选，如果不用集合去重会把同一个数字往 path 里重复推两次，生成
    两条路径拼出同一个整数（虽然结果集合去重后不影响正确性，但会做双倍无意义
    的递归）；3) 忘记候选必须落在 [0,9] 区间，直接把 `last+k` 或 `last-k` 当成
    合法数字使用，会拼出十进制以外的"数字"。
    """
    res: list[int] = []
    path: list[int] = []

    def backtrack() -> None:
        if len(path) == n:
            res.append(int("".join(map(str, path))))
            return
        if not path:
            for d in range(1, 10):
                path.append(d)
                backtrack()
                path.pop()
            return
        last = path[-1]
        for nd in {last + k, last - k}:
            if 0 <= nd <= 9:
                path.append(nd)
                backtrack()
                path.pop()

    backtrack()
    return res


# ── LC401 二进制手表 ─────────────────────────────────────────────────────
def read_binary_watch(turned_on: int) -> list[str]:
    """
    【题意】二进制手表用 4 个 LED 表示小时（0-11）、6 个 LED 表示分钟（0-59），
    给定亮着的 LED 总数 turned_on，返回所有可能表示的时间字符串（"H:MM" 格式，
    小时不补零、分钟必须两位且可以补零）。
    【思路】把 4 个小时 LED 和 6 个分钟 LED 看成 10 个可以"选或不选"的候选（前 4
    个 LED 的权重分别是 8/4/2/1 对应小时的贡献，后 6 个 LED 的权重分别是
    32/16/8/4/2/1 对应分钟的贡献），回溯"从这 10 个候选里选恰好 turned_on 个"——
    这正是"组合"（LC77）的变体，唯一区别是候选不是数字 1~n，而是"点亮某个 LED
    带来的数值贡献"，且要在最后额外校验小时 <=11、分钟 <=59（选出的组合可能让
    小时或分钟溢出合法范围，比如同时选中所有 4 个小时 LED 会得到 15 点，这是不
    合法的，需要在收集答案前过滤掉）。
    【复杂度】时间 O(C(10,turned_on))（最多枚举 2^10=1024 种 LED 开关组合的一个
    子集，规模很小）；空间 O(1) 结果外的递归深度。
    【易错点】1) 把"选 LED"和"直接枚举小时分钟再数二进制 1 的个数"这两种思路的
    复杂度搞混——本题用回溯枚举 LED 子集是为了呼应"组合"这一主题，直接枚举
    0-11 小时、0-59 分钟数 popcount 也完全正确且更直观，两种写法都是 O(1) 级别，
    不存在谁更优；2) 分钟格式化忘记补零（`f"{minute:02d}"` 写成
    `f"{minute}"`），比如分钟是 5 时应该输出 "05" 而不是 "5"；3) 忘记在选出一组
    LED 之后额外校验小时 <=11、分钟 <=59 这两个上限，直接把所有点亮 turned_on
    个 LED 的组合都当成合法时间收集，会把 12:00~15:63 这类不存在的时间也输出。
    """
    hour_bits = [8, 4, 2, 1]
    minute_bits = [32, 16, 8, 4, 2, 1]
    all_bits = hour_bits + minute_bits
    res: list[str] = []

    def backtrack(start: int, remain: int, hour: int, minute: int) -> None:
        if remain == 0:
            if hour <= 11 and minute <= 59:
                res.append(f"{hour}:{minute:02d}")
            return
        for i in range(start, len(all_bits)):
            if len(all_bits) - i < remain:
                break
            if i < 4:
                backtrack(i + 1, remain - 1, hour + all_bits[i], minute)
            else:
                backtrack(i + 1, remain - 1, hour, minute + all_bits[i])

    backtrack(0, turned_on, 0, 0)
    return res


# ── LC1219 黄金矿工 ──────────────────────────────────────────────────────
def get_maximum_gold(grid: list[list[int]]) -> int:
    """
    【题意】给定 m x n 的金矿网格 grid，每个格子是这个位置的金子数量（0 表示没有
    金子）；可以从任意一个有金子的格子出发，每步向上下左右任意一个相邻格子走，
    不能走 0 的格子，也不能重复走同一个格子（走过的格子不能再走第二次），求能
    收集到的最大金子总数。
    【思路】DFS + 原地标记回溯：因为出发点不固定，外层要枚举网格里每一个有金子
    的格子作为起点，各自跑一次 DFS 求"从这个起点出发能收集到的最大金子数"，取
    全局最大值。DFS 本身是"淹没-递归-复原"的标准回溯写法：进入 `dfs(r, c)` 时
    先记下当前格子的金子数、把这个格子临时改成 0（相当于"标记已访问"，因为 0
    本身就是"不能走"的格子，天然复用了这个语义，不需要额外的 visited 集合）；
    向四个方向递归，取四个方向里能收集到的最大值；离开这个格子之前把原来的金子
    数复原（"撤销选择"），这样其他方向或者其他起点的搜索仍然能正常走到这个格子。
    【复杂度】时间最坏 O(4^25)（题目保证有金子的格子最多 25 个，这是一个已知
    很小的常数上界，不随网格整体大小 m,n<=15 增长）；空间 O(25) 递归深度。
    【易错点】1) 用独立的 `visited` 集合而不是原地把格子改成 0 再复原，逻辑上
    也完全正确，但这里选择复用"0 表示不能走"这个语义，让淹没操作同时承担"标记
    访问"和"没有金子提前剪枝"两个作用，是本题最简洁的写法；2) 忘记在递归返回后
    把格子的金子数复原，会导致后续从其他起点出发的搜索发现这个格子"凭空消失"，
    算出偏小的答案；3) 忘记外层要枚举所有有金子的格子作为起点（只从第一个非零
    格子出发），会漏掉从其他起点出发才能达到的更优路径。
    """
    rows, cols = len(grid), len(grid[0])

    def dfs(r: int, c: int) -> int:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
            return 0
        gold = grid[r][c]
        grid[r][c] = 0
        best = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            best = max(best, dfs(r + dr, c + dc))
        grid[r][c] = gold
        return gold + best

    return max(dfs(r, c) for r in range(rows) for c in range(cols))


# ── LC87 扰乱字符串（重点难题：回溯 + 记忆化）───────────────────────────
def is_scramble(s1: str, s2: str) -> bool:
    """
    【题意】给定等长的两个字符串 s1、s2，判断 s2 是否是 s1 的"扰乱字符串"：把 s1
    看成一棵递归构造出的二叉树——每次可以把当前字符串从任意位置切成左右两个非空
    子串，然后（可选地）交换这两个子串的左右顺序，再对每个子串递归做同样的操作，
    如果通过若干次这样的"切割+可选交换"能把 s1 变成 s2，就返回 True。
    【思路】朴素回溯的写法很直接："枚举切割点 cut，不交换时递归检查左半对左半、
    右半对右半；交换时递归检查左半对（s2 的）右半、右半对（s2 的）左半"——但这
    个朴素版本会指数级爆炸：因为不同的切割路径会反复问同一个问题，比如判断
    `is_scramble(s1[2:5], s2[3:6])` 这个子问题，可能在"s1 整体切 3 刀"和"s1 整体
    切 4 刀又不交换"这两条完全不同的搜索路径里被重复计算很多次，且随着字符串
    变长，重复的量级是指数级的。**用记忆化把重复子问题降下来**——发现一个子问题
    只由三个量唯一确定：s1 的起始下标 i、s2 的起始下标 j、子串长度 length（两个
    子串长度必须相等，所以只需要一个 length，不需要两个），用 `memo[(i,j,length)]`
    缓存这个子问题的布尔结果；第一次算出来存起来，之后任何切割路径再次问到完全
    相同的 `(i,j,length)`，直接查表返回，不再重新展开递归树。除了记忆化，还加了
    两个提前剪枝：1) 两个子串完全相等，直接 True（不需要再往下切）；2) 两个子串
    的字符多重集合（`sorted()` 之后比较）不同，说明不可能通过任何切割+交换互相
    转换，直接 False，不需要浪费时间枚举切割点。
    【复杂度】时间 O(n⁴)（状态数 O(n³)——i、j 各 O(n)、length 额外 O(n)，每个状态
    内部要枚举 O(n) 个切割点，且每次比较子串/排序子串还有 O(n) 或 O(n log n) 的
    常数）；不加记忆化的朴素版本最坏是指数级（子问题被重复展开的次数随字符串
    长度指数增长）。空间 O(n³)（memo 字典最多缓存这么多不同的 (i,j,length) 状态）
    + 递归栈。
    【易错点】1) 不加记忆化直接朴素递归，在长度稍长（比如 s1、s2 长度超过
    15~20）的数据上会因为重复子问题的指数级展开而严重超时——这正是本题作为
    "回溯+记忆化经典难题"被收录的原因；2) 漏掉"字符多重集合不同就提前剪枝"这
    一步，虽然不影响正确性（最终枚举所有切割点也会得出 False），但会让每个
    "本来一望而知不可能"的子问题都要完整展开 O(n) 个切割点才能确认失败，浪费
    大量时间；3) 交换情况的下标算错——不交换时是
    `s1[i:i+cut]` 对 `s2[j:j+cut]`、`s1[i+cut:i+length]` 对 `s2[j+cut:j+length]`；
    交换时左边的 `s1[i:i+cut]` 要去和 s2 的**右边** `s2[j+length-cut:j+length]`
    比较，右边的 `s1[i+cut:i+length]` 要去和 s2 的**左边** `s2[j:j+length-cut]`
    比较——如果下标算反，交换分支永远查不到真正应该匹配的子串对，会把本该是
    True 的情况误判为 False。
    """
    n = len(s1)
    if len(s2) != n:
        return False

    memo: dict[tuple[int, int, int], bool] = {}

    def dfs(i: int, j: int, length: int) -> bool:
        key = (i, j, length)
        if key in memo:
            return memo[key]
        if s1[i : i + length] == s2[j : j + length]:
            memo[key] = True
            return True
        if sorted(s1[i : i + length]) != sorted(s2[j : j + length]):
            memo[key] = False
            return False
        for cut in range(1, length):
            if dfs(i, j, cut) and dfs(i + cut, j + cut, length - cut):
                memo[key] = True
                return True
            if dfs(i, j + length - cut, cut) and dfs(i + cut, j, length - cut):
                memo[key] = True
                return True
        memo[key] = False
        return False

    return dfs(0, 0, n)


# ── LC473 火柴拼正方形 ───────────────────────────────────────────────────
def makesquare(matchsticks: list[int]) -> bool:
    """
    【题意】给定一个整数数组 matchsticks，每个元素是一根火柴的长度，判断能否不
    折断任何一根火柴、把所有火柴恰好用一次拼出一个正方形（四条边长度都相等）。
    【思路】"把 n 根火柴分成 4 组，每组和相等"的回溯：先算出总长度，如果不能被
    4 整除直接 False；否则目标边长 `side = total // 4`。用一个长度为 4 的
    `sides` 数组表示当前四条边各自已经累积的长度，按顺序尝试把每一根火柴放进
    某一条边（`sides[i] += matchsticks[idx]`），只有放入后不超过 `side` 才继续
    递归处理下一根火柴，处理完（撤销）后再尝试放入下一条边。两个关键剪枝：
    1) 火柴从大到小排序——大火柴的可选桶更少（更容易触发"超过 side 就剪掉"），
    优先放大火柴能更快地让不可行的分支提前失败，是回溯里"最受约束变量优先"的
    经典应用；2) 对称性剪枝——如果 `sides[i]` 当前是 0，说明把当前火柴放进这个
    "空桶"和放进"另一个同样是空桶的边"是完全对称的选择（不管放进哪一个空桶，
    子问题的结构都一样），只需要尝试其中一个空桶就够了，尝试完之后立刻 break，
    不用把剩下的空桶都试一遍。
    【复杂度】时间最坏 O(4ⁿ)（每根火柴有 4 个桶可选，n 是火柴数，但两条剪枝在
    实际数据上大幅削减分支)；空间 O(n) 递归深度 + O(4) 的 sides 数组。
    【易错点】1) 忘记先判断 `matchsticks[0] > side`（排序后最大的一根比目标边
    长还长）就直接进入回溯，会做一堆注定失败的搜索，应该提前直接返回 False；
    2) 漏掉对称性剪枝（`sides[i]==0` 时 break），在火柴数量较多时会有大量重复
    子问题被反复搜索（"这根火柴放空桶A"和"这根火柴放空桶B"其实是同一件事却被
    当成两条不同分支展开）；3) 不排序或升序排序，导致小火柴先被放进桶里、大
    火柴到最后才尝试，很晚才能触发"超过 side"的剪枝，回溯树规模明显更大。
    """
    total = sum(matchsticks)
    if total % 4 != 0:
        return False
    side = total // 4
    matchsticks = sorted(matchsticks, reverse=True)
    if matchsticks[0] > side:
        return False

    n = len(matchsticks)
    sides = [0, 0, 0, 0]

    def backtrack(idx: int) -> bool:
        if idx == n:
            return True
        for i in range(4):
            if sides[i] + matchsticks[idx] <= side:
                sides[i] += matchsticks[idx]
                if backtrack(idx + 1):
                    return True
                sides[i] -= matchsticks[idx]
            if sides[i] == 0:
                break
        return False

    return backtrack(0)


# ── LC1863 找出所有子集的异或总和再求和 ──────────────────────────────────
def subset_xor_sum(nums: list[int]) -> int:
    """
    【题意】数组的"异或总和"定义为数组所有元素按位异或的结果（空数组的异或总和
    是 0）。给定 nums，对它的每一个子集（含空集和它本身，一共 2ⁿ 个），求出各自
    的异或总和，返回这些异或总和的和。
    【思路】和 Part II 的目标和 `find_target_sum_ways`、Part I 子集 `subsets` 是
    同一种"每个元素独立做二选一决策"的回溯：对下标 `idx` 处的元素，要么把它异或
    进当前累计值 `cur_xor`，要么不异或（保持 `cur_xor` 不变），递归到
    `idx==len(nums)` 时说明这个子集的选择已经决定完毕，把 `cur_xor` 累加进全局
    答案 `res`。这是纯粹的穷举计数，不需要 path 列表记录具体子集，因为题目只
    关心异或总和的和，不关心是哪个子集贡献的。
    【复杂度】时间 O(2ⁿ)（2ⁿ 个子集，每个子集的异或总和在递归过程中用 O(1) 增量
    维护，不需要重新计算）；空间 O(n) 递归深度。
    【易错点】1) 误以为可以像"只出现一次的数字"那样用位运算规律走捷径而跳过
    枚举——本题确实存在 O(n) 的位运算公式解法（每一位如果在 nums 中至少出现一次
    非零，这一位对答案的贡献是 `该位的值 << (n-1)`，因为其余 n-1 个元素自由选择
    都不影响这一位是否被异或进去），但这里选择回溯写法是为了呼应"每个元素二选
    一"这一贯穿本类的主题，两种写法复杂度分别是 O(2ⁿ) 和 O(n)，回溯版在 nums
    较大时明显更慢；2) 把"异或"误写成"求和"（比如写成 `cur_sum + nums[idx]`），
    这是最容易犯的手误，异或用 `^` 不是 `+`；3) 忘记空集也要贡献一次异或总和 0
    进最终的和——本实现的空集分支最终会走到 `idx==len(nums)` 且 `cur_xor==0`，
    自动把 0 累加进 res，不需要特殊处理，但如果换一种写法容易漏掉这一支。
    """
    res = 0
    n = len(nums)

    def backtrack(idx: int, cur_xor: int) -> None:
        nonlocal res
        if idx == n:
            res += cur_xor
            return
        backtrack(idx + 1, cur_xor ^ nums[idx])
        backtrack(idx + 1, cur_xor)

    backtrack(0, 0)
    return res


def _self_test() -> None:
    assert sorted(map(tuple, combine(4, 2))) == sorted(
        [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    )
    assert combine(1, 1) == [[1]]

    assert num_tile_possibilities("AAB") == 8
    assert num_tile_possibilities("AAABBC") == 188
    assert num_tile_possibilities("V") == 1

    assert sorted(numbers_same_consec_diff(3, 7)) == sorted([181, 292, 707, 818, 929])
    assert sorted(numbers_same_consec_diff(2, 1)) == sorted(
        [10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98]
    )

    assert sorted(read_binary_watch(1)) == sorted(
        ["0:01", "0:02", "0:04", "0:08", "0:16", "0:32", "1:00", "2:00", "4:00", "8:00"]
    )
    assert read_binary_watch(9) == []

    assert get_maximum_gold([[0, 6, 0], [5, 8, 7], [0, 9, 0]]) == 24
    assert (
        get_maximum_gold([[1, 0, 7], [2, 0, 6], [3, 4, 5], [0, 3, 0], [9, 0, 20]]) == 28
    )

    assert is_scramble("great", "rgeat") is True
    assert is_scramble("abcde", "caebd") is False
    assert is_scramble("a", "a") is True

    assert makesquare([1, 1, 2, 2, 2]) is True
    assert makesquare([3, 3, 3, 3, 4]) is False

    assert subset_xor_sum([1, 3]) == 6
    assert subset_xor_sum([5, 1, 6]) == 28
    assert subset_xor_sum([3, 4, 5, 6, 7, 8]) == 480

    print(
        "[PASS] p10_backtracking_iii: 8 道回溯竞赛级补充题"
        "（组合/活字印刷/连续差相同的数字/二进制手表/黄金矿工/"
        "扰乱字符串/火柴拼正方形/找出所有子集的异或总和再求和）全部通过"
        "（优美的排列改归入 31-bitmask-dp 类，避免和该类重复）"
    )


if __name__ == "__main__":
    _self_test()
