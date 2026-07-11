"""分类 31（Phase 3 竞赛级新分类）：位运算与状态压缩DP(Bitmask DP)。当"子集"本身就是
状态时（比如"哪些数字已经被用过""哪些技能已经被覆盖""哪些客户已经被满足"），用一个
整数的二进制位表示这个子集，dp 的下标从"一个数"变成"一个整数 + 一个位掩码(bitmask)"
——这是状态空间随 n 指数增长(通常 n<=20)的问题的标准应对方式：位运算(&, |, ^, 取反,
枚举子集)天然契合"子集"这个数学对象，比用哈希集合表示子集快得多、省得多。
"""
from __future__ import annotations

from collections import Counter, deque


def can_i_win(max_choosable_integer: int, desired_total: int) -> bool:
    """
    【题意】LeetCode 464·Medium。两人轮流从 1..maxChoosableInteger 中选一个"没被选过"
    的整数累加到总分上（不能重复选），谁先让总分 >= desiredTotal 谁赢。假设双方都
    绝顶聪明，判断先手是否必赢。
    【题意补充】desiredTotal <= 0 时先手直接算赢；如果所有数字加起来都不够
    desiredTotal，谁都赢不了，返回 False。
    【思路】"哪些数字已经被选过"正是一个子集，用位掩码 `used_mask` 表示（第 i 位为
    1 表示数字 i 已经被选走）。状态是 `(还需要凑够的分数 remaining, used_mask)`，
    但 `remaining` 其实可以由 `used_mask` 反推出来（已用数字之和 = 总和 - 未用
    数字之和），所以只用 `used_mask` 做记忆化的 key 就够了。递归 `dfs(remaining,
    mask)`：枚举当前玩家能选的每个数字 i，如果选了 i 之后 `i >= remaining`（直接
    达标）或者"轮到对手时对手在新状态下必败"（`not dfs(remaining-i, mask|bit)`），
    说明当前玩家有一种必胜的选法，返回 True；所有数字都试过仍找不到必胜选法，才
    返回 False。这是博弈论"存在一种走法让对手进入必败态，当前状态就是必胜态"的
    标准递归 + 状态压缩记忆化。
    【复杂度】时间 O(2^n * n)（n = maxChoosableInteger，最多 2^n 个不同的 mask，
    每个 mask 展开 O(n) 个选择）；空间 O(2^n)（记忆化表）。
    【易错点】1) 忘记先判断"所有数字之和 < desiredTotal"这个必输的边界情况，会
    在这种输入上退化成不必要的指数级搜索（虽然结果仍然正确，但会浪费大量时间，
    某些实现甚至会因为状态定义错误在这种情况下出错）；2) 位掩码从下标 1 开始用
    （`1 << i` 对应数字 i），如果从 0 开始容易在"数字"和"下标"之间搞混；3) 记忆化
    只用 `used_mask` 做 key 是因为 `remaining` 可以被 mask 唯一确定，如果两者都
    存反而会让缓存"看起来不同"实际上等价的状态，浪费记忆化的效果（不算错误，
    但会让复杂度退化）。
    """
    if desired_total <= 0:
        return True
    total = max_choosable_integer * (max_choosable_integer + 1) // 2
    if total < desired_total:
        return False

    memo: dict[int, bool] = {}

    def dfs(remaining: int, used_mask: int) -> bool:
        if used_mask in memo:
            return memo[used_mask]
        result = False
        for i in range(1, max_choosable_integer + 1):
            bit = 1 << i
            if used_mask & bit:
                continue
            if i >= remaining or not dfs(remaining - i, used_mask | bit):
                result = True
                break
        memo[used_mask] = result
        return result

    return dfs(desired_total, 0)


def can_partition_k_subsets(nums: list[int], k: int) -> bool:
    """
    【题意】LeetCode 698·Medium。给定数组 nums 和整数 k，判断能否把数组划分成 k 个
    非空子集，使得每个子集的和都相等。
    【思路】"每个数字要么用、要么不用"，n 个数字的使用情况正是一个位掩码 `mask`。
    先算出目标子集和 `target = sum(nums) / k`（如果不能整除，或者最大的数字本身
    就超过 target，直接无解）。用 `dfs(mask, current_sum)` 表示"已经用掉 mask
    里的数字，当前正在装的这个桶已经装了 current_sum"：枚举一个还没用过的数字 i
    尝试放进当前桶，如果放进去不超过 target 就递归；如果 `current_sum + nums[i]`
    恰好等于 target，说明当前桶装满了，下一次递归自然从 `new_sum % target == 0`
    开始一个新桶（用取模操作巧妙地把"装满就换桶"编码进同一个递归里，不需要显式
    传"当前是第几个桶"）。降序排序是关键的剪枝——大的数字更"挑剔"（能放的桶更少），
    优先尝试大数字能更快地让不可行的分支提前失败。
    【复杂度】时间 O(2^n * n)（最坏情况下每个 mask 都要展开 O(n) 种选择，配合
    记忆化后同一个 (mask, current_sum) 不会重复计算）；空间 O(2^n)。
    【易错点】1) 求和之后不整除 k、或者排序后最大元素超过 target，必须提前判定
    无解并直接返回 False，否则递归会在无解的情况下白白搜索到底；2) 记忆化的 key
    是 `(mask, current_sum)` 而不只是 `mask`——同一个 mask 不可能对应两种不同的
    current_sum（因为 mask 本身唯一确定了已用数字的和、进而唯一确定 current_sum
    在当前桶内的余数），但显式带上 current_sum 能让代码逻辑更清晰、避免推导错误；
    3) 降序排序只是性能优化，不影响正确性，但如果误以为"必须"降序才能得出正确
    答案就理解错了剪枝和正确性的关系。
    """
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    nums = sorted(nums, reverse=True)
    n = len(nums)
    if nums[0] > target:
        return False
    full_mask = (1 << n) - 1

    memo: dict[tuple[int, int], bool] = {}

    def dfs(mask: int, current_sum: int) -> bool:
        if mask == full_mask:
            return True
        if (mask, current_sum) in memo:
            return memo[(mask, current_sum)]
        result = False
        for i in range(n):
            if mask & (1 << i):
                continue
            if current_sum + nums[i] > target:
                continue
            new_sum = (current_sum + nums[i]) % target
            if dfs(mask | (1 << i), new_sum):
                result = True
                break
        memo[(mask, current_sum)] = result
        return result

    return dfs(0, 0)


def shortest_path_all_nodes(graph: list[list[int]]) -> int:
    """
    【题意】LeetCode 847·Hard。给定 n 个节点的无向连通图，可以从任意节点出发、允许
    重复经过节点和边，求"访问过所有节点"所需的最短路径长度（边数）。
    【思路】朴素 BFS 只能求"从某个固定起点到某个固定终点"的最短路，但这里的"目标"
    是一个集合条件——"已经访问过哪些节点"。这正是状态压缩的典型场景：把 BFS 的状态
    从单纯的"当前节点"扩展成 `(当前节点, 已访问节点集合的位掩码)`，因为"接下来该
    往哪走、还需要走多远"完全由这两个信息共同决定，缺一不可——只知道当前节点，
    不知道"还差哪些节点没访问"，就没法判断是否已经达成目标，也没法避免历史状态
    的重复搜索。由于可以从任意节点出发，用"多源 BFS"同时把所有 `(i, 1<<i)` 作为
    起始状态压入队列，第一次有某个状态的 mask 等于全集 `(1<<n)-1` 时，对应的
    距离就是答案——BFS 保证按层扩展，第一次达成"访问过所有节点"这个条件的时刻，
    一定是最短的。
    【复杂度】时间 O(n * 2^n)（状态数是 n * 2^n，每个状态最多扩展 O(n) 条边）；
    空间 O(n * 2^n)（visited 集合和 BFS 队列）。
    【易错点】1) 状态去重必须用 `(mask, node)` 二元组，只用 mask 去重会让"同一个
    已访问集合、但停在不同节点上"这两种本质不同的情况被错误地合并；2) 起点不是
    单一固定值，必须把所有节点作为可能的起点同时入队（多源 BFS），只从节点 0
    出发会错过"从别的节点出发反而更短"的情况；3) n <= 12 是这道题状态压缩可行的
    前提（2^12 = 4096），如果 n 很大，位掩码状态数会爆炸，这也是为什么"状态压缩"
    通常只在 n 较小时使用。
    """
    n = len(graph)
    if n == 1:
        return 0
    full_mask = (1 << n) - 1
    queue: deque[tuple[int, int, int]] = deque((1 << i, i, 0) for i in range(n))
    visited = {(1 << i, i) for i in range(n)}
    while queue:
        mask, node, dist = queue.popleft()
        if mask == full_mask:
            return dist
        for nxt in graph[node]:
            new_mask = mask | (1 << nxt)
            if (new_mask, nxt) not in visited:
                visited.add((new_mask, nxt))
                queue.append((new_mask, nxt, dist + 1))
    return -1  # 题目保证连通，理论上不会执行到这里


def smallest_sufficient_team(req_skills: list[str], people: list[list[str]]) -> list[int]:
    """
    【题意】LeetCode 1125·Hard。给定所需技能列表 req_skills 和每个人掌握的技能
    people[i]，求一个人数最少的"充分团队"——团队里所有人的技能并集要覆盖所有
    req_skills。返回团队成员的下标（任意合法答案均可，题目保证有解）。
    【思路】"已经被覆盖的技能集合"是一个位掩码 `state`（第 j 位为 1 表示第 j 个
    技能已被团队覆盖）。把每个人的技能也编码成一个位掩码 `person_masks[i]`。这是
    一个"最小集合覆盖"问题，用 dp[state] 表示"覆盖 state 这个技能集合所需的最少
    人数"，转移是 `dp[state | person_masks[i]] = min(dp[state] + 1)`，枚举所有
    "已有状态 state" 和 "候选加入的人 i"，用新增一个人来扩展已覆盖的技能集合。
    为了在最后还原出具体的团队名单（而不只是数量），额外维护 `choice[new_state]`
    （达到 new_state 时最后加入的是哪个人）和 `prev_mask[new_state]`（加入这个人
    之前的状态），从 `full_mask` 开始沿着 choice/prev_mask 往回倒推，就能拼出
    完整的团队。
    【复杂度】时间 O(2^m * n)（m 是技能数，n 是人数，枚举每个状态尝试加入每个人）；
    空间 O(2^m)（dp/choice/prev_mask 数组）。
    【易错点】1) dp 数组要初始化成"不可达"（用 +inf 或者足够大的数），只有
    `dp[0] = 0` 是已知的合法起点，如果全部初始化成 0 会让"根本没组队"被误判成
    已经覆盖了所有技能；2) 还原路径时要用 `choice`/`prev_mask` 而不是重新跑一遍
    正向 dp 去猜，正向猜测在有多个并列最优解时可能找到和 dp 值不一致的路径；
    3) 有些人掌握的技能不在 req_skills 里（不需要的技能），编码位掩码时要用一个
    "技能名 -> 位下标"的映射，忽略掉不在这个映射里的技能，不能直接按位置编码。
    """
    skill_to_bit = {skill: i for i, skill in enumerate(req_skills)}
    m = len(req_skills)
    full_mask = (1 << m) - 1
    person_masks = []
    for skills in people:
        mask = 0
        for s in skills:
            if s in skill_to_bit:
                mask |= 1 << skill_to_bit[s]
        person_masks.append(mask)

    INF = float("inf")
    dp: list[float] = [INF] * (1 << m)
    dp[0] = 0
    choice: list[int] = [-1] * (1 << m)
    prev_mask: list[int] = [-1] * (1 << m)

    for state in range(1 << m):
        if dp[state] == INF:
            continue
        for idx, pm in enumerate(person_masks):
            new_state = state | pm
            if dp[state] + 1 < dp[new_state]:
                dp[new_state] = dp[state] + 1
                choice[new_state] = idx
                prev_mask[new_state] = state

    team: list[int] = []
    state = full_mask
    while state != 0:
        person = choice[state]
        team.append(person)
        state = prev_mask[state]
    return team


def max_students_taking_exam(seats: list[list[str]]) -> int:
    """
    【题意】LeetCode 1349·Hard。'#' 表示坏座位、'.' 表示好座位。学生能看到左右、
    左上、右上相邻座位的同学答案，但看不到正前方/正后方。求在不发生"能互相看到
    答案"的前提下，最多能安排多少学生参加考试（学生只能坐在好座位上）。
    【思路】"这一行哪些座位坐了人"是一个位掩码——一行最多 8 个座位(n<=8)，天然
    适合状态压缩。同一行内部的约束（左右相邻不能都坐人）只和这一行自己的 mask
    有关：`mask & (mask << 1) == 0` 就是"没有两个相邻的 1"。跨行约束（左上、
    右上）只和"这一行的 mask"与"上一行的 mask"有关：上一行的 mask 左移一位/右移
    一位，和这一行的 mask 做按位与，如果不为 0 说明有对角冲突。把 dp 定义成
    `dp(row, prev_mask)` ="从第 row 行开始（含），已知上一行选择是 prev_mask，
    后续能坐的最多人数"，枚举这一行所有"内部无相邻冲突、和上一行无对角冲突、
    且只坐在好座位上"的候选状态，取"这一行坐的人数 + 递归到下一行的最优值"的
    最大值。这是把"同一行的选择"整体打包成状态转移的一步，而不是一个座位一个
    座位地做决策。
    【复杂度】时间 O(m * 2^n * 2^n)（最坏情况下每一行要和上一行的每个候选状态
    做比较，n<=8 使得 2^n=256 完全可接受）；空间 O(m * 2^n)（记忆化表）。
    【易错点】1) 判断"内部无相邻冲突"要用 `mask & (mask << 1) == 0`，注意是
    检查 mask 和它左移一位的按位与，而不是错误地写成 `mask & (mask >> 1)`
    只检查了一个方向（虽然对称性上这两种写法對稱本身没问题，但必须和"候选状态
    生成"的位序约定保持一致）；2) 判断"和上一行是否合法"要同时检查 mask 相对
    prev_mask 左移和右移两个方向的对角，只检查一个方向会漏掉另一侧的对角冲突；
    3) 候选状态必须先过滤掉"坐在了坏座位(#)上"的可能，即 `state & ~row_mask
    == 0`，不能等到后面统计人数时才发现非法。
    """
    m = len(seats)
    n = len(seats[0])
    row_masks = []
    for row in seats:
        mask = 0
        for j, c in enumerate(row):
            if c == ".":
                mask |= 1 << j
        row_masks.append(mask)

    def no_adjacent(mask: int) -> bool:
        return (mask & (mask << 1)) == 0

    candidates = [s for s in range(1 << n) if no_adjacent(s)]

    memo: dict[tuple[int, int], int] = {}

    def dp(row: int, prev_mask: int) -> int:
        if row == m:
            return 0
        if (row, prev_mask) in memo:
            return memo[(row, prev_mask)]
        best = 0
        for state in candidates:
            if state & ~row_masks[row]:
                continue
            if state & (prev_mask << 1) or state & (prev_mask >> 1):
                continue
            best = max(best, bin(state).count("1") + dp(row + 1, state))
        memo[(row, prev_mask)] = best
        return best

    return dp(0, 0)


def minimum_number_of_semesters(n: int, relations: list[list[int]], k: int) -> int:
    """
    【题意】LeetCode 1494·Hard。n 门课程编号 1..n，relations[i]=[prev,next] 表示
    prev 必须在 next 之前修完。每学期最多能修 k 门课（且必须已修完所有先修课）。
    求修完所有课程所需的最少学期数（题目保证课程图无环、一定有解）。
    【思路】"已经修完哪些课程"是位掩码 `taken_mask`。dp(taken_mask) = 从当前状态
    出发，修完剩余所有课程所需的最少学期数。每一步先算出"先修课都已满足、且本身
    还没修"的课程集合 `available`（对每门课检查它的先修课位掩码是不是 taken_mask
    的子集）；这一学期具体修 available 里的哪个子集，只要子集大小 <= k 都是合法
    选择——需要枚举 available 的**所有子集**（用经典的 "子集枚举" 技巧
    `sub = (sub-1) & available` 遍历一个集合的所有子集），对每个合法子集递归到
    "修完这些课之后"的状态，取最少学期数。之所以要枚举所有满足条件的子集而不是
    "贪心地一次修满 k 门"，是因为有时候少修几门反而能让下一学期解锁更多新课程，
    贪心并不总是最优。
    【复杂度】时间 O(3^n)（枚举每个 mask 的所有子集，根据组合数性质，对所有
    mask 的子集计数总和是 O(3^n)，n<=15 完全可接受）；空间 O(2^n)（记忆化表）。
    【易错点】1) `available` 只包含"先修课已满足"的课程，判断条件是
    `(taken_mask & prereq[course]) == prereq[course]`，不能写成"有交集"就算
    满足——必须是先修课**全部**被满足；2) 子集枚举必须包含"子集大小 <= k"这个
    过滤，不能默认所有子集都合法（可能凑出比 k 还多的课程组合）；3) 子集枚举
    必须用 `while subset:` 排除空集——如果写成"从 `sub=available` 迭代到
    `sub==0` 且把 0 也处理一遍"，"这一学期一门课都不修"会转移到
    `dp(taken_mask | 0) == dp(taken_mask)`，也就是自己递归调用自己而状态完全
    没有变化，在记忆化结果写回之前就发生自环，直接导致无限递归（这是实现这道题
    时最容易踩的坑，务必确保"这一学期"的候选子集非空）。
    """
    prereq = [0] * (n + 1)
    for prev, nxt in relations:
        prereq[nxt] |= 1 << (prev - 1)

    full_mask = (1 << n) - 1
    memo: dict[int, float] = {}

    def dp(taken_mask: int) -> float:
        if taken_mask == full_mask:
            return 0
        if taken_mask in memo:
            return memo[taken_mask]
        available = 0
        for course in range(1, n + 1):
            bit = 1 << (course - 1)
            if taken_mask & bit:
                continue
            if (taken_mask & prereq[course]) == prereq[course]:
                available |= bit

        best = float("inf")
        subset = available
        while subset:  # 只枚举非空子集：这一学期必须至少修一门课，否则会原地自环
            if bin(subset).count("1") <= k:
                best = min(best, 1 + dp(taken_mask | subset))
            subset = (subset - 1) & available
        memo[taken_mask] = best
        return best

    return int(dp(0))


def can_distribute(nums: list[int], quantity: list[int]) -> bool:
    """
    【题意】LeetCode 1655·Hard。数组 nums 里最多有 50 个不同的值，quantity 是 m
    个客户的订购数量。判断能否把 nums 分给这 m 个客户，使得每个客户拿到的数量恰好
    等于 quantity[i]、且同一个客户拿到的必须是**相同的数字**（不同客户可以拿相同
    或不同的数字，nums 里没用上的数字可以浪费）。
    【思路】"哪些客户已经被满足"是位掩码 `remaining_mask`（这里取的是"还没被满足
    的客户集合"，从全集开始逐步清空）。先统计 nums 里每个数字的出现频次，把频次
    降序排序——因为频次是"能一次性满足多大的量"的资源，从大到小分配更容易剪枝。
    dfs(i, remaining_mask) 表示"用第 i 大的频次组开始，尝试满足 remaining_mask
    里的客户，是否可行"：枚举 remaining_mask 的每个非空子集 sub（这个频次组
    打算同时满足哪些客户），如果这批客户的总需求量 `subset_sum[sub] <= freq[i]`，
    就递归到 `dfs(i+1, remaining_mask ^ sub)`（这批客户被满足，从待满足集合里
    去掉）；也要考虑"这个频次组一个客户都不满足"（sub 取 0，直接跳到下一个频次
    组）。只要存在一种分配方案能让 remaining_mask 最终清空（变成 0），就是可行的。
    【复杂度】时间 O(u * 3^m)（u 是不同数字个数，m 是客户数，对每个频次组枚举
    remaining_mask 的所有子集）；空间 O(u * 2^m)（记忆化表）。
    【易错点】1) 子集枚举要基于"当前还没被满足的客户集合"`remaining_mask`，而
    不是固定的全集，否则会把已经被满足过的客户重复计算进新的分配方案；2)
    `subset_sum` 需要预处理（用"最低位"递推：`subset_sum[mask] =
    subset_sum[mask ^ lowbit] + quantity[idx]`），如果每次都重新累加子集里的
    quantity 会让复杂度多一个 O(m) 因子；3) 频次降序排序只是让"大资源优先分配"
    从而更快剪枝，但记忆化的正确性不依赖这个顺序，不能把它当成正确性的必要条件。
    """
    freq = sorted(Counter(nums).values(), reverse=True)
    m = len(quantity)
    full_mask = (1 << m) - 1
    subset_sum = [0] * (1 << m)
    for mask in range(1, 1 << m):
        low_bit = mask & (-mask)
        idx = low_bit.bit_length() - 1
        subset_sum[mask] = subset_sum[mask ^ low_bit] + quantity[idx]

    memo: dict[tuple[int, int], bool] = {}

    def dfs(i: int, remaining_mask: int) -> bool:
        if remaining_mask == 0:
            return True
        if i == len(freq):
            return False
        if (i, remaining_mask) in memo:
            return memo[(i, remaining_mask)]
        result = False
        subset = remaining_mask
        while subset:
            if subset_sum[subset] <= freq[i] and dfs(i + 1, remaining_mask ^ subset):
                result = True
                break
            subset = (subset - 1) & remaining_mask
        if not result:
            result = dfs(i + 1, remaining_mask)
        memo[(i, remaining_mask)] = result
        return result

    return dfs(0, full_mask)


def count_arrangement(n: int) -> int:
    """
    【题意】LeetCode 526·Medium。1..n 的一个排列 perm(1-indexed)称为"优美排列"，
    如果对每个位置 i(1<=i<=n)，`perm[i] % i == 0` 或 `i % perm[i] == 0` 至少
    有一个成立。求优美排列的总数。
    【思路】"哪些数字已经被放到前面的位置里"是位掩码 `used_mask`。按位置从 1 到
    n 逐个决定"这个位置放哪个数字"：`dfs(pos, used_mask)` 枚举一个还没被使用的
    数字 num，只要满足 `num % pos == 0 or pos % num == 0` 这个整除关系，就可以
    把 num 放到位置 pos，递归到 `dfs(pos+1, used_mask | bit)`；当 `pos > n`
    时说明前面所有位置都成功放置，这是一种合法排列，贡献 1 种方案。这是回溯法
    加上状态压缩记忆化的组合——虽然本质是"枚举全排列"，但记忆化让"以同一个
    used_mask 到达同一个 pos"的子问题不会被重复展开。
    【复杂度】时间 O(2^n * n)（最坏情况下每个 (pos, mask) 状态要枚举 O(n) 个候选
    数字，n<=15 决定了这仍然可行）；空间 O(2^n)（记忆化表，实际状态数远少于
    2^n * n，因为 pos 和 mask 里 1 的个数是绑定的）。
    【易错点】1) 位置和数字都是 1-indexed，位掩码建议开到 `1 << (n+1)` 大小或者
    统一在心里明确"第 num 位对应数字 num"，避免 0/1-indexed 混用导致位对错；
    2) 整除条件是"或"关系（两个方向任意一个成立即可），不能写成"且"；3) 记忆化
    的 key 用 `(pos, used_mask)` 二元组，虽然 `used_mask` 的二进制 1 的个数
    理论上等于 `pos - 1`（可以只用 mask 做 key），显式带上 pos 能让代码逻辑
    更直观，避免在推导"由 mask 反推 pos"时出错。
    """
    memo: dict[tuple[int, int], int] = {}

    def dfs(pos: int, used_mask: int) -> int:
        if pos > n:
            return 1
        if (pos, used_mask) in memo:
            return memo[(pos, used_mask)]
        total = 0
        for num in range(1, n + 1):
            bit = 1 << num
            if used_mask & bit:
                continue
            if num % pos == 0 or pos % num == 0:
                total += dfs(pos + 1, used_mask | bit)
        memo[(pos, used_mask)] = total
        return total

    return dfs(1, 0)


def minimum_time_required(jobs: list[int], k: int) -> int:
    """
    【题意】LeetCode 1723·Hard。给定 jobs 数组和 k 个工人，把每个工作分配给恰好
    一个工人，工人的"工作时间"是分给他的所有工作耗时之和。求一种分配方案，使得
    "工作时间最长的那个工人"的工作时间尽量小，返回这个最小的最大工作时间。
    【思路】"这个工作子集里包含哪些工作"是位掩码，先预处理 `subset_sum[mask]`
    （这个子集里所有工作的总耗时，用最低位递推 O(1) 求出）。把问题转化成"用
    k 个不相交的子集覆盖全部工作，最小化这些子集里 subset_sum 的最大值"。
    `dp[i][mask]` 表示"前 i 个工人一共恰好分到了 mask 这些工作时，这 i 个工人里
    最大工作时间的最小可能值"，转移时枚举"剩余工作(full_mask 异或 mask)"里的
    每个子集 sub 分给第 i+1 个工人：`dp[i+1][mask|sub] = min(max(dp[i][mask],
    subset_sum[sub]))`。这里"取 max 再取 min"体现了"最小化最大值"的核心：每加入
    一个工人，新的最大工作时间是"之前已经确定的最大值"和"这个新工人自己的工作
    量"两者的较大值，我们要在所有分配方式里选出这个较大值最小的方案。
    【复杂度】时间 O(k * 3^n)（对每个工人、每个 mask 都要枚举"剩余部分"的所有
    子集，根据组合数性质是 O(3^n)；n<=12，k<=12）；空间 O(2^n)（滚动数组，每轮
    只保留当前工人层的 dp）。
    【易错点】1) 转移必须保证同一个工作不会被两个工人重复分配——通过"只枚举
    full_mask 异或已分配 mask 的子集"来保证新分配的子集和已有 mask 不相交；
    2) `dp[0][0] = 0` 是唯一已知的起点，其余状态要初始化为 +inf，避免"0 个工人
    就分到了非空工作集合"这种不可能状态被误判为可行；3) 最终答案是
    `dp[k][full_mask]`，不是 `dp[k]` 数组里的最小值——必须恰好用满 k 个工人、
    恰好分完所有工作。
    """
    n = len(jobs)
    full_mask = (1 << n) - 1
    subset_sum = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & (-mask)
        idx = low.bit_length() - 1
        subset_sum[mask] = subset_sum[mask ^ low] + jobs[idx]

    INF = float("inf")
    dp = [INF] * (1 << n)
    dp[0] = 0
    for _ in range(k):
        new_dp = [INF] * (1 << n)
        for mask in range(1 << n):
            if dp[mask] == INF:
                continue
            remaining = full_mask ^ mask
            sub = remaining
            while True:
                new_mask = mask | sub
                cost = max(dp[mask], subset_sum[sub])
                if cost < new_dp[new_mask]:
                    new_dp[new_mask] = cost
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        dp = new_dp
    return int(dp[full_mask])


def minimum_sessions(tasks: list[int], session_time: int) -> int:
    """
    【题意】LeetCode 1986·Medium。n 个任务，每个任务耗时 tasks[i] 小时。一个"工作
    时段"最多连续工作 session_time 小时，一旦开始某个任务必须在同一时段内做完
    （但做完一个可以立刻在同一时段接着做下一个）。求完成所有任务所需的最少工作
    时段数（可以按任意顺序完成任务）。
    【思路】"哪些任务已经完成"是位掩码。预处理 `subset_sum[mask]`（这个子集所有
    任务的总耗时）和 `valid[mask]`（这个子集能否装进同一个时段，即
    `subset_sum[mask] <= session_time`）。`dp[mask]` = 完成 mask 这些任务所需
    的最少时段数：枚举 mask 的每个非空子集 `sub`，如果 `sub` 是一个合法的时段
    （`valid[sub]` 为真），就可以把 sub 当作"最后一个时段"完成的任务，转移是
    `dp[mask] = min(dp[mask ^ sub] + 1)`——先假设最后一个时段做了子集 sub 的
    任务，那么在这之前需要用最优方案完成剩下的 `mask ^ sub`。这是"枚举最后一步
    决策，转移到更小的子问题"这一 DP 通用范式在子集上的具体化。
    【复杂度】时间 O(3^n)（n<=14，对每个 mask 枚举它的所有子集，总和是 O(3^n)）；
    空间 O(2^n)（dp、subset_sum、valid 数组）。
    【易错点】1) `subset_sum` 必须用"最低位递推"预处理，如果每次转移时都现算
    子集的和会让复杂度多一个因子，在 n=14 时可能明显变慢；2) 转移时要枚举 mask
    的**非空**子集作为"最后一个时段"，从 `sub = mask` 开始、`sub = (sub-1) &
    mask` 迭代到 `sub == 0` 之前结束（`while sub:` 天然不包含 0），因为一个
    时段不能不做任何任务；3) 题目保证 `max(tasks) <= session_time`，否则会有
    任务在任何时段里都装不下、导致无解，实现时可以依赖这个保证，不需要额外处理
    "无解"分支。
    """
    n = len(tasks)
    full_mask = (1 << n) - 1
    subset_sum = [0] * (1 << n)
    valid = [False] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & (-mask)
        idx = low.bit_length() - 1
        subset_sum[mask] = subset_sum[mask ^ low] + tasks[idx]
        valid[mask] = subset_sum[mask] <= session_time

    INF = float("inf")
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if valid[sub] and dp[mask ^ sub] + 1 < dp[mask]:
                dp[mask] = dp[mask ^ sub] + 1
            sub = (sub - 1) & mask
    return int(dp[full_mask])


def _self_test() -> None:
    assert can_i_win(10, 11) is False
    assert can_i_win(10, 0) is True
    assert can_i_win(10, 1) is True
    assert can_i_win(4, 6) is True

    assert can_partition_k_subsets([4, 3, 2, 3, 5, 2, 1], 4) is True
    assert can_partition_k_subsets([1, 2, 3, 4], 3) is False

    assert shortest_path_all_nodes([[1, 2, 3], [0], [0], [0]]) == 4
    assert shortest_path_all_nodes([[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]) == 4

    team1 = smallest_sufficient_team(
        ["java", "nodejs", "reactjs"], [["java"], ["nodejs"], ["nodejs", "reactjs"]]
    )
    assert sorted(team1) == [0, 2]
    team2 = smallest_sufficient_team(
        ["algorithms", "math", "java", "reactjs", "csharp", "aws"],
        [
            ["algorithms", "math", "java"],
            ["algorithms", "math", "reactjs"],
            ["java", "csharp", "aws"],
            ["reactjs", "csharp"],
            ["csharp", "math"],
            ["aws", "java"],
        ],
    )
    assert sorted(team2) == [1, 2]

    assert (
        max_students_taking_exam(
            [
                ["#", ".", "#", "#", ".", "#"],
                [".", "#", "#", "#", "#", "."],
                ["#", ".", "#", "#", ".", "#"],
            ]
        )
        == 4
    )
    assert (
        max_students_taking_exam(
            [[".", "#"], ["#", "#"], ["#", "."], ["#", "#"], [".", "#"]]
        )
        == 3
    )
    assert (
        max_students_taking_exam(
            [
                ["#", ".", ".", ".", "#"],
                [".", "#", ".", "#", "."],
                [".", ".", "#", ".", "."],
                [".", "#", ".", "#", "."],
                ["#", ".", ".", ".", "#"],
            ]
        )
        == 10
    )

    assert minimum_number_of_semesters(4, [[2, 1], [3, 1], [1, 4]], 2) == 3
    assert minimum_number_of_semesters(5, [[2, 1], [3, 1], [4, 1], [1, 5]], 2) == 4

    assert can_distribute([1, 2, 3, 4], [2]) is False
    assert can_distribute([1, 2, 3, 3], [2]) is True
    assert can_distribute([1, 1, 2, 2], [2, 2]) is True

    assert count_arrangement(1) == 1
    assert count_arrangement(2) == 2
    assert count_arrangement(3) == 3

    assert minimum_time_required([3, 2, 3], 3) == 3
    assert minimum_time_required([1, 2, 4, 7, 8], 2) == 11

    assert minimum_sessions([1, 2, 3], 3) == 2
    assert minimum_sessions([3, 1, 3, 1, 1], 8) == 2
    assert minimum_sessions([1, 2, 3, 4, 5], 15) == 1

    print(
        "[PASS] p31_bitmask_dp: 10/10 通过 "
        "(LC464/LC698/LC847/LC1125/LC1349/LC1494/LC1655/LC526/LC1723/LC1986)"
    )


if __name__ == "__main__":
    _self_test()
