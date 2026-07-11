"""分类 32（Phase 3 竞赛级新分类）：博弈论与组合数学。博弈论题的核心是"先手是否
必胜"这类问题通常能通过"当前状态是必胜态还是必败态"的记忆化搜索/DP 解决——一个
状态是必胜态，当且仅当存在至少一种走法能让对手进入必败态；反过来，一个状态是必败态，
当且仅当无论怎么走，对手都能进入必胜态。这一类还包含"两人轮流取最优策略、最大化/
最小化得分差"的经典区间 DP 模型，以及少量组合数学分析（不需要显式搜索、靠数学
性质直接推出结论）。
"""
from __future__ import annotations

from collections import deque


def can_win_nim(n: int) -> bool:
    """
    【题意】LeetCode 292·Easy。桌上有 n 颗石子，两人轮流取 1~3 颗，先手先取，
    取走最后一颗石子的人获胜。假设双方都绝顶聪明，判断先手是否必赢。
    【思路】这是最基础的"必胜态/必败态"模型，且它的必胜态/必败态呈现简单的周期
    规律，可以不必显式记忆化搜索就推出结论：当 n 是 4 的倍数时，无论当前玩家取
    1、2 还是 3 颗，对手总能取"4 减去你取的数量"颗，把剩余数量重新变回 4 的倍数
    ——这样对手总能把"4 的倍数"这个必败态重新甩回给你，直到剩 0 颗，轮到你面对
    "0 颗可取"的必败局面。反过来，只要 n 不是 4 的倍数，当前玩家总能取
    `n % 4` 颗，把剩余数量变成 4 的倍数，甩给对手一个必败态。这个结论也能用
    记忆化搜索 `dfs(n) = any(not dfs(n-i) for i in (1,2,3) if n-i>=0)` 验证，
    只是对这道题而言数学归纳出的周期规律比显式搜索更直接。
    【复杂度】时间 O(1)（直接判断 n % 4）；空间 O(1)。
    【易错点】1) 边界 n=0 时当前玩家无子可取，属于必败态，n=0 时 `0 % 4 == 0`
    自动落入"必败"分支，不需要特判；2) 容易把"取最后一颗石子的人获胜"错记成
    "取到第 n 颗输"，导致必胜/必败条件正好写反；3) 这个 O(1) 结论是"因为博弈的
    必胜态呈现 4 的周期"这个具体数学性质带来的，不是所有博弈问题都能这样化简
    成公式，多数题目仍然需要显式的记忆化搜索。
    """
    return n % 4 != 0


def can_win_flip_game(s: str) -> bool:
    """
    【题意】LeetCode 294·Medium。字符串只包含 '+' 和 '-'，两人轮流把其中相邻的
    两个 "++" 翻转成 "--"，谁先无法操作（找不到 "++"）就输。判断先手是否必赢。
    【思路】标准的"必胜态 = 存在一种走法让对手进入必败态"记忆化搜索：枚举字符串
    里每一个能翻转的位置 i（`s[i]=='+' and s[i+1]=='+'`），翻转后得到新字符串
    `nxt`，如果 `win(nxt)` 是 False（对手在新局面下必败），说明当前玩家选择这一
    步翻转就能赢，直接返回 True；如果试遍所有可翻转位置，对手在每种翻转后的局面
    下都必胜，那么当前局面就是必败态，返回 False。用字符串本身作为记忆化的 key
    （相同的字符串局面对应相同的胜负结果，与"是谁的回合"无关，因为每次都是"当前
    要行动的玩家"在问自己能不能赢）。
    【复杂度】时间最坏 O(2^n * n)（状态数最多是所有可能的 +/- 组合，记忆化让
    相同的字符串局面不会被重复搜索，但字符串本身作为 key 需要 O(n) 的哈希/比较
    开销）；空间 O(2^n * n)（记忆化表存储的字符串状态）。
    【易错点】1) 字符串是不可变对象，翻转生成新字符串时要用切片拼接
    `s[:i] + '--' + s[i+2:]`，不能尝试原地修改；2) 如果字符串里根本没有 "++"
    （比如全是 '-'，或者长度为 0/1），当前玩家没有任何操作可做，直接输掉，
    这种情况下循环体一次都不会进入 `if not win(nxt)` 分支，`result` 保持初始值
    False，天然正确，不需要额外特判；3) 记忆化的 key 用整个字符串本身，如果
    误用"这一步翻转的位置"之类的局部信息做 key，会把不同局面错误地合并。
    """
    memo: dict[str, bool] = {}

    def win(state: str) -> bool:
        if state in memo:
            return memo[state]
        result = False
        for i in range(len(state) - 1):
            if state[i] == "+" and state[i + 1] == "+":
                nxt = state[:i] + "--" + state[i + 2 :]
                if not win(nxt):
                    result = True
                    break
        memo[state] = result
        return result

    return win(s)


def get_money_amount(n: int) -> int:
    """
    【题意】LeetCode 375·Medium。系统在 1..n 里选定一个数字，你每次猜一个数字 x，
    猜错了要支付 x 元、并被告知目标数字比 x 大还是小，直到猜中为止。求"无论目标
    数字是多少，都能保证猜中"所需准备的最少金额（也就是最坏情况下的最小总花费，
    按最优策略进行）。
    【思路】这是"极小化极大(minimax)"的区间 DP：`dp[i][j]` 表示"目标数字确定在
    [i, j] 范围内时，为了保证猜中所需准备的最少金额"。枚举第一次猜的数字 pivot
    (i<=pivot<=j)：猜错后金额已经花掉 pivot 元，且会被告知目标在 pivot 左侧
    `[i, pivot-1]` 还是右侧 `[pivot+1, j]`——因为要"保证"猜中，必须按最坏情况
    准备，所以要素取"左右两个子区间里花费更大的那个"（`max(dp[i][pivot-1],
    dp[pivot+1][j])`），这一步的总花费是 `pivot + max(...)`。对每个区间，
    枚举所有可能的第一次猜测 pivot，取花费最小的那个——这是"当前决策者主动
    选择对自己最有利的第一步（min），但要为对手/环境可能造成的最坏结果做准备
    （max）"的经典 minimax 结构。
    【复杂度】时间 O(n^3)（区间数 O(n^2)，每个区间枚举 O(n) 个 pivot）；
    空间 O(n^2)（dp 表）。
    【易错点】1) 长度为 1 的区间（`i==j`，只剩一个候选数字）花费一定是 0——
    不用猜，直接确定就是它，dp 数组默认初始化为 0 恰好覆盖了这个基例，不需要
    单独判断"猜错"的情况；2) 计算 `max(dp[i][pivot-1], dp[pivot+1][j])` 时
    要注意 pivot 取到区间端点时其中一侧是空区间，空区间的花费视为 0（不能越界
    访问 dp 数组）；3) 容易把问题理解成"最小化期望花费"，但题目要求的是**保证
    猜中**的花费，也就是要覆盖最坏情况(极大化)，用错优化目标（比如直接取所有
    pivot 花费的最小值而不做 max 那一层）会得到偏低的错误答案。
    """
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for start in range(1, n - length + 2):
            end = start + length - 1
            best = float("inf")
            for pivot in range(start, end + 1):
                left = dp[start][pivot - 1] if pivot > start else 0
                right = dp[pivot + 1][end] if pivot < end else 0
                cost = pivot + max(left, right)
                best = min(best, cost)
            dp[start][end] = best
    return dp[1][n]


def stone_game(piles: list[int]) -> bool:
    """
    【题意】LeetCode 877·Medium。piles 是偶数堆石子，两人轮流从行首或行尾取走
    一整堆（先手先取），石子总数为奇数（不会平局），得到石子总数更多的人获胜。
    假设双方都绝顶聪明，判断先手(Alex)是否获胜。
    【思路】用区间 DP："当前玩家面对区间 [i, j] 时，能比对手多拿多少分"（分差，
    可正可负）定义 `dp[i][j]`。当前玩家只有两种选择——拿走最左边 `piles[i]`
    或最右边 `piles[j]`，拿走之后轮到对手面对剩下的区间，对手在剩下区间里也会
    按照同样的策略最大化"对手自己"相对当前玩家的分差，这个"对手的分差"从"当前
    玩家"的视角看就是要被减去的量。所以 `dp[i][j] = max(piles[i] - dp[i+1][j],
    piles[j] - dp[i][j-1])`——两种取法分别对应"这一步拿到的分" 减去 "对手在
    剩余区间里能占的优势"，取较大的那个（当前玩家会选对自己最有利的一种）。
    最终 `dp[0][n-1] > 0` 就表示先手能获得正的分差，也就是获胜。
    【复杂度】时间 O(n^2)（区间数 O(n^2)，每个区间转移 O(1)）；空间 O(n^2)。
    【易错点】1) `dp[i][j]` 存的是"分差"而不是"分数"，容易把递推写成
    单纯的"这一步能拿多少"而漏掉减去对手后续优势这一项；2) 长度为 1 的区间
    （`i==j`）基例是 `dp[i][i] = piles[i]`（只剩一堆，直接拿走），如果漏掉
    这个基例初始化，递推到长度 2 时会用到未初始化的值；3) 本题在"石子堆数为
    偶数、总数为奇数"的约束下，先手其实总是能获胜（可以证明先手总能选择只拿
    奇数下标或只拿偶数下标的所有堆，必然拿到更多），但这里仍然实现通用的区间
    DP，因为它能推广到约束被放宽（比如堆数为奇数）的变体，而"总是 True"的
    结论在约束改变后不再成立。
    """
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
    return dp[0][n - 1] > 0


def cat_and_mouse(graph: list[list[int]]) -> int:
    """
    【题意】LeetCode 913·Hard。无向图上老鼠从节点 1、猫从节点 2 出发，老鼠先走，
    双方交替沿一条边移动（猫不能走到节点 0 这个"洞"）。猫追上老鼠（同一节点）猫赢；
    老鼠跑到洞(节点0)老鼠赢；如果某个 (老鼠位置, 猫位置, 轮到谁走) 的局面重复出现，
    判定平局。返回 1(老鼠赢)、2(猫赢) 或 0(平局)。
    【思路】这道题不能像"必胜态/必败态"的简单博弈那样单纯往前搜索——因为存在
    "平局"这个第三种结局，前向递归 + 记忆化会在遇到环状依赖（A 依赖 B、B 又依赖
    A）时死循环或者需要复杂的"正在访问中"标记，而且平局的判定本身就来自于
    "无限循环而没有分出胜负"。标准解法是**逆向拓扑传播(retrograde analysis)**：
    从"已经确定胜负"的终局状态出发，反过来推导它们的"前驱状态"。终局状态是
    `(0, c, *)`（老鼠在洞里，无论轮到谁、猫在哪，老鼠必胜）和 `(c, c, *)`
    （猫和老鼠同位置，猫必胜，c != 0）。对一个已确定颜色（胜负）的状态
    `(m, c, turn)`，找到它的所有"前驱"——即上一步谁移动到这个状态：如果
    `turn == MOUSE_TURN`（说明上一步是猫在移动），前驱是 `(m, c', CAT_TURN)`
    使得猫能从 c' 走到 c；如果 `turn == CAT_TURN`，前驱是 `(m', c, MOUSE_TURN)`
    使得老鼠能从 m' 走到 m。传播规则：如果前驱状态"该走的那个玩家"恰好就是
    这个已确定状态的赢家，前驱可以直接染上同样的颜色（那个玩家可以主动选择走
    向这个必胜的后继状态）；否则说明这一步"该走的玩家"被迫走向一个对自己不利
    的状态，只有当**这个前驱状态的所有可能走法**都被证明对当前玩家不利时
    （用 `degree` 数组统计每个状态还有多少条"未被证明不利"的出边，减到 0），
    才能确定这个前驱状态的胜负（属于对手）。没有被染色的状态最终就是平局。
    【复杂度】时间 O(V^2)（状态数是 O(n^2)，边数/度数总和也是 O(n^2) 级别，
    整个 BFS 传播是线性于总的状态-边数量）；空间 O(n^2)。
    【易错点】1) 猫不能走到洞(节点 0)，统计猫方状态的 `degree`（可用走法数）
    时要把"走向节点 0"这条边排除掉，同时构造"猫方状态的前驱"时也要过滤掉
    "猫来自节点 0"这种不可能的情况；2) 传播时"谁的回合"和"这个状态归属于谁
    获胜"要严格对应：只有当前驱状态该走的玩家的回合恰好等于已确定后继状态的
    赢家时，才能直接传播颜色，判断条件写反会导致胜负关系被错误地传播；3) 最终
    答案要查询 `(1, 2, MOUSE_TURN)` 这个具体状态（老鼠在 1、猫在 2、老鼠先走），
    如果查询了错误的初始状态会得到不对应题目设定的结果；查询不到（没被染色）
    的状态就是平局，返回 0。
    """
    n = len(graph)
    DRAW, MOUSE_WIN, CAT_WIN = 0, 1, 2
    MOUSE_TURN, CAT_TURN = 0, 1

    color: dict[tuple[int, int, int], int] = {}
    degree: dict[tuple[int, int, int], int] = {}
    for m in range(n):
        for c in range(n):
            degree[(m, c, MOUSE_TURN)] = len(graph[m])
            degree[(m, c, CAT_TURN)] = len(graph[c]) - (1 if 0 in graph[c] else 0)

    queue: deque[tuple[int, int, int]] = deque()
    for c in range(n):
        for t in (MOUSE_TURN, CAT_TURN):
            color[(0, c, t)] = MOUSE_WIN
            queue.append((0, c, t))
            if c != 0:
                color[(c, c, t)] = CAT_WIN
                queue.append((c, c, t))

    def parents_of(m: int, c: int, t: int) -> list[tuple[int, int, int]]:
        res: list[tuple[int, int, int]] = []
        if t == MOUSE_TURN:
            # 当前局面轮到老鼠走，说明上一步是猫从 c_prev 走到了 c
            for c_prev in graph[c]:
                if c_prev != 0:
                    res.append((m, c_prev, CAT_TURN))
        else:
            # 当前局面轮到猫走，说明上一步是老鼠从 m_prev 走到了 m
            for m_prev in graph[m]:
                res.append((m_prev, c, MOUSE_TURN))
        return res

    while queue:
        m, c, t = queue.popleft()
        result = color[(m, c, t)]
        for pm, pc, pt in parents_of(m, c, t):
            if (pm, pc, pt) in color:
                continue
            if (pt == MOUSE_TURN and result == MOUSE_WIN) or (
                pt == CAT_TURN and result == CAT_WIN
            ):
                color[(pm, pc, pt)] = result
                queue.append((pm, pc, pt))
            else:
                degree[(pm, pc, pt)] -= 1
                if degree[(pm, pc, pt)] == 0:
                    color[(pm, pc, pt)] = CAT_WIN if pt == MOUSE_TURN else MOUSE_WIN
                    queue.append((pm, pc, pt))

    return color.get((1, 2, MOUSE_TURN), DRAW)


def predict_the_winner(nums: list[int]) -> bool:
    """
    【题意】LeetCode 486·Medium。两人轮流从数组两端取数字累加到自己分数上，先手
    先取。假设双方都绝顶聪明，判断先手的分数是否 >= 后手（分数相等也算先手赢）。
    【思路】和 LC877 石子游戏共享完全相同的区间 DP 结构——`dp[i][j]` 表示当前
    玩家面对区间 [i,j] 时能比对手多拿的分差，`dp[i][j] = max(nums[i] -
    dp[i+1][j], nums[j] - dp[i][j-1])`。区别只在最终判断：这里允许平局也算
    先手赢，所以是 `dp[0][n-1] >= 0`（而 LC877 因为总分数是奇数、不可能平局，
    严格用 `> 0`）。这说明"区间 DP 求最优分差"这个模型本身与具体的胜负判定
    规则是解耦的——模型算出通用的"分差"，最后一步再套上题目具体的胜负条件。
    【复杂度】时间 O(n^2)；空间 O(n^2)。
    【易错点】1) 边界判断是 `>=` 而不是 `>`——这道题允许平局判先手赢，与
    LC877 的判断条件不同，两题容易在这一步互相搞混；2) 同 LC877，基例
    `dp[i][i] = nums[i]` 必须先初始化再计算更长区间；3) `nums[i]` 允许是
    任意实数（本题约束是非负整数，但算法本身不依赖非负这个前提），如果照抄
    "先手必胜的充要条件是总和为奇数"这类只对 LC877 特定约束成立的结论，会在
    这道题上得出错误答案。
    """
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = nums[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0


def sum_game(num: str) -> bool:
    """
    【题意】LeetCode 1927·Medium。num 是长度为偶数、由数字和 '?' 组成的字符串，
    Alice 和 Bob 轮流把某个 '?' 替换成 0-9 的任意数字（Alice 先手），直到没有
    '?' 为止。如果最终前一半数字之和等于后一半数字之和，Bob 赢；否则 Alice 赢。
    双方都绝顶聪明，判断 Alice 是否获胜。
    【思路】这道题如果按"逐个 '?' 展开搜索"会有指数级的分支，但可以用纯数学分析
    直接得出结论，不需要显式博弈搜索。关键观察一：如果 '?' 的总数是奇数，Alice
    总能留到最后一步再下手——不管前面怎么变化，Alice 可以把最后一个 '?' 填成
    一个能打破平衡的数字，所以奇数个 '?' 时 Alice 必胜。关键观察二：如果 '?'
    数量是偶数，双方交替填，每一对回合里 Alice 想拉大两半的差距、Bob 想缩小
    差距——由于每个数字最大是 9，Alice 能达成的"最大可能偏差"和 Bob 能"抵消"的
    偏差在数学上正好抵消，胜负完全由**已确定数字之和的初始差**和**两侧 '?'
    个数的差**这两个量的固定关系决定：设 `diff = 前半已知和 - 后半已知和`，
    `cnt1, cnt2` 分别是前后半 '?' 的个数，Bob 能赢当且仅当
    `diff == 9 * (cnt2 - cnt1) / 2`（这是双方都用最优策略互相拉扯、最终能够
    到达的"必然结果"，本质上是一个可以严格证明的数学恒等式，不需要搜索）。
    【复杂度】时间 O(n)（扫一遍字符串统计每一半的已知数字和与 '?' 个数）；
    空间 O(1)。
    【易错点】1) '?' 总数的奇偶性判断必须放在最前面单独处理——奇数直接
    返回 True，不能跳过这一步直接套用"差值公式"（差值公式只在偶数个 '?'
    时成立）；2) `diff` 的定义方向（前半减后半还是后半减前半）必须和公式
    `9*(cnt2-cnt1)/2` 里 cnt1、cnt2 的角色保持一致，方向弄反会导致符号性
    错误；3) 这个结论依赖"每个数字的可填范围是 0-9"这个具体约束（9 是这个
    范围的极差），如果题目改成其他数字范围，系数 9 也要相应改变。
    """
    n = len(num)
    half = n // 2
    sum1 = cnt1 = 0
    sum2 = cnt2 = 0
    for i in range(half):
        if num[i] == "?":
            cnt1 += 1
        else:
            sum1 += int(num[i])
    for i in range(half, n):
        if num[i] == "?":
            cnt2 += 1
        else:
            sum2 += int(num[i])

    total_q = cnt1 + cnt2
    if total_q % 2 == 1:
        return True
    diff = sum1 - sum2
    return diff != 9 * (cnt2 - cnt1) // 2


def stone_game_iii(stone_value: list[int]) -> str:
    """
    【题意】LeetCode 1406·Hard。一排石子每堆有一个值(可能为负)，两人轮流从行首
    连续取 1、2 或 3 堆，得分是取走石子值之和。双方都绝顶聪明，返回 "Alice"、
    "Bob" 或 "Tie"（分数相等）。
    【思路】用"后缀分差" `dp[i]` 表示"从下标 i 开始（只剩 stone_value[i:]），
    当前玩家能比对手多拿多少分"。当前玩家有三种选择：拿 1、2 或 3 堆，拿完之后
    轮到对手面对剩下的后缀，对手在剩下后缀里的最优分差要从当前玩家视角"倒扣"掉——
    `dp[i] = max(取 j-i+1 堆的和 - dp[j+1])`，枚举 `j` 从 i 到 `min(i+2, n-1)`
    （最多连续取 3 堆）。这是"必胜态/必败态"思想在"带负数、比较分差而非纯粹
    胜负"场景下的推广——不再是布尔值的"能不能赢"，而是数值化的"最多能领先
    多少"，最后根据 `dp[0]` 的正负号/是否为零翻译回 "Alice"/"Bob"/"Tie"。
    【复杂度】时间 O(n)（每个 `dp[i]` 只需要枚举最多 3 种取法，均摊 O(1)）；
    空间 O(n)。
    【易错点】1) 石子值可以是负数，这意味着"贪心地尽量多拿"不一定是最优策略
    ——有时候当前玩家宁愿少拿甚至不拿某堆负值堆，把它甩给对手；DP 天然覆盖了
    这种情况，但如果试图用贪心简化会得出错误答案；2) 边界情况 `j` 不能超过
    `n-1`（数组末尾），`dp[n]`（越界之后的位置）要初始化为 0 作为递归基例；
    3) 最终结果的判断是 `dp[0] > 0` → Alice，`< 0` → Bob，`== 0` → Tie，
    不能想当然地认为"先手数值分差非负就算获胜"（这道题允许 Tie 单独作为一种
    结果，不是"分差 >= 0 就算 Alice 赢"）。
    """
    n = len(stone_value)
    dp = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        best = float("-inf")
        take = 0
        for j in range(i, min(i + 3, n)):
            take += stone_value[j]
            best = max(best, take - dp[j + 1])
        dp[i] = best

    if dp[0] > 0:
        return "Alice"
    if dp[0] < 0:
        return "Bob"
    return "Tie"


def stone_game_vii(stones: list[int]) -> int:
    """
    【题意】LeetCode 1690·Medium。一排石子，两人轮流从行首或行尾拿走一颗，拿走
    后获得"剩下所有石子值之和"的分数（不是拿走的那颗的值）。Alice 先手，目标是
    最大化"Alice 分数 - Bob 分数"，Bob 的目标是最小化这个差值（也就是让自己分
    更接近或超过 Alice）。假设双方都最优策略，返回最终的分差。
    【思路】和 LC877、LC486 是同一个区间 DP 家族，但"这一步能得到的分数"不是
    固定值（不是 `piles[i]` 或 `piles[j]` 本身），而是"剩下堆的总和"，需要先
    预处理前缀和 `prefix` 来 O(1) 求出任意区间的总和。`dp[i][j]` 表示面对
    区间 [i,j] 时，当前玩家能比对手多拿的分差：拿走最左边 `stones[i]` 能得到
    `total - stones[i]` 分（total 是区间 [i,j] 的总和，减去被拿走的那一颗，
    剩下的就是这一步的得分），之后轮到对手面对 [i+1,j]；拿最右边同理。所以
    `dp[i][j] = max(total - stones[i] - dp[i+1][j], total - stones[j] -
    dp[i][j-1])`——这一步得分减去对手在剩余区间里的最优分差。
    【复杂度】时间 O(n^2)（区间 DP，配合预处理的前缀和使得每个区间的"剩余总和"
    是 O(1) 查询）；空间 O(n^2)（dp 表）+ O(n)（前缀和）。
    【易错点】1) "这一步的得分"是**剩下堆的总和**，不是拿走的那一堆的值——
    正好和 LC877/LC486 反过来（那两题是"拿走的值就是这一步得分"），容易把
    转移公式里 `total - stones[i]` 误写成 `stones[i]` 本身；2) 前缀和的下标
    偏移要处理清楚，`total = prefix[j+1] - prefix[i]` 对应区间 [i,j]（闭区间）
    的和；3) 长度为 1 的区间(i==j)基例 `dp[i][i] = 0`（只剩一堆时，拿走它、
    剩下堆总和为 0，双方在这一步都不会再有后续分差），数组默认初始化为 0
    恰好覆盖这个基例，不需要额外处理。
    """
    n = len(stones)
    prefix = [0] * (n + 1)
    for i, v in enumerate(stones):
        prefix[i + 1] = prefix[i] + v

    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            total = prefix[j + 1] - prefix[i]
            dp[i][j] = max(
                total - stones[i] - dp[i + 1][j], total - stones[j] - dp[i][j - 1]
            )
    return dp[0][n - 1]


def _self_test() -> None:
    assert can_win_nim(1) is True
    assert can_win_nim(4) is False
    assert can_win_nim(7) is True
    assert can_win_nim(8) is False

    assert can_win_flip_game("++++") is True
    assert can_win_flip_game("+") is False
    assert can_win_flip_game("-") is False
    assert can_win_flip_game("++") is True

    assert get_money_amount(1) == 0
    assert get_money_amount(2) == 1
    assert get_money_amount(4) == 4

    assert stone_game([5, 3, 4, 5]) is True
    assert stone_game([3, 7, 2, 3]) is True

    assert cat_and_mouse([[2, 5], [3], [0, 4, 5], [1, 4, 5], [2, 3], [0, 2, 3]]) == 0
    assert cat_and_mouse([[1, 2], [0], [0]]) == 1  # 鼠 1->0 一步直达洞口，鼠必胜
    assert (
        cat_and_mouse([[2], [3], [0, 3], [1, 2]]) == 2
    )  # 鼠被迫走向 3，猫唯一可走的非洞边正好是 3，猫必胜

    assert predict_the_winner([1, 5, 2]) is False
    assert predict_the_winner([1, 5, 233, 7]) is True

    assert sum_game("5023") is False
    assert sum_game("25??") is True
    assert sum_game("?3295???") is False

    assert stone_game_iii([1, 2, 3, 7]) == "Bob"
    assert stone_game_iii([1, 2, 3, 6]) == "Tie"
    assert stone_game_iii([-1, -2, -3]) == "Tie"

    assert stone_game_vii([5, 3, 1, 4, 2]) == 6
    assert stone_game_vii([7, 90]) == 90

    print(
        "[PASS] p32_game_theory_combinatorics: 9/9 通过 "
        "(LC292/LC294/LC375/LC877/LC913/LC486/LC1927/LC1406/LC1690)"
    )


if __name__ == "__main__":
    _self_test()
