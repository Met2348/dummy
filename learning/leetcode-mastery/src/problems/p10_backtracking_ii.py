"""回溯专题 Part II（进阶补充）：子集II / 全排列II / 组合总和II / 组合总和III /
电话号码的字母组合 / 复原IP地址 / 分割回文串 / 解数独 / 目标和 / N皇后II /
删除无效的括号 / 找出不同的二进制字符串。

不重复讲 Part I 已经讲过的"做选择 -> 递归 -> 撤销选择"通用模板，本文件聚焦两条新
线索：1) 候选集合里出现重复值时，如何用"排序 + 同层跳过重复"避免生成内容相同的
答案（子集II/全排列II/组合总和II 三题共享同一个技巧，但要注意排列和组合/子集的
跳过条件写法不同）；2) 回溯的变体应用——用集合剪枝的数独填数、只统计数量不构造
具体解的 N 皇后计数、以及用 BFS 按删除次数分层实现"最少删除"的删除无效括号。
"""
from __future__ import annotations


# ── LC90 子集 II ─────────────────────────────────────────────────────────
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    """
    【题意】给定可能包含重复数字的整数数组 nums，返回它的所有不重复子集（幂集），
    结果中不允许出现两个内容相同（不计顺序）的子集。
    【思路】和 Part I 的 subsets 几乎同一套骨架（每个递归节点都是答案，下一层从
    start+1 开始防止重复选同一个下标），唯一新增的动作是"去重"：先把 nums 排序，
    让值相同的元素在数组里彼此相邻；在同一层 for 循环里，如果当前下标 i 满足
    i>start 且 nums[i]==nums[i-1]，说明这一层在更早的一次迭代里已经"选过一次这个
    值"了，此时再选一次会让产出的子集和之前产出的某个子集完全相同，直接 continue
    跳过整根同层分支。注意"跳过同层重复"必须配合排序：不排序就没法把"值相同"变成
    "下标相邻"，也就没法用一个简单的 i>start 判断同层是否已经选过同样的值。
    【复杂度】时间 O(n·2^n)（和不去重版本同阶，只是被剪掉的分支不需要再展开，常数
    更小）；空间 O(n) 递归栈 + path，不含结果。
    【易错点】1) 去重条件 nums[i]==nums[i-1] 前忘记先排序，重复元素在数组里不相邻，
    条件形同虚设，无法去重；2) 把 i>start 误写成 i>0——去重比较的是"同一层内是否
    重复"，不是"整个数组内是否重复"。比如 nums=[2,2]，选中第一个 2 之后本该允许
    递归进去再选第二个 2（这是纵向选择，不是同层重复），如果误用 i>0 会把这次合法
    选择也剪掉，漏掉子集 [2,2]。
    """
    nums = sorted(nums)
    res: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        res.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res


# ── LC47 全排列 II ───────────────────────────────────────────────────────
def permute_unique(nums: list[int]) -> list[list[int]]:
    """
    【题意】给定可能包含重复数字的数组 nums，返回所有不重复的全排列。
    【思路】排列要求"关心顺序、必须用光所有元素"，这一点和 Part I 的 permute 一致
    （叶子节点收集答案、用 visited 数组标记已用下标）。新增的去重逻辑同样是"排序 +
    同层跳过重复值"，但排列每层是"从头扫到尾选一个未用的下标"而不是"从 start 开始
    往后扫"，同层去重的判断条件也要跟着变：跳过条件是
    `nums[i]==nums[i-1] and not used[i-1]`——这里必须显式检查 `not used[i-1]`，
    因为排列问题里同一个值可能出现在路径的不同"深度"，不只是同一层：如果
    used[i-1] 为 True，说明 nums[i-1] 这个值已经被用在当前路径更靠前的位置（是
    "纵向"使用，不是"同层横向"重复选择），此时选 nums[i] 完全合法，不应该跳过；
    只有 used[i-1] 为 False（意味着 nums[i-1] 和当前 nums[i] 是"同一层"里两个下标
    不同但值相同的候选，前一个刚被撤销或还没被选到）时，才是真正需要剪掉的同层
    重复。
    【复杂度】时间最坏 O(n·n!)（重复元素越多，实际剪掉的分支越多，可能远小于此
    上界）；空间 O(n)（visited 数组 + 递归栈 + path）。
    【易错点】1) 沿用子集/组合去重的 "i>start" 条件——排列这里根本没有 start 概念，
    照抄会导致语义错误；2) 漏写 `not used[i-1]`，把"值相同但用在不同深度"的合法
    情况也剪掉，结果数量会少于应有的排列数；3) 忘记给 nums 排序，值相同的元素不
    相邻，去重判断直接失效。
    """
    nums = sorted(nums)
    res: list[list[int]] = []
    path: list[int] = []
    used = [False] * len(nums)

    def backtrack() -> None:
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return res


# ── LC40 组合总和 II ─────────────────────────────────────────────────────
def combination_sum2(candidates: list[int], target: int) -> list[list[int]]:
    """
    【题意】给定可能含重复数字的数组 candidates 和目标数 target，找出所有和为
    target 的组合；candidates 中每个位置的数字在一个组合里只能用一次（下标唯一，
    但值相同的多个下标可以各自出现在同一个组合里），且不允许出现两个内容相同的
    组合。
    【思路】和 Part I 的 combination_sum 结构几乎相同（remain 剪枝、remain==0
    收集），核心差异有两处：1) 因为每个下标只能用一次，下一层必须从 i+1 开始
    （不能像组合总和那样传 i 本身，那是"可重复选自己"的写法）；2) 因为数组本身
    可能有重复值，需要和子集 II 一样先排序再"同层跳过重复值"
    （`i>start and candidates[i]==candidates[i-1]`），防止两个下标不同但值相同的
    候选各自展开出一份内容相同的组合。排序后还能顺带做一个提前终止的剪枝：
    `candidates[i] > remain` 时直接 break（后面的候选只会更大，不可能再凑出解，
    没必要继续这层的 for 循环）。
    【复杂度】时间和空间量级与组合总和相同（指数级，取决于 target 和候选数分布），
    但去重剪枝和提前 break 在实际数据上通常显著更快。
    【易错点】1) 下一层写成 backtrack(i, ...)（组合总和的写法），会让同一个下标被
    无限重复选中；2) 漏掉同层去重条件，[1,1,6] 这种组合会因为两个 1 来自不同下标
    而被重复生成两次；3) 提前 break 的前提是数组已排序，如果忘记排序就加这个
    break，会错误地跳过后面本该被检查、实际更小的候选。
    """
    candidates = sorted(candidates)
    res: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            if candidates[i] > remain:
                break
            path.append(candidates[i])
            backtrack(i + 1, remain - candidates[i])
            path.pop()

    backtrack(0, target)
    return res


# ── LC216 组合总和 III ───────────────────────────────────────────────────
def combination_sum3(k: int, n: int) -> list[list[int]]:
    """
    【题意】找出所有相加之和为 n、且由 k 个 1-9 之间各不相同的数字组成的组合（每个
    数字最多用一次），不允许出现两个内容相同的组合。
    【思路】候选集合固定是 1~9 且天然没有重复值，所以不需要"同层去重"这一步——这
    正是它和上面两题的区别（放在这里对比，提醒"去重"不是回溯的必修课，只有候选集
    本身有重复值时才需要）。核心约束反而是"必须恰好选 k 个数"：终止条件从"remain
    ==0 就收集"变成"len(path)==k 时才检查 remain 是否恰好为 0"——凑够了 k 个数但
    remain 不为 0，或者 remain 提前到 0 但还没凑够 k 个数，都不是合法答案。下一层
    依然从 i+1 开始（每个数字只能用一次），并加上 `i > remain: break` 剪枝（候选
    天然升序排列，后面的数只会更大，不可能再凑出剩余的 remain）。
    【复杂度】时间 O(C(9,k))（候选池固定只有 9 个数，组合数远小于一般组合总和问题
    的指数级）；空间 O(k) 递归深度。
    【易错点】1) 把终止条件写成"remain==0 就收集"而不检查 len(path)==k，会把"不够
    k 个数但提前凑够 target"的不合法结果也收集进去；2) 反过来漏判——len(path)==k
    但 remain!=0 时不能收集，两个条件必须同时满足；3) 剪枝 `i > remain` 依赖候选
    严格递增遍历，这里 1-9 天然升序不需要额外排序，但如果误把候选顺序打乱，剪枝会
    错误地跳过还未检查、实际满足条件的候选。
    """
    res: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int, remain: int) -> None:
        if len(path) == k:
            if remain == 0:
                res.append(path[:])
            return
        for i in range(start, 10):
            if i > remain:
                break
            path.append(i)
            backtrack(i + 1, remain - i)
            path.pop()

    backtrack(1, n)
    return res


# ── LC17 电话号码的字母组合 ──────────────────────────────────────────────
def letter_combinations(digits: str) -> list[str]:
    """
    【题意】给定一个仅包含数字 2-9 的字符串 digits，按电话按键上的字母映射（比如 2
    对应 "abc"，7/9 各对应 4 个字母），返回它能表示的所有字母组合，顺序不作要求；
    如果 digits 是空字符串，返回空列表（不是 [""]）。
    【思路】本类里最"轻量"的回溯——没有去重、没有复杂剪枝，纯粹是"每一位有固定几
    种候选，依次选一种"的笛卡尔积展开：递归到 idx==len(digits) 说明每一位都选完
    了，把当前 path 拼成字符串收集；每一层的候选就是 mapping[digits[idx]] 这个
    字符串里的每个字母。放在这里对比的意义是提醒它和前面几道"排列组合去重"题的
    本质区别——这里每一位的候选集合互不相干（第 0 位选什么完全不影响第 1 位能选
    什么），不存在"同一层跳过重复"的问题。
    【复杂度】时间 O(3^m·4^n)（m 是对应 3 个字母的数字个数，n 是对应 4 个字母的
    数字个数如 7/9，这是所有组合数的上界）；空间 O(len(digits)) 递归深度。
    【易错点】1) 忘记特判 digits=="" 直接返回 []——如果不特判、直接跑
    backtrack(0)，因为 0==len("") 成立会在第一次调用就收集一个空字符串 ""，返回
    [""] 而不是题目要求的 []；2) 数字到字母的映射表写错（尤其 7 对应 "pqrs"、9
    对应 "wxyz" 各是 4 个字母，容易和其他数字一样按 3 个字母处理）。
    """
    if not digits:
        return []
    mapping = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz",
    }
    res: list[str] = []
    path: list[str] = []

    def backtrack(idx: int) -> None:
        if idx == len(digits):
            res.append("".join(path))
            return
        for ch in mapping[digits[idx]]:
            path.append(ch)
            backtrack(idx + 1)
            path.pop()

    backtrack(0)
    return res


# ── LC93 复原 IP 地址 ────────────────────────────────────────────────────
def restore_ip_addresses(s: str) -> list[str]:
    """
    【题意】给定只包含数字的字符串 s，返回所有能由它切割出的合法 IP 地址（切成 4
    段，每段是 0-255 之间的整数，且不能有前导 0，除非这一段本身就是单独一个
    "0"）。
    【思路】回溯"在字符串上切 4 刀"：每一层尝试把从 start 开始的接下来 1~3 个字符
    切成一段，切之前先用 valid() 检查这一段是否合法（长度大于 1 但以 '0' 开头直接
    判非法；否则要求数值在 0~255 之间）；当 path 已经攒够 4 段时，只有恰好把整个
    字符串用完（start==n）才收集为一个答案——4 段已切完但字符串还剩字符，或者
    字符串用完了但还不够 4 段，都不是合法结果。这里的关键剪枝是"提前检查每一段的
    合法性"，而不是切完 4 段之后再整体验证，能大幅减少无意义的递归分支。
    【复杂度】时间 O(1)（每段最多 3 种切法、最多 4 段，分支数是严格的常数上界
    3^4，和字符串长度无关，只有极少数分支能通过合法性检查）；空间 O(1) 递归深度
    （最多 4 层）。
    【易错点】1) 漏判"长度大于 1 且以 0 开头"这个前导零情况，会把 "012" 这种非法
    段当成合法；2) valid() 里如果不先判断长度再转 int，可能对 "256" 这种长度为 3
    但数值超界的段漏判（这里两个判断都做了：先前导零、再数值范围）；3) 收集答案
    的条件漏掉 start==n 这一半（只检查 len(path)==4），会把"提前凑够 4 段但字符串
    还有剩余字符没用上"的情况也当成合法答案。
    """
    res: list[str] = []
    path: list[str] = []
    n = len(s)

    def valid(seg: str) -> bool:
        if len(seg) > 1 and seg[0] == "0":
            return False
        return 0 <= int(seg) <= 255

    def backtrack(start: int) -> None:
        if len(path) == 4:
            if start == n:
                res.append(".".join(path))
            return
        for length in range(1, 4):
            if start + length > n:
                break
            seg = s[start : start + length]
            if not valid(seg):
                continue
            path.append(seg)
            backtrack(start + length)
            path.pop()

    backtrack(0)
    return res


# ── LC131 分割回文串 ─────────────────────────────────────────────────────
def partition_palindrome(s: str) -> list[list[str]]:
    """
    【题意】给定字符串 s，将其分割成若干子串，使得每个子串都是回文串，返回所有
    可能的分割方案。
    【思路】回溯"在字符串上找切割点"：每一层尝试把从 start 到某个 end 的子串切
    下来，只有这个子串本身是回文（is_pal 判断）才允许递归继续切剩下的部分；
    start==n 说明整个字符串都被成功切成了回文段，收集当前 path。和上面
    restore_ip_addresses 结构几乎一样（都是"在字符串上找切割点"的回溯），区别只在
    "每一段的合法性检查"——那题检查数值范围+前导零，这题检查回文。
    【复杂度】时间最坏 O(n·2^n)（最坏情况比如全部同一字符构成的字符串，任意切割
    点都合法，分割方案数是指数级；每种方案还要花 O(n) 拼接/拷贝）；空间 O(n) 递归
    深度。
    【易错点】1) 这里为了和课程风格保持一致选择了朴素的 sub==sub[::-1] 判断回文，
    当 s 很长时会有大量重复计算，可优化为提前用 O(n²) 动态规划算出所有子串是否
    回文的表——这是一个已知的可优化点，不是本实现的 bug；2) 切割区间弄反，比如
    把 s[start:end] 写成 s[start:end+1] 导致下标越界或多算一个字符。
    """
    res: list[list[str]] = []
    path: list[str] = []
    n = len(s)

    def is_pal(sub: str) -> bool:
        return sub == sub[::-1]

    def backtrack(start: int) -> None:
        if start == n:
            res.append(path[:])
            return
        for end in range(start + 1, n + 1):
            seg = s[start:end]
            if is_pal(seg):
                path.append(seg)
                backtrack(end)
                path.pop()

    backtrack(0)
    return res


# ── LC37 解数独 ──────────────────────────────────────────────────────────
def solve_sudoku(board: list[list[str]]) -> None:
    """
    【题意】给定一个部分填好的 9x9 数独棋盘（"." 表示空格），原地把它填成一个完整
    且合法的数独（每行、每列、每个 3x3 宫内 1-9 各出现一次），函数无返回值，直接
    修改传入的 board。
    【思路】这是"剪枝用集合、不用重新扫描"这一技巧最典型的应用（和 Part I 的 N
    皇后异曲同工，只是把"列/两条对角线"换成了"行/列/宫"三种约束）：预处理阶段先
    扫一遍棋盘，把每一行、每一列、每个 3x3 宫已经出现的数字分别记录进
    rows[r]/cols[c]/boxes[b] 三组集合（b=(r//3)*3+c//3 是"宫编号"的标准计算方式），
    同时收集所有空格坐标 empties；回溯时按 empties 的顺序逐个尝试填数，某个空格
    尝试填 val 前只需要查三个集合（各 O(1)）就知道合不合法，合法就把 val 加入三个
    集合、写入棋盘，递归填下一个空格；如果递归到最后返回 True 说明整个棋盘填完
    了，直接逐层返回 True 不再回溯；如果 1-9 都试过还是无法让后面的空格填满，才把
    val 从三个集合和棋盘里移除，尝试下一个候选数字。返回 True/False 表示"这一分支
    能否导致一个完整解"，一旦找到就要让这个 True 一路返回到最外层、不再继续尝试
    其他候选——这是"填数独只要一个解"和"N 皇后要收集所有解"的关键差异。
    【复杂度】时间理论上界很高（最坏情况下接近指数级，与空格数相关），但行/列/宫
    三重剪枝配合"一旦找到解就立刻停止搜索"通常让实际运行远小于暴力枚举；空间
    O(1)（三个集合最多各装 9 个数）+ O(空格数) 递归深度。
    【易错点】1) 每次判断合法性时重新扫描整行整列整宫，而不是维护并查询三个集合，
    在 board 接近填满时会有大量重复扫描，是本题最容易犯的"忘记维护增量状态"的
    错误；2) 找到解之后忘记让 backtrack 返回 True 一路"穿透"到最外层调用（比如
    错误地忽略返回值、继续把 for 循环跑完），会在已经填出完整解之后继续尝试其他
    数字，把已经正确的棋盘改错；3) 回溯失败时忘记把 val 从 rows/cols/boxes 三个
    集合里同时移除（比如只清了 board 上的字符，漏了集合），导致后续兄弟分支的
    合法性判断出错。
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empties: list[tuple[int, int]] = []

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == ".":
                empties.append((r, c))
            else:
                rows[r].add(val)
                cols[c].add(val)
                boxes[(r // 3) * 3 + c // 3].add(val)

    def backtrack(idx: int) -> bool:
        if idx == len(empties):
            return True
        r, c = empties[idx]
        b = (r // 3) * 3 + c // 3
        for val in "123456789":
            if val in rows[r] or val in cols[c] or val in boxes[b]:
                continue
            rows[r].add(val)
            cols[c].add(val)
            boxes[b].add(val)
            board[r][c] = val
            if backtrack(idx + 1):
                return True
            rows[r].remove(val)
            cols[c].remove(val)
            boxes[b].remove(val)
            board[r][c] = "."
        return False

    backtrack(0)


# ── LC494 目标和 ─────────────────────────────────────────────────────────
def find_target_sum_ways(nums: list[int], target: int) -> int:
    """
    【题意】给定一个非负整数数组 nums 和一个目标数 target，给每个数字前添加 '+'
    或 '-'，返回可以使最终表达式结果等于 target 的不同"添加符号方法"的数目。
    【思路】"回溯计数"而不是"回溯收集具体方案"的例子——每个数字只有两种候选（取
    +nums[idx] 或 -nums[idx]），递归到 idx==len(nums) 时判断当前累计和是否恰好
    等于 target，是就把计数器加一，不需要把具体的选择序列存下来。这是本题最朴素
    的解法（时间指数级），更快的做法是转换成"选一个子集取正号、其余取负号"的 0/1
    背包型 DP（时间 O(n·sum)），但既然本类是回溯专题的补充，这里选择用回溯写法
    呼应"每个元素做二选一决策"这一贯穿整个 backtracking 分类的主题。
    【复杂度】时间 O(2^n)（每个数字独立二选一，没有任何剪枝，是一棵满二叉递归
    树）；空间 O(n) 递归深度。
    【易错点】1) 误以为可以像组合总和那样提前剪枝"remain<0 就返回"——这里不行，
    因为后面的数字可以取负号，当前和超过 target 不代表这条分支已经不可能凑出最终
    解；2) 数组元素允许为 0，此时 +0 和 -0 是两种不同的"符号选择"但对结果毫无影响，
    容易被误认为需要去重（实际不需要，题目按"符号方案"计数，即使数值效果相同也
    算两种不同方案）；3) 忘记这是纯计数题、不需要 path 列表，多写一份无用的路径
    记录会浪费空间。
    """
    count = 0

    def backtrack(idx: int, cur_sum: int) -> None:
        nonlocal count
        if idx == len(nums):
            if cur_sum == target:
                count += 1
            return
        backtrack(idx + 1, cur_sum + nums[idx])
        backtrack(idx + 1, cur_sum - nums[idx])

    backtrack(0, 0)
    return count


# ── LC52 N 皇后 II ───────────────────────────────────────────────────────
def total_n_queens(n: int) -> int:
    """
    【题意】只要求返回 n 皇后问题"有多少种不同的合法摆法"这个数目，不需要像
    Part I 的 solve_n_queens 那样把每一种摆法转换成棋盘字符串返回。
    【思路】剪枝逻辑和 Part I 的 solve_n_queens 完全一样（cols/diag1/diag2 三个
    集合做 O(1) 冲突检测），唯一区别是收集答案的方式——不需要维护 queens 列表、
    不需要在终止时把解转换成字符串棋盘，只需要在 row==n 时把计数器 +1。放在这里
    对比"收集具体解"和"只统计解的数量"这两种回溯变体：当只关心数量时，可以省掉
    所有和"构造最终输出格式"相关的额外工作，让回溯本身更轻量。
    【复杂度】时间和空间量级与 solve_n_queens 完全一致（远小于 O(n^n) 的暴力
    枚举），但常数更小（没有构造棋盘字符串的开销）。
    【易错点】1) 误以为既然不需要具体棋盘，可以偷懒不维护 diag1/diag2 只判断
    列——这仍然会漏判对角线冲突，产生错误的计数（剪枝逻辑不能因为"只要数量"就
    简化）；2) 用 nonlocal count 时忘记声明，导致在内层函数里创建了一个新的局部
    变量，外层计数器实际没有被更新。
    """
    cols: set[int] = set()
    diag1: set[int] = set()  # r - c
    diag2: set[int] = set()  # r + c
    count = 0

    def backtrack(row: int) -> None:
        nonlocal count
        if row == n:
            count += 1
            return
        for c in range(n):
            if c in cols or (row - c) in diag1 or (row + c) in diag2:
                continue
            cols.add(c)
            diag1.add(row - c)
            diag2.add(row + c)
            backtrack(row + 1)
            cols.remove(c)
            diag1.remove(row - c)
            diag2.remove(row + c)

    backtrack(0)
    return count


# ── LC301 删除无效的括号 ─────────────────────────────────────────────────
def _is_balanced_parens(t: str) -> bool:
    """辅助函数：只检查字符串里的圆括号是否合法配对（其他字符一律忽略），供
    remove_invalid_parentheses 内部和自测共用。"""
    balance = 0
    for ch in t:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
            if balance < 0:
                return False
    return balance == 0


def remove_invalid_parentheses(s: str) -> list[str]:
    """
    【题意】给定一个可能包含多余括号的字符串 s（可能同时包含小写字母和圆括号），
    删除最少数量的无效括号，使得剩下的字符串是合法的括号字符串（只关心圆括号的
    合法性，其他字符原样保留、不参与合法性判断），返回所有可能的合法结果（不能有
    重复）。
    【思路】用 BFS 而不是"回溯+统计最少删除数"：本题要找"删除次数最少"的所有解，
    BFS 按"删除了几个字符"分层展开天然满足这个要求——第 0 层是原串本身，第 1 层是
    删除任意 1 个括号后的所有字符串，第 2 层是删除 2 个……只要某一层出现了合法的
    字符串，立刻停止扩展到下一层，因为再往下走删除数只会更多、不可能比当前层更
    优。每一层：对这一层里的每个字符串，枚举删除它的每一个括号字符（跳过普通
    字母，删字母不可能让括号更合法），把删除后的新字符串（如果没访问过）放进下
    一层候选集合，用 visited 去重防止同一个字符串被多条路径重复访问、重复扩展。
    合法性判断用独立的 _is_balanced_parens 辅助函数。
    【复杂度】时间最坏 O(2^n)（每一步都可能删或不删每个括号字符，层数最多是括号
    总数）；空间同量级（visited 集合 + 逐层候选集合）。
    【易错点】1) 用 DFS/回溯"删到某个位置就检查合法性"而不按删除次数分层，很难
    保证第一次找到的解就是删除最少的，需要额外维护一个全局最小删除数再过滤——
    BFS 天然按层保证"第一次出现合法解的那一层就是最优层"，这也是"删除次数最少"
    和"单词接龙求最短路径"在思路上的相似之处；2) 忘记去重（visited），同一个
    字符串会被多条不同的删除路径重复生成，既浪费时间又可能在结果里出现重复项；
    3) 枚举删除位置时忘记跳过字母字符（只处理 '(' 或 ')'），删除字母不会让括号
    更平衡，纯粹浪费搜索空间。
    """
    if _is_balanced_parens(s):
        return [s]

    level = {s}
    visited = {s}
    while level:
        valid_at_level = [t for t in level if _is_balanced_parens(t)]
        if valid_at_level:
            return valid_at_level
        next_level: set[str] = set()
        for t in level:
            for i in range(len(t)):
                if t[i] not in "()":
                    continue
                candidate = t[:i] + t[i + 1 :]
                if candidate not in visited:
                    visited.add(candidate)
                    next_level.add(candidate)
        level = next_level
    return [""]


# ── LC1980 找出不同的二进制字符串 ────────────────────────────────────────
def find_different_binary_string(nums: list[str]) -> str:
    """
    【题意】给定 n 个长度均为 n 的、各不相同的二进制字符串组成的数组 nums，返回
    任意一个长度为 n、且不在 nums 中出现过的二进制字符串（题目保证一定存在这样的
    字符串，因为长度为 n 的二进制字符串共有 2^n 个，严格多于 nums 的 n 个）。
    【思路】用回溯逐位构造：每一位尝试放 '0' 或 '1'，递归到凑够 n 位时检查这个
    候选字符串是否在 nums 构成的哈希集合里出现过，没出现过就是一个合法答案、直接
    层层返回给最外层调用方；这个写法呼应本类"每一位做二选一决策"的主题（和
    find_target_sum_ways 的"每个数选 + 或 -"是同一种回溯形状）。这不是最优写法——
    更快的经典解法是"对角线法"：直接令 result[i] 和 nums[i][i] 不同，这样 result
    与 nums 中第 i 个字符串至少在第 i 位上不同，从而保证 result 不等于 nums 中的
    任何一个，只需要一次 O(n) 遍历、不需要搜索；这里选择回溯写法是为了呼应本类
    主题，复杂度取舍见下方。
    【复杂度】回溯写法最坏 O(n·2^n)（找到答案前可能要尝试多个长度为 n 的候选，
    每个候选还要花 O(n) 判断是否在 set 里/拼接）；对角线法是 O(n)，作为对比写在
    这里，实际工程中更推荐对角线法。
    【易错点】1) 忘记先把 nums 转成 set，导致"候选是否已存在"退化成对列表的线性
    扫描，回溯的每一层都变慢；2) 误以为 n 很大时这个回溯写法也能稳定跑得很快——
    最坏情况下（比如 nums 覆盖了字典序前面几乎所有字符串）可能要尝试接近 2^n 个
    候选，这时应该换成对角线法而不是硬跑回溯；3) 忘记题目保证解一定存在，如果
    调用场景变化导致找不到解，回溯函数会返回 None，调用方直接对返回值做字符串
    操作会抛异常。
    """
    n = len(nums)
    seen = set(nums)
    path: list[str] = []

    def backtrack() -> str | None:
        if len(path) == n:
            candidate = "".join(path)
            return candidate if candidate not in seen else None
        for bit in "01":
            path.append(bit)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
        return None

    return backtrack()


def _self_test() -> None:
    assert {tuple(sorted(x)) for x in subsets_with_dup([1, 2, 2])} == {
        (),
        (1,),
        (1, 2),
        (1, 2, 2),
        (2,),
        (2, 2),
    }

    assert sorted(map(tuple, permute_unique([1, 1, 2]))) == sorted(
        [(1, 1, 2), (1, 2, 1), (2, 1, 1)]
    )

    assert sorted(map(tuple, combination_sum2([10, 1, 2, 7, 6, 1, 5], 8))) == sorted(
        [(1, 1, 6), (1, 2, 5), (1, 7), (2, 6)]
    )

    assert combination_sum3(3, 7) == [[1, 2, 4]]
    assert sorted(combination_sum3(3, 9)) == sorted([[1, 2, 6], [1, 3, 5], [2, 3, 4]])

    assert sorted(letter_combinations("23")) == sorted(
        ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
    )
    assert letter_combinations("") == []

    assert sorted(restore_ip_addresses("25525511135")) == sorted(
        ["255.255.11.135", "255.255.111.35"]
    )
    assert restore_ip_addresses("0000") == ["0.0.0.0"]

    assert sorted(map(tuple, partition_palindrome("aab"))) == sorted(
        [("a", "a", "b"), ("aa", "b")]
    )

    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    original = [row[:] for row in board]
    solve_sudoku(board)
    for r in range(9):
        for c in range(9):
            if original[r][c] != ".":
                assert board[r][c] == original[r][c]
    for r in range(9):
        assert sorted(board[r]) == list("123456789")
    for c in range(9):
        col = [board[r][c] for r in range(9)]
        assert sorted(col) == list("123456789")
    for br in range(3):
        for bc in range(3):
            cell = [board[br * 3 + dr][bc * 3 + dc] for dr in range(3) for dc in range(3)]
            assert sorted(cell) == list("123456789")

    assert find_target_sum_ways([1, 1, 1, 1, 1], 3) == 5
    assert find_target_sum_ways([1], 1) == 1

    assert total_n_queens(4) == 2
    assert total_n_queens(1) == 1

    results = remove_invalid_parentheses("()())()")
    assert all(_is_balanced_parens(r) for r in results)
    assert len({len(r) for r in results}) == 1
    assert len(set(results)) >= 2

    result = find_different_binary_string(["01", "10"])
    assert len(result) == 2 and result not in ["01", "10"]

    print(
        "[PASS] p10_backtracking_ii: 12 道回溯进阶题"
        "（子集II/全排列II/组合总和II/组合总和III/电话号码的字母组合/复原IP地址/"
        "分割回文串/解数独/目标和/N皇后II/删除无效的括号/找出不同的二进制字符串）"
        "全部通过"
    )


if __name__ == "__main__":
    _self_test()
