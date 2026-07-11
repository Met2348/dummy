"""回溯专题：子集 / 全排列 / 组合总和 / 括号生成 / 单词搜索 / N 皇后。

统一模板：做选择 -> 递归 -> 撤销选择；六道题的差异只在"做选择"的方式（选哪些候选、
能不能重复选）和"剪枝条件"（什么情况下这条分支注定无效，可以提前放弃）。
"""
from __future__ import annotations

import itertools


# ── LC78 子集 ────────────────────────────────────────────────────────────
def subsets(nums: list[int]) -> list[list[int]]:
    """
    【题意】给定不包含重复元素的整数数组 nums，返回它的所有子集（幂集），子集顺序、
    子集内部元素顺序均不作要求。
    【思路】每个元素只有"选"或"不选"两种命运，用递归模拟"依次决定每个元素选不选"这一
    过程。把 res.append(path[:]) 放在递归入口而不是递归树的叶子处，是因为子集问题里
    "递归树上的每一个节点"（不只是叶子）都是一个合法答案——子集长度可以是 0~n 之间任意
    值，不像后面全排列那样必须凑满 n 个元素才算数。递归时下一层从 start+1 开始（不能
    从 0 重新扫），保证同一个元素在一个子集里只被考虑一次、且不会因为顺序不同产生重复
    子集（比如 [1,2] 和 [2,1] 被当成同一个子集，不会被生成两次）。
    【复杂度】时间 O(n*2^n)（2^n 个子集，每个平均长度 O(n) 需要拷贝）；空间 O(n) 递归
    栈深度 + path。
    【易错点】1) 忘记 path[:] 拷贝，直接 res.append(path) 会让所有结果引用同一个列表
    对象，最终被后续的 append/pop 全部改空；2) start 参数忘记 +1（误写成一直传 0），
    会导致同一个元素在同一个子集里被重复选中，或产生大量重复子集。
    """
    res: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res


# ── LC46 全排列 ──────────────────────────────────────────────────────────
def permute(nums: list[int]) -> list[list[int]]:
    """
    【题意】给定不含重复数字的数组 nums，返回其所有全排列，每个排列必须用光全部 n 个
    元素，顺序不同视为不同排列。
    【思路】和子集不同，排列要求"用光所有元素且关心顺序"，所以答案只在递归深度达到 n
    时才产生（只在叶子节点收集，而不是像子集那样每个节点都收集）；每一层要从头到尾
    遍历所有"还没被用过"的元素（而不是像子集/组合那样从 start 下标开始），因为
    [1,2] 和 [2,1] 是两个不同的合法答案，不能靠下标去重。用一个 visited 数组标记
    "谁已经出现在当前路径里"，是最直观的"防止同一个下标被用两次"的办法。
    【复杂度】时间 O(n*n!)（n! 个排列，每个排列需要 O(n) 构造/拷贝）；空间 O(n)
    （visited 数组 + 递归栈 + path）。
    【易错点】1) 照搬子集的写法在这里也写"下一层从 start 开始"，会漏掉大量排列
    （比如永远选不出 [2,1] 这种"后面的数排在前面"的结果）；2) 忘记回溯
    visited[i]=False，导致某个下标在其他分支里被错误地认为"已用过"。
    """
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
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return res


# ── LC39 组合总和 ────────────────────────────────────────────────────────
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """
    【题意】给定无重复元素的正整数数组 candidates 和目标数 target，找出所有使数字和
    等于 target 的组合；candidates 中的同一个数字可以被无限次重复选取，但不允许出现
    两个内容相同（不计顺序）的组合。
    【思路】核心 insight：因为同一个数字可以重复选，所以"下一层递归从哪个下标开始"不能
    像子集/排列那样从 start+1（避免重复选同一个下标），而要从 start 本身开始——允许
    下一层再次选中当前这个数。同时递归下标只允许往后走、不能回头选更靠前的
    candidates[i-1]，这一条保证了"2+2+3"只会被生成一次，而不会因为选数顺序不同又
    生成一次"3+2+2"这种重复组合。用 remain（还差多少凑够 target）代替每次重新
    求和：remain<0 直接剪枝放弃，remain==0 说明凑够了、收集当前 path 为一个答案。
    【复杂度】时间最坏情况是指数级（回溯树大小取决于 target 和候选数的最小值，
    约 O(target 的相关幂次)）；空间 O(target/min(candidates)) 递归深度。
    【易错点】1) 把递归起始下标写成 i+1（那是"每个数字最多用一次"的组合总和 II 的
    做法），会漏掉 [2,2,3] 这种重复使用同一个数的解；2) 不加 start 下限、每层都从 0
    开始遍历，会把 [2,2,3] 和 [2,3,2] 当成两个不同答案，产生大量重复组合。
    """
    res: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path[:])
            return
        if remain < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, remain - candidates[i])  # 同一个下标可以重复选，所以传 i 不是 i+1
            path.pop()

    backtrack(0, target)
    return res


# ── LC22 括号生成 ────────────────────────────────────────────────────────
def generate_parenthesis(n: int) -> list[str]:
    """
    【题意】给定 n 对括号，生成所有由 n 对括号组成的、且合法（每个左括号都有匹配的右
    括号，且任意前缀里左括号数量不少于右括号数量）的括号组合。
    【思路】不用先生成所有长度为 2n 的括号串再逐一检查合法性，而是在回溯的每一步就用
    "剩余可放的左括号数 left、剩余可放的右括号数 right"两个计数器直接剪掉非法分支：
    只要 left>0 就可以放一个左括号（left 减一）；只要 right>left（当前剩余右括号数
    严格多于剩余左括号数，意味着目前已放的左括号数多于已放的右括号数）才允许放一个
    右括号（right 减一）。这样生成过程中每一步都自动维持"任意前缀左>=右"这条合法性
    约束，完全不需要生成完之后再过滤。
    【复杂度】时间 O(4^n / sqrt(n))（卡特兰数量级，即合法括号串本身的个数量级）；
    空间同量级用于存结果，另加 O(n) 递归深度。
    【易错点】1) 把"能不能放右括号"的条件写反或写成只看总数（比如写成 right<n 而不是
    right>left），会生成 ")(" 这种非法前缀；2) 如果改用可变的 list 拼接字符（而不是
    这里用不可变字符串直接传参），必须记得在回溯返回前 pop 撤销，否则各分支会共享
    同一个可变对象、结果相互污染。
    """
    res: list[str] = []

    def backtrack(cur: str, left: int, right: int) -> None:
        if len(cur) == 2 * n:
            res.append(cur)
            return
        if left > 0:
            backtrack(cur + "(", left - 1, right)
        if right > left:
            backtrack(cur + ")", left, right - 1)

    backtrack("", n, n)
    return res


# ── LC79 单词搜索 ────────────────────────────────────────────────────────
def exist(board: list[list[str]], word: str) -> bool:
    """
    【题意】给定 m*n 的字符网格 board 和字符串 word，判断 word 是否可以从网格中某个
    格子出发，沿上下左右相邻方向逐字符连接拼出来（同一个格子在同一条路径中不能被
    使用两次）。
    【思路】以网格中每个格子为起点尝试 DFS：如果当前格子字符等于 word 当前要匹配的
    字符，就把它临时标记为"已访问"（这里直接把 board[r][c] 改写成占位符 "#"，比额外
    开一个 visited 集合更省内存，也天然利用了"这一格在当前调用栈里已被占用"这一
    事实），然后递归匹配 word 的下一个字符；四个方向都失败就回溯——把刚才改掉的字符
    复原（这是本题"做选择 -> 递归 -> 撤销选择"三步里最容易被漏掉的"撤销"环节），
    换其他起点或路径继续尝试。
    【复杂度】时间最坏 O(m*n*4^L)（L=len(word)，每一步最多 4 个方向可走，起点最多
    m*n 个）；空间 O(L) 递归深度（原地修改 board，不需要额外的 O(m*n) visited）。
    【易错点】1) 忘记在函数返回前把 board[r][c] 复原回原字符，导致其他起点的搜索
    里这一格"凭空消失"，产生错误的 False；2) 边界越界检查和字符是否匹配的检查顺序
    写反，会先访问越界下标抛出异常。
    """
    if not board or not board[0]:
        return False
    rows, cols = len(board), len(board[0])

    def backtrack(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[idx]:
            return False
        saved = board[r][c]
        board[r][c] = "#"
        found = (
            backtrack(r + 1, c, idx + 1)
            or backtrack(r - 1, c, idx + 1)
            or backtrack(r, c + 1, idx + 1)
            or backtrack(r, c - 1, idx + 1)
        )
        board[r][c] = saved
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


# ── LC51 N 皇后 ──────────────────────────────────────────────────────────
def solve_n_queens(n: int) -> list[list[str]]:
    """
    【题意】在 n*n 的棋盘上放置 n 个皇后，使得任意两个皇后都不在同一行、同一列、同一条
    对角线上；返回所有合法摆法，每个摆法用长度为 n 的字符串列表表示（'Q' 表示皇后，
    '.' 表示空位）。
    【思路】按行回溯：因为每行必须恰好放一个皇后，天然保证了"不同行"，所以只需要在
    每一行里选一个"当前合法"的列。合法性判断的关键优化：不要每次放置前重新扫描整个
    棋盘去检查同列/同对角线，而是维护三个集合，一边放置一边更新——cols：已用的列
    下标；diag1：已用的"主对角线"标识 r-c（同一条 \\ 对角线上所有格子的 r-c 是同一个
    常数）；diag2：已用的"副对角线"标识 r+c（同一条 / 对角线上所有格子的 r+c 是同一个
    常数）。这样每一步的合法性判断从 O(n) 扫描降到 O(1) 集合查询。内部先用一个"每行
    皇后所在列下标"的整数列表 queens 表示当前解，找到一组完整解后再统一转换成棋盘
    字符串，避免在回溯过程中频繁拼接字符串。
    【复杂度】时间理论上界 O(n!)，实际因为三重剪枝远小于暴力枚举的 O(n^n)；空间
    O(n) 递归深度 + 三个集合（各最多 O(n) 大小）。
    【易错点】1) 只判断同列同行、忘记两条对角线，会漏判很多实际非法的摆法；
    2) 把 r-c 和 r+c 两条对角线标识搞混（比如都用 r+c），会导致本该独立的两条对角线
    被当成同一条判断，漏掉真正的对角线冲突；3) 回溯返回前忘记把 cols/diag1/diag2
    里当前这一步加入的值移除，导致后续兄弟分支被这一步"已撤销"的选择错误剪枝。
    """
    res: list[list[str]] = []
    cols: set[int] = set()
    diag1: set[int] = set()  # r - c
    diag2: set[int] = set()  # r + c
    queens = [-1] * n  # queens[row] = 该行皇后所在的列

    def backtrack(row: int) -> None:
        if row == n:
            board = []
            for r in range(n):
                line = ["."] * n
                line[queens[r]] = "Q"
                board.append("".join(line))
            res.append(board)
            return
        for c in range(n):
            if c in cols or (row - c) in diag1 or (row + c) in diag2:
                continue
            cols.add(c)
            diag1.add(row - c)
            diag2.add(row + c)
            queens[row] = c
            backtrack(row + 1)
            cols.remove(c)
            diag1.remove(row - c)
            diag2.remove(row + c)
            queens[row] = -1

    backtrack(0)
    return res


def _self_test() -> None:
    all_subsets = subsets([1, 2, 3])
    assert len(all_subsets) == 8
    assert set(map(tuple, map(sorted, all_subsets))) == set(
        map(tuple, map(sorted, [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]))
    )

    all_perms = permute([1, 2, 3])
    assert len(all_perms) == 6
    assert sorted(map(tuple, all_perms)) == sorted(itertools.permutations([1, 2, 3]))

    def norm(lists: list[list[int]]) -> list[tuple[int, ...]]:
        return sorted(tuple(sorted(x)) for x in lists)

    assert norm(combination_sum([2, 3, 6, 7], 7)) == norm([[2, 2, 3], [7]])
    assert norm(combination_sum([2, 3, 5], 8)) == norm([[2, 2, 2, 2], [2, 3, 3], [3, 5]])

    assert len(generate_parenthesis(3)) == 5
    assert set(generate_parenthesis(3)) == {"((()))", "(()())", "(())()", "()(())", "()()()"}

    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    assert exist(board, "ABCCED") is True
    assert exist(board, "SEE") is True
    assert exist(board, "ABCB") is False

    assert len(solve_n_queens(4)) == 2
    assert len(solve_n_queens(1)) == 1

    print(
        "[PASS] p10_backtracking: 6 道回溯题"
        "（子集/全排列/组合总和/括号生成/单词搜索/N皇后）全部通过"
    )


if __name__ == "__main__":
    _self_test()
