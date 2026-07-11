"""分类 23（Phase 3 竞赛级补充）：矩阵与模拟进阶 —— 螺旋生长、对角线分组、
列重排贪心、稀疏矩阵压缩乘法、多线索胜负判定，在基础篇"边界收缩"套路之上
再加一层下标推导和数据结构选择的复杂度。"""
from __future__ import annotations


def spiral_matrix_iii(rows: int, cols: int, r_start: int, c_start: int) -> list[list[int]]:
    """
    【题意】在 rows x cols 的网格上，从坐标 (r_start, c_start) 出发，起始朝向东（右），
    按顺时针螺旋的方式行走。每当移动到网格边界之外时，仍然要在网格外继续按规则走
    （不计入结果），之后可能会再绕回网格内部。要求返回访问到网格内每个坐标的顺序
    （一共 rows*cols 个坐标）。
    【思路】这不是"遍历一个已经填满的矩阵"（如 54/59 题那样靠 top/bottom/left/right
    四条边界收缩），而是"以给定起点向外扩张的螺旋"：边界不是矩形的四条边，而是不断
    增长的步长。观察规律：从起点出发先向东走 1 步，再向南走 1 步，再向西走 2 步，
    再向北走 2 步，再向东走 3 步，再向南走 3 步……即方向按"东→南→西→北"循环，
    每走完两个方向（一次"东+南"或"西+北"）步长才加一。按这个规律不断前进，每走
    到一个新坐标就判断它是否落在 `[0,rows) x [0,cols)` 范围内——在范围内才加入结果，
    不在范围内的坐标仍然要继续前进（因为后续可能绕回来），直到收集满 rows*cols 个
    坐标为止。这个"故意在网格外空走"的做法看似浪费，但保证了逻辑的一致性：不需要
    对"起点靠近边缘、螺旋提前撞墙"这种情况写任何特判分支。
    【复杂度】时间 O(max(rows, cols)^2)（螺旋边长增长到能覆盖整个网格所需的步数级别
    是 O(max(rows,cols))，故总共走过的格子数——含网格外空走的部分——是这个量级的
    平方；网格内的 rows*cols 个坐标只是其中的一个子集）；空间 O(rows*cols)（返回的
    结果列表本身）。
    【易错点】1) 步长递增的时机是"每两个方向切换一次"，不是"每次方向切换都加一"——
    错误地每次切换方向就 +1 步长，会让螺旋"长"得过快，提前跳过还没访问到的格子；
    2) 判断坐标是否落在网格内必须在**每一步**都做（而不是等一整条边走完再检查终点），
    因为一条边中间的某些格子可能在界内、两端在界外（起点靠边时常见）；3) 循环的
    终止条件必须用"已收集到的坐标数是否等于 rows*cols"，不能用步数或者边界作为
    停止条件——步长本身是无界增长的，只有"收集满了"这个条件是保证会成立、且必须
    成立的终止信号。
    """
    total = rows * cols
    result: list[list[int]] = []
    r, c = r_start, c_start
    if 0 <= r < rows and 0 <= c < cols:
        result.append([r, c])
    # 方向顺序：东、南、西、北，循环切换
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    d = 0
    step = 1
    while len(result) < total:
        for _ in range(2):  # 每两个方向共用同一个步长，走完这两个方向步长才 +1
            dr, dc = directions[d]
            for _ in range(step):
                r += dr
                c += dc
                if 0 <= r < rows and 0 <= c < cols:
                    result.append([r, c])
            d = (d + 1) % 4
        step += 1
    return result


def diagonal_sort(mat: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定 m x n 矩阵 mat。"矩阵对角线"是从矩阵最上面一行或最左侧一列的某个
    元素开始、沿右下方向一直到矩阵边界的所有元素。要求把每一条对角线上的元素都
    按升序排序，返回排序后的矩阵。
    【思路】关键观察：同一条对角线上的元素，行号减列号 `r - c` 的值是常数（沿右下
    方向移动时 r 和 c 同时 +1，差值不变）。于是分两步：第一步按 `r - c` 分组，把
    每条对角线上的元素依次收集到一个列表里（按行优先遍历矩阵自然会按对角线从左上
    到右下的顺序把元素追加进对应分组）；第二步把每条对角线的分组列表按**降序**
    排序，写回矩阵时再按行优先遍历一遍，每次用 `list.pop()`（弹出末尾、也就是当前
    分组里剩下的最小值）填回对应位置——降序排列 + 从末尾弹出，效果正好等价于"先进
    先出"地按升序依次取值，不需要额外引入双端队列。
    【复杂度】时间 O(m*n*log(min(m,n)))（每条对角线长度不超过 min(m,n)，一共
    m+n-1 条对角线，排序是主要开销）；空间 O(m*n)（哈希表存下所有对角线分组）。
    【易错点】1) 分组的 key 必须是 `r - c`，不能写成 `r + c`——`r + c` 相同的是
    "反对角线"（方向是左下到右上），题目要求的方向是"从左上到右下"，两者是完全
    不同的分组；2) 写回时必须保证"先被消费的是最小值"，如果对分组列表按**升序**
    排序又用 `pop()`（从末尾弹出，此时弹出的是最大值），写回顺序会整体颠倒；本文件
    选择"降序排序 + pop 弹出末尾（即最小值）"这个组合，如果改成升序排序就必须换成
    `pop(0)` 或者用双端队列的 `popleft()`，不能不做改动地混用。
    """
    rows, cols = len(mat), len(mat[0])
    diagonals: dict[int, list[int]] = {}
    for r in range(rows):
        for c in range(cols):
            diagonals.setdefault(r - c, []).append(mat[r][c])
    for values in diagonals.values():
        values.sort(reverse=True)
    for r in range(rows):
        for c in range(cols):
            mat[r][c] = diagonals[r - c].pop()
    return mat


def largest_submatrix(matrix: list[list[int]]) -> int:
    """
    【题意】给定 m x n 的 0/1 矩阵 matrix，可以将矩阵的**列**按任意顺序重新排列
    （行的相对顺序不能变）。求重新排列之后，能构成的全 1 子矩阵的最大面积。
    【思路】因为只能整列整列地移动，某一列在某一行"值不值得选"完全取决于"从这一行
    往上数，这一列连续为 1 的高度"——如果两列在同一行的这个"向上连续 1 高度"相同，
    把它们相邻放置就能拼出一个更宽的全 1 矩形，谁排在哪个具体列号完全不重要。
    具体分两步：第一步预处理出"高度矩阵"：把原矩阵里每个 1 替换成"从当前行往上数、
    连续为 1 的格子数"（`matrix[i][j] = matrix[i-1][j] + 1`，第 0 行保持不变），这样
    `matrix[i][j]` 就表示"以第 i 行为底边，第 j 列这根柱子能向上长多高"。第二步，
    因为列可以任意重排，对高度矩阵的**每一行单独降序排序**——排序后前 k 个位置里最
    小的高度就是排序后下标为 k-1 的那个值（因为已经降序），于是"以当前行为底边、
    宽度为 k"的最大矩形面积就是 `第k个位置的高度 * k`；枚举每一行、每一个前缀长度 k，
    取所有这些乘积里的最大值就是答案。
    【复杂度】时间 O(m*n*log n)（每一行排序 O(n log n)，共 m 行）；空间 O(m*n)
    （为了不破坏输入矩阵，这里复制了一份高度矩阵；如果允许原地修改传入的 matrix，
    可以做到额外空间 O(1)）。
    【易错点】1) "连续 1 高度"的递推方向必须是从上到下（`matrix[i]` 依赖
    `matrix[i-1]`），不能写反成从下往上；2) 排序后计算面积时，宽度和高度值的下标
    必须对应正确——排序后（降序）下标为 `j`（从 0 开始）的元素，代表"前 j+1 个位置
    里最小的高度"，对应宽度是 `j+1` 而不是 `j`，漏掉 +1 会让面积算少一整列；3) 全 0
    矩阵或某一列全是 0 的情况，高度矩阵对应位置会一直是 0，乘出来的面积自然是 0，
    不需要也不应该额外加特判分支。
    """
    rows, cols = len(matrix), len(matrix[0])
    heights = [row[:] for row in matrix]
    for r in range(1, rows):
        for c in range(cols):
            if heights[r][c]:
                heights[r][c] = heights[r - 1][c] + 1
    best = 0
    for row in heights:
        sorted_row = sorted(row, reverse=True)
        for width, height in enumerate(sorted_row, start=1):
            best = max(best, width * height)
    return best


def sparse_matrix_multiply(mat1: list[list[int]], mat2: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定两个稀疏矩阵 mat1（m x k）和 mat2（k x n，保证可以相乘），返回它们
    的乘积矩阵。"稀疏矩阵"指矩阵中绝大多数元素是 0。
    【思路】朴素的三重循环矩阵乘法是 O(m*k*n)，对稀疏矩阵而言，这里面绝大多数运算
    都是在算"0 乘任何数"这种毫无意义的浪费。既然大部分元素是 0，更高效的做法是先
    把两个矩阵都压缩成"每一行只记录非零元素的 (列号, 值)"这样的稀疏表示；做乘法时，
    枚举 mat1 第 i 行的每个非零元素 `(k, val1)`，再枚举 mat2 第 k 行的每个非零元素
    `(j, val2)`，把 `val1 * val2` 累加到结果矩阵的 `result[i][j]` 上——只要 mat1 或
    mat2 里有一个是 0，这次运算天然被压缩表示"跳过"了，不需要显式的 if 判断。这样
    总运算次数正比于"两个矩阵各自非零元素规模的乘积"，在真正稀疏的输入下远小于
    朴素算法的 m*k*n。
    【复杂度】时间 O(m*k + 实际枚举到的非零元素对的数量)——最坏情况下（矩阵其实
    并不稀疏）会退化到 O(m*k*n)，但真正稀疏时远小于此；空间 O(非零元素个数)（两份
    压缩表示的存储开销）+ O(m*n)（结果矩阵——乘积不一定继续稀疏，通常仍需要用稠密
    数组存储结果）。
    【易错点】1) 压缩表示要按"行号"对应到列表下标，即使某一行完全没有非零元素，
    也要保留这一行对应一个空列表占位，不能因为这一行是空的就整体跳过导致后续按
    行号索引时出现下标错位；2) mat1 第 i 行非零元素里的"列号"要和 mat2 对应"行号"
    的非零元素对齐（矩阵乘法定义里 `mat1[i][k] * mat2[k][j]`，这两个 k 必须是同一个
    维度），实现时容易把行列语义搞混；3) 结果矩阵必须预先初始化成全 0 的稠密 m x n
    数组再逐步累加，不能用稀疏表示直接构造——同一个 `(i, j)` 位置可能被多个不同的
    `k` 命中，需要真正做加法而不是覆盖写入。
    """
    m = len(mat1)
    n = len(mat2[0])
    mat1_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in mat1]
    mat2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in mat2]
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for col_k, val1 in mat1_sparse[i]:
            for j, val2 in mat2_sparse[col_k]:
                result[i][j] += val1 * val2
    return result


def find_winner(moves: list[list[int]]) -> str:
    """
    【题意】两名玩家 A、B 在 3x3 棋盘上轮流落子（A 用 "X" 先手，B 用 "O" 后手）。给定
    落子顺序 moves（`moves[i] = [row, col]`），如果某一方先让某一行/某一列/某条对角线
    的三个格子都是自己的棋子，游戏结束并返回获胜方（"A" 或 "B"）；如果棋盘下满仍
    无人获胜，返回 "Draw"；如果 moves 还没下满且未分出胜负，返回 "Pending"。
    【思路】不需要真的维护一个 3x3 的字符棋盘去逐条判断"哪三个格子连成一线"，更
    简洁的写法是给每一行、每一列、两条对角线各维护一个整数计数器：A 每下一步棋，
    把这一步涉及到的行计数器、列计数器（以及如果这一步落在对角线/副对角线上，对应
    的对角线计数器）都加 1；B 的落子则让对应计数器减 1。由于每个格子只会被下一次
    棋（moves 里不会有重复坐标），A 的 +1 和 B 的 -1 不会互相干扰。每下一步之后，
    只需要检查这一步实际影响到的那几个计数器——如果某个计数器的绝对值达到 3，说明
    对应的行/列/对角线被同一个玩家占满，刚下这一步的玩家就是赢家，可以立刻返回，
    不需要每步都重新扫描全部 8 条可能获胜的线。moves 全部处理完仍未分出胜负时，
    根据 `len(moves)` 是否等于 9（棋盘下满）判断是 "Draw" 还是 "Pending"。
    【复杂度】时间 O(len(moves))（每一步只需要更新常数个计数器并检查一次）；空间
    O(1)（3 个行计数器、3 个列计数器、2 个对角线计数器，是固定大小，不随输入变化）。
    【易错点】1) 判断赢家不能只看步数的奇偶性就直接下结论，必须先根据当前这一步
    更新对应的计数器，再检查这一步实际影响到的那几条线是否被占满；2) 副对角线的
    判断条件容易和主对角线搞混——主对角线是 `row == col`，副对角线是
    `row + col == n - 1`（3x3 棋盘即 `row + col == 2`），两者是相互独立的两个条件，
    一步棋可能同时命中主对角线和副对角线（比如正中心格子），需要分别检查；3) 平局
    的判断要用 `len(moves) == 9`（棋盘全部格子都已落子）而不是简单地"moves 遍历完"
    就认为是平局——如果 moves 长度本身小于 9 且没有人获胜，正确结果是 "Pending"
    而不是 "Draw"。
    """
    n = 3
    row_counts = [0] * n
    col_counts = [0] * n
    diag = 0
    anti_diag = 0
    for idx, (row, col) in enumerate(moves):
        mark = 1 if idx % 2 == 0 else -1  # A 用 +1 计数，B 用 -1 计数
        row_counts[row] += mark
        col_counts[col] += mark
        if row == col:
            diag += mark
        if row + col == n - 1:
            anti_diag += mark
        if (
            abs(row_counts[row]) == n
            or abs(col_counts[col]) == n
            or abs(diag) == n
            or abs(anti_diag) == n
        ):
            return "A" if mark == 1 else "B"
    return "Draw" if len(moves) == 9 else "Pending"


def _self_test() -> None:
    assert spiral_matrix_iii(1, 4, 0, 0) == [[0, 0], [0, 1], [0, 2], [0, 3]]
    assert spiral_matrix_iii(5, 6, 1, 4) == [
        [1, 4], [1, 5], [2, 5], [2, 4], [2, 3], [1, 3], [0, 3], [0, 4], [0, 5],
        [3, 5], [3, 4], [3, 3], [3, 2], [2, 2], [1, 2], [0, 2],
        [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [3, 1], [2, 1], [1, 1], [0, 1],
        [4, 0], [3, 0], [2, 0], [1, 0], [0, 0],
    ]

    mat1 = [[3, 3, 1, 1], [2, 2, 1, 2], [1, 1, 1, 2]]
    assert diagonal_sort(mat1) == [[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3]]

    assert largest_submatrix([[0, 0, 1], [1, 1, 1], [1, 0, 1]]) == 4
    assert largest_submatrix([[1, 0, 1, 0, 1]]) == 3
    assert largest_submatrix([[1, 1, 0], [1, 0, 1]]) == 2
    assert largest_submatrix([[0, 0], [0, 0]]) == 0

    assert sparse_matrix_multiply([[1, 0, 0], [-1, 0, 3]], [[7, 0, 0], [0, 0, 0], [0, 0, 1]]) == [
        [7, 0, 0],
        [-7, 0, 3],
    ]

    assert find_winner([[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]) == "A"
    assert find_winner([[0, 0], [1, 1], [0, 1], [0, 2], [1, 0], [2, 0]]) == "B"
    assert (
        find_winner([[0, 0], [1, 1], [2, 0], [1, 0], [1, 2], [2, 1], [0, 1], [0, 2], [2, 2]])
        == "Draw"
    )
    assert find_winner([[0, 0], [1, 1]]) == "Pending"

    print(
        "[PASS] p23_matrix_simulation_iii: 5/5 题通过 "
        "(螺旋矩阵III/将矩阵按对角线排序/重新排列后的最大子矩阵/稀疏矩阵的乘法/找出井字棋的获胜者)"
    )


if __name__ == "__main__":
    _self_test()
