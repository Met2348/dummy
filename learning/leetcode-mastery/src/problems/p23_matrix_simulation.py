"""分类 23：矩阵与模拟 —— 没有高深算法，考的是边界条件、下标偏移、原地修改约束的严谨性。"""
from __future__ import annotations


def rotate_image(matrix: list[list[int]]) -> None:
    """
    【题意】给定一个 n x n 的二维矩阵表示图像，原地把它顺时针旋转 90 度（不能返回新
    矩阵，必须直接修改传入的 matrix）。
    【思路】直接按"旋转后的坐标公式"一个个搬运容易在下标上出错，更稳妥的写法是把
    一次 90 度旋转拆成两个更简单、更不容易出错的基本操作的组合：先沿主对角线转置
    （`matrix[i][j]` 和 `matrix[j][i]` 互换，让"第 i 行"变成"第 i 列"），再把每一行
    左右反转——转置之后其实已经是"逆时针转 90 度 + 上下翻转"的效果，再对每一行做一次
    反转，两步叠加起来的净效果正好等于顺时针旋转 90 度。比起直接推导"新坐标 = f(旧
    坐标)"的通用公式，"转置 + 逐行反转"这两步都是耳熟能详、不容易出错的基础操作，
    组合出来的正确性也更容易在草稿上验证。
    【复杂度】时间 O(n^2)（转置和反转都要访问每个格子常数次）；空间 O(1)（原地修改，
    转置时只交换上三角和下三角，不需要额外矩阵）。
    【易错点】1) 转置时如果两层循环都从 0 开始跑到 n（`for j in range(n)`），会把
    每一对 (i,j) 交换两次（一次在处理 (i,j) 时，一次在处理 (j,i) 时），相当于白做，
    必须让内层循环从 `i+1` 开始，只交换上三角到下三角这一半；2) 忘记转置之后还要
    "每行反转"这一步，只转置的结果是"沿主对角线镜像"，不是旋转。
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()


def spiral_order(matrix: list[list[int]]) -> list[int]:
    """
    【题意】给定 m x n 矩阵，按顺时针螺旋顺序（从左上角开始，右→下→左→上循环）
    返回矩阵中的所有元素。
    【思路】维护 `top/bottom/left/right` 四条还未走过的边界，每一轮按"上边界从左到右
    →右边界从上到下→下边界从右到左→左边界从下到上"的顺序各走一遍，每走完一条边就把
    对应的边界往里收缩一格（因为这条边已经走完，不会再走第二遍）；一轮螺旋走完后，
    边界圈缩小了一圈，重复这个过程直到边界相遇或交错。这样每个格子只会被访问一次，
    整个逻辑完全由四个边界变量的相对位置决定，不需要额外的 visited 矩阵。
    【复杂度】时间 O(m*n)（每个格子恰好访问一次）；空间 O(1)（除去返回结果本身，
    不需要额外的辅助矩阵）。
    【易错点】1) 走完"上边界"和"右边界"之后，在走"下边界"和"左边界"之前，必须
    分别再检查一次 `top <= bottom` 和 `left <= right`——如果矩阵只有一行或一列，
    上边界走完之后下边界其实和上边界是同一行，如果不加这个检查会把这一行/列重复
    走一遍；2) 四个边界收缩的时机容易搞错顺序（应该是"走完一条边，立刻收缩对应的
    边界"，而不是等一整轮都走完再统一收缩），顺序错了会导致下一轮用到过期的边界。
    """
    if not matrix or not matrix[0]:
        return []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    result: list[int] = []
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
    return result


def set_zeroes(matrix: list[list[int]]) -> None:
    """
    【题意】给定 m x n 矩阵，如果某个元素是 0，就把它所在的整行和整列都置为 0，
    要求原地修改，且额外空间复杂度为 O(1)（不能用一个同样大小的矩阵去记录哪些
    行/列需要置零）。
    【思路】如果不限制空间，最直接的想法是用两个集合分别记录"哪些行需要置零"、
    "哪些列需要置零"。要把这两个集合的空间压缩到 O(1)，可以借用矩阵**自身的第一行
    和第一列**当作这两个集合——`matrix[i][0]` 被标记成 0 就表示"第 i 行需要置零"，
    `matrix[0][j]` 被标记成 0 就表示"第 j 列需要置零"。但第一行、第一列自己本身也是
    数据的一部分，用它们当标记位之前，必须先额外用两个布尔变量单独记下"第一行本身
    有没有 0"、"第一列本身有没有 0"，因为等会儿第一行第一列会被用来记录其它行列的
    置零信息，它们原本"自己是不是 0"这个信息会被覆盖掉、必须提前备份。具体三步：
    1) 先扫一遍矩阵，把每个 0 的信息"记"到它所在行的第一列格子和所在列的第一行格子
    上；2) 根据第一行第一列（除了它们自身之外的位置，`matrix[1:][1:]`）里的标记，
    把对应的行和列置零；3) 最后单独处理第一行、第一列自己该不该被置零（用第 0 步
    备份的两个布尔变量判断）。
    【复杂度】时间 O(m*n)（三次遍历，都是常数倍）；空间 O(1)（只多用了两个布尔
    变量，没有额外的矩阵或集合）。
    【易错点】1) 必须先备份"第一行/第一列自身是否含 0"，再开始用它们记录别的行列
    信息——如果先做标记再判断"第一行本身要不要置零"，这时候第一行早被别的 0 污染
    过了，会把不该置零的第一行也置零（或者相反）；2) 第二步"根据标记置零"时，循环
    范围必须从下标 1 开始（`range(1, rows)` / `range(1, cols)`），不能碰到第一行第
    一列自己，否则会提前破坏还没读完的标记信息；3) 第一步"记录 0 的位置"和第二步
    "根据记录置零"必须分两趟完整遍历分开做，不能合并成一趟——如果边扫描边置零，
    后面扫到的格子会被前面刚置的 0 污染，把不该置零的格子也误判成"遇到了原始的 0"。
    """
    rows, cols = len(matrix), len(matrix[0])
    first_row_has_zero = any(matrix[0][c] == 0 for c in range(cols))
    first_col_has_zero = any(matrix[r][0] == 0 for r in range(rows))

    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[r][0] = 0
                matrix[0][c] = 0

    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    if first_row_has_zero:
        for c in range(cols):
            matrix[0][c] = 0
    if first_col_has_zero:
        for r in range(rows):
            matrix[r][0] = 0


def game_of_life(board: list[list[int]]) -> None:
    """
    【题意】给定 m x n 的 0/1 矩阵表示"生命游戏"的当前状态（1=活细胞，0=死细胞），
    按规则（活细胞周围活邻居少于2或多于3个则死亡；死细胞周围恰好3个活邻居则复活；
    活细胞周围2或3个活邻居保持存活）原地计算出下一代状态，要求不能用额外的同样大小
    的矩阵存储中间结果。
    【思路】难点在于"原地更新"和"计算每个格子的新状态需要用到邻居的**旧**状态"之间
    的冲突——如果直接把某个格子改成新状态，它的邻居在计算自己的新状态时，读到的就
    是被提前更新过的"新值"而不是本该用的"旧值"。解决办法是用比 0/1 更多的中间状态
    做编码，把"旧状态"和"新状态"同时压缩存在同一个格子里：2 表示"旧值是活、新值是
    死"，3 表示"旧值是死、新值是活"。这样在第一遍遍历计算每个格子的新状态时，判断
    "某个邻居原来是不是活的"只需要判断它的值是 1 或 2（这两个值都意味着"旧状态是
    活的"，不区分它的新状态是否已经算出来），已经计算过的邻居不会误导后续的判断；
    第一遍遍历全部结束后，再统一做第二遍遍历，把 2 转回 0、3 转回 1，得到最终的下一代。
    【复杂度】时间 O(m*n*8)（每个格子固定检查 8 个方向的邻居）；空间 O(1)（原地
    编码，不使用额外矩阵）。
    【易错点】1) 判断"某邻居原本是否是活细胞"时，如果只判断 `== 1`，会漏掉那些
    "原本是活的、但已经被计算出新状态是死（编码成 2）"的邻居，把它们错误地当成
    "原本是死的"；正确写法是判断这个值是否在 `(1, 2)` 这个集合里；2) 两遍遍历不能
    合并成一遍——必须等所有格子的新状态都用编码方式记录完，才能统一转换回 0/1，
    否则会重复出现"用了别的格子的新值而不是旧值"这个问题；3) 网格边界外的邻居要
    做边界检查，遗漏会导致角落/边缘格子的邻居计数偏小。
    """
    rows, cols = len(board), len(board[0])

    def count_live_neighbors(r: int, c: int) -> int:
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] in (1, 2):
                    count += 1
        return count

    for r in range(rows):
        for c in range(cols):
            live_neighbors = count_live_neighbors(r, c)
            if board[r][c] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    board[r][c] = 2  # 活变死
            else:
                if live_neighbors == 3:
                    board[r][c] = 3  # 死变活

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 2:
                board[r][c] = 0
            elif board[r][c] == 3:
                board[r][c] = 1


def generate_matrix(n: int) -> list[list[int]]:
    """
    【题意】给定正整数 n，生成一个 n x n 的矩阵，按顺时针螺旋顺序填入 1 到 n^2
    这些数字。
    【思路】和 spiral_order（54 题）是完全对称的逆过程——54 题是"给一个已经填好的
    矩阵，按螺旋顺序读出来"，这题是"按螺旋顺序把 1..n^2 依次写进去"。维护同样的
    `top/bottom/left/right` 四个边界，循环"上边界从左到右填→右边界从上到下填→
    下边界从右到左填→左边界从下到上填"，每填完一条边就收缩对应边界、且每次收缩后
    都要重新检查边界是否还合法（n 为奇数时最中心会剩下一个格子，需要靠边界检查
    保证它被正确填到而不会被跳过或重复填）。
    【复杂度】时间 O(n^2)（每个格子填一次）；空间 O(n^2)（返回的矩阵本身，除此
    之外不需要额外空间）。
    【易错点】和 spiral_order 的坑完全一致：1) 填完上边界、右边界之后，填下边界和
    左边界之前，必须分别检查 `top <= bottom`、`left <= right`，否则 n 为奇数时最后
    剩下的那"一行"或"一列"会被重复填两次；2) 数字计数器 `num` 要在每次填格子后
    立刻自增，不能在一整条边填完之后才统一累加，否则同一条边上填的会是同一个数字。
    """
    matrix = [[0] * n for _ in range(n)]
    top, bottom = 0, n - 1
    left, right = 0, n - 1
    num = 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            matrix[top][c] = num
            num += 1
        top += 1
        for r in range(top, bottom + 1):
            matrix[r][right] = num
            num += 1
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                matrix[bottom][c] = num
                num += 1
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                matrix[r][left] = num
                num += 1
            left += 1
    return matrix


def is_valid_sudoku(board: list[list[str]]) -> bool:
    """
    【题意】给定 9x9 的数独棋盘（未填的格子用 '.' 表示），只需要判断当前已经填入的
    数字是否符合数独规则——每一行、每一列、每个 3x3 宫内 1-9 不能重复出现（不要求
    判断这个棋盘能否被解出完整答案）。
    【思路】用三组各 9 个集合分别记录"每一行已经出现过的数字"、"每一列已经出现过的
    数字"、"每个 3x3 宫已经出现过的数字"。扫描棋盘的每一个非 '.' 格子时，先检查它的
    值是否已经出现在对应的行集合、列集合、宫集合里的**任意一个**——只要出现在其中
    一个里，就说明违反了对应那条规则，直接判定无效；否则把这个值同时加进它所属的
    行集合、列集合、宫集合。宫的编号用 `(r // 3) * 3 + c // 3` 计算：`r // 3` 和
    `c // 3` 分别把行、列坐标压缩到 0/1/2 三个"宫的行号/列号"，再按"宫的行号 * 3 +
    宫的列号"编号成 0~8 唯一的宫下标。
    【复杂度】时间 O(81)（固定 9x9 大小，本质是 O(1)，但写成正比于格子数更直观）；
    空间 O(81)（27 个集合，每个集合最多存 9 个数字）。
    【易错点】1) 宫的编号公式容易写错——必须先做整数除法把坐标压缩到宫格坐标系
    再组合，直接用 `r*3+c` 之类的公式是错的；2) 忘记跳过 '.' 格子，把它也当成需要
    判重的字符会导致把多个空格子误判成"重复的 '.' "；3) 加入集合的时机必须在
    "确认没有冲突"之后，如果先加入再判断会把这个数字自己和自己判定为冲突。
    """
    rows: list[set[str]] = [set() for _ in range(9)]
    cols: list[set[str]] = [set() for _ in range(9)]
    boxes: list[set[str]] = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == ".":
                continue
            box_id = (r // 3) * 3 + c // 3
            if val in rows[r] or val in cols[c] or val in boxes[box_id]:
                return False
            rows[r].add(val)
            cols[c].add(val)
            boxes[box_id].add(val)
    return True


def _self_test() -> None:
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_image(matrix)
    assert matrix == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]

    m = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes(m)
    assert m == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]

    board = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    game_of_life(board)
    assert board == [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]

    assert generate_matrix(3) == [[1, 2, 3], [8, 9, 4], [7, 6, 5]]

    valid_board = [
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
    assert is_valid_sudoku(valid_board) is True
    invalid_board = [row[:] for row in valid_board]
    invalid_board[0][0] = "8"  # 和第一列已有的 8(第4行第0列)冲突
    assert is_valid_sudoku(invalid_board) is False

    print(
        "[PASS] p23_matrix_simulation: 6/6 题通过 "
        "(旋转图像/螺旋矩阵/矩阵置零/生命游戏/螺旋矩阵II/有效的数独)"
    )


if __name__ == "__main__":
    _self_test()
