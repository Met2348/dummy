"""图 BFS/DFS 专题 Part II（进阶补充）：被围绕的区域 / 太平洋大西洋水流问题 /
岛屿的最大面积 / 二进制矩阵中的最短路径 / 矩阵中的最长递增路径 / 01矩阵 /
蛇梯棋 / 最短的桥 / 所有可能的路径 / 寻找图中是否存在路径 / 重新规划路线 /
找出星型图的中心节点。

不重复讲 Part I 已经建立的"BFS 求最短路、DFS 求连通性/穷举路径"选择依据，本文件
聚焦网格类图论的几种变体：双源反向搜索（正向想很难、反向想很简单的经典案例）、
DFS + 记忆化、多源 BFS、以及"先建图再遍历"时的一些额外技巧（编号转坐标、给边打
方向标签、利用特殊图结构做 O(1) 判断）。
"""
from __future__ import annotations

from collections import defaultdict, deque


# ── LC130 被围绕的区域 ───────────────────────────────────────────────────
def solve_surrounded_regions(board: list[list[str]]) -> None:
    """
    【题意】给定一个由 'X' 和 'O' 组成的二维网格 board，把所有被 'X' 完全包围的
    'O' 区域（区域由水平/竖直相邻的 'O' 连接而成，且这片区域不与棋盘边界直接或
    间接相连）原地改写成 'X'；与边界相连的 'O'（哪怕通过一长串 'O' 间接连到边界）
    保持不变。函数无返回值，直接修改传入的 board。
    【思路】直接的思路是"对每个内部的 O 判断它所在的连通块是否触达边界"，但这样
    同一个连通块里的每个格子都要重新走一次判断，效率低还重复计算。更好的思路是
    反过来想：**先找出所有"一定安全"的 O**（从边界出发、通过相邻 O 能到达的所有
    O，这些一定不会被包围），再把其余所有 O 一律视为被包围。具体做法：从四条边界
    上的每一个 'O' 出发做 DFS，把能连通到的所有 'O' 暂时标记成第三态 '#'（"安全，
    不能被吃掉"）；扫描完边界后，棋盘上剩下的 'O'（没被标记成 '#' 的）就一定是
    被包围的，直接改成 'X'；最后把所有 '#' 复原成 'O'。这个"先从边界反向标记安全
    区域，再处理剩下的"思路，和下面深挖的"太平洋大西洋水流"是同一类技巧。
    【复杂度】时间 O(R·C)（每个格子最多被 DFS 访问一次）；空间最坏 O(R·C)（整个
    棋盘都是 O 相连时的递归栈深度）。
    【易错点】1) 直接对每个内部 O 单独判断是否连到边界，会对同一个连通块里的多个
    格子重复扫描，时间复杂度明显更差；2) 忘记用第三个标记状态 '#'，直接在 DFS
    过程中就把边界能到达的 O 改成别的最终状态，会和"被包围、要改成 X"的 O 混淆，
    最后一步无法区分谁该保留、谁该修改；3) 只从四个角出发 DFS 而不是从整条边界
    出发，会漏掉边界中间的 O。
    """
    if not board or not board[0]:
        return
    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != "O":
            return
        board[r][c] = "#"
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)
    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == "#":
                board[r][c] = "O"


# ── LC417 太平洋大西洋水流问题 ───────────────────────────────────────────
def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定一个 m×n 的整数矩阵 heights 表示大陆地形高度，左边界和上边界靠着
    "太平洋"，右边界和下边界靠着"大西洋"；水从每个格子出发只能流向高度小于等于
    自己的相邻格子（水往低处流，允许流向等高的格子），返回所有"既能流到太平洋、
    又能流到大西洋"的格子坐标。
    【思路】本讲深挖的核心技巧——"反过来想"。正向思路是对每个格子分别做
    DFS/BFS，判断水从这个格子出发能不能流到太平洋边界、能不能流到大西洋边界，
    这样每个格子都要重新搜索一次，非常低效。反向思路：既然"水从格子 A 流到边界"
    等价于"从边界出发反着水流方向走，能走到格子 A"，那么直接从太平洋边界的所有
    格子、大西洋边界的所有格子分别出发做多源 BFS，搜索规则也要反过来——正向水流
    规则是"流向更低或相等的格子"，反过来从边界向内走，能走的条件就变成"走向更高
    或相等的格子"（`heights[nxt] >= heights[cur]`）。分别得到"能被太平洋边界反向
    搜索到的格子集合"和"能被大西洋边界反向搜索到的格子集合"后，两个集合的交集就
    是答案——只需要两次多源 BFS（而不是对每个格子各搜一次），复杂度从平方级降到
    线性级。
    【复杂度】时间 O(R·C)（两次多源 BFS，各自访问每个格子至多一次）；空间
    O(R·C)（两个 visited 集合 + 队列）。
    【易错点】1) 正向对每个格子分别判断能否流到两个大洋，复杂度爆炸且大量重复
    计算；2) 反向搜索时条件写反——正向是"流向更低"，反向应该是"走向更高或
    相等"，如果照抄正向条件（写成 `heights[nxt] <= heights[cur]`），搜索方向就
    和实际水流方向一致而非相反，结果完全错误；3) 忘记边界四条边都要作为起点
    （不只是四个角），漏掉某一条完整的边会让该条边界上原本可达的格子被漏判。
    """
    if not heights or not heights[0]:
        return []
    rows, cols = len(heights), len(heights[0])

    def bfs(starts: list[tuple[int, int]]) -> set[tuple[int, int]]:
        visited = set(starts)
        q = deque(starts)
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and heights[nr][nc] >= heights[r][c]
                ):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return visited

    pacific_starts = [(0, c) for c in range(cols)] + [(r, 0) for r in range(rows)]
    atlantic_starts = [(rows - 1, c) for c in range(cols)] + [
        (r, cols - 1) for r in range(rows)
    ]
    pacific = bfs(pacific_starts)
    atlantic = bfs(atlantic_starts)
    return [list(cell) for cell in sorted(pacific & atlantic)]


# ── LC695 岛屿的最大面积 ─────────────────────────────────────────────────
def max_area_of_island(grid: list[list[int]]) -> int:
    """
    【题意】给定只含 0（水）和 1（陆地）的二维网格 grid，岛屿由水平/竖直相邻的
    陆地连接而成，返回网格中最大的岛屿面积（连通块的格子数），如果没有岛屿返回
    0。
    【思路】和 Part I 的 num_islands（求连通块个数）几乎同一套骨架，区别只是这次
    不只是"计数 +1"，而是要让 DFS 函数返回"这个连通块从这个格子往下探到底一共
    淹没了多少格子"：`dfs(r, c)` 如果越界或不是陆地就返回 0（对总面积没有贡献）；
    否则先把当前格子"淹没"（改成 0，防止被重复计数），再返回 1（自己）加上四个
    方向递归结果之和。外层扫描每个格子，遇到还是陆地（1）的格子就触发一次 dfs
    拿到这个连通块的面积，用它更新全局最大值。
    【复杂度】时间 O(R·C)（每个格子最多被访问一次）；空间最坏 O(R·C)（整个网格
    是一整块陆地时的递归栈深度）。
    【易错点】1) 把 dfs 设计成"直接在函数内部维护一个全局面积变量再返回 None"
    （类似 num_islands 的风格）本身也可行，但如果混用两种写法（既想用返回值又想
    用外部变量叠加），容易造成面积被重复计算；2) 忘记原地"淹没"当前格子就直接
    递归四个方向，会在同一片陆地上反复横跳、重复计数甚至无限递归；3) 混淆"求
    最大面积"和"求岛屿总数"，如果只是把每次 dfs 的返回值累加成一个全局统计量
    （而不是取 max），会把所有岛屿的面积加在一起而不是取最大的那一个。
    """
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r: int, c: int) -> int:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

    best = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                best = max(best, dfs(r, c))
    return best


# ── LC1091 二进制矩阵中的最短路径 ────────────────────────────────────────
def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    """
    【题意】给定一个 n×n 的二进制矩阵 grid（0 表示可通过的格子，1 表示障碍），从
    左上角 (0,0) 走到右下角 (n-1,n-1)，每一步可以走向当前格子的 8 个方向（上下
    左右 + 4 个对角线方向）里任意一个值为 0 的相邻格子，返回最短路径的长度（长度
    = 经过的格子数，含起点和终点）；如果起点、终点本身是障碍或者根本无法到达，
    返回 -1。
    【思路】和"腐烂的橘子""单词接龙"是同一类"无权图最短路用 BFS"问题，唯一的
    新变化是**移动方向从 4 个变成 8 个**（把对角线也算作相邻）。BFS 逐层扩展，
    第一次到达终点时所在的层数就是最短路径长度（这里直接把"路径长度"作为队列
    元素的一部分随 BFS 一起传递，出队时立即检查是否到达终点）。起点或终点本身是
    障碍（值为 1）时提前返回 -1。
    【复杂度】时间 O(n²)（每个格子最多入队一次，每次出队检查 8 个方向）；空间
    O(n²)（visited 集合 + 队列）。
    【易错点】1) 只写 4 个方向（照抄岛屿/橘子系列的写法），会漏掉对角线方向，
    得到偏大甚至错误的最短路径长度；2) 忘记特判起点或终点本身就是障碍的情况，
    直接跑 BFS 会从一个非法起点开始扩展，得到错误结果；3) 队列里存储的路径长度
    从 1 开始计（因为起点本身就算 1 个格子）而不是从 0 开始，如果初始值或后续
    +1 的时机记错，容易产生长度多算或少算 1 的偏差——起点 (0,0) 入队时就应该带
    着 dist=1，而不是 dist=0 再在别处补偿。
    """
    n = len(grid)
    if grid[0][0] != 0 or grid[n - 1][n - 1] != 0:
        return -1

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    visited = {(0, 0)}
    q = deque([(0, 0, 1)])
    while q:
        r, c, dist = q.popleft()
        if (r, c) == (n - 1, n - 1):
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < n
                and 0 <= nc < n
                and (nr, nc) not in visited
                and grid[nr][nc] == 0
            ):
                visited.add((nr, nc))
                q.append((nr, nc, dist + 1))
    return -1


# ── LC329 矩阵中的最长递增路径 ───────────────────────────────────────────
def longest_increasing_path(matrix: list[list[int]]) -> int:
    """
    【题意】给定 m×n 的整数矩阵 matrix，找出其中最长递增路径的长度；路径可以从
    任意格子开始、任意格子结束，每一步只能移动到上下左右四个相邻格子之一，且要求
    严格递增（下一个格子的值必须大于当前格子）。
    【思路】"DFS + 记忆化"的经典场景。如果不加记忆化，朴素 DFS 会对每个格子都
    独立地重新搜索它出发能达到的最长递增路径，而不同起点的搜索会大量重复访问
    相同的格子（比如从格子 A 经过格子 B，和直接从格子 B 出发，B 之后的搜索结果
    完全一样），导致指数级重复计算。用一个 memo 字典记录"以某个格子为起点，出发
    能走出的最长递增路径长度"，第一次算出来就存起来，以后任何路径再次经过这个
    格子时直接查表，不用重新展开子树——这正是"记忆化搜索"和普通回溯最大的区别：
    普通回溯很少缓存中间结果（同一状态很少被不同路径重复到达），而这里"以某格子
    为起点的最长递增路径长度"是一个和"如何到达这个格子"完全无关的固定值，天然
    适合缓存。
    【复杂度】时间 O(R·C)（每个格子的 dfs 结果只计算一次，均摊后每个格子和它的
    4 条出边各处理常数次）；空间 O(R·C)（memo 字典 + 递归栈）。
    【易错点】1) 不加记忆化直接朴素 DFS，在网格较大、递增路径交织复杂时会指数级
    爆炸超时；2) memo 的 key 含义搞混（比如把"从格子出发的最长路径"和"到达格子
    为止已经走过的长度"这两个不同含义的量混用同一份缓存），会读出语义不一致的
    缓存值；3) 移动条件写反（写成 `matrix[nxt] < matrix[cur]` 却忘了这是"寻找
    递增路径"，应该是 `matrix[nxt] > matrix[cur]`），会搜出递减路径的长度。
    """
    if not matrix or not matrix[0]:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    memo: dict[tuple[int, int], int] = {}

    def dfs(r: int, c: int) -> int:
        if (r, c) in memo:
            return memo[(r, c)]
        best = 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        memo[(r, c)] = best
        return best

    return max(dfs(r, c) for r in range(rows) for c in range(cols))


# ── LC542 01 矩阵 ────────────────────────────────────────────────────────
def update_matrix(mat: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定一个只含 0 和 1 的矩阵 mat，对每一个格子，返回它到最近的 0 的
    距离（相邻格子的距离为 1，只能上下左右移动）。
    【思路】又一个"多源 BFS"的场景，和 Part I 的"腐烂的橘子"是同一个模式，
    唯一区别是这里"同时出发的源"是所有值为 0 的格子（而不是所有已经腐烂的
    橘子）：把所有 0 格子的距离初始化为 0 并同时入队，再统一向外扩散；每向外
    扩展一层，说明这一层格子到最近的 0 的距离，恰好比上一层多 1（因为 BFS 保证
    第一次到达某个格子时经过的层数就是最短距离）。用一个和 mat 同形状的 dist
    矩阵（初值 -1 表示"还没被更新过"）同时充当"距离结果"和"visited 标记"两个
    作用，省掉一份额外的 visited 集合。
    【复杂度】时间 O(R·C)（每个格子最多入队一次）；空间 O(R·C)（dist 矩阵 +
    队列）。
    【易错点】1) 只从一个 0 出发做单源 BFS 再取所有 0 源结果的最小值——虽然结果
    一样但效率是对每个 0 都做一次全图 BFS，远不如多源 BFS 一次搞定；2) 忘记用
    `dist==-1` 兼职当 visited 标记，如果没有这个判断，同一个格子会被多个不同源
    头的扩展重复更新；3) 把 1 格子的初始距离误设为 0 或者忘记初始化（应该保留
    -1，等 BFS 扩展到它才被赋值）。
    """
    rows, cols = len(mat), len(mat[0])
    dist = [[-1] * cols for _ in range(rows)]
    q: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                dist[r][c] = 0
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist


# ── LC909 蛇梯棋 ─────────────────────────────────────────────────────────
def snakes_and_ladders(board: list[list[int]]) -> int:
    """
    【题意】给定一个 n×n 的棋盘 board，玩家从格子 1 出发，每一轮掷骰子可以走到
    编号 [当前编号+1, 当前编号+6] 之间任意一个格子（不能超过 n²）；如果目标格子
    上有蛇或梯子（board 上对应位置不是 -1），会被直接传送到 board 里记录的目标
    编号；求从 1 走到 n² 的最少骰子轮数，如果无法到达返回 -1。棋盘编号规则是
    "之字形"（Boustrophedon）：编号从棋盘最后一行开始、从左到右是 1~n，倒数第二
    行从右到左是 n+1~2n，再上一行又从左到右，如此交替向上。
    【思路】这题的"图"藏在两层翻译里：第一层是把"棋盘上的格子"看成图节点、"一次
    骰子（1~6）"看成边，是标准的无权图最短路，直接 BFS；第二层翻译更麻烦——题目
    给的 board 是二维的，但格子编号是按"之字形"排列的一维序列，所以第一步必须
    先写一个 `get_cell(label)` 函数把"一维编号"转换成"二维行列坐标"，才能去
    board 里查这个格子上有没有蛇/梯子。转换规则：编号从 1 开始，先算它是"从底
    往上数第几行"（`row_from_bottom = (label-1)//n`），再换算成真实行号
    `r = n-1-row_from_bottom`；同一行内部编号方向交替——`row_from_bottom` 是
    偶数（从底数起第 0/2/4… 行）的行从左到右编号，是奇数的行从右到左编号，所以
    列号在奇数行要做一次镜像 `c = n-1-col`。转换出坐标后查 `board[r][c]`，如果
    不是 -1（有蛇或梯子）就要跳到 `board[r][c]` 这个新编号，否则留在当前编号。
    BFS 时，状态就是"当前所在的编号"（已经把蛇梯效果处理完毕的编号），每一层
    尝试 1~6 步之内所有可能的下一个编号。
    【复杂度】时间 O(n²)（每个编号最多入队一次，每次尝试最多 6 种骰子结果）；
    空间 O(n²)（visited 集合 + 队列）。
    【易错点】1) "之字形"坐标转换写反方向（比如奇偶行判断反了，或者忘记翻转
    列），会查到棋盘上错误位置的蛇/梯子，得到完全错误的传送目标；2) 忘记"如果
    目标格子有蛇/梯子，最终落脚点是 board 上记录的编号，而不是骰子本来要走到的
    编号"，直接把骰子编号当成新状态入队，等于没有实现蛇梯效果；3) BFS 的 visited
    标记应该标记"处理完蛇梯之后的最终落脚编号"，而不是骰子直接算出来的中间编号，
    否则同一个真实格子可能被反复展开。
    """
    n = len(board)

    def get_cell(label: int) -> tuple[int, int]:
        row_from_bottom, col = divmod(label - 1, n)
        r = n - 1 - row_from_bottom
        c = n - 1 - col if row_from_bottom % 2 == 1 else col
        return r, c

    visited = {1}
    q = deque([(1, 0)])
    while q:
        label, steps = q.popleft()
        if label == n * n:
            return steps
        for nxt in range(label + 1, min(label + 6, n * n) + 1):
            r, c = get_cell(nxt)
            dest = board[r][c] if board[r][c] != -1 else nxt
            if dest not in visited:
                visited.add(dest)
                q.append((dest, steps + 1))
    return -1


# ── LC934 最短的桥 ───────────────────────────────────────────────────────
def shortest_bridge(grid: list[list[int]]) -> int:
    """
    【题意】给定一个只含 0、1 的二维网格 grid，其中恰好有两座岛屿（分别是一个由
    1 组成的四连通连通块），返回把两座岛屿连接起来（把一些 0 变成 1）所需要改变
    的最少 0 的数量。
    【思路】分两阶段：第一阶段用 DFS 找到"第一座岛屿"的所有格子（从网格里第一个
    遇到的 1 出发，DFS 淹没并收集这片连通块里的每一个坐标）；第二阶段把这些坐标
    同时作为多源 BFS 的第 0 层起点，向外扩散——BFS 每碰到一个还没访问过的 0，说明
    这是从第一座岛屿"架桥"要经过的一格水，距离 +1 继续扩散；一旦碰到一个属于
    第二座岛屿的 1，说明桥已经搭到了第二座岛屿，当前记录的 dist 就是答案（dist
    表示"从第一座岛屿出发、已经跨过了多少格水才走到当前这一格"，当前这一格恰好
    与第二座岛屿相邻，所以 dist 就是需要改变的 0 的数量）。这一题是"DFS 找起点
    集合 + 多源 BFS 求最短距离"两个技巧的组合，和"腐烂的橘子/01矩阵"里"预先知道
    所有源头"不同的地方在于——这里第一批源头（第一座岛屿的格子）需要先用 DFS
    探测出来，而不是像橘子/01矩阵那样直接从输入里一次性读出。
    【复杂度】时间 O(R·C)（DFS 一次遍历第一座岛屿 + BFS 一次遍历整个网格）；
    空间 O(R·C)（visited 矩阵 + 队列/递归栈）。
    【易错点】1) 把"距离"算错一格——需要想清楚 BFS 出队时的 dist 到底表示什么：
    第一座岛屿的格子本身入队时 dist=0（还没跨过任何水），从岛屿扩展出去的第一格
    水 dist=1，以此类推；碰到第二座岛屿时，直接返回"当前处理格子"（也就是与
    第二座岛屿相邻的那格水）的 dist，不需要再 +1；2) 忘记第一阶段找"第一座岛屿"
    时也要维护 visited（避免和第二阶段 BFS 的 visited 混用导致同一座岛屿被
    DFS 重复访问）；3) 如果代码写错导致 DFS 把两座岛屿都探测成"第一座"，会让
    后续 BFS 从错误的起点集合开始搜索，得到错误的最短距离。
    """
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    def dfs(r: int, c: int, island: list[tuple[int, int]]) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 1:
            return
        visited[r][c] = True
        island.append((r, c))
        dfs(r + 1, c, island)
        dfs(r - 1, c, island)
        dfs(r, c + 1, island)
        dfs(r, c - 1, island)

    first_island: list[tuple[int, int]] = []
    found = False
    for r in range(rows):
        if found:
            break
        for c in range(cols):
            if grid[r][c] == 1:
                dfs(r, c, first_island)
                found = True
                break

    q = deque((r, c, 0) for r, c in first_island)
    while q:
        r, c, dist = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                if grid[nr][nc] == 1:
                    return dist
                visited[nr][nc] = True
                q.append((nr, nc, dist + 1))
    return -1


# ── LC797 所有可能的路径 ─────────────────────────────────────────────────
def all_paths_source_target(graph: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定一个有向无环图（DAG），用邻接表 graph 表示（`graph[i]` 是节点 i
    指向的所有邻居），求从节点 0 到节点 n-1 的所有路径（n=len(graph)），路径以
    节点编号列表形式返回，顺序不作要求。
    【思路】DAG 天然无环，不需要 visited 标记——这是和前面"克隆图""课程表"等题
    最大的区别：那些题的图可能有环，必须显式处理"已访问"逻辑；DAG 保证沿着有向
    边一直往前走不会绕回自己，所以可以放心地用最朴素的 DFS 穷举"每一步都试遍
    所有出边"，不用担心死循环。DFS 到达 n-1 时收集当前 path 为一条完整路径；
    否则对当前节点的每一个邻居递归下去，回溯时把刚加入 path 的节点弹出。
    【复杂度】时间 O(2ⁿ·n)（最坏情况下 DAG 的路径数可以是指数级，比如每个节点
    都指向后面所有节点，每条路径长度最多 n）；空间 O(n) 递归深度（不含结果
    本身）。
    【易错点】1) 误以为需要和有环图一样加 visited 标记——DAG 不需要，加了反而
    会漏掉"同一个节点在不同路径里被再次经过"的合法情况（DAG 中同一个节点确实
    可能出现在多条不同路径中，只是不会在单条路径内部出现两次，而这一点由"沿着
    有向边往前走、图本身无环"这个前提天然保证，不需要额外的 visited 去强制）；
    2) 忘记 `path[:]` 拷贝直接 `append(path)` 引用，导致后续 pop 把已收集的答案
    改空（和 Part I 子集题的经典错误一样）。
    """
    n = len(graph)
    res: list[list[int]] = []
    path = [0]

    def dfs(node: int) -> None:
        if node == n - 1:
            res.append(path[:])
            return
        for nxt in graph[node]:
            path.append(nxt)
            dfs(nxt)
            path.pop()

    dfs(0)
    return res


# ── LC1971 寻找图中是否存在路径 ──────────────────────────────────────────
def valid_path(n: int, edges: list[list[int]], source: int, destination: int) -> bool:
    """
    【题意】给定 n 个节点（编号 0~n-1）和一个无向边列表 edges，判断是否存在从
    source 到 destination 的路径。
    【思路】本类里最基础的"连通性判断"，可以用 BFS/DFS，也可以用并查集，这里
    选择 BFS：先把边列表转换成邻接表（无向图，两个方向都要加边），从 source 出发
    做 BFS，只要在扩展过程中遇到 destination 就返回 True；BFS 耗尽（队列空）还
    没遇到，说明两点不连通，返回 False。放在这里作为"图论最小可行版本"的对照——
    后面几道题（重新规划路线、找中心节点）都是在这题"建图 + 遍历"的基础上叠加
    一点点额外信息（边的原始方向、特殊结构）。
    【复杂度】时间 O(V+E)；空间 O(V+E)。
    【易错点】1) 只加了一个方向的边（比如只写 `graph[a].append(b)` 忘记
    `graph[b].append(a)`），把无向图当成有向图处理，会漏判很多实际连通的情况；
    2) source 等于 destination 时要能正确返回 True——当前实现把 source 一开始
    就加入 visited 并在出队时检查是否等于 destination，第一次弹出的就是 source
    本身，能正确处理这一边界情况。
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    visited = {source}
    q = deque([source])
    while q:
        cur = q.popleft()
        if cur == destination:
            return True
        for nxt in graph[cur]:
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return False


# ── LC1466 重新规划路线 ──────────────────────────────────────────────────
def min_reorder(n: int, connections: list[list[int]]) -> int:
    """
    【题意】有 n 座城市（编号 0~n-1）由 n-1 条道路连成一棵树，每条道路
    `connections[i]=[a,b]` 表示有一条从 a 到 b 的单向道路；每个城市都需要能够
    到达城市 0（首都），求最少需要改变多少条道路的方向。
    【思路】把每条有向边 `a->b` 建成一条"无向边"，但给这条边打上"方向标签"：
    沿着 `a->b` 走（原方向）标记 cost=1——这个 1 不是"距离"，而是"如果走这条边
    需要花费的改动次数"，因为从 a 走到 b 是顺着原方向走，恰恰是"背离首都"的
    方向，需要反过来；沿着 `b->a` 走（反着原方向走）标记 cost=0——这才是"朝着
    首都方向走"，不需要改。然后从节点 0（首都）出发做 DFS，累加沿途经过的
    cost——这样统计出的，就是"从首都出发，把所有边都走一遍时需要反转的边数"，
    恰好等于"让所有城市都能到达首都"所需的最少反转数（因为原图是一棵树，从首都
    出发的 DFS 会恰好覆盖 n-1 条边各一次）。"给边打标签"而不是真的建两张图
    （正向图 + 反向图）是本题最关键的技巧：把"方向信息"编码进边的附加数据里，
    而不是用边的存在与否来表示方向。
    【复杂度】时间 O(V+E)（V=n，E=n-1）；空间 O(V+E)。
    【易错点】1) cost 标签打反（把"顺着原方向走"标记成 0、"反着走"标记成 1），
    会算出恰好相反的答案（"至少要保留的边数"而不是"至少要反转的边数"）；
    2) 忘记这是一棵树，DFS 时如果没有正确维护 visited，可能在父子节点之间反复
    横跳、重复累加同一条边的 cost；3) 混淆"边的方向"和"DFS 遍历的方向"——DFS
    从 0 出发访问邻居时，"邻居是通过哪条有向边和当前节点相连"才是决定 cost 的
    唯一依据，和 DFS 本身"从谁走到谁"的方向无关。
    """
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in connections:
        graph[a].append((b, 1))  # a -> b 是原方向，沿着它走需要反转，标记 1
        graph[b].append((a, 0))  # 沿着 b -> a 走是朝着首都方向，不需要反转，标记 0

    visited = {0}
    count = 0

    def dfs(node: int) -> None:
        nonlocal count
        for nxt, cost in graph[node]:
            if nxt not in visited:
                count += cost
                visited.add(nxt)
                dfs(nxt)

    dfs(0)
    return count


# ── LC1791 找出星型图的中心节点 ──────────────────────────────────────────
def find_center(edges: list[list[int]]) -> int:
    """
    【题意】给定一个"星型图"（有一个中心节点与其余所有节点直接相连，除此之外
    没有其他边）的边列表 edges，返回这个中心节点的编号。
    【思路】本类里最"讨巧"的一题，甚至不需要真正建图或做任何遍历：星型图的边数
    是 n-1（n 个节点），每条边都包含中心节点；换句话说，中心节点在**每一条边**
    里都会出现。那么只需要看前两条边 `edges[0]` 和 `edges[1]`——它们一定共享
    恰好一个端点，这个共享的端点就是中心（因为除了中心节点，其余任意两个节点
    之间都没有边直接相连的公共伙伴——每个非中心节点只连着中心这一个邻居，如果两
    条边不是都连着中心，四个端点里就不会有重复，共享端点这件事就不会发生）。
    判断"共享哪个端点"只需要几次比较：`edges[0]` 的第一个端点是否出现在
    `edges[1]` 的两个端点之一中，是就是它，不是就是 `edges[0]` 的第二个端点。
    【复杂度】时间 O(1)（只看前两条边，不需要遍历整个边列表）；空间 O(1)。
    【易错点】1) 想复杂了，去统计每个节点的出现次数（度数）再取最大值，这样做
    时间是 O(E) 而不是 O(1)，虽然也能得到正确答案，但错过了这题真正的考点——
    利用星型图"中心节点在每条边都出现"这个结构特性，只看前两条边就够了；
    2) 判断"共享端点"时把两个端点的比较搞混（比如没有分别检查 `edges[0]` 的
    两个端点各自是否出现在 `edges[1]` 中），可能在个别数据下得出错误答案。
    """
    a, b = edges[0]
    c, d = edges[1]
    return a if a in (c, d) else b


def _self_test() -> None:
    board1 = [
        ["X", "X", "X", "X"],
        ["X", "O", "O", "X"],
        ["X", "X", "O", "X"],
        ["X", "O", "X", "X"],
    ]
    solve_surrounded_regions(board1)
    assert board1 == [
        ["X", "X", "X", "X"],
        ["X", "X", "X", "X"],
        ["X", "X", "X", "X"],
        ["X", "O", "X", "X"],
    ]

    heights = [
        [1, 2, 2, 3, 5],
        [3, 2, 3, 4, 4],
        [2, 4, 5, 3, 1],
        [6, 7, 1, 4, 5],
        [5, 1, 1, 2, 4],
    ]
    assert sorted(map(tuple, pacific_atlantic(heights))) == sorted(
        [(0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)]
    )

    island_grid = [
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ]
    assert max_area_of_island([row[:] for row in island_grid]) == 6

    assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2
    assert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
    assert shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]) == -1

    assert longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4
    assert longest_increasing_path([[3, 4, 5], [3, 2, 6], [2, 2, 1]]) == 4

    assert update_matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    assert update_matrix([[0, 0, 0], [0, 1, 0], [1, 1, 1]]) == [
        [0, 0, 0],
        [0, 1, 0],
        [1, 2, 1],
    ]

    snl_board = [
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 35, -1, -1, 13, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 15, -1, -1, -1, -1],
    ]
    assert snakes_and_ladders(snl_board) == 4

    assert shortest_bridge([[0, 1], [1, 0]]) == 1
    assert shortest_bridge([[0, 1, 0], [0, 0, 0], [0, 0, 1]]) == 2
    assert (
        shortest_bridge(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        == 1
    )

    assert sorted(map(tuple, all_paths_source_target([[1, 2], [3], [3], []]))) == sorted(
        [(0, 1, 3), (0, 2, 3)]
    )

    assert valid_path(3, [[0, 1], [1, 2], [2, 0]], 0, 2) is True
    assert valid_path(6, [[0, 1], [0, 2], [3, 5], [5, 4], [4, 3]], 0, 5) is False

    assert min_reorder(6, [[0, 1], [1, 3], [2, 3], [4, 0], [4, 5]]) == 3
    assert min_reorder(5, [[1, 0], [1, 2], [3, 2], [3, 4]]) == 2

    assert find_center([[1, 2], [2, 3], [4, 2]]) == 2
    assert find_center([[1, 2], [5, 1], [1, 3], [1, 4]]) == 1

    print(
        "[PASS] p11_graph_bfs_dfs_ii: 12 道图 BFS/DFS 进阶题"
        "（被围绕的区域/太平洋大西洋水流问题/岛屿的最大面积/二进制矩阵中的最短路径/"
        "矩阵中的最长递增路径/01矩阵/蛇梯棋/最短的桥/所有可能的路径/"
        "寻找图中是否存在路径/重新规划路线/找出星型图的中心节点）全部通过"
    )


if __name__ == "__main__":
    _self_test()
