"""14 贪心 · 进阶补充(Part II)：扩大"区间贪心"和"贪心+单调栈/树形贪心"的覆盖面，
不重复 Part I 已经讲过的框架，只挑变体。"""
from __future__ import annotations

import heapq
from collections import deque


# ── LC56 合并区间(Medium) ────────────────────────────────────────────────
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    【题意】给定若干区间的列表，合并所有重叠的区间，返回一个不重叠区间的列表
    （区间按什么顺序给出不作保证）。
    【思路】重叠只可能发生在"起点相邻"的区间之间，所以先按**起点**排序，排序后
    只需要线性扫描一次：维护一个"当前正在合并的区间" cur，如果下一个区间的起点
    <= cur 的终点，说明有重叠（或恰好相接），把 cur 的终点更新为
    max(cur终点, 下一个区间终点)；否则说明 cur 已经"合完"，把它放进结果，
    开启一个新的 cur。贪心成立的原因：排序之后，只要当前区间管不到下一个区间的
    起点，后面任何区间也不可能再跟 cur 重叠（起点只会越来越大），所以 cur 可以
    放心地"收尾"。
    【复杂度】时间 O(n log n)（排序主导），空间 O(n)（存结果，不计排序本身开销）。
    【易错点】容易只判断"当前区间和上一个原始区间"是否重叠，而不是和"当前已经
    合并到的区间"比较——如果 [1,4] 和 [2,3] 合并成 [1,4] 后，接下来要比较的应该是
    [1,4] 的终点而不是 [2,3] 的终点，否则会漏合并。
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


# ── LC57 插入区间(Medium) ────────────────────────────────────────────────
def insert_interval(
    intervals: list[list[int]], new_interval: list[int]
) -> list[list[int]]:
    """
    【题意】给定一组已经按起点排好序且互不重叠的区间，再给一个新区间，把新区间
    插入进去并合并所有必要的重叠区间，返回合并后仍然有序、互不重叠的区间列表。
    【思路】既然原区间已经有序，就不需要重新整体排序，只需要把区间分成三段贪心
    处理：① 完全在新区间左边、不重叠的区间（终点 < 新区间起点），原样放入结果；
    ② 和新区间有重叠的区间（起点 <= 新区间当前终点），不断把它们"吃"进新区间——
    用 min 扩展新区间起点、用 max 扩展新区间终点，这一步做完把扩展后的新区间整体
    放入结果；③ 剩下完全在新区间右边、不重叠的区间，原样放入结果。三段贪心的
    正确性来自"原区间互不重叠且有序"这个前提——一旦某个区间不再和当前扩展中的
    新区间重叠，后面的区间（起点更大）也不可能再重叠。
    【复杂度】时间 O(n)（一次线性扫描，n 为原区间个数），空间 O(n)（存结果）。
    【易错点】① 判断"是否重叠"要用新区间**当前已经被扩展过**的 start/end，而不是
    题目传入的原始 new_interval——扩展是动态累积的；② 第一段的终止条件是
    `intervals[i][1] < start`（严格小于，端点相接不算不重叠，需要合并）。
    """
    result: list[list[int]] = []
    i, n = 0, len(intervals)
    start, end = new_interval
    while i < n and intervals[i][1] < start:
        result.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= end:
        start = min(start, intervals[i][0])
        end = max(end, intervals[i][1])
        i += 1
    result.append([start, end])
    while i < n:
        result.append(intervals[i])
        i += 1
    return result


# ── LC452 用最少数量的箭引爆气球(Medium) ─────────────────────────────────
def find_min_arrow_shots(points: list[list[int]]) -> int:
    """
    【题意】气球是水平方向的区间 [start, end]，一支箭垂直射出可以扎爆所有"横坐标
    区间包含箭所在横坐标"的气球，求引爆所有气球最少需要的箭数。
    【思路】和 LC435 无重叠区间是同一个"活动选择"母题：按**结束坐标**排序，贪心地
    让每一支箭都尽量射在"当前这批还没被扎爆的气球里，结束最早的那个气球的结束点"
    上——因为这一箭同时能扎爆所有和它重叠的气球（起点 <= 这支箭的位置），并且把
    箭放在"结束最早"的位置能让这一箭覆盖尽量多、同时不耽误后面的气球。扫描时维护
    当前箭的位置 arrow_end；如果下一个气球的起点 > arrow_end，说明这支箭已经扎不到
    它了，必须新开一箭，并把新箭的位置设为这个气球的结束坐标。
    【复杂度】时间 O(n log n)（排序主导），空间 O(1)（不计排序开销）。
    【易错点】容易和 LC435 混淆按哪个维度排序或返回值语义——LC435 返回"要删除几个
    区间"（保留的区间数 = n - 返回值），这题直接返回"需要几支箭"，且判断"能否被
    同一箭扎爆"要用非严格的 `start <= arrow_end`（区间闭合，端点相切也算重叠）。
    """
    if not points:
        return 0
    ordered = sorted(points, key=lambda p: p[1])
    arrows = 1
    arrow_end = ordered[0][1]
    for start, end in ordered[1:]:
        if start > arrow_end:
            arrows += 1
            arrow_end = end
    return arrows


# ── LC861 翻转矩阵后的得分(Medium) ───────────────────────────────────────
def matrix_score(grid: list[list[int]]) -> int:
    """
    【题意】0/1 矩阵，每行、每列都可以整体翻转任意次（0 变 1，1 变 0），每行看成
    一个二进制数，求所有翻转操作后能得到的"各行二进制数之和"的最大值。
    【思路】两层贪心，分别针对最高位（行）和其余位（列）：① 二进制数里最高位
    权重最大，所以第一步贪心是保证每一行的最高位（第一列）都是 1——如果某一行
    首位是 0，翻转整行必然让首位变 1，且不会让该行变差（首位从 0 变 1 是所有位
    翻转里收益最大的，翻转整行才能同时把首位翻正）。② 行首位固定为 1 之后，
    每一列的操作和其它列相互独立，对某一列而言，1 的数量越多这一列对总和的贡献
    越大，所以贪心地对每一列单独判断：如果这一列里 0 比 1 多，翻转整列能让 1
    的数量反超，翻转；否则不翻转。两步贪心互不冲突（行翻转只发生在预处理阶段，
    列翻转只发生在此后），所以可以分两个阶段独立完成。
    【复杂度】时间 O(rows * cols)，空间 O(1)（原地翻转，除结果外不额外分配，
    这里为了不破坏调用者传入的原始数据先复制了一份）。
    【易错点】容易颠倒两步贪心的顺序——如果先按列贪心、再按行翻转，行翻转会把
    刚刚调整好的列比例全部打乱，必须先固定"每行首位是 1"这个更高优先级的约束，
    再处理列。
    """
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        if grid[r][0] == 0:
            for c in range(cols):
                grid[r][c] ^= 1
    for c in range(cols):
        ones = sum(grid[r][c] for r in range(rows))
        if ones < rows - ones:
            for r in range(rows):
                grid[r][c] ^= 1
    return sum(int("".join(map(str, row)), 2) for row in grid)


# ── LC1046 最后一块石头的重量(Easy) ──────────────────────────────────────
def last_stone_weight(stones: list[int]) -> int:
    """
    【题意】每次任选两块最重的石头相撞：若重量相等，两块都碎；不等则较重的石头
    剩下"重量差"。反复操作直到最多剩一块石头，返回它的重量（没有剩余则返回 0）。
    【思路】"每次都要取当前最重的两块"这个操作本身就是堆的定义，不需要额外证明
    什么贪心性质——用一个最大堆（Python 用取负数模拟）维护所有石头，每次弹出堆顶
    两个最大值相撞，把差值（如果非零）重新压回堆里，直到堆里只剩 0 或 1 个元素。
    【复杂度】时间 O(n log n)（n 次弹出/压入，每次 O(log n)），空间 O(n)。
    【易错点】容易忘记"两块相撞后重量相等则都消失，不需要压回任何东西"这一分支，
    只处理了"不等时压回差值"，导致相等的情况被误当成压回了 0（虽然数值上不影响
    最终答案，但会让堆里多出无意义的 0 元素，浪费但不算错——真正的错误常发生在
    忘记判断堆是否为空就直接取 heap[0]）。
    """
    heap = [-s for s in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)
        if first != second:
            heapq.heappush(heap, -(first - second))
    return -heap[0] if heap else 0


# ── LC135 分发糖果(Hard) ─────────────────────────────────────────────────
def candy(ratings: list[int]) -> int:
    """
    【题意】n 个孩子站成一排，每人有一个评分 ratings[i]。分糖果规则：每人至少
    一颗；评分比两侧邻居高的孩子，糖果必须比对应邻居多。求满足规则的最少糖果总数。
    【思路】这个约束同时涉及"比左边邻居高"和"比右边邻居高"两个方向，一次遍历
    没法同时兼顾（只看左边邻居调整，会破坏和右边邻居的关系，反之亦然），所以拆成
    两次单方向贪心：① 从左到右扫描，只保证"比左边评分高就必须比左边多"——
    candies[i] = candies[i-1] + 1（当 ratings[i] > ratings[i-1]）；② 从右到左
    扫描，只保证"比右边评分高就必须比右边多"——但这次不能直接赋值，因为第一次
    扫描已经给这个位置分过糖果了，必须取 max(现有糖果数, candies[i+1] + 1)，
    否则会打破第一步已经满足的"比左边多"这个约束。两次扫描各自独立满足一个方向
    的约束，取两者的较大值就能同时满足两个方向。
    【复杂度】时间 O(n)（两次线性扫描），空间 O(n)（糖果数组）。
    【易错点】第二次扫描（从右到左）最容易写成直接赋值
    `candies[i] = candies[i+1] + 1`，忘记和第一次扫描的结果取 max——这样会
    悄悄地让"比左边评分高"这个第一次扫描已经保证的约束在某些位置失效。
    """
    n = len(ratings)
    candies = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    return sum(candies)


# ── LC402 移掉K位数字(Medium) ────────────────────────────────────────────
def remove_k_digits(num: str, k: int) -> str:
    """
    【题意】给定字符串形式的非负整数 num 和整数 k，从中移除 k 位数字，使得剩下
    的数字组成的新数最小（结果不能有多余的前导零，除非结果本身就是 "0"）。
    【思路】要让剩下的数最小，核心贪心原则是"高位数字越小，整个数越小"——所以
    从左到右扫描，用一个"单调不递减"的栈维护结果的候选数字：只要当前数字比栈顶
    小，且还有删除次数可用，就应该把栈顶弹出（相当于删掉那个更大的高位数字，让
    当前更小的数字顶上来占据更高的位），因为"删掉一个更大的高位换一个更小的高位"
    永远不会让结果变差。扫描结束后，如果 k 还有剩余（说明整个序列是单调不减的，
    比如 "12345" 删 2 位），只能从末尾删（末尾是最低位，对整体数值影响最小）。
    最后要 strip 掉前导零，并且如果 strip 之后是空串，说明结果是 0。
    【复杂度】时间 O(n)（n = len(num)，每个字符最多入栈出栈各一次），空间 O(n)。
    【易错点】① 栈顶弹出条件必须同时检查 `k > 0`——k 用完之后即使栈顶更大也不能
    再删；② 处理完主循环后如果 k 仍 > 0，要从栈**末尾**截掉剩余的 k 个（而不是
    忘记这一步，只在循环内删）；③ 结果 strip 前导零后如果变成空字符串，必须
    返回 "0" 而不是 ""。
    """
    stack: list[str] = []
    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    if k > 0:
        stack = stack[:-k]
    result = "".join(stack).lstrip("0")
    return result if result else "0"


# ── LC968 监控二叉树(Hard) ───────────────────────────────────────────────
class TreeNode:
    """本文件内独立定义的最小二叉树节点，字段与标准 LeetCode TreeNode 一致。"""

    def __init__(self, val: int = 0, left: "TreeNode | None" = None, right: "TreeNode | None" = None) -> None:
        self.val = val
        self.left = left
        self.right = right


def build_tree(values: list[int | None]) -> TreeNode | None:
    """按 LeetCode 标准层序（显式 None 表示空节点，空节点不再展开子节点）构建二叉树。"""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue: deque[TreeNode] = deque([root])
    i = 1
    n = len(values)
    while queue and i < n:
        node = queue.popleft()
        if i < n:
            left_val = values[i]
            i += 1
            if left_val is not None:
                node.left = TreeNode(left_val)
                queue.append(node.left)
        if i < n:
            right_val = values[i]
            i += 1
            if right_val is not None:
                node.right = TreeNode(right_val)
                queue.append(node.right)
    return root


def min_camera_cover(root: TreeNode | None) -> int:
    """
    【题意】给定一棵二叉树，在若干节点上安装摄像头，每个摄像头可以监控它自己、
    父节点和左右孩子。求覆盖整棵树所有节点最少需要多少个摄像头。
    【思路】这是"树形贪心"：从叶子往根后序遍历，每个节点根据左右子树的状态决定
    自己该不该装摄像头。给每个节点定义三种状态：0=未被覆盖，1=已被覆盖但自己
    没装摄像头，2=自己装了摄像头。后序遍历时：① 如果左右孩子中有任意一个状态是
    "未被覆盖"，那么当前节点必须装摄像头（状态变成 2，且计数 +1）——因为如果
    不装，那个未被覆盖的孩子就永远没机会被覆盖了（它自己的孩子已经在更早的递归
    里处理过，唯一能覆盖它的只剩它的父节点，也就是当前节点）；② 否则如果左右
    孩子中有任意一个装了摄像头，当前节点就被那个摄像头覆盖了，状态是"已覆盖但
    自己没装"（状态 1）；③ 否则（两个孩子都是"已覆盖无摄像头"）当前节点自己
    既没装摄像头也没被孩子的摄像头覆盖，状态是"未覆盖"（状态 0），把这个决策
    权交给它的父节点。这里的贪心之处在于：**叶子节点优先被判定为"未覆盖"**（叶子
    的左右孩子都是 None，None 按约定视为"已覆盖"，两个 None 都是"已覆盖无
    摄像头"，走到第③支，状态 0）——这就"逼"着叶子的父节点必须装摄像头，而不是
    在叶子上装。这样做是最优的：在叶子上装摄像头只能覆盖"叶子+它的父节点"两层，
    而在叶子的父节点上装摄像头能覆盖"父节点+它的父节点+左右两个叶子孩子"更多层，
    同一个摄像头覆盖的节点数更多，用摄像头的数量自然更少。递归结束后，如果根节点
    最终状态是"未覆盖"（说明根节点自己也没人能罩着它），还要再补一个摄像头。
    【复杂度】时间 O(n)（每个节点访问一次），空间 O(h)（h 为树高，递归栈开销）。
    【易错点】① 空节点（None）必须返回"已覆盖"状态而不是"未覆盖"——否则每个
    叶子节点都会误判成"孩子未覆盖"从而被迫装摄像头，摄像头数量会远超最优解；
    ② 根节点的最终状态如果是"未覆盖"，容易忘记在返回前再补一个摄像头（根节点
    没有父节点能替它兜底）。
    """
    count = 0
    NOT_COVERED, COVERED, HAS_CAMERA = 0, 1, 2

    def dfs(node: TreeNode | None) -> int:
        nonlocal count
        if node is None:
            return COVERED
        left = dfs(node.left)
        right = dfs(node.right)
        if left == NOT_COVERED or right == NOT_COVERED:
            count += 1
            return HAS_CAMERA
        if left == HAS_CAMERA or right == HAS_CAMERA:
            return COVERED
        return NOT_COVERED

    root_state = dfs(root)
    return count + (1 if root_state == NOT_COVERED else 0)


def _self_test() -> None:
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [
        [1, 6],
        [8, 10],
        [15, 18],
    ]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]

    assert insert_interval([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert insert_interval(
        [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]
    ) == [[1, 2], [3, 10], [12, 16]]

    assert find_min_arrow_shots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
    assert find_min_arrow_shots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
    assert find_min_arrow_shots([[1, 2], [2, 3], [3, 4], [4, 5]]) == 2

    assert matrix_score([[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0]]) == 39

    assert last_stone_weight([2, 7, 4, 1, 8, 1]) == 1
    assert last_stone_weight([1]) == 1

    assert candy([1, 0, 2]) == 5
    assert candy([1, 2, 2]) == 4

    assert remove_k_digits("1432219", 3) == "1219"
    assert remove_k_digits("10200", 1) == "200"
    assert remove_k_digits("10", 2) == "0"

    assert min_camera_cover(build_tree([0, 0, None, 0, 0])) == 1
    assert (
        min_camera_cover(build_tree([0, 0, None, 0, None, 0, None, None, 0])) == 2
    )

    print(
        "[PASS] p14_greedy_ii: 8 题(合并区间/插入区间/最少箭/翻转矩阵得分/"
        "最后一块石头/分发糖果/移掉K位数字/监控二叉树)全部通过"
    )


if __name__ == "__main__":
    _self_test()
