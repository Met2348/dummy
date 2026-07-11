"""分类 29（Phase 3 竞赛级新分类）：计算几何 —— 叉积判断转向、线段相交、凸包构造，
以及几道以这些几何原语为基础的真实 LeetCode 题（三角形面积、矩形碰撞/最小面积）。"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations


Point = tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    """
    【custom】二维叉积：给定三个点 o（起点/公共点）、a、b，计算向量 (a-o) 和
    (b-o) 的叉积 `(a-o) x (b-o)`。符号含义：结果 > 0 表示从 o->a 转到 o->b 是
    "逆时针"（左转）；结果 < 0 表示"顺时针"（右转）；结果 == 0 表示 o、a、b
    三点共线。这是几乎所有计算几何算法（凸包、线段相交、多边形面积）的最基础
    原语。
    【思路】叉积公式：`(a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x)`。几何意义是
    "以 o 为公共顶点、a 和 b 为另外两个顶点"的平行四边形的有向面积（三角形
    o-a-b 面积的两倍，带符号）。符号由"从 o->a 方向转到 o->b 方向"是逆时针
    还是顺时针决定——这正是凸包算法（Andrew 单调链）判断"要不要弹出栈顶点"、
    线段相交判断"两点是否在一条线段的两侧"的共同依据。
    【复杂度】时间 O(1)；空间 O(1)。
    【易错点】1) 容易把 a 和 b 的角色搞反，导致符号整体取反（o->a->b 逆时针
    和 o->b->a 逆时针符号相反，虽然不影响"是否共线"的判断，但会影响"左转/
    右转"的意义，用的时候要固定好方向约定并在整个算法里保持一致）；2) 浮点数
    比较时 `== 0` 可能因为精度问题永远不成立，实际工程代码通常需要一个小的
    容差 epsilon，这里用整数/精确点坐标演示时可以直接比较。
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    【custom】判断线段 p1-p2 和线段 p3-p4 是否相交（包括端点接触、共线部分重叠
    的退化情况）。
    【思路】核心是叉积法："一般位置"下两条线段相交，当且仅当 p3、p4 分别在
    直线 p1-p2 的两侧，同时 p1、p2 也分别在直线 p3-p4 的两侧——用符号来说：
    `cross(p1,p2,p3)` 和 `cross(p1,p2,p4)` 符号相反（严格小于 0 的判据是
    `d1*d2 < 0`），并且 `cross(p3,p4,p1)` 和 `cross(p3,p4,p2)` 符号也相反。
    这个判据处理不了"某个点恰好落在另一条线段所在直线上"（共线，叉积为 0）的
    退化情况，需要额外分支：如果四个叉积中有任何一个等于 0，说明出现了共线，
    这时改用"包围盒 + 点在线段上"的方式单独检查那个共线的点是否真的落在对应
    线段的坐标范围内（`on_segment` 辅助函数：共线的前提下，只需要检查该点的
    x、y 坐标是否都落在线段两端点坐标构成的矩形范围内）。
    【复杂度】时间 O(1)（常数次叉积和比较）；空间 O(1)。
    【易错点】1) 只写"一般位置"的叉积异号判断、忘记处理共线退化情况，会导致
    "一条线段的端点恰好落在另一条线段中间"这种情况被误判为不相交；2) 共线
    情况下漏判"两条线段共线但完全不重叠"（比如 p1-p2 是 (0,0)-(1,0)，p3-p4
    是 (2,0)-(3,0)，共线但不重叠）——`on_segment` 检查必须真的验证坐标范围
    重叠，不能只要共线就判定相交。
    """

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        # 前提：p, q, r 共线（cross(p, q, r) == 0），检查 r 是否落在线段 p-q 的坐标范围内
        return (
            min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        )

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True
    return False


def convex_hull(points: list[Point]) -> list[Point]:
    """
    【custom】Andrew 单调链算法构造凸包：给定平面上一组点，返回构成凸包边界的
    点（按逆时针顺序排列，不包含严格位于凸包内部的点）。
    【思路】先把所有点按 (x, y) 字典序排序（x 相同按 y 排序）。分别构造"下凸包"
    和"上凸包"：下凸包从排序后的第一个点开始，依次尝试把每个点加入一个栈式
    结果列表；每加入一个新点之前，检查栈里最后两个点和即将加入的新点是否构成
    "顺时针"或"共线"（`cross(stack[-2], stack[-1], point) <= 0`）——如果是，
    说明栈顶这个点让当前路径往内凹（不满足凸性，逆时针方向应该一直左转），
    必须弹出栈顶，重复这个检查直到栈里少于 2 个点或者不再出现凹陷为止，再把
    新点压入栈。这和 LC84（柱状图中最大的矩形）用单调栈"栈顶不满足单调性就弹出"
    的思路是同一个模式，只是这里的"单调性"换成了"凸性"。上凸包用同样的方法
    对逆序遍历的点做一遍。最后把下凸包和上凸包拼接起来（各自去掉末尾一个点，
    避免首尾点被重复计入）就是完整凸包。
    【复杂度】时间 O(n log n)（排序占主导，单调链构造本身均摊 O(n)，每个点
    最多入栈出栈各一次）；空间 O(n)（结果列表和排序后的点列表）。
    【易错点】1) 判断"是否弹出栈顶"时如果把条件写成 `< 0`（严格顺时针才弹出）
    而不是 `<= 0`（顺时针或共线都弹出），会导致共线的中间点被错误保留在凸包
    边界上（题目通常要求凸包只保留真正的顶点，共线的中间点应被排除，除非题目
    明确要求保留边界上所有点如 LC587）；2) 输入点数少于 3 个时（0、1、2 个点）
    构不成有意义的多边形，需要单独处理，直接返回原始点列表；3) 拼接下凸包和
    上凸包时，如果不去掉各自的最后一个点，首尾两个点会被重复计入结果。
    """
    pts = sorted(set(points))
    n = len(pts)
    if n <= 2:
        return pts

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def polygon_area(points: list[Point]) -> float:
    """
    【custom】Shoelace 公式（鞋带公式）计算简单多边形面积：给定按顺序（顺时针或
    逆时针均可）排列的多边形顶点，返回多边形面积（非负值）。
    【思路】多边形面积等于"把每条边和原点组成的有向三角形面积"依次累加：
    `area = 0.5 * |sum((x_i * y_{i+1} - x_{i+1} * y_i))|`，下标按顺序循环
    （最后一个点和第一个点相连）。这个公式本质上是"用叉积累加"的推广——每一项
    `x_i*y_{i+1} - x_{i+1}*y_i` 正是 `cross(origin, p_i, p_{i+1})`，把凸包（或
    任意简单多边形）拆分成以原点为公共顶点的一系列三角形，这些三角形的有向
    面积之和恰好等于多边形本身的面积（多边形外部被重复计算的部分会因为方向
    相反而相互抵消）。
    【复杂度】时间 O(n)（遍历一遍所有顶点）；空间 O(1)。
    【易错点】1) 忘记取绝对值——如果顶点是顺时针给出的，累加和会是负数，直接
    返回负的"面积"没有意义；2) 循环到最后一个点时容易漏掉"最后一个点和第一个
    点相连"这条边，必须用 `(i+1) % n` 或等价方式回绕到下标 0。
    """
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def largest_triangle_area(points: list[Point]) -> float:
    """
    【题意】LeetCode 812·Easy。给定平面上若干点，返回任意三个不同点构成的三角形
    中面积最大的那个（答案允许 1e-5 误差）。
    【思路】题目规模很小（约束里点数 <= 50），直接三重循环枚举所有 C(n,3) 种
    三点组合，对每一组用叉积算三角形面积：以其中一点为公共顶点 o，另外两点为
    a、b，`|cross(o,a,b)| / 2` 就是三角形面积（叉积的几何意义是"平行四边形
    有向面积"，三角形面积正好是一半），取所有组合里的最大值。
    【复杂度】时间 O(n^3)（三重循环枚举所有点组合，n<=50 时完全可接受）；空间
    O(1)。
    【易错点】1) 面积公式忘记除以 2（把平行四边形面积当成了三角形面积）；
    2) 叉积结果可能为负（取决于三点顺时针还是逆时针给出），必须取绝对值，否则
    会把"顺时针的大三角形"误判成面积很小甚至负数。
    """
    best = 0.0
    for o, a, b in combinations(points, 3):
        area = abs(cross(o, a, b)) / 2.0
        if area > best:
            best = area
    return best


def min_area_rectangle(points: list[Point]) -> float:
    """
    【题意】LeetCode 939·Medium。给定平面上若干点，返回由其中四个点构成的、
    边平行于坐标轴的矩形的最小面积；如果无法构成这样的矩形，返回 0。
    【思路】轴对齐矩形的关键观察：矩形的四个顶点可以看成"两条竖直线段"——同一
    个 x 坐标下的两个不同 y 坐标的点对 (x, y1) 和 (x, y2)。如果两个不同的 x
    坐标下，都出现了同一对 (y1, y2)，那么这四个点 (x_a,y1)、(x_a,y2)、
    (x_b,y1)、(x_b,y2) 就构成一个轴对齐矩形，面积是 `|x_a-x_b| * |y1-y2|`。
    做法：按 x 坐标把点分组，每组内部按 y 排序；用一个字典记录"看到过的
    (y1,y2) 竖直线段 -> 它所在的 x 坐标"，按 x 从小到大处理每一组，组内枚举
    所有 y 的两两组合，如果这个 (y1,y2) 之前在某个更小的 x 出现过，就用两个 x
    的差乘以 y 的差算出一个候选矩形面积，更新最小值；无论是否命中，都要把
    当前 x 记录/更新到这个 (y1,y2) 对应的"最近一次出现的 x"里（因为要找的是
    最小面积，两个 x 越近越好，所以要用最近一次出现的 x，而不是第一次出现的）。
    【复杂度】时间 O(n^2)（最坏情况下每个 x 分组内点数为 O(n)，两两组合是
    O(n^2)，虽然理论上界较松，但对约束内的 n<=500 完全可接受，等价于官方题解
    的复杂度）；空间 O(n^2)（记录 (y1,y2) 对到 x 坐标的字典，最坏情况下
    每个 x 组都贡献 O(n) 对）。
    【易错点】1) 记录 "最近出现的 x" 时如果错误地保留"第一次出现的 x"而不是
    更新为"最近一次"，会漏掉面积更小的矩形组合；2) 找不到任何矩形时要返回 0
    而不是初始化用的 `float("inf")`，容易忘记在返回前做这个转换。
    """
    x_to_ys: dict[float, list[float]] = defaultdict(list)
    for x, y in points:
        x_to_ys[x].append(y)

    seen: dict[tuple[float, float], float] = {}
    best = float("inf")
    for x in sorted(x_to_ys):
        ys = sorted(x_to_ys[x])
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                y1, y2 = ys[i], ys[j]
                if (y1, y2) in seen:
                    width = x - seen[(y1, y2)]
                    height = y2 - y1
                    best = min(best, width * height)
                seen[(y1, y2)] = x
    return 0.0 if best == float("inf") else best


def min_area_rectangle_ii(points: list[Point]) -> float:
    """
    【题意】LeetCode 963·Medium。给定平面上若干点，返回由其中四个点构成的、
    边不必平行于坐标轴（可以任意旋转）的矩形的最小面积；如果无法构成这样的
    矩形，返回 0（答案允许 1e-5 误差）。
    【思路】不再能用"轴对齐"的简化假设，需要用矩形的几何性质："矩形的两条
    对角线互相平分且长度相等"——反过来，任意两条线段，如果它们的中点相同、
    长度也相同，那么这两条线段的四个端点必然构成一个矩形（这是"平行四边形对
    角线互相平分"加上"矩形对角线长度相等"两个性质的组合判据，充分且必要）。
    做法：枚举所有点对 (p_i, p_j)，把每一对的"中点坐标 + 对角线长度平方"作为
    key（用长度平方避免开根号的精度问题），分组记录属于同一个 key 的所有点对；
    对每个 key 下的多个点对，两两组合（每一对点对贡献矩形的两条对角线），
    用两条对角线的四个端点、通过叉积计算的面积公式 `|AB| * |AD|`（AB、AD 是
    以矩形一个顶点为公共点的两条邻边向量长度的乘积）算出这个矩形的面积，
    维护最小值。
    【复杂度】时间 O(n^2 log n)（枚举 O(n^2) 个点对是主要开销，对分组内的点对
    两两组合在最坏情况下可能退化到 O(n^4)，但由于矩形数量在随机/约束数据下
    通常远小于最坏理论上界，这是本题公认的标准做法；n<=50 时完全可接受）；
    空间 O(n^2)（记录所有点对的分组）。
    【易错点】1) 用中点和对角线长度分组时，如果只用长度而不用长度的平方，
    会引入浮点开方的精度误差，导致本应属于同一组的点对因为浮点误差被分到
    不同组，从而漏判有效矩形；2) 从两条对角线还原矩形四个顶点后，计算面积时
    容易用错哪两个点是"邻边"（矩形的边应该是对角线端点间"交叉"配对，不是
    对角线本身）——两条对角线为 (p1,p3) 和 (p2,p4)，矩形四条边是 p1-p2、
    p2-p3、p3-p4、p4-p1，面积可以用相邻两边长度相乘（前提是已验证是矩形，
    邻边必然垂直）。
    """
    n = len(points)
    if n < 4:
        return 0.0

    groups: dict[tuple[float, float, float], list[tuple[Point, Point]]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            p, q = points[i], points[j]
            mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
            dist_sq = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            key = (mid[0], mid[1], dist_sq)
            groups[key].append((p, q))

    best = float("inf")
    for diag_pairs in groups.values():
        if len(diag_pairs) < 2:
            continue
        for (p1, p3), (p2, p4) in combinations(diag_pairs, 2):
            side1 = math.dist(p1, p2)
            side2 = math.dist(p2, p3)
            area = side1 * side2
            if area > 0:
                best = min(best, area)
    return 0.0 if best == float("inf") else best


def circle_rectangle_overlap(
    radius: float,
    x_center: float,
    y_center: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    """
    【题意】LeetCode 1401·Medium。给定圆心 (x_center, y_center) 和半径 radius
    的圆，以及左下角 (x1,y1)、右上角 (x2,y2) 定义的轴对齐矩形，判断两者是否
    有重叠部分（包括边界接触）。
    【思路】几何直觉：判断圆和矩形是否重叠，只需要找到"矩形上离圆心最近的
    那个点"，再看这个最近点到圆心的距离是否 <= 半径——如果矩形上都没有一个
    点能落进圆里，圆和矩形必然不重叠；如果矩形上离圆心最近的点都能落进圆里，
    说明圆至少覆盖了矩形的这一个点，两者重叠。"矩形上离圆心最近的点"可以
    通过"钳制（clamp）"直接算出：把圆心的 x 坐标限制在 `[x1, x2]` 范围内（如果
    圆心 x 本来就在范围内，钳制后不变；如果偏左就钳到 x1，偏右就钳到 x2），
    y 坐标同理钳制到 `[y1, y2]`，这样得到的点 `(clamped_x, clamped_y)`
    就是矩形上离圆心欧几里得距离最近的点。
    【复杂度】时间 O(1)（钳制和距离计算都是常数次算术运算）；空间 O(1)。
    【易错点】1) 用平方距离比较（`dx*dx+dy*dy <= radius*radius`）而不是先
    开方再比较，可以避免浮点开方带来的精度损失，但如果偷懒直接用
    `math.dist(...) <= radius` 也是对的，只是要留意开方本身的浮点误差在极端
    数据下可能刚好卡在边界值；2) 钳制的方向容易写反——应该是
    `max(x1, min(x_center, x2))`（先和上界比小的，再和下界比大的），如果写成
    单纯的 `min` 或单纯的 `max` 会在圆心落在矩形范围之外时得到错误的钳制结果。
    """
    closest_x = max(x1, min(x_center, x2))
    closest_y = max(y1, min(y_center, y2))
    dx = x_center - closest_x
    dy = y_center - closest_y
    return dx * dx + dy * dy <= radius * radius


def erect_the_fence(trees: list[Point]) -> list[Point]:
    """
    【题意】LeetCode 587·Hard。给定花园里若干棵树的坐标，用最短的绳子把所有
    树围起来（绳子必须是一个不自交的封闭多边形），返回绳子经过的所有树的坐标
    ——注意和普通凸包的区别：如果有树恰好落在凸包边界的线段上（不是顶点，而是
    严格夹在两个顶点之间的共线点），这些树也必须被绳子经过（因为围栅栏的绳子
    贴着边界走，边界线段上的树自然会被绳子"路过"），所以要保留所有边界共线点，
    而不能像普通凸包那样只保留严格的转折顶点。
    【思路】复用 `convex_hull` 里 Andrew 单调链的框架，但把"弹出栈顶"的判定
    条件从 `<= 0`（顺时针或共线都弹出）改成 `< 0`（只有严格顺时针、也就是真的
    往内凹时才弹出，共线的点不弹出、允许它留在结果里）。这正好对应本文件
    `convex_hull` 文档里提到的"如果题目要求保留边界上所有点（如 LC587）"这个
    分支：这道题的语义要求就是"保留边界上所有点"，所以复用同样的单调链算法，
    只是共线判据从"弹出"改成"保留"。另外，如果所有点都共线（构不成有面积的
    多边形，退化成一条线段），应直接返回按坐标排序后的全部点。
    【复杂度】时间 O(n log n)（排序为主，单调链构造均摊线性）；空间 O(n)。
    【易错点】1) 直接复用 `convex_hull` 的 `<= 0` 判据会把边界上的共线点
    错误地弹出，导致落在栅栏边上的树没有被计入结果（这是本题和普通凸包题最
    容易被忽略的语义差异）；2) 所有点共线的退化情况需要单独处理——正常单调
    链在只有共线点时，下凸包和上凸包会各自退化成把所有点都装进去，需要验证
    这种退化路径不会引入重复点。
    """
    pts = sorted(set(trees))
    n = len(pts)
    if n <= 2:
        return pts

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    # 去重（共线情况下下凸包和上凸包可能都收纳了同一个端点）
    seen = set()
    result = []
    for p in hull:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _self_test() -> None:
    # ---- 叉积：手工构造已知转向的三点组验证符号 ----
    assert cross((0, 0), (1, 0), (1, 1)) > 0  # 逆时针（左转）
    assert cross((0, 0), (1, 1), (1, 0)) < 0  # 顺时针（右转）
    assert cross((0, 0), (2, 2), (4, 4)) == 0  # 共线
    assert cross((0, 0), (1, 0), (2, 0)) == 0  # 共线（水平线）

    # ---- 线段相交：手工构造"确定相交"和"确定不相交"的线段组 ----
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True  # 经典交叉 X 形
    assert segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)) is False  # 共线但不重叠
    assert segments_intersect((0, 0), (2, 0), (1, 0), (3, 0)) is True  # 共线且重叠
    assert segments_intersect((0, 0), (1, 1), (1, 0), (2, 1)) is False  # 平行不相交
    assert segments_intersect((0, 0), (2, 0), (1, -1), (1, 1)) is True  # T 形交叉（端点落在另一条线段中间）
    assert segments_intersect((0, 0), (1, 0), (0, 1), (1, 1)) is False  # 两条平行线段不相交

    # ---- 凸包：正方形 + 内部点，验证只保留 4 个角点 ----
    square_with_interior = [
        (0, 0), (0, 4), (4, 0), (4, 4),  # 4 个角
        (1, 1), (2, 2), (3, 1), (1, 3),  # 4 个严格内部点
    ]
    hull = convex_hull(square_with_interior)
    assert set(hull) == {(0, 0), (0, 4), (4, 0), (4, 4)}, f"凸包应恰好是4个角点，实际: {hull}"
    assert len(hull) == 4

    # 三角形 + 边上的点 + 内部点：验证边上共线点被排除（普通凸包只保留顶点）
    triangle_with_edge_point = [(0, 0), (4, 0), (0, 4), (2, 0), (1, 1)]
    hull2 = convex_hull(triangle_with_edge_point)
    assert set(hull2) == {(0, 0), (4, 0), (0, 4)}, f"凸包应只保留3个顶点，实际: {hull2}"

    # 一条直线上的点（退化情况）
    collinear = [(0, 0), (1, 1), (2, 2), (3, 3)]
    hull3 = convex_hull(collinear)
    assert set(hull3) == {(0, 0), (3, 3)}

    # ---- Shoelace 多边形面积：正方形和三角形手算验证 ----
    assert polygon_area([(0, 0), (4, 0), (4, 4), (0, 4)]) == 16.0
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == 6.0
    # 顺时针给出顶点，面积应仍为正
    assert polygon_area([(0, 4), (4, 4), (4, 0), (0, 0)]) == 16.0
    # 凸包结果的面积应该等于用暴力方式（Shoelace）算出的同一个正方形面积
    assert polygon_area(hull) == 16.0

    # ---- LC812 Largest Triangle Area ----
    assert abs(largest_triangle_area([(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)]) - 2.0) < 1e-5

    # ---- LC939 Minimum Area Rectangle ----
    assert min_area_rectangle([(1, 1), (1, 3), (3, 1), (3, 3), (2, 2)]) == 4
    assert min_area_rectangle([(1, 1), (1, 3), (3, 1), (3, 3), (4, 1), (4, 3)]) == 2
    assert min_area_rectangle([(1, 1), (1, 3), (3, 1)]) == 0

    # ---- LC963 Minimum Area Rectangle II ----
    assert abs(min_area_rectangle_ii([(1, 2), (2, 1), (1, 0), (0, 1)]) - 2.0) < 1e-5
    assert abs(min_area_rectangle_ii([(0, 1), (2, 1), (1, 1), (1, 0), (2, 0)]) - 1.0) < 1e-5
    assert min_area_rectangle_ii([(0, 3), (1, 2), (3, 1), (1, 3), (2, 1)]) == 0

    # ---- LC1401 Circle and Rectangle Overlapping ----
    assert circle_rectangle_overlap(1, 0, 0, 1, -1, 3, 1) is True
    assert circle_rectangle_overlap(1, 1, 1, 1, -3, 2, -1) is False
    assert circle_rectangle_overlap(1, 0, 0, -1, 0, 0, 1) is True
    assert circle_rectangle_overlap(1, 1, 1, -3, -3, 3, 3) is True

    # ---- LC587 Erect the Fence：正方形+内部点，边界共线点也要保留 ----
    fence_input = [(1, 1), (2, 2), (2, 0), (0, 0), (3, 3), (2, 4), (4, 2)]
    # 手工确认：这些点里 (0,0)(2,0)(4,2)(3,3)(2,4) 构成外围（含边上共线点），(1,1)(2,2) 是内部点
    fence = erect_the_fence(fence_input)
    assert (1, 1) not in fence and (2, 2) not in fence, "内部点不应出现在栅栏结果里"
    for corner in [(0, 0), (4, 2), (2, 4)]:
        assert corner in fence, f"角点 {corner} 应该在栅栏结果里"

    # 简单矩形加一个边上的共线点：验证共线点确实被保留（和普通凸包的关键区别）
    fence_simple = [(0, 0), (2, 0), (4, 0), (4, 4), (0, 4)]
    fence2 = erect_the_fence(fence_simple)
    assert (2, 0) in fence2, "边上的共线点 (2,0) 应该被栅栏绳子经过"
    hull_simple = convex_hull(fence_simple)
    assert (2, 0) not in hull_simple, "普通凸包应该排除共线的边上点"

    print(
        "[PASS] p29_computational_geometry: 9/9 项通过 "
        "(叉积/线段相交/凸包/多边形面积/812 Largest Triangle Area/"
        "939 Minimum Area Rectangle/963 Minimum Area Rectangle II/"
        "1401 Circle and Rectangle Overlapping/587 Erect the Fence)"
    )


if __name__ == "__main__":
    _self_test()
