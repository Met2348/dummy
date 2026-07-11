"""分类 08·二叉树遍历与构造 · 竞赛级补充（Part III）：不重讲框架，专攻"递归
返回值形状更复杂"和"需要额外维护辅助结构（父指针/列表/坐标）"的 9 道题，
以 Hard 深挖题收尾。

Part I/II 已经把"递归函数对子树回答什么问题、怎么组合"这套心法用了个遍。这批
题在此基础上再加一条主线：**当树本身提供的信息不够时，主动给每个节点"外挂"
一份额外信息**——可以是"父指针"（LC863，把树临时看成图）、"到根的坐标"
（LC987，把树映射到二维网格）、"上一层节点的引用"（LC116/117 的 next
指针）、"方向状态"（LC1372 的锯齿方向）。
"""
from __future__ import annotations


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = None  # 仅 LC116/LC117 使用，其余题目不涉及这个字段


def build_tree(vals: list) -> "TreeNode | None":
    """LeetCode 层序数组（None 表示空位）构造二叉树。"""
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    queue = [root]
    i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals):
            v = vals[i]
            i += 1
            if v is not None:
                node.left = TreeNode(v)
                queue.append(node.left)
        if i < len(vals):
            v = vals[i]
            i += 1
            if v is not None:
                node.right = TreeNode(v)
                queue.append(node.right)
    return root


def _tree_to_level_list(root: "TreeNode | None") -> list:
    """仅用于测试校验：把树转回层序数组（截掉末尾多余的 None）。"""
    if root is None:
        return []
    result: list = []
    queue: list = [root]
    while queue:
        node = queue.pop(0)
        if node is None:
            result.append(None)
            continue
        result.append(node.val)
        queue.append(node.left)
        queue.append(node.right)
    while result and result[-1] is None:
        result.pop()
    return result


def _preorder(root: "TreeNode | None") -> list[int]:
    """仅用于测试校验的朴素前序遍历。"""
    if root is None:
        return []
    return [root.val] + _preorder(root.left) + _preorder(root.right)


def _next_chain_from(node: "TreeNode | None") -> list[int]:
    """仅用于测试校验 LC116/LC117：从某个节点出发，沿 next 指针收集一条链。"""
    vals: list[int] = []
    while node is not None:
        vals.append(node.val)
        node = node.next
    return vals


def _find_node(root: "TreeNode | None", val: int) -> "TreeNode | None":
    """仅用于测试校验：在树里找到值等于 val 的第一个节点（假设值互不相同）。"""
    if root is None:
        return None
    if root.val == val:
        return root
    return _find_node(root.left, val) or _find_node(root.right, val)


# ── LC100 相同的树（Easy） ────────────────────────────────────────────────
def is_same_tree(p: "TreeNode | None", q: "TreeNode | None") -> bool:
    """
    【题意】给两棵二叉树的根节点 p、q，判断它们是否结构完全相同、且对应位置
        上的节点值也都相同。
    【思路】最直接的"信任递归"范例：一个节点对能不能判定为"相同"，只取决于
        三件事——1）两者是否同时为空；2）如果都非空，值是否相等；3）左子树
        对、右子树对是否分别也相同。这三条判断天然就是递归定义本身，不需要
        额外设计。
    【复杂度】时间 O(min(m, n))（一旦某一步发现不同就提前剪掉，最坏情况要
        访问两棵树中较小的那棵的所有节点），空间 O(h)（递归栈）。
    【易错点】"两者都为空"和"其中一个为空"必须分开判断——先检查都为空返回
        True，再检查只有一个为空返回 False，顺序不能反、也不能合并成一个
        条件，否则会漏掉"一个空一个非空"这种情况。
    """
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


# ── LC129 求根节点到叶节点数字之和（Medium） ──────────────────────────────
def sum_numbers(root: "TreeNode | None") -> int:
    """
    【题意】二叉树每个节点存一个 0-9 的数字，每条从根到叶子的路径都能拼成一个
        数字（比如根到叶子依次是 1→2→3，拼成 123）。求所有根到叶路径拼出的
        数字之和。
    【思路】递归函数额外带一个参数 `current`，表示"从根走到当前节点为止，已经
        拼出的数字"——每往下走一层，新的数字就是 `current * 10 + node.val`
        （十进制左移一位再加新数字，和手动拼多位数字的过程一模一样）。走到
        叶子节点时，`current` 就是这条路径完整拼出的数字，直接作为这条路径的
        贡献返回；非叶子节点则把左右子树的贡献加起来网上传。
    【复杂度】时间 O(n)，空间 O(h)（递归栈）。
    【易错点】判断"叶子节点"要同时检查 `left is None and right is None`，
        只看其中一侧会把只有单侧子树的节点错误地当成叶子提前结算；空节点
        （递归到 None）应该返回 0 而不是继续处理 `current`，否则会重复计入
        不存在的路径。
    """

    def dfs(node: "TreeNode | None", current: int) -> int:
        if node is None:
            return 0
        current = current * 10 + node.val
        if node.left is None and node.right is None:
            return current
        return dfs(node.left, current) + dfs(node.right, current)

    return dfs(root, 0)


# ── LC116 填充每个节点的下一个右侧节点指针（Medium） ──────────────────────
def connect(root: "TreeNode | None") -> "TreeNode | None":
    """
    【题意】给一棵**完美二叉树**（所有叶子在同一层，每个父节点都有两个孩子），
        每个节点多一个 `next` 指针，要求把它指向同一层的右边相邻节点（没有
        就指向 None）。要求 O(1) 额外空间（不算递归栈/输出本身）。
    【思路】如果用队列做标准的分层 BFS 当然能解决，但那是 O(n) 额外空间。
        O(1) 空间的技巧是：**利用"上一层已经连好的 next 指针"去把"下一层"
        连起来**，一层一层往下推进，不需要队列。具体地，站在某一层的最左边
        节点 `leftmost` 开始，沿着这一层已经连好的 `next` 链从左到右走一遍
        （`head = leftmost` 开始，靠 `head = head.next` 移动），对每个
        `head` 做两件事：1）`head.left.next = head.right`（同一个父节点的
        左右孩子必然相邻，直接连上）；2）如果 `head.next` 存在，
        `head.right.next = head.next.left`（当前父节点的右孩子，和"下一个
        父节点"的左孩子，在下一层里也是相邻的）。这一层处理完，`leftmost`
        下移一层（`leftmost = leftmost.left`），重复直到叶子层。
    【复杂度】时间 O(n)（每个节点的 next 指针只被设置一次），空间 O(1)
        （不计递归/输出，只用了几个指针变量）。
    【易错点】这个技巧**依赖"完美二叉树"这个前提**——正是因为每个父节点都
        必然有两个孩子，`head.left` 和 `head.right` 才能保证不为 None，
        才能放心直接访问 `.left`/`.right` 不做判空；如果树不是完美的（比如
        LC117 的场景），这套写法会在访问 `head.left`/`head.right` 时直接
        崩溃，必须换成 LC117 那种更保守的写法。
    """
    leftmost = root
    while leftmost is not None and leftmost.left is not None:
        head = leftmost
        while head is not None:
            head.left.next = head.right
            if head.next is not None:
                head.right.next = head.next.left
            head = head.next
        leftmost = leftmost.left
    return root


# ── LC117 填充每个节点的下一个右侧节点指针 II（Medium） ───────────────────
def connect_ii(root: "TreeNode | None") -> "TreeNode | None":
    """
    【题意】和 LC116 相同的目标（把每层节点用 next 指针从左到右连起来），但
        这次树**不保证是完美二叉树**——任意节点可能只有一个孩子、或者没有
        孩子，仍要求 O(1) 额外空间。
    【思路】LC116 那种"直接访问 head.left/head.right"的写法在这里不安全
        （孩子可能不存在）。改用一个更保守但同样 O(1) 空间的技巧：**用一个
        临时的 `dummy_head` 节点，把"下一层"的节点们暂时串成一条链**——
        在处理当前层的过程中，沿着当前层已经连好的 next 链从左到右扫描，
        每遇到一个非空的 `left`/`right` 孩子，就把它接到 `dummy_head` 引出
        的这条临时链的尾部（`tail.next = child; tail = tail.next`）。当前层
        扫描完毕后，`dummy_head.next` 就是下一层从左到右的完整链表（因为
        每个孩子在被接入链表的同时，实际上正是在给"下一层"的 next 指针赋
        值），把 `dummy_head` 重置、`node` 指向 `dummy_head.next`，就可以
        进入下一层重复同样的过程。
    【复杂度】时间 O(n)，空间 O(1)（`dummy_head`/`tail`/`node` 都是 O(1)
        的额外指针，不算递归栈——这里本来就是迭代写法，没有递归栈）。
    【易错点】每层开始处理前必须把 `dummy_head.next` 重置为 None，否则会
        把上一层遗留的链表残留连到新一层里；`tail` 每次接入新孩子后要立刻
        跟着移动（`tail = tail.next`），忘记移动会导致后续的孩子覆盖掉之前
        接入的孩子而不是追加在链表尾部。
    """
    node = root
    dummy_head = TreeNode(0)  # 临时头节点，用来串起"下一层"的链表
    while node is not None:
        tail = dummy_head
        while node is not None:
            if node.left is not None:
                tail.next = node.left
                tail = tail.next
            if node.right is not None:
                tail.next = node.right
                tail = tail.next
            node = node.next
        node = dummy_head.next
        dummy_head.next = None
    return root


# ── LC863 二叉树中所有距离为 K 的结点（Medium） ───────────────────────────
def distance_k(root: "TreeNode | None", target_val: int, k: int) -> list[int]:
    """
    【题意】给二叉树根节点、一个目标节点的值 target_val，以及整数 k，返回
        所有与目标节点"距离"恰好为 k 的节点值（距离定义为两节点之间需要经过
        的边数，可以往父节点方向走，也可以往孩子方向走）。
    【思路】二叉树本身只有"父指向孩子"的单向指针，没有"孩子指向父"的指针，
        导致没法直接从目标节点往上走。解法是**先给每个节点补上一份"指向父
        节点"的映射**（一次 DFS/BFS 预处理，把树临时"升级"成一张无向图：
        每个节点有 left、right、parent 三个方向的邻居）。补上这份信息之后，
        问题就变成了一次标准的"图上 BFS 求距离为 k 的节点"：从目标节点出发，
        每一轮把当前层所有节点的三个方向邻居（left、right、parent）都尝试
        加入下一层（跳过已访问过的，避免走回头路），走 k 轮之后，当前这一层
        剩下的节点就是答案。
    【复杂度】时间 O(n)（建父指针映射 O(n) + BFS 最坏遍历所有节点 O(n)），
        空间 O(n)（父指针映射 + visited 集合 + BFS 队列）。
    【易错点】BFS 时必须把"父节点方向"也当成一个可以走的邻居，只考虑
        `left`/`right` 会退化成普通子树 BFS，找不到目标节点祖先方向的答案；
        必须维护一个 `visited` 集合防止往回走（比如从父节点走到目标节点的
        父节点后，又想沿着这个父节点的 `left` 走回目标节点自己），否则会
        死循环或者重复计数。
    """
    parent: dict[TreeNode, "TreeNode | None"] = {}

    def build_parent(node: "TreeNode | None", par: "TreeNode | None") -> None:
        if node is None:
            return
        parent[node] = par
        build_parent(node.left, node)
        build_parent(node.right, node)

    build_parent(root, None)
    target_node = _find_node(root, target_val)

    visited = {target_node}
    queue = [target_node]
    dist = 0
    while queue and dist < k:
        next_queue: list[TreeNode] = []
        for node in queue:
            for neighbor in (node.left, node.right, parent[node]):
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    next_queue.append(neighbor)
        queue = next_queue
        dist += 1
    return [node.val for node in queue]


# ── LC1123 最深叶节点的最近公共祖先（Medium） ─────────────────────────────
def lca_deepest_leaves(root: "TreeNode | None") -> "TreeNode | None":
    """
    【题意】给二叉树根节点，找出"所有深度最大的叶子节点"的最近公共祖先。
    【思路】后序遍历，让每次递归调用同时返回两件事：`(这棵子树的最大深度,
        这棵子树里"最深叶子们"的最近公共祖先)`。组合逻辑是本题的核心——比较
        左右子树各自算出的最大深度：如果左右深度相等，说明"全局最深的叶子"
        左右子树里都有，当前节点自然就是它们的最近公共祖先（因为再往上任何
        一个祖先都不如当前节点更"近"）；如果左边更深，说明最深的叶子全部在
        左子树里，最近公共祖先就是左子树递归算出来的那个 LCA（原样网上传，
        当前节点不参与）；右边更深同理。
    【复杂度】时间 O(n)，空间 O(h)（递归栈）。
    【易错点】比较的是"左右子树各自的最大深度"，不是"左右子树是否都非空"——
        即使左右子树都非空，只要深度不相等，答案也不会是当前节点，而是更深
        那一侧递归传上来的 LCA；空节点的深度约定为 0（不是 -1 或其他值），
        这样叶子节点的深度自然算出来是 1，和"根深度为 0"的定义体系保持一致。
    """

    def dfs(node: "TreeNode | None") -> tuple[int, "TreeNode | None"]:
        if node is None:
            return 0, None
        left_depth, left_lca = dfs(node.left)
        right_depth, right_lca = dfs(node.right)
        if left_depth == right_depth:
            return left_depth + 1, node
        if left_depth > right_depth:
            return left_depth + 1, left_lca
        return right_depth + 1, right_lca

    _, lca = dfs(root)
    return lca


# ── LC998 最大二叉树 II（Medium） ─────────────────────────────────────────
def insert_into_max_tree(root: "TreeNode | None", val: int) -> "TreeNode | None":
    """
    【题意】"最大树"定义为：树中每个节点的值都大于它子树中所有其他节点的值
        （根是全局最大值）。给一棵由数组 a 构造出的最大树 root，以及在 a
        末尾追加一个新数字 val 后的新数组 b，返回 b 对应的最大树。
    【思路】关键观察：因为 val 是**新追加在数组末尾**的，它在"最大树的构造
        规则"（取区间最大值当根，左边分左子树、右边分右子树）里只可能出现在
        "整棵树最右侧那条链"上——不会插到任何节点的左子树里。于是只需要沿着
        `root` 的右孩子链往下走：如果 val 比当前节点的值还大，说明 val 应该
        成为**当前这整棵（子）树的新根**，原来的（子）树整体变成 val 的左
        子树（因为原树的值全都更小，只能屈居左边）；否则说明 val 应该插在
        更右边，递归地在 `root.right` 这棵子树里继续插入。
    【复杂度】时间 O(h)（h 是这条最右链的长度，最坏 O(n)），空间 O(h)
        （递归栈；也可以改写成迭代做到 O(1) 额外空间）。
    【易错点】判断条件是 `root is None or root.val < val`——`root is None`
        这个终止条件不能漏（说明已经走到最右链的末尾，val 直接成为新的叶子）；
        val 成为新根时，新节点的 `right` 应该保持 None（val 是数组最后一个
        元素，它的右边没有更多数字了），只有 `left` 挂上原来的整棵子树。
    """
    if root is None or root.val < val:
        new_root = TreeNode(val)
        new_root.left = root
        return new_root
    root.right = insert_into_max_tree(root.right, val)
    return root


# ── LC1372 二叉树中的最长交错路径（Medium） ───────────────────────────────
def longest_zigzag(root: "TreeNode | None") -> int:
    """
    【题意】从树中任意节点出发，选一个初始方向（左或右），每一步都要往选定
        方向的孩子走，然后**方向必须反转**（左变右、右变左），一直走到走不
        动为止。这样一条"锯齿路径"的长度定义为经过的边数。求整棵树里最长的
        锯齿路径。
    【思路】递归函数额外带两个参数：`went_left`（表示"刚才这一步是往左走的
        还是往右走的"）和 `length`（表示"到当前节点为止，这条锯齿路径已经
        累积的边数"）。在每个节点上，都有两种延续方式：1）**保持锯齿、方向
        反转**——如果刚才是往左走的，现在必须往右走，长度 `+1`；2）**放弃
        当前这条锯齿、从当前节点开始重新起跳**——往同一个方向再走一步，但
        这条新路径长度只能算 1（因为方向没有反转，锯齿链在这里断开重新计数）。
        每个节点都要同时尝试这两种延续，因为"最优的锯齿路径"不一定要用满
        每一层的反转，也可能在中途重新起跳反而更划算。根节点本身两个方向都
        要各自入口探一次（`dfs(root, True, 0)` 和 `dfs(root, False, 0)`），
        因为根节点没有"上一步方向"这个概念，需要用长度 0 作为两种方向共同的
        起点。
    【复杂度】时间 O(n)（每个节点最多被以常数种状态访问，不是指数级——虽然
        每个节点触发两条递归分支，但因为其中一条分支的 `went_left` 参数值
        和另一条互斥，不会产生指数级重复访问），空间 O(h)（递归栈）。
    【易错点】"重新起跳"的分支长度必须重置成 1（不是延续 `length`），这是
        本题最容易漏掉的一步——如果两个分支都用 `length + 1`，会把"方向没有
        反转、锯齿其实已经断开"的情况也当成锯齿的延续来计数，得到偏大的错误
        答案；根节点的两次入口调用都要做（只做一次会漏掉"从根节点往另一个
        方向出发"的路径）。
    """
    best = 0

    def dfs(node: "TreeNode | None", went_left: bool, length: int) -> None:
        nonlocal best
        if node is None:
            return
        best = max(best, length)
        if went_left:
            dfs(node.right, False, length + 1)  # 方向反转，锯齿继续
            dfs(node.left, True, 1)              # 方向不变，锯齿断开重新起跳
        else:
            dfs(node.left, True, length + 1)
            dfs(node.right, False, 1)

    dfs(root, True, 0)
    dfs(root, False, 0)
    return best


# ── LC987 二叉树的垂序遍历（Hard） ────────────────────────────────────────
def vertical_traversal(root: "TreeNode | None") -> list[list[int]]:
    """
    【题意】把二叉树画在一个二维网格上：根节点位于 (row=0, col=0)，某节点的
        左孩子位于 (row+1, col-1)，右孩子位于 (row+1, col+1)。按列从左到右、
        同一列内按行从上到下，返回每一列的节点值列表；**如果同一列同一行
        出现多个节点（多个节点恰好落在完全相同的坐标上），这些节点必须按
        节点值升序排列**。
    【思路】做法见本文件对应 lecture 的深挖部分——核心是一次 DFS 收集每个
        节点的 `(col, row, val)` 三元组，然后**把元组按 `(col, row, val)`
        的顺序整体排序**（不是先按 col 分组、组内再各自排序，而是直接对
        整个三元组列表做一次排序）：Python 对 tuple 的默认排序规则是"先比较
        第一个元素，相等再比较第二个，再相等才比较第三个"，这恰好和题目
        "先按列分组、组内按行排序、行也相同时按值排序"这套三级排序规则完全
        对应，一次排序调用就能把三条规则都满足，不需要手写分组再排序的
        额外逻辑。排序完成后，只需要顺序扫一遍，"col 值和上一个不同"就开启
        一个新的分组。
    【复杂度】时间 O(n log n)（一次 DFS O(n) + 一次排序 O(n log n)），
        空间 O(n)（存储所有节点的三元组）。
    【易错点】这题最容易出错的地方就是"同一行同一列多个节点"的打破平局
        规则——如果三元组只存 `(col, row)`、把 val 单独放在旁边不参与排序，
        或者用字典按 `(row, col)` 分组后不再对组内的 val 排序，都会在"两个
        节点恰好落在同一个坐标"时得到和官方顺序不一致的结果（这种情况通常
        发生在一个节点是另一个节点的"左孩子的右孩子"和"右孩子的左孩子"这类
        对称路径上）；元组的字段顺序必须是 `(col, row, val)`，而不是
        `(row, col, val)`——col 必须放在最前面才能保证排序优先按列分组。
    """
    items: list[tuple[int, int, int]] = []

    def dfs(node: "TreeNode | None", row: int, col: int) -> None:
        if node is None:
            return
        items.append((col, row, node.val))
        dfs(node.left, row + 1, col - 1)
        dfs(node.right, row + 1, col + 1)

    dfs(root, 0, 0)
    items.sort()

    result: list[list[int]] = []
    prev_col: int | None = None
    for col, _row, val in items:
        if col != prev_col:
            result.append([])
            prev_col = col
        result[-1].append(val)
    return result


def _self_test() -> None:
    assert is_same_tree(build_tree([1, 2, 3]), build_tree([1, 2, 3])) is True
    assert is_same_tree(build_tree([1, 2]), build_tree([1, None, 2])) is False

    assert sum_numbers(build_tree([1, 2, 3])) == 25
    assert sum_numbers(build_tree([4, 9, 0, 5, 1])) == 1026

    tree116 = build_tree([1, 2, 3, 4, 5, 6, 7])
    connect(tree116)
    assert _next_chain_from(tree116) == [1]
    assert _next_chain_from(tree116.left) == [2, 3]
    assert _next_chain_from(tree116.left.left) == [4, 5, 6, 7]

    tree117 = build_tree([1, 2, 3, 4, 5, None, 7])
    connect_ii(tree117)
    assert _next_chain_from(tree117) == [1]
    assert _next_chain_from(tree117.left) == [2, 3]
    assert _next_chain_from(tree117.left.left) == [4, 5, 7]

    tree863 = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    assert sorted(distance_k(tree863, 5, 2)) == sorted([7, 4, 1])

    assert lca_deepest_leaves(
        build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    ).val == 2
    assert lca_deepest_leaves(build_tree([1])).val == 1
    assert lca_deepest_leaves(build_tree([0, 1, 3, None, 2])).val == 2

    inserted998 = insert_into_max_tree(build_tree([4, 1, 3, None, None, 2]), 5)
    assert _tree_to_level_list(inserted998) == [5, 4, None, 1, 3, None, None, 2]
    inserted998b = insert_into_max_tree(build_tree([5, 2, 4, None, 1]), 3)
    assert _tree_to_level_list(inserted998b) == [5, 2, 4, None, 1, None, 3]

    assert longest_zigzag(build_tree(
        [1, None, 1, 1, 1, None, None, 1, 1, None, 1, None, None, None, 1]
    )) == 3
    assert longest_zigzag(build_tree(
        [1, 1, 1, None, 1, None, None, 1, 1, None, 1]
    )) == 4

    assert vertical_traversal(build_tree([3, 9, 20, None, None, 15, 7])) == [
        [9], [3, 15], [20], [7]
    ]
    assert vertical_traversal(build_tree([1, 2, 3, 4, 5, 6, 7])) == [
        [4], [2], [1, 5, 6], [3], [7]
    ]
    assert vertical_traversal(build_tree([1, 2, 3, 4, 6, 5, 7])) == [
        [4], [2], [1, 5, 6], [3], [7]
    ]

    print(
        "[PASS] p08_binary_tree_iii: 9 题"
        "（相同的树/求根到叶数字之和/填充next右侧指针/填充next右侧指针II/"
        "所有距离为K的结点/最深叶节点的最近公共祖先/最大二叉树II/"
        "最长交错路径/垂序遍历）全部通过"
    )


if __name__ == "__main__":
    _self_test()
