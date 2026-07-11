"""分类 08·二叉树遍历与构造 —— 7 道题。

二叉树递归题的通用心法只有一句话：**想清楚"这个函数对一棵子树做什么、需要子树
返回什么信息"，剩下的就是把左右子树的结果组合起来**。每道题的 docstring 都会先
点出"这次要子树返回什么信息"，再讲怎么组合。
"""
from __future__ import annotations


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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


def tree_to_level_list(root: "TreeNode | None") -> list:
    """把树转回层序数组，方便和期望结果比较。

    只有"真实存在的节点"才会把自己的两个孩子（哪怕是 None）入队；一个 None 节点
    本身不会再展开出下一层的 None——这样序列化出来的末尾不会有无限多余的 None，
    只需要再把结尾处连续的 None 截掉即可，和 build_tree 的输入格式一一对应。
    """
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


def find_node(root: "TreeNode | None", val) -> "TreeNode | None":
    """按层序遍历找到第一个 val 等于给定值的节点（用于测试里按值取节点引用）。"""
    if root is None:
        return None
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.val == val:
            return node
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return None


# ── LC94 二叉树的中序遍历（Easy） ─────────────────────────────────────────
def _inorder_recursive(root: "TreeNode | None") -> list[int]:
    """递归版：代码最短、最直白，但每层调用都占一份调用栈，树很深时可能栈溢出。"""
    if root is None:
        return []
    return _inorder_recursive(root.left) + [root.val] + _inorder_recursive(root.right)


def inorder_traversal(root: "TreeNode | None") -> list[int]:
    """
    【题意】给二叉树根节点，返回它的中序遍历结果（左子树 → 根 → 右子树）。
    【思路】递归版天然符合"左-根-右"的定义，是理解语义的最快方式；但真实工程/面试
        里更常考迭代版——用一个显式栈模拟"一路往左走到底，走不动了就弹出访问，
        再转向右子树"的过程：先把当前节点和它所有的左孩子依次压栈（对应"能往左
        走就往左走"），栈顶弹出即访问（这一刻左边已经没有更小的了），然后转向
        该节点的右子树重复这个过程。两种写法**递归转迭代的关键 insight** 是：
        递归调用栈本来就是隐式地在做这件事，迭代版只是把这个隐式栈换成显式的
        Python list。本文件主函数用迭代版（工程上更常用、不受栈深度限制），
        递归版作为 `_inorder_recursive` 保留用于理解语义。
    【复杂度】时间 O(n)（每个节点入栈出栈各一次），空间 O(h)（h 为树高，最坏 O(n)）。
    【易错点】迭代版容易忘记"访问完当前节点后要转向它的右子树"这一步，导致漏掉
        右子树；空树直接返回空列表，不要忘记单独处理。
    """
    result: list[int] = []
    stack: list[TreeNode] = []
    node = root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result


# ── LC104 二叉树的最大深度（Easy） ────────────────────────────────────────
def max_depth(root: "TreeNode | None") -> int:
    """
    【题意】给二叉树根节点，返回从根到最远叶子节点的最长路径上的节点数。
    【思路】要求这个函数对一棵子树做什么：**返回"这棵子树的最大深度"**。空树深度
        为 0；非空树的深度 = 1（自己这一层）+ 左右子树深度的较大值。递归只需要
        "相信"左右子树已经算好了各自的深度，直接组合即可，不需要手动维护计数器。
    【复杂度】时间 O(n)，空间 O(h)（递归调用栈）。
    【易错点】忘记 +1（把当前节点自己这一层算漏）；把 max 写成 sum（那是"节点总数"
        不是"深度"）。
    """
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


# ── LC226 翻转二叉树（Easy） ──────────────────────────────────────────────
def invert_tree(root: "TreeNode | None") -> "TreeNode | None":
    """
    【题意】给二叉树根节点，把树中每个节点的左右子树互换，返回翻转后的根节点。
    【思路】要求这个函数对一棵子树做什么：**把这棵子树翻转好，并返回翻转后的根**。
        先分别递归翻转左子树、右子树（这一步"相信"递归已经把更深层都翻转好了），
        再把当前节点自己的 left/right 互换。递归到叶子/空节点时天然是"翻转后
        还是自己"，是最简单的递归终止条件（base case），无需额外特判。
    【复杂度】时间 O(n)（每个节点访问一次），空间 O(h)。
    【易错点】只交换了子节点的值而没有交换子树引用（值交换在有些题里凑巧也能通过
        小样例，但语义是错的，树的结构没变）；忘记对空节点提前 return None，会在
        None.left 处报错。
    """
    if root is None:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


# ── LC102 二叉树的层序遍历（Medium） ──────────────────────────────────────
def level_order(root: "TreeNode | None") -> list[list[int]]:
    """
    【题意】给二叉树根节点，按"从上到下、每一层从左到右"分组返回节点值，
        结果是"每层一个子列表"的二维列表。
    【思路】这是 BFS 的标准应用：普通 BFS 只关心访问顺序，这题多了一个要求——
        "按层分组"。技巧是在每一轮 while 循环开始时，先记录当前队列的长度
        `level_size`（这就是"这一层还剩几个节点没处理"），然后只处理这么多个
        节点、把它们的值收进当前层的列表，处理过程中把它们的孩子依次入队——
        这些孩子自然而然会形成"下一层"，不会和当前层混在一起。
    【复杂度】时间 O(n)，空间 O(n)（最坏情况下最后一层能有 n/2 个节点在队列里）。
    【易错点】不记录 level_size、直接对当前动态变化的队列长度做循环，会把下一层
        的节点也算进当前层；空树要单独返回空列表，不要返回 [[]]。
    """
    if root is None:
        return []
    result: list[list[int]] = []
    queue: list[TreeNode] = [root]
    while queue:
        level_size = len(queue)
        level_vals: list[int] = []
        for _ in range(level_size):
            node = queue.pop(0)
            level_vals.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level_vals)
    return result


# ── LC105 从前序与中序遍历序列构造二叉树（Medium） ────────────────────────
def build_tree_from_pre_in(preorder: list[int], inorder: list[int]) -> "TreeNode | None":
    """
    【题意】给一棵二叉树的前序遍历序列 preorder 和中序遍历序列 inorder（节点值
        互不相同），重建出这棵二叉树，返回根节点。
    【思路】两个关键性质缺一不可：
        1) **前序遍历的第一个元素永远是当前（子）树的根**——因为前序是"根-左-右"，
           第一个访问的必然是根。
        2) **中序遍历能把"根左边"和"根右边"切开**——中序是"左-根-右"，一旦知道
           根是谁，在中序序列里找到这个根的位置，根左边的一段就是整棵左子树的
           中序序列，右边一段就是整棵右子树的中序序列（因为中序遍历不会把左右
           子树的节点交叉排列）。
        于是递归框架是：每次从 preorder 当前游标处取出一个值当根 → 在 inorder
        里查这个值的下标 mid → 递归时，左子树对应 inorder 的 [in_left, mid-1]，
        右子树对应 inorder 的 [mid+1, in_right]，而 preorder 的游标要在递归进入
        左子树之前先 +1（因为前序序列消耗的顺序始终是"根、根的整棵左子树、根的
        整棵右子树"，游标必须严格按这个顺序往前走）。用一个 值→下标 的哈希表
        预处理 inorder，把"在 inorder 里查根的位置"从 O(n) 降到 O(1)，避免整体
        退化成 O(n^2)。
    【复杂度】时间 O(n)（哈希表预处理 O(n) + 每个节点 O(1) 定位），空间 O(n)。
    【易错点】preorder 的游标必须用可变的外部状态（这里用长度为 1 的 list 模拟
        nonlocal）而不能用函数参数传值——否则左子树递归时游标的变化不会体现在
        右子树的递归里；递归终止条件写成 in_left >= in_right 会在只剩一个节点
        时提前结束，正确写法是 in_left > in_right 才为空。
    """
    index_of = {val: i for i, val in enumerate(inorder)}
    pre_idx = [0]                            # 用单元素 list 模拟"可变的外部游标"

    def helper(in_left: int, in_right: int) -> "TreeNode | None":
        if in_left > in_right:
            return None
        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)
        mid = index_of[root_val]
        root.left = helper(in_left, mid - 1)
        root.right = helper(mid + 1, in_right)
        return root

    return helper(0, len(inorder) - 1)


def _preorder_values(root: "TreeNode | None") -> list[int]:
    """仅用于测试校验：把树重新做一次前序遍历，应该能还原输入的 preorder。"""
    if root is None:
        return []
    return [root.val] + _preorder_values(root.left) + _preorder_values(root.right)


# ── LC236 二叉树的最近公共祖先（Medium） ──────────────────────────────────
def lowest_common_ancestor(root: "TreeNode | None", p: "TreeNode", q: "TreeNode") -> "TreeNode":
    """
    【题意】给二叉树根节点和树中已存在的两个节点 p、q（注意是节点引用，不是值），
        返回 p 和 q 的最近公共祖先节点（一个节点也可以是它自己的祖先）。
    【思路】要求这个函数对一棵子树做什么：**在这棵子树范围内，告诉我 p/q 是否
        存在、以及如果两个都存在，它们的 LCA 是谁**。后序遍历（先递归左右子树，
        再处理当前节点）天然适合这种"先问子树要答案，再综合"的题：
        - 如果当前节点本身就是 p 或 q，直接返回当前节点（它自己就是"目前找到的
          最深的候选人"，不需要再往下找）。
        - 否则分别问左子树、右子树"你那边有没有 p 或 q"。如果左右子树各自都
          返回了非 None 结果，说明 **p 和 q 分别在当前节点的两侧**，那么当前
          节点就是 LCA（它是能同时"看到"两者的最深节点）。
        - 如果只有一侧非 None，说明 p、q（或至少已经找到的那个）都在这一侧，
          把这一侧的结果原样网上传递即可。
    【复杂度】时间 O(n)（最坏遍历整棵树），空间 O(h)（递归栈）。
    【易错点】把"当前节点是 p 或 q"的判断放漏了会导致找不到"某节点是另一节点祖先"
        的情况（比如 p 是 q 的祖先，标准答案应该是 p 本身）；一定要用 `is`（同一个
        对象引用）而不是 `==`（值相等）比较，因为题目给的是节点引用，树里可能有
        值相同但不是目标节点的其他节点。
    """
    if root is None or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left is not None and right is not None:
        return root
    return left if left is not None else right


# ── LC124 二叉树中的最大路径和（Hard） ────────────────────────────────────
def max_path_sum(root: "TreeNode | None") -> int:
    """
    【题意】二叉树中"路径"定义为沿着节点间的边走的一个节点序列，同一个节点在
        一条路径中最多出现一次，路径不要求经过根节点。求所有路径里节点值之和
        的最大值。
    【思路】这题最容易搞混的地方是要分清楚**两个不同的量**：
        1) "以某节点为**拐点**、经过该节点的最大路径和"——这条路径可以同时向
           左、右两个方向延伸（呈 ⋀ 形），是"最终候选答案"之一，每个节点都要
           算一次，取全局最大值。
        2) "某节点能贡献给它**父节点**的最大链和"——链的定义是"只能往一个方向
           延伸"（不能同时往左又往右，否则父节点再往上连接时，路径就会在这个
           节点分叉，不再是一条简单路径）。这个值是递归函数**真正返回**的东西，
           因为父节点只能拿"一条腿"去拼接。
        所以递归函数 gain(node) 的语义是"这个节点能贡献给父节点的最大链和"
        （量 2），但在计算它的过程中，顺手用 (node.val + left_gain + right_gain)
        （量 1，两条腿都算上）去更新一个记录全局答案的外部变量 best。两条腿的
        gain 如果是负数，还不如不要这条腿（对路径和只有负贡献），所以子树的
        gain 要先和 0 取 max 再使用——这对应"路径可以不经过某个方向，因为负数
        子链只会拉低总和"。
    【复杂度】时间 O(n)（每个节点访问一次），空间 O(h)（递归栈）。
    【易错点】把"返回值"（只能一条腿）和"更新 best 时用的值"（两条腿都算上）
        搞成同一个表达式，会把不合法的"经过节点后又分叉"的路径也当成合法链
        返回给父节点；忘记把负的子树 gain 截断成 0，遇到全是负数的树会算错
        （应该退化成"只选权值最大的单个节点"）。
    """
    best = float("-inf")

    def gain(node: "TreeNode | None") -> int:
        nonlocal best
        if node is None:
            return 0
        left_gain = max(gain(node.left), 0)
        right_gain = max(gain(node.right), 0)
        best = max(best, node.val + left_gain + right_gain)
        return node.val + max(left_gain, right_gain)

    gain(root)
    return int(best)


def _self_test() -> None:
    assert inorder_traversal(build_tree([1, None, 2, 3])) == [1, 3, 2]
    assert inorder_traversal(build_tree([])) == []
    assert inorder_traversal(build_tree([1])) == [1]
    # 递归版和迭代版必须给出同样的结果
    for vals in ([1, None, 2, 3], [], [1], [5, 3, 8, 1, 4]):
        t = build_tree(vals)
        assert inorder_traversal(t) == _inorder_recursive(t)

    assert max_depth(build_tree([3, 9, 20, None, None, 15, 7])) == 3
    assert max_depth(build_tree([1, None, 2])) == 2

    assert tree_to_level_list(invert_tree(build_tree([4, 2, 7, 1, 3, 6, 9]))) == [4, 7, 2, 9, 6, 3, 1]
    assert tree_to_level_list(invert_tree(build_tree([2, 1, 3]))) == [2, 3, 1]
    assert tree_to_level_list(invert_tree(build_tree([]))) == []

    assert level_order(build_tree([3, 9, 20, None, None, 15, 7])) == [[3], [9, 20], [15, 7]]
    assert level_order(build_tree([1])) == [[1]]
    assert level_order(build_tree([])) == []

    preorder, inorder = [3, 9, 20, 15, 7], [9, 3, 15, 20, 7]
    rebuilt = build_tree_from_pre_in(preorder, inorder)
    assert _preorder_values(rebuilt) == preorder
    assert inorder_traversal(rebuilt) == inorder

    lca_root = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p5, q1 = find_node(lca_root, 5), find_node(lca_root, 1)
    assert lowest_common_ancestor(lca_root, p5, q1).val == 3
    p5b, q4 = find_node(lca_root, 5), find_node(lca_root, 4)
    assert lowest_common_ancestor(lca_root, p5b, q4).val == 5

    assert max_path_sum(build_tree([1, 2, 3])) == 6
    assert max_path_sum(build_tree([-10, 9, 20, None, None, 15, 7])) == 42

    print("[PASS] p08_binary_tree: 7 题（LC94/104/226/102/105/236/124）全部通过")


if __name__ == "__main__":
    _self_test()
