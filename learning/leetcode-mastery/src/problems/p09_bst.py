"""分类 09·二叉搜索树（BST）—— 4 道题。

BST 的核心不变量只有一句话：**对任意节点，中序遍历一定是严格升序**（等价说法：
左子树所有值 < 该节点 < 右子树所有值，且这个约束对所有祖先都成立，不只是直接父子）。
本文件里的每一道题、每一个技巧，都是这条性质的直接推论。

本文件独立自包含（不依赖 p08_binary_tree.py 的 import 路径），复制了一份等价的
TreeNode / build_tree 定义，保证单独用
`python learning/leetcode-mastery/src/problems/p09_bst.py` 就能跑通。
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


def _inorder_values(root: "TreeNode | None") -> list[int]:
    """仅用于测试校验的朴素中序遍历。"""
    if root is None:
        return []
    return _inorder_values(root.left) + [root.val] + _inorder_values(root.right)


# ── LC98 验证二叉搜索树（Medium） ─────────────────────────────────────────
def is_valid_bst(root: "TreeNode | None", lower: float = float("-inf"), upper: float = float("inf")) -> bool:
    """
    【题意】给二叉树根节点，判断它是否是一棵合法的二叉搜索树。
    【思路】新手最经典的陷阱：**只比较父子节点**（`node.left.val < node.val <
        node.right.val`）是错的。反例：
            ```
                5
               / \\
              1   4
                 / \\
                3   6
            ```
        节点 4 和它的直接孩子 3、6 之间的父子关系看起来都满足"左 < 父 < 右"，但
        4 整体是根节点 5 的右子树，右子树里所有值都必须 > 5，而 3 < 5，所以整棵
        树不合法。问题出在"3 只跟它的直接父亲 4 比较"，却没有跟更高层的祖先 5
        比较。正确做法是给每个节点传递一个**开区间 (lower, upper)**，表示"这个
        节点的值必须落在这个区间里"——区间由所有祖先决定：往左子树递归时，
        新的上界收紧为当前节点的值（右边界不能超过父亲）；往右子树递归时，
        新的下界收紧为当前节点的值。这样区间是"层层收紧"传下去的，天然就把
        所有祖先的约束都带上了，而不只是直接父节点。
    【复杂度】时间 O(n)，空间 O(h)（递归栈）。
    【易错点】只比较父子（漏掉更高层祖先的约束），本题最典型的错误写法；
        用 <= 而不是 < 判断——BST 定义里值互不相等，边界写成 <= 会把值相等的
        非法情况误判为合法。
    """
    if root is None:
        return True
    if not (lower < root.val < upper):
        return False
    return is_valid_bst(root.left, lower, root.val) and is_valid_bst(root.right, root.val, upper)


# ── LC108 将有序数组转换为二叉搜索树（Easy） ──────────────────────────────
def sorted_array_to_bst(nums: list[int]) -> "TreeNode | None":
    """
    【题意】给一个升序数组，构造一棵"高度平衡"的二叉搜索树（每个节点左右子树高度
        差不超过 1），返回根节点（答案不唯一，只要满足性质即可）。
    【思路】反过来利用"中序遍历升序"这条性质：如果把数组的**中点**当根，那么
        数组左半部分天然都比根小、右半部分天然都比根大——直接满足 BST 的定义，
        不需要再比较大小。同时因为每次都从正中间切开，左右两半的元素数量差不超
        过 1，递归下去自然保证了"高度平衡"（这是这题"顺带"送的性质，不需要额外
        维护平衡逻辑，换成快排的每一步"选中点做根"就是这题的核心 insight）。
    【复杂度】时间 O(n)（每个元素恰好建一个节点），空间 O(log n)（平衡树的递归
        深度；加上构造出的树本身占 O(n)）。
    【易错点】数组长度为偶数时中点可以偏左也可以偏右，两种取法都能构造出合法的
        平衡 BST，不要误以为必须精确复现 LeetCode 官方给出的某一个特定答案；
        切片 nums[:mid] / nums[mid+1:] 时下标要跟"根取 nums[mid]"对应一致，
        不要多算或漏算根本身这个元素。
    """
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid + 1:])
    return root


def is_balanced(root: "TreeNode | None") -> bool:
    """仅用于测试校验：树是否"高度平衡"（每个节点左右子树高度差 <= 1）。"""

    def height(node: "TreeNode | None") -> int:
        if node is None:
            return 0
        lh = height(node.left)
        if lh == -1:
            return -1
        rh = height(node.right)
        if rh == -1:
            return -1
        if abs(lh - rh) > 1:
            return -1
        return 1 + max(lh, rh)

    return height(root) != -1


# ── LC230 二叉搜索树中第 K 小的元素（Medium） ─────────────────────────────
def kth_smallest(root: "TreeNode | None", k: int) -> int:
    """
    【题意】给 BST 根节点和整数 k，返回树中第 k 小的元素值（1-indexed，k 保证合法）。
    【思路】直接利用"中序遍历 = 升序序列"这条核心不变量：中序遍历走到第 k 个
        被访问的节点，它的值就是第 k 小。这里用显式栈做迭代中序遍历（和普通二叉
        树的中序遍历完全一样，BST 只是保证了这个序列一定是升序），每弹出一个
        节点就把计数器 +1，等计数器等于 k 时立刻返回，不需要真的遍历完整棵树、
        也不需要把所有值都收集到一个 list 里再取下标——这样最坏情况下也只多访问
        到第 k 个节点为止，如果 k 远小于 n 会更快退出。
    【复杂度】时间最坏 O(n)（k 接近 n 时），平均更快；空间 O(h)。
    【易错点】用递归收集全部中序遍历结果再取 `result[k-1]`，功能上没错但没有利用
        "找到第 k 个就能提前退出"这个优化，属于可以更好但不算 bug；真正的 bug
        是 k 的 1-indexed 和数组下标 0-indexed 搞混，或者忘记处理 k 大于节点总数
        的情况（题目保证合法可以不处理，但要清楚这是题目给的前提）。
    """
    stack: list[TreeNode] = []
    node = root
    count = 0
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        count += 1
        if count == k:
            return node.val
        node = node.right
    raise ValueError("k 超出了树中节点的数量")


# ── LC235 二叉搜索树的最近公共祖先（Medium） ──────────────────────────────
def lowest_common_ancestor_bst(root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
    """
    【题意】给 BST 根节点和树中已存在的两个节点 p、q，返回它们的最近公共祖先。
    【思路】对比"普通二叉树的最近公共祖先"（LC236）：那道题因为不知道值的大小
        关系，必须递归左右两侧子树、看两侧是否都能找到 p/q，时间复杂度 O(n)
        （最坏要遍历整棵树）。而 BST 的有序性质让这题**不需要检查两侧子树**：
        从根开始，只需要比较 p.val、q.val 和当前节点 val 的大小——如果 p、q
        的值都比当前节点小，说明它们的公共祖先只可能在**左子树**里（BST 的定义
        保证右子树里不可能有比当前节点小的值），直接往左走；都比当前节点大同理
        往右走；只要不满足"同侧"，说明当前节点正好把 p、q 分到了两侧（或者当前
        节点本身就是 p/q 之一），那这个节点就是 LCA。整个过程只沿着一条从根到
        LCA 的路径往下走，不需要两侧都递归。
    【复杂度】时间 O(h)（h 为树高，只走一条路径；平衡树是 O(log n)，普通二叉树
        LCA 是 O(n)——这是这题相对 LC236 的复杂度提升，来自"排序性质省掉了
        一侧子树的搜索"）。空间 O(1)（迭代写法不需要递归栈；写成递归也只有
        O(h) 而不需要额外遍历两侧）。
    【易错点】照抄 LC236 的"两侧都递归再合并"写法在这题里也能得到正确答案，
        但会浪费掉 BST 的有序性质、复杂度上不去，属于"能通过但没抓住这题真正
        考的点"；比较大小时漏掉"当前节点本身等于 p 或 q"这一分支，会导致
        p 是 q 祖先时死循环或多走一步。
    """
    node = root
    while node is not None:
        if p.val < node.val and q.val < node.val:
            node = node.left
        elif p.val > node.val and q.val > node.val:
            node = node.right
        else:
            return node
    raise ValueError("root 不能为空")


def _self_test() -> None:
    root108 = sorted_array_to_bst([-10, -3, 0, 5, 9])
    assert is_valid_bst(root108) is True
    assert is_balanced(root108) is True
    assert _inorder_values(root108) == [-10, -3, 0, 5, 9]

    assert is_valid_bst(build_tree([2, 1, 3])) is True
    assert is_valid_bst(build_tree([5, 1, 4, None, None, 3, 6])) is False

    assert kth_smallest(build_tree([3, 1, 4, None, 2]), 1) == 1
    assert kth_smallest(build_tree([5, 3, 6, 2, 4, None, None, 1]), 3) == 3

    lca_root = build_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
    p2, q8 = find_node(lca_root, 2), find_node(lca_root, 8)
    assert lowest_common_ancestor_bst(lca_root, p2, q8).val == 6
    p2b, q4 = find_node(lca_root, 2), find_node(lca_root, 4)
    assert lowest_common_ancestor_bst(lca_root, p2b, q4).val == 2

    print("[PASS] p09_bst: 4 题（LC108/98/230/235）全部通过")


if __name__ == "__main__":
    _self_test()
