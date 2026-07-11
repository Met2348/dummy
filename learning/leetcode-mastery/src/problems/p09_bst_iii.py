"""分类 09·二叉搜索树（BST）· 竞赛级补充（Part III）：不重讲框架，专攻
"中序遍历升序"这条核心不变量的更多变体用法，以"重建平衡树"深挖收尾的 7 道题。

Part I/II 已经把这条不变量用在了查找/插入/删除/计数/累加/自愈（LC99）上。
这批题继续在这条不变量上做文章：把它当"排序好的数组"来做双指针（LC653）、
用它检测"众数"和"最小差"这类统计量（LC501/LC530）、反过来利用它做范围裁剪
（LC669）、利用它从前序序列反推结构（LC1008）、把两棵树的有序序列做归并
（LC1305），最后用它把一棵退化的树重新"摊平再对折"成平衡树（LC1382）。

本文件独立自包含（不依赖 p09_bst.py / p09_bst_ii.py 的 import 路径），复制
了一份等价的 TreeNode / build_tree 定义，保证单独用
`python learning/leetcode-mastery/src/problems/p09_bst_iii.py` 就能跑通。
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


def _inorder(root: "TreeNode | None") -> list[int]:
    """仅用于测试校验的朴素中序遍历。"""
    if root is None:
        return []
    return _inorder(root.left) + [root.val] + _inorder(root.right)


def _tree_to_level_list(root: "TreeNode | None") -> list:
    """仅用于测试校验：把（子）树转回层序数组（截掉末尾多余的 None）。"""
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


def is_valid_bst(root: "TreeNode | None", lower: float = float("-inf"), upper: float = float("inf")) -> bool:
    """仅用于测试校验：判断是否为合法 BST（区间收紧法，做法见 p09_bst.py LC98）。"""
    if root is None:
        return True
    if not (lower < root.val < upper):
        return False
    return is_valid_bst(root.left, lower, root.val) and is_valid_bst(root.right, root.val, upper)


def _is_balanced(root: "TreeNode | None") -> bool:
    """仅用于测试校验：判断是否为高度平衡树（任意节点左右子树深度差 <= 1）。"""

    def check(node: "TreeNode | None") -> tuple[int, bool]:
        if node is None:
            return 0, True
        left_h, left_ok = check(node.left)
        right_h, right_ok = check(node.right)
        balanced = left_ok and right_ok and abs(left_h - right_h) <= 1
        return max(left_h, right_h) + 1, balanced

    return check(root)[1]


# ── LC653 两数之和 IV - 输入 BST（Easy） ──────────────────────────────────
def find_target(root: "TreeNode | None", k: int) -> bool:
    """
    【题意】给 BST 根节点和目标值 k，判断树中是否存在两个不同节点，值之和
        恰好等于 k。
    【思路】这题的"BST"身份其实不是必需的——用一个哈希集合边遍历边记录
        "已经见过的值"，对每个新节点检查 `k - node.val` 是否已经在集合里
        出现过（如果出现过，说明更早访问的某个节点和当前节点凑成了 k）。
        这个做法对任意二叉树（甚至任意集合）都成立，只是因为输入恰好是 BST，
        很容易让人误以为要利用有序性做双指针（那也是一种正确解法，但不是
        唯一或更简单的解法）——这里刻意选哈希集合的写法，提醒"BST"不代表
        "必须用有序性"，有时候更朴素的哈希做法反而更直接。
    【复杂度】时间 O(n)，空间 O(n)（哈希集合最坏存下所有节点值）。
    【易错点】"两个不同节点"这个要求容易在 `k` 恰好是某个节点值的两倍时
        被忽略——比如树里只有一个值为 5 的节点、k=10，正确答案应该是
        False（没有第二个节点能和它配对），这份写法天然正确是因为"检查
        `k - node.val` 是否已经在集合里"时，当前节点自己还没被加入集合
        （加入是在检查之后），不会把自己和自己配对；DFS 遍历左右子树都要
        做（用 `or` 短路），不能只探一侧。
    """
    seen: set[int] = set()

    def dfs(node: "TreeNode | None") -> bool:
        if node is None:
            return False
        if k - node.val in seen:
            return True
        seen.add(node.val)
        return dfs(node.left) or dfs(node.right)

    return dfs(root)


# ── LC501 二叉搜索树中的众数（Easy） ──────────────────────────────────────
def find_mode(root: "TreeNode | None") -> list[int]:
    """
    【题意】给一棵**允许存在重复值**的 BST（左子树 <= 根 <= 右子树），返回
        出现次数最多的值（众数），可能有多个，任意顺序返回。
    【思路】利用"中序遍历升序"这条核心不变量——一旦允许重复值，中序遍历会
        变成**非严格升序**，这意味着**所有相同的值在中序序列里必然连续排在
        一起**，不会被其他值打断。于是只需要一次中序遍历，边走边维护
        "上一个访问的值 `prev`"和"当前值连续出现的次数 `count`"：值和
        `prev` 相同就 `count += 1`，否则说明换了一个新值，`count` 重置为
        1。每次更新 `count` 之后，和全局最大次数 `max_count` 比较：如果
        破了纪录（`count > max_count`），众数列表整体清空重新只放当前值；
        如果打平纪录（`count == max_count`），把当前值追加进众数列表。
    【复杂度】时间 O(n)，空间 O(1)（不计输出和递归栈——只用了几个计数变量）。
    【易错点】`prev` 初始必须是一个"哨兵"（这里用 `None`，配合"是否是第一个
        节点"的判断），不能直接初始化成 0 或者某个具体数值，否则如果树里
        真的存在值为 0（或那个初始值）的节点，会在第一次比较时被误判为
        "和上一个值相同"；破纪录时要**清空**众数列表重新开始（`modes =
        [node.val]`），而不是直接 `append`，否则旧纪录的值会残留在结果里。
    """
    modes: list[int] = []
    max_count = 0
    count = 0
    prev: int | None = None

    def dfs(node: "TreeNode | None") -> None:
        nonlocal modes, max_count, count, prev
        if node is None:
            return
        dfs(node.left)
        if prev is not None and node.val == prev:
            count += 1
        else:
            count = 1
        if count > max_count:
            max_count = count
            modes = [node.val]
        elif count == max_count:
            modes.append(node.val)
        prev = node.val
        dfs(node.right)

    dfs(root)
    return modes


# ── LC530 二叉搜索树的最小绝对差（Easy） ──────────────────────────────────
def min_diff_in_bst(root: "TreeNode | None") -> int:
    """
    【题意】给 BST 根节点，求树中任意两个不同节点值之间差的绝对值的最小值。
    【思路】"中序遍历升序"直接给出一个关键结论：**最小差一定出现在中序序列
        里相邻的两个数之间**——如果 a < b < c，那么 `c - a` 一定不小于
        `b - a` 或 `c - b` 中较小的那个（三角不等式的直接推论），所以任何
        "隔着至少一个数"的两数之差都不可能是全局最小值，只需要比较中序序列
        里每一对相邻的数。于是只需要一次中序遍历，边走边维护"上一个访问的值
        `prev`"，每访问一个新节点就用 `node.val - prev` 更新全局最小值。
    【复杂度】时间 O(n)，空间 O(1)（不计递归栈）。
    【易错点】只在 `prev is not None`（即不是第一个被访问的节点）时才计算
        差值并更新最小值，第一个节点没有"上一个值"可比较；这题默认树中
        节点值互不相同（不同于 LC501 允许重复），不需要额外处理"差为 0"
        的情况，但如果误把这题和 LC501 的输入假设搞混，会错误地认为需要
        跳过差为 0 的相邻对。
    """
    best = float("inf")
    prev: int | None = None

    def dfs(node: "TreeNode | None") -> None:
        nonlocal best, prev
        if node is None:
            return
        dfs(node.left)
        if prev is not None:
            best = min(best, node.val - prev)
        prev = node.val
        dfs(node.right)

    dfs(root)
    return int(best)


# ── LC669 修剪二叉搜索树（Medium） ────────────────────────────────────────
def trim_bst(root: "TreeNode | None", low: int, high: int) -> "TreeNode | None":
    """
    【题意】给 BST 根节点和区间 `[low, high]`，删除所有值不在这个区间内的
        节点，但要保持剩下节点之间原有的祖先/后代关系不变（不能把某个节点
        的后代重新接到别的地方），返回修剪后的根节点（根本身也可能被换掉）。
    【思路】利用 BST 的有序性做剪枝，而不是老实地逐节点判断再重新拼接：
        如果当前节点的值**小于** low，说明当前节点自己以及它的**整棵左
        子树**（值只会更小）全部超出区间下界，可以整体丢弃——直接递归地
        用"当前节点的右子树修剪结果"取代当前节点，不需要访问左子树；对称地，
        如果当前节点的值**大于** high，直接丢弃当前节点和整棵右子树，用
        "左子树修剪结果"取代。只有当前节点的值落在区间内时，才需要保留
        当前节点，并对左右子树分别递归修剪、把结果重新接回 `left`/`right`。
    【复杂度】时间 O(n)（最坏情况仍要访问每个节点一次），空间 O(h)（递归
        栈）。
    【易错点】`root.val < low` 分支要递归 `root.right`（不是 `root.left`）
        ——容易凭直觉写反，其实是因为"值太小"要往"更大的方向"（右子树）里找
        还能保留的节点；同理 `root.val > high` 分支要递归 `root.left`；
        对递归调用的返回值必须重新赋回 `root.left`/`root.right`
        （`root.left = trim_bst(root.left, ...)`），忘记赋值会导致被修剪
        的子树虽然逻辑上"不该存在"，但物理上仍然挂在树上。
    """
    if root is None:
        return None
    if root.val < low:
        return trim_bst(root.right, low, high)
    if root.val > high:
        return trim_bst(root.left, low, high)
    root.left = trim_bst(root.left, low, high)
    root.right = trim_bst(root.right, low, high)
    return root


# ── LC1008 前序遍历构造二叉搜索树（Medium） ───────────────────────────────
def bst_from_preorder(preorder: list[int]) -> "TreeNode | None":
    """
    【题意】给一个数组 preorder，它是某棵 BST 的前序遍历结果（节点值互不
        相同），根据它重建出这棵 BST，返回根节点。
    【思路】和普通二叉树不同，BST 不需要额外的中序序列辅助定位左右子树的
        边界——因为"值小于根往左、值大于根往右"这条 BST 定义本身就足够
        切分。用一个**单调栈**在 O(n) 内一次扫描完成构造：栈里保存"当前
        路径上、还没确定右孩子的祖先链"，从栈顶到栈底值递减。对每个新值：
        如果比栈顶还小，说明它是栈顶节点的左孩子（BST 定义），直接挂上去
        再入栈；如果比栈顶大，说明它不可能是栈顶的孩子（会破坏"栈顶及其
        祖先链值递增"这个结构），要不断弹出栈里所有比新值小的节点，最后
        弹出的那个（丢失的最后一个）就是新值的父节点，新值作为它的右孩子。
    【复杂度】时间 O(n)（每个节点最多入栈出栈各一次），空间 O(n)（栈 +
        递归无关——这是迭代写法）。
    【易错点】判断"新值应该接在谁的右边"时，父节点是**最后一个被弹出**的
        节点，不是"弹出前的栈顶"——容易在弹栈循环写完后忘记正确保存这个
        引用，导致挂错父节点；栈里天然维护的是一条"值递减"的链，如果新值
        比栈顶小就不弹栈直接作为左孩子，这一步不能颠倒判断顺序（必须先判断
        `value < stack[-1].val` 这个更简单的分支，再进入需要弹栈的分支）。
    """
    root = TreeNode(preorder[0])
    stack = [root]
    for value in preorder[1:]:
        node = TreeNode(value)
        if value < stack[-1].val:
            stack[-1].left = node
        else:
            parent = stack[-1]
            while stack and stack[-1].val < value:
                parent = stack.pop()
            parent.right = node
        stack.append(node)
    return root


# ── LC1305 两棵二叉搜索树中的所有元素（Medium） ───────────────────────────
def get_all_elements(root1: "TreeNode | None", root2: "TreeNode | None") -> list[int]:
    """
    【题意】给两棵 BST 的根节点，返回一个列表，包含两棵树里所有节点的值，
        按升序排列。
    【思路】直接把两棵树的所有值收集起来再统一排序当然可行（O((m+n)
        log(m+n))），但更能体现"利用 BST 有序性"这条主线的做法是：分别对
        两棵树做中序遍历，各自得到一个天然有序的数组，然后像"合并两个有序
        链表"（LC21）一样用双指针把它们归并成一个有序数组——不需要对合并
        后的结果再排序一次。
    【复杂度】时间 O(m + n)（两次中序遍历各 O(树大小) + 一次线性归并），
        空间 O(m + n)（存储两个中序数组和合并结果）。
    【易错点】归并时比较用 `<=` 而不是 `<`（`list1[i] <= list2[j]` 就先取
        `list1[i]`），这只是为了保证结果稳定、并不影响正确性（相等时先取
        哪个都不影响最终排序结果，但统一用 `<=` 更符合"稳定归并"的直觉）；
        循环结束后不要忘记把其中一个数组**剩余的尾巴**整体接上去
        （`merged.extend(list1[i:])` 和 `merged.extend(list2[j:])`），这
        一步漏掉会丢失较长那个数组末尾没被比较到的元素。
    """
    list1 = _inorder(root1)
    list2 = _inorder(root2)
    merged: list[int] = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged


# ── LC1382 将二叉搜索树变平衡（Medium） ───────────────────────────────────
def balance_bst(root: "TreeNode | None") -> "TreeNode | None":
    """
    【题意】给一棵 BST（可能因为插入顺序问题严重退化，比如变成一条链），
        返回一棵值集合相同、但**高度平衡**的 BST（任意节点左右子树深度差
        不超过 1）。答案不唯一，返回任意一个满足条件的即可。
    【思路】做法见本文件对应 lecture 的深挖部分——核心是"忘记原来的树形状，
        只保留信息"：先对原树做一次中序遍历，得到一个升序数组 `values`
        （这一步就是"摊平"，中序遍历的结果只和值的集合与相对顺序有关，和原树
        的具体退化形状完全无关）。然后用 Part I LC108（将有序数组转换为
        BST）的思路，对这个有序数组"对折"重建：递归地取区间正中间的元素当
        根，递归构造左右两半，因为每次都从正中间切开，两侧元素数量差不超过
        1，天然保证了高度平衡。
    【复杂度】时间 O(n)（一次中序遍历 O(n) + 重建 O(n)），空间 O(n)（存储
        中序数组 + 递归栈）。
    【易错点】这题的解法本质上是"LC530 系列的中序遍历"（获取有序值）加上
        "LC108 的重建"（有序数组转平衡 BST）两个已经学过的技巧的直接拼接，
        容易被"退化树""重新平衡"这些新名词唬住而想复杂——原树具体长什么
        形状（一条左链还是一条右链）对解法完全没有影响，因为第一步中序遍历
        已经把这些形状信息"抹平"成了一个只保留顺序的数组；重建时取中点用
        `(lo + hi) // 2`（偏左中点），这不是唯一正确的取法（偏右中点同样
        合法），但必须和"递归切分左右区间"的写法保持一致，不能中途混用不同
        的取中点方式。
    """
    values = _inorder(root)

    def build(lo: int, hi: int) -> "TreeNode | None":
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = TreeNode(values[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(values) - 1)


def _self_test() -> None:
    assert find_target(build_tree([5, 3, 6, 2, 4, None, 7]), 9) is True
    assert find_target(build_tree([5, 3, 6, 2, 4, None, 7]), 28) is False

    assert find_mode(build_tree([1, None, 2, 2])) == [2]

    assert min_diff_in_bst(build_tree([4, 2, 6, 1, 3])) == 1
    assert min_diff_in_bst(build_tree([1, None, 3, 2])) == 1

    trimmed1 = trim_bst(build_tree([1, 0, 2]), 1, 2)
    assert _tree_to_level_list(trimmed1) == [1, None, 2]
    trimmed2 = trim_bst(build_tree([3, 0, 4, None, 2, None, None, 1]), 1, 3)
    assert _tree_to_level_list(trimmed2) == [3, 2, None, 1]

    built1008 = bst_from_preorder([8, 5, 1, 7, 10, 12])
    assert is_valid_bst(built1008) is True

    def _preorder(node: "TreeNode | None") -> list[int]:
        if node is None:
            return []
        return [node.val] + _preorder(node.left) + _preorder(node.right)

    assert _preorder(built1008) == [8, 5, 1, 7, 10, 12]

    assert get_all_elements(build_tree([2, 1, 4]), build_tree([1, 0, 3])) == [0, 1, 1, 2, 3, 4]
    assert get_all_elements(build_tree([1, None, 8]), build_tree([8, 1])) == [1, 1, 8, 8]

    balanced = balance_bst(build_tree([1, None, 2, None, 3, None, 4, None, None]))
    assert is_valid_bst(balanced) is True
    assert _is_balanced(balanced) is True
    assert _inorder(balanced) == [1, 2, 3, 4]
    assert _tree_to_level_list(balanced) == [2, 1, 3, None, None, None, 4]

    print(
        "[PASS] p09_bst_iii: 7 题"
        "（两数之和IV-输入BST/BST中的众数/BST的最小绝对差/修剪二叉搜索树/"
        "前序遍历构造二叉搜索树/两棵BST中的所有元素/将二叉搜索树变平衡）"
        "全部通过"
    )


if __name__ == "__main__":
    _self_test()
