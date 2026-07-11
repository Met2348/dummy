"""分类 09·二叉搜索树（BST）· 进阶补充（Part II）：不重讲框架，扩大 BST
增删改+计数变体覆盖面的 7 道题。

Part I 的核心不变量依旧成立——"中序遍历一定是严格升序"。这批题再加一条主线：
**BST 的"有序性"不仅能用来查找，还能用来插入、删除、计数、枚举、累加、甚至
定位并修复被破坏的有序性**。

本文件独立自包含（不依赖 p09_bst.py 的 import 路径），复制了一份等价的
TreeNode / build_tree 定义，保证单独用
`python learning/leetcode-mastery/src/problems/p09_bst_ii.py` 就能跑通。
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


# ── LC700 二叉搜索树中的搜索（Easy） ──────────────────────────────────────
def search_bst(root: "TreeNode | None", val: int) -> "TreeNode | None":
    """
    【题意】给 BST 根节点和目标值 val，返回以"值等于 val 的节点"为根的子树；
        如果不存在这样的节点，返回 None。
    【思路】直接利用 BST 的有序性做单向查找：当前节点值等于 val 就是答案；小于
        当前节点就说明目标只可能在左子树（右子树全都更大），大于就只可能在右
        子树——每一步都能排除一半的搜索空间，不需要像普通二叉树那样两侧都探查。
    【复杂度】时间 O(h)（h 为树高，只走一条路径；平衡树是 O(log n)），空间 O(1)
        （迭代写法不需要递归栈）。
    【易错点】判断分支写反（比如 val 更小时却往右子树走）会导致查找方向和 BST
        的有序性质相悖，多数情况下会直接返回 None（找不到本该存在的节点）；
        没有及时 return，导致 while 循环把 node 更新为 None 之后再访问
        `node.val` 会报错。
    """
    node = root
    while node is not None:
        if node.val == val:
            return node
        node = node.left if val < node.val else node.right
    return None


# ── LC701 二叉搜索树中的插入操作（Medium） ────────────────────────────────
def insert_into_bst(root: "TreeNode | None", val: int) -> "TreeNode | None":
    """
    【题意】给 BST 根节点和一个不存在于树中的值 val，把 val 插入这棵 BST 中的
        合适位置，返回插入后的根节点（插入方案不唯一，只要保持 BST 性质即可）。
    【思路】沿着"搜索这个值会走的路径"往下走，走到某个节点的孩子位置本该存在
        但实际是 None 时，就在那个位置新建一个值为 val 的叶子节点——本质上是
        `search_bst` 的"查找失败"分支，只是失败的地方不返回 None，而是在那里
        长出一个新节点。因为新插入的值一定是原树中不存在的，"新建叶子节点"这个
        位置是唯一确定的，不需要考虑插入到中间打断已有结构。
    【复杂度】时间 O(h)，空间 O(h)（递归写法的调用栈；改成迭代可以做到 O(1)
        额外空间）。
    【易错点】递归时必须把子递归的返回值**重新赋回** `root.left`/`root.right`
        （`root.left = insert_into_bst(root.left, val)`），如果只调用不赋值，
        新建的节点不会真正挂到树上；空树（`root is None`）时直接返回新建的
        `TreeNode(val)` 作为这棵子树的根，是递归的终止条件。
    """
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    return root


# ── LC450 删除二叉搜索树中的节点（Medium） ────────────────────────────────
def delete_node(root: "TreeNode | None", key: int) -> "TreeNode | None":
    """
    【题意】给 BST 根节点和一个值 key，删除树中值为 key 的节点（如果存在），
        返回删除后仍然合法的 BST 根节点。
    【思路】先像 `search_bst` 一样利用大小比较沿路径找到目标节点，找到之后分
        三种情况处理：
        1) 目标节点没有左孩子：直接让它的右孩子顶替它的位置（返回 `root.right`）。
        2) 目标节点没有右孩子：同理直接返回 `root.left`。
        3) 左右孩子都存在：不能直接删掉这个节点（会留下两个孤儿子树无法安放），
           标准做法是找**右子树中最小的节点**（即右子树里一路往左走到底的那个
           节点，它是"比目标节点大的所有值里最小的一个"，天然满足"放在目标节点
           原来的位置仍然保持 BST 性质"），把目标节点的值**替换**成这个后继节点
           的值，然后递归地去右子树里删除这个后继节点（此时后继节点最多只有
           右孩子，问题规模缩小成了情况 1/2）。
    【复杂度】时间 O(h)（查找 + 定位后继各一次，最坏两次树高的路径），空间
        O(h)（递归调用栈）。
    【易错点】情况 3 里"用后继节点的值覆盖目标节点的值"之后，一定要**递归删除
        右子树里那个后继节点**（不能只改值就结束），否则后继节点的值会在树里
        出现两次；找后继节点时容易写成"左子树找最大"和"右子树找最小"搞反——
        必须是右子树找最小（也可以对称地用左子树找最大，两种写法都合法，但
        不能混用查找方向和删除方向）。
    """
    if root is None:
        return None
    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        successor = root.right
        while successor.left is not None:
            successor = successor.left
        root.val = successor.val
        root.right = delete_node(root.right, successor.val)
    return root


# ── LC96 不同的二叉搜索树（Medium） ───────────────────────────────────────
def num_trees(n: int) -> int:
    """
    【题意】给整数 n，求"由 1..n 这 n 个不同的值恰好能组成多少种结构不同的
        BST"，只统计**数量**，不需要构造出具体的树。
    【思路】枚举"谁是根"：如果固定根节点的值是 `i`（`1 <= i <= n`），那么
        `1..i-1` 这 `i-1` 个更小的值必须全部落在左子树里、`i+1..n` 这
        `n-i` 个更大的值必须全部落在右子树里——这是 BST 定义直接决定的，不需要
        额外证明。而"用 `i-1` 个连续值能构成多少种 BST"这个子问题，答案只和
        "有多少个值"这个数量有关、和具体是哪些值无关（因为可以整体平移映射到
        `1..i-1`）。于是设 `dp[k]` 表示"k 个值能构成多少种不同 BST"，枚举根
        把 k 个值分成左边 `j` 个、右边 `k-1-j` 个（`j` 从 0 到 k-1），
        `dp[k] = sum(dp[j] * dp[k-1-j] for j in range(k))`——这正是卡特兰数
        的递推式。
    【复杂度】时间 O(n^2)（两层循环），空间 O(n)（dp 数组）。
    【易错点】`dp[0] = 1` 这个边界容易被忽略或理解错——它表示"0 个节点（空
        子树）只有 1 种形态"，是保证乘法 `dp[j] * dp[k-1-j]` 在某一侧为空时
        依然算对的关键（空子树不是"0 种可能"而是"1 种确定的可能：什么都没
        有"）；内层循环范围写成 `range(k)`（对应 `j` 从 0 到 k-1）容易漏掉
        边界值 `j = k-1`（根是最大值，右子树为空）或 `j = 0`（根是最小值，
        左子树为空）。
    """
    dp = [0] * (n + 1)
    dp[0] = 1
    for nodes in range(1, n + 1):
        total = 0
        for left_size in range(nodes):
            total += dp[left_size] * dp[nodes - 1 - left_size]
        dp[nodes] = total
    return dp[n]


# ── LC95 不同的二叉搜索树 II（Medium） ────────────────────────────────────
def generate_trees(n: int) -> list["TreeNode | None"]:
    """
    【题意】给整数 n，构造出"由 1..n 这 n 个不同的值能组成的所有结构不同的
        BST"，返回这些树的根节点列表（这次要求具体构造出每一棵树，不只是计数）。
    【思路】和 LC96 是同一个"枚举谁是根、左边小值、右边大值"的框架，但因为要
        构造具体的树，只能用回溯而不能用 DP 压缩成一个数字：对区间 `[lo, hi]`
        内的每个值 `root_val` 依次尝试当根，递归求出"左区间 `[lo, root_val-1]`
        能构成的所有子树列表"和"右区间 `[root_val+1, hi]` 能构成的所有子树
        列表"，然后做**笛卡尔积**——左边的每一种可能搭配右边的每一种可能，各自
        拼出一棵完整的树，都是合法答案。区间为空时返回 `[None]`（"这个位置的
        子树有且只有一种可能：空"，呼应 LC96 里 `dp[0] = 1` 的含义）。
    【复杂度】时间和空间都是 O(卡特兰数(n) * n)（结果本身有卡特兰数量级那么多
        棵树，每棵树有 n 个节点）。
    【易错点】空区间必须返回 `[None]` 而不是 `[]`——如果返回空列表，笛卡尔积
        的双重循环会因为其中一边是空列表而直接跳过、什么都不生成，导致"左子树
        为空"或"右子树为空"这些合法情况全部丢失；每次循环新建 `TreeNode` 时
        要注意是"同一个左子树对象"可能被多棵不同的树共享——本题只要求返回的
        树各自结构合法，不要求彼此的节点对象互不相同，这不算 bug。
    """
    if n == 0:
        return []

    def build(lo: int, hi: int) -> list["TreeNode | None"]:
        if lo > hi:
            return [None]
        result: list["TreeNode | None"] = []
        for root_val in range(lo, hi + 1):
            left_trees = build(lo, root_val - 1)
            right_trees = build(root_val + 1, hi)
            for left in left_trees:
                for right in right_trees:
                    root = TreeNode(root_val)
                    root.left = left
                    root.right = right
                    result.append(root)
        return result

    return build(1, n)


# ── LC538 把二叉搜索树转换为累加树（Medium） ──────────────────────────────
def convert_bst(root: "TreeNode | None") -> "TreeNode | None":
    """
    【题意】给 BST 根节点，把每个节点的值改成"原树中大于等于该节点值的所有
        节点值之和"，返回修改后的根节点（原地修改）。
    【思路】"大于等于当前节点的所有值之和"，如果按**降序**访问节点、边访问边
        累加一个全局和，那么访问到某节点时，这个全局和恰好就是"所有已经访问过
        的（也就是更大的）值 + 当前节点自己的值"——正是题目要求的新值。而
        "降序访问 BST"只需要把中序遍历"左-根-右"的顺序反过来，变成"右-根-左"
        （先访问右子树，因为右子树都更大；再访问自己；最后访问左子树，因为
        左子树都更小）。
    【复杂度】时间 O(n)，空间 O(h)（递归栈）。
    【易错点】必须先递归右子树、再更新当前节点、最后递归左子树——顺序错了
        （比如先更新自己再递归右子树）会导致累加的全局和还没把更大的值算进来，
        当前节点算出的新值是错的；全局累加变量必须在递归之间共享（用
        `nonlocal`），不能每次递归都重新从 0 开始。
    """
    total = 0

    def dfs(node: "TreeNode | None") -> None:
        nonlocal total
        if node is None:
            return
        dfs(node.right)
        total += node.val
        node.val = total
        dfs(node.left)

    dfs(root)
    return root


# ── LC99 恢复二叉搜索树（Hard） ───────────────────────────────────────────
def recover_tree(root: "TreeNode | None") -> None:
    """
    【题意】一棵 BST 中恰好有两个节点的值被错误地交换了，导致它不再是一棵合法
        的 BST。在不改变树结构（只能修改节点的 val，不能新建/删除节点）的前提
        下，找到这两个节点并把值换回来，使树重新合法。不返回值，原地修改。
    【思路】做法见本文件对应 lecture 的深挖部分——核心是"中序遍历本该严格
        递增，被交换后会出现 1 或 2 处逆序（相邻两个数前者反而更大）"，只要
        在中序遍历过程中记录前一个访问的节点 `prev`，每次和当前节点比较：一旦
        出现 `prev.val > node.val`，就说明发生了逆序。第一次出现逆序时，
        `prev` 就是"两个被换节点中，应该更靠后却排到前面"的那一个（记为
        `first`）；`second` 先临时设为当前的 `node`。如果之后（相邻节点，即
        两个被换的节点在中序序列中不相邻的情况）又出现了第二次逆序，说明还有
        一次交叉，这时候只更新 `second`（不再更新 `first`），因为 `first`
        必然是第一次逆序里那个"应该更靠后"的节点。遍历结束后交换
        `first.val` 和 `second.val` 即可。
    【复杂度】时间 O(n)（一次中序遍历），空间 O(h)（递归栈；`prev`/`first`/
        `second` 都是 O(1) 的额外变量）。
    【易错点】第二次发现逆序时如果错误地又更新了 `first`（而不是只更新
        `second`），会丢失第一次逆序里真正需要交换的节点；判断逆序的比较
        必须严格用"前一个访问的节点"（中序序列里紧邻的前驱），而不是父子
        关系上的父节点——这两者在 BST 里通常不是同一个节点。
    """
    first = second = prev = None

    def dfs(node: "TreeNode | None") -> None:
        nonlocal first, second, prev
        if node is None:
            return
        dfs(node.left)
        if prev is not None and prev.val > node.val:
            if first is None:
                first = prev
            second = node
        prev = node
        dfs(node.right)

    dfs(root)
    if first is not None and second is not None:
        first.val, second.val = second.val, first.val


def _self_test() -> None:
    assert _tree_to_level_list(search_bst(build_tree([4, 2, 7, 1, 3]), 2)) == [2, 1, 3]
    assert search_bst(build_tree([4, 2, 7, 1, 3]), 5) is None

    inserted = insert_into_bst(build_tree([4, 2, 7, 1, 3]), 5)
    assert _inorder(inserted) == [1, 2, 3, 4, 5, 7]

    deleted = delete_node(build_tree([5, 3, 6, 2, 4, None, 7]), 3)
    assert _inorder(deleted) == [2, 4, 5, 6, 7]
    assert is_valid_bst(deleted) is True

    assert num_trees(3) == 5
    assert num_trees(1) == 1

    trees = generate_trees(3)
    assert len(trees) == 5
    for t in trees:
        assert _inorder(t) == [1, 2, 3]
        assert is_valid_bst(t) is True

    converted = convert_bst(build_tree([0, None, 1]))
    assert _inorder(converted) == [1, 1]

    recovered = build_tree([1, 3, None, None, 2])
    recover_tree(recovered)
    assert _inorder(recovered) == [1, 2, 3]

    print(
        "[PASS] p09_bst_ii: 7 题"
        "（BST中的搜索/BST插入/删除BST节点/不同的二叉搜索树/不同的二叉搜索树II/"
        "把BST转换为累加树/恢复二叉搜索树）全部通过"
    )


if __name__ == "__main__":
    _self_test()
