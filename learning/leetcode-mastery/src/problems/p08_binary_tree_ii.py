"""分类 08·二叉树遍历与构造 · 进阶补充（Part II）：不重讲框架，扩大遍历/构造/
序列化变体覆盖面的 12 道题。

Part I 的心法依旧成立——"想清楚这个函数对一棵子树做什么、需要子树返回什么信息"。
这批题在此基础上再加一条主线：**递归/遍历的返回值，除了"一个值"，还可以是
"一整条路径""一片森林""一个字符串编码""还剩多少个节点没揭晓"**——返回值的形状
决定了整道题怎么设计。
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


def _tree_to_level_list(root: "TreeNode | None") -> list:
    """仅用于测试校验：把树转回层序数组（截掉末尾多余的 None），方便和期望结果比较。"""
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


# ── LC144 二叉树的前序遍历（Easy） ────────────────────────────────────────
def preorder_traversal(root: "TreeNode | None") -> list[int]:
    """
    【题意】给二叉树根节点，返回它的前序遍历结果（根 → 左子树 → 右子树）。
    【思路】迭代版用一个显式栈：先访问栈顶（弹出即访问，因为"根"永远最先被访问），
        然后把它的右孩子、左孩子依次压栈——注意压栈顺序必须"先右后左"，这样左孩子
        才会压在栈顶、下一次弹出时先被访问，保证"左子树先于右子树"这条前序的定义。
    【复杂度】时间 O(n)（每个节点入栈出栈各一次），空间 O(h)（h 为树高，最坏 O(n)）。
    【易错点】压栈顺序写反（先压左孩子再压右孩子）会导致右子树被先访问，得到的是
        "根-右-左"而不是"根-左-右"；空树要单独返回空列表。
    """
    if root is None:
        return []
    result: list[int] = []
    stack: list[TreeNode] = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result


# ── LC145 二叉树的后序遍历（Easy） ────────────────────────────────────────
def postorder_traversal(root: "TreeNode | None") -> list[int]:
    """
    【题意】给二叉树根节点，返回它的后序遍历结果（左子树 → 右子树 → 根）。
    【思路】后序遍历直接用栈写并不直观，但有一个经典技巧：**后序"左-右-根"正好是
        "根-右-左"的逆序**。而"根-右-左"只需要在前序遍历的基础上把压栈顺序反过来
        （先压左孩子、再压右孩子，让右孩子压在栈顶先被访问）就能得到——本质上是
        `preorder_traversal` 的镜像写法。所以整体流程是：用镜像前序（根-右-左）
        收集一遍结果，最后把结果整体反转，就得到了后序（左-右-根）。
    【复杂度】时间 O(n)，空间 O(h)。
    【易错点】容易把"镜像前序"的压栈顺序和真前序搞反——这里必须先压 left 再压
        right（右孩子在栈顶先出栈），是和 `preorder_traversal` 刻意相反的写法；
        最后一步反转 result 不能漏，漏了得到的只是"根-右-左"而不是后序。
    """
    if root is None:
        return []
    result: list[int] = []
    stack: list[TreeNode] = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    result.reverse()
    return result


# ── LC543 二叉树的直径（Easy） ────────────────────────────────────────────
def diameter_of_binary_tree(root: "TreeNode | None") -> int:
    """
    【题意】二叉树的直径定义为任意两个节点之间最长路径的**边数**（这条路径可能不
        经过根节点）。给根节点，求直径。
    【思路】和 LC104 最大深度是同一个递归骨架，但这次要求这个函数**顺带**多算一件
        事：递归函数本身仍然只返回"以当前节点为顶点的最大深度"（`1 + max(左深度,
        右深度)`，和 LC104 一模一样），但在计算的过程中，用一个外部变量记录
        "经过当前节点、把左右两侧深度加起来"的值（`左深度 + 右深度`，这就是"以
        当前节点为拐点"的最长路径边数），每个节点都更新一次这个全局最大值。
        这是"函数返回值"和"顺带更新的全局答案"分离的又一个例子（呼应 Part I 里
        LC124 最大路径和的写法）。
    【复杂度】时间 O(n)，空间 O(h)（递归栈）。
    【易错点】直径的单位是"边数"不是"节点数"，所以更新全局答案时用的是
        `左深度 + 右深度`（两段边长之和），而不是 `左深度 + 右深度 + 1`（那会把
        当前节点自己也算成一条边）；把"返回值"错误地写成"左深度+右深度"会导致
        父节点用错误的量继续计算深度。
    """
    best = 0

    def depth(node: "TreeNode | None") -> int:
        nonlocal best
        if node is None:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        best = max(best, left_depth + right_depth)
        return 1 + max(left_depth, right_depth)

    depth(root)
    return best


# ── LC113 路径总和 II（Medium） ───────────────────────────────────────────
def path_sum_ii(root: "TreeNode | None", target_sum: int) -> list[list[int]]:
    """
    【题意】给二叉树根节点和目标值 target_sum，找出所有"从根节点到叶子节点"路径
        上节点值之和等于 target_sum 的路径，返回这些路径（每条路径是节点值列表）。
    【思路】标准回溯模板：用一个共享的 `path` 列表记录"当前从根走到这里的路径"，
        每进入一个节点就把它的值追加进去、把剩余目标减去当前值；如果走到叶子节点
        且剩余目标恰好归零，说明这条路径合法，把 `path` **拷贝一份**收进结果
        （必须拷贝，因为 `path` 之后还会被继续修改）；不管合不合法，递归返回前都要
        把当前节点的值从 `path` 弹出（回溯），恢复现场给兄弟分支使用。
    【复杂度】时间 O(n^2) 最坏情况（每条根到叶子的路径长度最坏 O(n)，且每次收集
        答案时拷贝 path 也要 O(n)），空间 O(h)（不计输出）。
    【易错点】收集答案时忘记用 `list(path)` 拷贝、直接把 `path` 本身的引用放进
        结果，后续回溯修改 `path` 会连带把已收集的答案也改乱；判断"叶子节点"要
        同时检查 `left is None and right is None`，只判断其中一个会把只有单侧
        子树的节点误判为叶子。
    """
    result: list[list[int]] = []
    path: list[int] = []

    def dfs(node: "TreeNode | None", remaining: int) -> None:
        if node is None:
            return
        path.append(node.val)
        remaining -= node.val
        if node.left is None and node.right is None and remaining == 0:
            result.append(list(path))
        else:
            dfs(node.left, remaining)
            dfs(node.right, remaining)
        path.pop()

    dfs(root, target_sum)
    return result


# ── LC437 路径总和 III（Medium） ──────────────────────────────────────────
def path_sum_iii(root: "TreeNode | None", target_sum: int) -> int:
    """
    【题意】给二叉树根节点和目标值 target_sum，统计路径数目，路径**不需要从根节点
        开始，也不需要在叶子节点结束**，但必须沿父到子的方向（不能往上走）。
    【思路】如果对每个节点都重新往下暴力枚举一遍所有路径，会是 O(n^2)。更好的做法
        借用"前缀和数组里区间和"的思路：DFS 沿途维护一个"从根到当前节点"的
        累加和 `curr_sum`，以及一个哈希表 `prefix_count` 记录"路径上每个前缀和
        出现过多少次"。对当前节点而言，"以它结尾、和为 target_sum 的路径数"等价
        于"前缀和数组里，有多少个更早的前缀和恰好等于 `curr_sum - target_sum`"
        （区间和 = 两个前缀和之差）。处理完当前节点、递归完左右子树之后，必须把
        当前节点贡献的前缀和计数**减回去**（回溯），因为这个前缀和只在"从根到
        当前节点"这条链上有效，兄弟分支不应该看到它。
    【复杂度】时间 O(n)（每个节点 O(1) 查询和更新哈希表），空间 O(h)（哈希表里
        最多同时存在从根到当前节点这条链上的 h 个前缀和）。
    【易错点】初始化 `prefix_count = {0: 1}` 这一条容易漏掉——它对应"路径恰好从根
        节点自己开始"这种情况（前缀和减去 target_sum 后如果正好是 0，说明从根到
        当前节点这整条路径就满足条件）；忘记回溯（递归返回后不把当前前缀和计数
        减回去）会让后面不相关的兄弟分支错误地复用这条路径的前缀和。
    """
    prefix_count: dict[int, int] = {0: 1}
    total = 0

    def dfs(node: "TreeNode | None", curr_sum: int) -> None:
        nonlocal total
        if node is None:
            return
        curr_sum += node.val
        total += prefix_count.get(curr_sum - target_sum, 0)
        prefix_count[curr_sum] = prefix_count.get(curr_sum, 0) + 1
        dfs(node.left, curr_sum)
        dfs(node.right, curr_sum)
        prefix_count[curr_sum] -= 1

    dfs(root, 0)
    return total


# ── LC662 二叉树最大宽度（Medium） ────────────────────────────────────────
def width_of_binary_tree(root: "TreeNode | None") -> int:
    """
    【题意】二叉树某一层的宽度定义为"该层最左、最右两个**非空**节点之间（包含它们
        自己）的节点数"，中间的空位也要算进宽度里。求所有层里最大的宽度。
    【思路】把整棵满二叉树按堆的方式编号：根节点编号 0，某节点编号为 `idx`，则左
        孩子编号 `2*idx`、右孩子编号 `2*idx+1`——这样"宽度"就等于"同一层里最大
        编号减最小编号再 +1"，把"数空位"变成了纯粹的编号减法。问题是编号在满
        二叉树深处会指数级增长、很快就溢出/变得没必要地大，所以每层开始处理时，
        都用"当前层第一个节点的编号"作为基准，把这一层所有节点的编号都减去这个
        基准做**归一化**——同一层内相对距离不变，但数值不会跟着树的深度一直往
        上累积。
    【复杂度】时间 O(n)（BFS 每个节点访问一次），空间 O(n)（最后一层最多 n/2 个
        节点在队列里）。
    【易错点】不做每层归一化、直接用"从根开始的绝对编号"，树深一点就会出现巨大
        整数（虽然 Python 不会整数溢出，但失去了归一化本该带来的直觉和效率）；
        算宽度时用的是"该层第一个节点"和"最后一个节点"的编号差，中间即使有
        大段空位（左右孩子都缺失的位置）也要被这段差值"隐性地"计入，不需要真的
        遍历中间的空位去数。
    """
    if root is None:
        return 0
    max_width = 0
    queue: list[tuple[TreeNode, int]] = [(root, 0)]
    while queue:
        level_size = len(queue)
        first_idx = queue[0][1]
        level_last_idx = 0
        for _ in range(level_size):
            node, idx = queue.pop(0)
            idx -= first_idx  # 相对本层最左节点做偏移归一化，避免编号无限增长
            level_last_idx = idx
            if node.left:
                queue.append((node.left, idx * 2))
            if node.right:
                queue.append((node.right, idx * 2 + 1))
        max_width = max(max_width, level_last_idx + 1)
    return max_width


# ── LC199 二叉树的右视图（Medium） ────────────────────────────────────────
def right_side_view(root: "TreeNode | None") -> list[int]:
    """
    【题意】给二叉树根节点，假设站在树的右侧，返回从上到下能看到的节点值（每层
        最右边那个能被看到的节点）。
    【思路】直接复用 LC102 层序遍历"记录 level_size、按层处理"的模板，只是这次
        每层只留下**最后一个**被处理的节点——因为 BFS 每层是严格从左到右处理，
        "这一层第 `level_size - 1` 个被弹出的节点"正好就是这一层最右边、从右侧
        能看到的那个节点。
    【复杂度】时间 O(n)，空间 O(n)（最坏情况下最后一层的队列大小）。
    【易错点】误以为"每层第一个节点"是右视图（那其实是左视图），把判断条件写反
        成 `i == 0`；空树要单独返回空列表。
    """
    if root is None:
        return []
    result: list[int] = []
    queue: list[TreeNode] = [root]
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.pop(0)
            if i == level_size - 1:
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result


# ── LC297 二叉树的序列化与反序列化（Hard） ────────────────────────────────
class Codec:
    """
    【题意】设计一种算法，把一棵二叉树编码成一个字符串（序列化），再把这个字符串
        还原回原来的树结构（反序列化）——不要求还原出"值相同的另一棵树"，而是要求
        结构完全一致。
    【思路】用前序遍历（根→左→右）拼接节点值，并且**显式地为每个空节点写一个占位
        符**（这里用 `"#"`），用逗号分隔。反序列化时按同样的前序顺序、用一个迭代器
        依次取值消费：取到占位符就返回 None；否则新建节点，再**先递归构造左子树、
        后构造右子树**（顺序必须和序列化时完全对应，因为迭代器是一次性顺序消费，
        不能回退）。为什么"前序 + 显式 None 占位"能唯一还原一棵树，见本文件对应
        lecture 的深挖部分。
    【复杂度】时间 O(n)（序列化、反序列化各遍历一次所有节点，包括空节点），
        空间 O(n)（字符串长度、递归栈）。
    【易错点】反序列化时如果用 `list.pop(0)` 从头部消费会是 O(n) 一次、整体退化
        成 O(n^2)，更好的写法是用 `iter(...)` + `next(...)` 保持 O(1) 消费；
        构造左右子树的顺序必须严格和序列化时的写入顺序一致（先左后右），写反会
        导致树的结构镜像错误。
    """

    def serialize(self, root: "TreeNode | None") -> str:
        values: list[str] = []

        def dfs(node: "TreeNode | None") -> None:
            if node is None:
                values.append("#")
                return
            values.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(values)

    def deserialize(self, data: str) -> "TreeNode | None":
        values = iter(data.split(","))

        def dfs() -> "TreeNode | None":
            val = next(values)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


# ── LC106 从中序与后序遍历序列构造二叉树（Medium） ────────────────────────
def build_tree_from_in_post(inorder: list[int], postorder: list[int]) -> "TreeNode | None":
    """
    【题意】给一棵二叉树的中序遍历序列 inorder 和后序遍历序列 postorder（节点值
        互不相同），重建这棵二叉树，返回根节点。
    【思路】和 Part I 的 LC105（前序+中序）是镜像关系：后序遍历是"左-右-根"，
        所以**最后一个元素永远是当前子树的根**（对应 LC105 里"前序第一个元素是
        根"）。同样用中序序列里根的位置 `mid` 把左右子树切开。区别在于游标的移动
        方向：既然根是从 postorder **末尾**取的，就必须**从右往左**消费
        postorder——而 postorder 的顺序是"左子树的后序、右子树的后序、根"，所以
        从末尾往前退的顺序天然是"根、右子树的后序（倒着看）、左子树的后序（倒着
        看）"，因此递归时必须**先构造右子树、再构造左子树**（和 LC105 的"先左后
        右"刚好相反），这样游标退到的位置才会一直对得上。
    【复杂度】时间 O(n)（哈希表预处理 O(n) + 每个节点 O(1) 定位根），空间 O(n)。
    【易错点】递归顺序写成"先左后右"（照抄 LC105 的顺序）会导致游标和实际值对不
        上，构造出结构错误的树——这是这题最容易掉进的陷阱，因为直觉上"先处理左
        子树"更符合阅读习惯，但这题游标是从后往前走，必须先处理右子树；游标初始
        值要设成 `len(postorder) - 1`（指向最后一个元素），不是 0。
    """
    index_of = {val: i for i, val in enumerate(inorder)}
    post_idx = [len(postorder) - 1]  # 从后往前消费 postorder 的游标

    def helper(in_left: int, in_right: int) -> "TreeNode | None":
        if in_left > in_right:
            return None
        root_val = postorder[post_idx[0]]
        post_idx[0] -= 1
        root = TreeNode(root_val)
        mid = index_of[root_val]
        root.right = helper(mid + 1, in_right)  # 必须先构造右子树
        root.left = helper(in_left, mid - 1)    # 游标才会自动落在左子树的根上
        return root

    return helper(0, len(inorder) - 1)


# ── LC958 二叉树的完全性检验（Medium） ────────────────────────────────────
def is_complete_tree(root: "TreeNode | None") -> bool:
    """
    【题意】给二叉树根节点，判断它是否是一棵"完全二叉树"（除最后一层外每层都被
        填满，且最后一层的节点都靠左连续排列，中间不能有空缺）。
    【思路】BFS 时不管节点是否为空都入队（包括 None），一旦在队列中"见过一个
        None"，之后就不应该再出现非 None 节点——因为完全二叉树的性质决定了所有
        空位必须"扎堆"出现在层序遍历的末尾，一旦中间出现了空位后面又冒出真实
        节点，说明这个空位不是"末尾的收尾空位"，而是"中间的缺口"，破坏了完全性。
    【复杂度】时间 O(n)，空间 O(n)（队列中包含 None 占位）。
    【易错点】容易只对"非空节点"入队（沿用普通 BFS 的写法），这样会漏掉"空位
        之后又出现非空节点"这一关键信息，必须连 None 也一起入队才能检测出中间的
        缺口；空树按定义视为合法的完全二叉树，直接返回 True。
    """
    if root is None:
        return True
    queue: list["TreeNode | None"] = [root]
    seen_none = False
    while queue:
        node = queue.pop(0)
        if node is None:
            seen_none = True
        else:
            if seen_none:
                return False
            queue.append(node.left)
            queue.append(node.right)
    return True


# ── LC1110 删点成林（Medium） ─────────────────────────────────────────────
def del_nodes(root: "TreeNode | None", to_delete: list[int]) -> list["TreeNode"]:
    """
    【题意】给二叉树根节点和一个待删除值列表 to_delete，删除树中值在该列表里的
        所有节点；删除后，原来连接在被删节点下面、且自身没被删除的子树各自独立
        成为一棵新树，返回森林（这些新树根节点组成的列表）。
    【思路】后序遍历（先处理左右子树、再处理当前节点），这样"当前节点是否会被
        整棵删掉"这个决定要等子树都递归处理完之后才做，天然符合"先剪枝、再判断"
        的顺序。递归函数额外带一个 `is_root` 参数，表示"当前节点的父节点是否已经
        被删除（或者当前节点本来就是整棵树的根）"：只有当 `is_root` 为真、且当前
        节点自己没被删除时，它才有资格作为森林里的一棵新树的根，加入结果列表。
        递归返回值是"这个节点处理完之后，应该挂在父节点对应位置上的结果"——如果
        自己被删了就返回 None（父节点对应的指针置空），否则返回自己。
    【复杂度】时间 O(n)（每个节点访问一次），空间 O(n)（to_delete 转成集合 O(1)
        查询 + 递归栈）。
    【易错点】`is_root` 判断的是"父节点是否被删"，不是"当前节点是否被删"——一个
        节点自己没被删，但如果它的父节点被删了，它同样有资格成为新的森林根；
        森林根的加入条件必须同时满足"父节点已删（或本来就是根）"**和**"自己没
        被删"，只判断其中一个都会漏掉或多算。
    """
    to_delete_set = set(to_delete)
    forest: list["TreeNode"] = []

    def dfs(node: "TreeNode | None", is_root: bool) -> "TreeNode | None":
        if node is None:
            return None
        deleted = node.val in to_delete_set
        if is_root and not deleted:
            forest.append(node)
        node.left = dfs(node.left, deleted)
        node.right = dfs(node.right, deleted)
        return None if deleted else node

    dfs(root, True)
    return forest


# ── LC1325 删除给定值的叶子节点（Medium） ─────────────────────────────────
def remove_leaf_nodes(root: "TreeNode | None", target: int) -> "TreeNode | None":
    """
    【题意】给二叉树根节点和目标值 target，重复地删除值等于 target 的叶子节点，
        直到树中不再存在值为 target 的叶子节点为止，返回处理后的根节点。
    【思路】"重复删除"这个描述容易让人以为要写一个外层循环反复扫描，但其实一次
        后序遍历就够：递归先处理完左右子树（这一步保证了"如果某个孩子删除后，
        自己变成了新的叶子"这种连锁反应，已经在更深层的递归调用里被处理过），
        再检查当前节点——如果处理完子树后，当前节点自己**现在**变成了叶子（左右
        都是 None）且值等于 target，就返回 None 把它删掉。因为是自底向上处理，
        删除会自然地"传导"到父节点：父节点在自己的这一层递归里会看到孩子变成
        了 None，如果这导致父节点自己也变成叶子且值等于 target，同样会被删除。
    【复杂度】时间 O(n)，空间 O(h)（递归栈）。
    【易错点】如果只做"自顶向下"的一次性判断（先判断当前节点是不是叶子，再决定
        要不要递归），会漏掉"删除子节点后父节点新变成叶子"这种连锁删除，必须
        保证是先递归处理子树、再检查当前节点（后序）；判断叶子必须在**递归赋值
        之后**检查 `root.left is None and root.right is None`，而不是用删除前
        的旧状态判断。
    """
    if root is None:
        return None
    root.left = remove_leaf_nodes(root.left, target)
    root.right = remove_leaf_nodes(root.right, target)
    if root.left is None and root.right is None and root.val == target:
        return None
    return root


def _has_leaf_with_value(root: "TreeNode | None", target: int) -> bool:
    """仅用于测试校验：检查树里是否还存在值为 target 的叶子节点。"""
    if root is None:
        return False
    if root.left is None and root.right is None:
        return root.val == target
    return _has_leaf_with_value(root.left, target) or _has_leaf_with_value(root.right, target)


def _self_test() -> None:
    assert preorder_traversal(build_tree([1, None, 2, 3])) == [1, 2, 3]
    assert preorder_traversal(build_tree([])) == []

    assert postorder_traversal(build_tree([1, None, 2, 3])) == [3, 2, 1]
    assert postorder_traversal(build_tree([])) == []

    assert diameter_of_binary_tree(build_tree([1, 2, 3, 4, 5])) == 3
    assert diameter_of_binary_tree(build_tree([1, 2])) == 1

    assert sorted(path_sum_ii(
        build_tree([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]), 22
    )) == sorted([[5, 4, 11, 2], [5, 8, 4, 5]])

    assert path_sum_iii(
        build_tree([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1]), 8
    ) == 3

    assert width_of_binary_tree(build_tree([1, 3, 2, 5, 3, None, 9])) == 4
    assert width_of_binary_tree(build_tree([1, 3, 2, 5])) == 2

    assert right_side_view(build_tree([1, 2, 3, None, 5, None, 4])) == [1, 3, 4]
    assert right_side_view(build_tree([1, None, 3])) == [1, 3]

    codec = Codec()
    ser_root = build_tree([1, 2, 3, None, None, 4, 5])
    restored = codec.deserialize(codec.serialize(ser_root))
    assert _tree_to_level_list(restored) == _tree_to_level_list(ser_root)
    assert _tree_to_level_list(codec.deserialize(codec.serialize(build_tree([])))) == []

    inorder, postorder = [9, 3, 15, 20, 7], [9, 15, 7, 20, 3]
    rebuilt = build_tree_from_in_post(inorder, postorder)
    assert preorder_traversal(rebuilt) == [3, 9, 20, 15, 7]

    assert is_complete_tree(build_tree([1, 2, 3, 4, 5, 6])) is True
    assert is_complete_tree(build_tree([1, 2, 3, 4, 5, None, 7])) is False

    forest = del_nodes(build_tree([1, 2, 3, 4, 5, 6, 7]), [3, 5])
    assert sorted(node.val for node in forest) == [1, 6, 7]

    recovered = remove_leaf_nodes(build_tree([1, 2, 3, 2, None, 2, 4]), 2)
    assert _has_leaf_with_value(recovered, 2) is False

    print(
        "[PASS] p08_binary_tree_ii: 12 题"
        "（前序遍历/后序遍历/二叉树的直径/路径总和II/路径总和III/二叉树最大宽度/"
        "右视图/序列化与反序列化/中序后序建树/完全性检验/删点成林/删除叶子节点）"
        "全部通过"
    )


if __name__ == "__main__":
    _self_test()
