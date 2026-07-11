"""分类 33：高级数据结构 —— Phase 3 竞赛级新分类。

这批是更专精的数据结构组件：不像前面大多数分类那样直接对应"一类算法技巧"，
而是各自解决一个具体的工程/竞赛场景——它们未必会在面试题里原封不动地出现，但
经常是解决其他难题时的"底层零件"（比如离线算法用可撤销并查集、静态区间最值用
稀疏表、前缀匹配类问题用字典树的变体）。这一章把 1 道真实 LeetCode 设计题
（时间键值存储）、1 个受 LC1206"设计跳表"启发但改成确定性层数规则的自实现
变体（LC1206 本身的标准随机化实现见 21 类），和 6 个其他自实现的经典数据
结构放在一起（频率栈 LC895、全 O(1) 数据结构 LC432 已在 21 类实现，不在这里
重复），重点不是"记住某道题的解法"，而是理解每个结构"为什么是这个形状"——
它用什么代价换取了什么操作的加速。
"""
from __future__ import annotations

import bisect
from collections import defaultdict


class TimeMap:
    """
    【题意】LC981·基于时间的键值存储（Medium）。`set(key, value, timestamp)` 在
    严格递增的 timestamp 下给 key 存一个新的 value（同一个 key 会被多次 set，
    timestamp 保证严格递增）；`get(key, timestamp)` 返回"该 key 在所有
    `timestamp_prev <= timestamp` 里，timestamp_prev 最大的那次 set 存的
    value"；如果不存在这样的记录，返回空字符串。

    【思路】既然同一个 key 的多次 set 是按 timestamp **严格递增**的顺序调用的，
    每个 key 对应的 `(timestamp, value)` 历史记录天然就是一个按 timestamp 排好
    序的数组，不需要额外排序。`get` 要找的是"历史记录里，timestamp_prev 不超过
    查询值的最后一条"——这是标准的"在有序序列里找最后一个 <= 目标值的位置"，用
    `bisect_right` 在只存 timestamp 的辅助数组上二分定位，取它减一的下标即可，
    不需要线性扫描整条历史（这个技巧和 21 类"快照数组"的查询完全同构）。

    【复杂度】`set` 均摊 O(1)（列表尾部追加）；`get` 为 O(log h)（h 是该 key 的
    历史记录条数，二分查找）；空间 O(总 set 调用次数)。

    【易错点】
    - 二分要用 `bisect_right`（不是 `bisect_left`）再减一——`bisect_right` 定位
      到的是"第一个严格大于目标值的位置"，减一才是"最后一个 <= 目标值的位置"；
      用 `bisect_left` 会在"历史记录里恰好存在等于目标 timestamp 的那条"时，
      漏掉这条记录本身。
    - 如果查询的 key 从未 `set` 过，或者查询的 timestamp 早于该 key 第一次
      `set` 的时刻，都要返回空字符串 `""`，而不是抛出下标越界异常——对应二分
      结果下标为 -1 的情况要单独判断。
    - 不能假设"最新的 value 一定是历史记录的最后一项就是答案"——只有当查询的
      timestamp 大于等于最后一次 set 的 timestamp 时才成立，题目允许查询任意
      historical timestamp，必须做真正的二分定位而不是直接取最后一项。
    """

    def __init__(self) -> None:
        self.times: dict[str, list[int]] = defaultdict(list)
        self.values: dict[str, list[str]] = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.times[key].append(timestamp)
        self.values[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        ts = self.times.get(key)
        if not ts:
            return ""
        i = bisect.bisect_right(ts, timestamp) - 1
        return self.values[key][i] if i >= 0 else ""


class _SkipNode:
    __slots__ = ("val", "forward")

    def __init__(self, val: float, level: int) -> None:
        self.val = val
        self.forward: list["_SkipNode | None"] = [None] * level


class SimpleSkipList:
    """
    【题意】自实现数据结构（不注册为 LC1206——LC1206"设计跳表"的标准随机化
    实现已在 21 类 `p21_design_iii.py` 的 `Skiplist` 里完成，这里是同一问题的
    **确定性变体**，用于对比"标准随机化"和"确定性替代"两种设计的取舍）。
    实现 `insert(num)`、
    `search(target) -> bool`、`erase(num) -> bool`（数值可能重复，erase 命中
    任意一个即可），要求期望 O(log n)。

    【思路】跳表是"用空间换时间的概率数据结构"的代表：在一条普通有序链表之上，
    再叠加若干层"稀疏"的链表——第 0 层包含所有元素，第 1 层大约包含一半元素，
    第 2 层大约四分之一……每一层都只是"跳过"下一层的部分节点，查找时从最高层
    开始，尽量往右走，走不动了再降一层，逼近二分查找的效果，期望 O(log n)。
    标准实现给每个新插入节点"随机"分配一个层数（抛硬币，抛到反面才停，越往上
    层数越少）；本实现按题目要求换成**完全确定性**的层数规则以保证可复现：
    第 k 次插入（从 1 开始计数）的层数 = "k 的二进制表示里末尾连续 0 的个数 + 1"
    （k=1→0 个尾随 0→层数 1；k=2→1 个→层数 2；k=4→2 个→层数 3；k=8→3 个→
    层数 4……）。这个规则模拟了"大约一半元素升到下一层、大约四分之一升到再上
    一层"的概率分布——因为在 1..k 里，恰好一半的数是偶数（至少 1 个尾随 0），
    四分之一是 4 的倍数（至少 2 个尾随 0），以此类推——但整个过程不依赖任何
    随机数，同一个插入序列永远得到同一棵跳表结构，便于测试和复现。`insert`/
    `search`/`erase` 都共享同一套"从最高层开始、每层尽量往右走、走不动再降一
    层"的定位逻辑：定位到的位置，要么是目标值所在的节点（search/erase 命中），
    要么是"目标值应该插入的位置"（insert）。

    【复杂度】期望（在本实现里是"按确定性规则模拟出的近似"）O(log n)（层数期望
    O(log n)，每层最多走常数步）；空间 O(n)（每个节点额外占用的层数之和期望
    O(n)）。

    【易错点】
    - 确定性层数规则是本题和标准跳表最大的区别——标准实现靠随机化保证"期望"
      O(log n)，如果测试数据的插入顺序恰好是这套确定性规则的最坏情况（比如
      故意构造成让所有元素都分到层数 1），单次操作会退化到 O(n)；这里选择
      "末尾 0 的个数"这个规则，是因为它对任意插入顺序都能给出和二进制计数器
      一致的、类似真实概率分布的层数分配，不会退化，但**这是一种工程简化，
      不是标准跳表实现**，需要在文档里明确写出这一点。
    - 允许重复值：`insert` 定位到的是"第一个 >= num 的位置"并插入在它前面，
      多个相同值的节点会作为独立节点共存；`erase` 命中的是"第一个值等于 num
      的节点"，删除任意一个都算合法。
    - `erase` 时必须在每一层分别检查"这一层的前驱节点的 forward 指针是否恰好
      指向被删除的节点"——被删除的节点不一定在所有层都出现（层数可能小于跳表
      当前最高层数），只在它自己拥有的那几层做指针摘除，其余层不做任何操作。
    """

    MAX_LEVEL = 20  # 题目约束下 (至多约 5*10^4 次调用) 足够覆盖的层数上限

    def __init__(self) -> None:
        self.head = _SkipNode(float("-inf"), self.MAX_LEVEL)
        self.level = 1
        self.insert_count = 0

    def _deterministic_level(self) -> int:
        self.insert_count += 1
        k = self.insert_count
        trailing_zeros = 0
        while k % 2 == 0:
            trailing_zeros += 1
            k //= 2
        return min(trailing_zeros + 1, self.MAX_LEVEL)

    def search(self, target: int) -> bool:
        node = self.head
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].val < target:
                node = node.forward[i]
        node = node.forward[0]
        return node is not None and node.val == target

    def insert(self, num: int) -> None:
        update: list[_SkipNode] = [self.head] * self.MAX_LEVEL
        node = self.head
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].val < num:
                node = node.forward[i]
            update[i] = node
        new_level = self._deterministic_level()
        if new_level > self.level:
            for i in range(self.level, new_level):
                update[i] = self.head
            self.level = new_level
        new_node = _SkipNode(num, new_level)
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def erase(self, num: int) -> bool:
        update: list[_SkipNode] = [self.head] * self.MAX_LEVEL
        node = self.head
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].val < num:
                node = node.forward[i]
            update[i] = node
        target = node.forward[0]
        if target is None or target.val != num:
            return False
        for i in range(self.level):
            if update[i].forward[i] is target:
                update[i].forward[i] = target.forward[i]
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        return True


class SparseTable:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：稀疏表（Sparse Table），
    专门用于"数组构造后不再修改"场景下的静态区间最值查询，要求预处理
    O(n log n)、每次查询 O(1)（比线段树的 O(log n) 查询更快，代价是完全不支持
    修改）。要求 `query_min(l, r) -> int` 返回下标 `[l, r]`（0-indexed 闭区间）
    内的最小值。

    【思路】核心 insight 是"区间最值有一个前缀和没有的特殊性质——重叠不会导致
    错误"：如果要查询的区间 `[l, r]` 能被两个**有重叠**的、长度相同的预处理好的
    区间覆盖（只要它们的并集等于 `[l, r]`），直接取这两个区间最小值的较小者，
    结果仍然完全正确——因为 min 运算满足"可重复贡献"（重复统计同一个元素不会
    影响最终的最小值），这一点是前缀和（只能做恰好不重叠的区间拼接）不具备的。
    利用这个性质：预处理 `table[j][i]` 表示"从下标 i 开始、长度为 `2^j` 的区间"
    的最小值，通过倍增递推
    `table[j][i] = min(table[j-1][i], table[j-1][i + 2^(j-1)])`（把长度
    `2^j` 的区间拆成两个长度 `2^(j-1)` 的子区间，取较小值），预处理表格一共
    O(n log n) 个状态，每个状态 O(1) 递推。查询 `[l, r]` 时，设区间长度为
    `len = r - l + 1`，取 `j = floor(log2(len))`，用两个长度为 `2^j`、起点分别
    是 `l` 和 `r - 2^j + 1` 的区间（这两个区间可能有重叠，但并集恰好覆盖
    `[l, r]`）,取 `min(table[j][l], table[j][r - 2^j + 1])` 即为答案——O(1)。

    【复杂度】预处理 O(n log n) 时间、O(n log n) 空间；每次查询 O(1)。不支持
    修改（一旦数组变化，整张表都要重新预处理）。

    【易错点】
    - 稀疏表能做到 O(1) 查询、且允许两个查询区间重叠，**仅仅因为**要维护的
      信息是"幂等"的（min/max/gcd/and/or 这类"重复统计不影响结果"的运算）——
      如果拿来维护区间和，重叠部分会被重复计算导致结果错误，这种场景必须用
      线段树或树状数组，不能用稀疏表。
    - `j = floor(log2(len))` 要预先打表（`self.log[i]`）而不是每次查询都调用
      `math.log2`——一是avoid 浮点误差导致 j 算错1（比如 `log2(8)` 因为浮点
      精度算成 2.999999），二是重复计算 `log2` 会让本该 O(1) 的查询多出一个
      不小的常数开销。
    - 两个覆盖区间的起点分别是 `l` 和 `r - 2^j + 1`，第二个起点容易写成
      `l + 2^j`（那是"不重叠拼接"的直觉，只对前缀和成立）——稀疏表恰恰是利用
      "允许重叠"才做到 O(1) 查询，起点必须保证两个区间的并集覆盖整个
      `[l, r]`，而不是恰好首尾相接。
    """

    def __init__(self, arr: list[int]) -> None:
        self.n = len(arr)
        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
        max_j = self.log[self.n] + 1 if self.n > 0 else 1
        self.table = [list(arr)] + [[0] * self.n for _ in range(max_j - 1)]
        j = 1
        while (1 << j) <= self.n:
            half = 1 << (j - 1)
            limit = self.n - (1 << j) + 1
            for i in range(limit):
                self.table[j][i] = min(self.table[j - 1][i], self.table[j - 1][i + half])
            j += 1

    def query_min(self, l: int, r: int) -> int:
        length = r - l + 1
        j = self.log[length]
        return min(self.table[j][l], self.table[j][r - (1 << j) + 1])


class UndoableUnionFind:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：支持撤销最近一次
    `union` 操作的并查集，常用于离线算法（比如按某种顺序回退操作、或者需要
    "尝试合并、发现不满足条件就撤回"的场景）。要求 `union(a, b) -> bool`
    （合并成功返回 True，本来就已经在同一集合返回 False）、`undo()`——撤销最近
    一次 `union` 调用（不管它当时是否真正发生了合并）、`connected(a, b) ->
    bool`。

    【思路】普通并查集（17 类）为了追求 amortized O(α(n))，会用"路径压缩"——
    但路径压缩会把 find 路径上**很多**节点的 parent 指针都改掉，一次 union
    可能连带修改任意多个节点，这些修改无法用一条简单的历史记录撤销（要撤销就
    要记住路径压缩之前每一个被压缩节点的原始 parent，代价和重新算一遍差不多）。
    要支持"撤销最近一次 union"，必须放弃路径压缩，只用**按秩合并（union by
    rank）**——这样一次 union 只会修改**一个**节点的 parent（被合并的子树的
    根）、以及可能让**一个**节点的 rank 加一，两处修改都可以用一条历史记录
    精确回退。`find` 因为没有路径压缩，最坏情况下要沿着树走 O(log n) 步（按秩
    合并保证树高不超过 O(log n)），比普通并查集稍慢，这是"支持撤销"必须付出的
    代价。`union(a, b)`：先分别 find 到根 ra、rb；如果相等，往历史里记一个
    "空操作"标记（保证 `undo()` 的调用次数和 `union` 调用次数一一对应，不用
    调用方自己去区分"这次 union 到底有没有真正发生"）；否则按秩合并（rank 小
    的挂到 rank 大的下面），记录"谁被挂到了谁下面、这次挂载是否顺带让根节点的
    rank 加一"，供 `undo` 使用。`undo()`：弹出最近一条历史记录——如果是空操作
    直接返回；否则把被挂载的子树根节点的 parent 重新指回自己（这就是它union
    之前的状态，因为它当时正是自己的根），如果这次 union 顺带增加过 rank，
    再把 rank 减回去。

    【复杂度】`find`/`union`/`undo`/`connected` 均为 O(log n)（按秩合并保证的
    树高上界，没有路径压缩带来的均摊优化）；空间 O(n + 历史记录条数)。

    【易错点】
    - 绝对不能加路径压缩——哪怕只是"顺手"在 `find` 里加一行路径压缩代码，就会
      让 `undo` 撤销不完全（历史记录里没有记录路径压缩改动过的那些节点），
      导致撤销之后并查集的状态和"从未执行过这次 union"时不一致。
    - `union(a, b)` 在 `find(a) == find(b)`（已经连通，本次不会有任何实际修改）
      时，也必须往历史栈里压入一条"空操作"占位记录，而不是什么都不做——否则
      `union` 和 `undo` 的调用次数不再一一对应，后续的 `undo()` 会撤销到"更早
      一次真正发生过的 union"，而不是调用方以为的"上一次 union"。
    - `undo` 只需要处理"根节点挂载"和"rank 是否 +1 过"这两项——不需要、也不能
      尝试恢复 `find` 过程中的任何状态，因为没有路径压缩，`find` 本身从不修改
      并查集的状态，历史记录只需要覆盖 `union` 真正做出的修改。
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history: list[tuple[int, int, bool] | None] = []

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.history.append(None)
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        rank_bumped = self.rank[ra] == self.rank[rb]
        self.parent[rb] = ra
        if rank_bumped:
            self.rank[ra] += 1
        self.history.append((ra, rb, rank_bumped))
        return True

    def undo(self) -> None:
        record = self.history.pop()
        if record is None:
            return
        ra, rb, rank_bumped = record
        self.parent[rb] = rb
        if rank_bumped:
            self.rank[ra] -= 1

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


class _DequeNode:
    __slots__ = ("val", "idx", "prev", "next")

    def __init__(self, val: int, idx: int) -> None:
        self.val = val
        self.idx = idx
        self.prev: "_DequeNode | None" = None
        self.next: "_DequeNode | None" = None


class MonotonicQueue:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：不借助
    `collections.deque`，手写一个双端队列，支持 `push_back(val, idx)`（从队尾
    压入一个"值 + 原始下标"）、`pop_front()`/`pop_back()`（从队首/队尾弹出并
    返回)、`get_max()`（O(1) 查询当前队列里的最大值），用于实现"滑动窗口最大值"
    这类问题（这一技巧在 16 类"单调栈/单调队列"里已经用 `collections.deque`
    的现成接口做过，这里的练习目的是理解"手写一个 O(1) 首尾操作的双端队列"
    本身需要什么底层结构）。

    【思路】如果用 Python 原生 `list` 模拟双端队列，`list.pop(0)`（从头部弹出）
    是 O(n)（需要搬动剩余所有元素），不满足"双端都是 O(1)"的要求——这正是
    为什么要"手写"而不是拿 list 图省事。用**双向链表**（自己实现 `_DequeNode`，
    每个节点有 `prev`/`next` 指针）可以让首尾的插入/删除都是 O(1)，这也是
    `collections.deque` 内部的实现方式。在这个双向链表之上叠加"单调"性质——
    `push_back` 在真正把新节点接到队尾之前，先不断从队尾弹出"比新值还小"的
    旧节点（这些节点即使还没离开滑动窗口，也永远不可能再成为未来任何窗口的
    最大值，因为新值比它们大、且比它们更晚离开窗口，彻底"支配"了它们，可以
    安全丢弃）——这一步维护之后，队列从队首到队尾严格单调不增，队首恒为当前
    队列里的最大值，`get_max()` 直接读队首，O(1)。`pop_front()` 由调用方在
    "滑动窗口左边界滑过某个下标"时主动调用（配合节点上记录的 `idx` 字段判断
    队首是否已经滑出窗口），这一步不内置在类里，是为了让这个结构保持通用（不
    和"滑动窗口"这个具体应用场景强绑定）。

    【复杂度】`push_back`/`pop_front`/`pop_back`/`get_max` 均为**均摊** O(1)
    （`push_back` 内部虽然有一个 while 循环弹出旧节点，但每个元素一生只会被
    push 一次、pop 一次，均摊下来仍是 O(1)）；空间 O(当前队列内的元素个数)。

    【易错点】
    - `push_back` 里"弹出比新值小的队尾"这一步必须用 `<`（严格小于）而不是
      `<=`——如果改成 `<=`，遇到重复的最大值时会把"更早但同样是最大值"的元素
      也弹出，虽然不影响 `get_max()` 的结果，但会让"这个值到底还在不在窗口里"
      的下标信息丢失，影响后续 `pop_front` 基于下标判断是否过期的正确性。
    - `push_back`/`pop_front`/`pop_back` 都要正确维护 `prev`/`next` 双向指针
      ——尤其是队列从"只剩一个节点"变成"空队列"、或者从"空队列"变成"只剩一个
      节点"这两个边界，`self._head`/`self._tail` 要同步更新，漏掉任何一处都
      会导致下一次操作在一个"半失效"的链表上出错。
    - 这个类本身只保证"当前队列里的最大值在队首"，**不会**自动帮你判断某个
      元素是否已经滑出窗口——`pop_front` 什么时候调用、判断队首下标是否过期，
      是调用方（业务逻辑）的责任，这是"通用数据结构"和"针对某道题定制的解法"
      之间职责边界的体现。
    """

    def __init__(self) -> None:
        self._head: _DequeNode | None = None
        self._tail: _DequeNode | None = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push_back(self, val: int, idx: int) -> None:
        while self._tail is not None and self._tail.val < val:
            self.pop_back()
        node = _DequeNode(val, idx)
        if self._tail is None:
            self._head = self._tail = node
        else:
            node.prev = self._tail
            self._tail.next = node
            self._tail = node
        self._size += 1

    def pop_front(self) -> tuple[int, int]:
        if self._head is None:
            raise IndexError("pop_front from empty MonotonicQueue")
        node = self._head
        self._head = node.next
        if self._head is not None:
            self._head.prev = None
        else:
            self._tail = None
        self._size -= 1
        return node.val, node.idx

    def pop_back(self) -> tuple[int, int]:
        if self._tail is None:
            raise IndexError("pop_back from empty MonotonicQueue")
        node = self._tail
        self._tail = node.prev
        if self._tail is not None:
            self._tail.next = None
        else:
            self._head = None
        self._size -= 1
        return node.val, node.idx

    def front_index(self) -> int:
        if self._head is None:
            raise IndexError("front_index from empty MonotonicQueue")
        return self._head.idx

    def get_max(self) -> int:
        if self._head is None:
            raise IndexError("get_max from empty MonotonicQueue")
        return self._head.val


class SqrtDecomposition:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：分块（Sqrt
    Decomposition），支持区间批量加 `update_range(l, r, val)` 和区间和查询
    `query_range(l, r) -> int`——和 27 类的线段树 `SegmentTree` 解决的是完全
    相同的问题，放在一起是为了对比"两种思路都能做到区间更新/查询较快，什么时候
    该选哪一种"。

    【思路】把长度为 n 的数组切成大约 `sqrt(n)` 个块，每块大约 `sqrt(n)` 个
    元素，每块维护一个"块内元素和" `block_sum` 和一个"整块懒标记"
    `block_lazy`（表示"这一整块还有一个 +val 没有真正下推到每个元素"）。一次
    区间更新 `[l, r]` 跨越的块，分成三段处理：**左边不完整的那一块**——直接
    逐点把 `val` 加到具体元素上（这一步之前要先"下推"这一块可能存在的旧懒
    标记，否则会破坏"块内元素值 + 懒标记 = 真实值"这个不变量）；**中间完整覆盖
    的那些块**——不需要逐点修改，只需要把 `val` 累加到这些块的 `block_lazy`
    上，同时同步更新 `block_sum`（`+= val * 块长度`），O(1) 每块；**右边不
    完整的那一块**——和左边同样处理（先下推、再逐点加）。因为一次区间更新最多
    只有两个"不完整"的边界块需要逐点处理（每块最多 `sqrt(n)` 个元素），中间的
    完整块只需要 O(1) 打标记，一次更新总共 O(sqrt(n))。区间查询同理：边界块
    "下推懒标记后直接扫描求和"，中间的完整块直接读 `block_sum`，也是
    O(sqrt(n))。

    【复杂度】`update_range`/`query_range` 均为 O(sqrt(n))；空间 O(n)。比线段
    树的 O(log n) 慢，但实现更简单、常数因子更小，在 n 不太大、或者需要维护的
    信息不方便写成线段树的"合并"逻辑时，分块是更朴素直接的替代方案。

    【易错点】
    - 对边界块做"逐点直接加"之前，必须先调用 `_push_down` 把这一块之前可能
      残留的 `block_lazy` 真正下推到每个元素、并清零该块的懒标记——如果跳过
      这一步直接改 `arr[i]`，会让"块内元素值 + 懒标记"这个不变量被破坏（相当
      于同一批加法效果被错误地叠加或丢失）。
    - 最后一个块的长度可能小于 `block_size`（当 n 不是 block_size 的整数倍
      时），所有涉及"块长度"的地方（`block_lazy` 更新时乘的系数、下推时遍历
      的范围）都要用 `min(块起点 + block_size, n)` 计算实际块长，不能无脑用
      `block_size` 常量。
    - `block_sum[b]` 应该随时保持"这一块的真实当前总和"（不管这块此刻是靠
      `block_lazy` 整体表示还是已经下推成逐点的 `arr` 值），每次触碰这一块
      （无论是整块打懒标记还是逐点直接改）都要同步更新 `block_sum`，否则中间
      完整块的查询会读到过期的和。
    """

    def __init__(self, arr: list[int]) -> None:
        self.n = len(arr)
        self.arr = list(arr)
        self.block_size = max(1, int(self.n ** 0.5))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size if self.n > 0 else 0
        self.block_sum = [0] * self.num_blocks
        self.block_lazy = [0] * self.num_blocks
        for i, x in enumerate(arr):
            self.block_sum[i // self.block_size] += x

    def _block_of(self, i: int) -> int:
        return i // self.block_size

    def _block_len(self, b: int) -> int:
        start = b * self.block_size
        end = min(start + self.block_size, self.n)
        return end - start

    def _push_down(self, b: int) -> None:
        if self.block_lazy[b] != 0:
            start = b * self.block_size
            end = min(start + self.block_size, self.n)
            for i in range(start, end):
                self.arr[i] += self.block_lazy[b]
            self.block_lazy[b] = 0

    def update_range(self, l: int, r: int, val: int) -> None:
        bl, br = self._block_of(l), self._block_of(r)
        if bl == br:
            self._push_down(bl)
            for i in range(l, r + 1):
                self.arr[i] += val
            self.block_sum[bl] += val * (r - l + 1)
            return
        self._push_down(bl)
        end_of_bl = min((bl + 1) * self.block_size, self.n) - 1
        for i in range(l, end_of_bl + 1):
            self.arr[i] += val
        self.block_sum[bl] += val * (end_of_bl - l + 1)

        for b in range(bl + 1, br):
            self.block_lazy[b] += val
            self.block_sum[b] += val * self._block_len(b)

        self._push_down(br)
        start_of_br = br * self.block_size
        for i in range(start_of_br, r + 1):
            self.arr[i] += val
        self.block_sum[br] += val * (r - start_of_br + 1)

    def query_range(self, l: int, r: int) -> int:
        bl, br = self._block_of(l), self._block_of(r)
        if bl == br:
            self._push_down(bl)
            return sum(self.arr[l:r + 1])
        self._push_down(bl)
        end_of_bl = min((bl + 1) * self.block_size, self.n) - 1
        total = sum(self.arr[l:end_of_bl + 1])

        for b in range(bl + 1, br):
            total += self.block_sum[b]

        self._push_down(br)
        start_of_br = br * self.block_size
        total += sum(self.arr[start_of_br:r + 1])
        return total


class _XorTrieNode:
    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: list["_XorTrieNode | None"] = [None, None]


class XorTrie:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：01-字典树（按二进制位
    构建的字典树，而不是按字符构建），支持 `insert(num)` 插入一个非负整数，
    `query_max_xor(num) -> int` 返回"num 和已插入的所有数中某一个数"的按位异或
    的最大值（这是 LC421"数组中两个数的最大异或值"背后的核心结构）。

    【思路】20 类的 `Trie` 是按字符（多叉，26 个分支）组织的；`XorTrie` 是它的
    二进制特化版本——每个数看成固定位数（这里取 31 位，覆盖非负 32 位整数）的
    二进制串，从最高位到最低位，每一位是 0 还是 1 决定走左孩子还是右孩子，`insert`
    就是把这条 31 步的路径在字典树里建出来（复用已有前缀）。`query_max_xor` 的
    核心贪心：要让异或结果最大，应该从最高位开始，**每一位都尽量选择和 num 当前
    这一位相反的分支**——因为异或运算里，两个不同的位异或才能得到 1，二进制数的
    高位对最终大小的影响远大于低位之和，所以"贪心地在每一位都尽量选相反"、且
    "只要相反分支存在就一定选它"，是保证整体异或值最大的正确策略（不需要回溯，
    因为高位的收益永远压倒低位，不存在"这一位牺牲一点、换低位更大收益"的情况）；
    如果字典树里恰好没有相反的分支（说明所有已插入的数在这一位上都和 num 相同），
    只能被迫走相同的分支（这一位对异或值贡献 0）。

    【复杂度】`insert`/`query_max_xor` 均为 O(位数)（这里是 31，视为常数）；
    空间 O(已插入数字个数 × 位数)（公共前缀的路径会被共享）。

    【易错点】
    - 位数（`BITS`）必须固定并且对所有数字一致（包括查询时用的数字）——如果
      某些数字按更少的位数处理（比如没有对齐到同样的位宽），高位对齐会出错，
      贪心比较的就不是"同一位"了。
    - 每一位判断"相反分支是否存在"时,要先看 `node.children[want]`（want = 1
      - bit）**是否为 None**,再决定往哪个方向走——如果先无条件走某个方向,再
      判断,容易在两个分支都不存在时抛异常(字典树非空时至少有一个分支存在,
      因为已经插入过至少一个数)。
    - `query_max_xor` 只能求"num 和字典树中某个数"的最大异或,如果字典树里
      还没有插入任何数字,不能直接调用它(根节点没有任何子节点,第一步判断
      `want` 分支是否存在就会失败,取到 None 而不是有效节点)。
    """

    BITS = 31

    def __init__(self) -> None:
        self.root = _XorTrieNode()

    def insert(self, num: int) -> None:
        node = self.root
        for i in range(self.BITS - 1, -1, -1):
            bit = (num >> i) & 1
            if node.children[bit] is None:
                node.children[bit] = _XorTrieNode()
            node = node.children[bit]

    def query_max_xor(self, num: int) -> int:
        node = self.root
        result = 0
        for i in range(self.BITS - 1, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit
            if node.children[want] is not None:
                result |= (1 << i)
                node = node.children[want]
            else:
                node = node.children[bit]
        return result


class SimpleBloomFilter:
    """
    【题意】自实现数据结构（不对应具体 LeetCode 编号）：布隆过滤器
    （Bloom Filter）的简化版，支持 `add(item)`、`might_contain(item) -> bool`
    ——一种"用固定大小的空间，判断一个元素'绝对不在'还是'可能在'集合里"的概率
    数据结构：保证**绝不漏判**（元素真的被 add 过，`might_contain` 一定返回
    True），但可能**误判**（元素从未被 add，`might_contain` 仍有小概率返回
    True，称为"假阳性"）。

    【思路】用一个长度为 m 的位数组和 k 个相互独立的哈希函数：`add(item)` 用
    k 个哈希函数分别算出 k 个下标，把这些下标位置全部置为 True；
    `might_contain(item)` 同样算出这 k 个下标，只要**有任意一个**位置是
    False，就能断定这个元素一定没被 add 过（因为如果它被 add 过，这 k 个位置
    早就全部置位了）——这是"绝不漏判"的来源；但如果这 k 个位置**恰好**都被
    其他元素的哈希碰撞置为 True，`might_contain` 就会误判这个从未出现过的元素
    "可能存在"——这是"假阳性"的来源，且位数组越小、插入的元素越多、k 越不
    合适，假阳性率越高。本实现用 k 个**固定的线性同余式**
    `(item * a_i + b_i) % m`（`a_i`、`b_i` 是几个不同的质数）作为哈希函数族，
    完全不依赖 Python 内置的 `hash()`（它对字符串启用了随机化种子，同一段代码
    在不同进程里跑出的哈希值不一致）和 `random` 模块，保证同一个输入序列在任意
    时候重跑都得到完全相同的位数组状态，便于测试和复现。

    【复杂度】`add`/`might_contain` 均为 O(k)（k 是哈希函数个数，视为常数）；
    空间 O(m)（位数组大小，和实际存了多少元素无关——这正是布隆过滤器相对于
    直接用 `set()` 的价值：用固定的、可控的空间换取"绝不漏判、小概率误判"的
    近似成员判断）。

    【易错点】
    - 布隆过滤器**不支持删除**——如果需要删除元素，天真地把对应的 k 个位清零
      会误伤其他哈希碰撞到同一批位置的元素（导致它们从"可能存在"变成"绝对
      不存在"这个更强的错误结论,比假阳性更糟糕:假阳性只是错误地判断'可能
      存在',而错误清零会导致真正存在的元素被判断为'绝对不存在',违反了'绝不
      漏判'这个核心保证)。这里没有实现删除,如果要支持删除需要换成"计数布隆
      过滤器"（每个位置存计数而不是布尔值）。
    - k 个哈希函数必须相互"足够独立"（这里用不同的质数系数），如果 k 个哈希
      函数实质上是同一个函数的简单变形（比如系数都取 1 的倍数关系），会导致
      k 个哈希值高度相关，起不到"多次独立检验降低误判率"的效果，等价于只有
      1 个哈希函数。
    - 位数组大小 m 和元素个数、k 的选取直接决定假阳性率——m 太小、插入元素
      太多，`might_contain` 对任意输入几乎总是返回 True，退化成没有辨别力的
      "全部放行"，这是布隆过滤器在容量规划上最容易踩的坑（不是本实现的 bug，
      而是使用这个数据结构时必须提前预估容量的原因）。
    """

    def __init__(self, size: int = 1024, num_hashes: int = 4) -> None:
        self.size = size
        self.bits = [False] * size
        primes = [1000003, 1500007, 2000003, 2500009, 3000017, 3500009, 4000037, 4500007]
        self._coeffs = [
            (primes[i % len(primes)], primes[(i + 3) % len(primes)]) for i in range(num_hashes)
        ]

    def _hash(self, item: int, a: int, b: int) -> int:
        return (item * a + b) % self.size

    def add(self, item: int) -> None:
        for a, b in self._coeffs:
            self.bits[self._hash(item, a, b)] = True

    def might_contain(self, item: int) -> bool:
        return all(self.bits[self._hash(item, a, b)] for a, b in self._coeffs)


def _self_test() -> None:
    # ---- LC981 基于时间的键值存储（两个官方示例） ----
    tm = TimeMap()
    tm.set("foo", "bar", 1)
    assert tm.get("foo", 1) == "bar"
    assert tm.get("foo", 3) == "bar"
    tm.set("foo", "bar2", 4)
    assert tm.get("foo", 4) == "bar2"
    assert tm.get("foo", 5) == "bar2"

    tm2 = TimeMap()
    tm2.set("love", "high", 10)
    tm2.set("love", "low", 20)
    assert tm2.get("love", 5) == ""
    assert tm2.get("love", 10) == "high"
    assert tm2.get("love", 15) == "high"
    assert tm2.get("love", 20) == "low"
    assert tm2.get("love", 25) == "low"

    # ---- 跳表确定性变体（对应 LC1206 官方示例，custom 版本） ----
    sl = SimpleSkipList()
    for v in [1, 2, 3]:
        sl.insert(v)
    assert sl.search(0) is False
    sl.insert(4)
    assert sl.search(1) is True
    assert sl.erase(0) is False
    assert sl.erase(1) is True
    assert sl.search(1) is False

    # ---- SparseTable：与暴力 O(n) 重新扫描区间最小值交叉验证 ----
    arr = [5, 2, 6, 1, 9, 3, 8, 4, 7, 0]
    st = SparseTable(arr)
    for l in range(len(arr)):
        for r in range(l, len(arr)):
            assert st.query_min(l, r) == min(arr[l:r + 1]), (l, r)

    # ---- UndoableUnionFind：union 后 undo 能回到之前的连通状态 ----
    uuf = UndoableUnionFind(6)
    assert uuf.union(0, 1) is True
    assert uuf.union(2, 3) is True
    assert uuf.connected(0, 1) is True
    assert uuf.connected(2, 3) is True
    assert uuf.connected(1, 3) is False
    assert uuf.union(1, 2) is True  # 合并 {0,1} 和 {2,3}
    assert uuf.connected(0, 3) is True
    uuf.undo()  # 撤销 union(1, 2)
    assert uuf.connected(0, 3) is False
    assert uuf.connected(0, 1) is True
    assert uuf.connected(2, 3) is True
    uuf.undo()  # 撤销 union(2, 3)
    assert uuf.connected(2, 3) is False
    assert uuf.connected(0, 1) is True
    assert uuf.union(4, 5) is True
    assert uuf.connected(4, 5) is True
    uuf.undo()  # 撤销 union(4, 5)
    assert uuf.connected(4, 5) is False
    # 对同一对已连通的元素重复 union 也要能被 undo 正确处理（空操作）
    assert uuf.union(0, 1) is False
    uuf.undo()
    assert uuf.connected(0, 1) is True  # 之前的连通状态不受影响

    # ---- MonotonicQueue：手写双端队列实现滑动窗口最大值，和暴力交叉验证 ----
    def sliding_window_max(nums: list[int], k: int) -> list[int]:
        mq = MonotonicQueue()
        result = []
        for i, x in enumerate(nums):
            mq.push_back(x, i)
            while mq.front_index() <= i - k:
                mq.pop_front()
            if i >= k - 1:
                result.append(mq.get_max())
        return result

    nums1 = [1, 3, -1, -3, 5, 3, 6, 7]
    assert sliding_window_max(nums1, 3) == [3, 3, 5, 5, 6, 7]

    nums2 = [9, 11, -1, 2, 7, 4, 3, 8, -5, 6]
    for k in (1, 2, 3, 4, 5):
        expected = [max(nums2[i:i + k]) for i in range(len(nums2) - k + 1)]
        assert sliding_window_max(nums2, k) == expected, k

    # ---- SqrtDecomposition：区间加 + 区间和，与暴力数组模拟交叉验证 ----
    base = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2]  # 17 个元素，非完全平方数长度
    brute = list(base)
    sq = SqrtDecomposition(base)
    range_ops = [(0, 16, 2), (3, 10, -5), (1, 1, 100), (5, 12, 7), (0, 8, -3), (16, 16, 50)]
    check_ranges = [(0, 16), (3, 10), (1, 1), (5, 12), (0, 8), (2, 14), (16, 16)]
    for l, r, val in range_ops:
        for i in range(l, r + 1):
            brute[i] += val
        sq.update_range(l, r, val)
        for ql, qr in check_ranges:
            assert sq.query_range(ql, qr) == sum(brute[ql:qr + 1]), (l, r, val, ql, qr)

    # ---- XorTrie：与暴力 O(n^2) 枚举所有数对交叉验证，并对照 LC421 官方样例 ----
    nums3 = [3, 10, 5, 25, 2, 8]
    trie = XorTrie()
    for x in nums3:
        trie.insert(x)
    trie_best = max(trie.query_max_xor(x) for x in nums3)
    brute_best = max(x ^ y for i, x in enumerate(nums3) for j, y in enumerate(nums3) if i != j)
    assert trie_best == brute_best == 28  # LC421 官方样例：5 ^ 25 = 28

    # ---- SimpleBloomFilter：确定性哈希，无假阴性 + 有辨别力（假阳性率不失控） ----
    bf = SimpleBloomFilter(size=1024, num_hashes=4)
    added = [10, 200, 3000, 45, 999, 123456]
    for x in added:
        bf.add(x)
    for x in added:
        assert bf.might_contain(x) is True  # 绝不漏判：已 add 的元素永远返回 True

    never_added = list(range(10_000, 10_200))
    false_positive_count = sum(1 for x in never_added if bf.might_contain(x))
    assert false_positive_count < len(never_added) * 0.2, (
        f"假阳性率过高：{false_positive_count}/{len(never_added)}，说明参数或哈希函数选择有问题"
    )

    print(
        "[PASS] p33_advanced_data_structures: "
        "981(TimeMap) + SimpleSkipList(custom,对应LC1206确定性变体)/SparseTable/"
        "UndoableUnionFind/MonotonicQueue/SqrtDecomposition/XorTrie/SimpleBloomFilter 全部正确"
    )


if __name__ == "__main__":
    _self_test()
