"""分类 21：设计题 —— Phase 3 竞赛级补充：多层链表/分桶/有序区间的组合数据结构设计。"""
from __future__ import annotations

import bisect
import random
from collections import defaultdict


class _FreqNode:
    """AllOne 内部双向链表节点：代表"计数恰好为 count"的一个桶，桶里存所有该计数的 key。"""

    __slots__ = ("count", "keys", "prev", "next")

    def __init__(self, count: int) -> None:
        self.count = count
        self.keys: set[str] = set()
        self.prev: "_FreqNode | None" = None
        self.next: "_FreqNode | None" = None


class AllOne:
    """
    【题意】设计一个字符串计数容器：inc(key) 把 key 的计数 +1（不存在则以计数 1 插入）；
    dec(key) 把 key 的计数 -1（保证调用前 key 一定存在，计数减到 0 就整体移除）；
    getMaxKey()/getMinKey() 分别返回当前计数最大/最小的任意一个 key（不存在任何 key
    时返回空字符串）。要求全部四个操作均为 O(1)。
    【思路】哈希表能 O(1) 存取"key -> 计数"，但无法 O(1) 知道"当前最大/最小计数是哪个
    key"；堆能维护最值但无法 O(1) 做"任意 key 的计数 +1/-1"（堆不支持随机位置的高效
    更新）。这里的关键 insight 是：不需要维护"计数值"本身的顺序，只需要维护"计数值
    对应的桶"之间的顺序——把所有出现过的计数值组织成一条双向链表，链表天然按计数值
    从小到大排列，头尾两个哨兵节点之外，第一个真实节点就是"当前最小计数"的桶，最后
    一个就是"当前最大计数"的桶；每个桶（_FreqNode）用一个 set 存放"计数恰好等于这个
    桶的 count"的所有 key。inc/dec 的操作本质是"把 key 从当前所在的桶，挪到 count+1
    或 count-1 对应的桶"——如果目标计数的桶还不存在，就在链表的正确位置**现场创建**
    一个新桶（用 O(1) 的双向链表插入，不需要保持"每个计数值都要有桶"这种连续性，只有
    真正用到的计数值才会有对应的桶节点，链表节点数不会超过当前 key 的种数）；如果挪
    走后原来的桶空了，就把这个空桶从链表里删除。getMaxKey/getMinKey 直接读链表尾/头
    哨兵旁边的真实节点，天然 O(1)。
    【复杂度】inc/dec/getMaxKey/getMinKey 均为 O(1)（双向链表插入删除和哈希表存取都是
    常数时间）；空间 O(不同 key 的个数)（链表节点数不会超过 key 的种数，因为每个非空
    桶至少对应一个 key）。
    【易错点】1) inc 一个新 key 时，如果链表头部已经有一个计数为 1 的桶就应该复用，
    不能每次都无脑新建，否则同一个计数值会出现两个桶，getMinKey/getMaxKey 可能只看到
    其中一个、漏掉另一个桶里的 key；2) 挪动 key 到新桶之后，必须检查原来的桶是否变空，
    变空就要从链表里摘除，否则空桶会一直占据链表头/尾的位置，导致 getMinKey/getMaxKey
    返回一个早已不存在于该计数的、空的桶（无 key 可返回）；3) dec 到计数为 0 时是
    "彻底从容器中移除这个 key"，不是"保留 key、计数记为 0"——这两者语义完全不同，
    保留计数 0 的 key 会污染 getMinKey 的语义（0 不是一个合法的桶计数，最小计数应该
    从 1 开始）。
    """

    def __init__(self) -> None:
        self.head = _FreqNode(0)
        self.tail = _FreqNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.key_node: dict[str, _FreqNode] = {}

    def _insert_after(self, anchor: _FreqNode, count: int) -> _FreqNode:
        node = _FreqNode(count)
        nxt = anchor.next
        anchor.next = node
        node.prev = anchor
        node.next = nxt
        nxt.prev = node
        return node

    def _remove(self, node: _FreqNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def inc(self, key: str) -> None:
        if key not in self.key_node:
            first = self.head.next
            if first is self.tail or first.count != 1:
                first = self._insert_after(self.head, 1)
            first.keys.add(key)
            self.key_node[key] = first
            return
        node = self.key_node[key]
        new_count = node.count + 1
        nxt = node.next
        if nxt is self.tail or nxt.count != new_count:
            nxt = self._insert_after(node, new_count)
        nxt.keys.add(key)
        node.keys.discard(key)
        self.key_node[key] = nxt
        if not node.keys:
            self._remove(node)

    def dec(self, key: str) -> None:
        node = self.key_node[key]
        new_count = node.count - 1
        node.keys.discard(key)
        if new_count == 0:
            del self.key_node[key]
        else:
            prev = node.prev
            if prev is self.head or prev.count != new_count:
                prev = self._insert_after(node.prev, new_count)
            prev.keys.add(key)
            self.key_node[key] = prev
        if not node.keys:
            self._remove(node)

    def getMaxKey(self) -> str:
        if self.tail.prev is self.head:
            return ""
        return next(iter(self.tail.prev.keys))

    def getMinKey(self) -> str:
        if self.head.next is self.tail:
            return ""
        return next(iter(self.head.next.keys))


class FreqStack:
    """
    【题意】设计一个"最大频率栈"：push(val) 正常压栈；pop() 弹出并返回当前栈中出现
    次数最多的元素，如果多个元素频率并列最高，弹出其中离栈顶最近（最后被压入）的那个。
    【思路】把"频率"和"栈序"这两个维度分别用两个结构维护：`freq` 是普通哈希表记录
    "每个值当前的出现次数"；`group` 是"频率 -> 该频率下按压入顺序排列的值列表（用
    列表模拟栈）"的映射——一个值每次被 push，它的频率 +1，同时被追加到"新频率对应
    的那个子栈"的栈顶。这样"频率并列时选最后压入的"这个条件被自动满足：因为同一个
    频率桶内的值本身就是按压入顺序排列的一个栈，桶内栈顶天然就是"这批同频率的值里
    最后被压入的"。用一个 `max_freq` 变量全局跟踪"当前最高频率是多少"，pop 时直接从
    `group[max_freq]` 这个子栈弹出栈顶——如果弹出后这个子栈空了，说明当前最高频率的
    所有值都已经被弹完，`max_freq` 要减一，回退到次高频率。
    【复杂度】push/pop 均为 O(1)（哈希表存取 + 列表首尾操作都是常数时间）；空间
    O(push 调用总次数)（最坏情况下每个值的频率都不同，需要维护和 push 次数同量级的
    分组数据）。
    【易错点】1) 弹出的值虽然“频率降低了 1”，但仍然留在容器里（除非频率降到 0 之前
    都不会被丢弃）——`freq[val] -= 1` 只是更新计数，绝不能把 val 从 `freq` 字典里
    删除，因为它可能之后又被重新 push；2) 弹出后如果 `group[max_freq]` 变空必须把
    `max_freq` 减一，忘记这一步会导致下次 pop 从一个空列表里弹出而报错；3) push 同
    一个值多次导致它出现在多个不同频率的子栈里（比如频率 1、2、3 各有一份），这是
    刻意保留的"历史痕迹"，不需要、也不能从旧频率的子栈里删除它，因为整个算法的正确性
    依赖于"每次 push 都会在新频率的子栈顶追加一份"这个不变量。
    """

    def __init__(self) -> None:
        self.freq: dict[int, int] = defaultdict(int)
        self.group: dict[int, list[int]] = defaultdict(list)
        self.max_freq = 0

    def push(self, val: int) -> None:
        f = self.freq[val] + 1
        self.freq[val] = f
        self.max_freq = max(self.max_freq, f)
        self.group[f].append(val)

    def pop(self) -> int:
        val = self.group[self.max_freq].pop()
        self.freq[val] -= 1
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        return val


class _SkipNode:
    """跳表节点：val 是节点存的数值，forward[i] 是这个节点在第 i 层的后继节点。"""

    __slots__ = ("val", "forward")

    def __init__(self, val: int, level: int) -> None:
        self.val = val
        self.forward: list["_SkipNode | None"] = [None] * level


class Skiplist:
    """
    【题意】不借助任何内置有序结构，自己设计一个 Skiplist（跳表），支持 search(target)
    判断是否存在、add(num) 插入（允许重复值）、erase(num) 删除任意一个等于 num 的节点
    并返回是否删除成功。要求 search/add/erase 期望时间复杂度都是 O(log n)。
    【思路】跳表的核心比喻是"一条排好序的链表，加装了几层直达电梯"：最底层（第 0 层）
    是一条完整的、包含所有元素的有序单链表，查找它退化成 O(n) 的线性扫描；跳表的技巧
    是在部分节点上"抛硬币"决定它要不要往上多长一层——如果硬币正面（这里用
    `random.random() < P` 模拟），这个节点就同时出现在更高一层，高层节点数量因为
    "每层都是上一层的一次伯努利筛选"而呈指数衰减，形成"层数越高、节点越稀疏、能跳得
    越远"的结构。查找时从最高层、最左侧的哨兵头节点开始，在当前层尽量往右走（不超过
    目标值），走不动了就下降一层继续走，直到最底层——因为每一层"平均能跳过的节点数"
    和"层数"成反比但和"晋升概率"成正比，可以证明这个"逐层下降"的过程期望只需要
    O(log n) 步，而不需要真的从头线性扫描。add/erase 都要先做一次这样的"逐层定位"，
    记录下每一层"待插入/删除位置前面最后一个节点"（update 数组），再据此在对应层级
    完成链表的插入或摘除。
    【复杂度】期望时间：search/add/erase 均为 O(log n)（层数是 O(log n) 的随机变量，
    每层期望只需要常数步）；空间 O(n)（每个节点期望额外占用 O(1/(1-P)) 层的指针，
    总体仍是线性）。
    【易错点】1) "多层链表 + 随机层数"是跳表能达到期望 O(log n) 的根本原因——如果
    偷懒让每个节点都只有 1 层（退化成普通单链表），所有操作都会退化成 O(n)；反过来
    如果不设上限，晋升概率恰好一直为真的极端情况下层数可能失控，因此需要一个
    `MAX_LEVEL` 上限截断；2) erase 时如果只删除了某一层的指针、忘记同步删除其余层
    的指针，会让跳表内部出现"半个节点"，后续 search 可能因为某一层还残留着指向已删
    节点的指针而出错；3) erase 找到的目标节点摘除后，如果整个跳表当前的最高有效层数
    因此变成"最高层已经没有任何真实节点"，应当把 `self.level` 相应下调，否则每次
    查找都会白白从一个已经空掉的高层开始扫描，虽不影响正确性但会拖慢期望复杂度。
    """

    MAX_LEVEL = 16
    P = 0.5

    def __init__(self) -> None:
        self.head = _SkipNode(-1, self.MAX_LEVEL)
        self.level = 1

    def _random_level(self) -> int:
        lvl = 1
        while random.random() < self.P and lvl < self.MAX_LEVEL:
            lvl += 1
        return lvl

    def _find_predecessors(self, target: int) -> list[_SkipNode]:
        update = [self.head] * self.MAX_LEVEL
        node = self.head
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].val < target:
                node = node.forward[i]
            update[i] = node
        return update

    def search(self, target: int) -> bool:
        update = self._find_predecessors(target)
        candidate = update[0].forward[0]
        return candidate is not None and candidate.val == target

    def add(self, num: int) -> None:
        update = self._find_predecessors(num)
        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level, lvl):
                update[i] = self.head
            self.level = lvl
        new_node = _SkipNode(num, lvl)
        for i in range(lvl):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def erase(self, num: int) -> bool:
        update = self._find_predecessors(num)
        candidate = update[0].forward[0]
        if candidate is None or candidate.val != num:
            return False
        for i in range(self.level):
            if update[i].forward[i] is not candidate:
                break
            update[i].forward[i] = candidate.forward[i]
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        return True


class MyCalendar:
    """
    【题意】设计一个日程表：book(start, end) 尝试添加一个半开区间 [start, end) 表示的
    事件，如果它和已有的任何事件存在非空交集（双重预订）就拒绝添加并返回 False；否则
    成功添加并返回 True。
    【思路】维护一个按起点排序的、内部互不重叠的区间列表 `bookings`。每次 book 新区间
    时，用二分（`bisect_right`）在 `bookings` 里定位"第一个起点 > start 的位置" i——
    因为列表本身有序，新区间只可能和"紧邻这个插入点前面的那个区间"（结束点是否晚于
    start）或"紧邻插入点后面的那个区间"（起点是否早于 end）发生重叠，不需要跟其余
    所有区间比较。只要这两个相邻检查都通过（不重叠），就可以用 `list.insert` 把新
    区间插入到位置 i，继续保持整体有序。
    【复杂度】每次 book: O(log n) 二分定位 + O(n) 的插入搬移（Python list.insert 在
    中间插入是线性的）；如果追求插入也是 O(log n) 需要换用平衡树/线段树，但对本题
    数据规模，"二分查找重叠 + 线性插入"已经足够高效且实现simple；空间 O(n)。
    【易错点】1) 区间是半开区间 [start, end)——`book(10,20)` 之后 `book(20,30)` 不
    冲突（20 是前一个区间的开区间端点，不算被占用），比较时要写成"现有区间结束
    > 新区间开始"而不是">="，写错一个等号就会把"恰好相邻、不重叠"的合法情况误判为
    冲突；2) 只检查"插入点前一个"和"插入点后一个"这两个相邻区间是不够的前提是
    `bookings` 本身必须严格保持有序且互不重叠——如果之前的插入逻辑有误导致列表失序，
    这个"只看相邻两个"的优化就会漏检；3) 二分时比较的 key 要用 `(start, ...)` 元组
    或明确只按起点比较，不能让 Python 的默认元组比较在起点相同时又去比较第二个字段
    导致意外的行为。
    """

    def __init__(self) -> None:
        self.bookings: list[tuple[int, int]] = []

    def book(self, start: int, end: int) -> bool:
        i = bisect.bisect_right(self.bookings, (start, float("inf")))
        if i > 0 and self.bookings[i - 1][1] > start:
            return False
        if i < len(self.bookings) and self.bookings[i][0] < end:
            return False
        self.bookings.insert(i, (start, end))
        return True


def _self_test() -> None:
    ao = AllOne()
    ao.inc("hello")
    ao.inc("hello")
    assert ao.getMaxKey() == "hello"
    assert ao.getMinKey() == "hello"
    ao.inc("leet")
    assert ao.getMaxKey() == "hello"
    assert ao.getMinKey() == "leet"

    fs = FreqStack()
    for v in [5, 7, 5, 7, 4, 5]:
        fs.push(v)
    assert [fs.pop(), fs.pop(), fs.pop(), fs.pop()] == [5, 7, 5, 4]

    sl = Skiplist()
    sl.add(1)
    sl.add(2)
    sl.add(3)
    assert sl.search(0) is False
    sl.add(4)
    assert sl.search(1) is True
    assert sl.erase(0) is False
    assert sl.erase(1) is True
    assert sl.search(1) is False

    cal = MyCalendar()
    assert cal.book(10, 20) is True
    assert cal.book(15, 25) is False
    assert cal.book(20, 30) is True

    print(
        "[PASS] p21_design_iii: 4/4 题通过 "
        "(全O(1)的数据结构/最大频率栈/设计跳表/我的日程安排表I；"
        "Range模块改归入 27-segment-tree-bit 类，避免和该类重复)"
    )


if __name__ == "__main__":
    _self_test()
