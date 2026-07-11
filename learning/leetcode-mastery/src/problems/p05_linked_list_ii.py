"""LeetCode 分类 05·链表 进阶补充（Part II）：分段反转（K个一组/区间反转）+
深拷贝随机链表 + 找环入口/找交点 + 归并排序在链表上的应用，共 8 道范例。"""
from __future__ import annotations


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_list(vals: list[int]) -> ListNode | None:
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def list_to_vals(node: ListNode | None) -> list[int]:
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out


def build_cyclic_list(vals: list[int], pos: int) -> ListNode | None:
    """辅助构造函数（非 LeetCode 原题，同 Phase 1 p05_linked_list.py 的写法）：按 vals
    建立单链表，再把尾节点的 next 指向下标为 pos 的节点构造出带环链表；pos=-1 表示不
    成环。带环链表无法用 list_to_vals 打印（会死循环），只能手工搭建后直接传给待测函数。
    """
    if not vals:
        return None
    nodes = [ListNode(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]
    return nodes[0]


class RandomListNode:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def build_random_list(pairs: list[list[int | None]]) -> RandomListNode | None:
    """辅助构造函数：pairs[i] = [val, random_index]，random_index 为 None 表示这个
    节点的 random 指针不指向任何节点。先建好所有节点、串好 next，再统一按下标补 random，
    这样 random 可以指向"后面还没建到"的节点，不用担心构造顺序。
    """
    if not pairs:
        return None
    nodes = [RandomListNode(val) for val, _ in pairs]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    for i, (_, ridx) in enumerate(pairs):
        if ridx is not None:
            nodes[i].random = nodes[ridx]
    return nodes[0]


def random_list_to_pairs(head: RandomListNode | None) -> list[list[int | None]]:
    """辅助校验函数：把 RandomListNode 链表还原成 [val, random_val] 的列表，方便断言
    克隆结果的 val 序列和 random 指向的 val 是否与原链表一致（random_val 用"指向节点
    的 val"而不是下标比较，因为克隆出来的是全新对象，下标在克隆链表里没有意义）。
    """
    out: list[list[int | None]] = []
    node = head
    while node is not None:
        out.append([node.val, node.random.val if node.random is not None else None])
        node = node.next
    return out


def reverse_k_group(head: ListNode | None, k: int) -> ListNode | None:
    """
    【题意】给定链表头节点 head 和整数 k，每 k 个节点为一组进行反转；如果剩下的节点数量
    不足 k 个，这一组保持原有顺序不变。返回处理后的头节点。
    【思路】先探测"从 head 开始数 k 个节点，够不够数"：用一个指针 node 往前走 k 步，
    如果中途提前碰到 None（说明总共不够 k 个），直接返回 head 本身，这一组（以及后面
    所有不再够 k 个的节点）保持原样，不做任何反转——这一步"先探测再决定要不要反转"是
    本题区别于普通反转链表的关键，必须先确认凑够了一整组，才能安全地开始反转，否则会把
    "不足 k 个也强行反转"这种不符合题意的情况错误地处理掉。探测成功后，node 恰好停在
    "下一组的第一个节点"上；递归调用 reverse_k_group(node, k) 处理从这里开始的剩余
    部分，得到"剩余部分处理完之后的新头节点" prev；然后把当前这一组的 k 个节点按普通
    链表反转的方式原地反转，反转时的初始"上一个节点"不是 None，而是刚才递归求出的
    prev——这样反转完当前组后，当前组最后一个节点（其实是反转前的第一个节点 head）
    的 next 自然就正确地接上了递归处理好的剩余部分，不需要额外再写一次"拼接"逻辑。
    【复杂度】时间 O(n)，每个节点被访问常数次；空间 O(n/k)，即递归深度（可以改写成
    迭代版本把空间降到 O(1)，这里为了突出"探测+反转+递归衔接"的结构用递归实现）。
    【易错点】
    1) 探测阶段必须先数完整整 k 个节点、确认足够之后才能开始反转，如果一边探测一边
    反转，一旦发现不够 k 个再想"撤销已经做的反转"会非常麻烦；
    2) 反转当前组时，起始的 prev 不是 None，而是"递归处理完剩余链表后返回的新头
    节点"——这是本题和普通反转链表最大的区别，如果 prev 从 None 开始，当前组反转完后
    会和后面的链表断开；
    3) 递归调用必须传入探测阶段停下来的那个节点 node（下一组的第一个节点），而不是
    简单地传 head.next 之类的固定偏移，否则组的边界会算错。
    """
    node = head
    count = 0
    while node is not None and count < k:
        node = node.next
        count += 1
    if count < k:
        return head

    prev = reverse_k_group(node, k)
    cur = head
    for _ in range(k):
        next_tmp = cur.next
        cur.next = prev
        prev = cur
        cur = next_tmp
    return prev


def reverse_between(head: ListNode | None, left: int, right: int) -> ListNode | None:
    """
    【题意】给定链表头节点 head 和两个位置 left、right（从 1 开始计数），只反转
    [left, right] 这一段区间内的节点，区间之外的节点保持原有顺序和位置不变。
    【思路】用哨兵节点 dummy 接在 head 前面，先把 prev 指针移动到 left 的前一个位置
    （统一处理"left=1 时要反转的区间恰好包含原头节点"这种情况，不需要单独特判）。之后
    用"头插法"原地反转区间内的节点：cur 固定指向区间的第一个节点不动，每一轮把 cur
    后面紧跟的那个节点（next_tmp）摘下来，直接插到 prev 的后面（也就是当前已反转部分
    的最前端），重复 right-left 次之后，区间内的节点顺序就被整体倒转了过来，而 cur
    本身因为全程没有移动，自然就停在了反转后区间的最后一个位置，天然和区间右侧剩余的
    链表保持连接，不需要额外再处理"反转后怎么和后面拼接"。
    【复杂度】时间 O(right)，只需要遍历到 right 位置为止；空间 O(1)。
    【易错点】
    1) cur 指针在整个反转过程中必须固定不动（它是反转后区间的新尾部），每一轮真正移动
    位置的是被摘下来的 next_tmp 和 prev，如果误把 cur 也跟着往后挪，会把"头插法"变成
    普通的正向遍历，反转不出正确的顺序；
    2) prev 必须先移动到 left 的前一个位置（走 left-1 步），而不是 left 位置本身，
    否则反转区间的起点会偏移一位；
    3) 用哨兵节点是必须的，因为 left=1 时"left 的前一个位置"就是原头节点之前，如果
    不用哨兵节点，这种情况需要对"要不要更新 head 本身"单独写一次特判。
    """
    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    cur = prev.next
    for _ in range(right - left):
        next_tmp = cur.next
        cur.next = next_tmp.next
        next_tmp.next = prev.next
        prev.next = next_tmp

    return dummy.next


def copy_random_list(head: RandomListNode | None) -> RandomListNode | None:
    """
    【题意】给定一个特殊链表，每个节点除了 val 和 next 指针外，还有一个 random 指针，
    可以指向链表中的任意一个节点或者 None。要求返回这个链表的**深拷贝**（返回全新的
    节点组成的链表，不能与原链表共享任何节点），且新链表里每个节点的 random 指向关系
    要和原链表严格对应。
    【思路】难点在于 random 可能指向"链表里任何一个节点"，包括还没被扫描到的后面的
    节点——如果按顺序遍历时想一步到位同时建好 next 和 random，会因为目标克隆节点还
    不存在而卡住。解法是拆成两遍扫描：第一遍只建克隆节点、只填 val，同时用一个哈希表
    mapping 记录"原节点对象 -> 对应克隆节点对象"的映射（这一遍完全不管 next/random，
    先保证"任意原节点都能查到它对应的克隆节点"这个前提成立）；第二遍再扫一次原链表，
    这时对每个原节点 node，它的 next 和 random 指向的目标节点必然已经在第一遍里建好
    并存进了 mapping（因为第一遍已经遍历过整条链表），直接用 mapping 查出对应的克隆
    节点填进克隆节点的 next/random 即可。
    【复杂度】时间 O(n)（两遍线性扫描），空间 O(n)（哈希表 + n 个克隆节点）。
    【易错点】
    1) 不能试图一遍扫描同时搞定 next 和 random——random 可能指向后面尚未创建的节点，
    这也是为什么必须先用一遍扫描把"原节点到克隆节点"的映射关系全部建好；
    2) random（或 next）为 None 时，`mapping.get(None)` 天然返回 None（因为 None
    从未被当作 key 存入 mapping），可以直接用 dict.get 处理"指向 None"这种情况，不
    需要额外写 `if node.random is None` 的特判分支；
    3) 返回的必须是 mapping[head]（克隆头节点），而不是原来的 head——克隆头节点和
    原头节点是两个不同的对象，用 `is` 判断会得到 False，这正是深拷贝的定义；如果不小心
    返回了原 head，测试里 `clone_head is not original_head` 这类断言就会失败，说明
    根本没做深拷贝。
    """
    if head is None:
        return None

    mapping: dict[RandomListNode, RandomListNode] = {}
    node = head
    while node is not None:
        mapping[node] = RandomListNode(node.val)
        node = node.next

    node = head
    while node is not None:
        clone = mapping[node]
        clone.next = mapping.get(node.next)
        clone.random = mapping.get(node.random)
        node = node.next

    return mapping[head]


def detect_cycle(head: ListNode | None) -> ListNode | None:
    """
    【题意】给定单链表头节点 head，如果链表存在环，返回环的入口节点；如果不存在环，
    返回 None（不能通过修改链表结构来判断，也不能使用额外的哈希集合记录访问过的节点）。
    【思路】Floyd 判圈法分两个阶段。第一阶段和判断"是否有环"完全一样：慢指针每次走
    一步、快指针每次走两步，如果两者在环内相遇，说明存在环；如果 fast 提前走到链表
    末尾（None），说明无环，直接返回 None。第二阶段是本题比"判断有没有环"更进一步的
    地方——需要找到环的**入口**：数学上可以证明，把其中一个指针重新放回链表头 head，
    另一个指针留在刚才相遇的位置，两个指针此后都改成每次只走一步，它们必然会在环的
    入口节点再次相遇（这是由"起点到入口的距离"和"相遇点到入口的距离（沿环方向）"之间
    的等量关系决定的，属于需要记住的结论，也可以通过设未知数推导环长和各段距离验证）。
    【复杂度】时间 O(n)，空间 O(1)（相比"用哈希集合记录访问过的节点"的 O(n) 空间解法，
    双指针法把空间降到常数）。
    【易错点】
    1) 循环条件必须同时检查 fast 和 fast.next 是否为 None，因为 fast 每次要走两步，
    只检查 fast is not None 会在 fast.next 恰好是 None 时对 fast.next.next 取值
    而抛异常；
    2) 第二阶段重新出发的指针必须从 head 开始，而不是从相遇点重新开始——这两个指针
    分别代表"从链表起点出发"和"从相遇点出发"，缺一不可，把其中一个也设成相遇点会得到
    错误的入口位置；
    3) 第二阶段两个指针都必须改成每次只走一步（不再是一步两步的组合），如果沿用第一
    阶段的速度差，两者再也无法保证在入口相遇。
    """
    slow, fast = head, head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            ptr = head
            while ptr is not slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None


def reorder_list(head: ListNode | None) -> None:
    """
    【题意】给定单链表 head：L0 -> L1 -> ... -> Ln-1 -> Ln，原地将其重新排列成
    L0 -> Ln -> L1 -> Ln-1 -> L2 -> Ln-2 -> ...（首尾交替），不能修改节点内部的值，
    只能改变节点之间的连接顺序；函数不返回新链表，直接原地修改 head 指向的链表。
    【思路】拆成三步，每一步都是前面学过的心法的直接复用：1) 用快慢指针找到链表中点，
    把链表从中点切成前后两半；2) 把后半部分整体反转（复用"反转链表"的标准写法）；
    3) 把反转后的后半部分和前半部分逐节点交替合并——从前半部分取一个节点、从反转后的
    后半部分取一个节点，依次穿插连接，直到后半部分耗尽为止（后半部分节点数不多于前半
    部分，所以合并总是以"后半部分耗尽"结束，前半部分剩下的最后一个节点的 next 自然
    保持为 None）。
    【复杂度】时间 O(n)（找中点、反转、合并各是一趟线性扫描），空间 O(1)（只重新拼接
    已有节点的指针，不新建节点）。
    【易错点】
    1) 找中点时如果链表长度是奇数，中点应该划给前半部分（前半部分比后半部分多一个
    节点），也就是快指针要从 head.next 出发或者用"fast.next 和 fast.next.next
    都非空才继续走"这种条件来控制，写错会导致前后两半分割点偏移一位；
    2) 切割前后两半时必须显式把前半部分的尾节点 next 设为 None，否则合并阶段会因为
    前半部分残留着指向后半部分的旧指针而把链表接错；
    3) 合并阶段交替穿插时要先把"下一个要用的节点"提前存下来（类似反转链表里的
    next_tmp），因为合并过程中会不断修改 next 指针，如果不提前保存，会在赋值之后就
    丢失原来的"下一个节点"这个引用。
    """
    if head is None or head.next is None:
        return

    def reverse(node: ListNode | None) -> ListNode | None:
        prev = None
        while node is not None:
            next_tmp = node.next
            node.next = prev
            prev = node
            node = next_tmp
        return prev

    slow, fast = head, head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next

    second = reverse(slow.next)
    slow.next = None

    first = head
    while second is not None:
        first_next, second_next = first.next, second.next
        first.next = second
        second.next = first_next
        first, second = first_next, second_next


def sort_list(head: ListNode | None) -> ListNode | None:
    """
    【题意】给定单链表头节点 head，返回按升序排序后的链表，要求时间复杂度为
    O(n log n)（这排除了插入排序等 O(n^2) 做法）。
    【思路】链表天然适合用归并排序而不是快速排序——归并排序的"合并两个有序序列"这一步
    在链表上是 O(1) 额外空间就能完成的（不像数组归并需要额外数组），而链表也没有"随机
    访问"的能力，快速排序依赖的"按下标交换"在链表上做起来很别扭。具体做法：用快慢指针
    找到链表中点，从中点切成两半，分别递归排序；两个已经各自有序的子链表，用和
    "合并两个有序链表"完全相同的双指针合并逻辑拼接成一条整体有序的链表。递归的终止
    条件是链表为空或只剩一个节点（天然有序，不需要再切分）。
    【复杂度】时间 O(n log n)：链表被切分 O(log n) 层，每一层所有子问题的合并操作
    加起来是 O(n)；空间 O(log n)（递归调用栈深度，不计输入本身）。
    【易错点】
    1) 找中点切分时，快指针要从 head.next 出发（而不是 head），这样在链表长度为偶数
    时，中点会划分到后半部分的开头，保证每次切分都能让两个子链表严格变短，否则长度为
    2 的子链表可能永远切不出"两个长度为 1 的子链表"，导致递归不终止；
    2) 递归排序前必须显式把前半部分的尾节点 next 设为 None，断开与后半部分的连接，
    否则递归排序前半部分时会把后半部分也带进去，得到错误的结果；
    3) 合并两个有序子链表时依然要用哨兵节点技巧统一处理"结果链表第一个节点选自哪一半"
    的问题，和"合并两个有序链表"（LC21）是完全相同的子过程，可以直接复用同一套写法。
    """
    if head is None or head.next is None:
        return head

    slow, fast = head, head.next
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)

    dummy = ListNode()
    tail = dummy
    while left is not None and right is not None:
        if left.val <= right.val:
            tail.next, left = left, left.next
        else:
            tail.next, right = right, right.next
        tail = tail.next
    tail.next = left if left is not None else right
    return dummy.next


def get_intersection_node(
    head_a: ListNode | None, head_b: ListNode | None
) -> ListNode | None:
    """
    【题意】给定两个单链表的头节点 head_a、head_b，如果两条链表在某个节点开始"相交"
    （从那个节点起，两条链表共享同一批节点对象，而不是值相同的不同对象），返回相交的
    起始节点；如果不相交，返回 None。不允许修改链表结构，且要求空间复杂度 O(1)。
    【思路】双指针"走完自己的链表再走对方的"技巧：设两条链表各自独有的前缀长度分别为
    a、b，公共尾部长度为 c。指针 pa 从 head_a 出发，走完 a+c 步到达链表 A 末尾后，
    紧接着从 head_b 重新出发；指针 pb 从 head_b 出发，走完 b+c 步到达链表 B 末尾后，
    紧接着从 head_a 重新出发。两个指针走过的总路程都是 a+b+c 步，因此会在同一时刻
    到达同一个节点——如果两条链表确实相交，这一步恰好落在相交起点上；如果不相交，
    c=0，两个指针会同时耗尽（都变成 None）而"相遇"于 None，直接返回 None 也是正确
    答案。这个技巧不需要预先分别遍历两条链表算出长度差再对齐，一次同步遍历就够了。
    【复杂度】时间 O(m+n)，m、n 为两条链表的长度；空间 O(1)。
    【易错点】
    1) 指针走到自己链表末尾（None）后，下一步必须切换到**对方**链表的头节点，而不是
    停在 None 不动或者回到自己链表头——切换到对方头节点是让两个指针路程相等、从而
    能对齐相遇点的关键；
    2) 判断相遇必须用对象身份 `is` 比较，而不是比较 val——两条链表在"相交"之前可能
    存在值相同但对象不同的节点（巧合），只有比较对象引用才能确认是不是同一批共享节点；
    3) 两条链表完全不相交时，两个指针最终会同时变成 None，循环条件 `pa is not pb`
    在两者都是 None 时也会成立为 False（None is None 为真）从而正常退出循环并返回
    None，不需要为"不相交"这种情况单独写特判。
    """
    if head_a is None or head_b is None:
        return None

    pa, pb = head_a, head_b
    while pa is not pb:
        pa = pa.next if pa is not None else head_b
        pb = pb.next if pb is not None else head_a
    return pa


def swap_pairs(head: ListNode | None) -> ListNode | None:
    """
    【题意】给定单链表头节点 head，两两相邻节点为一组交换位置（不能只交换节点内部的
    val 值，必须真正改变节点之间的连接关系），返回交换后的头节点；节点总数为奇数时，
    最后落单的一个节点保持原位不变。
    【思路】用哨兵节点 dummy 接在 head 前面，统一处理"交换发生在链表最前端、头节点
    本身要被换下去"这种情况——不用哨兵的话，第一组交换需要单独把新的头节点提出来赋值，
    和后面组的交换逻辑不一致。用 prev 指针指向"每一组交换前，这一组前面已经处理好的
    最后一个节点"，每一轮取出紧跟在 prev 后面的两个节点 first、second，按固定的三步
    改指针：second 先接到 prev 后面 -> first 改接到 second 原来的下一个节点 ->
    second 改接到 first；交换完这一组后，prev 移动到 first（也就是交换后这一组的
    最后一个节点），继续处理下一组。循环条件同时检查 `prev.next` 和
    `prev.next.next` 是否存在，天然处理了"剩余节点数为奇数、只剩一个节点不用交换"
    和"剩余节点数为 0"两种终止情况。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】
    1) 三步改指针的顺序不能打乱：必须先把 first.next 指向 second 原来的下一个节点
    （在 second.next 被覆盖之前先读出来，或者像本实现里先做 `first.next =
    second.next`），再让 second.next 指向 first，最后 prev.next 指向 second——
    顺序错了会导致提前覆盖掉还需要用到的指针，链表断裂；
    2) 循环条件必须同时检查 `prev.next is not None` 和 `prev.next.next is not
    None`，只检查前者会在只剩一个节点时尝试访问 `second`（`prev.next.next`）而
    得到 None 后继续操作 None.next 抛异常；
    3) 交换完一组之后 prev 必须移动到 first（新排在后面的那个节点），而不是
    second——移动错节点会导致下一组交换时 prev 位置偏移，把不该交换的节点错误地
    连接在一起。
    """
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next is not None and prev.next.next is not None:
        first = prev.next
        second = first.next
        first.next = second.next
        second.next = first
        prev.next = second
        prev = first
    return dummy.next


def _self_test() -> None:
    assert list_to_vals(reverse_k_group(build_list([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
    assert list_to_vals(reverse_k_group(build_list([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]

    assert list_to_vals(reverse_between(build_list([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
    assert list_to_vals(reverse_between(build_list([5]), 1, 1)) == [5]

    original = build_random_list([[7, None], [13, 0], [11, 4], [10, 2], [1, 0]])
    clone = copy_random_list(original)
    assert clone is not original
    assert [v for v, _ in random_list_to_pairs(clone)] == [7, 13, 11, 10, 1]
    assert random_list_to_pairs(clone) == random_list_to_pairs(original)

    assert detect_cycle(build_cyclic_list([3, 2, 0, -4], 1)).val == 2
    assert detect_cycle(build_cyclic_list([1, 2], 0)).val == 1
    assert detect_cycle(build_cyclic_list([1], -1)) is None

    head4 = build_list([1, 2, 3, 4])
    reorder_list(head4)
    assert list_to_vals(head4) == [1, 4, 2, 3]
    head5 = build_list([1, 2, 3, 4, 5])
    reorder_list(head5)
    assert list_to_vals(head5) == [1, 5, 2, 4, 3]

    assert list_to_vals(sort_list(build_list([4, 2, 1, 3]))) == [1, 2, 3, 4]
    assert list_to_vals(sort_list(build_list([-1, 5, 3, 4, 0]))) == [-1, 0, 3, 4, 5]

    shared = build_list([8, 4, 5])
    head_a = build_list([4, 1])
    tail_a = head_a
    while tail_a.next is not None:
        tail_a = tail_a.next
    tail_a.next = shared
    head_b = build_list([5, 6, 1])
    tail_b = head_b
    while tail_b.next is not None:
        tail_b = tail_b.next
    tail_b.next = shared
    assert get_intersection_node(head_a, head_b) is shared

    assert list_to_vals(swap_pairs(build_list([1, 2, 3, 4]))) == [2, 1, 4, 3]
    assert swap_pairs(build_list([])) is None
    assert list_to_vals(swap_pairs(build_list([1]))) == [1]

    print(
        "[PASS] p05_linked_list_ii: 8/8 题通过 "
        "(K个一组翻转链表/反转链表II/随机链表的复制/环形链表II/"
        "重排链表/排序链表/相交链表/两两交换链表中的节点)"
    )


if __name__ == "__main__":
    _self_test()
