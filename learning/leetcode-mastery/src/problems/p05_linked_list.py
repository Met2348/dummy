"""LeetCode 分类 05·链表：哨兵节点消解头节点特判 + 快慢指针找中点/找环/找倒数第N个。"""
from __future__ import annotations

import heapq


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
    """
    【题意】辅助构造函数（非 LeetCode 原题）：按 vals 建立单链表，再把最后一个节点的
    next 指向下标为 pos 的节点，构造出"带环"的链表；pos=-1 表示不成环。用于给 LC141
    准备无法用普通 build_list 表达的测试输入（带环链表打印会死循环，只能手工搭建）。
    【思路】先把每个值包装成节点并存进一个列表 nodes（保留引用，不是只存值），依次
    把 nodes[i].next 指向 nodes[i+1] 串成普通链表，最后如果 pos != -1，把尾节点
    nodes[-1].next 指向 nodes[pos]，形成环。
    【复杂度】时间 O(n)，空间 O(n)（记录 n 个节点引用）。
    【易错点】pos=-1 时绝不能建环，必须保持尾节点 next 为 None（默认值本身就是 None，
    不写 else 分支反而更不容易出错）。
    """
    if not vals:
        return None
    nodes = [ListNode(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]
    return nodes[0]


def reverse_list(head: ListNode | None) -> ListNode | None:
    """
    【题意】给定单链表头节点 head，返回反转后的新头节点（原地反转，不新建节点）。
    【思路】维护两个指针 prev（已反转部分的新头，初始为 None）和 cur（还没处理的节点，
    初始为 head）。每一轮循环要做的三件事，顺序不能乱：
    1) 先用 next_tmp 把 cur.next 存下来——因为下一步要修改 cur.next，如果不先存，
       原来的"下一个节点"就永久丢失了，链表从这里断掉；
    2) 把 cur.next 改指向 prev，完成"这一个节点"的反转；
    3) prev 和 cur 都各自向前移动一位（prev = cur, cur = next_tmp），去处理下一个节点。
    循环结束（cur 变成 None）时，prev 正好停在原来的最后一个节点上，也就是新链表的头。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 忘记先把 cur.next 存到 next_tmp，直接写 cur.next = prev，会导致后面
    的链表片段找不到，反转出一个只有一两个节点的短链表；2) 空链表（head=None）时循环
    直接不执行，prev 保持 None 直接返回，这正是期望结果，不需要额外的 if 特判——但要
    能想清楚"为什么不需要"，而不是心虚地加一层判断。
    """
    prev = None
    cur = head
    while cur is not None:
        next_tmp = cur.next
        cur.next = prev
        prev = cur
        cur = next_tmp
    return prev


def merge_two_lists(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    """
    【题意】给定两个本身已经升序排列的单链表 l1、l2，合并成一个新的升序链表并返回头节点
    （可以直接复用原节点，不需要新建 ListNode 存值）。
    【思路】用一个哨兵节点 dummy 当"假头"，消掉"新链表第一个节点该接谁"这个特殊情况：
    不管最终第一个节点来自 l1 还是 l2，统一都先挂在 dummy.next 之后，最后返回
    dummy.next 即可，不需要为"谁当第一个节点"单独写一次判断逻辑。用一个 tail 指针
    始终指向"结果链表当前的最后一个节点"，每轮比较 l1、l2 当前节点的值，谁小就把谁接到
    tail 后面并让对应链表指针前进一步；某一条链表先耗尽后，把另一条剩余部分整体接上去
    （因为剩余部分本身已经有序，不需要再逐个比较）。
    【复杂度】时间 O(m+n)，空间 O(1)（不新建节点，只是重新拼接已有节点的指针）。
    【易错点】1) 不用哨兵节点的话，"第一个节点选 l1 还是 l2"需要在主循环之外单独判断，
    容易和循环体内的逻辑写重复；2) 主循环因为某条链表耗尽而结束后，忘记把非空的那条
    剩余部分接到 tail.next 后面，导致结果链表少了一段尾巴。
    """
    dummy = ListNode()
    tail = dummy
    while l1 is not None and l2 is not None:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 if l1 is not None else l2
    return dummy.next


def has_cycle(head: ListNode | None) -> bool:
    """
    【题意】判断单链表是否存在环：某个节点的 next 最终指回了链表中前面出现过的某个节点。
    【思路】Floyd 判圈法（"龟兔赛跑"）：慢指针 slow 每次走一步，快指针 fast 每次走两步。
    如果链表无环，fast 会先于 slow 到达链表末尾（None）；如果链表有环，fast 迟早会
    进入环内绕圈，而 slow 与 fast 的速度差恒为 1，在环内每走一步这个差距就缩小 1
    （模环长意义下），所以 fast 必然会在某一步追上 slow，两者相遇即说明存在环。这个
    方法不需要额外的哈希集合去记录"访问过的节点"，把空间复杂度从 O(n) 降到 O(1)。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 循环条件必须同时检查 fast 和 fast.next 是否为 None（fast 每次要走
    两步，如果只检查 fast is not None，会在 fast.next 恰好是 None 时对
    fast.next.next 取值而抛异常）；2) 初始时 slow 和 fast 都等于 head，不能在进入
    循环之前就检查 slow is fast（那样对任何输入都会立即判定为真），必须先各走一步
    再判断，也就是把相遇判断放在循环体内部、移动指针之后。
    """
    slow, fast = head, head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """
    【题意】给定链表头节点 head 和整数 n，删除链表的倒数第 n 个节点，返回删除后的头节点。
    要求只遍历一趟（不能先数出链表长度、换算成正数第几个节点后再遍历第二趟——虽然那样
    也能做对，但本题练的是"一趟"这个更节省的技巧）。
    【思路】用一个哨兵节点 dummy 接在 head 前面，这样即使要删除的恰好是原头节点
    （n 等于链表长度），也统一走"删除某节点的下一个节点"这一种情况，不用单独判断
    "删的是不是头节点"。具体做法：先让 fast 指针从 dummy 出发向前走 n+1 步；这样
    fast 和"最终要删除节点的前一个节点"之间，永远隔着 n 个节点的距离。之后 slow
    （从 dummy 出发）和 fast 同时逐步向后移动，直到 fast 走到链表末尾（None）——
    此时 slow 正好停在待删除节点的前一个节点上，执行 slow.next = slow.next.next
    即可把待删除节点从链表中摘除。
    【复杂度】时间 O(L)，L 为链表长度（只遍历一趟）；空间 O(1)。
    【易错点】1) fast 要多走一步（走 n+1 步而不是 n 步）：如果只走 n 步，slow 最终会
    停在待删除节点本身而不是它的前一个节点，没法直接摘除；2) 不用哨兵节点的话，"n 恰好
    等于链表长度、要删除的是原头节点"这种情况需要单独处理返回 head.next，容易被漏掉。
    """
    dummy = ListNode(0, head)
    fast = dummy
    for _ in range(n + 1):
        fast = fast.next
    slow = dummy
    while fast is not None:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next


def add_two_numbers(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    """
    【题意】两个非负整数分别以链表**逆序**存储每一位数字（例如整数 342 存成
    2 -> 4 -> 3，个位在链表头），返回这两个数相加之和，同样以逆序链表返回。
    【思路】正因为是逆序存储（个位在链表头、高位在链表尾），从头开始逐位相加，正好
    就是从个位到高位相加，和小学竖式加法的进位方式完全一致：每一位把 l1 当前位数字
    + l2 当前位数字 + 上一位带来的进位 carry 相加，结果对 10 取余就是这一位应该
    记录的数字，结果对 10 取整（0 或 1）就是要带到下一位的新进位；某条链表已经走完
    时按 0 处理（相当于这一位没有数字，只有另一条链表的数字和进位在参与）。
    【复杂度】时间 O(max(m,n))，空间 O(max(m,n))（新建结果链表，不含输入本身）。
    【易错点】1) 两条链表长度不同时，短的那条走完之后循环不能停止，要继续把它当作 0
    参与运算，直到两条链表都走完**并且**进位 carry 也清零才能结束（循环条件必须包含
    or carry，只判断两条链表是否为空会漏掉最后一次进位）；2) 最高位相加后如果还剩
    进位 1（例如 999+1=1000，最后要多出一位"1"），必须再额外补一个值为 1 的节点，
    这一步很容易被遗漏导致结果少一位。
    """
    dummy = ListNode()
    tail = dummy
    carry = 0
    while l1 is not None or l2 is not None or carry:
        x = l1.val if l1 is not None else 0
        y = l2.val if l2 is not None else 0
        total = x + y + carry
        carry, digit = divmod(total, 10)
        tail.next = ListNode(digit)
        tail = tail.next
        l1 = l1.next if l1 is not None else None
        l2 = l2.next if l2 is not None else None
    return dummy.next


def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
    """
    【题意】给定 k 个各自升序排列的链表组成的列表 lists，合并成一个升序链表并返回头节点。
    【思路】如果两两合并（先合并第 1、2 条得到一条更长的链表，再和第 3 条合并……），
    每一次合并的代价正比于"当前已经合并出来的链表长度"，最坏情况下把全部 k 条链表
    合并完的总代价是 O(n·k)（n 是所有节点总数）——这是因为越到后面的合并，一边的
    链表越长，但每次只吃掉一条新链表，没有充分利用"其余 k-2 条链表也早就排好序"这一点。
    更好的做法：维护一个大小最多为 k 的最小堆，堆里始终放着"每条尚未耗尽的链表里，
    当前最靠前（未被消费）的那个节点"。每次弹出堆顶（全局当前最小值）接到结果链表尾部，
    再把它所在链表的下一个节点压回堆——堆的大小自始至终不超过 k，每个节点入堆、出堆
    各恰好一次，单次堆操作是 O(log k)，所以总代价是 O(n log k)，比两两合并快一个
    log 因子。因为 ListNode 对象之间没有定义大小比较（直接放 (val, node) 进堆，
    一旦 val 相同就会尝试比较 node 本身，而 ListNode 没有实现 __lt__，会抛
    TypeError），所以堆里存的是 (val, 唯一序号, node) 三元组：先比较 val，
    val 相同时比较序号（序号全局递增、两两不同，保证永远不会退化到比较 node）。
    【复杂度】时间 O(n log k)，n 为节点总数，k 为链表条数；空间 O(k)（堆的大小）。
    【易错点】1) 直接把 (val, node) 放进堆而不带唯一序号，一旦出现相同的 val 就会
    在比较时尝试比较 node 对象本身而报错；2) lists 里可能包含 None（空链表），
    初始化堆时要跳过它们，且 lists 本身也可能是空列表或者全是 None，这两种情况都要
    能正确返回 None，而不是在堆为空时尝试弹出报错。
    """
    heap: list[tuple[int, int, ListNode]] = []
    counter = 0
    for node in lists:
        if node is not None:
            heapq.heappush(heap, (node.val, counter, node))
            counter += 1

    dummy = ListNode()
    tail = dummy
    while heap:
        _, _, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1
    return dummy.next


def _self_test() -> None:
    s = [1, 2, 3, 4, 5]
    assert list_to_vals(reverse_list(build_list(s))) == [5, 4, 3, 2, 1]
    assert list_to_vals(reverse_list(build_list([1, 2]))) == [2, 1]
    assert reverse_list(build_list([])) is None

    merged = merge_two_lists(build_list([1, 2, 4]), build_list([1, 3, 4]))
    assert list_to_vals(merged) == [1, 1, 2, 3, 4, 4]

    assert has_cycle(build_cyclic_list([3, 2, 0, -4], 1)) is True
    assert has_cycle(build_cyclic_list([1, 2], 0)) is True
    assert has_cycle(build_cyclic_list([1], -1)) is False

    assert list_to_vals(remove_nth_from_end(build_list([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
    assert remove_nth_from_end(build_list([1]), 1) is None
    assert list_to_vals(remove_nth_from_end(build_list([1, 2]), 1)) == [1]

    assert list_to_vals(add_two_numbers(build_list([2, 4, 3]), build_list([5, 6, 4]))) == [7, 0, 8]
    assert list_to_vals(add_two_numbers(build_list([0]), build_list([0]))) == [0]
    assert list_to_vals(
        add_two_numbers(build_list([9, 9, 9, 9, 9, 9, 9]), build_list([9, 9, 9, 9]))
    ) == [8, 9, 9, 9, 0, 0, 0, 1]

    merged_k = merge_k_lists(
        [build_list([1, 4, 5]), build_list([1, 3, 4]), build_list([2, 6])]
    )
    assert list_to_vals(merged_k) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert merge_k_lists([]) is None
    assert merge_k_lists([None]) is None

    print(
        "[PASS] p05_linked_list: 6/6 题通过 "
        "(反转链表/合并两个有序链表/环形链表/删除倒数第N个结点/两数相加/合并K个升序链表)"
    )


if __name__ == "__main__":
    _self_test()
