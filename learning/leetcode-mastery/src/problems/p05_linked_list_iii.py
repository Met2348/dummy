"""LeetCode 分类 05·链表 Phase 3 竞赛级补充（Part III）：断环重接做整体旋转（旋转
链表）+ 双哨兵拆分再拼接（分隔链表）+ 排序链表里"全删/留一"两种去重变体（删除排序
链表中的重复元素 I/II）+ 用栈翻转计算顺序（两数相加 II）+ 找中点+反转实现 O(1) 空间
回文判断（回文链表）+ 双指针定位首尾对称节点（交换链表中的节点），共 7 道范例，本文件
不依赖任何其它模块，独立定义 ListNode 及配套辅助函数。"""
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


def rotate_right(head: ListNode | None, k: int) -> ListNode | None:
    """
    【题意】给定链表头节点 head 和整数 k，把整条链表向右整体旋转 k 个位置（原来排在
    末尾的 k 个节点整体挪到最前面），返回旋转后的头节点。
    【思路】k 可能远大于链表长度 n（题目允许 k 达到 2×10^9），如果真的一步一步旋转
    k 次，必然超时；而且旋转 n 次等于没转，所以真正有效的旋转次数是 `k % n`。核心
    技巧是"先接成环、再从正确位置断开"：先遍历一遍算出链表长度 n 并让 `tail` 停在
    最后一个节点，把 `tail.next` 指向 `head`，链表此刻变成一个环；新的头节点应该
    是"原链表倒数第 `k%n` 个节点"，等价于"从 head 出发走 `n - k%n` 步"到达的节点的
    下一个节点——只要从 head 出发走 `n - k%n - 1` 步，停在的这个节点就是新链表的
    尾节点，它的 `next` 就是新头节点，断开这里（把它的 `next` 设为 `None`）即可。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】
    1) 必须先对 `k` 取模 `n`（`k %= n`），如果直接用原始 k 去走"n - k"步会因为
    k 远大于 n 导致这个数是负数，走负数步没有意义（Python 的 `range(负数)` 不会
    报错但直接变成空循环，会悄悄跳过旋转，返回一个和预期不符但看起来"没崩溃"的结果，
    这类静默错误比直接报错更难发现）；
    2) 取模之后如果 `k == 0`（要么原本就是 0，要么正好是 n 的整数倍），说明旋转后
    和原链表完全一样，应该直接返回原 head，不需要走后面"接环再断开"这一整套流程；
    3) 空链表（`head is None`）和单节点链表（`head.next is None`）旋转结果都是它
    自身，必须在最前面直接处理，否则后面计算长度的循环对单节点链表也能正常工作，但
    对空链表访问 `head.next` 会直接抛异常。
    """
    if head is None or head.next is None:
        return head

    n = 1
    tail = head
    while tail.next is not None:
        tail = tail.next
        n += 1

    k %= n
    if k == 0:
        return head

    tail.next = head
    steps_to_new_tail = n - k
    new_tail = head
    for _ in range(steps_to_new_tail - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head


def partition(head: ListNode | None, x: int) -> ListNode | None:
    """
    【题意】给定链表头节点 head 和整数 x，重新排列链表，使得所有小于 x 的节点都排在
    大于等于 x 的节点前面，且两部分内部各自节点的原始相对顺序必须保持不变（不能像
    快排那样任意交换）。
    【思路】"保持相对顺序"这个约束天然排除了"原地交换指针"的做法（交换容易打乱同一
    分组内的相对顺序），最直接的办法是准备两条独立的哨兵链表——`less` 只收集小于 x
    的节点，`ge` 只收集大于等于 x 的节点，各自维护一个尾指针，一次遍历原链表，按
    条件把每个节点"摘下来"接到对应链表的尾部即可，由于是顺序遍历、顺序追加，两条
    子链表内部天然保持了原有的相对顺序。遍历结束后，把 `less` 链表的尾部接上 `ge`
    链表的头部，整体拼接成一条链表返回。
    【复杂度】时间 O(n)，空间 O(1)（只重新拼接已有节点，不新建节点，两个哨兵节点除外）。
    【易错点】
    1) `ge` 链表的尾节点必须显式把 `next` 设为 `None`——它在原链表里的 `next` 可能
    还指向某个已经被摘到 `less` 链表里的节点，如果不清空，拼接后会出现"链表分叉"
    或者形成环；
    2) 两个哨兵节点（`less` 和 `ge` 各自的 dummy）只是用来简化"第一个节点往哪接"的
    判断，最终返回的是 `less_dummy.next`（可能为 None，如果所有节点都 >= x，此时
    要返回 `ge_dummy.next`——用 "`less` 接上 `ge`" 这种统一拼接方式，`less` 为空时
    `less_dummy.next` 本身就等于拼接后的 `ge_dummy.next`，不需要额外判断）；
    3) 判断条件是 `< x`（严格小于），大于等于 x 的节点（包括恰好等于 x 的节点）都要
    分到第二组，不能把 "等于 x" 的情况误分到第一组。
    """
    less_dummy = ListNode()
    ge_dummy = ListNode()
    less_tail, ge_tail = less_dummy, ge_dummy

    node = head
    while node is not None:
        if node.val < x:
            less_tail.next = node
            less_tail = less_tail.next
        else:
            ge_tail.next = node
            ge_tail = ge_tail.next
        node = node.next

    ge_tail.next = None
    less_tail.next = ge_dummy.next
    return less_dummy.next


def delete_duplicates_all(head: ListNode | None) -> ListNode | None:
    """
    【题意】给定一个已排序的链表 head，把所有"存在重复值"的节点全部删除（一个值只要
    出现超过一次，这个值对应的所有节点都要删掉，而不是只删多余的、留一个），返回只
    包含"原本就只出现一次"的那些节点组成的链表。
    【思路】链表已排序，意味着值相同的节点必然连续排列在一起，一次遍历就能识别出
    完整的"重复段"。用哨兵节点 `dummy` 接在 head 前面，`prev` 指向"当前已经确认保留
    下来的最后一个节点"，`cur` 向前扫描：一旦发现 `cur.next` 存在且和 `cur` 值相同，
    说明遇到了一整段重复值，用一个 `while` 循环把这个值对应的所有节点都跳过（包括
    `cur` 自己），跳到"这一段重复值结束后的第一个不同值节点"，再把 `prev.next` 直接
    指向这里（相当于把整段重复值一次性摘除）；如果 `cur.next` 不存在或者值不同，说明
    `cur` 是一个独立值，正常保留，`prev` 和 `cur` 都前移一位。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】
    1) 判断"是否进入重复段"要看 `cur.next` 和 `cur.val` 是否相等，而不是等删完了再
    判断——一旦确认这是重复段的开头，必须用 `while` 循环把这个值对应的**全部**节点
    都跳过（包括开头这一个），如果只删除"多余的"而保留第一个（那是 LC83 的做法），
    会得到和本题要求不符的结果；
    2) 只有在"当前不是重复段"的分支里，`prev` 才前移到 `cur`；一旦确认是重复段，
    `prev` 必须保持不动（`prev.next` 直接指向跳过重复段之后的位置），因为重复段里
    的所有节点都要被丢弃，`prev` 不能指向任何一个即将被丢弃的节点；
    3) 哨兵节点是必须的，因为链表最前面的若干个节点本身就可能是一段重复值（比如
    `[1,1,1,2,3]`），这种情况下原头节点会被整体删除，返回值不再是 `head`，必须
    返回 `dummy.next`。
    """
    dummy = ListNode(0, head)
    prev = dummy
    cur = head
    while cur is not None:
        if cur.next is not None and cur.next.val == cur.val:
            val = cur.val
            while cur is not None and cur.val == val:
                cur = cur.next
            prev.next = cur
        else:
            prev = cur
            cur = cur.next
    return dummy.next


def delete_duplicates_keep_one(head: ListNode | None) -> ListNode | None:
    """
    【题意】给定一个已排序的链表 head，删除重复出现的节点，使每个值只保留一个（和上一
    题 `delete_duplicates_all` 的区别是：这里重复值要留一份，不是全部删光），返回
    去重后仍然有序的链表。
    【思路】因为链表已排序、重复值必然相邻，且这里只需要"留一份、删多余的"，不需要
    像 LC82 那样借助 `prev` 指针去整体摘除一整段——只要发现 `cur.next` 和 `cur` 值
    相同，直接把 `cur.next` 短接成 `cur.next.next`（跳过那个多余的重复节点），`cur`
    本身不需要移动，因为 `cur` 后面新接上来的节点仍然可能和 `cur` 值相同（比如
    `[1,1,1]`，跳过第一个多余的 1 之后，新的 `cur.next` 还是 1，需要继续跳）；只有
    当 `cur.next` 和 `cur` 值不同时，`cur` 才前移一位，去检查下一个值是否有重复。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】
    1) 不需要引入哨兵节点——因为本题一定会保留原链表的第一个节点（第一个值不可能被
    完全删除，最多是它后面的重复值被删掉），所以 `head` 本身不会改变，直接返回
    `head` 即可，这一点和 `delete_duplicates_all` 恰好相反；
    2) 跳过重复节点后 `cur` 不能立刻前移，必须停在原地重新检查新的 `cur.next`——
    如果每跳过一个就无条件前移一位，遇到 3 个连续相同值时会漏删中间那一个；
    3) 循环条件要同时检查 `cur is not None` 和 `cur.next is not None`，避免在
    链表末尾对 `cur.next.val` 取值时因为 `cur.next` 是 `None` 而抛异常。
    """
    cur = head
    while cur is not None and cur.next is not None:
        if cur.next.val == cur.val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return head


def add_two_numbers_ii(
    l1: ListNode | None, l2: ListNode | None
) -> ListNode | None:
    """
    【题意】给定两个非空链表 l1、l2，分别表示两个非负整数（**最高位在链表头部**，和
    LC2"两数相加"的低位在前正好相反），返回它们相加之和，同样以"最高位在头部"的链表
    形式表示（假设输入不含前导 0，除非数字本身就是 0）。
    【思路】加法必须从个位（也就是链表的**尾部**）开始算起，但链表只能从头部顺序
    访问——这个"想要的访问顺序"和"实际能访问的顺序"正好相反，天然需要借助一个后进
    先出的结构来"倒转访问顺序"，这正是栈的用武之地。把 l1、l2 的每一位数字分别压入
    两个栈 `s1`、`s2`（压栈顺序是从最高位到最低位，栈顶自然就是最低位/个位）。之后
    不断从两个栈顶各弹出一位（某个栈已经耗尽就补 0），加上进位 `carry` 算出这一位
    的结果和新的进位；每算出一位，就把它作为新节点"插到当前结果链表的最前面"（而
    不是追加到末尾）——因为个位最先算出来，但它在最终结果里应该排在最后面（最低位），
    "每次都往头部插入"这个操作天然把计算顺序倒转回了正确的展示顺序。循环持续到两个
    栈都空、且进位也清零为止（进位不为 0 时即使两个栈都空了也要再生成一位，例如
    99+1=100 的最高位新进位）。
    【复杂度】时间 O(m+n)，空间 O(m+n)（两个栈存储链表数值 + 结果链表本身）。
    【易错点】
    1) 循环终止条件必须是"`s1` 非空 或 `s2` 非空 或 `carry` 非零"三者任意一个成立
    就继续，只检查两个栈是否为空会漏掉"最高位相加产生新的进位"这一位（比如
    5+5=10，两个栈弹完之后还需要再生成一个新的最高位节点 '1'）；
    2) 新生成的节点必须"接到当前结果链表的最前面"（`node.next = result_head` 再把
    `result_head` 更新为 `node`），如果写成"接到末尾"，个位会被错误地排到结果链表
    的最前面（变成最高位），整个数字就算反了；
    3) 某个栈提前耗尽后，取值要用"栈空则按 0 处理"（`s1.pop() if s1 else 0`），不能
    直接假设两个栈长度相等同步弹出——两个输入数字的位数完全可以不同（比如
    "9999" 和 "1"）。
    """
    def to_stack(node: ListNode | None) -> list[int]:
        stack: list[int] = []
        while node is not None:
            stack.append(node.val)
            node = node.next
        return stack

    s1, s2 = to_stack(l1), to_stack(l2)
    carry = 0
    result_head: ListNode | None = None
    while s1 or s2 or carry:
        v1 = s1.pop() if s1 else 0
        v2 = s2.pop() if s2 else 0
        total = v1 + v2 + carry
        carry, digit = divmod(total, 10)
        node = ListNode(digit)
        node.next = result_head
        result_head = node
    return result_head


def is_palindrome(head: ListNode | None) -> bool:
    """
    【题意】给定单链表头节点 head，判断这条链表从头到尾读出来的数值序列是否是回文
    （正读和反读结果相同）。进阶要求：只用 O(1) 额外空间完成（不能把所有值先存进
    数组再判断）。
    【思路】数组判断回文可以直接双指针从两端向中间收敛，但单链表没有"从后往前"的
    直接访问能力。做法是复用前面题目学过的两个心法拼在一起：先用快慢指针（和 LC876
    找中点、LC143 重排链表完全一样的写法）找到链表中点，把后半部分**原地反转**（复用
    "反转链表"LC206 的标准写法），这样后半部分就变成了"从中点往前"的可访问顺序；然后
    用两个指针分别从"原链表头部"和"反转后的后半部分头部"同步往前走，逐位比较值是否
    相等——如果整个后半部分走完都没有不相等的位置，说明是回文。
    【复杂度】时间 O(n)（找中点、反转、比较各是一趟线性扫描）；空间 O(1)（只重新
    拼接已有节点的指针，不新建节点、不使用额外数组）。
    【易错点】
    1) 找中点时快指针要从 `head` 出发（而不是 `head.next`），配合循环条件
    `fast.next and fast.next.next`，这样在链表长度为奇数时，中点会划给前半部分——
    这和 LC143 重排链表用的是同一种切分方式，此时不需要额外处理"奇数长度中间那个
    节点单独比不比较"的问题（比较只在后半部分走完时停止，奇数长度时中间节点留在
    前半部分、天然不参与比较）；
    2) 反转后半部分之前，必须先把前半部分的尾节点 `next` 设为 `None`，断开前后两半
    的连接，否则前半部分遍历时会越过中点、继续遍历到已经被反转的后半部分，比较会
    错位；
    3) 只需要循环到"反转后的后半部分"走完为止（`while p2:`），不需要也不能循环到
    前半部分走完——后半部分长度不超过前半部分（奇数长度时严格更短），用较短的一侧
    控制循环次数，避免在较长的一侧访问越界的 `None`。
    """
    if head is None or head.next is None:
        return True

    slow, fast = head, head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next

    def reverse(node: ListNode | None) -> ListNode | None:
        prev = None
        while node is not None:
            next_tmp = node.next
            node.next = prev
            prev = node
            node = next_tmp
        return prev

    second = reverse(slow.next)
    slow.next = None

    p1, p2 = head, second
    while p2 is not None:
        if p1.val != p2.val:
            return False
        p1 = p1.next
        p2 = p2.next
    return True


def swap_nodes(head: ListNode | None, k: int) -> ListNode | None:
    """
    【题意】给定链表头节点 head 和整数 k（从 1 开始计数），交换"正数第 k 个节点"和
    "倒数第 k 个节点"的**值**（不需要真的改变节点之间的连接关系，只交换 val），返回
    交换后的头节点。
    【思路】"正数第 k 个节点"很直接——从 head 出发走 `k-1` 步就到了。"倒数第 k 个
    节点"如果先算出链表总长度 n 再换算成"正数第 n-k+1 个"，需要遍历两趟；更省事的
    做法是复用"快慢指针间隔 k 步"的经典技巧（和 LC19"删除链表的倒数第 N 个结点"是
    同一个思路的变体）：一个指针 `runner` 先走到"正数第 k 个节点"的位置，另一个指针
    `second` 从 head 出发，两个指针此后同步前进，`runner` 每前进一步 `second` 也
    前进一步——当 `runner` 走到链表最后一个节点时，`second` 和 `runner` 之间的距离
    始终保持不变（都是 k-1 步的间隔），这个恒定间隔正好保证 `second` 停在"倒数第 k
    个节点"上，一次遍历就同时定位出两个目标节点，不需要额外遍历算长度。定位完成后，
    直接交换两个节点的 `val` 即可（题目只要求交换值，不要求真的重新连接指针，比"交换
    两两节点"LC24 这种要求真正交换指针连接关系的题简单）。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】
    1) 定位"正数第 k 个节点"时循环要走 `k-1` 步（从 head 出发，走 0 步就停在第 1
    个节点上，所以走到第 k 个节点需要走 k-1 步），走成 k 步会多走一位，定位到第
    k+1 个节点；
    2) 双指针同步前进的循环条件是 `runner.next is not None`（`runner` 还没到最后
    一个节点），退出循环时 `runner` 恰好停在最后一个节点、`second` 恰好停在倒数第
    k 个节点——如果写成 `runner is not None`，会在 `runner` 已经是 `None` 之后才
    停止，`second` 会多走一步，定位到倒数第 k-1 个节点；
    3) 交换的是 `.val` 属性，不是交换两个节点对象本身或者它们的 `.next` 指针——
    题目只要求最终链表遍历出来的数值序列符合要求，直接换值是最简单且不容易引入
    "指针接错导致成环/断链"这类 bug 的写法。
    """
    first = head
    for _ in range(k - 1):
        first = first.next
    kth_from_start = first

    second = head
    runner = first
    while runner.next is not None:
        runner = runner.next
        second = second.next
    kth_from_end = second

    kth_from_start.val, kth_from_end.val = kth_from_end.val, kth_from_start.val
    return head


def _self_test() -> None:
    assert list_to_vals(rotate_right(build_list([1, 2, 3, 4, 5]), 2)) == [4, 5, 1, 2, 3]
    assert list_to_vals(rotate_right(build_list([0, 1, 2]), 4)) == [2, 0, 1]
    assert rotate_right(build_list([]), 5) is None
    assert list_to_vals(rotate_right(build_list([1]), 99)) == [1]

    assert list_to_vals(partition(build_list([1, 4, 3, 2, 5, 2]), 3)) == [
        1, 2, 2, 4, 3, 5,
    ]

    assert list_to_vals(
        delete_duplicates_all(build_list([1, 2, 3, 3, 4, 4, 5]))
    ) == [1, 2, 5]
    assert list_to_vals(delete_duplicates_all(build_list([1, 1, 1, 2, 3]))) == [2, 3]

    assert list_to_vals(delete_duplicates_keep_one(build_list([1, 1, 2]))) == [1, 2]
    assert list_to_vals(
        delete_duplicates_keep_one(build_list([1, 1, 2, 3, 3]))
    ) == [1, 2, 3]

    assert list_to_vals(
        add_two_numbers_ii(build_list([7, 2, 4, 3]), build_list([5, 6, 4]))
    ) == [7, 8, 0, 7]
    assert list_to_vals(
        add_two_numbers_ii(build_list([2, 4, 3]), build_list([5, 6, 4]))
    ) == [8, 0, 7]
    assert list_to_vals(add_two_numbers_ii(build_list([0]), build_list([0]))) == [0]

    assert is_palindrome(build_list([1, 2, 2, 1])) is True
    assert is_palindrome(build_list([1, 2])) is False
    assert is_palindrome(build_list([1])) is True

    assert list_to_vals(swap_nodes(build_list([1, 2, 3, 4, 5]), 2)) == [1, 4, 3, 2, 5]
    assert list_to_vals(
        swap_nodes(build_list([7, 9, 6, 6, 7, 8, 3, 0, 9, 5]), 5)
    ) == [7, 9, 6, 6, 8, 7, 3, 0, 9, 5]
    assert list_to_vals(swap_nodes(build_list([1]), 1)) == [1]

    print(
        "[PASS] p05_linked_list_iii: 7/7 题通过 "
        "(旋转链表/分隔链表/删除排序链表中的重复元素II/删除排序链表中的重复元素/"
        "两数相加II/回文链表/交换链表中的节点)"
    )


if __name__ == "__main__":
    _self_test()
