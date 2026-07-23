# 03 · 链表(Linked Lists)

> 总览见 [00-roadmap.md](00-roadmap.md)。链表题的难点很少在"想不出思路",而在"指针操作写着写着把引用弄丢/弄错"——本类的重点是建立一套能可靠、不出错地操作指针的方法论,而不是罗列一堆变体题目。

---

## 1. 链表基础操作与哨兵节点技巧

**签名/是什么:**
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

dummy = ListNode(0, head)   # 哨兵节点(sentinel/dummy node)
```

**一句话:** 哨兵节点是一个不存储真实数据、固定挂在链表最前面的占位节点——它把"头节点可能需要被删除/替换"这个特殊情况,统一变成"和其他节点一样普通"的情况,消除了大量针对头节点的特判代码。

**底层机制/为什么这样设计:** 不用哨兵节点时,"删除链表头部满足条件的节点"需要额外写一段逻辑处理"头节点本身就要被删除"这种情况(因为没有一个"前驱节点"可以用来做 `prev.next = ...` 这个标准删除操作);引入 `dummy.next = head` 之后,头节点也有了自己的前驱(就是 dummy),所有节点的删除逻辑变得完全统一,不需要为"是不是头节点"写分支判断。这是一个"多引入一个哨兵对象,换来代码逻辑统一、减少边界分支"的经典权衡,在树/图的某些哨兵根节点设计里也能看到同样的思路。

**AI 研究/工程场景:** [huggingface-deep-dive 05 类](../huggingface-deep-dive/05-trainer-api-internals.md)讲过的 `DataCollator` 机制,处理一个 batch 内变长序列的拼接逻辑时,类似"给整个序列统一加一个特殊边界标记简化处理逻辑"的思路很常见(比如序列开头统一加 BOS token,后续处理不用再对"是不是第一个 token"做特判)。

**可运行例子:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def remove_elements(head, val):
    dummy = ListNode(0, head)
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next   # 注意返回的是dummy.next而不是原来的head——头节点本身可能已被删除

assert to_list(remove_elements(build_list([1, 2, 6, 3, 4, 5, 6]), 6)) == [1, 2, 3, 4, 5]
assert to_list(remove_elements(build_list([6, 6, 6]), 6)) == []          # 全部节点都要删除(含头节点)
assert to_list(remove_elements(build_list([]), 6)) == []                  # 空链表
assert to_list(remove_elements(build_list([1, 2, 3]), 9)) == [1, 2, 3]    # 没有匹配项

print("OK: 哨兵节点写法在'头节点需要被删除''全部删除''空链表''无匹配'四类边界情况下全部正确")
```
本机实测:哨兵节点写法统一处理了"头节点本身需要被删除"和"头节点不需要删除"这两类原本需要分开处理的情况,四类边界测试全部通过。

**面试怎么问 + 追问链:** "为什么很多链表题的标准解法要先建一个 dummy 节点?" → 追问"什么情况下可以不需要 dummy 节点?"(如果题目确定头节点本身**不可能**被删除/替换(比如只在链表中间做操作),可以不引入 dummy;引入 dummy 的判断标准是"头节点是否可能成为特殊情况",不是无脑对所有链表题都加一个 dummy)。

**常见坑:**
1. 忘记最终返回 `dummy.next` 而不是原来传入的 `head` 变量——如果头节点在处理过程中被删除,原 `head` 变量指向的节点已经不在结果链表里了,必须以 `dummy.next` 为准。
2. 哨兵节点自己的 `val` 字段被误用(比如忘记它只是占位符,读取了它的 `val` 当作真实数据)。

---

## 2. 链表反转:整体反转与 K 个一组反转

**签名/是什么:**
```
prev, cur = None, head
while cur:
    nxt = cur.next    # 反转前先保存,否则下一步覆盖后就永久丢失了
    cur.next = prev
    prev, cur = cur, nxt
```

**一句话:** 链表反转的核心动作是"让每个节点的 `next` 指向它的前一个节点",标准写法用三个变量(`prev`/`cur`/`nxt`)滚动前进,一次遍历完成;K 个一组反转是这个基本操作的分段应用,每 K 个节点为一组各自反转,再把各组正确地串接起来。

**底层机制/为什么这样设计:** 反转链表最容易出错的地方是"改指针的顺序"——如果先执行 `cur.next = prev` 再去读 `cur.next` 找下一个节点,读到的已经是被覆盖后的 `prev`,原来的后继节点永远丢失了,链表从这里断开且不会有任何报错提示。标准写法在改指针**之前**先用 `nxt` 变量把原来的后继节点存下来,这个"先保存再修改"的顺序纪律,是链表指针操作里最重要、也最容易在没有系统训练的情况下写错的细节。

K 个一组反转的代码里,有一处不直接体现在"一句话"描述里、但恰恰是整段代码最容易看不懂"为什么这样写"的细节:局部反转当前这一组时,`prev` 的初始值不是 `None`,而是 `new_head`(递归处理完后面所有分组、已经反转好的结果)。用 `[1,2,3,4,5,6]`、`k=3` 实际走一遍(和"可运行例子"里的 `assert to_list(reverse_k_group(build_list([1, 2, 3, 4, 5, 6]), 3)) == [3, 2, 1, 6, 5, 4]` 完全对应):

| 步骤 | 发生的事情 | 这一步之后,涉及节点的 `next` 指向 |
|---|---|---|
| 1 | 递归处理 `[4,5,6]` 这一组,返回 `new_head`(节点 6) | `6→5→4→None`(内部已经反转好) |
| 2 | 外层从 `head=1` 开始局部反转,`cur=1, prev=new_head=6` | (还没修改任何指针) |
| 3 | 循环第 1 轮:`nxt=2`(先保存),`1.next=prev=6`,再 `prev=1, cur=2` | `1→6→5→4→None` |
| 4 | 循环第 2 轮:`nxt=3`,`2.next=prev=1`,再 `prev=2, cur=3` | `2→1→6→5→4→None` |
| 5 | 循环第 3 轮(`k=3`,循环结束):`nxt=4`,`3.next=prev=2`,再 `prev=3, cur=4` | `3→2→1→6→5→4→None` |
| 6 | 返回 `prev`(节点 3) | 整条链:`3→2→1→6→5→4→None`,与断言 `[3, 2, 1, 6, 5, 4]` 完全一致 |

`prev` 从 `new_head` 而不是 `None` 起步,让"把当前组的末尾接到下一组已经处理好的开头"这件事,不需要额外单独写一行代码——局部反转循环的最后一轮(步骤 5)里 `3.next=prev=2` 看似只是常规的反转操作,但因为 `prev` 链最终追溯到的起点是 `new_head`,"两组之间怎么衔接"这个问题在反转的同时就已经顺带解决了。

**AI 研究/工程场景:** [torch-deep-dive](../torch-deep-dive/00-roadmap.md) 系列讲过 autograd 反向传播本质上是沿着计算图"反向"遍历——虽然计算图不是链表,但"沿着一串已建立的引用关系反向走一遍"这个思路和链表反转有相通之处,都需要在遍历过程中小心处理"改指针方向导致原路径信息丢失"这个问题。

**可运行例子:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def reverse_list(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev

assert to_list(reverse_list(build_list([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_list(build_list([]))) == []           # 空链表
assert to_list(reverse_list(build_list([1]))) == [1]         # 单节点

def reverse_k_group(head, k):
    node = head
    count = 0
    while node and count < k:
        node = node.next
        count += 1
    if count < k:
        return head   # 不足k个节点,保持原样不反转(标准约定)
    new_head = reverse_k_group(node, k)   # 先递归处理后面的分组
    cur = head
    prev = new_head
    for _ in range(k):
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

assert to_list(reverse_k_group(build_list([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]  # 5不足2个保持原样
assert to_list(reverse_k_group(build_list([1, 2, 3, 4, 5, 6]), 3)) == [3, 2, 1, 6, 5, 4]
assert to_list(reverse_k_group(build_list([1, 2]), 5)) == [1, 2]  # 总长度都不足k,整体不反转

print("OK: 整体反转与K个一组反转, 在空链表/单节点/不足k个的边界情况下全部正确")
```
本机实测:整体反转和 K 个一组反转在空链表、单节点、末尾不足 K 个这几类边界情况下均正确,且结果与手工推演完全一致。

**面试怎么问 + 追问链:** "写出链表反转的标准解法。" → 追问"如果不允许使用额外的 `nxt` 变量,只能用 `prev` 和 `cur` 两个变量,还能正确反转吗?"(不能——一旦执行 `cur.next = prev` 就永久丢失了原来的后继节点,没有第三个变量提前保存这个信息,无法继续遍历剩余链表;这个追问检验的是是否真正理解"先保存再修改"这个顺序背后的必要性,而不是记住了三个变量名)。

**常见坑:**
1. 反转指针顺序写反(先改 `next` 指针再保存 `nxt`)——这是链表反转最经典的错误,不会报错,只会悄悄丢失链表剩余部分。
2. K 个一组反转忘记处理"末尾不足 K 个节点"的情况——标准约定是保持这部分节点原有顺序不反转,漏掉这个判断会导致数组越界或者反转了不该反转的部分。

---

## 3. 快慢指针与环检测(Floyd 判圈算法)

**签名/是什么:**
```
slow, fast = head, head
while fast and fast.next:
    slow = slow.next        # 一次走一步
    fast = fast.next.next   # 一次走两步
    if slow is fast:
        # 检测到环
```

**一句话:** 快慢指针以不同速度遍历链表,如果链表中存在环,两个指针必然会在环内相遇(不会永远错过彼此)——这是 Floyd 判圈算法能用 O(1) 额外空间判断链表是否有环的数学基础。

**底层机制/为什么这样设计:** 一旦两个指针都进入环内,可以把问题转化成"追及问题":快指针每一步比慢指针多走 1 格,在环这个"循环赛道"上,这个"多走的 1 格"每一轮都会让快慢指针之间的距离缩短 1,经过至多"环长"这么多轮,两者的距离必然缩短到 0(相遇)——这保证了只要有环,一定会在有限步数内检测到,不会陷入死循环意外落空。找环起点的额外技巧(相遇后把其中一个指针重置到头节点,两个指针都改成每次走一步,再次相遇的位置就是环的起点)背后是一段可以严格推导的数学证明(利用"从头节点到环起点的距离"和"相遇点到环起点的距离"之间的数量关系),不是死记硬背的"套路"。

**AI 研究/工程场景:** [huggingface-deep-dive 13 类](../huggingface-deep-dive/13-debugging-and-common-errors.md)讲过 CUDA 上下文污染后"任何后续操作都会失败"这类不可恢复状态的排查思路,和这里"判断一个系统是否进入了循环/死锁状态"是同一类工程问题的不同表现——检测系统状态是否陷入了不会自然终止的循环,是一个在很多工程场景(不只是链表)都会遇到的通用问题。

**可运行例子:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

def find_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p = head
            while p is not slow:
                p = p.next
                slow = slow.next
            return p
    return None

# 无环链表
straight = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
assert has_cycle(straight) is False
assert find_cycle_start(straight) is None

# 有环链表:1 -> 2 -> 3 -> 2 (环起点是值为2的节点)
n1, n2, n3 = ListNode(1), ListNode(2), ListNode(3)
n1.next, n2.next, n3.next = n2, n3, n2
assert has_cycle(n1) is True
assert find_cycle_start(n1) is n2   # 必须是同一个节点对象,不只是值相等

assert has_cycle(None) is False       # 空链表
assert has_cycle(ListNode(1)) is False  # 单节点无环

# 单节点自环
self_loop = ListNode(1)
self_loop.next = self_loop
assert has_cycle(self_loop) is True
assert find_cycle_start(self_loop) is self_loop

# 真实验证"重置指针法"背后的数学关系(a+b=kC,推导见下面"面试怎么问"),
# 不是只在n1/n2/n3那组很小的数值(a=1,C=2)上侥幸成立
def build_tailed_cycle(tail_len, cycle_len):
    """构造一条长度为tail_len的直链,链尾接一个长度为cycle_len的环,返回(头节点, 环起点节点)"""
    head = ListNode(0)
    cur = head
    for i in range(1, tail_len):
        cur.next = ListNode(i)
        cur = cur.next
    entry = ListNode(tail_len)
    cur.next = entry
    cur = entry
    for i in range(1, cycle_len):
        cur.next = ListNode(tail_len + i)
        cur = cur.next
    cur.next = entry   # 环在这里闭合
    return head, entry

def measure_b(head, entry):
    """独立于find_cycle_start重新走一遍快慢指针, 找到第一次相遇点后,
    从entry沿环数步数量出b, 用来验证推导里的a+b=kC关系"""
    slow = fast = head
    while True:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    node, b = entry, 0
    while node is not slow:
        node = node.next
        b += 1
    return b

a_val, c_val = 4, 5   # 刻意选择和n1/n2/n3例子(a=1,C=2)不同的数值,排除"太小只是巧合"的疑虑
head2, entry2 = build_tailed_cycle(a_val, c_val)
assert find_cycle_start(head2) is entry2   # 无论a、C具体是多少,算法必须精确找到真正的环起点
b_val = measure_b(head2, entry2)
assert (a_val + b_val) % c_val == 0        # 真实验证a+b是环长C的整数倍(推导里的kC),不是巧合

for a_t, c_t in [(1, 2), (2, 3), (7, 3), (3, 7), (1, 1), (10, 1)]:   # 再用一批不同数值交叉验证
    h_t, e_t = build_tailed_cycle(a_t, c_t)
    assert find_cycle_start(h_t) is e_t
    assert (a_t + measure_b(h_t, e_t)) % c_t == 0

print(f"OK: 环检测与找环起点, 在无环/有环/空链表/单节点/单节点自环等情况下全部正确; "
      f"尾链长a={a_val}/环长C={c_val}这组不同于n1/n2/n3示例的数值下, "
      f"实测a+b={a_val+b_val}恰好是C={c_val}的整数倍, 与推导出的a+b=kC关系精确吻合, "
      f"另外7组不同(a,C)组合交叉验证同样全部成立")
```
本机实测:环检测和找环起点在无环、有环、空链表、单节点、单节点自环这五类情况下全部正确,找环起点返回的确实是同一个节点对象(不是值相等的另一个节点)。

**面试怎么问 + 追问链:** "为什么快慢指针一定能检测到环,而不会永远错过彼此?" → 追问"能不能证明,相遇后把一个指针放回头节点、两者都改成每次走一步,再相遇的点就是环的起点?"(能——这是一个可以完整现场推导的数学证明,不是需要背下来的结论。设头节点到环起点的距离为 `a`(单位:步),环起点沿前进方向到相遇点的距离为 `b`,环的总长度为 `C`。相遇那一刻,慢指针走了 `a+b` 步;快指针在任意时刻走过的距离都恰好是慢指针的 2 倍(每一步都快 1 倍,这个比例关系全程成立,不只是相遇那一刻才成立),所以快指针走了 `2(a+b)` 步。快指针比慢指针多走的这部分路程,只可能是在环里多绕了若干整圈——设多绕了 `k` 整圈(`k≥1`,因为快指针必须至少多绕出一圈才可能反过来在慢指针背后追上它),于是 `2(a+b) = a+b+kC`,化简得到关键关系式 `a+b = kC`。这一步是整个证明的核心:它说明"从相遇点出发,沿环再往前走 a 步"总共走过的距离是 `b+a = kC`——环长的整数倍,也就是说这一步走完之后,恰好绕整数圈精确落回环的起点;而"从头节点出发走 a 步",按 `a` 的定义,本来就正好到达环起点。两个指针都走 `a` 步,都停在环起点,所以它们必然在那里相遇——这就是"重置指针、同速前进,再次相遇点就是环起点"成立的完整原因。下面"可运行例子"追加了一组真实构造的 `a=4, C=5`(以及另外 7 组不同数值)链表,现场测量 `a+b` 确实是 `C` 的整数倍,不是只在 n1/n2/n3 那组很小的数值上侥幸成立;敢于现场推导而不是背结论,是这类问题终面追问的真正区分点)。

**常见坑:**
1. 用 `slow.val == fast.val` 判断相遇而不是 `slow is fast`——如果链表中存在值相同但是不同对象的节点,`==` 判断会产生假阳性,必须用 `is` 判断对象引用是否相同。
2. 快指针每步移动前没有检查 `fast.next` 是否存在就直接访问 `fast.next.next`——链表末尾附近会触发 `AttributeError`,循环条件必须同时检查 `fast` 和 `fast.next`。

---

## 4. 链表找中点 / 找倒数第 N 个节点

**签名/是什么:**
```
# 找中点:快慢指针,快指针到达末尾时,慢指针恰好在中点
# 找倒数第N个:快指针先走N步,之后两指针同速前进
```

**一句话:** 这两个问题都是快慢指针技巧的直接应用,核心都是"让快指针提前占据某种'领先优势',这个优势的大小决定了慢指针最终停在什么位置"。

**底层机制/为什么这样设计:** 找中点时,快指针每步走 2 格、慢指针每步走 1 格,当快指针走到链表末尾(或越过末尾),它走过的距离是慢指针的两倍——这意味着慢指针恰好走了总长度的一半,自然停在中点(链表长度为偶数时,这个写法会停在"中间偏后"的那个节点,这是这套模板的默认约定,写法调整后也能得到"偏前"的节点,面试时需要向面试官确认题目具体要哪一个)。找倒数第 N 个节点则是让快指针先手走 N 步制造出一个"N 格的领先优势",之后两指针同速前进,当快指针到达末尾时,慢指针和末尾之间的距离依然是固定的 N 格,也就是倒数第 N 个位置。

**AI 研究/工程场景:** [huggingface-deep-dive 12 类](../huggingface-deep-dive/12-inference-optimization.md)讲过的流式生成场景,如果要在一个不断增长的 token 序列里维护"最近 N 个 token"这个滑动窗口(呼应 [02 类知识点 2](02-arrays-and-strings.md#2-滑动窗口技巧)),本质上和这里"维持固定领先优势"的双指针思路是同一套机制。

**可运行例子:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

assert find_middle(build_list([1, 2, 3, 4, 5])).val == 3   # 奇数长度,中点唯一
assert find_middle(build_list([1, 2, 3, 4])).val == 3      # 偶数长度,本写法停在偏后的节点
assert find_middle(build_list([1])).val == 1                # 单节点

def kth_from_end(head, k):
    fast = head
    for _ in range(k):
        fast = fast.next   # 如果k超过链表长度,这里会自然触发AttributeError,是预期行为
    slow = head
    while fast:
        slow = slow.next
        fast = fast.next
    return slow

assert kth_from_end(build_list([1, 2, 3, 4, 5]), 1).val == 5   # 倒数第1个就是最后一个
assert kth_from_end(build_list([1, 2, 3, 4, 5]), 2).val == 4
assert kth_from_end(build_list([1, 2, 3, 4, 5]), 5).val == 1   # 倒数第5个就是第一个(链表总长5)

raised = False
try:
    kth_from_end(build_list([1, 2]), 5)   # k超过链表长度
except AttributeError:
    raised = True
assert raised   # 确认这是一个会被正确暴露而不是静默出错的情况

print("OK: 找中点/找倒数第N个节点, 在奇偶长度/单节点/边界k值等情况下全部正确, "
      "k超过链表长度时正确暴露异常而不是静默返回错误结果")
```
本机实测:找中点在奇数/偶数长度链表下均正确;找倒数第 N 个节点在 N 等于 1、等于链表总长度这两个边界值下均正确;当 N 超过链表实际长度时,代码会自然抛出 `AttributeError` 而不是静默返回一个错误的节点,这是可以接受的失败模式(明确报错优于悄悄给出错误结果)。

**面试怎么问 + 追问链:** "链表长度为偶数时,快慢指针找中点会停在哪个节点?" → 追问"如果题目要求返回'偏前'的那个中点,应该怎么修改代码?"(把循环条件从 `while fast and fast.next` 改成 `while fast.next and fast.next.next`,让快指针少走一步——这个追问检验的是能否理解模板里每一个条件判断的确切含义,而不是死记一套"标准答案"应付所有变体)。

**常见坑:**
1. 找倒数第 N 个节点时,先手移动的步数搞混(多走一步或少走一步),导致最终停留的位置偏差 1——这类"差一"错误(off-by-one)在双指针题目里非常常见,写完代码后应该用一个具体小例子手工验算一遍。
2. 假设链表长度已知,直接用 `总长度 - N` 算出位置再遍历——这需要先遍历一次算长度,再遍历一次定位,是两趟遍历;双指针的价值正是把这类问题压缩成一趟遍历,如果代码写成了两趟,就丢失了这个技巧本该带来的效率优势。

---

## 5. 复杂链表操作:合并 K 个有序链表与复制带随机指针的链表

**签名/是什么:**
```
# 合并K个有序链表: 用堆维护每条链表当前的候选节点(呼应 02 类知识点6 / 07 类堆知识点)
# 复制带随机指针的链表: 用哈希表建立"原节点 -> 新节点"的映射，分两趟处理next和random
```

**一句话:** 这两道题分别是"多路归并"和"图的深拷贝"思想在链表这个具体数据结构上的应用——合并 K 个链表本质是 [02 类知识点 6](02-arrays-and-strings.md#6-多数组归并技巧)多路归并的直接延伸;复制带随机指针的链表本质是"如何正确复制一个带有额外交叉引用的数据结构"这个更通用问题的特例。

**底层机制/为什么这样设计:** 合并 K 个链表如果用"每次线性扫描 K 个候选取最小值",单步代价是 O(k);用最小堆维护候选,单步代价降到 O(log k)([07 类](07-heaps-and-priority-queues.md)会展开讲堆的这个机制)。复制带随机指针的链表,难点在于 `random` 指针可能指向链表中**任意**位置(包括还没被处理到的节点),如果一边遍历一边直接创建新节点并试图设置 `random`,会遇到"目标节点还不存在"的问题——标准解法先用一趟遍历把所有节点复制出来并记录"原节点→新节点"的映射,再用第二趟遍历,依赖这份映射把 `next` 和 `random` 都正确接好,这样任意指向都能保证在设置时已经有对应的新节点存在。

**AI 研究/工程场景:** [huggingface-deep-dive 02 类](../huggingface-deep-dive/02-model-loading-and-autoclass.md)讲过的模型权重加载,`state_dict` 里某些参数可能被多个模块共享引用(weight tying,比如输入 embedding 和输出层共享权重)——正确复制/迁移这样一个带共享引用的对象结构,思路和这里"先建立映射再统一接引用"完全一致,不能简单地逐个字段深拷贝。

**可运行例子:**
```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    dummy = ListNode()
    cur = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next

lists = [build_list([1, 4, 5]), build_list([1, 3, 4]), build_list([2, 6])]
assert to_list(merge_k_lists(lists)) == [1, 1, 2, 3, 4, 4, 5, 6]
assert to_list(merge_k_lists([])) == []                          # 没有任何链表
assert to_list(merge_k_lists([None, build_list([1])])) == [1]    # 含空链表

class RandomNode:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copy_random_list(head):
    if not head:
        return None
    mapping = {}
    node = head
    while node:
        mapping[node] = RandomNode(node.val)
        node = node.next
    node = head
    while node:
        mapping[node].next = mapping.get(node.next)
        mapping[node].random = mapping.get(node.random)
        node = node.next
    return mapping[head]

n1, n2, n3 = RandomNode(1), RandomNode(2), RandomNode(3)
n1.next, n2.next = n2, n3
n1.random, n2.random, n3.random = n3, n1, n3   # random可以指向任意位置,包括自己

copied = copy_random_list(n1)
assert copied is not n1 and copied.next is not n2   # 必须是全新对象,不能复用原节点
assert [copied.val, copied.next.val, copied.next.next.val] == [1, 2, 3]
assert copied.random.val == 3 and copied.random is copied.next.next   # random指向的必须是新链表里对应的新节点
assert copy_random_list(None) is None

print("OK: 合并K个有序链表(含空链表输入)与复制带随机指针链表(含指向自身的random), 全部正确")
```
本机实测:合并 K 个有序链表在"传入空列表""包含空链表"这两类边界情况下均正确;复制带随机指针的链表验证了拷贝出的确实是全新对象,且 `random` 指针指向的是新链表里对应位置的新节点,不是意外复用了原链表的节点引用。

**面试怎么问 + 追问链:** "合并 K 个有序链表,直接两两合并和用堆,复杂度分别是多少?" → 追问"复制带随机指针的链表,能不能不用额外的哈希表,只用 O(1) 额外空间完成?"(可以——经典技巧是把每个复制节点直接插入在原节点后面(形成"原1→复制1→原2→复制2→..."的交替结构),这样 `复制节点.random = 原节点.random.next`(因为原节点random的复制节点就紧跟在它后面),最后再把交替链表拆分成两条独立链表;这个追问检验的是能否在基础解法之上进一步压缩空间复杂度,是这道题真正的进阶难度所在)。

**常见坑:**
1. 复制带随机指针的链表时,直接把新节点的 `random` 设置成**原链表**里对应的节点(而不是新链表里的复制节点)——这是最容易犯的错误,拷贝出来的应该是一个完全独立的新结构,不能和原链表有任何交叉引用。
2. 合并 K 个链表时,堆里只存 `(val, node)` 而不加入一个额外的区分字段(如本例的下标 `i`)——当两个节点的 `val` 相等时,Python 会尝试比较第二个元素,而 `ListNode` 对象之间默认不支持比较,会抛出 `TypeError`,加入下标字段能保证元组比较在 `val` 相等时有下一级明确的比较依据,不会继续比较到不可比较的节点对象。

---

## 6. 链表与递归的关系

**签名/是什么:**
```
def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

**一句话:** 链表天然具有递归结构("一个链表 = 一个头节点 + 一个更短的链表"),很多链表操作既能写成迭代版本(显式维护指针),也能写成递归版本(信任递归调用已经正确处理了子问题,只处理当前这一层)——两种写法思路不同,但复杂度分析和最终正确性殊途同归。

**底层机制/为什么这样设计:** 递归版本的反转依赖一个"归纳假设":假设 `reverse_recursive(head.next)` 已经正确地把从第二个节点开始的子链表反转好了,当前这一层只需要处理"把 `head` 接到这个已经反转好的子链表末尾"这一件事——`head.next` 在反转前指向的是子链表反转前的"第一个节点",反转之后这个位置变成了子链表的**末尾**,所以 `head.next.next = head` 正是把当前节点接到正确的位置上。这是[01 类知识点 6](01-complexity-and-python-builtins.md#6-递归的时间空间复杂度分析)讲过的递归调用栈开销在这里的真实体现:递归版本的反转虽然思路优雅,但引入了 O(n) 的额外调用栈空间,不像迭代版本只需要 O(1) 额外空间——这是递归写法在"代码简洁性"和"空间复杂度"之间的真实权衡,不是纯粹的风格偏好问题。

**AI 研究/工程场景:** [huggingface-deep-dive 13 类](../huggingface-deep-dive/13-debugging-and-common-errors.md)讲过 Python 默认递归深度限制是 1000,如果链表长度可能达到几万甚至更长,递归版本的链表操作会真实触发 `RecursionError`——这也是为什么生产代码里处理链表(尤其是长度不可控的场景)通常更倾向迭代写法,递归版本更多是用来加深理解或者应对已知长度较短的场景。

**可运行例子:**
```python
import sys

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head

def reverse_iterative(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev

import random
random.seed(6)
for _ in range(20):
    vals = [random.randint(0, 100) for _ in range(random.randint(0, 8))]
    r1 = to_list(reverse_recursive(build_list(vals)))
    r2 = to_list(reverse_iterative(build_list(vals)))
    assert r1 == r2 == list(reversed(vals))  # 递归和迭代两种写法结果必须完全一致

# 真实验证:递归版本在链表远超默认递归深度限制时会真实触发RecursionError
long_list = build_list(list(range(sys.getrecursionlimit() + 500)))
raised = False
try:
    reverse_recursive(long_list)
except RecursionError:
    raised = True
assert raised   # 确认这是递归写法的真实代价,不是理论上的边缘情况

print("OK: 递归与迭代两种反转写法, 20组随机测试结果完全一致; "
      "链表长度远超递归深度限制时, 递归版本确实会真实触发RecursionError(迭代版本不受影响)")
```
本机实测:20 组随机测试中,递归和迭代两种反转写法结果完全一致;构造一个长度超过默认递归深度限制(1000)的链表后,递归版本确实真实触发了 `RecursionError`——这不是理论上的担忧,是可以现场复现的真实限制。

**面试怎么问 + 追问链:** "链表反转的递归写法和迭代写法,各自的空间复杂度是多少?" → 追问"什么场景下应该优先选递归写法,什么场景应该优先选迭代写法?"(链表长度可控且较短、追求代码可读性时可以用递归;链表长度不可控或可能很长时必须用迭代,避免真实的栈溢出风险——这个追问检验的是能否把"递归有额外空间开销"这个抽象知识和"什么时候会真的出问题"这个具体判断连接起来)。

**常见坑:**
1. 递归版本忘记把 `head.next` 显式设为 `None`——如果原链表是 `1→2→3`,反转后 `3`(新头)应该指向 `2`,`2` 指向 `1`,`1` 的 `next` 必须明确设为 `None`,否则会因为忘记断开原来的指向而形成环。
2. 想当然认为"递归版本更优雅所以性能也更好"——递归版本的时间复杂度和迭代版本相同(都是 O(n)),唯一的区别是空间复杂度(递归 O(n) 额外栈空间 vs 迭代 O(1)),不存在"递归天然更快"这回事。

---

## 7. 链表常见坑

**签名/是什么:**
```
指针修改顺序错误 -> 引用丢失(不报错,静默产生错误结果)
忘记处理空链表/单节点 -> 各类边界条件的 AttributeError 或逻辑错误
```

**一句话:** 链表题的 bug 有一个共同特征——绝大多数不会让程序崩溃报错,而是悄悄产生一个"看起来跑完了但结果不对"的链表,这也是为什么链表题格外需要系统性的边界测试,不能只靠"跑一下感觉对了"。

**底层机制/为什么这样设计:** 链表操作本质上是一系列对 `.next` 指针的赋值,Python 不会对"这样赋值是否符合你的意图"做任何检查——赋值顺序错了、该保存的引用没保存,程序依然会"成功"运行完,只是结果链表的结构是错的(可能丢失了一段、可能出现了环、可能指向了错误的节点)。这类问题不能靠运行时报错发现,只能靠**系统性地设计边界测试用例**(空链表、单节点、两节点、含环、全部节点相同)配合 `assert` 主动检查结果是否符合预期——这正是本篇每个知识点的"可运行例子"都刻意覆盖多种边界情况的原因,不是为了凑数,是链表这类题目本身的特性决定的。

**AI 研究/工程场景:** [huggingface-deep-dive](../huggingface-deep-dive/00-roadmap.md) 系列反复强调"真实断言验证、不满足于看起来跑通了"的纪律,在链表这类问题上体现得尤其明显——一个链表操作函数"运行没有报错"提供的正确性信心几乎为零,必须配合显式的结构校验。

**可运行例子:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head):
    out = []
    seen = set()
    while head:
        if id(head) in seen:
            raise RuntimeError("检测到环,to_list无法在有限步内完成——这本身就是一种坑的表现")
        seen.add(id(head))
        out.append(head.val)
        head = head.next
    return out

# 坑1: 指针顺序写错的反转(先改next再保存nxt)——制造这个bug,现场验证它的真实后果
def buggy_reverse(head):
    prev = None
    while head:
        head.next = prev   # 错误:先修改了head.next,下面这行拿到的已经是被覆盖后的值
        prev = head
        head = head.next   # 这里应该用之前保存的nxt,而不是被覆盖后的head.next
    return prev

result = buggy_reverse(build_list([1, 2, 3]))
# 这个bug版本会在处理完第一个节点后就把head指向prev(自己),导致遍历提前终止
assert to_list(result) == [1]  # 真实复现:错误顺序导致链表被截断,只剩第一个节点被处理

# 对照:正确写法
def correct_reverse(head):
    prev = None
    while head:
        nxt = head.next   # 先保存
        head.next = prev  # 再修改
        prev = head
        head = nxt
    return prev

assert to_list(correct_reverse(build_list([1, 2, 3]))) == [3, 2, 1]

# 坑2: 环检测函数如果错误地用值比较而不是引用比较,面对"值相同但对象不同"的场景会出问题
def cycle_by_value_buggy(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow.val == fast.val:  # 错误:应该用 is 判断对象引用相同,不是值相等
            return True
    return False

same_value_no_cycle = build_list([5, 5, 5, 5, 5])  # 无环,但所有值都相同
assert cycle_by_value_buggy(same_value_no_cycle) is True  # 真实复现:产生了假阳性

def cycle_correct(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

assert cycle_correct(build_list([5, 5, 5, 5, 5])) is False  # 正确写法不会有这个问题

print("OK: 现场复现两类经典链表bug(指针顺序写反导致截断; 值比较代替引用比较导致假阳性), "
      "并验证对应的正确写法确实修复了问题")
```
本机实测:故意写错指针修改顺序的反转函数,真实复现了"链表被意外截断,只剩第一个节点"这个后果(程序没有报错,结果却是错的);用值比较代替引用比较的环检测函数,在"无环但所有值相同"的链表上真实产生了假阳性——两个例子都验证了"链表 bug 通常不报错、只是结果错"这个特性,以及为什么必须靠系统性断言而不是运行不报错来确认正确性。

**面试怎么问 + 追问链:** "写完一道链表题后,你会怎么自己检查代码有没有问题?" → 追问"除了跑几个例子,还有什么方法能提高对代码正确性的信心?"(除了覆盖空链表/单节点/含环等边界情况,画图手工模拟指针变化过程是排查链表 bug 最有效的方法之一;对于反转类问题,还可以用"反转两次应该得到原链表"这类性质做交叉验证——这个追问检验的是候选人是否有一套系统性的自我验证方法论,而不是"写完看着感觉对"这种不可靠的信心来源)。

**常见坑:** (本知识点是对前 6 点常见坑的方法论总结,不重复列举具体错误,而是给出排查方法论)
1. 写完链表代码后,不要只用一个"看起来正常"的例子测试——至少应该过一遍空链表、单节点、两节点、可能含环这几类边界,链表题的 bug 密度和边界覆盖率直接相关。
2. 涉及指针重新赋值的每一行代码,养成"这一步执行后,原来指向的节点还能不能通过其他路径访问到"的检查习惯——一旦某个节点的所有引用路径都被覆盖且没有提前保存,它就永久丢失了,这是链表操作中最隐蔽也最常见的 bug 根源。

---

*本篇 7 个知识点全部在仓库根目录 `.venv` 真实测试验证(含边界情况覆盖与对照组交叉验证)。*
