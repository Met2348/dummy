"""LeetCode 分类 06·栈与队列：栈是处理"嵌套/最近匹配"结构（括号、表达式、单调性）的天然工具。"""
from __future__ import annotations


def is_valid(s: str) -> bool:
    """
    【题意】给定只包含 '(' ')' '{' '}' '[' ']' 的字符串，判断括号是否"有效匹配"：
    每一个左括号都必须有同类型的右括号，并且按正确的嵌套顺序闭合。
    【思路】栈（后进先出）天然表达了"最近打开的括号必须最先被关闭"这条规则：遇到左括号
    就入栈（记录"欠着一个对应右括号"）；遇到右括号时，检查栈顶（也就是最近一个还没
    闭合的左括号）是不是它的同类型——如果是，说明这一次配对成功，把栈顶弹出（这笔"债"
    还上了）；如果类型不匹配，或者该出现右括号时栈已经空了（说明右括号没有对应的左括号），
    直接判定无效。整个字符串扫描完之后，栈必须正好清空，否则说明还有左括号没等到它的
    右括号（比如 "(("）。
    【复杂度】时间 O(n)，空间 O(n)（最坏情况全是左括号，栈里存满 n 个）。
    【易错点】1) 用一个哈希表把"右括号 -> 对应左括号"的映射列好，扫描到右括号时直接
    查表比较，不要写一长串 if/elif 手动枚举六种字符的组合，容易漏写或写反；
    2) 扫描到右括号但此时栈为空，必须立刻返回 False，不能对空栈调用 pop/取栈顶
    （会直接抛异常或者需要额外判断）；3) 扫描完整个字符串后必须再检查一次栈是否为空——
    像 "([" 这种只有左括号、没有任何右括号来匹配的输入，扫描过程中永远不会触发 False，
    必须在最后补一次"栈是否清空"的检查才能正确判定无效。
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:
            stack.append(ch)
    return not stack


class MyQueue:
    """
    【题意】只用栈（后进先出）这一种数据结构，实现队列（先进先出）的四个接口：
    push(x) 入队、pop() 弹出并返回队首元素、peek() 查看队首元素但不弹出、
    empty() 判断队列是否为空。
    【思路】用两个栈分工协作：in_stack 只负责接收新元素（push 直接压入 in_stack，
    不做任何额外处理）；out_stack 负责提供"先进先出"的读取顺序。当需要 pop/peek
    而 out_stack 恰好是空的，就把 in_stack 里的元素整体倒进 out_stack——栈本身是
    后进先出，倒了两次（一次在 in_stack 里压入，一次倒进 out_stack）之后顺序正好
    翻转回先进先出。之后只要 out_stack 还有元素，就一直从它这里 pop/peek，完全不需要
    重新倒。这样每个元素一生只会被"倒腾"这一次操作最多处理一次（从 in_stack 倒到
    out_stack），单次操作看起来最坏是 O(n)，但均摊到每个元素上是 O(1)。
    【复杂度】push: O(1)；pop/peek：最坏 O(n)（触发倒栈时），均摊 O(1)；空间 O(n)。
    【易错点】1) 每次 pop/peek 都无条件执行"倒栈"，会把 out_stack 里已经排好顺序、
    还没被消费的元素和 in_stack 里新来的元素搅在一起，破坏先进先出的顺序——只应该在
    out_stack 为空时才触发倒栈；2) 倒栈这一步必须把 in_stack 完全倒空（while 循环
    到底），不能只倒一部分，否则两个栈里的相对顺序会乱。
    """

    def __init__(self) -> None:
        self.in_stack: list[int] = []
        self.out_stack: list[int] = []

    def _shift(self) -> None:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        self._shift()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._shift()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack


class MinStack:
    """
    【题意】设计一个栈，除了普通的 push(val)、pop()、top()（查看栈顶）之外，还要支持
    get_min()（获取栈中当前所有元素的最小值），并且要求这四个操作都是 O(1)。
    【思路】难点在于 get_min 要做到 O(1)，但栈里的最小值会随着 push/pop 动态变化
    （比如当前最小值被 pop 掉之后，"曾经第二小"的那个值必须能立刻顶上来）。解法是
    额外维护一个和主栈严格同步增减的"最小值栈" min_stack：每次 push(val) 时，
    同时把 min(val, 当前已知最小值) 压入 min_stack（如果 min_stack 还是空的，
    就直接压 val 本身）；每次主栈 pop 时，min_stack 也同步 pop 一次。这样 min_stack
    的栈顶，永远等于"主栈从栈底到当前栈顶为止这一段的最小值"——因为它在每个时刻都记录了
    "历史最小值"，而不是只存一个会被 pop 破坏的全局变量。
    【复杂度】push/pop/top/get_min 均为 O(1)；空间 O(n)（辅助栈和主栈始终等长）。
    【易错点】1) 如果只用一个全局的 min 变量而不是"最小值栈"，当当前最小值恰好被
    pop 掉之后，就再也找不回"次小值"是多少了；2) push 和 pop 必须让两个栈严格
    同步增减（每次 push 都往两个栈各加一个元素，每次 pop 都各弹一个），漏了同步会导致
    两个栈长度不一致，get_min 读到过期的值。
    """

    def __init__(self) -> None:
        self.stack: list[int] = []
        self.min_stack: list[int] = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        cur_min = val if not self.min_stack else min(val, self.min_stack[-1])
        self.min_stack.append(cur_min)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.min_stack[-1]


def calculate(s: str) -> int:
    """
    【题意】给定一个只含非负整数、'+' '-' '*' '/'（整数除法结果向 0 截断）和空格的
    表达式字符串（不含括号），按四则运算的优先级（先乘除、后加减）计算出结果。
    【思路】乘除法的优先级比加减法高，但这道题并不需要真的建一棵表达式树去处理优先级——
    因为乘除法只和"紧挨着它前面的那一个数"发生关系，不会影响更前面已经确定下来的数。
    用一个栈存"已经确定符号、不会再被后续运算修改的数字"：扫描时维护一个 prev_sign
    （上一个遇到的运算符，初始视为 '+'，表示"当前数字之前的运算符"）和一个正在累积的
    数字 num；每当遇到一个新的运算符（或者已经扫到字符串末尾），就"结算"刚刚累积完的
    num：如果 prev_sign 是 '+'，把 num 原样入栈；是 '-'，把 -num 入栈（负号在入栈
    这一步就处理掉了，后面只需要把栈里所有数加起来）；是 '*' 或 '/'，则弹出栈顶
    （也就是"紧邻的前一个数"），和 num 做乘法/除法后把结果重新压回栈——这一步就是
    "只操作栈顶就地修正了优先级"，不需要真的构建表达式树或做两次扫描。全部字符扫描完后，
    栈里剩下的都是待相加的数（正负号已经在各自入栈时处理好了），把栈内所有元素求和就是
    最终答案。
    【复杂度】时间 O(n)，空间 O(n)（最坏情况栈里存 n/2 个数，比如全是加减法）。
    【易错点】1) 整数除法必须"向 0 截断"（truncate toward zero），而 Python 的
    // 对负数是向下取整（比如 -7 // 2 == -4，但题目要求的截断结果是 -3），必须写成
    int(a / b) 或者手动判断符号后再做整除；2) "结算"的触发时机是"遇到新运算符
    或者到达字符串末尾"两种情况都要触发——如果只在遇到运算符时结算，字符串最后一个
    数字永远不会被结算入栈，这里通过 i == n - 1 这个条件把"到达末尾"也当成一次
    触发信号来处理。
    """
    stack: list[int] = []
    num = 0
    prev_sign = "+"
    n = len(s)
    for i, ch in enumerate(s):
        if ch.isdigit():
            num = num * 10 + int(ch)
        if ch in "+-*/" or i == n - 1:
            if prev_sign == "+":
                stack.append(num)
            elif prev_sign == "-":
                stack.append(-num)
            elif prev_sign == "*":
                stack.append(stack.pop() * num)
            else:  # prev_sign == "/"
                stack.append(int(stack.pop() / num))
            prev_sign = ch
            num = 0
    return sum(stack)


def calculate_with_parens(s: str) -> int:
    """
    【题意】给定一个只含非负整数、'+' '-' '(' ')' 和空格的表达式字符串（没有乘除法，
    但含有括号嵌套），计算出结果。
    【思路】括号嵌套的本质是"计算到一半时，突然必须先算完一个子表达式，再拿子表达式的
    结果去参与外层的计算"——这正是栈擅长处理的"挂起当前状态、先处理内层、再恢复外层
    状态"模式。用 result 记录"当前这一层已经累加到的和"，sign 记录"接下来读到的这个
    数应该以什么符号加到 result 上"。遇到 '(' 时，把当前层的 (result, sign) 打包
    压栈"冻结"起来，然后把 result 和 sign 重置为初始状态，转而去计算括号内部的子
    表达式；遇到 ')' 时，说明内层子表达式已经算完并存在当前的 result 里了，这时候
    弹出栈顶被冻结的 (prev_result, prev_sign)，做
    result = prev_result + prev_sign * result，也就是把内层算出来的结果，
    按外层原本准备用的符号"揉回"外层的累加和里，再继续外层的计算。
    【复杂度】时间 O(n)，空间 O(n)（栈的深度等于括号的最大嵌套深度）。
    【易错点】1) 弹栈后合并的顺序容易写反——正确顺序是"外层已有的 result + 外层的
    sign * 内层刚算完的 result"，而不是反过来用内层结果做基准；2) 数字可能不止一位，
    扫描到数字字符时要持续累积到 num 里，遇到运算符、左括号或右括号（也就是非数字
    字符）才把 num 结算进 result；3) 遇到 ')' 时，如果内层最后一个数字还没有被结算
    进 result（比如恰好在读完一个数字之后就直接遇到 ')'），必须先把这个 num 结算
    进 result，再弹栈做外层合并——同理，整个字符串扫描完之后也要补一次结算，否则
    最外层最后一个数字会被漏加。
    """
    stack: list[tuple[int, int]] = []
    result = 0
    num = 0
    sign = 1
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch in "+-":
            result += sign * num
            num = 0
            sign = 1 if ch == "+" else -1
        elif ch == "(":
            stack.append((result, sign))
            result = 0
            sign = 1
        elif ch == ")":
            result += sign * num
            num = 0
            prev_result, prev_sign = stack.pop()
            result = prev_result + prev_sign * result
    result += sign * num
    return result


def _self_test() -> None:
    assert is_valid("()") is True
    assert is_valid("()[]{}") is True
    assert is_valid("(]") is False
    assert is_valid("([)]") is False
    assert is_valid("{[]}") is True

    q = MyQueue()
    q.push(1)
    q.push(2)
    assert q.peek() == 1
    assert q.pop() == 1
    assert q.empty() is False

    ms = MinStack()
    ms.push(-2)
    ms.push(0)
    ms.push(-3)
    assert ms.get_min() == -3
    ms.pop()
    assert ms.top() == 0
    assert ms.get_min() == -2

    assert calculate("3+2*2") == 7
    assert calculate(" 3/2 ") == 1
    assert calculate(" 3+5 / 2 ") == 5

    assert calculate_with_parens("1 + 1") == 2
    assert calculate_with_parens(" 2-1 + 2 ") == 3
    assert calculate_with_parens("(1+(4+5+2)-3)+(6+8)") == 23

    print(
        "[PASS] p06_stack_queue: 5/5 题通过 "
        "(有效的括号/用栈实现队列/最小栈/基本计算器II/基本计算器)"
    )


if __name__ == "__main__":
    _self_test()
