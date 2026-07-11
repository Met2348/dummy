"""栈与队列 · 进阶补充（Part II）：不重讲框架，扩大"栈处理嵌套/模拟调用栈/设计题"覆盖面的 8 道题。"""
from __future__ import annotations


def eval_rpn(tokens: list[str]) -> int:
    """
    【题意】给定逆波兰表达式（后缀表达式）的 token 列表，求值并返回整数结果。除法结果
    向 0 截断。
    【思路】后缀表达式天然适合用栈求值：从左到右扫描，遇到数字就入栈；遇到运算符，说明
    "最近入栈的两个数"已经凑齐了一次运算的操作数，直接弹出这两个数做运算，再把结果压回
    栈——运算符右边的数在栈里永远比左边的数晚入栈、先出栈，栈的后进先出顺序恰好对应了
    "先算最近凑齐的这一对"这个后缀表达式的求值顺序，不需要额外解析表达式树。
    【复杂度】时间 O(n)，空间 O(n)（最坏情况全是数字，栈里存满 n/2 个操作数）。
    【易错点】1) 弹出顺序不能颠倒：先弹出的是右操作数 b，后弹出的才是左操作数 a，减法
    和除法必须算 a - b / a / b，写反成 b - a 结果会错；2) Python 的 `//` 对负数是向下
    取整（`-7 // 2 == -4`），题目要求向 0 截断，必须写成 `int(a / b)`；3) 判断"是运算符
    还是数字"要用精确匹配（`tok in {"+","-","*","/"}`），不能用 `tok[0] in "+-*/"` 之类
    的写法，否则会把负数 token（比如 "-2"）误判成减法运算符。
    """
    stack: list[int] = []
    operators = {"+", "-", "*", "/"}
    for tok in tokens:
        if tok in operators:
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(tok))
    return stack[-1]


def decode_string(s: str) -> str:
    """
    【题意】给定形如 "3[a2[c]]" 的编码字符串（数字表示紧跟其后中括号内容的重复次数，
    可以嵌套），解码返回展开后的字符串。
    【思路】遇到 '[' 时，"当前已经累积的重复次数 cur_num" 和 "当前已经拼好的字符串
    cur_str" 都必须先"冻结"保存起来，因为接下来要转去处理括号内部这个全新的子问题，
    子问题算完之后还要回来接着用外层的这两个状态——这正是"挂起当前现场、处理完内层
    再恢复"的通用范式，用两个栈（一个存数字、一个存字符串）分别保存现场。遇到 ']' 时，
    说明内层子串已经在 cur_str 里拼好了，弹出外层冻结的 (上一层数字, 上一层字符串)，
    用 `上一层字符串 + cur_str * 上一层数字` 把内层结果按重复次数展开后拼回外层，
    再继续外层的扫描。
    【复杂度】时间 O(n · k)（k 是展开后相对输入的膨胀倍数，最坏情况下嵌套重复会指数级
    放大输出长度），空间 O(n)（两个栈的深度等于括号最大嵌套深度）。
    【易错点】1) 数字可能不止一位（比如 "12[a]"），必须用 `cur_num = cur_num*10+int(ch)`
    持续累积，不能只读一位；2) 遇到 ']' 时用来拼接的顺序是"外层字符串在前、内层展开
    结果在后"（`prev_str + cur_str * prev_num`），拼反顺序会打乱字母出现的先后次序；
    3) 遇到 '[' 或 ']' 之后必须重置/更新 cur_num 和 cur_str，忘记重置 cur_num 会让
    下一层的重复次数被污染成"外层数字 * 10 + 新数字"这种错误的累积结果。
    """
    num_stack: list[int] = []
    str_stack: list[str] = []
    cur_num = 0
    cur_str = ""
    for ch in s:
        if ch.isdigit():
            cur_num = cur_num * 10 + int(ch)
        elif ch == "[":
            num_stack.append(cur_num)
            str_stack.append(cur_str)
            cur_num = 0
            cur_str = ""
        elif ch == "]":
            prev_num = num_stack.pop()
            prev_str = str_stack.pop()
            cur_str = prev_str + cur_str * prev_num
        else:
            cur_str += ch
    return cur_str


def validate_stack_sequences(pushed: list[int], popped: list[int]) -> bool:
    """
    【题意】给定两个长度相同、元素各不相同的数组 pushed（入栈顺序）和 popped（声称的
    出栈顺序），判断这个出栈顺序是否可能由某种合法的栈操作序列产生。
    【思路】不需要枚举所有可能的操作序列，只需要老老实实"模拟"一遍：用一个辅助栈，
    严格按 pushed 的顺序依次入栈；每入栈一个元素后，就贪心地检查栈顶是否等于
    popped 里下一个待弹出的值——只要相等就立刻弹出（因为如果现在不弹，后面也没有
    更好的时机弹出这个栈顶元素，贪心弹出不会让后续变得更差），直到栈顶不再匹配
    popped 的下一个值，才继续入栈下一个 pushed 元素。全部入栈操作结束后，如果辅助
    栈恰好被弹空，说明这个出栈顺序合法。
    【复杂度】时间 O(n)（虽然有一层 while 嵌套在 for 里，但每个元素总共只会入栈一次、
    出栈一次），空间 O(n)（辅助栈）。
    【易错点】1) 弹出判断必须写成 while 循环而不是 if，因为一次入栈可能连续触发多次
    弹出（比如 pushed=[1,2,3], popped=[3,2,1] 时入栈 3 后要连续弹出 3、2、1）；
    2) while 条件里要同时检查 `j < len(popped)`，防止 popped 指针越界；3) 最终的
    判断依据是"辅助栈是否清空"，而不是"j 是否等于 len(popped)"（虽然两者在这题里
    等价，但用栈是否为空更直接对应"是否所有入栈的元素都找到了匹配的弹出时机"）。
    """
    stack: list[int] = []
    j = 0
    for x in pushed:
        stack.append(x)
        while stack and j < len(popped) and stack[-1] == popped[j]:
            stack.pop()
            j += 1
    return not stack


def _is_balanced_parens(s: str) -> bool:
    """辅助函数：校验字符串里的圆括号是否合法匹配（忽略非括号字符）。"""
    balance = 0
    for ch in s:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
            if balance < 0:
                return False
    return balance == 0


def min_remove_to_make_valid(s: str) -> str:
    """
    【题意】给定包含小写字母和圆括号的字符串，删除最少数量的括号，使剩余字符串的括号
    合法匹配，返回任意一个合法结果。
    【思路】一次扫描分两类"待删除"的右括号和左括号：遇到 '(' 就把它的下标压栈（表示
    "欠着一个待匹配的右括号"）；遇到 ')' 时，如果栈不空就弹栈（说明找到了配对的左括号，
    这一对都合法，不删除），如果栈是空的，说明这个右括号根本没有左括号能配它，把它的
    下标记录进待删除集合。扫描完整个字符串后，栈里如果还剩下下标，说明这些左括号从头
    到尾都没等到配对的右括号，同样要标记删除。最后按"下标是否在待删除集合里"过滤一遍
    原字符串即可，不需要真的做插入/移动操作。
    【复杂度】时间 O(n)，空间 O(n)（栈和待删除集合最坏情况各存 O(n) 个下标）。
    【易错点】1) 扫描结束后不能只处理"多余的右括号"，栈里剩下的下标（多余的左括号）
    也必须并入待删除集合，否则会漏删 "((" 这种只有左括号没有右括号的情况；2) 删除
    时不能一边遍历一边直接从字符串/列表里删字符（下标会实时错位），要先收集好全部待
    删除下标，最后统一用"下标是否命中"过滤一遍；3) 这题允许多种合法答案，只要结果
    括号合法且删除字符数最少即可，测试时应该用一个独立的括号合法性校验函数去验证，
    而不是死抠字符串是否和某一个特定答案逐字符相等（不过对本题给定的官方样例，两者
    恰好是唯一确定的）。
    """
    to_remove: set[int] = set()
    stack: list[int] = []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                to_remove.add(i)
    to_remove.update(stack)
    return "".join(ch for i, ch in enumerate(s) if i not in to_remove)


def exclusive_time(n: int, logs: list[str]) -> list[int]:
    """
    【题意】单线程 CPU 顺序执行 n 个函数（可以嵌套调用），logs 是形如
    "函数id:start|end:时间戳" 的日志（按时间戳升序），求每个函数的独占执行时间
    （不包含被它调用的子函数占用的时间）。
    【思路】函数调用的嵌套关系天然对应函数调用栈：栈顶永远是"当前正在执行的那个函数"。
    维护 prev_time（上一条日志发生的时间点）：遇到 "start" 日志，先把这段时间
    （从 prev_time 到当前时间戳）计给当前栈顶函数（如果栈非空——它是被新函数打断
    之前一直在跑的那个），再把新函数压栈；遇到 "end" 日志，把从 prev_time 到
    "当前时间戳 + 1"（因为时间戳是包含结束那一刻的，独占时间要多算一个单位）这段
    时间计给栈顶函数（此时栈顶正是即将结束的这个函数自己），再弹栈。
    【复杂度】时间 O(m)（m 是日志条数），空间 O(n)（调用栈深度最多 n）。
    【易错点】1) "end" 日志的时间戳是"闭区间"的（比如 0 到 5 表示占用了 0,1,2,3,4,5
    共 6 个单位时间），所以结算时间时要用 `time - prev_time + 1`，比 "start" 日志少
    的那个 +1 很容易漏掉；2) "end" 日志处理完之后，`prev_time` 要更新成
    `time + 1` 而不是 `time`，否则外层函数恢复执行后会把已经计给内层函数的这一个
    单位时间重复计算一次；3) "start" 日志触发的结算，是把时间计给"当前栈顶"（也就是
    被打断的外层函数），而不是即将入栈的新函数本身，两者的结算对象容易搞反。
    """
    result = [0] * n
    stack: list[int] = []
    prev_time = 0
    for log in logs:
        fid_str, typ, time_str = log.split(":")
        fid, time = int(fid_str), int(time_str)
        if typ == "start":
            if stack:
                result[stack[-1]] += time - prev_time
            stack.append(fid)
            prev_time = time
        else:
            result[stack[-1]] += time - prev_time + 1
            stack.pop()
            prev_time = time + 1
    return result


class BrowserHistory:
    """
    【题意】设计浏览器历史记录：`visit(url)` 访问新页面（并清空所有"前进"历史）、
    `back(steps)` 后退最多 steps 步并返回当前页面、`forward(steps)` 前进最多 steps
    步并返回当前页面，退后/前进都不能越过历史记录的两端。
    【思路】浏览器的"后退/前进"关系本质是一条线性历史 + 一个"当前所在位置"的指针，
    并不需要真的用两个栈分别模拟"后退栈"和"前进栈"——因为 visit 会清空前进历史，
    这意味着"前进方向"的内容其实一直隐式存在于"当前指针右边尚未被截断的历史"里，
    直到下一次 visit 才需要真正丢弃。所以用一个列表存整条历史、一个指针 cur 指向
    当前页面：back/forward 只需要移动指针（并 clamp 在 [0, len-1] 范围内）；
    visit 才需要"截断指针右边的旧历史，再追加新页面"，这一步就等价于丢弃了所有
    "前进"的可能性。
    【复杂度】push（这里是 visit）均摊 O(1)（截断操作最坏 O(n)，但均摊到每个曾经
    visit 过的元素上是 O(1)）；back/forward O(1)；空间 O(n)。
    【易错点】1) visit 必须先截断 `history[:cur+1]` 再 append，如果忘记截断，
    之前后退过又重新 visit 时，旧的"前进历史"会错误地保留在列表尾部；2) back/forward
    的步数可能超过实际能移动的范围，必须用 `max(0, ...)` / `min(len-1, ...)` 做
    clamp，而不是直接加减（会越界）；3) back 和 forward 都要返回移动之后的当前
    页面（而不是移动之前的），指针先更新、再用更新后的指针取值。
    """

    def __init__(self, homepage: str) -> None:
        self.history: list[str] = [homepage]
        self.cur = 0

    def visit(self, url: str) -> None:
        self.history = self.history[: self.cur + 1]
        self.history.append(url)
        self.cur += 1

    def back(self, steps: int) -> str:
        self.cur = max(0, self.cur - steps)
        return self.history[self.cur]

    def forward(self, steps: int) -> str:
        self.cur = min(len(self.history) - 1, self.cur + steps)
        return self.history[self.cur]


class CustomStack:
    """
    【题意】设计一个最多容纳 max_size 个元素的栈：`push(x)` 超过容量则忽略；
    `pop()` 弹出栈顶，空栈返回 -1；`increment(k, val)` 把栈底最多 k 个元素的值
    都加上 val（如果栈里元素不足 k 个，就全部加上）。
    【思路】如果 increment 真的遍历栈底 k 个元素逐一相加，最坏情况下每次
    increment 都是 O(size)，多次 increment 叠加起来会退化成 O(size^2)。关键
    优化是"延迟结算"：额外维护一个和主栈等长的差分数组 inc，`inc[i]` 表示
    "主栈第 i 个位置（以及它下面所有位置，也就是 [0, i] 这一段）将来 pop 出去时，
    应该额外加上的累计增量"，而不是立刻把增量摊到每一个元素上。increment(k, val)
    这一步就只需要把 val 加到 `inc[min(k, size) - 1]` 这一个位置（因为它代表
    "影响到这个下标为止的所有更靠底的元素"），是 O(1)。pop() 的时候，栈顶元素
    的真实值 = 原值 + inc[-1]；同时把 inc[-1] 这份"欠给下面元素的增量"下推一层，
    累加到 inc[-2] 上（因为下面的元素以后弹出时也应该享受这份增量），这样每个
    元素的增量最终都会在它被弹出的那一刻被正确结算，整个过程均摊 O(1)。
    【复杂度】push/pop/increment 均为 O(1)（均摊）；空间 O(max_size)。
    【易错点】1) increment 作用的下标是 `min(k, len(stack)) - 1`，而不是
    `k - 1`——如果当前元素个数比 k 少，只能影响到栈顶这个位置，不能越界写到
    不存在的下标；2) pop 时必须先把 `inc[-1]` 累加进 `inc[-2]`（如果还有更底下
    的元素）再弹出两个数组的最后一位，顺序颠倒或者忘记下推会让底下元素的增量
    丢失；3) push 超过容量时必须真正忽略（既不追加到 stack，也不追加到 inc），
    两个数组必须时刻保持等长，否则后续的下标对应关系全部错位。
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.stack: list[int] = []
        self.inc: list[int] = []

    def push(self, x: int) -> None:
        if len(self.stack) < self.max_size:
            self.stack.append(x)
            self.inc.append(0)

    def pop(self) -> int:
        if not self.stack:
            return -1
        if len(self.inc) >= 2:
            self.inc[-2] += self.inc[-1]
        return self.stack.pop() + self.inc.pop()

    def increment(self, k: int, val: int) -> None:
        i = min(k, len(self.stack)) - 1
        if i >= 0:
            self.inc[i] += val


def cal_points(operations: list[str]) -> int:
    """
    【题意】给定一串棒球比赛记录 operations：数字表示这一轮得分；"+" 表示这一轮得分
    是前两轮有效得分之和；"D" 表示这一轮得分是前一轮有效得分的两倍；"C" 表示前一轮
    的记录是无效的，要撤销（移除）。求所有有效得分之和。
    【思路】"+"、"D"、"C" 都只依赖"最近的一到两轮有效得分"，而"撤销"操作又要求
    移除的必须是"最近"的那一条记录——这正是栈的典型应用场景：只维护有效得分，"C"
    直接弹栈（撤销最近一条），"+" 和 "D" 只需要查看（必要时结合）栈顶的一两个元素，
    算出新分数后压栈，不需要额外的下标或链表结构。
    【复杂度】时间 O(n)，空间 O(n)（栈里最多存 n 条有效记录）。
    【易错点】1) "+" 需要栈顶两个元素相加后作为新记录压栈，而不是替换掉原来那两个——
    原来那两条记录仍然是各自独立有效的得分，新算出的这一条只是"追加"的第三条；
    2) "C" 只是弹栈丢弃，不产生新的得分，容易误写成"弹栈后再做某种运算"；3) 数字
    字符串可能带负号（比如 "-2"），要用 `int(op)` 直接转换而不是先判断是否以数字
    字符开头，避免负号被误处理。
    """
    stack: list[int] = []
    for op in operations:
        if op == "+":
            stack.append(stack[-1] + stack[-2])
        elif op == "D":
            stack.append(stack[-1] * 2)
        elif op == "C":
            stack.pop()
        else:
            stack.append(int(op))
    return sum(stack)


def _self_test() -> None:
    assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
    assert eval_rpn(["4", "13", "5", "/", "+"]) == 6

    assert decode_string("3[a]2[bc]") == "aaabcbc"
    assert decode_string("3[a2[c]]") == "accaccacc"
    assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"

    assert validate_stack_sequences([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]) is True
    assert validate_stack_sequences([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]) is False

    r1 = min_remove_to_make_valid("lee(t(c)o)de)")
    assert _is_balanced_parens(r1) and r1 == "lee(t(c)o)de"
    r2 = min_remove_to_make_valid("a)b(c)d")
    assert _is_balanced_parens(r2) and r2 == "ab(c)d"
    r3 = min_remove_to_make_valid("))((")
    assert _is_balanced_parens(r3) and r3 == ""

    assert exclusive_time(
        2, ["0:start:0", "1:start:2", "1:end:5", "0:end:6"]
    ) == [3, 4]

    bh = BrowserHistory("leetcode.com")
    bh.visit("google.com")
    bh.visit("facebook.com")
    bh.visit("youtube.com")
    assert bh.back(1) == "facebook.com"
    assert bh.back(1) == "google.com"
    assert bh.forward(1) == "facebook.com"
    bh.visit("linkedin.com")
    assert bh.forward(2) == "linkedin.com"
    assert bh.back(2) == "google.com"
    assert bh.back(7) == "leetcode.com"

    st = CustomStack(3)
    st.push(1)
    st.push(2)
    assert st.pop() == 2
    st.push(2)
    st.push(3)
    st.push(4)  # 4 超容量，忽略
    st.increment(5, 100)
    st.increment(2, 100)
    assert st.pop() == 103
    assert st.pop() == 202
    assert st.pop() == 201
    assert st.pop() == -1

    assert cal_points(["5", "2", "C", "D", "+"]) == 30
    assert cal_points(["5", "-2", "4", "C", "D", "9", "+", "+"]) == 27

    print(
        "[PASS] p06_stack_queue_ii: 8 题"
        "（逆波兰表达式求值/字符串解码/验证栈序列/移除无效的括号/"
        "函数的独占时间/设计浏览器历史记录/设计支持增量操作的栈/棒球比赛）"
        "全部通过"
    )


if __name__ == "__main__":
    _self_test()
