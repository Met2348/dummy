# 05 · 栈与队列,含设计题(Stacks, Queues and Design Problems)

> 总览见 [00-roadmap.md](00-roadmap.md)。栈和队列本身很简单(先进后出/先进先出),但"单调栈/单调队列"这类变体技巧,以及 LRU/LFU 这类"手写一个真实可用的数据结构"的设计题,是终面里出现频率极高的题材——本类把简单结构和复杂应用之间的落差讲透。

---

## 1. 栈的基础应用:括号匹配与表达式求值

**签名/是什么:**
```
栈：后进先出(LIFO)，只支持在同一端 push/pop
括号匹配：遇到左括号入栈，遇到右括号检查栈顶是否是对应的左括号
```

**一句话:** 栈天然适合处理"最近发生的事情要最先被处理/撤销"这类场景——括号匹配的本质是"最近打开的括号必须最先被闭合",逆波兰表达式求值的本质是"最近入栈的两个操作数参与当前运算",两者都直接对应栈的 LIFO 特性。

**底层机制/为什么这样设计:** 括号匹配用栈的正确性来自一个直接的对应关系:遇到右括号时,它能且只能匹配"最近一个还没被闭合的左括号"——这正是栈顶元素;如果这个假设不成立(比如括号顺序错乱),栈顶元素和当前右括号不匹配,直接可以判定整体不合法。逆波兰(后缀)表达式求值同理:运算符总是作用在"最近两个还没被使用的操作数"上,用栈暂存操作数,遇到运算符就弹出栈顶两个参与运算,再把结果压回栈,不需要额外的括号或优先级判断——这也是逆波兰表达式在早期计算器/编译器里被广泛使用的原因,求值逻辑天然适合用栈实现且没有歧义。

**AI 研究/工程场景:** [huggingface-deep-dive 01 类](../huggingface-deep-dive/01-tokenizer-internals.md)讲过的 chat template 渲染,处理嵌套的条件语法结构(比如 Jinja 模板里的 `{% if %}...{% endif %}` 配对)时,模板引擎内部做语法校验用的就是这里讲的括号匹配思路。

**可运行例子:**
```python
def valid_parens(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack

assert valid_parens("()[]{}") is True
assert valid_parens("(]") is False
assert valid_parens("((") is False        # 全部未闭合
assert valid_parens("") is True             # 空字符串合法
assert valid_parens("([)]") is False       # 交叉嵌套不合法(顺序错乱)
assert valid_parens("{[]}") is True         # 正确嵌套

def eval_rpn(tokens):
    stack = []
    for t in tokens:
        if t in "+-*/":
            b, a = stack.pop(), stack.pop()
            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            else: stack.append(int(a / b))   # 向零截断,匹配大多数语言的整数除法约定
        else:
            stack.append(int(t))
    return stack[-1]

assert eval_rpn(["2", "1", "+", "3", "*"]) == 9    # (2+1)*3
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6    # 4+(13/5)=4+2=6
assert eval_rpn(["5"]) == 5                            # 单个数字,不涉及运算

print("OK: 括号匹配在空串/未闭合/交叉嵌套等边界情况下全部正确; 逆波兰表达式求值验证通过")
```
本机实测:括号匹配在空字符串、全部未闭合、交叉嵌套(顺序错乱)这几类边界情况下均正确;逆波兰表达式求值在含除法运算、单个数字这两类情况下均正确。

**面试怎么问 + 追问链:** "为什么括号匹配问题天然适合用栈解决?" → 追问"如果字符串里同时存在括号和其他普通字符,怎么修改算法?"(遍历时直接跳过非括号字符,不影响栈的逻辑——这个追问检验的是能否把"括号匹配"这个抽象逻辑和"处理真实字符串里混杂其他字符"这个具体工程细节区分开,不少候选人会把两者耦合在一起写出更复杂的代码)。

**常见坑:**
1. 判断右括号匹配时,忘记先检查栈是否为空就直接访问栈顶——空栈时遇到右括号应该直接判定不合法,不能假设栈里一定有元素。
2. 遍历结束后忘记检查栈是否为空——即使遍历过程中所有右括号都成功匹配,如果栈里还剩下未闭合的左括号,整体依然不合法。

---

## 2. 单调栈:下一个更大元素与柱状图最大矩形

**签名/是什么:**
```
单调栈：栈内元素始终保持单调递增或递减，
新元素入栈前，先弹出所有破坏单调性的栈顶元素
```

**一句话:** 单调栈用来高效解决"对每个元素,找左边/右边第一个比它大/小的元素"这类问题——利用"被弹出的元素,恰好是当前元素的答案"这个性质,整个数组只需要一次遍历,每个元素最多入栈出栈各一次,总复杂度 O(n)。

**底层机制/为什么这样设计:** 以"下一个更大元素"为例,维护一个从栈底到栈顶单调**递减**的栈:遍历到新元素 `x` 时,只要栈顶元素比 `x` 小,就把栈顶弹出——**弹出的这个元素,`x` 正是它"右边第一个更大的元素"**(因为在弹出之前,栈顶和 `x` 之间的所有元素都已经被处理过,`x` 是第一个让它满足"找到了"条件的新元素)。这个"该元素被弹出的那一刻,当前正在处理的新元素就是它的答案"的性质,是单调栈能省去嵌套循环的核心原因——朴素解法需要对每个元素都往右扫描到找到答案为止(最坏 O(n²)),单调栈保证每个元素只会被压入和弹出各一次。柱状图最大矩形问题用的是同一个机制:对每根柱子,找到"左边第一个比它矮的柱子"和"右边第一个比它矮的柱子",这两个边界之间就是以当前柱子高度为限制的最大矩形宽度。

**AI 研究/工程场景:** [huggingface-deep-dive 09 类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过的训练 loss 曲线分析,如果要找"loss 第一次低于当前值的位置"这类"下一个更小/更大元素"问题(比如判断训练在哪一步开始稳定收敛),单调栈是比对每个点都重新扫描一遍更高效的做法。

**可运行例子:**
```python
def next_greater(nums):
    res = [-1] * len(nums)
    stack = []   # 存下标，保持nums[stack]从栈底到栈顶单调递减
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            res[stack.pop()] = x
        stack.append(i)
    return res

assert next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
assert next_greater([]) == []                    # 空数组
assert next_greater([5]) == [-1]                  # 单元素,不存在更大的
assert next_greater([1, 1, 1]) == [-1, -1, -1]    # 全部相同,不存在严格更大的
assert next_greater([1, 2, 3, 4]) == [2, 3, 4, -1]  # 严格递增,每个的答案就是下一个

def largest_rectangle(heights):
    stack = []
    max_area = 0
    for i, h in enumerate(heights + [0]):   # 末尾补0,确保所有柱子最终都会被弹出结算
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area

assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10   # 5和6两根柱子组成的矩形
assert largest_rectangle([]) == 0                       # 空输入
assert largest_rectangle([5]) == 5                       # 单根柱子
assert largest_rectangle([2, 2, 2]) == 6                 # 等高柱子,整体组成矩形

# 交叉验证:暴力解法(枚举每根柱子作为矩形高度,向两边扩展)结果必须一致
def brute_largest_rectangle(heights):
    max_area = 0
    for i, h in enumerate(heights):
        left = i
        while left > 0 and heights[left-1] >= h:
            left -= 1
        right = i
        while right < len(heights)-1 and heights[right+1] >= h:
            right += 1
        max_area = max(max_area, h * (right - left + 1))
    return max_area

import random
random.seed(7)
for _ in range(20):
    hs = [random.randint(0, 10) for _ in range(random.randint(0, 10))]
    assert largest_rectangle(hs) == brute_largest_rectangle(hs)

print("OK: 单调栈解法在边界情况下全部正确, 20组随机测试与暴力解法结果完全一致")
```
本机实测:下一个更大元素和柱状图最大矩形在空输入、单元素、全部相同、严格递增/等高这几类边界情况下均正确;20 组随机测试中,单调栈解法与暴力解法结果完全一致。

**面试怎么问 + 追问链:** "为什么单调栈能把'找下一个更大元素'的复杂度做到 O(n)?" → 追问"每个元素最多入栈出栈各一次,这个结论是怎么保证的?"(每个下标只会被 `stack.append(i)` 压入一次;它只有在被后续某个更大的元素弹出时才会离开栈,一旦弹出就不会再被压入——这保证了总的入栈/出栈操作次数都不超过 n 次,均摊到每个元素是 O(1),这正是[01类知识点2](01-complexity-and-python-builtins.md#2-均摊复杂度分析amortized-analysis)均摊复杂度分析在这里的具体应用)。

**常见坑:**
1. 单调栈存的是**下标**还是**值**没想清楚——大多数场景需要同时知道值和位置(比如计算矩形宽度需要下标),只存值会在需要位置信息时无法使用。
2. 柱状图最大矩形问题忘记在末尾补一个高度为 0 的哨兵柱子——如果不补,栈里剩余的柱子不会被结算,会漏掉包含最后几根柱子的矩形方案。

---

## 3. 单调队列:滑动窗口最大值

**签名/是什么:**
```
collections.deque 维护一个下标队列，
队列从队首到队尾对应的值单调递减(队首始终是当前窗口最大值)
```

**一句话:** 单调队列解决"滑动窗口内的最大值"问题——和单调栈解决"下一个更大元素"是同一种思想(维护单调性、及时淘汰不可能是答案的候选),区别在于单调队列还需要处理"候选元素超出窗口范围就要被移除"这个额外维度。

**底层机制/为什么这样设计:** 维护一个从队首到队尾单调递减的下标队列:新元素进入窗口时,从队尾开始把所有"比新元素小"的候选弹出(它们不可能再成为任何未来窗口的最大值,因为新元素比它们大且比它们晚离开窗口);队首元素如果已经滑出当前窗口范围,也要从队首移除。这样队首始终是当前窗口内的最大值——整个过程每个下标只会入队出队各一次,是均摊 O(1) 每步,总复杂度 O(n),比对每个窗口都重新扫描一遍求最大值(O(n·k))快得多。

**AI 研究/工程场景:** [huggingface-deep-dive 12 类](../huggingface-deep-dive/12-inference-optimization.md)讲过的 KV-cache 滑动窗口注意力机制,如果需要维护"最近 k 个位置里 attention score 最大的那个",单调队列正是支持这类"滑动窗口极值查询"的标准算法工具。

**可运行例子:**
```python
from collections import deque

def sliding_window_max(nums, k):
    dq = deque()   # 存下标,nums[dq]从队首到队尾单调递减
    res = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:      # 队首元素已经滑出窗口范围
            dq.popleft()
        if i >= k - 1:            # 窗口已经形成,开始记录结果
            res.append(nums[dq[0]])
    return res

assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert sliding_window_max([1], 1) == [1]                    # 单元素,窗口大小1
assert sliding_window_max([1, 2, 3], 3) == [3]              # 窗口等于数组长度
assert sliding_window_max([9, 8, 7, 6], 2) == [9, 8, 7]      # 严格递减,每个窗口最大值都是窗口首元素

# 交叉验证:暴力解法(每个窗口直接max())结果必须一致
def brute_sliding_max(nums, k):
    return [max(nums[i:i+k]) for i in range(len(nums) - k + 1)]

import random
random.seed(8)
for _ in range(20):
    nums = [random.randint(-20, 20) for _ in range(random.randint(1, 15))]
    k = random.randint(1, len(nums))
    assert sliding_window_max(nums, k) == brute_sliding_max(nums, k)

print("OK: 单调队列滑动窗口最大值在边界情况下全部正确, 20组随机测试与暴力解法结果完全一致")
```
本机实测:滑动窗口最大值在单元素、窗口等于数组长度、严格递减序列这几类边界情况下均正确;20 组随机测试中,单调队列解法与暴力逐窗口求最大值的结果完全一致。

**面试怎么问 + 追问链:** "单调队列和单调栈解决的问题有什么本质联系和区别?" → 追问"能不能只用单调栈(不用双端队列)解决滑动窗口最大值问题?"(不能直接照搬——单调栈只支持在一端操作,而滑动窗口需要"队首可能因为超出窗口范围而被移除"这个操作,必须是双端可操作的结构;这个追问检验的是能否看清"单调性维护"和"两端都需要移除"这两个需求分别对应什么数据结构能力)。

**常见坑:**
1. 判断队首是否滑出窗口时,条件写错(比如用 `dq[0] < i - k` 而不是 `dq[0] <= i - k`)——这类"差一"错误在滑动窗口类问题里非常常见,写完后应该用具体小例子验证窗口边界。
2. 先做"移除滑出窗口的队首"判断,再做"记录当前窗口结果"判断,顺序搞反或者遗漏其中一步——两个判断都要在每次循环里执行,遗漏任何一步都会导致结果错误。

---

## 4. 用队列实现栈 / 用栈实现队列

**签名/是什么:**
```
双栈实现队列: in_stack负责入队, out_stack负责出队(为空时才整体倒一次)
双队列/单队列实现栈: 每次入栈后, 把新元素前面的所有元素重新入队一遍(排到新元素后面)
```

**一句话:** 这类"用一种结构实现另一种结构接口"的设计题,考察的不是算法技巧,是对两种结构底层行为差异(LIFO vs FIFO)的理解——核心手段都是"用额外的搬运操作,把顺序倒过来一次"。

**底层机制/为什么这样设计:** 双栈实现队列的关键优化在于"`out_stack` 不是每次出队都重新倒一遍,只在 `out_stack` 为空时才把 `in_stack` 整体倒过去"——这个设计让"倒一次"这个 O(n) 的操作被分摊到了多次出队操作上,单次出队操作的**均摊**复杂度依然是 O(1)(呼应[01类知识点2](01-complexity-and-python-builtins.md#2-均摊复杂度分析amortized-analysis)),而不是每次出队都花 O(n) 重新倒一遍。这是"看起来用了错误的底层结构,但通过均摊分析依然能得到不错的整体效率"的一个典型例子。

**AI 研究/工程场景:** [huggingface-deep-dive 06 类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的梯度累加机制,某些实现里需要把一批操作"延迟执行、按特定顺序重新排列后再统一处理",这类"先暂存后按需重新排序处理"的思路和这里双栈倒腾的手法是同一类工程套路。

**可运行例子:**
```python
class QueueViaStacks:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

    def empty(self):
        return not self.in_stack and not self.out_stack

q = QueueViaStacks()
q.push(1); q.push(2); q.push(3)
assert q.pop() == 1   # 先进先出:1最先入队,应该最先出队
assert q.pop() == 2
q.push(4)
assert q.pop() == 3   # in_stack空了才会重新倒腾,验证倒腾时机正确
assert q.pop() == 4
assert q.empty() is True

class StackViaQueue:
    def __init__(self):
        self.q = __import__("collections").deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):        # 把新元素前面的元素依次搬到它后面
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def empty(self):
        return not self.q

s = StackViaQueue()
s.push(1); s.push(2); s.push(3)
assert s.pop() == 3   # 后进先出:3最后入栈,应该最先出栈
assert s.pop() == 2
s.push(4)
assert s.pop() == 4
assert s.pop() == 1
assert s.empty() is True

print("OK: 双栈实现队列(先进先出)与单队列实现栈(后进先出), 行为验证均正确")
```
本机实测:双栈实现的队列确认保持先进先出语义,且验证了 `out_stack` 只在为空时才重新倒腾(不是每次都倒);单队列实现的栈确认保持后进先出语义。

**面试怎么问 + 追问链:** "用两个栈实现队列,`pop` 操作的均摊复杂度是多少?" → 追问"最坏情况下单次 `pop` 操作的复杂度呢?这和均摊复杂度矛盾吗?"(最坏情况单次 `pop` 是 O(n)(恰好触发一次整体倒腾);均摊复杂度是 O(1)——两者不矛盾,这正是[01类知识点2](01-complexity-and-python-builtins.md#2-均摊复杂度分析amortized-analysis)讲过的"均摊 O(1) 不等于每次都是 O(1)"这个概念在一个新场景下的再次应用,能不能主动联系到这一点是检验知识是否真正融会贯通的好问题)。

**常见坑:**
1. 双栈实现队列时,`out_stack` 不为空时依然执行了倒腾操作——这会打乱已经存在于 `out_stack` 里的正确顺序,必须严格保证只在 `out_stack` 为空时才倒腾。
2. 单队列实现栈时,`push` 里"倒腾"的次数算错(比如用 `len(self.q)` 而不是 `len(self.q) - 1`)——多倒腾一次或少倒腾一次都会打乱正确的栈顺序。

---

## 5. LRU 缓存设计与手写实现

**签名/是什么:**
```
LRU(Least Recently Used)：容量满时，淘汰"最久未被访问"的元素
要求 get/put 都是 O(1) —— 需要哈希表(O(1)定位) + 双向链表(O(1)增删)组合
```

**一句话:** LRU 缓存是"手写一个真实可用的数据结构"这类设计题的标杆——单独用哈希表做不到"知道谁最久没被访问",单独用链表做不到"O(1) 定位到某个 key 对应的节点",必须把两者组合起来,各自发挥各自的优势。

**底层机制/为什么这样设计:** 双向链表维护"访问新旧顺序"(头部是最近访问,尾部是最久未访问),每次访问(`get` 命中或 `put`)都把对应节点挪到链表头部;哈希表存储 `key -> 链表节点` 的映射,让"根据 key 找到对应节点"这一步降到 O(1),不需要遍历链表查找。**双向**链表(而不是单向)是必需的——因为"把一个节点从链表中间移除"需要同时知道它的前驱和后继,单向链表做不到 O(1) 移除中间节点(呼应[03类知识点7](03-linked-lists.md#7-链表常见坑)提到的链表操作特性)。用两个哨兵节点(头哨兵和尾哨兵)固定链表两端,能让"添加到头部"和"从尾部淘汰"这两个操作不需要对"链表是否为空"做额外特判(呼应[03类知识点1](03-linked-lists.md#1-链表基础操作与哨兵节点技巧)的哨兵节点技巧)。

两种结构怎么拼在一起,跟着"可运行例子"里 `LRUCache(2)` 的真实调用顺序走一遍会比单看代码更直观(`head`/`tail` 是不存数据的哨兵节点,`⇄` 表示双向链接):

```
初始(容量2):  head ⇄ tail                                        map = {}

put(1,1):     head ⇄ [1:1] ⇄ tail                                map = {1:节点1}

put(2,2):     head ⇄ [2:2] ⇄ [1:1] ⇄ tail                        map = {1:节点1, 2:节点2}
              (新节点插到头部, 2是"最近访问", 1相对更久未访问)

get(1):       head ⇄ [1:1] ⇄ [2:2] ⇄ tail                        map映射关系不变(还是同两个节点对象)
              (命中后把节点1挪到头部, 1变成"最近访问", 2变成"最久未访问")

put(3,3):     ①先把新节点插到头部(此刻暂时超出容量2):
              head ⇄ [3:3] ⇄ [1:1] ⇄ [2:2] ⇄ tail
              ②发现超出容量, 淘汰 tail.prev(此刻正是最久未访问的节点2):
              head ⇄ [3:3] ⇄ [1:1] ⇄ tail                        map = {1:节点1, 3:节点3}(2被删除)
```

淘汰的节点2,正是"可运行例子"里 `assert lru.get(2) == -1` 验证的对象——它之所以被淘汰而不是1,是因为 `get(1)` 那一步已经把1挪到了头部,真正"最久未访问"的变成了2。

**AI 研究/工程场景:** [huggingface-deep-dive 11 类](../huggingface-deep-dive/11-hub-and-sharing.md)讲过的模型缓存机制(`~/.cache/huggingface/hub`),真实的模型/数据集缓存淘汰策略在概念上就是 LRU 的应用场景——本地磁盘空间有限时,优先保留最近被使用过的模型权重文件,淘汰长期未被访问的旧版本。

**可运行例子:**
```python
class LRUCache:
    class _Node:
        __slots__ = ("key", "val", "prev", "nxt")
        def __init__(self, key=0, val=0):
            self.key, self.val = key, val
            self.prev = self.nxt = None

    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}
        self.head = self._Node()   # 头哨兵,head.nxt永远是最近访问的节点
        self.tail = self._Node()   # 尾哨兵,tail.prev永远是最久未访问的节点
        self.head.nxt = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.nxt = node.nxt
        node.nxt.prev = node.prev

    def _add_front(self, node):
        node.nxt = self.head.nxt
        node.prev = self.head
        self.head.nxt.prev = node
        self.head.nxt = node

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_front(node)   # 访问过就要挪到最前面
        return node.val

    def put(self, key, val):
        if key in self.map:
            self._remove(self.map[key])
        node = self._Node(key, val)
        self.map[key] = node
        self._add_front(node)
        if len(self.map) > self.cap:
            lru_node = self.tail.prev   # 尾部前一个就是最久未访问的
            self._remove(lru_node)
            del self.map[lru_node.key]

lru = LRUCache(2)
lru.put(1, 1); lru.put(2, 2)
assert lru.get(1) == 1     # 访问1,1变成最近使用
lru.put(3, 3)               # 容量满,淘汰最久未用的2(不是1,因为1刚被访问过)
assert lru.get(2) == -1
assert lru.get(3) == 3

cache1 = LRUCache(1)
cache1.put(1, 100)
cache1.put(2, 200)          # 容量为1,put新key直接淘汰旧key
assert cache1.get(1) == -1
assert cache1.get(2) == 200

# 交叉验证:用"最简单但低效"的list实现同样语义,和上面的O(1)实现结果必须一致
class NaiveLRU:
    def __init__(self, capacity):
        self.cap = capacity
        self.order = []   # 末尾是最近使用
        self.data = {}
    def get(self, key):
        if key not in self.data:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.data[key]
    def put(self, key, val):
        if key in self.data:
            self.order.remove(key)
        self.data[key] = val
        self.order.append(key)
        if len(self.data) > self.cap:
            oldest = self.order.pop(0)
            del self.data[oldest]

import random
random.seed(9)
fast = LRUCache(3)
naive = NaiveLRU(3)
for _ in range(200):
    key = random.randint(1, 5)
    if random.random() < 0.5:
        assert fast.get(key) == naive.get(key)
    else:
        val = random.randint(1, 1000)
        fast.put(key, val); naive.put(key, val)

print("OK: LRU缓存在基础场景/容量为1的边界情况下全部正确, "
      "200次随机get/put操作序列与朴素低效实现的行为完全一致")
```
本机实测:基础淘汰场景和容量为 1 的边界场景均正确;200 次随机 get/put 操作序列的完整交互过程中,O(1) 实现和朴素但显然正确的低效实现,每一步的返回结果完全一致。

**面试怎么问 + 追问链:** "为什么 LRU 缓存需要双向链表,单向链表不行吗?" → 追问"如果只要求 `put` 是 O(1),`get` 允许 O(n),设计会有什么不同?"(如果放宽 `get` 的复杂度要求,可以完全不用哈希表,只用一个数组/链表按访问顺序排列,`get` 时线性查找、找到后挪到最前面即可——这个追问检验的是能否理解"复杂度要求本身决定了数据结构的选择",而不是不假思索地照搬"LRU=哈希表+双向链表"这个结论)。

**常见坑:**
1. `get` 命中时忘记把对应节点挪到链表头部——这会导致"最近访问"这个信息没有被正确更新,淘汰逻辑会做出错误判断(可能淘汰了刚被访问过的元素)。
2. `put` 一个已存在的 key 时,忘记先从原位置移除旧节点就直接添加新节点——会导致同一个 key 在链表里出现两次,破坏内部一致性。

---

## 6. LFU 缓存设计与手写实现

**签名/是什么:**
```
LFU(Least Frequently Used)：容量满时，淘汰"访问频率最低"的元素；
频率相同时，淘汰其中最久未被访问的(LFU + LRU 的组合语义)
```

**一句话:** LFU 比 LRU 多了一个维度——不仅要知道"谁最久没被访问",还要知道"谁被访问的次数最少",且要在频率相同的候选里再按 LRU 规则决出淘汰顺序,这是一个"按频率分层、每层内部再按时间排序"的组合结构。

**底层机制/为什么这样设计:** 核心数据结构是"频率 → 该频率对应的 key 集合(且集合内部保持插入/访问顺序)"这样一个二级结构——用 `frequency` 分桶,每个桶内部是一个能保持顺序的结构(有序字典或双向链表,LFU 内部实现事实上是很多个并列的、LRU 风格的子结构)。每次访问一个 key,它的频率 +1,需要从旧频率桶移除、加入新频率桶;额外维护一个 `min_freq` 变量跟踪当前最小频率是多少,避免每次淘汰都要扫描所有频率桶找最小值。这个设计是"多维度排序需求"(先按频率、再按时间)转化成"分层结构,每层解决一个维度"的典型思路。

**AI 研究/工程场景:** LFU 语义比 LRU 更适合"访问频率有长期规律"的场景——比如缓存 tokenizer 词表里的高频 token 编码结果,访问次数本身就是比"最近是否用过"更可靠的信号(高频词长期都会被频繁访问,不会因为短期没出现就被误淘汰),这是 LFU 和 LRU 语义差异在真实缓存策略选型时的实际考量。

**可运行例子:**
```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.min_freq = 0
        self.key_val = {}
        self.key_freq = {}
        self.freq_keys = defaultdict(OrderedDict)   # freq -> {key: True}，保持插入顺序

    def _touch(self, key):
        freq = self.key_freq[key]
        del self.freq_keys[freq][key]
        if not self.freq_keys[freq] and freq == self.min_freq:
            self.min_freq += 1
        self.key_freq[key] = freq + 1
        self.freq_keys[freq + 1][key] = True

    def get(self, key):
        if key not in self.key_val:
            return -1
        self._touch(key)
        return self.key_val[key]

    def put(self, key, val):
        if self.cap <= 0:
            return
        if key in self.key_val:
            self.key_val[key] = val
            self._touch(key)
            return
        if len(self.key_val) >= self.cap:
            evict_key, _ = self.freq_keys[self.min_freq].popitem(last=False)  # 同频率里最久未访问的
            del self.key_val[evict_key]
            del self.key_freq[evict_key]
        self.key_val[key] = val
        self.key_freq[key] = 1
        self.freq_keys[1][key] = True
        self.min_freq = 1

lfu = LFUCache(2)
lfu.put(1, 1); lfu.put(2, 2)
assert lfu.get(1) == 1        # key1频率变成2
lfu.put(3, 3)                   # 容量满,key1频率2 vs key2频率1,淘汰key2(频率更低)
assert lfu.get(2) == -1
assert lfu.get(3) == 3

lfu2 = LFUCache(0)               # 容量为0,任何put都不应该生效
lfu2.put(1, 1)
assert lfu2.get(1) == -1

lfu3 = LFUCache(2)
lfu3.put(1, 1); lfu3.put(2, 2)
lfu3.get(1); lfu3.get(1); lfu3.get(2)   # key1频率3, key2频率2
lfu3.put(3, 3)                            # 淘汰key2(频率更低)
assert lfu3.get(2) == -1 and lfu3.get(1) == 1 and lfu3.get(3) == 3

print("OK: LFU缓存在'频率决定淘汰''容量为0''多次访问后频率正确累计'等场景下全部验证正确")
```
本机实测:基础的"频率更低者被淘汰"场景、容量为 0 的边界场景、多次访问后频率正确累计并影响淘汰决策的场景,全部验证通过。

**面试怎么问 + 追问链:** "LFU 和 LRU 淘汰策略的本质区别是什么,分别适合什么场景?" → 追问"如果两个 key 的访问频率相同,LFU 应该淘汰哪一个?"(淘汰频率相同的候选里最久未被访问的那个——这正是为什么 LFU 内部的每个频率桶依然要维护插入/访问顺序,不是简单的一个哈希集合;这个追问检验的是能否说清 LFU 内部"按频率分层,层内按时间排序"这个组合结构的必要性,而不是把 LFU 简化成"只看频率"的错误理解)。

**常见坑:**
1. 淘汰时忘记同时处理"频率相同"这个维度,只按频率淘汰、不管同频率内谁更久未访问——这会让实现退化成一个不完整的 LFU,不满足标准 LFU 语义的完整定义。
2. `min_freq` 更新时机出错(比如某个 key 从当前最小频率桶移走后,忘记检查该桶是否已经清空、是否需要把 `min_freq` 加一)——这会导致后续淘汰操作在一个已经不存在候选的空频率桶上操作,抛出异常或者取到错误的淘汰对象。

---

## 7. 优先级队列与栈的组合应用场景

**签名/是什么:**
```
任务调度器：给定任务序列和冷却时间，同一任务两次执行之间必须间隔至少cooldown个单位时间，
求完成所有任务的最短总时间 —— 用堆(优先级队列)决定"下一个执行哪个任务"，
用队列追踪"哪些任务正在冷却中、什么时候能重新变为可选"
```

**一句话:** 有些问题不是单一数据结构能解决的——任务调度器需要"每一步都优先选择当前剩余次数最多的任务"(堆的强项)和"追踪一批任务各自的冷却截止时间"(队列的强项)两种能力组合在一起,分别用最适合的结构承担各自的职责。

**底层机制/为什么这样设计:** 每个时间单位,先看堆顶(当前剩余次数最多的任务类型)是否可执行,执行后如果这个任务还有剩余次数,把它连同"冷却结束时间"一起放进一个 FIFO 队列(因为任务们会按照它们被放入冷却的顺序依次冷却结束,先冷却的先解冻,这正是队列的天然语义);每一步同时检查队列头部的任务是否冷却期已满,如果满了就重新放回堆中参与下一轮的优先级竞争。这个设计让"优先级决策"(堆)和"时间顺序追踪"(队列)两个关注点保持解耦,各自用最匹配的结构实现,而不是勉强用一个结构同时应付两种不同性质的需求。

**AI 研究/工程场景:** [huggingface-deep-dive 05 类](../huggingface-deep-dive/05-trainer-api-internals.md)讲过的训练回调机制(比如 `EarlyStoppingCallback`),如果要实现一个"优先处理最紧急的回调,但同一个回调触发后有最小间隔限制"的调度系统,本质上和这里的任务调度器是同一类"优先级 + 冷却时间"组合问题。

**可运行例子:**
```python
import heapq
from collections import Counter, deque

def task_scheduler(tasks, cooldown):
    counts = Counter(tasks)
    heap = [-c for c in counts.values()]   # 用负数模拟大根堆(Python heapq只支持小根堆)
    heapq.heapify(heap)
    time = 0
    cooling = deque()   # (可重新变为可选的时间点, 剩余次数的负数)
    while heap or cooling:
        time += 1
        if heap:
            remaining = heapq.heappop(heap) + 1   # 执行一次,剩余次数(负数)绝对值减1
            if remaining != 0:
                cooling.append((time + cooldown, remaining))
        if cooling and cooling[0][0] == time:
            _, remaining = cooling.popleft()
            heapq.heappush(heap, remaining)
    return time

assert task_scheduler(['A', 'A', 'A', 'B', 'B', 'B'], 2) == 8   # 经典用例:A B _ A B _ A B
assert task_scheduler(['A', 'A', 'A', 'B', 'B', 'B'], 0) == 6   # 无冷却限制,直接顺序执行
assert task_scheduler(['A'], 5) == 1                              # 单任务,不涉及冷却
assert task_scheduler([], 3) == 0                                  # 空任务列表
assert task_scheduler(['A', 'A'], 1) == 3                          # A _ A (冷却1个单位)

print("OK: 任务调度器(堆+队列组合)在经典用例/无冷却/单任务/空任务/最小冷却等场景下全部正确")
```
本机实测:经典用例(3 个 A、3 个 B、冷却 2)得到最短总时间 8,与手工推演的 `A B _ A B _ A B` 排布完全一致;无冷却、单任务、空任务列表、最小冷却间隔这几类边界情况也全部验证正确。

**面试怎么问 + 追问链:** "任务调度器问题为什么要同时用堆和队列,只用其中一个不行吗?" → 追问"能不能不模拟每一个时间单位,用数学公式直接算出最短总时间?"(可以——用最高频次的任务出现次数 `max_count` 和有多少种任务并列拥有这个最高频次 `max_count_ties`,可以推导出一个 O(1) 的公式:`max((max_count-1)*(cooldown+1) + max_count_ties, len(tasks))`;这个追问检验的是能否从"模拟过程"跳出来,发现问题背后更本质的数学结构——这是终面里"仅仅代码正确"和"体现出更深入的问题理解"的一个真实分水岭)。

**常见坑:**
1. 用负数模拟大根堆(Python `heapq` 只原生支持小根堆)时,加减符号搞混——取出时要 `+1`(因为存的是负数,执行一次意味着绝对值减少,即负数本身增加),这个符号方向很容易搞反。
2. 冷却队列判断"是否到期"的条件写错(比如用 `>=` 而不是 `==`)——因为每个时间单位都会检查一次,只要冷却入队时记录的到期时间准确,用 `==` 精确判断即可,过度防御性地写成 `>=` 反而可能掩盖记录到期时间时的真实 bug。

---

## 8. 单调栈 / 单调队列常见坑

**签名/是什么:**
```
维护顺序反了：该保持递增却写成递减(或反过来)
弹出时机错误：该在入队前弹出，却写成入队后弹出（或反过来）
```

**一句话:** 单调栈/单调队列的 bug 大多集中在两类:维护的单调方向和问题要求的方向搞反、弹出候选的时机和入队新元素的时机顺序颠倒——这两类错误往往不会让程序报错,只会让结果"看起来合理但是错的"。

**底层机制/为什么这样设计:** 以"下一个更大元素"为例,如果错误地维护成单调**递增**栈(而不是递减),弹出栈顶的条件和触发逻辑会完全变味,得到的会是"上一个更大元素"这类完全不同语义的结果,而不是运行时错误——这类 bug 必须靠对照"预期语义"精心设计的测试用例才能发现,不能指望程序自己报错。滑动窗口最大值的"弹出时机"错误同理:如果把"移除滑出窗口的队首"这一步和"记录当前窗口答案"这一步的顺序写反,会导致队首元素在该被移除之前就被记录成了答案,产生一个"曾经在窗口内、但现在已经不在了"的错误值。

**AI 研究/工程场景:** 这类"逻辑顺序颠倒导致结果错误但程序不报错"的问题,是本篇([03类知识点7](03-linked-lists.md#7-链表常见坑))反复强调过的"链表/栈/队列这类需要精细维护内部顺序的结构,bug 密度和边界测试覆盖程度直接相关"这条纪律的又一次真实体现。

**可运行例子:**
```python
def next_greater_correct(nums):
    res = [-1] * len(nums)
    stack = []   # 正确:维护单调递减栈
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            res[stack.pop()] = x
        stack.append(i)
    return res

def next_greater_wrong_direction(nums):
    """故意维护成错误的方向(递增),看会产生什么真实后果"""
    res = [-1] * len(nums)
    stack = []
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] > x:   # 错误:条件方向反了
            res[stack.pop()] = x
        stack.append(i)
    return res

correct = next_greater_correct([2, 1, 2, 4, 3])
wrong = next_greater_wrong_direction([2, 1, 2, 4, 3])
assert correct == [4, 2, 4, -1, -1]
assert wrong != correct   # 真实复现:方向写反后,结果确实和正确答案不一样
assert wrong == [1, -1, -1, 3, -1]  # 这实际上计算出的是完全不同语义的结果(下一个更小元素)

from collections import deque

def sliding_max_correct(nums, k):
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res

def sliding_max_wrong_order(nums, k):
    """故意把'移除滑出窗口的队首'和'记录答案'顺序写反"""
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])       # 错误:先记录答案
        if dq[0] <= i - k:
            dq.popleft()                    # 再移除过期队首,这一步的移除对本次记录已经太晚了
    return res

correct_result = sliding_max_correct([1, 3, -1, -3, 5, 3, 6, 7], 3)
wrong_result = sliding_max_wrong_order([1, 3, -1, -3, 5, 3, 6, 7], 3)
assert correct_result == [3, 3, 5, 5, 6, 7]
# 这个经典用例里,顺序颠倒"恰好"没有暴露问题(结果和正确版本完全一样)——
# 如果只用这一个例子测试,会误以为"顺序其实无所谓",这正是本知识点要警惕的陷阱
assert wrong_result == correct_result

# 换一个窗口大小为1的例子,顺序错误就真实暴露出来了:
# 窗口大小1时,"记录答案"和"移除过期队首"这两步实际上处理的是同一个下标,
# 顺序颠倒会让本该在这一步就过期的旧值被多记录了一次
correct_k1 = sliding_max_correct([5, 4, 3, 2, 1], 1)
wrong_k1 = sliding_max_wrong_order([5, 4, 3, 2, 1], 1)
assert correct_k1 == [5, 4, 3, 2, 1]   # 窗口大小1,正确答案就是数组本身
assert wrong_k1 == [5, 5, 4, 3, 2]     # 错误版本:每个位置都多"拖延"了一步,答案整体错位
assert wrong_k1 != correct_k1            # 真实复现:顺序错误在这个例子里确实造成了错误结果

print("OK: 现场复现单调栈方向写反导致语义完全变化(下一个更大->下一个更小)这个真实后果; "
      "单调队列顺序错误在k=3经典用例里被巧合掩盖, 换成k=1才真实暴露出结果错位——"
      "证明不能仅凭个别测试用例通过就断言逻辑顺序无关紧要")
```
本机实测:单调栈维护方向写反后,得到的实际上是"下一个更小元素"这个完全不同语义的结果,而不是简单的数值偏差——这是一个"程序正常运行、结果看似合理、实则语义完全错误"的真实案例,比"程序崩溃"更难被发现。单调队列"记录答案"和"移除过期队首"顺序颠倒的 bug 更具迷惑性:在窗口大小为 3 的经典用例上,错误版本和正确版本给出**完全相同**的结果(纯属巧合,触发条件没有恰好重合);换成窗口大小为 1 的例子,错误版本立刻暴露出结果整体错位(`[5,5,4,3,2]` vs 正确的 `[5,4,3,2,1]`)——这个真实对比直接证明了"少数几个测试用例通过"不能作为"逻辑一定正确"的可靠证据。

**面试怎么问 + 追问链:** "写单调栈代码时,怎么确定应该维护递增还是递减?" → 追问"如果我说'我测试了几个例子,结果都对,所以顺序应该没错',这个论证有什么问题?"(个别测试用例通过不能证明逻辑普遍正确——本知识点最后一个例子就演示了这一点:两种不同(一个正确一个有顺序错误)的实现在某个特定输入上恰好给出了相同结果,这是一个真实的陷阱,提醒"用测试验证正确性"必须选择**有区分度**的测试用例,不能随便挑一个例子就下结论,这也是[02类知识点7](02-arrays-and-strings.md#7-循环不变量方法论)"循环不变量"方法论想要替代的、不可靠的"跑几个例子看着对"的验证方式)。

**常见坑:**
1. 只用一两个"容易想到"的例子测试单调栈/单调队列代码——本知识点最后展示的现象证明,这种验证方式可能被"恰好没有暴露问题的输入"欺骗,应该刻意构造多组、覆盖不同触发顺序的测试用例。
2. 调试单调栈/单调队列 bug 时,只检查"最终结果对不对",不检查"中间过程栈/队列的内容是否符合预期单调性"——在关键位置打印栈/队列的中间状态,往往比只看最终输出更快定位到顺序或方向错误的具体发生位置。

---

*本篇 8 个知识点全部在仓库根目录 `.venv` 真实测试验证(含边界情况覆盖、随机交叉验证、以及"错误方向/错误顺序"的真实复现对照)。*
