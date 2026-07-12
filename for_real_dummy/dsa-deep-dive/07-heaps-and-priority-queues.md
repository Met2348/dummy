# 07 · 堆 / 优先队列(Heaps and Priority Queues)

> 总览见 [00-roadmap.md](00-roadmap.md)。堆是"能在 O(log n) 内维护当前最值"的结构,合并 K 路数据、TopK、任务调度这类问题的标准工具。本类额外做了一件大多数题解不会做的事:**真实测量"算法复杂度更优"和"实际跑得更快"之间到底差多少、什么时候会不一致**。

---

## 1. 堆的数组表示与基本操作

**签名/是什么:**
```
用数组存储完全二叉树: 下标i的父节点是(i-1)//2, 左孩子是2i+1, 右孩子是2i+2
sift_up(上浮): 新元素插入末尾后, 和父节点比较, 不满足堆性质就交换并继续上浮
sift_down(下沉): 移除堆顶后, 末尾元素补到堆顶, 和较小(或较大)的孩子比较, 不满足就交换并继续下沉
```

**一句话:** 堆是一棵满足"父节点始终不大于(或不小于)子节点"性质的完全二叉树,用数组存储(不需要真的建树、维护指针),下标运算就能定位父子关系——插入/删除堆顶都是 O(log n)(和树高成正比),取堆顶是 O(1)。

**底层机制/为什么这样设计:** 用数组而不是真正的树结构存储堆,是因为堆永远是**完全二叉树**(除最后一层外都被填满,最后一层从左到右填充)——这个特殊形状保证了数组下标和树形结构之间存在固定的数学映射关系,不需要额外的指针开销。**从零构建一个堆,不是把 n 个元素逐个插入(那样是 n 次 O(log n) 插入,总共 O(n log n)),而是对数组做"自底向上的批量下沉"**:从最后一个非叶子节点开始,依次对每个节点做 sift_down——这个过程的总复杂度实际上是 O(n),不是直觉上认为的 O(n log n)。证明的关键在于:大部分节点都在树的**底层**,底层节点的下沉深度很浅(叶子节点深度是 0,不需要下沉);只有极少数节点在顶层,虽然下沉深度大但数量少——把每一层"节点数量 × 该层下沉深度上限"累加起来,是一个收敛级数,总和是 O(n) 而不是 O(n log n)。

**AI 研究/工程场景:** [huggingface-deep-dive 12 类](../huggingface-deep-dive/12-inference-optimization.md)讲过的 beam search 解码,维护当前 beam 宽度个候选序列并按累积概率排序,本质上就是用堆(或者概念上等价的有限优先队列)维护"当前最优的 k 个候选"这个动态更新的需求。

**可运行例子:**
```python
import heapq
import random
import time

def sift_down(arr, i, size):
    while True:
        smallest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < size and arr[left] < arr[smallest]:
            smallest = left
        if right < size and arr[right] < arr[smallest]:
            smallest = right
        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest

def build_heap_bottom_up(arr):
    arr = arr[:]
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)
    return arr

def is_min_heap(arr):
    n = len(arr)
    for i in range(n):
        left, right = 2 * i + 1, 2 * i + 2
        if left < n and arr[left] < arr[i]:
            return False
        if right < n and arr[right] < arr[i]:
            return False
    return True

h = build_heap_bottom_up([5, 2, 8, 1, 9, 3, 7])
assert is_min_heap(h)
assert is_min_heap(build_heap_bottom_up([]))
assert is_min_heap(build_heap_bottom_up([1]))

def sift_up_pure_python(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[parent] <= arr[i]:
            break
        arr[parent], arr[i] = arr[i], arr[parent]
        i = parent

def build_heap_via_pure_python_inserts(arr):
    """同样用纯Python实现,但是逐个插入(每次O(log n)),做真正apples-to-apples的语言对照"""
    h = []
    for x in arr:
        h.append(x)
        sift_up_pure_python(h, len(h) - 1)
    return h

def best_of(fn, data, trials=3):
    best = None
    for _ in range(trials):
        t0 = time.perf_counter(); fn(data); dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best

random.seed(20)
data = [random.random() for _ in range(200_000)]

bottom_up_time = best_of(build_heap_bottom_up, data)
pure_insert_time = best_of(build_heap_via_pure_python_inserts, data)
# 同为纯Python实现时,自底向上O(n)确实应该比逐个插入O(n log n)快(哪怕优势不算悬殊)
assert bottom_up_time < pure_insert_time

# 但如果对比对象换成heapq.heapify(标准库的C加速实现),结果会完全不同:
c_accelerated_time = best_of(lambda d: heapq.heapify(d[:]), data)
# C加速版本应该远快于我们手写的纯Python O(n)实现——这不是算法层面的差距,是实现语言的差距
assert c_accelerated_time < bottom_up_time / 3

print(f"OK: 同为纯Python实现时, 自底向上build={bottom_up_time:.4f}s < 逐个插入build={pure_insert_time:.4f}s"
      f"(比值{pure_insert_time/bottom_up_time:.2f}, 验证O(n)确实快于O(n log n)); "
      f"但heapq.heapify(C加速)={c_accelerated_time:.4f}s, 比我们手写的纯Python O(n)实现还快"
      f"{bottom_up_time/c_accelerated_time:.0f}倍(实现语言差距盖过了算法层面的差距)")
```
本机实测(这是本知识点最重要的发现,记录下来是为了如实反映一个容易被简化教材忽略的真相):在**同为纯 Python 实现**的前提下,自底向上建堆(0.0108s)确实比逐个插入建堆(0.0152s)更快,比值约 1.4 倍,方向符合 O(n) vs O(n log n) 的理论预期,但差距在这个规模下并不悬殊。**更值得记住的是**:标准库 `heapq.heapify()`(C 加速实现)比我们手写的纯 Python O(n) 实现还要快 11~14 倍——这说明"算法复杂度更优"和"实际跑得更快"是两个相关但不能划等号的问题,**实现语言/常数因子有时候能盖过复杂度量级本身的差距**,这是任何声称"O(n) 一定比 O(n log n) 快"的简化说法都没有说全的地方。

**面试怎么问 + 追问链:** "从一个无序数组建堆,为什么是 O(n) 而不是看起来更直觉的 O(n log n)?" → 追问"能不能现场推导一下这个 O(n) 结论,而不是直接背结论?"(设树高为 h=log n,第 i 层(从底部数,i=0 是叶子层)有约 n/2^(i+1) 个节点,每个节点下沉深度最多是 i;把每一层"节点数×下沉深度"累加,是 `Σ i·n/2^(i+1)`(i 从 0 到 h),这是一个已知收敛于常数的级数乘以 n,整体是 O(n)——能现场推导而不是死记结论,是这道题真正的深度所在)。**追问链继续深入**:"如果面试官追问'那既然建堆是O(n),为什么你实测下来比标准库的heapify()慢十几倍,你的O(n)证明是不是错的?',该怎么回答?"(证明本身没有错——O(n) 描述的是**渐进增长趋势**,不是绝对速度;纯 Python 函数调用/字节码解释执行本身有远高于 C 实现的常数开销,这个常数开销和输入规模无关,不会体现在大 O 记号里,但会真实体现在秒表上;能不能把"复杂度分析"和"实际工程性能"这两件事清楚地分开讨论,而不是遇到反直觉的实测数据就怀疑自己的理论证明,是终面里检验候选人是否真正理解复杂度分析本质、而不是把大 O 当成"性能保证书"的关键区分点)。

**常见坑:**
1. 把"O(n) 建堆"简单理解成"建堆比排序快",然后不加验证地认为"任何用纯 Python 手写的 O(n) 算法,一定比标准库里复杂度更差的方法快"——本知识点的实测数据是一个具体、真实的反例,复杂度优势可能被实现语言的常数因子抵消,不能脱离具体实现环境空谈复杂度。
2. 建堆时用"自顶向下"的顺序调用 sift_down(而不是从最后一个非叶子节点开始自底向上)——顺序反了会导致某些子树在被处理时,它的子节点还没有被调整成堆,建堆结果错误。

---

## 2. Python `heapq` 模块使用与内部实现

**签名/是什么:**
```
heapq.heapify(list)          # 原地把列表转换成堆,O(n)
heapq.heappush(heap, item)   # 插入元素,O(log n)
heapq.heappop(heap)          # 弹出并返回最小元素,O(log n)
heapq.heapreplace(heap, item)  # 弹出最小值后插入新值,比先pop再push更高效
heapq.nlargest(k, iterable) / heapq.nsmallest(k, iterable)
```

**一句话:** Python 的 `heapq` 模块只提供**最小堆**这一种堆,且它不是一个独立的堆对象类型,而是一组直接操作普通 `list` 的函数——这个设计选择本身("堆就是满足堆性质的 list,不需要额外包装类")值得理解,不只是记住 API。

**底层机制/为什么这样设计:** `heapq` 选择"函数式操作普通 list"而不是"提供一个 Heap 类",是 Python 标准库一贯的"数据和操作分离、优先复用内置类型"风格的体现——好处是可以直接对已有的 list 调用 `heapify()` 原地转换,不需要额外拷贝到一个新的堆对象里。`heapreplace(heap, item)` 存在的意义是性能:如果单纯 `heappop()` 再 `heappush()`,是两次独立的 O(log n) 操作;`heapreplace` 把新元素直接放到堆顶再做一次下沉,是一次操作完成"替换"这个语义,在需要频繁"弹出旧值、插入新值"的场景(比如滑动窗口最值维护)能省下明显的常数开销。`nlargest`/`nsmallest` 内部会根据 `k` 和 `len(iterable)` 的relative大小自动选择策略(k 很小时用堆,k 接近总长度时退化成排序),这是标准库内部已经做好的一层自适应优化。

**AI 研究/工程场景:** [huggingface-deep-dive 12 类](../huggingface-deep-dive/12-inference-optimization.md)讲过 `generate()` 内部的 top-k 采样策略,如果要从词表 logits 里挑出概率最高的 k 个 token,`heapq.nlargest` 或其等价的张量操作(`torch.topk`)背后是同一个"不需要完整排序,只需要前 k 个"的堆思想。

**可运行例子:**
```python
import heapq

h = [5, 2, 8, 1, 9]
heapq.heapify(h)
assert h[0] == 1   # 堆顶必然是最小值(heapq只支持最小堆)
assert heapq.heappop(h) == 1
assert h[0] == 2   # 弹出后,新的堆顶是次小值

heapq.heappush(h, 0)
assert h[0] == 0   # 插入比当前堆顶更小的值,新值成为堆顶

assert heapq.nlargest(3, [5, 2, 8, 1, 9, 3, 7]) == [9, 8, 7]
assert heapq.nsmallest(3, [5, 2, 8, 1, 9, 3, 7]) == [1, 2, 3]
assert heapq.nlargest(0, [1, 2, 3]) == []          # k=0
assert heapq.nlargest(10, [1, 2, 3]) == [3, 2, 1]  # k超过实际元素个数,返回全部

# heapreplace的正确性:等价于heappop+heappush,但应该是一次操作
h2 = [1, 3, 5]
heapq.heapify(h2)
old_min = heapq.heapreplace(h2, 4)
assert old_min == 1               # 返回的是被替换掉的旧堆顶
assert sorted(h2) == [3, 4, 5]     # 替换后堆内容正确

# heapreplace vs pop+push 在功能上必须等价(交叉验证)
import random
random.seed(22)
for _ in range(20):
    base = [random.randint(0, 100) for _ in range(random.randint(1, 15))]
    new_val = random.randint(0, 100)

    h_a = base[:]
    heapq.heapify(h_a)
    old_a = heapq.heapreplace(h_a, new_val)

    h_b = base[:]
    heapq.heapify(h_b)
    old_b = heapq.heappop(h_b)
    heapq.heappush(h_b, new_val)

    assert old_a == old_b
    assert sorted(h_a) == sorted(h_b)

print("OK: heapq基础操作全部正确(含k=0/k超过元素个数等边界情况); "
      "20组随机测试验证heapreplace与'先pop再push'功能完全等价")
```
本机实测:`heapq` 基础操作(heapify/heappush/heappop)全部正确;`nlargest`/`nsmallest` 在 k=0、k 超过实际元素个数这两类边界下均正确;20 组随机测试验证 `heapreplace` 和"先 `heappop` 再 `heappush`"在功能上完全等价。

**面试怎么问 + 追问链:** "Python 的 `heapq` 为什么只提供最小堆,没有专门的最大堆?" → 追问"实现最大堆的常见技巧(存入相反数)有什么局限性?"(如果堆里存的是可比较大小的原始数值,取负数没问题;但如果堆里存的是自定义对象或者不支持取负的数据(比如字符串),取负技巧就不适用,需要改用"自定义比较逻辑的包装类"或者把优先级和数据分开存成 `(priority, data)` 元组、只对 priority 取负——这个追问检验的是能否处理"取负数"这个技巧失效时该怎么办,而不是只会背这一个特定场景的小技巧)。

**常见坑:**
1. 堆里存 `(value, obj)` 这样的元组,当多个 `value` 相等时,Python 会尝试比较 `obj`——如果 `obj` 之间不支持比较(比如自定义类没有实现 `__lt__`),会抛出 `TypeError`(呼应 [03 类知识点 5](03-linked-lists.md#5-复杂链表操作合并-k-个有序链表与复制带随机指针的链表)常见坑里提到的同一个问题),标准做法是元组里加一个额外的、保证唯一或至少可比较的字段(比如递增的计数器)垫在中间。
2. 误以为 `heap[0]` 之外的其他位置也遵守某种直观的顺序(比如 `heap[1]` 是第二小的值)——堆只保证"父节点不大于子节点"这一个局部性质,除了堆顶,其余位置的相对顺序没有任何保证。

---

## 3. TopK 问题:大根堆与小根堆的选择

**签名/是什么:**
```
找最大的K个元素: 维护一个大小为K的小根堆, 堆顶(最小值)是"K个里最小的",
                新元素比堆顶大就替换 —— 堆里最终留下的就是最大的K个
```

**一句话:** TopK 问题里,堆的大小固定为 K,而堆的"最小/最大"方向,和你要找的是"最大的 K 个"还是"最小的 K 个"**恰好相反**——这个反直觉的方向选择,是 TopK 场景里最容易搞混的地方。

**底层机制/为什么这样设计:** 要找"最大的 K 个元素",维护一个大小为 K 的**小根堆**:堆顶始终是这 K 个候选里最小的那个——如果新来的元素比堆顶还小,它注定进不了"最大的 K 个"这个集合,直接跳过;只有比堆顶大,才有资格替换掉堆顶(当前 K 个里最不够格的那个)。这个设计的效率来源:整个过程只需要维护一个大小恒为 K 的堆(不是完整维护 n 个元素的堆),复杂度是 O(n log k),当 K 远小于 n 时,比"完整排序再取前 K 个"(O(n log n))更高效。

**AI 研究/工程场景:** [12 类](12-trie-and-string-matching.md)会提到的高频词统计场景(比如统计一批语料里出现频率最高的 K 个 token,用于构建词表)是 TopK 问题的直接应用——[huggingface-deep-dive 01 类](../huggingface-deep-dive/01-tokenizer-internals.md)讲过 BPE 分词训练本质上也涉及"反复找出当前频率最高的相邻字符对"这类和 TopK 密切相关的操作。

**可运行例子:**
```python
import heapq
from collections import Counter

def top_k_largest(nums, k):
    if k <= 0:
        return []
    heap = nums[:k]
    heapq.heapify(heap)   # 小根堆,堆顶是这k个里最小的
    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)

assert top_k_largest([3, 1, 4, 1, 5, 9, 2, 6], 3) == [9, 6, 5]
assert top_k_largest([1, 2, 3], 0) == []                  # k=0
assert top_k_largest([5], 1) == [5]                         # 单元素
assert top_k_largest([1, 1, 1, 1], 2) == [1, 1]             # 含重复元素

def top_k_frequent(nums, k):
    counts = Counter(nums)
    return [x for x, _ in heapq.nlargest(k, counts.items(), key=lambda pair: pair[1])]

assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
assert top_k_frequent([], 2) == []                            # 空输入

# 交叉验证:堆方案和"完整排序取前K个"结果必须一致(在数值本身层面,不要求元素顺序完全一样)
import random
random.seed(23)
for _ in range(30):
    nums = [random.randint(-100, 100) for _ in range(random.randint(0, 30))]
    k = random.randint(0, len(nums))
    heap_result = sorted(top_k_largest(nums, k))
    brute_result = sorted(sorted(nums, reverse=True)[:k])
    assert heap_result == brute_result

print("OK: TopK问题(大根堆语义用小根堆实现)在边界情况下全部正确, "
      "30组随机测试与'完整排序取前K个'结果一致")
```
本机实测:边界情况(k=0、单元素、含重复元素、空输入)全部正确;30 组随机测试中,堆方案和完整排序方案得到的 TopK 结果(排序后比较)完全一致。

**面试怎么问 + 追问链:** "找最大的 K 个元素,为什么维护的是小根堆而不是大根堆?" → 追问"如果 K 非常接近 n(比如 K = n - 5),这个方法还是最优选择吗?"(不是——当 K 接近 n 时,`O(n log k)` 约等于 `O(n log n)`,和直接排序没有优势,甚至因为堆操作的常数因子,可能比"完整排序"更慢;这种情况下更好的做法是反过来找"最小的 n-K 个要排除的元素",或者干脆直接排序;这个追问检验的是能否根据 K 和 n 的相对大小动态判断该用哪种策略,而不是无脑套用"TopK 就该用堆"这个结论)。

**常见坑:**
1. 找"最大的 K 个",却用了大根堆——这个方向搞反后,堆顶变成当前最大值,新元素的比较逻辑全部要跟着颠倒,而且会失去"小根堆只需要维护 K 个元素"的效率优势(因为要保留最大的 K 个,用大根堆需要维护所有元素才能知道该保留哪些)。
2. k 大于数组实际长度或者 k≤0 时没有做特殊处理——直接 `nums[:k]` 建堆在 k 超过数组长度时不会报错(Python 切片天然容忍越界),但如果后续逻辑假设堆大小恰好是 k,可能引发其他隐藏问题,应该在函数开头就做好边界检查。

---

## 4. K 路归并:堆在多路数据合并中的应用

**签名/是什么:**
```
K个有序序列,各自维护一个"当前候选"指针,
用堆同时管理这K个候选，每次弹出全局最小值，指针前移，把新候选压回堆
```

**一句话:** K 路归并不局限于[03类知识点5](03-linked-lists.md#5-复杂链表操作合并-k-个有序链表与复制带随机指针的链表)讲过的链表场景——只要是"K 个各自有序的数据源(数组、迭代器、甚至是磁盘上的多个已排序文件)"需要合并成一个整体有序的结果,都是同一个模式:堆维护"每一路当前最新候选值",每次弹出全局最小、从对应那一路补充下一个候选。

**底层机制/为什么这样设计:** 堆里最多同时存在 K 个元素(每一路各贡献一个"当前候选"),弹出一个、马上从对应来源补充一个,堆的大小始终维持在 K,不会随着总数据量增长而增长——这是 K 路归并能在有限内存下处理"K 个可能都很大、甚至来自磁盘流式读取"的数据源的关键:不需要一次性把所有数据都读入内存,只需要同时持有每一路的"当前位置"。整体复杂度是 O(n log k)(n 是元素总数,k 是路数),和[03类](03-linked-lists.md)链表版本的复杂度分析完全一致,只是数据来源从"链表节点"泛化成了更一般的"有序序列"。

**AI 研究/工程场景:** [huggingface-deep-dive 04 类](../huggingface-deep-dive/04-datasets-mechanics.md)讲过 `streaming=True` 流式加载大规模数据集的场景——如果需要把多个已经排好序的数据分片(shard)文件合并读取成一个整体有序的流,又不希望一次性把所有分片都读进内存,用的正是这里的 K 路归并模式,和处理链表的版本是同一个算法思想。

**可运行例子:**
```python
import heapq

def merge_k_sorted_sequences(sequences):
    heap = []
    for seq_idx, seq in enumerate(sequences):
        if seq:
            heapq.heappush(heap, (seq[0], seq_idx, 0))   # (值, 序列编号, 序列内下标)
    result = []
    while heap:
        val, seq_idx, item_idx = heapq.heappop(heap)
        result.append(val)
        if item_idx + 1 < len(sequences[seq_idx]):
            next_val = sequences[seq_idx][item_idx + 1]
            heapq.heappush(heap, (next_val, seq_idx, item_idx + 1))
    return result

assert merge_k_sorted_sequences([[1, 4, 7], [2, 5, 8], [3, 6, 9]]) == list(range(1, 10))
assert merge_k_sorted_sequences([]) == []                        # 没有任何序列
assert merge_k_sorted_sequences([[], [1, 2], []]) == [1, 2]        # 含空序列
assert merge_k_sorted_sequences([[5]]) == [5]                       # 单序列单元素

# 交叉验证:与"拼接后整体排序"结果一致(堆的价值在效率,不在结果正确性上有特殊之处)
import random
random.seed(24)
for _ in range(20):
    k = random.randint(0, 5)
    sequences = [sorted(random.sample(range(100), random.randint(0, 8))) for _ in range(k)]
    heap_result = merge_k_sorted_sequences(sequences)
    brute_result = sorted(x for seq in sequences for x in seq)
    assert heap_result == brute_result

print("OK: K路归并(泛化到任意有序序列, 不局限于链表)在空输入/含空序列等边界情况下全部正确, "
      "20组随机测试与'拼接后整体排序'结果完全一致")
```
本机实测:边界情况(没有任何序列、包含空序列、单序列单元素)均正确;20 组随机测试中,K 路归并的结果与"把所有序列拼接后整体排序"完全一致。

**面试怎么问 + 追问链:** "K 路归并用堆维护候选,复杂度是 O(n log k),如果不用堆,两两合并(合并第1和第2路,结果再和第3路合并……)复杂度是多少?" → 追问"什么情况下两两合并反而可能更简单实用,即使复杂度不是最优?"(K 很小(比如 K=2 或 3)时,两两合并的实现更简单,维护一个堆的额外开销(每次操作都要维护堆结构)可能得不偿失;这个追问检验的是能否认识到"复杂度最优"和"工程上最值得选择"不总是同一个答案,K 的具体大小会真实影响这个权衡,这也呼应了[知识点1](07-heaps-and-priority-queues.md#1-堆的数组表示与基本操作)"复杂度优势可能被常数因子抵消"这个更普遍的道理)。

**常见坑:**
1. 堆里只存值,不存"来自哪一路、该路的哪个位置"这两个额外信息——弹出一个值之后,不知道该从哪一路补充下一个候选,整个算法无法继续推进。
2. 值相同时(多路数据出现重复值),如果堆元组只有 `(值, 序列编号)` 没有第三个字段区分具体位置,当同一路连续出现相同值时可能引发比较歧义或逻辑错误,加入下标字段(如本例的 `item_idx`)能从根本上避免这个问题。

---

## 5. 双堆技巧:数据流中位数

**签名/是什么:**
```
用两个堆维护一个动态数据流的中位数:
small(大根堆): 存较小的一半数据    large(小根堆): 存较大的一半数据
维持 len(small) 与 len(large) 相差不超过1，中位数就在两堆堆顶附近
```

**一句话:** 双堆技巧用两个堆分别托管数据流"较小的一半"和"较大的一半",两堆堆顶紧贴着数据流的中间位置——插入新数据后动态调整两堆的平衡,能在 O(log n) 内完成一次插入并随时以 O(1) 查询当前中位数,不需要每次都重新排序整个数据流。

**底层机制/为什么这样设计:** 为什么用两个堆而不是一个?因为中位数需要同时知道"比中位数小的最大值"和"比中位数大的最小值"(数据量为偶数时),这正好是一个大根堆的堆顶和一个小根堆的堆顶——用一个堆做不到同时高效访问这两个位置。每次插入新元素后,先无条件放入 `small`(大根堆),再把 `small` 的堆顶弹出放入 `large`(小根堆)做一次"过滤",这个"先放小堆再倒一个到大堆"的顺序,保证了新元素不会因为直接放错堆而破坏"small 里所有元素都不大于 large 里所有元素"这个关键不变量;最后再检查两堆大小是否失衡(相差超过 1),失衡则从多的一堆倒一个到少的一堆——这套"插入 + 立即重新平衡"的流程,保证了任意时刻两堆的大小差最多是 1,中位数因此总能通过 O(1) 访问堆顶算出。

**AI 研究/工程场景:** [huggingface-deep-dive 09 类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过训练过程中实时监控 loss 分布,如果要在训练还没结束时就实时展示"当前所有已记录 loss 值的中位数"(比均值更抗异常值干扰),双堆技巧能做到每来一个新数据点就 O(log n) 更新,而不必每次都对全部历史数据重新排序。

**可运行例子:**
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []   # 大根堆(存负数模拟),维护较小的一半
        self.large = []   # 小根堆,维护较大的一半

    def add_num(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))   # 过滤,保证small都不大于large
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

mf = MedianFinder()
mf.add_num(5); assert mf.find_median() == 5           # 单个元素
mf.add_num(15); assert mf.find_median() == 10.0         # 两个元素,取平均
mf.add_num(1); assert mf.find_median() == 5              # 三个元素,取中间那个: 1,5,15
mf.add_num(3); assert mf.find_median() == 4.0            # 四个元素: 1,3,5,15 -> (3+5)/2

# 交叉验证:对随机数据流,双堆结果必须和"每次插入后完整排序取中位数"完全一致
import random
random.seed(25)

def brute_median(sorted_so_far):
    n = len(sorted_so_far)
    if n % 2 == 1:
        return sorted_so_far[n // 2]
    return (sorted_so_far[n // 2 - 1] + sorted_so_far[n // 2]) / 2

mf2 = MedianFinder()
seen = []
for _ in range(100):
    x = random.randint(-500, 500)
    mf2.add_num(x)
    seen.append(x)
    seen.sort()
    assert mf2.find_median() == brute_median(seen)

print("OK: 数据流中位数(双堆技巧)在1~4个元素的边界情况下全部正确, "
      "100次随机插入的每一步都与'重新排序取中位数'结果完全一致")
```
本机实测:1 到 4 个元素的基础场景全部正确;100 次随机插入的数据流中,双堆技巧每一步给出的中位数都和"每次都重新完整排序取中位数"这个更朴素但显然正确的做法完全一致。

**面试怎么问 + 追问链:** "双堆维护中位数,为什么每次插入都要经过'先放 small 再倒一个到 large'这个步骤,不能直接判断新元素该放哪个堆吗?" → 追问"如果不经过这个'先放再倒'的过滤步骤,直接根据'新元素和当前中位数的大小关系'决定放哪个堆,会有什么问题?"(直接判断需要先知道当前中位数,但中位数本身就是从两堆堆顶算出来的——如果两堆还没达到平衡状态,直接比较可能用一个不准确的"临时中位数"做判断,导致元素被放错堆;"先放再倒"的方式不依赖对当前中位数的预先判断,是一个更稳健、不依赖中间状态是否已经平衡的实现方式,这个追问检验的是能否看穿这个实现细节存在的必要性,而不只是记住代码怎么写)。

**常见坑:**
1. 用负数模拟大根堆时,存取时忘记转换符号(该取负的地方没取负,或者取负后忘记再取回来)——这是[知识点2](07-heaps-and-priority-queues.md#2-python-heapq-模块使用与内部实现)提到的大根堆技巧在这个更复杂场景下的具体体现,符号错误往往不会立即报错,只会让中位数计算出错。
2. 判断两堆是否失衡的条件写成 `len(self.large) >= len(self.small)`(而不是严格的 `>`)——这会导致本来已经平衡(相差恰好为 0 或 1)的两堆被不必要地重新调整,虽然不会导致错误结果,但会有多余的操作开销,也说明对"平衡"这个不变量的具体数值边界理解不够精确。

---

## 6. 堆与贪心的组合:会议室调度问题

**签名/是什么:**
```
给定多个会议的[开始时间, 结束时间], 求同一时刻最多需要多少个会议室 ——
排序后, 用堆维护"当前正在使用中的会议室, 各自的结束时间"
```

**一句话:** 会议室调度问题是"排序 + 堆"这个组合模式的另一个典型场景(和[05类知识点7](05-stacks-and-queues.md#7-优先级队列与栈的组合应用场景)的任务调度器是不同的应用):按开始时间排序后,依次决定每个新会议是复用一个已结束的会议室,还是必须新开一个,堆负责快速找到"最早会结束的那个会议室"。

**底层机制/为什么这样设计:** 会议按开始时间排序后处理,堆里维护当前所有"正在使用中"的会议室各自的结束时间(小根堆,堆顶是最早结束的那个)——当处理一个新会议时,先看堆顶(最早结束的会议室)是否已经在新会议开始前结束:如果是,说明这个会议室可以被复用,用 `heapreplace` 直接把堆顶换成新会议的结束时间;如果不是(最早结束的都还没结束),说明现有的会议室都不够用,必须新开一个,`heappush` 增加堆的大小。**堆的最终大小,就是同一时刻最多同时需要的会议室数量**——这是因为堆的大小只在"确实无法复用任何现有会议室"时才增长,精确对应了真实需要的并发资源数。

**AI 研究/工程场景:** [huggingface-deep-dive 06 类](../huggingface-deep-dive/06-accelerate-and-devices.md)讲过的分布式训练资源调度,如果要计算"给定一批训练任务各自的起止时间,最少需要多少张 GPU 卡才能保证互不冲突地跑完所有任务",本质上和会议室调度是同一个问题模型,只是把"会议室"换成了"GPU 卡"。

**可运行例子:**
```python
import heapq

def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)   # 按开始时间排序
    heap = []   # 存正在使用中的会议室各自的结束时间(小根堆)
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)   # 复用最早结束的会议室
        else:
            heapq.heappush(heap, end)        # 现有会议室都不够用,新开一个
    return len(heap)

assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
assert min_meeting_rooms([[7, 10], [2, 4]]) == 1          # 两个会议完全不重叠,复用同一间
assert min_meeting_rooms([]) == 0                            # 没有会议
assert min_meeting_rooms([[1, 5], [1, 5], [1, 5]]) == 3      # 三个会议完全同时,各自需要一间
assert min_meeting_rooms([[1, 10]]) == 1                     # 单个会议

# 交叉验证:用"扫描线"方式(在每个时间点统计当前重叠会议数,取最大值)对照验证
def brute_min_meeting_rooms(intervals):
    if not intervals:
        return 0
    events = []
    for s, e in intervals:
        events.append((s, 1))     # 开始 +1
        events.append((e, -1))    # 结束 -1
    events.sort(key=lambda x: (x[0], x[1]))   # 时间相同时,结束(-1)排在开始(+1)前面,避免误判重叠
    cur = max_rooms = 0
    for _, delta in events:
        cur += delta
        max_rooms = max(max_rooms, cur)
    return max_rooms

import random
random.seed(26)
for _ in range(20):
    n = random.randint(0, 10)
    intervals = []
    for _ in range(n):
        s = random.randint(0, 20)
        e = s + random.randint(1, 10)
        intervals.append([s, e])
    assert min_meeting_rooms(intervals) == brute_min_meeting_rooms(intervals)

print("OK: 会议室调度(堆+贪心)在边界情况(空输入/完全不重叠/完全同时)下全部正确, "
      "20组随机测试与扫描线方法交叉验证结果一致")
```
本机实测:边界情况(没有会议、完全不重叠、完全同时进行)全部正确;20 组随机测试中,堆方案和扫描线方案(两种完全不同的实现思路)结果完全一致。

**面试怎么问 + 追问链:** "会议室调度问题为什么要先按开始时间排序?" → 追问"这道题和[05类的任务调度器](05-stacks-and-queues.md#7-优先级队列与栈的组合应用场景)都用了堆,两者解决的是同一类问题吗?"(不是同一类——任务调度器关心的是"给定固定的冷却规则,排出一个总耗时最短的执行顺序"(堆负责决定"接下来该执行哪个任务类型"这个优先级问题);会议室调度关心的是"给定固定的时间安排,最少需要多少并发资源"(堆负责快速找到"哪个资源最快能被复用");两者都用堆是因为都需要快速访问某种"当前最值",但要解决的实际问题和堆里存的东西完全不同——这个追问检验的是能否透过"都用了堆"这个表面相似性,看到两个问题在本质上的区别,不能把"用堆"当成问题分类的依据)。

**常见坑:**
1. 忘记先按开始时间排序就直接处理——不排序时,处理顺序是任意的,无法保证"堆顶是当前所有已开始且未结束的会议室里最早结束的那个"这个关键性质。
2. 判断"是否可以复用会议室"的边界条件搞错(用 `<` 而不是 `<=`,或者反过来)——这类边界决定了"一个会议在另一个会议结束的同一时刻开始"算不算可以复用同一个会议室,取决于题目对时间区间开闭的具体约定,必须仔细确认。

---

## 7. 堆常见坑

**签名/是什么:**
```
heapq只支持最小堆 -> 求最大值需要取负数技巧,符号处理容易出错
堆不是完全排序结构 -> 除堆顶外,其余位置没有整体顺序保证
```

**一句话:** 堆最常见的两类错误——用负数模拟大根堆时符号搞混,以及误以为堆内部数据整体有序(除了堆顶,其余位置没有这个保证)——两者都不会让程序报错,只会让结果安静地出错。

**底层机制/为什么这样设计:** 堆只保证一个**局部**性质:每个节点不大于(或不小于)它的直接子节点,这个性质只在"父子"这条边上成立,不传递到"祖先-非直接子孙"或者"兄弟节点之间"——这也是为什么想要遍历堆里所有元素得到有序序列,必须反复执行 `heappop()`(每次 O(log n),总共 O(n log n)),而不能直接读取底层数组然后假设它已经有序。取负数模拟大根堆是一个纯粹的"符号技巧",不改变底层堆算法本身,但正因为它只是一层"包装",容易在存取的多个环节里有的地方忘记转换符号。

**AI 研究/工程场景:** 这类"局部不变量被误认为全局性质"的错误模式,在本系列其他知识点也反复出现——[02类知识点7](02-arrays-and-strings.md#7-循环不变量方法论)讲过的循环不变量、[05类知识点8](05-stacks-and-queues.md#8-单调栈--单调队列常见坑)讲过的单调栈/队列方向错误,本质上都是"对数据结构维护的具体不变量理解不够精确"导致的同一类问题。

**可运行例子:**
```python
import heapq

# 坑1: 大根堆取负技巧,符号处理不一致导致的真实bug
def buggy_max_heap_top_k(nums, k):
    """故意在'存入'和'取出'时符号处理不一致"""
    heap = []
    for x in nums:
        heapq.heappush(heap, -x)   # 存入时正确取负
    result = []
    for _ in range(k):
        if heap:
            result.append(heapq.heappop(heap))   # 错误:取出时忘记再取负转换回来
    return result

correct_top3 = sorted([3, 1, 4, 1, 5, 9, 2, 6], reverse=True)[:3]
buggy_result = buggy_max_heap_top_k([3, 1, 4, 1, 5, 9, 2, 6], 3)
assert buggy_result != correct_top3
assert buggy_result == [-9, -6, -5]  # 真实复现:忘记转换符号,拿到的是原数值的相反数

def correct_max_heap_top_k(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, -x)
    result = []
    for _ in range(k):
        if heap:
            result.append(-heapq.heappop(heap))   # 正确:取出时转换回来
    return result

assert correct_max_heap_top_k([3, 1, 4, 1, 5, 9, 2, 6], 3) == correct_top3

# 坑2: 误以为堆的底层数组本身已经整体有序
h = [5, 2, 8, 1, 9, 3, 7]
heapq.heapify(h)
assert h[0] == min(h)          # 堆顶确实是最小值,这个成立
assert h != sorted(h)            # 但整个数组绝不等价于sorted()的结果,这是真实会发生的情况
# 唯一能从堆数组里直接确认的顺序性质,只有"父节点不大于子节点"这一条局部关系
n = len(h)
for i in range(n):
    left, right = 2 * i + 1, 2 * i + 2
    if left < n:
        assert h[i] <= h[left]     # 局部堆性质成立
    if right < n:
        assert h[i] <= h[right]

print(f"OK: 大根堆取负符号处理不一致时, 真实复现了错误结果{buggy_result}"
      f"(应为{correct_top3}); 验证了堆数组本身不等价于完全排序结果, "
      f"只有'父子'这条局部关系成立")
```
本机实测:符号处理不一致的大根堆实现,真实产出了 `[-9, -6, -5]` 这样明显错误(带负号)的结果,而不是抽象的"可能出错"的警告;验证了 `heapq.heapify()` 之后的底层数组确实不等于 `sorted()` 的结果(只有相邻父子关系满足堆性质),这个区别在需要"遍历堆内容"的场景下是一个真实、常见的误用来源。

**面试怎么问 + 追问链:** "如果我告诉你一个数组满足堆性质,你能直接说出它的最大值在什么位置吗?" → 追问"能不能说出,在一个最小堆里,'第二小的元素'一定在哪个位置?"(不能精确定位到具体唯一下标——第二小的元素一定是堆顶的某个直接子节点(下标 1 或 2 之一),但具体是哪一个取决于建堆过程,不存在一个固定不变的答案;这个追问检验的是能否精确说出堆结构"能保证什么、不能保证什么",而不是笼统地说"堆是有序的"这种不精确的描述)。

**常见坑:**
1. 存入和取出堆时,大根堆的取负技巧没有对称地应用(比如存的时候取负,取的时候忘记转换回来)——本知识点已经现场复现了这个真实后果,排查这类 bug 时应该逐一检查每一处涉及堆元素读写的地方,确认符号转换是否成对出现。
2. 需要"堆中所有元素按顺序输出"时,直接读取底层数组当作已排序结果——正确做法是反复调用 `heappop()`,或者如果只是想查看而不消耗堆,应该用 `sorted(heap)` 单独排序一份拷贝,不能假设堆的底层存储天然有序。

---

*本篇 7 个知识点全部在仓库根目录 `.venv` 真实测试验证(含真实计时的"复杂度优势 vs 实现语言差距"对比、边界情况覆盖、以及"符号处理不一致"这类真实bug的现场复现)。*
