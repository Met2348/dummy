# 04 高级并发模式与无锁编程

> 板块 II:并发同步与死锁。03 类讲了基础同步原语,本类往上一层,讲这些原语组合出来的高级并发设计模式,以及"不用锁"这条路径本身。

---

## 1. CAS与ABA问题

**签名/是什么**

CAS(Compare-And-Swap,比较并交换)是硬件提供的一条原子指令:`CAS(内存地址, 期望值, 新值)`——只有当内存地址当前的值等于期望值时,才把它替换成新值,并返回是否成功,整个"比较+替换"过程不可被打断。ABA 问题指:一个值从 A 变成 B 又变回 A,单纯基于"当前值是否等于期望值"的 CAS 检查会误判"没有发生过变化",但实际上中间已经发生了别的线程的修改,如果这次修改改变了某些不体现在"值"本身的状态(比如链表节点被换成了另一个恰好持有相同值的新节点),会导致错误的结果。

**一句话**

CAS 是"无锁编程"的基石指令,ABA 问题是"只比较值、不比较历史"这种朴素判断方式的致命盲区。

**底层机制/为什么这样设计**

CAS 之所以能替代锁,是因为它把"检查条件+修改"这个复合操作压缩成一条硬件保证原子性的指令,配合"失败就重试"的循环,可以在不阻塞任何线程的情况下实现线程安全——这是无锁编程(Lock-Free Programming)的核心手法:不是"没有同步",而是"同步的实现方式换成了乐观重试而不是悲观加锁"。ABA 问题的根源在于 CAS 只能比较"当前值"这一个快照,无法感知"这个值从上次检查到现在,有没有经历过任何中间变化"——如果两次检查之间值虽然回到了原样,但依赖的其他不变量(比如某个节点对象的身份、或者与这个值关联的其他状态)已经被破坏,朴素 CAS 会给出错误的"一切正常"判断。

**AI研究/工程场景**

高性能推理服务里的无锁计数器/无锁队列(比如统计已处理请求数、批处理任务队列)大量使用 CAS 循环重试而不是加锁,以避免高并发下频繁加锁解锁的开销;但如果这类无锁结构里有"节点复用"(为了减少内存分配,把用过的节点放回一个空闲池重新使用)这类优化,就必须特别小心 ABA 问题——一个节点被回收后又被复用、恰好装的还是原来那个逻辑值,如果没有额外的版本号保护,基于 CAS 的无锁数据结构可能因此产生错误的接入(经典学术文献里这被称为"内存复用导致的 ABA",是无锁数据结构工程实现里公认的高难度陷阱之一)。

**可运行例子**(验证环境:`.venv`;Python 没有暴露原生 CAS 指令,这里用锁模拟"比较并交换"这个原子操作的语义,重点验证 ABA 问题的逻辑本身,不是验证真实硬件指令)

```python
import threading

class AtomicRef:
    def __init__(self, value):
        self._value = value
        self._lock = threading.Lock()
    def get(self):
        with self._lock:
            return self._value
    def compare_and_swap(self, expected, new_value):
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

ref = AtomicRef(10)
assert ref.compare_and_swap(10, 20) == True, "CAS should succeed when current value matches expected"
assert ref.compare_and_swap(10, 30) == False, "CAS should fail once the value no longer matches expected"
print("CAS_BASIC_TEST=PASS")

# ABA问题:值从A变成B又变回A,朴素CAS("值还是A吗")会误判"没人动过"
aba_ref = AtomicRef('A')
aba_ref.compare_and_swap('A', 'B')  # 模拟另一个线程抢先做了 A -> B
aba_ref.compare_and_swap('B', 'A')  # 又做了 B -> A
naive_cas_succeeds = aba_ref.compare_and_swap('A', 'FINAL')  # 朴素CAS只看"现在是不是A"
print('naive_cas_succeeds=%s current_value=%s' % (naive_cas_succeeds, aba_ref.get()))
assert naive_cas_succeeds == True, "naive value-only CAS incorrectly succeeds even though the value went A->B->A in between - this IS the ABA problem"
print("ABA_PROBLEM_TEST=PASS")

# 解决方案:带版本号的CAS能正确检测出"值虽然看起来一样,但已经被改过"
class VersionedAtomicRef:
    def __init__(self, value):
        self._value = value
        self._version = 0
        self._lock = threading.Lock()
    def get_with_version(self):
        with self._lock:
            return self._value, self._version
    def compare_and_swap(self, expected_value, expected_version, new_value):
        with self._lock:
            if self._value == expected_value and self._version == expected_version:
                self._value = new_value
                self._version += 1
                return True
            return False

vref = VersionedAtomicRef('A')
val, ver = vref.get_with_version()
vref.compare_and_swap('A', 0, 'B')
vref.compare_and_swap('B', 1, 'A')
versioned_cas_succeeds = vref.compare_and_swap(val, ver, 'FINAL')
print('versioned_cas_succeeds=%s' % versioned_cas_succeeds)
assert versioned_cas_succeeds == False, "versioned CAS correctly detects the intermediate A->B->A change via the version number, preventing the ABA bug"
print("VERSIONED_CAS_FIX_TEST=PASS")
```

**面试怎么问+追问链**

- **方案批判迭代轴**:"我们用 CAS 实现了一个无锁计数器"——追问1:计数器场景(值单调递增或递减)一般不会遇到 ABA 问题,为什么?候选人如果答不上来——追问2:ABA 问题的前提是"值可能变回原来的样子",纯粹递增的计数器值不会"变回去",天然不受 ABA 影响;真正需要警惕 ABA 的是那些值可能循环往复、或者对象会被复用的场景(如链表节点、对象池),这条追问检验候选人是否理解 ABA 问题的适用边界,而不是把它当成所有 CAS 场景都要担心的万能警告。

**常见坑**

- 把 ABA 问题当成"CAS 本身有 bug"——CAS 指令本身的原子性保证是完全正确的,ABA 是"只用值做判断依据"这个更高层设计选择的局限,解法是加入额外的、单调变化的信息(版本号、指针的高位打标记等),不是放弃使用 CAS。

---

## 2. 无锁队列初步

**签名/是什么**

无锁数据结构(Lock-Free Data Structure)是不使用互斥锁,而是用 CAS 循环重试来保证并发正确性的数据结构。Treiber 栈是最经典的入门级无锁数据结构:用一个指向栈顶节点的原子引用,`push`/`pop` 都通过"读取当前栈顶→构造新状态→CAS 尝试替换,失败就重读重试"这个模式实现。

**一句话**

无锁结构不是"没有同步开销",而是把"排队等锁"换成了"乐观地假设不冲突,冲突了就重试"。

**底层机制/为什么这样设计**

传统加锁的问题在于:如果持锁线程被操作系统调度器暂停(比如时间片用完、或者更糟——被抢占后很久都没被重新调度),所有等待这把锁的线程都会被阻塞陪跑,即使它们本可以继续做别的事。无锁结构完全不存在"某个线程持有资源导致别人无法前进"这个问题——每个线程各自尝试用 CAS 完成自己的操作,失败了就基于最新状态重试,不会因为另一个线程被挂起而被拖累(这个性质叫 lock-freedom,保证系统整体总能取得进展,即使个别线程的重试次数没有上限)。代价是:高竞争场景下,大量线程同时重试可能导致 CAS 失败率很高、有效工作占比反而下降,并且无锁结构的正确性证明和实现难度显著高于加锁方案,真正工业级的无锁队列(如 Michael-Scott 队列)需要比 Treiber 栈复杂得多的设计来处理队列特有的头尾指针协调问题。

**AI研究/工程场景**

高吞吐量的任务分发系统(比如把待处理的推理请求分发给多个 worker 线程)如果用传统加锁队列,在极高并发下锁本身会成为瓶颈;换成无锁队列可以让多个生产者/消费者线程真正并行推进,不会因为某个线程持锁期间被系统调度器"晾在一边"而拖慢所有人——这是"高吞吐、低延迟服务"这类系统在锁竞争成为可观测瓶颈之后,才值得投入的进阶优化手段(不是所有并发场景的默认最优选择,见"常见坑")。

**可运行例子**(验证环境:`.venv`;真实无锁数据结构依赖硬件 CAS 指令,这里继续用第 1 点的锁模拟 CAS 语义来验证 Treiber 栈的正确性逻辑,不是验证真实硬件层面的无锁性能)

```python
import threading

class Node:
    __slots__ = ('value', 'next')
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

class LockFreeStack:
    def __init__(self):
        self._top = None
        self._cas_lock = threading.Lock()  # 仅模拟硬件CAS指令的原子性,不是用来互斥保护整个push/pop流程

    def _compare_and_swap_top(self, expected, new):
        with self._cas_lock:
            if self._top is expected:
                self._top = new
                return True
            return False

    def push(self, value):
        while True:
            old_top = self._top
            new_node = Node(value, old_top)
            if self._compare_and_swap_top(old_top, new_node):
                return

    def pop(self):
        while True:
            old_top = self._top
            if old_top is None:
                return None
            new_top = old_top.next
            if self._compare_and_swap_top(old_top, new_top):
                return old_top.value

stack = LockFreeStack()
pushed_items = []
push_lock = threading.Lock()

def pusher(tid, count):
    for i in range(count):
        item = (tid, i)
        stack.push(item)
        with push_lock:
            pushed_items.append(item)

n_threads, count_per_thread = 6, 300
threads = [threading.Thread(target=pusher, args=(t, count_per_thread)) for t in range(n_threads)]
for t in threads: t.start()
for t in threads: t.join()

popped_items = []
while True:
    item = stack.pop()
    if item is None:
        break
    popped_items.append(item)

print('pushed=%d popped=%d' % (len(pushed_items), len(popped_items)))
assert len(popped_items) == len(pushed_items) == n_threads * count_per_thread, \
    "lock-free stack must not lose any pushed item even under 6-thread concurrent pushing without a traditional mutex protecting the whole push operation"
assert sorted(popped_items) == sorted(pushed_items), "every pushed item must be popped exactly once, no duplication"
print("LOCK_FREE_STACK_TEST=PASS")
```

**面试怎么问+追问链**

- **决策依据追问轴**:什么时候该用无锁结构,什么时候老老实实用锁?——追问:无锁结构的实现和验证复杂度远高于加锁方案,只有在"锁竞争已经被实测证明是瓶颈"且"操作本身足够简单能用 CAS 表达"的场景才值得投入,盲目追求无锁是常见的过早优化——这条追问检验候选人有没有"性能优化要基于测量"这个基本工程素养,而不是把"无锁"当成炫技指标。

**常见坑**

- 认为无锁一定比加锁快——高竞争场景下,大量线程反复 CAS 失败重试消耗的 CPU 周期可能比线程阻塞等待(不占 CPU,见 03 类知识点3自旋锁的教训同理)更多,无锁结构的性能优势主要体现在"低到中等竞争"且"临界区操作简单"的场景,竞争激烈时不一定占优。

---

## 3. 线程池设计(核心线程数/队列策略/拒绝策略)

**签名/是什么**

线程池(Thread Pool)预先创建一组可复用的工作线程,任务提交后从池中分配线程执行,避免"每个任务都创建销毁一个新线程"的开销。核心设计参数包括:核心线程数(池中常驻的最小线程数)、最大线程数(负载高时能临时扩展到的上限)、任务队列(核心线程都忙时,新任务排队等待的容器,策略上可以是有界队列或无界队列)、拒绝策略(队列也满了、线程数也到上限了,新任务应该怎么处理——直接丢弃、抛异常、调用者自己执行、还是丢弃队列里最老的任务腾地方)。

**一句话**

线程池用"提前准备好一批可复用的工人"替代"来一个任务就现雇一个工人、干完就辞退",省下反复创建销毁线程的真实开销。

**底层机制/为什么这样设计**

创建一个操作系统线程有实打实的开销(内核对象分配、栈内存分配、初始化调度信息),如果任务量大且单个任务耗时很短,这份开销占比会变得不可忽视甚至反客为主。线程池把"线程的生命周期管理"和"任务的执行调度"解耦:线程创建一次,反复处理不同的任务;核心线程数决定"稳态下能同时干多少活",队列决定"来不及处理的任务先存哪里、存多少",拒绝策略决定"系统真的扛不住了应该怎么应对而不是无限制地让请求堆积拖垮整个系统"——这几个参数共同决定了系统在过载时的行为特征,是"优雅降级"设计思想在并发资源管理层面的具体体现。

**AI研究/工程场景**

模型推理服务的请求处理线程池是最直接的应用场景:核心线程数通常和可用 CPU 核心数/GPU 并发处理能力挂钩,设置过大会导致过多线程争抢有限的计算资源(见 02 类知识点8的调度开销),设置过小会导致明明有空闲资源却排队;拒绝策略的选择直接决定服务过载时的行为——是快速失败返回"服务繁忙"(保护系统不被压垮)还是让请求排队等待(可能导致所有请求的延迟都被拖累),这是设计高并发推理服务时必须显式做出的工程决策,不存在放之四海皆准的默认值。

**可运行例子**(验证环境:`.venv`)

```python
import threading
import concurrent.futures
import time

def thread_pool_concurrency_test():
    max_concurrent_seen = [0]
    current_concurrent = [0]
    lock = threading.Lock()
    def task():
        with lock:
            current_concurrent[0] += 1
            max_concurrent_seen[0] = max(max_concurrent_seen[0], current_concurrent[0])
        time.sleep(0.1)
        with lock:
            current_concurrent[0] -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task) for _ in range(10)]
        concurrent.futures.wait(futures)
    return max_concurrent_seen[0]

max_seen = thread_pool_concurrency_test()
print('max_concurrent_seen=%d (pool max_workers=3)' % max_seen)
assert max_seen == 3, "thread pool with max_workers=3 should never run more than 3 tasks concurrently, even with 10 tasks submitted at once - excess tasks queue and wait"
print("THREAD_POOL_LIMIT_TEST=PASS")
```

**面试怎么问+追问链**

- **规模递增轴**:任务量从 10 个变成 1000 万个,线程池设计需要重新考虑什么?——追问:10 个任务下,`max_workers=3` 加一个无界队列完全够用;1000 万个任务下,如果用无界队列,即使线程数限制住了,排队等待的任务对象本身会持续占用内存直到被处理,可能在真正执行完之前就把内存耗尽——这时候需要有界队列 + 明确的拒绝策略(比如让提交任务的调用方感知到"系统已经饱和,应该自己限流或重试"),这条追问在检验候选人是否理解"限制并发数"和"限制排队总量"是两个独立的、都需要考虑的资源边界。
- **诊断真实数据(新题型)**:线上服务监控显示线程池的活跃线程数长期贴着 `max_workers` 的上限,同时任务队列长度持续增长——追问:这说明消费(处理任务)速度长期跟不上生产(提交任务)速度,需要判断根因是"单个任务本身处理太慢"(需要优化任务内部逻辑或增加 `max_workers`)还是"整体流量超出系统设计容量"(需要限流/扩容/降级),不能简单地"调大线程数"了事——如果任务本身是 CPU 密集型,盲目调大线程数在核心数有限的机器上反而会因为过度的上下文切换让情况更糟(见 02 类知识点8)。

**常见坑**

- 把 `max_workers` 设置得远超 CPU 核心数,期望"线程越多处理越快"——对 CPU 密集型任务,线程数超过核心数之后不会带来额外的真实并行度(还会受 GIL 限制,见 01 类知识点3),反而增加调度开销;线程池的合理线程数需要结合任务是 CPU 密集还是 IO 密集来定(IO 密集型可以远超核心数,因为大部分线程时间花在等待上不占用 CPU)。

---

## 4. 协程调度器实现思想

**签名/是什么**

协程调度器是负责在多个协程之间轮流切换执行权的组件——`asyncio` 的事件循环本质上就是一个协程调度器。理解其实现思想的最小方式是用 Python 生成器(`yield`)手写一个极简的协作式调度器("trampoline"模式):每个"协程"是一个生成器,调度器循环调用 `next()` 推进它,生成器每次 `yield` 就是主动交还控制权的时刻。

**一句话**

`asyncio` 的 `await` 表面上是新语法,底层做的事情和"生成器 `yield` 一下、外层循环再 `next()` 一下"是同一件事——调度器反复"推一下、看它要不要继续"。

**底层机制/为什么这样设计**

生成器天然具备"执行到一半可以暂停、之后从暂停点继续"的能力(每次调用 `next()` 运行到下一个 `yield` 就暂停,状态——包括局部变量、执行到哪一行——都被保留在生成器对象内部),这正是协程需要的核心能力。一个最简单的调度器只需要维护一个"待运行任务"列表,循环取出一个任务推进一步(调用 `next()`),如果它还没结束就放回列表末尾等下一轮,结束了(抛出 `StopIteration`)就不再放回——这就是一个基本能工作的协作式轮转调度器。`asyncio` 的真实实现远比这复杂(需要处理 IO 事件通知、定时器、异常传播、`Future`/`Task` 封装等),但核心的"反复推进生成器直到它暂停或结束"这个调度思想和这个最小实现是完全一致的,理解这一点能让"await 到底在干什么"这个问题不再是黑箱。

**AI研究/工程场景**

理解协程调度器的最小实现思想,有助于理解为什么 `asyncio` 代码里"忘记 `await` 一个协程"会导致它完全不执行(生成器不会自己运行,必须被调度器反复 `next()` 推进)——这是异步 Python 代码里一个极其高频的真实 bug:写了 `async def foo(): ...` 但调用时忘了 `await foo()`,只是创建了一个协程对象(相当于创建了生成器但没有任何调度器去推进它),代码不报错但也什么都没发生,新手极容易在这里卡住,而理解"协程本质是需要被外部调度器主动推进的生成器"能立刻定位这类问题。

**可运行例子**(验证环境:`.venv`)

```python
def task_a():
    for i in range(3):
        print('A step %d' % i)
        yield  # 让出控制权给调度器

def task_b():
    for i in range(3):
        print('B step %d' % i)
        yield

class MiniScheduler:
    def __init__(self):
        self.tasks = []
        self.execution_order = []
    def add(self, gen, name):
        self.tasks.append((name, gen))
    def run(self):
        while self.tasks:
            name, gen = self.tasks.pop(0)
            try:
                next(gen)
                self.execution_order.append(name)
                self.tasks.append((name, gen))  # 没结束,放回队尾等下一轮
            except StopIteration:
                pass  # 任务结束,不再放回

sched = MiniScheduler()
sched.add(task_a(), 'A')
sched.add(task_b(), 'B')
sched.run()
print('execution_order=%s' % sched.execution_order)
assert sched.execution_order == ['A', 'B', 'A', 'B', 'A', 'B'], \
    "the mini round-robin cooperative scheduler interleaves A and B exactly at each yield point - the same interleaving principle asyncio's event loop applies to coroutines under the hood"
print("MINI_SCHEDULER_TEST=PASS")
```

**面试怎么问+追问链**

- **底层机制追问轴**:`asyncio.sleep(1)` 底层是真的让这个协程"睡着"占用一个线程等 1 秒吗?——追问:不是,`asyncio.sleep()` 内部把当前协程挂起并向事件循环注册一个"1 秒后到期的定时器",然后立刻把控制权交还给调度器去运行其他就绪的协程;事件循环内部维护一个按到期时间排序的定时器队列,每一轮循环检查"有没有定时器到期了/有没有 IO 事件就绪了",到期的协程才会被重新推进——这也是为什么协程的"睡眠"完全不会阻塞其他协程,而线程的 `time.sleep()` 会阻塞调用它的那个线程(操作系统真的把这个线程挂起不参与调度,见 01 类知识点1)。

**常见坑**

- 在协程函数里调用同步阻塞的 `time.sleep()` 而不是 `await asyncio.sleep()`——这会真正阻塞当前线程(通常是承载整个事件循环的唯一线程),导致所有其他协程都被卡住,这是"协程和普通函数混用时最容易踩的坑",本质原因就是本知识点讲的调度机制:`time.sleep()` 不会主动把控制权交还给调度器(它是同步的,没有 `yield` 语义),调度器压根没有机会去推进其他任务。

---

## 5. 并发容器设计模式

**签名/是什么**

并发安全容器(Concurrent-Safe Container)指内部已经实现好同步保护、可以直接被多线程安全访问而不需要调用方自己额外加锁的数据结构,比如 Python 标准库的 `queue.Queue`(内部用锁和条件变量保护)。

**一句话**

用一个"已经想清楚怎么保护自己"的容器,比要求每一处使用它的代码都记得正确加锁,可靠得多。

**底层机制/为什么这样设计**

如果一个数据结构本身不是并发安全的,每一处使用它的代码都要自己记得在访问前后加锁解锁——这种"责任分散"的模式极其容易出错:只要有一处代码忘了加锁,整个保护就形同虚设,而且这类遗漏往往只在真实并发场景下才会暴露(见 03 类知识点1),代码审查很难发现。把同步逻辑封装进容器内部、对外暴露"本来就是线程安全的"接口,是把"正确性责任"从"每一处调用方"收敛到"容器自身实现"这一处,大幅降低了出错面。这也是"最小知识原则"在并发编程里的具体体现——调用方不需要知道、也不应该需要关心容器内部用了什么同步机制,只需要知道"这是并发安全的"这个契约。

**AI研究/工程场景**

多线程数据预处理流水线里,worker 线程之间传递数据的队列几乎总是直接用标准库的 `queue.Queue` 而不是手写一个 list 加锁,除了省去手写同步逻辑的麻烦,更重要的是 `queue.Queue` 已经是经过广泛验证的成熟实现,自己手写在真正的高并发场景下更容易留下未被发现的边界情况漏洞。

**可运行例子**(验证环境:`.venv`)

```python
import threading
import queue as queue_module

def concurrent_container_test(use_safe_queue):
    if use_safe_queue:
        container = queue_module.Queue()
        def push(item): container.put(item)
        def drain():
            items = []
            while not container.empty():
                try:
                    items.append(container.get_nowait())
                except queue_module.Empty:
                    break
            return items
    else:
        container = []
        def push(item): container.append(item)  # 单次 list.append 在CPython里因GIL天然原子
        def drain(): return list(container)

    n_threads, items_per_thread = 8, 500
    def worker(tid):
        for i in range(items_per_thread):
            push((tid, i))
    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()
    return drain()

result_queue = concurrent_container_test(use_safe_queue=True)
result_list = concurrent_container_test(use_safe_queue=False)
expected_total = 8 * 500
print('queue.Queue total=%d list.append total=%d expected=%d' % (len(result_queue), len(result_list), expected_total))
assert len(result_queue) == expected_total, "queue.Queue must reliably receive every item with no loss under concurrent access"
assert len(result_list) == expected_total, \
    "list.append is ALSO safe here because CPython's GIL makes a SINGLE append() call atomic - a real, non-obvious CPython detail, but this is fragile and operation-specific, not a general guarantee (see common pitfalls)"
print("CONCURRENT_CONTAINER_TEST=PASS")
```

**面试怎么问+追问链**

- **真实性验证轴**:验证代码显示朴素 `list.append` 在这个测试里"也是安全的",这是不是意味着 Python 的 list 天生线程安全,不需要用 `queue.Queue`?——追问:这是一个需要谨慎回答的陷阱问题——单次 `append()` 调用确实因为 CPython 的 GIL 而对这一个操作是原子的,但这只覆盖"添加"这一个动作;一旦涉及"读取长度再决定要不要添加"(检查-然后-操作)、或者"弹出并处理"这类复合操作,GIL 提供的单指令原子性完全不够用(参见 03 类知识点1 的"读-改-写"问题),`queue.Queue` 提供的是对"生产者消费者"这整套语义的正确并发保护(包括阻塞式的 `get`/`put`、边界检查),不能简单地用"list.append 恰好安全"来替代专门设计的并发容器。

**常见坑**

- 把"某个具体操作恰好因为实现细节而安全"泛化成"这类数据结构整体上是并发安全的"——这是一个真实存在的认知陷阱,`list.append`/`list.pop()`(无参数,从末尾弹出)在 CPython 里确实是原子的,但 `list[i] = x`(带索引访问)、`list.sort()`、以及任何"先读取状态再基于状态做决策"的复合操作都不享受同样的保证,依赖这种"侥幸安全"而不使用专门设计的并发容器,是脆弱且容易在未来被无意间破坏的做法。

---

## 6. 乐观锁到数据库场景的映射

**签名/是什么**

乐观锁(Optimistic Locking)不在读取数据时加锁阻止别人访问,而是在提交更新时检查数据自读取以来是否被别人修改过(通常靠版本号或时间戳字段实现)——如果被改过就拒绝这次更新(通常交由上层重试),没被改过则正常提交。这是本类前几点(CAS、版本号解决 ABA)在数据库并发控制场景下的具体应用形式,和 01 类知识点6"进程vs线程vs协程选型"呼应的思路一样,是同一套并发思想在不同层面的复用。

**一句话**

乐观锁赌的是"大概率不会冲突,冲突了再说",相对于"先把资源锁住谁都别想动"的悲观锁,是完全不同的风险取舍。

**底层机制/为什么这样设计**

传统的悲观锁(比如数据库的行锁)在读取数据时就加锁,直到事务结束才释放,这在读多写少、冲突概率低的场景下会造成不必要的等待——大部分读操作根本不会真的和别的写操作冲突,却要为了"万一冲突"付出加锁解锁的开销和阻塞代价。乐观锁把这个赌注反过来:读的时候完全不加锁(不阻塞任何人),只在真正要写回的那一刻才检查"我读到的版本,和现在数据库里的版本,是不是同一个"——如果是,说明这段时间没人动过,可以安全写入;如果不是,说明有人抢先修改了,这次更新作废,交给应用层决定是重试还是放弃。这本质上和第 1 点的 CAS+版本号解决 ABA 问题是同一个数学结构,只是把"内存里的一个值"换成了"数据库里的一行记录"。

**AI研究/工程场景**

多个训练任务并发更新同一份共享的实验元数据记录(比如"当前最优模型指标"这类需要被多个并行运行的评测任务更新的记录)是乐观锁的典型应用场景——评测任务量大但真正触发"打破当前最优记录"的写入很少,用乐观锁(每次更新先检查版本号)既避免了所有评测任务读取阶段互相阻塞,又能在真正发生并发写冲突时正确检测出来并重试,而不是简单的"后写的覆盖先写的"造成结果丢失。

**可运行例子**(验证环境:`.venv`)

```python
class OptimisticRecord:
    def __init__(self, value):
        self.value = value
        self.version = 0
    def read(self):
        return self.value, self.version
    def update(self, new_value, expected_version):
        if self.version != expected_version:
            return False  # 版本不匹配,拒绝更新(有其他人先改过了)
        self.value = new_value
        self.version += 1
        return True

record = OptimisticRecord(100)
val_tx1, ver_tx1 = record.read()  # 两个"事务"并发读到同一版本 (100, 0)
val_tx2, ver_tx2 = record.read()  # (100, 0)

tx1_success = record.update(val_tx1 + 10, ver_tx1)  # tx1先提交
tx2_success = record.update(val_tx2 + 20, ver_tx2)  # tx2基于过期版本提交,应被拒绝

print('tx1_success=%s tx2_success=%s final_value=%d final_version=%d' % (tx1_success, tx2_success, record.value, record.version))
assert tx1_success == True, "the first transaction to commit with a still-valid expected version should succeed"
assert tx2_success == False, \
    "the second transaction, using a now-stale version number, must be rejected rather than silently overwriting tx1's update - this is exactly the lost-update problem optimistic locking exists to prevent"
assert record.value == 110, "final value should only reflect tx1's update (100+10); tx2's would-be update (100+20=120) must NOT have been applied"
print("OPTIMISTIC_LOCK_TEST=PASS")
```

**面试怎么问+追问链**

- **方案批判迭代轴**:"我们把所有数据库更新都改成了乐观锁,减少了锁等待"——追问1:写冲突频繁的场景(比如高并发抢购同一件库存)会发生什么?候选人如果说"冲突了就重试呗"——追问2:极端高并发下,大量事务反复检测到版本冲突、反复重试,会不会比老老实实用悲观锁排队更慢?正确认识是:乐观锁在"冲突概率低"时收益明显,但在"冲突概率高、大家都在抢同一行"的场景下,重试风暴本身会消耗大量资源,这时候悲观锁(甚至专门的库存扣减方案,比如预扣减+异步对账)反而更合适——这条追问检验候选人是否理解乐观锁不是无条件优于悲观锁的"银弹"。

**常见坑**

- 用乐观锁但没有设计合理的重试逻辑(检测到冲突就直接放弃、把失败暴露给最终用户)——乐观锁的正确使用几乎总是需要配合"检测到冲突后自动重试几次"的上层逻辑,否则用户会在正常的并发场景下遇到本不该由他们承担的"提交失败,请重试"这类体验问题。

---

*本文件 6 个知识点,验证环境:全部 `.venv`(CAS/无锁结构用锁模拟原子语义验证逻辑正确性,`concurrent.futures`/`queue` 用标准库真实并发原语验证)。*
